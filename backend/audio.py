from __future__ import annotations

import io
import math
import wave
from dataclasses import dataclass
from typing import List

import numpy as np
import torch
import torch.nn.functional as F


TARGET_SAMPLE_RATE = 22050
CHUNK_DURATION_SECONDS = 6.0
MODEL_FRAMES = 128
MODEL_MELS = 128
N_FFT = 2048
HOP_LENGTH = 1024
MIN_DB = -80.0
MAX_DB = 0.0
EPSILON = 1e-10


@dataclass(frozen=True)
class DecodedAudio:
    samples: np.ndarray
    sample_rate: int

    @property
    def duration(self) -> float:
        return float(self.samples.shape[0] / self.sample_rate) if self.sample_rate else 0.0


@dataclass(frozen=True)
class AudioChunk:
    spectrogram: torch.Tensor
    start_time: float
    duration: float


_MEL_FILTERBANK: torch.Tensor | None = None
_HANN_WINDOW: torch.Tensor | None = None


def decode_wav_bytes(content: bytes) -> DecodedAudio:
    if not content:
        raise ValueError("Audio request body is empty.")

    try:
        with wave.open(io.BytesIO(content), "rb") as wav_file:
            channel_count = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            sample_rate = wav_file.getframerate()
            frame_count = wav_file.getnframes()
            compression = wav_file.getcomptype()
            frames = wav_file.readframes(frame_count)
    except wave.Error as error:
        raise ValueError(f"Expected PCM WAV audio: {error}") from error

    if compression != "NONE":
        raise ValueError("Compressed WAV audio is not supported.")
    if channel_count <= 0:
        raise ValueError("WAV audio must contain at least one channel.")
    if sample_rate <= 0:
        raise ValueError("WAV audio must declare a positive sample rate.")
    if frame_count <= 0 or not frames:
        raise ValueError("WAV audio contains no samples.")

    samples = _pcm_bytes_to_float32(frames, sample_width)
    if samples.size % channel_count != 0:
        samples = samples[: samples.size - (samples.size % channel_count)]

    samples = samples.reshape(-1, channel_count).mean(axis=1)
    samples = np.nan_to_num(samples, nan=0.0, posinf=0.0, neginf=0.0)
    samples = np.clip(samples, -1.0, 1.0).astype(np.float32, copy=False)

    if not samples.size:
        raise ValueError("WAV audio contains no complete samples.")

    return DecodedAudio(samples=samples, sample_rate=sample_rate)


def audio_bytes_to_model_chunks(
    content: bytes,
    *,
    chunk_duration: float = CHUNK_DURATION_SECONDS,
    target_sample_rate: int = TARGET_SAMPLE_RATE,
) -> List[AudioChunk]:
    audio = decode_wav_bytes(content)
    samples = resample_mono(audio.samples, audio.sample_rate, target_sample_rate)
    chunk_length = max(1, int(round(chunk_duration * target_sample_rate)))
    chunk_count = max(1, math.ceil(samples.shape[0] / chunk_length))
    prepared_chunks = []

    for index in range(chunk_count):
        start_sample = index * chunk_length
        end_sample = min(samples.shape[0], start_sample + chunk_length)
        segment = samples[start_sample:end_sample]
        mel_power = waveform_to_mel_power(
            segment,
            target_sample_rate=target_sample_rate,
            target_samples=chunk_length,
        )
        prepared_chunks.append((index, segment, mel_power))

    reference_power = max(float(mel_power.max()) for _, _, mel_power in prepared_chunks)
    reference_power = max(reference_power, EPSILON)

    chunks: List[AudioChunk] = []
    for index, segment, mel_power in prepared_chunks:
        start_sample = index * chunk_length
        duration = max(0.05, float(segment.shape[0] / target_sample_rate))
        chunks.append(
            AudioChunk(
                spectrogram=mel_power_to_db_spectrogram(mel_power, reference_power=reference_power),
                start_time=float(start_sample / target_sample_rate),
                duration=duration,
            )
        )

    return chunks


def resample_mono(samples: np.ndarray, source_sample_rate: int, target_sample_rate: int) -> np.ndarray:
    if source_sample_rate == target_sample_rate:
        return samples.astype(np.float32, copy=False)

    target_length = max(1, int(round(samples.shape[0] * target_sample_rate / source_sample_rate)))
    tensor = torch.from_numpy(samples.astype(np.float32, copy=True)).view(1, 1, -1)
    resampled = F.interpolate(tensor, size=target_length, mode="linear", align_corners=False)
    return resampled.view(-1).numpy().astype(np.float32, copy=False)


def waveform_to_log_mel_spectrogram(
    samples: np.ndarray,
    *,
    target_sample_rate: int = TARGET_SAMPLE_RATE,
    target_samples: int | None = None,
) -> torch.Tensor:
    mel_power = waveform_to_mel_power(
        samples,
        target_sample_rate=target_sample_rate,
        target_samples=target_samples,
    )
    return mel_power_to_db_spectrogram(
        mel_power,
        reference_power=max(float(mel_power.max()), EPSILON),
    )


def waveform_to_mel_power(
    samples: np.ndarray,
    *,
    target_sample_rate: int = TARGET_SAMPLE_RATE,
    target_samples: int | None = None,
) -> torch.Tensor:
    if target_samples is None:
        target_samples = int(round(CHUNK_DURATION_SECONDS * target_sample_rate))

    padded = np.zeros(target_samples, dtype=np.float32)
    copy_length = min(samples.shape[0], target_samples)
    if copy_length:
        padded[:copy_length] = samples[:copy_length]

    waveform = torch.from_numpy(padded)
    window = _hann_window()
    spectrum = torch.stft(
        waveform,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=N_FFT,
        window=window,
        center=False,
        return_complex=True,
    )
    power = spectrum.abs().pow(2)
    return torch.matmul(_mel_filterbank(), power)


def mel_power_to_db_spectrogram(mel_power: torch.Tensor, *, reference_power: float) -> torch.Tensor:
    mel_db = 10.0 * torch.log10(torch.clamp(mel_power / reference_power, min=EPSILON))
    mel_db = torch.clamp(mel_db, min=MIN_DB, max=MAX_DB)

    if mel_db.shape[1] < MODEL_FRAMES:
        mel_db = F.pad(mel_db, (0, MODEL_FRAMES - mel_db.shape[1]), value=MIN_DB)
    elif mel_db.shape[1] > MODEL_FRAMES:
        mel_db = mel_db[:, :MODEL_FRAMES]

    return mel_db.to(dtype=torch.float32)


def _pcm_bytes_to_float32(frames: bytes, sample_width: int) -> np.ndarray:
    if sample_width == 1:
        return (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    if sample_width == 2:
        return np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if sample_width == 3:
        raw = np.frombuffer(frames, dtype=np.uint8)
        complete = raw.size - (raw.size % 3)
        raw = raw[:complete].reshape(-1, 3)
        padded = np.zeros((raw.shape[0], 4), dtype=np.uint8)
        padded[:, :3] = raw
        padded[:, 3] = np.where((raw[:, 2] & 0x80) != 0, 0xFF, 0x00)
        return padded.view("<i4").reshape(-1).astype(np.float32) / 8388608.0
    if sample_width == 4:
        return np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0

    raise ValueError(f"Unsupported WAV sample width: {sample_width} bytes.")


def _hann_window() -> torch.Tensor:
    global _HANN_WINDOW
    if _HANN_WINDOW is None:
        _HANN_WINDOW = torch.hann_window(N_FFT, periodic=True)
    return _HANN_WINDOW


def _mel_filterbank() -> torch.Tensor:
    global _MEL_FILTERBANK
    if _MEL_FILTERBANK is None:
        fft_frequencies = np.linspace(0.0, TARGET_SAMPLE_RATE / 2.0, N_FFT // 2 + 1)
        mel_edges = np.linspace(
            _hz_to_mel(0.0),
            _hz_to_mel(TARGET_SAMPLE_RATE / 2.0),
            MODEL_MELS + 2,
        )
        hz_edges = _mel_to_hz(mel_edges)
        filters = np.zeros((MODEL_MELS, fft_frequencies.shape[0]), dtype=np.float32)

        for index in range(MODEL_MELS):
            lower = hz_edges[index]
            center = hz_edges[index + 1]
            upper = hz_edges[index + 2]

            lower_slope = (fft_frequencies - lower) / max(center - lower, EPSILON)
            upper_slope = (upper - fft_frequencies) / max(upper - center, EPSILON)
            filters[index] = np.maximum(0.0, np.minimum(lower_slope, upper_slope))

            enorm = 2.0 / max(upper - lower, EPSILON)
            filters[index] *= enorm

        _MEL_FILTERBANK = torch.from_numpy(filters)

    return _MEL_FILTERBANK


def _hz_to_mel(frequency: float | np.ndarray) -> float | np.ndarray:
    return 2595.0 * np.log10(1.0 + np.asarray(frequency) / 700.0)


def _mel_to_hz(mel: float | np.ndarray) -> float | np.ndarray:
    return 700.0 * (np.power(10.0, np.asarray(mel) / 2595.0) - 1.0)

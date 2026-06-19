from pathlib import Path
from typing import Dict, Iterable, List

import torch
from torch import nn


NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
TRAINED_MIDI_MIN = 39
TRAINED_MIDI_MAX = 52


class AlternativeCRNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128 * 128),
        )

    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        features = self.cnn(spectrogram)
        features = features.flatten(start_dim=1)
        logits = self.fc(features)
        return logits.view(-1, 128, 128)


def midi_to_note_name(midi: int) -> str:
    octave = midi // 12 - 1
    return f"{NOTE_NAMES[midi % 12]}{octave}"


def midi_to_frequency(midi: int) -> float:
    return 440.0 * (2 ** ((midi - 69) / 12))


def load_model(model_path: Path, device: torch.device) -> Dict[str, object]:
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = AlternativeCRNN().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return {
        "model": model,
        "epoch": checkpoint.get("epoch"),
        "train_loss": checkpoint.get("train_loss"),
        "val_loss": checkpoint.get("val_loss"),
        "architecture": checkpoint.get("model_architecture", "AlternativeCRNN"),
        "path": str(model_path),
    }


def validate_spectrogram(values: List[List[float]]) -> torch.Tensor:
    if len(values) != 128 or any(len(row) != 128 for row in values):
        raise ValueError("Spectrogram must be a 128 x 128 matrix.")

    tensor = torch.tensor(values, dtype=torch.float32)
    tensor = torch.nan_to_num(tensor, nan=-80.0, posinf=0.0, neginf=-80.0)
    tensor = torch.clamp(tensor, min=-80.0, max=0.0)
    return tensor


def predict_piano_roll(
    model: nn.Module,
    spectrograms: Iterable[torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    batch = torch.stack(list(spectrograms), dim=0).unsqueeze(1).to(device)
    with torch.no_grad():
        logits = model(batch)
        return torch.sigmoid(logits).cpu()


def piano_roll_to_events(
    piano_roll: torch.Tensor,
    start_time: float,
    duration: float,
    threshold: float,
    max_events: int = 128,
) -> List[Dict[str, float]]:
    frame_duration = duration / 128 if duration > 0 else 0.05
    decoded_frames = _decode_monophonic_frames(piano_roll, threshold)
    events: List[Dict[str, float]] = []
    active_midi = None
    active_start = None
    best_confidence = 0.0

    for frame, frame_note in enumerate(decoded_frames):
        midi = frame_note["midi"]
        confidence = frame_note["confidence"]

        if midi is None:
            if active_midi is not None:
                events.append(
                    _event_from_span(active_midi, active_start, frame, start_time, frame_duration, best_confidence)
                )
                active_midi = None
                active_start = None
                best_confidence = 0.0
            continue

        if active_midi is None:
            active_midi = midi
            active_start = frame
            best_confidence = confidence
        elif midi == active_midi:
            best_confidence = max(best_confidence, confidence)
        else:
            events.append(
                _event_from_span(active_midi, active_start, frame, start_time, frame_duration, best_confidence)
            )
            active_midi = midi
            active_start = frame
            best_confidence = confidence

    if active_midi is not None:
        events.append(
            _event_from_span(active_midi, active_start, 128, start_time, frame_duration, best_confidence)
        )

    events = _merge_close_events(events, max_gap=max(frame_duration * 3, 0.12))
    min_duration = max(frame_duration * 2, 0.09)
    events = [
        event for event in events
        if event["end"] - event["start"] >= min_duration or event["confidence"] >= threshold + 0.12
    ]
    events.sort(key=lambda event: (event["start"], event["midi"]))
    return events[:max_events]


def _decode_monophonic_frames(
    piano_roll: torch.Tensor,
    threshold: float,
    smoothing_radius: int = 2,
) -> List[Dict[str, float]]:
    note_slice = piano_roll[TRAINED_MIDI_MIN:TRAINED_MIDI_MAX + 1, :]
    raw_frames: List[Dict[str, float]] = []

    for frame in range(128):
        values = note_slice[:, frame]
        confidence, note_index = torch.max(values, dim=0)
        confidence_value = float(confidence)
        raw_frames.append({
            "midi": TRAINED_MIDI_MIN + int(note_index) if confidence_value >= threshold else None,
            "confidence": confidence_value,
        })

    smoothed_frames: List[Dict[str, float]] = []
    bridge_threshold = threshold * 0.8

    for frame in range(128):
        window_start = max(0, frame - smoothing_radius)
        window_end = min(128, frame + smoothing_radius + 1)
        candidates: Dict[int, Dict[str, float]] = {}

        for window_frame in range(window_start, window_end):
            midi = raw_frames[window_frame]["midi"]
            if midi is None:
                continue
            confidence = float(piano_roll[midi, window_frame])
            bucket = candidates.setdefault(midi, {"count": 0.0, "confidence": 0.0})
            bucket["count"] += 1.0
            bucket["confidence"] += confidence

        if not candidates:
            smoothed_frames.append({"midi": None, "confidence": 0.0})
            continue

        midi, score = max(
            candidates.items(),
            key=lambda candidate: (candidate[1]["count"], candidate[1]["confidence"]),
        )
        center_confidence = float(piano_roll[midi, frame])

        if center_confidence >= bridge_threshold:
            smoothed_frames.append({"midi": midi, "confidence": center_confidence})
        else:
            smoothed_frames.append({"midi": None, "confidence": 0.0})

    return smoothed_frames


def _merge_close_events(
    events: List[Dict[str, float]],
    max_gap: float,
) -> List[Dict[str, float]]:
    if not events:
        return []

    merged = [events[0]]
    for event in events[1:]:
        previous = merged[-1]
        if event["midi"] == previous["midi"] and event["start"] - previous["end"] <= max_gap:
            previous["end"] = max(previous["end"], event["end"])
            previous["confidence"] = max(previous["confidence"], event["confidence"])
            previous["id"] = f"ai-{round(previous['start'] * 1000)}-{previous['midi']}"
        else:
            merged.append(event)

    return merged


def _event_from_span(
    midi: int,
    start_frame: int,
    end_frame: int,
    start_time: float,
    frame_duration: float,
    confidence: float,
) -> Dict[str, float]:
    event_start = start_time + start_frame * frame_duration
    event_end = start_time + max(end_frame, start_frame + 1) * frame_duration
    return {
        "id": f"ai-{round(event_start * 1000)}-{midi}",
        "start": event_start,
        "end": event_end,
        "midi": midi,
        "note": midi_to_note_name(midi),
        "frequency": midi_to_frequency(midi),
        "confidence": confidence,
        "engine": "ai-model",
    }

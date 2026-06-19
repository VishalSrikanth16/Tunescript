from pathlib import Path
from typing import List, Optional

import torch
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.audio import audio_bytes_to_model_chunks
from backend.model import (
    TRAINED_MIDI_MAX,
    TRAINED_MIDI_MIN,
    load_model,
    midi_to_note_name,
    piano_roll_to_events,
    predict_piano_roll,
    validate_spectrogram,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "models" / "guitar_transcription_final.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_AUDIO_DURATION_SECONDS = 600.0

app = FastAPI(title="TuneScript AI Model API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
    ],
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_STATE = load_model(DEFAULT_MODEL, DEVICE)


class SpectrogramChunk(BaseModel):
    spectrogram: List[List[float]]
    start_time: float = Field(default=0.0, ge=0.0)
    duration: float = Field(default=6.0, gt=0.0)


class TranscriptionRequest(BaseModel):
    source_name: str = "browser-audio"
    threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    chunks: Optional[List[SpectrogramChunk]] = None
    spectrogram: Optional[List[List[float]]] = None
    duration: Optional[float] = Field(default=None, gt=0.0)


@app.get("/api/health")
def health() -> dict:
    return {
        "ok": True,
        "api_version": 2,
        "features": {
            "audio_transcription": True,
            "spectrogram_transcription": True,
        },
        "device": str(DEVICE),
        "model": {
            "architecture": MODEL_STATE["architecture"],
            "epoch": MODEL_STATE["epoch"],
            "train_loss": MODEL_STATE["train_loss"],
            "val_loss": MODEL_STATE["val_loss"],
            "trained_note_range": {
                "min_midi": TRAINED_MIDI_MIN,
                "max_midi": TRAINED_MIDI_MAX,
                "min_note": midi_to_note_name(TRAINED_MIDI_MIN),
                "max_note": midi_to_note_name(TRAINED_MIDI_MAX),
            },
            "path": MODEL_STATE["path"],
        },
    }


@app.post("/api/transcribe/spectrogram")
def transcribe_spectrogram(request: TranscriptionRequest) -> dict:
    chunks = request.chunks
    if chunks is None:
        if request.spectrogram is None:
            raise HTTPException(status_code=400, detail="Provide either chunks or spectrogram.")
        chunks = [
            SpectrogramChunk(
                spectrogram=request.spectrogram,
                start_time=0.0,
                duration=request.duration or 6.0,
            )
        ]

    if not chunks:
        raise HTTPException(status_code=400, detail="At least one spectrogram chunk is required.")

    try:
        tensors = [validate_spectrogram(chunk.spectrogram) for chunk in chunks]
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    return build_transcription_response(
        source_name=request.source_name,
        threshold=request.threshold,
        tensors=tensors,
        start_times=[chunk.start_time for chunk in chunks],
        durations=[chunk.duration for chunk in chunks],
    )


@app.post("/api/transcribe/audio")
async def transcribe_audio(
    request: Request,
    source_name: str = Query(default="browser-audio"),
    threshold: float = Query(default=0.3, ge=0.0, le=1.0),
) -> dict:
    content = await request.body()

    try:
        chunks = audio_bytes_to_model_chunks(content)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    duration = max(chunk.start_time + chunk.duration for chunk in chunks)
    if duration > MAX_AUDIO_DURATION_SECONDS:
        raise HTTPException(status_code=413, detail="Recordings longer than 10 minutes are not supported.")

    return build_transcription_response(
        source_name=source_name,
        threshold=threshold,
        tensors=[chunk.spectrogram for chunk in chunks],
        start_times=[chunk.start_time for chunk in chunks],
        durations=[chunk.duration for chunk in chunks],
    )


def build_transcription_response(
    *,
    source_name: str,
    threshold: float,
    tensors: List[torch.Tensor],
    start_times: List[float],
    durations: List[float],
) -> dict:
    piano_rolls = predict_piano_roll(MODEL_STATE["model"], tensors, DEVICE)
    events = []
    max_probability = 0.0

    for index, piano_roll in enumerate(piano_rolls):
        max_probability = max(max_probability, float(piano_roll.max()))
        events.extend(
            piano_roll_to_events(
                piano_roll,
                start_time=start_times[index],
                duration=durations[index],
                threshold=threshold,
            )
        )

    events.sort(key=lambda event: (event["start"], event["midi"]))
    return {
        "success": True,
        "source_name": source_name,
        "engine": "ai-model",
        "threshold": threshold,
        "max_probability": max_probability,
        "notes_detected": len(events),
        "notes": events,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.server:app", host="127.0.0.1", port=8000, reload=False)

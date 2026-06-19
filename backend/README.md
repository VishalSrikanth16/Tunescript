# TuneScript AI Model API

This backend loads `models/guitar_transcription_final.pth` and exposes it to the
React app at `http://127.0.0.1:8000/api`.

## Start

```powershell
cd "C:\React Projects\Tunescript"
& "$env:LOCALAPPDATA\Python\bin\python.exe" -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt
python -m uvicorn backend.server:app --host 127.0.0.1 --port 8000
```

Then start the frontend in another terminal:

```powershell
npm run dev
```

The frontend checks `/api/health`. If the backend is running, uploaded audio and
short microphone takes are decoded in the browser, encoded as WAV, and sent to
`/api/transcribe/audio`. The backend converts that audio into `128 x 128`
log-mel spectrogram chunks and runs checkpoint inference. The
`/api/transcribe/spectrogram` endpoint remains available for testing recovered
training chunks directly.

## Important

The checkpoint can now be loaded, but its recovered labels are very sparse. If
the model returns no confident notes, the frontend reports that model result
directly. It does not substitute a browser pitch transcription.

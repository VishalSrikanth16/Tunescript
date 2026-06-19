# TuneScript

TuneScript is a transcription workspace that Transcribes Music into Musical Notations.

Microphone can be used to record music in real time and uploaded files are decoded in the browser, encoded as .WAV,
sent to the backend, converted there into `128 x 128` log-mel spectrogram
chunks, passed through the PyTorch checkpoint, and then mapped to note events,
suggested guitar positions, ASCII tabs, and TXT/JSON exports.

## Start The Frontend

```powershell
cd "C:\React Projects\Tunescript"
npm install
npm run dev
```

Open the Vite URL, usually `http://localhost:5173`.

## Start The backend which is an AI model that does the conversion

Use a second terminal:

```powershell
cd "C:\React Projects\Tunescript"
& "$env:LOCALAPPDATA\Python\bin\python.exe" -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt
python -m uvicorn backend.server:app --host 127.0.0.1 --port 8000
```

The backend runs on `http://127.0.0.1:8000`. In dev, Vite proxies frontend
`/api/*` calls to that backend. The frontend checks `/api/health`; transcription
is locked until the model backend is reachable. Audio transcription posts WAV
data to `/api/transcribe/audio`, where Python performs model preprocessing and
inference.

You can also start the backend after the `.venv` dependencies are installed with:

```powershell
npm run backend
```

## What Works

- AI model inference for uploaded audio.
- AI model inference for short microphone takes
- Live microphone recording with a real Web Audio visualizer
- Note timing, frequency, confidence, and suggested fret positions
- Standard, Drop D, half-step-down, and Open G tunings
- ASCII tablature preview
- Browser-local project persistence
- TXT and JSON downloads

## Model Artifacts

The checkpoints contain a `model_state_dict` for a recovered architecture named
`AlternativeCRNN`. Its tensor shapes map to:

- Three convolution blocks: `1 -> 64 -> 128 -> 256`
- Batch normalization after each convolution
- Adaptive average pooling
- Fully connected head: `256 -> 512 -> 16384`
- Output reshaped as a `128 x 128` piano roll

The recovered training data is very sparse. The `processed_data/` directory
contains 1,696 NPZ chunks. Each chunk contains a `128 x 128` float32 spectrogram
and a `128 x 128` int64 piano roll. Across all chunks, only 72,245 of
27,787,264 label cells are nonzero (`0.259993%`), and 515 chunks have completely
empty piano rolls. Because of that, the app reports low/no-confidence model
results directly instead of substituting a non-model transcription.

## Scope

The AI backend is required for transcription, but checkpoint quality is still
limited by the recovered training labels. Treat model output as experimental.
The app does not yet provide chord separation, polished polyphonic transcription,
PDF, MusicXML, or MIDI generation. Still WIP.

## Quality Checks

```bash
npm run lint
npm test
npm run build
```

## Model Details

Model was fully trained by myself. 
The model architecture is a custom CRNN style PyTorch network named `AlternativeCRNN`.
It uses convolutional layers to learn patterns from spectrogram images, then maps those learned features into a piano roll style output. The output represents which notes are active across time frames. The model was trained using processed guitar audio data converted into spectrogram and piano-roll pairs. The spectrograms represent the sound of the guitar, while the piano roll labels tell the model which musical notes should be active at each moment.


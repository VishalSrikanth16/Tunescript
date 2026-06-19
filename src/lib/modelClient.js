const DEFAULT_MODEL_API_BASE = '/api';

export const MODEL_API_BASE = (
  import.meta.env.VITE_MODEL_API_BASE || DEFAULT_MODEL_API_BASE
).replace(/\/$/, '');

export async function checkModelHealth() {
  const response = await fetch(`${MODEL_API_BASE}/health`);

  if (!response.ok) {
    throw new Error(`Model API returned ${response.status}`);
  }

  return response.json();
}

export async function transcribeWithModel({ chunks, sourceName, threshold = 0.3 }) {
  const response = await fetch(`${MODEL_API_BASE}/transcribe/spectrogram`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      source_name: sourceName,
      threshold,
      chunks,
    }),
  });

  const payload = await response.json().catch(() => null);

  if (!response.ok) {
    if (response.status === 404) {
      throw new Error('Audio transcription route was not found. Stop the backend and restart npm run backend.');
    }
    throw new Error(payload?.detail || `Model API returned ${response.status}`);
  }

  return payload;
}

export async function transcribeAudioWithModel({ audioBlob, sourceName, threshold = 0.3 }) {
  const params = new URLSearchParams({
    source_name: sourceName || 'browser-audio',
    threshold: String(threshold),
  });
  const response = await fetch(`${MODEL_API_BASE}/transcribe/audio?${params.toString()}`, {
    method: 'POST',
    headers: {
      'Content-Type': 'audio/wav',
    },
    body: audioBlob,
  });

  const payload = await response.json().catch(() => null);

  if (!response.ok) {
    throw new Error(payload?.detail || `Model API returned ${response.status}`);
  }

  return payload;
}

import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react';
import {
  TUNINGS,
  buildAsciiTab,
  formatTranscript,
  getTabPosition,
} from '../lib/audioAnalysis';
import { audioBufferToWavBlob } from '../lib/audioEncoding';
import BrandMark from './BrandMark';
import {
  checkModelHealth,
  transcribeAudioWithModel,
} from '../lib/modelClient';
import {
  clearProject,
  loadProject,
  saveProject,
} from '../lib/projectStorage';

const DEFAULT_BARS = Array.from({ length: 36 }, (_, index) => 12 + ((index * 11) % 24));
const LIVE_CAPTURE_SECONDS = 6;
const MODEL_THRESHOLD = 0.3;

function getAudioContext() {
  const AudioContextClass = window.AudioContext || window.webkitAudioContext;
  return AudioContextClass ? new AudioContextClass() : null;
}

function downloadFile(filename, content, type) {
  const blob = new Blob([content], { type });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

function formatTime(seconds) {
  const minutes = Math.floor(seconds / 60);
  const remainder = seconds - minutes * 60;
  return `${minutes}:${remainder.toFixed(1).padStart(4, '0')}`;
}

function normalizeModelEvents(notes) {
  return notes
    .filter((note) => Number.isFinite(note.midi) && Number.isFinite(note.start))
    .map((note, index) => ({
      id: note.id || `ai-${index}-${Math.round(note.start * 1000)}`,
      start: note.start,
      end: Math.max(note.end ?? note.start + 0.05, note.start + 0.05),
      midi: note.midi,
      note: note.note || `MIDI ${note.midi}`,
      frequency: note.frequency ?? 0,
      confidence: note.confidence ?? 0,
      cents: 0,
      engine: 'ai-model',
    }));
}

function TranscriptionPage({ onBack }) {
  const [events, setEvents] = useState([]);
  const [tuningKey, setTuningKey] = useState('standard');
  const [sourceName, setSourceName] = useState('Untitled session');
  const [isListening, setIsListening] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [audioLevel, setAudioLevel] = useState(0);
  const [bars, setBars] = useState(DEFAULT_BARS);
  const [currentDetection, setCurrentDetection] = useState(null);
  const [statusMessage, setStatusMessage] = useState('Start the AI backend, then use microphone or file input.');
  const [dragActive, setDragActive] = useState(false);
  const [exportFormat, setExportFormat] = useState('txt');
  const [savedAt, setSavedAt] = useState('');
  const [analysisEngine, setAnalysisEngine] = useState('ai-model');
  const [emptyResultMessage, setEmptyResultMessage] = useState('');
  const [modelStatus, setModelStatus] = useState({
    state: 'checking',
    message: 'Checking AI model backend...',
    info: null,
  });

  const audioContextRef = useRef(null);
  const analyserRef = useRef(null);
  const sourceRef = useRef(null);
  const processorRef = useRef(null);
  const mediaStreamRef = useRef(null);
  const animationFrameRef = useRef(null);
  const listeningRef = useRef(false);
  const liveChunksRef = useRef([]);
  const liveSampleCountRef = useRef(0);
  const liveFinalizingRef = useRef(false);
  const eventsRef = useRef(events);
  const fileInputRef = useRef(null);

  useEffect(() => {
    eventsRef.current = events;
  }, [events]);

  const cleanupLiveAudio = useCallback(() => {
    listeningRef.current = false;

    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }

    processorRef.current?.disconnect();
    processorRef.current = null;
    sourceRef.current?.disconnect();
    sourceRef.current = null;

    mediaStreamRef.current?.getTracks().forEach((track) => track.stop());
    mediaStreamRef.current = null;

    if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
      audioContextRef.current.close();
    }

    audioContextRef.current = null;
    analyserRef.current = null;
    liveChunksRef.current = [];
    liveSampleCountRef.current = 0;
    liveFinalizingRef.current = false;
    setIsListening(false);
    setAudioLevel(0);
    setBars(DEFAULT_BARS);
  }, []);

  const stopLiveTranscription = useCallback(() => {
    cleanupLiveAudio();
    setCurrentDetection(null);
    setProgress(0);
    setStatusMessage('AI microphone capture cancelled before model inference.');
  }, [cleanupLiveAudio]);

  useEffect(() => {
    const savedProject = loadProject();

    if (savedProject?.events?.length) {
      setEvents(savedProject.events);
      setSourceName(savedProject.sourceName || 'Saved session');
      setTuningKey(savedProject.tuningKey || 'standard');
      setAnalysisEngine(savedProject.analysisEngine || 'ai-model');
      setSavedAt(savedProject.savedAt || '');
      setStatusMessage('Restored the last locally saved project.');
    }

    return () => {
      listeningRef.current = false;
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      processorRef.current?.disconnect();
      sourceRef.current?.disconnect();
      mediaStreamRef.current?.getTracks().forEach((track) => track.stop());
      if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
        audioContextRef.current.close();
      }
    };
  }, []);

  const refreshModelStatus = useCallback(async () => {
    setModelStatus((previous) => ({
      ...previous,
      state: 'checking',
      message: 'Checking AI model backend...',
    }));

    try {
      const health = await checkModelHealth();
      if (!health.features?.audio_transcription) {
        throw new Error('Backend is running old code. Stop it and restart npm run backend.');
      }
      const trainedRange = health.model?.trained_note_range;
      const rangeLabel = trainedRange
        ? ` (${trainedRange.min_note}-${trainedRange.max_note} labels)`
        : '';
      setModelStatus({
        state: 'ready',
        message: `AI model ready on ${health.device}${rangeLabel}`,
        info: health.model,
      });
    } catch (error) {
      setModelStatus({
        state: 'offline',
        message: `AI model backend offline: ${error.message}`,
        info: null,
      });
    }
  }, []);

  useEffect(() => {
    refreshModelStatus();
  }, [refreshModelStatus]);

  const transcript = useMemo(
    () => formatTranscript(events, tuningKey, sourceName),
    [events, sourceName, tuningKey],
  );

  const tablature = useMemo(
    () => buildAsciiTab(events, tuningKey),
    [events, tuningKey],
  );

  const statistics = useMemo(() => {
    if (!events.length) {
      return { duration: 0, confidence: 0, lowest: '--', highest: '--' };
    }

    const sorted = [...events].sort((a, b) => a.midi - b.midi);
    const confidence = events.reduce((sum, event) => sum + event.confidence, 0) / events.length;

    return {
      duration: Math.max(...events.map((event) => event.end)),
      confidence,
      lowest: sorted[0].note,
      highest: sorted.at(-1).note,
    };
  }, [events]);

  const transcribeAudioBufferWithModel = useCallback(async (audioBuffer, nextSourceName) => {
    if (modelStatus.state !== 'ready') {
      throw new Error('AI model backend is not ready. Start it with npm run backend.');
    }

    setStatusMessage('Encoding audio for the AI model backend...');
    setProgress(10);
    const audioBlob = audioBufferToWavBlob(audioBuffer);
    setProgress(30);

    setStatusMessage('Sending audio to the Torch AI model...');
    const modelResult = await transcribeAudioWithModel({
      audioBlob,
      sourceName: nextSourceName,
      threshold: MODEL_THRESHOLD,
    });
    setProgress(90);

    const modelEvents = normalizeModelEvents(modelResult.notes || []);
    setEvents(modelEvents);
    setAnalysisEngine('ai-model');
    setProgress(100);

    if (modelEvents.length > 0) {
      setEmptyResultMessage('');
      setStatusMessage(
        `AI model finished ${nextSourceName}: ${modelEvents.length} note events detected.`,
      );
    } else {
      const noNotesMessage = `The AI model returned no notes above ${Math.round(MODEL_THRESHOLD * 100)}% confidence. Max probability: ${Math.round((modelResult.max_probability || 0) * 100)}%.`;
      setEmptyResultMessage(noNotesMessage);
      setStatusMessage(
        `AI model completed ${nextSourceName}. ${noNotesMessage}`,
      );
    }

    return { modelEvents, modelResult };
  }, [modelStatus.state]);

  const finalizeLiveModelCapture = useCallback(async () => {
    if (liveFinalizingRef.current) {
      return;
    }

    liveFinalizingRef.current = true;
    const audioContext = audioContextRef.current;
    const sampleRate = audioContext?.sampleRate || 44100;
    const liveChunks = liveChunksRef.current;
    const totalSamples = liveSampleCountRef.current;
    cleanupLiveAudio();
    setCurrentDetection(null);

    if (!totalSamples) {
      setStatusMessage('AI microphone capture ended without audio samples.');
      return;
    }

    const samples = new Float32Array(totalSamples);
    let offset = 0;
    liveChunks.forEach((chunk) => {
      samples.set(chunk.subarray(0, Math.min(chunk.length, totalSamples - offset)), offset);
      offset += chunk.length;
    });

    const audioBuffer = {
      sampleRate,
      numberOfChannels: 1,
      length: samples.length,
      getChannelData: () => samples,
    };

    setIsAnalyzing(true);
    setSourceName('Live AI microphone take');

    try {
      const { modelEvents } = await transcribeAudioBufferWithModel(audioBuffer, 'Live AI microphone take');
      setCurrentDetection(modelEvents[0] ? { ...modelEvents[0], name: modelEvents[0].note } : null);
    } catch (error) {
      setEvents([]);
      setAnalysisEngine('ai-model');
      setEmptyResultMessage('');
      setStatusMessage(`AI microphone transcription failed: ${error.message}`);
    } finally {
      setIsAnalyzing(false);
      setProgress(0);
    }
  }, [cleanupLiveAudio, transcribeAudioBufferWithModel]);

  const startLiveTranscription = async () => {
    if (modelStatus.state !== 'ready') {
      setStatusMessage('AI model backend is required. Start it with npm run backend, then try microphone capture again.');
      return;
    }

    if (!navigator.mediaDevices?.getUserMedia) {
      setStatusMessage('This browser does not support microphone capture.');
      return;
    }

    const audioContext = getAudioContext();
    if (!audioContext) {
      setStatusMessage('The Web Audio API is not available in this browser.');
      return;
    }

    try {
      setStatusMessage('Requesting microphone permission...');
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false,
          channelCount: 1,
        },
      });

      if (audioContext.state === 'suspended') {
        await audioContext.resume();
      }

      const analyser = audioContext.createAnalyser();
      analyser.fftSize = 4096;
      analyser.smoothingTimeConstant = 0.7;
      const source = audioContext.createMediaStreamSource(stream);
      const processor = audioContext.createScriptProcessor(4096, 1, 1);
      source.connect(analyser);
      source.connect(processor);
      processor.connect(audioContext.destination);

      audioContextRef.current = audioContext;
      analyserRef.current = analyser;
      sourceRef.current = source;
      processorRef.current = processor;
      mediaStreamRef.current = stream;
      listeningRef.current = true;
      liveChunksRef.current = [];
      liveSampleCountRef.current = 0;
      liveFinalizingRef.current = false;
      setSourceName('Live microphone session');
      setAnalysisEngine('ai-model');
      setIsListening(true);
      setCurrentDetection(null);
      setProgress(0);
      setStatusMessage(`Recording ${LIVE_CAPTURE_SECONDS}s for AI model transcription...`);

      const frequencyData = new Uint8Array(analyser.frequencyBinCount);
      let lastVisualUpdate = 0;

      processor.onaudioprocess = (event) => {
        if (!listeningRef.current) {
          return;
        }

        const input = event.inputBuffer.getChannelData(0);
        const output = event.outputBuffer.getChannelData(0);
        output.fill(0);

        liveChunksRef.current.push(new Float32Array(input));
        liveSampleCountRef.current += input.length;

        const capturedSeconds = liveSampleCountRef.current / audioContext.sampleRate;
        setProgress(Math.min(100, Math.round((capturedSeconds / LIVE_CAPTURE_SECONDS) * 100)));

        if (capturedSeconds >= LIVE_CAPTURE_SECONDS) {
          finalizeLiveModelCapture();
        }
      };

      const update = (timestamp) => {
        if (!listeningRef.current) {
          return;
        }

        if (timestamp - lastVisualUpdate >= 50) {
          analyser.getByteFrequencyData(frequencyData);
          const nextBars = Array.from({ length: 36 }, (_, index) => {
            const normalized = index / 35;
            const bin = Math.min(
              frequencyData.length - 1,
              Math.floor(normalized * normalized * 420),
            );
            return Math.max(5, (frequencyData[bin] / 255) * 100);
          });
          const level = nextBars.reduce((sum, value) => sum + value, 0) / nextBars.length;
          setBars(nextBars);
          setAudioLevel(Math.min(100, level * 1.6));
          lastVisualUpdate = timestamp;
        }

        animationFrameRef.current = requestAnimationFrame(update);
      };

      animationFrameRef.current = requestAnimationFrame(update);
    } catch (error) {
      if (audioContext.state !== 'closed') {
        audioContext.close();
      }

      const message = error.name === 'NotAllowedError'
        ? 'Microphone permission was denied. Allow it in the browser site settings and try again.'
        : `Microphone capture failed: ${error.message}`;
      setStatusMessage(message);
    }
  };

  const toggleListening = () => {
    if (isListening) {
      stopLiveTranscription();
    } else {
      startLiveTranscription();
    }
  };

  const processFile = useCallback(async (file) => {
    if (!file) {
      return;
    }

    if (file.size > 100 * 1024 * 1024) {
      setStatusMessage('Choose an audio file smaller than 100 MB.');
      return;
    }

    if (modelStatus.state !== 'ready') {
      setStatusMessage('AI model backend is required. Start it with npm run backend, then upload again.');
      return;
    }

    const audioContext = getAudioContext();
    if (!audioContext) {
      setStatusMessage('The Web Audio API is not available in this browser.');
      return;
    }

    stopLiveTranscription();
    setIsAnalyzing(true);
    setProgress(0);
    setEvents([]);
    setEmptyResultMessage('');
    setSourceName(file.name);
    setStatusMessage(`Decoding ${file.name}...`);

    try {
      const arrayBuffer = await file.arrayBuffer();
      const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);

      if (audioBuffer.duration > 600) {
        throw new Error('Recordings longer than 10 minutes are not supported.');
      }

      await transcribeAudioBufferWithModel(audioBuffer, file.name);
    } catch (error) {
      setEvents([]);
      setAnalysisEngine('ai-model');
      setEmptyResultMessage(`Upload failed before model inference: ${error.message}`);
      setStatusMessage(`AI model could not analyze this file: ${error.message}`);
    } finally {
      if (audioContext.state !== 'closed') {
        await audioContext.close();
      }
      setIsAnalyzing(false);
      setProgress(0);
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  }, [modelStatus.state, stopLiveTranscription, transcribeAudioBufferWithModel]);

  const handleClear = () => {
    stopLiveTranscription();
    setEvents([]);
    setEmptyResultMessage('');
    setSourceName('Untitled session');
    setAnalysisEngine('ai-model');
    setSavedAt('');
    clearProject();
    setStatusMessage('Cleared the current session and local save.');
  };

  const handleSave = () => {
    const project = saveProject({ events, sourceName, tuningKey, analysisEngine });
    if (project) {
      setSavedAt(project.savedAt);
      setStatusMessage('Project saved in this browser.');
    } else {
      setStatusMessage('Browser storage is unavailable, so the project was not saved.');
    }
  };

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(transcript);
      setStatusMessage('Transcription copied to the clipboard.');
    } catch {
      setStatusMessage('Clipboard access was unavailable. Use the download action instead.');
    }
  };

  const handleExport = () => {
    const safeName = sourceName
      .replace(/\.[^/.]+$/, '')
      .replace(/[^a-z0-9]+/gi, '-')
      .replace(/^-|-$/g, '')
      .toLowerCase() || 'tunescript';

    if (exportFormat === 'json') {
      downloadFile(
        `${safeName}.json`,
        JSON.stringify({
          version: 1,
          sourceName,
          tuning: TUNINGS[tuningKey],
          exportedAt: new Date().toISOString(),
          events,
        }, null, 2),
        'application/json',
      );
    } else {
      downloadFile(`${safeName}.txt`, transcript, 'text/plain;charset=utf-8');
    }

    setStatusMessage(`Downloaded ${exportFormat.toUpperCase()} export.`);
  };

  const handleDrop = (event) => {
    event.preventDefault();
    setDragActive(false);
    processFile(event.dataTransfer.files?.[0]);
  };

  const currentPosition = currentDetection
    ? getTabPosition(currentDetection.midi, tuningKey)
    : null;

  return (
    <main className="studio-page">
      <header className="studio-header">
        <button className="back-button" type="button" onClick={onBack}>
          <span className="back-arrow" aria-hidden="true" />
          <span className="sr-only">Home</span>
        </button>
        <a className="brand studio-brand" href="#home" onClick={onBack}>
          <BrandMark />
          <span>TuneScript</span>
        </a>
      </header>

      <section className="studio-intro">
        <div>
          <p className="eyebrow">Transcription workspace</p>
          <h1>Turn a guitar line into a useful pitch sketch.</h1>
        </div>
        <p>
          Microphone takes and uploaded files are sent to the AI backend, where
          they are converted into model-ready spectrogram chunks and passed
          through the recovered PyTorch checkpoint. Transcription is locked
          until the AI backend is running.
        </p>
      </section>

      <div className="studio-grid">
        <section className="panel input-panel">
          <div className="panel-heading">
            <div>
              <p className="panel-kicker">Input</p>
              <h2>Listen and analyze</h2>
            </div>
            <div className={`level-label ${isListening ? 'is-live' : ''}`}>
              {isListening ? 'Recording' : isAnalyzing ? 'AI inference' : analysisEngine === 'ai-model' ? 'AI model' : 'Idle'}
            </div>
          </div>

          <div className="visualizer">
            <div className="visualizer-topline">
              <span>{currentDetection ? currentDetection.name : isListening ? 'Recording AI take' : 'Waiting for AI model'}</span>
              <span>{currentDetection ? `${currentDetection.frequency.toFixed(1)} Hz` : isListening ? `${progress}% captured` : `${Math.round(audioLevel)}% level`}</span>
            </div>
            <div className="visualizer-bars" aria-hidden="true">
              {bars.map((height, index) => (
                <span key={index} style={{ height: `${height}%` }} />
              ))}
            </div>
            <div className="visualizer-footer">
              <span>
                {currentPosition
                  ? `${currentPosition.label} string, fret ${currentPosition.fret}`
                  : 'Suggested position will appear here'}
              </span>
              <span>{TUNINGS[tuningKey].shortName}</span>
            </div>
          </div>

          <button
            className={`record-button ${isListening ? 'is-recording' : ''}`}
            type="button"
            onClick={toggleListening}
            disabled={isAnalyzing || modelStatus.state !== 'ready'}
          >
            <span className="record-dot" aria-hidden="true" />
            {isListening ? 'Cancel AI microphone take' : 'Record 6s AI microphone take'}
          </button>

          <div className={`model-note ${modelStatus.state === 'ready' ? 'is-ready' : ''}`}>
            <strong>AI backend</strong>
            <span>{modelStatus.message}</span>
            <button
              className="model-refresh-button"
              type="button"
              onClick={refreshModelStatus}
              disabled={modelStatus.state === 'checking'}
            >
              Recheck backend
            </button>
          </div>

          <div className="input-divider"><span>or use a recording</span></div>

          <div
            className={`drop-zone ${dragActive ? 'is-dragging' : ''}`}
            onDragEnter={(event) => {
              event.preventDefault();
              setDragActive(true);
            }}
            onDragOver={(event) => event.preventDefault()}
            onDragLeave={() => setDragActive(false)}
            onDrop={handleDrop}
          >
            <div className="upload-symbol" aria-hidden="true">&uarr;</div>
            <strong>Drop an audio file here</strong>
            <span>MP3, WAV, M4A, OGG, or any browser-decodable audio up to 100 MB. Requires the AI backend.</span>
            <button
              className="secondary-button"
              type="button"
              onClick={() => fileInputRef.current?.click()}
              disabled={isAnalyzing || modelStatus.state !== 'ready'}
            >
              {isAnalyzing ? 'Analyzing...' : 'Choose audio file'}
            </button>
            <input
              ref={fileInputRef}
              type="file"
              accept="audio/*"
              hidden
              onChange={(event) => processFile(event.target.files?.[0])}
            />
          </div>

          {isAnalyzing && (
            <div className="analysis-progress" aria-label={`Analysis ${progress}% complete`}>
              <div><span>AI model analysis</span><strong>{progress}%</strong></div>
              <div className="progress-track"><span style={{ width: `${progress}%` }} /></div>
            </div>
          )}

        </section>

        <section className="panel results-panel">
          <div className="panel-heading results-heading">
            <div>
              <p className="panel-kicker">Output</p>
              <h2>Detected notes</h2>
            </div>
            <div className="result-actions">
              <button type="button" onClick={handleCopy} disabled={!events.length}>Copy</button>
              <button type="button" onClick={handleClear} disabled={!events.length && !savedAt}>Clear</button>
            </div>
          </div>

          <div className="stats-grid">
            <div><span>Events</span><strong>{events.length}</strong></div>
            <div><span>Length</span><strong>{formatTime(statistics.duration)}</strong></div>
            <div><span>Range</span><strong>{statistics.lowest} - {statistics.highest}</strong></div>
            <div><span>Avg. confidence</span><strong>{events.length ? `${Math.round(statistics.confidence * 100)}%` : '--'}</strong></div>
          </div>

          <div className="event-list">
            {events.length ? (
              <>
                <div className="event-row event-row-header">
                  <span>Time</span>
                  <span>Note</span>
                  <span>Pitch</span>
                  <span>Position</span>
                </div>
                {events.map((event) => {
                  const position = getTabPosition(event.midi, tuningKey);
                  return (
                    <div className="event-row" key={event.id}>
                      <span>{formatTime(event.start)}</span>
                      <strong>{event.note}</strong>
                      <span>{event.frequency.toFixed(1)} Hz</span>
                      <span>{position ? `${position.label} / ${position.fret}` : 'Out of range'}</span>
                    </div>
                  );
                })}
              </>
            ) : (
              <div className="empty-state">
                <BrandMark className="empty-brand-mark" />
                <h3>No note events yet</h3>
                <p>{emptyResultMessage || 'Start the AI backend, then record a microphone take or choose a file for model inference.'}</p>
              </div>
            )}
          </div>
        </section>
      </div>

      <section className="panel tab-panel">
        <div className="panel-heading">
          <div>
            <p className="panel-kicker">Tab preview</p>
            <h2>Suggested guitar positions</h2>
          </div>
          <label className="select-field">
            <span>Tuning</span>
            <select value={tuningKey} onChange={(event) => setTuningKey(event.target.value)}>
              {Object.entries(TUNINGS).map(([key, tuning]) => (
                <option key={key} value={key}>{tuning.name} ({tuning.shortName})</option>
              ))}
            </select>
          </label>
        </div>
        <pre className="tab-output">{tablature}</pre>
      </section>

      <section className="project-bar">
        <div>
          <p className="panel-kicker">Project</p>
          <strong>{sourceName}</strong>
          <span>
            {savedAt
              ? `Saved locally ${new Date(savedAt).toLocaleString()}`
              : 'Not saved yet'}
          </span>
        </div>
        <div className="project-actions">
          <button className="secondary-button" type="button" onClick={handleSave} disabled={!events.length}>
            Save in browser
          </button>
          <label className="select-field compact-select">
            <span>Format</span>
            <select value={exportFormat} onChange={(event) => setExportFormat(event.target.value)}>
              <option value="txt">TXT with tablature</option>
              <option value="json">JSON note events</option>
            </select>
          </label>
          <button className="primary-button export-button" type="button" onClick={handleExport} disabled={!events.length}>
            Download
          </button>
        </div>
      </section>

      <p className="status-message" role="status" aria-live="polite">{statusMessage}</p>
    </main>
  );
}

export default TranscriptionPage;

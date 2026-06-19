import { useEffect, useState } from 'react';
import BrandMark from './components/BrandMark';
import TranscriptionPage from './components/TranscriptionPage';

function HomePage({ onStart }) {
  return (
    <main className="home-page">
      <nav className="home-nav" aria-label="Primary navigation">
        <a className="brand" href="#home" aria-label="TuneScript home">
          <BrandMark />
          <span>TuneScript</span>
        </a>
        <button className="nav-cta" type="button" onClick={onStart}>
          Open studio
        </button>
      </nav>

      <section className="hero">
        <div className="hero-copy">
          <p className="eyebrow">Powered by AI</p>
          <h1>Your Guitar Note Transciber is here.</h1>
          <p className="hero-description">
            TuneScript records a short microphone take or decodes an uploaded
            audio file, sends WAV audio to the local FastAPI backend, runs your
            recovered checkpoint, and displays the model note events with timing,
            confidence, and suggested guitar positions.
          </p>
          <div className="hero-actions">
            <button className="primary-button" type="button" onClick={onStart}>
              Start transcribing
            </button>
            <a className="text-link" href="#how-it-works">
              See how it works
            </a>
          </div>
        </div>

        <div className="hero-instrument" aria-hidden="true">
          <div className="signal-card signal-card-main">
            <div className="signal-card-header">
              <span>AI model output</span>
              <span className="status-pill">Ready</span>
            </div>
            <div className="detected-note">F#2</div>
            <div className="frequency">92.5 Hz</div>
            <div className="waveform">
              {Array.from({ length: 28 }).map((_, index) => (
                <span
                  key={index}
                  style={{ height: `${24 + ((index * 17) % 58)}%` }}
                />
              ))}
            </div>
            <div className="fret-readout">
              <span>E2 string</span>
              <strong>fret 2</strong>
            </div>
          </div>
          <div className="hero-orbit hero-orbit-one" />
          <div className="hero-orbit hero-orbit-two" />
        </div>
      </section>

      <section className="workflow-section" id="how-it-works">
        <div>
          <p className="eyebrow">What it does ?</p>
          <h2>Convert your audio files into guitar notes or use your microphone to record notes in real time!</h2>
        </div>
        <div className="workflow-grid">
          <article>
            <span>01</span>
            <h3>Start the backend</h3>
            <p>Run the local FastAPI service so the PyTorch checkpoint is loaded and ready.</p>
          </article>
          <article>
            <span>02</span>
            <h3>Send audio to the model</h3>
            <p>Record six seconds or upload a browser-decodable file for backend preprocessing and inference.</p>
          </article>
          <article>
            <span>03</span>
            <h3>Review model detections</h3>
            <p>Inspect low-range experimental note events, fret suggestions, confidence, and TXT or JSON exports.</p>
          </article>
        </div>
      </section>
    </main>
  );
}

export default function App() {
  const [view, setView] = useState(
    window.location.hash === '#studio' ? 'studio' : 'home',
  );

  useEffect(() => {
    const handleHashChange = () => {
      setView(window.location.hash === '#studio' ? 'studio' : 'home');
    };

    window.addEventListener('hashchange', handleHashChange);
    return () => window.removeEventListener('hashchange', handleHashChange);
  }, []);

  const navigate = (nextView) => {
    window.location.hash = nextView === 'studio' ? 'studio' : 'home';
    setView(nextView);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  if (view === 'studio') {
    return <TranscriptionPage onBack={() => navigate('home')} />;
  }

  return <HomePage onStart={() => navigate('studio')} />;
}

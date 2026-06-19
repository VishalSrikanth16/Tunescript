import test from 'node:test';
import assert from 'node:assert/strict';
import {
  buildAsciiTab,
  formatTranscript,
  getTabPosition,
  midiToNoteName,
} from './audioAnalysis.js';
import { audioBufferToWavArrayBuffer } from './audioEncoding.js';

function sineWave(frequency, sampleRate = 44100, duration = 0.2) {
  const samples = new Float32Array(Math.floor(sampleRate * duration));

  for (let i = 0; i < samples.length; i += 1) {
    samples[i] = Math.sin((2 * Math.PI * frequency * i) / sampleRate) * 0.7;
  }

  return samples;
}

test('converts MIDI notes to display names', () => {
  assert.equal(midiToNoteName(69), 'A4');
  assert.equal(midiToNoteName(40), 'E2');
});

function readAscii(view, offset, length) {
  return Array.from({ length }, (_, index) => String.fromCharCode(view.getUint8(offset + index))).join('');
}

test('encodes an AudioBuffer-like object as mono PCM WAV', () => {
  const sampleRate = 44100;
  const samples = sineWave(110, sampleRate, 0.25);
  const wav = audioBufferToWavArrayBuffer({
    sampleRate,
    numberOfChannels: 1,
    length: samples.length,
    getChannelData: () => samples,
  });
  const view = new DataView(wav);

  assert.equal(readAscii(view, 0, 4), 'RIFF');
  assert.equal(readAscii(view, 8, 4), 'WAVE');
  assert.equal(readAscii(view, 12, 4), 'fmt ');
  assert.equal(readAscii(view, 36, 4), 'data');
  assert.equal(view.getUint16(20, true), 1);
  assert.equal(view.getUint16(22, true), 1);
  assert.equal(view.getUint32(24, true), sampleRate);
  assert.equal(view.getUint16(34, true), 16);
  assert.equal(view.getUint32(40, true), samples.length * 2);
});

test('maps notes to playable positions in the selected tuning', () => {
  assert.deepEqual(getTabPosition(40, 'standard'), {
    stringIndex: 0,
    fret: 0,
    label: 'E2',
  });
  assert.deepEqual(getTabPosition(38, 'dropD'), {
    stringIndex: 0,
    fret: 0,
    label: 'D2',
  });
});

test('builds exportable text and tablature', () => {
  const events = [{
    id: 'event-1',
    start: 0,
    end: 0.5,
    midi: 40,
    note: 'E2',
    frequency: 82.41,
    confidence: 0.95,
    cents: 0,
  }];

  assert.match(buildAsciiTab(events), /E2\s+\|-0/);
  assert.match(formatTranscript(events), /Detected note events: 1/);
});

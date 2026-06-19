const NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];

export const TUNINGS = {
  standard: {
    name: 'Standard',
    shortName: 'E A D G B E',
    strings: [
      { label: 'E2', midi: 40 },
      { label: 'A2', midi: 45 },
      { label: 'D3', midi: 50 },
      { label: 'G3', midi: 55 },
      { label: 'B3', midi: 59 },
      { label: 'E4', midi: 64 },
    ],
  },
  dropD: {
    name: 'Drop D',
    shortName: 'D A D G B E',
    strings: [
      { label: 'D2', midi: 38 },
      { label: 'A2', midi: 45 },
      { label: 'D3', midi: 50 },
      { label: 'G3', midi: 55 },
      { label: 'B3', midi: 59 },
      { label: 'E4', midi: 64 },
    ],
  },
  halfStepDown: {
    name: 'Half step down',
    shortName: 'Eb Ab Db Gb Bb Eb',
    strings: [
      { label: 'Eb2', midi: 39 },
      { label: 'Ab2', midi: 44 },
      { label: 'Db3', midi: 49 },
      { label: 'Gb3', midi: 54 },
      { label: 'Bb3', midi: 58 },
      { label: 'Eb4', midi: 63 },
    ],
  },
  openG: {
    name: 'Open G',
    shortName: 'D G D G B D',
    strings: [
      { label: 'D2', midi: 38 },
      { label: 'G2', midi: 43 },
      { label: 'D3', midi: 50 },
      { label: 'G3', midi: 55 },
      { label: 'B3', midi: 59 },
      { label: 'D4', midi: 62 },
    ],
  },
};

export function midiToNoteName(midi) {
  const roundedMidi = Math.round(midi);
  const octave = Math.floor(roundedMidi / 12) - 1;
  return `${NOTE_NAMES[((roundedMidi % 12) + 12) % 12]}${octave}`;
}

export function getTabPosition(midi, tuningKey = 'standard') {
  const tuning = TUNINGS[tuningKey] ?? TUNINGS.standard;
  const candidates = tuning.strings
    .map((string, stringIndex) => ({
      stringIndex,
      fret: midi - string.midi,
      label: string.label,
    }))
    .filter(({ fret }) => fret >= 0 && fret <= 24)
    .sort((a, b) => a.fret - b.fret || b.stringIndex - a.stringIndex);

  return candidates[0] ?? null;
}

export function buildAsciiTab(events, tuningKey = 'standard') {
  const tuning = TUNINGS[tuningKey] ?? TUNINGS.standard;
  const displayStrings = [...tuning.strings].reverse();
  const lines = displayStrings.map((string) => `${string.label.padEnd(3, ' ')}|`);
  const times = ['    '];

  events.slice(0, 48).forEach((event) => {
    const position = getTabPosition(event.midi, tuningKey);
    const tokenWidth = Math.max(4, String(position?.fret ?? '').length + 2);

    displayStrings.forEach((string, displayIndex) => {
      const sourceIndex = tuning.strings.indexOf(string);
      const value = position?.stringIndex === sourceIndex ? String(position.fret) : '';
      lines[displayIndex] += `-${value.padEnd(tokenWidth - 1, '-')}`;
    });

    times[0] += `${event.start.toFixed(1).padEnd(tokenWidth, ' ')} `;
  });

  if (events.length === 0) {
    return 'No notes detected yet.';
  }

  const suffix = events.length > 48
    ? `\n\nShowing the first 48 of ${events.length} note events.`
    : '';

  return `${times[0].trimEnd()}\n${lines.join('\n')}${suffix}`;
}

export function formatTranscript(events, tuningKey = 'standard', sourceName = 'Untitled session') {
  const tuning = TUNINGS[tuningKey] ?? TUNINGS.standard;
  const rows = events.map((event) => {
    const position = getTabPosition(event.midi, tuningKey);
    const tab = position ? `${position.label}, fret ${position.fret}` : 'outside 24-fret range';
    return `${event.start.toFixed(2)}s  ${event.note.padEnd(4, ' ')}  ${event.frequency.toFixed(1)} Hz  ${tab}`;
  });

  return [
    'TuneScript transcription',
    `Source: ${sourceName}`,
    `Tuning: ${tuning.name} (${tuning.shortName})`,
    `Detected note events: ${events.length}`,
    '',
    ...rows,
    '',
    'ASCII tablature',
    buildAsciiTab(events, tuningKey),
  ].join('\n');
}

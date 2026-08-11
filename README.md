# Suno → MIDI Converter

Convert Suno (or any) audio into per-stem MIDI files ready for your DAW.

## Quick start

```bash
pip install flask demucs basic-pitch librosa pretty_midi soundfile
python app.py
# Open http://localhost:5050
```

## What it does

1. **Stem separation** — Demucs `htdemucs` splits your audio into 4 tracks:
   Vocals, Bass, Drums, Other (pads/chords/guitar)
2. **Melodic → MIDI** — Spotify's `basic-pitch` runs polyphonic pitch detection
   on Vocals, Bass, and Other stems
3. **Drums → MIDI** — librosa onset detection + spectral classification maps
   drum hits to General MIDI channel 10 (kick 36, snare 38, hi-hat 42)
4. **Download** — all 4 `.mid` files packaged in a ZIP

## DAW import

| DAW | How to import |
|-----|--------------|
| Ableton Live | Drag `.mid` into Arrangement / Session view |
| FL Studio | Piano Roll → File → Import MIDI |
| Logic Pro | Drag `.mid` into Tracks area |
| Pro Tools | File → Import → MIDI |

## Processing time (CPU)
- 3-min song: ~4–8 minutes
- Demucs is the slowest step; a GPU cuts it to under a minute

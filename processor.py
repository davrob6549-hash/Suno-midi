"""
Audio processing pipeline:
1. Separate stems with Demucs (vocals, drums, bass, other)
2. Convert melodic stems to MIDI via basic-pitch
3. Convert drum stem to MIDI via onset detection + drum classification
"""

import os
import shutil
import logging
import numpy as np
import soundfile as sf
import librosa
import pretty_midi

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# General MIDI drum note map (channel 9, 0-indexed)
DRUM_NOTES = {
    "kick":     36,  # Bass Drum 1
    "snare":    38,  # Acoustic Snare
    "hihat":    42,  # Closed Hi-Hat
    "crash":    49,  # Crash Cymbal 1
    "ride":     51,  # Ride Cymbal 1
    "tom_hi":   50,  # High Floor Tom
    "tom_mid":  47,  # Low-Mid Tom
    "tom_lo":   41,  # Low Floor Tom
}

VELOCITY_DEFAULT = 100


# ---------------------------------------------------------------------------
# Stem separation
# ---------------------------------------------------------------------------

def separate_stems(audio_path: str, output_dir: str, update_cb=None) -> dict:
    """
    Run Demucs htdemucs model to separate audio into 4 stems.
    Returns dict: {stem_name: wav_path}
    """
    import subprocess, glob

    if update_cb:
        update_cb("stem_separation", "running", "Separating stems with Demucs...")

    # Demucs writes: <output_dir>/htdemucs/<song_name>/{vocals,drums,bass,other}.wav
    cmd = [
        "python", "-m", "demucs",
        "--name", "htdemucs",
        "--out", output_dir,
        audio_path
    ]
    log.info(f"Running demucs: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    if result.returncode != 0:
        log.error(result.stderr)
        raise RuntimeError(f"Demucs failed:\n{result.stderr[-500:]}")

    # Find separated stem files
    song_name = os.path.splitext(os.path.basename(audio_path))[0]
    stem_dir = os.path.join(output_dir, "htdemucs", song_name)

    stems = {}
    for stem in ["vocals", "drums", "bass", "other"]:
        path = os.path.join(stem_dir, f"{stem}.wav")
        if os.path.exists(path):
            stems[stem] = path
        else:
            log.warning(f"Stem not found: {path}")

    if update_cb:
        update_cb("stem_separation", "done", f"Separated {len(stems)} stems")

    return stems


# ---------------------------------------------------------------------------
# Melodic stem → MIDI (via basic-pitch)
# ---------------------------------------------------------------------------

def melodic_stem_to_midi(audio_path: str, midi_path: str, stem_name: str,
                          update_cb=None) -> str:
    """Convert a melodic audio stem to MIDI using Spotify's basic-pitch."""
    import os
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

    from basic_pitch.inference import predict, Model
    from basic_pitch import ICASSP_2022_MODEL_PATH

    if update_cb:
        update_cb(stem_name, "running", f"Detecting pitches in {stem_name}...")

    log.info(f"Running basic-pitch on {stem_name}")

    # onset_threshold: lower = more notes detected; 0.5 is a good default
    # frame_threshold: minimum note duration threshold
    # minimum_note_length: in ms
    model_output, midi_data, note_events = predict(
        audio_path,
        onset_threshold=0.5,
        frame_threshold=0.3,
        minimum_note_length=58,       # ms
        minimum_frequency=None,
        maximum_frequency=None,
        multiple_pitch_bends=False,
        melodia_trick=True,
    )

    midi_data.write(midi_path)
    note_count = sum(len(inst.notes) for inst in midi_data.instruments)
    log.info(f"{stem_name}: wrote {note_count} notes → {midi_path}")

    if update_cb:
        update_cb(stem_name, "done", f"{stem_name}: {note_count} notes detected")

    return midi_path


# ---------------------------------------------------------------------------
# Drum stem → MIDI (librosa onset detection + spectral classification)
# ---------------------------------------------------------------------------

def _classify_onset(y_full, sr, onset_time, frame_len=1024):
    """
    Rough spectral classification of a drum onset into kick/snare/hihat.
    Uses spectral centroid and low-frequency energy ratio.
    """
    start = int(onset_time * sr)
    end = min(start + frame_len, len(y_full))
    frame = y_full[start:end]
    if len(frame) == 0:
        return "kick"

    # Compute FFT and frequency bins
    fft = np.abs(np.fft.rfft(frame, n=frame_len))
    freqs = np.fft.rfftfreq(frame_len, d=1.0 / sr)

    total_energy = np.sum(fft) + 1e-9
    low_energy = np.sum(fft[freqs < 200]) / total_energy    # kick territory
    mid_energy = np.sum(fft[(freqs >= 200) & (freqs < 3000)]) / total_energy  # snare
    hi_energy  = np.sum(fft[freqs >= 3000]) / total_energy  # hihat / cymbal

    if low_energy > 0.45:
        return "kick"
    elif hi_energy > 0.55:
        return "hihat"
    else:
        return "snare"


def drum_stem_to_midi(audio_path: str, midi_path: str, update_cb=None) -> str:
    """
    Detect drum onsets in the drum stem and create a MIDI drum track.
    Uses General MIDI channel 9 (drums).
    """
    if update_cb:
        update_cb("drums", "running", "Detecting drum hits...")

    log.info(f"Processing drum stem: {audio_path}")
    y, sr = librosa.load(audio_path, sr=44100, mono=True)

    # --- Onset detection ---
    onset_frames = librosa.onset.onset_detect(
        y=y, sr=sr,
        units="frames",
        hop_length=512,
        backtrack=True,
        pre_max=3, post_max=3, pre_avg=10, post_avg=10,
        delta=0.07, wait=8,
    )
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=512)
    log.info(f"Detected {len(onset_times)} drum onsets")

    # --- Estimate tempo for quantization reference ---
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    if hasattr(tempo, '__len__'):
        tempo = float(tempo[0])
    else:
        tempo = float(tempo)
    tempo = max(60.0, min(200.0, tempo))
    log.info(f"Estimated tempo: {tempo:.1f} BPM")

    # --- Classify each onset ---
    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    drum_inst = pretty_midi.Instrument(program=0, is_drum=True, name="Drums")

    note_duration = 0.05  # 50ms — short drum hits

    for t in onset_times:
        drum_type = _classify_onset(y, sr, t)
        pitch = DRUM_NOTES[drum_type]
        note = pretty_midi.Note(
            velocity=VELOCITY_DEFAULT,
            pitch=pitch,
            start=float(t),
            end=float(t) + note_duration,
        )
        drum_inst.notes.append(note)

    midi.instruments.append(drum_inst)
    midi.write(midi_path)
    log.info(f"Drums: wrote {len(drum_inst.notes)} hits → {midi_path}")

    if update_cb:
        update_cb("drums", "done", f"Drums: {len(drum_inst.notes)} hits detected")

    return midi_path


# ---------------------------------------------------------------------------
# Main pipeline entry point
# ---------------------------------------------------------------------------

def process_audio(audio_path: str, job_dir: str, update_cb=None) -> dict:
    """
    Full pipeline: audio → stems → MIDI files.
    Returns dict with paths to all generated MIDI files.
    """
    stems_dir = os.path.join(job_dir, "stems")
    midi_dir  = os.path.join(job_dir, "midi")
    os.makedirs(stems_dir, exist_ok=True)
    os.makedirs(midi_dir,  exist_ok=True)

    results = {}

    # 1. Separate stems
    stems = separate_stems(audio_path, stems_dir, update_cb=update_cb)

    # 2. Convert each stem
    for stem_name, stem_path in stems.items():
        midi_path = os.path.join(midi_dir, f"{stem_name}.mid")
        try:
            if stem_name == "drums":
                drum_stem_to_midi(stem_path, midi_path, update_cb=update_cb)
            else:
                melodic_stem_to_midi(stem_path, midi_path, stem_name, update_cb=update_cb)
            results[stem_name] = midi_path
        except Exception as e:
            log.error(f"Failed to convert {stem_name}: {e}")
            if update_cb:
                update_cb(stem_name, "error", str(e))

    return results

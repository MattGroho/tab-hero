"""
Feature extraction from .tab files for time-domain analysis.

Extracts de-identified features that can be rejoined via content_hash.
Designed for downstream ML analysis without exposing source audio.
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

from tab_hero.dataio.tab_format import TabData, load_tab
from tab_hero.dataio.tokenizer import ChartTokenizer


def _entropy(probs: np.ndarray) -> float:
    """Compute Shannon entropy of probability distribution."""
    probs = probs[probs > 0]  # Filter zeros to avoid log(0)
    return float(-np.sum(probs * np.log(probs)))


@dataclass
class TabFeatures:
    """De-identified features extracted from a .tab file."""

    # Key for rejoining with .tab files
    content_hash: str

    # Metadata (de-identified)
    difficulty_id: int
    instrument_id: int

    # Time-domain features (from mel spectrogram)
    duration_sec: float
    rms_energy_mean: float
    rms_energy_std: float
    amplitude_envelope_min: float
    amplitude_envelope_max: float
    amplitude_envelope_range: float
    tempo_bpm: float

    # Note-based features
    n_notes: int
    notes_per_second_mean: float
    inter_note_ms_mean: float
    inter_note_ms_std: float
    inter_note_ms_median: float
    inter_note_ms_p25: float
    inter_note_ms_p75: float

    # Note characteristics
    sustain_ratio: float
    chord_ratio: float
    fret_entropy: float
    hopo_ratio: float
    star_power_ratio: float
    tap_ratio: float

    # Spectral summary from mel
    mel_rms_mean: float
    mel_rms_std: float
    spectral_centroid_mean: float

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return asdict(self)


def estimate_tempo_from_mel(
    mel: np.ndarray,
    sample_rate: int = 22050,
    hop_length: int = 256,
) -> float:
    """Estimate tempo from mel spectrogram using onset strength envelope."""
    try:
        import warnings
        import librosa

        # Convert log-mel back to linear scale
        mel_linear = np.exp(mel.astype(np.float32))

        # Compute onset strength envelope
        onset_env = librosa.onset.onset_strength(
            S=mel_linear,
            sr=sample_rate,
            hop_length=hop_length,
        )

        # Estimate tempo (suppress deprecation warning for older librosa)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            if hasattr(librosa.feature, "rhythm"):
                tempo = librosa.feature.rhythm.tempo(
                    onset_envelope=onset_env,
                    sr=sample_rate,
                    hop_length=hop_length,
                )
            else:
                tempo = librosa.beat.tempo(
                    onset_envelope=onset_env,
                    sr=sample_rate,
                    hop_length=hop_length,
                )
        return float(tempo[0]) if len(tempo) > 0 else 0.0

    except Exception:
        return 0.0


def compute_spectral_centroid_from_mel(
    mel: np.ndarray,
    n_mels: int = 128,
) -> float:
    """Approximate spectral centroid from mel bands."""
    # Convert log-mel to linear
    mel_linear = np.exp(mel.astype(np.float32))

    # Frequency bin indices (normalized)
    freq_bins = np.arange(n_mels).reshape(-1, 1)

    # Weighted mean frequency per frame
    total_energy = mel_linear.sum(axis=0) + 1e-10
    centroids = (freq_bins * mel_linear).sum(axis=0) / total_energy

    return float(np.mean(centroids))


def extract_features(data: TabData, tokenizer: Optional[ChartTokenizer] = None) -> TabFeatures:
    """Extract time-domain and note-based features from TabData."""
    if tokenizer is None:
        tokenizer = ChartTokenizer()

    mel = data.mel_spectrogram  # (n_mels, n_frames)
    n_mels, n_frames = mel.shape
    frame_rate = data.sample_rate / data.hop_length
    duration_sec = n_frames / frame_rate

    # Convert log-mel to linear for energy calculations
    mel_linear = np.exp(mel.astype(np.float32))

    # RMS energy per frame (sum across mel bands)
    frame_energy = np.sqrt(np.mean(mel_linear ** 2, axis=0))
    rms_mean = float(np.mean(frame_energy))
    rms_std = float(np.std(frame_energy))

    # Amplitude envelope statistics
    amp_min = float(np.min(frame_energy))
    amp_max = float(np.max(frame_energy))
    amp_range = amp_max - amp_min

    # Tempo estimation
    tempo = estimate_tempo_from_mel(mel, data.sample_rate, data.hop_length)

    # Spectral centroid
    spectral_centroid = compute_spectral_centroid_from_mel(mel, n_mels)

    # Decode notes from tokens
    tokens = data.note_tokens.tolist()
    notes = tokenizer.decode_tokens(tokens)
    n_notes = len(notes)

    # Note timing features
    if n_notes > 0:
        notes_per_sec = n_notes / duration_sec if duration_sec > 0 else 0.0

        # Inter-note intervals
        timestamps = [n.timestamp_ms for n in notes]
        if len(timestamps) > 1:
            inter_note = np.diff(timestamps)
            inter_mean = float(np.mean(inter_note))
            inter_std = float(np.std(inter_note))
            inter_median = float(np.median(inter_note))
            inter_p25 = float(np.percentile(inter_note, 25))
            inter_p75 = float(np.percentile(inter_note, 75))
        else:
            inter_mean = inter_std = inter_median = inter_p25 = inter_p75 = 0.0
    else:
        notes_per_sec = 0.0
        inter_mean = inter_std = inter_median = inter_p25 = inter_p75 = 0.0

    # Note characteristics (will continue in next edit)
    sustain_ratio = chord_ratio = fret_entropy_val = 0.0
    hopo_ratio = star_power_ratio = tap_ratio = 0.0

    if n_notes > 0:
        durations = [n.duration_ms for n in notes]
        sustain_ratio = sum(1 for d in durations if d > 100) / n_notes

        chord_ratio = sum(1 for n in notes if len(n.frets) > 1) / n_notes
        hopo_ratio = sum(1 for n in notes if n.is_hopo) / n_notes
        tap_ratio = sum(1 for n in notes if n.is_tap) / n_notes
        star_power_ratio = sum(1 for n in notes if n.is_star_power) / n_notes

        # Fret distribution entropy
        all_frets = [f for n in notes for f in n.frets]
        if all_frets:
            fret_counts = np.bincount(all_frets, minlength=7)
            fret_probs = fret_counts / fret_counts.sum()
            fret_entropy_val = _entropy(fret_probs + 1e-10)

    return TabFeatures(
        content_hash=data.content_hash,
        difficulty_id=data.difficulty_id,
        instrument_id=data.instrument_id,
        duration_sec=duration_sec,
        rms_energy_mean=rms_mean,
        rms_energy_std=rms_std,
        amplitude_envelope_min=amp_min,
        amplitude_envelope_max=amp_max,
        amplitude_envelope_range=amp_range,
        tempo_bpm=tempo,
        n_notes=n_notes,
        notes_per_second_mean=notes_per_sec,
        inter_note_ms_mean=inter_mean,
        inter_note_ms_std=inter_std,
        inter_note_ms_median=inter_median,
        inter_note_ms_p25=inter_p25,
        inter_note_ms_p75=inter_p75,
        sustain_ratio=sustain_ratio,
        chord_ratio=chord_ratio,
        fret_entropy=fret_entropy_val,
        hopo_ratio=hopo_ratio,
        star_power_ratio=star_power_ratio,
        tap_ratio=tap_ratio,
        mel_rms_mean=rms_mean,
        mel_rms_std=rms_std,
        spectral_centroid_mean=spectral_centroid,
    )


def extract_features_from_file(path: Path, tokenizer: Optional[ChartTokenizer] = None) -> TabFeatures:
    """Load .tab file and extract features."""
    data = load_tab(path)
    return extract_features(data, tokenizer)


def extract_features_batch(
    paths: List[Path],
    tokenizer: Optional[ChartTokenizer] = None,
    progress: bool = True,
) -> List[TabFeatures]:
    """Extract features from multiple .tab files."""
    if tokenizer is None:
        tokenizer = ChartTokenizer()

    results = []
    iterator = paths
    if progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(paths, desc="Extracting features")
        except ImportError:
            pass

    for path in iterator:
        try:
            features = extract_features_from_file(path, tokenizer)
            results.append(features)
        except Exception as e:
            print(f"Error processing {path}: {e}")
            continue

    return results


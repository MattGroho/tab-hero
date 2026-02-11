"""Feature extraction from .tab files."""

import multiprocessing as mp
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from tqdm import tqdm

from tab_hero.dataio.tab_format import TabData, load_tab
from tab_hero.dataio.tokenizer import ChartTokenizer


def _entropy(probs: np.ndarray) -> float:
    """Shannon entropy of a probability distribution."""
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log(probs)))


@dataclass
class TabFeatures:
    """Features extracted from a .tab file, keyed by content_hash."""

    content_hash: str
    difficulty_id: int
    instrument_id: int
    song_id: int

    # Audio features
    duration_sec: float
    rms_energy_mean: float
    rms_energy_std: float
    amplitude_envelope_min: float
    amplitude_envelope_max: float
    amplitude_envelope_range: float
    tempo_bpm: float
    mel_rms_mean: float
    mel_rms_std: float
    spectral_centroid_mean: float

    # Note statistics
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

    def to_dict(self) -> Dict:
        return asdict(self)


def estimate_tempo_from_mel(mel: np.ndarray, sr: int = 22050, hop: int = 256) -> float:
    """Estimate tempo (BPM) from mel spectrogram via onset strength."""
    try:
        import librosa

        mel_linear = np.exp(mel.astype(np.float32))
        onset_env = librosa.onset.onset_strength(S=mel_linear, sr=sr, hop_length=hop)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            if hasattr(librosa.feature, "rhythm"):
                tempo = librosa.feature.rhythm.tempo(onset_envelope=onset_env, sr=sr, hop_length=hop)
            else:
                tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr, hop_length=hop)

        return float(tempo[0]) if len(tempo) > 0 else 0.0
    except Exception:
        return 0.0


def compute_spectral_centroid(mel: np.ndarray, n_mels: int = 128) -> float:
    """Mean spectral centroid from mel bands."""
    mel_linear = np.exp(mel.astype(np.float32))
    freq_bins = np.arange(n_mels).reshape(-1, 1)
    total_energy = mel_linear.sum(axis=0) + 1e-10
    centroids = (freq_bins * mel_linear).sum(axis=0) / total_energy
    return float(np.mean(centroids))


def extract_features(data: TabData, tokenizer: Optional[ChartTokenizer] = None) -> TabFeatures:
    """Extract audio and note features from TabData."""
    if tokenizer is None:
        tokenizer = ChartTokenizer()

    mel = data.mel_spectrogram
    n_mels, n_frames = mel.shape
    frame_rate = data.sample_rate / data.hop_length
    duration_sec = n_frames / frame_rate

    mel_linear = np.exp(mel.astype(np.float32))
    frame_energy = np.sqrt(np.mean(mel_linear ** 2, axis=0))
    rms_mean = float(np.mean(frame_energy))
    rms_std = float(np.std(frame_energy))
    amp_min = float(np.min(frame_energy))
    amp_max = float(np.max(frame_energy))

    tempo = estimate_tempo_from_mel(mel, data.sample_rate, data.hop_length)
    spectral_centroid = compute_spectral_centroid(mel, n_mels)

    notes = tokenizer.decode_tokens(data.note_tokens.tolist())
    n_notes = len(notes)

    if n_notes > 0:
        notes_per_sec = n_notes / duration_sec if duration_sec > 0 else 0.0
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

        sustain_ratio = sum(1 for n in notes if n.duration_ms > 100) / n_notes
        chord_ratio = sum(1 for n in notes if len(n.frets) > 1) / n_notes
        hopo_ratio = sum(1 for n in notes if n.is_hopo) / n_notes
        tap_ratio = sum(1 for n in notes if n.is_tap) / n_notes
        star_power_ratio = sum(1 for n in notes if n.is_star_power) / n_notes

        all_frets = [f for n in notes for f in n.frets]
        if all_frets:
            fret_counts = np.bincount(all_frets, minlength=7)
            fret_probs = fret_counts / fret_counts.sum()
            fret_entropy_val = _entropy(fret_probs + 1e-10)
        else:
            fret_entropy_val = 0.0
    else:
        notes_per_sec = 0.0
        inter_mean = inter_std = inter_median = inter_p25 = inter_p75 = 0.0
        sustain_ratio = chord_ratio = hopo_ratio = tap_ratio = star_power_ratio = 0.0
        fret_entropy_val = 0.0

    return TabFeatures(
        content_hash=data.content_hash,
        difficulty_id=data.difficulty_id,
        instrument_id=data.instrument_id,
        song_id=data.song_id,
        duration_sec=duration_sec,
        rms_energy_mean=rms_mean,
        rms_energy_std=rms_std,
        amplitude_envelope_min=amp_min,
        amplitude_envelope_max=amp_max,
        amplitude_envelope_range=amp_max - amp_min,
        tempo_bpm=tempo,
        mel_rms_mean=rms_mean,
        mel_rms_std=rms_std,
        spectral_centroid_mean=spectral_centroid,
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
    )


def extract_features_from_file(path: Path, tokenizer: Optional[ChartTokenizer] = None) -> TabFeatures:
    """Load .tab file and extract features."""
    return extract_features(load_tab(path), tokenizer)


def _extract_single_file(path: Path) -> Optional[tuple[TabFeatures, str]]:
    """Worker for parallel extraction. Returns (features, filename)."""
    try:
        features = extract_features(load_tab(path), ChartTokenizer())
        return (features, path.name)
    except Exception as e:
        print(f"Error processing {path}: {e}")
        return None


def extract_features_batch(
    paths: List[Path],
    tokenizer: Optional[ChartTokenizer] = None,
    progress: bool = True,
    n_workers: Optional[int] = None,
) -> List[TabFeatures]:
    """Extract features from multiple .tab files."""
    if n_workers is None:
        tokenizer = tokenizer or ChartTokenizer()
        results = []
        iterator = tqdm(paths, desc="Extracting features") if progress else paths

        for path in iterator:
            try:
                results.append(extract_features(load_tab(path), tokenizer))
            except Exception as e:
                print(f"Error processing {path}: {e}")
        return results

    if n_workers == 0:
        n_workers = mp.cpu_count()

    results = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_extract_single_file, p): i for i, p in enumerate(paths)}
        iterator = tqdm(as_completed(futures), total=len(paths), desc="Extracting features") if progress else as_completed(futures)
        for future in iterator:
            result = future.result()
            if result is not None:
                results.append(result[0])  # Just the features, not filename

    return results


def extract_features_batch_incremental(
    paths: List[Path],
    output_path: Path,
    tokenizer: Optional[ChartTokenizer] = None,
    progress: bool = True,
    n_workers: Optional[int] = None,
    save_interval: int = 100,
) -> List[TabFeatures]:
    """Extract features with incremental saving for resume support.

    Writes results to JSONL file incrementally to prevent data loss.
    Each line includes source_file for resume tracking.
    """
    import json

    results: List[TabFeatures] = []
    pending_writes: List[tuple[TabFeatures, str]] = []

    def flush_to_disk():
        """Append pending results to JSONL file."""
        nonlocal pending_writes
        if not pending_writes:
            return
        with open(output_path, "a") as f:
            for feat, filename in pending_writes:
                data = feat.to_dict()
                data["source_file"] = filename
                f.write(json.dumps(data) + "\n")
        pending_writes = []

    if n_workers is None:
        # Sequential mode
        tokenizer = tokenizer or ChartTokenizer()
        iterator = tqdm(paths, desc="Extracting features") if progress else paths

        for i, path in enumerate(iterator):
            try:
                feat = extract_features(load_tab(path), tokenizer)
                results.append(feat)
                pending_writes.append((feat, path.name))

                # Save periodically
                if (i + 1) % save_interval == 0:
                    flush_to_disk()
                    if progress and hasattr(iterator, "set_postfix"):
                        iterator.set_postfix(saved=len(results))
            except Exception as e:
                print(f"Error processing {path}: {e}")

        # Final flush
        flush_to_disk()
        return results

    # Parallel mode
    if n_workers == 0:
        n_workers = mp.cpu_count()

    processed_count = 0
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_extract_single_file, p): p for p in paths}
        iterator = tqdm(as_completed(futures), total=len(paths), desc="Extracting features") if progress else as_completed(futures)

        for future in iterator:
            result = future.result()
            if result is not None:
                feat, filename = result
                results.append(feat)
                pending_writes.append((feat, filename))
                processed_count += 1

                # Save periodically
                if processed_count % save_interval == 0:
                    flush_to_disk()
                    if progress and hasattr(iterator, "set_postfix"):
                        iterator.set_postfix(saved=len(results))

    # Final flush
    flush_to_disk()
    return results


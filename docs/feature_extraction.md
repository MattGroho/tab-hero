# Feature Extraction

Tab Hero includes a feature extraction pipeline that computes per-chart summary statistics from processed `.tab` files. These features characterize each chart's audio content, note patterns, and gameplay attributes in a flat, tabular form suitable for exploratory analysis, dataset profiling, and downstream tasks like difficulty classification or genre clustering.

## Overview

Feature extraction operates on `.tab` files (not raw audio). Each `.tab` file is loaded via `tab_format.load_tab`, its mel spectrogram is analyzed for audio-level statistics, and its token sequence is decoded via `ChartTokenizer.decode_tokens` to recover `NoteEvent` objects for note-level statistics. The result is a single `TabFeatures` dataclass per file.

**Source**: `src/tab_hero/dataio/feature_extractor.py`
**CLI**: `scripts/extract_features.py`
**Visualization**: `notebooks/feature_visualization.ipynb`

## Extracted Features

### Metadata Fields

Carried directly from the `.tab` header. These identify the chart and its conditioning axes.

| Feature | Type | Description |
|---------|------|-------------|
| `content_hash` | str | 16-char hex hash identifying the `.tab` file |
| `difficulty_id` | int | 0=easy, 1=medium, 2=hard, 3=expert |
| `instrument_id` | int | 0=lead, 1=bass, 2=rhythm, 3=keys |
| `genre_id` | int | 0=unknown through 15=other (see `tab_format.GENRE_MAP`) |
| `song_id` | int | Groups all difficulty/instrument variants of the same song |

### Audio Features

Derived from the mel spectrogram stored in the `.tab` file. The mel is converted from log scale to linear (`np.exp`) before computing energy statistics.

| Feature | Type | Description |
|---------|------|-------------|
| `duration_sec` | float | Song duration computed from `n_frames / (sample_rate / hop_length)` |
| `rms_energy_mean` | float | Mean frame-level RMS energy across the mel spectrogram |
| `rms_energy_std` | float | Standard deviation of frame-level RMS energy |
| `amplitude_envelope_min` | float | Minimum frame RMS energy (quietest frame) |
| `amplitude_envelope_max` | float | Maximum frame RMS energy (loudest frame) |
| `amplitude_envelope_range` | float | `max - min` of frame RMS energy |
| `tempo_bpm` | float | Estimated tempo via librosa onset strength autocorrelation |
| `mel_rms_mean` | float | Mean mel RMS (same as `rms_energy_mean`) |
| `mel_rms_std` | float | Mel RMS std (same as `rms_energy_std`) |
| `spectral_centroid_mean` | float | Mean spectral centroid across mel bands (higher = brighter) |

**RMS energy** is computed as `sqrt(mean(mel_linear^2))` per frame, where `mel_linear = exp(mel)`.

**Tempo estimation** uses `librosa.onset.onset_strength` on the linearized mel, followed by `librosa.feature.rhythm.tempo` (or `librosa.beat.tempo` on older librosa versions). This gives a single BPM estimate per chart.

**Spectral centroid** is computed directly from mel bands as the energy-weighted mean bin index: `sum(bin_index * mel_linear) / sum(mel_linear)` per frame, then averaged across frames.

### Note Statistics

Derived by decoding the token sequence back to `NoteEvent` objects via `ChartTokenizer.decode_tokens`. Each note event carries `timestamp_ms`, `frets`, `duration_ms`, `is_hopo`, `is_tap`, and `is_star_power`.

| Feature | Type | Description |
|---------|------|-------------|
| `n_notes` | int | Total note count in the chart |
| `notes_per_second_mean` | float | `n_notes / duration_sec` |
| `inter_note_ms_mean` | float | Mean time delta between consecutive notes |
| `inter_note_ms_std` | float | Standard deviation of inter-note deltas |
| `inter_note_ms_median` | float | Median inter-note delta |
| `inter_note_ms_p25` | float | 25th percentile of inter-note deltas |
| `inter_note_ms_p75` | float | 75th percentile of inter-note deltas |

### Note Characteristics

Ratios and distributional measures computed over all notes in the chart.

| Feature | Type | Description |
|---------|------|-------------|
| `sustain_ratio` | float | Fraction of notes with `duration_ms > 100` |
| `chord_ratio` | float | Fraction of notes with more than one fret active |
| `fret_entropy` | float | Shannon entropy of the fret distribution (higher = more uniform fret usage) |
| `hopo_ratio` | float | Fraction of notes flagged as HOPO (hammer-on / pull-off) |
| `tap_ratio` | float | Fraction of notes flagged as TAP |
| `star_power_ratio` | float | Fraction of notes within a star power phrase |

**Fret entropy** is computed over 7 bins (frets 0-5 + open). A uniform distribution across all frets yields maximum entropy (~1.95); a chart using only one fret yields 0.

## Usage

### CLI Script

`scripts/extract_features.py` extracts features from all `.tab` files in a directory and writes the results to CSV (default), JSON, or JSONL.

```bash
# Extract all features to CSV
python scripts/extract_features.py --data-dir data/processed --output features.csv

# JSON output
python scripts/extract_features.py --data-dir data/processed --output features.json --format json

# Sample 500 random files
python scripts/extract_features.py --data-dir data/processed --output features.csv --sample 500

# Parallel extraction (auto-detect CPU count)
python scripts/extract_features.py --data-dir data/processed --output features.csv --workers 0

# Sequential (single-process)
python scripts/extract_features.py --data-dir data/processed --output features.csv --workers -1
```

**Key options**:

| Flag | Default | Description |
|------|---------|-------------|
| `--data-dir` | `data/processed` | Directory containing `.tab` files |
| `--output` | `features.csv` | Output file path |
| `--format` | `csv` | Output format: `csv`, `json`, or `jsonl` |
| `--sample N` | all | Extract from N randomly sampled files |
| `--seed` | 42 | Random seed for `--sample` |
| `--workers N` | 0 (auto) | `0`=auto, `-1`=sequential, `N`=N processes |
| `--save-interval` | 100 | Flush to disk every N files |
| `--no-resume` | false | Start fresh, ignore existing checkpoint |

### Resume Support

The extraction script uses incremental JSONL checkpointing. If interrupted, re-running the same command automatically resumes from where it left off by checking which files have already been processed. The checkpoint file (`*.jsonl.checkpoint`) is written every `--save-interval` files and removed after the final output is written.

### Python API

```python
from pathlib import Path
from tab_hero.dataio.feature_extractor import (
    extract_features,
    extract_features_from_file,
    extract_features_batch,
)
from tab_hero.dataio.tab_format import load_tab

# Single file
features = extract_features_from_file(Path("data/processed/a1b2c3d4.tab"))
print(features.notes_per_second_mean, features.chord_ratio)

# From already-loaded TabData
data = load_tab(Path("data/processed/a1b2c3d4.tab"))
features = extract_features(data)

# Batch extraction with progress bar
paths = list(Path("data/processed").glob("*.tab"))
all_features = extract_features_batch(paths, progress=True)

# Convert to pandas DataFrame
import pandas as pd
df = pd.DataFrame([f.to_dict() for f in all_features])
```

## Feature Behavior by Difficulty

Several extracted features exhibit clear variation across difficulty levels, which validates both the feature extraction and the dataset labeling:

- **Note density** (`notes_per_second_mean`): Increases from Easy to Expert. Expert charts are significantly denser than Easy charts.
- **Chord ratio** (`chord_ratio`): Higher difficulties use more chords. Easy charts tend toward single notes.
- **Timing variability** (`inter_note_ms_std`): Higher difficulty charts have more varied rhythmic patterns (faster bursts mixed with gaps).
- **HOPO ratio** (`hopo_ratio`): HOPO notes are rare in Easy charts and common in Expert, reflecting the gameplay mechanic's higher skill requirement.
- **Fret entropy** (`fret_entropy`): Higher difficulties use a wider spread of frets.

These patterns are visible in the visualizations in `notebooks/feature_visualization.ipynb`.

## Implementation Details

### Pipeline

1. Load `.tab` file via `tab_format.load_tab` &rarr; `TabData` (mel spectrogram + token array + metadata)
2. Compute audio features from `TabData.mel_spectrogram` (shape: `n_mels x n_frames`, log-scaled float32)
3. Decode `TabData.note_tokens` via `ChartTokenizer.decode_tokens` &rarr; list of `NoteEvent` objects
4. Compute note statistics and ratios from the decoded events
5. Return `TabFeatures` dataclass

### Parallelism

`extract_features_batch` supports multiprocessing via `ProcessPoolExecutor`. Each worker loads and processes a `.tab` file independently. Since each file is self-contained (no shared state), this scales linearly with core count. The `extract_features_batch_incremental` variant adds periodic disk flushing for crash resilience on large datasets.

### Edge Cases

- **Empty charts** (0 notes): All note statistics default to 0.
- **Single-note charts**: Inter-note statistics default to 0 (no deltas to compute).
- **Tempo estimation failure**: Returns 0.0 BPM (librosa errors are caught and suppressed).

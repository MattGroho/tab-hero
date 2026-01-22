# Data Format Documentation

## Overview

Tab Hero uses a custom `.tab` binary format for training data. This format stores preprocessed audio features and note sequences in a compact form optimized for neural network training.

The `.tab` format is designed to be **non-reconstructable** - the stored representations cannot be used to recover the original source material.

## Training Data (.tab format)

Training data is stored as `.tab` files containing:
- Mel spectrogram features (lossy audio representation)
- Tokenized note sequences (time-discretized)
- Difficulty and instrument IDs

### Non-Reconstruction Properties

The `.tab` format intentionally prevents reconstruction of source material:

1. **Audio**: Stored as mel spectrogram, a lossy transform that cannot be inverted to audio
2. **Notes**: Time-discretized to a fixed grid, original timing precision is lost
3. **Metadata**: No song names, artist names, or other identifying information
4. **Filenames**: Content-hash based, no source identification

### Directory Structure

```
data/
└── processed/
    ├── a1b2c3d4e5f6g7h8.tab
    ├── b2c3d4e5f6g7h8i9.tab
    ├── manifest.json
    └── ...
```

Files are named by content hash only.

### Binary Layout

All multi-byte values are little-endian.

```
Offset  Size    Field
0       4       Magic ("TABH")
4       2       Version (uint16)
6       1       Difficulty ID (uint8)
7       1       Instrument ID (uint8)
8       4       Sample rate (uint32)
12      4       Hop length (uint32)
16      4       N mels (uint32)
20      4       N frames (uint32)
24      4       N tokens (uint32)
28      16      Content hash (ascii, null-padded)
44      4       Compressed mel size (uint32)
48      var     Mel data (zlib compressed float16)
var     var     Token data (int16 array)
```

#### Field Descriptions

| Field | Description |
|-------|-------------|
| **Magic** | File type identifier. Always `TABH` (0x54414248). Used to validate file format. |
| **Version** | Format version for compatibility. Current version: 1. Readers reject versions higher than supported. |
| **Difficulty ID** | Chart difficulty level. Maps to: `0=easy`, `1=medium`, `2=hard`, `3=expert`. |
| **Instrument ID** | Target instrument. Maps to: `0=lead`, `1=bass`, `2=rhythm`, `3=keys`. |
| **Sample rate** | Audio sample rate in Hz. Default: 22050. Used to calculate frame timestamps. |
| **Hop length** | Samples between mel frames. Default: 256. Frame time = `hop_length / sample_rate` (~11.6ms). |
| **N mels** | Number of mel frequency bins. Default: 128. Defines mel spectrogram height. |
| **N frames** | Number of time frames in mel spectrogram. Determines audio duration: `n_frames * hop_length / sample_rate`. |
| **N tokens** | Length of the tokenized note sequence (including BOS/EOS tokens). |
| **Content hash** | 16-character hex hash derived from mel and token content. Used for deduplication and filename. |
| **Compressed mel size** | Byte length of the zlib-compressed mel data that follows. |
| **Mel data** | Zlib-compressed mel spectrogram. Original shape: `(n_mels, n_frames)` as float16, row-major order. Decompress with zlib, cast to float32 for use. |
| **Token data** | Note sequence as int16 array. Length = `n_tokens`. See Tokenization section for token vocabulary. |

#### Example File Size Calculation

For a 3-minute song at default settings:
- Duration: 180 seconds
- Frames: `180 * 22050 / 256 ≈ 15506` frames
- Mel size (uncompressed): `128 * 15506 * 2 bytes ≈ 3.8 MB`
- Mel size (compressed): ~400-600 KB typical (3-5x compression)
- Tokens: ~2000-4000 tokens for expert chart → 4-8 KB
- **Total file size: ~400-600 KB per chart**

### Preprocessing Pipeline

The preprocessing pipeline transforms raw audio and chart files into the `.tab` training format:

```
Raw Song Directory
├── song.ogg (or .mp3/.wav/.mp4/.m4a)
├── notes.chart (or notes.mid)
└── (optional) guitar.ogg, bass.ogg, keys.ogg  ← pre-separated stems
                    │
                    ▼
         ┌─────────────────────┐
         │  Source Separation  │  (HTDemucs or pre-existing stems)
         └─────────────────────┘
                    │
    ┌───────────────┼───────────────┐
    ▼               ▼               ▼
 guitar          bass            piano
  stem           stem             stem
    │               │               │
    ▼               ▼               ▼
┌─────────────────────────────────────┐
│        Mel Spectrogram Extraction   │
│  (22050 Hz, 128 mels, hop=256)      │
└─────────────────────────────────────┘
                    │
                    ▼
         ┌─────────────────────┐
         │   Chart Parsing     │  (.chart or .mid → NoteEvents)
         │  + Note Modifiers   │  (HOPO, TAP, Star Power)
         └─────────────────────┘
                    │
                    ▼
         ┌─────────────────────┐
         │    Tokenization     │  (TIME, FRET, MOD, DUR quads)
         └─────────────────────┘
                    │
                    ▼
         ┌─────────────────────┐
         │   .tab File Output  │  (zlib-compressed mel + tokens)
         └─────────────────────┘
```

#### Source Separation

By default, preprocessing uses **HTDemucs** (htdemucs_6s model) to separate mixed audio into instrument stems. Each chart receives its corresponding instrument's mel spectrogram:

| Chart Instrument | Separated Stem |
|------------------|----------------|
| lead             | guitar         |
| rhythm           | guitar         |
| bass             | bass           |
| keys             | piano          |

Pre-existing stems (common in Clone Hero/Rock Band packs) are automatically detected and used:
- `guitar.ogg`, `bass.ogg`, `keys.ogg`, `rhythm.ogg`

Silent stems (RMS < 0.01) are skipped to avoid training on empty audio.

#### Mel Spectrogram Configuration

| Parameter   | Value  | Description                      |
|-------------|--------|----------------------------------|
| sample_rate | 22050  | Audio sample rate (Hz)           |
| n_fft       | 2048   | FFT window size                  |
| hop_length  | 256    | Samples between frames (~11.6ms) |
| n_mels      | 128    | Mel frequency bins               |

Mel spectrograms are log-scaled and per-song normalized (mean=0, std=1).

#### Running the Preprocessor

```bash
# Process all songs with source separation (recommended, requires GPU)
python scripts/preprocess.py --input_dir data/raw --output_dir data/processed --workers 8

# Process without source separation (faster, for testing or CPU-only)
python scripts/preprocess.py --input_dir data/raw --output_dir data/processed --skip-separation

# Process and remove source files afterward
python scripts/preprocess.py --input_dir data/raw --output_dir data/processed --cleanup

# Process specific difficulties and instruments
python scripts/preprocess.py --input_dir data/raw --output_dir data/processed \
    --difficulties expert hard --instruments lead bass

# Filter by duration and note count
python scripts/preprocess.py --input_dir data/raw --output_dir data/processed \
    --min-notes 50 --min-duration 30 --max-duration 600
```

Key options:
- `--workers N`: Parallel processing (default: CPU count)
- `--skip-separation`: Use mixed audio instead of separated stems
- `--stem-cache-dir`: Cache separated stems to avoid reprocessing
- `--min-stem-rms`: RMS threshold for valid stems (default: 0.01)
- `--val-split`: Fraction for validation set (default: 0.05)
- `--cleanup`: Remove source files after processing

The preprocessor supports resuming interrupted runs via `.progress_manifest.json`.

#### Output Manifest

The `manifest.json` in the output directory contains:
- Train/validation split file lists
- Processing configuration (difficulties, instruments, filters)
- Mel spectrogram parameters for reproducibility
- Source separation model used

## Model Output Format

The model outputs standard rhythm game song folders:

```
output/
└── my_song/
    ├── notes.mid       # MIDI chart
    ├── song.ini        # Metadata
    ├── song.ogg        # Audio (user-provided)
    └── album.png       # Album art
```

### notes.mid

Standard MIDI format with note events:
- Expert: notes 96-100
- Hard: notes 84-88
- Medium: notes 72-76
- Easy: notes 60-64

### song.ini

```ini
[song]
name = Song Title
artist = Artist Name
charter = Tab Hero
diff_guitar = -1
```

## Tokenization

Notes are encoded as quads: `[TIME_DELTA] [FRET_TOKEN] [MODIFIER] [DURATION]`

### Token Layout (Default Config)

| Range     | Type              | Count | Description |
|-----------|-------------------|-------|-------------|
| 0         | PAD               | 1     | Padding token |
| 1         | BOS               | 1     | Beginning of sequence |
| 2         | EOS               | 1     | End of sequence |
| 3-503     | TIME_0 to TIME_500| 501   | Time delta bins (10ms resolution, max 5000ms) |
| 504-630   | FRET combinations | 127   | All non-empty fret subsets (2^7 - 1) |
| 631-638   | MOD combinations  | 8     | Note modifier combinations (see below) |
| 639-739   | DUR_0 to DUR_100  | 101   | Duration bins (50ms resolution, max 5000ms) |

**Total vocabulary size: 740 tokens**

### Fret Combinations

6 frets (indices 0-5) plus open note (index 6) = 7 elements.
All non-empty subsets: 2^7 - 1 = 127 combinations.

Examples:
- Single notes: `FRET_0`, `FRET_1`, ..., `FRET_6` (open)
- Chords: `FRET_0_1`, `FRET_0_2_4`, `FRET_1_2_3_4_5`
- Order-invariant: `[0,2,4]` and `[4,0,2]` encode to the same token

### Note Modifiers

Modifier tokens encode combinations of three boolean flags using 3 bits:

| Token    | Value | HOPO | TAP | Star Power | Description |
|----------|-------|------|-----|------------|-------------|
| MOD_NONE | 0     | -    | -   | -          | No modifiers |
| MOD_H    | 1     | X    | -   | -          | Hammer-on/Pull-off |
| MOD_T    | 2     | -    | X   | -          | Tap note |
| MOD_HT   | 3     | X    | X   | -          | HOPO + Tap |
| MOD_S    | 4     | -    | -   | X          | Star Power phrase |
| MOD_HS   | 5     | X    | -   | X          | HOPO + Star Power |
| MOD_TS   | 6     | -    | X   | X          | Tap + Star Power |
| MOD_HTS  | 7     | X    | X   | X          | All modifiers |

**Modifier definitions:**
- **HOPO (Hammer-On/Pull-Off)**: Note can be played without strumming if previous note was hit.
  - .chart files: Parsed from `N 5` events
  - MIDI files: Parsed from note `base+6` (e.g., note 102 for Expert)
- **TAP**: Note can be tapped on the fretboard without strumming.
  - .chart files: Parsed from `N 6` events
  - MIDI files: Parsed from note `base+5` (e.g., note 101 for Expert)
- **Star Power**: Note is within a star power phrase, granting bonus multiplier when hit.
  - .chart files: Parsed from `S 2` phrase markers
  - MIDI files: Parsed from note 103 (Rock Band style) or note 116 (Guitar Hero style)

### Duration Bins

- Resolution: 50ms per bin
- Range: 0-5000ms (101 bins)
- DUR_0 = 0-49ms (tap), DUR_1 = 50-99ms, ..., DUR_100 = 5000ms+ (long sustain)

### Legacy Mode (Without Modifiers)

For backward compatibility, modifiers can be disabled via `TokenizerConfig(include_modifiers=False)`.
In this mode, notes are encoded as triplets: `[TIME_DELTA] [FRET_TOKEN] [DURATION]` with vocab size 636.

# Data Format

Tab Hero uses a custom `.tab` binary format for training data. This format stores preprocessed audio features and note sequences in a compact form optimized for training. The format is intentionally **non-reconstructable** -- the stored representations cannot recover the original source material.

## Non-Reconstruction Properties

1. **Audio**: Stored as mel spectrograms, a lossy transform that discards phase information and cannot be inverted to audio.
2. **Notes**: Time-discretized to fixed bins (10ms for timing, 50ms for duration), losing original timing precision.
3. **Metadata**: No song names, artist names, or identifying information is stored.
4. **Filenames**: Content-hash based, with no link to source material.

## .tab Binary Layout

All multi-byte values are little-endian. Header is 54 bytes.

```
Offset  Size    Type      Field
0       4       bytes     Magic ("TABH")
4       2       uint16    Version
6       1       uint8     Difficulty ID
7       1       uint8     Instrument ID
8       1       uint8     Genre ID
9       1       uint8     Reserved (always 0)
10      4       uint32    Sample rate
14      4       uint32    Hop length
18      4       uint32    N mels
22      4       uint32    N frames
26      4       uint32    N tokens
30      16      ascii     Content hash (null-padded)
46      4       uint32    Song ID
50      4       uint32    Compressed mel size
54      var     bytes     Mel data (zlib-compressed float16)
var     var     int16[]   Token data
```

### Field Descriptions

| Field | Description |
|-------|-------------|
| **Magic** | `TABH` (0x54414248). Validates file format. |
| **Version** | Format version. Current: 1. Readers reject unsupported versions. |
| **Difficulty ID** | `0`=easy, `1`=medium, `2`=hard, `3`=expert |
| **Instrument ID** | `0`=lead, `1`=bass, `2`=rhythm, `3`=keys |
| **Genre ID** | `0`=unknown, `1`=rock, `2`=metal, `3`=alternative, `4`=punk, `5`=pop, `6`=electronic, `7`=indie, `8`=country, `9`=blues, `10`=jazz, `11`=classical, `12`=hiphop, `13`=reggae, `14`=folk, `15`=other |
| **Sample rate** | Audio sample rate in Hz. Default: 22050. |
| **Hop length** | Samples between mel frames. Default: 256. Frame time = hop_length / sample_rate (~11.6ms). |
| **N mels** | Mel frequency bins. Default: 128. |
| **N frames** | Time frames in the mel spectrogram. Duration = n_frames * hop_length / sample_rate. |
| **N tokens** | Token sequence length (including BOS/EOS). |
| **Content hash** | 16-character hex hash of mel + token content. Used for deduplication and as the filename. |
| **Song ID** | Groups all variants (difficulty/instrument) of the same song. Not reversible to source. |
| **Compressed mel size** | Byte length of the zlib-compressed mel data. |
| **Mel data** | Zlib-compressed mel spectrogram. Shape: `(n_mels, n_frames)` as float16, row-major. Decompress with zlib, then cast to float32. |
| **Token data** | Note sequence as int16 array. Length = n_tokens. See [architecture.md](architecture.md) for token vocabulary. |

### Example File Size

For a 3-minute song at default settings:
- Frames: 180s * 22050 / 256 = ~15,500
- Mel (uncompressed): 128 * 15,500 * 2 bytes = ~3.8 MB
- Mel (compressed): ~400-600 KB (3-5x zlib compression)
- Tokens: ~2,000-4,000 for an expert chart = 4-8 KB
- **Total: ~400-600 KB per .tab file**

### Directory Structure

```
data/processed/
  a1b2c3d4e5f6g7h8.tab
  b2c3d4e5f6g7h8i9.tab
  ...
  manifest.json
```

## Preprocessing Pipeline

The preprocessing pipeline transforms raw Clone Hero song directories into `.tab` training files.

```
Raw Song Directory
  song.ogg (or .mp3/.wav/.m4a)
  notes.chart (or notes.mid)
  (optional) guitar.ogg, bass.ogg, keys.ogg
         |
         v
  Source Separation (HTDemucs or pre-existing stems)
         |
    +---------+---------+
    v         v         v
  guitar    bass      piano
  stem      stem      stem
    |         |         |
    v         v         v
  Mel Spectrogram Extraction
  (22050 Hz, 128 mels, hop=256)
         |
         v
  Chart Parsing (.chart or .mid -> NoteEvents)
  + Note Modifiers (HOPO, TAP, Star Power)
         |
         v
  Tokenization (TIME, FRET, MOD, DUR quads)
         |
         v
  .tab File Output (zlib-compressed mel + tokens)
```

### Source Separation

By default, preprocessing uses **HTDemucs** (`htdemucs_6s` model) to separate mixed audio into instrument stems. Each chart receives its corresponding instrument's mel spectrogram:

| Chart Instrument | Separated Stem |
|------------------|----------------|
| lead             | guitar         |
| rhythm           | guitar         |
| bass             | bass           |
| keys             | piano          |

Pre-existing stems (common in Clone Hero / Rock Band packs) are detected automatically and used directly: `guitar.ogg`, `bass.ogg`, `keys.ogg`, `rhythm.ogg`.

Silent stems (RMS < 0.01) are skipped to avoid training on empty audio.

### Mel Spectrogram Configuration

| Parameter   | Value  | Description                      |
|-------------|--------|----------------------------------|
| sample_rate | 22050  | Audio sample rate (Hz)           |
| n_fft       | 2048   | FFT window size                  |
| hop_length  | 256    | Samples between frames (~11.6ms) |
| n_mels      | 128    | Mel frequency bins               |

Mel spectrograms are log-scaled and per-song normalized (mean=0, std=1).

### Running the Preprocessor

```bash
# Process all songs with source separation (recommended, requires GPU)
python scripts/preprocess.py --input_dir data/raw --output_dir data/processed --workers 8

# Without source separation (faster, CPU-only)
python scripts/preprocess.py --input_dir data/raw --output_dir data/processed --skip-separation

# Remove source files after processing
python scripts/preprocess.py --input_dir data/raw --output_dir data/processed --cleanup

# Filter by difficulty, instrument, duration, and note count
python scripts/preprocess.py --input_dir data/raw --output_dir data/processed \
    --difficulties expert hard --instruments lead bass \
    --min-notes 50 --min-duration 30 --max-duration 600
```

**Key options**:

| Flag | Description |
|------|-------------|
| `--workers N` | Parallel processing (default: CPU count) |
| `--skip-separation` | Use mixed audio instead of separated stems |
| `--stem-cache-dir` | Cache separated stems to avoid reprocessing |
| `--min-stem-rms` | RMS threshold for valid stems (default: 0.01) |
| `--val-split` | Fraction for validation set (default: 0.05) |
| `--cleanup` | Remove source files after processing |

The preprocessor supports resuming interrupted runs via `.progress_manifest.json`.

### Output Manifest

The `manifest.json` in the output directory records:

- Train/validation file lists
- Processing configuration (difficulties, instruments, filters)
- Mel spectrogram parameters
- Source separation model used

## Model Output Format

Generated charts are exported as standard Clone Hero song folders:

```
output/my_song/
  notes.mid       # MIDI chart
  song.ini        # Metadata
  song.ogg        # Audio (copied from input)
  album.png       # Album art (placeholder)
```

### notes.mid

Standard MIDI format. Note events are mapped to difficulty-specific ranges:

| Difficulty | MIDI Notes |
|------------|------------|
| Expert     | 96-100     |
| Hard       | 84-88      |
| Medium     | 72-76      |
| Easy       | 60-64      |

### song.ini

```ini
[song]
name = Song Title
artist = Artist Name
charter = Tab Hero
diff_guitar = -1
preview_start_time = 0
```

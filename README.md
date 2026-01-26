# Tab Hero

Automatic guitar chart generation from audio using an encoder-decoder transformer. Given a song, Tab Hero produces playable [Clone Hero](https://clonehero.net/) charts across multiple difficulty levels and instruments.

Developed as a Milestone project for SIADS 696 at the University of Michigan by Matthew Grohotolski, Susan Hatem, and Christopher Pleman.

## Highlights

- Generates charts for all 4 difficulty levels (Easy, Medium, Hard, Expert)
- Supports multiple instruments (Lead, Bass, Rhythm, Keys)
- Includes note modifiers (HOPO, TAP, Star Power) for gameplay authenticity
- Handles full-length songs via streaming/chunked generation
- Integrates source separation ([HTDemucs](https://github.com/facebookresearch/demucs)) to isolate instrument stems
- Stores training data as non-reconstructable mel spectrograms for data privacy

## Architecture

Tab Hero uses an encoder-decoder transformer that processes mel spectrograms and autoregressively generates note token sequences:

```
Audio File (.ogg/.mp3/.wav)
         |
         v
  Source Separation (HTDemucs, optional)
         |
         v
  Mel Spectrogram (22kHz, 128 mels, hop=256)
         |
         v
  Audio Encoder (Linear + Conv Adapter)
         |
         v
  Transformer Decoder (Cross-Attention + Difficulty/Instrument Conditioning)
         |
         v
  Note Tokens --> Chart Exporter (.mid)
```

Notes are tokenized as quads: `[TIME_DELTA] [FRET_COMBINATION] [MODIFIER] [DURATION]` with a 740-token vocabulary. See [docs/architecture.md](docs/architecture.md) for full details.

### Model Sizes

| Size  | Layers | Dim | Heads | FFN  | Params |
|-------|--------|-----|-------|------|--------|
| Small | 4      | 384 | 6     | 1536 | ~30M   |
| Base  | 6      | 512 | 8     | 2048 | ~75M   |
| Large | 8      | 768 | 12    | 3072 | ~150M  |

## Installation

**Prerequisites:** Python 3.11+, CUDA-capable GPU (recommended), ffmpeg

### Conda (recommended)

```bash
conda env create -f conda.yaml
conda activate tab-hero
pip install -e .
```

### Pip

```bash
pip install -r requirements.txt
pip install -e .
```

## Usage

### Preprocessing

Transform raw Clone Hero song directories into `.tab` training files:

```bash
# With source separation (recommended, requires GPU)
python scripts/preprocess.py \
    --input_dir data/raw \
    --output_dir data/processed \
    --workers 8

# Without source separation (CPU-friendly)
python scripts/preprocess.py \
    --input_dir data/raw \
    --output_dir data/processed \
    --skip-separation

# Filter by difficulty, instrument, duration, or note count
python scripts/preprocess.py \
    --input_dir data/raw \
    --output_dir data/processed \
    --difficulties expert hard \
    --instruments lead bass \
    --min-notes 50 \
    --min-duration 30 \
    --max-duration 600
```

The preprocessor supports resuming interrupted runs and caching separated stems. See [docs/data_format.md](docs/data_format.md) for the `.tab` binary format specification.

### Training

Training is configured via [Hydra](https://hydra.cc/). Configuration files live in `configs/`.

```bash
# Default (Large model, as specified in configs/default.yaml)
python scripts/train.py

# Select a model size
python scripts/train.py +model=small
python scripts/train.py +model=large

# Override parameters
python scripts/train.py data.batch_size=32 training.learning_rate=5e-5
```

Key training features:
- Mixed precision (bf16) with gradient accumulation
- Cosine annealing + warmup learning rate schedule
- Early stopping and top-k checkpoint management

### Generation

Generate a Clone Hero chart from any audio file:

```bash
# Basic usage
python scripts/generate.py --audio song.ogg --output ./output/MySong

# Specify difficulty, instrument, and sampling parameters
python scripts/generate.py \
    --audio song.mp3 \
    --output ./output/MySong \
    --difficulty hard \
    --instrument bass \
    --temperature 0.8

# With metadata and custom checkpoint
python scripts/generate.py \
    --audio song.ogg \
    --output ./output/MySong \
    --song-name "My Song" \
    --artist "My Artist" \
    --model checkpoints/best_model.pt
```

Output is a Clone Hero-compatible song folder:

```
output/MySong/
  notes.mid       # MIDI chart
  song.ini        # Metadata
  song.ogg        # Audio (copied from input)
  album.png       # Album art
```

## Project Structure

```
tab-hero/
  src/tab_hero/
    model/            # Encoder-decoder transformer architecture
    dataio/           # Data loading, preprocessing, tokenization, chart parsing
    training/         # Training loop, loss functions
    inference/        # Generation pipeline, chart export
  scripts/            # CLI entry points (train, generate, preprocess)
  configs/            # Hydra configuration (default + model size variants)
  docs/               # Architecture, data format, and research documentation
  data/
    raw/              # Clone Hero song directories (input)
    processed/        # .tab training files + manifest.json (output)
```

## Data

### Input Format

Standard Clone Hero song directories:

```
song_directory/
  song.ogg            # Audio (ogg/mp3/wav/m4a)
  notes.chart         # Chart (or notes.mid)
  song.ini            # Metadata (name, artist, genre)
```

### Training Format

The custom `.tab` binary format stores compressed mel spectrograms and tokenized note sequences. Files are named by content hash and contain no identifying metadata, ensuring training data cannot be used to reconstruct original audio. See [docs/data_format.md](docs/data_format.md) for the full specification.

### Dataset

- ~6,000 song-difficulty-instrument combinations from Clone Hero community charts
- 95/5 train/validation split

## Documentation

- [Architecture](docs/architecture.md) - Model design, encoder/decoder details, tokenization
- [Data Format](docs/data_format.md) - `.tab` binary format specification, preprocessing pipeline
- [Feature Extraction](docs/feature_extraction.md) - Per-chart feature extraction from `.tab` files
- [Research Notes](docs/research_notes.md) - Design decisions, related work, evaluation metrics

## License

MIT

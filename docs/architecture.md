# Architecture

Tab Hero uses an encoder-decoder transformer to generate guitar charts from audio. The system processes audio into mel spectrograms (a lossy, non-invertible representation), then autoregressively generates note tokens conditioned on difficulty and instrument.

## System Overview

```
Audio File (.ogg/.mp3/.wav)
         |
         v
  Source Separation (HTDemucs, optional)
  guitar / bass / piano stems
         |
         v
  Mel Spectrogram (22050 Hz, 128 mels, hop=256)
         |
         v  (batch, n_frames, 128)
  Audio Encoder
  Linear projection + Conv1D stack with downsampling
         |
         v  (batch, n_frames / downsample, encoder_dim)
  Encoder Projection (if encoder_dim != decoder_dim)
         |
         v  (batch, n_frames', decoder_dim)
  Transformer Decoder
  Causal self-attention (RoPE + Flash Attention)
  Cross-attention to encoder output
  Difficulty + Instrument conditioning
         |
         v  (batch, seq_len, vocab_size)
  Chart Exporter (.mid / .ini)
```

## Audio Processing

- **Source separation** (optional): HTDemucs (`htdemucs_6s` model) separates mixed audio into guitar, bass, and piano stems. Each chart receives the mel spectrogram of its corresponding instrument stem.
- **Mel spectrogram**: 22050 Hz sample rate, 128 mel bins, 256 hop length (~11.6ms per frame), log-scaled, per-song normalized (mean=0, std=1).
- Mel spectrograms are lossy and cannot be inverted to recover the original audio.

## Audio Encoder

The `AudioEncoder` projects mel spectrogram frames from 128 dimensions to the transformer hidden dimension, then applies temporal downsampling via a convolutional stack.

**Pipeline**: Linear projection + LayerNorm &rarr; Conv1D stack &rarr; LayerNorm

**Downsampling**: Configurable 2x, 4x, or 8x reduction through cascaded stride-2 convolutions. Default is 4x (two stride-2 layers), which maps ~11.6ms mel frames to ~46.4ms encoder frames.

**Convolution details**:
- First layer: standard Conv1D (kernel=5, stride=2)
- Subsequent layers: depthwise separable convolutions (depthwise Conv1D + pointwise Conv1D) for parameter efficiency
- GELU activations with GroupNorm(8) after each conv layer
- Final refinement Conv1D (kernel=3, stride=1, no downsampling)

**Weight initialization**: Kaiming normal for convolutions, truncated normal (std=0.02) for linear layers.

## Transformer Decoder

The `ChartDecoder` generates note tokens autoregressively, attending to both its own output history (causal self-attention) and the encoder output (cross-attention).

### Decoder Block

Each `DecoderBlock` follows a pre-norm architecture:

1. **RMSNorm** &rarr; **Causal self-attention** (with RoPE) &rarr; residual + dropout
2. **RMSNorm** &rarr; **Cross-attention** to encoder output &rarr; residual + dropout
3. **RMSNorm** &rarr; **SwiGLU feed-forward** &rarr; residual + dropout

### Positional Encoding

**RoPE (Rotary Position Embeddings)** is the default and recommended mode. RoPE encodes relative position information directly into the attention computation, enabling the model to generate sequences longer than those seen during training. The RoPE cache extends dynamically at inference time.

Learned absolute position embeddings are available as a fallback (`use_rope: false`), but these are limited to `max_seq_len` positions.

### Attention

**Flash Attention 2** is used by default via PyTorch's `scaled_dot_product_attention`, which automatically selects the best backend (Flash Attention, memory-efficient, or math) based on hardware. This reduces attention memory from O(n^2) to O(n).

Self-attention uses a causal mask. Cross-attention supports optional encoder padding masks.

### Feed-Forward Network

Uses **SwiGLU** (gated linear unit with SiLU activation):

```
FFN(x) = W2(SiLU(W1(x)) * W3(x))
```

Three linear projections with no bias: `dim` &rarr; `ffn_dim` (gate + value), then `ffn_dim` &rarr; `dim`.

### Conditioning

Difficulty (4 levels) and instrument (4 types) are each represented as learned embeddings of size `decoder_dim`. These are added to the token embeddings at each decoder position, allowing a single model to generate charts for any difficulty/instrument combination.

### Output Projection

The output projection is **weight-tied** with the token embedding matrix: `output_proj.weight = token_embedding.weight`. This reduces parameters and improves training stability.

### Gradient Checkpointing

When enabled, decoder layers use `torch.utils.checkpoint` to trade compute for memory during training. This allows training larger models or longer sequences within the same VRAM budget.

## Tokenization

Each note is encoded as a quad: `[TIME_DELTA] [FRET_COMBINATION] [MODIFIER] [DURATION]`

Full sequences are wrapped with special tokens: `[BOS] [T1] [F1] [M1] [D1] [T2] [F2] [M2] [D2] ... [EOS]`

### Vocabulary (740 tokens)

| Range   | Type            | Count | Description                                  |
|---------|-----------------|-------|----------------------------------------------|
| 0       | PAD             | 1     | Padding token                                |
| 1       | BOS             | 1     | Beginning of sequence                        |
| 2       | EOS             | 1     | End of sequence                              |
| 3-503   | TIME_DELTA      | 501   | Time since previous note (10ms bins, 0-5000ms) |
| 504-630 | FRET            | 127   | All non-empty subsets of 7 frets (2^7 - 1)   |
| 631-638 | MODIFIER        | 8     | Combinations of HOPO, TAP, Star Power        |
| 639-739 | DURATION        | 101   | Note sustain length (50ms bins, 0-5000ms)    |

### Fret Combinations

6 frets (indices 0-5) plus open note (index 6) = 7 elements. All 127 non-empty subsets are enumerated, so single notes and chords of any size map to a single token. The encoding is order-invariant: frets `[0, 2, 4]` and `[4, 0, 2]` produce the same token.

### Note Modifiers

Three boolean flags encoded as a 3-bit field (8 combinations):

| Token    | HOPO | TAP | Star Power | Description            |
|----------|------|-----|------------|------------------------|
| MOD_NONE | -    | -   | -          | No modifiers           |
| MOD_H    | X    | -   | -          | Hammer-on / Pull-off   |
| MOD_T    | -    | X   | -          | Tap note               |
| MOD_HT   | X    | X   | -          | HOPO + Tap             |
| MOD_S    | -    | -   | X          | Star Power phrase      |
| MOD_HS   | X    | -   | X          | HOPO + Star Power      |
| MOD_TS   | -    | X   | X          | Tap + Star Power       |
| MOD_HTS  | X    | X   | X          | All modifiers          |

**Parsing sources**:
- **HOPO**: `.chart` `N 5` events / MIDI note `base+6`
- **TAP**: `.chart` `N 6` events / MIDI note `base+5`
- **Star Power**: `.chart` `S 2` phrases / MIDI notes 103 or 116

### Legacy Mode

Modifiers can be disabled via `TokenizerConfig(include_modifiers=False)`, reducing notes to triplets `[TIME_DELTA] [FRET] [DURATION]` with a 636-token vocabulary.

## Model Sizes

| Size  | Layers | Dim | Heads | FFN  | Params |
|-------|--------|-----|-------|------|--------|
| Small | 4      | 384 | 6     | 1536 | ~30M   |
| Base  | 6      | 512 | 8     | 2048 | ~75M   |
| Large | 8      | 768 | 12    | 3072 | ~150M  |

Configuration files are in `configs/model/`. The default config (`configs/default.yaml`) uses Large-equivalent settings (768 dim, 8 layers, 12 heads).

## Training

- **Loss**: Cross-entropy with PAD token ignored (`ignore_index=0`). The standalone `ChartLoss` class supports pad downweighting (weight=0.1) and optional label smoothing.
- **Optimizer**: AdamW with weight decay=0.01, betas=(0.9, 0.95)
- **Scheduler**: Cosine annealing with eta_min = lr * 0.01 (per-epoch). OneCycleLR is available as an alternative.
- **Warmup**: Linear warmup over configurable steps (default: 1000)
- **Mixed precision**: bf16-mixed by default (no loss scaling needed). Falls back to fp16 with GradScaler on hardware without bf16 support.
- **Gradient accumulation**: Configurable steps to simulate larger batch sizes (default effective batch = batch_size * accumulation_steps)
- **Gradient clipping**: Max norm 1.0
- **Checkpointing**: Top-k checkpoints by validation loss (default: keep 5). Always saves `best_model.pt` and `last_model.pt`.
- **Early stopping**: Configurable patience (default: 15 epochs without improvement)
- **Experiment tracking**: Optional Weights & Biases integration

## Inference

### Standard Generation

Autoregressive token-by-token generation with KV caching for O(1) per-step cost. The audio is encoded once, then the decoder generates tokens until EOS or a maximum length estimated from audio duration:

- **Constrained decoding** (default, via `vocab_ranges`): 12 tokens/sec (3 notes/sec × 4 tokens/note), capped at 32,768 tokens. No budget is wasted on structurally invalid tokens.
- **Unconstrained decoding**: 16 tokens/sec (4 notes/sec × 4 tokens/note × ~5x waste factor to account for malformed quads), capped at 32,768 tokens.

`ChartGenerator` always passes `vocab_ranges`, so constrained decoding is the default in practice.

**Sampling options**:
- Temperature scaling
- Top-k filtering
- Top-p (nucleus) filtering

### Streaming Generation

For songs longer than `chunk_size` frames, `generate_streaming` processes the audio in overlapping chunks. Each chunk is encoded independently, and the decoder carries forward a context window from the previous chunk. RoPE position offsets maintain coherent position encoding across chunk boundaries.

### Export

The `SongExporter` converts generated note events into a Clone Hero-compatible song folder:
- `notes.mid`: Standard MIDI with note events mapped to difficulty-specific ranges (Easy: 60-64, Medium: 72-76, Hard: 84-88, Expert: 96-100)
- `song.ini`: Metadata (name, artist, charter)
- `album.png`: Placeholder album art
- Audio file copied from input

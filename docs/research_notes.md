# Research Notes

Design decisions, related work, and evaluation strategy for Tab Hero.

## Related Work

### Audio-to-MIDI Transcription

1. **Basic Pitch** (Spotify, 2022). A lightweight neural network for polyphonic pitch detection. Good for melody extraction but not tailored to rhythm-game chart generation. [github.com/spotify/basic-pitch](https://github.com/spotify/basic-pitch)

2. **Onsets and Frames** (Magenta, 2018). A piano transcription model with a dual-head architecture for onset detection and frame prediction. Informs our approach to note onset timing.

3. **MT3** (Google, 2022). Multi-instrument transcription using a T5-style encoder-decoder transformer with tokenized MIDI events. The closest architectural precedent to Tab Hero's approach.

### Audio Representations

1. **Encodec** (Meta, 2022). A neural audio codec producing discrete tokens. The 24kHz model uses 8 codebooks with 1024 codes each at a 75 Hz frame rate (~13.3ms). High-quality compression but fully reconstructable.

2. **Jukebox** (OpenAI, 2020). A VQ-VAE for music generation with hierarchical tokens at different time scales.

3. **MusicGen** (Meta, 2023). A transformer decoder over Encodec tokens for text-to-music generation. Uses a delay pattern for codebook interleaving.

### Rhythm Game Chart Generation

1. **StepMania AI** (Various). A collection of rule-based and ML approaches for DDR step charts, with focus on difficulty scaling and playability.

2. **Dance Dance Convolution** (Donahue et al., 2017). A CNN-based step placement model for DDR that maps audio features to step timing.

## Key Design Decisions

### Audio Representation: Mel Spectrograms

We chose mel spectrograms over Encodec embeddings. The primary constraint was **non-reconstructability**: training data must not be invertible to original audio.

| Aspect          | Mel Spectrogram    | Encodec            |
|-----------------|--------------------|--------------------|
| Reconstructable | No (phase lost)    | Yes (neural codec) |
| Frame rate      | 86 Hz (11.6ms)     | 75 Hz (13.3ms)     |
| Dimension       | 128 (n_mels)       | 128 (encoder dim)  |
| Pretrained      | No                 | Yes                |

Encodec embeddings can be decoded back to audio, violating the non-reconstructability requirement. Mel spectrograms discard phase information during the STFT, making inversion impossible without the original phase.

### Time Resolution

- **Mel spectrogram**: 86 Hz (hop_length=256 at 22050 Hz) = 11.6ms per frame
- **Time delta tokens**: 10ms resolution, max 5000ms (501 bins)
- **Duration tokens**: 50ms resolution, max 5000ms (101 bins)

The 11.6ms frame rate provides sufficient temporal precision for note onset detection. The 10ms token resolution allows sub-frame precision for timing, while the coarser 50ms duration resolution is adequate for sustain lengths.

### Tokenization: Quad Format with Modifiers

Each note is encoded as a quad: `[TIME_DELTA] [FRET_COMBINATION] [MODIFIER] [DURATION]`

| Component        | Tokens | Description                                       |
|------------------|--------|---------------------------------------------------|
| TIME_DELTA       | 501    | Delta from previous note in 10ms bins (0-5000ms)  |
| FRET_COMBINATION | 127    | All non-empty subsets of 7 elements (6 frets + open) |
| MODIFIER         | 8      | Combinations of HOPO, TAP, Star Power flags       |
| DURATION         | 101    | Sustain length in 50ms bins (0-5000ms)            |
| Special          | 3      | PAD, BOS, EOS                                     |
| **Total**        | **740**|                                                   |

**Rejected alternatives**:
- Fret-only encoding (no explicit timing/duration): insufficient for accurate chart generation
- Event-based encoding (NOTE_ON/OFF pairs): produces longer sequences and adds complexity without clear benefit

### Modifier Tokens

Modifier tokens encode three gameplay-critical boolean flags as a 3-bit field:

- **HOPO** (bit 0): Hammer-on/pull-off. Parsed from `.chart` `N 5` events or MIDI note `base+6`.
- **TAP** (bit 1): Tap notes. Parsed from `.chart` `N 6` events or MIDI note `base+5`.
- **STAR_POWER** (bit 2): Star power phrases. Parsed from `.chart` `S 2` markers or MIDI notes 103/116.

Including modifiers adds only 8 tokens to the vocabulary while capturing attributes that significantly affect gameplay feel.

### Difficulty and Instrument Conditioning

Both difficulty (Easy/Medium/Hard/Expert) and instrument (Lead/Bass/Rhythm/Keys) are represented as learned embedding vectors added to each decoder position. This allows a single model to handle all combinations, sharing audio understanding across conditions.

The alternative of separate decoder heads or separate models per difficulty/instrument was rejected for parameter efficiency. The bulk of the model's knowledge (audio understanding, timing, musical structure) is shared across all conditions.

### Positional Encoding: RoPE

RoPE (Rotary Position Embeddings) was chosen over learned absolute embeddings because it enables **length extrapolation**: the model can generate sequences longer than those seen during training. This is critical for handling full-length songs (5+ minutes) when training on shorter chunks.

RoPE also supports position offsets for streaming generation, where audio is processed in overlapping chunks with coherent position encoding across chunk boundaries.

## Evaluation Metrics

1. **Token Accuracy**: Exact match of predicted vs. ground-truth tokens
2. **Note F1**: Precision and recall of note events (timing + fret match)
3. **Onset F1**: Timing accuracy within a tolerance window
4. **Playability Score**: Human evaluation of generated chart quality (flow, difficulty appropriateness, musical alignment)

## Dataset

**Source**: Clone Hero community-created custom songs.

| Statistic | Value |
|-----------|-------|
| Songs scanned | 6,892 |
| Successfully processed | ~6,000 (varies by difficulty/instrument availability) |
| Skipped (missing difficulty/instrument) | ~800 |
| Errors (corrupt files) | ~5 |
| Format | .chart/.mid files + .ogg/.mp3/.opus audio |

The dataset covers a broad range of genres, tempos, and difficulty levels from the Clone Hero community.

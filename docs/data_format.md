# Data Format Specification

## .tab Binary Format

The `.tab` format stores preprocessed training data with audio features and tokenized charts.

### File Structure

```
Header (16 bytes):
    Magic:   4 bytes  "TABH"
    Version: 2 bytes  uint16
    Flags:   2 bytes  reserved
    Hash:    8 bytes  content hash

Payload (zlib compressed):
    mel_len:    4 bytes  uint32
    mel_data:   N bytes  float32 array
    token_len:  4 bytes  uint32
    token_data: M bytes  int32 array
    meta_len:   4 bytes  uint32
    meta_data:  K bytes  JSON utf-8
```

### Metadata Fields

| Field | Type | Description |
|-------|------|-------------|
| instrument | string | "lead", "bass", "rhythm", "keys" |
| difficulty | string | "easy", "medium", "hard", "expert" |
| source_audio | string | Original audio filename |
| source_chart | string | Original chart filename |
| duration_ms | float | Total duration in milliseconds |
| mel_shape | [int, int] | Shape of mel spectrogram |
| num_tokens | int | Number of tokens |

### Audio Features

- Sample rate: 22050 Hz
- Mel bins: 128
- Hop length: 512 samples
- Frame rate: ~43 fps
- Log-mel spectrogram

### Token Format

Sequence: `[BOS, time, frets, duration, time, frets, duration, ..., EOS]`

- BOS = 1, EOS = 2, PAD = 0
- Time deltas: 10ms resolution, max 5000ms
- Durations: 50ms resolution, max 2000ms
- Frets: all combinations of 0-4

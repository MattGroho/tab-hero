"""
.tab binary format for training data.

This format is designed EXCLUSIVELY for training neural network weights.
It stores preprocessed, lossy representations that CANNOT be used to
reconstruct the original source material.

Non-reconstructable by design:
- Audio is stored as mel spectrogram (lossy, non-invertible transform)
- Notes are stored as token IDs (time-discretized, no original timing)
- No metadata, filenames, or identifying information is preserved
- Files are identified only by content hash

The format is intentionally one-way: source -> .tab is possible,
but .tab -> source is not.

Binary Layout:
    4 bytes:  magic (TABH)
    2 bytes:  version (uint16)
    2 bytes:  difficulty_id + instrument_id (uint8 each)
    4 bytes:  sample_rate (uint32)
    4 bytes:  hop_length (uint32)
    4 bytes:  n_mels (uint32)
    4 bytes:  n_frames (uint32)
    4 bytes:  n_tokens (uint32)
    16 bytes: content_hash (ascii)
    4 bytes:  compressed mel size (uint32)
    variable: zlib compressed mel data (float16)
    variable: token data (int16)
"""

import struct
import hashlib
import zlib

import numpy as np
from dataclasses import dataclass
from pathlib import Path


# Format version
TAB_FORMAT_VERSION = 1

# Magic bytes for file identification
TAB_MAGIC = b"TABH"

# zlib compression level (6 is default, good balance)
ZLIB_COMPRESSION_LEVEL = 6


@dataclass
class TabData:
    """
    Container for .tab file contents.

    This is a training-only format. The stored representations are:
    - Lossy (mel spectrogram cannot be inverted to audio)
    - Time-discretized (original note timing is quantized)
    - Anonymous (no source identification possible)
    """

    # Audio features (mel spectrogram, not raw audio)
    # This is a lossy transform - original audio cannot be recovered
    mel_spectrogram: np.ndarray  # (n_mels, n_frames)
    sample_rate: int
    hop_length: int

    # Note sequence as token IDs (from ChartTokenizer)
    # Original timing is discretized to a fixed grid
    note_tokens: np.ndarray  # (seq_len,) int16

    # Conditioning metadata (numeric IDs only, no names)
    difficulty_id: int  # 0=easy, 1=medium, 2=hard, 3=expert
    instrument_id: int  # 0=lead, 1=bass, 2=rhythm, 3=keys

    # Content hash for deduplication only
    content_hash: str


def compute_content_hash(mel: np.ndarray, tokens: np.ndarray) -> str:
    """Compute a hash from features for deduplication."""
    h = hashlib.sha256()
    h.update(mel.tobytes()[:4096])  # partial hash only
    h.update(tokens.tobytes())
    return h.hexdigest()[:16]


def save_tab(data: TabData, path: Path) -> None:
    """Save TabData to a .tab file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    n_mels, n_frames = data.mel_spectrogram.shape
    n_tokens = len(data.note_tokens)

    # Compress mel spectrogram with zlib
    mel_f16 = data.mel_spectrogram.astype(np.float16)
    mel_compressed = zlib.compress(mel_f16.tobytes(), level=ZLIB_COMPRESSION_LEVEL)

    with open(path, "wb") as f:
        f.write(TAB_MAGIC)
        f.write(struct.pack("<H", TAB_FORMAT_VERSION))
        f.write(struct.pack("<BB", data.difficulty_id, data.instrument_id))
        f.write(struct.pack("<I", data.sample_rate))
        f.write(struct.pack("<I", data.hop_length))
        f.write(struct.pack("<I", n_mels))
        f.write(struct.pack("<I", n_frames))
        f.write(struct.pack("<I", n_tokens))
        f.write(data.content_hash.encode("ascii").ljust(16, b"\x00"))

        # Write compressed mel size then data
        f.write(struct.pack("<I", len(mel_compressed)))
        f.write(mel_compressed)

        # Write tokens
        tokens_i16 = data.note_tokens.astype(np.int16)
        f.write(tokens_i16.tobytes())


def load_tab(path: Path) -> TabData:
    """Load a .tab file."""
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != TAB_MAGIC:
            raise ValueError(f"Invalid .tab file: bad magic {magic}")

        version = struct.unpack("<H", f.read(2))[0]
        if version > TAB_FORMAT_VERSION:
            raise ValueError(f"Unsupported .tab version {version}")

        difficulty_id, instrument_id = struct.unpack("<BB", f.read(2))
        sample_rate = struct.unpack("<I", f.read(4))[0]
        hop_length = struct.unpack("<I", f.read(4))[0]
        n_mels = struct.unpack("<I", f.read(4))[0]
        n_frames = struct.unpack("<I", f.read(4))[0]
        n_tokens = struct.unpack("<I", f.read(4))[0]
        content_hash = f.read(16).rstrip(b"\x00").decode("ascii")

        mel_size = struct.unpack("<I", f.read(4))[0]
        mel_compressed = f.read(mel_size)
        mel_bytes = zlib.decompress(mel_compressed)

        mel = np.frombuffer(mel_bytes, dtype=np.float16).reshape(n_mels, n_frames)
        mel = mel.astype(np.float32)

        tokens = np.frombuffer(f.read(n_tokens * 2), dtype=np.int16)

    return TabData(
        mel_spectrogram=mel,
        sample_rate=sample_rate,
        hop_length=hop_length,
        note_tokens=tokens,
        difficulty_id=difficulty_id,
        instrument_id=instrument_id,
        content_hash=content_hash,
    )


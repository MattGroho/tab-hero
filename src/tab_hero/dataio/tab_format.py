"""Binary format for storing preprocessed tab data.

File format (.tab):
    Header (16 bytes):
        - Magic number: 4 bytes (b"TABH")
        - Format version: 2 bytes (uint16)
        - Flags: 2 bytes (reserved)
        - Content hash: 8 bytes (xxhash64 of uncompressed content)
    
    Compressed payload (zlib):
        - mel_spectrogram: float32 tensor
        - tokens: int32 tensor
        - metadata: JSON bytes
"""

import hashlib
import json
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch


TAB_MAGIC = b"TABH"
TAB_FORMAT_VERSION = 1


@dataclass
class TabData:
    """Container for tab file data."""
    mel_spectrogram: torch.Tensor
    tokens: List[int]
    instrument: str
    difficulty: str
    source_audio: str
    source_chart: str
    duration_ms: float


def compute_content_hash(data: bytes) -> bytes:
    """Compute 8-byte hash of content."""
    h = hashlib.sha256(data).digest()
    return h[:8]


def save_tab(
    path: Union[str, Path],
    mel_spectrogram: torch.Tensor,
    tokens: List[int],
    instrument: str = "lead",
    difficulty: str = "expert",
    source_audio: str = "",
    source_chart: str = "",
    duration_ms: float = 0.0,
) -> None:
    """Save preprocessed data to .tab file."""
    path = Path(path)

    metadata = {
        "instrument": instrument,
        "difficulty": difficulty,
        "source_audio": source_audio,
        "source_chart": source_chart,
        "duration_ms": duration_ms,
        "mel_shape": list(mel_spectrogram.shape),
        "num_tokens": len(tokens),
    }

    mel_bytes = mel_spectrogram.numpy().astype(np.float32).tobytes()
    token_bytes = np.array(tokens, dtype=np.int32).tobytes()
    meta_bytes = json.dumps(metadata).encode("utf-8")

    payload = struct.pack("<I", len(mel_bytes)) + mel_bytes
    payload += struct.pack("<I", len(token_bytes)) + token_bytes
    payload += struct.pack("<I", len(meta_bytes)) + meta_bytes

    content_hash = compute_content_hash(payload)
    compressed = zlib.compress(payload, level=6)

    header = TAB_MAGIC
    header += struct.pack("<H", TAB_FORMAT_VERSION)
    header += struct.pack("<H", 0)
    header += content_hash

    path.write_bytes(header + compressed)


def load_tab(path: Union[str, Path]) -> TabData:
    """Load preprocessed data from .tab file."""
    path = Path(path)
    data = path.read_bytes()

    if data[:4] != TAB_MAGIC:
        raise ValueError(f"Invalid magic: {data[:4]}")

    version = struct.unpack("<H", data[4:6])[0]
    if version != TAB_FORMAT_VERSION:
        raise ValueError(f"Unsupported version: {version}")

    stored_hash = data[8:16]
    compressed = data[16:]
    payload = zlib.decompress(compressed)

    actual_hash = compute_content_hash(payload)
    if stored_hash != actual_hash:
        raise ValueError("Content hash mismatch")

    offset = 0
    mel_len = struct.unpack("<I", payload[offset:offset+4])[0]
    offset += 4
    mel_bytes = payload[offset:offset+mel_len]
    offset += mel_len

    token_len = struct.unpack("<I", payload[offset:offset+4])[0]
    offset += 4
    token_bytes = payload[offset:offset+token_len]
    offset += token_len

    meta_len = struct.unpack("<I", payload[offset:offset+4])[0]
    offset += 4
    meta_bytes = payload[offset:offset+meta_len]

    metadata = json.loads(meta_bytes.decode("utf-8"))
    mel_array = np.frombuffer(mel_bytes, dtype=np.float32)
    mel_spectrogram = torch.from_numpy(mel_array.reshape(metadata["mel_shape"]))
    tokens = np.frombuffer(token_bytes, dtype=np.int32).tolist()

    return TabData(
        mel_spectrogram=mel_spectrogram,
        tokens=tokens,
        instrument=metadata["instrument"],
        difficulty=metadata["difficulty"],
        source_audio=metadata["source_audio"],
        source_chart=metadata["source_chart"],
        duration_ms=metadata["duration_ms"],
    )

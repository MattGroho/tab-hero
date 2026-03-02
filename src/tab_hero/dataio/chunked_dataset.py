"""Chunked dataset for memory-efficient training on long songs.

Splits long songs into fixed-length chunks with optional overlap.
Handles token alignment to chunk at TIME_DELTA boundaries.
"""

import hashlib
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from .audio_processor import spec_augment
from .tab_format import load_tab, peek_tab_header

logger = logging.getLogger(__name__)


class ChunkedTabDataset(Dataset):
    """
    Dataset that chunks long songs for memory-efficient training.

    Each song is split into chunks of max_mel_frames. Token sequences are
    aligned to TIME_DELTA token boundaries to avoid splitting notes.

    Args:
        data_dir: Directory containing .tab files
        split: "train", "val", or None for all files
        max_mel_frames: Maximum mel frames per chunk (before downsampling)
        max_token_length: Maximum tokens per chunk
        chunk_overlap_frames: Overlap between chunks for continuity
        min_chunk_frames: Minimum frames to form a valid chunk
        time_delta_base: Base token ID for TIME_DELTA tokens
        time_delta_count: Number of TIME_DELTA tokens in vocabulary
    """

    # Token ranges from tokenizer
    TIME_DELTA_BASE = 3      # TIME_0 starts at 3
    TIME_DELTA_COUNT = 501   # TIME_0 to TIME_500

    def __init__(
        self,
        data_dir: str,
        split: Optional[str] = None,
        max_mel_frames: int = 8192,
        max_token_length: int = 4096,
        chunk_overlap_frames: int = 512,
        min_chunk_frames: int = 1024,
        audio_downsample: int = 4,
        training: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.max_mel_frames = max_mel_frames
        self.max_token_length = max_token_length
        self.chunk_overlap_frames = chunk_overlap_frames
        self.min_chunk_frames = min_chunk_frames
        self.audio_downsample = audio_downsample
        self.training = training

        # Load file list
        self.tab_files = self._load_file_list(split)

        # Build chunk index (cached to disk) and per-file metadata
        self.file_metadata: List[Dict] = []
        self.chunks = self._build_chunk_index()

        logger.info(f"ChunkedTabDataset: {len(self.tab_files)} files -> {len(self.chunks)} chunks")

    def _load_file_list(self, split: Optional[str]) -> List[Path]:
        """Load list of .tab files, optionally filtered by split."""
        manifest_path = self.data_dir / "manifest.json"
        split_files: Optional[Set[str]] = None

        if manifest_path.exists() and split is not None:
            with open(manifest_path) as f:
                manifest = json.load(f)
            if split in manifest:
                split_files = set(manifest[split])
                logger.info(f"Using {split} split: {len(split_files)} files")

        if split_files is not None:
            return [
                self.data_dir / f"{name}.tab"
                for name in split_files
                if (self.data_dir / f"{name}.tab").exists()
            ]
        else:
            return list(self.data_dir.glob("*.tab"))

    def _cache_key(self) -> str:
        """Compute a deterministic cache key from file list + chunking params.

        The key is an MD5 hex digest of sorted file names concatenated with
        the chunking parameters that affect the index layout.
        """
        h = hashlib.md5()
        for p in sorted(str(f) for f in self.tab_files):
            h.update(p.encode())
        h.update(f"{self.max_mel_frames}:{self.chunk_overlap_frames}:{self.min_chunk_frames}".encode())
        return h.hexdigest()

    def _cache_path(self) -> Path:
        """Return the path for the chunk index cache file."""
        return self.data_dir / f".chunk_cache_{self._cache_key()}.json"

    def _read_headers_parallel(self, max_workers: int = 32) -> List[Optional[Dict]]:
        """Read headers for all tab files using a thread pool.

        Returns a list aligned with ``self.tab_files``.  Entries are ``None``
        for files that could not be read.
        """
        results: List[Optional[Dict]] = [None] * len(self.tab_files)

        def _read_one(idx_path: Tuple[int, Path]) -> Tuple[int, Optional[Dict]]:
            idx, path = idx_path
            try:
                hdr = peek_tab_header(path)
                return idx, {
                    "n_frames": hdr["n_frames"],
                    "difficulty_id": hdr["difficulty_id"],
                    "instrument_id": hdr["instrument_id"],
                    "genre_id": hdr["genre_id"],
                    "song_id": hdr["song_id"],
                }
            except Exception as e:
                logger.warning(f"Error reading header {path}: {e}")
                return idx, None

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_read_one, (i, p)): i for i, p in enumerate(self.tab_files)}
            for future in as_completed(futures):
                idx, hdr = future.result()
                results[idx] = hdr

        return results

    def _build_chunk_index(self) -> List[Tuple[int, int, int]]:
        """Build index of chunks across all files.

        On first call the index is built by reading every ``.tab`` header in
        parallel (via :pymethod:`_read_headers_parallel`) and then cached to a
        JSON file next to the data.  Subsequent calls with the same file list
        and chunking parameters load the cache in < 1 s.

        Side-effect: populates ``self.file_metadata`` with per-file dicts
        containing ``difficulty_id``, ``instrument_id``, ``genre_id``,
        ``song_id``, and ``n_frames``.
        """
        cache_path = self._cache_path()

        # --- try loading from cache ---
        if cache_path.exists():
            try:
                with open(cache_path) as f:
                    cached = json.load(f)
                self.file_metadata = cached["file_metadata"]
                chunks = [tuple(c) for c in cached["chunks"]]
                logger.info(f"Loaded chunk index from cache ({len(chunks)} chunks)")
                return chunks
            except Exception as e:
                logger.warning(f"Cache load failed, rebuilding: {e}")

        # --- cache miss: read all headers in parallel ---
        logger.info(f"Building chunk index for {len(self.tab_files)} files (parallel)...")
        headers = self._read_headers_parallel()

        chunks: List[Tuple[int, int, int]] = []
        file_metadata: List[Dict] = []
        stride = max(1, self.max_mel_frames - self.chunk_overlap_frames)

        for file_idx, hdr in enumerate(headers):
            if hdr is None:
                file_metadata.append({})
                continue

            file_metadata.append(hdr)
            n_frames = hdr["n_frames"]

            if n_frames <= self.max_mel_frames:
                chunks.append((file_idx, 0, n_frames))
            else:
                start = 0
                while start < n_frames:
                    end = min(start + self.max_mel_frames, n_frames)
                    if (end - start) >= self.min_chunk_frames:
                        chunks.append((file_idx, start, end))
                    start += stride

        self.file_metadata = file_metadata

        # --- persist cache ---
        try:
            payload = {
                "file_metadata": file_metadata,
                "chunks": chunks,
            }
            with open(cache_path, "w") as f:
                json.dump(payload, f)
            logger.info(f"Saved chunk index cache to {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to write chunk cache: {e}")

        return chunks

    def __len__(self) -> int:
        return len(self.chunks)

    def _find_token_boundary(self, tokens: np.ndarray, target_frame: int,
                              mel_frames: int, is_end: bool = False) -> int:
        """
        Find token index that aligns with audio frame boundary.

        Searches for TIME_DELTA tokens and estimates position based on
        cumulative time deltas.
        """
        if len(tokens) <= 4:
            return len(tokens) if is_end else 0

        # Simple approach: proportional position
        # More accurate: track cumulative time from TIME_DELTA tokens
        proportion = target_frame / mel_frames
        target_idx = int(proportion * len(tokens))

        # Align to quad boundary (tokens come in groups of 4)
        target_idx = (target_idx // 4) * 4

        if is_end:
            return min(target_idx + 4, len(tokens))
        else:
            return max(0, target_idx)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        file_idx, start_frame, end_frame = self.chunks[idx]
        tab_path = self.tab_files[file_idx]

        try:
            data = load_tab(tab_path)

            mel = data.mel_spectrogram  # (n_mels, n_frames)
            tokens = data.note_tokens
            total_frames = mel.shape[1]

            # Resolve end_frame (-1 is a legacy sentinel for "full file")
            if end_frame == -1 or end_frame > total_frames:
                end_frame = total_frames
            if end_frame - start_frame > self.max_mel_frames:
                end_frame = start_frame + self.max_mel_frames

            # Extract mel chunk
            mel_chunk = mel[:, start_frame:end_frame]

            # Find corresponding token range
            if start_frame == 0 and end_frame >= total_frames:
                # Full song, use all tokens
                token_chunk = tokens
            else:
                # Find token boundaries
                token_start = self._find_token_boundary(
                    tokens, start_frame, total_frames, is_end=False
                )
                token_end = self._find_token_boundary(
                    tokens, end_frame, total_frames, is_end=True
                )
                token_chunk = tokens[token_start:token_end]

            # Truncate tokens if needed
            if len(token_chunk) > self.max_token_length:
                token_chunk = token_chunk[:self.max_token_length - 1]
                # Ensure EOS at end
                token_chunk = np.append(token_chunk, [2])  # EOS = 2

            # Ensure BOS at start if this is first chunk
            if start_frame == 0 and len(token_chunk) > 0 and token_chunk[0] != 1:
                token_chunk = np.insert(token_chunk, 0, 1)  # BOS = 1

            # Ensure EOS at end if this is last chunk
            if end_frame >= total_frames and len(token_chunk) > 0 and token_chunk[-1] != 2:
                token_chunk = np.append(token_chunk, 2)  # EOS = 2

            audio = torch.from_numpy(mel_chunk.copy()).T.float()  # (n_frames, n_mels)
            if self.training:
                audio = spec_augment(audio)

            return {
                "audio_embeddings": audio,
                "note_tokens": torch.from_numpy(token_chunk.copy()).long(),
                "difficulty_id": torch.tensor(data.difficulty_id),
                "instrument_id": torch.tensor(data.instrument_id),
                "position_offset": torch.tensor(start_frame // self.audio_downsample),
            }

        except Exception as e:
            logger.warning(f"Failed to load {tab_path}: {e}")
            return None


def chunked_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate function for ChunkedTabDataset.

    Returns an ``audio_mask`` boolean tensor (True = valid, False = pad).
    """
    # Filter out failed samples (None from __getitem__)
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        raise RuntimeError("All samples in batch failed to load")

    audio_emb = [item["audio_embeddings"] for item in batch]
    note_tokens = [item["note_tokens"] for item in batch]
    difficulty_ids = torch.stack([item["difficulty_id"] for item in batch])
    instrument_ids = torch.stack([item["instrument_id"] for item in batch])
    position_offsets = torch.stack([item["position_offset"] for item in batch])

    audio_lengths = torch.tensor([a.size(0) for a in audio_emb], dtype=torch.long)

    audio_padded = pad_sequence(audio_emb, batch_first=True, padding_value=0.0)
    tokens_padded = pad_sequence(note_tokens, batch_first=True, padding_value=0)

    max_frames = audio_padded.size(1)
    audio_mask = torch.arange(max_frames).unsqueeze(0) < audio_lengths.unsqueeze(1)

    return {
        "audio_embeddings": audio_padded,
        "audio_mask": audio_mask,
        "note_tokens": tokens_padded,
        "difficulty_id": difficulty_ids,
        "instrument_id": instrument_ids,
        "position_offset": position_offsets,
    }




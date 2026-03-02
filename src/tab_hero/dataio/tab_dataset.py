"""Dataset for loading preprocessed .tab files."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Set
import logging

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from .audio_processor import spec_augment
from .tab_format import load_tab

logger = logging.getLogger(__name__)


class TabDataset(Dataset):
    """
    Dataset for preprocessed .tab training files.

    Much faster than TabHeroDataset since audio is already processed
    to mel spectrograms.

    Args:
        data_dir: Path to directory containing .tab files and manifest.json
        split: Which split to use ("train", "val", or None for all files)
        max_mel_frames: Maximum mel spectrogram frames to use
        max_token_length: Maximum token sequence length
    """

    def __init__(
        self,
        data_dir: str,
        split: Optional[str] = None,
        max_mel_frames: int = 4096,
        max_token_length: int = 4096,
        training: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.max_mel_frames = max_mel_frames
        self.max_token_length = max_token_length
        self.training = training

        # Load manifest if available for train/val split
        manifest_path = self.data_dir / "manifest.json"
        split_files: Optional[Set[str]] = None

        if manifest_path.exists() and split is not None:
            with open(manifest_path) as f:
                manifest = json.load(f)
            if split in manifest:
                split_files = set(manifest[split])
                logger.info(f"Using {split} split: {len(split_files)} files from manifest")
            else:
                logger.warning(f"Split '{split}' not found in manifest, using all files")

        # Collect .tab files, optionally filtered by split
        if split_files is not None:
            self.tab_files = [
                self.data_dir / f"{name}.tab"
                for name in split_files
                if (self.data_dir / f"{name}.tab").exists()
            ]
        else:
            self.tab_files = list(self.data_dir.glob("*.tab"))

        logger.info(f"Found {len(self.tab_files)} .tab files in {data_dir}"
                    + (f" ({split} split)" if split else ""))

    def __len__(self) -> int:
        return len(self.tab_files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        tab_path = self.tab_files[idx]

        try:
            data = load_tab(tab_path)

            # Truncate mel if needed
            mel = data.mel_spectrogram
            if mel.shape[1] > self.max_mel_frames:
                mel = mel[:, :self.max_mel_frames]

            # Truncate tokens if needed
            tokens = data.note_tokens
            if len(tokens) > self.max_token_length:
                tokens = tokens[:self.max_token_length - 1]
                # Append EOS (token 2)
                tokens = list(tokens) + [2]

            audio = torch.from_numpy(mel).T  # (n_frames, n_mels)
            if self.training:
                audio = spec_augment(audio)

            return {
                # Use audio_embeddings key for trainer compatibility
                "audio_embeddings": audio,
                "note_tokens": torch.from_numpy(tokens).long(),
                "difficulty_id": torch.tensor(data.difficulty_id),
                "instrument_id": torch.tensor(data.instrument_id),
            }

        except Exception as e:
            logger.warning(f"Failed to load {tab_path}: {e}")
            return None


def tab_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate function for TabDataset.

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
    }


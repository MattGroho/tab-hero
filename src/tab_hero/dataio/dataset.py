"""
PyTorch Dataset for Tab Hero training.

Handles loading and preprocessing of audio-chart pairs for training.
"""

from pathlib import Path
from typing import Dict, List, Optional
import logging

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from .audio_processor import AudioProcessor
from .chart_parser import ChartParser
from .tokenizer import ChartTokenizer

logger = logging.getLogger(__name__)


class TabHeroDataset(Dataset):
    """Dataset of audio-chart pairs for training the chart generation model."""

    DIFFICULTY_TO_ID = {"easy": 0, "medium": 1, "hard": 2, "expert": 3}
    INSTRUMENT_TO_ID = {"lead": 0, "bass": 1, "rhythm": 2, "keys": 3}

    def __init__(
        self,
        data_dir: str,
        instrument: str = "lead",
        difficulty: str = "expert",
        max_audio_duration_s: float = 60.0,
        max_sequence_length: int = 4096,
        device: str = "cpu",
        tokenizer: Optional[ChartTokenizer] = None,
    ):
        self.data_dir = Path(data_dir)
        self.instrument = instrument
        self.difficulty = difficulty
        self.max_audio_duration_s = max_audio_duration_s
        self.max_sequence_length = max_sequence_length

        self.audio_processor = AudioProcessor(device=device)
        self.chart_parser = ChartParser()
        self.tokenizer = tokenizer or ChartTokenizer()

        self.samples = self._scan_data_directory()
        logger.info(f"Found {len(self.samples)} valid samples in {data_dir}")

    def _scan_data_directory(self) -> List[Dict[str, Path]]:
        """Scan data directory recursively for valid song folders."""
        samples = []

        if not self.data_dir.exists():
            return samples

        for chart_file in self.data_dir.rglob("*.mid"):
            self._add_sample_if_valid(chart_file.parent, chart_file, samples)

        for chart_file in self.data_dir.rglob("*.chart"):
            # Avoid duplicates if we already found a .mid in same dir
            if not any(s["song_dir"] == chart_file.parent for s in samples):
                self._add_sample_if_valid(chart_file.parent, chart_file, samples)

        return samples

    def _add_sample_if_valid(
        self,
        song_dir: Path,
        chart_file: Path,
        samples: List[Dict[str, Path]]
    ) -> None:
        """Add a sample if the directory contains valid audio."""
        # Prefer guitar.ogg (isolated stem) over the full mix
        audio_file = None

        guitar_file = song_dir / "guitar.ogg"
        if guitar_file.exists():
            audio_file = guitar_file
        else:
            # Fall back to full mix
            song_file = song_dir / "song.ogg"
            if song_file.exists():
                audio_file = song_file
            else:
                # Try any audio file
                for ext in ["*.ogg", "*.mp3", "*.wav", "*.opus", "*.mp4", "*.m4a"]:
                    audio_files = list(song_dir.glob(ext))
                    if audio_files:
                        audio_file = audio_files[0]
                        break

        if audio_file:
            samples.append({
                "audio": audio_file,
                "chart": chart_file,
                "song_dir": song_dir,
            })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a training sample.

        Returns:
            Dictionary containing:
            - audio_embeddings: Mel spectrogram features (n_frames, n_mels)
            - note_tokens: Tokenized note sequence (seq_len,)
            - difficulty_id: Difficulty conditioning ID
            - instrument_id: Instrument conditioning ID
        """
        sample = self.samples[idx]

        try:
            audio_embeddings, _ = self.audio_processor.process_audio_file(
                sample["audio"],
                max_duration_sec=self.max_audio_duration_s
            )

            chart_data = self.chart_parser.parse(
                sample["chart"],
                instrument=self.instrument,
                difficulty=self.difficulty
            )

            note_tokens = self.tokenizer.encode_chart(chart_data)

            if len(note_tokens) > self.max_sequence_length:
                note_tokens = note_tokens[:self.max_sequence_length - 1] + [self.tokenizer.eos_token_id]

            return {
                "audio_embeddings": audio_embeddings,  # (n_frames, 128)
                "note_tokens": torch.tensor(note_tokens, dtype=torch.long),
                "difficulty_id": torch.tensor(self.DIFFICULTY_TO_ID[self.difficulty]),
                "instrument_id": torch.tensor(self.INSTRUMENT_TO_ID[self.instrument]),
            }
        except Exception as e:
            logger.warning(f"Error loading sample {idx}: {e}")
            return None


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collate function for DataLoader.

    Pads audio embeddings and note tokens to the same length within batch.
    Returns an ``audio_mask`` boolean tensor (True = valid, False = pad).
    """
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        raise RuntimeError("All samples in batch failed to load")

    audio_embeddings = [item["audio_embeddings"] for item in batch]
    note_tokens = [item["note_tokens"] for item in batch]
    difficulty_ids = torch.stack([item["difficulty_id"] for item in batch])
    instrument_ids = torch.stack([item["instrument_id"] for item in batch])

    audio_lengths = torch.tensor([a.size(0) for a in audio_embeddings], dtype=torch.long)
    audio_padded = pad_sequence(audio_embeddings, batch_first=True, padding_value=0.0)

    max_frames = audio_padded.size(1)
    audio_mask = torch.arange(max_frames).unsqueeze(0) < audio_lengths.unsqueeze(1)

    tokens_padded = pad_sequence(note_tokens, batch_first=True, padding_value=0)

    return {
        "audio_embeddings": audio_padded,
        "audio_mask": audio_mask,
        "note_tokens": tokens_padded,
        "difficulty_id": difficulty_ids,
        "instrument_id": instrument_ids,
    }


def create_dataloader(
    data_dir: str,
    batch_size: int = 4,
    num_workers: int = 0,
    shuffle: bool = True,
    **dataset_kwargs
) -> DataLoader:
    """Create a DataLoader for training."""
    dataset = TabHeroDataset(data_dir, **dataset_kwargs)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

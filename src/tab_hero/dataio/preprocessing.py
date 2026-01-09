"""Preprocessing pipeline for Clone Hero songs.

Discovers songs, extracts audio features, parses charts, and saves
preprocessed data in .tab format.
"""

import os
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple, Union

import numpy as np
import torch
from tqdm import tqdm

from .audio_processor import AudioProcessor, DEFAULT_MEL_CONFIG
from .chart_parser import ChartParser, ChartData
from .tab_format import save_tab
from .tokenizer import ChartTokenizer, TokenizerConfig


DIFFICULTY_MAP = {
    "easy": "easy",
    "medium": "medium",
    "hard": "hard",
    "expert": "expert",
}

INSTRUMENT_MAP = {
    "lead": "lead",
    "guitar": "lead",
    "bass": "bass",
    "rhythm": "rhythm",
    "keys": "keys",
}

AUDIO_EXTENSIONS = {".ogg", ".mp3", ".wav", ".opus"}
CHART_EXTENSIONS = {".chart", ".mid", ".midi"}


def find_audio_file(song_dir: Path) -> Optional[Path]:
    """Find the main audio file in a song directory."""
    for name in ["song", "guitar", "audio"]:
        for ext in AUDIO_EXTENSIONS:
            path = song_dir / f"{name}{ext}"
            if path.exists():
                return path
    for path in song_dir.iterdir():
        if path.suffix.lower() in AUDIO_EXTENSIONS:
            return path
    return None


def find_chart_file(song_dir: Path) -> Optional[Path]:
    """Find the chart file in a song directory."""
    for name in ["notes"]:
        for ext in CHART_EXTENSIONS:
            path = song_dir / f"{name}{ext}"
            if path.exists():
                return path
    for path in song_dir.iterdir():
        if path.suffix.lower() in CHART_EXTENSIONS:
            return path
    return None


def is_song_directory(path: Path) -> bool:
    """Check if directory contains a valid song."""
    if not path.is_dir():
        return False
    return find_audio_file(path) is not None and find_chart_file(path) is not None


def discover_song_directories(root: Path) -> Generator[Path, None, None]:
    """Recursively find all song directories."""
    root = Path(root)
    if is_song_directory(root):
        yield root
        return
    for entry in root.iterdir():
        if entry.is_dir():
            yield from discover_song_directories(entry)


def extract_mel_spectrogram(
    audio_path: Path,
    processor: Optional[AudioProcessor] = None,
) -> Tuple[torch.Tensor, float]:
    """Extract mel spectrogram from audio file."""
    if processor is None:
        processor = AudioProcessor()
    mel = processor.process_audio_file(audio_path)
    duration_ms = processor.get_duration_ms(mel)
    return mel, duration_ms


def process_song_all_variants(
    song_dir: Path,
    output_dir: Path,
    difficulties: Optional[List[str]] = None,
    instruments: Optional[List[str]] = None,
    audio_processor: Optional[AudioProcessor] = None,
    chart_parser: Optional[ChartParser] = None,
    tokenizer: Optional[ChartTokenizer] = None,
) -> List[Path]:
    """Process a song for all difficulty/instrument combinations."""
    if difficulties is None:
        difficulties = ["expert"]
    if instruments is None:
        instruments = ["lead"]

    audio_path = find_audio_file(song_dir)
    chart_path = find_chart_file(song_dir)

    if audio_path is None or chart_path is None:
        return []

    if audio_processor is None:
        audio_processor = AudioProcessor()
    if chart_parser is None:
        chart_parser = ChartParser()
    if tokenizer is None:
        tokenizer = ChartTokenizer()

    mel, duration_ms = extract_mel_spectrogram(audio_path, audio_processor)

    output_files = []
    for difficulty in difficulties:
        for instrument in instruments:
            try:
                chart = chart_parser.parse(chart_path, instrument, difficulty)
                tokens = tokenizer.encode_chart(chart)

                output_name = f"{song_dir.name}_{instrument}_{difficulty}.tab"
                output_path = output_dir / output_name
                output_dir.mkdir(parents=True, exist_ok=True)

                save_tab(
                    output_path,
                    mel_spectrogram=mel,
                    tokens=tokens,
                    instrument=instrument,
                    difficulty=difficulty,
                    source_audio=str(audio_path.name),
                    source_chart=str(chart_path.name),
                    duration_ms=duration_ms,
                )
                output_files.append(output_path)
            except Exception:
                continue

    return output_files

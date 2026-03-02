"""
End-to-end inference pipeline for Tab Hero.

Takes audio file + metadata and outputs Clone Hero-compatible song folder.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union
import logging

from .generator import ChartGenerator
from .exporter import SongExporter

logger = logging.getLogger(__name__)


@dataclass
class SongMetadata:
    """Clone Hero song metadata."""

    name: str = "Generated Tab"
    artist: str = "Unknown Artist"
    album: str = ""
    genre: str = ""
    charter: str = "Tab Hero"
    bpm: float = 120.0


class TabHeroPipeline:
    """
    High-level inference pipeline for generating Clone Hero charts.

    Usage:
        pipeline = TabHeroPipeline("checkpoints/best_model.pt")
        output_path = pipeline.generate_song(
            audio_path="song.mp3",
            output_dir="output/MySong",
            metadata=SongMetadata(name="My Song", artist="My Artist"),
            difficulty="expert",
            instrument="lead",
        )
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        data_dir: Optional[str] = None,
    ):
        """
        Initialize the inference pipeline.

        Args:
            model_path: Path to trained model checkpoint
            device: Device for inference ("cuda" or "cpu")
            data_dir: Path to processed data dir (to load mel config for consistency)
        """
        self.generator = ChartGenerator(
            model_path=model_path, device=device, data_dir=data_dir
        )
        self.exporter = SongExporter()
        self.device = device

    def generate_song(
        self,
        audio_path: Union[str, Path],
        output_dir: Union[str, Path],
        metadata: Optional[SongMetadata] = None,
        difficulty: str = "expert",
        instrument: str = "lead",
        temperature: float = 1.0,
        include_audio: bool = True,
    ) -> Path:
        """
        Generate a complete Clone Hero song folder from audio.

        Args:
            audio_path: Path to input audio file (mp3, ogg, wav, etc.)
            output_dir: Output directory for the song folder
            metadata: Song metadata (name, artist, etc.)
            difficulty: Target difficulty level
            instrument: Target instrument
            temperature: Sampling temperature (higher = more random)
            include_audio: Whether to copy audio to output folder

        Returns:
            Path to the created song folder
        """
        audio_path = Path(audio_path)
        output_dir = Path(output_dir)

        if metadata is None:
            metadata = SongMetadata(name=audio_path.stem)

        logger.info(f"Generating {difficulty} {instrument} chart for {audio_path.name}")

        # Generate notes from audio
        result = self.generator.generate(
            audio_path=audio_path,
            difficulty=difficulty,
            instrument=instrument,
            temperature=temperature,
        )

        logger.info(f"Generated {len(result['notes'])} notes")

        # Export to Clone Hero format
        song_folder = self.exporter.export(
            notes=result["notes"],
            output_dir=str(output_dir),
            audio_path=str(audio_path) if include_audio else None,
            difficulty=difficulty,
            instrument=instrument,
            song_name=metadata.name,
            artist=metadata.artist,
            album=metadata.album,
            genre=metadata.genre,
            bpm=metadata.bpm,
        )

        logger.info(f"Song exported to {song_folder}")
        return song_folder

    def generate_multi_difficulty(
        self,
        audio_path: Union[str, Path],
        output_dir: Union[str, Path],
        metadata: Optional[SongMetadata] = None,
        difficulties: tuple = ("easy", "medium", "hard", "expert"),
        instrument: str = "lead",
        temperature: float = 1.0,
    ) -> Path:
        """
        Generate charts for multiple difficulties.

        Note: Currently generates each difficulty separately. Future versions
        may optimize by encoding audio once and generating all difficulties.
        """
        audio_path = Path(audio_path)
        output_dir = Path(output_dir)

        if metadata is None:
            metadata = SongMetadata(name=audio_path.stem)

        all_notes = {}
        for diff in difficulties:
            result = self.generator.generate(
                audio_path=audio_path,
                difficulty=diff,
                instrument=instrument,
                temperature=temperature,
            )
            all_notes[diff] = result["notes"]
            logger.info(f"Generated {len(result['notes'])} notes for {diff}")

        # Multi-difficulty export is not yet implemented in SongExporter;
        # fall back to writing the expert (or last) difficulty only.
        song_folder = self.exporter.export(
            notes=all_notes.get("expert", all_notes[difficulties[-1]]),
            output_dir=str(output_dir),
            audio_path=str(audio_path),
            difficulty="expert",
            instrument=instrument,
            song_name=metadata.name,
            artist=metadata.artist,
            album=metadata.album,
            genre=metadata.genre,
            bpm=metadata.bpm,
        )

        return song_folder


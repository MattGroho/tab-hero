#!/usr/bin/env python3
"""
Chart generation script.

Generates Clone Hero-compatible song folders from audio files.

Usage:
    python scripts/generate.py --audio song.ogg --output ./output/MySong
    python scripts/generate.py --audio song.mp3 --difficulty hard --instrument bass
    python scripts/generate.py --audio song.ogg --song-name "My Song" --artist "My Artist"
    python scripts/generate.py --audio song.ogg --zip
"""

import argparse
import logging
import shutil
from pathlib import Path

import torch

from tab_hero.inference import TabHeroPipeline, SongMetadata

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate Clone Hero charts from audio files"
    )
    parser.add_argument(
        "--audio",
        type=str,
        required=True,
        help="Path to input audio file (mp3, ogg, wav, etc.)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output folder path (default: ./output/<audio_name>)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="checkpoints/best_model.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--difficulty",
        type=str,
        default="expert",
        choices=["easy", "medium", "hard", "expert"],
        help="Target difficulty level",
    )
    parser.add_argument(
        "--instrument",
        type=str,
        default="lead",
        choices=["lead", "bass", "rhythm", "keys"],
        help="Target instrument",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (higher = more random)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for inference (cuda or cpu)",
    )
    parser.add_argument(
        "--song-name",
        type=str,
        default=None,
        help="Song name for chart metadata",
    )
    parser.add_argument(
        "--artist",
        type=str,
        default="Unknown Artist",
        help="Artist name for chart metadata",
    )
    parser.add_argument(
        "--album",
        type=str,
        default="",
        help="Album name for chart metadata",
    )
    parser.add_argument(
        "--genre",
        type=str,
        default="",
        help="Genre for chart metadata (e.g. rock, metal, pop)",
    )
    parser.add_argument(
        "--bpm",
        type=float,
        default=120.0,
        help="Song BPM (for MIDI timing)",
    )
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Don't copy audio file to output folder",
    )
    parser.add_argument(
        "--zip",
        action="store_true",
        help="Zip the output song folder after generation",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/processed",
        help="Path to processed data dir (to load mel config from manifest)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Validate input
    audio_path = Path(args.audio)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Set output folder
    if args.output is None:
        output_dir = Path("output") / audio_path.stem
    else:
        output_dir = Path(args.output)

    # Check device
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = "cpu"

    # Build metadata
    metadata = SongMetadata(
        name=args.song_name or audio_path.stem,
        artist=args.artist,
        album=args.album,
        genre=args.genre,
        bpm=args.bpm,
    )

    # Load pipeline with mel config from training data
    logger.info(f"Loading model from {args.model}")
    pipeline = TabHeroPipeline(
        model_path=args.model,
        device=device,
        data_dir=args.data_dir,
    )

    # Generate and export
    song_folder = pipeline.generate_song(
        audio_path=audio_path,
        output_dir=output_dir,
        metadata=metadata,
        difficulty=args.difficulty,
        instrument=args.instrument,
        temperature=args.temperature,
        include_audio=not args.no_audio,
    )

    logger.info(f"Clone Hero song folder created: {song_folder}")

    if args.zip:
        zip_base = song_folder.parent / song_folder.name
        zip_path = Path(shutil.make_archive(str(zip_base), "zip", song_folder.parent, song_folder.name))
        logger.info(f"Zipped to: {zip_path}")


if __name__ == "__main__":
    main()


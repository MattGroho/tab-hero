#!/usr/bin/env python3
"""
Demo inference script.

Downloads royalty-free audio samples (CC-licensed) and runs the Tab Hero
inference pipeline on each, producing Clone Hero-compatible song folders
and optionally zipping them.

Samples used (all Kevin MacLeod, incompetech.com, CC BY 4.0):
  - "Funk Game Loop"    - funk/guitar instrumental, driving groove.
  - "Volatile Reaction" - hard rock/action, electric guitar-forward.
  - "Aggressor"         - rock harder collection, aggressive guitar riff.

Usage:
    python scripts/run_demo.py
    python scripts/run_demo.py --difficulty hard --zip
    python scripts/run_demo.py --model checkpoints/best_model.pt --output demo_output
"""

import argparse
import logging
import shutil
import urllib.request
from pathlib import Path

import torch

from tab_hero.inference import SongMetadata, TabHeroPipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Royalty-free samples
# All tracks: Kevin MacLeod (incompetech.com), CC BY 4.0
# https://creativecommons.org/licenses/by/4.0/
# ---------------------------------------------------------------------------
_CC_BY = "CC BY 4.0 - https://creativecommons.org/licenses/by/4.0/"
_BASE  = "https://incompetech.com/music/royalty-free/mp3-royaltyfree"

DEMO_SAMPLES = [
    {
        "name": "Funk Game Loop",
        "artist": "Kevin MacLeod",
        "album": "incompetech.com",
        "genre": "funk",
        "url": f"{_BASE}/Funk%20Game%20Loop.mp3",
        "filename": "funk_game_loop.mp3",
        "license": _CC_BY,
    },
    {
        "name": "Volatile Reaction",
        "artist": "Kevin MacLeod",
        "album": "incompetech.com",
        "genre": "rock",
        "url": f"{_BASE}/Volatile%20Reaction.mp3",
        "filename": "volatile_reaction.mp3",
        "license": _CC_BY,
    },
    {
        "name": "Aggressor",
        "artist": "Kevin MacLeod",
        "album": "incompetech.com",
        "genre": "rock",
        "url": f"{_BASE}/Aggressor.mp3",
        "filename": "aggressor.mp3",
        "license": _CC_BY,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Tab Hero inference on royalty-free demo samples"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="checkpoints/best_model.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="demo_output",
        help="Root output directory for generated song folders",
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
        help="Sampling temperature",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for inference (cuda or cpu)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/processed",
        help="Path to processed data dir (for mel config)",
    )
    parser.add_argument(
        "--zip",
        action="store_true",
        help="Zip each output song folder after generation",
    )
    parser.add_argument(
        "--samples-dir",
        type=str,
        default="data/sample/demo",
        help="Directory to cache downloaded audio samples",
    )
    return parser.parse_args()


def download_sample(url: str, dest: Path) -> bool:
    """Download a file if it does not already exist. Returns True on success."""
    if dest.exists():
        logger.info(f"Already downloaded: {dest.name}")
        return True
    logger.info(f"Downloading {dest.name} ...")
    try:
        urllib.request.urlretrieve(url, dest)
        logger.info(f"Saved to {dest}")
        return True
    except Exception as exc:
        logger.warning(f"Failed to download {url}: {exc}")
        return False


def main() -> None:
    args = parse_args()

    samples_dir = Path(args.samples_dir)
    samples_dir.mkdir(parents=True, exist_ok=True)

    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = "cpu"

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    logger.info(f"Loading model from {model_path}")
    pipeline = TabHeroPipeline(
        model_path=str(model_path),
        device=device,
        data_dir=args.data_dir,
    )

    results = []
    for sample in DEMO_SAMPLES:
        audio_path = samples_dir / sample["filename"]

        if not download_sample(sample["url"], audio_path):
            logger.warning(f"Skipping {sample['name']} (download failed)")
            continue

        song_slug = audio_path.stem
        output_dir = output_root / song_slug

        metadata = SongMetadata(
            name=sample["name"],
            artist=sample["artist"],
            album=sample.get("album", ""),
            genre=sample.get("genre", ""),
        )

        logger.info(f"Generating chart: {sample['name']} by {sample['artist']}")
        try:
            song_folder = pipeline.generate_song(
                audio_path=audio_path,
                output_dir=output_dir,
                metadata=metadata,
                difficulty=args.difficulty,
                instrument=args.instrument,
                temperature=args.temperature,
                include_audio=True,
            )

            if args.zip:
                zip_base = song_folder.parent / song_folder.name
                zip_path = Path(
                    shutil.make_archive(str(zip_base), "zip", song_folder.parent, song_folder.name)
                )
                results.append(zip_path)
                logger.info(f"Zipped: {zip_path}")
            else:
                results.append(song_folder)
                logger.info(f"Song folder: {song_folder}")

        except Exception as exc:
            logger.error(f"Failed to generate chart for {sample['name']}: {exc}")

    print("\n--- Demo complete ---")
    for r in results:
        print(f"  {r}")


if __name__ == "__main__":
    main()


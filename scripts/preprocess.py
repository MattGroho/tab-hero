#!/usr/bin/env python3
"""Preprocess Clone Hero songs to .tab format.

Usage:
    python -m scripts.preprocess --input /path/to/songs --output /path/to/output
"""

import argparse
import sys
from pathlib import Path

from tqdm import tqdm

from tab_hero.dataio.preprocessing import (
    discover_song_directories,
    process_song_all_variants,
    AudioProcessor,
    ChartParser,
    ChartTokenizer,
)


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess Clone Hero songs to .tab format"
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        required=True,
        help="Input directory containing songs",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        required=True,
        help="Output directory for .tab files",
    )
    parser.add_argument(
        "--difficulties",
        nargs="+",
        default=["expert"],
        help="Difficulties to process (default: expert)",
    )
    parser.add_argument(
        "--instruments",
        nargs="+",
        default=["lead"],
        help="Instruments to process (default: lead)",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: Input directory does not exist: {args.input}")
        sys.exit(1)

    args.output.mkdir(parents=True, exist_ok=True)

    print(f"Discovering songs in {args.input}...")
    song_dirs = list(discover_song_directories(args.input))
    print(f"Found {len(song_dirs)} songs")

    if not song_dirs:
        print("No songs found")
        sys.exit(0)

    audio_processor = AudioProcessor()
    chart_parser = ChartParser()
    tokenizer = ChartTokenizer()

    success = 0
    failed = 0

    for song_dir in tqdm(song_dirs, desc="Processing"):
        try:
            outputs = process_song_all_variants(
                song_dir,
                args.output,
                difficulties=args.difficulties,
                instruments=args.instruments,
                audio_processor=audio_processor,
                chart_parser=chart_parser,
                tokenizer=tokenizer,
            )
            if outputs:
                success += 1
            else:
                failed += 1
        except Exception as e:
            print(f"\nError processing {song_dir.name}: {e}")
            failed += 1

    print(f"\nDone: {success} succeeded, {failed} failed")


if __name__ == "__main__":
    main()

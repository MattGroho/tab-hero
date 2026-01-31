#!/usr/bin/env python3
"""
Extract time-domain features from .tab files.

Outputs a de-identified dataset with content_hash as the key column
for rejoining with .tab file contents.

Usage:
    python scripts/extract_features.py --data-dir data/processed --output features.csv
    python scripts/extract_features.py --sample 100 --output sample_features.csv
"""

import argparse
import csv
import json
import random
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tab_hero.dataio.feature_extractor import (
    extract_features_batch,
)
from tab_hero.dataio.tokenizer import ChartTokenizer


def main():
    parser = argparse.ArgumentParser(
        description="Extract time-domain features from .tab files"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory containing .tab files (default: data/processed)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("features.csv"),
        help="Output file path (default: features.csv)",
    )
    parser.add_argument(
        "--format",
        choices=["csv", "json", "jsonl"],
        default="csv",
        help="Output format (default: csv)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Random sample N files instead of processing all",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42)",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bar",
    )
    args = parser.parse_args()

    # Discover .tab files
    data_dir = args.data_dir
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        sys.exit(1)

    tab_files = sorted(data_dir.glob("*.tab"))
    if not tab_files:
        print(f"Error: No .tab files found in {data_dir}")
        sys.exit(1)

    print(f"Found {len(tab_files):,} .tab files in {data_dir}")

    # Sample if requested
    if args.sample is not None and args.sample < len(tab_files):
        random.seed(args.seed)
        tab_files = random.sample(tab_files, args.sample)
        print(f"Randomly sampled {args.sample} files (seed={args.seed})")

    # Initialize tokenizer
    tokenizer = ChartTokenizer()

    # Extract features
    print("Extracting features...")
    features_list = extract_features_batch(
        tab_files,
        tokenizer=tokenizer,
        progress=not args.no_progress,
    )

    print(f"Successfully extracted features from {len(features_list):,} files")

    # Write output
    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.format == "csv":
        write_csv(features_list, output_path)
    elif args.format == "json":
        write_json(features_list, output_path)
    elif args.format == "jsonl":
        write_jsonl(features_list, output_path)

    print(f"Wrote features to {output_path}")


def write_csv(features_list: list, output_path: Path) -> None:
    """Write features to CSV file."""
    if not features_list:
        return

    fieldnames = list(features_list[0].to_dict().keys())

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for features in features_list:
            writer.writerow(features.to_dict())


def write_json(features_list: list, output_path: Path) -> None:
    """Write features to JSON file."""
    data = [f.to_dict() for f in features_list]
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


def write_jsonl(features_list: list, output_path: Path) -> None:
    """Write features to JSON Lines file."""
    with open(output_path, "w") as f:
        for features in features_list:
            f.write(json.dumps(features.to_dict()) + "\n")


if __name__ == "__main__":
    main()


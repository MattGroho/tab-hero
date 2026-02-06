#!/usr/bin/env python3
"""Extract features from .tab files with resume support."""

import argparse
import csv
import json
import multiprocessing as mp
import random
import sys
from pathlib import Path
from typing import List, Set

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tab_hero.dataio.feature_extractor import TabFeatures, extract_features_batch_incremental


def load_existing_features(output_path: Path) -> tuple[List[TabFeatures], Set[str]]:
    """Load existing features from JSONL file for resume support."""
    features = []
    processed_files = set()

    if not output_path.exists():
        return features, processed_files

    try:
        with open(output_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                # Track by filename (stored in content_hash field for now)
                if "source_file" in data:
                    processed_files.add(data["source_file"])
                features.append(TabFeatures(**{k: v for k, v in data.items() if k != "source_file"}))
    except Exception as e:
        print(f"Warning: Could not load existing features: {e}")

    return features, processed_files


def main():
    parser = argparse.ArgumentParser(description="Extract features from .tab files")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output", type=Path, default=Path("features.csv"))
    parser.add_argument("--format", choices=["csv", "json", "jsonl"], default="csv")
    parser.add_argument("--sample", type=int, default=None, help="Sample N random files")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--workers", type=int, default=0, help="0=auto, -1=sequential")
    parser.add_argument("--save-interval", type=int, default=100,
                        help="Save progress every N files (default: 100)")
    parser.add_argument("--no-resume", action="store_true",
                        help="Start fresh, ignore existing progress")
    args = parser.parse_args()

    if not args.data_dir.exists():
        print(f"Error: {args.data_dir} not found")
        sys.exit(1)

    tab_files = sorted(args.data_dir.glob("*.tab"))
    if not tab_files:
        print(f"Error: No .tab files in {args.data_dir}")
        sys.exit(1)

    print(f"Found {len(tab_files):,} .tab files")

    # JSONL used internally for checkpointing, regardless of final format
    args.output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output.with_suffix(".jsonl.checkpoint")

    existing_features = []
    processed_files: Set[str] = set()

    if not args.no_resume and checkpoint_path.exists():
        existing_features, processed_files = load_existing_features(checkpoint_path)
        if processed_files:
            print(f"Resuming: {len(processed_files):,} files already processed")

    # Filter out already-processed files
    pending_files = [f for f in tab_files if f.name not in processed_files]
    already_done = len(tab_files) - len(pending_files)

    if already_done > 0:
        print(f"Skipping {already_done:,} already-processed files")

    if not pending_files:
        print("All files already processed!")
        all_features = existing_features
    else:
        if args.sample and args.sample < len(pending_files):
            random.seed(args.seed)
            pending_files = random.sample(pending_files, args.sample)
            print(f"Sampled {args.sample} files (seed={args.seed})")

        n_workers = None if args.workers == -1 else args.workers
        if n_workers is not None:
            actual = mp.cpu_count() if n_workers == 0 else n_workers
            print(f"Using {actual} workers")
        else:
            print("Sequential mode")

        # Extract with incremental saving to checkpoint
        new_features = extract_features_batch_incremental(
            pending_files,
            output_path=checkpoint_path,
            progress=not args.no_progress,
            n_workers=n_workers,
            save_interval=args.save_interval,
        )
        all_features = existing_features + new_features

    print(f"Total features: {len(all_features):,}")

    # Always write final output in requested format
    if args.format == "jsonl":
        # Just rename checkpoint to final output
        if checkpoint_path.exists():
            checkpoint_path.rename(args.output)
    else:
        convert_output(all_features, args.output, args.format)
        # Clean up checkpoint after successful conversion
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            print(f"Removed checkpoint file")

    print(f"Wrote {args.output}")


def convert_output(features: List[TabFeatures], path: Path, fmt: str) -> None:
    """Convert features to the requested output format."""
    if fmt == "csv":
        write_csv(features, path)
    elif fmt == "json":
        write_json(features, path)
    # jsonl is already written incrementally


def write_csv(features: List[TabFeatures], path: Path) -> None:
    if not features:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(features[0].to_dict().keys()))
        writer.writeheader()
        writer.writerows(feat.to_dict() for feat in features)


def write_json(features: List[TabFeatures], path: Path) -> None:
    with open(path, "w") as f:
        json.dump([feat.to_dict() for feat in features], f, indent=2)


if __name__ == "__main__":
    main()


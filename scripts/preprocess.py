#!/usr/bin/env python3
"""Preprocess audio + charts into .tab training format with source separation."""

import argparse
import atexit
import json
import logging
import multiprocessing
import os
import signal
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed, BrokenExecutor
from dataclasses import dataclass
from datetime import datetime
from multiprocessing import cpu_count
from pathlib import Path
from typing import List, Optional, Set

from tqdm import tqdm

# CRITICAL: Use 'spawn' instead of 'fork' for CUDA compatibility
# Fork + CUDA causes race conditions and crashes
# This must be set before any CUDA operations
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Already set

# Global executor reference for cleanup on interrupt
_active_executor: Optional[ProcessPoolExecutor] = None


def _cleanup_executor():
    """Force shutdown of executor and GPU processes on exit."""
    global _active_executor
    if _active_executor is not None:
        try:
            _active_executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
        _active_executor = None

    # Clear any lingering CUDA memory
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _signal_handler(signum, frame):
    """Handle interrupt signals gracefully."""
    print("\nInterrupt received, cleaning up...")
    _cleanup_executor()
    raise KeyboardInterrupt


# Register cleanup handlers
atexit.register(_cleanup_executor)
signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)

from tab_hero.dataio.preprocessing import (
    DIFFICULTY_MAP,
    INSTRUMENT_MAP,
    MEL_CONFIG,
    cleanup_source_directory,
    discover_song_directories,
    process_song_all_variants,
)
from tab_hero.dataio.source_separation import (
    MIN_STEM_RMS,
    check_demucs_available,
    find_preexisting_stems,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class HardwareInfo:
    """Detected hardware specifications."""
    cpu_cores: int
    cpu_cores_physical: int
    ram_gb: float
    gpu_available: bool
    gpu_name: Optional[str]
    gpu_memory_gb: Optional[float]
    disk_free_gb: float


@dataclass
class OptimalSettings:
    """Auto-tuned preprocessing settings."""
    workers: int
    workers_cpu_only: int
    use_gpu_mel: bool
    use_separation: bool
    manifest_save_interval: int
    reasoning: List[str]


def detect_hardware(output_dir: Path) -> HardwareInfo:
    """Detect CPU, RAM, GPU, and disk for optimal tuning."""
    import torch

    # CPU info - with WSL2 workaround
    cpu_cores = cpu_count()
    try:
        # Try to get physical cores (excluding hyperthreads)
        cpu_cores_physical = len(os.sched_getaffinity(0))
        # WSL2 sometimes reports only 1 core incorrectly
        # Fall back to /proc/cpuinfo if sched_getaffinity seems wrong
        if cpu_cores_physical == 1 and cpu_cores > 1:
            cpu_cores_physical = cpu_cores
        elif cpu_cores_physical == 1:
            # Try reading from /proc/cpuinfo for WSL2
            try:
                with open("/proc/cpuinfo") as f:
                    physical_ids = set()
                    core_ids = set()
                    current_phys = None
                    for line in f:
                        if line.startswith("physical id"):
                            current_phys = line.split(":")[1].strip()
                            physical_ids.add(current_phys)
                        elif line.startswith("core id") and current_phys is not None:
                            core_ids.add((current_phys, line.split(":")[1].strip()))
                    if core_ids:
                        cpu_cores_physical = len(core_ids)
                    elif physical_ids:
                        # Fallback: count processor entries
                        cpu_cores_physical = sum(1 for line in open("/proc/cpuinfo") if line.startswith("processor"))
            except Exception:
                pass
    except (AttributeError, OSError):
        cpu_cores_physical = cpu_cores

    # RAM info
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    except ImportError:
        # Fallback: read from /proc/meminfo on Linux
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        ram_kb = int(line.split()[1])
                        ram_gb = ram_kb / (1024 ** 2)
                        break
                else:
                    ram_gb = 16.0  # Assume 16GB if unknown
        except Exception:
            ram_gb = 16.0

    # GPU info
    gpu_available = torch.cuda.is_available()
    gpu_name = None
    gpu_memory_gb = None
    if gpu_available:
        try:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        except Exception:
            pass

    # Disk free space at output directory
    try:
        disk_stats = shutil.disk_usage(output_dir.parent if not output_dir.exists() else output_dir)
        disk_free_gb = disk_stats.free / (1024 ** 3)
    except Exception:
        disk_free_gb = 100.0  # Assume plenty if unknown

    return HardwareInfo(
        cpu_cores=cpu_cores,
        cpu_cores_physical=cpu_cores_physical,
        ram_gb=ram_gb,
        gpu_available=gpu_available,
        gpu_name=gpu_name,
        gpu_memory_gb=gpu_memory_gb,
        disk_free_gb=disk_free_gb,
    )


def compute_optimal_settings(
    hw: HardwareInfo,
    use_separation: bool,
    n_songs: int,
    n_songs_with_stems: int = 0,
) -> OptimalSettings:
    """Compute optimal settings based on hardware. Two phases: CPU-parallel for stems, single GPU for Demucs."""
    reasoning = []

    base_workers = hw.cpu_cores_physical
    reasoning.append(f"Base workers from {hw.cpu_cores_physical} physical cores")

    ram_per_worker_gb = 2.0
    max_workers_by_ram = max(1, int(hw.ram_gb / ram_per_worker_gb) - 1)
    if max_workers_by_ram < base_workers:
        reasoning.append(f"RAM limited: {hw.ram_gb:.1f}GB allows ~{max_workers_by_ram} workers")
        base_workers = max_workers_by_ram

    workers_cpu_only = max(1, min(base_workers, 16))

    use_gpu_mel = False
    workers_gpu = 1

    if use_separation and hw.gpu_available:
        use_gpu_mel = True
        reasoning.append(f"GPU detected: {hw.gpu_name} ({hw.gpu_memory_gb:.1f}GB VRAM)")
        workers_gpu = 1  # Single worker for GPU to avoid CUDA context issues
        reasoning.append(f"Two-phase optimization enabled:")
        reasoning.append(f"  Phase 1: {workers_cpu_only} workers for {n_songs_with_stems} songs with pre-existing stems")
        reasoning.append(f"  Phase 2: {workers_gpu} worker for {n_songs - n_songs_with_stems} songs needing Demucs")
    elif use_separation:
        reasoning.append("No GPU: Demucs will run on CPU (slow)")
        workers_cpu_only = min(workers_cpu_only, 4)
        workers_gpu = min(base_workers, 4)
    else:
        reasoning.append("Source separation disabled: CPU-only mel extraction")
        workers_gpu = workers_cpu_only

    manifest_save_interval = max(10, min(100, workers_cpu_only * 5))
    reasoning.append(f"Manifest save interval: every {manifest_save_interval} songs")

    return OptimalSettings(
        workers=workers_gpu,
        workers_cpu_only=workers_cpu_only,
        use_gpu_mel=use_gpu_mel,
        use_separation=use_separation,
        manifest_save_interval=manifest_save_interval,
        reasoning=reasoning,
    )


def print_hardware_info(hw: HardwareInfo) -> None:
    """Print detected hardware information."""
    print("\n" + "=" * 60)
    print("HARDWARE DETECTION")
    print("=" * 60)
    print(f"CPU: {hw.cpu_cores} logical cores ({hw.cpu_cores_physical} physical)")
    print(f"RAM: {hw.ram_gb:.1f} GB")
    if hw.gpu_available:
        print(f"GPU: {hw.gpu_name} ({hw.gpu_memory_gb:.1f} GB VRAM)")
    else:
        print("GPU: Not available")
    print(f"Disk free: {hw.disk_free_gb:.1f} GB")


def print_optimal_settings(settings: OptimalSettings) -> None:
    """Print auto-tuned settings with reasoning."""
    print("\n" + "=" * 60)
    print("AUTO-TUNED SETTINGS")
    print("=" * 60)
    print(f"Workers: {settings.workers}")
    print(f"GPU mel extraction: {'enabled' if settings.use_gpu_mel else 'disabled'}")
    print(f"Source separation: {'enabled' if settings.use_separation else 'disabled'}")
    print("\nReasoning:")
    for reason in settings.reasoning:
        print(f"  - {reason}")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess source data into .tab training format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
The .tab format is designed for training neural network weights only.
It stores lossy mel spectrograms and tokenized note sequences that
cannot be used to reconstruct the original audio or chart files.

Source separation (default, requires GPU):
  Uses HTDemucs to separate audio into instrument stems (guitar, bass, piano).
  Each chart gets a mel spectrogram from its corresponding instrument.
  This provides cleaner training signal and reduces storage by eliminating
  duplicated mel spectrograms across different instrument charts.

Examples:
  # Process with auto-tuned settings (recommended)
  python scripts/preprocess.py --input_dir data/raw --output_dir data/processed --auto

  # Process all songs with source separation
  python scripts/preprocess.py --input_dir data/raw --output_dir data/processed --workers 8

  # Process without source separation (faster, for testing or CPU-only)
  python scripts/preprocess.py --input_dir data/raw --output_dir data/processed --skip-separation

  # Process and remove source files
  python scripts/preprocess.py --input_dir data/raw --output_dir data/processed --cleanup

  # Process specific difficulties and instruments
  python scripts/preprocess.py --input_dir data/raw --output_dir data/processed \\
      --difficulties expert hard --instruments lead bass
        """
    )
    parser.add_argument("--input_dir", type=Path, required=True,
                        help="Root directory containing song folders")
    parser.add_argument("--output_dir", type=Path, required=True,
                        help="Output directory for .tab files")
    parser.add_argument("--difficulties", nargs="+",
                        default=["expert", "hard", "medium", "easy"],
                        choices=list(DIFFICULTY_MAP.keys()),
                        help="Difficulties to extract (default: all)")
    parser.add_argument("--instruments", nargs="+",
                        default=["lead", "bass", "rhythm", "keys"],
                        choices=list(INSTRUMENT_MAP.keys()),
                        help="Instruments to extract (default: all)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers (default: auto-detect or CPU count)")
    parser.add_argument("--physical-cores", type=int, default=None,
                        help="Override detected physical CPU cores (useful for WSL2 where detection fails)")
    parser.add_argument("--cleanup", action="store_true",
                        help="Remove source files after successful processing")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be processed without doing it")
    parser.add_argument("--resume", action="store_true",
                        help="Skip songs where output already exists (auto-enabled)")
    # Auto-tuning
    parser.add_argument("--auto", action="store_true",
                        help="Auto-detect hardware and tune settings for optimal performance")
    parser.add_argument("--show-hardware", action="store_true",
                        help="Show detected hardware info and recommended settings, then exit")
    # Source separation options
    parser.add_argument("--skip-separation", action="store_true",
                        help="Skip source separation (use mixed audio for all charts)")
    parser.add_argument("--stem-cache-dir", type=Path, default=None,
                        help="Directory to cache mel spectrograms (saves reprocessing)")
    parser.add_argument("--min-stem-rms", type=float, default=MIN_STEM_RMS,
                        help=f"Minimum RMS energy for valid stems (default: {MIN_STEM_RMS}). "
                             "Stems below this threshold are skipped as empty/silent.")
    # Quality filters (unset by default to process all songs)
    parser.add_argument("--min-notes", type=int, default=1,
                        help="Minimum notes required per chart (default: 1)")
    parser.add_argument("--min-duration", type=float, default=None,
                        help="Minimum song duration in seconds (default: no limit)")
    parser.add_argument("--max-duration", type=float, default=None,
                        help="Maximum song duration in seconds (default: no limit)")
    # Train/val split
    parser.add_argument("--val-split", type=float, default=0.05,
                        help="Fraction of songs for validation set (default: 0.05)")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Discover all song directories recursively
    print(f"Scanning {args.input_dir} for song directories...")
    song_dirs = discover_song_directories(args.input_dir)
    print(f"Found {len(song_dirs)} song directories")

    if args.dry_run:
        print("Dry run - no files will be processed or removed")
        for sd in song_dirs[:10]:
            print(f"  Would process: {sd}")
        if len(song_dirs) > 10:
            print(f"  ... and {len(song_dirs) - 10} more")
        return

    # Load progress manifest for resume support
    manifest_path = args.output_dir / ".progress_manifest.json"
    processed_manifest: Set[str] = set()
    song_id_map: dict[str, int] = {}  # song_dir -> song_id mapping
    next_song_id = 1  # Start at 1 (0 reserved for legacy/unknown)
    if manifest_path.exists():
        try:
            with open(manifest_path, "r") as f:
                manifest_data = json.load(f)
                
                processed_manifest = set(manifest_data.get("processed", []))
                song_id_map = manifest_data.get("song_id_map", {})
                next_song_id = manifest_data.get("next_song_id", 1)
            print(f"Loaded progress manifest: {len(processed_manifest)} songs already processed")
            if song_id_map:
                print(f"  Song ID mappings: {len(song_id_map)} (next_id={next_song_id})")
        except Exception as e:
            print(f"Warning: Could not load manifest: {e}")

    # Filter out already-processed songs for fast resume
    pending_dirs = [sd for sd in song_dirs if str(sd) not in processed_manifest]
    already_done = len(song_dirs) - len(pending_dirs)
    if already_done > 0:
        print(f"Resuming: skipping {already_done} already-processed songs")

    # Determine source separation mode (initial check)
    use_separation = not args.skip_separation
    if use_separation and not check_demucs_available():
        print("Source separation: DISABLED (demucs not installed)")
        print("  Install with: pip install demucs")
        use_separation = False

    # Partition songs by whether they have pre-existing stems
    # This enables two-phase optimization: parallel CPU for stems, single GPU for Demucs
    print("Analyzing songs for two-phase optimization...")
    songs_with_stems = []
    songs_need_demucs = []
    for song_dir in pending_dirs:
        if find_preexisting_stems(song_dir):
            songs_with_stems.append(song_dir)
        else:
            songs_need_demucs.append(song_dir)
    print(f"  Songs with pre-existing stems (fast, parallel): {len(songs_with_stems)}")
    print(f"  Songs needing Demucs (GPU, sequential): {len(songs_need_demucs)}")

    # Hardware detection and auto-tuning
    use_gpu_mel = True  # Default
    manifest_save_interval = 50  # Default
    n_workers_cpu = cpu_count()  # Default for CPU-only phase
    n_workers_gpu = 1  # Default for GPU phase

    if args.auto or args.show_hardware:
        hw = detect_hardware(args.output_dir)
        # Apply physical cores override if specified (for WSL2 workaround)
        if args.physical_cores:
            hw = HardwareInfo(
                cpu_cores=hw.cpu_cores,
                cpu_cores_physical=args.physical_cores,
                ram_gb=hw.ram_gb,
                gpu_available=hw.gpu_available,
                gpu_name=hw.gpu_name,
                gpu_memory_gb=hw.gpu_memory_gb,
                disk_free_gb=hw.disk_free_gb,
            )
        print_hardware_info(hw)

        optimal = compute_optimal_settings(
            hw=hw,
            use_separation=use_separation,
            n_songs=len(pending_dirs),
            n_songs_with_stems=len(songs_with_stems),
        )
        print_optimal_settings(optimal)

        if args.show_hardware:
            return

        # Apply auto-tuned settings
        n_workers_cpu = args.workers if args.workers else optimal.workers_cpu_only
        n_workers_gpu = optimal.workers
        use_gpu_mel = optimal.use_gpu_mel
        manifest_save_interval = optimal.manifest_save_interval

        # Auto-enable mel cache if not specified
        if args.stem_cache_dir is None and use_separation:
            args.stem_cache_dir = args.output_dir / ".mel_cache"
            args.stem_cache_dir.mkdir(parents=True, exist_ok=True)
            print(f"Auto-enabled mel cache: {args.stem_cache_dir}")
    else:
        # Manual mode: use specified workers or CPU count
        n_workers_cpu = args.workers if args.workers else cpu_count()
        n_workers_gpu = 1  # Always 1 for GPU safety

    # Print separation status
    if use_separation:
        print("Source separation: ENABLED (htdemucs_6s)")
        if args.stem_cache_dir:
            args.stem_cache_dir.mkdir(parents=True, exist_ok=True)
            print(f"  Mel cache: {args.stem_cache_dir}")
    else:
        print("Source separation: DISABLED")

    # Build filters dict
    filters = {
        "min_notes": args.min_notes,
        "min_duration": args.min_duration,
        "max_duration": args.max_duration,
        "use_separation": use_separation,
        "stem_cache_dir": args.stem_cache_dir,
        "min_stem_rms": args.min_stem_rms,
        "use_gpu_mel": use_gpu_mel,
    }

    print(f"Stem RMS threshold: {args.min_stem_rms} (stems below this are skipped)")
    print(f"GPU mel extraction: {'enabled' if use_gpu_mel else 'disabled'}")

    # Assign song_ids to pending songs (preserving existing mappings for resume)
    for song_dir in pending_dirs:
        song_key = str(song_dir)
        if song_key not in song_id_map:
            song_id_map[song_key] = next_song_id
            next_song_id += 1

    # Build task lists for two-phase processing
    # Phase 1: Songs with pre-existing stems (CPU-only, parallel)
    # Phase 2: Songs needing Demucs (GPU, sequential)
    def make_task(song_dir, use_gpu):
        """Create a task tuple with song_id in filters."""
        task_filters = filters.copy()
        task_filters["use_gpu_mel"] = use_gpu
        task_filters["song_id"] = song_id_map.get(str(song_dir), 0)
        return (song_dir, args.output_dir, args.difficulties, args.instruments, task_filters)

    tasks_phase1 = [make_task(song_dir, use_gpu=False)
                    for song_dir in songs_with_stems if str(song_dir) not in processed_manifest]
    tasks_phase2 = [make_task(song_dir, use_gpu=True)
                    for song_dir in songs_need_demucs if str(song_dir) not in processed_manifest]

    n_variants = len(args.difficulties) * len(args.instruments)
    print(f"\nProcessing {len(pending_dirs)} songs x {n_variants} variants each")
    if args.min_duration or args.max_duration:
        print(f"Duration filters: min={args.min_duration}s, max={args.max_duration}s")
    if args.min_notes > 1:
        print(f"Minimum notes per chart: {args.min_notes}")

    if len(pending_dirs) == 0:
        print("All songs already processed!")
        return

    # Process in parallel with progress bar
    successful = 0
    skipped = 0  # Expected: missing difficulty/instrument
    errors = 0   # Unexpected errors
    worker_crashes = 0  # Track worker crashes separately
    processed_dirs: Set[str] = set()
    error_samples: List[str] = []
    last_manifest_save = 0  # Track when we last saved

    # manifest_save_interval is set above (auto-tuned or default 50)
    # But save more aggressively at the start to ensure resume works
    early_save_threshold = 5  # Save after first 5 songs regardless

    def save_manifest():
        """Save progress manifest to allow resume (v2 format with song_id tracking)."""
        nonlocal last_manifest_save
        manifest_data = {
            "processed": list(processed_manifest),
            "song_id_map": song_id_map,
            "next_song_id": next_song_id,
        }
        with open(manifest_path, "w") as f:
            json.dump(manifest_data, f)
        last_manifest_save = len(processed_manifest)

    def process_batch(batch_tasks, pbar, num_workers):
        """Process a batch of tasks with graceful error handling."""
        global _active_executor
        nonlocal successful, skipped, errors, worker_crashes

        # Note: max_tasks_per_child was removed - it caused worker restart issues
        # where new workers failed to properly initialize CUDA/Demucs.
        # Memory leaks are now handled via explicit cleanup in processing functions.
        executor = ProcessPoolExecutor(max_workers=num_workers)
        _active_executor = executor  # Register for cleanup on interrupt

        try:
            futures = {executor.submit(process_song_all_variants, task): task for task in batch_tasks}

            for future in as_completed(futures):
                task = futures[future]
                song_dir = task[0]  # First element is song_dir

                try:
                    result = future.result(timeout=600)  # 10 min timeout per song
                    successful += result["successful"]
                    skipped += result["skipped"]
                    errors += result["errors"]
                    processed_manifest.add(result["song_dir"])
                    if result["successful"] > 0:
                        processed_dirs.add(result["song_dir"])
                    if result["error_msgs"] and len(error_samples) < 10:
                        for msg in result["error_msgs"]:
                            error_samples.append(f"{Path(result['song_dir']).name}: {msg}")

                except BrokenExecutor:
                    # Worker pool crashed - this is fatal for this batch
                    worker_crashes += 1
                    processed_manifest.add(str(song_dir))  # Mark as attempted
                    error_samples.append(f"{song_dir.name}: Worker pool crashed")
                    save_manifest()  # Save progress immediately
                    raise  # Re-raise to trigger batch restart

                except TimeoutError:
                    errors += 1
                    processed_manifest.add(str(song_dir))
                    if len(error_samples) < 10:
                        error_samples.append(f"{song_dir.name}: Timeout (>10 min)")

                except Exception as e:
                    errors += 1
                    processed_manifest.add(str(song_dir))
                    if len(error_samples) < 10:
                        error_samples.append(f"{song_dir.name}: {type(e).__name__}: {e}")

                pbar.update(1)
                pbar.set_postfix({
                    "ok": successful,
                    "skip": skipped,
                    "err": errors,
                    "crash": worker_crashes if worker_crashes else None
                })

                # Save manifest: aggressively at start, then periodically
                songs_since_save = len(processed_manifest) - last_manifest_save
                should_save = (
                    len(processed_manifest) <= early_save_threshold or  # First few songs
                    songs_since_save >= manifest_save_interval  # Normal interval
                )
                if should_save:
                    save_manifest()
        finally:
            # Always clean up executor to prevent orphaned GPU processes
            executor.shutdown(wait=False, cancel_futures=True)
            _active_executor = None

    # Create initial empty manifest to ensure file exists for resume
    save_manifest()

    def run_phase(tasks, phase_name, num_workers):
        """Run a processing phase with the given tasks and worker count."""
        if not tasks:
            return

        batch_size = min(500, max(100, len(tasks) // 10))
        remaining = tasks.copy()

        print(f"\n{'='*60}")
        print(f"{phase_name}: {len(tasks)} songs with {num_workers} worker(s)")
        print(f"{'='*60}")

        with tqdm(total=len(tasks), desc=phase_name, unit="song") as pbar:
            while remaining:
                remaining = [t for t in remaining if str(t[0]) not in processed_manifest]
                if not remaining:
                    break

                batch = remaining[:batch_size]
                remaining = remaining[batch_size:]

                try:
                    process_batch(batch, pbar, num_workers)
                except BrokenExecutor:
                    print(f"\nWorker crash detected. Saving progress and continuing...")
                    save_manifest()
                    batch_size = max(50, batch_size // 2)
                    print(f"Reduced batch size to {batch_size}")
                    for task in batch:
                        if str(task[0]) not in processed_manifest:
                            remaining.insert(0, task)
                    continue
                except KeyboardInterrupt:
                    print(f"\nInterrupted. Saving progress...")
                    save_manifest()
                    raise

    # PHASE 1: Songs with pre-existing stems (parallel CPU workers)
    run_phase(tasks_phase1, "Phase 1 (pre-existing stems)", n_workers_cpu)

    # PHASE 2: Songs needing Demucs (sequential GPU worker)
    run_phase(tasks_phase2, "Phase 2 (Demucs separation)", n_workers_gpu)

    # Final manifest save
    save_manifest()

    print(f"\nCompleted: {successful} .tab files, {skipped} skipped, {errors} errors")
    if worker_crashes > 0:
        print(f"Worker crashes recovered: {worker_crashes}")
    print(f"Processed {len(processed_dirs)} songs with at least one valid chart")

    if error_samples:
        print("\nSample errors:")
        for err in error_samples[:5]:
            print(f"  {err}")

    # Cleanup source files if requested
    if args.cleanup and processed_dirs:
        print(f"Removing source files from {len(processed_dirs)} directories...")
        for song_dir_str in tqdm(processed_dirs, desc="Cleanup"):
            cleanup_source_directory(Path(song_dir_str))
        print("Cleanup complete")

    # Collect all .tab files and create train/val split
    import random
    all_tab_files = sorted([f.stem for f in args.output_dir.glob("*.tab")])
    random.seed(42)  # Reproducible split
    random.shuffle(all_tab_files)

    n_val = int(len(all_tab_files) * args.val_split)
    val_files = set(all_tab_files[:n_val])
    train_files = set(all_tab_files[n_val:])

    print(f"\nTrain/val split: {len(train_files)} train, {len(val_files)} val "
          f"({args.val_split * 100:.1f}% val)")

    # Write processing manifest with full config for reproducibility
    final_manifest_path = args.output_dir / "manifest.json"
    manifest = {
        "created": datetime.now().isoformat(),
        "format_version": 3,  # Bumped for source separation
        "total_files": len(all_tab_files),
        "train_files": len(train_files),
        "val_files": len(val_files),
        "val_split": args.val_split,
        "difficulties": args.difficulties,
        "instruments": args.instruments,
        "source_cleaned": args.cleanup,
        "source_separation": use_separation,
        "separation_model": "htdemucs_6s" if use_separation else None,
        "mel_config": MEL_CONFIG,  # Store audio processing parameters
        "filters": {
            "min_notes": args.min_notes,
            "min_duration": args.min_duration,
            "max_duration": args.max_duration,
        },
        "train": sorted(train_files),
        "val": sorted(val_files),
    }
    with open(final_manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Manifest written to {final_manifest_path}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Train a tab generation model.

Usage:
    python scripts/train.py
    python scripts/train.py data.max_samples=100
    python scripts/train.py +model=large  # Use large model config
    python scripts/train.py +model=small  # Use small model config
"""

import logging
import sys
from collections import Counter

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.data import DataLoader

from tab_hero.dataio.dataset import TabHeroDataset, collate_fn
from tab_hero.dataio.tab_dataset import TabDataset, tab_collate_fn
from tab_hero.dataio.chunked_dataset import ChunkedTabDataset, chunked_collate_fn
from tab_hero.dataio.tokenizer import ChartTokenizer
from tab_hero.dataio.samplers import WeightedInstrumentSampler, CurriculumSampler
from tab_hero.dataio.tab_format import peek_tab_header
from tab_hero.model.chart_transformer import ChartTransformer
from tab_hero.training.trainer import Trainer


# Custom handler that flushes after each log message
class FlushHandler(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()


# Configure logging with immediate flush
handler = FlushHandler(sys.stderr)
handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
logging.root.handlers = []
logging.root.addHandler(handler)
logging.root.setLevel(logging.INFO)
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg: DictConfig) -> None:
    logger.info("Configuration:\n" + OmegaConf.to_yaml(cfg))

    # Set device
    device = cfg.hardware.device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = "cpu"
    logger.info(f"Using device: {device}")

    # Create tokenizer
    tokenizer = ChartTokenizer()
    logger.info(f"Tokenizer vocab size: {tokenizer.vocab_size}")

    # Create dataset - use preprocessed .tab files if available
    use_tab_format = cfg.data.get("use_tab_format", False)
    use_chunked = cfg.data.get("use_chunked_dataset", False)
    data_dir = cfg.data.data_dir

    val_dataset = None

    if use_chunked and use_tab_format:
        # Use chunked dataset for long songs (recommended for production)
        logger.info(f"Loading chunked dataset from {data_dir}")
        train_dataset = ChunkedTabDataset(
            data_dir=data_dir,
            split=cfg.data.get("train_split", "train"),
            max_mel_frames=cfg.data.get("max_mel_frames", 8192),
            max_token_length=cfg.data.get("max_sequence_length", 4096),
            chunk_overlap_frames=cfg.data.get("chunk_overlap_frames", 512),
            audio_downsample=cfg.model.get("audio_downsample", 4),
            training=True,
        )
        # Create validation dataset
        val_split = cfg.data.get("val_split", "val")
        if val_split:
            val_dataset = ChunkedTabDataset(
                data_dir=data_dir,
                split=val_split,
                max_mel_frames=cfg.data.get("max_mel_frames", 8192),
                max_token_length=cfg.data.get("max_sequence_length", 4096),
                chunk_overlap_frames=cfg.data.get("chunk_overlap_frames", 512),
                audio_downsample=cfg.model.get("audio_downsample", 4),
                training=False,
            )
            logger.info(f"Validation samples: {len(val_dataset)}")
        collate = chunked_collate_fn
    elif use_tab_format:
        # Use preprocessed .tab files (faster, recommended)
        logger.info(f"Loading preprocessed .tab files from {data_dir}")
        train_dataset = TabDataset(
            data_dir=data_dir,
            max_mel_frames=cfg.data.get("max_mel_frames", 8192),
            max_token_length=cfg.data.get("max_sequence_length", 4096),
            training=True,
        )
        collate = tab_collate_fn
    else:
        # Use raw audio + chart files (slower, processes on-the-fly)
        logger.info(f"Loading raw data from {data_dir}")
        train_dataset = TabHeroDataset(
            data_dir=data_dir,
            instrument=cfg.data.instrument,
            difficulty=cfg.data.difficulty,
            max_audio_duration_s=cfg.data.get("max_audio_duration_s", 60.0),
            max_sequence_length=cfg.data.get("max_sequence_length", 4096),
            device="cpu",
            tokenizer=tokenizer,
        )
        collate = collate_fn

    # Limit samples for testing
    max_samples = cfg.data.get("max_samples")
    if max_samples:
        if hasattr(train_dataset, "samples"):
            train_dataset.samples = train_dataset.samples[:max_samples]
        elif hasattr(train_dataset, "tab_files"):
            # For ChunkedTabDataset: limit files first, then rebuild chunks
            train_dataset.tab_files = train_dataset.tab_files[:max_samples]
            if hasattr(train_dataset, "_build_chunk_index"):
                # Rebuild chunk index with limited files
                train_dataset.chunks = train_dataset._build_chunk_index()
        logger.info(f"Limited to {len(train_dataset)} samples")

    # Get batch size from config (may be overridden by model config)
    batch_size = cfg.data.get("batch_size", 8)
    if hasattr(cfg, "training") and cfg.training.get("batch_size"):
        batch_size = cfg.training.batch_size

    # Optionally use weighted instrument sampling to address class imbalance
    train_sampler = None
    if use_tab_format and cfg.data.get("weighted_instrument_sampling", False):
        logger.info("Building weighted instrument sampler...")
        if hasattr(train_dataset, "file_metadata") and train_dataset.file_metadata:
            # Use cached metadata from ChunkedTabDataset (no extra I/O)
            if hasattr(train_dataset, "chunks"):
                instrument_ids = [
                    train_dataset.file_metadata[c[0]].get("instrument_id", 0)
                    for c in train_dataset.chunks
                ]
            else:
                instrument_ids = [
                    m.get("instrument_id", 0) for m in train_dataset.file_metadata
                ]
        else:
            # Fallback: read headers directly
            instrument_ids = [
                peek_tab_header(f)["instrument_id"] for f in train_dataset.tab_files
            ]
        train_sampler = WeightedInstrumentSampler(instrument_ids)
        logger.info(f"Weighted sampler: {len(instrument_ids)} samples, "
                     f"instrument distribution: {dict(Counter(instrument_ids))}")

    # Optionally use curriculum learning (difficulty-based progressive training)
    # Note: curriculum sampler overrides weighted instrument sampler if both enabled.
    if use_tab_format and cfg.data.get("curriculum_learning", False):
        logger.info("Building curriculum sampler...")
        if hasattr(train_dataset, "file_metadata") and train_dataset.file_metadata:
            if hasattr(train_dataset, "chunks"):
                difficulty_ids = [
                    train_dataset.file_metadata[c[0]].get("difficulty_id", 0)
                    for c in train_dataset.chunks
                ]
            else:
                difficulty_ids = [
                    m.get("difficulty_id", 0) for m in train_dataset.file_metadata
                ]
        else:
            difficulty_ids = [
                peek_tab_header(f)["difficulty_id"] for f in train_dataset.tab_files
            ]
        schedule = cfg.data.get("curriculum_schedule", [[0, 1], [10, 2], [25, 3]])
        schedule = [tuple(s) for s in schedule]
        train_sampler = CurriculumSampler(difficulty_ids, schedule=schedule)
        logger.info(f"Curriculum sampler: {len(difficulty_ids)} samples, schedule={schedule}")

    num_workers = cfg.data.get("num_workers", 0)
    persistent = cfg.data.get("persistent_workers", False) and num_workers > 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=device == "cuda",
        prefetch_factor=cfg.data.get("prefetch_factor", 2) if num_workers > 0 else None,
        persistent_workers=persistent,
    )

    logger.info(f"Training samples: {len(train_dataset)}, batch_size: {batch_size}")

    # Create validation dataloader if validation dataset exists
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,  # No need to shuffle validation
            num_workers=num_workers,
            collate_fn=collate,
            pin_memory=device == "cuda",
            prefetch_factor=cfg.data.get("prefetch_factor", 2) if num_workers > 0 else None,
            persistent_workers=persistent,
        )
        logger.info(f"Validation batches: {len(val_loader)}")

    # Create model with new architecture features
    logger.info("Creating model")
    model = ChartTransformer(
        vocab_size=tokenizer.vocab_size,
        audio_input_dim=cfg.model.get("audio_input_dim", 128),
        encoder_dim=cfg.model.encoder_dim,
        decoder_dim=cfg.model.decoder_dim,
        n_decoder_layers=cfg.model.n_decoder_layers,
        n_heads=cfg.model.n_heads,
        ffn_dim=cfg.model.ffn_dim,
        max_seq_len=cfg.model.get("max_seq_len", 8192),
        dropout=cfg.model.dropout,
        audio_downsample=cfg.model.get("audio_downsample", 4),
        use_flash=cfg.model.get("use_flash", True),
        use_rope=cfg.model.get("use_rope", True),
        gradient_checkpointing=cfg.model.get("gradient_checkpointing", False),
    )

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    n_params_non_emb = model.get_num_params(non_embedding=True)
    logger.info(f"Model parameters: {n_params:,} (non-embedding: {n_params_non_emb:,})")

    # Optional: compile model for speedup
    if cfg.hardware.get("compile", False) and hasattr(torch, "compile"):
        logger.info("Compiling model with torch.compile")
        model = torch.compile(model)

    # Create trainer with new efficiency features
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        max_epochs=cfg.training.max_epochs,
        checkpoint_dir=cfg.training.checkpoint_dir,
        device=device,
        # New efficiency parameters
        gradient_accumulation_steps=cfg.training.get("gradient_accumulation_steps", 1),
        precision=cfg.hardware.get("precision", "bf16-mixed"),
        max_grad_norm=cfg.training.get("gradient_clip", 1.0),
        warmup_steps=cfg.training.get("warmup_steps", 1000),
        use_onecycle_lr=cfg.training.get("use_onecycle_lr", False),
        log_every_n_steps=cfg.training.get("log_every_n_steps", 100),
        # Checkpoint management
        keep_top_k_checkpoints=cfg.training.get("keep_top_k_checkpoints", 5),
        early_stopping_patience=cfg.training.get("early_stopping_patience", 0),
    )

    # Resume from checkpoint if specified
    resume_checkpoint = cfg.training.get("resume_checkpoint", None)
    if resume_checkpoint:
        from pathlib import Path
        checkpoint_path = Path(cfg.training.checkpoint_dir) / resume_checkpoint
        if checkpoint_path.exists():
            logger.info(f"Resuming from checkpoint: {checkpoint_path}")
            trainer.load_checkpoint(str(checkpoint_path))
        else:
            logger.error(f"Checkpoint not found: {checkpoint_path}")
            return

    # Train
    logger.info("Starting training")
    trainer.train()

    logger.info("Training complete")


if __name__ == "__main__":
    main()

"""Training loop for Tab Hero models."""

from pathlib import Path
from typing import Dict, Literal, Optional
import logging
import os
import sys

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from tqdm import tqdm

from ..model.chart_transformer import ChartTransformer
from .losses import ChartLoss

logger = logging.getLogger(__name__)


def log_with_flush(message: str, level: int = logging.INFO) -> None:
    """Log a message to both stderr and a dedicated log file in the project root."""
    from datetime import datetime
    import os

    # Use absolute path to project root for log file
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    log_file = os.path.join(project_root, "training_progress.log")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_msg = f"[{timestamp}] {message}"

    # Write to dedicated log file with immediate flush
    with open(log_file, "a") as f:
        f.write(formatted_msg + "\n")
        f.flush()
        os.fsync(f.fileno())  # Force OS-level flush

    # Also print to stderr
    print(formatted_msg, file=sys.stderr, flush=True)

PrecisionType = Literal["fp32", "fp16", "bf16", "bf16-mixed"]


class Trainer:
    """Handles training, validation, checkpointing, and logging."""

    def __init__(
        self,
        model: ChartTransformer,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        max_epochs: int = 100,
        checkpoint_dir: str = "checkpoints",
        device: str = "cuda",
        gradient_accumulation_steps: int = 1,
        precision: PrecisionType = "bf16-mixed",
        max_grad_norm: float = 1.0,
        warmup_steps: int = 1000,
        use_onecycle_lr: bool = False,
        log_every_n_steps: int = 100,
        # Checkpoint management
        keep_top_k_checkpoints: int = 5,
        early_stopping_patience: int = 0,  # 0 = disabled
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.max_epochs = max_epochs
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.precision = precision
        self.max_grad_norm = max_grad_norm
        self.warmup_steps = warmup_steps
        self.log_every_n_steps = log_every_n_steps

        self._setup_precision()

        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.95),  # Standard for transformers
            eps=1e-8,
        )

        total_steps = len(train_loader) * max_epochs // gradient_accumulation_steps

        if use_onecycle_lr:
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=learning_rate,
                total_steps=total_steps,
                pct_start=warmup_steps / total_steps if warmup_steps > 0 else 0.1,
                anneal_strategy='cos',
            )
            self.scheduler_step_per_batch = True
        else:
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=max_epochs,
                eta_min=learning_rate * 0.01,
            )
            self.scheduler_step_per_batch = False

        self.criterion = ChartLoss()

        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')

        # Checkpoint management: keep top-k checkpoints by val_loss
        self.keep_top_k_checkpoints = keep_top_k_checkpoints
        self.top_checkpoints: list[tuple[float, str]] = []  # (val_loss, filename)

        self.early_stopping_patience = early_stopping_patience
        self.epochs_without_improvement = 0

        self._log_config(learning_rate, weight_decay)

    def _setup_precision(self):
        """Configure mixed precision training."""
        self.use_amp = self.precision in ("fp16", "bf16", "bf16-mixed")

        if self.precision == "bf16-mixed":
            # bf16 doesn't need loss scaling
            self.amp_dtype = torch.bfloat16
            self.scaler = None  # bf16 is numerically stable
        elif self.precision == "bf16":
            self.amp_dtype = torch.bfloat16
            self.scaler = None
        elif self.precision == "fp16":
            self.amp_dtype = torch.float16
            # fp16 needs GradScaler for numerical stability
            self.scaler = torch.cuda.amp.GradScaler()
        else:  # fp32
            self.amp_dtype = torch.float32
            self.scaler = None

        # Check hardware support for bf16
        if self.amp_dtype == torch.bfloat16:
            if not torch.cuda.is_available():
                logger.warning("CUDA not available, falling back to fp32")
                self.amp_dtype = torch.float32
                self.use_amp = False
            elif not torch.cuda.is_bf16_supported():
                logger.warning("bf16 not supported on this GPU, falling back to fp16")
                self.amp_dtype = torch.float16
                self.scaler = torch.cuda.amp.GradScaler()

    def _log_config(self, learning_rate: float, weight_decay: float):
        """Log trainer configuration."""
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Trainer configuration:")
        logger.info(f"  Model parameters: {n_params:,}")
        logger.info(f"  Precision: {self.precision}")
        logger.info(f"  Gradient accumulation: {self.gradient_accumulation_steps}")
        logger.info(f"  Effective batch size: {len(self.train_loader.dataset) // len(self.train_loader) * self.gradient_accumulation_steps}")
        logger.info(f"  Learning rate: {learning_rate}")
        logger.info(f"  Weight decay: {weight_decay}")

    def train_epoch(self) -> Dict[str, float]:
        """
        Run one training epoch with mixed precision and gradient accumulation.
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        accumulated_loss = 0.0
        epoch_step = 0  # Track steps within this epoch for accurate loss logging

        self.optimizer.zero_grad()

        is_file_output = not sys.stderr.isatty()
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch}",
            file=sys.stderr,
            dynamic_ncols=not is_file_output,
            mininterval=30.0 if is_file_output else 0.1,  # Log every 30s to file
            disable=False,
        )

        for batch_idx, batch in enumerate(pbar):
            audio_embeddings = batch["audio_embeddings"].to(self.device)
            note_tokens = batch["note_tokens"].to(self.device)
            difficulty_id = batch.get("difficulty_id")
            instrument_id = batch.get("instrument_id")
            audio_mask = batch.get("audio_mask")

            if difficulty_id is not None:
                difficulty_id = difficulty_id.to(self.device)
            if instrument_id is not None:
                instrument_id = instrument_id.to(self.device)
            if audio_mask is not None:
                audio_mask = audio_mask.to(self.device)

            with torch.amp.autocast('cuda', enabled=self.use_amp, dtype=self.amp_dtype):
                outputs = self.model(
                    audio_embeddings, note_tokens, difficulty_id, instrument_id,
                    audio_mask=audio_mask,
                )
                # Scale loss for gradient accumulation
                loss = outputs["loss"] / self.gradient_accumulation_steps

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            accumulated_loss += loss.item() * self.gradient_accumulation_steps
            n_batches += 1

            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )
                    self.optimizer.step()

                # Step scheduler if per-batch
                if self.scheduler_step_per_batch:
                    self.scheduler.step()

                self.optimizer.zero_grad()
                self.global_step += 1
                epoch_step += 1

                total_loss += accumulated_loss

                avg_loss = total_loss / epoch_step
                lr = self.optimizer.param_groups[0]["lr"]
                pbar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'lr': f'{lr:.2e}',
                    'step': self.global_step
                })

                if self.global_step % self.log_every_n_steps == 0:
                    log_with_flush(
                        f"Epoch {self.current_epoch} Step {self.global_step}: "
                        f"loss={avg_loss:.4f}, lr={lr:.2e}"
                    )

                accumulated_loss = 0.0

        # Handle remaining gradients (incomplete accumulation at end)
        remaining_batches = n_batches % self.gradient_accumulation_steps
        if remaining_batches > 0:
            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
                self.optimizer.step()

            self.optimizer.zero_grad()
            self.global_step += 1
            total_loss += accumulated_loss

        avg_loss = total_loss / max(n_batches // self.gradient_accumulation_steps, 1)
        return {"train_loss": avg_loss, "global_step": self.global_step}

    def validate(self) -> Dict[str, float]:
        """Run validation with mixed precision."""
        if self.val_loader is None:
            return {}

        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        is_file_output = not sys.stderr.isatty()

        pbar = tqdm(
            self.val_loader,
            desc=f"Validating",
            file=sys.stderr,
            dynamic_ncols=not is_file_output,
            mininterval=30.0 if is_file_output else 0.1,
            disable=False,
        )

        with torch.no_grad():
            for batch in pbar:
                audio_embeddings = batch["audio_embeddings"].to(self.device)
                note_tokens = batch["note_tokens"].to(self.device)
                difficulty_id = batch.get("difficulty_id")
                instrument_id = batch.get("instrument_id")
                audio_mask = batch.get("audio_mask")

                if difficulty_id is not None:
                    difficulty_id = difficulty_id.to(self.device)
                if instrument_id is not None:
                    instrument_id = instrument_id.to(self.device)
                if audio_mask is not None:
                    audio_mask = audio_mask.to(self.device)

                with torch.amp.autocast('cuda', enabled=self.use_amp, dtype=self.amp_dtype):
                    outputs = self.model(
                        audio_embeddings, note_tokens, difficulty_id, instrument_id,
                        audio_mask=audio_mask,
                    )

                total_loss += outputs["loss"].item()
                n_batches += 1

                pbar.set_postfix({'val_loss': f'{total_loss / n_batches:.4f}'})

        avg_loss = total_loss / max(n_batches, 1)
        return {"val_loss": avg_loss}

    def _get_base_model(self):
        """Return the underlying ChartTransformer, unwrapping torch.compile if needed."""
        # torch.compile wraps the model; the original is accessible via _orig_mod
        return getattr(self.model, "_orig_mod", self.model)

    def save_checkpoint(self, filename: str = "checkpoint.pt"):
        """Save model checkpoint with training state."""
        base_model = self._get_base_model()
        model_config = getattr(base_model, "model_config", None)

        checkpoint = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "precision": self.precision,
        }
        if model_config is not None:
            checkpoint["config"] = model_config

        # Save scaler state if using fp16
        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        torch.save(checkpoint, self.checkpoint_dir / filename)
        logger.info(f"Saved checkpoint to {self.checkpoint_dir / filename}")

    def _save_top_k_checkpoint(self, val_loss: float):
        """Save checkpoint if it's in the top-k by validation loss.

        Maintains a list of top-k checkpoints sorted by val_loss (ascending).
        Removes the worst checkpoint when exceeding k.
        """
        filename = f"checkpoint_epoch{self.current_epoch}_loss{val_loss:.4f}.pt"

        if len(self.top_checkpoints) < self.keep_top_k_checkpoints:
            self.save_checkpoint(filename)
            self.top_checkpoints.append((val_loss, filename))
            self.top_checkpoints.sort(key=lambda x: x[0])
            logger.info(f"Saved top-k checkpoint: {filename} (val_loss={val_loss:.4f})")
        elif val_loss < self.top_checkpoints[-1][0]:
            worst_loss, worst_filename = self.top_checkpoints.pop()
            worst_path = self.checkpoint_dir / worst_filename
            if worst_path.exists():
                worst_path.unlink()
                logger.info(f"Removed checkpoint: {worst_filename} (val_loss={worst_loss:.4f})")

            self.save_checkpoint(filename)
            self.top_checkpoints.append((val_loss, filename))
            self.top_checkpoints.sort(key=lambda x: x[0])
            logger.info(f"Saved top-k checkpoint: {filename} (val_loss={val_loss:.4f})")

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint and resume training state.

        Tolerates architecture differences between the checkpoint and the
        current model (e.g. when a module is removed between runs).  Any
        keys present in the checkpoint but absent from the model are logged
        as warnings and skipped; any keys present in the model but absent
        from the checkpoint are logged and left at their randomly-initialised
        values.
        """
        import re

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        saved_state = checkpoint["model_state_dict"]
        model_state = self.model.state_dict()

        unexpected = [k for k in saved_state if k not in model_state]
        missing    = [k for k in model_state if k not in saved_state]

        if unexpected:
            logger.warning(
                f"Checkpoint contains {len(unexpected)} key(s) not present in "
                f"the current model (will be ignored): {unexpected}"
            )
        if missing:
            logger.warning(
                f"Current model has {len(missing)} key(s) not present in the "
                f"checkpoint (will use random init): {missing}"
            )

        self.model.load_state_dict(saved_state, strict=False)
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        # Resume from NEXT epoch since checkpoint was saved after epoch completed
        self.current_epoch = checkpoint["epoch"] + 1
        self.global_step = checkpoint.get("global_step", 0)
        self.best_val_loss = checkpoint["best_val_loss"]

        # Load scaler state if present
        if self.scaler is not None and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        # Restore top_checkpoints list from existing checkpoint files
        self.top_checkpoints = []
        pattern = re.compile(r"checkpoint_epoch(\d+)_loss([\d.]+)\.pt")
        for f in self.checkpoint_dir.iterdir():
            match = pattern.match(f.name)
            if match:
                val_loss = float(match.group(2))
                self.top_checkpoints.append((val_loss, f.name))
        self.top_checkpoints.sort(key=lambda x: x[0])
        # Keep only top k
        while len(self.top_checkpoints) > self.keep_top_k_checkpoints:
            worst_loss, worst_filename = self.top_checkpoints.pop()
            worst_path = self.checkpoint_dir / worst_filename
            if worst_path.exists():
                worst_path.unlink()
                logger.info(f"Cleaned up excess checkpoint: {worst_filename}")

        logger.info(f"Resumed from {checkpoint_path}, starting at epoch {self.current_epoch}")
        logger.info(f"Best val_loss so far: {self.best_val_loss:.4f}")
        logger.info(f"Restored {len(self.top_checkpoints)} top-k checkpoints")

    def train(self):
        """Full training loop."""
        logger.info(f"Starting training for {self.max_epochs} epochs")
        logger.info(f"Training with {self.precision} precision, "
                    f"gradient accumulation steps={self.gradient_accumulation_steps}")
        if self.early_stopping_patience > 0:
            logger.info(f"Early stopping enabled with patience={self.early_stopping_patience}")
        logger.info(f"Keeping top {self.keep_top_k_checkpoints} checkpoints by val_loss")

        for epoch in range(self.current_epoch, self.max_epochs):
            self.current_epoch = epoch

            # Update curriculum / distributed samplers that need the epoch
            if hasattr(self.train_loader.sampler, "set_epoch"):
                self.train_loader.sampler.set_epoch(epoch)

            train_metrics = self.train_epoch()
            val_metrics = self.validate()

            if not self.scheduler_step_per_batch:
                self.scheduler.step()

            lr = self.optimizer.param_groups[0]["lr"]
            metrics = {
                **train_metrics,
                **val_metrics,
                "epoch": epoch,
                "learning_rate": lr,
            }
            logger.info(f"Epoch {epoch}: {metrics}")

            # Check for improvement and handle checkpoints
            val_loss = val_metrics.get("val_loss", float('inf'))
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
                self.save_checkpoint("best_model.pt")
            else:
                self.epochs_without_improvement += 1

            self._save_top_k_checkpoint(val_loss)
            # Always save last checkpoint so training can be resumed
            self.save_checkpoint("last_model.pt")

            if self.early_stopping_patience > 0:
                if self.epochs_without_improvement >= self.early_stopping_patience:
                    logger.info(
                        f"Early stopping triggered after {self.epochs_without_improvement} "
                        f"epochs without improvement. Best val_loss: {self.best_val_loss:.4f}"
                    )
                    break

        logger.info(f"Training complete. Best val_loss: {self.best_val_loss:.4f}")


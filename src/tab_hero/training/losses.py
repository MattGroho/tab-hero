"""
Loss functions for Tab Hero training.

Implements multi-head loss computation for note prediction,
timing, and auxiliary attributes.
"""

from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChartLoss(nn.Module):
    """Cross-entropy loss with pad token downweighting and label smoothing."""

    def __init__(
        self,
        pad_token_id: int = 0,
        pad_weight: float = 0.1,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.pad_token_id = pad_token_id
        self.pad_weight = pad_weight
        self.label_smoothing = label_smoothing

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        pad_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute loss.

        Args:
            logits: Predicted logits (batch, seq_len, vocab_size)
            targets: Target token IDs (batch, seq_len)
            pad_mask: Optional padding mask

        Returns:
            Dict with 'loss' and component losses
        """
        batch_size, seq_len, vocab_size = logits.shape

        logits_flat = logits.reshape(-1, vocab_size)
        targets_flat = targets.reshape(-1)

        loss_per_token = F.cross_entropy(
            logits_flat,
            targets_flat,
            reduction='none',
            label_smoothing=self.label_smoothing,
        ).reshape(batch_size, seq_len)

        # Downweight pad positions to avoid the model optimizing for padding
        weights = torch.ones_like(loss_per_token)
        pad_positions = targets == self.pad_token_id
        weights[pad_positions] = self.pad_weight

        loss = (loss_per_token * weights).sum() / weights.sum()

        with torch.no_grad():
            predictions = logits.argmax(dim=-1)
            accuracy = (predictions == targets).float().mean()

            non_pad_mask = ~pad_positions
            if non_pad_mask.any():
                non_pad_accuracy = (
                    (predictions == targets)[non_pad_mask].float().mean()
                )
            else:
                non_pad_accuracy = torch.tensor(0.0, device=logits.device)

        return {
            "loss": loss,
            "accuracy": accuracy,
            "non_pad_accuracy": non_pad_accuracy,
        }


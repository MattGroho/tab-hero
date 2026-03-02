"""Audio encoder for mel spectrogram features."""

import math
from typing import Optional
import torch
import torch.nn as nn


class AudioEncoder(nn.Module):
    """
    Projects mel spectrograms to transformer hidden dimension with temporal downsampling.

    Supports configurable downsampling (2x, 4x, 8x) through multi-layer Conv1D stack.
    Uses depthwise separable convolutions for efficiency at higher downsample factors.
    """

    def __init__(
        self,
        input_dim: int = 128,  # n_mels from mel spectrogram
        embed_dim: int = 512,
        downsample_factor: int = 4,  # Total downsampling: 2, 4, or 8
        use_depthwise_sep: bool = True,  # Use depthwise separable convs for efficiency
    ):
        super().__init__()

        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.downsample_factor = downsample_factor

        # Validate downsample factor
        valid_factors = {2: [2], 4: [2, 2], 8: [2, 2, 2]}
        if downsample_factor not in valid_factors:
            raise ValueError(f"downsample_factor must be one of {list(valid_factors.keys())}")

        strides = valid_factors[downsample_factor]

        # Project input to embed_dim with layer norm
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        # Build multi-layer conv stack for progressive downsampling
        layers = []
        for i, stride in enumerate(strides):
            kernel_size = 5 if stride > 1 else 3
            padding = kernel_size // 2

            if use_depthwise_sep and i > 0:
                # Depthwise separable: depthwise conv + pointwise conv
                layers.extend([
                    nn.Conv1d(
                        embed_dim, embed_dim, kernel_size=kernel_size,
                        stride=stride, padding=padding, groups=embed_dim
                    ),
                    nn.Conv1d(embed_dim, embed_dim, kernel_size=1),
                    nn.GELU(),
                    nn.GroupNorm(8, embed_dim),  # GroupNorm works well with conv
                ])
            else:
                # Standard conv
                layers.extend([
                    nn.Conv1d(
                        embed_dim, embed_dim, kernel_size=kernel_size,
                        stride=stride, padding=padding,
                    ),
                    nn.GELU(),
                    nn.GroupNorm(8, embed_dim),
                ])

        # Add refinement layer (no downsampling)
        layers.extend([
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.GELU(),
        ])

        self.conv_stack = nn.Sequential(*layers)

        # Final layer norm
        self.output_norm = nn.LayerNorm(embed_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize conv weights with appropriate scaling."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        audio_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            audio_embeddings: (batch, n_frames, input_dim) mel spectrogram
            attention_mask: Optional bool mask (batch, n_frames).
                True = valid frame, False = padding.  After the conv stack the
                mask is downsampled and padded positions are zeroed out so they
                do not contribute to downstream cross-attention.

        Returns:
            (batch, n_frames // downsample_factor, embed_dim)
        """
        # Project: (B, T, input_dim) -> (B, T, embed_dim)
        x = self.input_proj(audio_embeddings)

        # Conv stack: (B, T, D) -> (B, D, T) -> conv -> (B, D, T//factor) -> (B, T//factor, D)
        x = x.transpose(1, 2)
        x = self.conv_stack(x)
        x = x.transpose(1, 2)

        x = self.output_norm(x)

        # Zero out padded positions so they don't leak into cross-attention
        if attention_mask is not None:
            # Downsample mask to match encoder output length
            out_len = x.size(1)
            if attention_mask.size(1) != out_len:
                # Pool the boolean mask: a downsampled frame is valid if any
                # of its source frames were valid.
                mask_float = attention_mask.float().unsqueeze(1)  # (B, 1, T)
                ds_mask = torch.nn.functional.max_pool1d(
                    mask_float,
                    kernel_size=self.downsample_factor,
                    stride=self.downsample_factor,
                    padding=0,
                )  # (B, 1, T//factor)
                # Trim or pad to match exact output length
                if ds_mask.size(2) >= out_len:
                    ds_mask = ds_mask[:, :, :out_len]
                else:
                    pad_size = out_len - ds_mask.size(2)
                    ds_mask = torch.nn.functional.pad(ds_mask, (0, pad_size), value=0.0)
                attention_mask = ds_mask.squeeze(1).bool()  # (B, T//factor)
            x = x * attention_mask.unsqueeze(-1)

        return x

    def get_output_length(self, input_length: int) -> int:
        """Calculate output sequence length given input length."""
        # Each stride-2 conv halves the length (with padding, approximately)
        length = input_length
        if self.downsample_factor == 2:
            length = math.ceil(length / 2)
        elif self.downsample_factor == 4:
            length = math.ceil(length / 2)
            length = math.ceil(length / 2)
        elif self.downsample_factor == 8:
            length = math.ceil(length / 2)
            length = math.ceil(length / 2)
            length = math.ceil(length / 2)
        return length

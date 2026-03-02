"""Autoregressive transformer decoder for chart generation.

Supports:
- Flash Attention 2 (via PyTorch scaled_dot_product_attention)
- RoPE (Rotary Position Embeddings) for length extrapolation
- Gradient checkpointing for memory efficiency
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class SwiGLU(nn.Module):
    """SwiGLU activation function."""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class RotaryPositionEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE).

    Enables length extrapolation - model can generate sequences longer than training.
    Key for handling arbitrary-length songs.
    """

    def __init__(self, dim: int, max_seq_len: int = 8192, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute frequency bands
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Cache cos/sin for efficiency
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        """Build cos/sin cache for given sequence length."""
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        # (seq_len, dim/2) -> (seq_len, dim) by duplicating
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get cos/sin for positions 0..seq_len-1."""
        if seq_len > self.cos_cached.size(0):
            # Extend cache dynamically
            self._build_cache(seq_len)
        return (
            self.cos_cached[:seq_len].to(device),
            self.sin_cached[:seq_len].to(device),
        )


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                          position_offset: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to query and key tensors.

    Args:
        q: Query tensor (batch, n_heads, seq_len, head_dim)
        k: Key tensor (batch, n_heads, seq_len, head_dim)
        cos: Cosine values (seq_len, head_dim)
        sin: Sine values (seq_len, head_dim)
        position_offset: Offset for chunked/streaming generation

    Returns:
        Rotated q and k tensors
    """
    seq_len = q.size(2)

    # Apply offset for streaming generation
    if position_offset > 0:
        cos = cos[position_offset:position_offset + seq_len]
        sin = sin[position_offset:position_offset + seq_len]
    else:
        cos = cos[:seq_len]
        sin = sin[:seq_len]

    # Reshape for broadcasting: (seq_len, dim) -> (1, 1, seq_len, dim)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    # Rotate half
    def rotate_half(x):
        x1, x2 = x[..., :x.size(-1)//2], x[..., x.size(-1)//2:]
        return torch.cat([-x2, x1], dim=-1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed


class FlashAttention(nn.Module):
    """
    Multi-head attention using PyTorch's scaled_dot_product_attention.

    Automatically uses Flash Attention 2 on compatible hardware (Ampere+).
    Falls back to memory-efficient attention or standard attention otherwise.
    """

    def __init__(self, dim: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv_proj = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
        position_offset: int = 0,
        is_causal: bool = True,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass with optional RoPE and KV cache.

        Args:
            x: Input tensor (batch, seq_len, dim)
            cos, sin: RoPE embeddings
            position_offset: For streaming generation
            is_causal: Apply causal mask
            kv_cache: Cached key/values for generation

        Returns:
            Output tensor and updated kv_cache
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, S, D)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Apply RoPE if provided
        if cos is not None and sin is not None:
            q, k = apply_rotary_pos_emb(q, k, cos, sin, position_offset)

        # Handle KV cache for incremental generation
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)

        new_kv_cache = (k, v)

        # Use Flash Attention via scaled_dot_product_attention
        # This automatically selects the best backend (Flash, Memory-efficient, or Math)
        dropout_p = self.dropout if self.training else 0.0

        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=dropout_p,
            is_causal=is_causal and kv_cache is None,  # Only causal for full sequence
        )

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.dim)
        output = self.out_proj(attn_output)

        return output, new_kv_cache


class CrossAttention(nn.Module):
    """Cross-attention for attending to encoder output."""

    def __init__(self, dim: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.kv_proj = nn.Linear(dim, 2 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        encoder_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Cross-attention to encoder output.

        Args:
            x: Decoder hidden states (batch, seq_len, dim)
            encoder_output: Encoder output (batch, enc_len, dim)
            encoder_mask: Padding mask for encoder
        """
        batch_size, seq_len, _ = x.shape
        enc_len = encoder_output.size(1)

        # Project
        q = self.q_proj(x)
        kv = self.kv_proj(encoder_output)

        # Reshape
        q = q.reshape(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        kv = kv.reshape(batch_size, enc_len, 2, self.n_heads, self.head_dim)
        kv = kv.permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        # Attention with Flash
        dropout_p = self.dropout if self.training else 0.0

        # Handle encoder mask - convert to attention bias
        attn_mask = None
        if encoder_mask is not None:
            # encoder_mask: (batch, enc_len) where True = padded
            # Convert to attention mask: (batch, 1, 1, enc_len)
            attn_mask = encoder_mask.unsqueeze(1).unsqueeze(2)
            attn_mask = attn_mask.to(q.dtype) * torch.finfo(q.dtype).min

        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=False,
        )

        # Reshape and project
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.dim)
        return self.out_proj(attn_output)


class DecoderBlock(nn.Module):
    """
    Single transformer decoder block with:
    - Causal self-attention with RoPE and Flash Attention
    - Cross-attention to encoder
    - SwiGLU FFN
    - Pre-norm with RMSNorm
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
        use_flash: bool = True,
    ):
        super().__init__()
        self.use_flash = use_flash

        self.self_attn_norm = RMSNorm(dim)
        self.cross_attn_norm = RMSNorm(dim)
        self.ffn_norm = RMSNorm(dim)

        if use_flash:
            self.self_attn = FlashAttention(dim, n_heads, dropout)
            self.cross_attn = CrossAttention(dim, n_heads, dropout)
        else:
            # Fallback to standard PyTorch attention
            self.self_attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
            self.cross_attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)

        self.ffn = SwiGLU(dim, ffn_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
        position_offset: int = 0,
        encoder_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass.

        Args:
            x: Decoder input (batch, seq_len, dim)
            encoder_output: Encoder output (batch, enc_len, dim)
            cos, sin: RoPE embeddings
            position_offset: For streaming generation
            encoder_mask: Padding mask for encoder
            kv_cache: Cached K/V for incremental generation

        Returns:
            Output tensor and updated kv_cache
        """
        new_kv_cache = None

        # Self-attention
        residual = x
        x = self.self_attn_norm(x)

        if self.use_flash:
            x, new_kv_cache = self.self_attn(
                x, cos=cos, sin=sin,
                position_offset=position_offset,
                is_causal=True,
                kv_cache=kv_cache,
            )
        else:
            seq_len = x.size(1)
            mask = torch.nn.Transformer.generate_square_subsequent_mask(
                seq_len, device=x.device, dtype=x.dtype
            )
            x, _ = self.self_attn(x, x, x, attn_mask=mask, is_causal=True)

        x = residual + self.dropout(x)

        # Cross-attention
        residual = x
        x = self.cross_attn_norm(x)

        if self.use_flash:
            x = self.cross_attn(x, encoder_output, encoder_mask)
        else:
            x, _ = self.cross_attn(x, encoder_output, encoder_output, key_padding_mask=encoder_mask)

        x = residual + self.dropout(x)

        # FFN
        residual = x
        x = self.ffn_norm(x)
        x = residual + self.dropout(self.ffn(x))

        return x, new_kv_cache


class ChartDecoder(nn.Module):
    """
    Transformer decoder with:
    - RoPE positional encoding for unlimited sequence length
    - Flash Attention for memory efficiency
    - Difficulty/instrument conditioning
    - Weight-tied output projection
    - Gradient checkpointing support
    """

    def __init__(
        self,
        vocab_size: int,
        n_layers: int = 6,
        dim: int = 512,
        n_heads: int = 8,
        ffn_dim: int = 2048,
        max_seq_len: int = 8192,
        dropout: float = 0.1,
        n_difficulties: int = 4,
        n_instruments: int = 4,
        use_flash: bool = True,
        use_rope: bool = True,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.max_seq_len = max_seq_len
        self.use_rope = use_rope
        self.use_flash = use_flash
        self.gradient_checkpointing = gradient_checkpointing

        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, dim)

        # Positional encoding: RoPE or learned
        if use_rope:
            self.rope = RotaryPositionEmbedding(self.head_dim, max_seq_len)
            self.position_embedding = None
        else:
            self.rope = None
            self.position_embedding = nn.Embedding(max_seq_len, dim)

        # Conditioning embeddings
        self.difficulty_embedding = nn.Embedding(n_difficulties, dim)
        self.instrument_embedding = nn.Embedding(n_instruments, dim)

        # Decoder layers
        self.layers = nn.ModuleList([
            DecoderBlock(dim, n_heads, ffn_dim, dropout, use_flash=use_flash)
            for _ in range(n_layers)
        ])

        # Output
        self.output_norm = RMSNorm(dim)
        self.output_proj = nn.Linear(dim, vocab_size, bias=False)
        self.output_proj.weight = self.token_embedding.weight  # Weight tying

        self.dropout = nn.Dropout(dropout)

        # Initialize
        self._init_weights()

    def _init_weights(self):
        """Initialize embeddings."""
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.difficulty_embedding.weight, std=0.02)
        nn.init.normal_(self.instrument_embedding.weight, std=0.02)
        if self.position_embedding is not None:
            nn.init.normal_(self.position_embedding.weight, std=0.02)

    def get_rope_embeddings(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get RoPE cos/sin embeddings."""
        if self.rope is not None:
            return self.rope(seq_len, device)
        return None, None


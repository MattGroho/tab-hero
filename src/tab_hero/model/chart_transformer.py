"""Audio -> chart transformer.

Encoder processes mel spectrograms, decoder generates note tokens autoregressively.
Supports arbitrary-length generation via RoPE, KV caching, and chunked streaming.
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .encoder import AudioEncoder
from .decoder import ChartDecoder


class ChartTransformer(nn.Module):
    """Encoder-decoder transformer for audio-to-chart generation."""

    def __init__(
        self,
        vocab_size: int = 740,  # From ChartTokenizer with modifiers
        audio_input_dim: int = 128,  # n_mels from mel spectrogram
        encoder_dim: int = 512,
        decoder_dim: int = 512,
        n_decoder_layers: int = 6,
        n_heads: int = 8,
        ffn_dim: int = 2048,
        max_seq_len: int = 8192,
        dropout: float = 0.1,
        audio_downsample: int = 4,
        use_flash: bool = True,
        use_rope: bool = True,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.audio_input_dim = audio_input_dim
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.n_decoder_layers = n_decoder_layers
        self.n_heads = n_heads
        self.ffn_dim = ffn_dim
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.audio_downsample = audio_downsample
        self.use_flash = use_flash
        self.use_rope = use_rope
        self.gradient_checkpointing = gradient_checkpointing

        self.encoder = AudioEncoder(
            input_dim=audio_input_dim,
            embed_dim=encoder_dim,
            downsample_factor=audio_downsample,
        )

        self.decoder = ChartDecoder(
            vocab_size=vocab_size,
            n_layers=n_decoder_layers,
            dim=decoder_dim,
            n_heads=n_heads,
            ffn_dim=ffn_dim,
            max_seq_len=max_seq_len,
            dropout=dropout,
            use_flash=use_flash,
            use_rope=use_rope,
            gradient_checkpointing=gradient_checkpointing,
        )

        self.encoder_proj = (
            nn.Linear(encoder_dim, decoder_dim)
            if encoder_dim != decoder_dim
            else nn.Identity()
        )

    @property
    def model_config(self) -> dict:
        """Return the constructor kwargs needed to recreate this architecture."""
        return {
            "vocab_size": self.vocab_size,
            "audio_input_dim": self.audio_input_dim,
            "encoder_dim": self.encoder_dim,
            "decoder_dim": self.decoder_dim,
            "n_decoder_layers": self.n_decoder_layers,
            "n_heads": self.n_heads,
            "ffn_dim": self.ffn_dim,
            "max_seq_len": self.max_seq_len,
            "dropout": self.dropout,
            "audio_downsample": self.audio_downsample,
            "use_flash": self.use_flash,
            "use_rope": self.use_rope,
            "gradient_checkpointing": self.gradient_checkpointing,
        }

    def _downsample_mask(
        self,
        audio_mask: torch.Tensor,
        target_len: int,
    ) -> torch.Tensor:
        """Downsample a boolean audio mask to match encoder output length.

        Args:
            audio_mask: (batch, n_frames) bool -- True = valid, False = pad.
            target_len: Expected encoder output sequence length.

        Returns:
            (batch, target_len) bool mask suitable for cross-attention
            (True = valid, False = pad).
        """
        if audio_mask.size(1) == target_len:
            return audio_mask
        mask_float = audio_mask.float().unsqueeze(1)  # (B, 1, T)
        ds = torch.nn.functional.max_pool1d(
            mask_float,
            kernel_size=self.audio_downsample,
            stride=self.audio_downsample,
            padding=0,
        )  # (B, 1, T // factor)
        ds = ds.squeeze(1)  # (B, T // factor)
        if ds.size(1) >= target_len:
            ds = ds[:, :target_len]
        else:
            pad_size = target_len - ds.size(1)
            ds = F.pad(ds, (0, pad_size), value=0.0)
        return ds.bool()

    def forward(
        self,
        audio_embeddings: torch.Tensor,
        note_tokens: torch.Tensor,
        difficulty_id: Optional[torch.Tensor] = None,
        instrument_id: Optional[torch.Tensor] = None,
        audio_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Training forward pass.

        Args:
            audio_embeddings: (batch, n_frames, 128) mel spectrogram
            note_tokens: (batch, seq_len) token sequence with BOS/EOS
            difficulty_id: (batch,) difficulty level
            instrument_id: (batch,) instrument type
            audio_mask: (batch, n_frames) bool -- True = valid, False = pad

        Returns:
            Dict with 'logits' and 'loss'
        """
        encoder_output = self.encoder(audio_embeddings, attention_mask=audio_mask)
        encoder_output = self.encoder_proj(encoder_output)

        # Build encoder mask: CrossAttention expects True = masked position (additive bias)
        encoder_mask: Optional[torch.Tensor] = None
        if audio_mask is not None:
            valid_mask = self._downsample_mask(audio_mask, encoder_output.size(1))
            encoder_mask = ~valid_mask

        decoder_input = note_tokens[:, :-1]
        targets = note_tokens[:, 1:]

        batch_size, seq_len = decoder_input.shape
        device = decoder_input.device

        x = self.decoder.token_embedding(decoder_input)

        if self.decoder.use_rope:
            cos, sin = self.decoder.get_rope_embeddings(seq_len, device)
        else:
            positions = torch.arange(seq_len, device=device)
            x = x + self.decoder.position_embedding(positions)
            cos, sin = None, None

        if difficulty_id is not None:
            x = x + self.decoder.difficulty_embedding(difficulty_id).unsqueeze(1)
        if instrument_id is not None:
            x = x + self.decoder.instrument_embedding(instrument_id).unsqueeze(1)

        x = self.decoder.dropout(x)

        for layer in self.decoder.layers:
            if self.gradient_checkpointing and self.training:
                x, _ = checkpoint(
                    layer, x, encoder_output, cos, sin, 0, encoder_mask, None,
                    use_reentrant=False
                )
            else:
                x, _ = layer(x, encoder_output, cos=cos, sin=sin,
                             encoder_mask=encoder_mask)

        x = self.decoder.output_norm(x)
        logits = self.decoder.output_proj(x)

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=0,  # PAD token
        )

        return {"logits": logits, "loss": loss}

    @torch.no_grad()
    def generate(
        self,
        audio_embeddings: torch.Tensor,
        difficulty_id: Optional[torch.Tensor] = None,
        instrument_id: Optional[torch.Tensor] = None,
        audio_mask: Optional[torch.Tensor] = None,
        max_length: Optional[int] = None,
        temperature: float = 1.0,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.95,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        use_kv_cache: bool = True,
        vocab_ranges: Optional[Dict[str, Tuple[int, int]]] = None,
        audio_frame_rate: float = 86.1,
    ) -> torch.Tensor:
        """
        Generate chart tokens from audio.

        Uses KV caching for efficient autoregressive generation.
        RoPE enables generation beyond training sequence length.

        Args:
            audio_embeddings: (batch, n_frames, 128) mel spectrogram
            difficulty_id: Difficulty level (0-3)
            instrument_id: Instrument type (0-3)
            audio_mask: (batch, n_frames) bool -- True = valid, False = pad
            max_length: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling (None to disable)
            top_p: Nucleus sampling threshold (None to disable)
            use_kv_cache: Use KV caching for efficiency
            vocab_ranges: Optional dict mapping slot names ("time", "fret", "mod", "dur")
                to (start_id, end_id) vocab ranges.  When provided, constrained decoding
                is applied: only tokens from the correct slot range (plus EOS) are allowed
                at each generation step.  This enforces the T-F-M-D quad structure even
                when the model is uncertain.
            audio_frame_rate: Mel-frame rate in Hz (default 86.1 = 22050 Hz / 256 hop).

        Returns:
            Generated token sequence (batch, seq_len)
        """
        self.eval()
        device = audio_embeddings.device
        batch_size = audio_embeddings.size(0)

        if max_length is None:
            # Estimate max length from audio duration.
            # n_frames is the number of mel frames; dividing by frame_rate gives seconds.
            # The encoder's audio_downsample is an internal pooling factor and does NOT
            # affect the relationship between mel frames and wall-clock time.
            n_frames = audio_embeddings.size(1)
            estimated_duration = n_frames / audio_frame_rate  # seconds
            if vocab_ranges is not None:
                # Constrained decoding: 0% token waste.
                # Budget: 3 notes/sec * 4 tokens/note = 12 tokens/sec.
                max_length = min(int(estimated_duration * 12) + 4, 32768)
            else:
                # Unconstrained: ~80% of tokens land in wrong positions, so use
                # a larger budget to keep enough structurally-valid tokens.
                # 4 notes/sec * 4 tokens/note * 5x waste factor ≈ 16 tokens/sec.
                max_length = min(int(estimated_duration * 16) + 4, 32768)

        # Build per-slot constraint masks (vocab_size,) once if constrained decoding enabled
        slot_masks: Optional[List[torch.Tensor]] = None
        if vocab_ranges is not None:
            slot_order = ["time", "fret", "mod", "dur"]
            slot_masks = []
            for key in slot_order:
                start, end = vocab_ranges[key]
                mask = torch.full((self.vocab_size,), float('-inf'), device=device)
                mask[start:end] = 0.0
                mask[eos_token_id] = 0.0  # EOS always allowed
                slot_masks.append(mask)

        encoder_output = self.encoder(audio_embeddings, attention_mask=audio_mask)
        encoder_output = self.encoder_proj(encoder_output)

        encoder_mask: Optional[torch.Tensor] = None
        if audio_mask is not None:
            valid_mask = self._downsample_mask(audio_mask, encoder_output.size(1))
            encoder_mask = ~valid_mask

        generated = torch.full(
            (batch_size, 1), bos_token_id,
            dtype=torch.long, device=device
        )

        kv_caches: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [
            None for _ in range(len(self.decoder.layers))
        ]

        for step in range(max_length - 1):
            if use_kv_cache and step > 0:
                current_token = generated[:, -1:]
                position_offset = step
            else:
                current_token = generated
                position_offset = 0

            seq_len = current_token.size(1)

            x = self.decoder.token_embedding(current_token)

            if self.decoder.use_rope:
                total_len = generated.size(1) if not use_kv_cache else position_offset + seq_len
                cos, sin = self.decoder.get_rope_embeddings(total_len, device)
            else:
                if use_kv_cache and step > 0:
                    positions = torch.tensor([step], device=device)
                else:
                    positions = torch.arange(seq_len, device=device)
                x = x + self.decoder.position_embedding(positions)
                cos, sin = None, None

            # Conditioning (only add at step 0 if using cache)
            if not use_kv_cache or step == 0:
                if difficulty_id is not None:
                    x = x + self.decoder.difficulty_embedding(difficulty_id).unsqueeze(1)
                if instrument_id is not None:
                    x = x + self.decoder.instrument_embedding(instrument_id).unsqueeze(1)

            new_kv_caches = []
            for i, layer in enumerate(self.decoder.layers):
                cache = kv_caches[i] if use_kv_cache else None
                x, new_cache = layer(
                    x, encoder_output,
                    cos=cos, sin=sin,
                    position_offset=position_offset,
                    encoder_mask=encoder_mask,
                    kv_cache=cache,
                )
                new_kv_caches.append(new_cache)

            if use_kv_cache:
                kv_caches = new_kv_caches

            x = self.decoder.output_norm(x)
            logits = self.decoder.output_proj(x[:, -1, :])  # (batch, vocab)

            # Constrained decoding: mask to the expected token type for this slot
            if slot_masks is not None:
                slot_idx = step % 4
                logits = logits + slot_masks[slot_idx]

            logits = logits / temperature

            if top_k is not None and top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')

            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(
                    dim=-1, index=sorted_indices, src=sorted_indices_to_remove
                )
                logits[indices_to_remove] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)

            if (next_token == eos_token_id).all():
                break

        return generated

    @torch.no_grad()
    def generate_streaming(
        self,
        audio_embeddings: torch.Tensor,
        difficulty_id: Optional[torch.Tensor] = None,
        instrument_id: Optional[torch.Tensor] = None,
        audio_mask: Optional[torch.Tensor] = None,
        chunk_size: int = 4096,
        overlap: int = 512,
        temperature: float = 1.0,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.95,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
    ) -> torch.Tensor:
        """
        Generate chart tokens for arbitrarily long songs using chunked streaming.

        Processes audio in overlapping chunks to avoid memory issues.
        Uses RoPE position offsets to maintain position coherence across chunks.

        Args:
            audio_embeddings: (batch, n_frames, 128) - can be any length
            audio_mask: (batch, n_frames) bool -- True = valid, False = pad
            chunk_size: Audio frames per chunk (before downsampling)
            overlap: Overlap between chunks in audio frames

        Returns:
            Generated token sequence (batch, seq_len)
        """
        self.eval()
        device = audio_embeddings.device
        batch_size = audio_embeddings.size(0)
        total_frames = audio_embeddings.size(1)

        # If short enough, use standard generation
        if total_frames <= chunk_size:
            return self.generate(
                audio_embeddings, difficulty_id, instrument_id,
                audio_mask=audio_mask,
                temperature=temperature, top_k=top_k, top_p=top_p,
                bos_token_id=bos_token_id, eos_token_id=eos_token_id,
            )

        # Chunked generation
        all_tokens = []
        position_offset = 0

        # Start with BOS
        current_context = torch.full(
            (batch_size, 1), bos_token_id,
            dtype=torch.long, device=device
        )

        # Process chunks
        start = 0
        while start < total_frames:
            end = min(start + chunk_size, total_frames)

            audio_chunk = audio_embeddings[:, start:end, :]
            chunk_mask = audio_mask[:, start:end] if audio_mask is not None else None

            encoder_output = self.encoder(audio_chunk, attention_mask=chunk_mask)
            encoder_output = self.encoder_proj(encoder_output)

            chunk_encoder_mask: Optional[torch.Tensor] = None
            if chunk_mask is not None:
                valid_mask = self._downsample_mask(chunk_mask, encoder_output.size(1))
                chunk_encoder_mask = ~valid_mask

            # Estimate tokens for this chunk.
            # (end - start) is in mel frames; dividing by frame_rate gives seconds.
            chunk_duration = (end - start) / 86.1
            chunk_max_tokens = int(chunk_duration * 16)  # ~4 notes/sec * 4 tokens/note

            chunk_tokens = self._generate_chunk(
                encoder_output=encoder_output,
                initial_tokens=current_context,
                difficulty_id=difficulty_id,
                instrument_id=instrument_id,
                encoder_mask=chunk_encoder_mask,
                max_tokens=chunk_max_tokens,
                position_offset=position_offset,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                eos_token_id=eos_token_id,
            )

            new_tokens = chunk_tokens[:, current_context.size(1):]
            all_tokens.append(new_tokens)

            position_offset += new_tokens.size(1)
            current_context = chunk_tokens[:, -overlap:] if overlap > 0 else chunk_tokens[:, -1:]
            start = end - (overlap if end < total_frames else 0)

            if (chunk_tokens[:, -1] == eos_token_id).all():
                break

        all_tokens.insert(0, torch.full((batch_size, 1), bos_token_id, dtype=torch.long, device=device))
        result = torch.cat(all_tokens, dim=1)

        return result

    def _generate_chunk(
        self,
        encoder_output: torch.Tensor,
        initial_tokens: torch.Tensor,
        difficulty_id: Optional[torch.Tensor],
        instrument_id: Optional[torch.Tensor],
        encoder_mask: Optional[torch.Tensor],
        max_tokens: int,
        position_offset: int,
        temperature: float,
        top_k: Optional[int],
        top_p: Optional[float],
        eos_token_id: int,
    ) -> torch.Tensor:
        """Generate tokens for a single audio chunk."""
        device = encoder_output.device
        batch_size = encoder_output.size(0)

        generated = initial_tokens.clone()

        for step in range(max_tokens):
            seq_len = generated.size(1)

            x = self.decoder.token_embedding(generated)

            if self.decoder.use_rope:
                total_pos = position_offset + seq_len
                cos, sin = self.decoder.get_rope_embeddings(total_pos, device)
            else:
                positions = torch.arange(seq_len, device=device) + position_offset
                positions = positions.clamp(max=self.decoder.max_seq_len - 1)
                x = x + self.decoder.position_embedding(positions)
                cos, sin = None, None

            if difficulty_id is not None:
                x = x + self.decoder.difficulty_embedding(difficulty_id).unsqueeze(1)
            if instrument_id is not None:
                x = x + self.decoder.instrument_embedding(instrument_id).unsqueeze(1)

            for layer in self.decoder.layers:
                x, _ = layer(x, encoder_output, cos=cos, sin=sin,
                             position_offset=position_offset,
                             encoder_mask=encoder_mask)

            x = self.decoder.output_norm(x)
            logits = self.decoder.output_proj(x[:, -1, :]) / temperature

            if top_k is not None and top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')

            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(
                    dim=-1, index=sorted_indices, src=sorted_indices_to_remove
                )
                logits[indices_to_remove] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)

            if (next_token == eos_token_id).all():
                break

        return generated

    def get_num_params(self, non_embedding: bool = True) -> int:
        """Return number of parameters, optionally excluding embeddings."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.decoder.token_embedding.weight.numel()
            if self.decoder.position_embedding is not None:
                n_params -= self.decoder.position_embedding.weight.numel()
        return n_params

"""Chart generation from audio files."""

from pathlib import Path
from typing import Dict, List, Optional, Union
import logging

import torch

from ..model.chart_transformer import ChartTransformer
from ..dataio.audio_processor import AudioProcessor
from ..dataio.tokenizer import ChartTokenizer

logger = logging.getLogger(__name__)


class ChartGenerator:
    """Generates guitar charts from audio files using a trained model."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        model: Optional[ChartTransformer] = None,
        device: str = "cuda",
        data_dir: Optional[str] = None,
    ):
        self.device = device
        # Load audio processor with config matching training data
        self.audio_processor = AudioProcessor(device=device, data_dir=data_dir)
        self.tokenizer = ChartTokenizer()

        if model is not None:
            self.model = model.to(device)
        elif model_path is not None:
            self.model = self._load_model(model_path)
        else:
            raise ValueError("Must provide either model_path or model")

        self.model.eval()

        logger.info(
            f"AudioProcessor: {self.audio_processor.sample_rate}Hz, "
            f"hop={self.audio_processor.hop_length}, "
            f"frame_rate={self.audio_processor.frame_rate:.1f}Hz"
        )

    @staticmethod
    def _infer_config_from_state_dict(state_dict: dict) -> dict:
        """Infer ChartTransformer constructor kwargs from tensor shapes in state_dict.

        Used as a fallback when the checkpoint was saved without an explicit
        "config" key (i.e. trained before trainer was updated to persist config).
        """
        # encoder.input_proj.0 is nn.Linear(audio_input_dim, encoder_dim)
        input_proj = state_dict.get("encoder.input_proj.0.weight")
        encoder_dim = int(input_proj.shape[0]) if input_proj is not None else 512
        audio_input_dim = int(input_proj.shape[1]) if input_proj is not None else 128

        # decoder.token_embedding is nn.Embedding(vocab_size, decoder_dim)
        token_emb = state_dict.get("decoder.token_embedding.weight")
        vocab_size = int(token_emb.shape[0]) if token_emb is not None else 740
        decoder_dim = int(token_emb.shape[1]) if token_emb is not None else 512

        # Count decoder layers by finding the highest layer index present
        layer_indices = set()
        for k in state_dict:
            parts = k.split(".")
            if len(parts) >= 3 and parts[0] == "decoder" and parts[1] == "layers":
                try:
                    layer_indices.add(int(parts[2]))
                except ValueError:
                    pass
        n_decoder_layers = max(layer_indices) + 1 if layer_indices else 6

        # decoder.layers.0.ffn.w1 is nn.Linear(decoder_dim, ffn_dim)
        ffn_w1 = state_dict.get("decoder.layers.0.ffn.w1.weight")
        ffn_dim = int(ffn_w1.shape[0]) if ffn_w1 is not None else decoder_dim * 4

        # n_heads: standard head_dim=64 heuristic; falls back to 8
        n_heads = max(1, decoder_dim // 64) if decoder_dim % 64 == 0 else 8

        config = {
            "vocab_size": vocab_size,
            "audio_input_dim": audio_input_dim,
            "encoder_dim": encoder_dim,
            "decoder_dim": decoder_dim,
            "n_decoder_layers": n_decoder_layers,
            "n_heads": n_heads,
            "ffn_dim": ffn_dim,
        }
        logger.info(
            "No 'config' key in checkpoint; inferred architecture from state dict: %s",
            config,
        )
        return config

    def _load_model(self, path: str) -> ChartTransformer:
        """Load model from checkpoint.

        Handles checkpoints saved from torch.compile-wrapped models (where all
        state dict keys carry a '_orig_mod.' prefix) and checkpoints saved
        before model config was persisted (pre-existing checkpoints).
        """
        checkpoint = torch.load(path, map_location=self.device)

        state_dict = checkpoint["model_state_dict"]

        # Strip the _orig_mod. prefix added by torch.compile when saving
        if any(k.startswith("_orig_mod.") for k in state_dict):
            state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}

        # Use the saved config when available; fall back to shape inference
        config = checkpoint.get("config") or self._infer_config_from_state_dict(state_dict)

        model = ChartTransformer(**config)
        model.load_state_dict(state_dict)
        model = model.to(self.device)

        return model

    def generate(
        self,
        audio_path: Union[str, Path],
        difficulty: str = "expert",
        instrument: str = "lead",
        temperature: float = 1.0,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.95,
    ) -> Dict:
        """
        Generate a chart from an audio file.
        """
        logger.info(f"Generating chart for {audio_path}")

        # Load and encode audio
        waveform, _ = self.audio_processor.load_audio(audio_path)
        audio_duration_ms = waveform.shape[-1] / self.audio_processor.sample_rate * 1000.0
        audio_codes = self.audio_processor.encode(waveform)
        audio_codes = audio_codes.unsqueeze(0)  # Add batch dimension

        # Map difficulty and instrument to IDs
        difficulty_map = {"easy": 0, "medium": 1, "hard": 2, "expert": 3}
        instrument_map = {"lead": 0, "bass": 1, "rhythm": 2, "keys": 3}

        difficulty_id = torch.tensor(
            [difficulty_map[difficulty]], device=self.device
        )
        instrument_id = torch.tensor(
            [instrument_map[instrument]], device=self.device
        )

        # Build constrained-decoding vocab ranges from the tokenizer
        tok = self.tokenizer
        vocab_ranges = {
            "time": (tok._time_token_start, tok._time_token_end),
            "fret": (tok._fret_token_start, tok._fret_token_end),
            "mod":  (tok._mod_token_start,  tok._mod_token_end),
            "dur":  (tok._dur_token_start,  tok._dur_token_end),
        }

        # Generate tokens
        with torch.no_grad():
            generated_tokens = self.model.generate(
                audio_codes,
                difficulty_id=difficulty_id,
                instrument_id=instrument_id,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                vocab_ranges=vocab_ranges,
                audio_frame_rate=self.audio_processor.frame_rate,
            )

        # Decode tokens to notes
        notes = self._decode_tokens(generated_tokens[0].cpu().tolist())

        # Clip notes to the actual audio duration; the model may generate
        # time deltas that push past the end of the file.
        notes = [n for n in notes if n["timestamp_ms"] <= audio_duration_ms]

        return {
            "notes": notes,
            "difficulty": difficulty,
            "instrument": instrument,
            "audio_path": str(audio_path),
            "audio_duration_ms": audio_duration_ms,
        }

    def _decode_tokens(self, tokens: List[int]) -> List[Dict]:
        """
        Convert token sequence to note events.

        Tokens are in quad format: [BOS] [TIME][FRET][MOD][DUR] ... [EOS]
        Each note is encoded as 4 consecutive tokens.
        """
        # Use the tokenizer's decode method which handles quads
        note_events = self.tokenizer.decode_tokens(tokens)

        # Convert NoteEvent objects to dicts
        notes = []
        for event in note_events:
            notes.append({
                "timestamp_ms": event.timestamp_ms,
                "frets": event.frets,
                "duration_ms": event.duration_ms,
            })

        return notes


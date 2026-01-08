"""Tokenizer for chart data to discrete token sequences."""

from dataclasses import dataclass
from itertools import combinations
from typing import List, Dict, Optional, Tuple


@dataclass
class TokenizerConfig:
    """Configuration for the chart tokenizer."""
    num_frets: int = 5
    max_time_delta_ms: int = 5000
    time_resolution_ms: int = 10
    max_duration_ms: int = 2000
    duration_resolution_ms: int = 50


class ChartTokenizer:
    """Converts chart data to discrete token sequences."""

    PAD_TOKEN = 0
    BOS_TOKEN = 1
    EOS_TOKEN = 2
    RESERVED_TOKENS = 3

    def __init__(self, config: Optional[TokenizerConfig] = None):
        self.config = config or TokenizerConfig()
        self._build_vocabulary()

    def _build_vocabulary(self) -> None:
        self.frets_to_token: Dict[Tuple[int, ...], int] = {}
        self.token_to_frets: Dict[int, Tuple[int, ...]] = {}

        token_idx = self.RESERVED_TOKENS

        # fret combinations
        all_frets = list(range(self.config.num_frets))
        for num_pressed in range(1, self.config.num_frets + 1):
            for combo in combinations(all_frets, num_pressed):
                self.frets_to_token[combo] = token_idx
                self.token_to_frets[token_idx] = combo
                token_idx += 1

        self.fret_token_end = token_idx

        # time delta tokens
        self.time_to_token: Dict[int, int] = {}
        self.token_to_time: Dict[int, int] = {}
        num_time_bins = self.config.max_time_delta_ms // self.config.time_resolution_ms
        for i in range(num_time_bins + 1):
            time_ms = i * self.config.time_resolution_ms
            self.time_to_token[time_ms] = token_idx
            self.token_to_time[token_idx] = time_ms
            token_idx += 1

        self.time_token_end = token_idx

        # duration tokens
        self.duration_to_token: Dict[int, int] = {}
        self.token_to_duration: Dict[int, int] = {}
        num_dur_bins = self.config.max_duration_ms // self.config.duration_resolution_ms
        for i in range(num_dur_bins + 1):
            dur_ms = i * self.config.duration_resolution_ms
            self.duration_to_token[dur_ms] = token_idx
            self.token_to_duration[token_idx] = dur_ms
            token_idx += 1

        self.vocab_size = token_idx

    def encode_frets(self, frets: List[int]) -> int:
        key = tuple(sorted(frets))
        return self.frets_to_token.get(key, self.PAD_TOKEN)

    def decode_frets(self, token: int) -> Tuple[int, ...]:
        return self.token_to_frets.get(token, ())

    def encode_time_delta(self, delta_ms: float) -> int:
        """Quantize time delta to nearest bin."""
        res = self.config.time_resolution_ms
        quantized = round(delta_ms / res) * res
        quantized = max(0, min(quantized, self.config.max_time_delta_ms))
        return self.time_to_token.get(quantized, self.time_to_token[0])

    def decode_time_delta(self, token: int) -> int:
        return self.token_to_time.get(token, 0)

    def encode_duration(self, duration_ms: float) -> int:
        """Quantize duration to nearest bin."""
        res = self.config.duration_resolution_ms
        quantized = round(duration_ms / res) * res
        quantized = max(0, min(quantized, self.config.max_duration_ms))
        return self.duration_to_token.get(quantized, self.duration_to_token[0])

    def decode_duration(self, token: int) -> int:
        return self.token_to_duration.get(token, 0)

    def encode_chart(self, chart_data) -> List[int]:
        """Convert chart data to token sequence."""
        tokens = [self.BOS_TOKEN]
        prev_time = 0.0

        for note in chart_data.notes:
            delta = note.timestamp_ms - prev_time
            tokens.append(self.encode_time_delta(delta))
            tokens.append(self.encode_frets(note.frets))
            tokens.append(self.encode_duration(note.duration_ms))
            prev_time = note.timestamp_ms

        tokens.append(self.EOS_TOKEN)
        return tokens

    def decode_tokens(self, tokens: List[int]) -> List[Dict]:
        """Convert tokens back to note events."""
        notes = []
        current_time = 0.0
        i = 0

        if tokens and tokens[0] == self.BOS_TOKEN:
            i = 1

        while i < len(tokens):
            if tokens[i] == self.EOS_TOKEN:
                break
            if i + 2 >= len(tokens):
                break

            time_token = tokens[i]
            fret_token = tokens[i + 1]
            dur_token = tokens[i + 2]

            delta = self.decode_time_delta(time_token)
            current_time += delta
            frets = self.decode_frets(fret_token)
            duration = self.decode_duration(dur_token)

            if frets:
                notes.append({
                    "timestamp_ms": current_time,
                    "frets": list(frets),
                    "duration_ms": duration,
                })
            i += 3

        return notes

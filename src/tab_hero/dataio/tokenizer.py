"""Tokenizer for .chart note sequences. Notes -> [TIME_DELTA, FRET, MODIFIER, DURATION] quads."""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
import itertools
import json
from pathlib import Path

from tab_hero.dataio.chart_parser import NoteEvent, ChartData


@dataclass
class TokenizerConfig:
    n_frets: int = 6
    include_open: bool = True
    include_modifiers: bool = True

    time_resolution_ms: int = 10
    max_time_delta_ms: int = 5000

    duration_resolution_ms: int = 50
    max_duration_ms: int = 5000

    max_sequence_length: int = 4096


class ChartTokenizer:
    """Maps note events <-> token IDs. Modifiers encode HOPO/TAP/STAR_POWER as bit flags."""

    PAD_TOKEN = 0
    BOS_TOKEN = 1
    EOS_TOKEN = 2

    # Modifier bit flags
    MOD_NONE = 0
    MOD_HOPO = 1
    MOD_TAP = 2
    MOD_STAR_POWER = 4

    def __init__(self, config: Optional[TokenizerConfig] = None):
        self.config = config or TokenizerConfig()
        self._build_vocabulary()

    def _build_vocabulary(self) -> None:
        self.id_to_token: Dict[int, str] = {}
        self.token_to_id: Dict[str, int] = {}

        # Separate mappings for frets
        self.token_to_frets: Dict[int, Tuple[int, ...]] = {}
        self.frets_to_token: Dict[Tuple[int, ...], int] = {}

        # Mappings for modifiers
        self.token_to_modifier: Dict[int, int] = {}
        self.modifier_to_token: Dict[int, int] = {}

        token_id = 0

        # Special tokens (0-2)
        for name in ["<PAD>", "<BOS>", "<EOS>"]:
            self.id_to_token[token_id] = name
            self.token_to_id[name] = token_id
            self.token_to_frets[token_id] = ()
            token_id += 1

        # Time delta tokens
        self._time_token_start = token_id
        num_time_bins = self.config.max_time_delta_ms // self.config.time_resolution_ms + 1
        for i in range(num_time_bins):
            name = f"TIME_{i}"
            self.id_to_token[token_id] = name
            self.token_to_id[name] = token_id
            token_id += 1
        self._time_token_end = token_id

        # Fret combination tokens
        self._fret_token_start = token_id
        n_elements = self.config.n_frets
        if self.config.include_open:
            n_elements += 1

        for r in range(1, n_elements + 1):
            for combo in itertools.combinations(range(n_elements), r):
                name = f"FRET_{'_'.join(str(f) for f in combo)}"
                self.id_to_token[token_id] = name
                self.token_to_id[name] = token_id
                self.token_to_frets[token_id] = combo
                self.frets_to_token[combo] = token_id
                token_id += 1
        self._fret_token_end = token_id

        # Modifier tokens (8 combinations: none, H, T, HT, S, HS, TS, HTS)
        self._mod_token_start = token_id
        if self.config.include_modifiers:
            mod_names = [
                "MOD_NONE",      # 0: no modifiers
                "MOD_H",         # 1: HOPO
                "MOD_T",         # 2: TAP
                "MOD_HT",        # 3: HOPO + TAP
                "MOD_S",         # 4: STAR_POWER
                "MOD_HS",        # 5: HOPO + STAR_POWER
                "MOD_TS",        # 6: TAP + STAR_POWER
                "MOD_HTS",       # 7: HOPO + TAP + STAR_POWER
            ]
            for i, name in enumerate(mod_names):
                self.id_to_token[token_id] = name
                self.token_to_id[name] = token_id
                self.token_to_modifier[token_id] = i
                self.modifier_to_token[i] = token_id
                token_id += 1
        self._mod_token_end = token_id

        # Duration tokens
        self._dur_token_start = token_id
        num_dur_bins = self.config.max_duration_ms // self.config.duration_resolution_ms + 1
        for i in range(num_dur_bins):
            name = f"DUR_{i}"
            self.id_to_token[token_id] = name
            self.token_to_id[name] = token_id
            token_id += 1
        self._dur_token_end = token_id

        self.vocab_size = token_id
    
    @property
    def pad_token_id(self) -> int:
        return self.PAD_TOKEN

    @property
    def bos_token_id(self) -> int:
        return self.BOS_TOKEN

    @property
    def eos_token_id(self) -> int:
        return self.EOS_TOKEN

    def encode_time_delta(self, delta_ms: float) -> int:
        bin_idx = min(
            int(delta_ms / self.config.time_resolution_ms),
            self.config.max_time_delta_ms // self.config.time_resolution_ms
        )
        return self._time_token_start + bin_idx

    def decode_time_delta(self, token_id: int) -> float:
        if not (self._time_token_start <= token_id < self._time_token_end):
            return 0.0
        bin_idx = token_id - self._time_token_start
        return bin_idx * self.config.time_resolution_ms

    def encode_frets(self, frets: List[int]) -> int:
        if not frets:
            if self.config.include_open:
                return self.frets_to_token.get((self.config.n_frets,), self.PAD_TOKEN)
            return self.PAD_TOKEN
        frets_tuple = tuple(sorted(frets))
        return self.frets_to_token.get(frets_tuple, self.PAD_TOKEN)

    def decode_frets(self, token_id: int) -> List[int]:
        frets = self.token_to_frets.get(token_id, ())
        return [f for f in frets if f < self.config.n_frets]

    def encode_duration(self, duration_ms: float) -> int:
        bin_idx = min(
            int(duration_ms / self.config.duration_resolution_ms),
            self.config.max_duration_ms // self.config.duration_resolution_ms
        )
        return self._dur_token_start + bin_idx

    def decode_duration(self, token_id: int) -> float:
        if not (self._dur_token_start <= token_id < self._dur_token_end):
            return 0.0
        bin_idx = token_id - self._dur_token_start
        return bin_idx * self.config.duration_resolution_ms

    def encode_modifiers(self, is_hopo: bool, is_tap: bool, is_star_power: bool) -> int:
        """Encode note modifiers into a single token."""
        if not self.config.include_modifiers:
            return self.PAD_TOKEN
        mod_value = 0
        if is_hopo:
            mod_value |= self.MOD_HOPO
        if is_tap:
            mod_value |= self.MOD_TAP
        if is_star_power:
            mod_value |= self.MOD_STAR_POWER
        return self.modifier_to_token.get(mod_value, self._mod_token_start)

    def decode_modifiers(self, token_id: int) -> Tuple[bool, bool, bool]:
        """Decode modifier token into (is_hopo, is_tap, is_star_power)."""
        if not self.config.include_modifiers:
            return (False, False, False)
        if not (self._mod_token_start <= token_id < self._mod_token_end):
            return (False, False, False)
        mod_value = self.token_to_modifier.get(token_id, 0)
        is_hopo = bool(mod_value & self.MOD_HOPO)
        is_tap = bool(mod_value & self.MOD_TAP)
        is_star_power = bool(mod_value & self.MOD_STAR_POWER)
        return (is_hopo, is_tap, is_star_power)

    @property
    def tokens_per_note(self) -> int:
        """Number of tokens per note event."""
        return 4 if self.config.include_modifiers else 3

    def encode_chart(self, chart_data: ChartData) -> List[int]:
        """Encode chart to tokens.

        With modifiers: [BOS, TIME, FRET, MOD, DUR, ..., EOS]
        Without: [BOS, TIME, FRET, DUR, ..., EOS]
        """
        tokens = [self.BOS_TOKEN]
        prev_time = 0.0

        for note in chart_data.notes:
            delta = note.timestamp_ms - prev_time
            tokens.append(self.encode_time_delta(delta))
            tokens.append(self.encode_frets(note.frets))
            if self.config.include_modifiers:
                tokens.append(self.encode_modifiers(
                    note.is_hopo, note.is_tap, note.is_star_power
                ))
            tokens.append(self.encode_duration(note.duration_ms))
            prev_time = note.timestamp_ms

        tokens.append(self.EOS_TOKEN)
        return tokens

    def decode_tokens(self, tokens: List[int]) -> List[NoteEvent]:
        """Decode tokens back to NoteEvent list."""
        notes = []
        current_time = 0.0
        i = 1 if tokens and tokens[0] == self.BOS_TOKEN else 0
        step = self.tokens_per_note

        while i < len(tokens):
            token = tokens[i]

            if token == self.EOS_TOKEN:
                break

            if i + step - 1 >= len(tokens):
                break  # incomplete note

            time_token = tokens[i]
            fret_token = tokens[i + 1]

            if self.config.include_modifiers:
                mod_token = tokens[i + 2]
                dur_token = tokens[i + 3]
                is_hopo, is_tap, is_star_power = self.decode_modifiers(mod_token)
            else:
                dur_token = tokens[i + 2]
                is_hopo, is_tap, is_star_power = False, False, False

            delta = self.decode_time_delta(time_token)
            current_time += delta
            frets = self.decode_frets(fret_token)
            duration = self.decode_duration(dur_token)

            notes.append(NoteEvent(
                timestamp_ms=current_time,
                frets=frets,
                duration_ms=duration,
                is_hopo=is_hopo,
                is_tap=is_tap,
                is_star_power=is_star_power,
            ))
            i += step

        return notes

    def get_vocab_size(self) -> int:
        return self.vocab_size

    def save(self, path: Path) -> None:
        config_dict = {
            "n_frets": self.config.n_frets,
            "include_open": self.config.include_open,
            "include_modifiers": self.config.include_modifiers,
            "time_resolution_ms": self.config.time_resolution_ms,
            "max_time_delta_ms": self.config.max_time_delta_ms,
            "duration_resolution_ms": self.config.duration_resolution_ms,
            "max_duration_ms": self.config.max_duration_ms,
            "max_sequence_length": self.config.max_sequence_length,
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(config_dict, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "ChartTokenizer":
        with open(path) as f:
            config_dict = json.load(f)
        config = TokenizerConfig(**config_dict)
        return cls(config)

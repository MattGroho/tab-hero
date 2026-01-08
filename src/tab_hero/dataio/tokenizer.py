from dataclasses import dataclass
from itertools import combinations
from typing import List, Dict, Optional, Tuple


@dataclass
class TokenizerConfig:
    num_frets: int = 5
    max_time_delta_ms: int = 5000
    time_resolution_ms: int = 10
    max_duration_ms: int = 2000
    duration_resolution_ms: int = 50


class ChartTokenizer:
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

        # generate all fret combinations (1 to 5 simultaneous frets)
        all_frets = list(range(self.config.num_frets))
        for num_pressed in range(1, self.config.num_frets + 1):
            for combo in combinations(all_frets, num_pressed):
                self.frets_to_token[combo] = token_idx
                self.token_to_frets[token_idx] = combo
                token_idx += 1

        self.fret_token_end = token_idx
        self.time_to_token: Dict[int, int] = {}
        self.token_to_time: Dict[int, int] = {}
        self.duration_to_token: Dict[int, int] = {}
        self.token_to_duration: Dict[int, int] = {}
        self.vocab_size = token_idx

    def encode_frets(self, frets: List[int]) -> int:
        key = tuple(sorted(frets))
        return self.frets_to_token.get(key, self.PAD_TOKEN)

    def decode_frets(self, token: int) -> Tuple[int, ...]:
        return self.token_to_frets.get(token, ())

    def encode_chart(self, chart_data) -> List[int]:
        raise NotImplementedError

    def decode_tokens(self, tokens: List[int]):
        raise NotImplementedError

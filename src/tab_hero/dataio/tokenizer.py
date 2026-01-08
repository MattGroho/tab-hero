from dataclasses import dataclass
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
        self.time_to_token: Dict[int, int] = {}
        self.token_to_time: Dict[int, int] = {}
        self.duration_to_token: Dict[int, int] = {}
        self.token_to_duration: Dict[int, int] = {}
        self.vocab_size = 0
        # TODO: build vocabulary

    def encode_chart(self, chart_data) -> List[int]:
        raise NotImplementedError

    def decode_tokens(self, tokens: List[int]):
        raise NotImplementedError

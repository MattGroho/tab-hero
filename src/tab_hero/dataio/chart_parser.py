from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any


@dataclass
class NoteEvent:
    timestamp_ms: float
    frets: List[int]
    duration_ms: float = 0.0


@dataclass
class ChartData:
    notes: List[NoteEvent]
    instrument: str
    difficulty: str
    resolution: int = 192
    bpm_events: List[Dict[str, Any]] = field(default_factory=list)
    song_length_ms: float = 0.0


class ChartParser:
    INSTRUMENTS = ["lead", "bass", "rhythm", "keys"]
    DIFFICULTIES = ["easy", "medium", "hard", "expert"]

    def parse(self, path: Path, instrument: str = "lead", difficulty: str = "expert"):
        # TODO: implement parsing
        raise NotImplementedError

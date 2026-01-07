from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional
import re


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

    def __init__(self):
        self._chart_data: Optional[ChartData] = None

    def parse(self, path: Path, instrument: str = "lead", difficulty: str = "expert") -> ChartData:
        path = Path(path)
        if path.suffix.lower() == ".chart":
            return self._parse_chart_file(path, instrument, difficulty)
        else:
            raise ValueError(f"Unsupported format: {path.suffix}")

    def _parse_chart_file(self, path: Path, instrument: str, difficulty: str) -> ChartData:
        content = path.read_text(encoding="utf-8", errors="replace")
        sections = self._parse_chart_sections(content)
        # TODO: extract notes from sections
        return ChartData(notes=[], instrument=instrument, difficulty=difficulty)

    def _parse_chart_sections(self, content: str) -> Dict[str, List[str]]:
        sections: Dict[str, List[str]] = {}
        current_section = None
        current_lines: List[str] = []

        for line in content.split("\n"):
            line = line.strip()
            match = re.match(r"^\[(.+)\]$", line)
            if match:
                if current_section:
                    sections[current_section] = current_lines
                current_section = match.group(1)
                current_lines = []
                continue
            if line in ["{", "}"]:
                continue
            if current_section and line:
                current_lines.append(line)

        if current_section:
            sections[current_section] = current_lines
        return sections

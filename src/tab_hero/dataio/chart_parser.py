from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
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

    CHART_SECTION_NAMES = {
        ("easy", "lead"): ["EasySingle"],
        ("medium", "lead"): ["MediumSingle"],
        ("hard", "lead"): ["HardSingle"],
        ("expert", "lead"): ["ExpertSingle"],
        ("expert", "bass"): ["ExpertDoubleBass"],
    }

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

        resolution = 192
        if "Song" in sections:
            for line in sections["Song"]:
                if "Resolution" in line:
                    match = re.search(r"Resolution\s*=\s*(\d+)", line)
                    if match:
                        resolution = int(match.group(1))

        section_names = self.CHART_SECTION_NAMES.get(
            (difficulty, instrument), [f"{difficulty.capitalize()}Single"]
        )
        note_section = None
        for name in section_names:
            if name in sections:
                note_section = sections[name]
                break

        if note_section is None:
            raise ValueError(f"No section for {difficulty} {instrument}")

        note_events: Dict[int, List[Tuple[int, int]]] = {}
        for line in note_section:
            match = re.match(r"\s*(\d+)\s*=\s*N\s+(\d+)\s+(\d+)", line)
            if match:
                tick = int(match.group(1))
                note_val = int(match.group(2))
                duration = int(match.group(3))
                if note_val <= 4:
                    if tick not in note_events:
                        note_events[tick] = []
                    note_events[tick].append((note_val, duration))

        # convert ticks to ms (assuming 120 bpm for now)
        ms_per_tick = (60000.0 / 120.0) / resolution
        notes = []
        for tick in sorted(note_events.keys()):
            frets = [f for f, _ in note_events[tick]]
            max_dur = max(d for _, d in note_events[tick])
            notes.append(NoteEvent(
                timestamp_ms=tick * ms_per_tick,
                frets=sorted(frets),
                duration_ms=max_dur * ms_per_tick,
            ))

        return ChartData(
            notes=notes,
            instrument=instrument,
            difficulty=difficulty,
            resolution=resolution,
        )

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

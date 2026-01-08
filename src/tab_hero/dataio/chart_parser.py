"""Parser for .chart and .mid format files."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import re
import chardet


@dataclass
class NoteEvent:
    """A single note or chord in a chart."""
    timestamp_ms: float
    frets: List[int]
    duration_ms: float = 0.0


@dataclass
class ChartData:
    """Container for parsed chart data."""
    notes: List[NoteEvent]
    instrument: str
    difficulty: str
    resolution: int = 192
    bpm_events: List[Dict[str, Any]] = field(default_factory=list)
    song_length_ms: float = 0.0


class ChartParser:
    """Parser for .chart and .mid files."""

    INSTRUMENTS = ["lead", "bass", "rhythm", "keys"]
    DIFFICULTIES = ["easy", "medium", "hard", "expert"]

    MIDI_DIFFICULTY_BASE = {
        "easy": 60,
        "medium": 72,
        "hard": 84,
        "expert": 96,
    }

    MIDI_TRACK_NAMES = {
        "lead": ["PART GUITAR", "T1 GEMS"],
        "bass": ["PART BASS"],
        "rhythm": ["PART RHYTHM"],
        "keys": ["PART KEYS"],
    }

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
        elif path.suffix.lower() in [".mid", ".midi"]:
            return self._parse_midi_file(path, instrument, difficulty)
        else:
            raise ValueError(f"Unsupported format: {path.suffix}")

    def _parse_midi_file(self, path: Path, instrument: str, difficulty: str) -> ChartData:
        import mido
        mid = mido.MidiFile(str(path))
        resolution = mid.ticks_per_beat

        target_track = None
        track_names = self.MIDI_TRACK_NAMES.get(instrument, ["PART GUITAR"])
        for track in mid.tracks:
            if track.name in track_names:
                target_track = track
                break
        if target_track is None:
            for track in mid.tracks:
                if "GUITAR" in track.name.upper():
                    target_track = track
                    break
        if target_track is None:
            raise ValueError(f"No track for {instrument}")

        bpm_events = []
        tick = 0
        for msg in mid.tracks[0]:
            tick += msg.time
            if msg.type == "set_tempo":
                bpm = 60000000.0 / msg.tempo
                bpm_events.append({"tick": tick, "bpm": bpm})
        if not bpm_events:
            bpm_events = [{"tick": 0, "bpm": 120.0}]

        base_note = self.MIDI_DIFFICULTY_BASE[difficulty]
        note_range = range(base_note, base_note + 5)

        active_notes: Dict[int, int] = {}
        note_events: Dict[int, List[Tuple[int, int]]] = {}

        current_tick = 0
        for msg in target_track:
            current_tick += msg.time
            if msg.type == "note_on" and msg.velocity > 0:
                if msg.note in note_range:
                    active_notes[msg.note] = current_tick
            elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                if msg.note in active_notes:
                    start_tick = active_notes.pop(msg.note)
                    fret = msg.note - base_note
                    duration = current_tick - start_tick
                    if start_tick not in note_events:
                        note_events[start_tick] = []
                    note_events[start_tick].append((fret, duration))

        notes = []
        for tick in sorted(note_events.keys()):
            frets = [f for f, _ in note_events[tick]]
            max_dur = max(d for _, d in note_events[tick])
            timestamp_ms = self._ticks_to_ms(tick, bpm_events, resolution)
            duration_ms = self._ticks_to_ms(tick + max_dur, bpm_events, resolution) - timestamp_ms
            notes.append(NoteEvent(
                timestamp_ms=timestamp_ms,
                frets=sorted(frets),
                duration_ms=duration_ms,
            ))

        song_length_ms = 0.0
        if notes:
            song_length_ms = notes[-1].timestamp_ms + notes[-1].duration_ms

        return ChartData(
            notes=notes,
            instrument=instrument,
            difficulty=difficulty,
            resolution=resolution,
            bpm_events=bpm_events,
            song_length_ms=song_length_ms,
        )

    def _read_chart_content(self, path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            raw = path.read_bytes()
            detected = chardet.detect(raw)
            encoding = detected.get("encoding", "utf-8") or "utf-8"
            return raw.decode(encoding, errors="replace")

    def _ticks_to_ms(self, ticks: int, bpm_events: List[Dict], resolution: int) -> float:
        if not bpm_events:
            return ticks * (60000.0 / 120.0) / resolution
        current_tick = 0
        current_time_ms = 0.0
        current_bpm = 120.0
        for event in bpm_events:
            event_tick = event["tick"]
            if event_tick > ticks:
                break
            tick_delta = event_tick - current_tick
            ms_per_tick = (60000.0 / current_bpm) / resolution
            current_time_ms += tick_delta * ms_per_tick
            current_tick = event_tick
            current_bpm = event["bpm"]
        tick_delta = ticks - current_tick
        ms_per_tick = (60000.0 / current_bpm) / resolution
        current_time_ms += tick_delta * ms_per_tick
        return current_time_ms

    def _parse_chart_file(self, path: Path, instrument: str, difficulty: str) -> ChartData:
        content = self._read_chart_content(path)
        sections = self._parse_chart_sections(content)
        resolution = 192
        if "Song" in sections:
            for line in sections["Song"]:
                if "Resolution" in line:
                    match = re.search(r"Resolution\s*=\s*(\d+)", line)
                    if match:
                        resolution = int(match.group(1))
        bpm_events = []
        if "SyncTrack" in sections:
            for line in sections["SyncTrack"]:
                match = re.match(r"\s*(\d+)\s*=\s*B\s+(\d+)", line)
                if match:
                    tick = int(match.group(1))
                    bpm = int(match.group(2)) / 1000.0
                    bpm_events.append({"tick": tick, "bpm": bpm})
        if not bpm_events:
            bpm_events = [{"tick": 0, "bpm": 120.0}]
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
        notes = []
        for tick in sorted(note_events.keys()):
            frets = [f for f, _ in note_events[tick]]
            max_dur = max(d for _, d in note_events[tick])
            timestamp_ms = self._ticks_to_ms(tick, bpm_events, resolution)
            duration_ms = self._ticks_to_ms(tick + max_dur, bpm_events, resolution) - timestamp_ms
            notes.append(NoteEvent(
                timestamp_ms=timestamp_ms,
                frets=sorted(frets),
                duration_ms=duration_ms,
            ))
        song_length_ms = 0.0
        if notes:
            song_length_ms = notes[-1].timestamp_ms + notes[-1].duration_ms
        return ChartData(
            notes=notes,
            instrument=instrument,
            difficulty=difficulty,
            resolution=resolution,
            bpm_events=bpm_events,
            song_length_ms=song_length_ms,
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

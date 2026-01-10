"""
Chart file parser for .chart and .mid formats.

Supports parsing Guitar Hero / Clone Hero chart files into a standardized
internal representation for training and evaluation.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import re
import logging

logger = logging.getLogger(__name__)


@dataclass
class NoteEvent:
    """Represents a single note or chord event in a chart."""

    timestamp_ms: float
    frets: List[int]  # List of fret indices (0-4 for 5-fret, 0-5 for 6-fret)
    duration_ms: float = 0.0
    is_hopo: bool = False  # Hammer-on / Pull-off
    is_tap: bool = False
    is_star_power: bool = False


@dataclass
class ChartData:
    """Container for parsed chart data."""

    notes: List[NoteEvent]
    instrument: str  # "lead", "bass", "rhythm", "keys"
    difficulty: str  # "easy", "medium", "hard", "expert"
    resolution: int = 192  # Ticks per beat
    bpm_events: List[Dict[str, Any]] = field(default_factory=list)
    time_signatures: List[Dict[str, Any]] = field(default_factory=list)
    song_length_ms: float = 0.0


class ChartParser:
    """
    Parser for .chart and .mid files.

    Handles reading and converting chart files into the internal
    NoteEvent representation used for training.
    """

    FRET_NAMES = ["green", "red", "yellow", "blue", "orange"]
    INSTRUMENTS = ["lead", "bass", "rhythm", "keys"]
    DIFFICULTIES = ["easy", "medium", "hard", "expert"]

    # MIDI note mappings for 5-fret guitar
    # Each difficulty has a base note, frets are offset 0-4
    MIDI_DIFFICULTY_BASE = {
        "easy": 60,
        "medium": 72,
        "hard": 84,
        "expert": 96,
    }

    # MIDI track names for instruments
    MIDI_TRACK_NAMES = {
        "lead": ["PART GUITAR", "PART GUITAR COOP", "T1 GEMS"],
        "bass": ["PART BASS", "PART RHYTHM"],
        "rhythm": ["PART RHYTHM", "PART GUITAR COOP"],
        "keys": ["PART KEYS"],
    }

    # .chart section names
    CHART_SECTION_NAMES = {
        ("easy", "lead"): ["EasySingle", "EasyGHLGuitar"],
        ("medium", "lead"): ["MediumSingle", "MediumGHLGuitar"],
        ("hard", "lead"): ["HardSingle", "HardGHLGuitar"],
        ("expert", "lead"): ["ExpertSingle", "ExpertGHLGuitar"],
        ("easy", "bass"): ["EasyDoubleBass"],
        ("medium", "bass"): ["MediumDoubleBass"],
        ("hard", "bass"): ["HardDoubleBass"],
        ("expert", "bass"): ["ExpertDoubleBass"],
    }

    def __init__(self):
        self._chart_data: Optional[ChartData] = None

    def parse(
        self,
        path: Path,
        instrument: str = "lead",
        difficulty: str = "expert"
    ) -> ChartData:
        """
        Parse a chart file.

        Args:
            path: Path to .chart or .mid file
            instrument: Instrument track to parse
            difficulty: Difficulty level to parse

        Returns:
            ChartData containing parsed notes and metadata
        """
        path = Path(path)

        if path.suffix.lower() == ".chart":
            return self._parse_chart_file(path, instrument, difficulty)
        elif path.suffix.lower() in [".mid", ".midi"]:
            return self._parse_midi_file(path, instrument, difficulty)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

    def _ticks_to_ms(
        self,
        ticks: int,
        bpm_events: List[Dict[str, Any]],
        resolution: int
    ) -> float:
        """Convert tick position to milliseconds using BPM events."""
        if not bpm_events:
            # Default 120 BPM
            return ticks * (60000.0 / 120.0) / resolution

        current_tick = 0
        current_time_ms = 0.0
        current_bpm = 120.0

        for event in bpm_events:
            event_tick = event["tick"]
            if event_tick > ticks:
                break

            # Add time from current position to this event
            tick_delta = event_tick - current_tick
            ms_per_tick = (60000.0 / current_bpm) / resolution
            current_time_ms += tick_delta * ms_per_tick

            current_tick = event_tick
            current_bpm = event["bpm"]

        # Add remaining time to target tick
        tick_delta = ticks - current_tick
        ms_per_tick = (60000.0 / current_bpm) / resolution
        current_time_ms += tick_delta * ms_per_tick

        return current_time_ms

    def _parse_midi_file(
        self,
        path: Path,
        instrument: str,
        difficulty: str
    ) -> ChartData:
        """Parse a .mid/.midi format file."""
        import mido

        mid = mido.MidiFile(str(path))
        resolution = mid.ticks_per_beat

        # Find the appropriate track
        target_track = None
        track_names = self.MIDI_TRACK_NAMES.get(instrument, ["PART GUITAR"])

        for track in mid.tracks:
            if track.name in track_names:
                target_track = track
                break

        if target_track is None:
            # Try to find any guitar-like track
            for track in mid.tracks:
                if "GUITAR" in track.name.upper() or "GEMS" in track.name.upper():
                    target_track = track
                    break

        if target_track is None:
            raise ValueError(f"No suitable track found for instrument '{instrument}'")

        # Extract tempo events from first track (usually contains tempo map)
        bpm_events = []
        current_tick = 0
        for msg in mid.tracks[0]:
            current_tick += msg.time
            if msg.type == "set_tempo":
                bpm = 60000000.0 / msg.tempo
                bpm_events.append({"tick": current_tick, "bpm": bpm})

        if not bpm_events:
            bpm_events = [{"tick": 0, "bpm": 120.0}]

        # Parse notes and modifiers from target track
        # MIDI note mappings for modifiers (relative to difficulty base):
        #   base+5 = Force Strum (tap)
        #   base+6 = Force HOPO
        #   103 = Star Power (Rock Band style)
        #   116 = Star Power (Guitar Hero style)
        base_note = self.MIDI_DIFFICULTY_BASE[difficulty]
        note_range = range(base_note, base_note + 5)  # 5 frets
        force_strum_note = base_note + 5  # 101 for Expert
        force_hopo_note = base_note + 6   # 102 for Expert
        star_power_notes = {103, 116}     # Both RB and GH style

        # Track active notes (note -> start_tick)
        active_notes: Dict[int, int] = {}
        note_events: Dict[int, List[Tuple[int, int]]] = {}  # tick -> [(fret, duration)]

        # Track modifier ranges
        force_hopo_ticks: set = set()
        force_strum_ticks: set = set()
        star_power_phrases: List[Tuple[int, int]] = []  # (start_tick, end_tick)

        # Active modifiers being tracked
        active_modifiers: Dict[int, int] = {}  # note -> start_tick

        current_tick = 0
        for msg in target_track:
            current_tick += msg.time

            if msg.type == "note_on" and msg.velocity > 0:
                if msg.note in note_range:
                    active_notes[msg.note] = current_tick
                elif msg.note == force_hopo_note:
                    active_modifiers[force_hopo_note] = current_tick
                elif msg.note == force_strum_note:
                    active_modifiers[force_strum_note] = current_tick
                elif msg.note in star_power_notes:
                    active_modifiers[msg.note] = current_tick

            elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                if msg.note in active_notes:
                    start_tick = active_notes.pop(msg.note)
                    fret = msg.note - base_note
                    duration = current_tick - start_tick

                    if start_tick not in note_events:
                        note_events[start_tick] = []
                    note_events[start_tick].append((fret, duration))

                # Handle modifier note-offs
                elif msg.note == force_hopo_note and force_hopo_note in active_modifiers:
                    start = active_modifiers.pop(force_hopo_note)
                    # Mark all ticks in this range as HOPO
                    for t in note_events.keys():
                        if start <= t < current_tick:
                            force_hopo_ticks.add(t)
                    # Also mark the start tick itself
                    force_hopo_ticks.add(start)

                elif msg.note == force_strum_note and force_strum_note in active_modifiers:
                    start = active_modifiers.pop(force_strum_note)
                    for t in note_events.keys():
                        if start <= t < current_tick:
                            force_strum_ticks.add(t)
                    force_strum_ticks.add(start)

                elif msg.note in star_power_notes and msg.note in active_modifiers:
                    start = active_modifiers.pop(msg.note)
                    star_power_phrases.append((start, current_tick))

        # Convert to NoteEvent objects
        notes = []
        for tick in sorted(note_events.keys()):
            frets = [f for f, _ in note_events[tick]]
            # Use max duration for the chord
            max_duration = max(d for _, d in note_events[tick])

            timestamp_ms = self._ticks_to_ms(tick, bpm_events, resolution)
            duration_ms = self._ticks_to_ms(tick + max_duration, bpm_events, resolution) - timestamp_ms

            # Apply modifiers
            is_hopo = tick in force_hopo_ticks
            is_tap = tick in force_strum_ticks
            is_star_power = any(start <= tick < end for start, end in star_power_phrases)

            notes.append(NoteEvent(
                timestamp_ms=timestamp_ms,
                frets=sorted(frets),
                duration_ms=duration_ms,
                is_hopo=is_hopo,
                is_tap=is_tap,
                is_star_power=is_star_power,
            ))

        # Calculate song length
        song_length_ms = 0.0
        if notes:
            last_note = notes[-1]
            song_length_ms = last_note.timestamp_ms + last_note.duration_ms

        return ChartData(
            notes=notes,
            instrument=instrument,
            difficulty=difficulty,
            resolution=resolution,
            bpm_events=bpm_events,
            song_length_ms=song_length_ms,
        )


    def _parse_chart_file(
        self,
        path: Path,
        instrument: str,
        difficulty: str
    ) -> ChartData:
        """Parse a .chart format file."""
        content = path.read_text(encoding="utf-8", errors="replace")

        # Parse sections
        sections = self._parse_chart_sections(content)

        # Get resolution from Song section
        resolution = 192
        if "Song" in sections:
            for line in sections["Song"]:
                if "Resolution" in line:
                    match = re.search(r"Resolution\s*=\s*(\d+)", line)
                    if match:
                        resolution = int(match.group(1))

        # Parse BPM events from SyncTrack
        bpm_events = []
        if "SyncTrack" in sections:
            for line in sections["SyncTrack"]:
                # Format: tick = B bpm_value (bpm * 1000)
                match = re.match(r"\s*(\d+)\s*=\s*B\s+(\d+)", line)
                if match:
                    tick = int(match.group(1))
                    bpm = int(match.group(2)) / 1000.0
                    bpm_events.append({"tick": tick, "bpm": bpm})

        if not bpm_events:
            bpm_events = [{"tick": 0, "bpm": 120.0}]

        # Find the appropriate note section
        section_names = self.CHART_SECTION_NAMES.get(
            (difficulty, instrument),
            [f"{difficulty.capitalize()}Single"]
        )

        note_section = None
        for name in section_names:
            if name in sections:
                note_section = sections[name]
                break

        if note_section is None:
            # Try case-insensitive search
            for sec_name, sec_content in sections.items():
                for target in section_names:
                    if sec_name.lower() == target.lower():
                        note_section = sec_content
                        break
                if note_section:
                    break

        if note_section is None:
            raise ValueError(
                f"No section found for {difficulty} {instrument}. "
                f"Available: {list(sections.keys())}"
            )

        # Parse notes and modifiers
        # Format: tick = N note_value duration
        # 5-fret: 0-4 are frets (green, red, yellow, blue, orange)
        # 6-fret (GHL): 0-2 top row (B1, B2, B3), 3-4 bottom row (W1, W2)
        #               8 is modifier for bottom row, 7 is open
        # Modifiers:
        #   N 5 = Force HOPO (hammer-on/pull-off)
        #   N 6 = Force Strum (or tap in some contexts)
        #   S 2 = Star Power phrase (tick = S 2 duration)
        note_events: Dict[int, List[Tuple[int, int]]] = {}
        force_hopo_ticks: set = set()  # Ticks with N 5 modifier
        force_strum_ticks: set = set()  # Ticks with N 6 modifier
        star_power_phrases: List[Tuple[int, int]] = []  # (start_tick, end_tick)
        is_ghl = "GHL" in str(note_section) or any("GHL" in name for name in section_names)

        for line in note_section:
            # Parse note events (N)
            match = re.match(r"\s*(\d+)\s*=\s*N\s+(\d+)\s+(\d+)", line)
            if match:
                tick = int(match.group(1))
                note_value = int(match.group(2))
                duration = int(match.group(3))

                # Note values:
                # 0-4: Standard frets (or 0-2 for GHL top row)
                # 5: Force HOPO modifier
                # 6: Force Strum modifier
                # 7: Open note
                # 8: GHL bottom row modifier (W3)
                if note_value <= 4:
                    if tick not in note_events:
                        note_events[tick] = []
                    note_events[tick].append((note_value, duration))
                elif note_value == 5:
                    force_hopo_ticks.add(tick)
                elif note_value == 6:
                    force_strum_ticks.add(tick)
                elif note_value == 7:
                    # Open note - use special value
                    if tick not in note_events:
                        note_events[tick] = []
                    # Mark as open but still valid
                    note_events[tick].append((-1, duration))  # -1 = open
                elif note_value == 8:
                    # GHL bottom row (W3) - treat as fret 5
                    if tick not in note_events:
                        note_events[tick] = []
                    note_events[tick].append((5, duration))
                continue

            # Parse star power phrases (S 2)
            match = re.match(r"\s*(\d+)\s*=\s*S\s+2\s+(\d+)", line)
            if match:
                start_tick = int(match.group(1))
                duration = int(match.group(2))
                end_tick = start_tick + duration
                star_power_phrases.append((start_tick, end_tick))

        # Convert to NoteEvent objects
        notes = []
        for tick in sorted(note_events.keys()):
            frets = [f for f, _ in note_events[tick] if f >= 0]
            has_open = any(f == -1 for f, _ in note_events[tick])
            max_duration = max(d for _, d in note_events[tick])

            # If only open note, use empty frets list (will be handled by tokenizer)
            if not frets and has_open:
                frets = []  # Open note = no frets pressed

            timestamp_ms = self._ticks_to_ms(tick, bpm_events, resolution)
            duration_ms = self._ticks_to_ms(tick + max_duration, bpm_events, resolution) - timestamp_ms

            # Apply modifiers
            is_hopo = tick in force_hopo_ticks
            is_tap = tick in force_strum_ticks  # N 6 can indicate tap notes
            is_star_power = any(
                start <= tick <= end for start, end in star_power_phrases
            )

            notes.append(NoteEvent(
                timestamp_ms=timestamp_ms,
                frets=sorted(frets),
                duration_ms=duration_ms,
                is_hopo=is_hopo,
                is_tap=is_tap,
                is_star_power=is_star_power,
            ))

        # Calculate song length
        song_length_ms = 0.0
        if notes:
            last_note = notes[-1]
            song_length_ms = last_note.timestamp_ms + last_note.duration_ms

        return ChartData(
            notes=notes,
            instrument=instrument,
            difficulty=difficulty,
            resolution=resolution,
            bpm_events=bpm_events,
            song_length_ms=song_length_ms,
        )

    def _parse_chart_sections(self, content: str) -> Dict[str, List[str]]:
        """Parse .chart file into sections."""
        sections: Dict[str, List[str]] = {}
        current_section = None
        current_lines: List[str] = []

        for line in content.split("\n"):
            line = line.strip()

            # Section header
            match = re.match(r"^\[(.+)\]$", line)
            if match:
                if current_section:
                    sections[current_section] = current_lines
                current_section = match.group(1)
                current_lines = []
                continue

            # Skip braces
            if line in ["{", "}"]:
                continue

            if current_section and line:
                current_lines.append(line)

        # Save last section
        if current_section:
            sections[current_section] = current_lines

        return sections

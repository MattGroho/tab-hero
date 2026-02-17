"""
Export generated tabs to Clone Hero format.

Produces a complete song folder with notes.mid, song.ini, and album.png.
"""

from pathlib import Path
from typing import Dict, List, Optional
import logging
import shutil

logger = logging.getLogger(__name__)

# MIDI note numbers for each difficulty
MIDI_BASE_NOTES = {"easy": 60, "medium": 72, "hard": 84, "expert": 96}


class SongExporter:
    """
    Export generated notes to Clone Hero song format.

    Creates a folder containing:
      - notes.mid: MIDI chart file
      - song.ini: Song metadata
      - album.png: Album art (placeholder)
    """

    def __init__(self, resolution: int = 480):
        self.resolution = resolution

    def export(
        self,
        notes: List[Dict],
        output_dir: str,
        audio_path: Optional[str] = None,
        difficulty: str = "expert",
        instrument: str = "lead",
        song_name: str = "Generated Tab",
        artist: str = "Unknown",
        album: str = "",
        genre: str = "",
        bpm: float = 120.0,
    ) -> Path:
        """
        Export notes to a Clone Hero song folder.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self._write_midi(notes, output_dir / "notes.mid", difficulty, bpm)
        self._write_ini(output_dir / "song.ini", song_name, artist, album, genre)
        self._write_placeholder_album(output_dir / "album.png")

        if audio_path:
            src = Path(audio_path)
            dst = output_dir / "song.ogg"
            if src.suffix.lower() == ".ogg":
                shutil.copy(src, dst)
            else:
                # would need ffmpeg to convert; just copy with original extension
                shutil.copy(src, output_dir / f"song{src.suffix}")

        logger.info(f"Exported song to {output_dir}")
        return output_dir

    def _write_midi(
        self, notes: List[Dict], path: Path, difficulty: str, bpm: float
    ) -> None:
        import mido

        mid = mido.MidiFile(ticks_per_beat=self.resolution)

        # Tempo track
        tempo_track = mido.MidiTrack()
        mid.tracks.append(tempo_track)
        tempo_track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm)))

        # Note track
        note_track = mido.MidiTrack()
        note_track.name = "PART GUITAR"
        mid.tracks.append(note_track)

        base_note = MIDI_BASE_NOTES[difficulty]
        ms_per_tick = (60000.0 / bpm) / self.resolution

        events = []
        for n in notes:
            tick_on = int(n["timestamp_ms"] / ms_per_tick)
            duration_ticks = max(1, int(n.get("duration_ms", 0) / ms_per_tick))
            for fret in n["frets"]:
                midi_note = base_note + fret
                events.append((tick_on, "on", midi_note))
                events.append((tick_on + duration_ticks, "off", midi_note))

        events.sort(key=lambda x: (x[0], x[1] == "on"))

        last_tick = 0
        for tick, event_type, note in events:
            delta = tick - last_tick
            if event_type == "on":
                note_track.append(mido.Message("note_on", note=note, velocity=100, time=delta))
            else:
                note_track.append(mido.Message("note_off", note=note, velocity=0, time=delta))
            last_tick = tick

        note_track.append(mido.MetaMessage("end_of_track", time=0))
        mid.save(str(path))

    def _write_ini(
        self,
        path: Path,
        song_name: str,
        artist: str,
        album: str = "",
        genre: str = "",
    ) -> None:
        lines = [
            "[song]",
            f"name = {song_name}",
            f"artist = {artist}",
            "charter = Tab Hero",
            "diff_guitar = -1",
            "preview_start_time = 0",
        ]
        if album:
            lines.append(f"album = {album}")
        if genre:
            lines.append(f"genre = {genre}")
        path.write_text("\n".join(lines) + "\n")

    def _write_placeholder_album(self, path: Path) -> None:
        # Minimal 1x1 PNG (placeholder)
        png_data = bytes([
            0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,
            0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52,
            0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
            0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x77, 0x53,
            0xDE, 0x00, 0x00, 0x00, 0x0C, 0x49, 0x44, 0x41,
            0x54, 0x08, 0xD7, 0x63, 0x60, 0x60, 0x60, 0x00,
            0x00, 0x00, 0x04, 0x00, 0x01, 0x5C, 0xCD, 0xFF,
            0x69, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E,
            0x44, 0xAE, 0x42, 0x60, 0x82,
        ])
        path.write_bytes(png_data)


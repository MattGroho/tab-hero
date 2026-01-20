"""
Tab Hero: Automatic Guitar Chart Generation from Audio

This package provides tools for automatically generating guitar charts
(Lead Guitar, Bass Guitar, 6-Fret Lead Guitar) at various difficulty
levels (Easy, Medium, Hard, Expert) from audio files.

The system uses a Transformer-based encoder-decoder architecture with:
- Mel spectrogram audio representation (lossy, non-invertible)
- Optional source separation via HTDemucs
- Time-discretized note prediction
- Multi-instrument and multi-difficulty support
"""

from importlib.metadata import metadata, version

_metadata = metadata("tab-hero")

__version__ = version("tab-hero")
# Author-email field is comma-separated, split into individual entries
_author_email = _metadata.get("Author-email") or ""
__authors__ = [author.strip() for author in _author_email.split(",") if author.strip()]


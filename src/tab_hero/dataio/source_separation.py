"""Audio source separation using HTDemucs (htdemucs_6s model)."""

import hashlib
import os
from pathlib import Path
from typing import Dict, Optional, Set

import torch
import torchaudio

_demucs_model = None
_demucs_available = None

# Performance tuning via env vars. DEMUCS_SEGMENT=7.8 is the model's native segment size.
_segment_env = os.environ.get("DEMUCS_SEGMENT", "7.8")
DEMUCS_SEGMENT = None if _segment_env.lower() == "none" else float(_segment_env)
DEMUCS_OVERLAP = float(os.environ.get("DEMUCS_OVERLAP", "0.25"))

REQUIRED_STEMS: Set[str] = {"guitar", "bass", "piano", "other"}

INSTRUMENT_TO_STEM: Dict[str, str] = {
    "lead": "guitar",
    "rhythm": "guitar",
    "bass": "bass",
    "keys": "piano",
}

# Pre-existing stems in Clone Hero/Rock Band packs (use instead of running demucs)
PREEXISTING_STEM_MAP: Dict[str, list] = {
    "guitar": ["guitar.ogg", "guitar.mp3", "guitar.wav"],
    "bass": ["bass.ogg", "bass.mp3", "bass.wav"],
    "piano": ["keys.ogg", "keys.mp3", "keys.wav", "piano.ogg", "piano.mp3"],
    "other": ["rhythm.ogg", "rhythm.mp3", "rhythm.wav"],
}

MIN_STEM_RMS: float = 0.01


def check_demucs_available() -> bool:
    """Check if demucs is available."""
    global _demucs_available
    if _demucs_available is None:
        try:
            from demucs.pretrained import get_model  # noqa: F401
            from demucs.apply import apply_model  # noqa: F401
            _demucs_available = True
        except ImportError:
            _demucs_available = False
    return _demucs_available


def get_demucs_model():
    """Get cached htdemucs_6s model instance."""
    global _demucs_model
    if _demucs_model is None:
        if not check_demucs_available():
            return None
        from demucs.pretrained import get_model
        _demucs_model = get_model("htdemucs_6s")
        if torch.cuda.is_available():
            _demucs_model = _demucs_model.cuda()
    return _demucs_model


def compute_stem_rms(waveform: torch.Tensor) -> float:
    """Compute RMS energy of waveform. Stems < MIN_STEM_RMS are silent."""
    if waveform.numel() == 0:
        return 0.0
    return float(torch.sqrt(torch.mean(waveform.float() ** 2)))


def find_preexisting_stems(song_dir: Path) -> Dict[str, Path]:
    """Find pre-existing stem files in song directory (higher quality than separation)."""
    found_stems = {}
    for stem_name, filenames in PREEXISTING_STEM_MAP.items():
        for filename in filenames:
            stem_path = song_dir / filename
            if stem_path.exists():
                found_stems[stem_name] = stem_path
                break
    return found_stems


def separate_audio_stems(
    waveform: torch.Tensor,
    sample_rate: int,
    cache_dir: Optional[Path] = None,
    audio_path: Optional[Path] = None,
) -> Dict[str, torch.Tensor]:
    """
    Separate audio into stems using HTDemucs.

    Returns dict mapping stem names to waveforms at 44100 Hz.
    """
    model = get_demucs_model()
    if model is None:
        raise RuntimeError("Demucs not available. Install with: pip install demucs")

    if cache_dir is not None and audio_path is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        audio_hash = hashlib.md5(audio_path.read_bytes()).hexdigest()[:12]
        cache_file = cache_dir / f"{audio_hash}_stems.pt"
        if cache_file.exists():
            try:
                return torch.load(cache_file, weights_only=True)
            except Exception:
                pass

    from demucs.apply import apply_model

    model_sr = model.samplerate
    if sample_rate != model_sr:
        waveform = torchaudio.transforms.Resample(sample_rate, model_sr)(waveform)

    # Ensure stereo
    if waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)
    elif waveform.shape[0] > 2:
        waveform = waveform[:2]

    wav = waveform.unsqueeze(0).to(next(model.parameters()).device)

    # Try fast path first, fall back to chunked on OOM
    with torch.no_grad():
        try:
            sources = apply_model(model, wav, progress=False, split=False)
        except (RuntimeError, ValueError, torch.cuda.OutOfMemoryError):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            sources = apply_model(
                model, wav, progress=False, split=True,
                segment=DEMUCS_SEGMENT, overlap=DEMUCS_OVERLAP,
            )

    del wav
    sources = sources.squeeze(0)

    stems = {name: sources[i].cpu() for i, name in enumerate(model.sources) if name in REQUIRED_STEMS}

    del sources
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if cache_dir is not None and audio_path is not None:
        try:
            torch.save(stems, cache_file)
        except Exception:
            pass

    return stems

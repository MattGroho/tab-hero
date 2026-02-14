"""Preprocessing pipeline for audio/charts to .tab format."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import hashlib
import numpy as np

import torch
import torchaudio

from tab_hero.dataio.chart_parser import ChartParser
from tab_hero.dataio.tokenizer import ChartTokenizer
from tab_hero.dataio.tab_format import TabData, save_tab, compute_content_hash, GENRE_MAP
from tab_hero.dataio.source_separation import (
    check_demucs_available,
    compute_stem_rms,
    find_preexisting_stems,
    separate_audio_stems,
    INSTRUMENT_TO_STEM,
    MIN_STEM_RMS,
)

DIFFICULTY_MAP: Dict[str, int] = {"easy": 0, "medium": 1, "hard": 2, "expert": 3}
INSTRUMENT_MAP: Dict[str, int] = {"lead": 0, "bass": 1, "rhythm": 2, "keys": 3}

# Mapping from song.ini genre strings to GENRE_MAP IDs
# Handles variations like "Classic Rock" -> "rock", "Nu Metal" -> "metal"
GENRE_STRING_MAP: Dict[str, int] = {
    # Rock variants
    "rock": GENRE_MAP["rock"],
    "classic rock": GENRE_MAP["rock"],
    "hard rock": GENRE_MAP["rock"],
    "southern rock": GENRE_MAP["rock"],
    "pop-rock": GENRE_MAP["rock"],
    "pop rock": GENRE_MAP["rock"],
    "pop/rock": GENRE_MAP["rock"],
    "glam": GENRE_MAP["rock"],
    "glam rock": GENRE_MAP["rock"],
    "prog": GENRE_MAP["rock"],
    "prog rock": GENRE_MAP["rock"],
    "progressive": GENRE_MAP["rock"],
    "progressive rock": GENRE_MAP["rock"],
    "grunge": GENRE_MAP["rock"],
    "new wave": GENRE_MAP["rock"],
    # Metal variants
    "metal": GENRE_MAP["metal"],
    "heavy metal": GENRE_MAP["metal"],
    "thrash metal": GENRE_MAP["metal"],
    "nu metal": GENRE_MAP["metal"],
    "nu-metal": GENRE_MAP["metal"],
    "power metal": GENRE_MAP["metal"],
    "metalcore": GENRE_MAP["metal"],
    "death metal": GENRE_MAP["metal"],
    "black metal": GENRE_MAP["metal"],
    "doom metal": GENRE_MAP["metal"],
    "speed metal": GENRE_MAP["metal"],
    "alternative metal": GENRE_MAP["metal"],
    # Alternative
    "alternative": GENRE_MAP["alternative"],
    "alternative rock": GENRE_MAP["alternative"],
    "alt rock": GENRE_MAP["alternative"],
    "emo": GENRE_MAP["alternative"],
    # Punk
    "punk": GENRE_MAP["punk"],
    "punk rock": GENRE_MAP["punk"],
    "pop punk": GENRE_MAP["punk"],
    "hardcore": GENRE_MAP["punk"],
    "hardcore punk": GENRE_MAP["punk"],
    # Pop
    "pop": GENRE_MAP["pop"],
    "dance": GENRE_MAP["pop"],
    "pop/dance/electronic": GENRE_MAP["pop"],
    "disco": GENRE_MAP["pop"],
    # Electronic
    "electronic": GENRE_MAP["electronic"],
    "edm": GENRE_MAP["electronic"],
    "synth": GENRE_MAP["electronic"],
    "synthwave": GENRE_MAP["electronic"],
    "industrial": GENRE_MAP["electronic"],
    # Indie
    "indie": GENRE_MAP["indie"],
    "indie rock": GENRE_MAP["indie"],
    "indie pop": GENRE_MAP["indie"],
    # Country
    "country": GENRE_MAP["country"],
    "country rock": GENRE_MAP["country"],
    # Blues
    "blues": GENRE_MAP["blues"],
    "blues rock": GENRE_MAP["blues"],
    # Jazz
    "jazz": GENRE_MAP["jazz"],
    "jazz fusion": GENRE_MAP["jazz"],
    "fusion": GENRE_MAP["jazz"],
    # Classical
    "classical": GENRE_MAP["classical"],
    "orchestral": GENRE_MAP["classical"],
    # Hip-hop
    "hip-hop": GENRE_MAP["hiphop"],
    "hip hop": GENRE_MAP["hiphop"],
    "hiphop": GENRE_MAP["hiphop"],
    "rap": GENRE_MAP["hiphop"],
    "r&b": GENRE_MAP["hiphop"],
    "r&b/soul/funk": GENRE_MAP["hiphop"],
    "hip-hop/rap": GENRE_MAP["hiphop"],
    "urban": GENRE_MAP["hiphop"],
    # Reggae
    "reggae": GENRE_MAP["reggae"],
    "ska": GENRE_MAP["reggae"],
    "ska punk": GENRE_MAP["reggae"],
    "reggae/ska": GENRE_MAP["reggae"],
    # Folk
    "folk": GENRE_MAP["folk"],
    "folk rock": GENRE_MAP["folk"],
    "acoustic": GENRE_MAP["folk"],
    # Additional metal subgenres
    "progressive metal": GENRE_MAP["metal"],
    "melodic death metal": GENRE_MAP["metal"],
    "glam metal": GENRE_MAP["metal"],
    "groove metal": GENRE_MAP["metal"],
    # Additional rock subgenres
    "psychedelic rock": GENRE_MAP["rock"],
    "instrumental rock": GENRE_MAP["rock"],
    "post-grunge": GENRE_MAP["rock"],
    # Additional alternative/punk
    "power pop": GENRE_MAP["alternative"],
    "post-hardcore": GENRE_MAP["punk"],
    # Other
    "other": GENRE_MAP["other"],
    "novelty": GENRE_MAP["other"],
    "soundtrack": GENRE_MAP["other"],
    "video game": GENRE_MAP["other"],
    "virtuoso": GENRE_MAP["other"],
    "comedy": GENRE_MAP["other"],
}

MEL_CONFIG: Dict[str, int] = {
    "sample_rate": 22050,
    "n_fft": 2048,
    "hop_length": 256,  # ~11.6ms per frame
    "n_mels": 128,
}


def parse_song_ini(song_dir: Path) -> Dict[str, str]:
    """
    Parse song.ini and return metadata as a dict.

    Args:
        song_dir: Directory containing song.ini

    Returns:
        Dict with keys like 'genre', 'artist', 'name', etc.
        Empty dict if song.ini not found or parsing fails.
    """
    ini_path = song_dir / "song.ini"
    if not ini_path.exists():
        return {}

    metadata: Dict[str, str] = {}
    try:
        with open(ini_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if "=" in line and not line.startswith("["):
                    key, _, value = line.partition("=")
                    metadata[key.strip().lower()] = value.strip()
    except Exception:
        return {}

    return metadata


def map_genre_to_id(genre_str: Optional[str]) -> int:
    """
    Map a genre string from song.ini to a GENRE_MAP ID.

    Args:
        genre_str: Genre string from song.ini (e.g., "Classic Rock", "Nu Metal")

    Returns:
        Genre ID from GENRE_MAP. Returns 0 (unknown) if genre_str is None/empty,
        or GENRE_MAP["other"] if the genre is not in GENRE_STRING_MAP.
    """
    if not genre_str:
        return GENRE_MAP["unknown"]

    genre_lower = genre_str.lower().strip()
    return GENRE_STRING_MAP.get(genre_lower, GENRE_MAP["other"])


# Cached transforms (lazy-initialized, separate CPU/GPU caches)
_mel_transform_cpu: Optional[torchaudio.transforms.MelSpectrogram] = None
_mel_transform_gpu: Optional[torchaudio.transforms.MelSpectrogram] = None
_resampler_44100_to_22050: Optional[torchaudio.transforms.Resample] = None


def get_mel_transform(device: str = "cpu") -> torchaudio.transforms.MelSpectrogram:
    """Get cached MelSpectrogram transform for the given device."""
    global _mel_transform_cpu, _mel_transform_gpu

    if device == "cpu" or not torch.cuda.is_available():
        if _mel_transform_cpu is None:
            _mel_transform_cpu = torchaudio.transforms.MelSpectrogram(
                sample_rate=MEL_CONFIG["sample_rate"],
                n_fft=MEL_CONFIG["n_fft"],
                hop_length=MEL_CONFIG["hop_length"],
                n_mels=MEL_CONFIG["n_mels"],
            )
        return _mel_transform_cpu
    else:
        if _mel_transform_gpu is None:
            _mel_transform_gpu = torchaudio.transforms.MelSpectrogram(
                sample_rate=MEL_CONFIG["sample_rate"],
                n_fft=MEL_CONFIG["n_fft"],
                hop_length=MEL_CONFIG["hop_length"],
                n_mels=MEL_CONFIG["n_mels"],
            ).cuda()
        return _mel_transform_gpu


def get_resampler(orig_sr: int, target_sr: int) -> torchaudio.transforms.Resample:
    """Get cached resampler. Caches 44100->22050 (Demucs output)."""
    global _resampler_44100_to_22050

    if orig_sr == 44100 and target_sr == 22050:
        if _resampler_44100_to_22050 is None:
            _resampler_44100_to_22050 = torchaudio.transforms.Resample(44100, 22050)
        return _resampler_44100_to_22050
    return torchaudio.transforms.Resample(orig_sr, target_sr)


def compute_audio_hash(audio_path: Path) -> str:
    """Compute MD5 hash of audio file (truncated to 12 hex chars)."""
    return hashlib.md5(audio_path.read_bytes()).hexdigest()[:12]


def get_mel_cache_path(cache_dir: Path, audio_hash: str, stem_name: str) -> Path:
    """Get cache file path for a mel spectrogram."""
    return cache_dir / f"{audio_hash}_{stem_name}_mel.npz"


def save_mel_to_cache(
    cache_path: Path,
    mel: np.ndarray,
    sample_rate: int,
    duration_sec: float,
) -> bool:
    """Save mel to cache as compressed npz. Returns True on success."""
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            mel=mel.astype(np.float16),  # float16 for space savings
            sample_rate=sample_rate,
            duration_sec=duration_sec,
        )
        return True
    except Exception:
        return False


def load_mel_from_cache(
    cache_path: Path,
) -> Optional[Tuple[np.ndarray, int, float]]:
    """Load mel from cache. Returns (mel, sample_rate, duration) or None."""
    if not cache_path.exists():
        return None
    try:
        data = np.load(cache_path)
        return data["mel"].astype(np.float32), int(data["sample_rate"]), float(data["duration_sec"])
    except Exception:
        return None


def find_audio_file(song_dir: Path) -> Optional[Path]:
    """Find the primary audio file in a song directory."""
    for name in ["song.ogg", "guitar.ogg", "song.mp3", "song.wav"]:
        p = song_dir / name
        if p.exists():
            return p
    for ext in [".ogg", ".mp3", ".wav", ".mp4", ".m4a"]:
        files = list(song_dir.glob(f"*{ext}"))
        if files:
            return files[0]
    return None


def find_chart_file(song_dir: Path) -> Optional[Path]:
    """Find the chart file in a song directory."""
    for name in ["notes.chart", "notes.mid"]:
        p = song_dir / name
        if p.exists():
            return p
    for ext in [".chart", ".mid", ".midi"]:
        files = list(song_dir.glob(f"*{ext}"))
        if files:
            return files[0]
    return None


def is_song_directory(path: Path) -> bool:
    """Check if a directory contains song files (audio + chart)."""
    return find_audio_file(path) is not None and find_chart_file(path) is not None


def discover_song_directories(root: Path) -> List[Path]:
    """Recursively discover all song directories under root."""
    song_dirs = []

    def _search(path: Path):
        if is_song_directory(path):
            song_dirs.append(path)
        else:
            for child in path.iterdir():
                if child.is_dir():
                    _search(child)

    _search(root)
    return song_dirs


def load_audio(audio_path: Path) -> Tuple[torch.Tensor, int]:
    """Load audio file using available backend."""
    try:
        waveform, sr = torchaudio.load(str(audio_path))
        return waveform, sr
    except (ImportError, RuntimeError):
        pass

    try:
        import librosa
        waveform, sr = librosa.load(str(audio_path), sr=None, mono=False)
        if waveform.ndim == 1:
            waveform = waveform[np.newaxis, :]
        return torch.from_numpy(waveform.astype(np.float32)), sr
    except ImportError:
        pass

    import soundfile as sf
    waveform, sr = sf.read(str(audio_path))
    if waveform.ndim == 1:
        waveform = waveform[np.newaxis, :]
    else:
        waveform = waveform.T
    return torch.from_numpy(waveform.astype(np.float32)), sr


def waveform_to_mel(
    waveform: torch.Tensor,
    sr: int,
    normalize: bool = True,
    use_gpu: bool = False,
) -> Tuple[np.ndarray, int, float]:
    """Convert waveform to mel spectrogram. Returns (mel, sample_rate, duration_sec)."""
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"

    if sr != MEL_CONFIG["sample_rate"]:
        waveform = get_resampler(sr, MEL_CONFIG["sample_rate"])(waveform)
        sr = MEL_CONFIG["sample_rate"]

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    duration_sec = waveform.shape[1] / sr

    if device == "cuda":
        waveform = waveform.cuda()

    mel = get_mel_transform(device)(waveform)
    mel = torch.log(mel.clamp(min=1e-5))

    if normalize:
        mel_std = mel.std()
        if mel_std > 1e-6:
            mel = (mel - mel.mean()) / mel_std

    return mel.squeeze(0).cpu().numpy(), MEL_CONFIG["sample_rate"], duration_sec


def extract_mel_spectrogram(
    audio_path: Path,
    normalize: bool = True
) -> Tuple[np.ndarray, int, float]:
    """Extract mel spectrogram from audio file."""
    waveform, sr = load_audio(audio_path)
    return waveform_to_mel(waveform, sr, normalize)


def process_song_all_variants(args: Tuple) -> Dict[str, Any]:
    """
    Process a single song for all difficulty/instrument combinations.

    Args:
        args: (song_dir, output_dir, difficulties, instruments, filters)
              filters dict may contain: min_notes, min_duration, max_duration,
              use_separation, stem_cache_dir, parser, tokenizer, use_gpu_mel,
              song_id (sequential index for grouping variants)
    """
    song_dir, output_dir, difficulties, instruments, filters = args
    song_id = filters.get("song_id", 0)

    audio_path = find_audio_file(song_dir)
    chart_path = find_chart_file(song_dir)

    if audio_path is None or chart_path is None:
        n_variants = len(difficulties) * len(instruments)
        return {"successful": 0, "skipped": n_variants, "errors": 0,
                "song_dir": str(song_dir), "error_msgs": [], "duration_sec": 0}

    use_separation = filters.get("use_separation", True)
    stem_cache_dir = filters.get("stem_cache_dir")
    min_stem_rms = filters.get("min_stem_rms", MIN_STEM_RMS)
    use_gpu_mel = filters.get("use_gpu_mel", True)

    # Reuse parser/tokenizer from filters if provided, otherwise create new
    parser = filters.get("parser") or ChartParser()
    tokenizer = filters.get("tokenizer") or ChartTokenizer()

    # Extract genre from song.ini
    song_metadata = parse_song_ini(song_dir)
    genre_id = map_genre_to_id(song_metadata.get("genre"))

    # Determine which stems we need based on requested instruments
    needed_stems: Set[str] = set()
    for inst in instruments:
        needed_stems.add(INSTRUMENT_TO_STEM.get(inst, "other"))

    stem_mels: Dict[str, Tuple[np.ndarray, int, float]] = {}
    skipped_stems: Set[str] = set()
    duration_sec = 0.0

    # Compute audio hash for mel caching (if cache enabled)
    audio_hash = compute_audio_hash(audio_path) if stem_cache_dir else None

    try:
        # First, check mel cache for all needed stems
        if stem_cache_dir and audio_hash:
            for stem_name in list(needed_stems):
                cache_path = get_mel_cache_path(stem_cache_dir, audio_hash, stem_name)
                cached_mel = load_mel_from_cache(cache_path)
                if cached_mel is not None:
                    mel, sr, dur = cached_mel
                    stem_mels[stem_name] = (mel, sr, dur)
                    duration_sec = max(duration_sec, dur)

        # Check for pre-existing stems (from song packs) for remaining stems
        remaining_after_cache = needed_stems - set(stem_mels.keys())
        preexisting = find_preexisting_stems(song_dir)
        has_preexisting_stems = len(preexisting) > 0

        for stem_name in remaining_after_cache:
            if stem_name in preexisting:
                stem_path = preexisting[stem_name]
                waveform, stem_sr = load_audio(stem_path)
                rms = compute_stem_rms(waveform)
                if rms < min_stem_rms:
                    skipped_stems.add(stem_name)
                    continue
                # Pre-existing stems are on CPU, use CPU mel
                mel, sr, dur = waveform_to_mel(waveform, stem_sr, use_gpu=False)
                stem_mels[stem_name] = (mel, sr, dur)
                duration_sec = max(duration_sec, dur)
                # Cache the mel for future runs
                if stem_cache_dir and audio_hash:
                    cache_path = get_mel_cache_path(stem_cache_dir, audio_hash, stem_name)
                    save_mel_to_cache(cache_path, mel, sr, dur)
                # Free waveform
                del waveform

        remaining_stems = needed_stems - set(stem_mels.keys()) - skipped_stems

        # If pre-existing stems were found, don't run Demucs for missing stems
        # Missing stems in a pre-separated pack likely means they don't exist in the song
        if remaining_stems and not has_preexisting_stems:
            if use_separation and check_demucs_available():
                waveform, audio_sr = load_audio(audio_path)
                stems = separate_audio_stems(
                    waveform, audio_sr,
                    cache_dir=None,  # Don't cache raw stems, we cache mels instead
                    audio_path=audio_path
                )
                # Free input waveform immediately
                del waveform
                demucs_sr = 44100

                for stem_name in remaining_stems:
                    stem_wav = None
                    if stem_name in stems:
                        stem_wav = stems[stem_name]
                    elif "other" in stems:
                        stem_wav = stems["other"]

                    if stem_wav is not None:
                        rms = compute_stem_rms(stem_wav)
                        if rms < min_stem_rms:
                            skipped_stems.add(stem_name)
                            continue
                        # Demucs output benefits from GPU mel extraction
                        mel, sr, dur = waveform_to_mel(
                            stem_wav, demucs_sr, use_gpu=use_gpu_mel
                        )
                        stem_mels[stem_name] = (mel, sr, dur)
                        duration_sec = max(duration_sec, dur)
                        # Cache the mel for future runs
                        if stem_cache_dir and audio_hash:
                            cache_path = get_mel_cache_path(
                                stem_cache_dir, audio_hash, stem_name
                            )
                            save_mel_to_cache(cache_path, mel, sr, dur)

                # Free stems after processing to prevent memory buildup
                del stems
                import gc
                gc.collect()
            else:
                mel, sr, duration_sec = extract_mel_spectrogram(audio_path)
                for stem_name in remaining_stems:
                    stem_mels[stem_name] = (mel, sr, duration_sec)
                    # Cache mixed audio mel too
                    if stem_cache_dir and audio_hash:
                        cache_path = get_mel_cache_path(
                            stem_cache_dir, audio_hash, stem_name
                        )
                        save_mel_to_cache(cache_path, mel, sr, duration_sec)

    except Exception as e:
        n_variants = len(difficulties) * len(instruments)
        return {"successful": 0, "skipped": 0, "errors": n_variants,
                "song_dir": str(song_dir), "error_msgs": [f"Audio error: {e}"],
                "duration_sec": 0}

    # Apply duration filters
    min_duration = filters.get("min_duration")
    max_duration = filters.get("max_duration")
    if min_duration is not None and duration_sec < min_duration:
        n_variants = len(difficulties) * len(instruments)
        return {"successful": 0, "skipped": n_variants, "errors": 0,
                "song_dir": str(song_dir),
                "error_msgs": [f"Duration {duration_sec:.1f}s < min {min_duration}s"],
                "duration_sec": duration_sec}
    if max_duration is not None and duration_sec > max_duration:
        n_variants = len(difficulties) * len(instruments)
        return {"successful": 0, "skipped": n_variants, "errors": 0,
                "song_dir": str(song_dir),
                "error_msgs": [f"Duration {duration_sec:.1f}s > max {max_duration}s"],
                "duration_sec": duration_sec}

    min_notes = filters.get("min_notes", 1)

    successful = 0
    skipped = 0
    errors = 0
    error_msgs: List[str] = []

    for difficulty in difficulties:
        for instrument in instruments:
            try:
                chart_data = parser.parse(
                    chart_path, instrument=instrument, difficulty=difficulty
                )

                if len(chart_data.notes) < min_notes:
                    skipped += 1
                    continue

                tokens = tokenizer.encode_chart(chart_data)
                tokens = np.array(tokens, dtype=np.int16)

                stem_name = INSTRUMENT_TO_STEM.get(instrument, "other")
                mel, sr, _ = stem_mels.get(
                    stem_name, stem_mels.get("other", (None, 0, 0))
                )
                if mel is None:
                    skipped += 1
                    continue

                content_hash = compute_content_hash(mel, tokens)
                variant_hash = hashlib.sha256(
                    f"{content_hash}_{difficulty}_{instrument}".encode()
                ).hexdigest()[:16]

                tab_data = TabData(
                    mel_spectrogram=mel,
                    sample_rate=sr,
                    hop_length=MEL_CONFIG["hop_length"],
                    note_tokens=tokens,
                    difficulty_id=DIFFICULTY_MAP[difficulty],
                    instrument_id=INSTRUMENT_MAP[instrument],
                    content_hash=variant_hash,
                    genre_id=genre_id,
                    song_id=song_id,
                )

                out_path = output_dir / f"{variant_hash}.tab"
                if out_path.exists():
                    skipped += 1
                    continue
                save_tab(tab_data, out_path)
                successful += 1

            except ValueError:
                skipped += 1
            except Exception as e:
                errors += 1
                if len(error_msgs) < 3:
                    error_msgs.append(f"{difficulty}/{instrument}: {e}")

    return {
        "successful": successful,
        "skipped": skipped,
        "errors": errors,
        "song_dir": str(song_dir),
        "error_msgs": error_msgs,
        "duration_sec": duration_sec,
    }


def cleanup_source_directory(song_dir: Path) -> None:
    """Remove all source files from a song directory after processing."""
    for f in song_dir.iterdir():
        if f.is_file():
            f.unlink()

    try:
        song_dir.rmdir()
        parent = song_dir.parent
        while parent.exists() and not any(parent.iterdir()):
            parent.rmdir()
            parent = parent.parent
    except OSError:
        pass


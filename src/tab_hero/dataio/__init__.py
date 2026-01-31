"""Data loading and processing."""

from .audio_processor import AudioProcessor
from .chart_parser import ChartParser
from .dataset import TabHeroDataset
from .preprocessing import (
    DIFFICULTY_MAP,
    INSTRUMENT_MAP,
    MEL_CONFIG,
    cleanup_source_directory,
    compute_audio_hash,
    discover_song_directories,
    extract_mel_spectrogram,
    find_audio_file,
    find_chart_file,
    get_mel_cache_path,
    get_mel_transform,
    get_resampler,
    load_audio,
    load_mel_from_cache,
    process_song_all_variants,
    save_mel_to_cache,
    waveform_to_mel,
)
from .source_separation import (
    INSTRUMENT_TO_STEM,
    MIN_STEM_RMS,
    check_demucs_available,
    compute_stem_rms,
    find_preexisting_stems,
    get_demucs_model,
    separate_audio_stems,
)
from .tab_dataset import TabDataset
from .tab_format import TabData, load_tab, save_tab
from .tokenizer import ChartTokenizer
from .feature_extractor import (
    TabFeatures,
    extract_features,
    extract_features_batch,
    extract_features_from_file,
)

__all__ = [
    "AudioProcessor",
    "ChartParser",
    "ChartTokenizer",
    "DIFFICULTY_MAP",
    "INSTRUMENT_MAP",
    "INSTRUMENT_TO_STEM",
    "MEL_CONFIG",
    "MIN_STEM_RMS",
    "TabData",
    "TabDataset",
    "TabFeatures",
    "TabHeroDataset",
    "check_demucs_available",
    "cleanup_source_directory",
    "compute_audio_hash",
    "compute_stem_rms",
    "discover_song_directories",
    "extract_features",
    "extract_features_batch",
    "extract_features_from_file",
    "extract_mel_spectrogram",
    "find_audio_file",
    "find_chart_file",
    "find_preexisting_stems",
    "get_demucs_model",
    "get_mel_cache_path",
    "get_mel_transform",
    "get_resampler",
    "load_audio",
    "load_mel_from_cache",
    "load_tab",
    "process_song_all_variants",
    "save_mel_to_cache",
    "save_tab",
    "separate_audio_stems",
    "waveform_to_mel",
]


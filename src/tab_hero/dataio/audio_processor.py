"""Audio processing utilities for mel spectrogram extraction."""

from pathlib import Path
from typing import Dict, Optional, Union
import numpy as np
import torch
import librosa


DEFAULT_MEL_CONFIG = {
    "sample_rate": 22050,
    "n_mels": 128,
    "n_fft": 2048,
    "hop_length": 512,
    "f_min": 20.0,
    "f_max": 8000.0,
}


class AudioProcessor:
    """Processes audio files to mel spectrograms."""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or DEFAULT_MEL_CONFIG.copy()

    @property
    def sample_rate(self) -> int:
        return self.config["sample_rate"]

    @property
    def hop_length(self) -> int:
        return self.config["hop_length"]

    @property
    def frame_rate(self) -> float:
        """Frames per second."""
        return self.sample_rate / self.hop_length

    @property
    def frame_duration_ms(self) -> float:
        """Duration of each mel frame in milliseconds."""
        return 1000.0 / self.frame_rate

    def load_audio(self, path: Union[str, Path]) -> np.ndarray:
        """Load audio file and resample to target sample rate."""
        path = Path(path)
        waveform, _ = librosa.load(str(path), sr=self.sample_rate, mono=True)
        return waveform

    def encode(self, waveform: np.ndarray) -> torch.Tensor:
        """Convert waveform to log-mel spectrogram.

        Args:
            waveform: 1D numpy array

        Returns:
            2D tensor of shape (n_mels, time_frames)
        """
        mel_spec = librosa.feature.melspectrogram(
            y=waveform,
            sr=self.sample_rate,
            n_mels=self.config["n_mels"],
            n_fft=self.config["n_fft"],
            hop_length=self.hop_length,
            fmin=self.config["f_min"],
            fmax=self.config["f_max"],
        )
        log_mel = np.log(mel_spec + 1e-9)
        return torch.from_numpy(log_mel).float()

    def process_audio_file(self, path: Union[str, Path]) -> torch.Tensor:
        """Load audio file and return log-mel spectrogram."""
        waveform = self.load_audio(path)
        return self.encode(waveform)

    def get_duration_ms(self, mel_spec: torch.Tensor) -> float:
        """Get duration in milliseconds from mel spectrogram."""
        num_frames = mel_spec.shape[-1]
        return num_frames * self.frame_duration_ms

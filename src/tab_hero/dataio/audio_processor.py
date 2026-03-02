"""Audio processing for mel spectrogram extraction."""

import json
from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np

import torch
import torchaudio

DEFAULT_MEL_CONFIG = {
    "sample_rate": 22050,
    "n_fft": 2048,
    "hop_length": 256,
    "n_mels": 128,
}


def load_mel_config_from_manifest(data_dir: Union[str, Path]) -> dict:
    """Load mel config from manifest.json if present."""
    manifest_path = Path(data_dir) / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
            if "mel_config" in manifest:
                return manifest["mel_config"]
    return DEFAULT_MEL_CONFIG


class AudioProcessor:
    """Converts audio to mel spectrograms for training/inference."""

    def __init__(
        self,
        device: str = "cuda",
        mel_config: Optional[dict] = None,
        data_dir: Optional[Union[str, Path]] = None,
    ):
        self.device = device

        if mel_config is not None:
            config = mel_config
        elif data_dir is not None:
            config = load_mel_config_from_manifest(data_dir)
        else:
            config = DEFAULT_MEL_CONFIG

        self.sample_rate = config["sample_rate"]
        self.hop_length = config["hop_length"]
        self.n_mels = config["n_mels"]
        self.n_fft = config["n_fft"]

        self._mel_transform = None

    @property
    def mel_transform(self):
        """Lazy-loaded mel spectrogram transform."""
        if self._mel_transform is None:
            self._mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.n_mels,
            ).to(self.device)
        return self._mel_transform

    def load_audio(
        self,
        path: Union[str, Path],
        normalize: bool = True
    ) -> Tuple[torch.Tensor, int]:
        """Load audio file, resample, and optionally normalize."""
        import librosa

        audio, sr = librosa.load(str(path), sr=self.sample_rate, mono=True)
        waveform = torch.from_numpy(audio).unsqueeze(0).float()

        if normalize:
            waveform = waveform / (waveform.abs().max() + 1e-8)

        return waveform, self.sample_rate

    def encode(self, waveform: torch.Tensor) -> torch.Tensor:
        """Encode waveform (1, samples) to mel spectrogram (n_frames, n_mels).

        Applies log-mel transform followed by per-file z-normalization to match
        the preprocessing pipeline (see preprocessing.waveform_to_mel).
        """
        waveform = waveform.to(self.device)
        with torch.no_grad():
            mel = self.mel_transform(waveform)
            mel = torch.log(mel.clamp(min=1e-5))
            # Z-normalize to match preprocessing (preprocessing.py:372-375)
            mel_std = mel.std()
            if mel_std > 1e-6:
                mel = (mel - mel.mean()) / mel_std
            mel = mel.squeeze(0).transpose(0, 1)
        return mel

    @property
    def embedding_dim(self) -> int:
        return self.n_mels

    @property
    def frame_rate(self) -> float:
        return self.sample_rate / self.hop_length

    @property
    def frame_duration_ms(self) -> float:
        return 1000.0 / self.frame_rate

    def get_frame_timestamps(self, n_frames: int) -> np.ndarray:
        """Get timestamps in milliseconds for each frame."""
        return np.arange(n_frames) * self.frame_duration_ms

    def process_audio_file(
        self,
        path: Union[str, Path],
        max_duration_sec: Optional[float] = None
    ) -> Tuple[torch.Tensor, np.ndarray]:
        """Load audio and compute mel spectrogram. Returns (mel, timestamps_ms)."""
        waveform, sr = self.load_audio(path)
        if max_duration_sec is not None:
            waveform = waveform[:, :int(max_duration_sec * sr)]
        mel = self.encode(waveform)
        return mel, self.get_frame_timestamps(mel.shape[0])


def spec_augment(
    mel: torch.Tensor,
    freq_mask_param: int = 27,
    time_mask_param: int = 100,
    n_freq_masks: int = 2,
    n_time_masks: int = 2,
) -> torch.Tensor:
    """Apply SpecAugment to a mel spectrogram tensor.

    Applies frequency and time masking as described in
    Park et al., "SpecAugment: A Simple Data Augmentation Method
    for Automatic Speech Recognition", 2019.

    Args:
        mel: (n_frames, n_mels) mel spectrogram tensor.
        freq_mask_param: Maximum width of each frequency mask.
        time_mask_param: Maximum width of each time mask.
        n_freq_masks: Number of frequency masks to apply.
        n_time_masks: Number of time masks to apply.

    Returns:
        Augmented mel tensor (same shape, in-place on a clone).
    """
    mel = mel.clone()
    n_frames, n_mels = mel.shape

    # Frequency masking
    for _ in range(n_freq_masks):
        f = torch.randint(0, min(freq_mask_param, n_mels) + 1, (1,)).item()
        if f == 0:
            continue
        f0 = torch.randint(0, max(n_mels - f, 1), (1,)).item()
        mel[:, f0 : f0 + f] = 0.0

    # Time masking
    for _ in range(n_time_masks):
        t = torch.randint(0, min(time_mask_param, n_frames) + 1, (1,)).item()
        if t == 0:
            continue
        t0 = torch.randint(0, max(n_frames - t, 1), (1,)).item()
        mel[t0 : t0 + t, :] = 0.0

    return mel
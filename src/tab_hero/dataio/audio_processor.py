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
        """Encode waveform (1, samples) to mel spectrogram (n_frames, n_mels)."""
        waveform = waveform.to(self.device)
        with torch.no_grad():
            mel = self.mel_transform(waveform)
            mel = torch.log(mel.clamp(min=1e-5))
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


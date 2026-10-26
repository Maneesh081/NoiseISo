"""Spectrogram (STFT) utilities.

An STFT nibbles a waveform into short overlapping slices, then reports how much of each
frequency is present in each slice. The result is a *magnitude* matrix: rows = frequency,
columns = time. The model treats this matrix like a grayscale image.
"""

import torch


class Spectrogram:
    def __init__(self, n_fft: int = 512, hop_length: int = 128, win_length: int = 512):
        self.n_fft = n_fft
        self.hop = hop_length
        self.win = win_length
        self.freq_bins = n_fft // 2 + 1  # 257 rows
        self._window = torch.hann_window(win_length)

    def to_device(self, device):
        self._window = self._window.to(device)
        return self

    def magnitude(self, waveform: torch.Tensor) -> torch.Tensor:
        """waveform: (B, T_samples) -> magnitude: (B, freq_bins, T_frames)."""
        spec = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop,
            win_length=self.win,
            window=self._window.to(waveform.device),
            center=True,
            return_complex=True,
        )
        return spec.abs()

    def phase(self, waveform: torch.Tensor) -> torch.Tensor:
        """waveform: (B, T_samples) -> phase: (B, freq_bins, T_frames)."""
        spec = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop,
            win_length=self.win,
            window=self._window.to(waveform.device),
            center=True,
            return_complex=True,
        )
        return spec.angle()

    def recover(self, magnitude: torch.Tensor, phase: torch.Tensor,
                length: int) -> torch.Tensor:
        """Combine enhanced magnitude with the (noisy) phase and invert to audio."""
        spec = magnitude * torch.exp(1j * phase)
        return torch.istft(
            spec,
            n_fft=self.n_fft,
            hop_length=self.hop,
            win_length=self.win,
            window=self._window.to(magnitude.device),
            center=True,
            length=length,
        )

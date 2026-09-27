"""SpectrogramExtractor: encodes waveforms to (log) linear spectrogram frames."""
from __future__ import annotations

import torch
import torch.nn as nn
import torchaudio

from vic.core import FrameGrid
from vic.data.audio_batch import AudioBatch


class SpectrogramExtractor(nn.Module):
    """Linear spectrogram feature extractor satisfying the FeatureExtractor protocol.

    No learned parameters — the STFT is fixed.  Kept as an nn.Module so it
    moves to the right device with .to(device).

    Parameters
    ----------
    sample_rate : target audio sample rate.
    n_fft       : FFT window size in samples (latent_dim = n_fft // 2 + 1).
    win_length  : analysis window in samples.  Defaults to n_fft.
    hop_length  : hop size in samples (determines frame_grid).
    power       : exponent for the magnitude spectrogram.  1.0 = magnitude,
                  2.0 = power spectrogram.
    log         : if True, apply log(spec + 1e-6) compression.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 1024,
        win_length: int | None = None,
        hop_length: int = 320,
        power: float = 2.0,
        log: bool = True,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.latent_dim = n_fft // 2 + 1
        self.frame_grid = FrameGrid(hop_size=hop_length, sample_rate=sample_rate)
        self._log = log
        self._spec = torchaudio.transforms.Spectrogram(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            power=power,
        )

    def encode(self, batch: AudioBatch) -> AudioBatch:
        """batch : AudioBatch (B, 1, T) → AudioBatch (B, n_fft//2+1, T_frames)."""
        spec = self._spec(batch.BCT)     # (B, 1, n_fft//2+1, T_frames+1) with center=True
        spec = spec.squeeze(1)           # (B, n_fft//2+1, T_frames+1)
        if self._log:
            spec = torch.log(spec + 1e-6)
        frame_lengths = batch.lengths // self.frame_grid.hop_size
        # Drop the extra frame produced by center=True padding (same as MelSpectrogramExtractor).
        spec = spec[..., :spec.shape[-1] - 1]
        return batch.as_frames(spec, frame_lengths)

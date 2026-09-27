"""MelSpectrogramExtractor: encodes waveforms to log-mel spectrogram frames."""
from __future__ import annotations

import torch
import torch.nn as nn
import torchaudio

from vic.core import FrameGrid
from vic.data.audio_batch import AudioBatch


class MelSpectrogramExtractor(nn.Module):
    """Mel spectrogram feature extractor satisfying the FeatureExtractor protocol.

    No learned parameters — the filter bank is fixed.  Kept as an nn.Module
    so it moves to the right device with .to(device).

    Parameters
    ----------
    sample_rate : target audio sample rate.
    n_mels      : number of mel filter banks (= latent_dim).
    n_fft       : FFT window size in samples.
    hop_length  : hop size in samples (determines frame_grid).
    log         : if True, apply log(mel + 1e-6) compression.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 128,
        n_fft: int = 1024,
        win_length = None,
        hop_length: int = 320,
        log: bool = True,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.latent_dim = n_mels
        self.frame_grid = FrameGrid(hop_size=hop_length, sample_rate=sample_rate)
        self._log = log
        self._mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
        )

    def encode(self, batch: AudioBatch) -> AudioBatch:
        """batch : AudioBatch (B, 1, T) → AudioBatch (B, n_mels, T_frames)."""
        mel = self._mel(batch.BCT)       # (B, 1, n_mels, T_frames+1) with center=True
        mel = mel.squeeze(1)             # (B, n_mels, T_frames+1)
        if self._log:
            mel = torch.log(mel + 1e-6)
        frame_lengths = batch.lengths // self.frame_grid.hop_size
        # Drop the extra frame produced by center=True padding: when T is a
        # multiple of hop_size (guaranteed by pad_to_multiple), torchaudio
        # yields T//hop + 1 frames; the last one is centred past the signal end.
        mel = mel[..., :mel.shape[-1] - 1]
        return batch.as_frames(mel, frame_lengths)

"""Stub codec, predictor and datasets shared by the training and conversion tests.

The real codec, predictor and corpus all live on the cluster, so the tests run the
real modules around these stand-ins.
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from vic.data.audio_batch import AudioBatch

SR, HOP, D = 16_000, 320, 8
EMBED = {"type": "sinusoidal", "embed_dim": 16, "d_sin": 16}
RANGE = (40.0, 80.0)


class _StubExtractor(nn.Module):
    """Waveform ↔ latent at the codec's frame rate, with a non-differentiable decode."""

    def __init__(self):
        super().__init__()
        self.enc = nn.Linear(HOP, D)
        self.dec = nn.Linear(D, HOP)

    def encode(self, wav_batch: AudioBatch) -> AudioBatch:
        wav = wav_batch.data[:, 0]                                  # (B, S)
        n = wav.shape[-1] // HOP
        frames = wav[:, : n * HOP].reshape(wav.shape[0], n, HOP)
        return AudioBatch(
            data=self.enc(frames).transpose(1, 2),                  # (B, D, T)
            lengths=wav_batch.lengths // HOP,
            sample_rate=SR // HOP,
        )

    @torch.no_grad()
    def decode(self, z: AudioBatch) -> AudioBatch:
        wav = self.dec(z.BTC).reshape(z.data.shape[0], 1, -1)       # (B, 1, S)
        return AudioBatch(data=wav, lengths=z.lengths * HOP, sample_rate=SR)


class _StubPipeline(nn.Module):
    def __init__(self, extractor, whitening):
        super().__init__()
        self.extractor = extractor
        self.whitening = whitening
        # The real ExtractionPipeline freezes its extractor (extraction_pipeline.py:81),
        # so z_real carries no graph back into the codec. Without this the D-step and the
        # G-step would both backward through the encoder and the second would find the
        # graph already freed — an artefact of the stub, not of the module.
        for param in self.parameters():
            param.requires_grad_(False)

    def encode(self, wav_batch, training=False):
        return self.extractor.encode(wav_batch)


class _StubPredictor(nn.Module):
    """Emits a per-frame dB curve, as the real P_φ does."""

    def __init__(self):
        super().__init__()
        self.head = nn.Linear(D, 1)

    def forward(self, z: AudioBatch) -> torch.Tensor:
        return 60.0 + 15.0 * torch.tanh(self.head(z.BTC).squeeze(-1))


class _WavDataset(Dataset):
    """Deliberately ragged: padding is where per-frame conditioning goes wrong."""

    def __init__(self, n=8):
        self.lengths = [(20 + i % 5) * HOP for i in range(n)]

    def __len__(self):
        return len(self.lengths)

    def __getitem__(self, i):
        return {"wav": torch.randn(1, self.lengths[i]) * 0.1}


class _LabelledDataset(_WavDataset):
    """What IntensityDataset yields: the waveform plus its calibrated per-frame SPL.

    ``frames_offset`` deliberately breaks the label/latent frame alignment, so the guard
    that catches it can be tested; ``silent_db`` reproduces FrameLevelTransform bottoming
    out on a silent frame, which is what the floor clamp exists for.
    """

    def __init__(self, n=8, frames_offset=0, silent_db=None):
        super().__init__(n)
        self._offset = frames_offset
        self._silent = silent_db

    def __getitem__(self, i):
        item = super().__getitem__(i)
        n_frames = self.lengths[i] // HOP + self._offset
        curve = torch.full((1, n_frames), 65.0)
        if self._silent is not None:
            curve[:, : n_frames // 2] = self._silent
        item["frame_intensity_db"] = curve
        return item

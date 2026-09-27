"""SpeechTokenizer codec wrapper implementing the AudioCodec protocol."""
from __future__ import annotations

import yaml

import torch
import torch.nn as nn

from vic.core import FrameGrid
from vic.data.audio_batch import AudioBatch


class SpeechTokenizerCodec(nn.Module):
    """Frozen SpeechTokenizer wrapper satisfying the AudioCodec protocol.

    All codec parameters are frozen at construction time and never updated.

    Parameters
    ----------
    config_path    : path to the SpeechTokenizer JSON config file.
    ckpt_path      : path to the model checkpoint (.pt / .ckpt).
    n_quantizers   : number of RVQ codebooks to use for quantization.
                     ``None`` (default) uses all codebooks in the checkpoint.

    Notes on the internal API
    -------------------------
    The wrapper calls ``model.encoder``, ``model.quantizer.encode``,
    ``model.quantizer.decode``, and ``model.decoder`` directly to work
    with *continuous* (pre-quantization) latents.  If a newer version of
    speechtokenizer changes these attribute names, adjust the three lines
    marked ``# API`` below.

    The ``hop_size`` (encoder total stride) is measured empirically by
    passing a 1-second dummy signal through the encoder, so it is always
    correct regardless of the checkpoint configuration.
    """

    def __init__(
        self,
        config_path: str,
        ckpt_path: str,
        n_quantizers: int | None = None,
    ):
        super().__init__()
        from speech_tokenizer import SpeechTokenizer  # lazy import

        with open(config_path) as f:
            config = yaml.safe_load(f)
        self.model = SpeechTokenizer.load_from_checkpoint(config, ckpt_path)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        # --- measure hop_size and latent_dim empirically ---
        with torch.no_grad():
            sr = self.model.sample_rate or 16_000
            dummy = torch.zeros(1, 1, sr)
            z = self.model.encoder(dummy)          # API
            _hop_size = sr // z.shape[-1]
            _latent_dim = z.shape[1]

        self._latent_dim = _latent_dim
        self._frame_grid = FrameGrid(
            hop_size=_hop_size,
            # Via the property, so the `or 16_000` fallback applies here too: some
            # checkpoints leave ``model.sample_rate`` unset, and a FrameGrid with
            # sample_rate=None breaks every duration it is asked to compute.
            sample_rate=self.sample_rate,
            causal=False,  # SpeechTokenizer encoder is non-causal (symmetric padding)
        )

        # resolve n_quantizers from model if not provided
        if n_quantizers is None:
            for attr in ("n_codebooks", "n_q", "num_quantizers"):
                if hasattr(self.model, attr):
                    n_quantizers = getattr(self.model, attr)
                    break
            if n_quantizers is None:
                raise AttributeError(
                    "Cannot determine n_quantizers automatically. "
                    "Pass it explicitly to SpeechTokenizerCodec(n_quantizers=...)."
                )
        self._n_quantizers = n_quantizers

    # ------------------------------------------------------------------
    # AudioCodec protocol properties
    # ------------------------------------------------------------------

    @property
    def sample_rate(self) -> int:
        return self.model.sample_rate or 16_000

    @property
    def latent_dim(self) -> int:
        return self._latent_dim

    @property
    def frame_grid(self) -> FrameGrid:
        return self._frame_grid

    # ------------------------------------------------------------------
    # Encode / decode
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode(self, batch: AudioBatch) -> AudioBatch:
        """Encode an audio AudioBatch to continuous latent frames.

        Parameters
        ----------
        batch : AudioBatch with ``data`` of shape (B, 1, T_samples).

        Returns
        -------
        AudioBatch with ``data`` of shape (B, D, T_frames), where
        D = ``self.latent_dim`` and T_frames = T_samples // hop_size.
        """
        z = self.model.encoder(batch.BCT)                   # API  (B, D, T_frames)
        frame_lengths = batch.lengths // self._frame_grid.hop_size
        return batch.as_frames(z, frame_lengths)

    @torch.no_grad()
    def decode(self, batch: AudioBatch) -> AudioBatch:
        """Quantize continuous latents and decode to a waveform.

        Parameters
        ----------
        batch : AudioBatch with ``data`` of shape (B, D, T_frames).

        Returns
        -------
        AudioBatch with ``data`` of shape (B, 1, T_samples).
        """
        z = batch.BCT  # (B, D, T_frames)
        codes = self.model.quantizer.encode(z, n_q=self._n_quantizers)  # API
        z_q   = self.model.quantizer.decode(codes)                       # API
        wav   = self.model.decoder(z_q)                                  # API  (B, 1, T)

        sample_lengths = (batch.lengths * self._frame_grid.hop_size).clamp(
            max=wav.shape[-1]
        )
        return AudioBatch(
            data=wav,
            lengths=sample_lengths,
            sample_rate=self.sample_rate,
        )

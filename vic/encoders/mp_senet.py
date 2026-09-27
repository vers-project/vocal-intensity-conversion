"""MP-SENet speech enhancement wrapper, compatible with AudioBatch."""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import torch
import torch.nn as nn

from vic.data.audio_batch import AudioBatch


class MPSENetDenoiser(nn.Module):
    """Pretrained MP-SENet generator used as a training-time speech denoiser.

    Accepts AudioBatch (B, 1, T) at 16 kHz and returns a denoised AudioBatch
    of the same shape.  Intended as a waveform augmentation applied before
    feature extraction — strips background noise while preserving vocal effort
    cues.  The model parameters are frozen; no gradients flow through it.

    Parameters
    ----------
    ckpt_path : path to the MP-SENet generator checkpoint file (e.g. ``g_best_vb``).
                ``config.json`` must be in the same directory.
    repo_path : path to the cloned MP-SENet repository root (the directory
                that contains ``models/``, ``env.py``, and ``dataset.py``).
    """
    
    SAMPLE_RATE = 16_000

    def __init__(self, ckpt_path: str, repo_path: str):
        super().__init__()

        repo = str(Path(repo_path).resolve())
        if repo not in sys.path:
            sys.path.insert(0, repo)

        # pesq is only used for evaluation metrics in model.py, not in MPNet.forward.
        # Stub it out so the import succeeds even when pesq is not installed.
        if "pesq" not in sys.modules:
            try:
                import pesq  # noqa: F401
            except ImportError:
                _stub = types.ModuleType("pesq")
                _stub.pesq = None  # type: ignore[attr-defined]
                sys.modules["pesq"] = _stub

        from env import AttrDict  # type: ignore[import]
        from models.model import MPNet  # type: ignore[import]

        ckpt_path = Path(ckpt_path)
        with open(ckpt_path.parent / "config.json") as f:
            h = AttrDict(json.load(f))
        self._h = h

        model = MPNet(h)
        state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        model.load_state_dict(state["generator"])
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        self.model = model

    # ------------------------------------------------------------------
    # STFT helpers (inlined from MP-SENet dataset.py to avoid top-level
    # module imports before sys.path is set up).
    # ------------------------------------------------------------------

    def _stft(self, wav: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Magnitude + phase STFT with power-law compression. wav: (B, T)."""
        h = self._h
        win = torch.hann_window(h.win_size, device=wav.device)
        spec = torch.stft(
            wav, h.n_fft, hop_length=h.hop_size, win_length=h.win_size,
            window=win, center=True, pad_mode="reflect",
            normalized=False, return_complex=True,
        )
        spec = torch.view_as_real(spec)                            # (B, F, T_f, 2)
        mag = torch.sqrt(spec.pow(2).sum(-1) + 1e-9)              # (B, F, T_f)
        pha = torch.atan2(spec[..., 1] + 1e-10, spec[..., 0] + 1e-5)
        mag = mag.pow(h.compress_factor)
        return mag, pha

    def _istft(self, mag: torch.Tensor, pha: torch.Tensor) -> torch.Tensor:
        """Magnitude decompression + ISTFT. Returns (B, T) waveform."""
        h = self._h
        mag = mag.pow(1.0 / h.compress_factor)
        com = torch.complex(mag * torch.cos(pha), mag * torch.sin(pha))
        win = torch.hann_window(h.win_size, device=com.device)
        return torch.istft(
            com, h.n_fft, hop_length=h.hop_size, win_length=h.win_size,
            window=win, center=True,
        )

    def forward(self, batch: AudioBatch) -> AudioBatch:
        """Denoise a batch of waveforms.

        Parameters
        ----------
        batch : AudioBatch (B, 1, T) at 16 kHz.

        Returns
        -------
        AudioBatch (B, 1, T) with background noise suppressed.
        """
        if batch.sample_rate != self.SAMPLE_RATE:
            raise ValueError(
                f"MPSENetDenoiser expects {self.SAMPLE_RATE} Hz, "
                f"got {batch.sample_rate} Hz."
            )
        wav = batch.data.squeeze(1)  # (B, T)
        T_orig = wav.shape[-1]

        # Per-utterance RMS normalisation — mirrors MP-SENet's inference.py.
        signal_power = wav.pow(2).sum(-1, keepdim=True).clamp(min=1e-10)  # (B, 1)
        norm_factor = torch.sqrt(T_orig / signal_power)
        wav = wav * norm_factor

        mag, pha = self._stft(wav)
        amp_out, pha_out, _ = self.model(mag, pha)
        wav_out = self._istft(amp_out, pha_out)[..., :T_orig]

        wav_out = (wav_out / norm_factor).unsqueeze(1)  # (B, 1, T_orig)
        return AudioBatch(
            data=wav_out,
            lengths=batch.lengths.clamp(max=T_orig),
            sample_rate=batch.sample_rate,
        )

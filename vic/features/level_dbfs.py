"""Uncalibrated RMS level in dBFS, aligned to a codec FrameGrid.

These functions compute a digital-domain loudness proxy used in speech
processing.  Values are in dBFS (decibels relative to full scale) and carry
no physical calibration.  For calibrated dB SPL labels see ``vic.features.spl``.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from vic.core import FrameGrid


def extract_sequence_intensity_db(wav: Tensor) -> Tensor:
    """RMS energy of the full signal in dBFS.

    Parameters
    ----------
    wav : (C, T) audio tensor, any number of channels.

    Returns
    -------
    Scalar tensor in dB.
    """
    rms = wav.pow(2).mean().sqrt()
    return 20.0 * torch.log10(rms + 1e-8)


def extract_frame_intensity_db(
    wav: Tensor,
    grid: FrameGrid,
    window_size: int | None = None,
) -> Tensor:
    """Per-frame RMS intensity aligned with a codec FrameGrid.

    Analysis windows are centred at each frame's nominal centre
    (``grid.frame_center(i)``), so the i-th output value directly
    corresponds to the i-th latent frame produced by the codec.

    Non-causal encoder (default)
    ----------------------------
    Window for frame i: wav[i·hop - win//2  :  i·hop + win//2]
    The signal is symmetrically padded on both sides to handle boundary frames.

    Causal encoder
    --------------
    Window for frame i: wav[i·hop  :  i·hop + win]
    Only left-padded (past context only).

    Parameters
    ----------
    wav         : (C, T) audio tensor.  Mixed to mono internally.
    grid        : FrameGrid of the target codec.
    window_size : analysis window in samples.  Defaults to grid.hop_size
                  (one frame, no overlap).  Longer windows smooth the estimate.

    Returns
    -------
    (T_frames,) tensor of dB values, one per codec frame.
    """
    wav_mono = wav.mean(0)  # (T,)
    win = window_size if window_size is not None else grid.hop_size
    half = win // 2

    if grid.causal:
        # Causal: frame i analyses wav[i*hop : i*hop + win]
        pad_left, pad_right = 0, win - 1
    else:
        # Non-causal: frame i analyses wav centered at i*hop
        pad_left, pad_right = half, half

    wav_padded = F.pad(wav_mono, (pad_left, pad_right))

    # unfold(dim, size, step): extract one window per stride step.
    # Truncate to the canonical frame count so the output always matches
    # grid.n_frames(T), regardless of how the padding interacts with unfold.
    n_frames = grid.n_frames(wav_mono.shape[0])
    frames = wav_padded.unfold(0, win, grid.hop_size)[:n_frames]  # (T_frames, win)
    rms = frames.pow(2).mean(-1).sqrt()
    return 20.0 * torch.log10(rms + 1e-8)

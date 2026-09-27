"""F0 extraction aligned to a codec FrameGrid.

Strategy
--------
Run a pitch tracker at a fine temporal resolution (default 1 ms), then
aggregate voiced F0 values within each codec frame's time span.  This
avoids resampling artefacts and handles unvoiced frames cleanly.

Requires: ``pip install librosa``
"""
from __future__ import annotations

import torch
from torch import Tensor

from vic.core import FrameGrid


def extract_f0_aligned(
    wav: Tensor,
    grid: FrameGrid,
    fine_hop_ms: float = 1.0,
    fmin: float = 50.0,
    fmax: float = 600.0,
) -> Tensor:
    """Per-frame F0 in Hz, aligned with a codec FrameGrid.

    Each output frame i aggregates the voiced F0 values that fall inside
    the half-open interval

        [frame_center(i) - hop//2,  frame_center(i) + hop//2)

    Unvoiced frames (no voiced fine-hop inside the window) are set to 0.

    Parameters
    ----------
    wav         : (C, T) audio tensor.  Mixed to mono internally.
    grid        : FrameGrid of the target codec.
    fine_hop_ms : temporal resolution of the pitch tracker in milliseconds.
                  Should be much smaller than grid.hop_duration_s.
    fmin / fmax : pitch range passed to the pyin estimator (Hz).

    Returns
    -------
    (T_frames,) tensor of F0 values in Hz (0 = unvoiced).
    """
    try:
        import librosa
        import numpy as np
    except ImportError as e:
        raise ImportError(
            "librosa is required for F0 extraction.  "
            "Install it with:  pip install librosa"
        ) from e

    wav_np = wav.mean(0).numpy()  # (T,)
    fine_hop = max(1, int(grid.sample_rate * fine_hop_ms / 1000))

    f0_fine, _, _ = librosa.pyin(
        wav_np,
        fmin=fmin,
        fmax=fmax,
        sr=grid.sample_rate,
        hop_length=fine_hop,
        fill_na=0.0,
    )
    f0_fine = torch.from_numpy(f0_fine.astype("float32"))  # (T_fine,)

    T_frames = grid.n_frames(wav.shape[-1])
    f0_frames = torch.zeros(T_frames, dtype=torch.float32)

    half_hop = grid.hop_size // 2
    for i in range(T_frames):
        center = grid.frame_center(i)
        lo_sample = max(0, center - half_hop)
        hi_sample = min(wav.shape[-1], center + half_hop)
        # Convert sample boundaries to fine-hop indices
        lo = lo_sample // fine_hop
        hi = hi_sample // fine_hop
        segment = f0_fine[lo:hi]
        voiced = segment[segment > 0]
        if len(voiced) > 0:
            f0_frames[i] = voiced.mean()

    return f0_frames

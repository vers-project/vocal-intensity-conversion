"""Per-frame intensity targets for a bounded-receptive-field converter.

Why a curve rather than a scalar
--------------------------------
The utterance-level formulation hands C_θ a single τ_tgt and expects it to infer
``Δ = τ_tgt − τ_src`` on its own.  τ_src is ``leq_aggregate`` over the whole chunk — a
chunk-global quantity.  A Transformer whose context covers the chunk can measure it; a
convolutional stack with a 1 s receptive field cannot.  It measures the level of *its own
window*, which for a pause is the pause level, so it infers the wrong Δ.

Concretely: a 2 s chunk of 0.9 s pause (≈34 dB) followed by 1.1 s of vowels (≈68 dB)
aggregates to τ_src ≈ 65.4 dB.  At τ_tgt = 80 the honest instruction is +14.6 dB at *both*
windows, but both receive the scalar ``80`` — read as "+46" at the pause and "+12" at the
vowels.  Across a corpus the same local content is paired with τ_tgt spanning the whole
range with nothing to distinguish the cases, so the conditional map is not a function and
the least-bad response is to under-react.

The fix is to give every frame a target it can actually reach from what it can see: the
source's own smoothed local Leq, displaced by Δ.

Two design points worth stating
-------------------------------
*Smoothing at the receptive field, not at some independent window.*  Displacing by a
constant Δ asserts that loud speech is quiet speech translated, which is false in detail —
vocal effort also expands local dynamic range.  Smoothing at the RF confines that assertion
to the scale the model can see and leaves everything finer to D_ψ's realism term, which is
what makes the approximation harmless rather than wrong.

*Silence is not displaced, and the law says so rather than a knee.*  Model the observed
pressure as speech plus an independent noise floor, so mean-square pressures add
(``P² = S² + N²``), and scale only the speech: silence stays where it is, loud frames move
by Δ, and the transition between them is smooth and monotone with no free parameters beyond
N.  A hard threshold would put a step in the target curve that no real recording contains.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from vic.features.spl import db_to_power, power_to_db


def moving_leq(
    frame_db: Tensor,
    window_frames: int,
    padding_mask: Tensor | None = None,
) -> Tensor:
    """Centred moving Leq of a per-frame dB curve.

    ``L_eq`` in the IEC sense, over a sliding window: the level of the **RMS pressure**
    across the window.  ``P_φ`` reports ``L_t = 20·log10(p_rms,t / P₀)``, so ``10^(L_t/10)``
    is the normalised mean-square pressure ``(p_rms,t / P₀)²``; averaging those and taking
    ``10·log10`` returns ``20·log10(RMS(p) / P₀)`` over the window.  The dB→linear→dB trip
    is how the RMS is recovered, not a separate definition — the same aggregation
    :func:`vic.features.spl.leq_aggregate` performs over a whole sequence, applied at every
    frame instead.

    Verified against a direct ``20·log10(RMS/P₀)`` on a non-stationary known-answer signal:
    exact to 0.0000 dB when ``FrameLevelTransform.window_length == grid.hop_size``.  With
    the usual ``window_length: 400`` against a 320-sample hop the analysis windows overlap
    1.25×, so the *labels* no longer tile the span uniformly and the agreement loosens to
    0.014 dB in the mean and 0.20 dB worst-case per window.  That residual belongs to the
    label definition — ``leq_aggregate`` carries the identical bias — not to this function.

    Parameters
    ----------
    frame_db      : (B, T) per-frame dB SPL, e.g. raw P_φ output.
    window_frames : window length in frames.  Must be odd, so the window is centred and the
                    output aligns with the input frame-for-frame.  A converter receptive
                    field is odd by construction (``1 + Σ(k−1)·d`` with odd ``k``).
    padding_mask  : (B, T) bool, True = valid frame.  ``None`` = all valid.

    Returns
    -------
    (B, T) smoothed dB curve.

    Note
    ----
    Normalisation is by the number of frames actually averaged, not by ``window_frames``:
    both numerator and denominator are pooled, so a window overhanging the start of a chunk
    averages the frames that exist rather than mixing in zeros.  Averaging in zeros is what
    produces the ≤3 dB edge droop documented for ``FrameLevelTransform`` (``spl.py``), and
    it would be considerably worse here — a window is 53 frames, not 400 samples.

    The same normalisation makes a window wider than the sequence well defined: every frame
    then reads the masked Leq of the whole sequence, which is what ``leq_aggregate`` would
    return.
    """
    if frame_db.dim() != 2:
        raise ValueError(f"frame_db must be (B, T), got shape {tuple(frame_db.shape)}")
    if window_frames < 1:
        raise ValueError(f"window_frames must be >= 1, got {window_frames}")
    if window_frames % 2 == 0:
        raise ValueError(
            f"window_frames must be odd for a centred window, got {window_frames}"
        )
    if window_frames == 1:
        return frame_db

    if padding_mask is None:
        weight = torch.ones_like(frame_db)
    else:
        weight = padding_mask.to(frame_db.dtype)
        # Zero the *dB* value rather than the power.  A padded frame holds whatever P_φ
        # happened to emit there, and ``db_to_power`` of a large dB value is ``inf``, which
        # multiplied by a zero weight gives nan — not nothing.
        frame_db = torch.where(padding_mask, frame_db, torch.zeros_like(frame_db))

    power = db_to_power(frame_db) * weight
    pad = window_frames // 2
    num = F.avg_pool1d(power.unsqueeze(1), window_frames, stride=1, padding=pad).squeeze(1)
    den = F.avg_pool1d(weight.unsqueeze(1), window_frames, stride=1, padding=pad).squeeze(1)
    return power_to_db(num / den.clamp(min=1e-12))


def shift_curve(
    curve_db: Tensor,
    delta_db: Tensor,
    floor_db: float | None,
) -> Tensor:
    """Displace an intensity curve by Δ dB, leaving the noise floor in place.

    Models the observed pressure as speech plus an independent noise floor, so mean-square
    pressures add, and scales only the speech RMS by ``10^(Δ/20)``::

        L_tgt = 10·log10( max(10^(L/10) − 10^(N/10), 0) · 10^(Δ/10) + 10^(N/10) )

    Frames far above the floor move by Δ; frames at the floor do not move at all; in
    between the transition is smooth and monotone.  ``floor_db=None`` gives the plain
    ``L + Δ``, which is the ablation that isolates what the floor is worth.

    Parameters
    ----------
    curve_db : (B, T) source dB curve.
    delta_db : (B,) or (B, 1) or scalar displacement in dB.
    floor_db : noise floor in the same units as ``curve_db``, or ``None``.

    Returns
    -------
    (B, T)

    Note
    ----
    At ``delta_db == 0`` this is the identity up to float round-off, not bit-exactly —
    ``(p − n) + n`` need not return ``p``.  Callers that need the exact source curve (the
    identity anchor, the cycle return trip) should pass the curve itself rather than route
    it through here with a zero displacement.
    """
    delta = delta_db if isinstance(delta_db, Tensor) else torch.as_tensor(delta_db)
    if delta.dim() == 1:
        delta = delta.unsqueeze(-1)                       # (B,) -> (B, 1)
    delta = delta.to(curve_db.dtype).to(curve_db.device)

    if floor_db is None:
        return curve_db + delta

    gain = db_to_power(delta)
    floor_power = float(10.0 ** (floor_db / 10.0))
    speech = (db_to_power(curve_db) - floor_power).clamp(min=0.0)
    return power_to_db(speech * gain + floor_power)

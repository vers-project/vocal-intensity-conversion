"""Reusable loss terms for converter training.

All functions are stateless and take the converter as a plain callable
``(AudioBatch, Tensor) -> AudioBatch``, so they work with ``FrameConverter``,
``ContextualConverter``, or anything with the same signature.

Content preservation without a proximity penalty
------------------------------------------------
The original ``loss_loc`` (MSE between converted and source latents) is a poor
content constraint: the NAC latent entangles content and intensity, so a single
scalar penalty on total movement cannot distinguish "changed the phonemes" from
"changed the vocal effort".  It fails in both directions at once — too weak to
prevent content drift on a large conversion, too strong to permit the movement a
large conversion requires — and by pushing the latent delta toward zero it also
pushes it into the codec's RVQ quantisation dead zone, where the converted
latent decodes to *identical* audio.

The two replacements here constrain content directly instead:

``cycle_consistency_loss``
    ``C(C(z, τ_tgt), τ_src) ≈ z``.  If the converter changed the words or the
    speaker, the return trip cannot recover z — the information is gone.  This
    is the mechanism CycleGAN/StarGAN rely on for content preservation without
    paired data, and it penalises *nothing* about legitimate intensity movement.
    Known limitation: cycle consistency can be partly satisfied by hiding
    information to invert later (Chu et al., "CycleGAN, a Master of
    Steganography"), so it is a strong constraint rather than a guarantee.

``identity_anchor_loss``
    ``C(z, τ_src) = z`` exactly.  Free, fully-supervised, paired supervision:
    when the target equals the source intensity the correct output is known.
    Constrains only that slice, so it complements rather than replaces the cycle.

``budgeted_locality_loss``
    A repaired ``loss_loc`` for anyone who wants to keep a proximity term:
    the allowed movement is scaled by ``|τ_tgt − τ_src|``, so the budget is zero
    at ``τ_tgt = τ_src`` and grows with the size of the change requested.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from vic.data.audio_batch import AudioBatch


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------


def masked_mse(a: Tensor, b: Tensor, mask: Tensor | None = None) -> Tensor:
    """Mean squared error over valid frames only.

    Parameters
    ----------
    a, b : (B, C, T) tensors.
    mask : (B, T) bool, True = valid frame.  ``None`` = all frames valid.

    Note
    ----
    This normalises by the number of *valid* elements, unlike
    ``mse_loss(a * mask, b * mask)`` as used by the old ``loss_loc``, which
    divides by ``B·C·T_max`` and therefore silently shrinks the loss in
    proportion to how much padding a batch happens to contain.
    """
    diff2 = (a - b).pow(2)
    if mask is None:
        return diff2.mean()
    m = mask.unsqueeze(1).to(diff2.dtype)                       # (B, 1, T)
    n_valid = (m.sum() * a.shape[1]).clamp(min=1.0)
    return (diff2 * m).sum() / n_valid


def lsgan_loss(score: Tensor, target: float) -> Tensor:
    """Least-squares GAN loss: ``MSE(score, target)``.

    ``score`` is a raw (unsigmoided) critic output of any shape; ``target`` is
    the scalar convention value (+1 for real, −1 for fake in this codebase).
    """
    return F.mse_loss(score, torch.full_like(score, target))


def masked_lsgan_loss(
    score: Tensor, target: float, mask: Tensor | None = None
) -> Tensor:
    """LSGAN loss over the entries ``mask`` selects.

    Parameters
    ----------
    score  : (B,) sequence-level scores, or (B, T) per-frame scores.
    target : +1 for real, −1 for fake.
    mask   : same shape as ``score``, True = counted.  ``None`` = all entries.

    Why this is not ``lsgan_loss`` on a masked tensor
    -------------------------------------------------
    ``lsgan_loss`` divides by the full element count, so on a (B, T) score it folds padding
    frames — whose critic output is meaningless — into the average, and the loss then
    shrinks with however much padding a batch happens to carry.  Same failure ``masked_mse``
    was written to avoid.

    What the per-frame form adds
    ----------------------------
    For a fully-valid sequence::

        mean_t (s_t − target)²  =  (s̄ − target)²  +  Var_t(s_t)

    The first term is exactly what the pooled critic optimises.  The per-frame form is
    therefore the pooled objective *plus* a penalty on within-sequence inconsistency — a
    chunk that is convincing on average while one window is wrong is no longer free.  That
    is the entire difference between a pooled and a patch critic here; moving the pooling
    inside the model changes nothing, because both the head and the projection are affine
    (see ``ConditionalDiscriminator.forward``).
    """
    diff2 = (score - target).pow(2)
    if mask is None:
        return diff2.mean()
    m = mask.to(diff2.dtype)
    return (diff2 * m).sum() / m.sum().clamp(min=1.0)


# ---------------------------------------------------------------------------
# Content-preservation losses
# ---------------------------------------------------------------------------


def cycle_consistency_loss(
    converter,
    z_fake: AudioBatch,
    tau_src: Tensor,
    z_real: AudioBatch,
) -> tuple[Tensor, AudioBatch]:
    """Round-trip loss ``C(C(z, τ_tgt), τ_src) ≈ z``.

    Parameters
    ----------
    converter : callable ``(AudioBatch, Tensor) -> AudioBatch``.
    z_fake    : output of the forward conversion ``C(z_real, τ_tgt)``.  Pass it
                **with its graph intact** — gradients should flow through both
                converter applications, which is what makes a shortcut delta
                costly (it has to be invertible).
    tau_src   : (B,) source intensity in the units the converter expects (raw dBSPL).
    z_real    : the original latents, used as the reconstruction target and for
                the padding mask.

    Returns
    -------
    (loss, z_cycle) — ``z_cycle`` is returned so callers can log or inspect it
    without a second forward pass.
    """
    z_cycle = converter(z_fake, tau_src)
    loss = masked_mse(z_cycle.data, z_real.data, z_real.padding_mask)
    return loss, z_cycle


def identity_anchor_loss(
    converter,
    z_real: AudioBatch,
    tau_src: Tensor,
) -> Tensor:
    """Identity constraint ``C(z, τ_src) = z``.

    Costs one extra converter forward pass.  Unlike a locality penalty this is
    an exact, paired target rather than a bias toward "don't move", so it does
    not fight the conversion objective at other targets.
    """
    z_id = converter(z_real, tau_src)
    return masked_mse(z_id.data, z_real.data, z_real.padding_mask)


def budgeted_locality_loss(
    z_fake: AudioBatch,
    z_real: AudioBatch,
    tau_tgt: Tensor,
    tau_src: Tensor,
    scale_db: float = 10.0,
) -> Tensor:
    """Locality penalty whose budget grows with the requested intensity change.

    Penalises the *excess* movement beyond a budget proportional to
    ``|τ_tgt − τ_src| / scale_db``, so a 0 dB conversion must not move at all
    while a 25 dB conversion is allowed substantial movement.  Optional — the
    cycle + identity pair is usually enough — but strictly better than a
    constant-weight ``loss_loc`` if a proximity term is wanted.

    Parameters
    ----------
    scale_db : dB of requested change that "buys" one unit of squared latent
               movement.  Larger = stricter.
    """
    per_sample = (z_fake.data - z_real.data).pow(2)              # (B, C, T)
    m = z_real.padding_mask.unsqueeze(1).to(per_sample.dtype)    # (B, 1, T)
    n_valid = (m.sum(dim=(1, 2)) * z_real.data.shape[1]).clamp(min=1.0)
    moved = (per_sample * m).sum(dim=(1, 2)) / n_valid           # (B,)

    budget = (tau_tgt - tau_src).abs() / scale_db                # (B,)
    return torch.relu(moved - budget).mean()

"""Utility helpers for training modules.

``configure_matmul_precision`` and ``apply_spectral_norm`` are re-exported from
``audio_utils.training.utils``; what lives here is the intensity-specific
machinery (label scaling, τ sampling and architecture reporting).
"""
from __future__ import annotations

import warnings
from pathlib import Path

import torch
import torch.nn as nn
from audio_utils.training.utils import apply_spectral_norm, configure_matmul_precision
from lightning.pytorch.utilities import rank_zero_info
from torch import Tensor

__all__ = [
    "apply_spectral_norm",
    "build_label_scaler",
    "configure_matmul_precision",
    "describe_model",
    "LABEL_SCALER_CKPT_KEY",
    "LabelScaler",
    "load_label_scaler",
    "IntensityDrawer",
    "VicinalSampler",
]


def describe_model(
    name: str,
    model: nn.Module,
    frame_rate_hz: float | None = None,
) -> None:
    """Log a model's parameter count and, for conv stacks, its receptive field.

    The receptive field is the number a conv config is actually tuned against,
    and it is only meaningful in milliseconds — which requires the frame rate,
    known to the training script and not to the model.  Reporting it at build
    time means a config that says ``dilations: [1, 2, 4, 1, 2, 4]`` never has to
    be mentally converted to "580 ms" by whoever reads the run log.

    Models without a ``receptive_field`` attribute (Transformer, MLP) report
    only their parameter count.
    """
    n_params = sum(p.numel() for p in model.parameters())
    line = f"{name}: {n_params / 1e6:.2f} M params"

    rf = getattr(model, "receptive_field", None)
    if rf is not None:
        line += f" | receptive field {rf} frames"
        if frame_rate_hz:
            line += f" = {1000.0 * rf / frame_rate_hz:.0f} ms"

    rank_zero_info(line)


class LabelScaler:
    """Affine normalisation for intensity labels (and predicted values).

    Stores mean and std computed on the training set so the same statistics
    are used consistently during training, validation, and inference.

    Parameters
    ----------
    mean : dataset mean of intensity_db values.
    std  : dataset std  of intensity_db values.
    """

    def __init__(self, mean: float, std: float):
        self.mean = mean
        self.std = max(std, 1e-8)

    def normalise(self, x: Tensor) -> Tensor:
        return (x - self.mean) / self.std

    def denormalise(self, x: Tensor) -> Tensor:
        return x * self.std + self.mean

    @classmethod
    def from_tensor(cls, values: Tensor) -> LabelScaler:
        return cls(float(values.mean()), float(values.std()))

    def state_dict(self) -> dict:
        return {"mean": self.mean, "std": self.std}

    @classmethod
    def from_state_dict(cls, d: dict) -> LabelScaler:
        return cls(d["mean"], d["std"])


LABEL_SCALER_CKPT_KEY = "label_scaler"


def build_label_scaler(config: dict) -> LabelScaler:
    """Fit the label scaler a converter run conditions on, from its config.

    Two sources, in order of preference:

    ``config["data_labels"]``
        A labelled corpus.  The scaler is fitted on the ``intensity_db`` column
        of its training split — the same statistics ``train_predictor.py`` uses,
        so the converter normalises τ on the scale P_φ was trained to emit.
    ``config["training"]["intensity_range_db"]``
        Fallback when no labelled corpus is configured.  The range is treated as
        ±2σ about its midpoint, which is an *approximation of a distribution by
        its support* — good enough to keep τ O(1), but not the same numbers.

    Whichever is used, the result is a **fitted** quantity that is not recoverable
    from the architecture.  Prefer :func:`load_label_scaler` when a trained
    checkpoint is at hand.
    """
    if "data_labels" in config:
        import pandas as pd

        labels = pd.read_csv(config["data_labels"]["metadata_csv"])
        train_labels = labels.loc[labels["split"] == "train", "intensity_db"]
        # One NaN makes the fitted mean and std NaN, which normalises every τ to
        # NaN for the whole run without raising anywhere.  Cheaper to refuse.
        if train_labels.isna().any():
            raise ValueError(
                f"{int(train_labels.isna().sum())} of {len(train_labels)} training "
                f"labels are null in {config['data_labels']['metadata_csv']}; the "
                "fitted scaler would be NaN and silently poison τ.  Re-run "
                "compute_labels.py over that table."
            )
        return LabelScaler.from_tensor(
            torch.tensor(train_labels.to_numpy(), dtype=torch.float32)
        )

    lo, hi = config["training"]["intensity_range_db"]
    return LabelScaler(mean=(lo + hi) / 2.0, std=(hi - lo) / 4.0)


def load_label_scaler(ckpt_path: str | Path, config: dict) -> LabelScaler:
    """Recover the label scaler a checkpoint was *actually* trained with.

    The scaler is fitted, not architectural, and rebuilding it from a config is
    only correct while that config still says what it said during training.  An
    edited ``intensity_range_db``, or a config copied between corpora, silently
    yields a converter conditioned on the wrong τ — plausible audio, meaningless
    numbers, nothing raised.

    So the checkpoint wins when it carries a scaler (written by
    ``ConverterCGANv2Module.on_save_checkpoint``), and the config is used only to
    cross-check it.  Checkpoints written before that hook existed carry nothing,
    and fall back to :func:`build_label_scaler` with a warning — which is exactly
    the situation the warning is there to make visible.
    """
    from vic.checkpoints import load_extra

    stored = load_extra(ckpt_path, LABEL_SCALER_CKPT_KEY)
    from_config = build_label_scaler(config)

    if stored is None:
        warnings.warn(
            f"{ckpt_path} stores no '{LABEL_SCALER_CKPT_KEY}', so the label scaler "
            f"was rebuilt from the config as mean={from_config.mean:.4f} "
            f"std={from_config.std:.4f}. That is correct only if the config's "
            "data_labels / training.intensity_range_db still hold the values this "
            "run was trained with.",
            RuntimeWarning,
            stacklevel=2,
        )
        return from_config

    scaler = LabelScaler.from_state_dict(stored)
    if (
        abs(scaler.mean - from_config.mean) > 1e-3
        or abs(scaler.std - from_config.std) > 1e-3
    ):
        warnings.warn(
            f"Label scaler in {ckpt_path} (mean={scaler.mean:.4f} "
            f"std={scaler.std:.4f}) disagrees with the one this config rebuilds "
            f"(mean={from_config.mean:.4f} std={from_config.std:.4f}). Using the "
            "checkpoint's, which is what the weights were trained against — but "
            "the config has drifted from the run and any τ range read out of it "
            "describes a different experiment.",
            RuntimeWarning,
            stacklevel=2,
        )
    return scaler


class IntensityDrawer:
    """Sample a target intensity τ from a curriculum distribution.

    During training the converter sees a range of target intensities drawn
    uniformly from [min_db, max_db].  The curriculum can optionally start
    narrow (around the source intensity) and widen over time, but the
    simplest and most common setting is full-range uniform sampling.

    Parameters
    ----------
    min_db  : lower bound of the sampling range (in dBFS or normalised units).
    max_db  : upper bound of the sampling range.
    """

    def __init__(self, min_db: float, max_db: float):
        self.min_db = min_db
        self.max_db = max_db

    def draw(self, batch_size: int, device: torch.device) -> Tensor:
        """Return (B,) uniform samples in [min_db, max_db]."""
        return torch.empty(batch_size, device=device).uniform_(self.min_db, self.max_db)


class VicinalSampler:
    """Draw the τ labels a *conditional* discriminator needs, given τ_src.

    A conditional discriminator D(z, τ) only learns to use τ if it is shown
    latents paired with *wrong* intensities and told they are fake.  Without
    such negatives, every real pair carries the correct τ and every fake pair
    carries τ_tgt, so D can score perfectly while ignoring τ entirely — which
    reduces it to an unconditional critic with extra parameters.  Checking for
    that degeneracy is what ``train/d_cond_gap`` is for.

    A second problem is specific to *continuous* conditions: no two real
    utterances share an intensity value, so "the real distribution at τ = 73.2"
    contains exactly one sample.  The fix is the vicinal treatment of CcGAN
    (Ding et al., ICLR 2021): treat real latents whose τ_src lies within a small
    window of τ as positives for τ.  Here that is implemented by jittering τ_src
    within ``vicinity_db`` for the positive pairs, which is the hard-vicinal
    variant.

    Parameters
    ----------
    min_db, max_db : bounds of the intensity range (raw dBSPL).
    vicinity_db    : half-width of the positive window.  Real pairs are labelled
                     ``τ_src + U(−vicinity_db, +vicinity_db)``.  Should be small
                     relative to the range (1–2 dB for a 30 dB span).
    min_offset_db  : minimum ``|τ_neg − τ_src|`` for mismatched negatives.  Must
                     be comfortably larger than ``vicinity_db`` or positives and
                     negatives overlap and the two losses fight each other.
    """

    def __init__(
        self,
        min_db: float,
        max_db: float,
        vicinity_db: float = 1.0,
        min_offset_db: float = 6.0,
    ):
        if min_offset_db <= vicinity_db:
            raise ValueError(
                f"min_offset_db ({min_offset_db}) must exceed vicinity_db "
                f"({vicinity_db}), otherwise positive and negative conditioning "
                "labels overlap."
            )
        if max_db - min_db < min_offset_db:
            raise ValueError(
                f"intensity range ({min_db}, {max_db}) is narrower than "
                f"min_offset_db ({min_offset_db}); no mismatched label can "
                "satisfy the margin."
            )
        self.min_db = min_db
        self.max_db = max_db
        self.vicinity_db = vicinity_db
        self.min_offset_db = min_offset_db

    def positive(self, tau_src: Tensor) -> Tensor:
        """Vicinal positive labels: τ_src jittered within ±vicinity_db."""
        jitter = torch.empty_like(tau_src).uniform_(-self.vicinity_db, self.vicinity_db)
        return (tau_src + jitter).clamp(self.min_db, self.max_db)

    def negative(self, tau_src: Tensor) -> Tensor:
        """Mismatched negative labels: at least ``min_offset_db`` away from τ_src.

        Vectorised with no rejection loop.  The direction is chosen from the room
        actually available on each side, then the magnitude is drawn within that
        room — so ``|τ_neg − τ_src| ≥ min_offset_db`` holds exactly and no clamp
        can collapse the margin.  (Drawing a signed offset first and reflecting it
        does not work: near a bound the reflection can overshoot the *other*
        bound, and the clamp then lands arbitrarily close to τ_src.)

        Sources near a bound can only be offset inward, so the sign is not
        balanced across the batch by construction; the magnitude distribution is
        uniform within whichever side was chosen.
        """
        room_up = self.max_db - tau_src
        room_down = tau_src - self.min_db
        up_ok = room_up >= self.min_offset_db
        down_ok = room_down >= self.min_offset_db

        # Both sides available → coin flip. Only one → take it. (The constructor
        # guarantees at least one side always has room.)
        coin = torch.rand_like(tau_src) < 0.5
        go_up = torch.where(up_ok & down_ok, coin, up_ok)

        room = torch.where(go_up, room_up, room_down)
        spare = (room - self.min_offset_db).clamp(min=0.0)
        magnitude = self.min_offset_db + torch.rand_like(tau_src) * spare

        tau_neg = torch.where(go_up, tau_src + magnitude, tau_src - magnitude)
        return tau_neg.clamp(self.min_db, self.max_db)

"""Modular conditioning of a discriminator on a scalar intensity τ.

Why conditioning matters here
-----------------------------
An *unconditional* realness discriminator answers "is this a plausible latent?".
That question is separately satisfiable from "does P_φ read τ_tgt off it?", so a
converter can take a genuinely-60-dB latent (which D_ψ happily accepts, because
it really is plausible) and add a small P_φ-fooling perturbation.  Both critics
are satisfied and nothing has been converted.

A *conditional* discriminator is shown the pair ``(z, τ)`` and answers "is this a
plausible latent **at intensity τ**?".  The perturbation that fools P_φ does
nothing to make a 60 dB latent plausible-as-85-dB, so the loophole closes.  The
two objectives also collapse into one adversarial game, which removes the
λ_fake / λ_int balancing problem.

Design
------
Three small pieces, composable:

``PooledFeatureModel``  (protocol)
    Anything that maps an AudioBatch to one fixed-width vector per sample and
    can score that vector unconditionally.  ``TrueFakeDiscriminator`` satisfies
    it; so would any future CNN/conformer critic.

``IntensityConditioner`` (abstract nn.Module)
    Maps ``(pooled_feature, τ) → (B,)``: the *conditional correction* added to
    the unconditional score.  Two implementations below.

``ConditionalDiscriminator``
    Glue: ``D(z, τ) = score(φ(z)) + conditioner(φ(z), τ)``.  It takes any
    ``PooledFeatureModel`` and any ``IntensityConditioner``, so a new backbone or
    a new conditioning scheme can be swapped in without touching the other.

Splitting the score this way is not arbitrary — it is exactly the projection
discriminator decomposition of Miyato & Koyama (ICLR 2018),
``D(x, y) = ψ(φ(x)) + yᵀ V φ(x)``, where ψ is the pre-existing unconditional
head and the conditioner supplies the second term.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol

import torch
import torch.nn as nn
from torch import Tensor

from vic.data.audio_batch import AudioBatch
from vic.models.blocks import build_intensity_embed
from vic.models.discriminator import ConvTrueFakeDiscriminator, TrueFakeDiscriminator


# ---------------------------------------------------------------------------
# Protocol for anything that can be conditioned
# ---------------------------------------------------------------------------


class PooledFeatureModel(Protocol):
    """A sequence model exposing a pooled per-sample feature and its own score.

    Used for typing only (not ``runtime_checkable``): the contract is three
    members, which ``TrueFakeDiscriminator`` implements.

        feature_dim : width of the pooled feature.
        frame_features(batch) -> (B, T, feature_dim)
        features(batch)       -> (B, feature_dim)
        score(feat)           -> (B,)  or (B, T) for per-frame features
    """

    feature_dim: int

    def frame_features(self, batch: AudioBatch) -> Tensor: ...

    def features(self, batch: AudioBatch) -> Tensor: ...

    def score(self, feat: Tensor) -> Tensor: ...


# ---------------------------------------------------------------------------
# Conditioners
# ---------------------------------------------------------------------------


class IntensityConditioner(nn.Module, ABC):
    """Conditional correction term added to an unconditional discriminator score.

    Subclasses implement ``forward(feat, tau) -> (B,)``.  Returning only the
    *correction* (not the whole score) keeps the unconditional head in one place
    and makes the projection decomposition explicit.
    """

    @abstractmethod
    def forward(self, feat: Tensor, tau: Tensor) -> Tensor:
        """
        feat : (B, feature_dim) pooled sequence feature, or (B, T, feature_dim) per-frame
               features.
        tau  : (B,) conditioning intensity, or (B, T) a target curve, in the *same units*
               the module was configured for (normalised label space — see
               ConditionalDiscriminator).
        returns (B,) or (B, T), matching the input rank.
        """
        raise NotImplementedError


class ProjectionConditioner(IntensityConditioner):
    """Projection conditioning: ``⟨V·e(τ), φ(z)⟩ / √d``.

    The recommended default.  An inner product between the pooled feature and a
    learned embedding of τ, which forces the discriminator to model the *joint*
    structure of (latent, intensity) rather than scoring each separately.  With
    a continuous scalar condition this is also the form CcGAN uses.

    The ``1/√feature_dim`` scale keeps the projection term the same order of
    magnitude as the unconditional score at init, so neither dominates early.

    Parameters
    ----------
    feature_dim     : width of the pooled feature φ(z).
    intensity_embed : embedder with ``embed_dim`` and ``forward((B,)) → (B, embed_dim)``
                      (``ScalarIntensityEmbedding`` or ``SinusoidalIntensityEmbedding``).
    """

    def __init__(self, feature_dim: int, intensity_embed: nn.Module):
        super().__init__()
        self.intensity_embed = intensity_embed
        self.embed_proj = nn.Linear(intensity_embed.embed_dim, feature_dim)
        self.scale = feature_dim ** -0.5

    def forward(self, feat: Tensor, tau: Tensor) -> Tensor:
        v = self.embed_proj(self.intensity_embed(tau))        # (B, feature_dim)
        return (v * feat).sum(dim=-1) * self.scale            # (B,)


class ConcatConditioner(IntensityConditioner):
    """Concatenate-then-MLP conditioning — a baseline for ablation.

    More expressive per-parameter than a projection, but empirically easier for
    the discriminator to *ignore* (the MLP can learn to zero the τ columns),
    which is the failure mode conditional GANs are prone to.  Kept so the
    projection choice can be ablated rather than assumed.

    Parameters
    ----------
    feature_dim     : width of the pooled feature φ(z).
    intensity_embed : embedder with ``embed_dim`` and ``forward((B,)) → (B, embed_dim)``.
    hidden_dim      : MLP width.  Defaults to ``feature_dim``.
    """

    def __init__(
        self,
        feature_dim: int,
        intensity_embed: nn.Module,
        hidden_dim: int | None = None,
    ):
        super().__init__()
        self.intensity_embed = intensity_embed
        h = hidden_dim if hidden_dim is not None else feature_dim
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim + intensity_embed.embed_dim, h),
            nn.SiLU(),
            nn.Linear(h, 1),
        )

    def forward(self, feat: Tensor, tau: Tensor) -> Tensor:
        x = torch.cat([feat, self.intensity_embed(tau)], dim=-1)
        return self.mlp(x).squeeze(-1)                         # (B,)


# ---------------------------------------------------------------------------
# Conditional discriminator
# ---------------------------------------------------------------------------


class ConditionalDiscriminator(nn.Module):
    """Wrap any :class:`PooledFeatureModel` into a τ-conditional critic.

        D(z, τ) = score(φ(z)) + conditioner(φ(z), τ)

    The wrapped model keeps working unconditionally on its own; this class only
    adds the conditional term, so D_ψ's architecture, spectral norm and
    ``attn_type`` settings are all unchanged and inherited.

    Parameters
    ----------
    feature_model : the backbone (e.g. ``TrueFakeDiscriminator``).
    conditioner   : an :class:`IntensityConditioner`.
    label_scaler  : optional ``LabelScaler``.  When given, raw dBSPL τ passed to
                    ``forward`` is normalised before it reaches the conditioner,
                    so callers never have to remember which space τ is in.
                    Pass ``None`` if you already hand it normalised values.
    """

    def __init__(
        self,
        feature_model: PooledFeatureModel,
        conditioner: IntensityConditioner,
        label_scaler=None,
    ):
        super().__init__()
        self.feature_model = feature_model      # type: ignore[assignment]
        self.conditioner = conditioner
        self.label_scaler = label_scaler

    @property
    def receptive_field(self) -> int | None:
        """Backbone receptive field in frames, or ``None`` if unbounded.

        Delegated so that reporting code can ask the critic directly rather than
        reaching into it.  A Transformer backbone has no bounded receptive field
        and returns ``None``.
        """
        return getattr(self.feature_model, "receptive_field", None)

    def forward(self, batch: AudioBatch, tau: Tensor) -> Tensor:
        """
        batch : AudioBatch of codec latents, data (B, D, T).
        tau   : (B,) sequence-level conditioning intensity, or (B, T) a per-frame target
                curve.  Raw dBSPL (normalised internally when a ``label_scaler`` was
                supplied).
        returns (B,) for a scalar condition, (B, T) for a curve.

        The two paths agree exactly on a flat curve.  ``head`` is a ``Linear`` and the
        projection is an inner product, so both are affine in the feature and commute with
        the mean::

            mean_t head(φ_t)      ≡ head(mean_t φ_t)
            mean_t ⟨v, φ_t⟩       ≡ ⟨v, mean_t φ_t⟩        for constant v

        Moving the pooling inside therefore changes nothing on its own.  What changes is
        where the pooling sits relative to the LSGAN loss: ``mean_t (s_t − t)²`` equals
        ``(s̄ − t)² + Var_t(s_t)``, i.e. the per-frame critic is the pooled one plus a
        penalty on within-chunk inconsistency.  That is the whole point of the per-frame
        form — a chunk convincing on average but wrong in one window now costs something.
        """
        if self.label_scaler is not None:
            tau = self.label_scaler.normalise(tau)
        if tau.dim() == 2:
            feat = self.feature_model.frame_features(batch)     # (B, T, d)
        else:
            feat = self.feature_model.features(batch)           # (B, d)
        return self.feature_model.score(feat) + self.conditioner(feat, tau)


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def build_conditioner(
    kind: str,
    feature_dim: int,
    intensity_embed: nn.Module,
    hidden_dim: int | None = None,
) -> IntensityConditioner:
    """Instantiate an :class:`IntensityConditioner` by name.

    kind : "projection" (default, recommended) or "concat".
    """
    if kind == "projection":
        return ProjectionConditioner(
            feature_dim=feature_dim, intensity_embed=intensity_embed
        )
    if kind == "concat":
        return ConcatConditioner(
            feature_dim=feature_dim,
            intensity_embed=intensity_embed,
            hidden_dim=hidden_dim,
        )
    raise ValueError(
        f"Unknown conditioner type: {kind!r} (expected 'projection' or 'concat')"
    )


def build_conditional_discriminator(
    cfg: dict,
    latent_dim: int,
    label_scaler=None,
) -> ConditionalDiscriminator:
    """Build D_ψ(z, τ): an unconditional backbone plus a conditioning term.

    The backbone is a ``PooledDiscriminator`` — Transformer (``type`` absent or
    ``"transformer"``) or convolutional (``type: conv``).  Only the backbone
    varies: the pooling, the head and the whole conditioning path below are
    shared, so a conv-vs-Transformer run differs in the sequence-mixing operator
    and nothing else.

    τ reaches the conditioner in *normalised* label space via ``label_scaler``,
    keeping the conditioning input O(1) whatever dB range is in use.
    """
    dc = cfg["model"]["discriminator"]
    kind = dc.get("type", "transformer")
    if kind == "conv":
        # No attn_type: the L2-attention option exists solely to stop
        # dot-product attention maps collapsing under spectral norm, which is
        # not a failure mode a convolution has.
        if "attn_type" in dc:
            raise ValueError(
                "model.discriminator.attn_type is set on a conv discriminator. "
                "It only applies to attention — remove the key."
            )
        backbone = ConvTrueFakeDiscriminator(latent_dim=latent_dim, conv_cfg=dc)
    elif kind == "transformer":
        backbone = TrueFakeDiscriminator(
            latent_dim=latent_dim,
            d_model=dc["d_model"],
            n_heads=dc["n_heads"],
            n_layers=dc["n_layers"],
            head_dim=dc.get("head_dim"),
            mlp_ratio=dc.get("mlp_ratio", 4),
            dropout=dc.get("dropout", 0.0),
            attn_type=dc.get("attn_type", "dot"),
            attn_window=dc.get("attn_window"),
        )
    else:
        raise ValueError(
            f"Unknown discriminator type: {kind!r} (expected 'conv' or 'transformer')"
        )

    cond_cfg = cfg["model"].get("conditioning", {})
    conditioner = build_conditioner(
        kind=cond_cfg.get("type", "projection"),
        feature_dim=backbone.feature_dim,
        intensity_embed=build_intensity_embed(
            cond_cfg.get("intensity_embed", {"type": "sinusoidal"})
        ),
        hidden_dim=cond_cfg.get("hidden_dim"),
    )

    return ConditionalDiscriminator(
        feature_model=backbone,
        conditioner=conditioner,
        label_scaler=label_scaler,
    )

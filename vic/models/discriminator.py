"""Real/fake discriminators over codec latents.

``PooledDiscriminator`` holds the part that never varies — pool the backbone's
per-frame features over time, score the pooled vector — and takes the backbone
by injection.  ``TrueFakeDiscriminator`` and ``ConvTrueFakeDiscriminator`` are
thin subclasses that build a Transformer or a convolutional backbone from the
usual config keys.

Why the split
-------------
The ``PooledFeatureModel`` protocol in ``vic/models/conditioning.py`` is what
lets a discriminator be wrapped into a τ-conditional critic, and it asks for
exactly three members: ``feature_dim``, ``features(batch)`` and ``score(feat)``.
Those three are backbone-independent, so implementing them once means a new
backbone costs a constructor and nothing else — ``ConditionalDiscriminator``,
``ProjectionConditioner`` and the training modules are untouched.

``TrueFakeDiscriminator`` keeps its original keyword signature, which is why the
refactor left its construction sites untouched.

Pooling and sequence length
---------------------------
``masked_mean_pool`` is a plain mean over valid frames, so unlike softmax
attention it does not change behaviour with sequence length — the pooling is not
the part of a Transformer critic that fails to generalise beyond the 1-s
training chunks.  It is kept as-is for the convolutional variant, which means
the conv-vs-Transformer comparison isolates the backbone.

Output convention
-----------------
Raw scalar logit per sample, trained with MSE loss and ±1 targets (matching the
original codebase convention):

    real latents   → target  +1
    fake latents   → target  -1

Spectral normalisation is applied externally by the LightningModule after
construction, via ``apply_spectral_norm`` from ``vic.training.utils``.
"""
from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from vic.data.audio_batch import AudioBatch
from vic.models.blocks import (
    ConvBackbone,
    ConvLayerConfig,
    LayerConfig,
    SequenceBackbone,
    masked_mean_pool,
)


class PooledDiscriminator(nn.Module):
    """Backbone → masked mean pool → linear score.

    Satisfies the ``PooledFeatureModel`` protocol.  The forward pass is split
    into :meth:`features` (backbone + pooling) and :meth:`score` (the linear
    head) so the pooled representation can be reused by a conditioner.

    Parameters
    ----------
    backbone : any module with an ``output_dim`` property whose ``forward``
               maps an :class:`AudioBatch` to ``(B, T, output_dim)`` —
               ``SequenceBackbone`` or ``ConvBackbone``.
    """

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(backbone.output_dim, 1)

    @property
    def feature_dim(self) -> int:
        """Width of the pooled feature returned by :meth:`features`."""
        return self.backbone.output_dim

    def frame_features(self, batch: AudioBatch) -> Tensor:
        """
        batch   : AudioBatch of codec latents, data (B, D, T).
        returns   (B, T, feature_dim) per-frame representation, before pooling.

        Exposed so a conditioner can score each frame against its own condition.  With a
        constant condition that is arithmetically identical to conditioning the pooled
        feature, since both the head and the projection are affine in it — see
        ``vic.models.conditioning``.
        """
        return self.backbone(batch)

    def features(self, batch: AudioBatch) -> Tensor:
        """
        batch   : AudioBatch of codec latents, data (B, D, T).
        returns   (B, feature_dim) pooled sequence representation.
        """
        return masked_mean_pool(self.frame_features(batch), batch.padding_mask)

    def score(self, feat: Tensor) -> Tensor:
        """
        feat    : (B, feature_dim) pooled feature from :meth:`features`.
        returns   (B,) unconditional real/fake score.
        """
        return self.head(feat).squeeze(-1)

    def forward(self, batch: AudioBatch) -> Tensor:
        """
        batch   : AudioBatch of codec latents, data (B, D, T).
        returns   (B, 1) real/fake logit per sample.

        Shape note: kept as (B, 1) for backward compatibility with the existing
        modules, which call ``.squeeze(-1)``.  New code should prefer
        ``score(features(batch))``, which returns (B,) directly.
        """
        return self.score(self.features(batch)).unsqueeze(-1)


class TrueFakeDiscriminator(PooledDiscriminator):
    """Transformer-backboned discriminator (the original architecture).

    Parameters
    ----------
    latent_dim : D, codec latent dimensionality.
    d_model    : internal model width.
    n_heads    : attention heads.
    n_layers   : Transformer depth.
    head_dim   : per-head dim.  Defaults to ``d_model // n_heads``.
    mlp_ratio  : FFN expansion ratio.
    dropout    : dropout probability.
    attn_type  : "dot" (standard scaled dot-product) or "l2" (negative-L2-distance
                 attention). "l2" avoids the spectral-norm / attention-collapse
                 issue: dot-product attention maps flatten under spectral norm,
                 starving the converter (generator) of useful gradients.  The
                 convolutional variant does not have this failure mode and so
                 has no corresponding option.
    """

    def __init__(
        self,
        latent_dim: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        head_dim: int | None = None,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
        attn_type: str = "dot",
        attn_window: int | None = None,
    ):
        hd = head_dim if head_dim is not None else d_model // n_heads
        layer_configs = [
            LayerConfig(
                d_model=d_model, head_dim=hd, mlp_ratio=mlp_ratio,
                dropout=dropout, attn_type=attn_type, attn_window=attn_window,
            )
        ] * n_layers
        super().__init__(
            SequenceBackbone(input_dim=latent_dim, layer_configs=layer_configs)
        )

    @property
    def receptive_field(self) -> int | None:
        """Receptive field in latent frames, or ``None`` under unrestricted attention.

        ``1 + 2·Σ w`` when every layer is windowed.  A bounded field is what makes the
        per-frame critic a *patch* critic: without it each frame's score is one global
        judgement read out at T positions, and a bad window can be excused by a good one
        anywhere in the sequence.
        """
        return self.backbone.receptive_field


class ConvTrueFakeDiscriminator(PooledDiscriminator):
    """Convolution-backboned discriminator: a bounded, length-robust critic.

    Same pooling and head as :class:`TrueFakeDiscriminator`; only the backbone
    differs.  Every frame is scored through a fixed receptive field rather than
    global attention, so the critic's judgement of a 5-s validation segment is
    made from the same evidence it was trained on with 1-s chunks.

    Note on spectral normalisation: ``apply_spectral_norm`` recurses into
    ``nn.Conv1d``, normalising each weight tensor reshaped to a matrix.  For a
    depthwise convolution that constrains the stacked per-channel kernels
    jointly rather than the true operator norm — the standard practice, and
    sufficient here since the goal is to bound the critic, not to certify a
    Lipschitz constant.

    Parameters
    ----------
    latent_dim : D, codec latent dimensionality.
    conv_cfg   : config dict for :meth:`ConvLayerConfig.expand` — ``channels``,
                 ``dilations``, and optionally ``kernel_size``/``mlp_ratio``/
                 ``dropout``/``groups``.
    """

    def __init__(self, latent_dim: int, conv_cfg: dict):
        super().__init__(
            ConvBackbone(
                input_dim=latent_dim,
                layer_configs=ConvLayerConfig.expand(conv_cfg),
            )
        )

    @property
    def receptive_field(self) -> int:
        """Receptive field in latent frames."""
        return self.backbone.receptive_field

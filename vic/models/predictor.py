"""Intensity predictors (P_φ): predict frame-level intensity from codec latents.

Three interchangeable architectures, all with the same contract — take an
``AudioBatch`` of latents, return ``(B, T)`` raw per-frame scalars in the units
of the training labels (calibrated dBSPL, no activation):

=================================  ==============================  ============
class                              temporal context                config type
=================================  ==============================  ============
``MLPIntensityPredictor``          none (frame-wise)               ``mlp``
``TransformerIntensityPredictor``  global (RoPE attention)         ``transformer``
``ConvIntensityPredictor``         fixed receptive field           ``conv``
=================================  ==============================  ============

Each is a sequence backbone from ``audio_utils`` plus ``Linear(width, 1)``
applied per frame, so they differ in the mixing operator and nothing else.

Why a convolutional predictor
-----------------------------
P_φ trains on 1-s chunks (50 latent frames at hop 320 / 16 kHz) but is applied
to whole 2–10 s utterances — both as the evaluation instrument and as the
τ_src labeller in converter training.  RoPE attention is *relatively* positioned
but not length-robust: training never presents relative distances beyond 49
frames, and softmax normalises over sequence length.  A convolutional stack has
a receptive field fixed at construction and applies the same
translation-equivariant operator at any length, which is what makes a prediction
at 5 s mean the same thing as one at 1 s.

Caveat worth knowing before choosing ``dilations``: ``nn.Conv1d`` zero-pads at
sequence ends, so a frame whose receptive field crosses an end is computed
partly from zeros.  A receptive field wider than the training chunk leaves *no*
frame with a clean field, and the stack then only ever extrapolates the interior
operator it meets at evaluation.  Keep the receptive field below the chunk
length in frames, or raise the chunk with it.

Building from config
--------------------
:func:`build_predictor` is the single factory for all three.  Use it rather than
re-deriving the ``type`` dispatch — training scripts, the converter's frozen
labeller and the packaged inference API in ``vic/predict.py`` all go through it,
so a new architecture reaches every one of them at once.
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
)

__all__ = [
    "TransformerIntensityPredictor",
    "MLPIntensityPredictor",
    "ConvIntensityPredictor",
    "build_predictor",
]


class TransformerIntensityPredictor(nn.Module):
    """Predict per-frame intensity from a sequence of codec latent frames.

    Used as:
      P_φ — pretrained predictor, frozen during converter training.
             Aggregate with ``leq_aggregate()`` to get a sequence-level estimate.
      D_ξ — intensity discriminator, initialised from P_φ weights and trained
             adversarially.  Load P_φ weights then unfreeze with
             ``requires_grad_(True)`` to switch roles.

    Architecture
    ------------
    SequenceBackbone (project → bidirectional Transformer) → Linear(d_model, 1)
    applied per frame.

    The output contains raw per-frame scalars in the same units as the training
    labels (calibrated dBSPL).  No activation is applied.  Use
    ``leq_aggregate(predictor(batch), batch.padding_mask)`` to obtain a
    sequence-level equivalent level.

    Parameters
    ----------
    latent_dim : D, codec latent dimensionality.
    d_model    : internal model width.
    n_heads    : attention heads.
    n_layers   : Transformer depth.
    head_dim   : per-head dim.  Defaults to d_model // n_heads.
    mlp_ratio  : FFN expansion ratio.
    dropout    : dropout probability.
    attn_type  : "dot" (standard scaled dot-product) or "l2" (negative-L2-distance
                 attention). "l2" avoids the spectral-norm / attention-collapse
                 issue when used as a spectrally-normalised discriminator (D_ξ).
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
    ):
        super().__init__()
        hd = head_dim if head_dim is not None else d_model // n_heads
        layer_configs = [
            LayerConfig(
                d_model=d_model, head_dim=hd, mlp_ratio=mlp_ratio,
                dropout=dropout, attn_type=attn_type,
            )
        ] * n_layers
        self.backbone = SequenceBackbone(input_dim=latent_dim, layer_configs=layer_configs)
        self.head = nn.Linear(d_model, 1)

    def forward(self, batch: AudioBatch) -> Tensor:
        """
        batch   : AudioBatch of codec latents, data (B, D, T).
        returns   (B, T) predicted intensity per frame.
        """
        return self.head(self.backbone(batch)).squeeze(-1)


class MLPIntensityPredictor(nn.Module):
    """Predict per-frame intensity from codec latents using a frame-wise MLP.

    Each frame is processed independently — no cross-frame context.  The same
    MLP is applied at every time step, equivalent to a 1-D pointwise convolution
    with kernel size 1.

    Shares the same ``forward`` interface as ``TransformerIntensityPredictor`` so it is a
    drop-in replacement in ``PredictorModule``.

    Parameters
    ----------
    latent_dim : D, codec latent dimensionality.
    d_model    : hidden width of every MLP layer.
    n_layers   : number of hidden Linear→LayerNorm→GELU blocks.
    dropout    : dropout probability applied before each hidden layer.
    """

    def __init__(
        self,
        latent_dim: int,
        d_model: int,
        n_layers: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Linear(latent_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        ]
        for _ in range(n_layers - 1):
            layers += [
                nn.Dropout(dropout),
                nn.Linear(d_model, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
            ]
        self.mlp = nn.Sequential(*layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, batch: AudioBatch) -> Tensor:
        """
        batch   : AudioBatch of codec latents, data (B, D, T).
        returns   (B, T) predicted intensity per frame.
        """
        return self.head(self.mlp(batch.BTC)).squeeze(-1)  # (B, T, 1) → (B, T)


class ConvIntensityPredictor(nn.Module):
    """Predict per-frame intensity through a fixed receptive field.

    ``TransformerIntensityPredictor`` with the attention backbone swapped for a
    dilated-convolution one — see the module docstring for why that matters for
    P_φ specifically.  ``ConvBackbone`` already has ``SequenceBackbone``'s
    contract, so this is the same two lines: backbone → per-frame linear head.

    No conditioning path is built (``film_dim`` stays ``None``): P_φ is
    unconditional, unlike the converter, which modulates on τ.

    Parameters
    ----------
    latent_dim : D, codec latent dimensionality.
    conv_cfg   : config dict for :meth:`ConvLayerConfig.expand` — ``channels``
                 and ``dilations``, optionally ``kernel_size`` / ``mlp_ratio`` /
                 ``dropout`` / ``groups``.  The ``dilations`` list sets the depth
                 *and* the receptive field, so the two cannot disagree.
    """

    def __init__(self, latent_dim: int, conv_cfg: dict):
        super().__init__()
        self.backbone = ConvBackbone(
            input_dim=latent_dim,
            layer_configs=ConvLayerConfig.expand(conv_cfg),
        )
        self.head = nn.Linear(self.backbone.output_dim, 1)

    @property
    def receptive_field(self) -> int:
        """Receptive field in latent frames."""
        return self.backbone.receptive_field

    def forward(self, batch: AudioBatch) -> Tensor:
        """
        batch   : AudioBatch of codec latents, data (B, D, T).
        returns   (B, T) predicted intensity per frame.
        """
        return self.head(self.backbone(batch)).squeeze(-1)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_predictor(
    model_cfg: dict,
    latent_dim: int,
    dropout_override: float | None = None,
) -> nn.Module:
    """Instantiate a predictor from a ``model``-section config dict.

    Parameters
    ----------
    model_cfg        : the ``model:`` block itself — not the whole config — with
                       a ``type`` key of ``transformer`` (default), ``mlp`` or
                       ``conv``, plus that architecture's parameters.
    latent_dim       : D, the feature width the extractor produces.
    dropout_override : forces this dropout regardless of what the config says.
                       Callers that rebuild a *frozen, already-trained* predictor
                       to load weights into pass ``0.0``: the checkpoint's own
                       training dropout is irrelevant at inference and reading it
                       from config would silently apply dropout to a model being
                       used as a fixed measuring instrument.  ``None`` (default)
                       means honour ``model_cfg["dropout"]``.
    """
    kind = model_cfg.get("type", "transformer")
    dropout = (
        model_cfg.get("dropout", 0.0) if dropout_override is None else dropout_override
    )

    if kind == "transformer":
        return TransformerIntensityPredictor(
            latent_dim=latent_dim,
            d_model=model_cfg["d_model"],
            n_heads=model_cfg["n_heads"],
            n_layers=model_cfg["n_layers"],
            head_dim=model_cfg.get("head_dim"),
            mlp_ratio=model_cfg.get("mlp_ratio", 4),
            dropout=dropout,
            attn_type=model_cfg.get("attn_type", "dot"),
        )
    if kind == "mlp":
        return MLPIntensityPredictor(
            latent_dim=latent_dim,
            d_model=model_cfg["d_model"],
            n_layers=model_cfg["n_layers"],
            dropout=dropout,
        )
    if kind == "conv":
        if "attn_type" in model_cfg:
            raise ValueError(
                "model.attn_type is set on a conv predictor. It only applies to "
                "attention — remove the key."
            )
        return ConvIntensityPredictor(
            latent_dim=latent_dim,
            conv_cfg={**model_cfg, "dropout": dropout},
        )
    raise ValueError(
        f"Unknown model type: {kind!r} (expected 'transformer', 'mlp' or 'conv')"
    )

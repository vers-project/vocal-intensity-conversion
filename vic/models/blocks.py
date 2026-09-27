"""Shared building blocks for all VIC models.

Sequence mixing
---------------
Two interchangeable families, both imported from ``audio_utils``:

``audio_utils.models.blocks``
    RoPE attention, encoder, sequence backbone.  See that module for
    ``attn_type`` options ("dot" for standard scaled dot-product, "l2" for
    negative-L2-distance attention, which avoids the spectral-norm / attention
    collapse issue in spectrally-normalised discriminators).

``audio_utils.models.conv``
    Dilated-convolution counterparts with a fixed receptive field.  ``ConvBlock``
    is ``TransformerBlock`` with attention replaced by a depthwise dilated
    convolution and nothing else changed, so a conv-vs-Transformer comparison at
    matched width isolates the mixing operator.  Relevant here because every VIC
    model trains on 1-s chunks but validates on 2–10 s sequences, and RoPE
    attention is not length-robust: training never presents relative distances
    beyond 49 frames, and softmax dilutes as the sequence grows.

Intensity conditioning
----------------------
Two drop-in embedders with the same interface:

    embed_dim  : int property — width of the output vector.
    forward    : (B,) → (B, embed_dim)

The converter concatenates the embedding with the projected z along the feature
axis and uses a single cond_proj to mix them, so the two embedders differ only
in what they produce — no converter code changes when swapping.

  ScalarIntensityEmbedding
      Returns the raw scalar as a (B, 1) vector.  No learned parameters.
      Equivalent to the original "cat(z, tau)" approach: the following
      cond_proj Linear(d_model+1, d_model) learns the same weighting.

  SinusoidalIntensityEmbedding
      Sinusoidal multi-frequency encoding → two-layer MLP → (B, embed_dim).
      Genuinely richer than the scalar approach: the non-linear MLP can
      produce different conditioning vectors for the same dB value at
      different parts of the intensity range.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

from audio_utils.models.blocks import (
    LayerConfig,
    RoPEAttention,
    SequenceBackbone,
    TransformerBlock,
    TransformerEncoder,
    masked_mean_pool,
)
from audio_utils.models.conv import (
    ConvBackbone,
    ConvBlock,
    ConvEncoder,
    ConvLayerConfig,
    FiLM,
)

__all__ = [
    "LayerConfig",
    "RoPEAttention",
    "SequenceBackbone",
    "TransformerBlock",
    "TransformerEncoder",
    "masked_mean_pool",
    "ConvBackbone",
    "ConvBlock",
    "ConvEncoder",
    "ConvLayerConfig",
    "FiLM",
    "ScalarIntensityEmbedding",
    "SinusoidalIntensityEmbedding",
]


# ---------------------------------------------------------------------------
# Intensity embedders
# ---------------------------------------------------------------------------


class ScalarIntensityEmbedding(nn.Module):
    """Return the scalar intensity as a (B, 1) vector — no learned parameters.

    This is equivalent to the original approach of concatenating the raw
    dB (or normalised dB) value to the latent before the input projection.
    The cond_proj layer in the converter plays the role of the original
    in_proj: it sees [z_projected | tau] and learns a weighting for tau
    as one column of its weight matrix.

    embed_dim is always 1.
    """

    embed_dim: int = 1

    def forward(self, tau: Tensor) -> Tensor:
        """tau : (B,)  → (B, 1)"""
        return tau.unsqueeze(-1)


class SinusoidalIntensityEmbedding(nn.Module):
    """Sinusoidal multi-frequency encoding followed by a two-layer MLP.

    Maps a scalar to ``embed_dim`` dimensions by encoding it at logarithmically
    spaced frequencies (fine-grained to coarse-grained), then learning a
    non-linear projection.  Provides richer conditioning than the scalar
    approach when generalising to unseen intensity values.

    Parameters
    ----------
    embed_dim : output width; typically a small fraction of d_model (e.g. 64).
    d_sin     : width of the sinusoidal encoding (must be even).
    """

    def __init__(self, embed_dim: int, d_sin: int = 64):
        super().__init__()
        assert d_sin % 2 == 0, "d_sin must be even"
        self.d_sin = d_sin
        self.embed_dim = embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(d_sin, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    @staticmethod
    def _sinusoidal(x: Tensor, dim: int) -> Tensor:
        half = dim // 2
        freqs = torch.exp(
            torch.arange(half, device=x.device, dtype=x.dtype)
            * (-math.log(10_000.0) / half)
        )
        args = x.unsqueeze(-1) * freqs.unsqueeze(0)   # (B, half)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

    def forward(self, tau: Tensor) -> Tensor:
        """tau : (B,)  → (B, embed_dim)"""
        return self.mlp(self._sinusoidal(tau, self.d_sin))


def build_intensity_embed(embed_cfg: dict) -> nn.Module:
    """Build an intensity embedder from a config block.

    Shared by the converter and the discriminator conditioner — they take
    independent config blocks so the two can differ (a scalar embedding is
    usually enough for the converter, while the projection conditioner tends to
    benefit from the sinusoidal one).
    """
    kind = embed_cfg.get("type", "scalar")
    if kind == "scalar":
        return ScalarIntensityEmbedding()
    elif kind == "sinusoidal":
        return SinusoidalIntensityEmbedding(
            embed_dim=embed_cfg.get("embed_dim", 64),
            d_sin=embed_cfg.get("d_sin", 64),
        )
    else:
        raise ValueError(
            f"Unknown intensity_embed type: {kind!r} (expected 'scalar' or 'sinusoidal')"
        )

"""Latent converters (C_θ): map codec latents at intensity ℓ to intensity τ.

Both converters share the same conditioning pattern:

    1. Project z:  Linear(latent_dim, d_model)           → (B, T, d_model)
    2. Normalise τ via ``label_scaler`` (if given), then embed it:
                   intensity_embed(tau)                  → (B, embed_dim)
    3. Condition:  cond_proj(cat([z_proj, tau_expanded])) → (B, T, d_model)
                   where cond_proj = Linear(d_model + embed_dim, d_model)
    4. Process
    5. Output:     z_hat = z + output_proj(processed)    residual connection

``label_scaler`` matters more than it looks.  ``cond_proj`` sees the projected
latent — LayerNorm'd to ~N(0,1) in ContextualConverter — concatenated with the
τ columns.  Handed raw dBSPL, τ arrives with magnitude ~60 against neighbours of
magnitude ~1, so at initialisation it contributes a large constant offset that
training must first suppress, and suppressing it scales down the informative
±12 dB deviation along with the useless 60 dB pedestal.  Normalising makes the
conditioning input comparable to the latent, and matches what
``ConditionalDiscriminator`` has always done on the discriminator side.
``None`` keeps the old raw-τ behaviour for the earlier converter modules.

The scaler lives on the converter rather than in the training module because
callbacks call ``converter(z, tau)`` directly with raw dBSPL; putting it here is
what keeps evaluation and training conditioned identically.

The intensity embedder is injected at construction time, making it trivial to
swap between ScalarIntensityEmbedding and SinusoidalIntensityEmbedding without
touching any converter code.

FrameConverter
    Feedforward-only: no attention, each frame processed independently.
    Fast, no temporal context.  Equivalent to the original frame-level
    experiment (exp_frame_conversion).

ContextualConverter
    Bidirectional TransformerEncoder: each frame attends to the full sequence.
    Richer but slower.  Equivalent to the original contextual experiment
    (exp_contextual_conversion), with the added benefit of proper padding masks.

ConvConverter
    Dilated convolutions with a bounded receptive field, in place of attention.
    Same conditioning options, same residual output; motivated by the fact that
    every converter trains on 1-s chunks and is evaluated on 2–10 s sequences,
    which is exactly where RoPE attention stops being trustworthy.  Adds a
    second conditioning route (per-block FiLM) that a bounded receptive field
    makes worth having — see below.

Where τ enters
--------------
``ContextualConverter`` injects τ once, at the input.  Under global attention
that is enough: every frame can reach the conditioned representation of every
other frame at any depth, so the conditioning cannot be locally diluted away.

A convolutional stack has no such global route.  τ enters as ``embed_dim``
columns out of ``channels`` at layer 0 and must then survive six residual blocks
of purely local mixing, and the training signal that would preserve it is one
scalar adversarial term per sample — the conditioning variable is exactly what a
CNN tends to let vanish.  ``ConvConverter`` therefore offers
``conditioning="film"``, which re-derives ``(γ, β)`` from τ at every block so the
condition is re-asserted at each depth and cannot be forgotten by construction.

``conditioning="concat"`` reproduces the Transformer's conditioning exactly and
remains available for ablation.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from vic.data.audio_batch import AudioBatch
from vic.models.blocks import (
    ConvEncoder,
    ConvLayerConfig,
    TransformerEncoder,
    build_intensity_embed,
)


# ---------------------------------------------------------------------------
# Frame-level converter
# ---------------------------------------------------------------------------


def _embed_tau(
    intensity_embed: nn.Module,
    tau: Tensor,
    label_scaler,
    n_frames: int | None = None,
) -> Tensor:
    """Normalise and embed the intensity condition.

    Parameters
    ----------
    intensity_embed : embedder with ``forward(tau) -> (..., embed_dim)``.
    tau             : (B,) one target intensity per sequence, or (B, T) a per-frame target
                      curve.  Raw dBSPL when a ``label_scaler`` is given.
    label_scaler    : optional ``LabelScaler``.
    n_frames        : when given, a sequence-level embedding is broadcast to
                      ``(B, n_frames, embed_dim)`` — what the concat route needs.  Leave
                      ``None`` for FiLM, which broadcasts over time itself, once per block.

    Returns
    -------
    ``(B, embed_dim)`` or ``(B, T, embed_dim)``.

    A (B, T) curve passes through the embedders unchanged: ``_sinusoidal`` broadcasts
    ``(B, T, 1) * (1, half)`` and the MLP acts on the last dimension, so no shape handling
    is needed beyond deciding whether to expand.
    """
    if label_scaler is not None:
        tau = label_scaler.normalise(tau)
    emb = intensity_embed(tau)
    if n_frames is not None and emb.dim() == 2:
        emb = emb.unsqueeze(1).expand(-1, n_frames, -1)
    return emb


class FrameConverter(nn.Module):
    """Frame-wise feedforward converter: no temporal context.

    Each frame is processed independently via a small MLP after conditioning
    on the target intensity.  Fast and parameter-efficient; useful as a
    baseline before adding attention.

    Architecture (per frame)
    ------------------------
    [z_i | tau_emb]                      (latent_dim + embed_dim)
    → cond_proj  Linear(*, d_model)      (d_model)
    → n_layers × (Linear(d_model, d_model) + GELU + Dropout)
    → output_proj  Linear(d_model, latent_dim)   (delta)
    → residual: z_hat_i = z_i + delta_i

    Parameters
    ----------
    latent_dim      : D, codec latent dimensionality.
    d_model         : hidden width of the MLP.
    intensity_embed : any embedder with ``embed_dim`` attribute and
                      ``forward(tau: (B,)) → (B, embed_dim)``.
    n_layers        : number of hidden MLP layers (depth after cond_proj).
    dropout         : dropout probability inside the MLP.
    """

    def __init__(
        self,
        latent_dim: int,
        d_model: int,
        intensity_embed: nn.Module,
        n_layers: int = 3,
        dropout: float = 0.0,
        label_scaler=None,
    ):
        super().__init__()
        self.intensity_embed = intensity_embed
        self.label_scaler = label_scaler
        embed_dim = intensity_embed.embed_dim

        self.cond_proj = nn.Linear(latent_dim + embed_dim, d_model)

        hidden_layers: list[nn.Module] = []
        for _ in range(n_layers):
            hidden_layers += [
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(p=dropout),
            ]
        self.hidden = nn.Sequential(*hidden_layers)

        self.output_proj = nn.Linear(d_model, latent_dim)

    def forward(self, batch: AudioBatch, tau: Tensor) -> AudioBatch:
        """
        batch : AudioBatch of codec latents, data (B, D, T).
        tau   : (B,) one target intensity per sequence, or (B, T) a per-frame
                target curve.
        returns AudioBatch with the same lengths/sr, converted data (B, D, T).
        """
        z_BTC = batch.BTC                                      # (B, T, D)
        _, T, _ = z_BTC.shape

        tau_exp = _embed_tau(self.intensity_embed, tau, self.label_scaler, n_frames=T)

        x = self.cond_proj(torch.cat([z_BTC, tau_exp], dim=-1))  # (B, T, d_model)
        x = self.hidden(x)                                     # (B, T, d_model)
        delta = self.output_proj(x)                            # (B, T, D)

        z_hat_BCT = (z_BTC + delta).transpose(1, 2).contiguous()  # (B, D, T)
        return AudioBatch(
            data=z_hat_BCT,
            lengths=batch.lengths,
            sample_rate=batch.sample_rate,
        )


# ---------------------------------------------------------------------------
# Contextual converter
# ---------------------------------------------------------------------------


class ContextualConverter(nn.Module):
    """Bidirectional Transformer converter with full temporal context.

    Each frame attends to all other frames (across time) before predicting
    the residual delta.  Padding frames are masked in attention so they do
    not contaminate the representations of valid frames.

    Architecture
    ------------
    z → input_proj + input_norm              (B, T, d_model)
    [z_proj | tau_emb] → cond_proj           (B, T, d_model)
    → TransformerEncoder (bidirectional)     (B, T, d_model)
    → output_norm → output_proj              (B, T, D)  delta
    → residual: z_hat = z + delta

    Parameters
    ----------
    latent_dim      : D, codec latent dimensionality.
    d_model         : Transformer model width.
    n_heads         : attention heads.
    n_layers        : Transformer depth.
    intensity_embed : embedder with ``embed_dim`` and ``forward(tau) → embed``.
    head_dim        : per-head dim.  Defaults to d_model // n_heads.
    mlp_ratio       : FFN expansion ratio.
    dropout         : dropout probability.
    """

    def __init__(
        self,
        latent_dim: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        intensity_embed: nn.Module,
        head_dim: int | None = None,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
        label_scaler=None,
        attn_window: int | None = None,
    ):
        super().__init__()
        self.intensity_embed = intensity_embed
        self.label_scaler = label_scaler
        embed_dim = intensity_embed.embed_dim

        self.input_proj = nn.Linear(latent_dim, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        self.cond_proj  = nn.Linear(d_model + embed_dim, d_model)

        self.transformer = TransformerEncoder(
            n_layers=n_layers,
            dim=d_model,
            n_heads=n_heads,
            head_dim=head_dim,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            causal=False,
            attn_window=attn_window,
        )

        self.output_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, latent_dim)

    @property
    def receptive_field(self) -> int | None:
        """Frames one output depends on, or ``None`` under unrestricted attention.

        ``1 + 2·Σ w`` when every layer is windowed.  Bounding it is what keeps the
        converter in distribution across sequence lengths: RoPE's failure mode is
        relative distances never seen in training, and a window caps the largest distance
        any attention op computes.

        Read this against the training chunk.  A field WIDER than the chunk is aggregated
        over a span training never supplied, which reintroduces the length dependence the
        window is meant to remove; a field much NARROWER than the chunk cannot measure the
        chunk-global τ_src that ``Δ = τ_tgt − τ_src`` needs, which is what the per-frame
        target curve exists to supply.
        """
        return self.transformer.receptive_field

    def forward(self, batch: AudioBatch, tau: Tensor) -> AudioBatch:
        """
        batch : AudioBatch of codec latents, data (B, D, T).
        tau   : (B,) one target intensity per sequence, or (B, T) a per-frame
                target curve.
        returns AudioBatch with the same lengths/sr, converted data (B, D, T).
        """
        z_BTC = batch.BTC                                      # (B, T, D)
        _, T, _ = z_BTC.shape

        z_proj = self.input_norm(self.input_proj(z_BTC))      # (B, T, d_model)

        tau_exp = _embed_tau(self.intensity_embed, tau, self.label_scaler, n_frames=T)

        x = self.cond_proj(torch.cat([z_proj, tau_exp], dim=-1))  # (B, T, d_model)
        x = self.transformer(x, key_padding_mask=batch.key_padding_mask)
        delta = self.output_proj(self.output_norm(x))         # (B, T, D)

        z_hat_BCT = (z_BTC + delta).transpose(1, 2).contiguous()  # (B, D, T)
        return AudioBatch(
            data=z_hat_BCT,
            lengths=batch.lengths,
            sample_rate=batch.sample_rate,
        )


# ---------------------------------------------------------------------------
# Convolutional converter
# ---------------------------------------------------------------------------


class ConvConverter(nn.Module):
    """Dilated-convolution converter with a fixed receptive field.

    Structurally parallel to :class:`ContextualConverter` — same input
    projection, same τ normalisation via ``label_scaler``, same residual output
    — with the TransformerEncoder replaced by a ``ConvEncoder``.

    Architecture
    ------------
    ::

        z → input_proj + input_norm                (B, T, C)
        concat:  [z_proj | tau_emb] → cond_proj    (B, T, C)
        film:    z_proj, with tau_emb modulating every block
        → ConvEncoder (bidirectional, dilated)     (B, T, C)
        → output_norm → output_proj                (B, T, D)  delta
        → residual: z_hat = z + delta

    The residual formulation matters more here than for the Transformer: the
    identity is available at zero cost, so the convolutional stack only has to
    model the *change* that raising or lowering intensity implies — plausibly a
    local spectral-tilt and voice-quality transform, which is the assumption
    that makes a bounded receptive field reasonable in the first place.

    Parameters
    ----------
    latent_dim      : D, codec latent dimensionality.
    conv_cfg        : config dict for :meth:`ConvLayerConfig.expand` —
                      ``channels``, ``dilations``, and optionally
                      ``kernel_size``/``mlp_ratio``/``dropout``/``groups``.
    intensity_embed : embedder with ``embed_dim`` and ``forward(tau) → embed``.
    conditioning    : ``"concat"`` (default, matches ContextualConverter) or
                      ``"film"`` (per-block modulation).
    label_scaler    : optional ``LabelScaler`` normalising raw dBSPL τ.
    """

    def __init__(
        self,
        latent_dim: int,
        conv_cfg: dict,
        intensity_embed: nn.Module,
        conditioning: str = "concat",
        label_scaler=None,
    ):
        super().__init__()
        if conditioning not in ("concat", "film"):
            raise ValueError(
                f"conditioning must be 'concat' or 'film', got {conditioning!r}"
            )
        self.intensity_embed = intensity_embed
        self.label_scaler = label_scaler
        self.conditioning = conditioning
        embed_dim = intensity_embed.embed_dim

        layer_configs = ConvLayerConfig.expand(conv_cfg)
        d_model = layer_configs[0].d_model

        self.input_proj = nn.Linear(latent_dim, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        # Exactly one conditioning route is built, so a "concat" run carries no
        # FiLM parameters and a "film" run no cond_proj — the two are comparable
        # at the width the config asks for, not at two different capacities.
        self.cond_proj = (
            nn.Linear(d_model + embed_dim, d_model) if conditioning == "concat" else None
        )

        self.encoder = ConvEncoder(
            layer_configs,
            film_dim=embed_dim if conditioning == "film" else None,
        )

        self.output_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, latent_dim)

    @property
    def receptive_field(self) -> int:
        """Receptive field in latent frames."""
        return self.encoder.receptive_field

    def forward(self, batch: AudioBatch, tau: Tensor) -> AudioBatch:
        """
        batch : AudioBatch of codec latents, data (B, D, T).
        tau   : (B,) one target intensity per sequence, or (B, T) a per-frame target
                curve.  Raw dBSPL when a ``label_scaler`` was supplied.
        returns AudioBatch with the same lengths/sr, converted data (B, D, T).
        """
        z_BTC = batch.BTC                                      # (B, T, D)
        _, T, _ = z_BTC.shape

        x = self.input_norm(self.input_proj(z_BTC))            # (B, T, C)

        if self.cond_proj is not None:
            tau_exp = _embed_tau(self.intensity_embed, tau, self.label_scaler, n_frames=T)
            x = self.cond_proj(torch.cat([x, tau_exp], dim=-1))
            cond = None
        else:
            cond = _embed_tau(self.intensity_embed, tau, self.label_scaler)

        x = self.encoder(x, key_padding_mask=batch.key_padding_mask, cond=cond)
        delta = self.output_proj(self.output_norm(x))         # (B, T, D)

        z_hat_BCT = (z_BTC + delta).transpose(1, 2).contiguous()  # (B, D, T)
        return AudioBatch(
            data=z_hat_BCT,
            lengths=batch.lengths,
            sample_rate=batch.sample_rate,
        )


def receptive_field_frames(cfg: dict) -> int | None:
    """Receptive field of the configured converter, in latent frames.

    ``None`` for an architecture with no bounded field (``contextual``, ``frame``).
    Reads the config without building the model, so a diagnostic script can ask what
    window a run will use without loading a codec first.
    """
    cc = cfg.get("model", {}).get("converter")
    if cc is None or cc.get("type", "contextual") != "conv":
        return None
    return ConvEncoder(ConvLayerConfig.expand(cc)).receptive_field


def build_converter(cfg: dict, latent_dim: int, label_scaler=None) -> nn.Module:
    """Build C_θ from a training config.

    ``cfg`` is the whole config; the converter block is read from
    ``cfg["model"]["converter"]``.

    ``label_scaler`` is anything exposing ``normalise(Tensor) -> Tensor``, and is
    handed to the converter so τ is normalised internally — exactly as
    ``ConditionalDiscriminator`` does.  ``None`` is accepted because the
    converter classes accept it, but it feeds raw dBSPL (~60) against LayerNorm'd
    latents (~1); see the module docstring above.
    """
    cc = cfg["model"]["converter"]
    kind = cc.get("type", "contextual")
    embed = build_intensity_embed(cc.get("intensity_embed", {}))
    if kind == "conv":
        return ConvConverter(
            latent_dim=latent_dim,
            conv_cfg=cc,
            intensity_embed=embed,
            conditioning=cc.get("conditioning", "concat"),
            label_scaler=label_scaler,
        )
    elif kind == "frame":
        return FrameConverter(
            latent_dim=latent_dim,
            d_model=cc["d_model"],
            intensity_embed=embed,
            n_layers=cc.get("n_layers", 3),
            dropout=cc.get("dropout", 0.0),
            label_scaler=label_scaler,
        )
    elif kind == "contextual":
        return ContextualConverter(
            latent_dim=latent_dim,
            d_model=cc["d_model"],
            n_heads=cc["n_heads"],
            n_layers=cc["n_layers"],
            intensity_embed=embed,
            head_dim=cc.get("head_dim"),
            mlp_ratio=cc.get("mlp_ratio", 4),
            dropout=cc.get("dropout", 0.0),
            label_scaler=label_scaler,
            attn_window=cc.get("attn_window"),
        )
    else:
        raise ValueError(
            f"Unknown converter type: {kind!r} "
            "(expected 'conv', 'frame' or 'contextual')"
        )

"""The per-frame condition must reduce exactly to the scalar one on a flat curve.

That equivalence is what makes the per-frame condition a strictly additive change to the
shared model layer: any drift here would silently move the scalar-τ converter.
"""
import torch

from vic.data.audio_batch import AudioBatch
from vic.models.conditioning import build_conditional_discriminator
from vic.models.converter import build_converter
from vic.training.losses import lsgan_loss, masked_lsgan_loss

D, B, T = 8, 4, 25
EMBED = {"type": "sinusoidal", "embed_dim": 16, "d_sin": 16}


def _batch(lengths=None):
    lengths = torch.full((B,), T) if lengths is None else lengths
    return AudioBatch(data=torch.randn(B, D, T), lengths=lengths, sample_rate=16_000)


def _disc():
    cfg = {"model": {
        "discriminator": {"type": "conv", "channels": 32, "kernel_size": 3,
                          "dilations": [1, 3], "mlp_ratio": 2, "dropout": 0.0},
        "conditioning": {"type": "projection", "intensity_embed": EMBED},
    }}
    return build_conditional_discriminator(cfg, latent_dim=D).eval()


# ------------------------------------------------------- discriminator equivalence

def test_flat_curve_matches_the_scalar_score():
    """``mean_t head(φ_t) ≡ head(mean_t φ_t)`` — head and projection are both affine."""
    torch.manual_seed(0)
    disc, z = _disc(), _batch()
    tau = torch.tensor([50.0, 60.0, 70.0, 80.0])
    with torch.no_grad():
        scalar = disc(z, tau)
        curve = disc(z, tau.unsqueeze(-1).expand(B, T))
    assert curve.shape == (B, T)
    assert torch.allclose(curve.mean(-1), scalar, atol=1e-5)


def test_flat_curve_matches_the_scalar_score_under_padding():
    torch.manual_seed(1)
    disc = _disc()
    z = _batch(lengths=torch.tensor([25, 20, 13, 7]))
    tau = torch.tensor([50.0, 60.0, 70.0, 80.0])
    with torch.no_grad():
        scalar = disc(z, tau)
        curve = disc(z, tau.unsqueeze(-1).expand(B, T))
    m = z.padding_mask.float()
    assert torch.allclose((curve * m).sum(-1) / m.sum(-1), scalar, atol=1e-5)


def test_a_varying_curve_actually_changes_the_score():
    torch.manual_seed(2)
    disc, z = _disc(), _batch()
    flat = torch.full((B, T), 65.0)
    varying = torch.linspace(40, 85, T).expand(B, T)
    with torch.no_grad():
        assert not torch.allclose(disc(z, flat), disc(z, varying), atol=1e-3)


# ------------------------------------------------------------ converter equivalence

def test_every_converter_accepts_a_flat_curve_identically():
    cases = [("frame", {"d_model": 16, "n_layers": 2}),
             ("contextual", {"d_model": 16, "n_heads": 2, "n_layers": 2}),
             ("conv", {"channels": 16, "kernel_size": 3, "dilations": [1, 3],
                       "conditioning": "concat"}),
             ("conv", {"channels": 16, "kernel_size": 3, "dilations": [1, 3],
                       "conditioning": "film"})]
    for kind, extra in cases:
        torch.manual_seed(3)
        cfg = {"model": {"converter": dict(type=kind, intensity_embed=EMBED, **extra)}}
        conv = build_converter(cfg, latent_dim=D).eval()
        # FiLM is zero-initialised (identity), so give it real parameters or the test
        # would pass without exercising the conditioning path at all.
        for m in conv.modules():
            if type(m).__name__ == "FiLM":
                torch.nn.init.normal_(m.proj.weight, std=0.1)
        z = _batch()
        tau = torch.tensor([50.0, 60.0, 70.0, 80.0])
        with torch.no_grad():
            a = conv(z, tau).data
            b = conv(z, tau.unsqueeze(-1).expand(B, T)).data
        assert torch.allclose(a, b, atol=1e-5), f"{kind}/{extra.get('conditioning')}"


# ---------------------------------------------------------------- masked LSGAN loss

def test_unmasked_matches_lsgan_loss():
    s = torch.randn(B, T)
    assert torch.allclose(masked_lsgan_loss(s, 1.0), lsgan_loss(s, 1.0))


def test_padding_is_excluded():
    s = torch.randn(B, T)
    mask = torch.zeros(B, T, dtype=torch.bool)
    mask[:, :6] = True
    s_junk = s.clone()
    s_junk[:, 6:] = 1e3                                   # padded scores are meaningless
    assert torch.allclose(masked_lsgan_loss(s_junk, -1.0, mask),
                          lsgan_loss(s[:, :6], -1.0))


def test_per_frame_is_pooled_plus_within_sequence_variance():
    """The entire difference between a pooled and a patch critic, as an identity."""
    s = torch.randn(B, T)
    per_frame = ((s - 1.0) ** 2).mean(-1)
    pooled = (s.mean(-1) - 1.0) ** 2
    assert torch.allclose(per_frame, pooled + s.var(-1, unbiased=False), atol=1e-5)


def test_an_empty_mask_contributes_nothing():
    """A batch where the margin holds nowhere must not divide by zero."""
    s = torch.randn(B, T)
    assert masked_lsgan_loss(s, 1.0, torch.zeros(B, T, dtype=torch.bool)).item() == 0.0

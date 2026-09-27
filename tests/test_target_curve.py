"""Unit tests for the per-frame intensity target."""
import pytest
import torch

from vic.features.spl import leq_aggregate
from vic.features.target_curve import moving_leq, shift_curve


# --------------------------------------------------------------------- moving_leq

def test_window_of_one_is_the_identity():
    x = torch.tensor([[34.0, 41.0, 68.0, 55.0, 30.0]])
    assert torch.equal(moving_leq(x, 1), x)


def test_constant_input_has_no_edge_droop():
    """The reason numerator and denominator are both pooled.

    Averaging a window that overhangs the start against zeros would pull the first frames
    down — the ≤3 dB artefact documented for ``FrameLevelTransform``, but over 53 frames
    rather than 400 samples.
    """
    x = torch.full((2, 40), 62.0)
    assert torch.allclose(moving_leq(x, 11), x, atol=1e-4)


def test_full_window_matches_leq_aggregate():
    x = torch.tensor([[30.0, 45.0, 70.0, 52.0, 38.0]])
    wide = moving_leq(x, 101)                      # wider than the sequence
    assert torch.allclose(wide, leq_aggregate(x).expand_as(x), atol=1e-4)


def test_padding_does_not_leak_into_valid_frames():
    """A sequence must smooth the same whether or not it shares a batch with a longer one."""
    short = torch.tensor([[30.0, 45.0, 70.0, 52.0, 38.0]])
    padded = torch.cat([short, torch.full((1, 7), 999.0)], dim=-1)
    mask = torch.zeros(1, 12, dtype=torch.bool)
    mask[:, :5] = True
    assert torch.allclose(moving_leq(padded, 3, mask)[:, :5], moving_leq(short, 3), atol=1e-4)


def test_is_an_rms_not_a_mean_of_decibels():
    """Leq is the level of the RMS pressure, so the loud frame dominates.

    A naive mean of the dB values would read 50.
    """
    x = torch.tensor([[20.0, 80.0]])
    assert moving_leq(x, 3)[0, 0].item() == pytest.approx(76.99, abs=0.01)


def test_even_window_is_rejected():
    with pytest.raises(ValueError, match="odd"):
        moving_leq(torch.zeros(1, 8), 4)


# -------------------------------------------------------------------- shift_curve

def test_no_floor_is_a_plain_shift():
    x = torch.tensor([[34.0, 68.0]])
    out = shift_curve(x, torch.tensor([14.6]), None)
    assert torch.allclose(out, x + 14.6, atol=1e-4)


def test_zero_displacement_is_the_identity():
    x = torch.tensor([[34.0, 50.0, 68.0]])
    assert torch.allclose(shift_curve(x, torch.tensor([0.0]), 33.0), x, atol=1e-4)


def test_frames_at_the_floor_do_not_move():
    x = torch.tensor([[33.0, 33.0]])
    out = shift_curve(x, torch.tensor([20.0]), 33.0)
    assert torch.allclose(out, x, atol=1e-4)


def test_frames_far_above_the_floor_move_by_delta():
    x = torch.tensor([[68.0]])
    out = shift_curve(x, torch.tensor([14.6]), 33.0)
    assert out.item() == pytest.approx(82.6, abs=0.01)


def test_the_worked_avid_example():
    """0.9 s pause at 34 dB + vowels at 68 dB, Δ = +14.6, floor 33 dB.

    The pause rises 8.3 dB rather than the 46 dB the scalar target implied — and rises at
    all only because it sits 1 dB above the assumed floor, which is why floor_db has to be
    read off the corpus rather than guessed.
    """
    out = shift_curve(torch.tensor([[34.0, 68.0]]), torch.tensor([14.6]), 33.0)
    assert out[0, 0].item() == pytest.approx(42.28, abs=0.01)
    assert out[0, 1].item() == pytest.approx(82.60, abs=0.01)


def test_monotone_in_the_input_level():
    x = torch.linspace(20, 90, 71).unsqueeze(0)
    out = shift_curve(x, torch.tensor([12.0]), 33.0)
    assert torch.all(out.diff() >= 0)                       # never inverts
    above = x[0] > 34.0
    assert torch.all(out[0][above].diff() > 0)              # strict above the floor


def test_below_floor_is_clamped_not_negative():
    out = shift_curve(torch.tensor([[10.0]]), torch.tensor([20.0]), 33.0)
    assert out.item() == pytest.approx(33.0, abs=1e-4)


def test_per_sample_displacement_broadcasts():
    x = torch.full((3, 4), 60.0)
    out = shift_curve(x, torch.tensor([-10.0, 0.0, 10.0]), None)
    assert torch.allclose(out[:, 0], torch.tensor([50.0, 60.0, 70.0]), atol=1e-4)

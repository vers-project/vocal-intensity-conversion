"""The perceived-level machinery, checked against data whose truth is known.

Every test here simulates forced choices from a stimulus whose perceived level
and whose listener precision were chosen in advance, then asks whether the fit
returns them.  That is the only way to know the decibels this module reports mean
what they claim: on real data there is nothing to compare them against, which is
exactly why the analysis validates on stimuli whose level was measured.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from vic.evaluation.psychometric import (
    Scale,
    bootstrap_scale,
    build_design,
    fit_scale,
    probit_fit,
)


def simulate(
    perceived: dict[str, float], anchors: np.ndarray, noise_db: float,
    repeats: int = 400, listeners: int = 20, seed: int = 0,
) -> pd.DataFrame:
    """Forced choices for stimuli at known perceived levels against known anchors."""
    rng = np.random.default_rng(seed)
    rows = []
    for name, level in perceived.items():
        for anchor in anchors:
            probability = stats.norm.cdf((level - anchor) / noise_db)
            for index in range(repeats):
                rows.append({
                    "stimulus": name,
                    "anchor_db": float(anchor),
                    "louder": bool(rng.random() < probability),
                    "listener": f"L{index % listeners}",
                })
    return pd.DataFrame(rows)


def test_probit_fit_recovers_coefficients():
    rng = np.random.default_rng(0)
    design = np.column_stack([np.ones(4000), rng.normal(size=4000)])
    truth = np.array([0.4, 1.3])
    response = rng.random(4000) < stats.norm.cdf(design @ truth)
    assert probit_fit(design, response) == pytest.approx(truth, abs=0.12)


def test_probit_fit_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="do not line up"):
        probit_fit(np.ones((5, 2)), np.ones(4, dtype=bool))


def test_scale_recovers_perceived_levels():
    """The point of the whole module: known levels in, the same levels out."""
    truth = {"soft": 50.0, "mid": 60.0, "loud": 70.0}
    trials = simulate(truth, np.arange(45.0, 76.0, 5.0), noise_db=4.0)
    fit = fit_scale(trials, response="louder", anchor_db="anchor_db", by="stimulus")
    for name, level in truth.items():
        assert fit.perceived({f"stimulus={name}": 1.0}) == pytest.approx(level, abs=0.6)


def test_scale_recovers_perceptual_noise():
    trials = simulate({"a": 60.0}, np.arange(50.0, 71.0, 2.5), noise_db=3.0)
    fit = fit_scale(trials, response="louder", anchor_db="anchor_db", by="stimulus")
    assert fit.noise_db == pytest.approx(3.0, rel=0.15)
    # The 75% point of a normal with that spread, which is what a JND means.
    assert fit.jnd_db == pytest.approx(stats.norm.ppf(0.75) * 3.0, rel=0.15)


def test_scale_recovers_a_linear_slope():
    """A covariate slope is 'perceived decibels per requested decibel'."""
    rng = np.random.default_rng(1)
    requested = np.repeat(np.arange(45.0, 76.0, 4.3), 900)
    anchors = rng.uniform(46.0, 78.0, size=len(requested))
    # Perceived level follows the request at 0.6 dB per dB, offset by 20 dB.
    perceived = 20.0 + 0.6 * requested
    trials = pd.DataFrame({
        "requested_db": requested,
        "anchor_db": anchors,
        "louder": rng.random(len(requested)) < stats.norm.cdf(
            (perceived - anchors) / 4.0),
        "listener": [f"L{i % 25}" for i in range(len(requested))],
    })
    fit = fit_scale(trials, response="louder", anchor_db="anchor_db",
                    covariates=("requested_db",))
    slope = fit.coefficients[fit.names.index("requested_db")] / fit.beta
    assert slope == pytest.approx(0.6, abs=0.06)
    assert fit.perceived({"intercept": 1.0, "requested_db": 60.0}) == pytest.approx(
        20.0 + 0.6 * 60.0, abs=1.0)


def test_build_design_places_the_anchor_last_and_labels_columns():
    trials = pd.DataFrame({"anchor_db": [1.0, 2.0], "x": [3.0, 4.0],
                           "g": ["a", "b"]})
    design, names, anchor = build_design(trials, "anchor_db", by="g",
                                         covariates=("x",))
    assert names == ("g=a", "g=b", "x", "anchor_db")
    assert anchor == 3
    assert design.shape == (2, 4)
    # Dummy coding, so each row flags exactly one group.
    assert design[:, :2].sum(axis=1).tolist() == [1.0, 1.0]


def test_perceived_ignores_the_anchor_column():
    scale = Scale(coefficients=np.array([12.0, -0.2]), names=("intercept", "l"),
                  anchor=1)
    assert scale.beta == pytest.approx(0.2)
    # 12 / 0.2 = 60 dB, and the anchor's own coefficient must not enter.
    assert scale.perceived({"intercept": 1.0, "l": 99.0}) == pytest.approx(60.0)


def test_bootstrap_interval_covers_the_truth_and_reports_its_draws():
    trials = simulate({"a": 62.0}, np.arange(50.0, 76.0, 5.0), noise_db=4.0,
                      repeats=120, listeners=15, seed=3)
    table = bootstrap_scale(
        trials, response="louder", anchor_db="anchor_db", cluster="listener",
        by="stimulus", n_boot=120, seed=0,
        quantities={"perceived": lambda f: f.perceived({"stimulus=a": 1.0})},
    )
    row = table.iloc[0]
    assert row["estimate"] == pytest.approx(62.0, abs=1.0)
    assert row["ci_low"] < 62.0 < row["ci_high"]
    assert row["n_boot"] + row["n_failed"] == 120

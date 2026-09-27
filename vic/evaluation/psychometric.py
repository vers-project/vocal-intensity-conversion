"""Turning forced-choice answers into a perceived level in decibels.

A listening test gives one bit per trial.  A *scale* comes out of those bits
because each stimulus under test was compared against reference recordings whose
level was measured: the references are a ruler, and the level at which a listener
would be at 50/50 is the stimulus' perceived level on that ruler.

The model
---------
With $\\mu$ the perceived level of the stimulus under test and $\\ell$ the
measured level of the reference it was compared against,

.. math::  P(\\text{test judged louder}) = \\Phi(\\beta (\\mu - \\ell))

$\\beta$ is precision and $1/\\beta$ is the perceptual noise in decibels.  The
useful part is that this is *linear inside* $\\Phi$, so letting $\\mu$ depend on
whatever the experiment manipulated turns it into an ordinary probit regression:

.. math::  P = \\Phi(\\alpha_0 + \\alpha_1 \\tau + \\alpha_2 \\ell)
           \\quad\\Rightarrow\\quad
           \\beta = -\\alpha_2, \\; \\mu(\\tau) = -(\\alpha_0 + \\alpha_1\\tau)/\\alpha_2

so the perceived level is a ratio of fitted coefficients, and it is in decibels
because $\\ell$ is.  Nothing here is specific to loudness: any forced choice
against a measured reference has this shape.

Why the fit is written out rather than imported
-----------------------------------------------
statsmodels is not a dependency of this project, and the only thing it would add
is analytic standard errors -- which are not used, because a listener contributes
many trials and everyone in a group sees the same pairs, so intervals come from
resampling **listeners**, not trials.  The likelihood is concave, so a plain
quasi-Newton solve is enough.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import optimize, stats


def probit_fit(design: np.ndarray, response: np.ndarray) -> np.ndarray:
    """Maximum-likelihood probit coefficients for ``design`` against ``response``.

    ``response`` is boolean.  The log-likelihood is concave in the coefficients,
    so any descent method lands on the global optimum; the gradient is supplied
    because the finite-difference one is slow enough to matter inside a bootstrap.
    """
    design = np.asarray(design, dtype=float)
    y = np.asarray(response, dtype=float)
    if design.ndim != 2 or len(design) != len(y):
        raise ValueError(
            f"design {design.shape} and response {y.shape} do not line up."
        )
    sign = 2.0 * y - 1.0  # +1 where the test member was chosen, -1 otherwise

    def objective(beta):
        z = sign * (design @ beta)
        # logcdf rather than log(cdf) so a confidently-classified trial does not
        # underflow to -inf and take the whole fit with it.
        log_likelihood = stats.norm.logcdf(z).sum()
        ratio = np.exp(stats.norm.logpdf(z) - stats.norm.logcdf(z))
        gradient = design.T @ (sign * ratio)
        return -log_likelihood, -gradient

    start = np.zeros(design.shape[1])
    result = optimize.minimize(objective, start, jac=True, method="L-BFGS-B")
    if not result.success and not np.all(np.isfinite(result.x)):
        raise RuntimeError(f"the probit fit did not converge: {result.message}")
    return result.x


@dataclass(frozen=True)
class Scale:
    """A fitted perceived-level scale.

    ``names`` labels the columns of the design; ``anchor`` is the position of the
    reference-level column, whose coefficient carries the decibel units.
    """

    coefficients: np.ndarray
    names: tuple[str, ...]
    anchor: int

    @property
    def beta(self) -> float:
        """Precision: how sharply judgements flip per decibel of difference."""
        return float(-self.coefficients[self.anchor])

    @property
    def noise_db(self) -> float:
        """Perceptual noise, in decibels. Large means judgements are inconsistent."""
        return float(1.0 / self.beta) if self.beta > 0 else float("inf")

    @property
    def jnd_db(self) -> float:
        """The difference giving 75% correct -- the conventional just-noticeable one."""
        return float(stats.norm.ppf(0.75) * self.noise_db)

    def perceived(self, terms: dict[str, float]) -> float:
        """Perceived level in dB for a stimulus described by ``terms``.

        ``terms`` maps design column names to their values for that stimulus; any
        column not named is taken as zero, which is what makes dummy-coded
        designs read naturally (pass only the dummy that is on).
        """
        total = 0.0
        for name, coefficient in zip(self.names, self.coefficients):
            if name == self.names[self.anchor]:
                continue
            total += coefficient * float(terms.get(name, 0.0))
        return float(total / self.beta)


def build_design(
    trials: pd.DataFrame, anchor_db: str, by: str | None = None,
    covariates: tuple[str, ...] = (), intercept: bool = True,
) -> tuple[np.ndarray, tuple[str, ...], int]:
    """Design matrix for a perceived-level fit.

    ``by`` dummy-codes a factor -- one perceived level per level of it, with no
    shape assumed across them.  ``covariates`` enter linearly, which is how a
    slope like "perceived decibels per requested decibel" is obtained.  The
    reference level ``anchor_db`` always enters, and its coefficient is what puts
    everything else in decibels.
    """
    columns: list[np.ndarray] = []
    names: list[str] = []
    if by is not None:
        for value in sorted(trials[by].dropna().unique()):
            columns.append((trials[by] == value).to_numpy(float))
            names.append(f"{by}={value}")
    elif intercept:
        columns.append(np.ones(len(trials)))
        names.append("intercept")
    if by is not None and intercept and not covariates:
        pass  # dummies already span the intercept; adding one would be collinear
    for name in covariates:
        columns.append(trials[name].to_numpy(float))
        names.append(name)
    anchor = len(columns)
    columns.append(trials[anchor_db].to_numpy(float))
    names.append(anchor_db)
    return np.column_stack(columns), tuple(names), anchor


def fit_scale(
    trials: pd.DataFrame, response: str, anchor_db: str, by: str | None = None,
    covariates: tuple[str, ...] = (), intercept: bool = True,
) -> Scale:
    """Fit the perceived-level scale described by ``by`` and ``covariates``."""
    design, names, anchor = build_design(
        trials, anchor_db, by=by, covariates=covariates, intercept=intercept)
    coefficients = probit_fit(design, trials[response].to_numpy(bool))
    return Scale(coefficients=coefficients, names=names, anchor=anchor)


def bootstrap_scale(
    trials: pd.DataFrame, response: str, anchor_db: str, cluster: str,
    quantities, by: str | None = None, covariates: tuple[str, ...] = (),
    intercept: bool = True, n_boot: int = 2000, seed: int = 0,
) -> pd.DataFrame:
    """Percentile intervals for any quantity of the fit, resampling ``cluster``.

    ``quantities`` maps a name to a function of the fitted :class:`Scale`.  The
    clusters are listeners: one contributes many trials and everyone in a group
    sees the same pairs, so resampling trials would give intervals several times
    too narrow.

    Resamples that fail to converge -- possible when a bootstrap draw happens to
    separate perfectly -- are dropped, and the count is reported so a fit that is
    mostly failures cannot be mistaken for a tight one.
    """
    point = fit_scale(trials, response, anchor_db, by=by, covariates=covariates,
                      intercept=intercept)
    groups = {name: part.index.to_numpy()
              for name, part in trials.groupby(cluster)}
    keys = list(groups)
    rng = np.random.default_rng(seed)

    draws: dict[str, list[float]] = {name: [] for name in quantities}
    failures = 0
    for _ in range(n_boot):
        picked = rng.choice(len(keys), size=len(keys), replace=True)
        rows = np.concatenate([groups[keys[i]] for i in picked])
        try:
            fit = fit_scale(trials.loc[rows], response, anchor_db, by=by,
                            covariates=covariates, intercept=intercept)
        except (RuntimeError, ValueError):
            failures += 1
            continue
        if fit.beta <= 0:
            # A non-positive slope means "louder references were chosen more
            # often", which inverts the ruler; such a draw cannot be placed on
            # the scale and is dropped rather than folded in as a huge value.
            failures += 1
            continue
        for name, function in quantities.items():
            draws[name].append(float(function(fit)))

    rows = []
    for name, function in quantities.items():
        values = np.asarray(draws[name], dtype=float)
        rows.append({
            "quantity": name,
            "estimate": float(function(point)),
            "ci_low": float(np.percentile(values, 2.5)) if len(values) else np.nan,
            "ci_high": float(np.percentile(values, 97.5)) if len(values) else np.nan,
            "n_boot": int(len(values)),
            "n_failed": int(failures),
        })
    return pd.DataFrame(rows)

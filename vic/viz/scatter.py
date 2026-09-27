"""Scatter-plot utilities for intensity prediction evaluation."""
from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure


def intensity_scatter(
    labels: np.ndarray,
    preds: np.ndarray,
    title: str | None = None,
    alpha: float = 1.0,
) -> Figure:
    """Prediction-vs-label scatter plot with identity line.

    Both axes share the same limits (derived from the joint range of labels
    and predictions) and the axes box is square, so the identity line always
    appears at exactly 45°.

    Uses the non-interactive ``matplotlib.figure.Figure`` API — no pyplot
    state is touched, making this safe to call from training callbacks.

    Parameters
    ----------
    labels : (N,) ground-truth intensity values in dB SPL.
    preds  : (N,) predicted intensity values in dB SPL.
    title  : optional figure title.
    alpha  : scatter point opacity.  Use a low value (e.g. 0.1) for
             frame-level plots where many overlapping points are drawn;
             use 1.0 for sequence-level plots where each point is distinct.

    Returns
    -------
    Square ``matplotlib.figure.Figure``; call ``fig.savefig(path)`` to persist.
    """
    lo = min(labels.min(), preds.min())
    hi = max(labels.max(), preds.max())
    margin = (hi - lo) * 0.05
    lim = (lo - margin, hi + margin)

    fig = Figure(figsize=(6, 6))
    ax = fig.add_subplot(1, 1, 1)

    ax.scatter(labels, preds, s=1, alpha=alpha, linewidths=0)
    ax.plot(lim, lim, color="tab:red", linewidth=1.2, linestyle="--", label="Identity (y = x)")

    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal")

    ticks = np.arange(np.floor(lim[0] / 5) * 5, np.ceil(lim[1] / 5) * 5 + 1, 5)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    ax.grid(visible=True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax.set_xlabel("Label (dB SPL)")
    ax.set_ylabel("Prediction (dB SPL)")
    if title is not None:
        ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)

    fig.tight_layout()
    return fig

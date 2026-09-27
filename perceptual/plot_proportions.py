"""The same result with no model at all: the proportion judged louder.

    uv run --extra cpu python perceptual/plot_proportions.py \\
        outputs/perceptual

Every requested level was compared against **the same 32 real recordings**, so
the eight proportions are directly comparable with no adjustment, no covariate
and no fitted curve.  That is the whole method: count, and put a binomial-style
interval on it.

What it cannot do is produce decibels, and therefore neither the slope nor the
real-speech ceiling -- converting a proportion into a level is exactly the step
the probit in ``scale_converted.py`` performs.  This script exists so the two can
be compared and the cost of dropping the model can be seen rather than argued
about.

One asymmetry to keep in mind on the right-hand panel: within a selection the
references are that selection's own four recordings, so a curve sitting high can
mean loud conversions *or* quiet references.  The pooled panel does not have that
problem, because there every level meets the identical reference set.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from perceptual.scale_converted import SPAN, source_classes


def proportions(trials: pd.DataFrame, by: list[str], n_boot: int = 2000,
                seed: int = 0) -> pd.DataFrame:
    """Share of trials where the conversion was judged louder, by ``by``.

    The interval resamples **listeners**, matching every other interval in this
    experiment: a listener contributes 64 trials and everyone in a group sees the
    same pairs, so a trial-level interval would be several times too narrow.
    """
    from speech_eval.compare.core import cluster_bootstrap_ci

    rows = []
    for name, part in trials.groupby(by):
        values = part["chose_a"].astype(float).to_numpy()
        interval = cluster_bootstrap_ci(
            values, part["listener"].to_numpy(), statistic=np.mean,
            n_boot=n_boot, seed=seed)
        row = dict(zip(by, name if isinstance(name, tuple) else (name,)))
        row.update({
            "share": float(values.mean()),
            "ci_low": float(interval["ci_low"]),
            "ci_high": float(interval["ci_high"]),
            "n_trials": int(len(values)),
            "n_listeners": int(part["listener"].nunique()),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def reference_curves(trials: pd.DataFrame, noise_db: float, ceiling: tuple[float, float],
                     grid: np.ndarray) -> pd.DataFrame:
    """What the proportion curve *should* look like, and at best could look like.

    This is the piece the raw proportions cannot supply on their own.  Whether a
    conversion is judged louder depends on which reference it met, so the shape
    of the pooled curve is set by the **distribution of the 32 reference
    levels** -- and without drawing that, there is no way to tell a good curve
    from a bad one.

    Three curves, from strictest to fairest:

    ``perfect``
        a converter that hits the requested level exactly, judged by listeners
        who never err: the share of references quieter than the request. Needs
        no model, but it is a step function no human could produce.
    ``attainable``
        the same converter judged by **these** listeners, whose comparisons
        carry ``noise_db`` of dispersion. This is the curve to read against.
    ``real_speech``
        what a real recording put through the same comparison achieves, using
        the measured real-speech line. The empirical ceiling.

    The references are weighted by how many trials each actually contributed,
    because the four groups differ in size and their pair sets are disjoint --
    an unweighted average would describe a design nobody ran.
    """
    from scipy import stats

    weights = trials.groupby("stimulus_b").agg(
        level=("b_level_db", "first"), n=("chose_a", "size"))
    level = weights["level"].to_numpy(float)
    weight = weights["n"].to_numpy(float)
    weight = weight / weight.sum()
    offset, slope = ceiling

    rows = []
    for requested in grid:
        rows.append({
            "requested_db": float(requested),
            "perfect": float(weight @ (level < requested)),
            "attainable": float(
                weight @ stats.norm.cdf((requested - level) / noise_db)),
            "real_speech": float(
                weight @ stats.norm.cdf(
                    (offset + slope * requested - level) / noise_db)),
        })
    return pd.DataFrame(rows)


def draw(pooled: pd.DataFrame, banded: pd.DataFrame, path: Path,
         curves: pd.DataFrame | None = None) -> None:
    from matplotlib.colors import LinearSegmentedColormap

    from speech_eval import figures

    plt = figures._plt()
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 5.0), layout="constrained")

    for ax in axes:
        figures.style_axes(ax)
        # Half the time is parity with the reference set, not chance: it is the
        # level at which a conversion is as often judged louder as not.
        ax.axhline(0.5, linestyle=(0, (4, 3)), color=figures.AXIS, linewidth=1.0,
                   zorder=1)
        ax.set_xlim(*SPAN)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("requested level (dB SPL)", color=figures.INK_PRIMARY,
                      fontsize=10)

    if curves is not None:
        axes[0].plot(curves["requested_db"], curves["perfect"],
                     linestyle=(0, (1, 2)), color=figures.INK_MUTED,
                     linewidth=1.2, zorder=2,
                     label="exact converter, perfect listeners")
        axes[0].plot(curves["requested_db"], curves["real_speech"],
                     color=figures.SERIES[1], linewidth=1.7, zorder=2,
                     label="real speech (ceiling)")
        axes[0].plot(curves["requested_db"], curves["attainable"],
                     linestyle=(0, (5, 2)), color=figures.SERIES[2],
                     linewidth=1.7, zorder=2,
                     label="exact converter, these listeners")

    ordered = pooled.sort_values("requested_db")
    axes[0].errorbar(
        ordered["requested_db"], ordered["share"],
        yerr=[ordered["share"] - ordered["ci_low"],
              ordered["ci_high"] - ordered["share"]],
        fmt="o-", color=figures.SERIES[0], markersize=5.5, capsize=2.5,
        elinewidth=1.2, linewidth=2.0, markeredgecolor=figures.SURFACE,
        markeredgewidth=0.8, zorder=4, label="measured")
    axes[0].set_ylabel("judged louder than the real recording",
                       color=figures.INK_PRIMARY, fontsize=10)
    axes[0].set_title("Pooled over the corpus", color=figures.INK_PRIMARY,
                      fontsize=10.5, pad=8)
    if curves is not None:
        first = axes[0].legend(loc="upper left", frameon=False, fontsize=7.6,
                               handlelength=2.0, borderaxespad=0.4)
        for text in first.get_texts():
            text.set_color(figures.INK_PRIMARY)

    ramp = LinearSegmentedColormap.from_list(
        "source_level", ["#9dc2ea", figures.SERIES[0], "#123f73"])
    bands = sorted(banded["source_band"].unique())
    for index, band in enumerate(bands):
        part = banded[banded["source_band"] == band].sort_values("requested_db")
        axes[1].plot(part["requested_db"], part["share"],
                     color=ramp(index / max(len(bands) - 1, 1)), linewidth=2.0,
                     marker="o", markersize=4.0, markeredgecolor=figures.SURFACE,
                     markeredgewidth=0.6, zorder=3,
                     label=part["source_band_label"].iloc[0])
    axes[1].set_title("Grouped by source level", color=figures.INK_PRIMARY,
                      fontsize=10.5, pad=8)
    legend = axes[1].legend(loc="upper left", frameon=False, fontsize=8.0,
                            handlelength=1.8, borderaxespad=0.4,
                            title="source level")
    legend.get_title().set_color(figures.INK_PRIMARY)
    legend.get_title().set_fontsize(8.5)
    for text in legend.get_texts():
        text.set_color(figures.INK_PRIMARY)

    figure.suptitle("How often a conversion was judged louder than a real recording",
                    color=figures.INK_PRIMARY, fontsize=12)
    figures.save(figure, path)


def main(output_dir: Path) -> None:
    trials = pd.read_csv(output_dir / "converted_trials.csv")
    trials = trials[trials["answered"]].copy()
    trials["chose_a"] = trials["picked_member"] == "a"
    trials["selection"] = (trials["a_speaker"].astype(str) + "_"
                           + trials["a_sentence_id"].astype(str))

    reference_sets = trials.groupby("target_index")["stimulus_b"].apply(
        lambda s: tuple(sorted(set(s))))
    if len(set(reference_sets)) != 1:
        raise ValueError(
            "the requested levels did not all meet the same reference "
            "recordings, so their raw proportions are not comparable and this "
            "model-free reading is invalid; use scale_converted.py instead.")
    print(f"every requested level met the same "
          f"{len(reference_sets.iloc[0])} reference recordings, so the "
          f"proportions are directly comparable")

    pooled = proportions(trials, ["target_index", "a_level_db"]).rename(
        columns={"a_level_db": "requested_db"})
    pooled.to_csv(output_dir / "share_louder_by_target.csv", index=False)
    print("\n  requested   judged louder        n")
    for r in pooled.sort_values("requested_db").itertuples():
        print(f"  {r.requested_db:9.1f}   {r.share:.3f} "
              f"[{r.ci_low:.3f}, {r.ci_high:.3f}]  {r.n_trials:4d}")

    banded = source_classes(
        trials.assign(source_level_db=trials["source_level_db"]), 4)
    banded = proportions(
        banded, ["source_band", "source_band_label", "a_level_db"]).rename(
        columns={"a_level_db": "requested_db"})
    banded.to_csv(output_dir / "share_louder_by_band.csv", index=False)

    # The reference curves need two numbers the proportions cannot supply:
    # the listeners' comparison noise and the real-speech line, both from the
    # ruler fitted in scale_converted.py. Without them the panel has no scale
    # against which "good" means anything.
    ruler = pd.read_csv(output_dir / "ruler.csv").set_index("quantity")["estimate"]
    noise_db = float(ruler[[q for q in ruler.index if "noise" in q][0]])
    ceiling = (float(ruler[[q for q in ruler.index if q.startswith("offset")][0]]),
               float(ruler[[q for q in ruler.index if q.startswith("slope")][0]]))
    curves = reference_curves(trials, noise_db, ceiling,
                              np.linspace(SPAN[0], SPAN[1], 200))
    curves.to_csv(output_dir / "share_louder_reference_curves.csv", index=False)
    print(f"\n  reference curves use listener noise {noise_db:.2f} dB and the "
          f"real-speech line {ceiling[1]:.3f}x{ceiling[0]:+.2f}")
    at = curves.set_index(curves["requested_db"].round(0))
    for requested in (45.0, 60.0, 75.0):
        row = curves.iloc[(curves["requested_db"] - requested).abs().idxmin()]
        print(f"   at {requested:.0f} dB requested: exact+perfect "
              f"{row['perfect']:.2f}, exact+these listeners "
              f"{row['attainable']:.2f}, real speech {row['real_speech']:.2f}")

    path = output_dir / "figures" / "share_louder_two_panel.png"
    draw(pooled, banded, path, curves)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "."))

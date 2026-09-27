"""Step 2: put the converted stimuli on a perceived-decibel scale.

    uv run --extra cpu python perceptual/scale_converted.py \\
        --config configs/paper/analyze_perceptual.yaml \\
        --output-dir outputs/perceptual

Each converted stimulus was compared against
real recordings whose level we measured, so those recordings are a ruler, and the
level at which a listener would be at 50/50 is the converted stimulus' perceived
level in dB SPL.

This file currently covers the go/no-go of that plan: loading the converted
trials for the screened listeners, and the **coverage check** that says whether a
perceived level can be estimated at all. A crossing point that is not bracketed
by the real levels a stimulus was actually paired with is an extrapolation, and
must be reported as a bound rather than a point.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from vic.evaluation.perceptual import TargetGrid, describe_trials, read_experiment, score
from vic.evaluation.psychometric import Scale, bootstrap_scale, fit_scale
from vic.evaluation.stimulus_levels import attach_measured_levels

from experiment_launcher import parse_args


def load_converted(config: dict, kept: set[str] | None) -> pd.DataFrame:
    """The converted-versus-real trials, scored, for the listeners we keep.

    The converted member's reference is the level it was **asked** for -- that is
    the control interface under test.  The real member's is measured.  Both the
    real anchor and the converted stimulus' own source recording are measured
    here, the latter because the requested *change* is what the direction
    asymmetry is expressed in.
    """
    trials = read_experiment(Path(config["experiment_dir"]))
    trials = describe_trials(trials, TargetGrid(**config["targets"]))
    total = trials["listener"].nunique()
    if kept is not None:
        trials = trials[trials["listener"].isin(kept)].reset_index(drop=True)
    print(f"{trials['listener'].nunique()} of {total} listener(s) kept, "
          f"{len(trials)} trial(s), "
          f"groups {trials.groupby('group')['listener'].nunique().to_dict()}")

    converted = trials[trials["synthetic_trial"]].reset_index(drop=True)
    converted = attach_measured_levels(converted, "b", "b_level_db", config)
    # The converted stimulus' own source recording is a real AVID utterance, so
    # it is measured the same way; `a_*` identifies it because a conversion keeps
    # its source's key.
    converted = attach_measured_levels(converted, "a", "source_level_db", config)
    converted["requested_change_db"] = (
        converted["a_level_db"] - converted["source_level_db"])
    converted["selection"] = (converted["a_speaker"].astype(str) + "_"
                              + converted["a_sentence_id"].astype(str))
    scored = score(converted)
    usable = scored[scored["correct"].notna()]
    print(f"  {len(scored)} converted trial(s), {len(usable)} scored "
          f"({int((~scored['answered']).sum())} unanswered)")
    print(f"  real anchors {scored['b_level_db'].min():.1f}-"
          f"{scored['b_level_db'].max():.1f} dB, "
          f"sources {scored['source_level_db'].min():.1f}-"
          f"{scored['source_level_db'].max():.1f} dB")
    return scored


def coverage(trials: pd.DataFrame) -> pd.DataFrame:
    """Can a perceived level be estimated for each requested target?

    A crossing point is only measured if the listeners were on both sides of it:
    some anchors judged quieter than the conversion, some louder.  If even the
    quietest anchor is judged louder than the conversion, the perceived level
    sits below everything the design offered and can only be bounded.
    """
    usable = trials[trials["correct"].notna()].copy()
    usable["chose_converted"] = usable["picked_member"] == "a"
    rows = []
    for (index, level), part in usable.groupby(["target_index", "a_level_db"]):
        # The proportion judged louder against the softest and the loudest
        # anchors available for this target: the two ends that bracket or fail to.
        by_anchor = part.groupby("stimulus_b").agg(
            anchor_db=("b_level_db", "first"),
            share=("chose_converted", "mean"),
            n=("chose_converted", "size"),
        ).sort_values("anchor_db")
        low = by_anchor.iloc[: max(len(by_anchor) // 3, 1)]
        high = by_anchor.iloc[-max(len(by_anchor) // 3, 1):]
        crosses = (by_anchor["share"] > 0.5).any() and (by_anchor["share"] < 0.5).any()
        rows.append({
            "target_index": int(index),
            "requested_db": float(level),
            "n_trials": int(len(part)),
            "n_anchors": int(len(by_anchor)),
            "anchor_min_db": float(by_anchor["anchor_db"].min()),
            "anchor_max_db": float(by_anchor["anchor_db"].max()),
            "share_louder": float(part["chose_converted"].mean()),
            "share_vs_softest": float(low["share"].mean()),
            "share_vs_loudest": float(high["share"].mean()),
            "brackets": bool(crosses),
        })
    return pd.DataFrame(rows).sort_values("target_index").reset_index(drop=True)


def build_ruler(control: pd.DataFrame, n_boot: int, seed: int) -> tuple[Scale, pd.DataFrame]:
    """Fit the scale on pairs where BOTH levels are known, and check it.

    This is the gate.  For real-versus-real trials the answer is already known --
    the perceived level of a recording should come back as its measured level --
    so the fit must return a slope of 1 and an offset of 0.  A slope far from 1
    would mean the ruler is stretched and every converted decibel below is wrong
    by that factor.
    """
    fit = fit_scale(control, response="chose_a", anchor_db="b_level_db",
                    covariates=("a_level_db",))
    slope = fit.coefficients[fit.names.index("a_level_db")] / fit.beta
    offset = fit.coefficients[fit.names.index("intercept")] / fit.beta
    table = bootstrap_scale(
        control, response="chose_a", anchor_db="b_level_db", cluster="listener",
        covariates=("a_level_db",), n_boot=n_boot, seed=seed,
        quantities={
            "slope (should be 1)":
                lambda f: f.coefficients[f.names.index("a_level_db")] / f.beta,
            "offset dB (should be 0)":
                lambda f: f.coefficients[f.names.index("intercept")] / f.beta,
            "perceptual noise dB": lambda f: f.noise_db,
            "JND dB": lambda f: f.jnd_db,
        })
    print("\n=== the ruler: fitted on natural pairs, where both levels are known ===")
    print(f"  {len(control)} trial(s), {control['listener'].nunique()} listener(s)")
    for r in table.itertuples():
        print(f"  {r.quantity:26s} {r.estimate:+8.3f}  [{r.ci_low:+.3f}, {r.ci_high:+.3f}]")
    row = table[table["quantity"].str.startswith("slope")].iloc[0]
    covers_one = row["ci_low"] <= 1.0 <= row["ci_high"]
    print(f"  perceived level of a real recording measured at "
          f"45 / 60 / 75 dB: "
          + " / ".join(f"{offset + slope * x:.1f}" for x in (45.0, 60.0, 75.0)) + " dB")
    print("  -> " + ("the ruler is unstretched (the interval covers 1)" if covers_one
                     else f"the ruler is compressed: a 1 dB change in a real recording "
                          f"moves perception by {slope:.2f} dB. Converted estimates "
                          f"below are on this same ruler, so they stay comparable to "
                          f"the real recordings -- but perceived intensity is not "
                          f"exactly measured L_eq."))
    return (offset, slope), table


def recover_known_levels(control: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    """Estimate each real recording's level from the votes, and compare.

    The same estimator that will be applied to the conversions, applied to
    stimuli whose answer is on file.  If these do not come back close, no
    converted number below is worth reporting.
    """
    fit = fit_scale(control, response="chose_a", anchor_db="b_level_db",
                    by="stimulus_a")
    names = [n for n in fit.names if n.startswith("stimulus_a=")]
    table = bootstrap_scale(
        control, response="chose_a", anchor_db="b_level_db", cluster="listener",
        by="stimulus_a", n_boot=n_boot, seed=seed,
        quantities={n: (lambda f, n=n: f.perceived({n: 1.0})) for n in names})
    measured = control.groupby("stimulus_a")["a_level_db"].first()
    counts = control.groupby("stimulus_a").size()
    # A stimulus louder (or quieter) than every anchor it met has no crossing
    # point in range: its estimate is a bound, and its interval will run away.
    spans = control.groupby("stimulus_a").apply(
        lambda d: bool((d["b_level_db"] > d["a_level_db"].iloc[0]).any()
                       and (d["b_level_db"] < d["a_level_db"].iloc[0]).any()),
        include_groups=False)
    table["stimulus"] = [n.split("=", 1)[1] for n in table["quantity"]]
    table["measured_db"] = table["stimulus"].map(measured)
    table["n_trials"] = table["stimulus"].map(counts)
    table["bracketed"] = table["stimulus"].map(spans)
    table["error_db"] = table["estimate"] - table["measured_db"]
    table["covers"] = ((table["ci_low"] <= table["measured_db"])
                       & (table["measured_db"] <= table["ci_high"]))
    return table.drop(columns=["quantity"])


def converted_scale(
    trials: pd.DataFrame, n_boot: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Perceived level of the conversions: free per target, then linear."""
    free = fit_scale(trials, response="chose_a", anchor_db="b_level_db",
                     by="target_index")
    names = [n for n in free.names if n.startswith("target_index=")]
    per_target = bootstrap_scale(
        trials, response="chose_a", anchor_db="b_level_db", cluster="listener",
        by="target_index", n_boot=n_boot, seed=seed,
        quantities={n: (lambda f, n=n: f.perceived({n: 1.0})) for n in names})
    asked = trials.groupby("target_index")["a_level_db"].first()
    per_target["target_index"] = [int(float(n.split("=", 1)[1]))
                                  for n in per_target["quantity"]]
    per_target["requested_db"] = per_target["target_index"].map(asked)
    per_target["shortfall_db"] = per_target["estimate"] - per_target["requested_db"]

    linear = bootstrap_scale(
        trials, response="chose_a", anchor_db="b_level_db", cluster="listener",
        covariates=("a_level_db",), n_boot=n_boot, seed=seed,
        quantities={
            "slope (perceived dB per requested dB)":
                lambda f: f.coefficients[f.names.index("a_level_db")] / f.beta,
            "offset dB":
                lambda f: f.coefficients[f.names.index("intercept")] / f.beta,
            "perceptual noise dB": lambda f: f.noise_db,
            "perceived at 45 dB requested":
                lambda f: f.perceived({"intercept": 1.0, "a_level_db": 45.0}),
            "perceived at 75 dB requested":
                lambda f: f.perceived({"intercept": 1.0, "a_level_db": 75.0}),
        })

    # Does where the source started matter as well as what was asked for?
    #
    # The target varies WITHIN every selection, so resampling listeners is the
    # right interval for its slope.  The source level does not: it takes one
    # value per selection, so a listener bootstrap holds all eight fixed and
    # reports an interval far too narrow for it.  Both clusterings are fitted
    # and the source verdict rests on the selection one.
    quantities = {
        "target slope":
            lambda f: f.coefficients[f.names.index("a_level_db")] / f.beta,
        "source slope (0 = source does not matter)":
            lambda f: f.coefficients[f.names.index("source_level_db")] / f.beta,
    }
    frames = []
    for cluster in ("listener", "selection"):
        table = bootstrap_scale(
            trials, response="chose_a", anchor_db="b_level_db", cluster=cluster,
            covariates=("a_level_db", "source_level_db"), n_boot=n_boot,
            seed=seed, quantities=quantities)
        table.insert(0, "resampled", cluster)
        frames.append(table)
    with_source = pd.concat(frames, ignore_index=True)
    return per_target, linear, with_source


def per_stimulus_levels(trials: pd.DataFrame) -> pd.DataFrame:
    """Perceived level of each individual conversion, one per rendered file.

    These are the grey points of the figure.  They carry no interval -- roughly
    35 responses each is enough to place a stimulus but not to bound it -- and
    their job is to show the scatter that a slope through the middle hides.

    A conversion outside the range of the four anchors it met has no crossing
    point in range; such a stimulus is flagged rather than plotted, because its
    estimate is the solver running off rather than a measurement.
    """
    fit = fit_scale(trials, response="chose_a", anchor_db="b_level_db",
                    by="stimulus_a")
    rows = []
    for name in fit.names:
        if not name.startswith("stimulus_a="):
            continue
        stem = name.split("=", 1)[1]
        part = trials[trials["stimulus_a"] == stem]
        perceived = fit.perceived({name: 1.0})
        anchors = part["b_level_db"]
        rows.append({
            "stimulus": stem,
            "selection": f"{part['a_speaker'].iloc[0]}_{part['a_sentence_id'].iloc[0]}",
            "target_index": int(part["target_index"].iloc[0]),
            "requested_db": float(part["a_level_db"].iloc[0]),
            "source_level_db": float(part["source_level_db"].iloc[0]),
            "perceived_db": float(perceived),
            "n_trials": int(len(part)),
            "bracketed": bool(anchors.min() <= perceived <= anchors.max()),
        })
    return pd.DataFrame(rows)


SPAN = np.array([43.0, 78.0])


def draw_scale_panel(
    ax, per_target: pd.DataFrame, per_stimulus: pd.DataFrame,
    converted_line: tuple[float, float], real_line: tuple[float, float],
    show_stimuli: bool = False, font_scale: float = 1.0,
) -> None:
    """What was asked for against what was heard, pooled over the corpus.

    Everything is on one pair of axes and in the same units, which is the whole
    point -- the distance from the diagonal is the result, and it is read
    directly rather than inferred from a number.  Real recordings get their own
    line because they are the ceiling: they are not on the diagonal either, so
    the converter must be judged against them and not against perfection.

    ``show_stimuli`` draws every individual conversion behind the fit.  It is
    off by default: the cloud is hard to read, and its content -- how far
    conversions sit from each other -- is carried far better by the banded panel
    beside it.  **When this panel is used alone, the spread is then not in the
    figure at all** and the surrounding text has to supply it, because the error
    bars are how well the average is known and say nothing about where an
    individual conversion lands.
    """
    from speech_eval import figures

    span = SPAN
    figures.style_axes(ax)

    ax.plot(span, span, linestyle=(0, (4, 3)), color=figures.AXIS, linewidth=1.0,
            zorder=1, label="requested = perceived")

    # When they are drawn, all 64 are, because the ones whose estimate falls
    # outside their own anchors are exactly the extremes -- dropping those would
    # shrink the visible spread and make the converter look more compressed than
    # it is.  They are marked as the weaker estimates they are rather than hidden.
    solid = per_stimulus[per_stimulus["bracketed"]]
    weak = per_stimulus[~per_stimulus["bracketed"]]
    inside = weak[weak["perceived_db"].between(*span)]
    beyond = weak[~weak["perceived_db"].between(*span)]

    if show_stimuli:
        ax.scatter(solid["requested_db"], solid["perceived_db"], s=18,
                   color=figures.INK_MUTED, alpha=0.38, linewidths=0, zorder=2,
                   label="individual conversions")
        ax.scatter(inside["requested_db"], inside["perceived_db"], s=20,
                   facecolors="none", edgecolors=figures.INK_MUTED, alpha=0.55,
                   linewidths=0.9, zorder=2,
                   label="estimate beyond its own anchors")
        # Off-scale ones sit at the edge as carets, so the reader sees that they are
        # off the chart rather than absent.
        for _, row in beyond.iterrows():
            low = row["perceived_db"] < span[0]
            ax.plot(row["requested_db"], span[0] + 0.6 if low else span[1] - 0.6,
                    marker="v" if low else "^", markersize=5,
                    markerfacecolor="none", markeredgecolor=figures.INK_MUTED,
                    markeredgewidth=0.9, alpha=0.7, zorder=2, linestyle="none")

    real_offset, real_slope = real_line
    ax.plot(span, real_offset + real_slope * span, color=figures.SERIES[1],
            linewidth=1.7, zorder=3,
            label="original recordings")

    offset, slope = converted_line
    ax.plot(span, offset + slope * span, color=figures.SERIES[0], linewidth=2.1,
            zorder=4, label="conversions")
    ordered = per_target.sort_values("requested_db")
    ax.errorbar(
        ordered["requested_db"], ordered["estimate"],
        yerr=[ordered["estimate"] - ordered["ci_low"],
              ordered["ci_high"] - ordered["estimate"]],
        fmt="o", color=figures.SERIES[0], markersize=5.5, capsize=2.5,
        elinewidth=1.2, markeredgecolor=figures.SURFACE, markeredgewidth=0.8,
        zorder=5, linestyle="none",
    )

    ax.set_xlim(*span)
    ax.set_ylim(*span)
    ax.set_aspect("equal")
    ax.set_xlabel("requested level (dB SPL)", color=figures.INK_PRIMARY,
                  fontsize=10 * font_scale)
    ax.set_ylabel("perceived level (dB SPL)", color=figures.INK_PRIMARY,
                  fontsize=10 * font_scale)
    ax.set_title("Pooled over the corpus", color=figures.INK_PRIMARY,
                 fontsize=10.5 * font_scale, pad=8)
    ax.tick_params(labelsize=9 * font_scale)
    legend = ax.legend(loc="upper left", frameon=False,
                       fontsize=8.0 * font_scale,
                       handlelength=1.8, borderaxespad=0.4)
    for text in legend.get_texts():
        text.set_color(figures.INK_PRIMARY)


def source_classes(per_stimulus: pd.DataFrame, n_classes: int = 4) -> pd.DataFrame:
    """Group the source recordings into ``n_classes`` bands of source level.

    Eight separate lines is more detail than the eye can hold, and the eight
    source levels are a sample of a continuum rather than eight meaningful
    categories -- so they are banded by level, which is the variable that orders
    them anyway.  Equal-count bands rather than equal-width ones, because the
    sources cluster and equal-width bands would leave some empty.
    """
    sources = per_stimulus.groupby("selection")["source_level_db"].first()
    bands = pd.qcut(sources, n_classes, labels=False, duplicates="drop")
    label = {}
    for band in sorted(set(bands)):
        members = sources[bands == band]
        label[band] = f"{members.min():.0f}–{members.max():.0f} dB"
    out = per_stimulus.copy()
    out["source_band"] = out["selection"].map(bands)
    out["source_band_label"] = out["source_band"].map(label)
    return out


def draw_selection_panel(ax, per_stimulus: pd.DataFrame, n_classes: int = 4,
                         hairlines: bool = False):
    """The same conversions, grouped by how loud their source recording was.

    The pooled panel puts a slope through a cloud 7 dB wide, which invites the
    reading that individual conversions are erratic.  They are not: within one
    source the eight requests come out in order, and almost all of the cloud is
    the *offset* between sources rather than disorder inside them.  Banding by
    source level is what makes that visible -- the request sets the change, the
    source sets the place.

    Each band gets a bold mean line.  ``hairlines`` puts the individual sources
    behind it; it is off by default because the panel is for a paper, where an
    extra series costs a paragraph of explanation and the banded means already
    carry the point.
    """
    from matplotlib.colors import LinearSegmentedColormap

    from speech_eval import figures

    span = SPAN
    figures.style_axes(ax)
    ax.plot(span, span, linestyle=(0, (4, 3)), color=figures.AXIS, linewidth=1.0,
            zorder=1)

    banded = source_classes(per_stimulus, n_classes)
    # A sequential ramp sampled once per band: the bands are ordered levels, not
    # unrelated categories, so they should read as light-to-dark and not as a
    # set of hues.
    ramp = LinearSegmentedColormap.from_list(
        "source_level", ["#9dc2ea", figures.SERIES[0], "#123f73"])
    bands = sorted(banded["source_band"].unique())
    colours = {band: ramp(index / max(len(bands) - 1, 1))
               for index, band in enumerate(bands)}

    for band in bands:
        part = banded[banded["source_band"] == band]
        colour = colours[band]
        if hairlines:
            for _, one in part.groupby("selection"):
                one = one.sort_values("requested_db")
                ax.plot(one["requested_db"], one["perceived_db"], color=colour,
                        linewidth=0.7, alpha=0.30, zorder=2)
        mean = part.groupby("requested_db")["perceived_db"].mean().sort_index()
        ax.plot(mean.index, mean.to_numpy(), color=colour, linewidth=2.0,
                marker="o", markersize=4.0, zorder=3,
                markeredgecolor=figures.SURFACE, markeredgewidth=0.6,
                label=part["source_band_label"].iloc[0])
        # Same convention as the pooled panel: a conversion off the axes is a
        # caret at the frame, never a silently clipped point.  Only meaningful
        # while the individual conversions are shown at all.
        if hairlines:
            for _, row in part[~part["perceived_db"].between(*span)].iterrows():
                low = row["perceived_db"] < span[0]
                ax.plot(row["requested_db"], span[0] + 0.6 if low else span[1] - 0.6,
                        marker="v" if low else "^", markersize=4.5, color=colour,
                        markerfacecolor="none", markeredgewidth=0.9, alpha=0.5,
                        zorder=4, linestyle="none")

    ax.set_xlim(*span)
    ax.set_ylim(*span)
    ax.set_aspect("equal")
    ax.set_xlabel("requested level (dB SPL)", color=figures.INK_PRIMARY, fontsize=10)
    ax.set_title("Grouped by source level", color=figures.INK_PRIMARY,
                 fontsize=10.5, pad=8)
    legend = ax.legend(loc="upper left", frameon=False, fontsize=8.0,
                       handlelength=1.8, borderaxespad=0.4,
                       title="source level")
    legend.get_title().set_color(figures.INK_PRIMARY)
    legend.get_title().set_fontsize(8.5)
    for text in legend.get_texts():
        text.set_color(figures.INK_PRIMARY)
    return ax


def make_figure(
    per_target: pd.DataFrame, per_stimulus: pd.DataFrame,
    converted_line: tuple[float, float], real_line: tuple[float, float],
    path: Path, two_panel_path: Path | None = None, titled: bool = True,
    small: bool = False,
) -> None:
    """Write the single pooled panel, and the two-panel version beside it.

    ``titled`` off leaves the titles out, for a figure whose caption carries them
    -- which is what a paper wants, and what a screen does not.

    ``small`` draws the pooled panel at roughly half the area, **keeping it
    square** so the diagonal stays at 45 degrees and a distance from it is to the
    same scale on both axes.  The lettering is scaled less than the axes, so a
    figure included at a fraction of the column width stays legible; place it at
    about 0.72 of the text width and accept white margins beside it.
    """
    from speech_eval import figures

    plt = figures._plt()

    figure, ax = plt.subplots(figsize=(2.6, 2.55) if small else (5.4, 5.0))
    draw_scale_panel(ax, per_target, per_stimulus, converted_line, real_line,
                     font_scale=0.80 if small else 1.0)
    ax.set_title("What listeners heard when a level was requested"
                 if titled else "", color=figures.INK_PRIMARY, fontsize=11, pad=10)
    figures.save(ax, path)

    if two_panel_path is None:
        return
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 5.0), layout="constrained")
    draw_scale_panel(axes[0], per_target, per_stimulus, converted_line, real_line)
    axes[0].set_ylabel("perceived level (dB SPL)", color=figures.INK_PRIMARY,
                       fontsize=10)
    draw_selection_panel(axes[1], per_stimulus)
    # constrained layout places the suptitle itself; pinning y puts it on top
    # of the two subplot titles.
    if titled:
        figure.suptitle("What listeners heard when a level was requested",
                        color=figures.INK_PRIMARY, fontsize=12)
    figures.save(figure, two_panel_path)


@parse_args
def main(config: dict, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    kept_path = output_dir / "kept_listeners.csv"
    if not kept_path.exists():
        raise FileNotFoundError(
            f"{kept_path} is missing; run perceptual/screen_control.py first -- "
            "this step is defined on the screened listeners."
        )
    kept = set(pd.read_csv(kept_path)["listener"])
    trials = load_converted(config, kept)
    trials.to_csv(output_dir / "converted_trials.csv", index=False)

    table = coverage(trials)
    table.to_csv(output_dir / "coverage.csv", index=False)
    print("\n=== coverage: is each target's crossing point bracketed? ===")
    print(f"  {'target':>6s} {'asked':>7s} {'trials':>7s} {'anchors span':>16s} "
          f"{'P louder':>9s} {'vs softest':>11s} {'vs loudest':>11s} {'bracketed':>10s}")
    for r in table.itertuples():
        print(f"  {r.target_index:6d} {r.requested_db:6.1f}  {r.n_trials:7d} "
              f"{r.anchor_min_db:7.1f}-{r.anchor_max_db:<7.1f} {r.share_louder:9.3f} "
              f"{r.share_vs_softest:11.3f} {r.share_vs_loudest:11.3f} "
              f"{'yes' if r.brackets else 'NO':>10s}")

    ok = table[table["brackets"]]
    print(f"\n  {len(ok)} of {len(table)} targets have a bracketed crossing point")
    if len(ok) < len(table):
        missing = table[~table["brackets"]]
        print("  not bracketed: " + ", ".join(
            f"target {int(r.target_index)} ({r.requested_db:.1f} dB, "
            f"P louder {r.share_louder:.2f})" for r in missing.itertuples()))
        print("  -> those get a bound, not a point estimate")

    n_boot = int(config.get("n_boot", 2000))
    seed = int(config.get("seed", 0))

    # ------------------------------------------------------------------
    # The ruler, and the validation that gates everything after it.
    # ------------------------------------------------------------------
    control = pd.read_csv(output_dir / "control_trials.csv")
    control = control[control["listener"].isin(kept) & control["answered"]].copy()
    control["chose_a"] = control["picked_member"] == "a"
    real_line, ruler = build_ruler(control, n_boot, seed)
    ruler.to_csv(output_dir / "ruler.csv", index=False)

    recovered = recover_known_levels(control, n_boot, seed)
    recovered.to_csv(output_dir / "recovered_levels.csv", index=False)
    print("\n=== validation: recovering levels we already know ===")
    print(f"  {'stimulus':32s} {'measured':>9s} {'estimated':>20s} {'error':>7s} {'n':>4s}")
    for r in recovered.sort_values("measured_db").itertuples():
        flag = ("" if r.bracketed else "  <- not bracketed, bound only"
                ) or ("" if r.covers else "  <- interval misses")
        print(f"  {r.stimulus:32s} {r.measured_db:9.1f} {r.estimate:9.1f} "
              f"[{r.ci_low:5.1f},{r.ci_high:5.1f}] {r.error_db:+7.1f} "
              f"{int(r.n_trials):4d}{flag}")
    usable_rows = recovered[recovered["bracketed"]]
    bias = usable_rows["error_db"].mean()
    spread = usable_rows["error_db"].abs().mean()
    rms = float((usable_rows["error_db"] ** 2).mean() ** 0.5)
    print(f"  over the {len(usable_rows)} bracketed stimuli: mean error "
          f"{bias:+.2f} dB, mean |error| {spread:.2f} dB, RMS {rms:.2f} dB")
    print(f"  {int(usable_rows['covers'].sum())}/{len(usable_rows)} intervals cover "
          f"the measured level")
    if spread > 5.0:
        print("  *** the estimator does not recover known levels; treat every "
              "converted number below as unreliable ***")

    # ------------------------------------------------------------------
    # The conversions.
    # ------------------------------------------------------------------
    usable = trials[trials["answered"]].copy()
    usable["chose_a"] = usable["picked_member"] == "a"
    per_target, linear, with_source = converted_scale(usable, n_boot, seed)
    per_target.to_csv(output_dir / "perceived_by_target.csv", index=False)
    linear.to_csv(output_dir / "perceived_linear.csv", index=False)
    with_source.to_csv(output_dir / "perceived_with_source.csv", index=False)

    print("\n=== perceived level of the conversions, one per requested target ===")
    print(f"  {'asked':>7s} {'perceived':>22s} {'shortfall':>10s}")
    for r in per_target.sort_values("requested_db").itertuples():
        print(f"  {r.requested_db:6.1f}  {r.estimate:8.1f} "
              f"[{r.ci_low:5.1f}, {r.ci_high:5.1f}] {r.shortfall_db:+10.1f}")

    print("\n=== the same, as a line ===")
    for r in linear.itertuples():
        print(f"  {r.quantity:38s} {r.estimate:+8.3f}  "
              f"[{r.ci_low:+.3f}, {r.ci_high:+.3f}]")

    print("\n=== does the source level matter as well as the target? ===")
    for cluster, part in with_source.groupby("resampled", sort=False):
        print(f"  resampling {cluster}s:")
        for r in part.itertuples():
            print(f"    {r.quantity:42s} {r.estimate:+8.3f}  "
                  f"[{r.ci_low:+.3f}, {r.ci_high:+.3f}]")
    source_row = with_source[
        (with_source["resampled"] == "selection")
        & with_source["quantity"].str.startswith("source")].iloc[0]
    if source_row["ci_low"] <= 0 <= source_row["ci_high"]:
        print("  -> with only 8 selections the source effect is NOT established: "
              "the interval spans zero once selections are resampled. The offsets "
              "between selections are real; attributing them to the source level "
              "is not supported by this design.")
    else:
        print("  -> the source level matters even with selections resampled")

    # ------------------------------------------------------------------
    # The figure.
    # ------------------------------------------------------------------
    per_stimulus = per_stimulus_levels(usable)
    per_stimulus.to_csv(output_dir / "perceived_by_stimulus.csv", index=False)
    weak = int((~per_stimulus["bracketed"]).sum())
    firm = per_stimulus[per_stimulus["bracketed"]]
    print(f"\n=== per-conversion scatter ===")
    print(f"  {len(per_stimulus)} conversions placed; {weak} fall outside their own "
          f"anchor range, so those are extrapolations and are marked as such")
    print(f"  perceived {per_stimulus['perceived_db'].min():.1f}-"
          f"{per_stimulus['perceived_db'].max():.1f} dB over all of them, "
          f"{firm['perceived_db'].min():.1f}-{firm['perceived_db'].max():.1f} dB "
          f"over the firm ones")
    print(f"  spread within one requested level: sd "
          f"{per_stimulus.groupby('requested_db')['perceived_db'].std().mean():.2f} dB "
          f"(firm only {firm.groupby('requested_db')['perceived_db'].std().mean():.2f})")

    line = (float(linear.loc[linear["quantity"] == "offset dB", "estimate"].iloc[0]),
            float(linear.loc[linear["quantity"].str.startswith("slope"),
                             "estimate"].iloc[0]))
    figure_dir = output_dir / "figures"
    make_figure(per_target, per_stimulus, line, real_line,
                figure_dir / "perceived_vs_requested.png",
                figure_dir / "perceived_vs_requested_two_panel.png")
    per_selection = per_stimulus.sort_values(["selection", "requested_db"]).groupby(
        "selection").apply(lambda p: pd.Series({
            "source_db": p["source_level_db"].iloc[0],
            "slope": np.polyfit(p["requested_db"], p["perceived_db"], 1)[0],
            "at_60db": np.polyval(
                np.polyfit(p["requested_db"], p["perceived_db"], 1), 60.0),
            "range_db": p["perceived_db"].max() - p["perceived_db"].min(),
            "rank_corr": p["requested_db"].corr(p["perceived_db"], method="spearman"),
        }), include_groups=False).sort_values("at_60db")
    per_selection.to_csv(output_dir / "perceived_by_selection.csv")
    print("\n=== per selection: ordered inside, offset between ===")
    print(f"  {'selection':10s} {'source':>7s} {'slope':>6s} {'@60 dB':>7s} "
          f"{'range':>6s} {'rank corr':>10s}")
    for name, r in per_selection.iterrows():
        print(f"  {name:10s} {r.source_db:7.1f} {r.slope:6.2f} {r.at_60db:7.1f} "
              f"{r.range_db:6.1f} {r.rank_corr:10.2f}")
    print(f"  offsets at a 60 dB request span "
          f"{per_selection['at_60db'].min():.1f}-{per_selection['at_60db'].max():.1f} dB "
          f"(sd {per_selection['at_60db'].std():.2f}); "
          f"every rank correlation positive: "
          f"{bool((per_selection['rank_corr'] > 0).all())}")
    print(f"  wrote {figure_dir / 'perceived_vs_requested.png'} and "
          f"{figure_dir / 'perceived_vs_requested_two_panel.png'}")

    print(f"\nWrote coverage.csv, ruler.csv, recovered_levels.csv, "
          f"perceived_by_target.csv, perceived_by_stimulus.csv, "
          f"perceived_linear.csv and perceived_with_source.csv to {output_dir}")


if __name__ == "__main__":
    main()

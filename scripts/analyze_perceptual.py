"""Score the two-alternative listening test and report its statistics.

    uv run --extra cpu scripts/analyze_perceptual.py \\
        --config configs/paper/analyze_perceptual.yaml \\
        --output-dir outputs/perceptual

What it does, in order:

1. parses every participant file, dropping the practice trial;
2. resolves each stimulus name into (speaker, sentence, pass, effort, target);
3. gives the converted member the level it was **asked** for, from the render
   grid, and the real member its **calibrated** level, measured over the span the
   VAD kept -- the audio as played, not the whole original segment;
4. scores each trial as whether the louder member was chosen;
5. reports accuracy against $|\\Delta|$, the real-versus-real control, and
   listener agreement per group.


Config
------
    experiment_dir : the directory of PsyToolkit participant files.
    metadata_csv / dataset_roots : the annotated corpus table the real stimuli
                     come from, for their calibration and their span.
    targets        : ``{min_db, max_db, n}``, copied from the render config.
                     Copied, not inherited, so the numbers this analysis reports
                     can be checked against the run that produced the audio.
    window_length  : SPL analysis window in samples, matching the run.
    vad            : Silero parameters, matching the run, so the trim is the same.
    exclude_listeners / exclude_below_control_accuracy : screening. The control
                     trials decide it -- a listener who cannot order two real
                     recordings was not doing the task -- but there are only a
                     handful per listener, so read the exact p, not the bare
                     proportion, before setting a floor.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from vic.evaluation.perceptual import (
    TargetGrid,
    accuracy,
    accuracy_by_delta,
    describe_trials,
    listener_agreement,
    accuracy_by,
    difficulty_model,
    metrics_table,
    pairwise_agreement,
    per_listener_table,
    source_target_cells,
    read_experiment,
    score,
)
from vic.evaluation.stimulus_levels import KEY, calibrated_levels, stimulus_keys
from speech_eval import figures

from experiment_launcher import parse_args


@parse_args
def main(config: dict, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)

    trials = read_experiment(Path(config["experiment_dir"]))
    grid = TargetGrid(**config["targets"])
    trials = describe_trials(trials, grid)
    print(f"{len(trials)} test trial(s), {trials['listener'].nunique()} listener(s), "
          f"group sizes {trials.groupby('group')['listener'].nunique().to_dict()}")
    print(f"  synthetic-vs-real {int(trials['synthetic_trial'].sum())}, "
          f"real-vs-real {int((~trials['synthetic_trial']).sum())}, "
          f"no response {int((~trials['answered']).sum())}")

    levels = calibrated_levels(stimulus_keys(trials, "b"), config)
    levels.to_csv(output_dir / "real_levels.csv", index=False)
    print(f"\ncalibrated {len(levels)} real stimulus level(s): "
          f"{levels['b_level_db'].min():.1f}-{levels['b_level_db'].max():.1f} dB "
          f"over {levels['played_s'].min():.2f}-{levels['played_s'].max():.2f} s")

    trials = trials.merge(
        levels.rename(columns={c: f"b_{c}" for c in KEY}),
        on=[f"b_{c}" for c in KEY], how="left",
    )
    # A real-vs-real trial has no requested level: member A is a recording too,
    # so its reference is measured the same way as member B's.
    control = ~trials["synthetic_trial"]
    if control.any():
        a_keys = trials.loc[control, [f"a_{c}" for c in KEY]].drop_duplicates()
        a_keys.columns = KEY
        a_levels = calibrated_levels(a_keys, config).rename(
            columns={"b_level_db": "a_measured_db"}
        )
        trials = trials.merge(
            a_levels[KEY + ["a_measured_db"]].rename(
                columns={c: f"a_{c}" for c in KEY}),
            on=[f"a_{c}" for c in KEY], how="left",
        )
        trials["a_level_db"] = trials["a_level_db"].fillna(trials["a_measured_db"])

    # The converted member's own SOURCE recording is a real AVID utterance, so
    # its level is measured the same way -- needed for the requested change and
    # for both axes of the intensity plane.
    src_keys = trials.loc[trials["synthetic_trial"],
                          [f"a_{c}" for c in KEY]].drop_duplicates()
    src_keys.columns = KEY
    src = calibrated_levels(src_keys, config).rename(
        columns={"b_level_db": "source_level_db", "played_s": "source_played_s"})
    src.to_csv(output_dir / "source_levels.csv", index=False)
    print(f"calibrated {len(src)} source recording level(s): "
          f"{src['source_level_db'].min():.1f}-{src['source_level_db'].max():.1f} dB")
    trials = trials.merge(
        src[KEY + ["source_level_db"]].rename(columns={c: f"a_{c}" for c in KEY}),
        on=[f"a_{c}" for c in KEY], how="left")
    # What the model was asked to change, signed: negative is a request to soften.
    trials["requested_change_db"] = trials["a_level_db"] - trials["source_level_db"]

    scored = score(trials)

    # ------------------------------------------------------------------
    # Screening, before anything is reported over the pool.
    # ------------------------------------------------------------------
    per_listener = per_listener_table(scored)
    per_listener.to_csv(output_dir / "per_listener.csv", index=False)
    print("\n=== per listener (sorted by accuracy within trial type) ===")
    print("   listener  grp  type       n  acc    prec   rec    spec   f1     d'     c      p(>chance)")
    for r in per_listener.itertuples():
        print(f"   {r.listener[:8]}  {r.group}    {r.trial_type:9s} {r.n:3d}  "
              f"{r.accuracy:.3f}  {r.precision:.3f}  {r.recall:.3f}  {r.specificity:.3f}  "
              f"{r.f1:.3f}  {r.d_prime:+.2f}  {r.criterion:+.2f}  {r.p_above_chance:.3f}")

    # The control trials are the screening instrument: a listener who cannot
    # order two REAL recordings was not doing the task.  There are only a handful
    # per listener, so the exact p is what decides, never the bare proportion.
    control_by_listener = per_listener[per_listener["trial_type"] == "control"]
    floor = config.get("exclude_below_control_accuracy")
    excluded = set(config.get("exclude_listeners") or [])
    if floor is not None:
        failing = control_by_listener[control_by_listener["accuracy"] < float(floor)]
        excluded |= set(failing["listener"])
    if excluded:
        print(f"\n*** excluding {len(excluded)} listener(s) on the control: "
              f"{sorted(x[:8] for x in excluded)}")
        scored = scored[~scored["listener"].isin(excluded)].reset_index(drop=True)
        print(f"    {scored['listener'].nunique()} listener(s) remain, "
              f"{int(scored['correct'].notna().sum())} scored trial(s)")
    scored.to_csv(output_dir / "trials.csv", index=False)

    # ------------------------------------------------------------------
    # Everything below is reported for both trial types side by side.
    # ------------------------------------------------------------------
    overall = metrics_table(scored, by=["trial_type"])
    with_ci = accuracy(scored, by=["trial_type"]).set_index("trial_type")
    overall = overall.join(
        with_ci[["ci_low", "ci_high", "n_listeners"]], on="trial_type"
    )
    overall.to_csv(output_dir / "overall.csv", index=False)
    print("\n=== overall, converted vs control (chance = 0.5) ===")
    for r in overall.itertuples():
        print(f"   {r.trial_type:9s} n={r.n:4d} ({r.n_listeners} listeners)  "
              f"acc {r.accuracy:.3f} [{r.ci_low:.3f}, {r.ci_high:.3f}]")
        print(f"             {'':9s}    precision {r.precision:.3f}  recall {r.recall:.3f}  "
              f"specificity {r.specificity:.3f}  F1 {r.f1:.3f}")
        print(f"             {'':9s}    d' {r.d_prime:+.3f}  criterion {r.criterion:+.3f}  "
              f"(TP {r.tp} FP {r.fp} FN {r.fn} TN {r.tn})")

    print("\n=== by |delta|, converted vs control ===")
    frames = []
    for kind, part in scored.groupby("trial_type"):
        by_delta = accuracy_by_delta(part)
        by_delta.insert(0, "trial_type", kind)
        frames.append(by_delta)
        print(f"  -- {kind}")
        for r in by_delta.itertuples():
            print(f"     {str(r.delta_bin):>12s} dB  {r.accuracy:.3f} "
                  f"[{r.ci_low:.3f}, {r.ci_high:.3f}]  n={r.n_trials}")
    pd.concat(frames, ignore_index=True).to_csv(
        output_dir / "accuracy_by_delta.csv", index=False)

    print("\n=== by direction (is the A member asked to be louder?) ===")
    directed = scored[scored["correct"].notna()].assign(
        asked=np.where(scored.loc[scored["correct"].notna(), "delta_db"] > 0,
                       "louder", "softer"))
    by_dir = accuracy(directed, by=["trial_type", "asked"])
    by_dir.to_csv(output_dir / "accuracy_by_direction.csv", index=False)
    for r in by_dir.itertuples():
        print(f"   {r.trial_type:9s} A asked {r.asked:7s} {r.accuracy:.3f} "
              f"[{r.ci_low:.3f}, {r.ci_high:.3f}]  n={r.n_trials}")

    print("\n=== agreement per group and trial type ===")
    agree = listener_agreement(scored)
    agree.to_csv(output_dir / "agreement.csv", index=False)
    for r in agree.itertuples():
        print(f"   Grp{r.group} {r.trial_type:9s} {r.n_listeners} listener(s), "
              f"{r.n_pairs:3d} pair(s): majority {r.majority_share:.3f}, "
              f"observed {r.observed_agreement:.3f}, expected {r.expected_agreement:.3f}, "
              f"Fleiss kappa {r.fleiss_kappa:+.3f}")

    pairs = pairwise_agreement(scored)
    pairs.to_csv(output_dir / "pairwise_agreement.csv", index=False)
    print("\n=== pairwise listener agreement (a lone dissenter shows up here) ===")
    for (grp, kind), part in pairs.groupby(["group", "trial_type"]):
        print(f"  -- Grp{grp} {kind}: mean {part['agreement'].mean():.3f}, "
              f"min {part['agreement'].min():.3f} "
              f"({part.loc[part['agreement'].idxmin(), 'listener_a'][:8]} vs "
              f"{part.loc[part['agreement'].idxmin(), 'listener_b'][:8]}), "
              f"max {part['agreement'].max():.3f}")

    # ------------------------------------------------------------------
    # The intensity plane.
    # ------------------------------------------------------------------
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(exist_ok=True)
    conv = scored[scored["trial_type"] == "converted"]
    curve = difficulty_model(conv)
    curve.to_csv(output_dir / "difficulty_curve.csv", index=False)

    change_edges = np.arange(-30, 31, 6.0)
    views = {
        "requested_change": (
            "requested_change_db", change_edges,
            "requested change, target - source (dB)",
            "Audibility of the requested change"),
        "difficulty": (
            "abs_delta_db", None,
            "|level difference| to the real recording (dB)",
            "Difficulty: how far apart the pair was"),
        "target_level": (
            "a_level_db", np.arange(43, 78, 4.3),
            "requested target level (dB SPL)",
            "Where in the range the target sat"),
        "source_level": (
            "source_level_db", np.arange(48, 73, 4.0),
            "source level (dB SPL)",
            "Where in the range the source sat"),
    }
    print("\n=== views on the intensity plane ===")
    for name, (column, edges, xlabel, title) in views.items():
        table = curve if edges is None else accuracy_by(conv, column, edges)
        table.to_csv(output_dir / f"accuracy_by_{name}.csv", index=False)
        ax = figures.proportion_curve(
            table, x="x", xlabel=xlabel, title=title, n="n_trials")
        figures.save(ax, figure_dir / f"accuracy_by_{name}.png")
        print(f"  {name:18s} {len(table)} bin(s), "
              f"accuracy {table['accuracy'].min():.2f}-{table['accuracy'].max():.2f}")

    cells = source_target_cells(conv, curve)
    cells.to_csv(output_dir / "source_target_cells.csv", index=False)
    print(f"\n=== source x target plane: {len(cells)} cell(s), "
          f"{int(cells['single_class'].sum())} with a single class ===")
    print(f"  above chance: mean {cells['above_chance'].mean():+.3f}, "
          f"{int((cells['above_chance'] > 0).sum())}/{len(cells)} cells positive")
    print(f"  residual vs the difficulty curve: mean {cells['residual'].mean():+.3f}, "
          f"sd {cells['residual'].std():.3f}")

    # Both maps are diverging about a neutral midpoint, which is the whole reason
    # `above_chance` is plotted rather than raw accuracy: on a diverging scale a
    # cell at chance must read as nothing, and raw accuracy would put its
    # midpoint at 0.5 only by accident of the data's range.
    for value, label in (("above_chance", "accuracy - chance"),
                         ("residual", "accuracy - difficulty curve")):
        ax = figures.cell_heatmap(
            cells.round({"source_level_db": 1, "target_level_db": 1}),
            row_col="source_level_db", col_col="target_level_db", value=value,
            row_label="source level (dB SPL)", col_label="target level (dB SPL)",
            title=label, symmetric=True, value_label=label,
        )
        figures.save(ax, figure_dir / f"plane_{value}.png")

    print(f"\nWrote tables and {len(list(figure_dir.glob('*.png')))} figure(s) "
          f"to {output_dir}")


if __name__ == "__main__":
    main()

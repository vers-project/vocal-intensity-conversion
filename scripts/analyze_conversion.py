"""Turn an evaluation run into the paper's figures and tables.

A thin consumer.  Every method here lives in ``speech_eval``: the displacement formulation
and its fits in ``speech_eval.compare.displacement``, the word-error pooling in
``compare.wer``, the speaker calibration in ``compare.speaker``, the figures in
``speech_eval.figures``.  What this script contributes is the part that is specific to
*this* experiment — which columns are displacement measures and which are preservation
ones, what the four effort levels are called, and where the run directory is.

Reads ``metrics.csv``, not ``pairs.csv``.  Each conversion needs three references — the
real recording at its source level, the codec round trip of that source, and the real
recording at its target level — and ``pairs.csv`` carries only the last.  They are joined
on ``source_real_utt_id``, which every row carries.

Usage
-----
    uv run --extra cpu --extra phonetics scripts/analyze_conversion.py \\
        --config configs/paper/analyze_conversion.yaml \\
        --output-dir outputs/analysis/avid_labels

Config
------
    run_dir            : the evaluation run directory (metrics.csv, artifacts.npz).
    displacement       : measures that SHOULD move with the conversion, as
                         ``{column: {label, unit}}``.  Each gets a slope, an intercept
                         and a residual spread.
    preservation       : measures that should NOT move — speaker similarity, WER.  A slope
                         is meaningless for these; they get a floor and a ceiling.
    levels             : index → name, for the source × target grid's tick labels.
    group_col          : what a slope is fitted within.  ``speaker_uid``, never a raw
                         speaker column: ids are unique only within a corpus.
    n_boot             : cluster-bootstrap resamples for the interval.

Outputs (in ``--output-dir``)
-----------------------------
    displacement.csv   one row per (conversion, measure): requested, achieved, error
    slopes.csv         per group × measure: slope, intercept, r, n
    summary.csv        per measure: mean slope with interval, bias, spread
    preservation.csv   speaker similarity (raw and calibrated) and pooled WER per condition
    figures/*.png      the scatter, the grids, the slope dot plot
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from speech_eval import figures
from vic.evaluation.conditions import CODEC
from vic.evaluation.monitor import summarise_conversions

from experiment_launcher import parse_args


def load_run(run_dir: Path) -> tuple[pd.DataFrame, dict]:
    metrics = pd.read_csv(run_dir / "metrics.csv")
    archive = run_dir / "artifacts.npz"
    arrays = dict(np.load(archive)) if archive.exists() else {}
    print(f"{len(metrics)} utterance(s), {len(arrays)} array(s) from {run_dir}")
    print("   " + metrics["condition"].value_counts().to_string().replace("\n", "\n   "))
    return metrics, arrays


@parse_args
def main(config: dict, output_dir: Path):

    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = Path(config["run_dir"])
    group_col = config.get("group_col", "speaker_uid")
    metrics, arrays = load_run(run_dir)

    displacement_spec = config["displacement"]
    levels = {int(k): v for k, v in (config.get("levels") or {}).items()}
    level_names = [levels[k] for k in sorted(levels)] if levels else None

    # One shared implementation with the training-time monitor: the three-reference
    # join, the per-speaker fits and the preservation summaries.  This script's job
    # is the configuration around it and the tables and figures that come out.
    result = summarise_conversions(
        metrics, arrays, list(displacement_spec),
        group_col=group_col, n_boot=int(config.get("n_boot", 10_000)),
    )
    table, slopes, cells = result.displacement, result.slopes, result.cells
    summary = result.summary
    if not summary.empty:
        summary.insert(1, "label", summary["metric"].map(
            {k: v.get("label", k) for k, v in displacement_spec.items()}))

    table.to_csv(output_dir / "displacement.csv", index=False)
    slopes.to_csv(output_dir / "slopes.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)
    cells.to_csv(output_dir / "cells.csv", index=False)

    # Written apart, because speaker similarity, word errors and predicted quality
    # share no columns and concatenating them yields a table half full of NaN that
    # reads as missing data.
    speaker, wer = result.speaker, result.wer
    speaker.to_csv(output_dir / "preservation_speaker.csv", index=False)
    wer.to_csv(output_dir / "preservation_wer.csv", index=False)
    result.quality.to_csv(output_dir / "preservation_quality.csv", index=False)

    # ------------------------------------------------------------------
    # Figures.
    # ------------------------------------------------------------------
    figure_dir = output_dir / "figures"
    for column, spec in displacement_spec.items():
        if column not in set(table["metric"]):
            continue
        label, unit = spec.get("label", column), spec.get("unit", "")
        figures.save(
            figures.displacement_scatter(table, metric=column, unit=unit, title=label),
            figure_dir / f"displacement_{column}.png",
        )
        figures.save(
            figures.cell_heatmap(
                cells, metric=column, row_col="source_level_index",
                col_col="target_level_index", labels=level_names, unit=unit, title=label),
            figure_dir / f"heatmap_{column}.png",
        )
    if not slopes.empty:
        labelled = slopes.copy()
        labelled["metric"] = labelled["metric"].map(
            {k: v.get("label", k) for k, v in displacement_spec.items()}).fillna(
            labelled["metric"])
        figures.save(
            figures.group_slopes(labelled, group_col=group_col,
                                 title="achieved / requested, per speaker"),
            figure_dir / "slopes.png",
        )

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    print(f"\n{len(table)} displacement rows over {table['metric'].nunique()} measure(s), "
          f"{slopes[group_col].nunique() if not slopes.empty else 0} {group_col}(s)")
    print("\nslope (1 = moves by the right amount, 0 = ignores the request):")
    for row in summary.itertuples():
        print(f"   {row.label:<28} {row.slope_mean:6.2f}  "
              f"[{row.slope_lo:5.2f}, {row.slope_hi:5.2f}]   "
              f"bias {row.error_mean:+6.2f}  spread {row.error_sd:5.2f}")
    if not speaker.empty:
        print("\nspeaker identity (1 = a real same-speaker pair, 0 = different speakers):")
        for row in speaker.itertuples():
            print(f"   {row.embedder:<14} {row.condition:<10} "
                  f"cos {row.cosine_mean:.3f}  calibrated {row.calibrated_mean:6.3f}   "
                  f"(ceiling {row.ceiling_mu_target:.3f}, floor {row.floor_mu_nontarget:.3f}, "
                  f"{row.short_fraction:.0%} short)")
    if not wer.empty:
        print("\nWER, pooled sum(n_err)/sum(n_ref):")
        for row in wer.itertuples():
            print(f"   {row.backend:<14} {row.condition:<10} {row.wer:6.3f}"
                  + (f"   {row.delta_vs_codec:+.3f} vs codec"
                     if row.condition != CODEC else "   (anchor)"))
    print(f"\nWrote tables and {len(list(figure_dir.glob('*.png')))} figure(s) "
          f"to {output_dir}")


if __name__ == "__main__":
    main()

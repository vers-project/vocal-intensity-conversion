"""The conversion evaluation as one callable, so training and test can share it.

``scripts/evaluate_conversion.py`` and ``scripts/analyze_conversion.py`` between
them do four things: measure τ_src for every source, render and measure each
condition, join the three references into requested-versus-achieved displacement,
and fit a slope per speaker.  All four are wanted *during* training too, because
the intensity RMSE that training used to report cannot see a model that hits the
requested level while sounding worse — and a checkpoint has to be chosen on
something that can.

What is here is only the sequencing.  The rendering is
:class:`~vic.evaluation.conditions.ConversionRenderer`, the measuring is
:class:`~vic.evaluation.harness.MetricHarness`, the statistics are
``speech_eval.compare.displacement``, and the preservation summaries are
:mod:`vic.evaluation.analysis`.  Nothing is reimplemented, and in particular
nothing is reimplemented *more cheaply* for the training-time caller: a monitor
that computes a number a slightly different way from the final table is worse
than no monitor, because the two disagree and neither is wrong.

No I/O and no config
--------------------
Neither function reads a file, writes a file or takes a config dict.  The script
keeps its tables and figures, the callback keeps its Lightning scalars, and this
module keeps the part they must agree on.  Cost is controlled by what the caller
puts in the dataset, not by a cheaper code path.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from speech_eval.compare.displacement import (
    cell_means,
    displacement_table,
    per_group_slopes,
    summarise,
)

from vic.evaluation.analysis import (
    quality_preservation,
    speaker_preservation,
    wer_preservation,
)
from vic.evaluation.conditions import (
    CODEC,
    CONVERTED,
    REAL,
    REFERENCE_UTT_ID,
    SOURCE_REAL_UTT_ID,
    ConversionRenderer,
)
from vic.evaluation.harness import MetricHarness

#: Identity columns carried onto the displacement rows when present.  Filtered
#: against the results table rather than demanded: a validation split need not
#: carry every column the annotated test split does.
CARRY_CANDIDATES = [
    "source_level_index", "target_level_index", "sentence_id", "repetition",
    "delta_tau_db",
]


def _identity(iterable: Iterable, **_: Any) -> Iterable:
    return iterable


@dataclass
class ConversionSummary:
    """Every table the evaluation produces, and the scalars worth logging."""

    displacement: pd.DataFrame
    slopes: pd.DataFrame
    summary: pd.DataFrame
    cells: pd.DataFrame
    speaker: pd.DataFrame
    wer: pd.DataFrame
    quality: pd.DataFrame
    results: pd.DataFrame = field(repr=False, default_factory=pd.DataFrame)
    arrays: dict = field(repr=False, default_factory=dict)

    def scalars(self, prefix: str = "val") -> dict[str, float]:
        """Flatten to ``{key: float}`` for a logger.

        One key per measure and per kind, rather than one dict per table, because
        a logger charts scalars and a table is not one.  Slopes are named by the
        measure they fit so ``slope/f0_praat.median_hz`` sits next to
        ``slope/intensity.leq_db`` on the same axes.
        """
        out: dict[str, float] = {}
        for row in self.summary.itertuples():
            out[f"{prefix}/slope/{row.metric}"] = float(row.slope_mean)
            out[f"{prefix}/bias/{row.metric}"] = float(row.error_mean)
            out[f"{prefix}/spread/{row.metric}"] = float(row.error_sd)
        for row in self.speaker.itertuples():
            out[f"{prefix}/speaker/{row.embedder}/{row.condition}"] = float(
                row.calibrated_mean)
        for row in self.wer.itertuples():
            out[f"{prefix}/wer/{row.backend}/{row.condition}"] = float(row.wer)
        for row in self.quality.itertuples():
            measure = str(row.measure).split(".")[0]
            out[f"{prefix}/{measure}/{row.condition}"] = float(row.mean)
        return {k: v for k, v in out.items() if np.isfinite(v)}


@torch.no_grad()
def measure_conversions(
    module,
    pipeline,
    loader_factory: Callable[[], Iterable[dict]],
    metrics: Sequence,
    targets_by_row: dict[int, list],
    stems: Sequence[str],
    group_ids: Sequence[str],
    records: Sequence[dict],
    device: torch.device,
    *,
    with_codec: bool = True,
    batch_size: int = 16,
    progress: Callable = _identity,
) -> tuple[pd.DataFrame, dict]:
    """Render every condition and measure it, entirely in memory.

    Two passes over the same sources, and the order is not an accident: a level
    target is another row's τ_src, so every source level is measured before any
    conversion is rendered.  τ_src is therefore computed exactly once and the two
    passes cannot disagree about it.

    Parameters
    ----------
    module          : the trained converter module; supplies ``source_tau`` and the
                      conversion path, so what is measured is what training built.
    pipeline        : its conversion-space pipeline.
    loader_factory  : called once per pass to get a fresh iterator of batches of
                      one source each.  A factory rather than a loader because the
                      two passes each need to start from the beginning, and a
                      consumed iterator silently yields an empty second pass.
    metrics         : ``speech_eval`` metrics, already built.
    targets_by_row  : row index -> level targets, as ``vic.data.levels`` produces.
    stems, group_ids, records : per-row identity, aligned with the loader's order.
    progress        : optional wrapper for a progress bar; defaults to a no-op so
                      the callback stays silent in a training log.

    Returns
    -------
    ``(results, arrays)`` exactly as :meth:`MetricHarness.finish` returns them.
    """
    n_sources = len(stems)

    tau_src = np.empty(n_sources, dtype=np.float64)
    for i, batch in enumerate(progress(
        loader_factory(), desc="pass 1  measuring tau_src", total=n_sources
    )):
        batch = {k: v.to(device) for k, v in batch.items()}
        tau_src[i] = float(module.source_tau(batch)[0])

    harness = MetricHarness(list(metrics), device, batch_size=batch_size)
    renderer = ConversionRenderer(
        module, pipeline, list(stems), list(group_ids), tau_src, list(records)
    )
    for i, batch in enumerate(progress(
        loader_factory(), desc="pass 2  rendering + measuring", total=n_sources
    )):
        batch = {k: v.to(device) for k, v in batch.items()}
        for rendered in renderer.render(
            batch, i, targets_by_row.get(i, []), with_codec=with_codec
        ):
            harness.add(
                rendered.utt_id, rendered.wav, pipeline.sample_rate,
                rendered.record, text=rendered.record.get("text"),
            )
    return harness.finish()


def summarise_conversions(
    results: pd.DataFrame,
    arrays: dict,
    displacement_metrics: Sequence[str],
    *,
    group_col: str = "speaker_uid",
    n_boot: int = 1_000,
    min_points: int = 3,
    seed: int = 0,
) -> ConversionSummary:
    """Join the three references, fit a slope per group, and summarise the rest.

    ``displacement_metrics`` names the columns that *should* move with the
    conversion.  Columns absent from ``results`` are skipped rather than raising,
    so one list can serve a monitor running a reduced metric set and a final
    analysis running the full one.

    ``n_boot`` defaults well below the analysis script's 10 000: a monitor reads
    the slope, not its interval, and the interval over a handful of validation
    speakers is wide whatever the resample count.
    """
    present = [m for m in displacement_metrics if m in results.columns]
    carry = [group_col] + [c for c in CARRY_CANDIDATES if c in results.columns]

    empty = pd.DataFrame()
    if not present or results.empty:
        return ConversionSummary(
            displacement=empty, slopes=empty, summary=empty, cells=empty,
            speaker=empty, wer=empty, quality=empty,
            results=results, arrays=arrays,
        )

    table = displacement_table(
        results, present,
        source_col=SOURCE_REAL_UTT_ID, target_col=REFERENCE_UTT_ID,
        source_condition=REAL, baseline_condition=CODEC,
        subject_conditions=[CONVERTED],
        carry=carry,
    )
    slopes = per_group_slopes(table, group_col=group_col, min_points=min_points)
    summary = (
        summarise(slopes, table, group_col=group_col, n_boot=n_boot, seed=seed)
        if not slopes.empty else empty
    )
    cells = (
        cell_means(table, row_col="source_level_index", col_col="target_level_index")
        if {"source_level_index", "target_level_index"} <= set(table.columns)
        else empty
    )
    return ConversionSummary(
        displacement=table,
        slopes=slopes,
        summary=summary,
        cells=cells,
        speaker=speaker_preservation(results, arrays, group_col),
        wer=wer_preservation(results),
        quality=quality_preservation(results),
        results=results,
        arrays=arrays,
    )

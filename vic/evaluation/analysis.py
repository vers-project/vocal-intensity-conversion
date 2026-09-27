"""Preservation summaries: the measures for which a slope would be meaningless.

The displacement measures of ``speech_eval.compare.displacement`` ask whether a
quantity moved the amount it was asked to.  Speaker identity and word error rate
ask the opposite — whether a quantity stayed put — and the same machinery does
not apply: there is no requested displacement to regress against, and a raw
cosine or an absolute WER is uninterpretable on its own.  Each therefore gets a
floor and a ceiling drawn from the corpus itself.

These two functions were originally written inside
``scripts/analyze_conversion.py``.  They live here because they are pure
table-to-table transforms with no I/O and no configuration, and because the
training-time monitor needs exactly the same numbers as the final analysis —
computing them twice, in two places, is how the two stop agreeing.

Everything method-shaped is still ``speech_eval``'s: the calibration in
``compare.speaker``, the pooling in ``compare.wer``.  What is here is the part
that knows this experiment's conditions.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from vic.evaluation.conditions import (
    CODEC,
    CONVERTED,
    REAL,
    REFERENCE_UTT_ID,
    SOURCE_REAL_UTT_ID,
)


def speaker_preservation(
    metrics: pd.DataFrame, arrays: dict, group_col: str
) -> pd.DataFrame:
    """Where converted audio sits between the same-speaker ceiling and the impostor floor.

    A raw cosine is uninterpretable: 0.72 is neither good nor bad without knowing how far
    apart this corpus's own same-speaker and different-speaker pairs sit.  The trials are
    built from the ``real`` rows — real speech only, so the scale is a property of the
    corpus and the embedder rather than of the system under test — and the conversions are
    then read on it.
    """
    from speech_eval.compare.speaker import (
        calibrate,
        calibrated_similarity,
        cosine,
        verification_trials,
    )

    embedders = sorted({
        key.split("|", 1)[1].rsplit(".", 1)[0]
        for key in arrays if "|" in key and key.endswith(".embedding")
    })
    rows = []
    for name in embedders:
        column = f"{name}.embedding"
        trials = verification_trials(
            metrics, arrays, embedding_col=column,
            speaker_col=group_col, conditions=[REAL],
        )
        if trials.empty:
            continue
        # Both classes are needed, and a subset covering one speaker has only
        # same-speaker pairs.  `calibrate` rightly raises there; a training-time
        # monitor must not die of it, so the embedder is skipped with a warning
        # and every other measure still gets reported.
        if "label" in trials.columns and trials["label"].astype(bool).nunique() < 2:
            warnings.warn(
                f"{name}: {len(trials)} trial(s) but only one class, so no "
                "same-speaker/different-speaker scale can be built. This subset "
                "covers a single speaker; widen it to calibrate speaker identity.",
                RuntimeWarning,
            )
            continue
        calibration = calibrate(trials)

        for condition in (CODEC, CONVERTED):
            part = metrics[metrics["condition"] == condition]
            # A codec row has no target of its own, so it is read against the source it
            # came from.  `pd.isna` rather than `or`: a float NaN is TRUTHY, so the
            # obvious `reference or source` keeps the NaN and looks up "nan|...".
            reference = part[REFERENCE_UTT_ID].where(
                part[REFERENCE_UTT_ID].notna(), part[SOURCE_REAL_UTT_ID])
            similarity = np.array([
                cosine(arrays[f"{utt}|{column}"], arrays[f"{ref}|{column}"])
                if f"{utt}|{column}" in arrays and f"{ref}|{column}" in arrays
                else np.nan
                for utt, ref in zip(part["utt_id"], reference)
            ], dtype=float)
            rows.append({
                "embedder": name,
                "condition": condition,
                "n": int(np.isfinite(similarity).sum()),
                "cosine_mean": float(np.nanmean(similarity)),
                # 1 = as similar as two real recordings of one speaker; 0 = as similar
                # as two different speakers. Not a probability, not bounded, and only
                # comparable across runs sharing a corpus and an embedder.
                "calibrated_mean": float(np.nanmean(
                    calibrated_similarity(similarity, calibration))),
                "ceiling_mu_target": calibration["mu_target"],
                "floor_mu_nontarget": calibration["mu_nontarget"],
                "d_prime": calibration["d_prime"],
                "eer": calibration.get("eer"),
                "short_fraction": float(
                    (part.get(f"{name}.status", pd.Series(dtype=str)) == "short").mean()),
            })
    return pd.DataFrame(rows)


def wer_preservation(metrics: pd.DataFrame) -> pd.DataFrame:
    """Pooled WER per backend and condition, and the change against the codec anchor.

    Pooled as ``sum(n_err) / sum(n_ref)``, never as a mean of per-utterance rates — which
    is why the evaluation stored counts.  Reported against the codec rather than in
    absolute terms, because part of any degradation is the codec's.
    """
    from speech_eval.compare.wer import pooled_wer, transcription_errors

    rows = []
    for column in [c for c in metrics.columns if c.endswith(".hypothesis")]:
        backend = column.split(".")[0]
        errors = transcription_errors(
            metrics, hypothesis_col=column, reference_column="text",
            carry=["condition"],
        )
        pooled = pooled_wer(errors, by="condition")
        pooled.insert(0, "backend", backend)
        anchor = pooled.loc[pooled["condition"] == CODEC, "wer"]
        pooled["delta_vs_codec"] = pooled["wer"] - (
            float(anchor.iloc[0]) if len(anchor) else np.nan)
        rows.append(pooled)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def quality_preservation(
    metrics: pd.DataFrame, columns: list[str] | None = None
) -> pd.DataFrame:
    """Mean predicted quality per condition, and the change against the codec anchor.

    Unlike speaker identity and word errors this needs no reference utterance: a
    naturalness predictor scores a signal on its own.  It still gets the codec
    anchor, because part of any degradation belongs to the encode/decode round
    trip rather than to the conversion, and the absolute value of a MOS predictor
    on this material is not a calibrated opinion score.

    ``columns`` defaults to every ``mos_*.score`` column present.
    """
    if columns is None:
        columns = [
            c for c in metrics.columns
            if c.startswith("mos_") and c.endswith(".score")
        ]
    rows = []
    for column in columns:
        if column not in metrics.columns:
            continue
        grouped = metrics.groupby("condition", dropna=False)[column]
        table = grouped.agg(
            mean="mean", sd="std", n=lambda s: int(s.notna().sum())
        ).reset_index()
        table.insert(0, "measure", column)
        anchor = table.loc[table["condition"] == CODEC, "mean"]
        table["delta_vs_codec"] = table["mean"] - (
            float(anchor.iloc[0]) if len(anchor) else np.nan)
        rows.append(table)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

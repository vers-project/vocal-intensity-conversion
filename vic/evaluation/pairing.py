"""Turn per-utterance features into real-vs-converted comparisons at equal τ.

``speech_eval`` metrics are per-utterance feature extractors on purpose: a recording that
serves as the reference for several conversions is measured once, and redesigning the
comparison costs no GPU time.  This module is the other half — the part that knows what a
condition means for *this* experiment, and pairs the rows accordingly.

The pairing
-----------
Every converted row names the real recording it was aimed at, in ``reference_utt_id``.  So
the comparison is a join, not a search: converted row ``spk9_…_to3`` against the ``real``
row of the group's loud take.  For each numeric metric column the table then carries the
converted value, the reference value, and their difference.

``codec`` rows are joined on too, against the *source's own* real row, because the question
they answer is different: not "did the conversion reach the target" but "what did the codec
alone cost".  Subtracting the codec's contribution from a converted delta is then a column
operation the analysis can do, rather than something baked in here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from vic.evaluation.conditions import CODEC, CONVERTED, REAL

#: Columns that identify a row rather than measure it, so they are never differenced.
IDENTITY_COLUMNS = {
    "utt_id", "group_id", "condition", "reference_utt_id", "source_real_utt_id",
    "target_level", "target_level_index", "source_level", "source_level_index",
    "tau_src_db", "tau_tgt_db", "delta_tau_db", "text",
    "speaker_uid", "subject_id", "sentence_id", "repetition", "sex", "corpus",
    "signal_path", "start_s", "end_s", "level",
}


def numeric_metric_columns(results: pd.DataFrame) -> list[str]:
    """Metric columns worth differencing: ``<metric>.<key>`` and numeric.

    Named by prefix rather than by an allow-list so a metric added later is picked up
    without editing this module.  Non-numeric metric output — an ASR hypothesis, a status
    flag — is carried through untouched instead, since subtracting two strings is not a
    comparison.
    """
    return [
        column for column in results.columns
        if "." in column
        and column not in IDENTITY_COLUMNS
        and pd.api.types.is_numeric_dtype(results[column])
        # bool is "numeric" to pandas but `True - False` raises, and the difference of two
        # flags is not a measurement anyway — a voiced/unvoiced flag is carried, not
        # subtracted.
        and not pd.api.types.is_bool_dtype(results[column])
    ]


def pair_with_references(results: pd.DataFrame, warn: bool = True) -> pd.DataFrame:
    """One row per converted (and codec) utterance, beside the real take it targets.

    Every metric column appears three times: ``<col>`` as produced, ``<col>.reference``
    from the real recording, and ``<col>.delta`` for the difference.  Rows whose reference
    was not measured — a sentence missing that level — keep NaN references rather than
    disappearing, so the count of comparisons stays honest.

    ``warn=False`` for an intermediate checkpoint of a run still in progress: there, a
    conversion aimed at a source not yet reached has an unresolved reference *by
    construction*, and saying so every hundred sources trains the reader to ignore the one
    time it means something.
    """
    if results.empty:
        return results.copy()

    reals = results[results["condition"] == REAL].set_index("utt_id")
    measured = numeric_metric_columns(results)
    carried = [c for c in results.columns if "." in c and c not in IDENTITY_COLUMNS]

    rows = results[results["condition"].isin([CONVERTED, CODEC])].copy()
    # A codec row has no target of its own: what it is read against is the source's own
    # real take, which is what makes it an anchor rather than a conversion.
    if "source_real_utt_id" in rows.columns:
        rows["reference_utt_id"] = rows["reference_utt_id"].fillna(
            rows["source_real_utt_id"]
        )
    else:
        # Older tables predate the explicit column; rewriting the suffix recovers it, at
        # the cost of breaking on a stem that itself contains "_codec".
        rows["reference_utt_id"] = rows["reference_utt_id"].fillna(
            rows["utt_id"].str.replace(r"_codec$", "_real", regex=True)
        )

    reference = rows["reference_utt_id"].map(
        lambda key: key if key in reals.index else None
    )
    missing = int(reference.isna().sum())
    if missing and warn:
        print(f"[pairing] {missing} of {len(rows)} row(s) name a real recording that was "
              f"not measured — those comparisons are NaN, not dropped")

    # Built as a dict and concatenated once. eGeMAPS alone is ~88 columns, so inserting
    # two per metric one at a time fragments the frame badly enough for pandas to warn —
    # and on a full split that is ~180 inserts over 7 000 rows.
    added: dict[str, pd.Series] = {}
    for column in carried:
        values = rows["reference_utt_id"].map(
            reals[column] if column in reals.columns else {}
        )
        added[f"{column}.reference"] = values
        if column in measured:
            added[f"{column}.delta"] = rows[column] - values

    return pd.concat([rows, pd.DataFrame(added, index=rows.index)], axis=1).reset_index(
        drop=True
    )


def add_word_errors(
    results: pd.DataFrame, asr_columns: list[str], normalizer: str = "whisper_en"
) -> pd.DataFrame:
    """Attach per-utterance word-error counts for each ASR backend, against ``text``.

    Delegated to ``speech_eval.compare.wer`` rather than reimplemented: it owns the text
    normalisation and the rule that WER is pooled as ``sum(n_err) / sum(n_ref)`` and never
    averaged over per-utterance rates.  Counts are what is stored here for exactly that
    reason — the corpus number cannot be recovered from rates.
    """
    from speech_eval.compare.wer import transcription_errors

    out = results
    for column in asr_columns:
        if column not in results.columns:
            continue
        errors = transcription_errors(
            results, hypothesis_col=column, reference_column="text",
            normalizer=normalizer, carry=[],
        )
        prefix = column.split(".")[0]
        errors = errors.rename(columns={
            c: f"{prefix}.{c}" for c in ("n_sub", "n_del", "n_ins", "n_err", "n_ref", "wer")
            if c in errors.columns
        })
        keep = ["utt_id"] + [c for c in errors.columns if c.startswith(f"{prefix}.")]
        out = out.merge(errors[keep], on="utt_id", how="left")
    return out


def add_speaker_similarity(
    results: pd.DataFrame, arrays: dict[str, np.ndarray]
) -> pd.DataFrame:
    """Cosine similarity per speaker embedder, plus the real-vs-real ceiling.

    Speaker embeddings are arrays, so they live in the artifact archive and never reach
    the results table — which means the one number worth reading from them has to be
    computed here.  Two columns per embedder:

    ``<metric>.cosine``
        between this utterance and the real recording it is paired with: the target-level
        take for a conversion, the source's own take for the codec anchor.
    ``<metric>.cosine_real_real``
        between the two *real* takes at those same two effort levels.  This is the
        ceiling, and it is not 1.0: vocal effort moves a speaker embedding on its own, and
        that movement is already in this number.  Reading a conversion's cosine without it
        charges the converter for what effort does to the embedder.

    Both are raw cosines, deliberately.  A raw cosine is uninterpretable on its own —
    0.72 is neither good nor bad without knowing this corpus's target and non-target
    distributions — so calibration against them is the analysis's job, for which
    ``speech_eval.compare.speaker`` provides ``verification_trials``, ``calibrate`` and
    ``calibrated_similarity``.  Emitting a calibrated number here would bake in a choice
    of trial set that belongs downstream.
    """
    from speech_eval.compare.speaker import cosine

    embedders = sorted({
        key.split("|", 1)[1].rsplit(".", 1)[0]
        for key in arrays
        if "|" in key and key.endswith(".embedding")
    })
    if not embedders:
        return results

    out = results.copy()
    for name in embedders:
        def look_up(utt_id) -> np.ndarray | None:
            return arrays.get(f"{utt_id}|{name}.embedding")

        pairwise, ceiling = [], []
        for row in out.itertuples():
            own = look_up(row.utt_id)
            other = look_up(getattr(row, "reference_utt_id", None))
            pairwise.append(
                float("nan") if own is None or other is None else cosine(own, other)
            )

            source = look_up(getattr(row, "source_real_utt_id", None))
            ceiling.append(
                float("nan") if source is None or other is None
                else cosine(source, other)
            )
        out[f"{name}.cosine"] = pairwise
        out[f"{name}.cosine_real_real"] = ceiling
    return out

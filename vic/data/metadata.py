"""Metadata DataFrame utilities — ``resolve_paths`` re-exported from ``audio_utils``."""
from __future__ import annotations

import pandas as pd
from audio_utils.data.metadata import resolve_paths

__all__ = ["resolve_paths", "require_non_null", "merge_annotations",
           "SEGMENT_KEY"]


def require_non_null(metadata: pd.DataFrame, columns: list[str], reason: str) -> None:
    """Raise if any of ``columns`` holds a null, naming which corpus and how many.

    Checked because a combined table NaN-fills every column that one of its
    corpora lacks, and each place that consumes such a column fails *silently*
    on the NaN rather than raising: an all-NaN label tensor, or a degenerate
    sampler.  Testing ``column in df.columns`` does not catch it — the column is
    present, its values are not.

    ``reason`` completes the sentence "…; <reason>", so the caller says what the
    null would have broken.
    """
    for column in columns:
        if column not in metadata.columns:
            raise ValueError(f"No '{column}' column in the metadata; {reason}")
        null = metadata[column].isna()
        if not null.any():
            continue
        where = ""
        if "corpus" in metadata.columns:
            counts = metadata.loc[null, "corpus"].value_counts()
            where = " — " + ", ".join(f"{name}: {n}" for name, n in counts.items())
        raise ValueError(
            f"{int(null.sum())} of {len(metadata)} row(s) have a null "
            f"'{column}'{where}; {reason}"
        )


#: What identifies a segment across tables.  The path alone is not enough — a
#: session file holds hundreds of segments — and ``segment_index`` is not enough
#: either, because it is assigned per table.
SEGMENT_KEY = ["signal_path", "start_s", "end_s"]


def merge_annotations(
    base: pd.DataFrame,
    annotations: dict[str, pd.DataFrame],
    key: list[str] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Fold hand annotations into the table every consumer reads.

    One combined table rather than one per split, because the split is already a
    column: a consumer selects ``split == "val"`` and gets annotated rows, and a
    future third annotated split changes nothing but this call.

    **A left join, never an inner one.**  The annotator runs with
    ``--task sentence``, so the paragraph segments of an annotated split are
    absent from its file — 74 of 909 validation rows and 128 of 1443 test rows in
    the AVID tables this was written for.  An inner join would silently delete
    them from the combined table, and they are legitimate training and
    measurement material that simply has no sentence structure to annotate.
    Unannotated rows keep empty annotation columns.

    An annotation row that matches nothing in ``base`` is an error, not a
    curiosity: it means the two tables were built from different indexes, or the
    split was re-drawn after the annotation, and every group key downstream would
    be quietly wrong.

    Parameters
    ----------
    base        : the combined table, all corpora and all splits.
    annotations : ``{name: frame}``; the name appears in the report only.
    key         : columns identifying a segment.  Defaults to :data:`SEGMENT_KEY`.

    Returns
    -------
    ``(merged, report)``.  ``report`` carries per-annotation match counts and the
    list of annotation columns added, for a caller to print or assert on.
    """
    key = list(key or SEGMENT_KEY)
    for column in key:
        if column not in base.columns:
            raise KeyError(f"base has no key column {column!r}")
    if base.duplicated(subset=key).any():
        n = int(base.duplicated(subset=key).sum())
        raise ValueError(
            f"base has {n} duplicate row(s) on {key}; the key does not identify "
            "a segment and a join on it would multiply rows."
        )

    # Aligned on the key as an index throughout: a positional boolean mask plus
    # a column assignment fights pandas' string dtypes, which refuse NaN, and
    # reindex expresses "the annotation's value where it has one" directly.
    merged = base.set_index(key)
    report: dict = {"annotations": {}, "added_columns": []}

    for name, frame in annotations.items():
        missing = [c for c in key if c not in frame.columns]
        if missing:
            raise KeyError(f"annotation {name!r} has no key column(s) {missing}")
        if frame.duplicated(subset=key).any():
            n = int(frame.duplicated(subset=key).sum())
            raise ValueError(
                f"annotation {name!r} has {n} duplicate row(s) on {key}."
            )

        indexed = frame.set_index(key)
        unknown = indexed.index.difference(merged.index)
        if len(unknown) == len(indexed):
            raise ValueError(
                f"annotation {name!r} matched no row of base on {key}. The two "
                "tables were built from different indexes, or the split was "
                "re-drawn after the annotation."
            )
        if len(unknown):
            raise ValueError(
                f"annotation {name!r} has {len(unknown)} row(s) matching nothing "
                f"in base, e.g. {list(unknown[:5])}. Every annotated segment must "
                "exist in base, or the annotation is keyed to a different index."
            )

        aligned = indexed.reindex(merged.index)
        new_columns = [c for c in frame.columns if c not in base.columns]
        shared = [c for c in frame.columns if c in base.columns and c not in key]

        for column in new_columns:
            incoming = aligned[column]
            if column in merged.columns:
                # A second annotation filling the same column on its own rows.
                merged[column] = incoming.where(incoming.notna(), merged[column])
            else:
                merged[column] = incoming
                report["added_columns"].append(column)

        # Shared columns: the annotation is the decision, so it wins on the rows
        # it covers.  Rows it does not cover keep whatever base had, which is what
        # protects corpora legitimately using the same column name — AVID's
        # `level` is empty in base while VIC's is meaningful, and VIC is never
        # matched here.
        conflicts = {}
        for column in shared:
            incoming = aligned[column]
            covered = incoming.notna()
            existing = merged[column]
            had = existing.notna() & (existing.astype("object").fillna("").astype(str)
                                      .str.strip() != "")
            differs = int(
                (covered & had & (existing.astype("object").astype(str)
                                  != incoming.astype("object").astype(str))).sum()
            )
            if differs:
                conflicts[column] = differs
            merged[column] = incoming.where(covered, existing)

        report["annotations"][name] = {
            "rows": int(len(indexed)),
            "matched": int(len(indexed)),
            "new_columns": new_columns,
            # Only the shared columns whose value actually moved.  Listing every
            # shared column is alarming and useless: the annotation files are
            # supersets of the base, so all 28 of them are "overwritten" with the
            # value they already had, which buries the one that changed.
            "changed_columns": conflicts,
            "unchanged_shared_columns": len(shared) - len(conflicts),
        }

    # An added column is NaN on every row the annotation did not cover, which
    # upcasts an integer column to float and serialises `1` as `1.0`.  That is not
    # cosmetic: `sentence_id` and `repetition` are group keys, and a config or a
    # hand-written selection saying `1` then matches nothing in a table saying
    # `1.0`.  Nullable Int64 keeps the integer and still holds the missing rows.
    for column in report["added_columns"]:
        values = merged[column].dropna()
        if values.empty:
            continue
        # Booleans must be left alone.  read_csv infers bool for a True/False
        # column and to_numeric(True) is 1, so an integral test accepts it and
        # rewrites the flag as 0/1 -- which silently breaks every consumer
        # comparing against "True", prepare_level_groups' exclusion included.
        if values.map(type).eq(bool).any():
            continue
        numeric = pd.to_numeric(values, errors="coerce")
        if numeric.notna().all() and (numeric == numeric.round()).all():
            merged[column] = pd.to_numeric(merged[column], errors="coerce").astype(
                "Int64"
            )
            report.setdefault("integer_columns", []).append(column)

    return merged.reset_index(), report

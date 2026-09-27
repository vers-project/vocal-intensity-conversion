"""Fold the hand annotations into the combined metadata table.

The effort and repetition annotations are produced one split at a time, by
``speech-annotator``, over the sentence segments of that split only.  Every
consumer — P_φ training, converter training, the render scripts, the
training-time monitor — wants one table instead: the split is already a column,
so a consumer selects the rows it needs and finds them annotated.

    all_corpora_prediction_metadata_with_labels.csv         22401 rows, no annotation
      + …_val_corrected_sentences.csv                         835 AVID val rows
      + …_directive_annotations_sentences_test_corrected.csv 1315 AVID test rows
      = one table, same 22401 rows, annotated where annotation exists

**The row count does not change.**  This is a left join: the annotator runs with
``--task sentence``, so an annotated split's *paragraph* segments are absent from
its file (74 of 909 validation rows, 128 of 1443 test rows).  They stay in the
output with empty annotation columns.  An inner join would delete them, and they
are legitimate training material that merely has no sentence structure to mark.

An annotated row matching nothing in the base table stops the run: it means the
two were built from different indexes, or the split was re-drawn after the
annotation, and every group key downstream would be quietly wrong.

Usage
-----
    uv run --extra cpu scripts/merge_annotations.py \\
        --config configs/paper/merge_annotations.yaml \\
        --output-dir outputs/metadata

Config
------
    base        : the combined table to fold into.
    annotations : ``{name: path}``.  The name appears only in the report.
    output      : where to write the merged table.  Relative paths land under
                  ``--output-dir``.
    key         : optional; columns identifying a segment.  Defaults to
                  ``signal_path, start_s, end_s``.
    group_columns / level_column : optional; if given, the report counts complete
                  level groups per split, which is the number the paired
                  evaluation can actually use.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from vic.data.metadata import SEGMENT_KEY, merge_annotations

from experiment_launcher import parse_args


def report_groups(
    merged: pd.DataFrame, group_columns: list[str], level_column: str
) -> None:
    """How many complete four-level groups each split ends up with.

    This is the number that matters: a group missing a level has destinations
    naming no real recording, so it cannot take part in the paired design.
    """
    missing = [c for c in group_columns if c not in merged.columns]
    if missing:
        print(f"  (cannot count groups: no column(s) {missing})")
        return
    # Every group column must be present on the row, not merely the level: VIC
    # fills `level` for all 1574 of its rows while carrying no sentence or
    # repetition, so a filter on level alone counts rows that can never form a
    # group and reports an inflated "usable" figure.
    usable = merged[
        (merged["excluded"].astype(str) != "True")
        & merged[level_column].notna()
        & (merged[level_column].astype(str).str.strip() != "")
        & merged[group_columns].notna().all(axis=1)
    ]
    print("\nComplete level groups, per split:")
    for split, part in usable.groupby("split"):
        sizes = part.groupby(group_columns).size()
        complete = int((sizes == 4).sum())
        print(f"  {split:6s} {len(part):6d} annotated+usable row(s)  "
              f"{len(sizes):5d} group(s)  {complete:5d} complete")


@parse_args
def main(config: dict, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    base = pd.read_csv(config["base"])
    print(f"base: {len(base)} row(s), {len(base.columns)} column(s) "
          f"from {config['base']}")

    annotations = {}
    for name, path in config["annotations"].items():
        frame = pd.read_csv(path)
        annotations[name] = frame
        print(f"  {name}: {len(frame)} row(s), {len(frame.columns)} column(s)")

    merged, report = merge_annotations(
        base, annotations, key=config.get("key", SEGMENT_KEY)
    )

    if len(merged) != len(base):
        raise RuntimeError(
            f"the merge changed the row count ({len(base)} -> {len(merged)}); "
            "a left join must not."
        )

    print("\nMerge report:")
    for name, entry in report["annotations"].items():
        print(f"  {name}: {entry['matched']}/{entry['rows']} matched")
        print(f"     added  : {', '.join(entry['new_columns']) or '-'}")
        changed = entry["changed_columns"]
        print(f"     changed: {changed or 'nothing'}"
              f"   ({entry['unchanged_shared_columns']} shared column(s) carried "
              f"a value identical to base's, or filled one base left empty)")

    # What a consumer will actually see.
    for column in report["added_columns"]:
        filled = int(merged[column].notna().sum())
        print(f"  {column:20s} filled on {filled}/{len(merged)} row(s)")

    if report.get("integer_columns"):
        print(f"  kept as integers (not 1.0): "
              f"{', '.join(report['integer_columns'])}")

    gc = config.get("group_columns")
    if gc:
        report_groups(merged, list(gc), config.get("level_column", "level"))

    out = Path(config["output"])
    if not out.is_absolute():
        out = output_dir / out
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out, index=False)
    print(f"\nWrote {out} ({len(merged)} rows, {len(merged.columns)} columns)")


if __name__ == "__main__":
    main()

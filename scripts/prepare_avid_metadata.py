"""Turn the AVID TextGrid segment index into converter- and predictor-training metadata.

Reads the ``segments_metadata.csv`` produced by ``dataset-metadata``'s
``avid/textgrid_segments.py`` → ``avid/add_metadata.py`` chain and adds the two
columns the training scripts need:

    corpus – always ``corpus_name`` (resolve_paths maps it to a dataset root)
    split  – "train", "val" or "test"

This is the **single source of AVID splits**.  P_φ and the converter must share
one partition, not merely one recipe: P_φ is the instrument every converter
metric is read through, so a converter test speaker that sat in P_φ's training
set has been seen by the ruler even though the converter never saw it.
``make_prediction_metadata.py`` therefore consumes this file with
``use_existing_split: true`` rather than re-drawing its own.

Nothing else is transformed.  ``subject_id``, ``sex``, ``channel`` and
``segment_duration_s`` are already correct in the input because the upstream
chain emits them that way.

The split is **speaker-disjoint and sex-stratified**; see
``vic/data/splits.py`` for why both properties are load-bearing.  With AVID's
50 speakers (25 M / 25 F) and ``{train: 40, val: 4, test: 6}`` every split is
exactly balanced: 20+20, 2+2, 3+3.  The uneven 4 / 6 is deliberate — an even
5 / 5 of the 10-speaker holdout cannot be sex-balanced on either side, whereas
4 = 2+2 and 6 = 3+3 both are.

No filtering is applied: every segment in the input reaches the output.  Dropping
``read_error`` rows, or the 60 s paragraph segments, is a later decision and is
deliberately left out until the diagnostics that would justify it have been run.

Why there is no ``channel`` guard here
--------------------------------------
AVID's ``channel`` is a genuine 0-based index (0 = speech mic, 1 = EGG) and
``AudioDataset`` **requires** it — mono-mixing the EGG into the speech is exactly
what it prevents.  Copying that guard would reject valid metadata.

YAML config keys
----------------
    segments_csv     : path to avid/add_metadata.py's segments_metadata.csv
    output_csv       : destination path
    corpus_name      : value for the corpus column (default: "avid")
    split_fractions  : mapping of split name → relative weight of speakers.
                       Weights are normalised, so speaker counts may be given
                       directly (default: train 40, val 4, test 6)
    random_seed      : RNG seed for the speaker assignment (default: 42)

Example config
--------------
    segments_csv: /path/to/AVID/AVID/segments_metadata.csv
    output_csv:   /path/to/AVID/AVID/avid_converter_metadata.csv
    corpus_name:  avid
    split_fractions:
      train: 40
      val:    4
      test:   6
    random_seed: 42

Usage
-----
    uv run --extra cpu python scripts/prepare_avid_metadata.py \\
        --config configs/paper/prepare_avid_metadata.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from vic.data.splits import stratified_speaker_split

# Written by the dataset-metadata AVID chain; a missing one means it has not been
# re-run since the columns settled.
REQUIRED_COLUMNS = [
    "signal_path", "start_s", "end_s", "segment_duration_s",
    "subject_id", "sex",
]

SPEAKER_COLUMN = "subject_id"
STRATIFY_COLUMN = "sex"

DEFAULT_FRACTIONS = {"train": 40, "val": 4, "test": 6}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build training metadata from the AVID TextGrid segment index."
    )
    parser.add_argument(
        "--config", required=True, type=Path,
        help="Path to the YAML configuration file.",
    )
    args = parser.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    segments_csv = Path(cfg["segments_csv"])
    output_path  = Path(cfg["output_csv"])
    corpus_name  = cfg.get("corpus_name", "avid")
    fractions    = cfg.get("split_fractions", DEFAULT_FRACTIONS)
    seed         = int(cfg.get("random_seed", 42))

    if not segments_csv.exists():
        raise SystemExit(f"Segment index not found: {segments_csv}")

    df = pd.read_csv(segments_csv)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise SystemExit(
            f"{segments_csv} is missing {missing}.\n"
            "Re-run dataset-metadata's avid/textgrid_segments.py and then "
            "avid/add_metadata.py, which is what emits 'sex' and the segment bounds."
        )

    df = df.drop(columns=["corpus", "split"], errors="ignore")
    df["corpus"] = corpus_name
    df["split"] = stratified_speaker_split(
        df, fractions, seed,
        speaker_column=SPEAKER_COLUMN, stratify_column=STRATIFY_COLUMN,
    )

    front = ["corpus", "split"]
    df = df[front + [c for c in df.columns if c not in front]]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    # --- report ---------------------------------------------------------------
    overlap = [
        s for s, n in df.groupby(SPEAKER_COLUMN)["split"].nunique().items() if n > 1
    ]
    if overlap:
        raise SystemExit(f"BUG: {len(overlap)} speaker(s) span several splits: {overlap[:5]}")

    speakers_per_split = df.groupby("split")[SPEAKER_COLUMN].nunique()

    print(f"Input   : {segments_csv}")
    print(f"Output  : {output_path}")
    print(f"Seed    : {seed}   fractions: {fractions}")
    print(f"\n{'split':6} {'subjects':>9} {'segments':>9} {'hours':>7} {'sex (subjects)':>20}")
    for split in fractions:
        part = df[df["split"] == split]
        sex_counts = part.drop_duplicates(SPEAKER_COLUMN)[STRATIFY_COLUMN].value_counts()
        sex_str = "  ".join(f"{k}:{v}" for k, v in sorted(sex_counts.items()))
        print(f"{split:6} {speakers_per_split.get(split, 0):9d} {len(part):9d} "
              f"{part['segment_duration_s'].sum() / 3600:7.2f} {sex_str:>20}")
    print(f"{'TOTAL':6} {df[SPEAKER_COLUMN].nunique():9d} {len(df):9d} "
          f"{df['segment_duration_s'].sum() / 3600:7.2f}")

    print("\nspeech hours by task and split:")
    hours = df.pivot_table(index="task", columns="split",
                           values="segment_duration_s", aggfunc="sum") / 3600
    print(hours.round(2).to_string())

    print("\nsegments by read_error and split:")
    print(df.pivot_table(index="read_error", columns="split",
                         values="segment_duration_s", aggfunc="size").to_string())

    # Printed because these are the speakers that get hand-annotated with
    # level/session by speech-annotator, and re-drawing the split later throws
    # that annotation away.
    speakers = df.drop_duplicates(SPEAKER_COLUMN)
    for split in ("val", "test"):
        ids = sorted(speakers.loc[speakers["split"] == split, SPEAKER_COLUMN])
        print(f"\n{split} speakers: {ids}")


if __name__ == "__main__":
    main()

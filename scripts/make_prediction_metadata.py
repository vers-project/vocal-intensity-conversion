"""Assemble the combined train/val/test metadata CSV for intensity prediction.

This file is the **single source of splits for both experiments** — the
intensity predictor P_φ and the converter that freezes P_φ inside its training
loop.  They must share one partition, not merely one recipe: P_φ is also the
instrument every converter metric is read through, so a converter test speaker
that sat in P_φ's training set has been seen by the ruler even though the
converter never saw it.  (Some converter configs still read AVID's own
``metadata.csv`` instead of this file; until they are repointed, the test split
below does not constrain those runs.)

AVID is the labelled training corpus and is split **three ways at the speaker
level**, sex-stratified: train, val, and a held-out test set used only for
objective evaluation of a trained converter.
With AVID's 50 speakers (25 M / 25 F) this yields **40 / 4 / 6**.  Every other
dataset listed under ``datasets`` is assigned to the test split.

The uneven 4 / 6 is deliberate.  Train takes 20 of each sex, leaving a holdout of
5 M + 5 F; an even 5 / 5 split of that holdout cannot be sex-balanced on either
side, whereas 4 = 2 + 2 and 6 = 3 + 3 both are.

Where the AVID split comes from
-------------------------------
For the **segment-format** AVID table (Repository 1 session recordings with
``start_s``/``end_s``), the split is drawn by ``prepare_avid_metadata.py`` and
this script honours it: set ``use_existing_split: true`` on the dataset entry.
That keeps one partition rather than two recipes, which is the whole point --
a split re-derived here from a differently ordered table would silently disagree
with the one the converter trained on.

``_gender_stratified_speaker_split`` below is the **old-format** path
(Repository 2, one file per sentence, columns ``speaker``/``gender``).  It
carves test out of the held-out remainder rather than sampling afresh, so the
training population stays bit-identical to the one the old checkpoints were
fitted on.  It applies only to a dataset entry without ``use_existing_split``.

Two columns are added to the combined output:
    corpus – name of the source dataset (key from the ``datasets`` config block)
    split  – "train", "val", or "test"

Any pre-existing ``corpus`` column is replaced, and so is ``split`` unless the
entry sets ``use_existing_split``.

YAML config keys
----------------
    output_csv     : destination path for the combined CSV
    datasets       : mapping of dataset name → dataset entry (below)
    train_fraction : fraction of AVID speakers assigned to train  (default: 0.8)
    val_speakers   : how many of the *held-out* AVID speakers go to val; the
                     rest go to test  (default: 4 → 4 val / 6 test)
    random_seed    : RNG seed for reproducible speaker assignment  (default: 42)

Dataset entry keys
------------------
    root               : dataset root (resolve_paths maps the corpus name to it)
    csv                : the corpus's metadata CSV
    use_existing_split : keep the CSV's own ``split`` column instead of drawing
                         one here  (default: false)
    speaker_column     : column holding the speaker id, for the printed summary
                         only  (default: "speaker"; the segment tables use
                         "subject_id")

**When a split is drawn here, the input CSV and its row order are part of its
definition.** Speakers are shuffled in the order they first appear, so deriving
the split from a differently ordered AVID table reassigns them.  Compute it
once, from the CSV named here, and let everything downstream read the result.
``prepare_avid_metadata.py`` does not have this property -- it sorts before
shuffling -- which is one more reason to prefer ``use_existing_split``.

Example config
--------------
    output_csv: /path/to/prediction_metadata.csv

    datasets:
      AVID:
        root: /path/to/AVID
        csv:  /path/to/avid_converter_metadata_with_labels.csv
        use_existing_split: true      # drawn by prepare_avid_metadata.py
        speaker_column: subject_id
      FLombard:
        root: /path/to/FLombard
        csv:  /path/to/flombard_metadata.csv
      VIC:
        root: /path/to/VIC
        csv:  /path/to/vic_metadata.csv

    train_fraction: 0.8
    val_speakers: 4
    random_seed: 42

Usage
-----
    uv run python scripts/make_prediction_metadata.py \\
        --config configs/paper/make_prediction_metadata.yaml
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

_TRAIN_VAL_DATASET = "AVID"


def _apportion(sizes: dict, total: int) -> dict:
    """Split ``total`` items across groups in proportion to ``sizes``.

    Largest-remainder apportionment, so the parts sum to ``total`` exactly
    instead of drifting by a speaker or two after rounding each group
    independently.  Ties break on the string form of the group key, which keeps
    the outcome reproducible rather than dependent on dict ordering.
    """
    population = sum(sizes.values())
    if population == 0:
        return {key: 0 for key in sizes}

    exact = {key: total * size / population for key, size in sizes.items()}
    quota = {key: int(math.floor(value)) for key, value in exact.items()}
    order = sorted(sizes, key=lambda key: (-(exact[key] - quota[key]), str(key)))
    for key in order[: total - sum(quota.values())]:
        quota[key] += 1
    return quota


def _gender_stratified_speaker_split(
    df: pd.DataFrame,
    train_fraction: float,
    val_speakers: int,
    seed: int,
) -> pd.Series:
    """Return a Series mapping each row to 'train', 'val' or 'test'.

    Splits at the speaker level.  Speakers are grouped by gender so that each
    gender contributes proportionally to all three splits.  Speakers with no
    gender information form a separate group.

    Train is derived exactly as the original two-way split did — same seed, same
    per-gender shuffle, same ``ceil(train_fraction · n)`` prefix — and only the
    held-out remainder is subdivided.  Keeping that prefix intact is what lets a
    test split be introduced without moving any speaker into or out of the
    population that existing checkpoints were fitted on.
    """
    rng = np.random.default_rng(seed)
    speakers = (
        df[["speaker", "gender"]]
        .drop_duplicates(subset="speaker")
        .reset_index(drop=True)
    )

    split_map: dict = {}
    holdout: dict = {}
    for gender, group in speakers.groupby("gender", dropna=False):
        ids = group["speaker"].tolist()
        rng.shuffle(ids)
        n_train = math.ceil(train_fraction * len(ids))
        for spk in ids[:n_train]:
            split_map[spk] = "train"
        holdout[gender] = ids[n_train:]

    n_holdout = sum(len(ids) for ids in holdout.values())
    if not 0 <= val_speakers <= n_holdout:
        raise SystemExit(
            f"val_speakers={val_speakers} but only {n_holdout} speaker(s) are held "
            f"out of train (train_fraction={train_fraction})."
        )
    val_quota = _apportion({g: len(ids) for g, ids in holdout.items()}, val_speakers)

    for gender, ids in holdout.items():
        for spk in ids[: val_quota[gender]]:
            split_map[spk] = "val"
        for spk in ids[val_quota[gender]:]:
            split_map[spk] = "test"

    return df["speaker"].map(split_map)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build combined train/val/test metadata for intensity experiments."
    )
    parser.add_argument(
        "--config", required=True, type=Path,
        help="Path to the YAML configuration file.",
    )
    args = parser.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    train_fraction = float(cfg.get("train_fraction", 0.8))
    val_speakers = int(cfg.get("val_speakers", 4))
    seed = int(cfg.get("random_seed", 42))

    # Kept per corpus, not just concatenated: after the concat every column
    # missing from one corpus is NaN-filled there, which silently turns integer
    # speaker ids into floats and makes a column one corpus lacks look present.
    # The summary below is only honest if it reads each corpus's own frame.
    frames: dict[str, pd.DataFrame] = {}
    speaker_columns: dict[str, str] = {}

    for name, ds_cfg in cfg["datasets"].items():
        df = pd.read_csv(ds_cfg["csv"])
        speaker_column = ds_cfg.get("speaker_column", "speaker")
        speaker_columns[name] = speaker_column

        if speaker_column not in df.columns:
            raise SystemExit(
                f"{name}: no '{speaker_column}' column in {ds_cfg['csv']}.  Set "
                "speaker_column on the dataset entry; the segment tables use "
                "'subject_id'."
            )

        if ds_cfg.get("use_existing_split", False):
            if "split" not in df.columns:
                raise SystemExit(
                    f"{name}: use_existing_split is set but {ds_cfg['csv']} has no "
                    "'split' column.  Run prepare_avid_metadata.py first."
                )
            df = df.drop(columns=["corpus"], errors="ignore")
            df["corpus"] = name
        else:
            df = df.drop(columns=["corpus", "split"], errors="ignore")
            df["corpus"] = name
            if name == _TRAIN_VAL_DATASET:
                df["split"] = _gender_stratified_speaker_split(
                    df, train_fraction, val_speakers, seed
                )
            else:
                df["split"] = "test"

        if df["split"].isna().any():
            raise SystemExit(
                f"{name}: {int(df['split'].isna().sum())} row(s) ended with no split.  "
                "Refusing to write a table that would silently drop them from every "
                "train/val/test filter downstream."
            )

        # Speaker ids are only unique *within* a corpus -- AVID's subject_id 1-50
        # and FLombard's speaker 1-38 overlap numerically while meaning different
        # people.  Anything that groups across corpora (the per-speaker
        # regressions of the evaluation) must group on this instead.
        df["speaker_uid"] = name + ":" + df[speaker_column].astype(str)

        frames[name] = df

    combined = pd.concat(frames.values(), ignore_index=True)

    front = ["corpus", "split", "speaker_uid"]
    rest = [c for c in combined.columns if c not in front]
    combined = combined[front + rest]

    output_path = Path(cfg["output_csv"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)

    print(f"Wrote {len(combined)} rows → {output_path}")

    # Counted per corpus rather than in one groupby: the corpora do not agree on
    # what the speaker column is called (Repository 2 says "speaker", the segment
    # tables say "subject_id"), and there is no reason to rename a whole corpus's
    # columns for a summary line.
    print(f"\n{'corpus':10} {'split':6} {'rows':>8} {'speakers':>9}")
    for name, frame in frames.items():
        speaker_column = speaker_columns[name]
        for split, part in frame.groupby("split"):
            print(f"{name:10} {split:6} {len(part):8d} "
                  f"{part[speaker_column].nunique():9d}")

    # Which columns each corpus does NOT have.  After the concat they are all
    # present and NaN-filled, so this is the only place the asymmetry is visible
    # -- and a column one corpus lacks is exactly what breaks a reader that
    # decides by column presence rather than per row.
    print("\ncolumns a corpus does not carry (NaN-filled in the output):")
    union = list(combined.columns)
    for name, frame in frames.items():
        absent = [c for c in union if c not in frame.columns]
        print(f"  {name:10} {', '.join(absent) if absent else '(none)'}")

    labelling = [c for c in ("calibration_rms", "intensity_db") if c in union]
    for column in labelling:
        null = combined[column].isna()
        if null.any():
            counts = combined.loc[null, "corpus"].value_counts()
            print(f"  → '{column}' is null for "
                  f"{', '.join(f'{k}: {v}' for k, v in counts.items())}; "
                  "run compute_labels.py over this table before training.")

    # Stratification is the claim worth checking, and it is only visible per
    # speaker: row counts happily hide an all-male test set behind a plausible
    # total.  The test speaker list is printed because it is what gets
    # hand-annotated with sentence_id/level/session, and re-drawing it later
    # throws that annotation away.
    avid = frames.get(_TRAIN_VAL_DATASET)
    if avid is None:
        return
    speaker_column = speaker_columns[_TRAIN_VAL_DATASET]
    # Read off AVID's own frame, so a sex column that only another corpus carries
    # cannot be picked here and reported as all-NaN.
    sex_column = next((c for c in ("gender", "sex") if c in avid.columns), None)
    if not avid.empty and sex_column is not None:
        speakers = avid.drop_duplicates(speaker_column)
        print(f"\n{_TRAIN_VAL_DATASET} speakers by split and {sex_column}:")
        print(
            speakers.groupby(["split", sex_column], dropna=False)
            .size().rename("n_speakers").to_string()
        )
        for split in ("val", "test"):
            ids = sorted(speakers.loc[speakers["split"] == split, speaker_column])
            print(f"\n{_TRAIN_VAL_DATASET} {split} speakers: {ids}")


if __name__ == "__main__":
    main()

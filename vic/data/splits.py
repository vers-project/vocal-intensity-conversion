"""Speaker-disjoint, stratified assignment of a segment table to splits.

Used by ``prepare_avid_metadata.py``.  The input is one row per speech segment,
with a ``subject_id`` and a ``sex``.

**Speaker-disjoint** because segments from one recording are hugely correlated:
splitting at the segment level leaks a speaker's voice into validation and makes
the score meaningless.  **Stratified** because a split's sex balance is only
visible per speaker -- row counts happily hide an all-male test set behind a
plausible total.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def stratified_speaker_split(
    df: pd.DataFrame,
    fractions: dict[str, float],
    seed: int,
    speaker_column: str = "subject_id",
    stratify_column: str = "sex",
) -> pd.Series:
    """Assign whole speakers to splits, keeping each stratum proportional.

    Returns a Series mapping each row of ``df`` to a split name.  Speakers with
    no value in the stratify column form their own group rather than being
    dropped.

    ``fractions`` are relative weights, normalised here, so a caller may state
    speaker counts directly -- ``{"train": 40, "val": 4, "test": 6}`` -- which is
    more legible than the equivalent 0.8 / 0.08 / 0.12 and gives the same answer.
    """
    rng = np.random.default_rng(seed)
    speakers = df[[speaker_column, stratify_column]].drop_duplicates(subset=speaker_column)

    names = list(fractions)
    weights = np.array([fractions[n] for n in names], dtype=float)
    weights /= weights.sum()

    split_map: dict = {}
    for _, group in speakers.groupby(stratify_column, dropna=False):
        # Sorted before shuffling so the result depends only on the seed and
        # not on the row order of the input CSV.
        ids = sorted(group[speaker_column].tolist())
        rng.shuffle(ids)

        # Largest-remainder allocation, so the per-split counts sum to exactly
        # len(ids) instead of losing or duplicating a speaker to rounding.
        exact = weights * len(ids)
        counts = np.floor(exact).astype(int)
        for i in np.argsort(-(exact - counts))[: len(ids) - counts.sum()]:
            counts[i] += 1

        start = 0
        for name, count in zip(names, counts):
            for speaker in ids[start : start + count]:
                split_map[speaker] = name
            start += count

    return df[speaker_column].map(split_map)

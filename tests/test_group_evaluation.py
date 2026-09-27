"""Tests for selecting the sentence groups a paired evaluation runs on.

The regression these protect against was found on the cluster: taking the first
N sorted group ids put all 24 validation groups on one speaker, because a
``sentence_group_id`` begins with the speaker.  The visible symptom was the
speaker metric raising ``0 nontarget``; the quieter and worse one was that every
per-speaker slope was fitted within a single group.
"""
from __future__ import annotations

import pandas as pd
import pytest

from vic.training.build import _take_groups


def rows(speakers=("s1", "s2", "s3", "s4"), per_speaker=10) -> pd.DataFrame:
    """Four levels per group, group ids prefixed by the speaker as the real ones are."""
    out = []
    for speaker in speakers:
        for g in range(per_speaker):
            for level in range(4):
                out.append({
                    "speaker_uid": speaker,
                    "sentence_group_id": f"{speaker}|1|1|{g}",
                    "level": level,
                })
    return pd.DataFrame(out)


def test_groups_are_spread_over_speakers_not_taken_in_sorted_order():
    sel = _take_groups(rows(), 24, "speaker_uid")
    per = sel.groupby("speaker_uid")["sentence_group_id"].nunique().to_dict()
    assert per == {"s1": 6, "s2": 6, "s3": 6, "s4": 6}


def test_the_sorted_prefix_would_have_used_one_speaker():
    """Pins the bug itself, so the fix cannot be quietly reverted.

    50 groups per speaker, as the real validation split has, so that a 24-group
    prefix fits entirely inside the alphabetically first speaker -- which is
    exactly what happened on the cluster.
    """
    frame = rows(per_speaker=50)
    ids = sorted(frame["sentence_group_id"].unique())
    prefix = frame[frame["sentence_group_id"].isin(ids[:24])]
    assert prefix["speaker_uid"].nunique() == 1
    # and the fix, on the same frame
    assert _take_groups(frame, 24, "speaker_uid")["speaker_uid"].nunique() == 4


def test_whole_groups_are_kept_so_no_level_is_orphaned():
    sel = _take_groups(rows(), 12, "speaker_uid")
    assert (sel.groupby("sentence_group_id").size() == 4).all()
    assert len(sel) == 48


def test_the_selection_does_not_depend_on_row_order():
    frame = rows()
    a = set(_take_groups(frame, 13, "speaker_uid")["sentence_group_id"])
    shuffled = frame.sample(frac=1, random_state=7).reset_index(drop=True)
    b = set(_take_groups(shuffled, 13, "speaker_uid")["sentence_group_id"])
    assert a == b


def test_fewer_groups_than_speakers_still_covers_distinct_speakers():
    sel = _take_groups(rows(), 3, "speaker_uid")
    assert sel["speaker_uid"].nunique() == 3


def test_asking_for_more_groups_than_exist_returns_all_of_them():
    frame = rows(per_speaker=2)
    sel = _take_groups(frame, 999, "speaker_uid")
    assert sel["sentence_group_id"].nunique() == 8


def test_a_missing_stratification_column_warns_rather_than_failing_silently():
    frame = rows().drop(columns=["speaker_uid"])
    with pytest.warns(RuntimeWarning, match="one speaker"):
        sel = _take_groups(frame, 4, "speaker_uid")
    assert sel["sentence_group_id"].nunique() == 4


def test_uneven_speakers_do_not_starve_the_smaller_ones():
    frame = pd.concat([rows(("s1",), per_speaker=20), rows(("s2",), per_speaker=2)],
                      ignore_index=True)
    sel = _take_groups(frame, 10, "speaker_uid")
    per = sel.groupby("speaker_uid")["sentence_group_id"].nunique().to_dict()
    assert per["s2"] == 2, "the speaker with fewer groups must contribute all of them"
    assert per["s1"] == 8

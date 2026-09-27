"""Tests for folding hand annotations into the metadata table.

Three of these guard failures that would be silent in the data rather than loud
at runtime: dropping the unannotated rows, coercing a boolean flag to 0/1, and
clobbering a column another corpus legitimately uses under the same name.
"""
from __future__ import annotations

import pandas as pd
import pytest

from vic.data.metadata import SEGMENT_KEY, merge_annotations


def base_table() -> pd.DataFrame:
    """Two AVID sentence rows, one AVID paragraph row, one VIC row.

    The paragraph row stands for what the annotator never sees (it runs with
    ``--task sentence``); the VIC row stands for a corpus that fills ``level``
    with its own vocabulary.
    """
    return pd.DataFrame([
        {"signal_path": "a.wav", "start_s": 0.0, "end_s": 1.0, "corpus": "AVID",
         "split": "val", "task": "sentence", "level": None},
        {"signal_path": "a.wav", "start_s": 1.0, "end_s": 2.0, "corpus": "AVID",
         "split": "val", "task": "sentence", "level": None},
        {"signal_path": "a.wav", "start_s": 2.0, "end_s": 9.0, "corpus": "AVID",
         "split": "val", "task": "paragraph", "level": None},
        {"signal_path": "v.wav", "start_s": 0.0, "end_s": 1.0, "corpus": "VIC",
         "split": "test", "task": "sentence", "level": "1b"},
    ])


def annotation() -> pd.DataFrame:
    """The two AVID sentence rows, annotated.  Note the paragraph is absent."""
    return pd.DataFrame([
        {"signal_path": "a.wav", "start_s": 0.0, "end_s": 1.0, "corpus": "AVID",
         "split": "val", "task": "sentence", "level": "soft",
         "excluded": False, "sentence_id": 1, "repetition": 1},
        {"signal_path": "a.wav", "start_s": 1.0, "end_s": 2.0, "corpus": "AVID",
         "split": "val", "task": "sentence", "level": "normal",
         "excluded": False, "sentence_id": 1, "repetition": 1},
    ])


# ---------------------------------------------------------------------------
# The join keeps every row
# ---------------------------------------------------------------------------


def test_unannotated_rows_survive_with_empty_annotation_columns():
    """A left join, never an inner one: the paragraph row is real material."""
    merged, _ = merge_annotations(base_table(), {"val": annotation()})
    assert len(merged) == 4
    paragraph = merged[merged["task"] == "paragraph"].iloc[0]
    assert pd.isna(paragraph["sentence_id"])
    assert pd.isna(paragraph["excluded"])


def test_the_annotation_columns_are_added_and_filled_where_matched():
    merged, report = merge_annotations(base_table(), {"val": annotation()})
    assert set(report["added_columns"]) == {"excluded", "sentence_id", "repetition"}
    annotated = merged[merged["level"].isin(["soft", "normal"])]
    assert len(annotated) == 2
    assert annotated["sentence_id"].tolist() == [1, 1]


def test_row_order_and_key_columns_are_preserved():
    base = base_table()
    merged, _ = merge_annotations(base, {"val": annotation()})
    assert merged[SEGMENT_KEY].equals(base[SEGMENT_KEY])


# ---------------------------------------------------------------------------
# A column another corpus owns is not clobbered
# ---------------------------------------------------------------------------


def test_a_shared_column_is_untouched_on_rows_the_annotation_does_not_cover():
    """AVID's `level` is empty in base while VIC's is meaningful, and the
    annotation covers no VIC row -- so VIC must come through unchanged."""
    merged, _ = merge_annotations(base_table(), {"val": annotation()})
    vic = merged[merged["corpus"] == "VIC"].iloc[0]
    assert vic["level"] == "1b"


def test_replacing_a_non_empty_base_value_is_reported():
    base = base_table()
    base.loc[0, "level"] = "WRONG"
    _, report = merge_annotations(base, {"val": annotation()})
    assert report["annotations"]["val"]["changed_columns"] == {"level": 1}


# ---------------------------------------------------------------------------
# Types: integers stay integers, booleans stay booleans
# ---------------------------------------------------------------------------


def test_integer_group_keys_do_not_become_floats():
    """`sentence_id` is a group key. NaN on the unannotated rows upcasts it to
    float and serialises 1 as 1.0, which then matches no config saying 1."""
    merged, report = merge_annotations(base_table(), {"val": annotation()})
    assert str(merged["sentence_id"].dtype) == "Int64"
    assert "sentence_id" in report["integer_columns"]
    assert merged["sentence_id"].dropna().tolist() == [1, 1]


def test_boolean_flags_are_not_coerced_to_zero_and_one():
    """to_numeric(True) is 1, so an integral test accepts a bool column and
    rewrites the flag -- silently breaking every consumer comparing to "True",
    prepare_level_groups' exclusion included."""
    merged, report = merge_annotations(base_table(), {"val": annotation()})
    assert "excluded" not in report.get("integer_columns", [])
    assert set(merged["excluded"].dropna().unique()) == {False}
    assert str(merged["excluded"].dtype) != "Int64"


# ---------------------------------------------------------------------------
# Key drift is an error, not a curiosity
# ---------------------------------------------------------------------------


def test_an_annotation_row_matching_nothing_is_refused():
    rogue = annotation()
    rogue.loc[0, "start_s"] = 99.0
    with pytest.raises(ValueError, match="matching nothing"):
        merge_annotations(base_table(), {"val": rogue})


def test_an_annotation_matching_nothing_at_all_names_the_likely_cause():
    rogue = annotation()
    rogue["signal_path"] = "elsewhere.wav"
    with pytest.raises(ValueError, match="re-drawn|different indexes"):
        merge_annotations(base_table(), {"val": rogue})


def test_duplicate_keys_in_base_are_refused_before_they_multiply_rows():
    base = pd.concat([base_table(), base_table().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        merge_annotations(base, {"val": annotation()})


def test_duplicate_keys_in_the_annotation_are_refused():
    dupe = pd.concat([annotation(), annotation().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        merge_annotations(base_table(), {"val": dupe})


def test_a_missing_key_column_is_named():
    with pytest.raises(KeyError, match="end_s"):
        merge_annotations(base_table(), {"val": annotation().drop(columns=["end_s"])})


# ---------------------------------------------------------------------------
# Several annotations at once
# ---------------------------------------------------------------------------


def test_two_annotations_each_fill_their_own_rows():
    base = base_table()
    second = pd.DataFrame([{
        "signal_path": "v.wav", "start_s": 0.0, "end_s": 1.0, "corpus": "VIC",
        "split": "test", "task": "sentence", "level": "loud",
        "excluded": False, "sentence_id": 7, "repetition": 2,
    }])
    merged, report = merge_annotations(base, {"val": annotation(), "test": second})
    assert report["annotations"]["val"]["matched"] == 2
    assert report["annotations"]["test"]["matched"] == 1
    assert merged["sentence_id"].dropna().tolist() == [1, 1, 7]
    # and the paragraph row is still unannotated
    assert merged["sentence_id"].isna().sum() == 1

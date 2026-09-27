"""Tests for the preservation summaries hoisted out of ``analyze_conversion.py``.

They had no coverage while they lived in a script.  They are now library code with
two callers — the final analysis and the training-time monitor — so the properties
that make them correct are worth pinning, in particular the two that are silent
when broken: pooling word errors by count rather than by rate, and resolving a
codec row's missing reference without tripping over NaN being truthy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vic.evaluation.analysis import speaker_preservation, wer_preservation
from vic.evaluation.conditions import (
    CODEC,
    CONVERTED,
    REAL,
    REFERENCE_UTT_ID,
    SOURCE_REAL_UTT_ID,
)


# ---------------------------------------------------------------------------
# Word error rate
# ---------------------------------------------------------------------------


LONG = "the quick brown fox jumps over"   # six words after normalisation
SHORT = "cat"                               # one word, transcribed wrongly


def wer_frame() -> pd.DataFrame:
    """Two conditions, with utterances of deliberately unequal reference length.

    The long utterance is clean and the short one is wrong, so a mean of
    per-utterance rates (0.5) and a count-pooled rate (1/7) differ.

    Deliberately *not* number words: the default normaliser is Whisper's
    ``EnglishTextNormalizer``, which concatenates a spoken digit sequence into one
    token -- "one two three four five six" normalises to "123456", a single word --
    so number words would make the reference length unrecognisable.
    """
    return pd.DataFrame([
        {"utt_id": "a_codec", "condition": CODEC, "text": LONG,
         "asr.hypothesis": LONG},
        {"utt_id": "b_codec", "condition": CODEC, "text": SHORT,
         "asr.hypothesis": "dog"},
        {"utt_id": "a_conv", "condition": CONVERTED, "text": LONG,
         "asr.hypothesis": LONG},
        {"utt_id": "b_conv", "condition": CONVERTED, "text": SHORT,
         "asr.hypothesis": "dog"},
    ])


def test_wer_is_pooled_by_count_not_averaged_over_utterances():
    out = wer_preservation(wer_frame())
    codec = out.loc[out["condition"] == CODEC, "wer"].iloc[0]
    # 1 error over 7 reference words, not the mean of 0.0 and 1.0.
    assert codec == pytest.approx(1 / 7)
    assert codec != pytest.approx(0.5)


def test_delta_is_measured_against_the_codec_anchor():
    out = wer_preservation(wer_frame())
    assert (out.loc[out["condition"] == CODEC, "delta_vs_codec"] == 0.0).all()
    converted = out.loc[out["condition"] == CONVERTED]
    assert converted["delta_vs_codec"].iloc[0] == pytest.approx(0.0)


def test_one_row_per_backend_and_condition():
    frame = wer_frame()
    frame["other.hypothesis"] = frame["asr.hypothesis"]
    out = wer_preservation(frame)
    assert set(out["backend"]) == {"asr", "other"}
    assert len(out) == 4


def test_no_asr_column_yields_an_empty_frame_rather_than_raising():
    frame = wer_frame().drop(columns=["asr.hypothesis"])
    assert wer_preservation(frame).empty


def test_a_missing_codec_condition_gives_nan_delta_not_an_exception():
    frame = wer_frame()
    frame = frame[frame["condition"] != CODEC]
    out = wer_preservation(frame)
    assert out["delta_vs_codec"].isna().all()


# ---------------------------------------------------------------------------
# Speaker identity
# ---------------------------------------------------------------------------


def speaker_fixture() -> tuple[pd.DataFrame, dict]:
    """Two speakers with two real recordings each, plus a codec and a converted row.

    The codec row deliberately has no ``reference_utt_id``: it has no target of its
    own and must be read against the source it came from.
    """
    rng = np.random.default_rng(0)

    def vector(base: np.ndarray) -> np.ndarray:
        v = base + 0.05 * rng.standard_normal(8)
        return (v / np.linalg.norm(v)).astype("float32")

    centre = {"s1": rng.standard_normal(8), "s2": rng.standard_normal(8)}
    rows, arrays = [], {}

    def add(utt, speaker, condition, reference, source):
        rows.append({
            "utt_id": utt, "speaker_uid": speaker, "condition": condition,
            REFERENCE_UTT_ID: reference, SOURCE_REAL_UTT_ID: source,
        })
        arrays[f"{utt}|spk.embedding"] = vector(centre[speaker])

    for speaker in ("s1", "s2"):
        for i in (1, 2):
            add(f"{speaker}_r{i}", speaker, REAL, None, f"{speaker}_r1")
    add("s1_codec", "s1", CODEC, None, "s1_r1")
    add("s1_conv", "s1", CONVERTED, "s1_r2", "s1_r1")
    return pd.DataFrame(rows), arrays


def test_reports_one_row_per_embedder_and_condition():
    metrics, arrays = speaker_fixture()
    out = speaker_preservation(metrics, arrays, group_col="speaker_uid")
    assert set(out["embedder"]) == {"spk"}
    assert set(out["condition"]) == {CODEC, CONVERTED}


def test_a_codec_row_falls_back_to_its_source_because_nan_is_truthy():
    """The documented trap: ``reference or source`` keeps a float NaN and looks up
    the key ``"nan|spk.embedding"``, which is absent, so every codec similarity
    would be NaN while looking like a legitimate missing measurement."""
    metrics, arrays = speaker_fixture()
    out = speaker_preservation(metrics, arrays, group_col="speaker_uid")
    codec = out.loc[out["condition"] == CODEC].iloc[0]
    assert codec["n"] == 1, "the codec row resolved to no reference at all"
    assert np.isfinite(codec["cosine_mean"])


def test_the_calibration_scale_comes_with_its_floor_and_ceiling():
    metrics, arrays = speaker_fixture()
    out = speaker_preservation(metrics, arrays, group_col="speaker_uid")
    row = out.iloc[0]
    for column in ("ceiling_mu_target", "floor_mu_nontarget", "d_prime"):
        assert np.isfinite(row[column]), f"{column} missing"
    # A raw cosine is uninterpretable without these two, which is why both are kept.
    assert row["ceiling_mu_target"] > row["floor_mu_nontarget"]


def test_an_embedding_absent_from_the_archive_is_nan_not_a_key_error():
    metrics, arrays = speaker_fixture()
    del arrays["s1_conv|spk.embedding"]
    out = speaker_preservation(metrics, arrays, group_col="speaker_uid")
    converted = out.loc[out["condition"] == CONVERTED].iloc[0]
    assert converted["n"] == 0
    assert np.isnan(converted["cosine_mean"])


def test_no_embeddings_at_all_yields_an_empty_frame():
    metrics, _ = speaker_fixture()
    assert speaker_preservation(metrics, {}, group_col="speaker_uid").empty


def test_a_single_speaker_subset_is_skipped_not_fatal():
    """One speaker means only same-speaker pairs, so no scale can be built.

    `calibrate` rightly raises there. A training-time monitor must not die of it,
    which is what happened on the cluster before the group selection was
    stratified: every other measure was lost with it.
    """
    metrics, arrays = speaker_fixture()
    one = metrics[metrics["speaker_uid"] == "s1"].reset_index(drop=True)
    kept = {k: v for k, v in arrays.items() if k.startswith("s1_")}
    with pytest.warns(RuntimeWarning, match="one class"):
        out = speaker_preservation(one, kept, group_col="speaker_uid")
    assert out.empty, "the embedder is skipped, and nothing else raises"

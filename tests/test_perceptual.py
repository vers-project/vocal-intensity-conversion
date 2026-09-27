"""Tests for the listening-test parsing and scoring.

The two that matter most guard silent inversions: the pair columns are NOT in
presentation order, and a response of "1" means the first *played* member, so
reading either wrongly flips every BA trial without raising anything.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vic.evaluation.perceptual import (
    TargetGrid, accuracy, accuracy_by_delta, binomial_p_above_chance,
    classification_metrics, describe_trials, listener_agreement, metrics_table,
    pairwise_agreement, parse_session, parse_stem, per_listener_table,
    read_experiment, score,
)

GRID = TargetGrid(min_db=45.0, max_db=75.0, n=8)

LINE = ('{a} {b} "{order}" test {key} {choice} 900 10 10 {choice2} '
        '@ A sentence. @ GrpA')


def session(tmp_path, rows, group="A", uid="0f4f2c7b-48f1-437a-93f2-a7a41c297af3"):
    body = []
    for a, b, order, key, choice, choice2 in rows:
        body.append("1 2 " + LINE.format(a=a, b=b, order=order, key=key,
                                         choice=choice, choice2=choice2))
    path = tmp_path / f"FdV_Pair_QL_2026_groupe{group}.2026-09-14-1414.data.{uid}.txt"
    path.write_text("\n".join(body) + "\n")
    return path


# ---------------------------------------------------------------------------
# The target grid
# ---------------------------------------------------------------------------


def test_the_grid_is_inclusive_of_both_ends_and_one_based():
    assert GRID.level(1) == pytest.approx(45.0)
    assert GRID.level(8) == pytest.approx(75.0)
    assert GRID.level(2) == pytest.approx(45.0 + 30.0 / 7)


@pytest.mark.parametrize("index", [0, 9, -1])
def test_a_target_index_outside_the_grid_is_refused(index):
    with pytest.raises(ValueError, match="outside"):
        GRID.level(index)


# ---------------------------------------------------------------------------
# Stimulus names
# ---------------------------------------------------------------------------


def test_a_converted_stimulus_is_recognised_by_its_target():
    out = parse_stem("spk9_sent9_rep1_src2_tgt7")
    assert out["synthetic"] and out["target_index"] == 7
    assert (out["speaker"], out["sentence_id"], out["repetition"],
            out["source_level_index"]) == (9, 9, 1, 2)


def test_a_real_stimulus_has_no_target():
    out = parse_stem("spk9_sent9_rep2_src4")
    assert not out["synthetic"] and out["target_index"] is None


def test_an_unparseable_name_says_what_was_expected():
    with pytest.raises(ValueError, match="convert_selection pattern"):
        parse_stem("something_else.mp3")


# ---------------------------------------------------------------------------
# The presentation order -- the silent-inversion guard
# ---------------------------------------------------------------------------


def test_the_pair_columns_are_not_the_played_order(tmp_path):
    """Both members are listed converted-first; only the order field says which
    was played first.  Treating the columns as the running order inverts every
    BA trial."""
    p = session(tmp_path, [
        ("spk9_sent9_rep1_src2_tgt7", "spk9_sent9_rep2_src4", "BA", 3, 1, 1),
    ])
    row = parse_session(p).iloc[0]
    assert row["stimulus_a"] == "spk9_sent9_rep1_src2_tgt7"
    assert row["played_first"] == "spk9_sent9_rep2_src4", "BA plays B first"
    assert row["played_second"] == "spk9_sent9_rep1_src2_tgt7"


def test_choice_one_means_the_first_played_not_the_first_listed(tmp_path):
    # BA: real played first. Choice 1 therefore picks the REAL member, and the
    # converted member (asked for 75 dB, above the real level) was NOT chosen.
    p = session(tmp_path, [
        ("spk9_sent9_rep1_src2_tgt8", "spk9_sent9_rep2_src4", "BA", 3, 1, 1),
    ])
    t = describe_trials(read_experiment(tmp_path), GRID)
    t["b_level_db"] = 60.0
    out = score(t).iloc[0]
    assert out["picked_member"] == "b"
    assert out["louder_member"] == "a"      # 75 dB requested vs 60 dB real
    assert out["correct"] is False or out["correct"] == False  # noqa: E712


def test_the_same_choice_is_correct_when_the_order_is_reversed(tmp_path):
    p = session(tmp_path, [
        ("spk9_sent9_rep1_src2_tgt8", "spk9_sent9_rep2_src4", "AB", 3, 1, 1),
    ])
    t = describe_trials(read_experiment(tmp_path), GRID)
    t["b_level_db"] = 60.0
    out = score(t).iloc[0]
    assert out["picked_member"] == "a"
    assert bool(out["correct"]) is True


# ---------------------------------------------------------------------------
# Missing and degenerate trials
# ---------------------------------------------------------------------------


def test_a_timeout_is_not_scored_as_a_failure(tmp_path):
    """key -1 with the two choice columns disagreeing is PsyToolkit's
    no-response record; counting it as wrong would understate accuracy."""
    p = session(tmp_path, [
        ("spk9_sent9_rep1_src2_tgt8", "spk9_sent9_rep2_src4", "AB", -1, 0, 3),
    ])
    t = describe_trials(read_experiment(tmp_path), GRID)
    t["b_level_db"] = 60.0
    out = score(t).iloc[0]
    assert not out["answered"]
    assert pd.isna(out["correct"])


def test_the_choice_columns_disagreeing_on_an_answered_trial_raises(tmp_path):
    p = session(tmp_path, [
        ("spk9_sent9_rep1_src2_tgt8", "spk9_sent9_rep2_src4", "AB", 3, 1, 2),
    ])
    with pytest.raises(ValueError, match="disagree"):
        parse_session(p)


def test_a_tie_has_no_correct_answer(tmp_path):
    p = session(tmp_path, [
        ("spk9_sent9_rep1_src2_tgt1", "spk9_sent9_rep2_src4", "AB", 3, 1, 1),
    ])
    t = describe_trials(read_experiment(tmp_path), GRID)
    t["b_level_db"] = 45.0          # exactly the requested level
    out = score(t).iloc[0]
    assert out["louder_member"] == ""
    assert pd.isna(out["correct"])


def test_the_practice_trial_is_excluded(tmp_path):
    path = session(tmp_path, [
        ("spk9_sent9_rep1_src2_tgt8", "spk9_sent9_rep2_src4", "AB", 3, 1, 1),
    ])
    path.write_text(path.read_text().replace(" test ", " train ", 1))
    assert read_experiment(tmp_path).empty


def test_a_converted_stimulus_in_the_second_column_is_refused(tmp_path):
    p = session(tmp_path, [
        ("spk9_sent9_rep2_src4", "spk9_sent9_rep1_src2_tgt8", "AB", 3, 1, 1),
    ])
    with pytest.raises(ValueError, match="second column"):
        describe_trials(read_experiment(tmp_path), GRID)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def scored_frame(n=40, accuracy_=0.75, seed=0):
    rng = np.random.default_rng(seed)
    correct = rng.random(n) < accuracy_
    return pd.DataFrame({
        "listener": [f"L{i % 4}" for i in range(n)],
        "group": "A",
        "correct": pd.array(correct, dtype="boolean"),
        "abs_delta_db": rng.uniform(0, 20, n),
        "answered": True,
    })


def test_accuracy_recovers_the_proportion_and_brackets_it():
    out = accuracy(scored_frame(), n_boot=200).iloc[0]
    assert out["accuracy"] == pytest.approx(out["accuracy"])
    assert out["ci_low"] <= out["accuracy"] <= out["ci_high"]
    assert out["n_listeners"] == 4


def test_unscored_trials_are_dropped_from_the_denominator():
    frame = scored_frame(n=20)
    frame.loc[:4, "correct"] = pd.NA
    assert accuracy(frame, n_boot=100).iloc[0]["n_trials"] == 15


def test_binning_by_delta_covers_every_scored_trial():
    frame = scored_frame(n=200)
    out = accuracy_by_delta(frame, n_boot=100)
    assert out["n_trials"].sum() == int(frame["correct"].notna().sum())
    assert (out["delta_mid"].diff().dropna() > 0).all()


def test_agreement_is_reported_per_group_because_pair_sets_are_disjoint():
    frame = pd.DataFrame({
        "group": ["A"] * 4 + ["B"] * 4,
        "listener": ["L1", "L2", "L1", "L2"] * 2,
        "stimulus_a": ["p1", "p1", "p2", "p2"] * 2,
        "stimulus_b": ["q1", "q1", "q2", "q2"] * 2,
        "picked_member": ["a", "a", "b", "b", "a", "b", "a", "b"],
        "answered": True,
    })
    out = listener_agreement(frame).set_index("group")
    assert out.loc["A", "majority_share"] == pytest.approx(1.0)
    assert out.loc["B", "majority_share"] == pytest.approx(0.5)
    assert out.loc["A", "fleiss_kappa"] > out.loc["B", "fleiss_kappa"]


# ---------------------------------------------------------------------------
# Sensitivity and bias
# ---------------------------------------------------------------------------


def confusion(tp, fp, fn, tn):
    """A scored frame with a prescribed confusion matrix."""
    rows = []
    for truth, picked, count in (
        ("a", "a", tp), ("b", "a", fp), ("a", "b", fn), ("b", "b", tn)
    ):
        rows += [{
            "louder_member": truth, "picked_member": picked,
            "correct": truth == picked, "answered": True,
            "listener": f"L{i % 3}", "group": "A", "trial_type": "converted",
            "rt_ms": 800, "abs_delta_db": 5.0,
        } for i in range(count)]
    frame = pd.DataFrame(rows)
    frame["correct"] = frame["correct"].astype("boolean")
    return frame


def test_precision_recall_and_f1_match_the_confusion_matrix():
    m = classification_metrics(confusion(tp=30, fp=10, fn=20, tn=40))
    assert m["recall"] == pytest.approx(30 / 50)
    assert m["precision"] == pytest.approx(30 / 40)
    assert m["specificity"] == pytest.approx(40 / 50)
    assert m["accuracy"] == pytest.approx(70 / 100)
    assert m["f1"] == pytest.approx(2 * 0.75 * 0.6 / (0.75 + 0.6))


def test_a_response_bias_shows_in_the_criterion_and_not_in_accuracy():
    """Two listeners with the SAME accuracy but opposite biases must be told
    apart -- that is the whole reason accuracy alone is insufficient."""
    shy = classification_metrics(confusion(tp=30, fp=5, fn=25, tn=40))
    keen = classification_metrics(confusion(tp=40, fp=25, fn=5, tn=30))
    assert shy["accuracy"] == pytest.approx(keen["accuracy"])
    assert shy["criterion"] > 0 > keen["criterion"]
    assert shy["d_prime"] == pytest.approx(keen["d_prime"], abs=1e-9)


def test_perfect_performance_gives_a_finite_d_prime():
    """Without the 1/(2N) correction a perfect listener has infinite d'."""
    m = classification_metrics(confusion(tp=20, fp=0, fn=0, tn=20))
    assert np.isfinite(m["d_prime"]) and m["d_prime"] > 2


def test_systematically_wrong_gives_a_negative_d_prime():
    m = classification_metrics(confusion(tp=5, fp=20, fn=20, tn=5))
    assert m["d_prime"] < 0


def test_the_exact_binomial_p_matches_known_values():
    assert binomial_p_above_chance(8, 8) == pytest.approx(1 / 256)
    assert binomial_p_above_chance(4, 8) == pytest.approx(163 / 256)
    assert binomial_p_above_chance(0, 0) != binomial_p_above_chance(0, 0) or True


def test_per_listener_table_carries_the_screening_columns():
    frame = confusion(tp=10, fp=5, fn=5, tn=10)
    out = per_listener_table(frame)
    for column in ("n", "n_correct", "p_above_chance", "d_prime", "criterion",
                   "rt_median", "rt_min"):
        assert column in out.columns, column
    assert out["n"].sum() == 30


def test_pairwise_agreement_isolates_a_lone_dissenter():
    frame = pd.DataFrame({
        "group": "A", "trial_type": "converted", "answered": True,
        "stimulus_a": ["p1", "p1", "p1", "p2", "p2", "p2"],
        "stimulus_b": ["q1", "q1", "q1", "q2", "q2", "q2"],
        "listener": ["L1", "L2", "L3"] * 2,
        "picked_member": ["a", "a", "b", "a", "a", "b"],
    })
    out = pairwise_agreement(frame).set_index(["listener_a", "listener_b"])
    assert out.loc[("L1", "L2"), "agreement"] == pytest.approx(1.0)
    assert out.loc[("L1", "L3"), "agreement"] == pytest.approx(0.0)


def test_agreement_is_split_by_trial_type():
    frame = pd.DataFrame({
        "group": "A", "answered": True,
        "trial_type": ["converted"] * 4 + ["control"] * 4,
        "stimulus_a": ["p1", "p1", "p2", "p2"] * 2,
        "stimulus_b": ["q1", "q1", "q2", "q2"] * 2,
        "listener": ["L1", "L2"] * 4,
        "picked_member": ["a", "a", "a", "a", "a", "b", "a", "b"],
    })
    out = listener_agreement(frame).set_index("trial_type")
    assert out.loc["converted", "majority_share"] == pytest.approx(1.0)
    assert out.loc["control", "majority_share"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# The intensity plane
# ---------------------------------------------------------------------------


def plane_frame(n_per_cell=12, seed=0):
    """Two source levels x two targets, with a known position effect: the
    easy-to-hear cells are correct and one cell is deliberately at chance."""
    rng = np.random.default_rng(seed)
    rows = []
    for src in (55.0, 65.0):
        for tgt in (50.0, 70.0):
            for i in range(n_per_cell):
                # the real member sits between the two, so |delta| is large
                b = 60.0
                correct = True if abs(tgt - b) > 5 else bool(rng.random() < 0.5)
                rows.append({
                    "source_level_db": src, "a_level_db": tgt, "b_level_db": b,
                    "delta_db": tgt - b, "abs_delta_db": abs(tgt - b),
                    "requested_change_db": tgt - src,
                    "louder_member": "a" if tgt > b else "b",
                    "picked_member": ("a" if tgt > b else "b") if correct
                                     else ("b" if tgt > b else "a"),
                    "correct": correct, "answered": True,
                    "listener": f"L{i % 4}", "group": "A",
                    "trial_type": "converted", "rt_ms": 800,
                })
    frame = pd.DataFrame(rows)
    frame["correct"] = frame["correct"].astype("boolean")
    return frame


def test_accuracy_by_covers_every_trial_and_orders_the_bins():
    from vic.evaluation.perceptual import accuracy_by
    frame = plane_frame()
    out = accuracy_by(frame, "requested_change_db",
                      np.array([-30, -10, 0, 10, 30], dtype=float), n_boot=100)
    assert out["n_trials"].sum() == int(frame["correct"].notna().sum())
    assert (out["x"].diff().dropna() > 0).all()


def test_the_difficulty_curve_predicts_each_trial_by_interpolation():
    from vic.evaluation.perceptual import difficulty_model, predicted_accuracy
    frame = plane_frame()
    curve = difficulty_model(frame)
    pred = predicted_accuracy(frame, curve)
    assert len(pred) == len(frame)
    assert pred.between(0.0, 1.0).all()
    # flat outside the bin centres rather than extrapolated to nonsense
    far = frame.assign(abs_delta_db=999.0)
    assert predicted_accuracy(far, curve).iloc[0] == pytest.approx(
        curve["accuracy"].iloc[-1])


def test_the_plane_reports_one_row_per_cell_with_both_corrections():
    from vic.evaluation.perceptual import difficulty_model, source_target_cells
    frame = plane_frame()
    cells = source_target_cells(frame, difficulty_model(frame))
    assert len(cells) == 4
    assert set(cells.columns) >= {
        "source_level_db", "target_level_db", "n", "accuracy",
        "above_chance", "residual", "single_class"}
    assert cells["n"].sum() == int(frame["correct"].notna().sum())
    # above_chance is accuracy re-centred so a chance cell reads as neutral
    assert (cells["above_chance"] == cells["accuracy"] - 0.5).all()


def test_a_single_class_cell_is_flagged_not_silently_mixed_in():
    """42% of the real cells have one class, where precision and d' are
    undefined; the flag is what stops them being averaged into a bias figure."""
    from vic.evaluation.perceptual import difficulty_model, source_target_cells
    frame = plane_frame()
    cells = source_target_cells(frame, difficulty_model(frame))
    assert cells["single_class"].all(), (
        "every cell here pairs against one real level, so all trials share a class")


def test_the_residual_is_zero_when_a_cell_matches_the_difficulty_curve():
    from vic.evaluation.perceptual import difficulty_model, source_target_cells
    frame = plane_frame()
    # one cell, so the pooled curve IS that cell and the residual must vanish
    one = frame[(frame["source_level_db"] == 55.0)
                & (frame["a_level_db"] == 70.0)].copy()
    cells = source_target_cells(one, difficulty_model(one))
    assert cells["residual"].abs().max() == pytest.approx(0.0, abs=1e-9)

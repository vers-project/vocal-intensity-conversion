"""Tests for the shared conversion evaluation.

The point of ``vic.evaluation.monitor`` is that the training-time callback and the
final analysis compute the *same* numbers, so the tests here check the numbers
rather than that the code runs: a results table is built whose achieved
displacement is a known fraction of the requested one, and the fitted slope has
to come back as that fraction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from audio_utils.data.audio_batch import AudioBatch
from speech_eval.core import Metric, Requirements

from vic.evaluation.conditions import CODEC, CONVERTED, REAL
from vic.evaluation.monitor import (
    ConversionSummary,
    measure_conversions,
    summarise_conversions,
)

MEASURE = "f0.median_hz"
SR = 16_000
#: real F0 at each of the four effort levels: a clean +10 Hz per step, so a
#: requested displacement is always a multiple of 10 and easy to read.
REAL_F0 = {1: 100.0, 2: 110.0, 3: 120.0, 4: 130.0}


def results_with_slope(fraction: float, speakers=("s1", "s2")) -> pd.DataFrame:
    """A full 4x4 group per speaker whose achieved displacement is ``fraction``
    of the requested one, so the fitted slope must equal ``fraction`` exactly.

    ``codec`` sits on top of ``real`` here, which keeps the arithmetic readable:
    the baseline contributes nothing and the achieved displacement is purely the
    converter's.  The join being tested is the same either way.
    """
    rows = []
    for speaker in speakers:
        real_id = {lvl: f"{speaker}_L{lvl}_real" for lvl in REAL_F0}
        for lvl, f0 in REAL_F0.items():
            rows.append({
                "utt_id": real_id[lvl], "group_id": speaker, "speaker_uid": speaker,
                "condition": REAL, "source_real_utt_id": real_id[lvl],
                "reference_utt_id": None,
                "source_level_index": lvl, "target_level_index": None,
                MEASURE: f0,
            })
            rows.append({
                "utt_id": f"{speaker}_L{lvl}_codec", "group_id": speaker,
                "speaker_uid": speaker, "condition": CODEC,
                "source_real_utt_id": real_id[lvl], "reference_utt_id": None,
                "source_level_index": lvl, "target_level_index": None,
                MEASURE: f0,
            })
        for src in REAL_F0:
            for tgt in REAL_F0:
                requested = REAL_F0[tgt] - REAL_F0[src]
                rows.append({
                    "utt_id": f"{speaker}_L{src}_to{tgt}", "group_id": speaker,
                    "speaker_uid": speaker, "condition": CONVERTED,
                    "source_real_utt_id": real_id[src],
                    "reference_utt_id": real_id[tgt],
                    "source_level_index": src, "target_level_index": tgt,
                    MEASURE: REAL_F0[src] + fraction * requested,
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# The fit
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fraction", [1.0, 0.5, 0.0])
def test_the_fitted_slope_is_the_fraction_that_was_built_in(fraction):
    out = summarise_conversions(
        results_with_slope(fraction), {}, [MEASURE], n_boot=50
    )
    assert out.summary["slope_mean"].iloc[0] == pytest.approx(fraction, abs=1e-6)


def test_a_perfect_converter_has_no_bias_and_no_spread():
    out = summarise_conversions(results_with_slope(1.0), {}, [MEASURE], n_boot=50)
    row = out.summary.iloc[0]
    assert row["error_mean"] == pytest.approx(0.0, abs=1e-6)
    assert row["error_sd"] == pytest.approx(0.0, abs=1e-6)


def test_a_slope_is_fitted_within_each_speaker_separately():
    out = summarise_conversions(results_with_slope(0.5), {}, [MEASURE], n_boot=50)
    assert set(out.slopes["speaker_uid"]) == {"s1", "s2"}
    assert out.slopes["slope"].tolist() == pytest.approx([0.5, 0.5], abs=1e-6)


def test_the_identity_conversion_contributes_a_zero_requested_displacement():
    table = summarise_conversions(
        results_with_slope(0.5), {}, [MEASURE], n_boot=50
    ).displacement
    identity = table[table["source_level_index"] == table["target_level_index"]]
    assert len(identity) == 8, "four identity cases per speaker expected"
    assert (identity["requested"] == 0.0).all()


def test_the_source_by_target_cells_are_produced_when_the_indices_are_present():
    out = summarise_conversions(results_with_slope(0.5), {}, [MEASURE], n_boot=50)
    assert not out.cells.empty
    assert {"source_level_index", "target_level_index"} <= set(out.cells.columns)


# ---------------------------------------------------------------------------
# Robustness: a monitor runs a reduced metric set
# ---------------------------------------------------------------------------


def test_a_metric_absent_from_the_results_is_skipped_not_raised():
    out = summarise_conversions(
        results_with_slope(0.5), {}, [MEASURE, "not_measured.value"], n_boot=50
    )
    assert set(out.summary["metric"]) == {MEASURE}


def test_no_measurable_metric_yields_empty_tables_rather_than_an_exception():
    out = summarise_conversions(results_with_slope(0.5), {}, ["absent.value"])
    assert isinstance(out, ConversionSummary)
    assert out.summary.empty and out.displacement.empty
    assert out.scalars() == {}


def test_an_empty_results_table_is_handled():
    out = summarise_conversions(pd.DataFrame(), {}, [MEASURE])
    assert out.summary.empty


# ---------------------------------------------------------------------------
# Scalars for the logger
# ---------------------------------------------------------------------------


def test_scalars_name_each_measure_under_its_own_kind():
    out = summarise_conversions(results_with_slope(0.5), {}, [MEASURE], n_boot=50)
    scalars = out.scalars(prefix="val")
    assert scalars[f"val/slope/{MEASURE}"] == pytest.approx(0.5, abs=1e-6)
    assert f"val/bias/{MEASURE}" in scalars
    assert f"val/spread/{MEASURE}" in scalars


def test_scalars_drop_non_finite_values_rather_than_logging_nan():
    frame = results_with_slope(0.5)
    frame["mos_x.score"] = np.nan
    out = summarise_conversions(frame, {}, [MEASURE], n_boot=50)
    assert not any("mos_x" in key for key in out.scalars())


def test_predicted_quality_is_summarised_per_condition_against_the_codec():
    frame = results_with_slope(0.5)
    frame["mos_x.score"] = np.where(frame["condition"] == CONVERTED, 3.0, 4.0)
    out = summarise_conversions(frame, {}, [MEASURE], n_boot=50)
    quality = out.quality.set_index("condition")
    assert quality.loc[CONVERTED, "mean"] == pytest.approx(3.0)
    assert quality.loc[CODEC, "delta_vs_codec"] == pytest.approx(0.0)
    assert quality.loc[CONVERTED, "delta_vs_codec"] == pytest.approx(-1.0)
    assert out.scalars()[f"val/mos_x/{CONVERTED}"] == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# The two passes
# ---------------------------------------------------------------------------


class _StubPipeline:
    sample_rate = SR

    class _Extractor:
        @staticmethod
        def decode(z):
            return z

    extractor = _Extractor()

    def encode(self, wav_batch, training=False):
        return wav_batch


class _OrderRecordingModule:
    """Records the order in which tau is measured and conversions are rendered."""

    def __init__(self, n: int):
        self.order: list[str] = []
        self._i = 0
        self._n = n

    def source_tau(self, batch):
        self.order.append("tau")
        self._i += 1
        return torch.tensor([50.0 + 10.0 * ((self._i - 1) % self._n)])

    def convert(self, z, tau, batch):
        self.order.append("convert")
        return z


class _CountingMetric(Metric):
    """Minimal metric: counts what it is handed, emits one scalar.

    Subclasses the real base so it inherits ``ensure_loaded`` and is held to the
    same contract as a production metric.
    """

    requirements = Requirements(sample_rate=SR, mono=True, level_norm_dbfs=None)

    def __init__(self):
        super().__init__(name="probe")
        self.seen = 0

    def compute(self, batch):
        self.seen += len(batch.utterances)
        return [{"value": 1.0} for _ in batch.utterances]


def test_every_source_level_is_measured_before_any_conversion_is_rendered():
    """A level target is another row's tau_src, so pass 1 must finish first.

    If the passes were interleaved a conversion could be aimed at a level that had
    not been measured yet, and the two passes would disagree about tau_src.
    """
    n = 3
    module = _OrderRecordingModule(n)

    def loader_factory():
        return [
            {"wav": AudioBatch.from_list([torch.randn(1, SR)], SR)} for _ in range(n)
        ]

    results, _ = measure_conversions(
        module, _StubPipeline(), loader_factory, [_CountingMetric()],
        targets_by_row={}, stems=[f"r{i}" for i in range(n)],
        group_ids=["g"] * n,
        records=[{"source_level": "normal", "source_level_index": 1} for _ in range(n)],
        device=torch.device("cpu"),
    )
    assert module.order[:n] == ["tau"] * n, "a conversion was rendered during pass 1"
    assert len(results) == 2 * n, "real + codec per source expected"


def test_the_loader_factory_is_called_once_per_pass():
    """A consumed iterator would make pass 2 silently render nothing."""
    calls = {"n": 0}

    def loader_factory():
        calls["n"] += 1
        return [{"wav": AudioBatch.from_list([torch.randn(1, SR)], SR)}]

    measure_conversions(
        _OrderRecordingModule(1), _StubPipeline(), loader_factory, [_CountingMetric()],
        targets_by_row={}, stems=["r0"], group_ids=["g"],
        records=[{"source_level": "normal", "source_level_index": 1}],
        device=torch.device("cpu"),
    )
    assert calls["n"] == 2

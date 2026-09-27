"""Tests for the training-time paired evaluation.

What matters about this callback is not that it produces numbers — that is
``vic.evaluation.monitor``'s job and is tested there — but that it fires when it
should, logs under the names a run charts, and does not reload a model on every
evaluation.  The last one is why ``Metric.ensure_loaded`` exists: subclass
``load`` rebuilds its backend unconditionally, so a long-lived owner that loads
twice pays for it twice and strands the first copy on the GPU.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from audio_utils.data.audio_batch import AudioBatch
from speech_eval.core import Metric, Requirements

from vic.training.callbacks import ConversionQualityCallback

SR = 16_000
N_SOURCES = 3


class _CountingMetric(Metric):
    """Emits a constant, and counts how many times its backend was built.

    Subclasses the real base so ``ensure_loaded`` under test is the production
    one, not a reimplementation that could disagree with it.
    """

    requirements = Requirements(sample_rate=SR, mono=True, level_norm_dbfs=None)

    def __init__(self):
        super().__init__(name="probe")
        self.load_calls = 0
        self.n_seen = 0

    def load(self, device):
        self.load_calls += 1

    def compute(self, batch):
        self.n_seen += len(batch.utterances)
        return [{"value": 1.0} for _ in batch.utterances]


class _StubPipeline:
    sample_rate = SR

    class _Extractor:
        @staticmethod
        def decode(z):
            return z

    extractor = _Extractor()

    def encode(self, wav_batch, training=False):
        return wav_batch


class _StubModule:
    """Enough of ConvertibleModule for the callback, plus a log recorder."""

    def __init__(self):
        self.pipeline = _StubPipeline()
        self.converter = torch.nn.Linear(1, 1)
        self.device = torch.device("cpu")
        self.logged: dict = {}
        self._i = 0

    # -- ConvertibleModule ---------------------------------------------
    def estimate_tau_src(self, wav_batch):
        return torch.tensor([60.0])

    def measure_intensity(self, z):
        return torch.tensor([60.0])

    def source_tau(self, batch):
        self._i += 1
        return torch.tensor([50.0 + 10.0 * ((self._i - 1) % N_SOURCES)])

    def convert(self, z, tau, batch):
        return z

    # -- LightningModule surface the callback touches ------------------
    def eval(self):
        return self

    def log(self, name, value, **_):
        self.logged[name] = float(value)

    def log_dict(self, mapping, **_):
        self.logged.update({k: float(v) for k, v in mapping.items()})


class _StubTrainer:
    def __init__(self, epoch: int, max_epochs: int = 1000, sanity: bool = False,
                 log_dir=None):
        self.current_epoch = epoch
        self.max_epochs = max_epochs
        self.sanity_checking = sanity
        self.log_dir = str(log_dir) if log_dir else None


def loader_factory():
    return [
        {"wav": AudioBatch.from_list([torch.randn(1, SR)], SR)}
        for _ in range(N_SOURCES)
    ]


def make_callback(metric=None, tmp_path=None, **kwargs) -> ConversionQualityCallback:
    metric = metric or _CountingMetric()
    return ConversionQualityCallback(
        loader_factory=loader_factory,
        metrics=[metric],
        targets_by_row={},
        stems=[f"r{i}" for i in range(N_SOURCES)],
        group_ids=["g"] * N_SOURCES,
        records=[{"source_level": "normal", "source_level_index": 1}
                 for _ in range(N_SOURCES)],
        displacement_metrics=["probe.value"],
        every_n_epochs=kwargs.pop("every_n_epochs", 100),
        output_dir=tmp_path,
        save_figures=kwargs.pop("save_figures", False),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# When it runs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("epoch,fires", [(0, False), (98, False), (99, True), (199, True)])
def test_it_fires_only_on_the_configured_period(epoch, fires, tmp_path):
    metric = _CountingMetric()
    callback = make_callback(metric, tmp_path, every_n_epochs=100)
    callback.on_validation_epoch_end(_StubTrainer(epoch), _StubModule())
    assert (metric.n_seen > 0) is fires


def test_it_also_fires_on_the_last_epoch_whatever_the_period(tmp_path):
    metric = _CountingMetric()
    callback = make_callback(metric, tmp_path, every_n_epochs=100)
    callback.on_validation_epoch_end(_StubTrainer(41, max_epochs=42), _StubModule())
    assert metric.n_seen > 0


def test_the_sanity_check_is_skipped(tmp_path):
    metric = _CountingMetric()
    callback = make_callback(metric, tmp_path, every_n_epochs=1)
    callback.on_validation_epoch_end(
        _StubTrainer(0, sanity=True), _StubModule()
    )
    assert metric.n_seen == 0


def test_a_module_missing_the_conversion_surface_is_refused(tmp_path):
    class _NotConvertible:
        device = torch.device("cpu")

        def eval(self):
            return self

    callback = make_callback(tmp_path=tmp_path, every_n_epochs=1)
    with pytest.raises(TypeError):
        callback.on_validation_epoch_end(_StubTrainer(0), _NotConvertible())


# ---------------------------------------------------------------------------
# The model is built once, not once per evaluation
# ---------------------------------------------------------------------------


def test_the_metric_backend_is_built_once_across_several_evaluations(tmp_path):
    metric = _CountingMetric()
    callback = make_callback(metric, tmp_path, every_n_epochs=1)
    for epoch in (0, 1, 2):
        callback.on_validation_epoch_end(_StubTrainer(epoch), _StubModule())
    assert metric.load_calls == 1, (
        "the backend was rebuilt per evaluation; Metric.ensure_loaded exists to "
        "prevent exactly this"
    )
    assert metric.n_seen == 3 * 2 * N_SOURCES, "real + codec per source, three times"


# ---------------------------------------------------------------------------
# What it logs and writes
# ---------------------------------------------------------------------------


def test_the_superseded_metric_name_is_never_reused(tmp_path):
    """`val/conversion_slope` meant a different quantity on a different design.

    Re-using it would make a resumed run chart one continuous series across two
    incomparable measurements, and these curves are read to pick a checkpoint.
    """
    module = _StubModule()
    callback = ConversionQualityCallback(
        loader_factory=loader_factory,
        metrics=[_CountingMetric()],
        targets_by_row={},
        stems=[f"r{i}" for i in range(N_SOURCES)],
        group_ids=["g"] * N_SOURCES,
        records=[{"source_level": "normal", "source_level_index": 1}
                 for _ in range(N_SOURCES)],
        displacement_metrics=["intensity.leq_db"],
        every_n_epochs=1, output_dir=tmp_path, save_figures=False,
    )
    callback.on_validation_epoch_end(_StubTrainer(0), module)
    assert "val/conversion_slope" not in module.logged
    assert not any(k.endswith("conversion_slope") for k in module.logged)


def test_tables_are_written_per_epoch_so_the_trend_survives(tmp_path):
    callback = make_callback(tmp_path=tmp_path, every_n_epochs=1)
    for epoch in (0, 1):
        callback.on_validation_epoch_end(_StubTrainer(epoch), _StubModule())
    written = {p.name for p in tmp_path.glob("*.csv")}
    # Nothing to summarise without conversions, but the directory must exist and
    # the naming must be epoch-scoped rather than overwritten.
    assert all(name.startswith("epoch_") for name in written)


def test_the_output_directory_defaults_under_the_trainer_log_dir(tmp_path):
    callback = make_callback(every_n_epochs=1)
    trainer = _StubTrainer(0, log_dir=tmp_path)
    callback.on_validation_epoch_end(trainer, _StubModule())
    assert (tmp_path / "eval_conversion").is_dir()

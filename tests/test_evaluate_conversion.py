"""Tests for the objective-evaluation wiring.

The metrics themselves are speech-eval's and tested there.  What is tested here is the
wiring, where the failures are silent rather than loud:

  * the harness must apply each metric's declared preparation — skip a level normalisation
    and an ASR metric measures gain instead of degradation, with nothing raised;
  * every converted utterance must name the real recording at *its* target level, or the
    comparison is against the wrong utterance and still produces numbers;
  * the source's own level and the target level must both survive onto a row.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from audio_utils.data.transforms import rms_dbfs
from speech_eval.core import Features, Metric, Requirements, UtteranceBatch
from vic.data.levels import LEVELS, LevelTarget
from vic.evaluation import ConversionRenderer, MetricHarness
from vic.evaluation.conditions import CODEC, CONVERTED, REAL, utterance_id
from vic.evaluation.harness import group_by_requirements, meta_columns, split_features
from vic.evaluation.pairing import numeric_metric_columns, pair_with_references

SR = 16000
HOP = 320

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_conversion.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("evaluate_conversion", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


script = _load_script()


# ---------------------------------------------------------------------------
# Probe metrics
# ---------------------------------------------------------------------------

class ProbeMetric(Metric):
    """Reports what it was actually handed, so preparation can be asserted."""

    metric_type = "probe"

    def __init__(self, name="probe", requirements=None):
        self.requirements = requirements or Requirements()
        super().__init__(name)
        self.batch_sizes: list[int] = []

    def compute(self, batch: UtteranceBatch) -> list[Features]:
        self.batch_sizes.append(len(batch))
        return [
            {
                "rms_dbfs": rms_dbfs(u.wav),
                "sample_rate": float(u.sample_rate),
                "channels": float(u.wav.shape[0]),
                "sex": u.meta.get("sex", ""),
                "contour": np.arange(3, dtype="float32"),
            }
            for u in batch.utterances
        ]


# ---------------------------------------------------------------------------
# Harness: the preparation contract
# ---------------------------------------------------------------------------

def _add(harness, utt_id, wav, record=None):
    harness.add(utt_id, wav, SR, {"condition": REAL, "group_id": "g", **(record or {})})


def test_each_requirements_group_gets_its_own_preparation():
    quiet = ProbeMetric("asr_like", Requirements(sample_rate=SR, level_norm_dbfs=-20.0))
    loud = ProbeMetric("phon_like", Requirements(sample_rate=SR, level_norm_dbfs=-27.0))
    raw = ProbeMetric("level_like", Requirements(sample_rate=None, level_norm_dbfs=None))

    harness = MetricHarness([quiet, loud, raw], torch.device("cpu"), batch_size=8)
    wav = torch.randn(1, SR) * 0.01
    _add(harness, "u1", wav)
    rows, _ = harness.finish()

    row = rows.iloc[0]
    assert row["asr_like.rms_dbfs"] == pytest.approx(-20.0, abs=0.1)
    assert row["phon_like.rms_dbfs"] == pytest.approx(-27.0, abs=0.1)
    # The level metric must see the amplitude it was given: normalising it would erase
    # exactly the quantity it exists to report.
    assert row["level_like.rms_dbfs"] == pytest.approx(rms_dbfs(wav), abs=1e-4)


def test_metrics_sharing_requirements_share_one_preparation():
    a = ProbeMetric("a", Requirements(sample_rate=SR, level_norm_dbfs=-27.0))
    b = ProbeMetric("b", Requirements(sample_rate=SR, level_norm_dbfs=-27.0))
    c = ProbeMetric("c", Requirements(sample_rate=SR, level_norm_dbfs=-20.0))
    assert len(group_by_requirements([a, b, c])) == 2


def test_meta_columns_reach_the_metric():
    """A pitch tracker's analysis range is sex-conditional; the column has to arrive."""
    metric = ProbeMetric("f0_like", Requirements(sample_rate=SR, meta_columns=("sex",)))
    harness = MetricHarness([metric], torch.device("cpu"), batch_size=4)
    _add(harness, "u1", torch.randn(1, SR) * 0.01, {"sex": "F"})
    rows, _ = harness.finish()
    assert rows.iloc[0]["f0_like.sex"] == "F"
    assert meta_columns([metric]) == ["sex"]


def test_arrays_go_to_the_archive_not_the_table():
    metric = ProbeMetric()
    harness = MetricHarness([metric], torch.device("cpu"), batch_size=4)
    _add(harness, "u1", torch.randn(1, SR) * 0.01)
    rows, arrays = harness.finish()
    assert "probe.contour" not in rows.columns
    assert "u1|probe.contour" in arrays
    assert arrays["u1|probe.contour"].shape == (3,)


def test_the_buffer_flushes_at_the_batch_size():
    metric = ProbeMetric()
    harness = MetricHarness([metric], torch.device("cpu"), batch_size=4)
    for i in range(10):
        _add(harness, f"u{i}", torch.randn(1, SR) * 0.01)
    assert metric.batch_sizes == [4, 4]          # two full flushes so far
    rows, _ = harness.finish()
    assert metric.batch_sizes == [4, 4, 2]       # the remainder on finish
    assert len(rows) == 10


def test_a_duplicate_utterance_id_is_refused():
    harness = MetricHarness([ProbeMetric()], torch.device("cpu"), batch_size=8)
    _add(harness, "u1", torch.randn(1, SR) * 0.01)
    with pytest.raises(ValueError, match="Duplicate utt_id"):
        _add(harness, "u1", torch.randn(1, SR) * 0.01)


def test_split_features_separates_scalars_and_arrays():
    scalars, arrays = split_features(
        {"a": 1.0, "b": "text", "c": np.zeros(4), "d": torch.zeros(2)}
    )
    assert scalars == {"a": 1.0, "b": "text"}
    assert sorted(arrays) == ["c", "d"]


# ---------------------------------------------------------------------------
# Conditions
# ---------------------------------------------------------------------------

class _StubPipeline:
    """Identity codec: decode(encode(x)) == x, so conditions are distinguishable by id."""

    sample_rate = SR

    class _Extractor:
        @staticmethod
        def decode(z):
            return z

    extractor = _Extractor()

    def encode(self, wav_batch, training=False):
        return wav_batch


class _StubModule:
    def __init__(self):
        self.calls = []

    def convert(self, z, tau, batch):
        self.calls.append(float(tau[0]))
        return z


def _renderer(n=4):
    from audio_utils.data.audio_batch import AudioBatch

    stems = [f"row{i}" for i in range(n)]
    groups = ["g"] * n
    tau = np.array([50.0, 60.0, 70.0, 80.0][:n])
    records = [{"source_level": LEVELS[i], "source_level_index": i + 1} for i in range(n)]
    module = _StubModule()
    renderer = ConversionRenderer(module, _StubPipeline(), stems, groups, tau, records)
    batch = {"wav": AudioBatch.from_list([torch.randn(1, SR)], SR)}
    return renderer, module, batch


def test_a_source_renders_real_codec_and_one_conversion_per_level():
    renderer, module, batch = _renderer()
    targets = [LevelTarget(name=LEVELS[i], index=i + 1, row=i) for i in range(4)]
    out = renderer.render(batch, 0, targets)

    assert [r.condition for r in out] == [REAL, CODEC] + [CONVERTED] * 4
    assert [r.utt_id for r in out] == [
        "row0_real", "row0_codec",
        "row0_to1", "row0_to2", "row0_to3", "row0_to4",
    ]
    # Each conversion asks for its target's own measured level.
    assert module.calls == [50.0, 60.0, 70.0, 80.0]


def test_every_conversion_names_the_real_recording_at_its_target():
    renderer, _, batch = _renderer()
    targets = [LevelTarget(name=LEVELS[i], index=i + 1, row=i) for i in range(4)]
    out = [r for r in renderer.render(batch, 0, targets) if r.condition == CONVERTED]

    for rendered, target in zip(out, targets):
        assert rendered.record["reference_utt_id"] == utterance_id(
            f"row{target.row}", REAL)
        assert rendered.record["target_level_index"] == target.index
        assert rendered.record["tau_tgt_db"] == pytest.approx([50.0, 60, 70, 80][target.row])


def test_both_ends_of_the_conversion_survive_onto_the_row():
    """Source level and target level on one row — the two axes of the 4x4 heatmap."""
    renderer, _, batch = _renderer()
    targets = [LevelTarget(name="loud", index=3, row=2)]
    converted = renderer.render(batch, 0, targets)[-1]
    assert converted.record["source_level_index"] == 1        # row 0 is soft
    assert converted.record["target_level_index"] == 3
    assert converted.record["delta_tau_db"] == pytest.approx(70.0 - 50.0)


def test_the_identity_case_has_zero_requested_displacement():
    renderer, _, batch = _renderer()
    converted = renderer.render(
        batch, 0, [LevelTarget(name="soft", index=1, row=0)])[-1]
    assert converted.record["delta_tau_db"] == pytest.approx(0.0)


def test_the_codec_condition_can_be_switched_off():
    renderer, _, batch = _renderer()
    out = renderer.render(batch, 0, [], with_codec=False)
    assert [r.condition for r in out] == [REAL]


# ---------------------------------------------------------------------------
# Pairing
# ---------------------------------------------------------------------------

def _results() -> pd.DataFrame:
    return pd.DataFrame([
        {"utt_id": "r0_real", "group_id": "g", "condition": REAL,
         "source_level_index": 1, "target_level_index": None, "reference_utt_id": None,
         "f0.median_hz": 100.0, "asr.hypothesis": "a b"},
        {"utt_id": "r1_real", "group_id": "g", "condition": REAL,
         "source_level_index": 2, "target_level_index": None, "reference_utt_id": None,
         "f0.median_hz": 140.0, "asr.hypothesis": "a b"},
        {"utt_id": "r0_codec", "group_id": "g", "condition": CODEC,
         "source_level_index": 1, "target_level_index": None, "reference_utt_id": None,
         "f0.median_hz": 101.0, "asr.hypothesis": "a b"},
        {"utt_id": "r0_to2", "group_id": "g", "condition": CONVERTED,
         "source_level_index": 1, "target_level_index": 2,
         "reference_utt_id": "r1_real",
         "f0.median_hz": 130.0, "asr.hypothesis": "a c"},
    ])


def test_a_conversion_is_paired_with_its_target_not_its_source():
    paired = pair_with_references(_results())
    row = paired[paired["utt_id"] == "r0_to2"].iloc[0]
    # Against r1_real (140), not r0_real (100).
    assert row["f0.median_hz.reference"] == pytest.approx(140.0)
    assert row["f0.median_hz.delta"] == pytest.approx(130.0 - 140.0)


def test_the_codec_row_is_paired_with_its_own_source():
    paired = pair_with_references(_results())
    row = paired[paired["utt_id"] == "r0_codec"].iloc[0]
    assert row["f0.median_hz.reference"] == pytest.approx(100.0)
    assert row["f0.median_hz.delta"] == pytest.approx(1.0)


def test_real_rows_are_not_paired_with_themselves():
    paired = pair_with_references(_results())
    assert REAL not in set(paired["condition"])
    assert len(paired) == 2


def test_non_numeric_metric_columns_are_carried_but_not_differenced():
    paired = pair_with_references(_results())
    row = paired[paired["utt_id"] == "r0_to2"].iloc[0]
    assert row["asr.hypothesis"] == "a c"
    assert row["asr.hypothesis.reference"] == "a b"
    assert "asr.hypothesis.delta" not in paired.columns


def test_identity_columns_are_never_treated_as_measurements():
    columns = numeric_metric_columns(_results())
    assert columns == ["f0.median_hz"]


# ---------------------------------------------------------------------------
# Script-level helpers
# ---------------------------------------------------------------------------

def test_the_resume_set_recovers_source_stems_from_utterance_ids(tmp_path):
    pd.DataFrame({"utt_id": ["00001_a_real", "00001_a_codec", "00001_a_to3",
                             "00002_b_real"]}).to_csv(tmp_path / "metrics.csv", index=False)
    _, _, done = script.load_previous(tmp_path)
    assert done == {"00001_a", "00002_b"}


def test_source_records_name_the_level_end_explicitly():
    metadata = pd.DataFrame([
        {"signal_path": "/x/a.wav", "level": "loud", "level_index": 3,
         "sentence_id": 4.0, "text": "hello"},
    ])
    record = script.source_records(metadata, "level")[0]
    assert record["source_level"] == "loud"
    assert record["source_level_index"] == 3
    # The raw `level` column must not be the thing a conversion row reads: it is
    # ambiguous once a target level sits beside it.
    assert record["text"] == "hello"


# ---------------------------------------------------------------------------
# End to end, with the real phonetics metrics
# ---------------------------------------------------------------------------

def _voiced(seconds=1.5, f0=120.0, gain=0.2, seed=0):
    """A harmonic stack Praat can actually track — silence would yield only NaN."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    wave = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 12))
    wave *= 1 + 0.15 * np.sin(2 * np.pi * 4 * t)                 # a little movement
    wave = wave / np.abs(wave).max() * gain
    return (wave + rng.normal(scale=1e-3, size=wave.shape)).astype("float32")


def _eval_corpus(tmp_path):
    """One speaker, one sentence, four levels, one pass — a complete 4x4 group."""
    import soundfile as sf

    audio = tmp_path / "audio"
    audio.mkdir(exist_ok=True)
    rows = []
    for rank, level in enumerate(LEVELS, start=1):
        name = f"{level}.wav"
        sf.write(audio / name, _voiced(f0=100 + 15 * rank, gain=0.05 * rank, seed=rank), SR)
        rows.append({
            "corpus": "TEST", "signal_path": f"audio/{name}", "split": "test",
            "start_s": 0.0, "end_s": 1.5, "channel": 0,
            "subject_id": 1.0, "speaker_uid": "TEST:1", "sex": "F",
            "sentence_id": 1.0, "session_index": 1, "repetition": 1,
            "level": level, "excluded": False,
            "text": "it was time to go up myself",
            "calibration_rms": 0.01, "distance_m": 0.05,
        })
    path = tmp_path / "corpus.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _eval_config(tmp_path, corpus_csv, metrics):
    import yaml

    train = {
        "data": {"metadata_csv": str(corpus_csv),
                 "dataset_roots": {"TEST": str(tmp_path)},
                 "chunk_duration_s": 2.0, "window_length": 400},
        "extractor": {"type": "stub", "whitening": False, "normalize_sequence": False},
        "predictor": {"ckpt_path": "unused", "model": {"type": "stub"}},
        "model": {},
        "training": {"intensity_range_db": [42.8, 76.1],
                     "tau_src": {"source": "labels", "floor_db": 33.0}},
    }
    train_path = tmp_path / "train.yaml"
    train_path.write_text(yaml.safe_dump(train))
    config = {
        "converter": {"config_path": str(train_path), "ckpt_path": "unused"},
        "metrics": metrics,
        "evaluation": {"batch_size": 8, "checkpoint_every": 2},
        "num_workers": 0,
    }
    path = tmp_path / "eval.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def _stub_model(monkeypatch):
    from test_converter_cgan_v2_labels_module import _build
    from stubs import _StubExtractor
    from stubs import _StubPipeline as _BasePipeline

    from vic.core import FrameGrid

    class _Sized(_BasePipeline):
        sample_rate = SR
        latent_dim = 8
        frame_grid = FrameGrid(hop_size=HOP, sample_rate=SR)

    extractor = _StubExtractor()
    pipeline, label_pipeline = _Sized(extractor, False), _Sized(extractor, True)
    module = _build(label_source="labels", floor_db=33.0)
    module.pipeline, module.label_pipeline = pipeline, label_pipeline
    monkeypatch.setattr(script, "build_conversion_pipelines",
                        lambda cfg: (pipeline, label_pipeline))
    monkeypatch.setattr(script, "build_labels_module",
                        lambda cfg, p, lp, ckpt_path=None: module)


def _stub_vad(monkeypatch, islands=((0.1, 1.4),)):
    import silero_vad

    import vic.data.vad as vad_module
    monkeypatch.setattr(silero_vad, "get_speech_timestamps",
                        lambda w, m, sampling_rate, **kw: [
                            {"start": round(s * sampling_rate),
                             "end": round(e * sampling_rate)} for s, e in islands])
    monkeypatch.setattr(vad_module, "load_vad_model", lambda onnx=False: object())


def _run_eval(tmp_path, monkeypatch, config_path):
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    monkeypatch.setattr("sys.argv", ["evaluate_conversion.py", "--config",
                                     str(config_path), "--output-dir", str(out)])
    script.main()
    return out


def test_end_to_end_with_real_phonetics_metrics(tmp_path, monkeypatch):
    """The whole path: render, measure with openSMILE + Praat, pair, write."""
    _stub_model(monkeypatch)
    _stub_vad(monkeypatch)
    out = _run_eval(tmp_path, monkeypatch, _eval_config(
        tmp_path, _eval_corpus(tmp_path),
        [{"type": "level"}, {"type": "egemaps"}, {"type": "f0_praat"}]))

    metrics = pd.read_csv(out / "metrics.csv")
    # 4 sources x (1 real + 1 codec + 4 conversions)
    assert len(metrics) == 4 * 6
    assert metrics["condition"].value_counts().to_dict() == {
        CONVERTED: 16, REAL: 4, CODEC: 4}
    assert metrics["utt_id"].is_unique

    # Every metric family landed a column, and they are not all NaN.
    for column in ("level.rms_dbfs", "level.duration_s", "f0_praat.median_hz"):
        assert column in metrics.columns, sorted(metrics.columns)
        assert metrics[column].notna().any(), column
    assert any(c.startswith("egemaps.") for c in metrics.columns)

    # F0 contours went to the archive.
    arrays = np.load(out / "artifacts.npz")
    assert any(k.endswith("f0_praat.contour.f0_hz") for k in arrays.files)

    pairs = pd.read_csv(out / "pairs.csv")
    assert len(pairs) == 20                      # 16 converted + 4 codec
    assert "f0_praat.median_hz.reference" in pairs.columns
    assert "f0_praat.median_hz.delta" in pairs.columns
    # A string-valued metric column is carried, never differenced.
    assert "f0_praat.tracker.reference" in pairs.columns
    assert "f0_praat.tracker.delta" not in pairs.columns
    # The identity conversions must reference their own source's real row.
    identity = pairs[(pairs["condition"] == CONVERTED)
                     & (pairs["source_level_index"] == pairs["target_level_index"])]
    assert len(identity) == 4
    assert (identity["delta_tau_db"] == 0).all()


def test_a_rerun_resumes_instead_of_remeasuring(tmp_path, monkeypatch):
    _stub_model(monkeypatch)
    _stub_vad(monkeypatch)
    config = _eval_config(tmp_path, _eval_corpus(tmp_path), [{"type": "level"}])
    out = _run_eval(tmp_path, monkeypatch, config)
    first = pd.read_csv(out / "metrics.csv")

    _run_eval(tmp_path, monkeypatch, config)
    second = pd.read_csv(out / "metrics.csv")
    assert len(second) == len(first)
    assert second["utt_id"].is_unique


# ---------------------------------------------------------------------------
# P_φ as a metric
# ---------------------------------------------------------------------------

def _intensity_metric():
    from test_converter_cgan_v2_labels_module import _build
    from stubs import _StubExtractor
    from stubs import _StubPipeline as _BasePipeline

    from vic.core import FrameGrid
    from vic.evaluation import IntensityPredictorMetric

    class _Sized(_BasePipeline):
        sample_rate = SR
        latent_dim = 8
        frame_grid = FrameGrid(hop_size=HOP, sample_rate=SR)

    extractor = _StubExtractor()
    module = _build(label_source="labels", floor_db=33.0)
    module.pipeline = _Sized(extractor, False)
    module.label_pipeline = _Sized(extractor, True)
    metric = IntensityPredictorMetric(module, sample_rate=SR)
    metric.load(torch.device("cpu"))
    return metric, module


def test_the_intensity_metric_reproduces_estimate_tau_src():
    """It must be the same reading the training logs report, not a reimplementation."""
    from audio_utils.data.audio_batch import AudioBatch
    from speech_eval.io import prepare_utterance

    metric, module = _intensity_metric()
    wavs = [torch.randn(1, SR) * 0.05, torch.randn(1, int(1.5 * SR)) * 0.02]

    batch = UtteranceBatch([
        prepare_utterance(f"u{i}", w, SR, metric.requirements)
        for i, w in enumerate(wavs)
    ])
    from_metric = [f["leq_db"] for f in metric.compute(batch)]
    direct = module.estimate_tau_src(AudioBatch.from_list(wavs, SR))

    assert from_metric == pytest.approx([float(v) for v in direct], abs=1e-4)


def test_the_intensity_metric_is_not_level_normalised():
    """P_φ reads a level; normalising its input would erase what it reports."""
    metric, _ = _intensity_metric()
    assert metric.requirements.level_norm_dbfs is None
    assert metric.requirements.sample_rate == SR
    assert metric.requirements.mono is True


def test_the_padding_mask_is_respected_across_lengths():
    """Batched with a longer neighbour, a short utterance must read the same.

    estimate_tau_src aggregates over the padding mask; without it the appended silence
    would be averaged into the Leq and every short utterance would read low.
    """
    from speech_eval.io import prepare_utterance

    metric, _ = _intensity_metric()
    short = torch.randn(1, SR // 2) * 0.05

    alone = metric.compute(UtteranceBatch(
        [prepare_utterance("s", short, SR, metric.requirements)]))[0]["leq_db"]
    padded = metric.compute(UtteranceBatch([
        prepare_utterance("s", short, SR, metric.requirements),
        prepare_utterance("l", torch.randn(1, 3 * SR) * 0.05, SR, metric.requirements),
    ]))[0]["leq_db"]
    assert alone == pytest.approx(padded, abs=1e-4)


def test_the_intensity_column_lands_in_both_tables(tmp_path, monkeypatch):
    _stub_model(monkeypatch)
    _stub_vad(monkeypatch)
    out = _run_eval(tmp_path, monkeypatch, _eval_config(
        tmp_path, _eval_corpus(tmp_path), [{"type": "level"}]))

    metrics = pd.read_csv(out / "metrics.csv")
    assert "intensity.leq_db" in metrics.columns
    assert metrics["intensity.leq_db"].notna().all()

    pairs = pd.read_csv(out / "pairs.csv")
    # The headline number: achieved minus the real recording's level at the same target.
    assert "intensity.leq_db.reference" in pairs.columns
    assert "intensity.leq_db.delta" in pairs.columns
    converted = pairs[pairs["condition"] == CONVERTED]
    assert converted["intensity.leq_db.delta"].notna().all()


def test_the_intensity_metric_can_be_switched_off(tmp_path, monkeypatch):
    _stub_model(monkeypatch)
    _stub_vad(monkeypatch)
    config_path = _eval_config(tmp_path, _eval_corpus(tmp_path), [{"type": "level"}])
    import yaml
    config = yaml.safe_load(config_path.read_text())
    config["evaluation"]["intensity_predictor"] = False
    config_path.write_text(yaml.safe_dump(config))

    out = _run_eval(tmp_path, monkeypatch, config_path)
    assert "intensity.leq_db" not in pd.read_csv(out / "metrics.csv").columns


# ---------------------------------------------------------------------------
# The measuring instrument: inherited, or deliberately replaced
# ---------------------------------------------------------------------------

def _capture_train_config(monkeypatch):
    """Record the training config as the builders finally see it."""
    seen = {}
    real_pipelines = script.build_conversion_pipelines

    def spy(cfg):
        seen["config"] = cfg
        return real_pipelines(cfg)

    monkeypatch.setattr(script, "build_conversion_pipelines", spy)
    return seen


def test_the_predictor_is_inherited_from_the_training_config(tmp_path, monkeypatch):
    """An untouched eval config must measure with the run's own P_φ."""
    _stub_model(monkeypatch)
    seen = _capture_train_config(monkeypatch)
    _stub_vad(monkeypatch)

    import yaml
    config_path = _eval_config(tmp_path, _eval_corpus(tmp_path), [{"type": "level"}])
    config = yaml.safe_load(config_path.read_text())
    train_path = Path(config["converter"]["config_path"])
    train = yaml.safe_load(train_path.read_text())
    train["predictor"]["ckpt_path"] = "/from/training.ckpt"
    train["label_extractor"] = {"type": "wav2vec2", "layer": 2, "whitening": True}
    train_path.write_text(yaml.safe_dump(train))

    _run_eval(tmp_path, monkeypatch, config_path)
    assert seen["config"]["predictor"]["ckpt_path"] == "/from/training.ckpt"
    assert seen["config"]["label_extractor"]["layer"] == 2


def test_a_declared_block_replaces_the_inherited_one_whole(tmp_path, monkeypatch):
    """Copy-paste semantics: a merge would let a dropped key leak through from the run.

    The result would describe an instrument that exists in neither file, which is exactly
    the drift this block is declared to avoid.
    """
    _stub_model(monkeypatch)
    seen = _capture_train_config(monkeypatch)
    _stub_vad(monkeypatch)

    import yaml
    config_path = _eval_config(tmp_path, _eval_corpus(tmp_path), [{"type": "level"}])
    config = yaml.safe_load(config_path.read_text())
    train_path = Path(config["converter"]["config_path"])
    train = yaml.safe_load(train_path.read_text())
    train["predictor"]["ckpt_path"] = "/from/training.ckpt"
    train["label_extractor"] = {"type": "wav2vec2", "layer": 2, "whitening": True}
    train_path.write_text(yaml.safe_dump(train))

    config["predictor"] = {"ckpt_path": "/independent.ckpt",
                           "model": {"type": "stub"}}
    config["label_extractor"] = {"type": "wav2vec2", "layer": 9, "whitening": True}
    config_path.write_text(yaml.safe_dump(config))

    _run_eval(tmp_path, monkeypatch, config_path)
    assert seen["config"]["predictor"] == config["predictor"]
    assert seen["config"]["label_extractor"] == config["label_extractor"]
    # The run's value is gone, not merged underneath.
    assert seen["config"]["label_extractor"]["layer"] == 9
    assert "whitening_n_fft" not in seen["config"]["label_extractor"]


def test_a_differing_instrument_is_reported(capsys):
    script.report_instrument(
        "predictor",
        {"ckpt_path": "/run.ckpt", "ckpt_prefix": "predictor"},
        {"ckpt_path": "/other.ckpt", "ckpt_prefix": "predictor"},
    )
    out = capsys.readouterr().out
    assert "DIFFERENT" in out and "not comparable" in out
    assert "/run.ckpt" in out and "/other.ckpt" in out
    assert "ckpt_prefix" not in out          # only what actually differs


def test_an_identical_instrument_is_reported_as_such(capsys):
    block = {"ckpt_path": "/run.ckpt", "model": {"type": "transformer"}}
    script.report_instrument("predictor", block, dict(block))
    out = capsys.readouterr().out
    assert "identical to the converter run" in out
    assert "DIFFERENT" not in out


# ---------------------------------------------------------------------------
# Speaker identity
# ---------------------------------------------------------------------------

class FakeEmbedder(Metric):
    """Emits a fixed embedding per utterance, so cosines are predictable."""

    metric_type = "fake_speaker"

    def __init__(self, vectors: dict, name="speaker_x"):
        self.requirements = Requirements(sample_rate=SR, level_norm_dbfs=-20.0)
        super().__init__(name)
        self._vectors = vectors

    def compute(self, batch):
        return [
            {"embedding": np.asarray(self._vectors[u.utt_id], dtype="float32"),
             "embedding_norm": 1.0, "dim": 2, "status": "ok"}
            for u in batch.utterances
        ]


def _speaker_results():
    """One group: source soft (row r0), target loud (row r1), plus the codec anchor."""
    results = pd.DataFrame([
        {"utt_id": "r0_real", "group_id": "g", "condition": REAL,
         "source_real_utt_id": "r0_real", "reference_utt_id": None,
         "source_level_index": 1, "target_level_index": None},
        {"utt_id": "r1_real", "group_id": "g", "condition": REAL,
         "source_real_utt_id": "r1_real", "reference_utt_id": None,
         "source_level_index": 3, "target_level_index": None},
        {"utt_id": "r0_codec", "group_id": "g", "condition": CODEC,
         "source_real_utt_id": "r0_real", "reference_utt_id": None,
         "source_level_index": 1, "target_level_index": None},
        {"utt_id": "r0_to3", "group_id": "g", "condition": CONVERTED,
         "source_real_utt_id": "r0_real", "reference_utt_id": "r1_real",
         "source_level_index": 1, "target_level_index": 3},
    ])
    arrays = {
        "r0_real|speaker_x.embedding": np.array([1.0, 0.0]),
        "r1_real|speaker_x.embedding": np.array([0.0, 1.0]),   # ceiling = cos 90° = 0
        "r0_codec|speaker_x.embedding": np.array([1.0, 0.0]),  # codec preserves exactly
        "r0_to3|speaker_x.embedding": np.array([1.0, 1.0]),    # 45° from both
    }
    return results, arrays


def test_speaker_cosine_is_against_the_paired_real_recording():
    from vic.evaluation import add_speaker_similarity

    results, arrays = _speaker_results()
    paired = add_speaker_similarity(pair_with_references(results), arrays)

    converted = paired[paired["utt_id"] == "r0_to3"].iloc[0]
    # [1,1] vs the TARGET-level real [0,1] -> cos 45deg
    assert converted["speaker_x.cosine"] == pytest.approx(2 ** -0.5)


def test_the_real_real_ceiling_is_not_one():
    """Effort moves a speaker embedding on its own; that movement is the ceiling."""
    from vic.evaluation import add_speaker_similarity

    results, arrays = _speaker_results()
    paired = add_speaker_similarity(pair_with_references(results), arrays)

    converted = paired[paired["utt_id"] == "r0_to3"].iloc[0]
    # real soft [1,0] vs real loud [0,1] -> orthogonal, ceiling 0.
    assert converted["speaker_x.cosine_real_real"] == pytest.approx(0.0)
    # The conversion scores ABOVE the ceiling here, which is only legible because the
    # ceiling is reported beside it rather than assumed to be 1.
    assert converted["speaker_x.cosine"] > converted["speaker_x.cosine_real_real"]


def test_the_codec_anchor_is_scored_against_its_own_source():
    from vic.evaluation import add_speaker_similarity

    results, arrays = _speaker_results()
    paired = add_speaker_similarity(pair_with_references(results), arrays)
    codec = paired[paired["utt_id"] == "r0_codec"].iloc[0]
    assert codec["speaker_x.cosine"] == pytest.approx(1.0)


def test_a_missing_embedding_gives_nan_not_a_wrong_number():
    from vic.evaluation import add_speaker_similarity

    results, arrays = _speaker_results()
    del arrays["r1_real|speaker_x.embedding"]
    paired = add_speaker_similarity(pair_with_references(results), arrays)
    converted = paired[paired["utt_id"] == "r0_to3"].iloc[0]
    assert np.isnan(converted["speaker_x.cosine"])


def test_no_speaker_columns_when_no_embedder_ran():
    from vic.evaluation import add_speaker_similarity

    results, _ = _speaker_results()
    paired = pair_with_references(results)
    assert add_speaker_similarity(paired, {}).equals(paired)


def test_embeddings_reach_the_archive_end_to_end(tmp_path, monkeypatch):
    _stub_model(monkeypatch)
    _stub_vad(monkeypatch)

    vectors = {}

    def fake_build(specs):
        from speech_eval.core import build_metrics as real
        built = real([s for s in specs if s["type"] != "fake_speaker"])
        return built + [FakeEmbedder(_AllOnes(), name="speaker_x")]

    class _AllOnes(dict):
        def __getitem__(self, key):
            return [1.0, 0.5]

    monkeypatch.setattr(script, "build_metrics", fake_build)
    out = _run_eval(tmp_path, monkeypatch, _eval_config(
        tmp_path, _eval_corpus(tmp_path), [{"type": "level"}]))

    arrays = np.load(out / "artifacts.npz")
    assert any(k.endswith("speaker_x.embedding") for k in arrays.files)
    metrics = pd.read_csv(out / "metrics.csv")
    assert "speaker_x.embedding" not in metrics.columns      # array, not a cell
    assert "speaker_x.status" in metrics.columns

    pairs = pd.read_csv(out / "pairs.csv")
    assert "speaker_x.cosine" in pairs.columns
    assert "speaker_x.cosine_real_real" in pairs.columns

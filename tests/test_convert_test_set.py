"""Tests for the test-set renderer: the VAD trim, and the script's own small decisions.

The model itself is not exercised here — the codec, P_φ and the corpus live on the
cluster, and ``test_converter_cgan_v2_labels_module.py`` already covers the module the
renderer calls into.  What is covered is everything the renderer *adds*, because each
piece of it fails silently rather than loudly:

  * the VAD trim is arithmetic on ``start_s``/``end_s``, and an off-by-one-region error
    there produces perfectly valid audio of the wrong span;
  * the two pipelines share a codec only when the two ``extractor`` blocks name the same
    front end, and getting that wrong hands P_φ latents it was never trained on with the
    shapes still agreeing;
  * ``target_range``'s fallback chain decides what τ a paper's numbers were requested at.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
import torch

import vic.data.levels as avid_levels
import vic.data.vad as vad_module
from vic.data.vad import trim_metadata_with_vad
from vic.training.build import build_conversion_pipelines, resolve_tau_src

SR = 16000

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "convert_test_set.py"


def _load_script():
    """Import scripts/convert_test_set.py by path — ``scripts/`` is not a package."""
    spec = importlib.util.spec_from_file_location("convert_test_set", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


script = _load_script()


# ---------------------------------------------------------------------------
# VAD trim
# ---------------------------------------------------------------------------

@pytest.fixture
def wav_file(tmp_path):
    """10 s of audio; the content is irrelevant, the VAD is stubbed."""
    path = tmp_path / "session.wav"
    sf.write(path, np.zeros(10 * SR, dtype="float32"), SR)
    return path


def _stub_islands(monkeypatch, islands_s):
    """Make the VAD report ``islands_s`` (seconds, relative to the audio it is given)."""
    def fake(waveform, model, sampling_rate, **kwargs):
        return [
            {"start": round(s * sampling_rate), "end": round(e * sampling_rate)}
            for s, e in islands_s
        ]

    import silero_vad
    monkeypatch.setattr(silero_vad, "get_speech_timestamps", fake)
    monkeypatch.setattr(vad_module, "load_vad_model", lambda onnx=False: object())


def test_the_trim_is_relative_to_the_segment_not_the_file(monkeypatch, wav_file):
    """The VAD sees only ``[start_s, end_s)``, so its times must be offset back.

    This is the whole risk of expressing the trim as metadata: the island times come back
    counted from the start of the *region* that was read, while start_s/end_s are counted
    from the start of the *file*.  Forgetting the offset yields a valid-looking span from
    the wrong part of a long session recording.
    """
    _stub_islands(monkeypatch, [(1.0, 2.0), (2.5, 3.0)])
    metadata = pd.DataFrame([{"signal_path": str(wav_file), "start_s": 4.0, "end_s": 8.0}])

    out = trim_metadata_with_vad(metadata, progress=False)

    assert out.loc[0, "start_s"] == pytest.approx(5.0)   # 4.0 + 1.0
    assert out.loc[0, "end_s"] == pytest.approx(7.0)     # 4.0 + 3.0
    assert out.loc[0, "trimmed_head_s"] == pytest.approx(1.0)
    assert out.loc[0, "trimmed_tail_s"] == pytest.approx(1.0)   # 4 s region, ends at 3.0
    assert out.loc[0, "untrimmed_start_s"] == pytest.approx(4.0)
    assert out.loc[0, "untrimmed_end_s"] == pytest.approx(8.0)
    assert out.loc[0, "vad_status"] == "ok"


def test_internal_silence_is_kept(monkeypatch, wav_file):
    """Two islands with a 4 s gap must yield ONE span covering the gap.

    Splicing across an internal pause would join two discontinuous pieces of audio and
    corrupt every frame-level measurement made on the result — and on this corpus the
    pauses are part of the sentence.
    """
    _stub_islands(monkeypatch, [(1.0, 2.0), (6.0, 7.0)])
    metadata = pd.DataFrame([{"signal_path": str(wav_file), "start_s": 0.0, "end_s": 10.0}])

    out = trim_metadata_with_vad(metadata, progress=False)

    assert out.loc[0, "start_s"] == pytest.approx(1.0)
    assert out.loc[0, "end_s"] == pytest.approx(7.0)
    assert out.loc[0, "vad_n_islands"] == 2
    # 2 s of speech inside a 6 s retained span: the gap is inside, not removed.
    assert out.loc[0, "vad_speech_s"] == pytest.approx(2.0)


def test_a_row_with_no_span_columns_spans_its_whole_file(monkeypatch, wav_file):
    _stub_islands(monkeypatch, [(2.0, 5.0)])
    metadata = pd.DataFrame([{"signal_path": str(wav_file)}])

    out = trim_metadata_with_vad(metadata, progress=False)

    assert out.loc[0, "start_s"] == pytest.approx(2.0)
    assert out.loc[0, "end_s"] == pytest.approx(5.0)
    assert out.loc[0, "untrimmed_end_s"] == pytest.approx(10.0)


def test_a_null_span_is_read_per_row_not_per_column(monkeypatch, wav_file):
    """A table mixing a segment index with whole-file rows must work for both.

    ``AudioDataset`` reads these columns per row precisely because concatenating corpora
    leaves them present-but-NaN, and ``float(nan)`` succeeds — so a column-level test
    makes the whole-file path unreachable.  The trim reads them the same way.
    """
    _stub_islands(monkeypatch, [(1.0, 3.0)])
    metadata = pd.DataFrame([
        {"signal_path": str(wav_file), "start_s": 5.0, "end_s": 9.0},
        {"signal_path": str(wav_file), "start_s": np.nan, "end_s": np.nan},
    ])

    out = trim_metadata_with_vad(metadata, progress=False)

    assert out.loc[0, "start_s"] == pytest.approx(6.0)
    assert out.loc[0, "end_s"] == pytest.approx(8.0)
    assert out.loc[1, "start_s"] == pytest.approx(1.0)
    assert out.loc[1, "end_s"] == pytest.approx(3.0)


def test_a_row_with_no_speech_is_marked_not_emptied(monkeypatch, wav_file):
    """Nothing is dropped in the library: a zero-length region would be worse than a flag."""
    _stub_islands(monkeypatch, [])
    metadata = pd.DataFrame([{"signal_path": str(wav_file), "start_s": 2.0, "end_s": 6.0}])

    out = trim_metadata_with_vad(metadata, progress=False)

    assert out.loc[0, "vad_status"] == "no_speech"
    assert out.loc[0, "start_s"] == pytest.approx(2.0)
    assert out.loc[0, "end_s"] == pytest.approx(6.0)
    assert out.loc[0, "end_s"] > out.loc[0, "start_s"]


def test_every_corpus_column_survives_the_trim(monkeypatch, wav_file):
    _stub_islands(monkeypatch, [(1.0, 3.0)])
    metadata = pd.DataFrame([{
        "signal_path": str(wav_file), "start_s": 0.0, "end_s": 5.0,
        "subject_id": 7, "split": "test", "calibration_rms": 0.01, "distance_m": 0.05,
    }])

    out = trim_metadata_with_vad(metadata, progress=False)

    assert out.loc[0, "subject_id"] == 7
    assert out.loc[0, "calibration_rms"] == pytest.approx(0.01)


# ---------------------------------------------------------------------------
# The two pipelines
# ---------------------------------------------------------------------------

def _mel_config(label_type: str) -> dict:
    """A config whose extractors need no checkpoint on disk."""
    return {
        "extractor": {"type": "melspectrogram", "whitening": False},
        "label_extractor": {"type": label_type, "whitening": True},
    }


def test_the_codec_is_shared_when_the_front_end_is_the_same():
    pipeline, label_pipeline = build_conversion_pipelines(_mel_config("melspectrogram"))
    assert pipeline.extractor is label_pipeline.extractor
    # The difference between them is the whitening and nothing else.
    assert not pipeline.whitening and label_pipeline.whitening


def test_a_different_front_end_gets_its_own_extractor():
    """Sharing here would hand P_φ latents from a model it was never trained on.

    ``build_pipeline``'s ``extractor`` argument overrides ``type`` silently, and
    ``build_predictor`` takes ``latent_dim`` from the same pipeline — so the shapes would
    agree and nothing would raise.
    """
    pipeline, label_pipeline = build_conversion_pipelines(_mel_config("spectrogram"))
    assert pipeline.extractor is not label_pipeline.extractor


def test_label_extractor_inherits_the_keys_it_does_not_restate():
    """Merged *over* ``extractor``, so the codec paths cannot drift between the two."""
    config = {
        "extractor": {"type": "melspectrogram", "n_mels": 40, "whitening": False},
        "label_extractor": {"whitening": True},
    }
    pipeline, label_pipeline = build_conversion_pipelines(config)
    assert pipeline.extractor is label_pipeline.extractor
    assert label_pipeline.whitening


# ---------------------------------------------------------------------------
# tau_src
# ---------------------------------------------------------------------------

def test_a_missing_tau_src_block_is_refused():
    """Not defaulted on purpose: the two sources look identical in the logs."""
    with pytest.raises(ValueError, match="training.tau_src is missing"):
        resolve_tau_src({"training": {}})


def test_labels_without_a_floor_is_refused():
    with pytest.raises(ValueError, match="floor_db"):
        resolve_tau_src({"training": {"tau_src": {"source": "labels"}}})


def test_the_predictor_source_needs_no_floor():
    assert resolve_tau_src({"training": {"tau_src": {"source": "predictor"}}}) == (
        "predictor", None
    )


# ---------------------------------------------------------------------------
# Script-level decisions
# ---------------------------------------------------------------------------

def test_the_target_range_prefers_the_runs_own_generation_block():
    train_config = {
        "training": {"intensity_range_db": [42.8, 76.1]},
        "generation": {"intensity_min_db": 45.0, "intensity_max_db": 75.0},
    }
    assert script.target_range({}, train_config) == (45.0, 75.0)


def test_the_target_range_falls_back_to_the_conditioning_range():
    train_config = {"training": {"intensity_range_db": [42.8, 76.1]}}
    assert script.target_range({}, train_config) == (42.8, 76.1)


def test_an_explicit_target_range_wins():
    train_config = {
        "training": {"intensity_range_db": [42.8, 76.1]},
        "generation": {"intensity_min_db": 45.0, "intensity_max_db": 75.0},
    }
    config = {"conversion": {"intensity_min_db": 50.0, "intensity_max_db": 70.0}}
    assert script.target_range(config, train_config) == (50.0, 70.0)


def test_utt_ids_are_unique_when_many_segments_share_a_file():
    metadata = pd.DataFrame({"signal_path": ["/data/a/s01.wav"] * 3 + ["/data/b/s02.wav"]})
    ids = script.make_utt_ids(metadata)
    assert len(set(ids)) == 4
    assert ids[0] == "00000_s01" and ids[3] == "00003_s02"


def test_an_output_above_full_scale_survives_the_write(tmp_path):
    """Unnormalised is the point: a loud target can decode past ±1.0.

    16-bit PCM would clip it and turn an overshoot the paper wants to report into a
    distortion it does not.  ``torchaudio.save`` does exactly that in torchaudio 2.11 —
    it accepts ``encoding="PCM_F"``, warns that TorchCodec does not fully support it, and
    writes 16-bit anyway — which is why the write goes through soundfile.
    """
    wav = torch.full((1, SR), 1.7)
    stats = script.save_unnormalized(wav, tmp_path / "loud.wav", SR)

    assert stats["peak_amplitude"] == pytest.approx(1.7)
    assert stats["duration_s"] == pytest.approx(1.0)
    assert stats["n_samples"] == SR

    read, sr = sf.read(tmp_path / "loud.wav", dtype="float32")
    assert sr == SR
    assert float(np.abs(read).max()) == pytest.approx(1.7, abs=1e-5)


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------

HOP = 320


def _stub_module(monkeypatch):
    """Point the script at the stub codec/predictor the module tests already use.

    Only the two builders are replaced.  Everything between reading the YAML and writing
    the CSV — the config merge, the split filter, the trim, the dataset choice, the
    conversion loop and the metadata — is the real code path.
    """
    from test_converter_cgan_v2_labels_module import _build
    from stubs import _StubExtractor, _StubPipeline

    from vic.core import FrameGrid

    class _SizedPipeline(_StubPipeline):
        """The stub plus the three attributes the renderer sizes its datasets from."""

        sample_rate = SR
        latent_dim = 8
        frame_grid = FrameGrid(hop_size=HOP, sample_rate=SR)

    extractor = _StubExtractor()
    pipeline = _SizedPipeline(extractor, whitening=False)
    label_pipeline = _SizedPipeline(extractor, whitening=True)

    module = _build(label_source="labels", floor_db=33.0)
    module.pipeline = pipeline
    module.label_pipeline = label_pipeline

    monkeypatch.setattr(
        script, "build_conversion_pipelines", lambda cfg: (pipeline, label_pipeline)
    )
    monkeypatch.setattr(
        script, "build_labels_module", lambda cfg, p, lp, ckpt_path=None: module
    )
    return module


def _corpus(tmp_path, groups=(("g1", ["soft", "normal", "loud", "veryloud"]),), n_train=1):
    """A miniature calibrated, level-annotated segment index over real files.

    One row per (group, level), plus ``n_train`` train rows to check the split filter.
    """
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    rows = []
    i = 0
    for sentence, (group, levels) in enumerate(groups, start=1):
        for level in levels:
            name = f"{group}_{level}.wav"
            sf.write(audio_dir / name,
                     np.random.randn(6 * SR).astype("float32") * 0.05, SR)
            rows.append({
                "corpus": "TEST", "signal_path": f"audio/{name}", "split": "test",
                "start_s": 0.5, "end_s": 5.5, "channel": 0,
                "speaker_uid": "TEST:1", "session_index": 1, "sentence_id": sentence,
                "repetition": 1,
                "level": level, "excluded": False,
                "calibration_rms": 0.01, "distance_m": 0.05,
            })
            i += 1
    for j in range(n_train):
        name = f"train{j:02d}.wav"
        sf.write(audio_dir / name, np.random.randn(6 * SR).astype("float32") * 0.05, SR)
        rows.append({
            "corpus": "TEST", "signal_path": f"audio/{name}", "split": "train",
            "start_s": 0.5, "end_s": 5.5, "channel": 0,
            "speaker_uid": "TEST:1", "session_index": 1, "sentence_id": 99,
            "repetition": 1,
            "level": "soft", "excluded": False,
            "calibration_rms": 0.01, "distance_m": 0.05,
        })
    csv = tmp_path / "metadata.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    return csv


def _configs(tmp_path, csv, levels=None, **conversion):
    import yaml

    train_config = {
        "data": {
            "metadata_csv": str(csv),
            "dataset_roots": {"TEST": str(tmp_path)},
            "chunk_duration_s": 2.0,
            "window_length": 400,
        },
        "extractor": {"type": "stub", "whitening": False, "normalize_sequence": False},
        "predictor": {"ckpt_path": "unused", "model": {"type": "stub"}},
        "model": {},
        "training": {
            "intensity_range_db": [42.8, 76.1],
            "tau_src": {"source": "labels", "floor_db": 33.0},
        },
        "generation": {"intensity_min_db": 45.0, "intensity_max_db": 75.0},
    }
    train_path = tmp_path / "train.yaml"
    train_path.write_text(yaml.safe_dump(train_config))

    config = {
        "converter": {"config_path": str(train_path), "ckpt_path": "unused"},
        "data": {"split": "test"},
        "levels": levels if levels is not None else {"enabled": True},
        "conversion": {"n_targets": 3, "save_codec": True, **conversion},
        "num_workers": 0,
    }
    config_path = tmp_path / "render.yaml"
    config_path.write_text(yaml.safe_dump(config))
    return config_path


def _run(tmp_path, monkeypatch, config_path):
    out_dir = tmp_path / "render"
    out_dir.mkdir()
    monkeypatch.setattr(
        "sys.argv",
        ["convert_test_set.py", "--config", str(config_path),
         "--output-dir", str(out_dir)],
    )
    script.main()
    return out_dir




def test_the_renderer_writes_a_file_and_a_row_per_target(tmp_path, monkeypatch):
    """One group of four levels: 4 real + 4 codec + 4x3 linspace + 4x4 level."""
    _stub_module(monkeypatch)
    _stub_islands(monkeypatch, [(0.5, 3.5)])
    out_dir = _run(tmp_path, monkeypatch, _configs(tmp_path, _corpus(tmp_path)))

    out = pd.read_csv(out_dir / "conversion_metadata.csv")
    assert len(out) == 4 + 4 + 4 * 3 + 4 * 4
    assert sorted(out["condition"].unique()) == ["codec", "converted", "real"]
    # The train row is filtered out before anything is rendered.
    assert out["source_utt_id"].nunique() == 4

    # utt_id is unique per FILE — speech-eval's manifest reader refuses duplicates.
    assert out["utt_id"].is_unique
    for _, row in out.iterrows():
        path = out_dir / row["signal_path"]
        assert path.exists(), row["signal_path"]
        assert sf.info(path).subtype == "FLOAT"


def test_the_two_target_families_are_indexed_separately(tmp_path, monkeypatch):
    """`linspace_index` runs 1..8 and `level_index` 1..4, and never both on one row."""
    _stub_module(monkeypatch)
    _stub_islands(monkeypatch, [(0.5, 3.5)])
    out_dir = _run(tmp_path, monkeypatch, _configs(tmp_path, _corpus(tmp_path)))

    conv = pd.read_csv(out_dir / "conversion_metadata.csv").query("condition == 'converted'")
    lin = conv[conv["target_kind"] == "linspace"]
    lvl = conv[conv["target_kind"] == "level"]

    assert sorted(lin["linspace_index"].unique()) == [1, 2, 3]      # n_targets=3 here
    assert lin["level_index"].isna().all()
    assert lin["level_name"].isna().all()

    assert sorted(lvl["level_index"].unique()) == [1, 2, 3, 4]
    assert lvl["linspace_index"].isna().all()
    # soft=1 .. veryloud=4, the elicited order, not an alphabetical one.
    ranks = lvl.drop_duplicates("level_name").set_index("level_name")["level_index"]
    assert ranks["soft"] == 1 and ranks["veryloud"] == 4


def test_a_level_target_names_the_real_recording_it_should_be_compared_with(
    tmp_path, monkeypatch
):
    """The point of the level family: the destination is a real file, not a number.

    `reference_utt_id` must resolve to a `real` row of the SAME sentence group at the
    requested level, and the τ asked for must be that recording's own measured τ_src.
    """
    _stub_module(monkeypatch)
    _stub_islands(monkeypatch, [(0.5, 3.5)])
    out_dir = _run(tmp_path, monkeypatch, _configs(tmp_path, _corpus(tmp_path)))

    out = pd.read_csv(out_dir / "conversion_metadata.csv")
    real = out[out["condition"] == "real"].set_index("utt_id")
    lvl = out.query("condition == 'converted' and target_kind == 'level'")
    assert len(lvl) == 16

    for _, row in lvl.iterrows():
        reference = real.loc[row["reference_utt_id"]]
        assert reference["sentence_group_id"] == row["sentence_group_id"]
        assert reference["source_level"] == row["level_name"]
        # The requested τ IS the reference recording's measured source level.
        assert reference["tau_src_db"] == pytest.approx(row["tau_tgt_db"])
        assert (out_dir / row["reference_signal_path"]).exists()


def test_the_source_level_is_not_overwritten_by_the_destination(tmp_path, monkeypatch):
    """Both axes of the 4x4 heatmap have to survive onto the same row.

    A conversion row carries the destination in `level_index`; the source's own level had
    to be renamed to `source_level_index` or the destination would silently overwrite it
    and the heatmap would lose its source axis entirely.
    """
    _stub_module(monkeypatch)
    _stub_islands(monkeypatch, [(0.5, 3.5)])
    out_dir = _run(tmp_path, monkeypatch, _configs(tmp_path, _corpus(tmp_path)))

    lvl = pd.read_csv(out_dir / "conversion_metadata.csv").query(
        "condition == 'converted' and target_kind == 'level'"
    )
    # All 16 ordered (source, destination) pairs are present and distinguishable.
    pairs = set(zip(lvl["source_level_index"], lvl["level_index"]))
    assert pairs == {(s, d) for s in (1, 2, 3, 4) for d in (1, 2, 3, 4)}
    # The four identity cases are exactly the diagonal.
    identity = lvl[lvl["source_level_index"] == lvl["level_index"]]
    assert len(identity) == 4
    assert np.allclose(identity["delta_tau_db"], 0.0)


def test_a_partial_group_converts_between_the_levels_it_has(tmp_path, monkeypatch):
    """A missing level costs that destination, not the sentence."""
    _stub_module(monkeypatch)
    _stub_islands(monkeypatch, [(0.5, 3.5)])
    csv = _corpus(tmp_path, groups=(("g1", ["soft", "normal", "loud"]),), n_train=0)
    out_dir = _run(tmp_path, monkeypatch, _configs(tmp_path, csv))

    out = pd.read_csv(out_dir / "conversion_metadata.csv")
    lvl = out.query("condition == 'converted' and target_kind == 'level'")
    assert len(lvl) == 3 * 3
    assert sorted(lvl["level_index"].unique()) == [1, 2, 3]
    # Flagged, so an analysis wanting only complete groups can filter without re-rendering.
    assert (out["group_n_levels"] == 3).all()


def test_excluded_rows_are_never_rendered(tmp_path, monkeypatch):
    _stub_module(monkeypatch)
    _stub_islands(monkeypatch, [(0.5, 3.5)])
    csv = _corpus(tmp_path, n_train=0)
    table = pd.read_csv(csv)
    table.loc[table["level"] == "veryloud", "excluded"] = True
    table.to_csv(csv, index=False)

    out_dir = _run(tmp_path, monkeypatch, _configs(tmp_path, csv))

    out = pd.read_csv(out_dir / "conversion_metadata.csv")
    assert out["source_utt_id"].nunique() == 3
    assert "veryloud" not in set(out["level_name"].dropna())
    assert "veryloud" not in set(out["source_level"].dropna())


def test_levels_can_be_switched_off(tmp_path, monkeypatch):
    """Without the annotation the linspace family must still render."""
    _stub_module(monkeypatch)
    _stub_islands(monkeypatch, [(0.5, 3.5)])
    csv = _corpus(tmp_path, n_train=0)
    table = pd.read_csv(csv).drop(columns=["level", "sentence_id", "session_index"])
    table.to_csv(csv, index=False)

    out_dir = _run(
        tmp_path, monkeypatch, _configs(tmp_path, csv, levels={"enabled": False})
    )

    out = pd.read_csv(out_dir / "conversion_metadata.csv")
    conv = out[out["condition"] == "converted"]
    assert (conv["target_kind"] == "linspace").all()
    assert len(conv) == 4 * 3


def test_missing_annotation_columns_are_refused_not_ignored(tmp_path, monkeypatch):
    """Silently rendering only the linspace family would look like a successful run."""
    _stub_module(monkeypatch)
    _stub_islands(monkeypatch, [(0.5, 3.5)])
    csv = _corpus(tmp_path, n_train=0)
    pd.read_csv(csv).drop(columns=["level"]).to_csv(csv, index=False)

    with pytest.raises(ValueError, match="sentence groups cannot be formed"):
        _run(tmp_path, monkeypatch, _configs(tmp_path, csv))


def test_the_rendered_audio_is_the_trimmed_span(tmp_path, monkeypatch):
    """The trim must reach the audio, not just the CSV.

    ``start_s``/``end_s`` are rewritten in the metadata and the dataset seeks with them,
    so a source render that is still 5 s long means the trim was recorded and ignored.
    """
    _stub_module(monkeypatch)
    _stub_islands(monkeypatch, [(1.0, 3.0)])       # 2 s of a 5 s segment
    out_dir = _run(tmp_path, monkeypatch, _configs(tmp_path, _corpus(tmp_path)))

    trim = pd.read_csv(out_dir / "vad_trim.csv")
    assert trim.loc[0, "start_s"] == pytest.approx(1.5)   # 0.5 + 1.0
    assert trim.loc[0, "end_s"] == pytest.approx(3.5)     # 0.5 + 3.0

    out = pd.read_csv(out_dir / "conversion_metadata.csv")
    source = out[out["condition"] == "real"].iloc[0]
    assert source["duration_s"] == pytest.approx(2.0, abs=HOP / SR)


def test_a_no_speech_split_is_refused(tmp_path, monkeypatch):
    _stub_module(monkeypatch)
    _stub_islands(monkeypatch, [])
    with pytest.raises(ValueError, match="found no speech in any"):
        _run(tmp_path, monkeypatch, _configs(tmp_path, _corpus(tmp_path)))


def test_the_default_group_key_works(tmp_path, monkeypatch):
    """The module default is a tuple, and pandas reads a tuple of names as ONE key.

    Every config passes a YAML list, so this only ever failed when the default was used —
    which is exactly the path a caller reaching for the library directly would take.
    """
    from vic.data.levels import prepare_level_groups

    table = pd.read_csv(_corpus(tmp_path, n_train=0))
    out = prepare_level_groups(table)          # no group_columns argument
    assert out["sentence_group_id"].nunique() == 1
    assert (out["group_n_levels"] == 4).all()


# ---------------------------------------------------------------------------
# Sentence selection
# ---------------------------------------------------------------------------

def _multi_sentence_corpus(tmp_path, sentences=4):
    groups = tuple((f"s{n}", list(avid_levels.LEVELS)) for n in range(1, sentences + 1))
    return _corpus(tmp_path, groups=groups, n_train=0)


def test_selecting_sentences_keeps_whole_sentences(tmp_path, monkeypatch):
    _stub_module(monkeypatch)
    _stub_islands(monkeypatch, [(0.5, 3.5)])
    csv = _multi_sentence_corpus(tmp_path)
    out_dir = _run(tmp_path, monkeypatch,
                   _configs(tmp_path, csv, levels={"enabled": True,
                                                   "sentence_ids": [2, 4]}))

    out = pd.read_csv(out_dir / "conversion_metadata.csv")
    assert sorted(out["sentence_id"].unique()) == [2, 4]
    # Both selected sentences keep all four levels, so both stay 4x4.
    assert out["source_utt_id"].nunique() == 8
    lvl = out.query("condition == 'converted' and target_kind == 'level'")
    assert len(lvl) == 2 * 16
    assert (out["group_n_levels"] == 4).all()


def test_dropping_other_sentences_does_not_break_a_group(tmp_path, monkeypatch):
    """Selection removes whole sentences, so no retained group loses a level."""
    table = pd.read_csv(_multi_sentence_corpus(tmp_path))
    selected = avid_levels.prepare_level_groups(table, sentence_ids=[1, 3])
    every = avid_levels.prepare_level_groups(table)

    kept = selected.groupby("sentence_group_id")["group_n_levels"].first()
    full = every.groupby("sentence_group_id")["group_n_levels"].first()
    assert kept.to_dict() == full.loc[kept.index].to_dict()
    assert set(kept.index) < set(full.index)          # a strict subset of the groups


def test_a_sentence_number_matching_nothing_is_refused(tmp_path):
    """A typo in a config would otherwise render fewer sentences than asked for."""
    table = pd.read_csv(_multi_sentence_corpus(tmp_path))
    with pytest.raises(ValueError, match=r"sentence_ids \[99\] match no row"):
        avid_levels.prepare_level_groups(table, sentence_ids=[1, 99])


def test_selecting_sentences_without_the_annotation_is_refused(tmp_path, monkeypatch):
    _stub_module(monkeypatch)
    _stub_islands(monkeypatch, [(0.5, 3.5)])
    csv = _multi_sentence_corpus(tmp_path)
    config = _configs(tmp_path, csv,
                      levels={"enabled": False, "sentence_ids": [1, 2]})
    with pytest.raises(ValueError, match="levels.enabled: false"):
        _run(tmp_path, monkeypatch, config)


def test_repetition_is_part_of_the_default_group_key(tmp_path):
    """The two passes must not merge into one eight-recording group.

    sentence_id restarts at 1 in the second reading of the list, so without `repetition`
    a group holds two takes at every level and every destination becomes ambiguous.
    """
    one_pass = pd.read_csv(_multi_sentence_corpus(tmp_path, sentences=2))
    second = one_pass.copy()
    second["repetition"] = 2
    # The second reading is renumbered from 1, which is exactly what makes sentence_id
    # ambiguous on its own.
    table = pd.concat([one_pass, second], ignore_index=True)
    assert table["sentence_id"].tolist() == [1] * 4 + [2] * 4 + [1] * 4 + [2] * 4

    grouped = avid_levels.prepare_level_groups(table)
    assert grouped["sentence_group_id"].nunique() == 4
    assert (grouped.groupby("sentence_group_id").size() == 4).all()

    merged = avid_levels.prepare_level_groups(
        table, group_columns=["speaker_uid", "session_index", "sentence_id"]
    )
    assert merged["sentence_group_id"].nunique() == 2      # the failure being guarded

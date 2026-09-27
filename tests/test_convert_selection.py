"""Tests for the hand-picked listening render.

Two things carry the whole result and both fail silently if wrong: the mapping from the
selection's column names onto the corpus's (the corpus has its own all-null ``speaker``
and ``sentence`` columns, so a name match finds nothing and an empty render looks like a
successful one), and the filename, which is the *only* record this script writes.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

import vic.data.vad as vad_module
from vic.data.levels import LEVELS

SR = 16000
HOP = 320

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "convert_selection.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("convert_selection", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


script = _load_script()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _corpus_table(tmp_path, speakers=(9, 34), sentences=(1, 3)):
    """Two passes of every (speaker, sentence, level), as the annotated corpus has it."""
    audio = tmp_path / "audio"
    audio.mkdir(exist_ok=True)
    rows = []
    for speaker in speakers:
        for repetition in (1, 2):
            for sentence in sentences:
                for level in LEVELS:
                    name = f"spk{speaker}_s{sentence}_r{repetition}_{level}.wav"
                    sf.write(audio / name,
                             np.random.randn(6 * SR).astype("float32") * 0.05, SR)
                    rows.append({
                        "corpus": "TEST", "signal_path": f"audio/{name}",
                        "split": "test", "start_s": 0.5, "end_s": 5.5, "channel": 0,
                        # float, as the real table stores them
                        "subject_id": float(speaker), "sentence_id": float(sentence),
                        "level": level, "repetition": repetition,
                        "speaker_uid": f"TEST:{speaker}", "session_index": 1,
                        "calibration_rms": 0.01, "distance_m": 0.05,
                        # The trap: present, and null on every row.
                        "speaker": np.nan, "sentence": np.nan,
                    })
    path = tmp_path / "corpus.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _selection(tmp_path, rows=((9, 1, "soft", 1), (34, 3, "veryloud", 1))):
    path = tmp_path / "selection.csv"
    pd.DataFrame(rows, columns=["speaker", "sentence", "level", "pass"]).to_csv(
        path, index=False)
    return path


def _stub_module(monkeypatch):
    from test_converter_cgan_v2_labels_module import _build
    from stubs import _StubExtractor, _StubPipeline

    from vic.core import FrameGrid

    class _SizedPipeline(_StubPipeline):
        sample_rate = SR
        latent_dim = 8
        frame_grid = FrameGrid(hop_size=HOP, sample_rate=SR)

    extractor = _StubExtractor()
    pipeline = _SizedPipeline(extractor, whitening=False)
    label_pipeline = _SizedPipeline(extractor, whitening=True)
    module = _build(label_source="labels", floor_db=33.0)
    module.pipeline, module.label_pipeline = pipeline, label_pipeline

    monkeypatch.setattr(script, "build_conversion_pipelines",
                        lambda cfg: (pipeline, label_pipeline))
    monkeypatch.setattr(script, "build_labels_module",
                        lambda cfg, p, lp, ckpt_path=None: module)
    return module


def _stub_islands(monkeypatch, islands_s=((0.5, 3.5),)):
    def fake(waveform, model, sampling_rate, **kwargs):
        return [{"start": round(s * sampling_rate), "end": round(e * sampling_rate)}
                for s, e in islands_s]

    import silero_vad
    monkeypatch.setattr(silero_vad, "get_speech_timestamps", fake)
    monkeypatch.setattr(vad_module, "load_vad_model", lambda onnx=False: object())


def _configs(tmp_path, corpus_csv, selection_csv, n_targets=8, **selection_extra):
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
        "generation": {"intensity_min_db": 45.0, "intensity_max_db": 75.0},
    }
    train_path = tmp_path / "train.yaml"
    train_path.write_text(yaml.safe_dump(train))

    config = {
        "converter": {"config_path": str(train_path), "ckpt_path": "unused"},
        "selection": {"csv": str(selection_csv), **selection_extra},
        "conversion": {"n_targets": n_targets},
        "num_workers": 0,
    }
    path = tmp_path / "render.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def _run(tmp_path, monkeypatch, config_path):
    out = tmp_path / "out"
    out.mkdir()
    monkeypatch.setattr("sys.argv", ["convert_selection.py", "--config",
                                     str(config_path), "--output-dir", str(out)])
    script.main()
    return out


# ---------------------------------------------------------------------------
# Naming — the only record this script writes
# ---------------------------------------------------------------------------

def test_the_stem_format():
    assert script.stem(9, 1, 1, 1) == "spk9_sent1_rep1_src1"
    assert script.stem(9, 1, 1, 1, 8) == "spk9_sent1_rep1_src1_tgt8"
    # Floats out of a CSV must not leak "9.0" into a filename.
    assert script.stem(9.0, 1.0, 2.0, 4.0, 3.0) == "spk9_sent1_rep2_src4_tgt3"


def test_the_expected_files_are_written(tmp_path, monkeypatch):
    _stub_module(monkeypatch)
    _stub_islands(monkeypatch)
    out = _run(tmp_path, monkeypatch,
               _configs(tmp_path, _corpus_table(tmp_path), _selection(tmp_path)))

    for directory in (script.RAW_DIR, script.NORM_DIR):
        names = sorted(p.name for p in (out / directory).glob("*.mp3"))
        # Per selection: 8 conversions + the whole sentence (4 levels x 2 passes).
        assert len(names) == 2 * (8 + 8)
        # Selection row 1: speaker 9, sentence 1, soft (src 1), pass 1.
        for k in range(1, 9):
            assert f"spk9_sent1_rep1_src1_tgt{k}.mp3" in names
        # …and every real take of that sentence, both passes, all four levels.
        for repetition in (1, 2):
            for rank in (1, 2, 3, 4):
                assert f"spk9_sent1_rep{repetition}_src{rank}.mp3" in names
                assert f"spk34_sent3_rep{repetition}_src{rank}.mp3" in names
        # Conversions are made from the selected recording only.
        assert not any(n.startswith("spk9_sent1_rep2_src1_tgt") for n in names)
        assert not any(n.startswith("spk9_sent1_rep1_src2_tgt") for n in names)


def test_the_src_index_follows_the_effort_rank(tmp_path, monkeypatch):
    _stub_module(monkeypatch)
    _stub_islands(monkeypatch)
    selection = _selection(tmp_path, rows=[(9, 1, level, 1) for level in LEVELS])
    out = _run(tmp_path, monkeypatch,
               _configs(tmp_path, _corpus_table(tmp_path), selection, n_targets=2))

    names = {p.name for p in (out / script.RAW_DIR).glob("*.mp3")}
    for rank, _ in enumerate(LEVELS, start=1):
        # Every level is both a source (it was selected) and an original.
        assert f"spk9_sent1_rep1_src{rank}.mp3" in names
        assert f"spk9_sent1_rep1_src{rank}_tgt1.mp3" in names
    # Four selections over ONE sentence: the block is shared, not written four times.
    assert len(names) == 4 * 2 + 4 * 2


def test_both_directories_hold_the_same_names_but_different_audio(tmp_path, monkeypatch):
    _stub_module(monkeypatch)
    _stub_islands(monkeypatch)
    out = _run(tmp_path, monkeypatch,
               _configs(tmp_path, _corpus_table(tmp_path), _selection(tmp_path),
                        n_targets=2))

    raw = sorted(p.name for p in (out / script.RAW_DIR).glob("*.mp3"))
    norm = sorted(p.name for p in (out / script.NORM_DIR).glob("*.mp3"))
    assert raw == norm

    peaks_norm = np.array(
        [float(np.abs(sf.read(out / script.NORM_DIR / n)[0]).max()) for n in norm])
    peaks_raw = np.array(
        [float(np.abs(sf.read(out / script.RAW_DIR / n)[0]).max()) for n in raw])

    # The property that matters: normalisation brings every file to a common peak, so
    # amplitude stops carrying information.  Asserted as spread rather than an absolute
    # window because MP3 is lossy and its reconstruction overshoots — about 6% on
    # speech-like content, much more on the random noise this stub decodes.
    assert peaks_norm.std() / peaks_norm.mean() < peaks_raw.std() / peaks_raw.mean()
    assert not np.allclose(peaks_raw, peaks_norm)


def test_the_files_are_mp3_at_the_codec_rate(tmp_path, monkeypatch):
    _stub_module(monkeypatch)
    _stub_islands(monkeypatch)
    out = _run(tmp_path, monkeypatch,
               _configs(tmp_path, _corpus_table(tmp_path), _selection(tmp_path),
                        n_targets=1))
    for path in (out / script.RAW_DIR).glob("*.mp3"):
        info = sf.info(path)
        assert info.format == "MP3"
        assert info.samplerate == SR


# ---------------------------------------------------------------------------
# The column mapping
# ---------------------------------------------------------------------------

def test_the_corpus_columns_of_the_same_name_are_not_used(tmp_path):
    """AVID's own `speaker`/`sentence` are null everywhere; matching them finds nothing."""
    corpus = pd.read_csv(_corpus_table(tmp_path))
    selection = pd.read_csv(_selection(tmp_path))
    assert corpus["speaker"].isna().all() and corpus["sentence"].isna().all()

    resolved = script.resolve_selection(
        selection, corpus,
        script.DEFAULT_SELECTION_COLUMNS, script.DEFAULT_CORPUS_COLUMNS)
    assert len(resolved) == 16                 # 2 sentences x 4 levels x 2 passes
    assert int(resolved["is_source"].sum()) == 2

    with pytest.raises(ValueError, match="matches 0 recording"):
        script.resolve_selection(
            selection, corpus, script.DEFAULT_SELECTION_COLUMNS,
            {**script.DEFAULT_CORPUS_COLUMNS, "speaker": "speaker"})


def test_a_renamed_selection_column_is_honoured(tmp_path, monkeypatch):
    _stub_module(monkeypatch)
    _stub_islands(monkeypatch)
    selection = pd.read_csv(_selection(tmp_path)).rename(columns={"speaker": "spk"})
    path = tmp_path / "renamed.csv"
    selection.to_csv(path, index=False)

    out = _run(tmp_path, monkeypatch,
               _configs(tmp_path, _corpus_table(tmp_path), path, n_targets=1,
                        columns={"speaker": "spk"}))
    assert (out / script.RAW_DIR / "spk9_sent1_rep1_src1.mp3").exists()


def test_a_numeric_type_mismatch_still_matches(tmp_path):
    """The corpus stores 9.0 and a hand-written selection says 9."""
    corpus = pd.read_csv(_corpus_table(tmp_path))
    assert corpus["subject_id"].dtype == float
    selection = pd.read_csv(_selection(tmp_path))
    assert selection["speaker"].dtype == int

    resolved = script.resolve_selection(
        selection, corpus,
        script.DEFAULT_SELECTION_COLUMNS, script.DEFAULT_CORPUS_COLUMNS)
    assert set(resolved["out_speaker"]) == {9, 34}


def test_a_missing_selection_column_is_named(tmp_path):
    corpus = pd.read_csv(_corpus_table(tmp_path))
    selection = pd.read_csv(_selection(tmp_path)).drop(columns=["pass"])
    with pytest.raises(ValueError, match=r"no \['pass'\] column"):
        script.resolve_selection(
            selection, corpus,
            script.DEFAULT_SELECTION_COLUMNS, script.DEFAULT_CORPUS_COLUMNS)


def test_an_unknown_level_is_refused(tmp_path):
    corpus = pd.read_csv(_corpus_table(tmp_path))
    selection = _selection(tmp_path, rows=((9, 1, "SOFT", 1),))
    with pytest.raises(ValueError, match="unknown level"):
        script.resolve_selection(
            pd.read_csv(selection), corpus,
            script.DEFAULT_SELECTION_COLUMNS, script.DEFAULT_CORPUS_COLUMNS)


def test_a_missing_level_costs_that_take_and_nothing_else(tmp_path):
    """Two sentences of the corpus really are short a level; the rest stays usable."""
    corpus = pd.read_csv(_corpus_table(tmp_path))
    corpus = corpus[~((corpus.subject_id == 9.0) & (corpus.sentence_id == 1.0)
                      & (corpus.level == "veryloud") & (corpus.repetition == 2))]
    selection = pd.read_csv(_selection(tmp_path, rows=((9, 1, "soft", 1),)))

    resolved = script.resolve_selection(
        selection, corpus,
        script.DEFAULT_SELECTION_COLUMNS, script.DEFAULT_CORPUS_COLUMNS)
    assert len(resolved) == 7                                  # 8 minus the absent take
    assert int(resolved["is_source"].sum()) == 1


def test_two_selections_on_one_sentence_do_not_duplicate_its_originals(tmp_path):
    """Both pull the whole block; the originals must be written once, and stay sources."""
    corpus = pd.read_csv(_corpus_table(tmp_path))
    selection = pd.read_csv(_selection(
        tmp_path, rows=((9, 1, "soft", 1), (9, 1, "loud", 2))))

    resolved = script.resolve_selection(
        selection, corpus,
        script.DEFAULT_SELECTION_COLUMNS, script.DEFAULT_CORPUS_COLUMNS)
    assert len(resolved) == 8                                  # one block, not two
    assert int(resolved["is_source"].sum()) == 2               # both selections survive


def test_a_selection_matching_nothing_is_refused(tmp_path):
    corpus = pd.read_csv(_corpus_table(tmp_path))
    selection = pd.read_csv(_selection(tmp_path, rows=((99, 1, "soft", 1),)))
    with pytest.raises(ValueError, match="matches 0 recording"):
        script.resolve_selection(
            selection, corpus,
            script.DEFAULT_SELECTION_COLUMNS, script.DEFAULT_CORPUS_COLUMNS)


# ---------------------------------------------------------------------------
# VAD
# ---------------------------------------------------------------------------

def test_the_originals_are_the_trimmed_span(tmp_path, monkeypatch):
    _stub_module(monkeypatch)
    _stub_islands(monkeypatch, islands_s=((1.0, 3.0),))      # 2 s of a 5 s segment
    out = _run(tmp_path, monkeypatch,
               _configs(tmp_path, _corpus_table(tmp_path), _selection(tmp_path),
                        n_targets=1))
    info = sf.info(out / script.RAW_DIR / "spk9_sent1_rep1_src1.mp3")
    assert info.frames / info.samplerate == pytest.approx(2.0, abs=0.05)


def test_a_silent_selected_recording_stops_the_run(tmp_path, monkeypatch):
    """A hand-picked set is small enough that a dropped row is a hole, not a statistic."""
    _stub_module(monkeypatch)
    _stub_islands(monkeypatch, islands_s=())
    with pytest.raises(ValueError, match="found no speech"):
        _run(tmp_path, monkeypatch,
             _configs(tmp_path, _corpus_table(tmp_path), _selection(tmp_path)))

"""Tests for the mel and WavLM conversion domains.

Split in two on purpose.  The first part needs no third-party checkout and pins the
arithmetic that decides whether the frame labels line up with the features -- a class of
bug that never raises, it just measures tau_src over the wrong span of audio.  The second
part loads the real encoders and decoders and is skipped unless ``VIC_VOCODER_DIR`` points
at a directory populated by ``scripts/download_vocoders.py``:

    VIC_VOCODER_DIR=/path/to/vocoders uv run --extra cpu pytest tests/test_vocoder_codecs.py

The exactness claim each codec's docstring makes -- that ``encode`` is upstream's own
function and not a reimplementation -- is checked there, by calling upstream directly and
requiring bit equality.
"""
import os
from pathlib import Path

import pytest
import torch

from vic.codec._vendor import add_repo_to_path
from vic.codec.wavlm_hifigan import _CONV_RECEPTIVE_FIELD
from vic.data.audio_batch import AudioBatch
from vic.data.transforms import resample_batch
from vic.training.extraction_pipeline import ExtractionPipeline, build_extractor

VOCODER_DIR = os.environ.get("VIC_VOCODER_DIR")
needs_weights = pytest.mark.skipif(
    not VOCODER_DIR,
    reason="set VIC_VOCODER_DIR to a directory from scripts/download_vocoders.py",
)


# ---------------------------------------------------------------------------
# Frame-grid arithmetic.  No weights involved.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_frames", [1, 2, 5, 40, 100, 313])
def test_wavlm_left_pad_recovers_the_frame_grid(n_frames):
    """The 200-sample left pad must turn WavLM's ``T//320 - 1`` into ``T//320``.

    WavLM's conv stack has stride 320 and receptive field 400, so it emits
    ``floor((T - 400) / 320) + 1`` frames -- one short of what ``FrameGrid.n_frames``
    and therefore ``frame_intensity_db`` assume.  Reflect-padding half the receptive
    field on the left fixes the count *and* centres frame i at sample ``320 * i``.
    ``ConverterCGANv2LabelsModule._check_alignment`` raises on a mismatch, so getting
    this wrong is loud -- but only once a real checkpoint is loaded, which is why the
    arithmetic is pinned here.
    """
    hop, left_pad = 320, _CONV_RECEPTIVE_FIELD // 2
    T = hop * n_frames

    unpadded = (T - _CONV_RECEPTIVE_FIELD) // hop + 1
    padded = (T + left_pad - _CONV_RECEPTIVE_FIELD) // hop + 1

    assert padded == T // hop
    assert unpadded == T // hop - 1        # what kNN-VC's own call returns


def test_wavlm_left_pad_is_half_the_receptive_field():
    """Frame i must analyse ``[320i - 200, 320i + 200)``, i.e. be centred at ``320i``."""
    assert _CONV_RECEPTIVE_FIELD // 2 == 200


@pytest.mark.parametrize("n_fft,hop", [(1024, 256), (1024, 320), (2048, 512)])
def test_mel_centre_offset_is_half_a_hop(n_fft, hop):
    """Upstream pads ``(n_fft - hop)//2`` where centring needs ``n_fft//2``.

    The difference is what ``FrameGrid(causal=True)`` encodes as ``center_offset``, and
    it is ``hop // 2`` for any even ``n_fft`` and ``hop`` -- which is why the mel codec
    can declare the offset from the flag rather than carrying its own field.
    """
    assert n_fft // 2 - (n_fft - hop) // 2 == hop // 2


# ---------------------------------------------------------------------------
# resample_batch, and the pipeline that calls it.
# ---------------------------------------------------------------------------

def _wav(B=2, T=16000, sr=16000, lengths=None):
    lengths = torch.tensor([T] * B) if lengths is None else torch.tensor(lengths)
    return AudioBatch(data=torch.randn(B, 1, T) * 0.1, lengths=lengths, sample_rate=sr)


def test_resample_batch_is_identity_at_the_same_rate():
    batch = _wav()
    assert resample_batch(batch, 16000) is batch


def test_resample_batch_scales_data_and_lengths():
    batch = _wav(T=16000, lengths=[16000, 8000])
    out = resample_batch(batch, 22050)

    assert out.sample_rate == 22050
    assert out.data.shape[-1] == pytest.approx(22050, abs=2)
    assert out.lengths.tolist() == [22050, 11025]
    assert out.lengths.dtype == torch.long


def test_resample_batch_never_reports_a_length_past_the_tensor():
    """The resampler's output length is a ceiling, so a scaled length can overshoot."""
    out = resample_batch(_wav(T=16000 + 1, lengths=[16001, 16001]), 22050)
    assert int(out.lengths.max()) <= out.data.shape[-1]


class _FakeExtractor(torch.nn.Module):
    """Records the rate it was handed.  Stands in for a codec at a non-16 kHz rate."""

    def __init__(self, sample_rate):
        super().__init__()
        self.sample_rate = sample_rate
        self.latent_dim = 4
        self.seen_rate = None

    def encode(self, batch):
        self.seen_rate = batch.sample_rate
        n = batch.data.shape[-1] // 100
        return batch.as_frames(torch.zeros(batch.batch_size, 4, n), batch.lengths // 100)


@pytest.mark.parametrize("extractor_rate", [16000, 22050])
def test_pipeline_hands_the_extractor_its_own_rate(extractor_rate):
    """A 22.05 kHz mel vocoder as the conversion domain must not put 22.05 kHz audio
    into the 16 kHz measurement pipeline, and vice versa.  ``measure_intensity`` decodes
    at the conversion rate and then calls the label pipeline, so the resample has to
    happen inside ``ExtractionPipeline.encode``."""
    extractor = _FakeExtractor(extractor_rate)
    ExtractionPipeline(extractor=extractor).encode(_wav(sr=16000))
    assert extractor.seen_rate == extractor_rate


# ---------------------------------------------------------------------------
# Vendored-repo resolution.
# ---------------------------------------------------------------------------

def test_add_repo_to_path_names_the_path_it_was_given(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        add_repo_to_path(tmp_path / "nope", name="BigVGAN", expect=["bigvgan.py"])


def test_add_repo_to_path_rejects_a_directory_that_is_not_the_repo(tmp_path):
    """A path pointing one level too deep is the likely mistake, and
    ``ModuleNotFoundError: No module named 'bigvgan'`` does not say so."""
    with pytest.raises(FileNotFoundError, match="missing"):
        add_repo_to_path(tmp_path, name="BigVGAN", expect=["bigvgan.py", "meldataset.py"])


def test_build_extractor_still_rejects_an_unknown_type():
    with pytest.raises(ValueError, match="Unknown extractor type"):
        build_extractor({"extractor": {"type": "not_a_codec"}})


def test_mel_vocoder_rejects_an_unknown_backend(tmp_path):
    from vic.codec.mel_vocoder import MelVocoderCodec

    with pytest.raises(ValueError, match="Unknown mel vocoder backend"):
        MelVocoderCodec(repo_path=tmp_path, model_dir=tmp_path, backend="vocos")


def test_wavlm_codec_refuses_a_layer_the_decoder_was_not_trained_on(tmp_path):
    """``layer`` documents the coupling to the checkpoint; it is not a sweep axis."""
    from vic.codec.wavlm_hifigan import WavLMHiFiGANCodec

    repo = tmp_path / "knn-vc"
    (repo / "wavlm").mkdir(parents=True)
    (repo / "hifigan").mkdir()
    with pytest.raises(ValueError, match="layer 6"):
        WavLMHiFiGANCodec(
            repo_path=repo, wavlm_ckpt=tmp_path / "w.pt",
            hifigan_ckpt=tmp_path / "g.pt", layer=9,
        )


# ---------------------------------------------------------------------------
# The domain sweep config.  Needs no weights -- it only expands the sweep.
# ---------------------------------------------------------------------------

_DOMAIN_SWEEP = (
    Path(__file__).resolve().parents[1] / "configs" / "paper" / "train_converter.yaml"
)

# Keys each extractor branch of build_extractor reads, and the repo checkout each type
# must be pointed at.  `repo_path` appears in both, with DIFFERENT values -- which is the
# whole reason this test exists.
#
# `config_path` is the same shape of trap one key further along: `nac` requires it and
# `wavlm_hifigan` reads it as an optional HiFi-GAN config override, so the WavLM cell
# carrying one would silently build its generator from a SpeechTokenizer YAML.  The
# foreign-key assertion below is what catches that.
_DOMAIN_KEYS = {
    "mel_vocoder": ({"repo_path", "model_dir", "backend"}, "BigVGAN", 22050),
    "wavlm_hifigan": ({"repo_path", "wavlm_ckpt", "hifigan_ckpt"}, "knn-vc", 16000),
    "nac": ({"config_path", "ckpt_path"}, None, 16000),
}
_SHARED_KEYS = {
    "type", "whitening", "normalize_features", "normalize_features_per_band",
    "normalize_sequence",
}


def test_domain_sweep_cells_are_self_consistent():
    """Every cell of the domain sweep must carry its own domain's keys and no others.

    ``GroupedConfigurationSweeper._apply_overrides`` assigns leaf keys into an existing
    dict, so it cannot swap a whole block.  An earlier draft of the config put the shared
    *name* ``repo_path`` in the base ``extractor`` block, and the WavLM cell silently
    inherited the BigVGAN checkout -- which raises, but only once the job is running on the
    cluster.  Expanding the sweep here turns that into a local failure.

    Also checks the pairing that makes the whole comparison valid: the SPL analysis window
    is a count of samples, so it must be 25 ms at whatever rate the cell's codec runs at.
    """
    yaml = pytest.importorskip("yaml")
    sweeper = pytest.importorskip(
        "experiment_launcher.grouped_configuration_sweeper"
    ).GroupedConfigurationSweeper()

    configs, _ = sweeper.get_sweeped_configs(yaml.safe_load(_DOMAIN_SWEEP.read_text()))
    assert len(configs) == 3

    seen = set()
    for config in configs:
        extractor = config["extractor"]
        kind = extractor["type"]
        required, checkout, rate = _DOMAIN_KEYS[kind]
        seen.add(kind)

        assert not required - set(extractor), f"{kind} is missing {required - set(extractor)}"
        foreign = set(extractor) - _SHARED_KEYS - required
        assert not foreign, f"{kind} carries another domain's keys: {foreign}"
        if checkout is not None:      # nac loads a checkpoint, not a checkout
            assert extractor["repo_path"].rstrip("/").endswith(checkout), (
                f"{kind} points at {extractor['repo_path']}, not a {checkout} checkout"
            )
        # 400 samples at 16 kHz and 551 at 22.05 kHz are both 25 ms, which is the window
        # P_φ was trained with; a mismatch makes val/ruler_bias_db unreadable.
        assert config["data"]["window_length"] == round(0.025 * rate)

    assert seen == set(_DOMAIN_KEYS), "every domain must appear exactly once"


def test_wavlm_cell_uses_the_plain_generator():
    """kNN-VC's prematching matches ITS inference distribution, not ours.

    Our decoder input is a transformed raw encoder output -- at tau_tgt == tau_src it is
    the raw encoder output exactly -- so the plain checkpoint is the one whose training
    inputs this pipeline produces.  Both files sit in the same directory with names one
    prefix apart, which is how a revert happens by accident.  See the rationale in
    ``vic/codec/wavlm_hifigan.py``.
    """
    yaml = pytest.importorskip("yaml")
    sweeper = pytest.importorskip(
        "experiment_launcher.grouped_configuration_sweeper"
    ).GroupedConfigurationSweeper()

    configs, _ = sweeper.get_sweeped_configs(yaml.safe_load(_DOMAIN_SWEEP.read_text()))
    cells = [c for c in configs if c["extractor"]["type"] == "wavlm_hifigan"]
    assert len(cells) == 1
    assert Path(cells[0]["extractor"]["hifigan_ckpt"]).name == "g_02500000.pt"


def test_domain_sweep_disables_both_attention_windows():
    """attn_window is a half-width in FRAMES and the cells differ in frame rate, so a
    non-null value would make the effective context differ per cell and confound the
    comparison with the representation."""
    yaml = pytest.importorskip("yaml")
    config = yaml.safe_load(_DOMAIN_SWEEP.read_text())

    assert config["model"]["converter"]["attn_window"] is None
    assert config["model"]["discriminator"]["attn_window"] is None


# ---------------------------------------------------------------------------
# The real thing.  Requires VIC_VOCODER_DIR.
# ---------------------------------------------------------------------------

def _mel_config():
    root = Path(VOCODER_DIR)
    return {"extractor": {
        "type": "mel_vocoder", "backend": "bigvgan",
        "repo_path": str(root / "BigVGAN"),
        "model_dir": str(root / "bigvgan_v2_22khz_80band_fmax8k_256x"),
    }}


def _wavlm_config():
    root = Path(VOCODER_DIR)
    return {"extractor": {
        "type": "wavlm_hifigan",
        "repo_path": str(root / "knn-vc"),
        "wavlm_ckpt": str(root / "knn-vc-weights" / "WavLM-Large.pt"),
        # The plain generator, matching the sweep config: the weight-gated tests should
        # load the checkpoint the runs actually use.
        "hifigan_ckpt": str(root / "knn-vc-weights" / "g_02500000.pt"),
    }}


@needs_weights
def test_mel_codec_matches_the_checkpoints_own_hyperparameters():
    codec = build_extractor(_mel_config())
    assert codec.sample_rate == 22050
    assert codec.latent_dim == 80
    assert codec.frame_grid.hop_size == 256
    # fmax must be the source band, not Nyquist -- see the module docstring.
    assert int(codec._h.fmax) == 8000


@needs_weights
def test_mel_encode_is_upstreams_own_function():
    """The exactness claim, checked rather than asserted in prose.

    A torchaudio mel with matching numeric parameters would still differ: HTK vs Slaney
    filters, power vs magnitude, ``center=True`` padding, ``log(x + 1e-6)`` vs
    ``log(clamp(x, 1e-5))``.  Bit equality is the only useful tolerance here.
    """
    codec = build_extractor(_mel_config())
    add_repo_to_path(
        _mel_config()["extractor"]["repo_path"],
        name="BigVGAN", expect=["meldataset.py"],
    )
    from meldataset import get_mel_spectrogram

    batch = _wav(B=2, T=256 * 60, sr=22050)
    reference = get_mel_spectrogram(batch.BCT.squeeze(1), codec._h)
    assert torch.equal(codec.encode(batch).data, reference)


@needs_weights
def test_wavlm_codec_matches_the_checkpoints_own_hyperparameters():
    codec = build_extractor(_wavlm_config())
    assert codec.sample_rate == 16000
    assert codec.latent_dim == 1024          # WavLM-Large, not Base
    assert codec.frame_grid.hop_size == 320
    assert codec.frame_grid.causal is False


@needs_weights
def test_wavlm_encode_reproduces_knnvc_on_an_unpadded_batch():
    """Our only deviation from kNN-VC's call is the frame-alignment pad and an explicit
    padding mask.  For a batch with nothing to mask, the result must be identical to
    upstream's ``padding_mask=None``, or the decoder is being fed something it was not
    trained on."""
    import torch.nn.functional as F

    codec = build_extractor(_wavlm_config())
    batch = _wav(B=1, T=320 * 60, sr=16000)

    padded = F.pad(batch.BCT, (_CONV_RECEPTIVE_FIELD // 2, 0), mode="reflect").squeeze(1)
    with torch.no_grad():
        reference, _ = codec.wavlm.extract_features(
            padded, padding_mask=None, output_layer=6, ret_layer_results=False,
        )
    assert torch.equal(codec.encode(batch).data, reference.transpose(1, 2))


@needs_weights
@pytest.mark.parametrize("config_fn", [_mel_config, _wavlm_config])
@pytest.mark.parametrize("chunk_s", [2.0, None])
def test_frame_labels_land_on_the_codecs_own_grid(config_fn, chunk_s, tmp_path):
    """``IntensityDataset``'s label count must equal the codec's frame count.

    ``ConverterCGANv2LabelsModule._check_alignment`` raises on a mismatch, so this is
    loud on the cluster -- but it fires an hour into a queued job, and the mel domain is
    the first codec whose rate is not 16 kHz and whose ``window_length`` is therefore not
    400.  Checked at both the chunked training length and a full sequence, because
    ``pad_to_multiple`` rounds the chunk *up* (2.0 s at 22050 Hz is 44100 samples, padded
    to 44288 = 173 * 256) and the labels are measured on the padded waveform.
    """
    import numpy as np
    import pandas as pd
    import soundfile as sf

    from vic.data.collate import make_intensity_collate_fn
    from vic.data.dataset import IntensityDataset

    codec = build_extractor(config_fn())
    window_length = round(0.025 * codec.sample_rate)      # 25 ms, as the configs set it

    src_sr = 48000
    rows = []
    for i in range(2):
        sig, cal = tmp_path / f"s{i}.wav", tmp_path / f"c{i}.wav"
        sf.write(sig, (np.random.randn(src_sr * 5) * 0.05).astype("float32"), src_sr)
        sf.write(cal, (np.random.randn(src_sr) * 0.02).astype("float32"), src_sr)
        rows.append({"signal_path": str(sig), "calibration_path": str(cal),
                     "calibration_rms": 0.02, "distance_m": 0.3,
                     "corpus": "AVID", "split": "train"})

    dataset = IntensityDataset(
        pd.DataFrame(rows), target_sr=codec.sample_rate, frame_grid=codec.frame_grid,
        window_length=window_length, chunk_s=chunk_s, train=chunk_s is not None,
        normalize_sequence=False,
    )
    collate = make_intensity_collate_fn(codec.sample_rate, codec.frame_grid.hop_size)
    batch = collate([dataset[i] for i in range(2)])

    n_labels = batch["frame_intensity_db"].data.shape[-1]
    n_frames = codec.encode(batch["wav"]).padding_mask.shape[-1]
    assert n_labels == n_frames


@needs_weights
@pytest.mark.parametrize("config_fn", [_mel_config, _wavlm_config])
def test_round_trip_preserves_the_sample_count(config_fn):
    """``decode(encode(wav))`` must return the same number of samples it was given, or
    every rendered file is silently short and the evaluation joins mismatched audio."""
    codec = build_extractor(config_fn())
    hop = codec.frame_grid.hop_size
    batch = _wav(B=2, T=hop * 50, sr=codec.sample_rate, lengths=[hop * 50, hop * 31])

    out = codec.decode(codec.encode(batch))
    assert out.sample_rate == codec.sample_rate
    assert out.data.shape[-1] == batch.data.shape[-1]
    assert out.lengths.tolist() == [hop * 50, hop * 31]

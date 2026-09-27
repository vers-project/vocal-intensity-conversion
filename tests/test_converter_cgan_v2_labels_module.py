"""Wiring test for the scalar-τ trainer anchored on calibrated labels.

The real codec, predictor and corpus live on the cluster, so the pipelines here are the
shared stubs of ``stubs.py``.

What this exercises is the one thing the fork changes: τ_src stops being P_φ's reading and
becomes ``leq_aggregate`` of the clamped frame labels, in the four places that anchor on it
(vicinal positives, mismatched negatives, the cycle return trip, the identity anchor).  The
architecture here is the Transformer C_θ / D_ψ pair the fork was written for, not the conv
stack the local tests use.
"""
import lightning as L
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from vic.data.collate import make_intensity_collate_fn, make_unlabeled_collate_fn
from vic.models.conditioning import build_conditional_discriminator
from vic.models.converter import build_converter
from vic.training.callbacks import (
    IntensityConversionCallback,
    IntensityEvaluationCallback,
)
from vic.training.converter_cgan_v2_labels_module import (
    ConverterCGANv2LabelsModule,
    _masked_mean_t,
)
from vic.training.utils import IntensityDrawer, LabelScaler, VicinalSampler

from stubs import (  # noqa: E402  — shared test harness, see docstring
    D,
    EMBED,
    HOP,
    RANGE,
    SR,
    _LabelledDataset,
    _StubExtractor,
    _StubPipeline,
    _StubPredictor,
    _WavDataset,
)


class _SilentDataset(_WavDataset):
    """Every frame at FrameLevelTransform's -200 dB floor: a chunk that is all pause.

    The scalar path aggregates before anything else sees the labels, so a *scattering* of
    -200 dB frames is harmless — mean-square-pressure averaging drops them.  A whole
    silent chunk is the case that bites, and about 3% of random 2-s AVID training chunks
    are one.
    """

    def __getitem__(self, i):
        item = super().__getitem__(i)
        item["frame_intensity_db"] = torch.full((1, self.lengths[i] // HOP), -200.0)
        return item


def _build(floor_db=33.0, label_source="predictor", predictor=None,
           lambda_frame=0.0, silence_db=None):
    """The Transformer pair from jeanzay_converter_cgan_v2_labels_avid_segments.yaml, small."""
    conv_cfg = {"type": "contextual", "d_model": 16, "n_heads": 2, "n_layers": 2,
                "head_dim": 8, "mlp_ratio": 2, "dropout": 0.0, "intensity_embed": EMBED}
    disc_cfg = {"d_model": 16, "n_heads": 2, "n_layers": 2, "head_dim": 8,
                "mlp_ratio": 2, "dropout": 0.0, "attn_type": "l2"}
    cfg = {"model": {"converter": conv_cfg, "discriminator": disc_cfg,
                     "conditioning": {"type": "projection", "intensity_embed": EMBED}}}
    scaler = LabelScaler(mean=60.0, std=10.0)
    extractor = _StubExtractor()
    return ConverterCGANv2LabelsModule(
        pipeline=_StubPipeline(extractor, whitening=False),
        label_pipeline=_StubPipeline(extractor, whitening=True),
        converter=build_converter(cfg, latent_dim=D, label_scaler=scaler),
        predictor=predictor if predictor is not None else _StubPredictor(),
        cond_disc=build_conditional_discriminator(cfg, latent_dim=D, label_scaler=scaler),
        label_scaler=scaler,
        intensity_drawer=IntensityDrawer(*RANGE),
        vicinal_sampler=VicinalSampler(*RANGE, vicinity_db=2.5, min_offset_db=6.0),
        lambda_id=1.0,                                   # exercise the identity anchor too
        spectral_norm=False,
        floor_db=floor_db,
        label_source=label_source,
        lambda_frame=lambda_frame,
        silence_db=silence_db,
    )


def _loader():
    return DataLoader(_WavDataset(), batch_size=4,
                      collate_fn=make_unlabeled_collate_fn(SR))


def _labelled_loader(dataset=None, batch_size=4, **kw):
    ds = dataset if dataset is not None else _LabelledDataset(**kw)
    return DataLoader(ds, batch_size=batch_size,
                      collate_fn=make_intensity_collate_fn(SR, HOP))


def test_the_predictor_source_is_exactly_the_predictor_reading():
    """``source: predictor`` must stay the parent module's behaviour, bit for bit."""
    module = _build(label_source="predictor").eval()
    batch = next(iter(_labelled_loader()))
    assert torch.equal(module.source_tau(batch), module.estimate_tau_src(batch["wav"]))


def test_the_label_source_never_calls_the_predictor():
    class _Exploding(nn.Module):
        def forward(self, z):
            raise AssertionError("P_φ was called on the conditioning path")

    module = _build(label_source="labels", predictor=_Exploding()).eval()
    tau = module.source_tau(next(iter(_labelled_loader())))     # must not raise
    # The stub labels every frame at 65 dB, so their Leq is 65 dB.
    assert torch.allclose(tau, torch.full((4,), 65.0), atol=1e-3)


def test_an_all_silent_chunk_is_held_at_the_floor():
    """Without the clamp this reads -200 dB — about -26σ into the vicinal sampler."""
    module = _build(label_source="labels", floor_db=33.0).eval()
    tau = module.source_tau(next(iter(_labelled_loader(dataset=_SilentDataset()))))
    assert torch.allclose(tau, torch.full((4,), 33.0), atol=1e-3)


def test_padding_is_excluded_from_the_source_level():
    """The batch is ragged, so the shortest item's Leq must not include padded frames.

    Padding arrives as zeros in the label tensor — 0 dB SPL, i.e. quiet but nowhere near
    the floor — so an unmasked aggregate would drag every short item down.
    """
    module = _build(label_source="labels").eval()
    batch = next(iter(_labelled_loader()))
    assert batch["frame_intensity_db"].lengths.min() < batch["frame_intensity_db"].lengths.max()
    assert torch.allclose(module.source_tau(batch), torch.full((4,), 65.0), atol=1e-3)


def test_a_floorless_label_run_is_refused_at_construction():
    with pytest.raises(ValueError, match="floor_db"):
        _build(label_source="labels", floor_db=None)


def test_an_unknown_label_source_is_refused():
    with pytest.raises(ValueError, match="label_source"):
        _build(label_source="phi")


def test_misaligned_labels_are_refused_rather_than_broadcast():
    """τ_src would otherwise be measured over a different span than the audio converted."""
    module = _build(label_source="labels")
    with pytest.raises(RuntimeError, match="disagree on frame count"):
        L.Trainer(fast_dev_run=True, accelerator="cpu", logger=False,
                  enable_checkpointing=False, enable_progress_bar=False,
                  enable_model_summary=False).fit(
            module, _labelled_loader(frames_offset=1))


def test_the_ruler_bias_metric_appears_only_on_the_label_path():
    """P_φ's error against the truth is only defined when the truth is what we condition on.

    Under `predictor` P_φ *is* τ_src, so the difference is identically zero and logging it
    would invite reading a constant as a result.
    """
    def _logged(label_source, loader):
        module = _build(label_source=label_source)
        keys = {}
        module.log_dict = lambda d, **kw: keys.update(d)
        module.log = lambda k, v, **kw: keys.update({k: v})
        L.Trainer(fast_dev_run=True, accelerator="cpu", logger=False,
                  enable_checkpointing=False, enable_progress_bar=False,
                  enable_model_summary=False).fit(module, loader, loader)
        return keys

    assert "val/ruler_bias_db" in _logged("labels", _labelled_loader())
    assert "val/ruler_bias_db" not in _logged("predictor", _labelled_loader())


def test_the_conversion_callbacks_run_on_both_sources(tmp_path):
    """Outside training, C_θ is reached only through the callbacks' ConvertibleModule contract.

    They read ``source_tau(batch)`` for the Δτ axis and ``convert(z_real, τ, batch)`` for
    the audio, so the label path has to survive a batch that carries frame labels and a
    device move that was written for ``batch["wav"]`` alone.  This is also the first thing
    that would fail on the cluster, an epoch into a two-hour job.

    Both callback loaders are batch_size=1, as the training script builds them:
    IntensityConversionCallback names its files from ``source_tau(batch)[0]`` and converts
    at a ``(1,)`` τ, so one utterance per batch is its contract, not an incidental choice.
    """
    for label_source in ("labels", "predictor"):
        torch.manual_seed(0)
        module = _build(label_source=label_source)
        out = tmp_path / label_source
        L.Trainer(
            fast_dev_run=True, accelerator="cpu", logger=False,
            enable_checkpointing=False, enable_progress_bar=False,
            enable_model_summary=False,
            callbacks=[
                IntensityEvaluationCallback(
                    loader=_labelled_loader(batch_size=1), n_targets=3, n_bins=3,
                    output_dir=out,
                ),
                IntensityConversionCallback(
                    loader=_labelled_loader(batch_size=1), sample_rate=SR, n_targets=2,
                    every_n_epochs=1, output_dir=out,
                ),
            ],
        ).fit(module, _labelled_loader(), _labelled_loader())
        assert list(out.rglob("*.wav")), f"{label_source}: no audio written"


def test_fast_dev_run_trains_and_validates_on_labels():
    torch.manual_seed(0)
    module = _build(label_source="labels")
    L.Trainer(fast_dev_run=True, accelerator="cpu", logger=False,
              enable_checkpointing=False, enable_progress_bar=False,
              enable_model_summary=False).fit(
        module, _labelled_loader(), _labelled_loader())


# --------------------------------------------------------------- per-frame critic


def test_the_pooled_score_is_the_time_average_of_the_frame_scores():
    """The identity the one-forward-pass trick rests on.

    ``score`` is Linear and the projection is an inner product, so both are affine in φ and
    commute with the mean over frames.  If this ever stops holding, the pooled term and the
    frame term are silently optimising two different quantities.
    """
    module = _build(lambda_frame=1.0).eval()
    batch = next(iter(_labelled_loader()))
    z = module.pipeline.encode(batch["wav"])
    tau = torch.full((4,), 65.0)

    frames, pooled = module._critic(z, tau)
    assert frames.shape == z.padding_mask.shape
    # The direct pooled call, i.e. what lambda_frame == 0 computes.
    assert torch.allclose(pooled, module.cond_disc(z, tau), atol=1e-5)


def test_padded_frames_do_not_enter_the_pooled_score():
    """The batch is ragged; a padded frame's critic output is meaningless."""
    module = _build(lambda_frame=1.0).eval()
    batch = next(iter(_labelled_loader()))
    z = module.pipeline.encode(batch["wav"])
    frames, pooled = module._critic(z, torch.full((4,), 65.0))

    corrupted = frames.masked_fill(~z.padding_mask, 1e4)
    assert torch.allclose(
        _masked_mean_t(corrupted, z.padding_mask), pooled, atol=1e-5
    ), "padding leaked into the pooled score"


def test_the_blend_is_normalised_and_reduces_to_the_pooled_loss():
    module = _build(lambda_frame=0.0)
    pooled, frame = torch.tensor(2.0), torch.tensor(6.0)
    assert module._blend(pooled, None) is pooled                       # λ = 0 path
    module = _build(lambda_frame=1.0)
    assert module._blend(pooled, frame) == pytest.approx(4.0)          # (2 + 1·6) / 2
    module = _build(lambda_frame=3.0)
    assert module._blend(pooled, frame) == pytest.approx(5.0)          # (2 + 3·6) / 4


def test_silence_is_dropped_from_the_mismatch_term_at_both_scales():
    """A pause carries no information about vocal effort, so it is not a valid negative.

    Checked through the logged fractions rather than by reaching into the loss: they are
    what the run is read by, so they are what has to be right.
    """
    def _fracs(dataset, silence_db):
        module = _build(label_source="labels", floor_db=12.5, lambda_frame=1.0,
                        silence_db=silence_db)
        logged = {}
        module.log_dict = lambda d, **kw: logged.update(d)
        module.log = lambda k, v, **kw: logged.update({k: v})
        L.Trainer(fast_dev_run=True, accelerator="cpu", logger=False,
                  enable_checkpointing=False, enable_progress_bar=False,
                  enable_model_summary=False).fit(module, _labelled_loader(dataset=dataset))
        return logged["train/mism_frame_frac"].item(), logged["train/mism_seq_frac"].item()

    # Speech everywhere (the stub labels every frame at 65 dB): nothing is dropped.
    assert _fracs(_LabelledDataset(), 32.5) == pytest.approx((1.0, 1.0))
    # Silence everywhere: every frame and every chunk leaves the mismatch term.
    assert _fracs(_SilentDataset(), 32.5) == pytest.approx((0.0, 0.0))
    # Half of each chunk silent: the frames go, the chunks stay (their Leq is speech).
    frame_frac, seq_frac = _fracs(_LabelledDataset(silent_db=-200.0), 32.5)
    assert 0.4 < frame_frac < 0.6 and seq_frac == pytest.approx(1.0)
    # Threshold off: the previous behaviour, everything counts.
    assert _fracs(_SilentDataset(), None) == pytest.approx((1.0, 1.0))


def test_silence_masking_without_labels_is_refused():
    with pytest.raises(ValueError, match="silence_db"):
        _build(label_source="predictor", lambda_frame=1.0, silence_db=32.5)


def test_a_non_projection_conditioner_is_refused_with_a_frame_critic():
    """The pooled score is only the frames' mean when the conditional term is affine in φ."""
    module = _build(lambda_frame=0.0)
    module.cond_disc.conditioner = nn.Identity()          # stand-in for a concat MLP
    with pytest.raises(ValueError, match="projection conditioner"):
        ConverterCGANv2LabelsModule(
            pipeline=module.pipeline, label_pipeline=module.label_pipeline,
            converter=module.converter, predictor=module.predictor,
            cond_disc=module.cond_disc, label_scaler=module.label_scaler,
            intensity_drawer=module.intensity_drawer,
            vicinal_sampler=module.vicinal_sampler,
            spectral_norm=False, lambda_frame=1.0,
        )


def test_fast_dev_run_trains_with_the_frame_critic():
    torch.manual_seed(0)
    module = _build(label_source="labels", floor_db=12.5, lambda_frame=1.0, silence_db=32.5)
    L.Trainer(fast_dev_run=True, accelerator="cpu", logger=False,
              enable_checkpointing=False, enable_progress_bar=False,
              enable_model_summary=False).fit(
        module, _labelled_loader(), _labelled_loader())


def test_a_windowed_converter_is_length_invariant():
    """The property `attn_window` on C_θ exists for: test-time sequences are longer.

    C_θ is the only network that runs at inference, on 2-5 s utterances against 2-s
    training chunks. With a bounded field, content beyond that field cannot change a
    frame's output — so the same audio converts identically whether it arrives alone or
    with 3 more seconds appended. Unbounded attention has no such guarantee, and RoPE
    meets relative distances it never saw in training.

    Tests the wiring as much as the property: config key → build_converter →
    TransformerEncoder → the mask actually being applied.
    """
    from vic.data.audio_batch import AudioBatch
    from vic.models.converter import build_converter

    embed = {"type": "sinusoidal", "embed_dim": 16, "d_sin": 16}
    cfg = {"model": {"converter": {
        "type": "contextual", "d_model": 16, "n_heads": 2, "n_layers": 3,
        "head_dim": 8, "mlp_ratio": 2, "dropout": 0.0,
        "intensity_embed": embed, "attn_window": 2,
    }}}
    torch.manual_seed(0)
    conv = build_converter(cfg, latent_dim=D, label_scaler=LabelScaler(60.0, 10.0)).eval()
    assert conv.receptive_field == 1 + 2 * 3 * 2                # 13 frames

    torch.manual_seed(1)
    short = torch.randn(1, D, 40)
    longer = torch.cat([short, torch.randn(1, D, 60)], dim=-1)
    tau = torch.tensor([60.0])
    with torch.no_grad():
        a = conv(AudioBatch(data=short, lengths=torch.tensor([40]), sample_rate=50), tau)
        b = conv(AudioBatch(data=longer, lengths=torch.tensor([100]), sample_rate=50), tau)

    clean = 40 - conv.receptive_field // 2      # frames the appended content cannot reach
    assert torch.allclose(a.data[..., :clean], b.data[..., :clean], atol=1e-5)


def test_fast_dev_run_trains_and_validates_on_the_predictor_source():
    """The unlabelled path still runs, on batches carrying no labels at all."""
    torch.manual_seed(0)
    module = _build(label_source="predictor")
    L.Trainer(fast_dev_run=True, accelerator="cpu", logger=False,
              enable_checkpointing=False, enable_progress_bar=False,
              enable_model_summary=False).fit(module, _loader(), _loader())

"""Config → objects for the scalar-τ / calibrated-label converter stack.

Everything ``scripts/train_converter_cgan_v2_labels.py`` used to do between reading the
YAML and handing a module to the Trainer now lives here, so that a script which only
*runs* a trained converter builds it along exactly the same path rather than along a
second one that has to be kept in step by hand.  That mattered enough to hoist: the two
pipelines share one frozen codec, the sharing is conditional on the two ``extractor``
blocks naming the same front end, and getting that condition wrong produces a run that
reports plausible nonsense with no exception anywhere (see
:func:`build_conversion_pipelines`).

The split into three functions is not cosmetic — it preserves the *order* in which the
training script constructs things.  The pipelines have to exist before the datasets (they
carry the sample rate and hop size the datasets are built on), and the models have to be
constructed after them, because model init draws from the global RNG that
``L.seed_everything`` set.  Collapsing this into one call would reorder those draws and
change the weights a given seed produces, which would silently break reproducibility of
the runs already reported.

Usage
-----
Training (``ckpt_path=None`` — fresh weights, scaler fitted from the config)::

    source, floor_db = resolve_tau_src(config)
    pipeline, label_pipeline = build_conversion_pipelines(config)
    ...                                     # datasets, using pipeline.sample_rate
    module = build_labels_module(config, pipeline, label_pipeline)

Inference (``ckpt_path`` given — C_θ's weights and the *fitted* label scaler both come
from the checkpoint)::

    module = build_labels_module(config, pipeline, label_pipeline, ckpt_path=ckpt)
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from vic.checkpoints import load_submodule
from vic.data.collate import make_intensity_collate_fn, make_unlabeled_collate_fn
from vic.data.dataset import IntensityDataset, UnlabeledAudioDataset
from vic.models.conditioning import build_conditional_discriminator
from vic.models.converter import build_converter
from vic.models.predictor import build_predictor
from vic.training.converter_cgan_v2_labels_module import ConverterCGANv2LabelsModule
from vic.training.extraction_pipeline import (
    ExtractionPipeline,
    build_extractor,
    build_pipeline,
)
from vic.training.utils import (
    IntensityDrawer,
    VicinalSampler,
    build_label_scaler,
    describe_model,
    load_label_scaler,
)

__all__ = [
    "resolve_tau_src",
    "build_conversion_pipelines",
    "build_labels_module",
    "build_inference_dataset",
]


def build_inference_dataset(
    metadata: pd.DataFrame,
    config: dict,
    source: str,
    pipeline: ExtractionPipeline,
):
    """A full-sequence dataset and its collate, for rendering rather than training.

    ``chunk_s=None`` and ``train=False``: whole utterances, taken from the start of their
    region, so a render is deterministic and nothing is cropped.

    The dataset class follows ``tau_src.source`` exactly as the training script chooses
    it, because ``labels`` needs ``frame_intensity_db`` in the batch for
    ``ConverterCGANv2LabelsModule.source_tau`` to read.  Shared by every rendering script
    so that the source level a render reports is measured the same way training measured
    it, down to the SPL analysis window.
    """
    sr = pipeline.sample_rate
    normalize = config["extractor"].get("normalize_sequence", False)
    if source == "labels":
        dataset = IntensityDataset(
            metadata,
            target_sr=sr, frame_grid=pipeline.frame_grid,
            window_length=config["data"]["window_length"],
            chunk_s=None, train=False,
            normalize_sequence=normalize,
        )
        return dataset, make_intensity_collate_fn(sr, pipeline.frame_grid.hop_size)

    dataset = UnlabeledAudioDataset(
        metadata,
        target_sr=sr, frame_grid=pipeline.frame_grid,
        chunk_s=None, train=False,
        normalize_sequence=normalize,
    )
    return dataset, make_unlabeled_collate_fn(sr)


def resolve_tau_src(config: dict) -> tuple[str, float | None]:
    """Read and validate ``training.tau_src``, returning ``(source, floor_db)``.

    Required, not optional.  A run anchored on the calibrated labels and a run anchored
    on P_φ are indistinguishable from their logs — same metrics, same keys, same
    magnitudes — so the config has to say which it is rather than inherit a default.

    Separate from :func:`build_labels_module` because ``source`` also decides which
    dataset class the caller builds, which happens before any model exists.
    """
    ts = config["training"].get("tau_src")
    if ts is None:
        raise ValueError(
            "training.tau_src is missing. Set source: labels to anchor the conditioning "
            "on the corpus's calibrated frame labels, or source: predictor to anchor it "
            "on P_phi's reading. It is not defaulted on purpose: the two "
            "runs look identical in their logs."
        )
    source = ts.get("source")
    if source not in ("labels", "predictor"):
        raise ValueError(
            f"training.tau_src.source must be 'labels' or 'predictor', got {source!r}."
        )
    if source == "labels" and ts.get("floor_db") is None:
        raise ValueError(
            "training.tau_src.floor_db is missing. The calibrated frame labels reach "
            "-200 dB on silent frames, so an all-silent chunk would hand -200 dB to the "
            "vicinal sampler as its source level. Set it from the corpus's own "
            "frame-level SPL distribution."
        )
    return source, ts.get("floor_db")


def build_conversion_pipelines(
    config: dict,
) -> tuple[ExtractionPipeline, ExtractionPipeline]:
    """Two pipelines, one codec: ``(pipeline, label_pipeline)``.

      pipeline       — conversion space.  No whitening: the converter, the
                       discriminator and the decoder all operate on latents the
                       NAC was actually trained to produce and consume.
      label_pipeline — measurement space.  Whitening on, because that is what
                       P_φ was trained on.  Only P_φ ever reads it.

    ``label_extractor`` is merged *over* ``extractor`` so the config states only the
    difference (the whitening flags) and cannot drift on the codec paths.  The
    already-built codec is passed in, so it is loaded once and lives on the GPU once, no
    matter how many pipelines wrap it.

    The codec is shared only when the measurement space is that same model differently
    pre-processed — the case the sharing was written for.  A different ``type`` means a
    different front end (a Wav2Vec2 P_φ, say), and ``build_pipeline``'s ``extractor``
    argument overrides ``type`` silently: the pipeline would keep the codec, P_φ would be
    handed NAC latents it was never trained on, and the run would report plausible
    nonsense with no exception anywhere — ``build_predictor`` takes ``latent_dim`` from
    this same pipeline, so even the shapes would agree.
    """
    codec = build_extractor(config)
    pipeline = build_pipeline(config, extractor=codec)

    label_cfg = {**config["extractor"], **config.get("label_extractor", {})}
    share = label_cfg.get("type") == config["extractor"].get("type")
    label_pipeline = build_pipeline(
        {**config, "extractor": label_cfg}, extractor=codec if share else None
    )
    return pipeline, label_pipeline


def build_labels_module(
    config: dict,
    pipeline: ExtractionPipeline,
    label_pipeline: ExtractionPipeline,
    ckpt_path: str | Path | None = None,
) -> ConverterCGANv2LabelsModule:
    """Assemble :class:`ConverterCGANv2LabelsModule` from a converter training config.

    Parameters
    ----------
    config         : the converter training YAML, already parsed.
    pipeline,
    label_pipeline : from :func:`build_conversion_pipelines`.  Passed in rather than
                     built here so the caller can size its datasets from
                     ``pipeline.sample_rate`` / ``pipeline.frame_grid`` first — see the
                     module docstring on construction order.
    ckpt_path      : a converter training checkpoint.  When given, C_θ's weights are
                     loaded from it and the label scaler is recovered from it rather
                     than refitted from the config, because the scaler is a *fitted*
                     quantity the architecture does not carry and a converter built with
                     the wrong one normalises τ wrongly while raising nothing.  D_ψ is
                     deliberately left at its init: nothing outside the training loop
                     reads the critic, and its spectral-norm reparametrisation is one
                     more thing that can refuse to load for reasons that would not
                     matter.

    P_φ is loaded whether or not ``ckpt_path`` is given — its weights come from
    ``predictor.ckpt_path``, never from the converter checkpoint, and it is the
    instrument that reads the *achieved* intensity of decoded audio.
    """
    source, floor_db = resolve_tau_src(config)

    # ------------------------------------------------------------------
    # LabelScaler: from a labelled corpus's training split, or approximated
    # from the drawer range.  Saved into every checkpoint by the module, so
    # inference does not have to trust this config not to have moved.
    # ------------------------------------------------------------------
    dc_range = config["training"]["intensity_range_db"]
    label_scaler = (
        build_label_scaler(config)
        if ckpt_path is None
        else load_label_scaler(ckpt_path, config)
    )

    # ------------------------------------------------------------------
    # Frozen P_φ.  It reads label_pipeline's output (whitened), never the conversion
    # latents, and contributes no gradient.  Built unconditionally: even under
    # `source: labels`, validation measures the *achieved* intensity of decoded audio
    # with it, and that reading has no ground truth to replace it.
    # ------------------------------------------------------------------
    pc = config["predictor"]
    predictor = build_predictor(
        pc["model"], latent_dim=label_pipeline.latent_dim, dropout_override=0.0
    )
    load_submodule(
        predictor, pc["ckpt_path"],
        prefix=pc.get("ckpt_prefix", "predictor"),
    )
    predictor.eval().requires_grad_(False)

    # ------------------------------------------------------------------
    # Optional independent test predictor (P_φ_test).  Applied to the decoded
    # waveform, so its own whitening flag now whitens clean audio exactly once.
    # ------------------------------------------------------------------
    test_pipeline_obj = None
    test_predictor_obj = None
    if "test_predictor" in config:
        tpc = config["test_predictor"]
        test_pipeline_obj = build_pipeline(tpc)
        test_predictor_obj = build_predictor(
            tpc["model"], latent_dim=test_pipeline_obj.latent_dim, dropout_override=0.0
        )
        load_submodule(
            test_predictor_obj, tpc["ckpt_path"],
            prefix=tpc.get("ckpt_prefix", "predictor"),
        )
        test_predictor_obj.eval().requires_grad_(False)

    # ------------------------------------------------------------------
    # Converter and conditional discriminator.
    # ------------------------------------------------------------------
    converter = build_converter(
        config, latent_dim=pipeline.latent_dim, label_scaler=label_scaler
    )
    cond_disc = build_conditional_discriminator(
        config, latent_dim=pipeline.latent_dim, label_scaler=label_scaler
    )

    frame_rate_hz = pipeline.sample_rate / pipeline.frame_grid.hop_size
    describe_model("C_θ converter", converter, frame_rate_hz)
    describe_model("D_ψ discriminator", cond_disc, frame_rate_hz)

    tc = config["training"]
    hp = tc.get("loss_weights", {})
    fc = tc.get("frame_critic") or {}
    drawer = IntensityDrawer(min_db=dc_range[0], max_db=dc_range[1])

    if "pred" in hp:
        raise ValueError(
            "training.loss_weights.pred is set, but this module has no direct "
            "predictor term: P_φ cannot see the converted latent at all, since "
            "the only route to it runs through the non-differentiable decoder. "
            "Remove the key (see the module docstring)."
        )

    cond_cfg = config["model"].get("conditioning", {})
    vicinal = VicinalSampler(
        min_db=dc_range[0],
        max_db=dc_range[1],
        vicinity_db=cond_cfg.get("vicinity_db", 1.0),
        min_offset_db=cond_cfg.get("min_offset_db", 6.0),
    )

    module = ConverterCGANv2LabelsModule(
        pipeline=pipeline,
        label_pipeline=label_pipeline,
        converter=converter,
        predictor=predictor,
        cond_disc=cond_disc,
        label_scaler=label_scaler,
        intensity_drawer=drawer,
        vicinal_sampler=vicinal,
        test_pipeline=test_pipeline_obj,
        test_predictor=test_predictor_obj,
        lr_g=tc.get("lr_g", 1e-4),
        lr_d=tc.get("lr_d", 1e-4),
        lambda_fake=hp.get("fake", 1.0),
        lambda_cycle=hp.get("cycle", 1.0),
        lambda_id=hp.get("identity", 0.0),
        lambda_mismatch=hp.get("mismatch", 1.0),
        spectral_norm=tc.get("spectral_norm", True),
        gradient_clip_val=tc.get("gradient_clip_val", 1.0),
        floor_db=floor_db,
        label_source=source,
        # Per-frame ("patch") critic. Absent or weight 0 → the pooled objective and code
        # path of the pooled critic, unchanged.
        lambda_frame=fc.get("weight", 0.0),
        silence_db=fc.get("silence_db"),
    )

    if ckpt_path is not None:
        load_submodule(module.converter, ckpt_path, prefix="converter")

    return module


def build_group_evaluation(
    metadata: pd.DataFrame,
    config: dict,
    source: str,
    pipeline: ExtractionPipeline,
    *,
    levels: dict | None = None,
    vad: dict | None = None,
    n_groups: int | None = None,
    stratify_by: str = "speaker_uid",
    num_workers: int = 4,
    seed: int = 42,
):
    """Everything the paired evaluation needs, from a metadata split.

    The level-group design needs real recordings at *both* ends of every
    conversion, so a split can only be evaluated this way once it carries the
    sentence and effort annotation.  This assembles it: group the rows into
    sentence groups, trim head and tail silence, work out which levels each row
    can be converted to, and build a full-sequence dataset.

    Used by the training-time
    :class:`~vic.training.callbacks.ConversionQualityCallback`.

    **Not yet used by ``scripts/evaluate_conversion.py``**, which still calls
    ``prepare_level_groups`` and ``trim_metadata_with_vad`` itself.  So the monitor
    and the final table currently select their material by two separate code
    paths, and the round-robin group selection below exists only on this one.
    That is a drift risk against the whole point of the monitor -- it is supposed
    to report the same quantity the paper reports -- and closing it means giving
    :func:`vic.evaluation.monitor.measure_conversions` the two hooks the script
    needs for resume (a ``skip`` set and an ``on_checkpoint`` callable).

    ``n_groups`` keeps whole sentence groups rather than sampling rows: dropping
    rows would leave partial groups, and a group missing a level has destinations
    that name no real recording.  The groups are taken **round-robin over
    ``stratify_by``**, which is load-bearing -- see :func:`_take_groups`.

    Returns
    -------
    ``(loader_factory, targets_by_row, stems, group_ids, records)`` — the
    arguments of :func:`vic.evaluation.monitor.measure_conversions`.
    ``loader_factory`` is a callable because each of its two passes restarts from
    the beginning.
    """
    from torch.utils.data import DataLoader

    from vic.data.levels import (
        DEFAULT_EXCLUDE_COLUMN,
        DEFAULT_GROUP_COLUMNS,
        DEFAULT_LEVEL_COLUMN,
        DEFAULT_SENTENCE_COLUMN,
        level_targets,
        prepare_level_groups,
    )
    from vic.data.vad import trim_metadata_with_vad
    from vic.evaluation.conditions import source_records, stems

    lc = dict(levels or {})
    level_column = lc.get("level_column", DEFAULT_LEVEL_COLUMN)
    rows = prepare_level_groups(
        metadata.reset_index(drop=True),
        level_column=level_column,
        group_columns=lc.get("group_columns", list(DEFAULT_GROUP_COLUMNS)),
        exclude_column=lc.get("exclude_column", DEFAULT_EXCLUDE_COLUMN),
        sentence_column=lc.get("sentence_column", DEFAULT_SENTENCE_COLUMN),
        sentence_ids=lc.get("sentence_ids"),
    )

    if n_groups is not None:
        rows = _take_groups(rows, int(n_groups), stratify_by)

    vad_cfg = dict(vad or {})
    if vad_cfg.pop("enabled", True):
        rows = trim_metadata_with_vad(rows, vad_cfg)
        kept = rows[rows["vad_status"] == "ok"].reset_index(drop=True)
        if kept.empty:
            raise ValueError(
                "The VAD found no speech in any row of the evaluation split."
            )
        rows = kept

    # After the trim, not before: a group loses a destination if the VAD dropped
    # one of its members, and a target naming a discarded row has no reference.
    targets_by_row = level_targets(rows, level_column=level_column)
    dataset, collate_fn = build_inference_dataset(rows, config, source, pipeline)

    def loader_factory():
        return DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=num_workers,
            pin_memory=True, collate_fn=collate_fn,
        )

    return (
        loader_factory,
        targets_by_row,
        stems(rows),
        rows["sentence_group_id"].tolist(),
        source_records(rows, level_column),
    )


def _take_groups(rows, n_groups: int, stratify_by: str):
    """Pick ``n_groups`` whole sentence groups, spread over speakers.

    Taking the first N sorted group ids is deterministic and was the first
    implementation, but a ``sentence_group_id`` *begins with the speaker*, so
    sorting and truncating collapses the whole selection onto one speaker.  On the
    AVID validation split the first 24 of 200 sorted groups were all one speaker,
    which has two consequences, the quieter one being the worse:

    * the per-speaker slopes are fitted within a single group, so there is no
      slope distribution and the bootstrap interval over speakers is meaningless;
    * the speaker metric gets no different-speaker pairs at all and
      ``calibrate`` raises ``0 nontarget``.

    Round-robin over ``stratify_by`` keeps the selection deterministic and
    independent of row order while covering every speaker: 24 groups over 4
    speakers is 6 each.  Falls back to the sorted prefix, with a warning, when the
    column is absent.
    """
    import warnings

    ids = sorted(rows["sentence_group_id"].unique())
    if stratify_by not in rows.columns:
        warnings.warn(
            f"{stratify_by!r} is not a column, so the {n_groups} evaluation "
            "groups are taken as a sorted prefix and may all come from one "
            "speaker; per-speaker statistics would then be undefined.",
            RuntimeWarning,
        )
        keep = ids[:n_groups]
    else:
        per_speaker = {
            speaker: sorted(part["sentence_group_id"].unique())
            for speaker, part in rows.groupby(stratify_by, sort=True)
        }
        keep, depth = [], 0
        while len(keep) < n_groups and any(
            len(g) > depth for g in per_speaker.values()
        ):
            for speaker in per_speaker:
                if len(keep) >= n_groups:
                    break
                if len(per_speaker[speaker]) > depth:
                    keep.append(per_speaker[speaker][depth])
            depth += 1
    return rows[rows["sentence_group_id"].isin(keep)].reset_index(drop=True)

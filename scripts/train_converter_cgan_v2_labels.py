"""Train the intensity converter C_θ with a τ-conditional critic and a **calibrated** τ_src.

The condition is a **scalar** τ.  P_φ is a frozen labeller, and the conditional critic,
cycle consistency and the identity anchor are described in
``vic/training/converter_cgan_v2_labels_module.py``.

τ_src comes from the calibrated labels: on a calibrated corpus the source level is
measurable rather than estimable, and τ_src is the anchor the vicinal positives, the mismatched negatives, the
cycle return trip and the identity anchor are all built on — so P_φ's error there is
trained into C_θ rather than merely reported.  See
``vic/training/converter_cgan_v2_labels_module.py``.

    ``labels``     ``leq_aggregate`` of the calibrated per-frame dB SPL
                   ``IntensityDataset`` computes from ``calibration_rms`` and
                   ``distance_m``, clamped at ``floor_db``.  AVID is calibrated, so this
                   is the ground truth.  P_φ is still loaded — it reads the *achieved*
                   intensity of decoded audio, which has no ground truth by construction,
                   and ``val/ruler_bias_db`` now measures its error against the labels.
    ``predictor``  P_φ estimates τ_src on the fly.  The only option on an
                   uncalibrated corpus.

Usage
-----
    launch_experiment --config configs/paper/train_converter.yaml \\
                      --script scripts/train_converter_cgan_v2_labels.py

Config sections
---------------
  extractor           : NAC pipeline for the conversion space (whitening: false).
  label_extractor     : whitening params for P_φ's measurement space.  Merged
                        over ``extractor``, so the codec paths are not repeated.
  predictor           : frozen P_φ matching ``label_extractor``.
  test_predictor      : (optional) independent predictor for test-time evaluation.
  data                : corpus with train/val/test splits.  ``window_length`` is
                        required under ``source: labels`` — the SPL analysis window in
                        samples, which should match the one P_φ was trained on.
  data_labels         : (optional) labelled corpus whose training-split
                        intensity_db column fits the LabelScaler.
  model.converter     : converter architecture.
  model.discriminator : D_ψ backbone architecture.
  model.conditioning  : how τ enters D_ψ (projection | concat) + vicinal sampling.
  training            : optimiser, batch size, loss weights.
  training.resume_from : (optional) a ``.ckpt`` (or a directory holding
                        ``last.ckpt``) to continue an EARLIER run from.  This
                        output_dir's own ``last.ckpt`` always wins, so a requeue
                        continues itself rather than jumping back to the seed.
                        In a sweep it belongs in the per-cell overrides — one
                        value in the shared block seeds every cell from one cell.
  training.tau_src    : where the source level comes from.  Required.
                        ``source``   — ``labels`` | ``predictor``, see above.
                        ``floor_db`` — the frame labels are clamped here before
                        aggregation; required under ``labels``, unused otherwise.
  training.frame_critic : (optional) per-frame adversarial term alongside the pooled one,
                        which is what stops the converter buzzing in silence.  Absent, or
                        ``weight: 0.0``, reproduces the pooled objective exactly.
                        ``weight``     — λ_frame; every adversarial term becomes
                                         ``(L_pooled + λ·L_frame) / (1 + λ)``.
                        ``silence_db`` — frames (and chunks) below this level are dropped
                                         from the mismatch term, because a pause carries
                                         no information about vocal effort.  Needs
                                         ``tau_src.source: labels``.
"""
from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.utilities import rank_zero_info
from torch.utils.data import DataLoader, Subset

from vic.checkpoints import resolve_resume
from vic.data.collate import make_intensity_collate_fn, make_unlabeled_collate_fn
from vic.data.metadata import resolve_paths
from vic.data.dataset import IntensityDataset, UnlabeledAudioDataset
from vic.data.sampling import duration_weighted_sampler, epoch_size
from vic.evaluation import IntensityPredictorMetric
from vic.training.build import (
    build_conversion_pipelines,
    build_group_evaluation,
    build_labels_module,
    resolve_tau_src,
)
from vic.training.callbacks import (
    ConversionQualityCallback,
    IntensityConversionCallback,
    IntensityEvaluationCallback,
    build_checkpoint_callbacks,
)
from vic.training.utils import configure_matmul_precision

from experiment_launcher import parse_args


def random_subset(dataset, n: int, seed: int) -> Subset:
    """A fixed random subset of ``dataset``, identical across epochs and ranks.

    Taking the first n rows instead would be wrong with a segment index: the
    metadata is ordered by file, so consecutive rows are consecutive segments of
    one recording -- n_samples=64 yielded 64 segments of a single speaker in a
    single effort condition.  Sampling spreads the callbacks over speakers,
    rooms and the whole intensity range.

    Drawn from its own generator rather than the global RNG so the subset
    depends on the seed alone, and stays the same from epoch to epoch: the
    monitored metric must move because the model changed, not because the
    material did, and the generation callback must offer the same utterances
    each time for listening comparisons to mean anything.
    """
    n = min(n, len(dataset))
    indices = np.random.default_rng(seed).choice(len(dataset), size=n, replace=False)
    return Subset(dataset, sorted(indices.tolist()))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@parse_args
def main(config: dict, output_dir: Path):

    configure_matmul_precision()
    subset_seed = config.get("seed", 42)
    L.seed_everything(subset_seed)

    # Two pipelines, one codec — see vic/training/build.py.
    pipeline, label_pipeline = build_conversion_pipelines(config)

    sr       = pipeline.sample_rate
    hop_size = pipeline.frame_grid.hop_size

    # `source` decides which dataset class the data section below builds, so the
    # tau_src block is validated here rather than at module construction.
    source, _ = resolve_tau_src(config)

    # ------------------------------------------------------------------
    # Data.
    # normalize_sequence=False keeps raw amplitudes so the NAC latents encode
    # amplitude information alongside spectral patterns.
    # ------------------------------------------------------------------
    metadata = pd.read_csv(config["data"]["metadata_csv"])
    metadata = resolve_paths(
        metadata, config["data"]["dataset_roots"], path_columns=["signal_path"]
    )

    frame_grid = pipeline.frame_grid
    chunk_s    = config["data"]["chunk_duration_s"]
    normalize  = config["extractor"].get("normalize_sequence", False)

    split_meta = {s: metadata[metadata["split"] == s] for s in ("train", "val", "test")}

    # `source: labels` needs the calibrated per-frame SPL alongside the waveform, which is
    # what IntensityDataset adds -- and it hard-checks calibration_rms/distance_m at
    # construction, so a table without them fails here rather than as NaN losses an hour in.
    # window_length is the SPL analysis window: match the value P_φ was trained with
    # (400 samples in configs/paper/train_predictor.yaml) so the labels and the ruler are
    # measured the same way.
    if source == "labels":
        window_length = config["data"]["window_length"]

        def make_split(split: str, train: bool, chunk: float | None) -> IntensityDataset:
            return IntensityDataset(
                split_meta[split],
                target_sr=sr, frame_grid=frame_grid,
                window_length=window_length,
                chunk_s=chunk, train=train,
                normalize_sequence=normalize,
            )

        collate_fn = make_intensity_collate_fn(sr, hop_size)
    else:
        def make_split(split: str, train: bool, chunk: float | None) -> UnlabeledAudioDataset:
            return UnlabeledAudioDataset(
                split_meta[split],
                target_sr=sr, frame_grid=frame_grid,
                chunk_s=chunk, train=train,
                normalize_sequence=normalize,
            )

        collate_fn = make_unlabeled_collate_fn(sr)

    train_ds = make_split("train", train=True,  chunk=chunk_s)
    val_ds   = make_split("val",   train=False, chunk=chunk_s)
    test_ds  = make_split("test",  train=False, chunk=None)   # full sequences at test

    loader_kw = dict(
        batch_size=config["training"]["batch_size"],
        num_workers=config["training"].get("num_workers", 4),
        pin_memory=True,
        collate_fn=collate_fn,
    )
    # With a segment index one row is a speech segment, not a file, so an epoch
    # would otherwise be ~100x longer and each segment's *seconds* would be drawn
    # at a rate inversely proportional to its duration.  Weighting by duration
    # makes the per-second probability uniform; num_samples sets epoch length
    # explicitly (total across ranks — DDP shards it).  Absent from the config,
    # sampling stays as it was.
    num_samples = epoch_size(config["training"], chunk_s)
    if num_samples is None:
        train_loader = DataLoader(train_ds, shuffle=True, **loader_kw)
    else:
        train_loader = DataLoader(
            train_ds,
            sampler=duration_weighted_sampler(split_meta["train"], num_samples),
            **loader_kw,
        )
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kw)
    test_loader  = DataLoader(
        test_ds, shuffle=False,
        batch_size=config["training"].get("test_batch_size", 1),
        num_workers=loader_kw["num_workers"],
        pin_memory=True,
        collate_fn=collate_fn,
    )

    # ------------------------------------------------------------------
    # Everything from the fitted LabelScaler to the assembled LightningModule —
    # P_φ, the optional test predictor, C_θ, D_ψ, the drawer and the vicinal
    # sampler — is built by vic/training/build.py, so scripts/convert_test_set.py
    # can construct exactly this object from exactly this config.  It is called
    # *here*, after the datasets, because model init draws from the global RNG
    # seed_everything set: moving it changes the weights a given seed produces.
    # ------------------------------------------------------------------
    tc       = config["training"]
    dc_range = tc["intensity_range_db"]
    module   = build_labels_module(config, pipeline, label_pipeline)

    # ------------------------------------------------------------------
    # Checkpointing / stopping.
    #
    # Checkpointing selects on nothing: last.ckpt every epoch, plus a numbered
    # snapshot every ``training.checkpoint_every_n_epochs`` as insurance against an
    # interrupt landing inside a write.  No validation scalar orders a GAN's
    # quality, so there is no metric a "best" checkpoint could be best by.
    #
    # Early stopping is **off by default** and only enabled by setting a positive
    # ``training.patience``; it is the only remaining reader of ``training.monitor``.
    # ------------------------------------------------------------------
    ckpt_dir = output_dir / "checkpoints"
    callbacks = build_checkpoint_callbacks(
        ckpt_dir, every_n_epochs=tc.get("checkpoint_every_n_epochs", 100)
    )

    # patience: null / 0 / absent -> no early stopping.
    patience = tc.get("patience")
    if patience:
        callbacks.append(EarlyStopping(
            monitor=tc.get("monitor", "val/rmse_pred_db"), patience=patience, mode="min",
        ))

    # The paired evaluation on the validation split: the same call the final
    # analysis makes, so a slope logged here means what a slope in the tables
    # means.  Supersedes the `evaluation` block below, which reported only how
    # close P_phi's reading came to the requested level and so could not see a
    # model that hits the level while sounding worse.
    if "conversion_eval" in config:
        cc = config["conversion_eval"]
        # The registering import first: build_metrics resolves against a registry
        # that only `speech_eval.metrics` populates, and without it the failure is
        # `Unknown metric type 'level'. Available: []` -- an empty list, not a typo.
        from speech_eval import metrics as _speech_eval_metrics  # noqa: F401
        from speech_eval.core import build_metrics

        eval_metrics = build_metrics(list(cc["metrics"]))
        if cc.get("intensity_predictor", True):
            eval_metrics = eval_metrics + [IntensityPredictorMetric(
                module, sample_rate=pipeline.sample_rate,
                name=cc.get("intensity_predictor_name", "intensity"),
            )]
        plan = build_group_evaluation(
            split_meta[cc.get("split", "val")], config, source, pipeline,
            levels=cc.get("levels"), vad=cc.get("vad"),
            n_groups=cc.get("n_groups", 24),
            num_workers=cc.get("num_workers", 4), seed=subset_seed,
        )
        loader_factory, targets_by_row, eval_stems, eval_groups, eval_records = plan
        rank_zero_info(
            f"[conversion_eval] {len(eval_stems)} source(s), "
            f"{sum(len(t) for t in targets_by_row.values())} conversion(s), "
            f"metrics: {', '.join(m.name for m in eval_metrics)}"
        )
        callbacks.append(ConversionQualityCallback(
            loader_factory=loader_factory,
            metrics=eval_metrics,
            targets_by_row=targets_by_row,
            stems=eval_stems,
            group_ids=eval_groups,
            records=eval_records,
            displacement_metrics=list(cc["displacement"]),
            group_col=cc.get("group_col", "speaker_uid"),
            every_n_epochs=cc.get("every_n_epochs", 100),
            n_boot=cc.get("n_boot", 1000),
            batch_size=cc.get("batch_size", 16),
            save_figures=cc.get("save_figures", True),
        ))

    if "evaluation" in config and "conversion_eval" not in config:
        ec = config["evaluation"]
        eval_ds = make_split("val", train=False, chunk=None)  # full signals
        n_eval  = min(ec.get("n_samples", 64), len(eval_ds))
        eval_loader = DataLoader(
            random_subset(eval_ds, n_eval, subset_seed),
            batch_size=1,
            shuffle=False,
            num_workers=ec.get("num_workers", 4),
            pin_memory=True,
            collate_fn=collate_fn,
        )
        callbacks.append(IntensityEvaluationCallback(
            loader=eval_loader,
            n_targets=ec.get("n_targets", 5),
            intensity_min_db=ec.get("intensity_min_db", dc_range[0]),
            intensity_max_db=ec.get("intensity_max_db", dc_range[1]),
            every_n_epochs=ec.get("every_n_epochs", 1),
            n_bins=ec.get("n_bins", 10),
        ))

    if "generation" in config:
        gc = config["generation"]
        gen_ds = make_split("val", train=False, chunk=None)  # full signals
        n_gen  = min(gc.get("n_samples", 8), len(gen_ds))
        gen_loader = DataLoader(
            random_subset(gen_ds, n_gen, subset_seed),
            batch_size=1,
            shuffle=False,
            num_workers=gc.get("num_workers", 0),
            pin_memory=True,
            collate_fn=collate_fn,
        )
        callbacks.append(IntensityConversionCallback(
            loader=gen_loader,
            sample_rate=sr,
            n_targets=gc.get("n_targets", 5),
            intensity_min_db=gc["intensity_min_db"],
            intensity_max_db=gc["intensity_max_db"],
            every_n_epochs=gc.get("every_n_epochs", 10),
        ))

    trainer = L.Trainer(
        default_root_dir=output_dir,
        max_epochs=tc["max_epochs"],
        accelerator=config.get("accelerator", "auto"),
        strategy=DDPStrategy(timeout=timedelta(minutes=5), find_unused_parameters=True),
        callbacks=callbacks,
        logger=CSVLogger(save_dir=output_dir, name=""),
        log_every_n_steps=tc.get("log_every_n_steps", 50),
    )
    # Where to continue from.  ``resolve_resume`` owns the precedence: this
    # output_dir's own last.ckpt (a requeue) beats ``training.resume_from`` (a seed
    # from an earlier run), and a resume_from that does not exist raises rather
    # than silently training from scratch.  See vic/checkpoints.py.
    #
    # Every rank resolves the same shared-filesystem paths and so reaches the same
    # decision.  Lightning restores optimiser state, the epoch/step counters and
    # callback state — and the FULL state_dict, frozen submodules included, so a
    # seed written before a codec or P_φ checkpoint changed will put the old frozen
    # weights back without complaint.  Seed only from a run whose frozen components
    # match this config.
    resume_from = resolve_resume(ckpt_dir, tc.get("resume_from"))
    if resume_from is not None:
        rank_zero_info(f"Resuming from {resume_from}")

    trainer.fit(module, train_loader, val_loader, ckpt_path=resume_from)
    trainer.test(module, test_loader)


if __name__ == "__main__":
    main()

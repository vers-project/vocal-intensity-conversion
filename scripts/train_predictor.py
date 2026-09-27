"""Train the intensity predictor P_φ.

Usage
-----
    uv run scripts/train_predictor.py --config configs/predictor.yaml

The extractor type is controlled by configs/predictor.yaml → extractor.type:
    nac           : SpeechTokenizer NAC encoder (frozen)
    melspectrogram: log-mel spectrogram
    spectrogram   : linear (STFT) spectrogram
    wav2vec2      : Wav2Vec2 transformer hidden states (frozen)
"""
from __future__ import annotations
from pytorch_lightning import Callback

from pathlib import Path

import pandas as pd
import torch
import yaml
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader

from vic.data.collate import make_intensity_collate_fn
from vic.data.metadata import resolve_paths
from vic.data.dataset import IntensityDataset
from vic.data.sampling import duration_weighted_sampler, epoch_size
from vic.data.transforms import (
    NORMALIZE_SEQUENCE_PREDICTOR,
    normalize_sequence_from_config,
)
from vic.models.predictor import build_predictor
from vic.training.callbacks import PredictionSaver, ScatterPlotSaver
from vic.training.extraction_pipeline import build_extractor, build_pipeline
from vic.training.predictor_module import PredictorModule
from vic.training.utils import LabelScaler, configure_matmul_precision, describe_model

from experiment_launcher import parse_args


@parse_args
def main(config: dict, output_dir: Path):

    configure_matmul_precision()
    L.seed_everything(config.get("seed", 42))

    # ------------------------------------------------------------------
    # Extraction pipeline — built first so sr and hop_size are authoritative.
    # ------------------------------------------------------------------
    pipeline = build_pipeline(config)
    sr       = pipeline.sample_rate
    hop_size = pipeline.frame_grid.hop_size

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    metadata = pd.read_csv(config["data"]["metadata_csv"])
    metadata = resolve_paths(metadata, config["data"]["dataset_roots"], path_columns=["signal_path", "calibration_path"])
    train_meta = metadata[metadata["split"]=="train"]
    val_meta = metadata[metadata["split"]=="val"]
    test_meta = metadata[metadata["split"]=="test"]

    frame_grid = pipeline.frame_grid
    chunk_s = config["data"]["chunk_duration_s"]
    window_length = config["data"]["window_length"]

    # Stated rather than inherited from IntensityDataset's default: it decides
    # whether P_φ can read level off the amplitude, and vic.predict must apply
    # the same answer at inference.  Labels are measured on the raw waveform
    # before this happens, so normalising does not corrupt them.
    normalize = normalize_sequence_from_config(config, NORMALIZE_SEQUENCE_PREDICTOR)

    train_ds = IntensityDataset(
        train_meta, target_sr=sr, frame_grid=frame_grid,
        window_length=window_length, chunk_s=chunk_s, train=True,
        normalize_sequence=normalize,
    )
    val_ds = IntensityDataset(
        val_meta, target_sr=sr, frame_grid=frame_grid,
        window_length=window_length, chunk_s=chunk_s, train=False,
        normalize_sequence=normalize,
    )

    collate_fn = make_intensity_collate_fn(sr, hop_size)
    loader_kw = dict(
        batch_size=config["training"]["batch_size"],
        num_workers=config["training"].get("num_workers", 4),
        pin_memory=True,
        collate_fn=collate_fn,
    )
    # One __getitem__ is one chunk whatever the row's length, so over a segment
    # index a uniform draw over rows under-samples long segments' seconds: on
    # AVID the paragraph task is 42% of the speech and 7.5% of the rows.
    # Weighting by duration makes the per-second probability uniform.  With
    # neither epoch key set this is exactly the previous behaviour.
    num_samples = epoch_size(config["training"], chunk_s)
    if num_samples is None:
        train_loader = DataLoader(train_ds, shuffle=True, **loader_kw)
    else:
        print(f"Duration-weighted sampling: {num_samples} chunks/epoch "
              f"({num_samples * chunk_s / 3600:.2f} h at {chunk_s}s chunks; "
              f"{len(train_meta)} rows in the split).")
        train_loader = DataLoader(
            train_ds,
            sampler=duration_weighted_sampler(train_meta, num_samples),
            **loader_kw,
        )
    # val is deliberately never weighted: a stochastic validation pass makes
    # checkpoint selection stochastic too.
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kw)

    # Build one test loader per corpus plus a combined "all" loader.
    # Test datasets use full signals (no chunking) to avoid boundary effects.
    test_loader_kw = dict(shuffle=False, batch_size=1,
                          num_workers=loader_kw["num_workers"],
                          pin_memory=True, collate_fn=collate_fn)

    def _make_test_loader(meta_subset: pd.DataFrame) -> DataLoader:
        ds = IntensityDataset(
            meta_subset, target_sr=sr, frame_grid=frame_grid,
            window_length=window_length, chunk_s=None, train=False,
            normalize_sequence=normalize,
        )
        return DataLoader(ds, **test_loader_kw)

    test_corpora = sorted(test_meta["corpus"].unique())
    # "all" first so dataloader_idx=0 always refers to the combined set; "val" last.
    test_dataset_names = ["all"] + test_corpora + ["val"]
    test_loaders = (
        [_make_test_loader(test_meta)]
        + [_make_test_loader(test_meta[test_meta["corpus"] == c]) for c in test_corpora]
        + [_make_test_loader(val_meta)]
    )

    # Label scaler: fit on the sequence-level CSV labels (same dBSPL domain).
    label_scaler = LabelScaler.from_tensor(
        torch.tensor(train_meta["intensity_db"].values, dtype=torch.float32)
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    predictor = build_predictor(config["model"], latent_dim=pipeline.latent_dim)
    describe_model("P_φ", predictor, 1.0 / pipeline.frame_grid.hop_duration_s)

    # ------------------------------------------------------------------
    # Lightning module + trainer
    # ------------------------------------------------------------------
    tc = config["training"]
    aug = tc.get("augmentation", {})

    codec_cfg = aug.get("codec_resynthesis")
    resynthesis_codec = build_extractor({"extractor": codec_cfg}) if codec_cfg is not None else None

    se_cfg = aug.get("speech_enhancement")
    if se_cfg is not None:
        from vic.encoders.mp_senet import MPSENetDenoiser
        denoiser = MPSENetDenoiser(ckpt_path=se_cfg["ckpt_path"], repo_path=se_cfg["repo_path"])
    else:
        denoiser = None

    module = PredictorModule(
        pipeline=pipeline,
        predictor=predictor,
        label_scaler=label_scaler,
        lr=tc.get("lr", 1e-4),
        weight_decay=tc.get("weight_decay", 1e-2),
        max_epochs=tc["max_epochs"],
        denoiser=denoiser,
        resynthesis_codec=resynthesis_codec,
        snr_min_db=aug.get("snr_min_db", None),
        snr_max_db=aug.get("snr_max_db", None),
        spectral_coloring_cfg=aug.get("spectral_coloring"),
        test_dataset_names=test_dataset_names,
        # Fixed duration, independent of chunk_duration_s, so leq_win is
        # comparable between configs that chunk differently.
        leq_window_s=tc.get("leq_window_s", 2.0),
    )

    ckpt_dir = output_dir / "checkpoints"
    callbacks: list = [
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="predictor-{epoch:03d}-{val/loss:.4f}",
            monitor="val/loss",
            mode="min",
            save_top_k=1,
        ),
        EarlyStopping(monitor="val/loss", patience=tc.get("patience", 10), mode="min"),
        PredictionSaver(output_dir, dataset_names=test_dataset_names),
        ScatterPlotSaver(output_dir / "plots", dataset_names=test_dataset_names),
    ]

    trainer = L.Trainer(
        default_root_dir=output_dir,
        max_epochs=tc["max_epochs"],
        accelerator=config.get("accelerator", "auto"),
        devices=config.get("devices", 1),
        callbacks=callbacks,
        logger=CSVLogger(save_dir=output_dir, name=""),
        log_every_n_steps=tc.get("log_every_n_steps", 50),
        gradient_clip_val=tc.get("gradient_clip_val", 1.0),
    )
    trainer.fit(module, train_loader, val_loader)
    trainer.test(module, test_loaders, ckpt_path="best")


if __name__ == "__main__":
    main()

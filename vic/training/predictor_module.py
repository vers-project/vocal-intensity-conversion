"""LightningModule for pretraining the intensity predictor P_φ.

Training objective
------------------
Frame-level MSE regression against calibrated per-frame SPL labels produced
by ``FrameLevelTransform`` in the dataset.  Using frame-level supervision
instead of a single sequence-level target gives T_frames training pairs per
clip rather than one, naturally increasing the effective dataset size.

The encoding pipeline (extractor + optional whitening / normalisation) is
encapsulated in an :class:`~vic.training.extraction_pipeline.ExtractionPipeline`
object that is always kept frozen.  Only the intensity predictor head is trained.

Batch format (from collate_fn)
------------------------------
    "wav"               : AudioBatch  (B, 1, T_samples)
    "frame_intensity_db": AudioBatch  (B, 1, T_frames)   per-frame labels

Three evaluation families, answering three different questions
--------------------------------------------------------------
``frame``    accuracy of one 20 ms frame.  Every valid frame is pooled, so an
             item contributes in proportion to its duration.
``leq``      accuracy of a **whole-item** level, one point per utterance
             regardless of length.  Kept because this is how P_φ is used: the
             converter's ``estimate_tau_src`` is ``leq_aggregate`` over a full
             sequence, and the evaluation design regresses per-utterance
             displacement.  Its difficulty depends on item length -- aggregating
             over ~3x more frames lowers its RMSE by 5-13% on this repo's runs --
             so it is not comparable between corpora of differing length mix.
``leq_win``  accuracy of a **fixed ``leq_window_s``** level, so every unit spans
             the same duration and a long item contributes proportionally more
             of them.  This is the one to compare across corpora, and the one to
             read when a test set mixes 2 s sentences with 46 s paragraphs.
             See ``vic.features.spl.leq_windows`` for how the windows are laid
             out inside an item and why that rule was chosen -- the placement is
             not a one-frame slide, and the choice is not free.

The windows are cut from the frame predictions *after* a full-sequence forward
pass, never by chunking the input, so no zero-padding is introduced at a window
edge -- which is why the test loaders stay un-chunked.

**val and test are not on the same footing.**  ``train_predictor.py`` chunks val
at ``data.chunk_duration_s`` but runs test on full signals, so ``val/leq/*`` and
``test/*/leq/*`` measure different things and must not be compared.  When the val
chunk is shorter than ``leq_window_s``, ``val/leq_win/*`` is identically
``val/leq/*``; that is expected, not a bug.  val's regime is left alone on
purpose: ``val/loss`` selects checkpoints, and changing it would invalidate
comparisons with every previous run.
"""
from __future__ import annotations

import lightning as L
import torch
import torch.nn as nn
import torchmetrics
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from vic.core import AudioCodec
from vic.data.audio_batch import AudioBatch
from vic.data.augmentation import add_noise, codec_resynthesis, random_spectral_coloring, speech_enhancement
from vic.features.spl import leq_aggregate, leq_windows
from vic.training.extraction_pipeline import ExtractionPipeline
from vic.training.utils import LabelScaler


class ErrorStdMetric(torchmetrics.Metric):
    """Streaming population standard deviation of signed errors (pred - labels)."""

    full_state_update = False

    def __init__(self) -> None:
        super().__init__()
        self.add_state("sum",    default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("sum_sq", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count",  default=torch.tensor(0),   dist_reduce_fx="sum")

    def update(self, errors: Tensor) -> None:
        self.sum    += errors.sum()
        self.sum_sq += errors.pow(2).sum()
        self.count  += errors.numel()

    def compute(self) -> Tensor:
        mean = self.sum / self.count
        return (self.sum_sq / self.count - mean.pow(2)).clamp(min=0).sqrt()


def _make_eval_metrics() -> torch.nn.ModuleDict:
    """One set of evaluation metrics for a single level (frame or Leq).

    Metrics
    -------
    ev   : Explained Variance
    r2   : R² score
    mae  : Mean Absolute Error  — average |error| in dB
    rmse : Root Mean Squared Error — like MAE but outlier-sensitive, same dB unit
    me   : Mean Error (signed bias) — detects systematic over/under-prediction
    std  : Std dev of signed errors — spread around the mean error
    """
    return torch.nn.ModuleDict({
        "ev":   torchmetrics.ExplainedVariance(),
        "r2":   torchmetrics.R2Score(),
        "mae":  torchmetrics.MeanAbsoluteError(),
        "rmse": torchmetrics.MeanSquaredError(squared=False),
        "me":   torchmetrics.MeanMetric(),
        "std":  ErrorStdMetric(),
    })


def _update_eval_metrics(
    metrics: torch.nn.ModuleDict, pred: Tensor, labels: Tensor
) -> None:
    errors = pred - labels
    metrics["ev"].update(pred, labels)
    metrics["r2"].update(pred, labels)
    metrics["mae"].update(pred, labels)
    metrics["rmse"].update(pred, labels)
    metrics["me"].update(errors)
    metrics["std"].update(errors)


class PredictorModule(L.LightningModule):
    """Pretrain the frame-level intensity predictor P_φ.

    Parameters
    ----------
    pipeline          : frozen ExtractionPipeline (whitening + extractor +
                        normalisation).  Replaces the individual extractor and
                        normalisation flags from the previous API.
    predictor         : intensity predictor head to train.
    label_scaler      : normalise/denormalise intensity labels.
    lr                : learning rate for AdamW.
    weight_decay      : weight decay for AdamW.
    max_epochs        : used by CosineAnnealingLR.
    denoiser          : if set, suppress background noise with this frozen
                        MP-SENet generator before feature extraction.  Applied
                        first in ``training_step``.  None disables it.
    resynthesis_codec : if set, resynthesize training waveforms through this
                        frozen codec (encode → RVQ → decode) before feature
                        extraction.  Applied after denoising.  None disables it.
    snr_min_db        : lower bound of noise augmentation SNR range (dB).
                        None disables noise augmentation entirely.
    snr_max_db        : upper bound of noise augmentation SNR range (dB).
    spectral_coloring_cfg : if not None, a dict forwarded to
                        ``random_spectral_coloring`` at training time only.
    """

    def __init__(
        self,
        pipeline: ExtractionPipeline,
        predictor: nn.Module,
        label_scaler: LabelScaler,
        lr: float = 1e-4,
        weight_decay: float = 1e-2,
        max_epochs: int = 100,
        denoiser=None,
        resynthesis_codec: AudioCodec | None = None,
        snr_min_db: float | None = None,
        snr_max_db: float | None = None,
        spectral_coloring_cfg: dict | None = None,
        test_dataset_names: list[str] | None = None,
        leq_window_s: float = 2.0,
    ):
        super().__init__()
        self.save_hyperparameters(
            ignore=["pipeline", "predictor", "label_scaler", "denoiser", "resynthesis_codec",
                    "test_dataset_names"]
        )

        self.pipeline = pipeline
        self.predictor = predictor
        self.label_scaler = label_scaler
        self.denoiser = denoiser
        self.resynthesis_codec = resynthesis_codec
        self._test_dataset_names = test_dataset_names or ["all"]

        # A fixed duration, NOT chunk_duration_s: the point of leq_win is to be
        # comparable across configs that differ in chunk length.
        grid = pipeline.frame_grid
        self.leq_window_frames = max(
            1, round(leq_window_s * grid.sample_rate / grid.hop_size)
        )

        families = ("frame", "leq", "leq_win")
        self.val_metrics = torch.nn.ModuleDict(
            {family: _make_eval_metrics() for family in families}
        )
        self.test_metrics = torch.nn.ModuleDict({
            name: torch.nn.ModuleDict(
                {family: _make_eval_metrics() for family in families}
            )
            for name in self._test_dataset_names
        })

    # ------------------------------------------------------------------

    def _step(
        self, batch: dict, stage: str, log_prefix: str | None = None
    ) -> tuple[Tensor, ...]:
        """Forward pass shared by all stages.

        Returns
        -------
        loss         : scalar MSE on normalised predictions.
        pred_valid   : (N,) predictions for valid frames, in dB SPL.
        labels_valid : (N,) labels   for valid frames, in dB SPL.
        leq_pred     : (B,) energy-equivalent level of predictions, in dB SPL.
        leq_labels   : (B,) energy-equivalent level of labels,     in dB SPL.
        win_pred     : (W,) per-window Leq of predictions, batch-flattened.
        win_labels   : (W,) per-window Leq of labels, batch-flattened.
        n_frames     : (B,) valid frame count per item.

        ``W`` is not ``B``: an item yields one window per ``leq_window_frames``
        of valid frames, so a long item contributes several.  See the module
        docstring for what each of the three levels measures.
        """
        wav_batch: AudioBatch = batch["wav"]
        label_batch: AudioBatch = batch["frame_intensity_db"]   # (B, 1, T_frames)

        prefix = log_prefix if log_prefix is not None else stage

        z_batch = self.pipeline.encode(wav_batch, training=(stage == "train"))
        pred_frames = self.predictor(z_batch)                   # (B, T_pred)

        frame_labels = label_batch.data[:, 0, :]                # (B, T_label)
        mask = z_batch.padding_mask                             # (B, T_pred) bool

        T_pred, T_label = pred_frames.shape[1], frame_labels.shape[1]
        if T_pred != T_label:
            diff = abs(T_pred - T_label)
            if diff > 2:
                import warnings
                warnings.warn(
                    f"[{stage}] Large frame-length mismatch: pred={T_pred}, "
                    f"label={T_label} (diff={diff}). Check frame-grid configuration.",
                    RuntimeWarning, stacklevel=2,
                )
            T = min(T_pred, T_label)
            pred_frames  = pred_frames[:, :T]
            frame_labels = frame_labels[:, :T]
            mask         = mask[:, :T]

        n_valid = mask.float().sum().clamp(min=1)

        pred_norm   = self.label_scaler.normalise(pred_frames)
        labels_norm = self.label_scaler.normalise(frame_labels)

        loss = ((pred_norm - labels_norm).pow(2) * mask.float()).sum() / n_valid
        self.log(f"{prefix}/loss", loss, prog_bar=(stage == "val"), sync_dist=True)

        leq_pred   = leq_aggregate(pred_frames,   mask)
        leq_labels = leq_aggregate(frame_labels,  mask)
        win_pred, win_labels = leq_windows(
            pred_frames, frame_labels, mask, self.leq_window_frames
        )
        return (loss, pred_frames[mask], frame_labels[mask],
                leq_pred, leq_labels, win_pred, win_labels, mask.sum(-1))

    def training_step(self, batch: dict, batch_idx: int) -> Tensor:
        wav = batch["wav"]
        if self.hparams.spectral_coloring_cfg is not None:
            wav = random_spectral_coloring(wav, **self.hparams.spectral_coloring_cfg)
        if self.denoiser is not None:
            wav = speech_enhancement(wav, self.denoiser)
        if self.resynthesis_codec is not None:
            wav = codec_resynthesis(wav, self.resynthesis_codec)
        if self.hparams.snr_min_db is not None and self.hparams.snr_max_db is not None:
            wav = AudioBatch(
                add_noise(wav.data, self.hparams.snr_min_db, self.hparams.snr_max_db),
                wav.lengths,
                wav.sample_rate,
            )
        if wav is not batch["wav"]:
            batch = {**batch, "wav": wav}
        loss, *_ = self._step(batch, "train")
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        (_, pred, labels, leq_pred, leq_labels,
         win_pred, win_labels, _) = self._step(batch, "val")
        _update_eval_metrics(self.val_metrics["frame"],   pred, labels)
        _update_eval_metrics(self.val_metrics["leq"],     leq_pred, leq_labels)
        _update_eval_metrics(self.val_metrics["leq_win"], win_pred, win_labels)
        self._log_eval_metrics(self.val_metrics["frame"],   "val",         prog_bar=True)
        self._log_eval_metrics(self.val_metrics["leq"],     "val/leq",     prog_bar=False)
        self._log_eval_metrics(self.val_metrics["leq_win"], "val/leq_win", prog_bar=False)

    def _log_eval_metrics(
        self, metrics: torch.nn.ModuleDict, prefix: str, prog_bar: bool = False
    ) -> None:
        """Log all metrics in an eval group under the given prefix."""
        self.log(f"{prefix}/explained_variance", metrics["ev"],   prog_bar=prog_bar, sync_dist=True)
        self.log(f"{prefix}/r2",                 metrics["r2"],   prog_bar=prog_bar, sync_dist=True)
        self.log(f"{prefix}/mae_db",             metrics["mae"],  prog_bar=prog_bar, sync_dist=True)
        self.log(f"{prefix}/rmse_db",            metrics["rmse"], prog_bar=prog_bar, sync_dist=True)
        self.log(f"{prefix}/me_db",              metrics["me"],   prog_bar=prog_bar, sync_dist=True)
        self.log(f"{prefix}/std_db",             metrics["std"],  prog_bar=prog_bar, sync_dist=True)

    def test_step(self, batch: dict, batch_idx: int, dataloader_idx: int = 0) -> dict:
        name = self._test_dataset_names[dataloader_idx]
        prefix = f"test/{name}"
        (_, pred, labels, leq_pred, leq_labels,
         win_pred, win_labels, n_frames) = self._step(batch, "test", log_prefix=prefix)
        m = self.test_metrics[name]
        _update_eval_metrics(m["frame"],   pred, labels)
        _update_eval_metrics(m["leq"],     leq_pred, leq_labels)
        _update_eval_metrics(m["leq_win"], win_pred, win_labels)
        self._log_eval_metrics(m["frame"],   prefix,              prog_bar=True)
        self._log_eval_metrics(m["leq"],     f"{prefix}/leq",     prog_bar=True)
        self._log_eval_metrics(m["leq_win"], f"{prefix}/leq_win", prog_bar=True)

        return {
            "pred":           pred.cpu(),
            "label":          labels.cpu(),
            "leq_pred":       leq_pred.cpu(),
            "leq_label":      leq_labels.cpu(),
            # Per item, so duration = n_frames * hop_size / sample_rate is
            # recoverable offline without assuming batch_size == 1.  `pred` and
            # `label` are already flattened by the mask, so nothing else in this
            # dict says how long each item was.
            "n_frames":       n_frames.cpu(),
            "dataloader_idx": dataloader_idx,
        }

    # ------------------------------------------------------------------

    def configure_optimizers(self):
        opt = AdamW(
            self.predictor.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        sched = CosineAnnealingLR(opt, T_max=self.hparams.max_epochs)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}

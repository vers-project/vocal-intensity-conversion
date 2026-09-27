"""Reusable Lightning callbacks for VIC training."""
from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import torch
import torchaudio
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from torch import Tensor
from torch.utils.data import DataLoader
from vic.data.audio_batch import AudioBatch
from vic.data.transforms import peak_normalize
from vic.training.extraction_pipeline import ExtractionPipeline
from vic.viz.scatter import intensity_scatter


@runtime_checkable
class ConvertibleModule(Protocol):
    """What the two conversion callbacks need from the module they observe.

    Declared once and asserted, rather than probed with ``hasattr`` at each use.
    The distinction matters because the callbacks previously *fell back* when a
    method was absent: they read intensity straight off the converted latent
    with ``P_φ(z_fake)``.  That reading is circular — the converter can move the
    latent wherever P_φ likes without the change surviving resynthesis — so a
    fallback silently selected by a renamed attribute would not disable the
    metric, it would replace it with a flattering one.
    """

    pipeline: ExtractionPipeline
    converter: torch.nn.Module

    def estimate_tau_src(self, wav_batch: AudioBatch) -> Tensor:
        """Sequence-level intensity of a waveform, in raw dBSPL."""
        ...

    def measure_intensity(self, z: AudioBatch) -> Tensor:
        """Sequence-level intensity of a raw latent, via decode → re-encode."""
        ...

    def source_tau(self, batch: dict) -> Tensor:
        """Sequence-level source intensity, from whatever source the module conditions on.

        Distinct from ``estimate_tau_src``: a module may condition on calibrated labels
        while still measuring decoded audio with P_φ.  This is the reading that defines
        Δτ = τ_tgt − τ_src, so the error bins agree with the request the model was given;
        ``estimate_tau_src`` stays the ruler, and the round-trip bias below must keep
        using it or it stops isolating the codec.
        """
        ...

    def convert(
        self, z_real: AudioBatch, tau_tgt: Tensor, batch: dict
    ) -> AudioBatch:
        """Apply C_θ at a sequence-level target intensity.

        Required rather than optional so that a module conditioning C_θ on something
        derived from the source — a per-frame target curve, say — cannot be driven by a
        callback that bypasses the derivation and calls ``converter`` directly.  The
        callback would still run; it would just be evaluating a different model from the
        one being trained.  The whole ``batch`` is passed because that derivation may need
        the source signal or its labels, not only the latent.
        """
        ...


def _require_convertible(pl_module: L.LightningModule, callback: str) -> None:
    """Raise unless ``pl_module`` satisfies :class:`ConvertibleModule`.

    Checked with ``hasattr`` rather than ``isinstance`` against the protocol, which cannot
    work here.  Since Python 3.12 an ``isinstance`` check on a runtime-checkable protocol
    resolves non-callable members with ``inspect.getattr_static``, and ``pipeline`` and
    ``converter`` are submodules living in ``nn.Module._modules`` — reachable only through
    ``nn.Module.__getattr__``, which ``getattr_static`` deliberately does not call.  Every
    LightningModule therefore failed the check on 3.12 regardless of what it implements,
    with a "is missing []" message naming nothing.  ``hasattr`` still catches what the
    protocol is there to catch: a renamed or absent method, which would otherwise let a
    callback silently fall back to a flattering measurement.
    """
    missing = [
        name for name in ("pipeline", "converter", "estimate_tau_src",
                          "measure_intensity", "convert", "source_tau")
        if not hasattr(pl_module, name)
    ]
    if missing:
        raise TypeError(
            f"{callback} needs a converter-style module; "
            f"{type(pl_module).__name__} is missing {missing}. Attach this callback "
            "only to a module implementing ConvertibleModule (see vic/training/callbacks.py)."
        )


def build_checkpoint_callbacks(
    dirpath: Path | str, every_n_epochs: int = 100
) -> list[ModelCheckpoint]:
    """Unconditional checkpointing: the latest epoch, plus periodic snapshots.

    No metric selects what is kept.  A GAN's validation scalars are not a quality
    ordering, so "best" here means "last", and the two callbacks returned are:

    ``last.ckpt``               rewritten at every epoch end.  Written in place, so an
                                interrupt during the write can leave it truncated -- which
                                is what the second callback exists for.
    ``converter-{epoch}.ckpt``  kept forever, one every ``every_n_epochs``.

    Both carry optimiser and epoch state, so either can be resumed from.

    ``monitor=None`` with ``save_top_k=1`` is what makes the first one save every epoch:
    ``save_last=True`` alone does NOT, because since Lightning 2.x ``_save_last_checkpoint``
    only runs when the top-k path already saved at this global step, and ``save_top_k=0``
    returns before that happens.  Writing straight to ``last.ckpt`` also halves the I/O of
    the usual "top-1 plus a mirrored last", which matters when an epoch is under a minute
    and the state dict carries the frozen codec.
    """
    return [
        ModelCheckpoint(
            dirpath=dirpath,
            filename=ModelCheckpoint.CHECKPOINT_NAME_LAST,
            monitor=None,
            save_top_k=1,
            save_last=False,
            # Without this a second epoch would write last-v1.ckpt rather than overwrite.
            enable_version_counter=False,
        ),
        ModelCheckpoint(
            dirpath=dirpath,
            filename="converter-{epoch:04d}",
            monitor=None,
            save_top_k=-1,
            save_last=False,
            every_n_epochs=every_n_epochs,
        ),
    ]


class PredictionSaver(L.Callback):
    """Collect per-sample predictions and labels during the test loop and save them.

    Hooks into ``on_test_batch_end`` to accumulate the dicts returned by
    ``test_step``, then writes one ``.pt`` file per dataset at the end of the epoch.

    Each file is a list of the dicts ``test_step`` returned, one per *batch*::

        [
            {
                "pred":      Tensor(N,),   # valid frames, mask-flattened
                "label":     Tensor(N,),
                "leq_pred":  Tensor(B,),   # one whole-item Leq per item
                "leq_label": Tensor(B,),
                "n_frames":  Tensor(B,),   # valid frames per item
                "dataloader_idx": int,
            },
            ...
        ]

    ``pred``/``label`` are already flattened across the batch by the padding
    mask, so ``n_frames`` is what makes per-item durations recoverable offline
    (``n_frames * hop_size / sample_rate``) without assuming ``batch_size == 1``.
    Records join back to the metadata by position: the test loaders are
    ``shuffle=False`` and ``AudioDataset`` keeps its paths positionally.

    Both tensors are 1-D, on CPU, in the original label units (dB SPL).
    Load with ``torch.load(path)``.

    Parameters
    ----------
    output_dir       : directory where prediction files are written.
    dataset_names    : ordered list of dataset names matching the test dataloaders.
                       Each name produces ``test_predictions_{name}.pt``.
    """

    def __init__(self, output_dir: Path, dataset_names: list[str]):
        self.output_dir = Path(output_dir)
        self._dataset_names = dataset_names
        self._records: dict[str, list[dict]] = {}

    def on_test_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._records = {name: [] for name in self._dataset_names}

    def on_test_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: dict | None,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if outputs is not None:
            name = self._dataset_names[dataloader_idx]
            self._records[name].append(outputs)

    def on_test_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for name, records in self._records.items():
            torch.save(records, self.output_dir / f"test_predictions_{name}.pt")


class ScatterPlotSaver(L.Callback):
    """Save frame-level and sequence-level Leq scatter plots after the test epoch.

    Collects the ``"pred"``, ``"label"``, ``"leq_pred"``, and ``"leq_label"``
    tensors returned by ``test_step``, then writes two PNG files per dataset:

    - ``scatter_frame_{name}.png`` — all valid frames (many points; alpha=0.1).
    - ``scatter_leq_{name}.png``   — one point per sequence (alpha=1.0).

    Parameters
    ----------
    output_dir    : directory where the PNGs are written.
    dataset_names : ordered list of dataset names matching the test dataloaders.
    """

    def __init__(self, output_dir: Path, dataset_names: list[str]):
        self.output_dir = Path(output_dir)
        self._dataset_names = dataset_names
        self._frame_preds:  dict[str, list[torch.Tensor]] = {}
        self._frame_labels: dict[str, list[torch.Tensor]] = {}
        self._leq_preds:    dict[str, list[torch.Tensor]] = {}
        self._leq_labels:   dict[str, list[torch.Tensor]] = {}

    def on_test_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._frame_preds  = {name: [] for name in self._dataset_names}
        self._frame_labels = {name: [] for name in self._dataset_names}
        self._leq_preds    = {name: [] for name in self._dataset_names}
        self._leq_labels   = {name: [] for name in self._dataset_names}

    def on_test_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: dict | None,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if outputs is None:
            return
        name = self._dataset_names[dataloader_idx]
        self._frame_preds[name].append(outputs["pred"])
        self._frame_labels[name].append(outputs["label"])
        if "leq_pred" in outputs:
            self._leq_preds[name].append(outputs["leq_pred"])
            self._leq_labels[name].append(outputs["leq_label"])

    def on_test_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)

        for name in self._dataset_names:
            if self._frame_preds[name]:
                preds  = torch.cat(self._frame_preds[name]).numpy()
                labels = torch.cat(self._frame_labels[name]).numpy()
                fig = intensity_scatter(
                    labels, preds,
                    title=f"Frame-level SPL prediction — {name}", alpha=0.1,
                )
                fig.savefig(self.output_dir / f"scatter_frame_{name}.png", dpi=150)

            if self._leq_preds[name]:
                preds  = torch.cat(self._leq_preds[name]).numpy()
                labels = torch.cat(self._leq_labels[name]).numpy()
                fig = intensity_scatter(
                    labels, preds,
                    title=f"Sequence-level Leq prediction — {name}", alpha=1.0,
                )
                fig.savefig(self.output_dir / f"scatter_leq_{name}.png", dpi=150)


class IntensityConversionCallback(L.Callback):
    """Periodically convert validation samples to multiple target intensities and save them.

    At every ``every_n_epochs`` epochs (and always at the last epoch), the
    callback iterates over ``loader`` (which should yield full, un-chunked
    signals), converts each to ``n_targets`` linearly-spaced target intensities,
    and saves both the original and all converted waveforms as WAV files.

    All saved waveforms are peak-normalised so that amplitude differences do not
    bias perceptual intensity judgements during listening tests — only spectral
    and temporal cues carry the intensity information.

    Output structure::

        {output_dir}/
          epoch_{N:04d}/
            sample_{i:04d}_src{src:+.1f}dB.wav                     ← original (peak-norm)
            sample_{i:04d}_src{src:+.1f}dB_tgt{tgt:+.1f}dB.wav    ← converted (peak-norm, ×n_targets)
            ...

    The dB values in filenames are the frozen NAC predictor's Leq estimates.

    Measurement path
    ----------------
    Both intensities come from ``pl_module.estimate_tau_src(wav)``, which reads
    the *waveform* through the module's own measurement pipeline — necessary
    when the conversion latents are raw and the predictor expects whitened ones.
    The converted waveform is already decoded, so its achieved intensity costs
    only a whitened re-encode and goes in the filename too::

        sample_{i:04d}_src{src:+.1f}dB_tgt{tgt:+.1f}dB_pred{pred:+.1f}dB.wav

    The module must satisfy :class:`ConvertibleModule`; one that does not raises
    rather than being skipped.

    Parameters
    ----------
    loader           : DataLoader yielding full-signal batches (batch_size=1).
                       Build it with ``chunk_s=None`` so complete utterances are
                       loaded regardless of the training chunk length.
    sample_rate      : audio sample rate for saving WAV files.
    n_targets        : number of target intensity levels.
    intensity_min_db : lower bound of the target intensity range (dB).
    intensity_max_db : upper bound of the target intensity range (dB).
    every_n_epochs   : save every this many epochs (1-indexed; also fires on the
                       last epoch regardless).
    output_dir       : root directory for saved audio.  Defaults to
                       ``trainer.log_dir/generations``.
    """

    def __init__(
        self,
        loader: DataLoader,
        sample_rate: int,
        n_targets: int = 5,
        intensity_min_db: float = 55.0,
        intensity_max_db: float = 85.0,
        every_n_epochs: int = 10,
        output_dir: str | Path | None = None,
    ):
        self._loader         = loader
        self._sr             = sample_rate
        self._n_targets      = n_targets
        self._intensity_min  = intensity_min_db
        self._intensity_max  = intensity_max_db
        self._every_n        = every_n_epochs
        self._out_root       = Path(output_dir) if output_dir is not None else None

    @torch.no_grad()
    def on_validation_epoch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        if trainer.sanity_checking:
            return

        epoch   = trainer.current_epoch
        is_last = (epoch + 1 == trainer.max_epochs)
        if not (is_last or (epoch + 1) % self._every_n == 0):
            return

        _require_convertible(pl_module, "IntensityConversionCallback")

        # Resolve output directory.
        out_root = self._out_root
        if out_root is None:
            log_dir = trainer.log_dir or trainer.default_root_dir
            out_root = Path(log_dir) / "generations"
        epoch_dir = out_root / f"epoch_{epoch + 1:04d}"
        epoch_dir.mkdir(parents=True, exist_ok=True)

        device     = pl_module.device
        targets_db = torch.linspace(self._intensity_min, self._intensity_max, self._n_targets)

        pl_module.eval()

        for i, batch in enumerate(self._loader):
            # The whole batch, not only the waveform: a module may derive its condition
            # from the calibrated labels collated alongside it, and those must be on the
            # same device as the model.
            batch     = {k: v.to(device) for k, v in batch.items()}
            wav_batch = batch["wav"]

            z_real = pl_module.pipeline.encode(wav_batch, training=False)

            # Source intensity as the module itself derives it, for file naming — the
            # same reading the conversion below is displaced from, so src and tgt in a
            # filename are on one scale.
            src_db = pl_module.source_tau(batch)[0].item()

            # Save peak-normalised source audio (before any encode/decode round-trip).
            src_wav  = peak_normalize(wav_batch.unbatch()[0])    # (1, T_samples)
            src_path = epoch_dir / f"sample_{i:04d}_src{src_db:+.1f}dB.wav"
            torchaudio.save(str(src_path), src_wav.cpu(), self._sr)

            # Convert to each target intensity and save (all peak-normalised).
            for tgt_db_val in targets_db:
                tau_tgt  = tgt_db_val.unsqueeze(0).to(device)    # (1,)
                z_fake   = pl_module.convert(z_real, tau_tgt, batch)
                wav_fake = pl_module.pipeline.extractor.decode(z_fake)

                tgt_db   = tgt_db_val.item()
                conv_wav = peak_normalize(wav_fake.unbatch()[0])  # (1, T_samples)

                # The waveform is already decoded, so the achieved intensity is
                # one whitened re-encode away — worth having in the filename when
                # listening: it says whether the file *should* sound converted.
                pred_db = pl_module.estimate_tau_src(wav_fake)[0].item()
                stem = (f"sample_{i:04d}_src{src_db:+.1f}dB"
                        f"_tgt{tgt_db:+.1f}dB_pred{pred_db:+.1f}dB")

                torchaudio.save(str(epoch_dir / f"{stem}.wav"), conv_wav.cpu(), self._sr)


class IntensityEvaluationCallback(L.Callback):
    """Periodically evaluate conversion accuracy across the target intensity range.

    For each sample in ``loader`` the frozen predictor estimates the source
    intensity τ_src.  The converter is applied for every target intensity τ_tgt
    in a fixed linspace and the post-conversion intensity τ_pred is estimated.
    Results are grouped by the **signed** target delta Δτ = τ_tgt − τ_src, and
    the reported statistic is the **signed** error τ_pred − τ_tgt: a perfect
    converter produces a flat curve at 0 dB across all bins.

    Signed, not absolute, on both axes — this is load-bearing.  Grouping by
    |Δτ| merges "convert down by 15 dB" with "convert up by 15 dB", and those
    are not the same task: measured on the Mixer 7 conv run, the downward half
    of the 15–20 dB bin sat at +1.7 ± 1.5 dB while the upward half sat at
    −19.0 ± 8.6 dB.  Averaging two populations that far apart produced a tall
    bar with a huge error bar and hid the fact that one direction worked.  The
    old plot also error-barred ``std(|error|)``, which folds the sign and so
    conflates location with spread; the spread of the signed error is the
    quantity that actually answers "how consistent is this across utterances".

    Measurement path
    ----------------
    Intensity is read with the module's own ``measure_intensity(z)`` /
    ``estimate_tau_src(wav)``, never by applying the predictor to the converted
    latent.  This matters for more than correctness: when the conversion latents
    are raw and the predictor expects whitened ones, reading the latent is not
    merely wrong, it is *circular* — the module's measurement routes through the
    decoder, so the reported RMSE reflects intensity that survived resynthesis
    rather than intensity the predictor can be talked into seeing.  The cost is
    one decode per (sample, target) pair.

    Because the direct-latent reading flatters the model, it is not offered as a
    fallback: a module not satisfying :class:`ConvertibleModule` raises.

    Logged metrics
    --------------
    val/rmse_intensity_db     : overall RMSE of τ_pred vs. τ_tgt across all
                                (sample, target) pairs, in dB.
    val/roundtrip_bias_db     : mean of τ_roundtrip − τ_src over the subset,
                                where τ_roundtrip is the *unconverted* latent
                                measured through the same decode → whiten →
                                re-encode → P_φ path as τ_pred.
    val/roundtrip_bias_sd_db  : spread of that same quantity across utterances.

    The spread is the point.  ``val/codec_bias_db`` already reports the mean
    offset the measurement path introduces, but a mean cannot distinguish "the
    ruler is uniformly 3 dB short" from "the ruler is 1 dB short on one
    utterance and 7 dB short on the next".  Only the second explains a
    per-utterance conversion offset that is stable across epochs, so this sd
    decides whether such an offset belongs to the converter or to the
    measurement.  It is computed here rather than in ``validation_step``
    because this loader is a seeded *random* subset spanning many recordings,
    whereas the validation loader runs unshuffled over a metadata file ordered
    by ``signal_path`` — consecutive rows there are consecutive segments of one
    recording, so a within-batch sd would measure within-recording spread and
    badly understate the across-recording spread.  Costs one extra decode per
    sample (not per sample × target).

    Saved artefacts
    ---------------
    {output_dir}/epoch_{N:04d}_error_curve.png : signed error vs. signed Δτ.

    Parameters
    ----------
    loader           : DataLoader over full-signal samples (chunk_s=None,
                       batch_size=1).  Use a Subset to control sample count.
    n_targets        : number of linearly-spaced target intensities.
    intensity_min_db : lower bound of the target range (dB).
    intensity_max_db : upper bound of the target range (dB).
    every_n_epochs   : evaluate every this many epochs (also fires on last epoch).
    n_bins           : number of |Δτ| bins for the RMSE curve.
    output_dir       : directory for PNG files.  Defaults to
                       ``trainer.log_dir/eval_intensity``.
    """

    def __init__(
        self,
        loader: DataLoader,
        n_targets: int = 5,
        intensity_min_db: float = 55.0,
        intensity_max_db: float = 85.0,
        every_n_epochs: int = 1,
        n_bins: int = 10,
        output_dir: str | Path | None = None,
    ):
        self._loader        = loader
        self._n_targets     = n_targets
        self._intensity_min = intensity_min_db
        self._intensity_max = intensity_max_db
        self._every_n       = every_n_epochs
        self._n_bins        = n_bins
        self._out_root      = Path(output_dir) if output_dir is not None else None

    @torch.no_grad()
    def on_validation_epoch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        if trainer.sanity_checking:
            return

        epoch   = trainer.current_epoch
        is_last = (epoch + 1 == trainer.max_epochs)
        if not (is_last or (epoch + 1) % self._every_n == 0):
            return

        _require_convertible(pl_module, "IntensityEvaluationCallback")

        device     = pl_module.device
        targets_db = torch.linspace(
            self._intensity_min, self._intensity_max, self._n_targets, device=device,
        )

        tau_src_list:  list[torch.Tensor] = []
        tau_tgt_list:  list[torch.Tensor] = []
        tau_pred_list: list[torch.Tensor] = []
        tau_rt_list:   list[torch.Tensor] = []
        # τ_src once per sample, not once per (sample, target) as tau_src_list
        # holds it — the round-trip bias is a per-utterance quantity.
        tau_src_once:  list[torch.Tensor] = []

        pl_module.eval()

        for batch in self._loader:
            batch     = {k: v.to(device) for k, v in batch.items()}
            wav_batch = batch["wav"]
            z_real    = pl_module.pipeline.encode(wav_batch, training=False)
            # The source level the module conditions on, so Δτ = τ_tgt − τ_src bins the
            # error by the request the converter actually received.
            tau_src   = pl_module.source_tau(batch)                                      # (B,)

            # The unconverted latent pushed through the *same* measurement path
            # as τ_pred.  Any offset between this and τ_src is the ruler, not
            # the converter, and must be subtracted before reading the bars.
            #
            # Both sides of this difference must be P_φ readings: it exists to isolate
            # what the decode → whiten → re-encode round trip does to the ruler, and a
            # calibrated τ_src would fold label-vs-P_φ disagreement into it.  So it keeps
            # estimate_tau_src even where the bins above use source_tau.
            tau_rt_list.append(pl_module.measure_intensity(z_real).cpu())
            tau_src_once.append(pl_module.estimate_tau_src(wav_batch).cpu())

            for tau_tgt_val in targets_db:
                tau_tgt  = tau_tgt_val.unsqueeze(0).expand(tau_src.shape[0])             # (B,)
                z_fake   = pl_module.convert(z_real, tau_tgt, batch)
                tau_pred = pl_module.measure_intensity(z_fake)                           # (B,)

                tau_src_list.append(tau_src.cpu())
                tau_tgt_list.append(tau_tgt.cpu())
                tau_pred_list.append(tau_pred.cpu())

        tau_src_all  = torch.cat(tau_src_list).numpy()   # (N,)
        tau_tgt_all  = torch.cat(tau_tgt_list).numpy()   # (N,)
        tau_pred_all = torch.cat(tau_pred_list).numpy()  # (N,)

        error       = tau_pred_all - tau_tgt_all         # (N,) SIGNED error per pair
        delta       = tau_tgt_all - tau_src_all          # (N,) SIGNED request size

        rmse = float(np.sqrt((error ** 2).mean()))
        pl_module.log("val/rmse_intensity_db", rmse, prog_bar=True, sync_dist=True)

        # Per-utterance bias of the measurement path itself.  sd ≈ 1 dB means the
        # ruler is consistent and a per-utterance conversion offset is the
        # converter's doing; sd of several dB means the offset is the ruler's and
        # the work belongs on P_φ, not on C_θ.
        if tau_rt_list:
            rt_bias = torch.cat(tau_rt_list).numpy() - torch.cat(tau_src_once).numpy()
            pl_module.log_dict({
                "val/roundtrip_bias_db":    float(rt_bias.mean()),
                "val/roundtrip_bias_sd_db": float(rt_bias.std()),
            }, prog_bar=False, sync_dist=True)

        # dτ_pred/dτ_tgt: how much of the requested intensity change was actually
        # realised. 1.0 = the request is honoured, 0.0 = the output ignores it and
        # reproduces the source. Unlike the RMSE it has an unambiguous target and
        # is unaffected by how far τ_tgt happens to be drawn from τ_src.
        # Computed within utterance (both series demeaned per source sample), so
        # variation in τ_src cannot leak into the slope.
        #
        # NOT a signal gain. τ_pred comes from P_φ reading *spectrally whitened*
        # audio, and whitening subtracts mean_t(log|X[f,t]|) per bin, so scaling
        # the waveform cancels exactly — a converter that only changed level would
        # score 0.000 here. The slope moves only if F0, formants, spectral tilt or
        # voicing structure move, which is what vocal intensity actually is.
        pl_module.log(
            "val/conversion_slope",
            self._conversion_slope(tau_src_all, tau_tgt_all, tau_pred_all),
            prog_bar=True, sync_dist=True,
        )

        self._save_error_curve(trainer, epoch, delta, error, rmse)

    @staticmethod
    def _conversion_slope(
        tau_src: np.ndarray, tau_tgt: np.ndarray, tau_pred: np.ndarray
    ) -> float:
        """Within-utterance least-squares slope of τ_pred on τ_tgt.

        Each source sample contributes one group (identified by its τ_src, which
        is constant across that sample's targets).  Demeaning both series inside
        the group makes this a fixed-effects slope: the between-utterance spread
        of τ_src, which a do-nothing converter would track perfectly, cannot
        inflate it.  Returns NaN when there is no variation to regress on.
        """
        x, y = [], []
        for s in np.unique(tau_src):
            k = tau_src == s
            if k.sum() < 2:
                continue
            x.append(tau_tgt[k] - tau_tgt[k].mean())
            y.append(tau_pred[k] - tau_pred[k].mean())
        if not x:
            return float("nan")
        x, y = np.concatenate(x), np.concatenate(y)
        denom = float((x * x).sum())
        return float((x * y).sum() / denom) if denom > 0 else float("nan")

    @staticmethod
    def _signed_bin_edges(delta: np.ndarray, n_bins: int) -> np.ndarray:
        """Bin edges over signed Δτ that always place a boundary at zero.

        Splitting the budget either side of zero rather than running a single
        linspace over [min, max] is the whole point of the chart: a bin that
        straddles zero re-creates, in miniature, the up/down pooling this plot
        exists to separate.
        """
        lo, hi = float(delta.min()), float(delta.max())
        if lo >= 0.0:
            return np.linspace(max(lo, 0.0), hi, n_bins + 1)
        if hi <= 0.0:
            return np.linspace(lo, min(hi, 0.0), n_bins + 1)
        n_neg = int(round(n_bins * (-lo) / (hi - lo)))
        n_neg = min(max(n_neg, 1), n_bins - 1)          # both sides get ≥ 1 bin
        return np.concatenate([
            np.linspace(lo, 0.0, n_neg + 1),
            np.linspace(0.0, hi, n_bins - n_neg + 1)[1:],
        ])

    def _save_error_curve(
        self,
        trainer: L.Trainer,
        epoch: int,
        delta: np.ndarray,
        error: np.ndarray,
        rmse: float,
    ) -> None:
        import matplotlib.pyplot as plt

        out_root = self._out_root
        if out_root is None:
            log_dir  = trainer.log_dir or trainer.default_root_dir
            out_root = Path(log_dir) / "eval_intensity"
        out_root.mkdir(parents=True, exist_ok=True)

        bin_edges   = self._signed_bin_edges(delta, self._n_bins)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        n_bins      = len(bin_centers)
        indices     = np.clip(np.digitize(delta, bin_edges[:-1]) - 1, 0, n_bins - 1)

        counts   = np.array([(indices == b).sum() for b in range(n_bins)])
        bin_mean = np.array([
            error[indices == b].mean() if counts[b] else np.nan for b in range(n_bins)
        ])
        # Spread of the SIGNED error: how consistent the converter is across
        # utterances asked for a comparable change.  std(|error|) would fold the
        # sign and mix location into scale.
        bin_sd = np.array([
            error[indices == b].std() if counts[b] else np.nan for b in range(n_bins)
        ])

        # Diverging pair (Okabe-Ito): downward vs upward requests are opposite
        # tasks, so they get opposite poles rather than one cycled hue.
        DOWN, UP = "#0072B2", "#D55E00"
        colors = [DOWN if c < 0 else UP for c in bin_centers]

        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        band = ax.axhspan(-3, 3, color="0.90", zorder=0)
        ax.bar(
            bin_centers, bin_mean,
            width=np.diff(bin_edges) * 0.8,
            color=colors, alpha=0.85, zorder=2,
            yerr=bin_sd, capsize=3,
            error_kw=dict(ecolor="0.25", lw=1.2, zorder=3),
        )
        ax.axhline(0.0, color="0.25", lw=1.2, zorder=4)
        ax.axvline(0.0, color="0.55", lw=1.0, ls=":", zorder=1)

        # Anchor the count outside the whisker, not on the bar tip, or it lands
        # on the error-bar cap.
        for c, m, s, n in zip(bin_centers, bin_mean, bin_sd, counts):
            if n:
                up_side = m >= 0
                tip = m + s if up_side else m - s
                ax.annotate(f"n={n}", (c, tip), textcoords="offset points",
                            xytext=(0, 5 if up_side else -13), ha="center",
                            fontsize=7, color="0.35", zorder=5)

        handles = [
            plt.Rectangle((0, 0), 1, 1, color=DOWN, alpha=0.85),
            plt.Rectangle((0, 0), 1, 1, color=UP,   alpha=0.85),
            band,
        ]
        ax.legend(handles,
                  ["asked quieter (Δτ < 0)", "asked louder (Δτ > 0)", "±3 dB of target"],
                  loc="lower left", fontsize=8, framealpha=0.95)

        ax.set_xlabel("Δτ = τ_tgt − τ_src  (dB)      ← quieter    louder →")
        ax.set_ylabel("signed error  τ_pred − τ_tgt  (dB)")
        ax.set_title(
            f"Conversion error by signed request — epoch {epoch + 1:04d}\n"
            f"bars = mean ± sd across utterances · overall RMSE = {rmse:.2f} dB",
            fontsize=10,
        )
        ax.grid(axis="y", color="0.85", lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        # Room for the n= labels, which sit beyond the whisker ends.
        finite = np.isfinite(bin_mean)
        if finite.any():
            lo = float(np.nanmin(bin_mean - bin_sd)); hi = float(np.nanmax(bin_mean + bin_sd))
            pad = 0.10 * max(hi - lo, 1.0)
            ax.set_ylim(min(lo - pad, -3.5), max(hi + pad, 3.5))
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        fig.tight_layout()
        fig.savefig(out_root / f"epoch_{epoch + 1:04d}_error_curve.png", dpi=150)
        plt.close(fig)


#: The one scalar put on the progress bar.  Named, not spelled out twice.
HEADLINE_SCALAR = "val/slope/intensity.leq_db"


class ConversionQualityCallback(L.Callback):
    """Run the *test* evaluation on the validation split, periodically.

    Replaces :class:`IntensityEvaluationCallback`, which reported one number —
    how close P_φ's reading of the output came to the requested level — and could
    therefore not see a model that hits the requested level while sounding worse.
    Checkpoint selection needs something that can.

    What it measures is not a cheaper approximation of the final evaluation: it is
    the same call.  Rendering, measuring, the three-reference join and the
    per-speaker fits all come from :mod:`vic.evaluation.monitor`, so a slope logged
    here and a slope in the paper's table differ in the split they were computed on
    and in nothing else.  ``val/slope/intensity.leq_db`` is the level axis,
    alongside the same fit for $f_0$, formants, spectral balance and predicted
    naturalness.

    It does NOT reuse the name ``val/conversion_slope``
    ---------------------------------------------------
    The superseded callback logged a slope under that name, and re-using it would
    make a resumed run chart one continuous series across two different
    quantities.  That one demeaned within utterance and regressed P_phi's reading
    on a linspace of targets spanning the whole configured range; this one
    regresses *achieved* on *requested* displacement against the four real
    recorded levels, with the codec round trip as the origin, fitted per speaker.
    Same intent, different material and different reference, so the numbers step
    at the point the callbacks were swapped.  Leaving the break visible is the
    point: these curves are read to choose a checkpoint, and a disguised step
    invites reading it as the model changing.

    Cost, and why it can afford to be this thorough
    ----------------------------------------------
    The callback it replaces decoded once per (sample, target) pair at batch 1 —
    with 64 samples and 10 targets, 640 decodes **every epoch**.  This one runs on
    a fixed set of sentence groups every ``every_n_epochs``, so at 24 groups and a
    100-epoch period it is both more informative and substantially cheaper.

    The utterance set must not vary between evaluations
    ---------------------------------------------------
    A naturalness predictor is trained on ordinary conversational loudness, so its
    absolute value on shouted speech is not a calibrated opinion score; only the
    trend across epochs is readable, and only if the material is held fixed.  The
    loader is therefore built once by the caller and replayed, never resampled.

    DDP
    ---
    Every rank runs the whole evaluation over the same non-distributed loader and
    logs with ``sync_dist=True``.  That is deliberate and matches the other
    callbacks here: guarding on rank zero leaves the other ranks at the metric
    all-reduce while rank zero is still measuring, which ends in an NCCL timeout
    rather than a speed-up.

    Parameters
    ----------
    loader_factory      : called once per pass to get a fresh iterator of
                          single-source batches.  A factory, not a loader: the two
                          passes each restart from the beginning, and a consumed
                          iterator yields a silently empty second pass.
    metrics             : ``speech_eval`` metrics, constructed once and reused, so
                          their backends load on the first evaluation only.
    targets_by_row      : row index -> level targets.
    stems, group_ids, records : per-row identity, aligned with the loader's order.
    displacement_metrics: columns that should move with the conversion.  Absent
                          ones are skipped, so one list serves several metric sets.
    group_col           : what a slope is fitted within.  ``speaker_uid``, never a
                          raw speaker column: ids are unique only within a corpus.
    every_n_epochs      : evaluate every this many epochs (also fires on the last).
    n_boot              : bootstrap resamples for the interval.  Small on purpose —
                          a monitor reads the slope, not its width.
    output_dir          : where the tables and figures go.  Defaults to
                          ``trainer.log_dir/eval_conversion``.
    save_figures        : write the scatter and per-speaker slope plots.
    """

    def __init__(
        self,
        loader_factory,
        metrics,
        targets_by_row: dict,
        stems,
        group_ids,
        records,
        displacement_metrics,
        group_col: str = "speaker_uid",
        every_n_epochs: int = 100,
        n_boot: int = 1_000,
        batch_size: int = 16,
        with_codec: bool = True,
        output_dir: str | Path | None = None,
        save_figures: bool = True,
    ):
        self._loader_factory = loader_factory
        self._metrics = list(metrics)
        self._targets_by_row = targets_by_row
        self._stems = list(stems)
        self._group_ids = list(group_ids)
        self._records = list(records)
        self._displacement = list(displacement_metrics)
        self._group_col = group_col
        self._every_n = int(every_n_epochs)
        self._n_boot = int(n_boot)
        self._batch_size = int(batch_size)
        self._with_codec = bool(with_codec)
        self._out_root = Path(output_dir) if output_dir is not None else None
        self._save_figures = bool(save_figures)

    def _output_dir(self, trainer: L.Trainer) -> Path:
        root = self._out_root or Path(trainer.log_dir or ".") / "eval_conversion"
        root.mkdir(parents=True, exist_ok=True)
        return root

    @torch.no_grad()
    def on_validation_epoch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        if trainer.sanity_checking:
            return
        epoch = trainer.current_epoch
        is_last = (epoch + 1 == trainer.max_epochs)
        if not (is_last or (epoch + 1) % self._every_n == 0):
            return

        _require_convertible(pl_module, "ConversionQualityCallback")

        from vic.evaluation.monitor import measure_conversions, summarise_conversions

        pl_module.eval()
        results, arrays = measure_conversions(
            pl_module,
            pl_module.pipeline,
            self._loader_factory,
            self._metrics,
            self._targets_by_row,
            self._stems,
            self._group_ids,
            self._records,
            pl_module.device,
            with_codec=self._with_codec,
            batch_size=self._batch_size,
        )
        summary = summarise_conversions(
            results, arrays, self._displacement,
            group_col=self._group_col, n_boot=self._n_boot,
        )

        scalars = summary.scalars(prefix="val")
        headline = scalars.pop(HEADLINE_SCALAR, None)
        if scalars:
            pl_module.log_dict(scalars, prog_bar=False, sync_dist=True)
        if headline is not None:
            # Separately, and on the progress bar, because it is the number a
            # human watches.  Logged under its own name and deliberately NOT
            # under the old `val/conversion_slope` -- see the class docstring.
            pl_module.log(HEADLINE_SCALAR, headline, prog_bar=True, sync_dist=True)

        self._write(trainer, epoch, summary)

    def _write(self, trainer: L.Trainer, epoch: int, summary) -> None:
        """Tables and figures, named by epoch so the trend survives the run."""
        out = self._output_dir(trainer)
        tag = f"epoch_{epoch:04d}"
        for name, table in (
            ("summary", summary.summary),
            ("slopes", summary.slopes),
            ("preservation_speaker", summary.speaker),
            ("preservation_wer", summary.wer),
            ("preservation_quality", summary.quality),
        ):
            if table is not None and not table.empty:
                table.to_csv(out / f"{tag}_{name}.csv", index=False)
        if not self._save_figures or summary.displacement.empty:
            return
        try:
            from speech_eval import figures
        except ImportError:
            # matplotlib is an optional extra; the tables are the point.
            return
        for column in sorted(set(summary.displacement["metric"])):
            figures.save(
                figures.displacement_scatter(summary.displacement, metric=column),
                out / f"{tag}_displacement_{column}.png",
            )
        if not summary.slopes.empty:
            figures.save(
                figures.group_slopes(summary.slopes, group_col=self._group_col,
                                     title="achieved / requested, per speaker"),
                out / f"{tag}_slopes.png",
            )

"""LightningModule: scalar-τ cGAN on raw NAC latents, with τ_src from **calibrated labels**.

The design: two latent spaces over one frozen codec, no predictor term in the objective,
the τ-conditional critic as the only intensity signal, and a *scalar* condition.  This
module takes **τ_src from the calibrated labels** rather than from P_φ's reading.

Why τ_src is worth changing
---------------------------
τ_src is not a diagnostic here.  It is the anchor four separate things are built on:

    tau_pos  = vicinal(τ_src)     the label D_ψ's *real* pairs are shown under
    tau_neg  = mismatch(τ_src)    the label that teaches D_ψ to use τ at all
    cycle    = C(z_fake, τ_src)   the return trip that defines content preservation
    identity = C(z_real, τ_src)   the anchor that says "already at τ_src, do nothing"

In the parent module every one of those is P_φ's reading of a whitened re-encode.  So the
ruler's error does not merely show up in the reported numbers — it defines the τ axis the
critic learns, and it defines the level the identity anchor calls "unchanged".  A
systematic bias in P_φ is therefore trained *into* C_θ, and no metric computed with the
same P_φ can see it.

AVID's metadata carries ``calibration_rms`` and ``distance_m``, so the per-frame dB SPL is
directly measurable, and ``leq_aggregate`` over it is the same sequence-level quantity P_φ
was fitted to approximate — the ground truth, not an estimate of it.  Under
``label_source="labels"`` that is what the four quantities above use.  P_φ is not
dismissed; it is confined to the one reading that has no ground truth by construction, the
*achieved* level of decoded audio, and to ``val/ruler_bias_db``, which is now computable:
P_φ's error against the truth on held-out audio.


``label_source="predictor"`` restores the parent's behaviour exactly, and is the only
option on an uncalibrated corpus.

Why ``floor_db`` is required under labels
-----------------------------------------
``FrameLevelTransform`` floors a silent frame at ``amplitude_to_db``'s -200 dB.  A scatter
of such frames inside an utterance is harmless here — everything passes through
``leq_aggregate`` first, and mean-square-pressure averaging makes them contribute nothing.
What bites is a *whole chunk* of silence: about 3% of random 2-s training chunks land in a
pause inside a long segment, and their Leq is then -200 dB, roughly -25σ after
``LabelScaler``, where P_φ would have read ~34 dB.  Clamping the frame labels at
``floor_db`` before aggregating keeps those chunks at the corpus's own noise floor instead.
Same requirement as the local module, arriving at a different scale for a different reason.

Reading the metrics
-------------------
Unchanged from the parent except:

  ``val/ruler_bias_db``  mean P_φ(source waveform) − Leq(labels), on held-out audio.
      Only defined under ``label_source="labels"``; under ``"predictor"`` it is
      identically zero and is not logged.  This is the direct read on whether a
      reproducible per-utterance conversion offset belongs to the ruler or to the
      converter — a question the P_φ-conditioned runs could not ask.
  ``val/codec_bias_db``  still isolates what decode → whiten → re-encode does to P_φ's
      reading, so **both** sides of the difference are P_φ readings.  Under labels that
      costs one extra P_φ pass per validation batch; folding a label in would have mixed
      label-vs-P_φ disagreement into a metric that means nothing once it contains any.
  ``train/tau_src_db``  is now the calibrated source level rather than P_φ's estimate of
      it, so it is not numerically comparable to the same key in the parent's runs.

Batch format
------------
    "wav"                : AudioBatch  (B, 1, T_samples)
    "frame_intensity_db" : AudioBatch  (B, 1, T_frames)   — label_source: labels only,
                           from make_intensity_collate_fn / IntensityDataset
"""
from __future__ import annotations

import warnings

import lightning as L
import torch
import torch.nn as nn
from torch import Tensor

from vic.data.audio_batch import AudioBatch
from vic.features.spl import leq_aggregate
from vic.models.conditioning import ConditionalDiscriminator, ProjectionConditioner
from vic.training.extraction_pipeline import ExtractionPipeline
from vic.training.losses import (
    cycle_consistency_loss,
    identity_anchor_loss,
    lsgan_loss,
    masked_lsgan_loss,
    masked_mse,
)
from vic.training.utils import (
    LABEL_SCALER_CKPT_KEY,
    IntensityDrawer,
    LabelScaler,
    VicinalSampler,
    apply_spectral_norm,
)

REAL_TARGET = 1.0
FAKE_TARGET = -1.0


def _masked_mean_t(x: Tensor, mask: Tensor) -> Tensor:
    """Mean of ``(B, T)`` over each sequence's valid frames → ``(B,)``."""
    m = mask.to(x.dtype)
    return (x * m).sum(dim=-1) / m.sum(dim=-1).clamp(min=1.0)


class ConverterCGANv2LabelsModule(L.LightningModule):
    """Conditional-realness GAN training for C_θ with a calibrated source level.

    Parameters
    ----------
    pipeline         : frozen ExtractionPipeline for the **conversion space**.
                       Must NOT whiten — the whole point is that the converter,
                       the discriminator and the decoder share the latent space
                       the codec was trained on.  Its extractor must implement
                       ``decode`` (i.e. be an AudioCodec, not a bare extractor).
    label_pipeline   : frozen ExtractionPipeline for the **measurement space**,
                       normally the same codec with ``whitening=True``.  Only
                       P_φ ever sees its output.
    converter        : C_θ — FrameConverter or ContextualConverter.
    predictor        : P_φ — frozen predictor matching ``label_pipeline``.  A measuring
                       instrument only, never a loss: it contributes no gradient anywhere.
                       Under ``label_source="predictor"`` it also supplies τ_src; under
                       ``"labels"`` it is confined to reading the achieved level of
                       decoded audio, and to ``val/ruler_bias_db``.
    cond_disc        : D_ψ — :class:`ConditionalDiscriminator` scoring (z, τ) pairs.
    label_scaler     : normalises dBSPL labels ↔ model output space.
    intensity_drawer : samples target intensities τ_tgt for each batch.
    vicinal_sampler  : builds the matched/mismatched conditioning labels for D_ψ.
    test_pipeline    : optional ExtractionPipeline for cross-predictor test
                       evaluation (e.g. whitening + Wav2Vec2).  Applied to the
                       *decoded* fake waveform, so its ``whitening`` flag now
                       whitens clean audio exactly once — in the whitened-latent
                       runs it was whitening an already-whitened signal.
    test_predictor   : predictor head used with ``test_pipeline``.  Must be
                       provided together with it.
    lr_g, lr_d       : learning rates for C_θ and D_ψ.  The remaining optimiser
                       settings are fixed, not configurable — see
                       ``configure_optimizers``.
    lambda_fake      : weight for the conditional adversarial loss.
    lambda_cycle     : weight for cycle consistency.
    lambda_id        : weight for the identity anchor.  ``0.0`` disables it and
                       saves one converter forward pass per step.
    lambda_mismatch  : relative weight of mismatched-real negatives against fake
                       negatives inside the D-step.  ``0.0`` reduces D_ψ to an
                       unconditional critic in all but name — ablation only.
    spectral_norm    : apply spectral normalisation to D_ψ at init.
    floor_db         : the frame labels are clamped here before aggregation, in the same
                       dBSPL units as the labels.  Required under
                       ``label_source="labels"`` — see the module docstring.  Unused
                       under ``"predictor"``.
    label_source     : ``"labels"`` reads the calibrated per-frame SPL the dataset
                       computed from ``calibration_rms`` and ``distance_m``;
                       ``"predictor"`` estimates τ_src with P_φ, which reproduces
                       :class:`ConverterCGANv2Module` exactly and is the only option on
                       an uncalibrated corpus.
    lambda_frame     : weight of the per-frame adversarial term against the pooled one.
                       ``0.0`` leaves the pooled objective and its code path unchanged.
    silence_db       : frames whose calibrated label is below this, and chunks whose τ_src
                       is below it, are dropped from the **mismatch term** — a non-speech
                       frame carries no information about vocal effort, so it cannot be
                       evidence against a claimed effort level.  ``None`` disables the
                       masking.  Requires ``label_source="labels"``.
    """

    def __init__(
        self,
        pipeline: ExtractionPipeline,
        label_pipeline: ExtractionPipeline,
        converter: nn.Module,
        predictor: nn.Module,
        cond_disc: ConditionalDiscriminator,
        label_scaler: LabelScaler,
        intensity_drawer: IntensityDrawer,
        vicinal_sampler: VicinalSampler,
        test_pipeline: ExtractionPipeline | None = None,
        test_predictor: nn.Module | None = None,
        lr_g: float = 1e-4,
        lr_d: float = 1e-4,
        lambda_fake: float = 1.0,
        lambda_cycle: float = 1.0,
        lambda_id: float = 0.0,
        lambda_mismatch: float = 1.0,
        spectral_norm: bool = True,
        gradient_clip_val: float = 1.0,
        floor_db: float | None = None,
        label_source: str = "predictor",
        lambda_frame: float = 0.0,
        silence_db: float | None = None,
    ):
        super().__init__()
        self.automatic_optimization = False  # manual: separate G and D steps

        if lambda_frame < 0.0:
            raise ValueError(f"lambda_frame must be >= 0, got {lambda_frame}.")
        # The pooled score is derived from the per-frame scores by averaging over time,
        # which is only the same number when the conditional term is affine in φ.  It is
        # for the projection conditioner (an inner product) and it is not for the concat
        # one (an MLP over [φ ; e(τ)]).  Refuse rather than silently optimise two
        # quantities that no longer agree.
        if lambda_frame > 0.0 and not isinstance(cond_disc.conditioner, ProjectionConditioner):
            raise ValueError(
                f"lambda_frame > 0 requires a projection conditioner; this critic has "
                f"{type(cond_disc.conditioner).__name__}. The pooled score is taken as the "
                "time-average of the per-frame scores, which holds only when the "
                "conditional term is affine in the feature. Set "
                "model.conditioning.type: projection, or run with frame_critic.weight: 0."
            )
        if label_source not in ("labels", "predictor"):
            raise ValueError(
                f"label_source must be 'labels' or 'predictor', got {label_source!r}."
            )
        if silence_db is not None and label_source != "labels":
            raise ValueError(
                "silence_db needs the calibrated per-frame labels to decide which frames "
                f"are non-speech, but label_source is {label_source!r}. Set "
                "tau_src.source: labels, or drop frame_critic.silence_db."
            )
        # A chunk that is silent end to end aggregates to -200 dB without a clamp, and
        # that value reaches the vicinal sampler, the intensity embedder and the identity
        # anchor.  See the module docstring for why this is a real case and not a
        # defensive one.
        if label_source == "labels" and floor_db is None:
            raise ValueError(
                "label_source='labels' with floor_db=None: the calibrated frame labels "
                "reach -200 dB on silent frames (amplitude_to_db's floor), so an "
                "all-silent chunk would hand -200 dB to the vicinal sampler and the "
                "intensity embedder as its source level. Set floor_db to the corpus's "
                "own noise floor."
            )

        self.save_hyperparameters(ignore=[
            "pipeline", "label_pipeline", "converter", "predictor", "cond_disc",
            "label_scaler", "intensity_drawer", "vicinal_sampler",
            "test_pipeline", "test_predictor",
        ])

        # The conversion space must be the codec's own space, and we must be able
        # to leave it: every intensity measurement in this module goes through
        # decode(). Catch both at construction rather than at the first val loop.
        if not hasattr(pipeline.extractor, "decode"):
            raise TypeError(
                f"pipeline.extractor ({type(pipeline.extractor).__name__}) has no "
                "decode(); this module measures intensity by decoding z_fake back "
                "to a waveform, so the conversion pipeline must wrap an AudioCodec."
            )
        if pipeline.whitening:
            raise ValueError(
                "pipeline.whitening is True, which puts the converter back in the "
                "whitened latent space this module exists to leave. Set "
                "extractor.whitening: false and put the whitening in label_extractor."
            )
        if not label_pipeline.whitening:
            warnings.warn(
                "label_pipeline.whitening is False, so P_φ will be fed raw latents. "
                "That is correct only if this P_φ checkpoint was trained without "
                "whitening (e.g. a distilled raw-latent student); the NAC+whitening "
                "predictors will report nonsense.",
                RuntimeWarning,
                stacklevel=2,
            )

        self.pipeline         = pipeline
        self.label_pipeline   = label_pipeline
        self.converter        = converter
        self.predictor        = predictor
        self.cond_disc        = cond_disc
        self.label_scaler     = label_scaler
        self.intensity_drawer = intensity_drawer
        self.vicinal_sampler  = vicinal_sampler
        self.test_pipeline    = test_pipeline
        self.test_predictor   = test_predictor

        # Freeze and put in eval *here*, not only in the train() override below:
        # Lightning enters the first training epoch without calling .train()
        # (the module is already in train mode from construction), so a module
        # relying on the override alone runs P_φ in train mode for all of epoch 0
        # and only settles into eval after the first validation loop.
        self.predictor.eval()
        for p in self.predictor.parameters():
            p.requires_grad_(False)
        if self.test_predictor is not None:
            self.test_predictor.eval()
            for p in self.test_predictor.parameters():
                p.requires_grad_(False)

        if spectral_norm:
            # Whole critic, conditioner included (as in Miyato & Koyama).
            # Restricting it to the backbone was tried and reverted. Seed-paired
            # measurement of the init cond/uncond ratio over 8 seeds:
            #     no SN 0.182 | SN backbone only 0.104 | SN whole 0.150
            # Normalising the conditioner *raises* the conditional term (7/8
            # seeds): spectral_norm divides W by σ_max, and at default init
            # σ_max < 1 for these layers, so it scales them up. Excluding the
            # conditioner would have shrunk the one term opposing C_θ = identity.
            # The ratio is strongly init-dependent (0.05–0.40 across seeds), so
            # read train/cond_ratio_* as a trajectory, not an absolute level.
            apply_spectral_norm(self.cond_disc)

    # ------------------------------------------------------------------

    def train(self, mode: bool = True) -> "ConverterCGANv2LabelsModule":
        super().train(mode)
        # Keep frozen predictors in eval so their dropout is off.
        self.predictor.eval()
        if self.test_predictor is not None:
            self.test_predictor.eval()
        # ExtractionPipeline.train() already keeps its extractor in eval.
        return self

    # ------------------------------------------------------------------

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        """Persist the label scaler alongside the weights.

        It is *fitted*, not architectural, and ``save_hyperparameters`` ignores
        it — so without this the only record of the τ normalisation a converter
        was trained under lives in a YAML file that nothing stops anyone editing.
        ``load_label_scaler`` reads it back and cross-checks the config.
        """
        checkpoint[LABEL_SCALER_CKPT_KEY] = self.label_scaler.state_dict()

    # ------------------------------------------------------------------
    # Intensity measurement.  ``estimate_tau_src`` is the only place P_φ is called on a
    # waveform, and every other P_φ reading routes through it.  Under
    # label_source='labels' it is off the conditioning path entirely.
    # Both are public: the evaluation and generation callbacks call them so the
    # measurement path is defined once, here, rather than duplicated per callback.
    # ------------------------------------------------------------------

    @torch.no_grad()
    def estimate_tau_src(self, wav_batch: AudioBatch) -> Tensor:
        """Sequence-level intensity of a *waveform*, in raw dBSPL, as read by P_φ.

        Whitens and re-encodes, so P_φ sees the representation it was trained on
        regardless of what the conversion pipeline does.
        """
        z_w = self.label_pipeline.encode(wav_batch, training=False)
        return leq_aggregate(self.predictor(z_w), z_w.padding_mask)

    @torch.no_grad()
    def measure_intensity(self, z: AudioBatch) -> Tensor:
        """Sequence-level intensity of a *raw latent*, in raw dBSPL.

        Leaves the conversion space the only way it can be left — decode →
        whiten → re-encode — and measures there.  Not differentiable, by
        construction: ``decode`` is ``@torch.no_grad()`` and quantises through the
        RVQ.  That is exactly why no gradient from P_φ can reach the converter.
        """
        wav = self.pipeline.extractor.decode(z)
        return self.estimate_tau_src(wav)

    @torch.no_grad()
    def source_tau(self, batch: dict) -> Tensor:
        """Sequence-level source intensity — the anchor the whole condition is built on.

        ``"predictor"``
            P_φ over a whitened re-encode of the waveform.  Identical to
            :class:`ConverterCGANv2Module`.
        ``"labels"``
            ``leq_aggregate`` of the calibrated per-frame dB SPL ``IntensityDataset``
            measured from ``calibration_rms`` and ``distance_m`` before any peak
            normalisation, clamped at ``floor_db``.  On AVID this is the ground truth.

        Part of the callbacks' ConvertibleModule contract, so the evaluation callback's
        Δτ = τ_tgt − τ_src axis is keyed on the same source level training used.
        ``estimate_tau_src`` stays P_φ everywhere else — ``val/codec_bias_db`` compares two
        readings of the *same ruler*, and folding a label into it would destroy what it
        isolates.
        """
        if self.hparams.label_source == "predictor":
            return self.estimate_tau_src(batch["wav"])

        labels: AudioBatch = batch["frame_intensity_db"]
        return leq_aggregate(self.source_frame_db(batch), labels.padding_mask)

    @torch.no_grad()
    def source_frame_db(self, batch: dict) -> Tensor:
        """The calibrated per-frame dB SPL of the source, clamped at ``floor_db``.

        ``(B, T)``.  This is what :meth:`source_tau` aggregates, and what ``silence_db``
        thresholds to decide which frames are non-speech.  Labels only — under
        ``label_source='predictor'`` there is no per-frame ground truth to threshold, which
        is why ``silence_db`` is refused there at construction.
        """
        return batch["frame_intensity_db"].data.squeeze(1).clamp(min=self.hparams.floor_db)

    # ------------------------------------------------------------------
    # The critic, at both resolutions
    # ------------------------------------------------------------------

    def _critic(self, z: AudioBatch, tau: Tensor) -> tuple[Tensor | None, Tensor]:
        """D's verdict on ``(z, τ)`` as ``(per-frame, pooled)``, from ONE forward pass.

        With ``lambda_frame == 0`` the frame scores are not computed and this is exactly
        the pooled call the parent module makes.

        Otherwise the scalar τ is broadcast across frames, which makes
        ``ConditionalDiscriminator`` return ``(B, T)``, and the pooled score is their masked
        mean over time.  That is the *same number* the pooled call returns — ``score`` is
        Linear and the projection is an inner product, so both are affine in φ and commute
        with the mean.  Taking it this way rather than running D twice halves the critic
        forward passes and makes it impossible for the two resolutions to disagree.
        """
        if self.hparams.lambda_frame == 0.0:
            return None, self.cond_disc(z, tau)
        n_frames = z.padding_mask.shape[-1]
        frames = self.cond_disc(z, tau.unsqueeze(1).expand(-1, n_frames))
        return frames, _masked_mean_t(frames, z.padding_mask)

    def _blend(self, pooled: Tensor, frame: Tensor | None) -> Tensor:
        """``(L_pooled + λ·L_frame) / (1 + λ)``.

        Normalised so the magnitude of every adversarial term is invariant to λ: the D/G
        balance was tuned at ``lr_d: 5.0e-5`` against the pooled loss alone, and letting λ
        scale the total would change two things at once.  λ = 0 returns the pooled loss
        untouched.
        """
        if frame is None:
            return pooled
        lam = self.hparams.lambda_frame
        return (pooled + lam * frame) / (1.0 + lam)

    def convert(
        self, z_real: AudioBatch, tau_tgt: Tensor, batch: dict
    ) -> AudioBatch:
        """Apply C_θ at a target intensity — here, simply the converter itself.

        Exists so the evaluation and generation callbacks have one entry point that also
        fits modules deriving C_θ's condition from the source signal.  ``batch`` is
        unused: this module's condition is τ_tgt and nothing else.
        """
        return self.converter(z_real, tau_tgt)

    @torch.no_grad()
    def _conditioning_balance(self, z: AudioBatch, tau: Tensor) -> Tensor:
        """|conditional term| / |unconditional term| in D_ψ's score.

        ``D(z,τ) = ψ(φ(z)) + ⟨V·e(τ), φ(z)⟩/√d``.  Only the second term can push
        C_θ away from the identity map — the first is minimised *by* it, since
        D_ψ's positive class is z_real, which is C_θ's own input.  A small ratio
        means the realness pull toward identity outweighs the conditional push
        toward conversion, whatever the loss weights say.
        """
        phi = self.cond_disc.feature_model.features(z)
        uncond = self.cond_disc.feature_model.score(phi)
        cond = self.cond_disc.conditioner(phi, self.label_scaler.normalise(tau))
        return cond.abs().mean() / uncond.abs().mean().clamp(min=1e-8)

    def _check_alignment(self, batch: dict, z_real: AudioBatch) -> None:
        """Frame labels and conversion latents must live on the same frame grid.

        τ_src is aggregated over the labels' own padding mask, so a mismatch here does not
        produce a shape error — it produces a source level quietly measured over a
        different span of audio than the one being converted, which is worse.  Both counts
        come from the same waveform (``FrameGrid.n_frames`` for the labels, the codec hop
        for the latents), so any disagreement is a padding bug upstream.
        """
        n_labels = batch["frame_intensity_db"].data.shape[-1]
        n_frames = z_real.padding_mask.shape[-1]
        if n_labels != n_frames:
            raise RuntimeError(
                f"the frame labels and the conversion encode disagree on frame count: "
                f"{n_labels} vs {n_frames}. Both are derived from the same waveform on "
                "the same frame grid, so a mismatch means one of them is padding "
                "differently and tau_src is being measured over the wrong span."
            )

    def _intensity_metrics(self, tau_pred: Tensor, tau_tgt: Tensor) -> dict[str, Tensor]:
        """Normalised MSE (comparable to the null baseline) plus raw-dB errors."""
        err = tau_pred - tau_tgt
        return {
            "loss_pred": nn.functional.mse_loss(
                self.label_scaler.normalise(tau_pred),
                self.label_scaler.normalise(tau_tgt),
            ),
            "rmse_pred_db": err.pow(2).mean().sqrt(),
            "me_pred_db": err.mean(),
        }

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def training_step(self, batch: dict, batch_idx: int) -> None:
        opt_g, opt_d = self.optimizers()

        wav_batch: AudioBatch = batch["wav"]
        B = wav_batch.batch_size
        device = wav_batch.data.device

        tau_tgt = self.intensity_drawer.draw(B, device)            # (B,) raw dBSPL

        # Conversion space: raw latents, in distribution for the codec.
        z_real: AudioBatch = self.pipeline.encode(wav_batch, training=True)

        # The source level: calibrated labels, or a second whitened encode read by P_φ.
        # Label extraction either way — no gradient path exists from here to C_θ.
        if self.hparams.label_source == "labels":
            self._check_alignment(batch, z_real)
        tau_src = self.source_tau(batch)                           # (B,) raw dBSPL

        # Conditioning labels for the discriminator's real pairs.
        tau_pos = self.vicinal_sampler.positive(tau_src)
        tau_neg = self.vicinal_sampler.negative(tau_src)

        mask = z_real.padding_mask                                 # (B, T) True = valid

        # Which frames and which chunks the MISMATCH term applies to.  A non-speech frame
        # carries no information about vocal effort, so pairing it with a wrong τ and
        # calling it fake teaches D that the behaviour we want -- leave the pause alone --
        # is a forgery.  Both scales come from the same threshold: frames below it, and
        # chunks whose whole τ_src is below it (the all-silence ones).
        # None here means "every frame / every chunk", i.e. the previous behaviour.
        if self.hparams.silence_db is None:
            mism_frame_mask, mism_seq_mask = mask, None
        else:
            frame_db = self.source_frame_db(batch)                 # (B, T) calibrated dB
            mism_frame_mask = mask & (frame_db > self.hparams.silence_db)
            mism_seq_mask = tau_src > self.hparams.silence_db

        # Forward conversion (graph retained for the G-step).
        z_fake: AudioBatch = self.converter(z_real, tau_tgt)

        # ==============================================================
        # D-step: update D_ψ
        # ==============================================================
        self.toggle_optimizer(opt_d)
        opt_d.zero_grad()

        d_real_f, d_real = self._critic(z_real, tau_pos)            # matched
        d_fake_f, d_fake = self._critic(z_fake.detach(), tau_tgt)   # generated
        d_mism_f, d_mism = self._critic(z_real, tau_neg)            # mismatched

        # Each term is the pooled loss blended with its per-frame form.  The two differ by
        # Var_t(s_t) -- a chunk convincing on average while one window is a scribble is
        # free under the pooled loss alone.
        loss_d_real = self._blend(
            lsgan_loss(d_real, REAL_TARGET),
            None if d_real_f is None else masked_lsgan_loss(d_real_f, REAL_TARGET, mask),
        )
        loss_d_fake = self._blend(
            lsgan_loss(d_fake, FAKE_TARGET),
            None if d_fake_f is None else masked_lsgan_loss(d_fake_f, FAKE_TARGET, mask),
        )
        loss_d_mism = self._blend(
            masked_lsgan_loss(d_mism, FAKE_TARGET, mism_seq_mask),
            None if d_mism_f is None
            else masked_lsgan_loss(d_mism_f, FAKE_TARGET, mism_frame_mask),
        )

        # Positive and negative mass are balanced 0.5/0.5; lambda_mismatch splits
        # the negative half between generated and mismatched-real negatives.
        lam_mm = self.hparams.lambda_mismatch
        loss_d = 0.5 * loss_d_real + 0.5 * (
            (loss_d_fake + lam_mm * loss_d_mism) / (1.0 + lam_mm)
        )

        self.manual_backward(loss_d)
        self.clip_gradients(opt_d, gradient_clip_val=self.hparams.gradient_clip_val)
        opt_d.step()
        self.untoggle_optimizer(opt_d)

        # Diagnostics — no gradient, no effect on either objective.
        with torch.no_grad():
            z_fake_d = z_fake.detach()
            # Does D_ψ's τ-knowledge reach the *fake* branch? d_cond_gap only ever
            # measured it on real latents, where it can be large while the fakes
            # are scored on the realness axis alone. The mismatched τ here is drawn
            # away from τ_tgt — the τ this fake was built for — not from τ_src.
            d_fake_mism = self._critic(
                z_fake_d, self.vicinal_sampler.negative(tau_tgt)
            )[1]
            cond_ratio_real = self._conditioning_balance(z_real, tau_pos)
            cond_ratio_fake = self._conditioning_balance(z_fake_d, tau_tgt)
            # Within-chunk spread of the critic's verdicts on real audio: the square root
            # of exactly the term the per-frame loss adds to the pooled one.  ~0 means the
            # frame loss is doing nothing the pooled loss was not already doing.
            frame_sd = (
                torch.zeros((), device=device) if d_real_f is None
                else _masked_mean_t(
                    (d_real_f - _masked_mean_t(d_real_f, mask).unsqueeze(1)).pow(2), mask
                ).mean().sqrt()
            )

        # ==============================================================
        # G-step: update C_θ only.
        # toggle_optimizer sets D_ψ params to requires_grad=False; gradients
        # still flow *through* its operations to z_fake → C_θ.
        # ==============================================================
        self.toggle_optimizer(opt_g)
        opt_g.zero_grad()

        # The conditional critic is now the only intensity signal in the objective.
        g_fake_f, g_fake = self._critic(z_fake, tau_tgt)
        loss_fake = self._blend(
            lsgan_loss(g_fake, REAL_TARGET),
            None if g_fake_f is None else masked_lsgan_loss(g_fake_f, REAL_TARGET, mask),
        )

        # Content preservation: round trip back to the source intensity.
        loss_cycle, _ = cycle_consistency_loss(
            self.converter, z_fake, tau_src, z_real
        )

        loss_g = (
            self.hparams.lambda_fake  * loss_fake
            + self.hparams.lambda_cycle * loss_cycle
        )

        # Optional identity anchor: C(z, τ_src) = z. Costs one forward pass.
        loss_id = torch.zeros((), device=device)
        if self.hparams.lambda_id > 0.0:
            loss_id = identity_anchor_loss(self.converter, z_real, tau_src)
            loss_g = loss_g + self.hparams.lambda_id * loss_id

        self.manual_backward(loss_g)
        self.clip_gradients(opt_g, gradient_clip_val=self.hparams.gradient_clip_val)
        opt_g.step()
        self.untoggle_optimizer(opt_g)

        self.log_dict({
            "train/loss_g":       loss_g,
            "train/loss_d":       loss_d,
            "train/loss_fake":    loss_fake,
            "train/loss_cycle":   loss_cycle,
            "train/loss_id":      loss_id,
            "train/loss_d_real":  loss_d_real,
            "train/loss_d_fake":  loss_d_fake,
            "train/loss_d_mism":  loss_d_mism,
            # Raw critic scores — the conditioning health checks.
            "train/d_real":       d_real.mean(),
            "train/d_fake":       d_fake.mean(),
            "train/d_mism":       d_mism.mean(),
            # ≈ 0 means D_ψ ignores τ and has degenerated to an unconditional critic.
            "train/d_cond_gap":   d_real.mean() - d_mism.mean(),
            # Same question asked of the fake branch. If this is ≈ 0 while
            # d_cond_gap is large, D_ψ knows τ but does not use it to judge
            # fakes — so C_θ receives no per-τ direction and cannot learn to convert.
            "train/d_cond_gap_fake": d_fake.mean() - d_fake_mism.mean(),
            # Share of D_ψ's score carried by the conditional term. The only term
            # in the whole objective that opposes C_θ = identity.
            "train/cond_ratio_real": cond_ratio_real,
            "train/cond_ratio_fake": cond_ratio_fake,
            # Generator progress against a negative at comparable τ. More honest
            # than loss_fake, whose +1 target may be far out of reach.
            "train/d_z_gap":      d_fake.mean() - d_mism.mean(),
            # Does the real τ_src distribution cover the τ_tgt being requested?
            # The calibrated truth under 'labels', P_φ's estimate under 'predictor'.
            "train/tau_src_db":   tau_src.mean(),
            # sqrt of the term the per-frame loss adds. 0 when lambda_frame is 0.
            "train/d_real_frame_sd": frame_sd,
            # Share of frames / of chunks the mismatch term still applies to. If either
            # collapses toward 0, silence_db is too high and the negatives have quietly
            # gone away -- and they are the only thing teaching D_ψ to use τ at all.
            "train/mism_frame_frac": (
                mism_frame_mask.float().sum() / mask.float().sum().clamp(min=1.0)
            ),
            "train/mism_seq_frac": (
                torch.ones((), device=device) if mism_seq_mask is None
                else mism_seq_mask.float().mean()
            ),
        }, prog_bar=False, sync_dist=True)

    # ------------------------------------------------------------------
    # Validation step — intensity measured through the decoder
    # ------------------------------------------------------------------

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        wav_batch: AudioBatch = batch["wav"]
        B = wav_batch.batch_size
        device = wav_batch.data.device

        tau_tgt = self.intensity_drawer.draw(B, device)

        z_real = self.pipeline.encode(wav_batch, training=False)
        tau_src = self.source_tau(batch)
        z_fake = self.converter(z_real, tau_tgt)

        # decode → whiten → re-encode → P_φ.  Costs one decode per val batch and
        # is the reason this metric cannot be gamed: no gradient reaches it.
        tau_pred = self.measure_intensity(z_fake)

        # The same measurement applied to the *unconverted* latent: how much dB
        # the codec round trip alone moves P_φ's reading.  τ_src is read from the
        # original waveform while τ_pred is read after decode, so any systematic
        # offset between them lands in val/rmse_pred_db without belonging to the
        # converter.  Subtract this before concluding the converter under-shoots.
        tau_roundtrip = self.measure_intensity(z_real)

        loss_cycle, _ = cycle_consistency_loss(
            self.converter, z_fake, tau_src, z_real
        )
        # Reported for comparability with the earlier runs' loss_loc only — it is
        # not in the objective and its scale differs from the whitened runs.
        loss_loc = masked_mse(z_fake.data, z_real.data, z_real.padding_mask)

        # val/codec_bias_* exists to isolate what decode → whiten → re-encode does to
        # P_φ's reading, so BOTH sides of the difference have to be P_φ readings.  Under
        # label_source='labels' τ_src above is the calibrated truth, which would fold
        # label-vs-P_φ disagreement into a metric that means nothing once it contains
        # any.  Take the extra reading and keep the two questions apart; under
        # 'predictor' this is τ_src itself, so the metric is unchanged.
        tau_src_phi = (
            self.estimate_tau_src(wav_batch)
            if self.hparams.label_source == "labels" else tau_src
        )
        codec_bias = tau_roundtrip - tau_src_phi

        metrics = self._intensity_metrics(tau_pred, tau_tgt)
        self.log_dict({
            "val/loss_pred":     metrics["loss_pred"],
            "val/rmse_pred_db":  metrics["rmse_pred_db"],
            "val/me_pred_db":    metrics["me_pred_db"],
            "val/loss_cycle":    loss_cycle,
            "val/loss_loc":      loss_loc,
            # Measurement floor, not a converter metric — see above.
            "val/codec_bias_db": codec_bias.mean(),
            # Spread of that floor WITHIN a batch.  The val loader is unshuffled
            # over metadata ordered by signal_path, so a batch is mostly
            # consecutive segments of a few recordings: read this as
            # within-recording spread, and val/roundtrip_bias_sd_db (eval
            # callback, random subset) as the across-recording spread.  The two
            # together say whether the measurement offset is a property of the
            # recording or of the individual segment.
            "val/codec_bias_sd_db": codec_bias.std(),
            # The source level the condition was built from: the calibrated truth under
            # 'labels', P_φ's estimate under 'predictor'.
            "val/tau_src_db":    tau_src.mean(),
        }, prog_bar=True, sync_dist=True)

        # P_φ's error against the ground truth, on held-out audio, in dB.  Only
        # computable when the labels are the source — when P_φ *is* the source this is
        # identically zero and says nothing.  It costs one extra P_φ pass per validation
        # batch and nothing at training time, and it is the measurement that decides
        # whether a per-utterance conversion offset belongs to the ruler or to the
        # converter — a question the P_φ-conditioned runs could not ask, since there was
        # no independent reading to compare against.
        if self.hparams.label_source == "labels":
            self.log(
                "val/ruler_bias_db",
                (tau_src_phi - tau_src).mean(),
                prog_bar=False, sync_dist=True,
            )

    # ------------------------------------------------------------------
    # Test step — cross-predictor evaluation on the same decoded audio
    # ------------------------------------------------------------------

    def test_step(self, batch: dict, batch_idx: int) -> None:
        wav_batch: AudioBatch = batch["wav"]
        B = wav_batch.batch_size
        device = wav_batch.data.device

        tau_tgt = self.intensity_drawer.draw(B, device)

        z_real = self.pipeline.encode(wav_batch, training=False)
        z_fake = self.converter(z_real, tau_tgt)

        # One decode, two independent readings of the same waveform.
        wav_fake: AudioBatch = self.pipeline.extractor.decode(z_fake)

        logs = {
            "test/loss_loc": masked_mse(
                z_fake.data, z_real.data, z_real.padding_mask
            ),
        }

        # (a) The training-time predictor, on decoded audio. Directly comparable
        #     to val/loss_pred — same measurement, held-out data.
        nac_metrics = self._intensity_metrics(
            self.estimate_tau_src(wav_fake), tau_tgt
        )
        logs["test/loss_pred_nac"]    = nac_metrics["loss_pred"]
        logs["test/rmse_pred_nac_db"] = nac_metrics["rmse_pred_db"]

        # (b) The architecturally independent predictor. Together with (a) on the
        #     same audio, the gap between them is predictor disagreement alone —
        #     no decode, chunking or data difference confounds it.
        if self.test_pipeline is not None and self.test_predictor is not None:
            z_test = self.test_pipeline.encode(wav_fake, training=False)
            tau_pred_test = leq_aggregate(
                self.test_predictor(z_test), z_test.padding_mask
            )
            test_metrics = self._intensity_metrics(tau_pred_test, tau_tgt)
            logs["test/loss_pred"]    = test_metrics["loss_pred"]
            logs["test/rmse_pred_db"] = test_metrics["rmse_pred_db"]
            logs["test/me_pred_db"]   = test_metrics["me_pred_db"]

        self.log_dict(logs, prog_bar=True, sync_dist=True)

    # ------------------------------------------------------------------
    # Optimisers
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        """Adam with β₁ = 0 and no weight decay for both players (ViT-GAN).

        Fixed, not configurable: β₁ = 0 drops momentum because each player's
        objective moves under it every step, and decay would pull D_ψ toward the
        constant critic when spectral norm is already the constraint it needs.
        """
        opt_g = torch.optim.AdamW(
            self.converter.parameters(),
            lr=self.hparams.lr_g,
            betas=(0.0, 0.99),
            weight_decay=0.0,
        )
        opt_d = torch.optim.AdamW(
            self.cond_disc.parameters(),
            lr=self.hparams.lr_d,
            betas=(0.0, 0.99),
            weight_decay=0.0,
        )
        return [opt_g, opt_d]

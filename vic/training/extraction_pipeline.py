"""ExtractionPipeline: reusable deterministic feature extraction pipeline.

Bundles three stages that appear in every training script and module:

    1. (optional) STFT-domain spectral whitening on the raw waveform
    2. Feature extraction via a frozen FeatureExtractor
    3. (optional) Feature-space normalisation (thresholding, utterance CMN,
       per-band CMN)

Training-only *stochastic* augmentations (spectral coloring, denoiser, codec
resynthesis, noise) are NOT part of the pipeline — they remain in each
LightningModule's ``training_step`` because they vary between experiments.

Using a pipeline object instead of passing individual flags to every module:
  - removes the 8+ normalisation/whitening flags from LightningModule __init__
  - makes teacher / student configurations explicit and independently composable
  - supports different extractor types for teacher and student in distillation
"""
from __future__ import annotations

import warnings

import torch
import torch.nn as nn

from vic.core import FeatureExtractor, FrameGrid
from vic.data.audio_batch import AudioBatch
from vic.data.augmentation import spectral_whitening, threshold_spectrum
from vic.data.transforms import resample_batch
from vic.features.normalization import utterance_normalize, utterance_normalize_per_band


class ExtractionPipeline(nn.Module):
    """Complete deterministic feature extraction pipeline (always frozen).

    Parameters
    ----------
    extractor                   : frozen FeatureExtractor (NAC, MelSpec, Wav2Vec2 …).
    whitening                   : apply STFT-domain per-frequency mean subtraction
                                  to the waveform before encoding.  Removes recording-
                                  chain spectral coloring for non-spectral encoders.
    whitening_n_fft             : FFT window size for spectral whitening.
    whitening_hop_length        : hop size for spectral whitening.
    whitening_train_only        : if True, whitening is skipped at val/test (use to
                                  assess coloration invariance without whitening at
                                  inference time).
    spec_threshold              : zero encoded feature values below this scalar (all
                                  splits).  Calibrated in raw feature space so the
                                  threshold transfers across recordings.
    normalize_features          : subtract the global per-utterance mean from all
                                  feature frames (all splits).  Removes crest-factor
                                  proxy while preserving spectral tilt.
    normalize_features_per_band : subtract the per-band per-utterance mean from
                                  feature frames (all splits).  Removes spectral
                                  coloring for spectral extractors (MelSpec, STFT).
                                  Mutually exclusive with ``normalize_features`` in
                                  practice; if both are True, per-band is applied last.
    """

    def __init__(
        self,
        extractor: FeatureExtractor,
        whitening: bool = False,
        whitening_n_fft: int = 1024,
        whitening_hop_length: int = 320,
        whitening_train_only: bool = False,
        spec_threshold: float | None = None,
        normalize_features: bool = False,
        normalize_features_per_band: bool = False,
    ):
        super().__init__()
        self.extractor = extractor
        self.whitening = whitening
        self.whitening_n_fft = whitening_n_fft
        self.whitening_hop_length = whitening_hop_length
        self.whitening_train_only = whitening_train_only
        self.spec_threshold = spec_threshold
        self.normalize_features = normalize_features
        self.normalize_features_per_band = normalize_features_per_band

        for p in self.extractor.parameters():
            p.requires_grad_(False)

    def train(self, mode: bool = True) -> "ExtractionPipeline":
        # Extractor must stay in eval mode so its batch-norm / dropout statistics
        # are never updated, regardless of what the Lightning Trainer does.
        super().train(mode)
        self.extractor.eval()
        return self

    # ------------------------------------------------------------------
    # Passthrough properties — callers can use the pipeline as a drop-in
    # for a bare FeatureExtractor when they only need these attributes.
    # ------------------------------------------------------------------

    @property
    def sample_rate(self) -> int:
        return self.extractor.sample_rate

    @property
    def latent_dim(self) -> int:
        return self.extractor.latent_dim

    @property
    def frame_grid(self) -> FrameGrid:
        return self.extractor.frame_grid

    # ------------------------------------------------------------------

    def encode(self, wav_batch: AudioBatch, training: bool = False) -> AudioBatch:
        """Run the full pipeline: resample → whiten → encode → normalise.

        Parameters
        ----------
        wav_batch : waveform batch to process.
        training  : set True when called from a training step so that
                    ``whitening_train_only`` is respected correctly.
        """
        # A pipeline hands its extractor the rate that extractor was trained at, whatever
        # rate the batch arrives at.  This is a no-op for every 16 kHz pipeline, and it is
        # what lets the 22.05 kHz mel vocoder be the conversion domain while P_φ's
        # measurement pipeline stays at 16 kHz: ``measure_intensity`` decodes at the
        # conversion rate and then calls the label pipeline, which lands here.  Before the
        # whitening, whose n_fft/hop are chosen for the extractor's rate.
        wav_batch = resample_batch(wav_batch, self.extractor.sample_rate)
        if self.whitening and (training or not self.whitening_train_only):
            wav_batch = spectral_whitening(
                wav_batch,
                n_fft=self.whitening_n_fft,
                hop_length=self.whitening_hop_length,
            )
        with torch.no_grad():
            z = self.extractor.encode(wav_batch)
        if self.spec_threshold is not None:
            z = threshold_spectrum(z, self.spec_threshold)
        if self.normalize_features:
            z = utterance_normalize(z)
        if self.normalize_features_per_band:
            z = utterance_normalize_per_band(z)
        return z


# ---------------------------------------------------------------------------
# Factory helpers — used by training scripts to build pipelines from YAML.
# ---------------------------------------------------------------------------

def build_extractor(config: dict) -> nn.Module:
    """Instantiate a FeatureExtractor from ``config["extractor"]``.

    Kept as a separate function so scripts that need a bare extractor (e.g.
    to build a codec-resynthesis augmentation) can call it directly without
    wrapping it in a pipeline.
    """
    ec = config["extractor"]
    kind = ec["type"]

    if kind == "nac":
        from vic.codec.speechtokenizer import SpeechTokenizerCodec
        return SpeechTokenizerCodec(
            config_path=ec["config_path"],
            ckpt_path=ec["ckpt_path"],
        )
    elif kind == "melspectrogram":
        from vic.encoders.mel import MelSpectrogramExtractor
        return MelSpectrogramExtractor(
            sample_rate=ec.get("sample_rate", 16000),
            n_mels=ec.get("n_mels", 128),
            n_fft=ec.get("n_fft", 1024),
            win_length=ec.get("win_length", 400),
            hop_length=ec.get("hop_length", 320),
            log=ec.get("log", True),
        )
    elif kind == "spectrogram":
        from vic.encoders.spectrogram import SpectrogramExtractor
        return SpectrogramExtractor(
            sample_rate=ec.get("sample_rate", 16000),
            n_fft=ec.get("n_fft", 1024),
            win_length=ec.get("win_length", None),
            hop_length=ec.get("hop_length", 320),
            power=ec.get("power", 2.0),
            log=ec.get("log", True),
        )
    elif kind == "wav2vec2":
        from vic.encoders.wav2vec2 import Wav2Vec2Extractor
        return Wav2Vec2Extractor(
            model_name=ec.get("model_name", "facebook/wav2vec2-base"),
            freeze=ec.get("freeze", True),
            layer=ec.get("layer", -1),
        )
    # The two invertible non-NAC domains.  Both satisfy `AudioCodec`, not just
    # `FeatureExtractor`, so they can be a *conversion* space and not only a predictor
    # front end -- which is what separates `mel_vocoder` from `melspectrogram` above.
    # See their module docstrings for provenance, citations and why these checkpoints.
    elif kind == "mel_vocoder":
        from vic.codec.mel_vocoder import MelVocoderCodec
        return MelVocoderCodec(
            repo_path=ec["repo_path"],
            model_dir=ec["model_dir"],
            backend=ec.get("backend", "bigvgan"),
        )
    elif kind == "wavlm_hifigan":
        from vic.codec.wavlm_hifigan import WavLMHiFiGANCodec
        return WavLMHiFiGANCodec(
            repo_path=ec["repo_path"],
            wavlm_ckpt=ec["wavlm_ckpt"],
            hifigan_ckpt=ec["hifigan_ckpt"],
            config_path=ec.get("config_path"),
            layer=ec.get("layer", 6),
        )
    else:
        raise ValueError(f"Unknown extractor type: {kind!r}")


def build_pipeline(config: dict, extractor: nn.Module | None = None) -> ExtractionPipeline:
    """Build an :class:`ExtractionPipeline` from a config dict.

    Parameters
    ----------
    config    : config dict with an ``extractor`` block (see layout below).
    extractor : optional pre-built extractor to wrap instead of instantiating a
                new one from ``config["extractor"]``.  Use this to build two
                pipelines that differ only in their pre/post-processing while
                **sharing one frozen model** — e.g. the converter's raw-latent
                pipeline and the predictor's whitened-latent pipeline, which are
                the same NAC weights applied to differently pre-processed audio.
                Loading the codec twice would double its GPU footprint for no
                reason.  When passed, the extractor-specific keys in
                ``config["extractor"]`` (``type``, ``ckpt_path``, …) are ignored;
                the whitening and normalisation keys still apply.

    Config layout (same as existing ``train_predictor.py`` YAML format)::

        extractor:
          type: nac | melspectrogram | spectrogram | wav2vec2
                | mel_vocoder | wavlm_hifigan
          # extractor-specific params ...
          whitening: false
          whitening_n_fft: 1024
          whitening_hop_length: 320
          whitening_train_only: false
          normalize_features: false
          normalize_features_per_band: false

        training:           # optional — only needed if spec_threshold is set
          augmentation:
            spec_threshold: null

    For the teacher pipeline in a distillation config, pass ``config["teacher"]``
    which mirrors this layout under its own ``extractor`` sub-key.
    """
    ec = config["extractor"]
    spec_threshold = (
        config.get("training", {})
              .get("augmentation", {})
              .get("spec_threshold", None)
    )
    return ExtractionPipeline(
        extractor=build_extractor(config) if extractor is None else extractor,
        whitening=ec.get("whitening", False),
        whitening_n_fft=ec.get("whitening_n_fft", 1024),
        whitening_hop_length=ec.get("whitening_hop_length", 320),
        whitening_train_only=ec.get("whitening_train_only", False),
        spec_threshold=spec_threshold,
        normalize_features=ec.get("normalize_features", False),
        normalize_features_per_band=ec.get("normalize_features_per_band", False),
    )

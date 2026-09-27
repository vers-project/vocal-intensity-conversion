"""WavLM conversion domain: WavLM-Large layer 6 encoder + kNN-VC's HiFi-GAN decoder.

Third of the three conversion domains, alongside SpeechTokenizer latents
(``vic/codec/speechtokenizer.py``) and log-mel (``vic/codec/mel_vocoder.py``).  Satisfies
the same ``AudioCodec`` protocol, so switching to it is a change to the ``extractor``
block of the YAML and nothing else.

Provenance
----------
Encoder, WavLM-Large:

    Sanyuan Chen, Chengyi Wang, Zhengyang Chen, Yu Wu, Shujie Liu, Zhuo Chen,
    Jinyu Li, Naoyuki Kanda, Takuya Yoshioka, Xiong Xiao, Jian Wu, Long Zhou,
    Shuo Ren, Yanmin Qian, Yao Qian, Jian Wu, Michael Zeng, Xiangzhan Yu, Furu Wei.
    "WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing."
    IEEE JSTSP 16(6), 2022.  https://arxiv.org/abs/2110.13900

    Code  https://github.com/microsoft/unilm/tree/master/wavlm

Decoder, the HiFi-GAN trained to vocode WavLM-Large layer 6 features:

    Matthew Baas, Benjamin van Niekerk, Herman Kamper.
    "Voice Conversion With Just Nearest Neighbors."  Interspeech 2023.
    https://arxiv.org/abs/2305.18975

    Code and weights  https://github.com/bshall/knn-vc                (MIT)
    Encoder weights mirrored on the same release, so both come from one place:
      https://github.com/bshall/knn-vc/releases/download/v0.1/WavLM-Large.pt
      https://github.com/bshall/knn-vc/releases/download/v0.1/prematch_g_02500000.pt
      https://github.com/bshall/knn-vc/releases/download/v0.1/g_02500000.pt

Why this encoder/decoder pair
-----------------------------
*Which of kNN-VC's two generators.*  The repo ships both at 2.5 M updates, same HiFi-GAN V1
architecture, same ``config_v1_wavlm.json``, same LibriSpeech train-clean-100.  The only
difference is the features each was trained to vocode: ``g_02500000.pt`` on layer-6
features as the encoder emits them, ``prematch_g_02500000.pt`` on *prematched* features --
each training frame replaced by the mean of its top-4 nearest neighbours drawn from other
utterances of the same speaker.

**We use the plain one.**  Prematching exists to match kNN-VC's own inference distribution,
where the vocoder is fed kNN outputs: "we pick these features from various points in time
with different phonetic contexts, leading to inconsistencies between adjacent frames"
(Baas et al., §3.4).  Our decoder input is not that.  C_theta emits a transformed version
of raw encoder output, and the identity and cycle anchors hold it near the raw manifold --
at tau_tgt == tau_src it *is* the raw encoder output.  So the plain checkpoint is the one
whose training inputs we actually produce.

The counter-argument, which is real and was why the prematched weights were used until
2026-09-12: prematched training is a form of input augmentation, so that checkpoint is the
more tolerant of the off-manifold sequences a converter produces once tau_tgt /= tau_src.
It is a genuine trade and neither paper settles it.  What decided it in practice was
listening -- background noise under quiet speech and artefacts on loud speech.  Note that
neither symptom is likely to be *fixed* by this swap: both checkpoints share the 100 h of
clean read speech and the level-blind input described below, so the swap tests the
prematch hypothesis rather than curing the symptoms.  Swapping back is one config key.

*Why not something newer.*  vec2wav 2.0 and Amphion's Vevo are more recent but have thin
third-party track records; FreeVC's decoder sits behind a content bottleneck and a speaker
embedding, so it would confound the representation with its own information bottleneck
rather than isolating it.

*Why layer 6 is not a free parameter.*  The decoder was trained on layer 6 (kNN-VC's
``SPEAKER_INFORMATION_LAYER``) and on nothing else, so ``layer`` exists to document the
coupling, not to sweep: any other value produces features the checkpoint has never seen.

What this domain can and cannot represent
-----------------------------------------
WavLM-Large is close to level-blind.  Its feature extractor runs in ``layer_norm`` mode,
which normalises every frame across channels, and the checkpoint's ``cfg.normalize`` is
True, meaning upstream's documented usage layer-norms the waveform as well.  kNN-VC does
*not* apply that waveform normalisation, and neither do we -- see "Exactness" below -- but
the per-frame normalisation inside the extractor is unavoidable.  Consequences:

  * Conversion in this domain can only change production character, never gain.  The
    converter cannot reach a target level by scaling, and the critic cannot judge realness
    by loudness.  For a project whose stated goal is predicting and controlling intensity
    from production cues rather than recording-level artefacts, that is the point of this
    arm, not a defect -- it enforces by construction what the other two arms merely
    encourage.
  * Absolute output level is whatever the vocoder emits.  kNN-VC itself re-imposes
    loudness after synthesis (``torchaudio.functional.gain``), which is the authors'
    own admission that the output level carries no information.  Any absolute SPL has to
    be applied downstream of ``decode``.
  * The measurement path is unaffected: ``estimate_tau_src`` whitens and re-encodes
    through a Wav2Vec2 P_phi that z-normalises its input, so the ruler was already reading
    production cues rather than gain.

Known limitation: this HiFi-GAN was trained on LibriSpeech train-clean-100 only -- clean
read speech at conversational effort.  Shouted AVID / Mixer 7 material is out of
distribution for it, which the resynthesis-ceiling probe is there to quantify.  Upstream
ships training code if fine-tuning turns out to be necessary.

Exactness
---------
Both halves are upstream's own code, reached by putting the kNN-VC checkout on
``sys.path`` (see ``vic/codec/_vendor.py``), so the features handed to the decoder are
produced by the same call kNN-VC makes.  Two deliberate deviations, both documented at
their call site in ``encode``:

  1. **No waveform layer-norm.**  Upstream's README layer-norms the input when
     ``cfg.normalize`` is True; kNN-VC does not, so the checkpoint was trained on features
     extracted from un-normalised audio.  Matching the decoder beats matching the README,
     and this is not exposed as a flag -- a wrong setting here degrades output silently.
  2. **A 200-sample left pad**, so the frame grid is the one the rest of the pipeline
     assumes.  See below.

Frame-grid alignment
--------------------
WavLM's convolutional feature extractor has total stride 320 and receptive field 400
samples, so an unpadded signal of ``T`` samples yields ``T // 320 - 1`` frames, with frame
*i* centred at sample ``320i + 200``.  Both facts are wrong for this codebase: the frame
labels are built at ``T // 320`` (``FrameGrid.n_frames``), and
``ConverterCGANv2LabelsModule._check_alignment`` raises on the resulting off-by-one rather
than silently measuring tau_src over the wrong span.

Reflect-padding 200 samples on the left fixes both at once: the count becomes
``floor((T + 200 - 400) / 320) + 1 == T // 320`` for ``T`` a multiple of 320, and frame
*i* then analyses ``[320i - 200, 320i + 200)``, centred at ``320i`` -- exactly what a
non-causal ``FrameGrid`` promises.  A uniform shift of the analysis grid is not a
distribution shift for the decoder (its training crops were at arbitrary offsets), so this
buys alignment for free.  Verified against upstream's own ``ConvFeatureExtractionModel``
at 640, 16000, 32000 and 100160 samples.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from vic.codec._vendor import add_repo_to_path
from vic.core import FrameGrid
from vic.data.audio_batch import AudioBatch

# kNN-VC's SPEAKER_INFORMATION_LAYER.  The decoder checkpoint is tied to it.
WAVLM_LAYER = 6

# The convolutional receptive field, in input samples (kernels [10,3,3,3,3,2,2] over
# strides [5,2,2,2,2,2,2]).  Half of it is the left pad that centres frame i at i*hop.
_CONV_RECEPTIVE_FIELD = 400


class WavLMHiFiGANCodec(nn.Module):
    """Frozen WavLM-Large layer-6 encoder + kNN-VC HiFi-GAN, satisfying ``AudioCodec``.

    Parameters
    ----------
    repo_path   : path to a clone of https://github.com/bshall/knn-vc -- the repository
                  root, i.e. the directory containing ``wavlm/`` and ``hifigan/``.
    wavlm_ckpt  : path to ``WavLM-Large.pt``.
    hifigan_ckpt: path to the HiFi-GAN generator checkpoint.  Pass the plain weights
                  (``g_02500000.pt``); ``prematch_g_02500000.pt`` loads the prematched
                  variant, which the module docstring explains we no longer use.  Either
                  is accepted -- they share an architecture and a config.
    config_path : path to the HiFi-GAN config JSON.  Defaults to the repo's own
                  ``hifigan/config_v1_wavlm.json``, which is the config the released
                  weights were trained with; override only to load a fine-tune.
    layer       : WavLM layer to read.  Fixed at 6 by the decoder checkpoint; see the
                  module docstring.
    """

    def __init__(
        self,
        repo_path: str | Path,
        wavlm_ckpt: str | Path,
        hifigan_ckpt: str | Path,
        config_path: str | Path | None = None,
        layer: int = WAVLM_LAYER,
    ):
        super().__init__()
        # Before the repo resolution and before 1.2 GB of encoder: a config typo should
        # report the typo, not whatever the next step happens to fail on.
        if layer != WAVLM_LAYER:
            raise ValueError(
                f"layer={layer} but the kNN-VC HiFi-GAN was trained exclusively on WavLM "
                f"layer {WAVLM_LAYER} features. Any other layer feeds the decoder a "
                "representation it has never seen; there is no checkpoint for it."
            )
        self._layer = layer

        repo = add_repo_to_path(repo_path, name="kNN-VC", expect=["wavlm", "hifigan"])

        from hifigan.models import Generator                # noqa: PLC0415
        from hifigan.utils import AttrDict                   # noqa: PLC0415
        from wavlm.WavLM import WavLM, WavLMConfig           # noqa: PLC0415

        # --- encoder ---
        # weights_only=False, unlike the BigVGAN load: a fairseq-lineage checkpoint stores
        # its own config alongside the tensors, and that dict is not restricted to the
        # types the safe loader admits.
        ckpt = torch.load(str(wavlm_ckpt), map_location="cpu", weights_only=False)
        cfg = WavLMConfig(ckpt["cfg"])
        wavlm = WavLM(cfg)
        wavlm.load_state_dict(ckpt["model"])
        wavlm.eval()
        for p in wavlm.parameters():
            p.requires_grad_(False)
        self.wavlm = wavlm
        self._cfg = cfg

        # --- decoder ---
        config_path = Path(config_path) if config_path else repo / "hifigan" / "config_v1_wavlm.json"
        with open(config_path) as f:
            h = AttrDict(json.load(f))
        generator = Generator(h)
        state = torch.load(str(hifigan_ckpt), map_location="cpu", weights_only=False)
        generator.load_state_dict(state["generator"])
        generator.eval()
        generator.remove_weight_norm()
        for p in generator.parameters():
            p.requires_grad_(False)
        self.generator = generator
        self._h = h

        # The two checkpoints are downloaded separately and there is nothing in either
        # file naming the other, so the pairing is checked here: a WavLM whose width
        # disagrees with the decoder's input width means the wrong pair was fetched.
        if int(h.hubert_dim) != int(cfg.encoder_embed_dim):
            raise ValueError(
                f"checkpoint mismatch: the HiFi-GAN expects {h.hubert_dim}-d features "
                f"(hubert_dim) but this WavLM emits {cfg.encoder_embed_dim}-d "
                f"(encoder_embed_dim). Check that wavlm_ckpt is WavLM-*Large*.pt and that "
                f"config_path matches hifigan_ckpt."
            )

        hop = 1
        for r in h.upsample_rates:
            hop *= int(r)
        self._frame_grid = FrameGrid(
            hop_size=hop,
            sample_rate=int(h.sampling_rate),
            # Non-causal, and true after the left pad below: frame i is centred at i*hop.
            causal=False,
        )
        self._left_pad = _CONV_RECEPTIVE_FIELD // 2

    # ------------------------------------------------------------------
    # AudioCodec protocol properties
    # ------------------------------------------------------------------

    @property
    def sample_rate(self) -> int:
        return int(self._h.sampling_rate)

    @property
    def latent_dim(self) -> int:
        return int(self._h.hubert_dim)

    @property
    def frame_grid(self) -> FrameGrid:
        return self._frame_grid

    # ------------------------------------------------------------------
    # Encode / decode
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode(self, batch: AudioBatch) -> AudioBatch:
        """Encode audio to WavLM layer-6 frames.

        Parameters
        ----------
        batch : AudioBatch with ``data`` of shape (B, 1, T_samples) at 16 kHz.

        Returns
        -------
        AudioBatch of shape (B, ``latent_dim``, T_frames) with
        ``T_frames = T_samples // hop_size``.
        """
        if batch.sample_rate != self.sample_rate:
            raise ValueError(
                f"{type(self).__name__} expects {self.sample_rate} Hz audio but got "
                f"{batch.sample_rate}. WavLM is a 16 kHz model; resample upstream."
            )
        wav = batch.BCT
        if wav.shape[1] != 1:
            raise ValueError(f"expected mono (B, 1, T), got {tuple(wav.shape)}")
        if wav.shape[-1] <= self._left_pad:
            raise ValueError(
                f"input is {wav.shape[-1]} samples; the frame-alignment pad reflects "
                f"{self._left_pad} samples and so needs more than that."
            )

        # No layer_norm on the waveform: the decoder was trained on features extracted
        # from un-normalised audio (deviation 1 in the module docstring).
        wav = wav.squeeze(1)                                        # (B, T)
        wav = F.pad(wav.unsqueeze(1), (self._left_pad, 0), mode="reflect").squeeze(1)

        # Sample-level mask over the padded signal, in fairseq's convention (True =
        # padding).  kNN-VC passes None because it vocodes one utterance at a time; for an
        # unpadded batch an all-False mask is identical, and for a padded one it is the
        # only correct choice.
        T_pad = wav.shape[-1]
        valid_to = (batch.lengths.to(wav.device) + self._left_pad).unsqueeze(1)
        padding_mask = torch.arange(T_pad, device=wav.device).unsqueeze(0) >= valid_to

        feats, _ = self.wavlm.extract_features(
            wav,
            padding_mask=padding_mask,
            output_layer=self._layer,
            ret_layer_results=False,
        )                                                           # (B, T_frames, D)

        hop = self._frame_grid.hop_size
        expected = batch.max_length // hop
        if feats.shape[1] != expected:
            raise RuntimeError(
                f"WavLM returned {feats.shape[1]} frames for {batch.max_length} samples; "
                f"the frame grid says {expected}. The left pad is calibrated for inputs "
                f"that are a multiple of {hop} samples -- check that the dataset's "
                f"pad_to_multiple is using this codec's frame_grid."
            )
        frame_lengths = batch.lengths // hop
        return batch.as_frames(feats.transpose(1, 2), frame_lengths)

    @torch.no_grad()
    def decode(self, batch: AudioBatch) -> AudioBatch:
        """Synthesise a waveform from WavLM layer-6 frames.

        Parameters
        ----------
        batch : AudioBatch of shape (B, ``latent_dim``, T_frames).

        Returns
        -------
        AudioBatch of shape (B, 1, T_samples) at 16 kHz.
        """
        # kNN-VC's Generator takes (B, T, D), unlike a mel HiFi-GAN's (B, D, T).
        wav = self.generator(batch.BTC)                             # (B, 1, T*hop)
        sample_lengths = (batch.lengths * self._frame_grid.hop_size).clamp(
            max=wav.shape[-1]
        )
        return AudioBatch(
            data=wav,
            lengths=sample_lengths,
            sample_rate=self.sample_rate,
        )

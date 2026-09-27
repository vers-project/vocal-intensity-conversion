"""Mel-spectrogram conversion domain: log-mel encoder + pretrained neural vocoder.

Second of the three conversion domains, alongside SpeechTokenizer latents
(``vic/codec/speechtokenizer.py``) and WavLM features
(``vic/codec/wavlm_hifigan.py``).  Satisfies the same ``AudioCodec`` protocol, so the
converter, the conditional critic, the callbacks and the render/evaluate chain all work
unchanged -- switching domain is a change to the ``extractor`` block of the YAML.

Provenance
----------
Decoder (``backend: bigvgan``), BigVGAN-v2:

    Sang-gil Lee, Wei Ping, Boris Ginsburg, Bryan Catanzaro, Sungroh Yoon.
    "BigVGAN: A Universal Neural Vocoder with Large-Scale Training."
    ICLR 2023.  https://arxiv.org/abs/2206.04658

    Code     https://github.com/NVIDIA/BigVGAN                     (MIT)
    Weights  https://huggingface.co/nvidia/bigvgan_v2_22khz_80band_fmax8k_256x

The architecture descends from HiFi-GAN, whose generator is the intended second backend
(``backend: hifigan``, not yet implemented -- see ``_BACKENDS``):

    Jungil Kong, Jaehyeon Kim, Jaekyoung Bae.  "HiFi-GAN: Generative Adversarial
    Networks for Efficient and High Fidelity Speech Synthesis."  NeurIPS 2020.
    https://arxiv.org/abs/2010.05646 -- https://github.com/jik876/hifi-gan (MIT)

Why BigVGAN-v2, and why this specific checkpoint
------------------------------------------------
*Why BigVGAN at all.*  The requirement was a vocoder that stays intact under conditions
it was not trained on, not one that wins a benchmark table.  BigVGAN's periodic (snake)
activations and anti-aliased up/downsampling were introduced for exactly that, and the
paper's headline result is zero-shot robustness: trained on clean LibriTTS, it holds up
on unseen speakers, unseen languages, real noisy recordings, singing and instrumental
audio.  The VIC corpora are shouted, breathy and close-mic'd speech, which is off the
read-speech manifold every TTS vocoder is trained on, so that property is the one that
matters here.  Vocos was rejected despite better VISQOL/PESQ -- higher scores on
in-distribution speech is the profile we were explicitly told not to optimise for.

*Why ``fmax8k``.*  The mel filterbank of this checkpoint spans 0--8000 Hz, which is
exactly the band 16 kHz source audio occupies.  The alternatives
(``bigvgan_v2_22khz_80band_256x``, ``..._24khz_100band_256x``) have ``fmax: null``, so
librosa spreads the filterbank to Nyquist -- 11025 and 12000 Hz respectively -- and the
top ~8 of 80 (resp. ~13 of 100) bands would sit entirely above the signal's content,
pinned at the ``log(clamp(x, 1e-5))`` floor of -11.51.  A rail of identical constant
bands is an input pattern the generator never saw in training.  ``fmax8k`` removes the
problem rather than hoping the generator tolerates it.

*Why ``22khz``, not a 16 kHz vocoder.*  There is no 16 kHz vocoder with a comparable
robustness claim; ``speechbrain/tts-hifigan-libritts-16kHz`` is LibriTTS-only and its mel
hyperparameters are not fully documented, which is precisely the ambiguity that produces
audio that is wrong but raises nothing.  Running at 22.05 kHz is the cheaper compromise:
the mel ceiling is 8 kHz either way, so nothing above the source band is *encoded*, and
what the generator synthesises above 8 kHz is discarded whenever the evaluation
band-limits back to 16 kHz.

Consequences for the rest of the pipeline
-----------------------------------------
This codec reports ``sample_rate = 22050``, so the dataset resamples to 22.05 kHz and the
frame grid is 256 samples / 86.13 Hz rather than SpeechTokenizer's 320 / 50 Hz.  That is
deliberate: 256 samples at 22050 Hz is 11.61 ms, which is *not* an integer number of
samples at 16 kHz, so a codec claiming 16 kHz could not describe its own frame grid with
an integer ``FrameGrid.hop_size``.  Two knock-on effects, both handled elsewhere:

  * ``data.window_length`` is a duration expressed in samples.  The 400 used by every
    16 kHz config is 25 ms; the equivalent here is **551**.  Configs for this domain must
    say so -- a stale 400 would measure the SPL labels over 18 ms while P_phi was trained
    on 25 ms.
  * P_phi and the whitening in the measurement pipeline are 16 kHz.
    ``ExtractionPipeline.encode`` resamples to its own extractor's rate, so the ruler
    keeps seeing 16 kHz audio regardless of the conversion domain.

Frame-grid alignment
--------------------
Upstream's ``mel_spectrogram`` reflect-pads by ``(n_fft - hop) // 2`` = 384 and calls
``torch.stft(center=False)``, where true centring would pad ``n_fft // 2`` = 512.  Mel
frame *i* therefore analyses input samples ``[256i - 384, 256i + 640)``, centred at
``256i + 128`` -- half a hop later than the ``i * hop`` a non-causal ``FrameGrid``
promises.  The offset is ``n_fft // 2 - (n_fft - hop) // 2 = hop // 2`` in general, which
is numerically what ``FrameGrid(causal=True)`` encodes via ``center_offset``.  So the grid
is declared causal: not a claim about causality, but the only field that expresses the
half-hop shift, and declaring it makes ``FrameLevelTransform`` measure each frame's SPL
label over the samples that frame actually saw.

Frame count is exact: ``T_frames == T_samples // hop_size`` for any ``T_samples`` that is
a multiple of ``hop_size`` (verified against upstream's own function, 512--99840 samples).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn

from vic.codec._vendor import add_repo_to_path
from vic.core import FrameGrid
from vic.data.audio_batch import AudioBatch


@dataclass(frozen=True)
class _VocoderBundle:
    """One pretrained mel vocoder, plus the exact front end it was trained with.

    ``mel_fn`` travels with ``generator`` on purpose.  A mel vocoder is only as good as
    the agreement between the spectrogram it is fed and the one it was trained on, and
    that agreement is not expressible as a set of numbers: it also covers the mel filter
    convention (librosa/Slaney, not torchaudio's HTK default), magnitude vs power,
    the padding mode, and the log compression.  Keeping the callable next to the weights
    makes a mismatch impossible by construction instead of a config to get right.
    """

    generator: nn.Module
    h: object                                        # upstream AttrDict of hyperparams
    mel_fn: Callable[[torch.Tensor, object], torch.Tensor]


def _load_bigvgan(repo_path: str | Path, model_dir: str | Path) -> _VocoderBundle:
    """Load a BigVGAN generator and BigVGAN's own ``get_mel_spectrogram``.

    ``model_dir`` is a local directory holding the two files
    ``scripts/download_vocoders.py`` fetches from the Hugging Face repo: ``config.json``
    and ``bigvgan_generator.pt``.

    Deliberately *not* via ``BigVGAN.from_pretrained``.  That is a
    ``PyTorchModelHubMixin`` hook, and upstream's override declares ``proxies`` and
    ``resume_download`` keyword-only, which current ``huggingface_hub`` no longer passes
    -- so the call raises ``TypeError`` before it reads a single weight.  Loading the
    config and the state dict directly is what that method does anyway once the files are
    local, it matches how MP-SENet and kNN-VC are loaded elsewhere in this codebase, and
    it leaves the hub API free to keep moving.
    """
    add_repo_to_path(repo_path, name="BigVGAN", expect=["bigvgan.py", "meldataset.py"])

    import bigvgan                                     # noqa: PLC0415  (needs sys.path)
    from env import AttrDict                           # noqa: PLC0415
    from meldataset import get_mel_spectrogram         # noqa: PLC0415

    model_dir = Path(model_dir).expanduser().resolve()
    for required in ("config.json", "bigvgan_generator.pt"):
        if not (model_dir / required).exists():
            raise FileNotFoundError(
                f"BigVGAN model_dir {model_dir} is missing {required}. Fetch it with "
                f"scripts/download_vocoders.py."
            )

    with open(model_dir / "config.json") as f:
        h = AttrDict(json.load(f))
    # use_cuda_kernel=False: upstream supports the fused kernel for inference only, and
    # building it needs a matching nvcc/ninja.  We only ever run this decoder frozen, so
    # the portable path is the right one.
    generator = bigvgan.BigVGAN(h, use_cuda_kernel=False)

    state = torch.load(
        model_dir / "bigvgan_generator.pt", map_location="cpu", weights_only=True
    )
    try:
        generator.load_state_dict(state["generator"])
    except RuntimeError:
        # Some published checkpoints were saved after weight norm was folded away; this
        # is upstream's own recovery path, kept so either flavour loads.
        generator.remove_weight_norm()
        generator.load_state_dict(state["generator"])
    else:
        generator.remove_weight_norm()                 # as upstream's inference.py does
    return _VocoderBundle(generator=generator, h=h, mel_fn=get_mel_spectrogram)


# Registry rather than a class hierarchy: a backend is three things travelling together,
# and adding one is a function.  `hifigan` (jik876/hifi-gan UNIVERSAL_V1) is the intended
# second entry -- its mel front end is numerically identical to BigVGAN's (22050 / 80 /
# 1024 / 256 / 1024, fmin 0, fmax 8000), so the two decoders are directly A/B-able on the
# same features.  It needs its own clone: kNN-VC vendors a HiFi-GAN `Generator` but with
# an extra `lin_pre` linear layer for 1024-d WavLM input, so it cannot load UNIVERSAL_V1.
_BACKENDS: dict[str, Callable[..., _VocoderBundle]] = {
    "bigvgan": _load_bigvgan,
}


class MelVocoderCodec(nn.Module):
    """Frozen log-mel encoder + pretrained neural vocoder, satisfying ``AudioCodec``.

    ``encode`` is upstream's own mel function and ``decode`` is upstream's own generator;
    this class contributes only the ``AudioBatch`` / ``FrameGrid`` bookkeeping around
    them.  Every hyperparameter -- sample rate, hop, band count, ``fmax`` -- is read from
    the loaded checkpoint's ``config.json`` rather than from our YAML, so a config cannot
    disagree with the weights.

    Parameters
    ----------
    repo_path : path to the cloned decoder repository root (for ``bigvgan``, a checkout of
                https://github.com/NVIDIA/BigVGAN -- the directory containing
                ``bigvgan.py`` and ``meldataset.py``).
    model_dir : local directory holding the checkpoint files for the chosen backend.
    backend   : which decoder to load; see ``_BACKENDS``.
    """

    def __init__(
        self,
        repo_path: str | Path,
        model_dir: str | Path,
        backend: str = "bigvgan",
    ):
        super().__init__()
        if backend not in _BACKENDS:
            raise ValueError(
                f"Unknown mel vocoder backend {backend!r}. "
                f"Available: {sorted(_BACKENDS)}."
            )
        bundle = _BACKENDS[backend](repo_path, model_dir)

        self._backend = backend
        self._h = bundle.h
        self._mel_fn = bundle.mel_fn
        self.generator = bundle.generator
        self.generator.eval()
        for p in self.generator.parameters():
            p.requires_grad_(False)

        self._frame_grid = FrameGrid(
            hop_size=int(self._h.hop_size),
            sample_rate=int(self._h.sampling_rate),
            # See "Frame-grid alignment" in the module docstring: this expresses
            # upstream's half-hop analysis offset, not a causal encoder.
            causal=True,
        )
        # Reflect padding requires the pad width to be strictly less than the input, so
        # upstream's front end cannot analyse a signal this short at all.
        self._min_samples = (int(self._h.n_fft) - int(self._h.hop_size)) // 2 + 1

    # ------------------------------------------------------------------
    # AudioCodec protocol properties
    # ------------------------------------------------------------------

    @property
    def sample_rate(self) -> int:
        return int(self._h.sampling_rate)

    @property
    def latent_dim(self) -> int:
        return int(self._h.num_mels)

    @property
    def frame_grid(self) -> FrameGrid:
        return self._frame_grid

    # ------------------------------------------------------------------
    # Encode / decode
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode(self, batch: AudioBatch) -> AudioBatch:
        """Encode audio to log-mel frames.

        Parameters
        ----------
        batch : AudioBatch with ``data`` of shape (B, 1, T_samples) at
                ``self.sample_rate``.

        Returns
        -------
        AudioBatch of shape (B, ``latent_dim``, T_frames) with
        ``T_frames = T_samples // hop_size``.
        """
        if batch.sample_rate != self.sample_rate:
            raise ValueError(
                f"{type(self).__name__} expects {self.sample_rate} Hz audio (the rate the "
                f"{self._backend} checkpoint was trained at) but got {batch.sample_rate}. "
                "The dataset's target_sr comes from the pipeline's sample_rate, so this "
                "means the batch was built for a different extractor."
            )
        wav = batch.BCT
        if wav.shape[1] != 1:
            raise ValueError(f"expected mono (B, 1, T), got {tuple(wav.shape)}")
        if wav.shape[-1] < self._min_samples:
            raise ValueError(
                f"input is {wav.shape[-1]} samples; upstream's mel front end reflect-pads "
                f"by {self._min_samples - 1} and so needs at least {self._min_samples}."
            )

        mel = self._mel_fn(wav.squeeze(1), self._h)     # (B, n_mels, T_samples // hop)
        frame_lengths = batch.lengths // self._frame_grid.hop_size
        return batch.as_frames(mel, frame_lengths)

    @torch.no_grad()
    def decode(self, batch: AudioBatch) -> AudioBatch:
        """Synthesise a waveform from log-mel frames.

        Parameters
        ----------
        batch : AudioBatch of shape (B, ``latent_dim``, T_frames).

        Returns
        -------
        AudioBatch of shape (B, 1, T_samples) at ``self.sample_rate``.
        """
        wav = self.generator(batch.BCT)                 # (B, 1, T_frames * hop)
        sample_lengths = (batch.lengths * self._frame_grid.hop_size).clamp(
            max=wav.shape[-1]
        )
        return AudioBatch(
            data=wav,
            lengths=sample_lengths,
            sample_rate=self.sample_rate,
        )

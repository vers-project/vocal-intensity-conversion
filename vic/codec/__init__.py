"""The invertible representations a converter can operate in.

Everything here satisfies ``vic.core.AudioCodec`` (``encode`` + ``decode``), which is what
separates a conversion domain from the encoder-only front ends in ``vic/encoders/``: the
converter needs to get back out to a waveform.  Three domains, one protocol, selected by
``extractor.type`` in the YAML:

    nac            SpeechTokenizerCodec    16 kHz, hop 320, RVQ latents
    mel_vocoder    MelVocoderCodec         22.05 kHz, hop 256, log-mel + BigVGAN-v2
    wavlm_hifigan  WavLMHiFiGANCodec       16 kHz, hop 320, WavLM-Large L6 + kNN-VC HiFi-GAN

Each module's docstring carries its provenance, citation and the reasoning behind the
specific checkpoint.  The two vocoder domains need weights fetched by
``scripts/download_vocoders.py`` and the ``vocoder`` extra installed.

Imports here are light: the third-party repositories are only touched when a codec is
*constructed*, not when this package is imported.
"""
from vic.codec.mel_vocoder import MelVocoderCodec
from vic.codec.speechtokenizer import SpeechTokenizerCodec
from vic.codec.wavlm_hifigan import WavLMHiFiGANCodec

__all__ = ["SpeechTokenizerCodec", "MelVocoderCodec", "WavLMHiFiGANCodec"]

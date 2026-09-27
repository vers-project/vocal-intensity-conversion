"""Wav2Vec2Extractor: encodes waveforms to Wav2Vec2 transformer hidden states."""
from __future__ import annotations

import torch
import torch.nn as nn

from vic.core import FrameGrid
from vic.data.audio_batch import AudioBatch


class Wav2Vec2Extractor(nn.Module):
    """Frozen Wav2Vec2 encoder satisfying the FeatureExtractor protocol.

    Wraps HuggingFace Wav2Vec2Model.  Requires the `transformers` package.

    The convolutional feature extractor in standard Wav2Vec2 models has a
    total stride of 320 samples at 16 kHz, giving 50 frames/s — the same
    frame rate as SpeechTokenizer.

    Input waveforms are normalised per-utterance (zero-mean, unit-variance)
    before being passed to the model, matching the Wav2Vec2FeatureExtractor
    ``do_normalize=True`` behaviour.  Normalisation statistics are computed
    on valid samples only (padding zeros are excluded).

    Parameters
    ----------
    model_name : HuggingFace model identifier (e.g. "facebook/wav2vec2-base").
    freeze     : if True, freeze all model parameters (strongly recommended).
    layer      : number of transformer layers to keep (0 = feature projection
                 only, k = first k transformer layers, -1 = all layers).
                 Layers beyond this index are dropped from ``encoder.layers``
                 at init time, so they consume no memory and no compute.
                 Both ``Wav2Vec2Encoder`` and ``Wav2Vec2EncoderStableLayerNorm``
                 apply a final ``layer_norm`` after the (truncated) stack,
                 so ``last_hidden_state`` is always properly normalised.
    """

    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-base",
        freeze: bool = True,
        layer: int = -1,
    ):
        super().__init__()
        from transformers import Wav2Vec2Model
        self._model = Wav2Vec2Model.from_pretrained(model_name)

        if layer != -1:
            self._model.encoder.layers = self._model.encoder.layers[:layer]

        if freeze:
            for p in self._model.parameters():
                p.requires_grad_(False)

        self.sample_rate = 16_000
        self.latent_dim = self._model.config.hidden_size
        self.frame_grid = FrameGrid(
            hop_size=self._compute_hop_size(),
            sample_rate=self.sample_rate,
        )

    def _compute_hop_size(self) -> int:
        """Total stride of the CNN feature extractor."""
        stride = 1
        for s in self._model.config.conv_stride:
            stride *= s
        return stride

    def _normalize(self, wav: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Per-utterance zero-mean unit-variance normalization on valid samples only.

        Replicates Wav2Vec2FeatureExtractor with do_normalize=True.
        Padding zeros are left at 0 after normalization; the attention_mask
        passed to the model already excludes them from self-attention.
        """
        normed = wav.clone()
        for i, n in enumerate(lengths):
            n = int(n)
            x = wav[i, :n]
            normed[i, :n] = (x - x.mean()) / (x.var() + 1e-7).sqrt()
        return normed

    def encode(self, batch: AudioBatch) -> AudioBatch:
        """batch : AudioBatch (B, 1, T) → AudioBatch (B, hidden_size, T_frames)."""
        wav = batch.BCT.squeeze(1)                      # (B, T)
        wav = self._normalize(wav, batch.lengths)       # per-utterance z-norm
        attention_mask = batch.padding_mask.long()      # (B, T), 1 = valid

        outputs = self._model(input_values=wav, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state              # (B, T_frames, D)
        hidden_BCT = hidden.transpose(1, 2)             # (B, D, T_frames)

        frame_lengths = self._model._get_feat_extract_output_lengths(
            batch.lengths.long()
        )
        return batch.as_frames(hidden_BCT, frame_lengths)

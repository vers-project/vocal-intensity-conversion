"""Shared types: FrameGrid and AudioCodec protocol."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from vic.data.audio_batch import AudioBatch


@dataclass(frozen=True)
class FrameGrid:
    """Encodes the temporal frame mapping of a convolutional audio encoder.

    A codec with total stride S maps input sample i·S to output frame i.
    For a non-causal (symmetric) encoder the analysis centre of frame i is
    exactly i·hop_size.  For a causal encoder it is shifted right by
    hop_size // 2.

    Attributes
    ----------
    hop_size    : number of input samples between consecutive frame centres.
    sample_rate : sample rate of the input audio.
    causal      : whether the encoder is causal (left-padded only).
    """

    hop_size: int
    sample_rate: int
    causal: bool = False

    def __post_init__(self):
        # Caught at construction rather than in whichever property first divides
        # by it: an extractor that leaves either field unset produces a grid that
        # looks fine until something asks it for a duration.
        for field in ("hop_size", "sample_rate"):
            if getattr(self, field) is None:
                raise ValueError(
                    f"FrameGrid.{field} is None — the extractor that built this "
                    f"grid failed to resolve it."
                )

    @property
    def hop_duration_s(self) -> float:
        """Duration of one frame hop in seconds."""
        return self.hop_size / self.sample_rate

    @property
    def center_offset(self) -> int:
        """Sample offset from i*hop_size to the true centre of frame i."""
        return self.hop_size // 2 if self.causal else 0

    def frame_center(self, i: int) -> int:
        """Centre of frame i expressed in input samples."""
        return i * self.hop_size + self.center_offset

    def sample_to_frame(self, sample: int) -> int:
        """Nearest frame index for a given input sample position."""
        return round((sample - self.center_offset) / self.hop_size)

    def n_frames(self, n_samples: int) -> int:
        """Number of output frames produced from n_samples input samples.

        Uses floor division, which matches the behaviour of a strided
        convolution without output padding.
        """
        return n_samples // self.hop_size


@runtime_checkable
class FeatureExtractor(Protocol):
    """Protocol for any frozen encoder that maps audio to latent frames.

    Implemented by NAC wrappers, MelSpectrogramExtractor, Wav2Vec2Extractor, etc.
    Used by PredictorModule so the predictor head can be trained with any
    feature representation without changing the training loop.
    """

    sample_rate: int
    latent_dim: int
    frame_grid: FrameGrid

    def encode(self, batch: "AudioBatch") -> "AudioBatch":
        """Encode an audio AudioBatch to a latent AudioBatch."""
        ...


@runtime_checkable
class AudioCodec(FeatureExtractor, Protocol):
    """Full codec: encode + decode.  Used by the converter experiments."""

    def decode(self, batch: "AudioBatch") -> "AudioBatch":
        """Decode a latent AudioBatch back to an audio AudioBatch."""
        ...

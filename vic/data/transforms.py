"""Audio transform functions.

I/O, chunking, channel and amplitude helpers are imported from
``audio_utils.data.transforms`` — see that module for the chunking convention
(seconds, and short signals are never zero-padded; the collate function pads
and records the true length).

What stays here is the intensity-domain extraction that depends on VIC's
``FrameGrid``.  ExtractF0 is a class because it wraps a pitch-extraction model
(expensive init).
"""
from __future__ import annotations

import torch
from torch import Tensor

from audio_utils.data.transforms import (
    audio_info,
    fixed_chunk,
    load_audio,
    load_audio_chunk,
    mono_mix,
    pad_to_multiple,
    peak_normalize,
    random_chunk,
    resample,
    select_channel,
)
from vic.core import FrameGrid
from vic.data.audio_batch import AudioBatch

__all__ = [
    "audio_info",
    "fixed_chunk",
    "load_audio",
    "load_audio_chunk",
    "mono_mix",
    "pad_to_multiple",
    "peak_normalize",
    "random_chunk",
    "resample",
    "resample_batch",
    "select_channel",
    "extract_sequence_intensity",
    "extract_frame_intensity",
    "normalize_sequence_from_config",
    "prepare_waveform",
    "ExtractF0",
]

# Peak normalisation defaults per task, because the two tasks genuinely differ
# and the difference used to live implicitly in a dataset class signature.
#
#   Predictor (True)   — P_φ is trained on peak-normalised audio, so absolute
#                        level cannot be read off the amplitude and must come
#                        from spectral shape.  Its labels are measured on the
#                        raw waveform *before* normalisation.
#   Converter (False)  — the conversion latents are raw NAC, whose whole job is
#                        to carry amplitude alongside spectral pattern; the
#                        module rejects a whitened conversion pipeline outright.
#
# Whitened pipelines are exactly gain-invariant — whitening subtracts
# mean_t(log|X[f,t]|) per bin, so a gain adds a constant that cancels — which is
# why the flag is inert for a whitened predictor and decisive for a raw one.
NORMALIZE_SEQUENCE_PREDICTOR = True
NORMALIZE_SEQUENCE_CONVERTER = False


# ---------------------------------------------------------------------------
# Inference-side preprocessing
# ---------------------------------------------------------------------------

def resample_batch(batch: AudioBatch, target_sr: int) -> AudioBatch:
    """Resample a waveform ``AudioBatch``, scaling its lengths with it.  No-op if equal.

    Needed because the conversion domain and the measurement domain no longer share a
    sample rate: the mel vocoder runs at 22.05 kHz (its hop is not an integer number of
    16 kHz samples, so it cannot describe its own frame grid at 16 kHz) while P_φ and the
    STFT whitening are 16 kHz.  ``ExtractionPipeline.encode`` applies this so a pipeline
    always hands its extractor the rate that extractor was trained at.

    ``lengths`` are rescaled and clamped rather than recomputed from the resampler, whose
    output length is ``ceil`` of the ratio and so can exceed the scaled length by a sample.
    """
    if batch.sample_rate == target_sr:
        return batch
    data = resample(batch.data, batch.sample_rate, target_sr)
    lengths = (
        batch.lengths.to(torch.float64) * target_sr / batch.sample_rate
    ).round().long().clamp(max=data.shape[-1])
    return AudioBatch(data=data, lengths=lengths, sample_rate=target_sr)


def normalize_sequence_from_config(config: dict, default: bool) -> bool:
    """Read ``extractor.normalize_sequence``, falling back to a task default.

    ``default`` has no safe value: see ``NORMALIZE_SEQUENCE_PREDICTOR`` /
    ``NORMALIZE_SEQUENCE_CONVERTER`` above.  It is required rather than defaulted
    so a caller cannot get the wrong one by omission.
    """
    return bool(config.get("extractor", {}).get("normalize_sequence", default))


def prepare_waveform(
    wav: Tensor,
    sr: int,
    target_sr: int,
    hop_size: int,
    normalize_sequence: bool,
) -> Tensor:
    """Preprocess one loaded waveform the way ``AudioDataset`` does at training.

    ``resample → mono-mix → pad-to-multiple → (optional) peak-normalise``.

    The dataset pads before normalising and this mirrors that, though the order
    is immaterial: padding appends zeros, which cannot change a peak.
    """
    wav = resample(wav, sr, target_sr)
    wav = mono_mix(wav)
    wav = pad_to_multiple(wav, hop_size)
    if normalize_sequence:
        wav = peak_normalize(wav)
    return wav


# ---------------------------------------------------------------------------
# Feature extraction (functions)
# ---------------------------------------------------------------------------

def extract_sequence_intensity(wav: Tensor) -> Tensor:
    """RMS intensity of the whole signal in dBFS. Returns a scalar Tensor."""
    rms = wav.pow(2).mean().sqrt()
    return 20 * torch.log10(rms.clamp(min=1e-8))


def extract_frame_intensity(
    wav: Tensor,
    grid: FrameGrid,
    window_size: int | None = None,
) -> Tensor:
    """Per-frame RMS intensity aligned with grid. Returns (T_frames,) Tensor."""
    win = window_size if window_size is not None else grid.hop_size
    n_frames = grid.n_frames(wav.shape[-1])
    out = wav.new_zeros(n_frames)
    for i in range(n_frames):
        centre = grid.frame_center(i)
        lo = max(0, centre - win // 2)
        hi = min(wav.shape[-1], lo + win)
        rms = wav[..., lo:hi].pow(2).mean().sqrt()
        out[i] = 20 * torch.log10(rms.clamp(min=1e-8))
    return out


# ---------------------------------------------------------------------------
# Stateful / expensive extractors
# ---------------------------------------------------------------------------

class ExtractF0:
    """Per-frame F0 extractor aligned with a FrameGrid.

    Kept as a class because it may eventually wrap a neural pitch estimator
    with costly model loading at construction time.

    Parameters
    ----------
    grid        : FrameGrid of the codec.
    fine_hop_ms : temporal resolution of the internal pyin call (ms).
    fmin / fmax : pitch range in Hz.
    """

    def __init__(
        self,
        grid: FrameGrid,
        fine_hop_ms: float = 1.0,
        fmin: float = 50.0,
        fmax: float = 600.0,
    ):
        self.grid = grid
        self.fine_hop_ms = fine_hop_ms
        self.fmin = fmin
        self.fmax = fmax

    def __call__(self, wav: Tensor) -> Tensor:
        """wav : (1, T) → (T_frames,) F0 in Hz (0 = unvoiced)."""
        from vic.features.f0 import extract_f0_aligned
        return extract_f0_aligned(wav, self.grid, self.fine_hop_ms, self.fmin, self.fmax)

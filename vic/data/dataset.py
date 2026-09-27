"""Audio datasets for intensity experiments.

Both classes are thin specialisations of ``audio_utils.data.dataset.AudioDataset``,
which owns the loading, chunking and padding logic — see that module for the
chunking convention.  Only the requested chunk is decoded, so cost does not
grow with file duration.
"""
from __future__ import annotations

import pandas as pd
from audio_utils.data.dataset import AudioDataset
from audio_utils.data.transforms import peak_normalize

from vic.core import FrameGrid
from vic.data.metadata import require_non_null
from vic.features.spl import FrameLevelTransform


class UnlabeledAudioDataset(AudioDataset):
    """Waveform-only dataset (e.g. LibriSpeech for distillation).

    ``AudioDataset`` with a ``FrameGrid`` in place of a raw padding multiple.

    Parameters
    ----------
    metadata           : DataFrame with a ``signal_path`` column and an optional
                         ``channel`` column (0-based; -1 or absent → mono mix).
    target_sr          : target sample rate.
    frame_grid         : FrameGrid of the codec/extractor — its hop size is what
                         each chunk is padded to a multiple of.
    chunk_s            : audio chunk duration in seconds, or None for the full signal.
    train              : True → random chunk; False → chunk from the start.
    normalize_sequence : if True, peak-normalize the waveform before returning.
                         Disable for models expecting raw amplitudes (NAC, Wav2Vec2).
    """

    def __init__(
        self,
        metadata: pd.DataFrame,
        target_sr: int,
        frame_grid: FrameGrid,
        chunk_s: float | None,
        train: bool,
        normalize_sequence: bool = True,
    ):
        super().__init__(
            metadata,
            target_sr=target_sr,
            chunk_s=chunk_s,
            train=train,
            pad_to_multiple_of=frame_grid.hop_size,
            normalize_sequence=normalize_sequence,
        )
        self.frame_grid = frame_grid


class IntensityDataset(UnlabeledAudioDataset):
    """Dataset for intensity-labelled speech audio.

    Adds per-frame calibrated SPL labels, computed by ``FrameLevelTransform``
    with the provided ``frame_grid`` and ``window_length`` — so they stay exactly
    aligned with the codec or feature extractor, and are measured before any peak
    normalization.

    Not on the *raw* waveform, though: ``AudioDataset._load`` has already applied
    ``pad_to_multiple``, so the last frame's analysis window can contain up to
    199 appended zero samples and read up to 3 dB low.  See
    ``vic.features.spl.FrameLevelTransform.__call__`` for that and for frame 0's
    unconditional -3.01 dB.

    Parameters
    ----------
    metadata      : as ``UnlabeledAudioDataset``, plus ``calibration_rms`` and
                    ``distance_m`` columns.
    window_length : SPL analysis window in samples (passed to ``FrameLevelTransform``).
    """

    def __init__(
        self,
        metadata: pd.DataFrame,
        target_sr: int,
        frame_grid: FrameGrid,
        window_length: int,
        chunk_s: float | None,
        train: bool,
        normalize_sequence: bool = True,
    ):
        super().__init__(
            metadata,
            target_sr=target_sr,
            frame_grid=frame_grid,
            chunk_s=chunk_s,
            train=train,
            normalize_sequence=normalize_sequence,
        )
        # Silent otherwise: a NaN propagates through FrameLevelTransform without
        # raising (``max(nan, 1e-10)`` is ``nan``), so every frame label becomes
        # NaN, the batch collates, and the loss goes NaN.
        require_non_null(
            metadata, ["calibration_rms", "distance_m"],
            "a null here silently produces all-NaN labels.  Run compute_labels.py "
            "over the metadata before training.",
        )
        self.calibration_rms = metadata["calibration_rms"].tolist()
        self.distance_m = metadata["distance_m"].tolist()
        self.frame_transform = FrameLevelTransform(frame_grid, window_length)

    def __getitem__(self, index: int) -> dict:
        wav = self._load(index)

        frame_labels = self.frame_transform(
            wav,
            self.calibration_rms[index],
            self.distance_m[index],
        )  # (T_frames,)

        if self.normalize_sequence:
            wav = peak_normalize(wav)

        return {
            "wav": wav,                                       # (1, T_samples)
            "frame_intensity_db": frame_labels.unsqueeze(0),  # (1, T_frames)
        }

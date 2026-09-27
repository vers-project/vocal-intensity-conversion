"""Collation for intensity training batches.

The waveform-only collate lives in ``audio_utils.data.collate``; only the
labelled one, which also packs the frame-level SPL targets, is VIC-specific.
"""
from __future__ import annotations

from audio_utils.data.collate import make_collate_fn as make_unlabeled_collate_fn

from vic.data.audio_batch import AudioBatch

__all__ = ["make_unlabeled_collate_fn", "make_intensity_collate_fn"]


def make_intensity_collate_fn(sample_rate: int, hop_size: int):
    """Return a DataLoader collate_fn for IntensityDataset batches.

    Parameters
    ----------
    sample_rate : audio sample rate — used for the waveform AudioBatch.
    hop_size    : codec hop size — used to derive the frame rate for the
                  frame-level intensity AudioBatch.
    """
    frame_rate = sample_rate // hop_size

    def collate_fn(samples: list[dict]) -> dict:
        wavs = [s["wav"] for s in samples]
        frame_labels = [s["frame_intensity_db"] for s in samples]
        return {
            "wav": AudioBatch.from_list(wavs, sample_rate=sample_rate),
            "frame_intensity_db": AudioBatch.from_list(frame_labels, sample_rate=frame_rate),
        }

    return collate_fn

"""Trim the leading and trailing silence off every row of a segment index.

Why this is a metadata operation and not an audio one
-----------------------------------------------------
``AudioDataset`` already reads a row as the region ``[start_s, end_s)`` of its file and
seeks there rather than decoding what precedes it, so "remove the silence before and
after the speech" is fully expressed by moving those two numbers inward.  Nothing has to
be re-cut on disk, and — the part that matters — ``IntensityDataset`` then computes its
calibrated per-frame dB SPL over the *trimmed* waveform, so the labels stay aligned with
the audio the converter is actually given.  Trimming the waveform after the dataset had
loaded it would leave the frame labels describing a different span.

Silence is never removed from *within* a segment: only the head before the first speech
island and the tail after the last one are dropped.  Cutting an internal pause would
splice two discontinuous pieces of audio together, which corrupts every frame-level
acoustic measurement made downstream — and, for this corpus, would destroy the pauses
that are part of the sentence.

Why a VAD and not a level gate
------------------------------
Silero thresholds a *speech posterior*, not a level, so a quiet talker is not penalised
the way a fixed-dB gate would penalise one.  On a corpus whose whole point is that
speakers vary by 30 dB of vocal effort, a level gate would trim the soft conditions
harder than the loud ones and confound the very axis under study.

The model weights ship inside the ``silero-vad`` wheel, so this needs no network access.
"""
from __future__ import annotations

import pandas as pd
from audio_utils.data.transforms import load_audio, resample, select_channel
from torch import Tensor
from tqdm import tqdm

# Sample rates the Silero VAD accepts natively; anything else is resampled.
NATIVE_RATES = (8000, 16000)
VAD_RATE = 16000

#: Defaults, named as in ``data-collection/scripts/vad_segments.py`` so a parameter set
#: reads the same in both places.  ``min_silence_ms`` is what decides whether a pause is
#: an island boundary at all; it is irrelevant to the head and tail this module keeps,
#: but it is what Silero uses to close an island, so it still shapes where the *first*
#: island starts and the *last* one ends.
DEFAULT_VAD_PARAMS = {
    "threshold": 0.5,
    "min_island_ms": 100,
    "min_silence_ms": 100,
    "speech_pad_ms": 30,
    "onnx": False,
}

#: Columns :func:`trim_metadata_with_vad` writes.  ``start_s``/``end_s`` are *overwritten*
#: — that is the whole point — so the originals are preserved under ``untrimmed_*``.
TRIM_COLUMNS = [
    "untrimmed_start_s",
    "untrimmed_end_s",
    "trimmed_head_s",
    "trimmed_tail_s",
    "vad_speech_s",
    "vad_n_islands",
    "vad_status",
]


def load_vad_model(onnx: bool = False):
    """Load the Silero VAD.  Imported lazily so importing this module stays cheap."""
    from silero_vad import load_silero_vad

    return load_silero_vad(onnx=onnx)


def speech_span(
    wav: Tensor, sample_rate: int, model, params: dict
) -> tuple[float, float, float, int] | None:
    """First island's start and last island's end, in seconds from ``wav``'s own start.

    Returns ``(start_s, end_s, speech_s, n_islands)``, or ``None`` when the VAD finds no
    speech at all — a real outcome on a segment index (a mis-annotated boundary, a
    dead-mic session), and one the caller must decide about rather than have silently
    turned into a zero-length region here.

    ``speech_s`` is the total island duration, so ``speech_s / (end_s - start_s)`` is the
    fraction of the *retained* span the VAD calls speech; it is < 1 exactly to the extent
    that the segment contains internal pauses, which are deliberately kept.
    """
    from silero_vad import get_speech_timestamps

    mono = wav.mean(dim=0)  # the VAD is mono-only
    if sample_rate not in NATIVE_RATES:
        mono = resample(mono.unsqueeze(0), sample_rate, VAD_RATE).squeeze(0)
        rate = VAD_RATE
    else:
        rate = sample_rate

    # return_seconds=False: the seconds mode rounds to 0.1 s, which is coarser than the
    # trim being asked for here.
    islands = get_speech_timestamps(
        mono,
        model,
        sampling_rate=rate,
        threshold=params["threshold"],
        min_speech_duration_ms=params["min_island_ms"],
        min_silence_duration_ms=params["min_silence_ms"],
        speech_pad_ms=params["speech_pad_ms"],
        return_seconds=False,
    )
    if not islands:
        return None

    start_s = islands[0]["start"] / rate
    end_s = islands[-1]["end"] / rate
    speech_s = sum(i["end"] - i["start"] for i in islands) / rate
    return start_s, end_s, speech_s, len(islands)


def trim_metadata_with_vad(
    metadata: pd.DataFrame,
    params: dict | None = None,
    model=None,
    progress: bool = True,
) -> pd.DataFrame:
    """Return a copy of ``metadata`` whose ``start_s``/``end_s`` bound the speech only.

    Parameters
    ----------
    metadata : rows with a ``signal_path`` already resolved to an absolute path, and
               optional ``start_s``/``end_s``/``channel`` columns read exactly as
               :class:`audio_utils.data.dataset.AudioDataset` reads them — per row and
               NaN-tolerant, so a table mixing a segment index with whole-file rows
               behaves correctly for both.
    params   : keys of :data:`DEFAULT_VAD_PARAMS`; missing ones take their default.
    model    : a preloaded Silero model, or None to load one.

    Rows where the VAD finds no speech keep their original span and are marked
    ``vad_status == "no_speech"``; every other row is ``"ok"``.  Nothing is dropped here —
    the caller reports and filters, because "how many of my test sentences contain no
    detectable speech" is a result, not an implementation detail.
    """
    params = {**DEFAULT_VAD_PARAMS, **(params or {})}
    if model is None:
        model = load_vad_model(onnx=params["onnx"])

    out = metadata.copy().reset_index(drop=True)

    has_span = "start_s" in out.columns and "end_s" in out.columns
    starts = (
        [0.0 if pd.isna(s) else float(s) for s in out["start_s"]]
        if has_span else [0.0] * len(out)
    )
    ends = (
        [None if pd.isna(e) else float(e) for e in out["end_s"]]
        if has_span else [None] * len(out)
    )
    channels = (
        [-1 if pd.isna(c) else int(c) for c in out["channel"]]
        if "channel" in out.columns else [-1] * len(out)
    )

    rows = []
    iterator = zip(out["signal_path"], starts, ends, channels)
    if progress:
        iterator = tqdm(iterator, total=len(out), desc="VAD trim", unit="seg")

    for path, start_s, end_s, channel in iterator:
        duration_s = None if end_s is None else end_s - start_s
        wav, sr = load_audio(path, start_s, duration_s)
        # Select the channel *before* the VAD, not the mono mix of all of them: on a
        # multi-track corpus the other channels can carry a second talker (AVID's ch1 is
        # the EGG), and mixing them in would move the speech boundaries.
        wav = select_channel(wav, channel)
        region_s = wav.shape[-1] / sr

        span = speech_span(wav, sr, model, params)
        if span is None:
            rows.append({
                "start_s": start_s,
                "end_s": start_s + region_s,
                "untrimmed_start_s": start_s,
                "untrimmed_end_s": start_s + region_s,
                "trimmed_head_s": 0.0,
                "trimmed_tail_s": 0.0,
                "vad_speech_s": 0.0,
                "vad_n_islands": 0,
                "vad_status": "no_speech",
            })
            continue

        head_s, tail_end_s, speech_s, n_islands = span
        rows.append({
            "start_s": start_s + head_s,
            "end_s": start_s + tail_end_s,
            "untrimmed_start_s": start_s,
            "untrimmed_end_s": start_s + region_s,
            "trimmed_head_s": head_s,
            "trimmed_tail_s": region_s - tail_end_s,
            "vad_speech_s": speech_s,
            "vad_n_islands": n_islands,
            "vad_status": "ok",
        })

    trimmed = pd.DataFrame(rows, index=out.index)
    for column in trimmed.columns:
        out[column] = trimmed[column]
    return out

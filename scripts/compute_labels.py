"""Compute LeqZF intensity labels for a calibrated speech dataset.

Reads a metadata CSV, computes calibrated sound pressure levels for every
speech signal, and writes two new columns back to the CSV:

    calibration_rms  : RMS of the calibration tone recording (float).
    intensity_db     : LeqZF in dB SPL at 1 m mouth distance (float).

This is the **only** place a calibration tone is turned into an RMS, so AVID,
VIC, Mixer 7 and FLombard all get the identical calculation.  A second
implementation upstream is how two calibrations silently drift apart.

Prerequisites
-------------
The input CSV must contain:
    signal_path      : path to the speech audio file
    calibration_path : path to the matching calibration tone recording
    distance_m       : mouth-to-microphone distance in metres

Whole files or segments
-----------------------
Two metadata shapes are labelled by the same code path:

    one row per audio file      -- the whole file is measured
    one row per speech segment  -- ``start_s``/``end_s`` bound the row, and only
                                   that window is decoded

The segment form matters because its ``signal_path`` is a *session* recording
tens of minutes long, shared by every segment of that speaker.  Measuring the
whole file would give all of them the same label, which is not a level at all.
Reading is seek-based, so the cost scales with the segment rather than with the
session.

A ``channel`` column, when present, selects one channel (0-based; ``-1``
mono-mixes).  AVID's session recordings are 2-channel with the
electroglottograph on channel 1, and averaging that into the speech would
corrupt every label.

Calibration tone RMS is computed once per unique calibration file and
cached in memory, so files sharing a calibration recording are not
reloaded multiple times.

Usage
-----
    uv run scripts/compute_labels.py \\
        --csv    data/metadata.csv \\
        --sr     16000 \\
        --output data/metadata.csv        # overwrite in-place, or give a new path
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
import torchaudio.functional as AF

from vic.data.transforms import load_audio, select_channel
from vic.features.spl import LeqZFTransform, compute_calibration_rms
from vic.data.metadata import resolve_paths

from experiment_launcher import parse_args


def load_signal(
    path: str,
    target_sr: int,
    start_s: float | None = None,
    end_s: float | None = None,
    channel: int | None = None,
) -> torch.Tensor:
    """Load audio (or one window of it), resample if needed, return (C, T).

    ``start_s``/``end_s`` are seek bounds, not a crop of an already-decoded
    file: what precedes the window is never read.  Either being ``None`` reads
    the whole file, which is what a one-row-per-file corpus wants.
    """
    offset_s = 0.0 if start_s is None else float(start_s)
    duration_s = None if (start_s is None or end_s is None) else float(end_s) - offset_s
    wav, sr = load_audio(path, offset_s, duration_s)
    if channel is not None:
        wav = select_channel(wav, int(channel))
    if sr != target_sr:
        wav = AF.resample(wav, sr, target_sr)
    return wav

@parse_args
def main(config: dict, output_dir: Path):

    metadata = pd.read_csv(config["csv"])
    # Stash original relative paths — resolve_paths replaces them with absolute
    # paths for file I/O, but we write the relative originals back to the output
    # CSV so that it remains portable across machines.
    path_cols = ["signal_path", "calibration_path"]
    original_paths = metadata[path_cols].copy()
    metadata = resolve_paths(metadata, config["dataset_roots"], path_columns=path_cols)

    required = {"signal_path", "calibration_path", "distance_m"}
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    transform = LeqZFTransform(sample_rate=config["sr"], tau_time_weighting=config["tau"], target_distance_m=config["db_at_x_m"])

    # Resolved per row, not per table: a combined CSV holds a segment-indexed
    # corpus beside one-row-per-file corpora, and after the concat the bound
    # columns are present but NaN on the file rows.  A table-level flag would
    # hand those NaNs to the reader.
    bounded = (
        metadata["start_s"].notna() & metadata["end_s"].notna()
        if {"start_s", "end_s"} <= set(metadata.columns)
        else pd.Series(False, index=metadata.index)
    )
    channels = (
        metadata["channel"] if "channel" in metadata.columns
        else pd.Series(pd.NA, index=metadata.index)
    )
    print(f"Measuring {int(bounded.sum())} segment row(s) seek-based and "
          f"{int((~bounded).sum())} whole file(s); "
          f"{int(channels.notna().sum())} row(s) select a channel.")

    # Cache calibration RMS per unique calibration file.
    cal_rms_cache: dict[str, float] = {}

    intensity_db_col: list[float] = []
    cal_rms_col: list[float] = []

    n = len(metadata)
    for i,(index, row) in enumerate(metadata.iterrows()):
        cal_path = str(row["calibration_path"])
        if cal_path not in cal_rms_cache:
            # The tone is always a whole file, never a segment of one.
            cal_wav = load_signal(cal_path, config["sr"])
            cal_rms_cache[cal_path] = compute_calibration_rms(cal_wav)

        cal_rms = cal_rms_cache[cal_path]
        channel = channels.at[index]
        wav = load_signal(
            str(row["signal_path"]), config["sr"],
            start_s=row["start_s"] if bounded.at[index] else None,
            end_s=row["end_s"] if bounded.at[index] else None,
            channel=channel if pd.notna(channel) else None,
        )
        level = transform(wav, calibration_rms=cal_rms, distance_m=float(row["distance_m"]))

        cal_rms_col.append(cal_rms)
        intensity_db_col.append(level)

        if (i + 1) % 100 == 0 or (i + 1) == n:
            print(f"  {i + 1}/{n}")

    metadata["calibration_rms"] = cal_rms_col
    metadata["intensity_db"]    = intensity_db_col

    metadata[path_cols] = original_paths

    output_path = config["output"]
    metadata.to_csv(output_path, index=False)
    print(f"Saved {n} labels → {output_path}")


if __name__ == "__main__":
    main()

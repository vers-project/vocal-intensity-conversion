"""Calibrated levels of the real recordings a listening test played.

The listening test scores a trial by which member was louder, so every *real*
member needs a level in dB SPL.  That level is measured here rather than read
from a stored column, for two reasons:

* the stimuli were **VAD-trimmed** before rendering, and a segment's leading and
  trailing silence pulls its whole-segment $L_\\mathrm{eq}$ down by several
  decibels -- so the untrimmed row is the wrong reference for what was heard;
* it must be the same quantity, measured the same way, as the conditioning
  labels the converter was trained on, which means the same
  ``FrameLevelTransform`` at the same window rather than a second convention.

Lifted out of ``scripts/analyze_perceptual.py`` so the screening analysis under
``perceptual/`` and the main scoring script share one implementation.
"""
from __future__ import annotations

import pandas as pd

from vic.data.metadata import resolve_paths
from vic.data.vad import trim_metadata_with_vad
from vic.features.spl import leq_aggregate

#: The columns that identify one real recording in the corpus table.
KEY = ["speaker", "sentence_id", "repetition", "source_level_index"]


def stimulus_keys(trials: pd.DataFrame, side: str) -> pd.DataFrame:
    """The distinct recordings on one side of the pair, as corpus keys."""
    keys = trials[[f"{side}_{c}" for c in KEY]].drop_duplicates()
    keys.columns = KEY
    return keys.reset_index(drop=True)


def calibrated_levels(keys: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Calibrated $L_\\mathrm{eq}$ of each real stimulus, over its played span.

    The same Silero parameters as the render are applied, which reproduces the
    span because the trim is deterministic.  Returns ``KEY`` plus ``b_level_db``
    and ``played_s``; callers rename the level column for the side they asked
    about.
    """
    metadata = pd.read_csv(config["metadata_csv"])
    metadata = resolve_paths(
        metadata, config["dataset_roots"], path_columns=["signal_path"]
    )
    metadata["speaker"] = (
        metadata["speaker_uid"].astype(str).str.split(":").str[-1].astype(int)
    )
    for column in ("sentence_id", "repetition"):
        metadata[column] = pd.to_numeric(metadata[column], errors="coerce")
    # `src` in a stimulus name is the effort RANK, 1..4.  The table stores the
    # level by name; the rank is derived by prepare_level_groups and not saved,
    # so it is re-derived here from the same mapping rather than assumed.
    from vic.data.levels import LEVEL_INDEX

    metadata["source_level_index"] = metadata["level"].map(LEVEL_INDEX)
    metadata = metadata[metadata["source_level_index"].notna()]

    wanted = keys.merge(metadata, on=KEY, how="left", indicator=True)
    missing = wanted[wanted["_merge"] != "both"]
    if len(missing):
        raise ValueError(
            f"{len(missing)} real stimulus/stimuli have no row in "
            f"{config['metadata_csv']}, e.g.\n{missing[KEY].head().to_string()}\n"
            "The experiment and the corpus table disagree; check the split and "
            "that this is the annotated table."
        )
    if wanted.duplicated(subset=KEY).any():
        raise ValueError(
            "a real stimulus matches more than one corpus row on "
            f"{KEY}; retakes must be resolved before levels can be assigned."
        )
    rows = wanted.drop(columns=["_merge"]).reset_index(drop=True)

    vad = dict(config.get("vad") or {})
    if vad.pop("enabled", True):
        rows = trim_metadata_with_vad(rows, vad)
        bad = rows[rows["vad_status"] != "ok"]
        if len(bad):
            raise ValueError(
                f"the VAD found no speech in {len(bad)} real stimulus/stimuli, "
                "which cannot be true of audio that was rendered and played; "
                "the parameters here differ from the render's."
            )

    # IntensityDataset would work here, but it needs a FrameGrid, which belongs
    # to an extractor this analysis has no reason to build.  The level is a
    # property of the waveform and the calibration, so the same frame transform
    # is applied directly, at the 25 ms window the conditioning labels used.
    from vic.data.transforms import load_audio, mono_mix, select_channel
    from vic.features.spl import FrameLevelTransform
    from audio_utils.data.transforms import resample as resample_wav
    from vic.core import FrameGrid

    sr = int(config.get("sample_rate", 16_000))
    grid = FrameGrid(sample_rate=sr, hop_size=int(config.get("hop_size", 320)))
    transform = FrameLevelTransform(grid, int(config["window_length"]))

    levels = []
    for row in rows.itertuples():
        wav, file_sr = load_audio(
            row.signal_path, offset_s=float(row.start_s),
            duration_s=float(row.end_s) - float(row.start_s),
        )
        # AVID stores the electroglottograph on channel 1, so the channel must
        # be selected and not mixed; absent or NaN means the row is already mono.
        channel = getattr(row, "channel", None)
        wav = (
            mono_mix(wav) if channel is None or pd.isna(channel)
            else select_channel(wav, int(channel))
        )
        if file_sr != sr:
            wav = resample_wav(wav, file_sr, sr)
        frames = transform(wav, float(row.calibration_rms), float(row.distance_m))
        levels.append(float(leq_aggregate(frames.unsqueeze(0))[0]))
    out = rows[KEY].copy()
    out["b_level_db"] = levels
    out["played_s"] = (rows["end_s"] - rows["start_s"]).to_numpy()
    return out


def attach_measured_levels(
    trials: pd.DataFrame, side: str, column: str, config: dict,
    cache: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Measure one side's real recordings and merge the level in as ``column``."""
    keys = stimulus_keys(trials, side)
    measured = calibrated_levels(keys, config)
    if cache is not None:
        cache[side] = measured
    measured = measured.rename(columns={"b_level_db": column})
    return trials.merge(
        measured[KEY + [column]].rename(columns={c: f"{side}_{c}" for c in KEY}),
        on=[f"{side}_{c}" for c in KEY], how="left",
    )

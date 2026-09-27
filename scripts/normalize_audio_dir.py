"""Mirror a directory of audio with every waveform peak-normalised.

The companion to ``scripts/convert_test_set.py``, which writes unnormalised 32-bit float
WAVs so the amplitude the decoder produced can be measured.  A listening test needs the
opposite: if the loud targets are louder on the way out, amplitude alone answers the
question the test is asking, and the listener never has to judge whether the *voice*
sounds loud.  Splitting the two means one render serves both, and the measured set is
never overwritten by the normalised one.

The applied gain is recorded in ``normalization.csv`` at the root of the mirror, so the
normalisation is reversible and the original level is still recoverable from the mirror
alone.

Non-audio files are copied verbatim, so ``conversion_metadata.csv`` travels with the
audio it describes.

Usage
-----
    launch_experiment --config path/to/normalize_audio_dir.yaml \\
                      --script scripts/normalize_audio_dir.py

Config
------
    data.input_dir    : the render to mirror.
    data.output_dir   : where the mirror is written.  Optional — without it the mirror
                        goes to ``<launcher output_dir>/normalized``, which keeps the
                        launcher's timestamped directory as the single record of the run.
    normalization.target_peak : peak absolute amplitude every file is scaled to (0.9).
    normalization.encoding    : ``pcm_s16`` (default) or ``pcm_f32``.

A full render is ~17 000 files, so this is worth a compute job rather than a terminal that
has to stay open; it is single-process and does no GPU work.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import soundfile as sf

from audio_utils.data.transforms import load_audio, rms_dbfs
from vic.data.transforms import peak_normalize

from experiment_launcher import parse_args

AUDIO_SUFFIXES = {".wav", ".flac"}

# soundfile, not ``torchaudio.save``: in torchaudio 2.11 that dispatches to TorchCodec's
# AudioEncoder, which warns that ``encoding``/``bits_per_sample`` "are not fully
# supported" and then writes 16-bit PCM regardless of what was asked for.  Harmless for
# pcm_s16 and wrong for pcm_f32, so neither goes through it.
SUBTYPES = {
    # Normalised output sits at target_peak < 1.0, so 16-bit PCM cannot clip it and is
    # what every listening-test tool expects.
    "pcm_s16": "PCM_16",
    # Kept for a mirror that is going to be measured as well as heard.
    "pcm_f32": "FLOAT",
}


@parse_args
def main(config: dict, output_dir: Path):

    dc = config["data"]
    nc = config.get("normalization") or {}

    in_root = Path(dc["input_dir"])
    out_root = Path(dc["output_dir"]) if dc.get("output_dir") else output_dir / "normalized"
    target_peak = float(nc.get("target_peak", 0.9))
    encoding = nc.get("encoding", "pcm_s16")

    if not in_root.is_dir():
        raise ValueError(f"data.input_dir {in_root} is not a directory.")
    if out_root.resolve() == in_root.resolve():
        raise ValueError(
            "data.output_dir must differ from data.input_dir; this writes a mirror, and "
            "normalising the render in place would destroy the amplitudes it exists to "
            "record."
        )
    if encoding not in SUBTYPES:
        raise ValueError(
            f"normalization.encoding must be one of {sorted(SUBTYPES)}, got {encoding!r}."
        )

    out_root.mkdir(parents=True, exist_ok=True)
    print(f"{in_root} → {out_root}  (peak {target_peak}, {encoding})")

    rows = []
    n_audio = n_copied = 0

    for src in sorted(in_root.rglob("*")):
        if src.is_dir():
            continue
        rel = src.relative_to(in_root)
        dst = out_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)

        if src.suffix.lower() not in AUDIO_SUFFIXES:
            shutil.copy2(src, dst)
            n_copied += 1
            continue

        wav, sr = load_audio(src)
        peak = float(wav.abs().max())
        normalized = peak_normalize(wav, target_peak=target_peak)
        sf.write(str(dst), normalized.numpy().T, sr, subtype=SUBTYPES[encoding])
        n_audio += 1

        rows.append({
            "path": str(rel),
            "source_peak_amplitude": peak,
            "source_rms_dbfs": rms_dbfs(wav),
            "target_peak": target_peak,
            # peak_normalize leaves a digitally silent file alone rather than dividing by
            # zero, so its gain is 1.0 and its peak stays 0 — recorded, not special-cased.
            "gain": target_peak / peak if peak > 0 else 1.0,
            "normalized_rms_dbfs": rms_dbfs(normalized),
        })

        if n_audio % 1000 == 0:
            print(f"  {n_audio} normalised", flush=True)

    pd.DataFrame(rows).to_csv(out_root / "normalization.csv", index=False)

    print(f"\nNormalised {n_audio} audio files, copied {n_copied} others.")
    print(f"Mirror at {out_root}, gains in {out_root / 'normalization.csv'}")


if __name__ == "__main__":
    main()

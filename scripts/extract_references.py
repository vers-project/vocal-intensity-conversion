"""Extract VAD-trimmed, peak-normalised copies of chosen recordings. No model involved.

Selects rows of an annotated metadata CSV by speaker / sentence / pass, trims the leading
and trailing silence with the same Silero settings the render scripts use
(``vic.data.vad.DEFAULT_VAD_PARAMS``), peak-normalises, and writes one file per row named
like the listening set:

    spk<speaker>_sent<sentence>_rep<pass>_src<1-4>.<ext>

Native sample rate by default — there is no codec here, so nothing forces 16 kHz. Pass
``--sample-rate 16000`` to match the converter renders when these are to be A/B'd
against them.

Usage
-----
    uv run --extra cpu scripts/extract_references.py \\
        --metadata     ~/Documents/Datasets/experiments_metadata/all_corpora_prediction_metadata_with_labels_directive_annotations_sentences_test_corrected.csv \\
        --dataset-root ~/Documents/Datasets/AVID/AVID \\
        --output-dir   ~/Documents/references_sent24

Defaults select sentence 24 of speakers 27 and 29, both passes — 16 files.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import soundfile as sf

from audio_utils.data.transforms import load_audio, peak_normalize, resample, select_channel
from vic.data.levels import LEVEL_INDEX
from vic.data.vad import trim_metadata_with_vad

SUBTYPES = {"wav": "PCM_16", "mp3": None, "flac": None}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--speakers", nargs="+", type=float, default=[27, 29])
    parser.add_argument("--sentences", nargs="+", type=float, default=[24])
    parser.add_argument("--passes", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--sample-rate", type=int, default=None,
                        help="Resample to this rate (default: leave the file's own).")
    parser.add_argument("--target-peak", type=float, default=0.9)
    parser.add_argument("--format", choices=sorted(SUBTYPES), default="wav")
    args = parser.parse_args()

    rows = pd.read_csv(args.metadata)
    rows = rows[
        rows["subject_id"].isin(args.speakers)
        & rows["sentence_id"].isin(args.sentences)
        & rows["repetition"].isin(args.passes)
        & rows["level"].notna()
    ].copy()
    if rows.empty:
        raise SystemExit(
            f"Nothing matched speakers={args.speakers} sentences={args.sentences} "
            f"passes={args.passes} in {args.metadata}."
        )
    rows["signal_path"] = rows["signal_path"].map(lambda p: str(args.dataset_root / p))
    print(f"{len(rows)} recording(s) selected")

    # Same VAD, same parameters as the render scripts: head and tail only, never a pause
    # inside the sentence.  It rewrites start_s/end_s, so the read below is the trim.
    rows = trim_metadata_with_vad(rows)
    silent = rows[rows["vad_status"] != "ok"]
    if not silent.empty:
        raise SystemExit(f"The VAD found no speech in {len(silent)} of them.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for row in rows.itertuples():
        wav, sr = load_audio(row.signal_path, row.start_s, row.end_s - row.start_s)
        wav = select_channel(wav, -1 if pd.isna(row.channel) else int(row.channel))
        if args.sample_rate and args.sample_rate != sr:
            wav, sr = resample(wav, sr, args.sample_rate), args.sample_rate
        wav = peak_normalize(wav, target_peak=args.target_peak)

        name = (f"spk{int(row.subject_id)}_sent{int(row.sentence_id)}"
                f"_rep{int(row.repetition)}_src{LEVEL_INDEX[row.level]}.{args.format}")
        path = args.output_dir / name
        subtype = SUBTYPES[args.format]
        sf.write(str(path), wav.numpy().T, sr,
                 format=args.format.upper(), **({"subtype": subtype} if subtype else {}))
        print(f"  {name}  {wav.shape[-1] / sr:5.2f} s  "
              f"(trimmed {row.trimmed_head_s:.2f}s + {row.trimmed_tail_s:.2f}s)")

    print(f"\nWrote {len(rows)} file(s) to {args.output_dir}")


if __name__ == "__main__":
    main()

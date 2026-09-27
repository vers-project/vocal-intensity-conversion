"""Render a hand-picked list of utterances at every target level, for listening.

The narrow sibling of ``scripts/convert_test_set.py``.  That script sweeps a whole split
and writes a metadata table for objective evaluation; this one takes a short CSV naming
individual recordings — "speaker 9, sentence 1, soft, pass 1" — and produces exactly the
files a listening test needs, named so that a human can find any of them without a
manifest.

Per selected utterance, sixteen files:

    8   conversions of the selected recording, one per linearly-spaced target τ
    8   the whole sentence as it was really spoken — all four effort levels, in **both**
        passes, VAD-trimmed (the selected recording is one of these)

Those eight originals are what makes the set worth listening to.  Each conversion asks
the model for a level, and the speaker actually *read* that sentence at all four levels,
twice — so every converted file can be paired with a real utterance of the same words by
the same voice at the requested effort.  Having both passes matters too: two real takes
at one level bound how much of a difference is just take-to-take variation rather than
anything the model did.

Two output directories, because the two questions are incompatible:

    no_normalization/  the decoder's own amplitudes.  MP3 does not clip these — measured
                       at ±3.0 it round-trips the peak with no distortion — so the loudness
                       the model produced is audible and comparable across files.
    peak_normalized/   every file scaled to peak 0.9 individually.  Amplitude then carries
                       no information at all, so a listener has to judge the *voice*.

Naming
------
    spk<speaker>_sent<sentence>_rep<pass>_src<1-4>_tgt<1-8>.mp3   converted
    spk<speaker>_sent<sentence>_rep<pass>_src<1-4>.mp3            originals

``src`` is the recording's own effort rank (1 soft … 4 veryloud) — for a conversion, the
rank of the source it was made from; for an original, its own.  ``rep`` is the pass the
*file* comes from, so two originals differing only in ``rep`` are the two real takes of
one sentence at one level.  ``tgt`` indexes the target sweep and is absent from an
original, where it would mean nothing.

Mapping the selection onto the corpus
-------------------------------------
The selection CSV uses its own column names and they do **not** match the corpus table:
AVID carries ``speaker`` and ``sentence`` columns of its own that are null on every row,
so a match by name would silently find nothing.  ``selection.columns`` states the mapping
explicitly, and every selected row must resolve to exactly one recording in each pass or
the run stops.

Usage
-----
    launch_experiment --config configs/paper/convert_selection.yaml \\
                      --script scripts/convert_selection.py

Config
------
    converter.config_path / ckpt_path : as convert_test_set.py.
    data.metadata_csv / dataset_roots : the annotated corpus table.
    selection.csv        : the list of recordings to render.
    selection.columns    : selection column → corpus column.  Defaults map
                           speaker/sentence/level/pass onto AVID's
                           subject_id/sentence_id/level/repetition.
    vad.*                : Silero parameters; ``enabled: false`` renders untrimmed.
    conversion.n_targets / intensity_min_db / intensity_max_db : the target sweep.
    device, num_workers.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import soundfile as sf
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from vic.data.levels import LEVEL_INDEX
from vic.data.metadata import resolve_paths
from vic.data.transforms import peak_normalize
from vic.data.vad import trim_metadata_with_vad
from vic.training.build import (
    build_conversion_pipelines,
    build_inference_dataset,
    build_labels_module,
    resolve_tau_src,
)
from vic.training.utils import configure_matmul_precision

from experiment_launcher import parse_args

#: The four things a selection row has to say.  Fixed roles; only the column *names* on
#: either side are configurable.
ROLES = ("speaker", "sentence", "level", "pass")

#: role → column in the selection CSV.
DEFAULT_SELECTION_COLUMNS = {role: role for role in ROLES}

#: role → column in the corpus table.  Not the same names, and that is the trap: AVID
#: carries its own ``speaker`` and ``sentence`` columns which are null on every row, so
#: matching by name would find nothing and report an empty selection as success.
DEFAULT_CORPUS_COLUMNS = {
    "speaker": "subject_id",
    "sentence": "sentence_id",
    "level": "level",
    "pass": "repetition",
}

RAW_DIR = "no_normalization"
NORM_DIR = "peak_normalized"


def stem(speaker, sentence, repetition, src_index, target_index=None) -> str:
    """``spk9_sent1_rep1_src1[_tgt3]`` — the whole identity of a file, in its name."""
    name = (f"spk{int(speaker)}_sent{int(sentence)}"
            f"_rep{int(repetition)}_src{int(src_index)}")
    return name if target_index is None else f"{name}_tgt{int(target_index)}"


def save_both(wav: torch.Tensor, output_dir: Path, name: str, sample_rate: int) -> float:
    """Write one waveform to both directories; return its raw peak.

    MP3 rather than WAV because these are for listening, and libsndfile's encoder carries
    amplitudes past full scale intact (verified to ±3.0), so the unnormalised copy stays
    faithful instead of clipping the loud targets — which is the one thing that would have
    made this set useless for the conditions that matter most.
    """
    data = wav.cpu().numpy().T
    peak = float(wav.abs().max())
    for directory, signal in (
        (RAW_DIR, data),
        (NORM_DIR, peak_normalize(wav).cpu().numpy().T),
    ):
        path = output_dir / directory / f"{name}.mp3"
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(path), signal, sample_rate, format="MP3")
    return peak


def resolve_selection(
    selection: pd.DataFrame,
    corpus: pd.DataFrame,
    selection_columns: dict,
    corpus_columns: dict,
) -> pd.DataFrame:
    """Locate each selected recording, and its counterpart in the other pass.

    Returns one row per *file to load*: the selected recording tagged ``is_source``, and
    the same (speaker, sentence, level) from the other pass tagged as a reference.  Both
    go through the same VAD trim and the same loader, so the pair a listener compares
    differs only in which take it is.

    Speaker and sentence are compared as numbers because the two tables disagree on their
    storage — the corpus holds ``9.0``, a hand-written selection holds ``9`` — and a string
    comparison would quietly match nothing.
    """
    missing = [c for c in selection_columns.values() if c not in selection.columns]
    if missing:
        raise ValueError(
            f"The selection CSV has no {missing} column(s). Its columns are "
            f"{list(selection.columns)}; set selection.columns to map them."
        )
    absent = [c for c in corpus_columns.values() if c not in corpus.columns]
    if absent:
        raise ValueError(
            f"The corpus table has no {absent} column(s). Level and pass come from the "
            f"speech-annotator output; point data.metadata_csv at the annotated table."
        )

    rows = []
    for position, item in selection.iterrows():
        level = item[selection_columns["level"]]
        if level not in LEVEL_INDEX:
            raise ValueError(
                f"Selection row {position}: unknown level {level!r}; "
                f"expected one of {list(LEVEL_INDEX)}."
            )
        speaker = float(item[selection_columns["speaker"]])
        sentence = float(item[selection_columns["sentence"]])
        pass_index = int(item[selection_columns["pass"]])
        pass_column = corpus_columns["pass"]

        # The whole sentence: every effort level, in both passes.  The selected recording
        # is one of these; the rest are the real material the conversions are paired
        # against in the perceptual experiment, so they are extracted through exactly the
        # same trim and the same normalisation rather than pulled separately.
        block = corpus[
            (pd.to_numeric(corpus[corpus_columns["speaker"]], errors="coerce") == speaker)
            & (pd.to_numeric(corpus[corpus_columns["sentence"]], errors="coerce") == sentence)
            & corpus[corpus_columns["level"]].isin(LEVEL_INDEX)
        ]
        passes = pd.to_numeric(block[pass_column], errors="coerce")
        selected = block[
            (block[corpus_columns["level"]] == level) & (passes == pass_index)
        ]

        where = f"speaker {speaker:g}, sentence {sentence:g}, {level}, pass {pass_index}"
        if len(selected) != 1:
            raise ValueError(
                f"Selection row {position}: {where} matches {len(selected)} recording(s) "
                f"in the corpus; exactly one is required."
            )

        source_key = (selected.index[0],)
        for index, row in block.iterrows():
            rows.append({
                **row.to_dict(),
                "selection_index": position,
                "is_source": index in source_key,
                # Frozen here so a file's name cannot drift from the row it came from
                # once the table is reindexed by the trim.  Level and pass are the ROW's
                # own, not the selection's — the block spans all four levels.
                "out_speaker": int(speaker),
                "out_sentence": int(sentence),
                "out_repetition": int(row[pass_column]),
                "out_src_index": LEVEL_INDEX[row[corpus_columns["level"]]],
            })

    out = pd.DataFrame(rows)
    # Two selections naming the same (speaker, sentence) would each pull the whole block,
    # so the originals would be written twice under one name.  The source flag survives
    # the merge: a row selected by any one of them is a source.
    out["is_source"] = out.groupby(
        ["out_speaker", "out_sentence", "out_repetition", "out_src_index"]
    )["is_source"].transform("any")
    out = out.drop_duplicates(
        subset=["out_speaker", "out_sentence", "out_repetition", "out_src_index"]
    )
    return out.reset_index(drop=True)


@parse_args
def main(config: dict, output_dir: Path):

    configure_matmul_precision()
    output_dir.mkdir(parents=True, exist_ok=True)

    cc = config.get("conversion", {})
    train_config = OmegaConf.to_container(
        OmegaConf.load(config["converter"]["config_path"]), resolve=True
    )
    ckpt_path = config["converter"]["ckpt_path"]
    train_config["data"] = {**train_config["data"], **config.get("data", {})}

    device = torch.device(
        config.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}\nCheckpoint: {ckpt_path}")

    source, _ = resolve_tau_src(train_config)
    pipeline, label_pipeline = build_conversion_pipelines(train_config)
    module = build_labels_module(
        train_config, pipeline, label_pipeline, ckpt_path=ckpt_path
    ).to(device).eval()
    sr = pipeline.sample_rate

    # ------------------------------------------------------------------
    # Corpus and selection.
    # ------------------------------------------------------------------
    corpus = pd.read_csv(train_config["data"]["metadata_csv"])
    corpus = resolve_paths(
        corpus, train_config["data"]["dataset_roots"], path_columns=["signal_path"]
    )

    sc = config["selection"]
    selection = pd.read_csv(sc["csv"])
    selection_columns = {**DEFAULT_SELECTION_COLUMNS, **(sc.get("columns") or {})}
    corpus_columns = {**DEFAULT_CORPUS_COLUMNS, **(sc.get("corpus_columns") or {})}
    print(f"{len(selection)} utterance(s) selected from {sc['csv']}")
    print("   mapping: " + ", ".join(
        f"{selection_columns[role]} → {corpus_columns[role]}" for role in ROLES))

    metadata = resolve_selection(
        selection, corpus, selection_columns, corpus_columns
    )
    per_pass = metadata.groupby(
        ["out_speaker", "out_sentence", "out_repetition"])["out_src_index"].nunique()
    print(f"   {len(metadata)} original(s): "
          f"{per_pass.index.droplevel(2).nunique()} sentence(s) x "
          f"{sorted(per_pass.unique())} level(s) x "
          f"{metadata['out_repetition'].nunique()} pass(es)")
    short = per_pass[per_pass < len(LEVEL_INDEX)]
    if len(short):
        # Reported, not refused: a missing level costs that one pairing, and the rest of
        # the sentence is still usable material for the experiment.
        print(f"   NOTE: {len(short)} (sentence, pass) group(s) have fewer than "
              f"{len(LEVEL_INDEX)} levels — {dict(short)}")

    vad_cfg = dict(config.get("vad") or {})
    if vad_cfg.pop("enabled", True):
        metadata = trim_metadata_with_vad(metadata, vad_cfg)
        silent = metadata[metadata["vad_status"] != "ok"]
        if not silent.empty:
            # A hand-picked list is small enough that every row matters; dropping one
            # would leave a gap in the set with nothing to say so.
            raise ValueError(
                f"The VAD found no speech in {len(silent)} selected recording(s): "
                f"{[stem(r.out_speaker, r.out_sentence, r.out_repetition, r.out_src_index) for r in silent.itertuples()]}. "
                f"Check the channel, or set vad.enabled: false."
            )
        cut = metadata["trimmed_head_s"] + metadata["trimmed_tail_s"]
        print(f"VAD: {cut.sum():.1f} s of silence removed over {len(metadata)} files "
              f"(median {cut.median():.2f} s each)")

    dataset, collate_fn = build_inference_dataset(
        metadata, train_config, source, pipeline
    )
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False,
        num_workers=config.get("num_workers", 4),
        pin_memory=True, collate_fn=collate_fn,
    )

    gen = train_config.get("generation", {})
    fallback = train_config["training"]["intensity_range_db"]
    lo = float(cc.get("intensity_min_db", gen.get("intensity_min_db", fallback[0])))
    hi = float(cc.get("intensity_max_db", gen.get("intensity_max_db", fallback[1])))
    n_targets = int(cc.get("n_targets", 8))
    targets_db = torch.linspace(lo, hi, n_targets)
    print(f"{n_targets} targets over [{lo:.1f}, {hi:.1f}] dB: "
          + ", ".join(f"{t:.1f}" for t in targets_db.tolist()))

    # ------------------------------------------------------------------
    # Render.
    # ------------------------------------------------------------------
    written, peaks = [], []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            meta = metadata.iloc[i]
            name = stem(meta["out_speaker"], meta["out_sentence"],
                        meta["out_repetition"], meta["out_src_index"])

            wav = batch["wav"].unbatch()[0]
            peaks.append(save_both(wav, output_dir, name, sr))
            written.append(name)

            if not meta["is_source"]:
                continue                     # the other pass is a reference, not a source

            z_real = pipeline.encode(batch["wav"], training=False)
            for k, tgt in enumerate(targets_db, start=1):
                tau = tgt.unsqueeze(0).to(device)
                wav_fake = pipeline.extractor.decode(
                    module.convert(z_real, tau, batch)
                )
                target_name = stem(meta["out_speaker"], meta["out_sentence"],
                                   meta["out_repetition"], meta["out_src_index"], k)
                peaks.append(save_both(wav_fake.unbatch()[0], output_dir, target_name, sr))
                written.append(target_name)

    n_sources = int(metadata["is_source"].sum())
    expected = n_sources * n_targets + len(metadata)
    print(f"\nWrote {len(written)} file(s) to each of "
          f"{output_dir/RAW_DIR} and {output_dir/NORM_DIR}")
    print(f"   {n_sources * n_targets} converted ({n_sources} source(s) x {n_targets} "
          f"targets) + {len(metadata)} real takes")
    if len(written) != expected:
        raise RuntimeError(f"expected {expected} files, wrote {len(written)}")
    if len(set(written)) != len(written):
        raise RuntimeError("two files share a name — the selection is not unique")

    above = sum(1 for p in peaks if p > 1.0)
    print(f"   peak amplitude: max {max(peaks):.2f}, {above} file(s) above 1.0 "
          f"(carried intact in {RAW_DIR}, scaled in {NORM_DIR})")


if __name__ == "__main__":
    main()

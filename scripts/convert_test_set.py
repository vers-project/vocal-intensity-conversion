"""Render a trained converter over a whole test split, at two families of target intensity.

For every sentence of the split this
  1. trims the leading and trailing silence with the Silero VAD — never anything inside
     the sentence — because the converter fills long pre-onset pauses with a buzz, and a
     listening test or an objective metric computed over that buzz measures the pause,
     not the conversion;
  2. converts the trimmed sentence to two families of target τ:

     ``linspace``  ``n_targets`` (8) linearly-spaced levels over the τ range, exactly as
                   ``IntensityConversionCallback`` draws them.  A dense, evenly-spaced
                   sweep of the conditioning axis, indexed 1..8 in ``linspace_index``.
     ``level``     the measured level of *every recording in this sentence's own group* —
                   the same sentence, same speaker, read at soft / normal / loud /
                   veryloud, indexed 1..4 in ``level_index``.  This is the family that
                   makes an objective metric possible: the destination of each conversion
                   is a real recording, named in ``reference_utt_id``, so converted and
                   real can be compared at the same level rather than against a model of
                   one.  Including the row's own level, which is the identity case and the
                   only confound-free artefact check in the design.

  3. writes every waveform **unnormalised**, as 32-bit float WAV, so the amplitude the
     decoder actually produced survives to be measured;
  4. writes one metadata row per output file.

Peak normalisation is a *separate* step — ``scripts/normalize_audio_dir.py`` mirrors an
output directory with everything normalised — because the two uses are incompatible.  A
listening test must be normalised, or amplitude alone gives the answer away; a
measurement of "what level did the model output" must not be, and within a sentence group
the real recordings share a calibration constant, so their dBFS differences *are* SPL
differences.  16-bit PCM would clip a converted loud target on top of that.

What it reuses, and why that is the point
-----------------------------------------
The model is assembled by :func:`vic.training.build.build_labels_module` — the same
function ``scripts/train_converter_cgan_v2_labels.py`` calls — from the run's own
training YAML, with C_θ's weights and the *fitted* label scaler taken from the
checkpoint.  Conversion is ``module.convert``, the source level is ``module.source_tau``
and the achieved level is ``module.estimate_tau_src``: the three measurement paths that
produced every number in the training logs, called here rather than reimplemented.  The
VAD trim moves ``start_s``/``end_s`` in the metadata rather than cutting the waveform, so
``IntensityDataset`` computes its calibrated frame labels over the trimmed span and
τ_src stays aligned with the audio being converted (see ``vic/data/vad.py``).

Two passes over the data, not one
---------------------------------
A level target is *another row's* τ_src, so every source level in the split has to be
known before the first conversion is made.  Pass 1 measures τ_src for every row and keeps
it; pass 2 converts.  τ_src is therefore computed exactly once and the two passes cannot
disagree about it.  Under ``tau_src.source: labels`` pass 1 touches no model at all — the
source level is the calibrated frame labels, and the cost is one read of the audio.

Conditions
----------
    ``real``       the VAD-trimmed source recording
    ``codec``      that source encoded and decoded with no conversion — the anchor every
                   measure is read against, so that the codec's own damage is excluded
                   from a converter's score
    ``converted``  the model's output, one row per target

Usage
-----
    launch_experiment --config path/to/convert_test_set.yaml \\
                      --script scripts/convert_test_set.py

Config
------
    converter.config_path  : the converter run's own training YAML.  Everything about
                             the model, the codec and P_φ is read from it, so nothing
                             here can drift from the run being rendered.
    converter.ckpt_path    : that run's checkpoint.

    data.*                 : overrides merged over the training config's ``data`` block.
                             The render normally points at a *different* metadata CSV
                             from the one training used — the one carrying the hand
                             annotation — so ``metadata_csv`` is usually set here.
    data.split             : which split to render (default ``test``).

    vad.*                  : Silero parameters, see ``vic.data.vad.DEFAULT_VAD_PARAMS``.
                             ``vad.enabled: false`` renders the untrimmed segments.

    levels.enabled         : render the sentence-group family (default true).
    levels.level_column,
    levels.group_columns,
    levels.sentence_column,
    levels.exclude_column  : where the annotation lives, see ``vic/data/levels.py``.
    levels.sentence_ids    : (optional) render only these sentence numbers, whole
                             sentences at every level.  Absent renders them all.  A
                             number matching no row is refused rather than skipped.

    conversion.n_targets        : how many linspace τ per sentence (default 8).
    conversion.intensity_min_db,
    conversion.intensity_max_db : the linspace bounds.  Default to the training config's
                             ``generation`` block — the range whose renders were listened
                             to during the run — and fall back to
                             ``training.intensity_range_db``.
    conversion.save_codec       : write the round-trip condition (default true).

    device                 : ``cuda`` / ``cpu`` (default: cuda when available).
    num_workers            : dataloader workers (default 4).

Outputs (in ``--output-dir``)
-----------------------------
    real/<utt>.wav                          VAD-trimmed source, unnormalised
    codec/<utt>.wav                         encode → decode, no conversion
    converted/<utt>_lin<k>_tgt<NN.N>dB.wav  linspace family
    converted/<utt>_lvl<k><name>_tgt<NN.N>dB.wav
                                            sentence-group family
    conversion_metadata.csv                 one row per written file
    vad_trim.csv                            the trimmed segment index, with what was cut
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from audio_utils.data.transforms import rms_dbfs
from vic.data.levels import (
    DEFAULT_EXCLUDE_COLUMN,
    DEFAULT_GROUP_COLUMNS,
    DEFAULT_LEVEL_COLUMN,
    DEFAULT_SENTENCE_COLUMN,
    LEVELS,
    level_targets,
    prepare_level_groups,
)
from vic.data.metadata import resolve_paths
from vic.data.vad import TRIM_COLUMNS, trim_metadata_with_vad
from vic.training.build import (
    build_conversion_pipelines,
    build_inference_dataset,
    build_labels_module,
    resolve_tau_src,
)
from vic.training.utils import configure_matmul_precision

from experiment_launcher import parse_args


def save_unnormalized(wav: torch.Tensor, path: Path, sample_rate: int) -> dict:
    """Write ``wav`` (1, T) as 32-bit float WAV and report what was written.

    Float, not the 16-bit PCM the rest of the repo saves for listening: these files exist
    to have their amplitude measured, and a converter asked for a loud target can decode
    past ±1.0, which 16-bit PCM would clip — silently turning an overshoot the paper
    wants to report into a distortion it does not.

    Written with soundfile rather than ``torchaudio.save``, which in torchaudio 2.11
    dispatches to TorchCodec's AudioEncoder and warns that ``encoding`` and
    ``bits_per_sample`` "are not fully supported" — then writes 16-bit PCM anyway.  A
    warning on stderr is not a defence for a file that was supposed to be measurable, and
    ``tests/test_convert_test_set.py`` pins the round trip past full scale.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), wav.cpu().numpy().T, sample_rate, subtype="FLOAT")
    return {
        "peak_amplitude": float(wav.abs().max()),
        "rms_dbfs": rms_dbfs(wav),
        "duration_s": wav.shape[-1] / sample_rate,
        "n_samples": int(wav.shape[-1]),
    }


def target_range(config: dict, train_config: dict) -> tuple[float, float]:
    """The linspace bounds, from this config, else the run's ``generation`` block.

    The generation block is the default rather than ``intensity_range_db`` because it is
    the range the run's own listening renders were made over, so a render produced here
    is comparable to the ones already judged by ear.  ``intensity_range_db`` is wider — it
    is the LabelScaler's fitting range, which includes τ the converter saw least often.
    """
    cc = config.get("conversion", {})
    gen = train_config.get("generation", {})
    fallback = train_config["training"]["intensity_range_db"]
    lo = cc.get("intensity_min_db", gen.get("intensity_min_db", fallback[0]))
    hi = cc.get("intensity_max_db", gen.get("intensity_max_db", fallback[1]))
    return float(lo), float(hi)


def make_utt_ids(metadata: pd.DataFrame) -> list[str]:
    """A stable, unique, filesystem-safe id per source sentence: ``<row>_<file stem>``.

    The row index is first because a segment index gives many rows the same stem, and
    because it keeps the rendered files in metadata order in a directory listing.  It is
    the position within the already-filtered split, so it is stable for a given metadata
    CSV and split and changes if either does — which is why the CSV also carries
    ``signal_path``/``start_s``/``end_s`` as the real identity.
    """
    return [
        f"{i:05d}_{Path(path).stem}"
        for i, path in enumerate(metadata["signal_path"])
    ]


@parse_args
def main(config: dict, output_dir: Path):

    configure_matmul_precision()
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # The run being rendered defines the model, the codec and P_φ.  Only `data` may be
    # overridden from here, and it is merged rather than replaced so a render can retarget
    # the metadata CSV without restating window_length.
    # ------------------------------------------------------------------
    cc = config.get("conversion", {})
    train_config = OmegaConf.to_container(
        OmegaConf.load(config["converter"]["config_path"]), resolve=True
    )
    ckpt_path = config["converter"]["ckpt_path"]
    train_config["data"] = {**train_config["data"], **config.get("data", {})}

    split = config.get("data", {}).get("split", "test")
    device = torch.device(
        config.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")
    print(f"Config: {config['converter']['config_path']}")
    print(f"Checkpoint: {ckpt_path}")

    # ------------------------------------------------------------------
    # Model, built by the training script's own builder.
    # ------------------------------------------------------------------
    source, _ = resolve_tau_src(train_config)
    pipeline, label_pipeline = build_conversion_pipelines(train_config)
    module = build_labels_module(
        train_config, pipeline, label_pipeline, ckpt_path=ckpt_path
    )
    module = module.to(device).eval()
    sr = pipeline.sample_rate

    # ------------------------------------------------------------------
    # Corpus.
    # ------------------------------------------------------------------
    metadata = pd.read_csv(train_config["data"]["metadata_csv"])
    metadata = resolve_paths(
        metadata, train_config["data"]["dataset_roots"], path_columns=["signal_path"]
    )
    metadata = metadata[metadata["split"] == split].reset_index(drop=True)
    if metadata.empty:
        raise ValueError(
            f"No rows with split == {split!r} in "
            f"{train_config['data']['metadata_csv']}."
        )
    print(f"{len(metadata)} rows in split {split!r}.")

    # ------------------------------------------------------------------
    # Sentence groups.  Done before the VAD so the excluded rows are never read.
    # ------------------------------------------------------------------
    lc = config.get("levels") or {}
    use_levels = lc.get("enabled", True)
    level_column = lc.get("level_column", DEFAULT_LEVEL_COLUMN)
    sentence_ids = lc.get("sentence_ids")
    if sentence_ids is not None and not use_levels:
        # The selection is stated per sentence, and without the annotation there are no
        # sentence numbers to select on. Refuse rather than render the whole split.
        raise ValueError(
            "levels.sentence_ids was given with levels.enabled: false. The sentence "
            "numbers come from the level annotation, so there is nothing to select on "
            "with it off."
        )
    if use_levels:
        n_before = len(metadata)
        metadata = prepare_level_groups(
            metadata,
            level_column=level_column,
            group_columns=lc.get("group_columns", list(DEFAULT_GROUP_COLUMNS)),
            exclude_column=lc.get("exclude_column", DEFAULT_EXCLUDE_COLUMN),
            sentence_column=lc.get("sentence_column", DEFAULT_SENTENCE_COLUMN),
            sentence_ids=sentence_ids,
        )
        sizes = metadata.groupby("sentence_group_id")["group_n_levels"].first()
        selected = (f", restricted to sentences {sorted(int(s) for s in sentence_ids)}"
                    if sentence_ids is not None else "")
        print(
            f"Levels: {len(metadata)}/{n_before} rows kept"
            f"{selected}; {len(sizes)} sentence groups, "
            f"{int((sizes == len(LEVELS)).sum())} of them complete with "
            f"{len(LEVELS)} levels."
        )
    else:
        print("Levels disabled: rendering the linspace family only.")

    # ------------------------------------------------------------------
    # The VAD trim, which is a rewrite of start_s/end_s — so everything downstream (the
    # seek, the calibrated frame labels, τ_src) follows automatically.
    # ------------------------------------------------------------------
    vad_cfg = dict(config.get("vad") or {})
    if vad_cfg.pop("enabled", True):
        metadata = trim_metadata_with_vad(metadata, vad_cfg)
        n_empty = int((metadata["vad_status"] == "no_speech").sum())
        kept = metadata[metadata["vad_status"] == "ok"].reset_index(drop=True)
        cut = metadata["trimmed_head_s"] + metadata["trimmed_tail_s"]
        print(
            f"VAD: {len(kept)}/{len(metadata)} rows retained, "
            f"{n_empty} with no detectable speech (dropped). "
            f"Silence removed: {cut.sum():.0f} s total, "
            f"median {cut.median():.2f} s per sentence "
            f"(head {metadata['trimmed_head_s'].median():.2f} s, "
            f"tail {metadata['trimmed_tail_s'].median():.2f} s)."
        )
        if kept.empty:
            # Every row silent means the VAD is looking at the wrong channel, or the
            # metadata's spans do not point where they claim. Say so here rather than
            # produce an empty render and a KeyError in the report.
            raise ValueError(
                f"The VAD found no speech in any of the {len(metadata)} rows of split "
                f"{split!r}. Check that data.dataset_roots resolves to the right audio "
                f"and that the metadata's `channel` column selects the speech track "
                f"(AVID's channel 1 is the electroglottograph), or set vad.enabled: "
                f"false to render the segments untrimmed."
            )
        metadata = kept
    else:
        print("VAD disabled: rendering the untrimmed segments.")

    metadata["utt_id"] = make_utt_ids(metadata)
    metadata.to_csv(output_dir / "vad_trim.csv", index=False)

    # A group loses a member if the VAD dropped it, so the destinations are recomputed
    # from the rows that survived rather than from the ones that were annotated.
    targets_by_row = level_targets(metadata) if use_levels else {}

    dataset, collate_fn = build_inference_dataset(
        metadata, train_config, source, pipeline
    )
    loader_kw = dict(
        batch_size=1,        # full sentences of unequal length, one output file each
        shuffle=False,
        num_workers=config.get("num_workers", 4),
        pin_memory=True,
        collate_fn=collate_fn,
    )

    n_targets = int(cc.get("n_targets", 8))
    lo, hi = target_range(config, train_config)
    linspace_db = torch.linspace(lo, hi, n_targets)
    save_codec = bool(cc.get("save_codec", True))
    cond_range = train_config["training"]["intensity_range_db"]
    print(
        f"{n_targets} linspace targets over [{lo:.1f}, {hi:.1f}] dB: "
        + ", ".join(f"{t:.1f}" for t in linspace_db.tolist())
    )

    # ------------------------------------------------------------------
    # Pass 1 — τ_src for every row.  A level target is another row's τ_src, so the whole
    # split has to be measured before the first conversion is made.  Under
    # `tau_src.source: labels` this touches no model: the source level is the calibrated
    # frame labels, and the cost is one read of the audio.
    # ------------------------------------------------------------------
    tau_src_all = np.empty(len(dataset), dtype=np.float64)
    with torch.no_grad():
        for i, batch in enumerate(DataLoader(dataset, **loader_kw)):
            batch = {k: v.to(device) for k, v in batch.items()}
            tau_src_all[i] = float(module.source_tau(batch)[0])
    print(f"Pass 1: τ_src measured over {len(dataset)} sentences "
          f"(mean {tau_src_all.mean():.2f} dB, "
          f"range {tau_src_all.min():.1f}–{tau_src_all.max():.1f} dB).")

    # Carry every corpus column through onto each output row, so the rendered set is
    # self-describing and downstream analysis never has to re-join against the corpus.
    #
    # The source's own level is carried under `source_level` / `source_level_index`, NOT
    # under the corpus's own column names.  A conversion row already has a `level_index`
    # — the *destination's* — and carrying the source's under the same name would have let
    # the destination silently overwrite it, leaving a 4x4 heatmap with no source axis.
    source_level_columns = [level_column, "level_index"] if use_levels else []
    carried = [
        c for c in metadata.columns
        if c not in TRIM_COLUMNS + ["utt_id"] + source_level_columns
    ]
    utt_ids = metadata["utt_id"].tolist()

    def relative(condition: str, name: str) -> str:
        return f"{condition}/{name}.wav"

    rows: list[dict] = []

    # ------------------------------------------------------------------
    # Pass 2 — convert and write.
    # ------------------------------------------------------------------
    with torch.no_grad():
        for i, batch in enumerate(DataLoader(dataset, **loader_kw)):
            batch = {k: v.to(device) for k, v in batch.items()}
            wav_batch = batch["wav"]
            meta = metadata.iloc[i]
            src_utt = utt_ids[i]
            base = {c: meta[c] for c in carried}

            z_real = pipeline.encode(wav_batch, training=False)

            # Two readings of the source, both kept: source_tau is what the conditioning
            # was built on (the calibrated Leq under `labels`), estimate_tau_src is P_φ's
            # reading of the same audio.  Their difference is the ruler bias, per
            # sentence — the quantity that decides whether a reproducible conversion
            # offset belongs to the converter or to the instrument measuring it.
            tau_src = float(tau_src_all[i])
            tau_src_pred = float(module.estimate_tau_src(wav_batch)[0])

            # The codec's own contribution: the *unconverted* latent through the same
            # decode → whiten → re-encode → P_φ path every converted reading takes.
            wav_codec = pipeline.extractor.decode(z_real)
            tau_codec = float(module.estimate_tau_src(wav_codec)[0])

            shared = {
                **base,
                "source_utt_id": src_utt,
                "source_signal_path": relative("real", src_utt),
                "source_level": meta[level_column] if use_levels else pd.NA,
                "source_level_index": meta["level_index"] if use_levels else pd.NA,
                "tau_src_db": tau_src,
                "tau_src_pred_db": tau_src_pred,
                "tau_codec_db": tau_codec,
            }
            empty = {
                "target_kind": pd.NA, "linspace_index": pd.NA,
                "level_index": pd.NA, "level_name": pd.NA,
                "reference_utt_id": pd.NA, "reference_signal_path": pd.NA,
                "tau_tgt_db": pd.NA, "tau_tgt_in_range": pd.NA,
                "error_db": pd.NA, "delta_tau_db": pd.NA,
            }

            stats = save_unnormalized(
                wav_batch.unbatch()[0], output_dir / "real" / f"{src_utt}.wav", sr
            )
            rows.append({
                **shared, **empty,
                "utt_id": f"{src_utt}_real",
                "signal_path": relative("real", src_utt),
                "condition": "real",
                "tau_pred_db": tau_src_pred,
                **stats,
            })

            if save_codec:
                codec_stats = save_unnormalized(
                    wav_codec.unbatch()[0], output_dir / "codec" / f"{src_utt}.wav", sr
                )
                rows.append({
                    **shared, **empty,
                    "utt_id": f"{src_utt}_codec",
                    "signal_path": relative("codec", src_utt),
                    "condition": "codec",
                    "tau_pred_db": tau_codec,
                    "delta_tau_db": 0.0,
                    **codec_stats,
                })

            # --- the two target families -------------------------------------------
            # (name, tau, extra columns) triples, converted by one loop so that a
            # linspace target and a level target cannot drift apart in how they are
            # measured or written.
            jobs: list[tuple[str, float, dict]] = []

            for k, tgt in enumerate(linspace_db.tolist(), start=1):
                jobs.append((
                    f"{src_utt}_lin{k}_tgt{tgt:05.1f}dB",
                    tgt,
                    {"target_kind": "linspace", "linspace_index": k},
                ))

            for target in targets_by_row.get(i, []):
                tau_tgt = float(tau_src_all[target.row])
                ref_utt = utt_ids[target.row]
                jobs.append((
                    f"{src_utt}_lvl{target.index}{target.name}_tgt{tau_tgt:05.1f}dB",
                    tau_tgt,
                    {
                        "target_kind": "level",
                        "level_index": target.index,
                        "level_name": target.name,
                        # The real recording this conversion is to be compared against.
                        # Naming it per row is what makes "real vs converted at the same
                        # level" a lookup instead of a reconstruction.
                        "reference_utt_id": f"{ref_utt}_real",
                        "reference_signal_path": relative("real", ref_utt),
                    },
                ))

            for name, tau_value, extra in jobs:
                tau_tgt = torch.tensor([tau_value], device=device)
                z_fake = module.convert(z_real, tau_tgt, batch)
                wav_fake = pipeline.extractor.decode(z_fake)

                tau_pred = float(module.estimate_tau_src(wav_fake)[0])
                stats = save_unnormalized(
                    wav_fake.unbatch()[0], output_dir / "converted" / f"{name}.wav", sr
                )
                rows.append({
                    **shared, **empty, **extra,
                    "utt_id": name,
                    "signal_path": relative("converted", name),
                    "condition": "converted",
                    "tau_tgt_db": tau_value,
                    # A few real levels sit outside the range the LabelScaler was fitted
                    # on. They are rendered rather than clamped — a silent clamp would
                    # report an error the converter was never asked to make — and flagged
                    # so the analysis can see them.
                    "tau_tgt_in_range": bool(cond_range[0] <= tau_value <= cond_range[1]),
                    "tau_pred_db": tau_pred,
                    # Signed, both of them.  Grouping conversions by |Δτ| merges "convert
                    # down by 15 dB" with "convert up by 15 dB", and on this model those
                    # are not the same task — see IntensityEvaluationCallback.
                    "error_db": tau_pred - tau_value,
                    "delta_tau_db": tau_value - tau_src,
                    **stats,
                })

            if (i + 1) % 50 == 0 or i + 1 == len(dataset):
                print(f"  {i + 1}/{len(dataset)} sentences rendered "
                      f"({len(rows)} files)", flush=True)

    out = pd.DataFrame(rows)
    if out["utt_id"].duplicated().any():
        # speech-eval's manifest reader refuses a duplicate utt_id, and so should this:
        # a collision means two different renders overwrote one file.
        clash = sorted(out.loc[out["utt_id"].duplicated(), "utt_id"])[:5]
        raise RuntimeError(f"Duplicate utt_id in the render: {clash}")
    out.to_csv(output_dir / "conversion_metadata.csv", index=False)

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    conv = out[out["condition"] == "converted"]
    src = out[out["condition"] == "real"]
    lin = conv[conv["target_kind"] == "linspace"]
    lvl = conv[conv["target_kind"] == "level"]

    print(f"\nWrote {len(out)} files from {len(dataset)} sentences to {output_dir}")
    print(f"  real {len(src)}   codec {int((out['condition'] == 'codec').sum())}   "
          f"converted {len(conv)}  ({len(lin)} linspace + {len(lvl)} level)")
    print(f"  τ_src (conditioning):  mean {src['tau_src_db'].mean():.2f} dB, "
          f"sd {src['tau_src_db'].std():.2f} dB")
    ruler = src["tau_src_pred_db"] - src["tau_src_db"]
    codec = src["tau_codec_db"] - src["tau_src_pred_db"]
    # The sd is the point of the round-trip line: a mean cannot tell "the ruler is
    # uniformly 3 dB short" from "1 dB short here and 7 dB short there", and only the
    # second explains a per-utterance conversion offset that is stable across epochs.
    print(f"  ruler bias P_φ − src:  mean {ruler.mean():+.2f} dB, sd {ruler.std():.2f} dB")
    print(f"  codec round-trip bias: mean {codec.mean():+.2f} dB, sd {codec.std():.2f} dB")

    if not lin.empty:
        print("\n  linspace family — achieved − requested, by target:")
        for k, tgt in enumerate(linspace_db.tolist(), start=1):
            cell = lin[lin["linspace_index"] == k]
            print(f"    {k}  τ_tgt={tgt:5.1f} dB   n={len(cell):<5} "
                  f"error {cell['error_db'].mean():+6.2f} ± "
                  f"{cell['error_db'].std():5.2f} dB")

    if not lvl.empty:
        print("\n  level family — achieved − requested, by destination level:")
        for name in ("soft", "normal", "loud", "veryloud"):
            cell = lvl[lvl["level_name"] == name]
            if cell.empty:
                continue
            print(f"    {int(cell['level_index'].iloc[0])}  {name:<9} n={len(cell):<5} "
                  f"τ_tgt {cell['tau_tgt_db'].mean():5.1f} dB   "
                  f"error {cell['error_db'].mean():+6.2f} ± "
                  f"{cell['error_db'].std():5.2f} dB")
        # The identity cases: asked to convert to the level the recording already has.
        # The only confound-free artefact check in the design — changing nothing should
        # change nothing — so it is worth reading before anything else in this block.
        identity = lvl[lvl["level_index"] == lvl["source_level_index"]]
        if not identity.empty:
            print(f"    identity (target = own level): n={len(identity)}  "
                  f"error {identity['error_db'].mean():+.2f} ± "
                  f"{identity['error_db'].std():.2f} dB")
        outside = int((~lvl["tau_tgt_in_range"].astype(bool)).sum())
        if outside:
            print(f"    {outside} level targets fall outside the conditioning range "
                  f"[{cond_range[0]:.1f}, {cond_range[1]:.1f}] dB — rendered, not clamped.")

    clipped = int((conv["peak_amplitude"] > 1.0).sum())
    if clipped:
        print(f"\n  {clipped} converted files peak above 1.0 — they are intact here "
              f"(32-bit float) but will be scaled by normalize_audio_dir.py.")
    print(f"\nWrote {output_dir}/conversion_metadata.csv and vad_trim.csv")


if __name__ == "__main__":
    main()

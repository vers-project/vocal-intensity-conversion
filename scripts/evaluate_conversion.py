"""Measure the converter against real recordings at the same vocal effort. No audio kept.

Every test utterance is converted to the levels of its own sentence group — the levels the
same speaker really read the same words at — and both sides are measured with
``speech_eval``.  So each comparison is converted-vs-real at equal τ, on matched content in
one voice, which is the strongest reference the corpus offers.

Per source recording, six utterances are rendered and measured (see
``vic/evaluation/conditions.py``):

    real        the VAD-trimmed recording, at the codec's rate — the exact tensor the
                converter is given, so real and converted differ only by the model
    codec       that tensor encoded and decoded, no conversion: the anchor that says what
                the NAC alone costs, so a degradation is attributable
    converted   one per level in the sentence group, including the identity case

**Nothing is written to disk but numbers.**  A source is rendered, handed to the metric
harness, and dropped; on the full AVID test split that is ~7 200 utterances measured and
none stored.

Two passes over the data
------------------------
A level target is *another row's* τ_src, so the whole split's source levels must be known
before the first conversion.  Pass 1 measures τ_src for every row (under
``tau_src.source: labels`` this touches no model at all — it is the calibrated frame
labels); pass 2 renders and measures.  τ_src is computed exactly once, so the passes cannot
disagree.

Outputs (in ``--output-dir``)
-----------------------------
    metrics.csv     one row per measured utterance: identity, τ, every metric column
    pairs.csv       one row per converted/codec utterance beside the real take it
                    targeted, with ``<metric>.reference`` and ``<metric>.delta``
    artifacts.npz   arrays too large for a cell — F0 contours — keyed ``<utt_id>|<column>``

Both tables are rewritten every ``checkpoint_every`` sources, and a rerun into the same
directory skips sources already measured: this is a multi-hour job and losing it to a
walltime kill would be expensive.

Usage
-----
    launch_experiment --config configs/paper/evaluate_conversion_wavlm.yaml \\
                      --script scripts/evaluate_conversion.py

Config
------
    converter.config_path / ckpt_path : the run to evaluate, as in convert_test_set.py.
    data.*        : overrides merged over the training config's ``data`` block.
    levels.*      : the sentence-group annotation, see ``vic/data/levels.py``.
                    ``levels.sentence_ids`` narrows the run to a few sentences.
    vad.*         : Silero parameters; ``enabled: false`` measures untrimmed segments.
    metrics       : a list of ``speech_eval`` metric specs, e.g. ``{type: f0_praat}``.
    evaluation.batch_size      : utterances per metric call.
    evaluation.checkpoint_every: sources between table rewrites.
    evaluation.wer_normalizer  : text normaliser for the word-error counts.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from speech_eval import metrics as _speech_eval_metrics  # noqa: F401  (registers builtins)
from speech_eval.core import available_metrics, build_metrics

from vic.data.levels import (
    DEFAULT_EXCLUDE_COLUMN,
    DEFAULT_GROUP_COLUMNS,
    DEFAULT_LEVEL_COLUMN,
    DEFAULT_SENTENCE_COLUMN,
    level_targets,
    prepare_level_groups,
)
from vic.data.metadata import resolve_paths
from vic.data.vad import trim_metadata_with_vad
from vic.evaluation import (
    CARRY_COLUMNS,
    ConversionRenderer,
    IntensityPredictorMetric,
    MetricHarness,
    add_speaker_similarity,
    add_word_errors,
    source_records,
    stems,
)
from vic.evaluation.pairing import pair_with_references
from vic.training.build import (
    build_conversion_pipelines,
    build_inference_dataset,
    build_labels_module,
    resolve_tau_src,
)
from vic.training.utils import configure_matmul_precision

from experiment_launcher import parse_args

#: Identity columns carried from the corpus onto every results row, when present.
def progress(iterable, desc: str, total: int | None = None, enabled: bool = True):
    """A progress bar paced for a log file, not a terminal.

    SLURM captures stderr to a file, where tqdm's default 0.1 s refresh writes tens of
    thousands of carriage-returned updates and turns a useful log into a 50 MB smear.  On
    a non-TTY the refresh drops to 30 s, which still gives a rate and an ETA — the two
    things worth having on a job whose runtime was, until it was measured, a guess — while
    costing a couple of hundred lines over a two-hour run.
    """
    if not enabled:
        return iterable
    return tqdm(
        iterable,
        desc=desc,
        total=total,
        unit="src",
        mininterval=0.5 if sys.stderr.isatty() else 30.0,
    )


def report_instrument(block: str, inherited: dict, declared: dict) -> None:
    """Say whether the declared instrument is the run's own, and how it differs if not.

    Measuring with a different predictor is legitimate and sometimes better, but it stops
    ``intensity.leq_db`` being comparable to that run's ``val/rmse_pred_db``.  That is a
    change of meaning, so it is reported at every launch rather than left to whoever next
    diffs two YAML files.
    """
    if not inherited:
        print(f"[config] {block}: declared here; the converter run has no such block")
        return
    keys = sorted(set(inherited) | set(declared))
    differing = [k for k in keys if inherited.get(k) != declared.get(k)]
    if not differing:
        print(f"[config] {block}: declared here, identical to the converter run's")
        return
    print(f"[config] {block}: declared here and DIFFERENT from the converter run's — "
          f"intensity.leq_db is not comparable to that run's val/rmse_pred_db")
    for key in differing:
        print(f"           {key}: run={inherited.get(key)!r} -> eval={declared.get(key)!r}")


def load_previous(output_dir: Path) -> tuple[pd.DataFrame, dict, set[str]]:
    """Results already on disk, and the source stems fully covered by them."""
    metrics_path = output_dir / "metrics.csv"
    if not metrics_path.exists():
        return pd.DataFrame(), {}, set()
    previous = pd.read_csv(metrics_path)
    archive = output_dir / "artifacts.npz"
    arrays = dict(np.load(archive)) if archive.exists() else {}
    done = {
        utt.rsplit("_", 1)[0] if utt.endswith(("_real", "_codec"))
        else utt.rsplit("_to", 1)[0]
        for utt in previous["utt_id"].astype(str)
    }
    print(f"[resume] {len(previous)} row(s) already measured over {len(done)} source(s)")
    return previous, arrays, done


def write_tables(
    output_dir: Path,
    results: pd.DataFrame,
    arrays: dict,
    normalizer: str,
    final: bool = False,
) -> None:
    """Rewrite metrics.csv / artifacts.npz / pairs.csv from what has been measured."""
    if results.empty:
        return
    results.to_csv(output_dir / "metrics.csv", index=False)
    if arrays:
        np.savez_compressed(output_dir / "artifacts.npz", **arrays)

    # Read off the table rather than off the metric list: a backend is an ASR backend
    # here precisely when it emitted a hypothesis, and nothing else has to declare it.
    asr_columns = [c for c in results.columns if c.endswith(".hypothesis")]
    scored = add_word_errors(results, asr_columns, normalizer) if asr_columns else results
    paired = pair_with_references(scored, warn=final)
    # After pairing, because a cosine needs both ends: the row's own embedding and the one
    # belonging to the real recording it is set against.
    paired = add_speaker_similarity(paired, arrays)
    paired.to_csv(output_dir / "pairs.csv", index=False)


@parse_args
def main(config: dict, output_dir: Path):

    configure_matmul_precision()
    output_dir.mkdir(parents=True, exist_ok=True)

    ec = config.get("evaluation", {})
    train_config = OmegaConf.to_container(
        OmegaConf.load(config["converter"]["config_path"]), resolve=True
    )
    ckpt_path = config["converter"]["ckpt_path"]
    train_config["data"] = {**train_config["data"], **config.get("data", {})}
    split = config.get("data", {}).get("split", "test")

    # The measuring instrument, declared where the measurement is configured.
    #
    # A `predictor` / `label_extractor` block here REPLACES the training config's, whole:
    # these are copy-pasted, and a shallow merge would let a key the paste dropped leak
    # through from the run and describe an instrument that exists in neither file.
    #
    # Absent, they are inherited from converter.config_path, which measures with the P_φ
    # the run was validated against.  Present and different, the script says so — see
    # report_instrument.  That is the point of allowing it: an independent predictor is a
    # stronger test, because one the converter has never answered to cannot flatter it.
    for block in ("predictor", "label_extractor"):
        if config.get(block):
            report_instrument(block, train_config.get(block, {}), config[block])
            train_config[block] = dict(config[block])
        else:
            print(f"[config] {block}: inherited from the converter run")

    device = torch.device(
        config.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}\nCheckpoint: {ckpt_path}")

    # ------------------------------------------------------------------
    # Metrics first: a bad metric name or a missing extra should fail in seconds, not
    # after the codec has loaded and the corpus has been trimmed.
    # ------------------------------------------------------------------
    specs = config.get("metrics")
    if not specs:
        raise ValueError(
            f"No metrics configured. Available: {available_metrics()}"
        )
    metric_list = build_metrics(list(specs))

    # ------------------------------------------------------------------
    # Model and corpus, exactly as the render scripts build them.
    # ------------------------------------------------------------------
    source, _ = resolve_tau_src(train_config)
    pipeline, label_pipeline = build_conversion_pipelines(train_config)
    module = build_labels_module(
        train_config, pipeline, label_pipeline, ckpt_path=ckpt_path
    ).to(device).eval()

    # P_φ is appended rather than named in `metrics`: it borrows the already-loaded
    # predictor and whitening pipeline out of the module, so the reading is made by the
    # exact checkpoint this run is evaluating and cannot drift to a separately configured
    # one.  It is the only intensity measurement converted audio can have — the utterance
    # never existed, so there is nothing to compare against but the ruler.
    if ec.get("intensity_predictor", True):
        metric_list = metric_list + [IntensityPredictorMetric(
            module,
            sample_rate=pipeline.sample_rate,
            name=ec.get("intensity_predictor_name", "intensity"),
        )]
    print("Metrics: " + ", ".join(m.name for m in metric_list))

    metadata = pd.read_csv(train_config["data"]["metadata_csv"])
    metadata = resolve_paths(
        metadata, train_config["data"]["dataset_roots"], path_columns=["signal_path"]
    )
    metadata = metadata[metadata["split"] == split].reset_index(drop=True)

    lc = config.get("levels") or {}
    level_column = lc.get("level_column", DEFAULT_LEVEL_COLUMN)
    metadata = prepare_level_groups(
        metadata,
        level_column=level_column,
        group_columns=lc.get("group_columns", list(DEFAULT_GROUP_COLUMNS)),
        exclude_column=lc.get("exclude_column", DEFAULT_EXCLUDE_COLUMN),
        sentence_column=lc.get("sentence_column", DEFAULT_SENTENCE_COLUMN),
        sentence_ids=lc.get("sentence_ids"),
    )

    vad_cfg = dict(config.get("vad") or {})
    if vad_cfg.pop("enabled", True):
        metadata = trim_metadata_with_vad(metadata, vad_cfg)
        kept = metadata[metadata["vad_status"] == "ok"].reset_index(drop=True)
        print(f"VAD: {len(kept)}/{len(metadata)} rows retained")
        if kept.empty:
            raise ValueError("The VAD found no speech in any row of the split.")
        metadata = kept

    # Recomputed after the trim: a group loses a member if the VAD dropped it.
    targets_by_row = level_targets(metadata, level_column=level_column)
    row_stems = stems(metadata)
    group_ids = metadata["sentence_group_id"].tolist()
    records = source_records(metadata, level_column)

    dataset, collate_fn = build_inference_dataset(
        metadata, train_config, source, pipeline
    )
    loader_kw = dict(batch_size=1, shuffle=False,
                     num_workers=config.get("num_workers", 4),
                     pin_memory=True, collate_fn=collate_fn)

    # ------------------------------------------------------------------
    # Pass 1 — τ_src for every row.  A level target is another row's τ_src.
    # ------------------------------------------------------------------
    tau_src = np.empty(len(dataset), dtype=np.float64)
    show_progress = bool(ec.get("progress", True))
    with torch.no_grad():
        for i, batch in enumerate(progress(
            DataLoader(dataset, **loader_kw), "pass 1  measuring tau_src",
            total=len(dataset), enabled=show_progress,
        )):
            batch = {k: v.to(device) for k, v in batch.items()}
            tau_src[i] = float(module.source_tau(batch)[0])
    n_conversions = sum(len(t) for t in targets_by_row.values())
    print(f"Pass 1: τ_src over {len(dataset)} sources "
          f"({tau_src.min():.1f}–{tau_src.max():.1f} dB); "
          f"{n_conversions} conversions to render")

    # ------------------------------------------------------------------
    # Pass 2 — render and measure.
    # ------------------------------------------------------------------
    previous, prior_arrays, done = load_previous(output_dir)
    with_codec = bool(ec.get("with_codec", True))
    harness = MetricHarness(
        metric_list, device, batch_size=int(ec.get("batch_size", 16))
    )
    renderer = ConversionRenderer(
        module, pipeline, row_stems, group_ids, tau_src, records
    )
    normalizer = ec.get("wer_normalizer", "whisper_en")
    checkpoint_every = int(ec.get("checkpoint_every", 100))

    def combine() -> tuple[pd.DataFrame, dict]:
        fresh, arrays = harness.finish()
        merged = pd.concat([previous, fresh], ignore_index=True) if len(previous) \
            else fresh
        return merged, {**prior_arrays, **arrays}

    n_done = 0
    bar = progress(
        DataLoader(dataset, **loader_kw),
        f"pass 2  rendering + {len(metric_list)} metrics",
        total=len(dataset), enabled=show_progress,
    )
    with torch.no_grad():
        for i, batch in enumerate(bar):
            if row_stems[i] in done:
                continue
            batch = {k: v.to(device) for k, v in batch.items()}
            for rendered in renderer.render(
                batch, i, targets_by_row.get(i, []), with_codec=with_codec
            ):
                harness.add(
                    rendered.utt_id, rendered.wav, pipeline.sample_rate,
                    rendered.record, text=rendered.record.get("text"),
                )
            n_done += 1
            # Cheap, and it is what turns the bar from "sources" into something that
            # says how much measuring has actually happened.
            if hasattr(bar, "set_postfix"):
                bar.set_postfix(utt=harness.n_measured, refresh=False)
            if n_done % checkpoint_every == 0:
                results, arrays = combine()
                write_tables(output_dir, results, arrays, normalizer)

    results, arrays = combine()
    write_tables(output_dir, results, arrays, normalizer, final=True)

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    print(f"\nMeasured {len(results)} utterance(s) from {n_done} source(s)")
    print(results["condition"].value_counts().to_string())
    print(f"\nWrote {output_dir}/metrics.csv, pairs.csv"
          + (f", artifacts.npz ({len(arrays)} arrays)" if arrays else ""))


if __name__ == "__main__":
    main()

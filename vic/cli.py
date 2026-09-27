"""Command-line interface: ``vic-predict``.

Predict vocal intensity (per-frame dBSPL and sequence-level LeqZF) from
one or more audio files.

Usage — from a training checkpoint + config (internal)
------------------------------------------------------
    vic-predict \\
        --checkpoint runs/.../checkpoints/predictor-best.ckpt \\
        --config     configs/paper/train_predictor.yaml \\
        --input      speech.wav \\
        --output     speech.json

Usage — from a pre-built model bundle or HF Hub repo (collaborators)
--------------------------------------------------------------------
    vic-predict \\
        --model your-org/vic-predictor \\
        --input "recordings/*.wav" \\
        --output-dir predictions/

Output formats
--------------
    json (default)
        {"leq_db": -12.3, "frame_db": [...], "frame_duration_ms": 20.0}

    txt
        Line 1   : LeqZF value (float)
        Lines 2+ : one frame_db value per line (valid frames only, no padding)
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

from vic.predict import VocalIntensityPredictor, PredictionResult


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="vic-predict",
        description="Predict vocal intensity from audio files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Model loading — mutually exclusive modes
    model_src = p.add_mutually_exclusive_group(required=True)
    model_src.add_argument(
        "--model", metavar="PATH_OR_REPO",
        help="Local model bundle directory or HuggingFace Hub repo ID.",
    )
    model_src.add_argument(
        "--checkpoint", metavar="PATH",
        help="Path to a PredictorModule .ckpt file (requires --config).",
    )

    p.add_argument(
        "--config", metavar="PATH",
        help="Training YAML config — required when using --checkpoint.",
    )

    # I/O
    p.add_argument(
        "--input", required=True, metavar="PATH",
        help="Input audio file or glob pattern (quote globs in the shell).",
    )
    out = p.add_mutually_exclusive_group()
    out.add_argument(
        "--output", metavar="PATH",
        help="Output file path (single-file mode).",
    )
    out.add_argument(
        "--output-dir", metavar="DIR",
        help="Output directory (batch mode — one file per input).",
    )

    p.add_argument(
        "--format", choices=["json", "txt"], default="json",
        help="Output format: json (default) or txt.",
    )
    p.add_argument(
        "--device", default=None,
        help="Compute device, e.g. 'cuda', 'cuda:1', 'cpu'. "
             "Default: cuda if available, else cpu.",
    )
    return p


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def _load_predictor(args: argparse.Namespace) -> VocalIntensityPredictor:
    if args.model:
        return VocalIntensityPredictor.from_pretrained(args.model, device=args.device)
    # checkpoint mode
    if not args.config:
        print("error: --config is required when --checkpoint is used.", file=sys.stderr)
        sys.exit(1)
    return VocalIntensityPredictor.from_checkpoint(
        args.checkpoint, args.config, device=args.device
    )


def _resolve_inputs(pattern: str) -> list[Path]:
    paths = sorted(Path(p) for p in glob.glob(pattern, recursive=True))
    if not paths:
        print(f"error: no files matched: {pattern!r}", file=sys.stderr)
        sys.exit(1)
    return paths


# ---------------------------------------------------------------------------
# Per-file inference + output
# ---------------------------------------------------------------------------

def _infer_and_write(
    input_path: Path,
    predictor: VocalIntensityPredictor,
    output_path: Path,
    fmt: str,
) -> None:
    import torchaudio

    wav, sr = torchaudio.load(str(input_path))
    result: PredictionResult = predictor.predict(wav, sr)

    # Single file: batch size 1 — strip padding and unwrap the batch dim.
    mask = result.padding_mask[0]                 # (T,)
    frame_db = result.frame_db[0][mask].tolist()  # (T_valid,) floats
    leq_db = float(result.leq_db[0])

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "json":
        payload = {
            "leq_db": leq_db,
            "frame_db": frame_db,
            "frame_duration_ms": predictor.frame_duration_ms,
        }
        output_path.write_text(json.dumps(payload, indent=2))
    else:  # txt
        lines = [f"{leq_db:.6f}"] + [f"{v:.6f}" for v in frame_db]
        output_path.write_text("\n".join(lines) + "\n")

    print(f"{input_path.name}  →  {output_path}  (leq = {leq_db:.1f} dBSPL)")


def _output_ext(fmt: str) -> str:
    return ".json" if fmt == "json" else ".txt"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    predictor = _load_predictor(args)
    inputs = _resolve_inputs(args.input)

    if args.output is not None:
        if len(inputs) > 1:
            print(
                f"error: --output expects a single file but {len(inputs)} matched.",
                file=sys.stderr,
            )
            sys.exit(1)
        _infer_and_write(inputs[0], predictor, Path(args.output), args.format)

    elif args.output_dir is not None:
        out_dir = Path(args.output_dir)
        ext = _output_ext(args.format)
        for src in inputs:
            _infer_and_write(src, predictor, out_dir / (src.stem + ext), args.format)

    else:
        # No output path: print LeqZF to stdout (single file only).
        if len(inputs) > 1:
            print(
                "error: multiple inputs require --output or --output-dir.",
                file=sys.stderr,
            )
            sys.exit(1)
        import torchaudio
        wav, sr = torchaudio.load(str(inputs[0]))
        result = predictor.predict(wav, sr)
        print(f"{float(result.leq_db[0]):.4f} dBSPL")


if __name__ == "__main__":
    main()

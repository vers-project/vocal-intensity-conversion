"""Export a trained PredictorModule checkpoint to a self-contained model bundle.

The bundle can be loaded with ``VocalIntensityPredictor.from_pretrained()``
and shared with collaborators via Hugging Face Hub.

Bundle contents
---------------
    config.yaml   — extractor + model architecture (same format as training YAMLs)
    predictor.pt  — predictor head weights only (~few MB)

NAC-based models
----------------
The SpeechTokenizer codec weights are NOT copied into the bundle (they are
large and already publicly available at ``fnlp/SpeechTokenizer`` on HF Hub).
The config.yaml will contain the extractor paths from the training YAML as-is.
To make the bundle fully portable you can either:

  a) Store the codec files inside the bundle directory and use relative paths
     in config.yaml (``from_pretrained`` resolves them automatically), or
  b) After export, edit config.yaml to point extractor.config_path /
     extractor.ckpt_path to wherever collaborators download SpeechTokenizer.

MelSpec-based models are fully self-contained — no extra files needed.

Usage
-----
    uv run --extra cpu scripts/export_predictor.py \\
        --checkpoint runs/.../checkpoints/predictor-best.ckpt \\
        --config     configs/paper/train_predictor.yaml \\
        --output     exports/vic-predictor/

Upload to HF Hub (private repo)
--------------------------------
    huggingface-cli login
    huggingface-cli repo create YOUR-ORG/vic-predictor --private
    huggingface-cli upload YOUR-ORG/vic-predictor exports/vic-predictor/

    # Grant access to collaborators via the repo Settings page, or share a
    # read-only access token (HF Settings > Access Tokens).

Collaborator install
--------------------
    pip install huggingface_hub
    huggingface-cli login      # paste their read token once
    vic-predict --model YOUR-ORG/vic-predictor --input speech.wav
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from vic.predict import VocalIntensityPredictor


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a PredictorModule checkpoint to a self-contained bundle.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--checkpoint", required=True,
                        help="Path to the PredictorModule .ckpt file.")
    parser.add_argument("--config", required=True,
                        help="Training YAML config used to produce the checkpoint.")
    parser.add_argument("--output", required=True,
                        help="Output directory for the bundle.")
    parser.add_argument("--device", default="cpu",
                        help="Device for loading weights (default: cpu).")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())

    print(f"Loading checkpoint: {args.checkpoint}")
    predictor = VocalIntensityPredictor.from_checkpoint(
        args.checkpoint, config, device=args.device
    )

    print(f"Saving bundle to:   {args.output}/")
    predictor.save_pretrained(args.output, config)

    out = Path(args.output)
    print(f"  {(out / 'predictor.pt').stat().st_size / 1e6:.1f} MB  predictor.pt")
    print(f"  config.yaml")
    print()
    print("To upload to HF Hub:")
    print("  huggingface-cli login")
    print("  huggingface-cli repo create YOUR-ORG/vic-predictor --private")
    print(f"  huggingface-cli upload YOUR-ORG/vic-predictor {args.output}/")


if __name__ == "__main__":
    main()

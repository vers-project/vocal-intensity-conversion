"""Export a converter training checkpoint to an inference bundle.

The bundle (``model.safetensors`` + ``config.yaml``, see ``vic/converter_bundle.py``)
holds the converter and its label scaler only.  After writing it, the script reloads
it and checks that it converts a random latent exactly as the checkpoint does.

    uv run --extra cpu scripts/export_converter.py \\
        --checkpoint /path/to/run/checkpoints/converter-epoch=0599.ckpt \\
        --config     /path/to/run/config.yaml \\
        --output     exports/converter-wavlm/

Uploading is a separate, deliberate step:

    hf upload <repo-id> exports/converter-wavlm/ converter-wavlm/
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from vic.converter_bundle import (
    CONFIG_FILE,
    WEIGHTS_FILE,
    load_converter_bundle,
    load_converter_from_ckpt,
    save_converter_bundle,
)
from vic.data.audio_batch import AudioBatch


@torch.no_grad()
def convert_random_latent(converter: torch.nn.Module, latent_dim: int, tau_db: float):
    torch.manual_seed(0)
    z = AudioBatch(
        data=torch.randn(2, latent_dim, 150),
        lengths=torch.tensor([150, 110]),
        sample_rate=50,
    )
    return converter(z, torch.tensor([tau_db, tau_db])).data


def main():
    parser = argparse.ArgumentParser(description="Export a converter checkpoint to a bundle.")
    parser.add_argument("--checkpoint", required=True, help="Converter training .ckpt.")
    parser.add_argument("--config", required=True, help="The run's config.yaml.")
    parser.add_argument("--output", required=True, help="Bundle directory to write.")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    converter = load_converter_from_ckpt(args.checkpoint, cfg)
    out_dir = save_converter_bundle(converter, cfg, args.output)

    reloaded, bundle_cfg = load_converter_bundle(out_dir)
    tau_db = sum(bundle_cfg["training"]["intensity_range_db"]) / 2
    latent_dim = bundle_cfg["latent_dim"]
    if not torch.equal(
        convert_random_latent(converter, latent_dim, tau_db),
        convert_random_latent(reloaded, latent_dim, tau_db),
    ):
        raise SystemExit("The reloaded bundle does not reproduce the checkpoint's output.")

    n_params = sum(p.numel() for p in converter.parameters())
    size_mb = (out_dir / WEIGHTS_FILE).stat().st_size / 2**20
    print(f"{out_dir}: {n_params / 1e6:.2f} M parameters, {WEIGHTS_FILE} {size_mb:.1f} MiB")
    print(f"label scaler: {bundle_cfg['label_scaler']}")
    print(f"reloaded bundle reproduces the checkpoint exactly; see {out_dir / CONFIG_FILE}")


if __name__ == "__main__":
    main()

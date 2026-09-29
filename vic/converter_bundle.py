"""Converter bundles: the inference-only form of a trained C_θ.

A converter training checkpoint is a Lightning pickle of about 1.5 GB.  Almost all
of it is the frozen encoder and vocoder, the frozen P_φ, the discriminator and the
optimizer state, none of which a conversion needs or which we should redistribute.
A bundle keeps only what a conversion reads::

    bundle/
    ├── model.safetensors   the converter's state_dict
    └── config.yaml         extractor type and settings, model.converter,
                            latent_dim, label_scaler, training.intensity_range_db

The extractor block keeps its settings but drops every file path (``repo_path``,
``wavlm_ckpt``, …): those point at the encoder and vocoder weights on the machine
that trained the run.  Whoever loads the bundle fetches those weights from their
sources (``scripts/download_vocoders.py``) and supplies the paths.

The label scaler is written into ``config.yaml`` and is required on load: a
converter built with a different scaler normalises τ wrongly and raises nothing.
``latent_dim`` is stored so the converter can be rebuilt without building the codec.
"""
from __future__ import annotations

from pathlib import Path

import torch.nn as nn
import yaml
from safetensors.torch import load_file, save_file

from vic.checkpoints import read_submodule_state
from vic.models.converter import build_converter
from vic.training.utils import LabelScaler, load_label_scaler

WEIGHTS_FILE = "model.safetensors"
CONFIG_FILE = "config.yaml"
EXTRACTOR_PATH_KEYS = (
    "repo_path", "model_dir", "wavlm_ckpt", "hifigan_ckpt", "config_path", "ckpt_path",
)


def load_converter_from_ckpt(ckpt_path: str | Path, cfg: dict) -> nn.Module:
    """Rebuild C_θ from a converter training checkpoint and its training config.

    The label scaler comes from the checkpoint when it is there and from the
    config otherwise (see :func:`load_label_scaler`).  Every converter type ends
    in ``output_proj = Linear(d_model, latent_dim)``, so the latent width is read
    from the weights and the codec need not be built.
    """
    state = read_submodule_state(ckpt_path, "converter")
    latent_dim = state["output_proj.weight"].shape[0]
    converter = build_converter(
        cfg, latent_dim, label_scaler=load_label_scaler(ckpt_path, cfg)
    )
    converter.load_state_dict(state)
    return converter.eval()


def bundle_config(cfg: dict, converter: nn.Module) -> dict:
    """The part of a training config that a conversion reads, without file paths."""
    extractor = {
        key: value for key, value in cfg["extractor"].items()
        if key not in EXTRACTOR_PATH_KEYS
    }
    return {
        "extractor": extractor,
        "model": {"converter": cfg["model"]["converter"]},
        "latent_dim": converter.output_proj.out_features,
        "label_scaler": converter.label_scaler.state_dict(),
        "training": {"intensity_range_db": list(cfg["training"]["intensity_range_db"])},
    }


def save_converter_bundle(converter: nn.Module, cfg: dict, out_dir: str | Path) -> Path:
    """Write ``converter`` and the inference part of ``cfg`` as a bundle."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    state = {key: value.contiguous() for key, value in converter.state_dict().items()}
    save_file(state, out_dir / WEIGHTS_FILE)
    (out_dir / CONFIG_FILE).write_text(
        yaml.safe_dump(bundle_config(cfg, converter), sort_keys=False)
    )
    return out_dir


def resolve_bundle(
    path_or_repo_id: str | Path,
    revision: str | None = None,
    subfolder: str | None = None,
) -> Path:
    """A local bundle directory, downloading it from the Hugging Face Hub if needed.

    ``path_or_repo_id`` is a local directory, or else a Hub repo id.  ``revision``
    (a tag, branch or commit) pins the Hub version, and ``subfolder`` selects one
    bundle in a repo that holds several.
    """
    path = Path(path_or_repo_id)
    if path.is_dir():
        return path / subfolder if subfolder else path

    from huggingface_hub import snapshot_download

    pattern = f"{subfolder}/*" if subfolder else None
    root = Path(snapshot_download(
        repo_id=str(path_or_repo_id), revision=revision, allow_patterns=pattern,
    ))
    return root / subfolder if subfolder else root


def load_converter_bundle(
    path_or_repo_id: str | Path,
    revision: str | None = None,
    subfolder: str | None = None,
) -> tuple[nn.Module, dict]:
    """Rebuild C_θ from a bundle; return it with the bundle's config.

    The returned config has no extractor file paths: the caller adds them before
    building the codec.
    """
    path = resolve_bundle(path_or_repo_id, revision=revision, subfolder=subfolder)
    cfg = yaml.safe_load((path / CONFIG_FILE).read_text())
    if "label_scaler" not in cfg:
        raise KeyError(
            f"{path / CONFIG_FILE} has no 'label_scaler'; a converter cannot be "
            "rebuilt without the scaler it was trained with."
        )
    converter = build_converter(
        cfg, cfg["latent_dim"], label_scaler=LabelScaler.from_state_dict(cfg["label_scaler"])
    )
    converter.load_state_dict(load_file(path / WEIGHTS_FILE))
    return converter.eval(), cfg

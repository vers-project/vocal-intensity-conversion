"""Converter bundles: a checkpoint exported and reloaded must convert identically."""
import pytest
import torch
import yaml
from safetensors.torch import load_file

from vic.converter_bundle import (
    CONFIG_FILE,
    EXTRACTOR_PATH_KEYS,
    WEIGHTS_FILE,
    load_converter_bundle,
    load_converter_from_ckpt,
    save_converter_bundle,
)
from vic.data.audio_batch import AudioBatch
from vic.models.converter import build_converter
from vic.training.utils import LABEL_SCALER_CKPT_KEY, LabelScaler

from stubs import D, EMBED, RANGE

CONVERTERS = {
    "contextual": {
        "type": "contextual", "d_model": 16, "n_heads": 2, "n_layers": 2,
        "head_dim": 8, "mlp_ratio": 2, "dropout": 0.0, "intensity_embed": EMBED,
    },
    "frame": {
        "type": "frame", "d_model": 16, "n_layers": 2, "dropout": 0.0,
        "intensity_embed": EMBED,
    },
}


def _config(converter_cfg):
    return {
        "extractor": {
            "type": "wavlm_hifigan",
            "normalize_sequence": False,
            "repo_path": "/cluster/user/knn-vc",
            "wavlm_ckpt": "/cluster/user/WavLM-Large.pt",
            "hifigan_ckpt": "/cluster/user/g_02500000.pt",
        },
        "model": {"converter": converter_cfg},
        "training": {"intensity_range_db": list(RANGE)},
    }


def _write_checkpoint(path, cfg, scaler):
    """A Lightning-shaped checkpoint: several submodules, the scaler beside them."""
    torch.manual_seed(0)
    converter = build_converter(cfg, latent_dim=D, label_scaler=scaler)
    state_dict = {f"converter.{k}": v for k, v in converter.state_dict().items()}
    state_dict["cond_disc.weight"] = torch.randn(3, 3)
    torch.save(
        {"state_dict": state_dict, LABEL_SCALER_CKPT_KEY: scaler.state_dict()}, path
    )


@torch.no_grad()
def _convert(converter):
    torch.manual_seed(1)
    z = AudioBatch(data=torch.randn(2, D, 30), lengths=torch.tensor([30, 21]), sample_rate=50)
    return converter(z, torch.tensor([45.0, 72.0])).data


@pytest.mark.parametrize("kind", sorted(CONVERTERS))
def test_bundle_reproduces_the_checkpoint(tmp_path, kind):
    cfg = _config(CONVERTERS[kind])
    # Deliberately not the scaler the config's range implies: the bundle must
    # carry the checkpoint's own.
    scaler = LabelScaler(57.0, 9.0)
    _write_checkpoint(tmp_path / "run.ckpt", cfg, scaler)

    with pytest.warns(RuntimeWarning, match="disagrees"):
        from_ckpt = load_converter_from_ckpt(tmp_path / "run.ckpt", cfg)
    save_converter_bundle(from_ckpt, cfg, tmp_path / "bundle")
    from_bundle, bundle_cfg = load_converter_bundle(tmp_path / "bundle")

    assert torch.equal(_convert(from_ckpt), _convert(from_bundle))
    assert bundle_cfg["label_scaler"] == {"mean": 57.0, "std": 9.0}
    assert bundle_cfg["latent_dim"] == D


def test_bundle_holds_the_converter_only_and_no_paths(tmp_path):
    cfg = _config(CONVERTERS["contextual"])
    scaler = LabelScaler(60.0, 10.0)
    _write_checkpoint(tmp_path / "run.ckpt", cfg, scaler)
    save_converter_bundle(
        load_converter_from_ckpt(tmp_path / "run.ckpt", cfg), cfg, tmp_path / "bundle"
    )

    weights = load_file(tmp_path / "bundle" / WEIGHTS_FILE)
    assert not any(key.startswith(("converter.", "cond_disc")) for key in weights)
    extractor = yaml.safe_load((tmp_path / "bundle" / CONFIG_FILE).read_text())["extractor"]
    assert extractor == {"type": "wavlm_hifigan", "normalize_sequence": False}
    assert not set(extractor) & set(EXTRACTOR_PATH_KEYS)


def test_bundle_without_label_scaler_is_refused(tmp_path):
    cfg = _config(CONVERTERS["contextual"])
    converter = build_converter(cfg, latent_dim=D, label_scaler=LabelScaler(60.0, 10.0))
    save_converter_bundle(converter, cfg, tmp_path / "bundle")
    config_path = tmp_path / "bundle" / CONFIG_FILE
    bundle_cfg = yaml.safe_load(config_path.read_text())
    del bundle_cfg["label_scaler"]
    config_path.write_text(yaml.safe_dump(bundle_cfg))

    with pytest.raises(KeyError, match="label_scaler"):
        load_converter_bundle(tmp_path / "bundle")


def test_a_hub_id_downloads_only_the_requested_subfolder(tmp_path, monkeypatch):
    huggingface_hub = pytest.importorskip("huggingface_hub")
    from vic.converter_bundle import resolve_bundle

    calls = []

    def snapshot_download(repo_id, revision, allow_patterns):
        calls.append((repo_id, revision, allow_patterns))
        return str(tmp_path)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    path = resolve_bundle("org/models", revision="v1.0", subfolder="converter-wavlm")

    assert path == tmp_path / "converter-wavlm"
    assert calls == [("org/models", "v1.0", "converter-wavlm/*")]

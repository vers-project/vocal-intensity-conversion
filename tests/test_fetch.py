"""vic.codec.fetch: the vocoder tree layout, with the network replaced by fakes."""
import io
import zipfile

import pytest

from vic.codec import fetch as fetch_module
from vic.codec.fetch import (
    HIFIGAN_FILE,
    KNNVC_REPO_DIR,
    KNNVC_WEIGHTS_DIR,
    WAVLM_FILE,
    ensure_knnvc,
)


@pytest.fixture
def network(monkeypatch):
    """Record every URL requested; serve a GitHub-shaped zip and dummy weight files."""
    requested = []

    def download(url, part):
        requested.append(url)
        part.write_bytes(b"weights")

    def urlopen(url):
        requested.append(url)
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("knn-vc-abc123/wavlm/WavLM.py", "")
            archive.writestr("knn-vc-abc123/hifigan/models.py", "")
        buffer.seek(0)
        return buffer

    monkeypatch.setattr(fetch_module, "_download", download)
    monkeypatch.setattr(fetch_module.urllib.request, "urlopen", urlopen)
    return requested


def test_ensure_knnvc_builds_the_layout_and_returns_extractor_keys(tmp_path, network):
    paths = ensure_knnvc(tmp_path)

    assert paths == {
        "repo_path": str(tmp_path / KNNVC_REPO_DIR),
        "wavlm_ckpt": str(tmp_path / KNNVC_WEIGHTS_DIR / WAVLM_FILE),
        "hifigan_ckpt": str(tmp_path / KNNVC_WEIGHTS_DIR / HIFIGAN_FILE),
    }
    assert (tmp_path / KNNVC_REPO_DIR / "wavlm" / "WavLM.py").exists()
    assert (tmp_path / KNNVC_REPO_DIR / "hifigan" / "models.py").exists()
    assert not list(tmp_path.rglob("*.part"))
    assert len(network) == 3


def test_a_populated_tree_is_used_without_downloading(tmp_path, network):
    ensure_knnvc(tmp_path)
    network.clear()

    ensure_knnvc(tmp_path)
    assert network == []


def test_default_root_is_the_vic_cache(tmp_path, network, monkeypatch):
    monkeypatch.setenv("VIC_CACHE", str(tmp_path / "cache"))

    paths = ensure_knnvc()
    assert paths["repo_path"] == str(tmp_path / "cache" / KNNVC_REPO_DIR)


def test_an_extractor_without_a_fetcher_is_refused():
    from vic.codec.fetch import ensure_extractor_files

    with pytest.raises(NotImplementedError, match="mel_vocoder"):
        ensure_extractor_files("mel_vocoder")

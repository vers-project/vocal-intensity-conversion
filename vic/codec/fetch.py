"""Fetch the third-party code and weights a conversion domain needs, into one layout.

Every vocoder tree in this project has the same shape, whether it is a user's cache, a
``--vocoder-dir`` or a cluster copy populated by ``scripts/download_vocoders.py``::

    root/
    ├── knn-vc/            kNN-VC code (https://github.com/bshall/knn-vc, MIT)
    └── knn-vc-weights/    WavLM-Large.pt, g_02500000.pt  (kNN-VC release v0.1)

``ensure_*`` functions download whatever is missing under ``root`` and return the
``extractor`` config keys that point at it.  Files already present are never
re-downloaded, so a hand-populated tree is used as it is.

The kNN-VC code is pinned to one commit and taken as GitHub's zip archive of it, so no
``git`` is needed.  The weights are NOT verified: both files are pickles loaded with
``weights_only=False``.  Hosting them as safetensors on the Hub is the planned fix.
"""
from __future__ import annotations

import io
import os
import shutil
import subprocess
import tempfile
import urllib.request
import zipfile
from pathlib import Path

KNNVC_COMMIT = "c616845c4e309e24d5927f15adbdf277a3d65358"
KNNVC_ARCHIVE = f"https://github.com/bshall/knn-vc/archive/{KNNVC_COMMIT}.zip"
KNNVC_RELEASE = "https://github.com/bshall/knn-vc/releases/download/v0.1"
KNNVC_REPO_DIR = "knn-vc"
KNNVC_WEIGHTS_DIR = "knn-vc-weights"
WAVLM_FILE = "WavLM-Large.pt"
HIFIGAN_FILE = "g_02500000.pt"


def default_cache_dir() -> Path:
    """``$VIC_CACHE``, else ``$XDG_CACHE_HOME/vic``, else ``~/.cache/vic``."""
    if os.environ.get("VIC_CACHE"):
        return Path(os.environ["VIC_CACHE"]).expanduser()
    xdg = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(xdg).expanduser() / "vic"


def _download(url: str, part: Path) -> None:
    """curl if available (follows redirects, resumes, retries), else urllib with retries.

    curl is preferred because a 1.2 GB transfer that drops at 90 % through a proxy
    resumes instead of starting over.
    """
    try:
        subprocess.run(
            ["curl", "-fL", "--retry", "5", "--retry-delay", "5", "--retry-all-errors",
             "-C", "-", "--connect-timeout", "30", "-o", str(part), url],
            check=True,
        )
        return
    except FileNotFoundError:
        pass

    last: Exception | None = None
    for attempt in range(1, 4):
        try:
            urllib.request.urlretrieve(url, part)
            return
        except Exception as e:                                  # noqa: BLE001
            last = e
            print(f"        attempt {attempt}/3 failed: {e}")
    raise RuntimeError(f"could not download {url}") from last


def fetch(url: str, dest: Path, force: bool = False) -> Path:
    """Download ``url`` to ``dest`` via a ``.part`` file, so a kill leaves no half-file."""
    if dest.exists() and not force:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    print(f"  [get] {url}")
    try:
        _download(url, part)
    except Exception as e:                                      # noqa: BLE001
        raise RuntimeError(
            f"Failed to download {url}: {e}\n"
            "This URL redirects to a CDN host, so reaching the host in the URL is not "
            "enough. Behind a proxy, run scripts/download_vocoders.py --check to see "
            "which host is refused."
        ) from e
    part.replace(dest)
    print(f"        -> {dest} ({dest.stat().st_size / 1e6:.0f} MB)")
    return dest


def fetch_github_archive(url: str, dest: Path) -> Path:
    """Unpack a GitHub zip archive's single top-level folder as ``dest``.

    Unpacked beside ``dest`` and renamed into place, so an interrupted download never
    leaves a partial ``dest`` that later calls would take as complete.
    """
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  [get] {url}")
    with urllib.request.urlopen(url) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))
    with tempfile.TemporaryDirectory(dir=dest.parent) as tmp:
        archive.extractall(tmp)
        (top,) = Path(tmp).iterdir()
        shutil.move(str(top), dest)
    print(f"        -> {dest}")
    return dest


def ensure_knnvc(root: str | Path | None = None, force: bool = False) -> dict[str, str]:
    """kNN-VC code and weights under ``root``, downloading what is missing.

    Returns the ``repo_path``, ``wavlm_ckpt`` and ``hifigan_ckpt`` keys of a
    ``wavlm_hifigan`` extractor block.  ``force`` re-downloads the weights (never the
    code: a present ``knn-vc/`` may be a user's own checkout).
    """
    root = Path(root).expanduser() if root is not None else default_cache_dir()
    repo = fetch_github_archive(KNNVC_ARCHIVE, root / KNNVC_REPO_DIR)
    weights = root / KNNVC_WEIGHTS_DIR
    return {
        "repo_path": str(repo),
        "wavlm_ckpt": str(fetch(f"{KNNVC_RELEASE}/{WAVLM_FILE}", weights / WAVLM_FILE, force)),
        "hifigan_ckpt": str(
            fetch(f"{KNNVC_RELEASE}/{HIFIGAN_FILE}", weights / HIFIGAN_FILE, force)
        ),
    }

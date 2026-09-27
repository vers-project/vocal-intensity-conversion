"""Fetch the code and weights for the mel and WavLM conversion domains.

Neither vocoder is pip-installable, and neither set of weights ships inside a wheel, so
both are resolved by path at runtime (see ``vic/codec/_vendor.py``).  This script is the
one place that says where they come from, so a cluster without outbound access from its
compute nodes can be populated from a login node in a single command and the configs then
point at local paths.

    # FIRST, on a machine behind a proxy: which hosts can we actually reach?
    uv run --extra cpu --extra vocoder scripts/download_vocoders.py --check

    # everything, into the conventional sibling-of-the-repo layout
    uv run --extra cpu --extra vocoder scripts/download_vocoders.py --dest ../../../vocoders

    # just one domain, e.g. while the other is already downloading
    uv run --extra cpu --extra vocoder scripts/download_vocoders.py --dest DIR --only mel
    uv run --extra cpu --extra vocoder scripts/download_vocoders.py --dest DIR --only wavlm

Behind an HTTP proxy (Jean-Zay), read this first
------------------------------------------------
Neither weight file is served by the host its URL names.  Both 302-redirect to a CDN:

    huggingface.co/.../bigvgan_generator.pt  ->  us.aws.cdn.hf.co
                                             or  *.xethub.hf.co   (Xet-backed repos)
    github.com/.../prematch_g_02500000.pt    ->  release-assets.githubusercontent.com

So reaching ``huggingface.co`` and ``github.com`` is not sufficient, and a proxy allowlist
written before HF moved large-file serving to ``*.cdn.hf.co`` will pass the API call and
the 1 KB ``config.json`` and then refuse the 449 MB generator with a CONNECT failure.
``--check`` probes every host involved and says which, so the answer is one command rather
than one failed download.

The BigVGAN repo is Xet-backed, which is worth knowing because it means ``hf_hub_download``
(used here when ``huggingface_hub`` is importable) fetches from ``*.xethub.hf.co`` rather
than from ``us.aws.cdn.hf.co`` -- a different host, which may be allowed where the other is
not.  ``HF_HUB_DISABLE_XET=1`` forces the classic CDN path if the reverse turns out to be
true.

If a CDN host is blocked, no client-side setting fixes it.  Two ways out:
  * ask IDRIS to allow ``*.cdn.hf.co`` and ``release-assets.githubusercontent.com``; or
  * run this script on a machine with open access and copy the tree up, which needs no
    ticket:  ``rsync -av --progress DIR/ jean-zay:/path/to/vocoders/``

What lands where, and what it is for
------------------------------------
``DIR/BigVGAN/``                     clone of https://github.com/NVIDIA/BigVGAN (MIT)
``DIR/bigvgan_v2_22khz_80band_fmax8k_256x/``
                                     ``config.json`` + ``bigvgan_generator.pt`` from
                                     https://huggingface.co/nvidia/bigvgan_v2_22khz_80band_fmax8k_256x
``DIR/knn-vc/``                      clone of https://github.com/bshall/knn-vc (MIT)
``DIR/knn-vc-weights/``              ``WavLM-Large.pt``, ``prematch_g_02500000.pt`` and
                                     ``g_02500000.pt`` from that repo's v0.1 release

Citations for both, and the reasoning behind these specific checkpoints, are in
``vic/codec/mel_vocoder.py`` and ``vic/codec/wavlm_hifigan.py``.  In short: BigVGAN-v2 for
documented zero-shot robustness rather than in-distribution benchmark scores, the
``fmax8k`` variant so no mel band sits above the source band, and kNN-VC's *plain*
HiFi-GAN because prematching matches kNN-VC's own inference distribution and not ours --
a converter emits a transformed version of raw encoder output.  Both are fetched.

Plain ``argparse`` rather than the ``@parse_args`` launcher paradigm: this is a one-shot
helper that runs once per machine on a login node, has no sweep and no SLURM job.

Weights total roughly 2.5 GB, dominated by WavLM-Large (~1.2 GB).  Existing files are left
alone unless ``--force`` is passed, so a re-run after an interrupted download is cheap.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

# --- mel domain -------------------------------------------------------------------
BIGVGAN_REPO = "https://github.com/NVIDIA/BigVGAN.git"
BIGVGAN_MODEL = "nvidia/bigvgan_v2_22khz_80band_fmax8k_256x"
BIGVGAN_FILES = ("config.json", "bigvgan_generator.pt")

# --- WavLM domain -----------------------------------------------------------------
KNNVC_REPO = "https://github.com/bshall/knn-vc.git"
KNNVC_RELEASE = "https://github.com/bshall/knn-vc/releases/download/v0.1"
KNNVC_FILES = (
    "WavLM-Large.pt",            # encoder
    "g_02500000.pt",             # decoder trained on raw features -- the one to use
    "prematch_g_02500000.pt",    # decoder trained on kNN-prematched features -- control
)


def clone(url: str, dest: Path, force: bool) -> None:
    """Shallow-clone ``url`` into ``dest``, or report that it is already there."""
    if dest.exists() and not force:
        print(f"  [skip] {dest} already exists")
        return
    if dest.exists():
        raise FileExistsError(
            f"{dest} exists; remove it by hand if you really want to re-clone. "
            "This script will not delete a directory for you."
        )
    print(f"  [clone] {url} -> {dest}")
    subprocess.run(
        ["git", "clone", "--depth", "1", url, str(dest)],
        check=True,
    )


# Every host the two downloads touch, and what fails if it is refused.  Kept next to the
# URLs so a redirect target changing upstream is a one-line edit here.
PROBE_HOSTS = [
    ("huggingface.co", "HF API + small files (config.json)"),
    ("us.aws.cdn.hf.co", "HF large files, classic CDN path"),
    ("cas-bridge.xethub.hf.co", "HF large files, Xet path (hf_hub_download)"),
    ("transfer.xethub.hf.co", "HF Xet content-addressed store"),
    ("github.com", "git clone of both repos"),
    ("release-assets.githubusercontent.com", "kNN-VC weights (WavLM-Large.pt, generators)"),
]


def _proxy_env() -> dict[str, str]:
    keys = ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "no_proxy", "NO_PROXY")
    return {k: os.environ[k] for k in keys if os.environ.get(k)}


def check_hosts() -> int:
    """Probe each required host and report reachable / blocked.  Returns an exit code.

    Any HTTP status counts as reachable -- a CDN root legitimately answers 403 to a bare
    GET.  What matters is whether the connection (through the proxy's CONNECT, if there is
    one) is established at all, which is exactly what failed with
    ``Tunnel connection failed: 500``.
    """
    proxies = _proxy_env()
    print("proxy environment:", proxies or "(none set)")
    print()
    blocked = []
    for host, purpose in PROBE_HOSTS:
        try:
            out = subprocess.run(
                ["curl", "-sS", "-o", os.devnull, "-w", "%{http_code}",
                 "--max-time", "25", f"https://{host}/"],
                capture_output=True, text=True, check=True,
            )
            print(f"  [ok]      {host:<38} HTTP {out.stdout.strip()}   {purpose}")
        except FileNotFoundError:
            print("curl not found; cannot probe. Install curl or check by hand.")
            return 2
        except subprocess.CalledProcessError as e:
            reason = (e.stderr or "").strip().splitlines()
            print(f"  [BLOCKED] {host:<38} {reason[-1] if reason else 'curl failed'}")
            print(f"            needed for: {purpose}")
            blocked.append(host)
    if blocked:
        print(
            "\nBlocked hosts are a proxy allowlist, not something a client flag fixes.\n"
            "Either ask IDRIS to allow them, or run this script where access is open and\n"
            "rsync the result up -- see the module docstring."
        )
    else:
        print("\nAll hosts reachable; a plain download should work.")
    return 1 if blocked else 0


def _fetch_url(url: str, part: Path) -> None:
    """curl if available (follows redirects, resumes, retries), else urllib with retries.

    ``urlretrieve`` was the original implementation and is kept only as a fallback: it does
    not resume, so a 1.2 GB transfer that drops at 90% through a proxy starts over.
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


def fetch(url: str, dest: Path, force: bool) -> None:
    """Download ``url`` to ``dest`` via a ``.part`` file, so a kill leaves no half-file."""
    if dest.exists() and not force:
        print(f"  [skip] {dest.name} already present ({dest.stat().st_size / 1e6:.0f} MB)")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    print(f"  [get] {url}")
    try:
        _fetch_url(url, part)
    except Exception as e:                                      # noqa: BLE001
        raise SystemExit(
            f"\nFailed to download {url}\n  {e}\n\n"
            f"proxy environment: {_proxy_env() or '(none set)'}\n\n"
            "This URL 302-redirects to a CDN host, so reaching the host in the URL is not\n"
            "enough. Run with --check to see which host is refused, then read the\n"
            "'Behind an HTTP proxy' section of this script's docstring."
        ) from e
    part.replace(dest)
    print(f"        -> {dest} ({dest.stat().st_size / 1e6:.0f} MB)")


def hf_file_url(model: str, filename: str) -> str:
    return f"https://huggingface.co/{model}/resolve/main/{filename}"


def fetch_hf(model: str, filename: str, dest_dir: Path, force: bool) -> None:
    """Fetch one file from the Hub, preferring ``hf_hub_download``.

    Two reasons to prefer it over the URL: it resumes and retries, and for a Xet-backed
    repo -- which ``nvidia/bigvgan_v2_*`` is -- it transfers via ``*.xethub.hf.co`` instead
    of ``us.aws.cdn.hf.co``.  Those are different hosts as far as a proxy allowlist is
    concerned, so this path can succeed where the plain URL is refused.  ``local_dir``
    writes the real file (no symlink into a shared cache), which is the layout
    ``MelVocoderCodec`` expects.
    """
    target = dest_dir / filename
    if target.exists() and not force:
        print(f"  [skip] {filename} already present ({target.stat().st_size / 1e6:.0f} MB)")
        return
    try:
        from huggingface_hub import hf_hub_download          # noqa: PLC0415
    except ImportError:
        print("  [note] huggingface_hub not installed; falling back to the plain URL.")
        fetch(hf_file_url(model, filename), target, force)
        return

    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"  [get] {model}/{filename} via hf_hub_download")
    try:
        hf_hub_download(
            repo_id=model, filename=filename,
            local_dir=str(dest_dir), force_download=force,
        )
    except Exception as e:                                      # noqa: BLE001
        print(f"  [note] hf_hub_download failed ({type(e).__name__}); trying the plain URL.")
        print("         If both fail, the CDN host is blocked -- run with --check.")
        fetch(hf_file_url(model, filename), target, force)
        return
    print(f"        -> {target} ({target.stat().st_size / 1e6:.0f} MB)")


def download_mel(dest: Path, force: bool) -> dict[str, Path]:
    print("\n=== mel domain: BigVGAN-v2 ===")
    repo = dest / "BigVGAN"
    clone(BIGVGAN_REPO, repo, force)

    model_dir = dest / BIGVGAN_MODEL.split("/")[-1]
    for name in BIGVGAN_FILES:
        fetch_hf(BIGVGAN_MODEL, name, model_dir, force)
    return {"repo_path": repo, "model_dir": model_dir}


def download_wavlm(dest: Path, force: bool) -> dict[str, Path]:
    print("\n=== WavLM domain: kNN-VC ===")
    repo = dest / "knn-vc"
    clone(KNNVC_REPO, repo, force)

    weights = dest / "knn-vc-weights"
    for name in KNNVC_FILES:
        fetch(f"{KNNVC_RELEASE}/{name}", weights / name, force)
    return {"repo_path": repo, "weights": weights}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dest", type=Path, default=None,
        help="directory to populate; created if absent. Required unless --check",
    )
    parser.add_argument(
        "--only", choices=("mel", "wavlm"), default=None,
        help="fetch just one domain (default: both)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="re-download files that are already present (does not re-clone)",
    )
    parser.add_argument(
        "--check", action="store_true",
        help="probe every host the downloads need and exit; downloads nothing. "
             "Run this first on a machine behind a proxy",
    )
    args = parser.parse_args()

    if args.check:
        return check_hosts()
    if args.dest is None:
        parser.error("--dest is required (or pass --check to probe connectivity only)")

    dest = args.dest.expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)

    mel = wavlm = None
    if args.only in (None, "mel"):
        mel = download_mel(dest, args.force)
    if args.only in (None, "wavlm"):
        wavlm = download_wavlm(dest, args.force)

    # Printed rather than written to a config, because the values belong to whichever
    # experiment config the user is about to write and the paths are machine-specific.
    print("\n" + "=" * 72)
    print("Paste into the `extractor:` block of the config for that domain.")
    print("=" * 72)
    if mel:
        print(f"""
extractor:
  type: mel_vocoder
  backend: bigvgan
  repo_path: {mel['repo_path']}
  model_dir: {mel['model_dir']}
  whitening: false
  normalize_sequence: false
# NOTE: this domain runs at 22050 Hz, so data.window_length must be 551 (25 ms),
#       not the 400 used by every 16 kHz config.
""")
    if wavlm:
        print(f"""
extractor:
  type: wavlm_hifigan
  repo_path: {wavlm['repo_path']}
  wavlm_ckpt: {wavlm['weights'] / 'WavLM-Large.pt'}
  hifigan_ckpt: {wavlm['weights'] / 'g_02500000.pt'}
  whitening: false
  normalize_sequence: false
""")
    return 0


if __name__ == "__main__":
    sys.exit(main())

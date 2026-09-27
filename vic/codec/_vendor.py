"""Import helper for third-party vocoder repositories that are not pip-installable.

BigVGAN and kNN-VC are both distributed as plain research repositories: no
``pyproject.toml``, no wheel, module-level imports that assume the repo root is on
``sys.path``.  The established pattern in this codebase is the one
``vic/encoders/mp_senet.py`` already uses for MP-SENet -- clone the repo, put its root
on ``sys.path``, import its modules by their own names -- and this module just
de-duplicates it so the two codecs fail the same way, with the same message, when the
path is wrong.

Why not vendor the files into ``vic/`` instead: the whole point of these two codecs is
that feature extraction matches the pretrained decoder *exactly*.  Calling upstream's own
``mel_spectrogram`` / ``WavLM.extract_features`` makes that true by construction, whereas
a copy is a snapshot that can drift from the checkpoint it has to match without anything
raising.  The cost is that the repo is a runtime dependency, resolved by path.

Known limitation, inherited from the pattern: these repos put modules at their root
(BigVGAN ships ``utils.py``, ``env.py``, ``meldataset.py``, ``activations.py``), so after
this call those names resolve to the vendored copies for the rest of the process.  Nothing
in ``vic`` imports an unqualified ``utils`` or ``env``, so this is latent rather than
active -- but it is the reason to keep the list of vendored repos short.
"""
from __future__ import annotations

import sys
from pathlib import Path


def add_repo_to_path(repo_path: str | Path, *, name: str, expect: list[str]) -> Path:
    """Put a cloned repository root on ``sys.path`` and check it is the right one.

    Parameters
    ----------
    repo_path : path to the cloned repository root.
    name      : human-readable repo name, used in error messages only.
    expect    : entries (files or directories) that must exist directly under
                ``repo_path``.  Checked *before* the import so a wrong or empty path
                reports the path it was given, rather than surfacing later as a
                ``ModuleNotFoundError`` for a module nobody has heard of.

    Returns
    -------
    The resolved repository root.
    """
    repo = Path(repo_path).expanduser().resolve()
    if not repo.is_dir():
        raise FileNotFoundError(
            f"{name} repo_path does not exist or is not a directory: {repo}. "
            f"Clone it with scripts/download_vocoders.py."
        )

    missing = [e for e in expect if not (repo / e).exists()]
    if missing:
        raise FileNotFoundError(
            f"{name} repo_path {repo} does not look like a {name} checkout: "
            f"missing {missing}. Expected the repository *root*, the directory that "
            f"contains {expect}."
        )

    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    return repo

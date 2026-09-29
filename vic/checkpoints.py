"""Reading weights out of Lightning checkpoints.

A ``LightningModule`` checkpoint stores every submodule in one flat
``state_dict`` keyed by attribute path: a ``PredictorModule`` holding
``self.predictor`` writes ``predictor.blocks.0.weight``, and a converter module
holding ``self.converter`` writes ``converter.…``.  Rebuilding one component on
its own — the predictor for inference, the converter for a conversion script —
therefore means selecting a key prefix and stripping it.

That is a two-line dict comprehension, which is why five call sites had each
written their own.  They disagreed on the parts that matter when something is
wrong: whether the prefix is configurable at all, and what happens when it
matches nothing.  An empty selection handed to ``load_state_dict(strict=True)``
reports itself as *every* key being missing, which reads like an architecture
mismatch and sends you looking at the model config — when the actual fault is a
one-word typo in a prefix string.  This module raises on the empty selection
instead, and names the prefixes the file does contain.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


def read_submodule_state(ckpt_path: str | Path, prefix: str) -> dict[str, torch.Tensor]:
    """One submodule's ``state_dict`` out of a Lightning checkpoint, prefix stripped.

    Parameters
    ----------
    ckpt_path : path to the ``.ckpt``.
    prefix    : the submodule's attribute name on the LightningModule, without
                the trailing dot — ``"predictor"``, ``"converter"``,
                ``"student_predictor"`` for an archived distillation checkpoint.

    Raises
    ------
    RuntimeError
        If no key in the checkpoint carries ``prefix``, naming the top-level
        prefixes the file actually has.
    """
    # weights_only=False: a Lightning checkpoint carries hyper_parameters and
    # callback state alongside the tensors, which the weights-only unpickler
    # refuses to touch.
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"]

    dotted = f"{prefix}."
    selected = {
        key.removeprefix(dotted): value
        for key, value in state_dict.items()
        if key.startswith(dotted)
    }
    if not selected:
        available = sorted({key.split(".")[0] for key in state_dict})
        raise RuntimeError(
            f"No keys with prefix '{dotted}' found in {ckpt_path}. "
            f"Available top-level prefixes: {available}"
        )
    return selected


def load_submodule(
    module: nn.Module,
    ckpt_path: str | Path,
    prefix: str = "predictor",
    strict: bool = True,
) -> nn.Module:
    """Load one submodule's weights in place from a Lightning checkpoint.

    ``module`` is the already-built submodule, returned so this composes with a
    builder: ``load_submodule(build_predictor(...), ...)``.  ``prefix`` is as in
    :func:`read_submodule_state`, and ``strict`` is passed to ``load_state_dict``.
    """
    module.load_state_dict(read_submodule_state(ckpt_path, prefix), strict=strict)
    return module


def load_extra(ckpt_path: str | Path, key: str, default=None):
    """Read a non-``state_dict`` entry a LightningModule wrote in its checkpoint.

    ``on_save_checkpoint`` can store anything alongside the weights — a fitted
    normaliser, a vocabulary, the label range a run was trained over.  Such an
    entry is absent from every checkpoint written before the hook existed, so
    ``default`` is returned rather than raising: the caller decides whether a
    missing entry is fatal or just means "fall back to the config".
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return ckpt.get(key, default)


def resolve_resume(ckpt_dir: str | Path, resume_from: str | Path | None) -> Path | None:
    """Which checkpoint a training run should continue from, if any.

    Two different needs, and getting the precedence backwards silently discards
    work, so it is decided in one place:

    ``ckpt_dir/last.ckpt``
        This run's own progress.  Exists only once the run has written an epoch,
        i.e. on a SLURM requeue or a relaunch into the same ``--output-dir``.
        **Wins**, because a run that has already advanced must not jump back to
        whatever seeded it.
    ``resume_from``
        A *seed* from an earlier run, named by ``training.resume_from``.  Used
        only on this output_dir's first start.  Accepts a ``.ckpt`` file or a
        directory, in which case ``last.ckpt`` inside it is taken.

    Absent both, the run starts from scratch.

    A ``resume_from`` that does not exist raises rather than falling back to a
    fresh start: it is set deliberately, a typo in a cluster path is easy, and
    discovering it as "the run silently retrained from zero" costs a queue slot.

    In a SWEEP this key must live in the per-cell overrides, never in the shared
    ``training`` block -- one value would point every cell at one cell's weights.
    Most such mistakes raise on the ``strict=True`` state_dict load (the cells'
    latent widths and frozen submodules differ), but do not rely on that.
    """
    ckpt_dir = Path(ckpt_dir)
    own = ckpt_dir / "last.ckpt"
    if own.exists():
        return own

    if resume_from is None:
        return None

    seed = Path(resume_from)
    if seed.is_dir():
        seed = seed / "last.ckpt"
    if not seed.exists():
        raise SystemExit(
            f"training.resume_from points at {seed}, which does not exist. "
            "Check the path (and that the run you are resuming actually wrote a "
            "last.ckpt); remove the key to train from scratch."
        )
    return seed

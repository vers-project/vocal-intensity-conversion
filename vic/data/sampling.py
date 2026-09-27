"""Duration-weighted sampling over a segment index, and what an epoch means.

One ``__getitem__`` yields exactly one chunk, whatever the row's length.  So over
a segment index, drawing rows uniformly draws each segment's *seconds* at a rate
inversely proportional to its duration: on AVID's training split the paragraph
task is 42% of the speech but only 7.5% of the rows, and would be seen 7% of the
time.  Weighting the draw by segment duration makes the per-second probability
uniform, which moves paragraphs to ~42% of draws.

Shared by ``train_predictor.py`` and ``train_converter_cgan_v2_labels.py``, so both
models are exposed to the same draw.
"""
from __future__ import annotations

import pandas as pd
import torch
from torch.utils.data import WeightedRandomSampler

from vic.data.metadata import require_non_null

# Not a parameter.  ``file_duration_s`` sits beside this column in a combined
# metadata table and means the whole session recording — on AVID, a mean of
# 1261 s against this column's 5.4 s.  Weighting by it would be a near-flat
# weight set by how long the session was, correcting nothing, and nothing would
# say so.  Owning the name here is what stops a call site picking the wrong one.
DURATION_COLUMN = "segment_duration_s"


def duration_weighted_sampler(
    metadata: pd.DataFrame,
    num_samples: int,
    generator: torch.Generator | None = None,
) -> WeightedRandomSampler:
    """Sample rows with probability proportional to their segment duration.

    ``metadata`` must be the split's own frame, in the same row order as the
    dataset it will index: ``WeightedRandomSampler`` emits *positional* indices,
    and ``AudioDataset`` builds its lists positionally too, so position ``i`` of
    the weights must be position ``i`` of the dataset.  Pass the frame you passed
    to the dataset, not the full table.
    """
    require_non_null(
        metadata, [DURATION_COLUMN],
        "a duration-weighted sampler cannot weight it.  A corpus with one row "
        "per file carries no segment duration, and pandas NaN-fills the column "
        "when such a corpus is concatenated with a segment index; "
        "WeightedRandomSampler does not raise on NaN weights, it degenerates.",
    )
    return WeightedRandomSampler(
        weights=metadata[DURATION_COLUMN].tolist(),
        num_samples=int(num_samples),
        replacement=True,
        generator=generator,
    )


def epoch_size(training_cfg: dict, chunk_s: float | None) -> int | None:
    """How many chunks make an epoch, or ``None`` for a plain shuffle over rows.

    Two ways to say it, at most one of them:

    ``hours_per_epoch``
        Hours of speech per epoch.  Preferred, because it is the only unit that
        survives a change of ``chunk_duration_s``: an epoch defined in rows or in
        chunks silently doubles the audio seen when the chunk length doubles, so
        two configs differing only in chunk length are not comparable at equal
        ``max_epochs``.  Needs ``chunk_s``; meaningless without one.
    ``samples_per_epoch``
        Number of chunks.  The converter's existing key, kept so its configs and
        past runs keep their meaning.

    Neither key present means sampling stays as it was — uniform over rows — so
    adding this to a script changes nothing until a config asks for it.
    """
    hours = training_cfg.get("hours_per_epoch")
    samples = training_cfg.get("samples_per_epoch")

    if hours is not None and samples is not None:
        raise SystemExit(
            "training.hours_per_epoch and training.samples_per_epoch both set; "
            "they define the same quantity in different units.  Keep one."
        )
    if hours is not None:
        if not chunk_s:
            raise SystemExit(
                "training.hours_per_epoch needs data.chunk_duration_s to convert "
                "hours into chunks.  Use samples_per_epoch on an unchunked run."
            )
        return max(1, round(float(hours) * 3600.0 / float(chunk_s)))
    if samples is not None:
        return int(samples)
    return None

"""Run ``speech_eval`` metrics over audio that never reaches disk.

``speech_eval``'s own runner reads a manifest and loads files.  A converter evaluating
itself has no files: it holds a decoded tensor that exists for as long as it takes to
measure it.  Writing ~7 000 rendered utterances out just to read them back would cost
several gigabytes and a lot of I/O for nothing.

So this is the runner's in-memory counterpart, and it reuses the parts that carry meaning:
metrics are built by ``speech_eval.core.build_metrics``, audio is prepared by
``speech_eval.io.prepare_utterance`` (the same function the file-backed dataset calls), and
features are split into scalars and arrays the same way.  What it adds is buffering: it
accumulates utterances as the renderer produces them and flushes a whole batch through
every metric at once, so a GPU metric still sees batches even though the producer emits six
utterances at a time.

Why the buffer holds *raw* utterances
-------------------------------------
Metrics declare different :class:`Requirements` — 16 kHz at -27 dBFS for the phonetics
family, 16 kHz at -20 dBFS for ASR, native rate and untouched amplitude for level — so the
same waveform is prepared several different ways.  Buffering the raw signal once and
preparing per group at flush time keeps one copy of the audio alive instead of one per
requirements group, and it is what lets the caller drop its own reference immediately.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from speech_eval.core import Features, Metric, Requirements, UtteranceBatch
from speech_eval.io import prepare_utterance

#: Identity columns every emitted row carries, in this order, before metric columns.
ID_COLUMNS = ["utt_id", "group_id", "condition"]


def group_by_requirements(metrics: list[Metric]) -> dict[Requirements, list[Metric]]:
    """Bucket metrics by the audio they need, preserving declaration order.

    ``meta_columns`` is cleared from the key because it names metadata, not a property of
    the waveform: two metrics differing only in it still share one preparation pass.  The
    same rule ``speech_eval.runner`` applies.
    """
    groups: dict[Requirements, list[Metric]] = defaultdict(list)
    for metric in metrics:
        groups[replace(metric.requirements, meta_columns=())].append(metric)
    return dict(groups)


def meta_columns(metrics: list[Metric]) -> list[str]:
    """Every metadata column some metric needs in order to compute.

    Distinct from the columns a report *shows*: a pitch tracker needs ``sex`` to set its
    analysis range whether or not the results table mentions it.
    """
    names: list[str] = []
    for metric in metrics:
        names.extend(metric.requirements.meta_columns)
    return list(dict.fromkeys(names))


def split_features(features: Features) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Separate scalars (CSV) from arrays (archive), as ``speech_eval.runner`` does."""
    scalars: dict[str, Any] = {}
    arrays: dict[str, np.ndarray] = {}
    for key, value in features.items():
        if isinstance(value, np.ndarray):
            arrays[key] = value
        elif isinstance(value, Tensor):
            arrays[key] = value.detach().cpu().numpy()
        else:
            scalars[key] = value
    return scalars, arrays


class MetricHarness:
    """Buffers utterances, then runs every metric over each full batch.

    Parameters
    ----------
    metrics    : built ``speech_eval`` metrics; ``load`` is called once here.
    device     : where metric models go.
    batch_size : how many utterances to accumulate before flushing.  This is the batch the
                 metrics see, so it trades GPU efficiency against peak memory.

    Usage::

        harness = MetricHarness(metrics, device, batch_size=16)
        for rendered in ...:
            harness.add(rendered.utt_id, rendered.wav, sr, record, text)
        rows, arrays = harness.finish()
    """

    def __init__(self, metrics: list[Metric], device: torch.device, batch_size: int = 16):
        if not metrics:
            raise ValueError("MetricHarness needs at least one metric.")
        self.metrics = metrics
        self.device = device
        self.batch_size = int(batch_size)
        self.groups = group_by_requirements(metrics)
        self.meta_columns = meta_columns(metrics)

        for metric in self.metrics:
            # ensure_loaded, not load: subclass load() rebuilds its backend
            # unconditionally, and a long-lived owner (the training-time monitor)
            # constructs a harness once per evaluation over the same instances.
            metric.ensure_loaded(device)

        self._buffer: list[dict[str, Any]] = []
        self._rows: dict[str, dict[str, Any]] = {}
        self._arrays: dict[str, np.ndarray] = {}

    # -- input ---------------------------------------------------------

    def add(
        self,
        utt_id: str,
        wav: Tensor,
        sample_rate: int,
        record: dict[str, Any],
        text: Any = None,
    ) -> None:
        """Queue one utterance.  ``record`` is carried onto its results row verbatim."""
        if utt_id in self._rows or any(item["utt_id"] == utt_id for item in self._buffer):
            raise ValueError(f"Duplicate utt_id {utt_id!r}.")
        # Detached and moved off the GPU now: the caller is about to drop its reference,
        # and holding a CUDA tensor in the buffer would pin decoder memory for the whole
        # batch.
        self._buffer.append({
            "utt_id": utt_id,
            "wav": wav.detach().to("cpu"),
            "sample_rate": sample_rate,
            "record": record,
            "text": text,
        })
        if len(self._buffer) >= self.batch_size:
            self.flush()

    # -- computation ---------------------------------------------------

    def flush(self) -> None:
        """Run every metric over the buffered utterances, then release the audio."""
        if not self._buffer:
            return

        for item in self._buffer:
            self._rows[item["utt_id"]] = {
                **item["record"],
                "utt_id": item["utt_id"],
            }

        for requirements, metrics in self.groups.items():
            batch = UtteranceBatch(utterances=[
                prepare_utterance(
                    utt_id=item["utt_id"],
                    wav=item["wav"],
                    sample_rate=item["sample_rate"],
                    requirements=requirements,
                    text=item["text"],
                    meta={k: item["record"].get(k) for k in self.meta_columns},
                )
                for item in self._buffer
            ])
            for metric in metrics:
                for utt_id, features in zip(batch.utt_ids, metric.compute(batch)):
                    scalars, arrays = split_features(features)
                    self._rows[utt_id].update(
                        {f"{metric.name}.{k}": v for k, v in scalars.items()}
                    )
                    for key, value in arrays.items():
                        self._arrays[f"{utt_id}|{metric.name}.{key}"] = value

        self._buffer.clear()

    # -- output --------------------------------------------------------

    def finish(self) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
        """Flush what remains and return the results table and the array archive."""
        self.flush()
        frame = pd.DataFrame(list(self._rows.values()))
        if not frame.empty:
            ordered = [c for c in ID_COLUMNS if c in frame.columns]
            frame = frame[ordered + [c for c in frame.columns if c not in ordered]]
        return frame, dict(self._arrays)

    @property
    def n_measured(self) -> int:
        return len(self._rows)

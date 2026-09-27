"""What each test utterance turns into: the real take, the codec anchor, four conversions.

The pairing this implements rests on the corpus design.  One
sentence is read by one speaker at four effort levels, so for every source recording there
exist four *real* recordings — its own and its three siblings — whose levels are the four
targets worth asking the converter for.  Converting the source to each of them yields
pairs that can be compared directly, because the destination of every conversion is an
actual utterance of the same words in the same voice at that effort, not a model of one.

Per source row, six utterances:

    ``real``       the VAD-trimmed recording, at the codec's rate.  This is the exact
                   tensor the converter is given, so real and converted differ by what the
                   model did and by nothing else — not by a resampler, not by a trim.
    ``codec``      that tensor encoded and decoded with no conversion.  The anchor: it
                   says what the NAC alone does to F0, to spectral slope and to WER, so a
                   degradation can be attributed between codec and converter instead of
                   being charged entirely to the converter.
    ``converted``  four, one per level present in the source's own sentence group,
                   including the source's own level — the identity case, and the only
                   confound-free artefact check available.

Nothing is written to disk.  A caller renders a source, hands the six waveforms to the
metric harness, and drops them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch import Tensor

from vic.data.levels import LevelTarget

REAL = "real"
CODEC = "codec"
CONVERTED = "converted"

#: Columns naming the two ends of a conversion.  Named rather than spelled out at
#: every call site because a row's *source* and its *target* are both real
#: recordings, and reading one where the other was meant is silent.
SOURCE_REAL_UTT_ID = "source_real_utt_id"
REFERENCE_UTT_ID = "reference_utt_id"

#: Metadata columns copied onto every results row.  Absent ones are skipped, so
#: one list serves corpora that carry different subsets.
CARRY_COLUMNS = [
    "corpus", "speaker_uid", "subject_id", "sex", "sentence_id", "repetition",
    "level", "text", "signal_path", "start_s", "end_s", "intensity_db",
]


@dataclass(frozen=True)
class Rendered:
    """One utterance to measure, plus everything needed to interpret the measurement."""

    utt_id: str
    wav: Tensor
    condition: str
    #: Columns copied onto the results row.  Identity of the source, and of the target.
    record: dict[str, Any] = field(default_factory=dict)


def utterance_id(stem: str, condition: str, target_index: int | None = None) -> str:
    """``<stem>_real`` / ``<stem>_codec`` / ``<stem>_to3``.

    Unique per file-that-would-have-been-written, which is what the results table keys on
    and what ``speech_eval``'s manifest reader would demand of a manifest.
    """
    if condition != CONVERTED:
        return f"{stem}_{condition}"
    return f"{stem}_to{int(target_index)}"


def stems(metadata: pd.DataFrame) -> list[str]:
    """A stable, unique, filesystem-safe stem per source row."""
    return [
        f"{i:05d}_{Path(path).stem}"
        for i, path in enumerate(metadata["signal_path"])
    ]


def source_records(metadata: pd.DataFrame, level_column: str) -> list[dict]:
    """Identity columns per row, with the source's own level named unambiguously.

    ``source_level`` rather than ``level``: a converted row also carries a
    ``target_level``, and two columns whose names do not say which end they describe is
    how a 4x4 heatmap loses its source axis.
    """
    columns = [c for c in CARRY_COLUMNS if c in metadata.columns]
    records = metadata[columns].to_dict(orient="records")
    for record, (_, row) in zip(records, metadata.iterrows()):
        record["source_level"] = row[level_column]
        record["source_level_index"] = int(row["level_index"])
    return records


class ConversionRenderer:
    """Turns one loaded source batch into its six utterances.

    Parameters
    ----------
    module      : the trained :class:`ConverterCGANv2LabelsModule`; ``convert`` and the
                  codec come from it, so the conversion path is the one training used.
    pipeline    : its conversion-space pipeline (encode / decode).
    stems       : per metadata row, the stem its utterances are named from.
    group_ids   : per metadata row, the sentence group it belongs to.
    tau_src     : per metadata row, its source level in dB.  A level target is *another
                  row's* τ_src, so the whole split has to be measured before the first
                  conversion — see the caller's first pass.
    records     : per metadata row, the identity columns to carry onto its results rows.
    """

    def __init__(
        self,
        module,
        pipeline,
        stems: list[str],
        group_ids: list[str],
        tau_src,
        records: list[dict[str, Any]],
    ):
        self.module = module
        self.pipeline = pipeline
        self.stems = stems
        self.group_ids = group_ids
        self.tau_src = tau_src
        self.records = records

    @torch.no_grad()
    def render(
        self, batch: dict, row: int, targets: list[LevelTarget], with_codec: bool = True
    ) -> list[Rendered]:
        """Render source ``row``.  ``batch`` is its (batch-of-one) loaded audio."""
        wav_batch = batch["wav"]
        stem, group = self.stems[row], self.group_ids[row]
        base = {
            **self.records[row],
            "group_id": group,
            "tau_src_db": float(self.tau_src[row]),
            # Carried on every condition, including `real` itself.  The speaker ceiling
            # needs "the real take this was converted FROM" alongside "the real take it
            # was converted TO", and recovering either by rewriting a filename breaks the
            # moment a stem contains the word it rewrites.
            "source_real_utt_id": utterance_id(stem, REAL),
        }
        empty = {
            "target_level": None, "target_level_index": None,
            "tau_tgt_db": None, "delta_tau_db": None, "reference_utt_id": None,
        }

        out = [Rendered(
            utt_id=utterance_id(stem, REAL),
            wav=wav_batch.unbatch()[0],
            condition=REAL,
            record={**base, **empty, "condition": REAL},
        )]

        z_real = self.pipeline.encode(wav_batch, training=False)

        if with_codec:
            out.append(Rendered(
                utt_id=utterance_id(stem, CODEC),
                wav=self.pipeline.extractor.decode(z_real).unbatch()[0],
                condition=CODEC,
                record={**base, **empty, "condition": CODEC, "delta_tau_db": 0.0},
            ))

        for target in targets:
            tau_value = float(self.tau_src[target.row])
            tau = torch.tensor([tau_value], device=wav_batch.data.device)
            z_fake = self.module.convert(z_real, tau, batch)
            out.append(Rendered(
                utt_id=utterance_id(stem, CONVERTED, target.index),
                wav=self.pipeline.extractor.decode(z_fake).unbatch()[0],
                condition=CONVERTED,
                record={
                    **base,
                    "condition": CONVERTED,
                    "target_level": target.name,
                    "target_level_index": target.index,
                    "tau_tgt_db": tau_value,
                    # Signed: converting up and converting down are different tasks and
                    # must never be pooled by magnitude.
                    "delta_tau_db": tau_value - float(self.tau_src[row]),
                    # The real recording this conversion is to be compared against.
                    "reference_utt_id": utterance_id(self.stems[target.row], REAL),
                },
            ))
        return out

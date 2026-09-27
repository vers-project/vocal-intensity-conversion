"""Objective evaluation of the converter, wired onto ``speech_eval``.

Three pieces, deliberately separable:

``conditions``  what each test utterance becomes — real, codec, four conversions — and
                what identifies each one.  Knows the corpus design, not the metrics.
``harness``     runs ``speech_eval`` metrics over audio held in memory.  Knows the metric
                protocol, not the experiment.
``intensity``   P_φ as a speech_eval metric — the run's own ruler, reading the level of
                its own output, which is the one measurement converted audio has no ground
                truth for.
``pairing``     joins converted rows to the real recording they targeted.  Knows the
                comparison, and runs on a results table without any audio.
``monitor``     the whole evaluation as two calls — render+measure, then fit — so the
                training-time callback and the final scripts cannot drift apart.
``analysis``    the preservation summaries — speaker identity, word error rate — for which
                a displacement slope would be meaningless.  Table to table, no I/O, so the
                training-time monitor and the final analysis read the same numbers.

The split follows ``speech_eval``'s own: measuring one utterance and comparing two are
different jobs, and keeping them apart is what lets the comparison be redesigned without
re-running a single model.
"""
from vic.evaluation.analysis import (
    quality_preservation,
    speaker_preservation,
    wer_preservation,
)
from vic.evaluation.conditions import (
    CARRY_COLUMNS,
    CODEC,
    CONVERTED,
    REAL,
    REFERENCE_UTT_ID,
    SOURCE_REAL_UTT_ID,
    ConversionRenderer,
    Rendered,
    source_records,
    stems,
)
from vic.evaluation.harness import MetricHarness
from vic.evaluation.monitor import (
    ConversionSummary,
    measure_conversions,
    summarise_conversions,
)
from vic.evaluation.intensity import IntensityPredictorMetric
from vic.evaluation.pairing import (
    add_speaker_similarity,
    add_word_errors,
    pair_with_references,
)

__all__ = [
    "CARRY_COLUMNS",
    "CODEC",
    "CONVERTED",
    "REAL",
    "REFERENCE_UTT_ID",
    "SOURCE_REAL_UTT_ID",
    "ConversionRenderer",
    "Rendered",
    "MetricHarness",
    "IntensityPredictorMetric",
    "pair_with_references",
    "add_word_errors",
    "add_speaker_similarity",
    "source_records",
    "stems",
    "speaker_preservation",
    "wer_preservation",
    "quality_preservation",
    "ConversionSummary",
    "measure_conversions",
    "summarise_conversions",
]

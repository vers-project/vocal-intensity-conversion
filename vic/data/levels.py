"""Sentence groups: the same sentence, by the same speaker, at four elicited effort levels.

AVID is recorded sentence-major — one sentence read soft → normal → loud → veryloud, then
the next sentence — so a group of four recordings holds four *measured* intensities of the
same words in the same voice.  That is what lets a conversion be compared against a real
recording at the destination level instead of against a model of one, and it is the corpus
property the evaluation protocol is built on.

This module turns the hand annotation (``speech-annotator``'s ``level`` / ``excluded`` /
``session_index`` / derived ``sentence_id``) into the two things a renderer needs: a group
key per row, and, for each row, the list of levels its own group actually contains.

Why the group key is not just ``sentence_id``
---------------------------------------------
``sentence_id`` counts within each (speaker, session), so on its own it collides across
every speaker in the split.  The key is the whole tuple, and the speaker component must be
``speaker_uid`` rather than a raw ``subject_id`` — ids are unique only within a corpus, and
all 38 FLombard speaker numbers collide with AVID subject_ids while meaning different
people.

Why a partial group is kept
---------------------------
A missing or excluded level costs that one destination, not the sentence.  A group of three
still yields nine ordered pairs, all of them real; discarding it would throw away three
usable recordings to avoid one absent number.  ``group_n_levels`` is written onto every row
so an analysis that wants only complete 4x4 groups can filter after the fact, without
re-rendering anything.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pandas as pd

#: Ordered as elicited, so the index is the effort rank and not an alphabetical accident.
LEVELS = ("soft", "normal", "loud", "veryloud")

#: 1-based: the index appears in output filenames and metadata, where a 0 reads as "none".
LEVEL_INDEX = {name: i + 1 for i, name in enumerate(LEVELS)}

DEFAULT_LEVEL_COLUMN = "level"
DEFAULT_EXCLUDE_COLUMN = "excluded"
DEFAULT_SENTENCE_COLUMN = "sentence_id"
#: ``repetition`` is part of the key, not decoration.  Once the second reading of the list
#: is renumbered from 1 (speech-annotator's ``avid_repetition``), ``sentence_id`` alone no
#: longer separates the two passes, and a group would silently hold eight recordings of
#: the same sentence instead of four.
DEFAULT_GROUP_COLUMNS = ("speaker_uid", "session_index", "repetition", "sentence_id")

#: Columns :func:`prepare_level_groups` adds.
GROUP_COLUMNS_ADDED = ["sentence_group_id", "level_index", "group_n_levels"]


@dataclass(frozen=True)
class LevelTarget:
    """One destination for a conversion: a level, and the real recording that defines it.

    ``row`` is the *sibling's* position in the rendered metadata, which is what makes the
    pairing row-to-row rather than row-to-level-average.  "Convert this soft take of
    sentence 12 to the level of that loud take of sentence 12" is the comparison the
    evaluation wants; a level mean would compare it against a number no recording has.
    """

    name: str
    index: int
    row: int


def prepare_level_groups(
    metadata: pd.DataFrame,
    level_column: str = DEFAULT_LEVEL_COLUMN,
    group_columns: list[str] | tuple[str, ...] = DEFAULT_GROUP_COLUMNS,
    exclude_column: str | None = DEFAULT_EXCLUDE_COLUMN,
    sentence_column: str = DEFAULT_SENTENCE_COLUMN,
    sentence_ids: Sequence | None = None,
) -> pd.DataFrame:
    """Drop unusable rows and add ``sentence_group_id`` / ``level_index`` / ``group_n_levels``.

    Dropped: rows flagged in ``exclude_column``, rows with no ``level``, and rows missing
    any part of the group key.  All three mean the same thing — the annotator did not place
    this segment in the level cycle — and a row that is not placed cannot be a source or a
    destination.

    ``sentence_ids`` further narrows the table to those sentence numbers, keeping whole
    sentences.  A group's completeness is unaffected: every retained sentence keeps all
    the levels it had, and dropping *other* sentences cannot remove one of them.  Useful
    because a render costs one decode per (source, target) pair and the full split is
    fifteen thousand of them, while a listening test or a first metric pass wants a
    handful of sentences at every level.

    Returns a copy with a fresh ``RangeIndex``, because :class:`LevelTarget` refers to rows
    by position.
    """
    # list(), not the caller's sequence: pandas reads a *tuple* of column names as one
    # composite key and raises KeyError on the whole tuple, so the module's own default
    # would fail while a config's YAML list worked.
    group_columns = list(group_columns)
    missing = [c for c in (level_column, *group_columns) if c not in metadata.columns]
    if missing:
        raise ValueError(
            f"The metadata has no {missing} column(s), so sentence groups cannot be "
            f"formed. Level conversions need the speech-annotator output (level, "
            f"excluded, session_index, and the derived sentence_id) joined onto the "
            f"corpus table, or levels.enabled: false to render only the linspace targets."
        )

    out = metadata.copy()
    keep = out[level_column].notna()
    for column in group_columns:
        keep &= out[column].notna()
    if exclude_column is not None and exclude_column in out.columns:
        keep &= ~out[exclude_column].fillna(False).astype(bool)
    out = out[keep].reset_index(drop=True)

    unknown = sorted(set(out[level_column]) - set(LEVELS))
    if unknown:
        raise ValueError(
            f"Unknown vocal-effort level(s) {unknown} in column {level_column!r}; "
            f"expected some of {list(LEVELS)}."
        )

    if sentence_ids is not None:
        if sentence_column not in out.columns:
            raise ValueError(
                f"sentence_ids was given but the metadata has no {sentence_column!r} "
                f"column to select on."
            )
        wanted = {int(value) for value in sentence_ids}
        numbers = pd.to_numeric(out[sentence_column], errors="coerce").astype("Int64")
        # Refuse a number that names nothing rather than quietly rendering fewer sentences
        # than asked for: the selection is stated in a config and nothing downstream would
        # reveal that one of its entries fell through.
        absent = sorted(wanted - set(numbers.dropna().astype(int)))
        if absent:
            raise ValueError(
                f"sentence_ids {absent} match no row of {sentence_column!r} "
                f"(present: {sorted(set(numbers.dropna().astype(int)))})."
            )
        out = out[numbers.isin(wanted).fillna(False)].reset_index(drop=True)

    # Stringified rather than a tuple so the key survives a CSV round trip unchanged, and
    # so one groupby column replaces three everywhere downstream.
    out["sentence_group_id"] = out[group_columns].astype(str).agg("|".join, axis=1)
    out["level_index"] = out[level_column].map(LEVEL_INDEX).astype("Int64")
    out["group_n_levels"] = (
        out.groupby("sentence_group_id")[level_column].transform("nunique").astype("Int64")
    )
    return out


def level_targets(
    metadata: pd.DataFrame,
    level_column: str = DEFAULT_LEVEL_COLUMN,
) -> dict[int, list[LevelTarget]]:
    """For each row position, the destinations its own sentence group offers.

    A row's own level is included: converting to the level a recording already has is the
    identity case, and it is the only confound-free artefact check in the design — asking
    the model to change nothing should change nothing.  Destinations are ordered by effort
    rank, so a filename's level index and its position in the list agree.

    A retake — two segments annotated at the same level in one group — yields two
    destinations at that rank, each naming its own reference recording.  Nothing collapses
    them, because they are two different real takes and the comparison is against a take.
    """
    by_group: dict[str, list[LevelTarget]] = {}
    for group, rows in metadata.groupby("sentence_group_id", sort=False):
        by_group[group] = sorted(
            (
                LevelTarget(name=row[level_column], index=LEVEL_INDEX[row[level_column]],
                            row=int(position))
                for position, row in rows.iterrows()
            ),
            key=lambda target: (target.index, target.row),
        )
    return {
        int(position): by_group[group]
        for position, group in enumerate(metadata["sentence_group_id"])
    }

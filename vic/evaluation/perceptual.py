"""The two-alternative listening test: parsing, scoring and its statistics.

The objective protocol asks whether a measure moved by the amount requested.  It
cannot say whether the change is *audible as* a change of vocal intensity, and it
reads the output through a model that could share a blind spot with the converter.
This module scores the experiment that answers that separately: a listener hears
one converted and one real recording and says which sounds louder.

The design, and what each part of it buys
-----------------------------------------
A trial pairs a converted utterance with a **real** one from the *other* pass of
the same speaker and sentence.  Taking the real member from a different pass is
deliberate: pairing a conversion against the very recording it was derived from
would let a listener answer from residual similarity rather than from loudness.

Scoring reduces each trial to one bit.  The converted member was asked for a
target level $\\tau_t$; the real member has a calibrated level measured from the
corpus.  Whichever is higher *should* be chosen, and a trial is correct when the
listener chose it.  Chance is one half, so the quantity of interest is how far
above one half the listeners land, and how that depends on how far apart the two
levels were.

Two parts of the design are controls rather than measurements:

``real`` vs ``real`` pairs
    Some trials pair two real recordings at different efforts.  Nothing is being
    tested about the model there; it measures whether these listeners, on this
    material and over this range, can do the task at all.  It is the ceiling the
    converted trials should be read against, exactly as the codec round trip is
    the ceiling for a displacement slope.
the identity and near-identity trials
    Because the eight targets are spread over the whole training range while the
    real member takes one of four efforts, some pairs differ by a fraction of a
    decibel.  Those trials *should* sit at chance, and an analysis that pools
    them with the 20 dB pairs reports a number that describes neither.  Accuracy
    is therefore always reported against $|\\Delta|$.

What the reference level is, and what it is not
-----------------------------------------------
For a converted utterance the reference is the level that was **requested**, not
the level $P_\\phi$ reads back from the output.  That is the point of the
experiment: the control interface is a number of decibels, and the question is
whether asking for it produces an audible difference.  Scoring against the
achieved level instead would answer a different and easier question.

For a real utterance the reference is the calibrated $L_\\mathrm{eq}$ of the audio
**as played** — that is, over the span the VAD kept.  The level of the whole
original segment is the wrong number: the stimuli were trimmed, and a segment's
leading and trailing silence drags its $L_\\mathrm{eq}$ down by several decibels.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

#: A PsyToolkit row, once split on whitespace.  Only these positions are read;
#: the rest are timing and cursor coordinates.
COL_FIRST, COL_SECOND, COL_ORDER, COL_SPLIT = 2, 3, 4, 5
COL_KEY, COL_CHOICE, COL_RT, COL_CHOICE2 = 6, 7, 8, 11

#: ``col7 == -1`` with ``col8 == 0`` and ``col12 == 3`` is PsyToolkit's no-response
#: record; the reaction time then sits at the 5 s ceiling.  The two choice columns
#: agree on every answered trial and disagree only here, which is what makes
#: their disagreement a usable integrity check rather than a nuisance.
NO_RESPONSE_KEY = "-1"

#: ``spk9_sent9_rep1_src2_tgt7`` — a converted stimulus.  Without ``_tgt`` it is
#: the real recording the conversion is compared against.
STEM = re.compile(
    r"^spk(?P<speaker>\d+)_sent(?P<sentence>\d+)_rep(?P<repetition>\d+)"
    r"_src(?P<source_level>\d+)(?:_tgt(?P<target>\d+))?$"
)


@dataclass(frozen=True)
class TargetGrid:
    """The requested levels, as ``convert_selection`` laid them out.

    ``n`` targets evenly spaced over ``[min_db, max_db]`` inclusive, indexed from
    1 so that ``tgt1`` is the softest.  Taken from the render config rather than
    assumed: the grid is a property of that run, and reading it from anywhere
    else is how a paper ends up reporting the wrong decibels.
    """

    min_db: float
    max_db: float
    n: int

    def level(self, index: int) -> float:
        if not 1 <= index <= self.n:
            raise ValueError(
                f"target index {index} outside 1..{self.n}; the render used "
                f"{self.n} targets over [{self.min_db}, {self.max_db}] dB."
            )
        return float(
            np.linspace(self.min_db, self.max_db, self.n)[index - 1]
        )


def parse_stem(stem: str) -> dict:
    """``spk9_sent9_rep1_src2_tgt7`` into its fields, or raise saying what failed."""
    match = STEM.match(stem)
    if match is None:
        raise ValueError(
            f"stimulus name {stem!r} does not match the convert_selection "
            "pattern spk<N>_sent<N>_rep<N>_src<N>[_tgt<N>]."
        )
    fields = match.groupdict()
    return {
        "speaker": int(fields["speaker"]),
        "sentence_id": int(fields["sentence"]),
        "repetition": int(fields["repetition"]),
        "source_level_index": int(fields["source_level"]),
        "target_index": (
            int(fields["target"]) if fields["target"] is not None else None
        ),
        "synthetic": fields["target"] is not None,
        "stem": stem,
    }


def parse_session(path: Path) -> pd.DataFrame:
    """One participant's file into one row per trial.

    ``group`` comes from the filename and ``listener`` from the UUID PsyToolkit
    puts there, so two sessions recorded in the same minute stay distinct.

    The two stimulus columns are **not** in presentation order: the converted
    member is always listed first, and the ``"AB"``/``"BA"`` field says which was
    played first.  Reading the columns as the running order silently inverts
    every ``BA`` trial, so the played positions are reconstructed here and the
    raw columns are kept only as ``stimulus_a`` / ``stimulus_b``.
    """
    group = re.search(r"groupe([A-D])", path.name)
    listener = re.search(r"data\.([0-9a-f-]{36})", path.name)
    rows = []
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        parts = line.split()
        if len(parts) <= COL_CHOICE2:
            continue
        order = parts[COL_ORDER].strip('"')
        if order not in ("AB", "BA"):
            raise ValueError(
                f"{path.name} line {number}: order field {order!r} is neither "
                "'AB' nor 'BA'."
            )
        a, b = parts[COL_FIRST], parts[COL_SECOND]
        answered = parts[COL_KEY] != NO_RESPONSE_KEY
        rows.append({
            "group": group.group(1) if group else "?",
            "listener": listener.group(1) if listener else path.name,
            "trial": number,
            "split": parts[COL_SPLIT],
            "stimulus_a": a,
            "stimulus_b": b,
            "order": order,
            # What was actually played first and second.
            "played_first": a if order == "AB" else b,
            "played_second": b if order == "AB" else a,
            "answered": answered,
            # 1 = the first played sounded louder, 2 = the second.
            "choice": int(parts[COL_CHOICE]) if answered else pd.NA,
            "choice_check": int(parts[COL_CHOICE2]) if answered else pd.NA,
            "rt_ms": int(parts[COL_RT]),
        })
    frame = pd.DataFrame(rows)
    disagree = frame["answered"] & (frame["choice"] != frame["choice_check"])
    if disagree.any():
        raise ValueError(
            f"{path.name}: the two choice columns disagree on "
            f"{int(disagree.sum())} answered trial(s); they should agree "
            "wherever a response was given."
        )
    return frame


def read_experiment(directory: Path, split: str = "test") -> pd.DataFrame:
    """Every participant's trials, with the practice trials dropped.

    The practice trial introduces the task and its pair is not part of the
    design, so it is excluded rather than scored.
    """
    paths = sorted(Path(directory).glob("*.txt"))
    if not paths:
        raise FileNotFoundError(f"no participant files under {directory}")
    frame = pd.concat([parse_session(p) for p in paths], ignore_index=True)
    return frame[frame["split"] == split].reset_index(drop=True)


def describe_trials(trials: pd.DataFrame, grid: TargetGrid) -> pd.DataFrame:
    """Add the stimulus fields and the requested level of the converted member."""
    left = pd.DataFrame([parse_stem(s) for s in trials["stimulus_a"]])
    right = pd.DataFrame([parse_stem(s) for s in trials["stimulus_b"]])
    if right["synthetic"].any():
        raise ValueError(
            "a converted stimulus appears in the second column; the render "
            "always lists it first, so the pair columns are not what this "
            "function expects."
        )
    out = trials.copy()
    out["synthetic_trial"] = left["synthetic"].to_numpy()
    for side, part in (("a", left), ("b", right)):
        for field in ("speaker", "sentence_id", "repetition", "source_level_index"):
            out[f"{side}_{field}"] = part[field].to_numpy()
    out["target_index"] = left["target_index"].to_numpy()
    out["a_level_db"] = [
        grid.level(int(i)) if pd.notna(i) else np.nan
        for i in left["target_index"]
    ]
    return out


def score(trials: pd.DataFrame) -> pd.DataFrame:
    """Was the louder member chosen?

    Requires ``a_level_db`` and ``b_level_db``.  Adds the signed difference
    ``delta_db`` (member A minus member B), which member *should* have been
    chosen, and whether it was.  A trial whose two levels are equal has no
    correct answer and is scored ``NA`` rather than counted as a failure.
    """
    out = trials.copy()
    out["delta_db"] = out["a_level_db"] - out["b_level_db"]
    out["abs_delta_db"] = out["delta_db"].abs()

    louder = np.where(out["delta_db"] > 0, "a", "b")
    louder = np.where(out["delta_db"] == 0, "", louder)
    out["louder_member"] = louder
    # Which member the listener picked: position 1/2 maps back through the
    # played order, which is why played_first/second exist.
    picked_stem = np.where(
        out["choice"] == 1, out["played_first"], out["played_second"]
    )
    out["picked_member"] = np.where(
        picked_stem == out["stimulus_a"], "a", "b"
    )
    out.loc[~out["answered"], "picked_member"] = ""
    out["trial_type"] = np.where(
        out["synthetic_trial"], "converted", "control"
    ) if "synthetic_trial" in out.columns else "converted"
    out["correct"] = pd.array(
        np.where(
            (~out["answered"]) | (out["louder_member"] == ""),
            pd.NA,
            out["picked_member"] == out["louder_member"],
        ),
        dtype="boolean",
    )
    return out


def accuracy(
    trials: pd.DataFrame, by: list[str] | None = None, n_boot: int = 10_000,
    seed: int = 0,
) -> pd.DataFrame:
    """Proportion correct with an interval resampled over **listeners**.

    A listener contributes 72 trials and a stimulus pair is seen by everyone in
    its group, so trials are not independent observations.  Resampling them would
    give an interval several times too narrow; resampling listeners keeps the
    dependence intact.  Chance is 0.5, so an interval whose lower bound clears
    0.5 is the claim worth making.
    """
    from speech_eval.compare.core import cluster_bootstrap_ci

    usable = trials[trials["correct"].notna()]
    keys = list(by or [])
    rows = []
    for name, part in (usable.groupby(keys) if keys else [((), usable)]):
        values = part["correct"].astype(float).to_numpy()
        interval = cluster_bootstrap_ci(
            values, part["listener"].to_numpy(),
            statistic=np.mean, n_boot=n_boot, seed=seed,
        )
        row = dict(zip(keys, name if isinstance(name, tuple) else (name,)))
        row.update({
            "n_trials": int(len(values)),
            "n_listeners": int(part["listener"].nunique()),
            "accuracy": float(values.mean()),
            "ci_low": float(interval["ci_low"]),
            "ci_high": float(interval["ci_high"]),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def accuracy_by_delta(
    trials: pd.DataFrame, edges: np.ndarray | None = None, **kwargs
) -> pd.DataFrame:
    """Accuracy against how far apart the two levels were.

    The single most informative view: near zero the task is impossible and
    accuracy must sit at chance, and where it leaves chance is the smallest
    requested difference these listeners could hear.  Pooling across $|\\Delta|$
    instead reports a number that depends mostly on how the grid was sampled.
    """
    edges = np.array([0, 2, 4, 6, 9, 12, 16, 100]) if edges is None else edges
    usable = trials[trials["correct"].notna()].copy()
    usable["delta_bin"] = pd.cut(usable["abs_delta_db"], bins=edges, right=False)
    out = accuracy(usable, by=["delta_bin"], **kwargs)
    out["delta_mid"] = [b.left + (b.right - b.left) / 2 for b in out["delta_bin"]]
    return out


def listener_agreement(trials: pd.DataFrame) -> pd.DataFrame:
    """How often listeners who saw the same pair chose the same member.

    Reported per group, because the four groups see **disjoint** pair sets: no
    two groups share a trial, so there is no pooled rater agreement to compute.
    Two statistics, because each is misleading alone:

    ``majority_share``  mean over pairs of the proportion agreeing with the
                        modal choice.  Directly interpretable, but its floor is
                        1/2 for two raters, not 0.
    ``fleiss_kappa``    agreement above what the marginal choice frequencies
                        would produce by themselves.  0 is chance, 1 is perfect;
                        it can go negative.  With only two raters this is
                        Cohen's kappa and is estimated from very little.
    """
    rows = []
    keys = ["group", "trial_type"] if "trial_type" in trials.columns else ["group"]
    for name, part in trials.groupby(keys):
        name = name if isinstance(name, tuple) else (name,)
        answered = part[part["answered"]]
        counts = []
        for _, pair in answered.groupby(["stimulus_a", "stimulus_b"]):
            picks = pair["picked_member"].value_counts()
            n = int(picks.sum())
            if n < 2:
                continue
            counts.append((n, int(picks.get("a", 0)), int(picks.get("b", 0))))
        if not counts:
            continue
        counts = np.array(counts, dtype=float)
        n_raters = counts[:, 0]
        majority = counts[:, 1:].max(axis=1) / n_raters
        # Fleiss: mean per-item agreement against the chance level implied by the
        # marginal proportions.  Items are weighted equally.
        agree = (
            (counts[:, 1:] ** 2).sum(axis=1) - n_raters
        ) / (n_raters * (n_raters - 1))
        marginals = counts[:, 1:].sum(axis=0) / counts[:, 1:].sum()
        expected = float((marginals ** 2).sum())
        observed = float(agree.mean())
        kappa = (
            (observed - expected) / (1 - expected) if expected < 1 else np.nan
        )
        rows.append({
            **dict(zip(keys, name)),
            "n_listeners": int(answered["listener"].nunique()),
            "n_pairs": int(len(counts)),
            "majority_share": float(majority.mean()),
            "observed_agreement": observed,
            "expected_agreement": expected,
            "fleiss_kappa": float(kappa),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Beyond accuracy: sensitivity and bias
# ---------------------------------------------------------------------------

#: Positive class: the **A** member -- the converted one wherever there is one --
#: is the louder of the pair.  Fixing it this way is what lets the converted
#: trials and the real-vs-real control be described by one metric set.
POSITIVE = "a"


def classification_metrics(trials: pd.DataFrame) -> dict:
    """Accuracy, precision, recall, F1, and the signal-detection pair.

    Accuracy alone hides a response bias: a listener who is reluctant to call the
    converted member louder scores the same as one who is reluctant to call the
    real member louder, and those are different behaviours with different causes.
    Splitting the two error kinds apart shows it.

    With the positive class fixed at :data:`POSITIVE`:

    ``recall``       of the trials where the A member really was louder, the share
                     the listener called louder.  Equals accuracy restricted to
                     the "asked louder" half.
    ``specificity``  the same for the other half, so ``1 - specificity`` is the
                     false-alarm rate.
    ``precision``    of the trials the listener *called* A-louder, the share where
                     it was.  Unlike recall it depends on how the design happens
                     to balance the two halves, which is why both are reported.
    ``d_prime``      sensitivity in standard-deviation units, independent of where
                     the listener put their criterion.  0 is chance.
    ``criterion``    the bias itself: positive means a reluctance to answer
                     "A is louder", negative an eagerness.

    ``d_prime`` and ``criterion`` use the ``1/(2N)`` correction so that a listener
    who never errs in one direction gets a large finite value rather than an
    infinite one.
    """
    from statistics import NormalDist

    usable = trials[trials["correct"].notna()]
    truth = (usable["louder_member"] == POSITIVE).to_numpy()
    picked = (usable["picked_member"] == POSITIVE).to_numpy()

    tp = int((truth & picked).sum())
    fp = int((~truth & picked).sum())
    fn = int((truth & ~picked).sum())
    tn = int((~truth & ~picked).sum())
    n_pos, n_neg = tp + fn, fp + tn

    def safe(numerator, denominator):
        return float(numerator / denominator) if denominator else float("nan")

    recall = safe(tp, n_pos)
    specificity = safe(tn, n_neg)
    precision = safe(tp, tp + fp)
    f1 = safe(2 * precision * recall, precision + recall) if (
        precision == precision and recall == recall and precision + recall > 0
    ) else float("nan")

    # Hit and false-alarm rates, corrected away from 0 and 1.
    def corrected(count, total):
        if not total:
            return float("nan")
        return min(max(count / total, 1 / (2 * total)), 1 - 1 / (2 * total))

    hit = corrected(tp, n_pos)
    fa = corrected(fp, n_neg)
    z = NormalDist().inv_cdf
    d_prime = z(hit) - z(fa) if hit == hit and fa == fa else float("nan")
    criterion = -(z(hit) + z(fa)) / 2 if hit == hit and fa == fa else float("nan")

    return {
        "n": int(len(usable)),
        "n_positive": n_pos,
        "n_negative": n_neg,
        "accuracy": safe(tp + tn, len(usable)),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "hit_rate": hit,
        "false_alarm_rate": fa,
        "d_prime": d_prime,
        "criterion": criterion,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def metrics_table(
    trials: pd.DataFrame, by: list[str] | None = None
) -> pd.DataFrame:
    """:func:`classification_metrics` over each group of ``by``."""
    keys = list(by or [])
    if not keys:
        return pd.DataFrame([classification_metrics(trials)])
    rows = []
    for name, part in trials.groupby(keys, dropna=False):
        row = dict(zip(keys, name if isinstance(name, tuple) else (name,)))
        row.update(classification_metrics(part))
        rows.append(row)
    return pd.DataFrame(rows)


def binomial_p_above_chance(correct: int, n: int) -> float:
    """One-sided exact p for ``correct`` of ``n`` against a chance rate of 1/2."""
    from math import comb

    if n == 0:
        return float("nan")
    return float(
        sum(comb(n, k) for k in range(correct, n + 1)) / 2 ** n
    )


def per_listener_table(trials: pd.DataFrame) -> pd.DataFrame:
    """One row per (listener, trial type), with an exact test against chance.

    The control trials are the screening instrument, but there are only a handful
    of them per listener, so a single listener's control accuracy is a very noisy
    estimate -- the exact p is reported precisely so that a low accuracy on eight
    trials is not mistaken for evidence of inattention.
    """
    out = metrics_table(trials, by=["listener", "group", "trial_type"])
    out["n_correct"] = (out["accuracy"] * out["n"]).round().astype(int)
    out["p_above_chance"] = [
        binomial_p_above_chance(c, n) for c, n in zip(out["n_correct"], out["n"])
    ]
    # Reaction time is the better inattention detector: a listener clicking
    # through is fast *and* at chance, whereas a low control accuracy on a
    # handful of trials is mostly sampling noise.
    rt = (
        trials[trials["answered"]]
        .groupby(["listener", "trial_type"])["rt_ms"]
        .agg(rt_median="median", rt_min="min")
        .reset_index()
    )
    out = out.merge(rt, on=["listener", "trial_type"], how="left")
    return out.sort_values(["trial_type", "accuracy"]).reset_index(drop=True)


def pairwise_agreement(trials: pd.DataFrame) -> pd.DataFrame:
    """Listener-by-listener agreement within each group and trial type.

    Every listener in a group saw the same pairs, so any two of them can be
    compared directly.  A single listener who agrees with nobody shows up here as
    a row of low values, which a group-level summary would average away.
    """
    rows = []
    answered = trials[trials["answered"]]
    for (group, kind), part in answered.groupby(["group", "trial_type"]):
        wide = part.pivot_table(
            index=["stimulus_a", "stimulus_b"], columns="listener",
            values="picked_member", aggfunc="first",
        )
        listeners = list(wide.columns)
        for i, one in enumerate(listeners):
            for other in listeners[i + 1:]:
                both = wide[[one, other]].dropna()
                if both.empty:
                    continue
                rows.append({
                    "group": group, "trial_type": kind,
                    "listener_a": one, "listener_b": other,
                    "n_pairs": int(len(both)),
                    "agreement": float((both[one] == both[other]).mean()),
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Views on the intensity plane
# ---------------------------------------------------------------------------

#: Bins for |Δ| to the paired real recording, i.e. how hard the trial was.
DELTA_EDGES = np.array([0, 2, 4, 6, 9, 12, 16, 100], dtype=float)


def accuracy_by(
    trials: pd.DataFrame, column: str, edges: np.ndarray, **kwargs
) -> pd.DataFrame:
    """Accuracy against any binned continuous covariate, with its bin centre."""
    usable = trials[trials["correct"].notna()].copy()
    usable["_bin"] = pd.cut(usable[column], bins=edges, right=False)
    out = accuracy(usable, by=["_bin"], **kwargs)
    out["x"] = [b.left + (b.right - b.left) / 2 for b in out["_bin"]]
    out["bin"] = out["_bin"].astype(str)
    return out.drop(columns=["_bin"]).sort_values("x").reset_index(drop=True)


def difficulty_model(
    trials: pd.DataFrame, edges: np.ndarray | None = None
) -> pd.DataFrame:
    """The pooled accuracy-versus-|Δ| curve, used as the difficulty baseline.

    Fitted on the converted trials as a whole so that it describes the *task*
    rather than any one region of the intensity plane.
    """
    edges = DELTA_EDGES if edges is None else edges
    return accuracy_by(trials, "abs_delta_db", edges, n_boot=2000)


def predicted_accuracy(
    trials: pd.DataFrame, curve: pd.DataFrame
) -> pd.Series:
    """What the pooled difficulty curve expects of each trial, from its |Δ|.

    Linear interpolation between bin centres, flat outside them.  Crude on
    purpose: it exists to remove the difficulty gradient from a map, not to model
    the psychometric function, and a smooth fit would imply precision the seven
    bins do not have.
    """
    return pd.Series(
        np.interp(trials["abs_delta_db"].to_numpy(dtype=float),
                  curve["x"].to_numpy(dtype=float),
                  curve["accuracy"].to_numpy(dtype=float)),
        index=trials.index,
    )


def source_target_cells(
    trials: pd.DataFrame, curve: pd.DataFrame, chance: float = 0.5
) -> pd.DataFrame:
    """One row per (source level, target level) cell of the intensity plane.

    Three quantities per cell, and the third is the one worth reading:

    ``accuracy``            raw proportion correct.
    ``above_chance``        ``accuracy - chance``, so that a cell at chance reads
                            as neutral on a diverging scale rather than as a
                            middling colour.
    ``residual``            ``accuracy`` minus what the pooled |Δ| curve expects
                            of that cell's trials.  A cell's difficulty is set by
                            the real recordings it happened to be paired against,
                            whose levels vary widely *within* one cell, so raw
                            accuracy confounds "the model failed here" with
                            "these trials were hard".  The residual removes the
                            difficulty and leaves the position effect.
    """
    usable = trials[trials["correct"].notna()].copy()
    usable["_predicted"] = predicted_accuracy(usable, curve)
    grouped = usable.groupby(["source_level_db", "a_level_db"])
    out = grouped.agg(
        n=("correct", "size"),
        accuracy=("correct", lambda v: float(v.astype(float).mean())),
        predicted=("_predicted", "mean"),
        n_positive=("louder_member", lambda v: int((v == POSITIVE).sum())),
    ).reset_index()
    out["above_chance"] = out["accuracy"] - chance
    out["residual"] = out["accuracy"] - out["predicted"]
    # A cell whose trials are all one class has no false-alarm rate, so nothing
    # bias-corrected can be computed for it; flagged rather than silently mixed in.
    out["single_class"] = (out["n_positive"] == 0) | (out["n_positive"] == out["n"])
    return out.rename(columns={"a_level_db": "target_level_db"})

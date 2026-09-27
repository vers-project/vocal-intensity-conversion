"""Step 1: how each listener did on the real-versus-real pairs, and who to keep.

    uv run --extra cpu python perceptual/screen_control.py \\
        --config configs/paper/analyze_perceptual.yaml \\
        --output-dir outputs/perceptual

The control pairs are two **real** recordings at different vocal efforts.
Nothing about the converter is tested there, so a listener who cannot order them
was not doing the task -- and it is the only screen available that does not
select on the quantity under test.  Screening on converted trials, or on all
trials pooled, would pick listeners by the very accuracy the experiment reports.

Both members are recordings, so both get a **measured** level: calibrated
$L_\\mathrm{eq}$ over the VAD-trimmed span, through the same
``FrameLevelTransform`` the conditioning labels used.  Which member is louder is
therefore a measurement, not a request.  (Verified against the rendered audio:
the file-domain level differences reproduce these to within 0.12 dB, so the
listeners really did hear the gaps scored here.)

Two properties of the design the screen has to respect, both printed below:

* the four groups see **disjoint** pair sets, so each group's eight control pairs
  have their own difficulty and a raw 8-trial accuracy is not comparable across
  groups;
* some control pairs differ by a fraction of a decibel -- group C even contains a
  recording paired with itself -- and those trials are coin flips for everyone.
  Counting them punishes listeners for the design rather than for inattention.

Hence the screening set is the control trials above ``floor_db``.  At the default
6 dB that is **exactly five trials in every group**, which is what makes the
count comparable; the script refuses to go on if a floor breaks that balance.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats

from vic.evaluation.perceptual import TargetGrid, describe_trials, read_experiment, score
from vic.evaluation.stimulus_levels import KEY, attach_measured_levels

from experiment_launcher import parse_args


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact interval for a proportion; the normal one is useless at n = 5."""
    lo = 0.0 if k == 0 else float(stats.beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(stats.beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


def homogeneity(k: np.ndarray, n: np.ndarray) -> dict:
    """Do these listeners differ by more than binomial noise?

    With a handful of trials each, a spread of accuracies is what pure chance
    looks like, so this is the question any screen rests on: if the observed
    spread matches the binomial-only spread, the ranking is noise and every cut
    is arbitrary.  It is worth re-running whenever the panel changes -- the
    answer was *no* on the 15-listener pull and is *yes* here.
    """
    pooled = float(k.sum() / n.sum())
    chi2 = float(np.sum((k - n * pooled) ** 2 / (n * pooled * (1 - pooled))))
    observed_var = float(np.var(k / n, ddof=1))
    expected_var = float(np.mean(pooled * (1 - pooled) / n))
    return {
        "pooled": pooled, "chi2": chi2, "dof": len(n) - 1,
        "p": float(stats.chi2.sf(chi2, len(n) - 1)),
        "observed_sd": float(np.sqrt(observed_var)),
        "binomial_sd": float(np.sqrt(expected_var)),
        "ability_sd": float(np.sqrt(max(observed_var - expected_var, 0.0))),
    }


def attentive_share(k: np.ndarray, n: int) -> dict:
    """Two-component fit: a share of listeners doing the task, the rest guessing.

    ``k`` counts of correct answers out of ``n`` easy control trials are modelled
    as a mixture of ``Binom(n, p_attentive)`` and ``Binom(n, 0.5)``.  It turns the
    question "how many of them were not really listening?" into a number, which
    no single threshold can give: a cut answers *who*, this answers *how many*.
    """
    counts = np.bincount(k.astype(int), minlength=n + 1)[: n + 1]
    support = np.arange(n + 1)

    def negative_log_likelihood(theta):
        share, p = theta
        mixed = (share * stats.binom.pmf(support, n, p)
                 + (1 - share) * stats.binom.pmf(support, n, 0.5))
        return -float(counts @ np.log(np.clip(mixed, 1e-12, None)))

    best = min(
        (optimize.minimize(negative_log_likelihood, x0, method="Nelder-Mead")
         for x0 in ((0.7, 0.9), (0.5, 0.95), (0.9, 0.8))),
        key=lambda r: r.fun,
    )
    share, p = best.x
    return {"share_attentive": float(np.clip(share, 0, 1)),
            "p_attentive": float(np.clip(p, 0, 1)),
            "n_guessing": float(len(k) * (1 - np.clip(share, 0, 1)))}


def load_trials(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every trial, and the real-versus-real ones with both members measured."""
    trials = read_experiment(Path(config["experiment_dir"]))
    trials = describe_trials(trials, TargetGrid(**config["targets"]))
    print(f"{len(trials)} test trial(s), {trials['listener'].nunique()} listener(s), "
          f"group sizes {trials.groupby('group')['listener'].nunique().to_dict()}")
    print(f"  converted-vs-real {int(trials['synthetic_trial'].sum())}, "
          f"real-vs-real {int((~trials['synthetic_trial']).sum())}, "
          f"no response {int((~trials['answered']).sum())}")

    control = trials[~trials["synthetic_trial"]].reset_index(drop=True)
    # Both members are recordings, so both levels are measured the same way.
    control = attach_measured_levels(control, "a", "a_measured_db", config)
    control = attach_measured_levels(control, "b", "b_level_db", config)
    control["a_level_db"] = control["a_measured_db"]
    scored = score(control)
    print(f"\nmeasured {control.drop_duplicates([f'a_{c}' for c in KEY]).shape[0]} A-members "
          f"and {control.drop_duplicates([f'b_{c}' for c in KEY]).shape[0]} B-members; "
          f"control |delta| {scored['abs_delta_db'].min():.2f}-"
          f"{scored['abs_delta_db'].max():.2f} dB")
    return trials, scored


def pair_table(scored: pd.DataFrame) -> pd.DataFrame:
    """The distinct control pairs, their measured gap, and how they were judged."""
    scored = scored.assign(pair=scored["stimulus_a"] + " | " + scored["stimulus_b"])
    rows = scored.groupby(["group", "pair"]).agg(
        delta_db=("delta_db", "first"),
        abs_delta_db=("abs_delta_db", "first"),
        louder=("louder_member", "first"),
        n_scored=("correct", "count"),
        accuracy=("correct", "mean"),
    ).reset_index()
    return rows.sort_values(["group", "abs_delta_db"])


def screening_set(scored: pd.DataFrame, floor_db: float) -> pd.DataFrame:
    """The control trials a listener doing the task cannot get wrong.

    The denominator is trials **presented**, not trials answered: above the floor
    every pair has a correct answer, so a trial that ran to the 5 s timeout is a
    listener who did not respond to an unmistakable difference, which is a
    failure of the screen and not missing data.  Scoring it as absent instead
    would quietly reward the least engaged listeners with a shorter test.

    Balance matters more than the exact floor: the count must come out the same
    in every group, or "4 correct" means a different thing per group.
    """
    easy = scored[scored["abs_delta_db"] >= floor_db].copy()
    easy["correct"] = easy["correct"].fillna(False).astype(bool)
    per_group = easy.groupby(["group", "listener"]).size().groupby("group").agg(
        ["min", "max"])
    if per_group["min"].nunique() > 1 or (per_group["min"] != per_group["max"]).any():
        raise ValueError(
            f"a {floor_db} dB floor leaves an uneven screening set "
            f"({per_group.to_dict()}); a count threshold would then mean a "
            "different thing in each group. Pick a floor that balances, or "
            "screen on a proportion."
        )
    return easy


def listener_table(
    trials: pd.DataFrame, scored: pd.DataFrame, easy: pd.DataFrame
) -> pd.DataFrame:
    """One row per listener: the control screen plus two independent flags."""
    usable = scored[scored["correct"].notna()].copy()
    usable["correct"] = usable["correct"].astype(bool)
    rows = []
    for listener, part in trials.groupby("listener"):
        mine = usable[usable["listener"] == listener]
        mine_easy = easy[easy["listener"] == listener]
        k, n = int(mine_easy["correct"].sum()), len(mine_easy)
        lo, hi = clopper_pearson(k, n) if n else (np.nan, np.nan)
        answered = part[part["answered"]]
        rows.append({
            "listener": listener,
            "group": part["group"].iloc[0],
            "n_easy": n, "k_easy": k,
            "easy_accuracy": k / n if n else np.nan,
            "easy_ci_low": lo, "easy_ci_high": hi,
            "p_above_chance": (
                float(stats.binomtest(k, n, 0.5, alternative="greater").pvalue)
                if n else np.nan),
            "n_control_scored": len(mine),
            "control_accuracy": float(mine["correct"].mean()) if len(mine) else np.nan,
            # Independent of the control: someone who skipped trials or answered
            # the same key throughout was not doing the task either.
            "n_missed": int((~part["answered"]).sum()),
            "missed_share": float((~part["answered"]).mean()),
            "side_bias": float(answered["choice"].mean()) if len(answered) else np.nan,
            "rt_median": float(part["rt_ms"].median()),
            "rt_q10": float(part["rt_ms"].quantile(0.10)),
        })
    return pd.DataFrame(rows).sort_values(
        ["easy_accuracy", "control_accuracy"], ascending=False).reset_index(drop=True)


@parse_args
def main(config: dict, output_dir: Path):
    # The launcher passes only the config and the output directory, so the
    # screening rule lives in the config -- which is where it belongs anyway: it
    # is the decision the analysis rests on and it has to travel with the run.
    screen = dict(config.get("screen") or {})
    floor_db = float(screen.get("floor_db", 6.0))
    min_easy_correct = int(screen.get("min_easy_correct", 5))
    max_missed_share = float(screen.get("max_missed_share", 0.10))

    output_dir.mkdir(parents=True, exist_ok=True)
    trials, scored = load_trials(config)
    scored.to_csv(output_dir / "control_trials.csv", index=False)

    pairs = pair_table(scored)
    pairs.to_csv(output_dir / "control_pairs.csv", index=False)
    print("\n=== the control pairs, per group (disjoint sets, so difficulty differs) ===")
    for group, part in pairs.groupby("group"):
        gaps = ", ".join(f"{d:.1f}" for d in part["abs_delta_db"])
        print(f"  Grp{group}: {len(part)} pairs, |delta| = {gaps} dB, "
              f"mean accuracy {part['accuracy'].mean():.3f}")
    identity = pairs[pairs["n_scored"] == 0]
    if len(identity):
        print(f"  ({len(identity)} pair(s) have no correct answer and are never "
              f"scored: {', '.join(identity['pair'])})")

    print("\n=== accuracy against the measured gap, pooled over listeners ===")
    binned = scored[scored["correct"].notna()].copy()
    binned["bin"] = pd.cut(binned["abs_delta_db"], [0, 1, 2, 3, 6, 9, 12, 100],
                           right=False)
    for b, part in binned.groupby("bin", observed=True):
        k, n = int(part["correct"].sum()), len(part)
        lo, hi = clopper_pearson(k, n)
        print(f"  {str(b):>10s} dB  {k / n:.3f} [{lo:.3f}, {hi:.3f}]  n={n}")

    easy = screening_set(scored, floor_db)
    per_listener = easy.groupby(["group", "listener"]).size()
    print(f"\n=== screening set: control trials with |delta| >= {floor_db:.0f} dB ===")
    print(f"  {len(easy)} trials, {per_listener.iloc[0]} per listener in every group, "
          f"pooled accuracy {easy['correct'].mean():.3f}")

    listeners = listener_table(trials, scored, easy)
    listeners.to_csv(output_dir / "control_per_listener.csv", index=False)

    print("\n=== is there anything to rank? ===")
    h = homogeneity(listeners["k_easy"].to_numpy(float),
                    listeners["n_easy"].to_numpy(float))
    print(f"  pooled {h['pooled']:.3f}; observed sd {h['observed_sd']:.3f} vs "
          f"binomial-only {h['binomial_sd']:.3f}")
    print(f"  chi2({h['dof']}) = {h['chi2']:.1f}, p = {h['p']:.2e}  ->  "
          f"between-listener ability sd {h['ability_sd']:.3f}")
    print("  " + ("listeners genuinely differ: a screen is measuring something"
                  if h["p"] < 0.05 else
                  "no more spread than chance: DO NOT screen, the ranking is noise"))

    scored_listeners = listeners[listeners["n_easy"] > 0]
    mix = attentive_share(scored_listeners["k_easy"].to_numpy(),
                          int(scored_listeners["n_easy"].iloc[0]))
    print(f"\n  mixture fit: {100 * mix['share_attentive']:.0f}% of listeners doing the "
          f"task at p = {mix['p_attentive']:.2f}, the remaining "
          f"{mix['n_guessing']:.0f} indistinguishable from guessing")

    n_easy = int(scored_listeners["n_easy"].iloc[0])
    print(f"\n=== the distribution of the {n_easy}-trial screen ===")
    print(f"  {'score':>7s} {'listeners':>9s} {'p vs chance':>12s} {'guessers passing':>17s}")
    for k in range(n_easy, -1, -1):
        count = int((scored_listeners["k_easy"] == k).sum())
        p = stats.binomtest(k, n_easy, 0.5, alternative="greater").pvalue
        print(f"  {k:3d}/{n_easy:<3d} {count:9d} {p:12.3f} {'':17s}")
    # What a threshold buys is not its own pass rate but the purity of what it
    # keeps: a strict cut that keeps 35 clean listeners can beat a loose one that
    # keeps 54 with a tenth of them still guessing.
    n_guess = mix["n_guessing"]
    n_att = len(scored_listeners) - n_guess
    print("  cumulative, and what the kept set would then contain:")
    print(f"    {'rule':>12s} {'kept':>5s} {'guessers pass':>14s} "
          f"{'expected guessers kept':>23s} {'purity':>7s}")
    for k in range(n_easy, 0, -1):
        passing = float(stats.binom.sf(k - 1, n_easy, 0.5))
        surviving_attentive = n_att * float(
            stats.binom.sf(k - 1, n_easy, mix["p_attentive"]))
        surviving_guessers = n_guess * passing
        kept = int((scored_listeners["k_easy"] >= k).sum())
        purity = surviving_attentive / max(surviving_attentive + surviving_guessers, 1e-9)
        print(f"    {f'>= {k}/{n_easy}':>12s} {kept:5d} {100 * passing:13.1f}% "
              f"{surviving_guessers:23.1f} {100 * purity:6.0f}%")

    # ------------------------------------------------------------------
    # The decision.
    # ------------------------------------------------------------------
    fails_control = listeners["k_easy"] < min_easy_correct
    fails_effort = listeners["missed_share"] > max_missed_share
    listeners["excluded"] = fails_control | fails_effort
    listeners["reason"] = np.where(
        fails_control & fails_effort, "control+missed",
        np.where(fails_control, "control", np.where(fails_effort, "missed", "")))
    listeners.to_csv(output_dir / "control_per_listener.csv", index=False)

    kept = listeners[~listeners["excluded"]]
    print(f"\n=== applying the rule: >= {min_easy_correct}/{n_easy} on the control "
          f"and <= {100 * max_missed_share:.0f}% trials missed ===")
    print(f"  excluded {int(listeners['excluded'].sum())} of {len(listeners)} "
          f"({100 * listeners['excluded'].mean():.0f}%), "
          f"{len(kept)} listeners kept")
    for reason, part in listeners[listeners["excluded"]].groupby("reason"):
        print(f"    {reason:15s} {len(part):3d}")
    print(f"  kept per group: {kept.groupby('group').size().to_dict()} "
          f"(was {listeners.groupby('group').size().to_dict()})")
    print(f"  control accuracy of the kept set: "
          f"{easy[easy['listener'].isin(kept['listener'])]['correct'].mean():.3f} "
          f"on the screen, "
          f"{scored[scored['listener'].isin(kept['listener'])]['correct'].mean():.3f} "
          f"on all control trials")

    kept[["listener", "group"]].to_csv(output_dir / "kept_listeners.csv", index=False)
    listeners.loc[listeners["excluded"], ["listener", "group", "k_easy", "n_easy",
                                          "missed_share", "reason"]].to_csv(
        output_dir / "excluded_listeners.csv", index=False)
    print(f"\nWrote control_trials.csv, control_pairs.csv, control_per_listener.csv, "
          f"kept_listeners.csv and excluded_listeners.csv to {output_dir}")


if __name__ == "__main__":
    main()

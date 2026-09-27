# Perceptual experiment analysis

The listening-test analysis, kept together here rather than spread across
`scripts/`. The scoring library stays in `vic/evaluation/perceptual.py` — this
directory holds the analysis steps that run on top of it. Their configuration is
`configs/paper/analyze_perceptual.yaml`.

| file | what it is |
|---|---|
| `screen_control.py` | step 1 — listener screening on the real-versus-real pairs |
| `scale_converted.py` | step 2 — putting conversions on a perceived-dB scale |
| `replot.py` | redraw the figures from the saved tables, refitting nothing |

```bash
uv run --extra cpu python perceptual/screen_control.py \
    --config configs/paper/analyze_perceptual.yaml \
    --output-dir outputs/perceptual
```

## Step 1: screening on the control pairs

A control pair is two **real** recordings, so both members carry a measured
level — calibrated `L_eq` over the VAD-trimmed span, through the same
`FrameLevelTransform` the conditioning labels used. Which member is louder is a
measurement, not a request, which is what makes these trials usable as a screen:
they say whether a listener did the task at all, without selecting on the
converted accuracy the experiment exists to report.

Three things the screen has to respect, all printed by the script:

* **The four groups see disjoint pair sets.** Each group's eight control pairs
  have their own difficulty, so a raw 8-trial accuracy is not comparable across
  groups. The screen uses the trials above an audibility floor, and refuses to
  run if the floor leaves an uneven count per group.
* **Some control pairs are coin flips.** The gaps run from 0.0 to 27.5 dB, and
  group C even contains a recording paired with itself. Counting those trials
  punishes listeners for the design. At the default 6 dB floor the screen is
  exactly five trials per listener in every group.
* **A timeout on an easy control trial is a failure, not missing data.** The
  denominator is trials presented. Treating a timeout as absent would give the
  least engaged listeners a shorter test.

Two flags are recorded per listener, and the rule uses both: the control score,
and the share of all 72 trials left unanswered.

### Before trusting any of it

`homogeneity()` asks whether the listeners differ by more than binomial noise. On
a small panel the answer can be *no*, and then the ranking is noise and every cut
is arbitrary — that was the case on the 15-listener pull of 2026-09-14
(χ²(14) = 13.8, p = 0.46). Re-run it whenever the panel changes; the script
prints the verdict in words.

`attentive_share()` then fits a two-component mixture — a share of listeners
doing the task, the rest guessing — which answers *how many* are bad, where a
threshold only answers *who*. Its estimate is what turns the cumulative table
into a purity column: the useful property of a cut is not its own pass rate but
how clean the set it keeps is.

## Adjusting a figure

`scale_converted.py` measures audio, fits probits and runs a 2000-draw bootstrap.
None of that changes when a marker size does, and everything the figures need is
written to the output directory. So iterate on the figures with

```bash
uv run --extra cpu python perceptual/replot.py outputs/perceptual
```

and re-run `scale_converted.py` only when the data or the model changes.

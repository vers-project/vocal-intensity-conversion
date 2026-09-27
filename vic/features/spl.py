"""Calibrated sound pressure level computation.

Offline workflow (label preprocessing)
---------------------------------------
1. ``compute_calibration_rms()`` — run once per calibration tone file; store
   the result in the metadata CSV as ``calibration_rms``.
2. ``LeqZFTransform`` — run once per speech file to produce the sequence-level
   ``intensity_db`` label (dB SPL at target distance, IEC "F" time-weighted).

Online workflow (dataset __getitem__)
--------------------------------------
3. ``FrameLevelTransform`` — applied per sample in the dataset to produce
   frame-aligned calibrated dB SPL labels without time-weighting.  The model's
   self-attention provides temporal context; smoothing is applied *after*
   prediction for perceptual experiments via ``apply_f_smoothing_frame()``.

Silence and numerical floors
----------------------------
A digitally silent frame does **not** produce -inf.  ``amplitude_to_db`` clamps
its argument at ``eps=1e-10``, so an all-zero frame reads -200 dB, and the
unconditional distance correction then shifts it:

    distance_m   20*log10(d/1.0)   label for exact digital silence
      0.05 (AVID)     -26.02              -226.02 dB
      0.30 (VIC, FLombard)  -10.46        -210.46 dB

The floor is a **constant**: the clamp fires before ``calibration_rms`` can
matter.  It is also unreachable on real audio -- measured over ~196 000 frames
across AVID, VIC and FLombard, the minimum is **+2.04 dB** and no frame falls
below -20 dB, because a recording's noise floor keeps the RMS far from zero.
Both float32 cliffs are ~290 dB away too: ``10**(x/10)`` overflows near +385 dB
and stays normal well past -450 dB.

The value is left absurd on purpose.  A label near -226 dB means the audio was
*exactly* zero -- a dropout, a gate, a broken file -- and clipping it to
something physical (0 dB SPL, say) would turn that into a plausible-looking
quiet frame and lose the only signal that something is wrong.

Two places where zeros do enter, both bounded at 3 dB and one frame each:
frame 0's analysis window is half zero-padding by construction (see
``FrameLevelTransform.__call__``), and the final frame picks up the zeros that
``pad_to_multiple`` appends upstream.  Both behave identically at train and test.

Three silence conventions coexist in this codebase and none of them is shared:
``audio_utils.data.transforms.SILENCE_DBFS`` is -120, ``vic/data/transforms.py``
uses -160, and this module's SPL path uses -200 (amplitude) / -300 (power).  The
first two are **dBFS** and are not used for labels.

Reference: IEC 61672-1:2013, "Electroacoustics — Sound level meters".
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor

from vic.core import FrameGrid


P0: float = 20e-6   # reference sound pressure: 20 µPa


def db_to_power(db: Tensor) -> Tensor:
    """Convert dB-scaled values to linear power ratios."""
    return 10.0 ** (db / 10.0)


def power_to_db(power: Tensor, eps: float = 1e-30) -> Tensor:
    """Convert linear power ratios to dB scale.

    ``eps`` floors the result at **-300 dB** for a zero input.  Note this does
    not agree with :func:`amplitude_to_db`, whose floor is -200 dB; the two
    functions disagree about what silence is, which is invisible at the call
    site.  Nothing in the SPL path reaches either floor -- see the module
    docstring.
    """
    return 10.0 * torch.log10(power.clamp(min=eps))


def db_to_amplitude(db: Tensor) -> Tensor:
    """Convert dB-scaled values to linear amplitude."""
    return 10.0 ** (db / 20.0)


def amplitude_to_db(amplitude: Tensor, eps: float = 1e-10) -> Tensor:
    """Convert linear amplitude values to dB scale.

    ``eps`` floors the result at **-200 dB** instead of -inf.  Callers pass the
    *ratio* ``p_rms / P0``, so the implied pressure floor is ``1e-10 * 20e-6 =
    2e-15 Pa`` -- only exact digital zero saturates it, and a frame at 1e-9 Pa
    still reads a true -86 dB.  See the module docstring for why the value is
    left absurd rather than raised to something physical.
    """
    return 20.0 * torch.log10(amplitude.clamp(min=eps))


def compute_calibration_rms(wav: Tensor) -> float:
    """Compute the RMS amplitude of a calibration tone recording.

    The calibration tone is produced by an acoustic calibrator (e.g. a
    pistonphone at 94 dB SPL = 1 Pa).  Its RMS in the recording corresponds
    to that reference pressure, so dividing any co-recorded signal by this
    value converts it to Pascal.

    Run this once per calibration file and store the result in the metadata
    CSV as ``calibration_rms`` to avoid reloading calibration audio at
    training time.

    Parameters
    ----------
    wav : (C, T) calibration tone recording.

    Returns
    -------
    RMS amplitude as a Python float.
    """
    return wav.pow(2).mean().sqrt().item()


class FrameLevelTransform:
    """Per-frame calibrated sound pressure level, without time-weighting.

    Computes the physical SPL at each codec frame position by taking the RMS of
    an analysis window centred on the frame, calibrating to Pascal, and
    converting to dB SPL at the target distance.  No IEC time-weighting is
    applied — use ``apply_f_smoothing()`` on the model's output for perceptual
    experiments.

    Parameters
    ----------
    grid               : FrameGrid of the codec/extractor — determines frame
                         centres and the total number of frames.
    window_length      : analysis window in samples.  Can be larger than
                         ``grid.hop_size`` for overlapping frames (e.g.
                         ``2 * grid.hop_size`` gives 50 % overlap).
    target_distance_m  : reference distance in metres at which the SPL is
                         expressed.  Default 1.0 m.
    """

    def __init__(self, grid: FrameGrid, window_length: int, target_distance_m: float = 1.0):
        self.grid = grid
        self.window_length = window_length
        self.target_distance_m = target_distance_m

    def __call__(
        self,
        wav: Tensor,
        calibration_rms: float,
        distance_m: float,
    ) -> Tensor:
        """Compute per-frame calibrated SPL for one utterance.

        Parameters
        ----------
        wav             : (C, T) speech waveform in raw ADC units.
        calibration_rms : RMS of the matching calibration tone.
        distance_m      : mouth-to-microphone distance in metres.

        Returns
        -------
        (T_frames,) calibrated dB SPL at ``self.target_distance_m``, one value per codec frame.

        Edge frames
        -----------
        The signal is ``F.pad``-ed by ``window_length // 2`` on each side, so in
        the non-causal case **frame 0's window is exactly half zeros** and reads
        a hard **-3.01 dB** low.  That happens on every item, unconditionally.

        The last frame is not touched by *this* padding -- with
        ``window_length <= 2 * hop_size`` the final kept window ends
        ``window_length // 2 - hop_size`` samples short of the signal end, which
        also means the trailing 120 samples (at 400/320) are never analysed by
        any frame.  It *is* touched by ``pad_to_multiple`` upstream:
        ``AudioDataset._load`` appends 0-319 zero samples **before** labels are
        computed here, and the last window overlaps them once more than 120 are
        added -- measured 0 dB at 120, -1.25 dB at 220, -2.99 dB at 319.

        Both artefacts are bounded at 3 dB, affect one frame each, and behave
        identically at train and test, so they introduce no mismatch.  An
        all-zero frame is a different matter entirely -- see the module
        docstring.
        """
        p = wav.mean(0) / calibration_rms          # (T,) mono, in Pa
        half = self.window_length // 2

        if self.grid.causal:
            # Causal: frame i analyses p[i*hop : i*hop + window_length]
            pad_left, pad_right = 0, self.window_length - 1
        else:
            # Non-causal: frame i analyses p centered at i*hop
            pad_left, pad_right = half, half

        p_padded = F.pad(p, (pad_left, pad_right))
        n_frames = self.grid.n_frames(p.shape[0])
        frames = p_padded.unfold(0, self.window_length, self.grid.hop_size)[:n_frames]  # (T_frames, window_length)

        p_rms = frames.pow(2).mean(-1).sqrt()       # (T_frames,)
        out = amplitude_to_db(p_rms / P0)

        # Distance correction  (free-field inverse-square law)
        #   L_target = L_d + 20·log10(d / d_target)
        return out + 20.0 * math.log10(
            max(distance_m, 1e-10) / max(self.target_distance_m, 1e-10)
        )


def leq_aggregate(frame_preds: Tensor, padding_mask: Tensor | None = None) -> Tensor:
    """Equivalent level of per-frame dB predictions (energy average in linear domain).

    Correct aggregation for dB-scaled predictions:
        L_eq = 10 * log10( mean_i( 10^(L_i / 10) ) )

    Parameters
    ----------
    frame_preds  : (B, T) per-frame predictions in dB.
    padding_mask : (B, T) True = valid frame (not padding).  If None, all
                   frames are treated as valid.

    Returns
    -------
    (B,) sequence-level equivalent level in dB.
    """
    if padding_mask is None:
        padding_mask = torch.ones_like(frame_preds, dtype=torch.bool)
    linear = db_to_power(frame_preds)
    mask = padding_mask.float()
    mean_power = (linear * mask).sum(-1) / mask.sum(-1).clamp(min=1)
    return power_to_db(mean_power)


def leq_windows(
    frame_preds: Tensor,
    frame_labels: Tensor,
    padding_mask: Tensor | None,
    window_frames: int,
) -> tuple[Tensor, Tensor]:
    """Leq of predictions and labels over fixed-length windows, batch-flattened.

    Two different hops are involved and they are easy to confuse
    -----------------------------------------------------------
    The **frame grid** hop (320 samples at 16 kHz) is what makes one SPL frame
    every 20 ms.  It is fixed by the extractor and is not a parameter here: it
    only decides that a 3.2 s utterance arrives as 160 frames.

    The **window** is ``window_frames`` of those frames -- 100 frames = 2 s at
    the usual grid -- and it is the unit the metric scores.  Where the windows
    sit inside an item is this function's only real choice, and it is not a
    one-frame slide: consecutive windows are roughly a whole window apart.

    Why window at all
    -----------------
    ``leq_aggregate`` gives one number per item, so an item's difficulty depends
    on its length: aggregating over more frames averages part of the error away.
    Measured on this repo's own runs, going from chunked to whole-utterance
    aggregation lowers Leq RMSE by 5-13%.  That makes a whole-utterance Leq
    incomparable between corpora whose length distributions differ -- and it
    weights a test set oddly, since one 46 s paragraph and one 2 s sentence each
    contribute a single point.

    Equal-length windows fix both: every unit spans the same duration, and a long
    item contributes proportionally more of them, which is how the frame-level
    metrics already count.

    Worked example: a 3.2 s sentence, 100-frame window
    --------------------------------------------------
    3.2 s = 160 frames, so proportional weighting would want 160/100 = 1.60
    windows.  Only whole numbers exist, and ``ceil(1.60) = 2``::

        frame   0        50       100      150 160
                |---------|---------|---------|--|
        win 1   [#########################       ]   frames   0-99
        win 2   [       #########################]   frames  60-159

    The last window is slid back so it *ends* at the item's final frame.  Nothing
    is missed; frames 60-99 fall in both windows and are counted twice.

    Why ``ceil`` and not something else
    -----------------------------------
    Covering every frame with windows of exactly ``window_frames`` needs
    ``n_windows * window_frames >= n``, i.e. ``n_windows >= ceil(n / W)``.  So
    **equal-length windows, full coverage and exactly duration-proportional
    counts cannot all hold at once** -- some rule has to give.  Measured on the
    AVID test split at a 2 s window (paragraphs are 46.8% of the audio, which is
    what "proportional" means here):

        rule            3.2 s   30.6 s   coverage   paragraph share   units
        ceil  (this)        2       16     100.0%             41.1%    4697
        round               2       15      95.5%             47.0%    3972
        dense (hop 1)      61     1431     100.0%             67.4%  257837
        floor               1       15      85.3%             52.6%    3423

    ``ceil`` rounds up always, and one extra window is +25% on a 1.6-window
    sentence but only +5% on a 15.3-window paragraph.  That systematic bonus to
    short items is the whole 41%-vs-47% gap; there is nothing subtler in it.

    ``round`` keeps the counts proportional but rounds *down* sometimes, and then
    the windows no longer reach across the item: 15 windows spaced 102 frames
    apart with a 100-frame window leave 2-frame holes, and 4.5% of the test audio
    lands in no window at all.

    ``floor`` was rejected for a stronger reason than arithmetic.  It keeps only
    the *first* ``window_frames`` of a short item -- a quarter of all sentence
    audio, always its tail.  Since a sentence segment's leading silence sits at
    its start, that would systematically over-sample silence.  A hole in the
    middle of a paragraph is a rounding artefact; always discarding the tail is a
    bias.

    A dense hop of one frame is a reasonable first instinct and does work.  It
    removes the arbitrary tiling phase, which is a genuine advantage.  But it
    yields ``n - W + 1`` windows, and losing a fixed ``W - 1`` start positions
    costs a 160-frame sentence almost everything while costing a 1530-frame
    paragraph almost nothing -- so long items end up over-weighted (67% of units
    against 47% of the audio), the mirror image of the problem being fixed.  It
    also buys no statistical precision: windows one frame apart share 99 of their
    100 frames, so 61 of them carry about the information of 2 independent ones,
    and an error bar computed from the unit count would be several times too
    narrow.

    Other guarantees
    ----------------
    Windows are cut from each item's **valid** frames only, so padding never
    enters one -- the same guarantee ``leq_aggregate`` gives through its mask.

    An item shorter than one window yields one short window rather than being
    dropped, because discarding short utterances would bias the metric toward the
    long ones this function exists to stop over-weighting.

    Parameters
    ----------
    frame_preds  : (B, T) per-frame predictions in dB.
    frame_labels : (B, T) per-frame labels in dB, same shape.
    padding_mask : (B, T) True = valid frame.  None treats every frame as valid.
    window_frames : window length in frames; must be >= 1.

    Returns
    -------
    ``(pred, label)``, both 1-D and of equal length: one entry per window, all
    items concatenated.  Empty tensors if no item has a valid frame.
    """
    if window_frames < 1:
        raise ValueError(f"window_frames must be >= 1, got {window_frames}")
    if frame_preds.shape != frame_labels.shape:
        raise ValueError(
            f"predictions {tuple(frame_preds.shape)} and labels "
            f"{tuple(frame_labels.shape)} must have the same shape"
        )
    if padding_mask is None:
        padding_mask = torch.ones_like(frame_preds, dtype=torch.bool)

    pred_out: list[Tensor] = []
    label_out: list[Tensor] = []
    for item in range(frame_preds.shape[0]):
        keep = padding_mask[item]
        pred, label = frame_preds[item][keep], frame_labels[item][keep]
        n = pred.shape[0]
        if n == 0:
            continue
        if n <= window_frames:
            pred_out.append(leq_aggregate(pred.unsqueeze(0)))
            label_out.append(leq_aggregate(label.unsqueeze(0)))
            continue
        n_windows = -(-n // window_frames)          # ceil
        starts = torch.linspace(
            0, n - window_frames, n_windows, device=pred.device
        ).round().long()
        index = starts.unsqueeze(1) + torch.arange(window_frames, device=pred.device)
        pred_out.append(leq_aggregate(pred[index]))
        label_out.append(leq_aggregate(label[index]))

    if not pred_out:
        empty = frame_preds.new_empty(0)
        return empty, empty.clone()
    return torch.cat(pred_out), torch.cat(label_out)


def apply_f_smoothing(
    frame_preds: Tensor,
    frame_rate: float,
    tau: float = 0.125,
) -> Tensor:
    """Apply IEC "F" exponential time-weighting to frame-level dB predictions.

    Operates at frame rate (not sample rate).  Converts predictions to linear
    power, applies a zero-phase first-order IIR (filtfilt), then converts back
    to dB — matching the behaviour of ``LeqZFTransform._time_weight``.

    Parameters
    ----------
    frame_preds : (..., T) per-frame predictions in dB.
    frame_rate  : frames per second (= sample_rate / hop_size).
    tau         : time constant in seconds.  Default 0.125 s = IEC "F" (Fast).

    Returns
    -------
    (..., T) smoothed level in dB.
    """
    from torchaudio.functional import filtfilt

    alpha = 1.0 - math.exp(-1.0 / (frame_rate * tau))
    b = torch.tensor([alpha, 0.0], dtype=frame_preds.dtype, device=frame_preds.device)
    a = torch.tensor([1.0, -(1.0 - alpha)], dtype=frame_preds.dtype, device=frame_preds.device)

    linear = db_to_power(frame_preds)
    smoothed = filtfilt(linear, a_coeffs=a, b_coeffs=b, clamp=False).clamp(min=1e-30)
    return power_to_db(smoothed)


def apply_f_smoothing_frame(
    frame_preds: Tensor,
    sample_rate: int,
    hop_size: int,
    tau: float = 0.125,
) -> Tensor:
    """Apply IEC "F" exponential time-weighting to frame-level dB predictions.

    Equivalent to ``apply_f_smoothing`` but takes ``sample_rate`` and
    ``hop_size`` explicitly so the frame rate (sample_rate / hop_size) is
    computed internally — avoiding the silent error of accidentally passing
    the audio sample rate instead of the frame rate.

    Parameters
    ----------
    frame_preds : (..., T) per-frame predictions in dB, one value per codec frame.
    sample_rate : audio sample rate in Hz.
    hop_size    : codec/extractor hop size in samples.
    tau         : time constant in seconds.  Default 0.125 s = IEC "F" (Fast).

    Returns
    -------
    (..., T) smoothed level in dB.
    """
    return apply_f_smoothing(frame_preds, frame_rate=sample_rate / hop_size, tau=tau)


class LeqZFTransform:
    """Sequence-level LeqZF (dB SPL at a target distance) for a single utterance.

    LeqZF: Equivalent Continuous Sound Level with Z-frequency weighting
    (flat response) and F (Fast, τ = 125 ms) exponential time-weighting,
    per IEC 61672-1:2013.

    Kept as a class (not a plain function) because ``sample_rate`` and
    ``tau`` are fixed configuration that should not be repeated at every
    call site — consistent with the ``ExtractF0`` pattern.

    Parameters
    ----------
    sample_rate        : audio sample rate in Hz.
    tau_time_weighting : exponential time constant in seconds.
                         Default 0.125 s = IEC "F" (Fast) weighting.
    target_distance_m  : reference distance in metres at which the SPL is
                         expressed.  Default 1.0 m.
    """

    def __init__(
        self,
        sample_rate: int,
        tau_time_weighting: float = 0.125,
        target_distance_m: float = 1.0,
    ):
        self.sample_rate = sample_rate
        self.tau = tau_time_weighting
        self.target_distance_m = target_distance_m

    # ------------------------------------------------------------------

    def _time_weight(self, p2: Tensor) -> Tensor:
        """Apply IEC "F" exponential time-weighting to squared pressure.

        First-order IIR:  y[n] = α·x[n] + (1−α)·y[n−1]
                          α = 1 − exp(−1 / (sr · τ))

        Applied zero-phase (filtfilt) to match the original implementation.

        Parameters
        ----------
        p2 : (..., T) squared pressure in Pa².

        Returns
        -------
        (..., T) time-weighted squared pressure (non-negative).
        """
        from torchaudio.functional import filtfilt
        alpha = 1.0 - math.exp(-1.0 / (self.sample_rate * self.tau))
        b = torch.tensor([alpha, 0.0], dtype=p2.dtype, device=p2.device)
        a = torch.tensor([1.0, -(1.0 - alpha)], dtype=p2.dtype, device=p2.device)
        return filtfilt(p2, a_coeffs=a, b_coeffs=b, clamp=False).clamp(min=0.0)

    # ------------------------------------------------------------------

    def __call__(
        self,
        wav: Tensor,
        calibration_rms: float,
        distance_m: float,
    ) -> float:
        """Compute LeqZF for one utterance.

        Parameters
        ----------
        wav             : (C, T) speech waveform in raw ADC units.
        calibration_rms : RMS of the matching calibration tone (from
                          ``compute_calibration_rms``). Converts the signal
                          to Pascal.
        distance_m      : mouth-to-microphone distance in metres.

        Returns
        -------
        LeqZF in dB SPL at ``target_distance_m``, as a Python float.
        """
        # 1. Mix to mono and convert to Pascal
        p = wav.mean(0) / calibration_rms          # (T,) in Pa

        # 2. Apply time-weighting to squared pressure
        p2_tw = self._time_weight(p.pow(2))         # (T,)

        # 3. Time-averaged mean-square pressure → RMS
        p_rms = p2_tw.mean().sqrt()                  # 0-dim tensor

        # 4. Convert to dB SPL  (ref 20 µPa)
        level = amplitude_to_db(p_rms / P0).item()

        # 5. Distance correction  (free-field inverse-square law)
        #    L_target = L_d + 20·log10(d / d_target)
        level += 20.0 * math.log10(
            max(distance_m, 1e-10) / max(self.target_distance_m, 1e-10)
        )

        return level

"""Feature-level normalization for batched AudioBatch tensors."""
from __future__ import annotations

from vic.data.audio_batch import AudioBatch


def utterance_normalize_per_band(z_batch: AudioBatch) -> AudioBatch:
    """Subtract per-band mean over valid frames from log-mel features (per-band CMN).

    Removes the per-band mean spectral shape — which conflates the recording chain's
    frequency response (microphone, room) with the speaker's static spectral tilt —
    while preserving the *temporal dynamics* of each band (voiced/unvoiced contrast,
    frame-to-frame tilt changes).

    Unlike ``utterance_normalize`` (one global scalar per utterance), this subtracts
    a different value per frequency band, zeroing the time-averaged spectral envelope.
    Spectral tilt is therefore not preserved.  Use ``utterance_normalize`` if spectral
    tilt should be available to the model; use this function to test whether removing
    the recording-chain spectral fingerprint improves cross-dataset generalisation.

    Parameters
    ----------
    z_batch : AudioBatch (B, C, T_frames) log-mel features with right-padding.

    Returns
    -------
    AudioBatch of the same shape with zero per-band mean per utterance.
    """
    data = z_batch.data                                        # (B, C, T_frames)
    mask = z_batch.padding_mask.float()                        # (B, T_frames)  1 = valid

    n_valid = mask.sum(-1).clamp(min=1)                        # (B,)
    band_means = (data * mask.unsqueeze(1)).sum(-1) / n_valid.unsqueeze(1)  # (B, C)

    return AudioBatch(
        data=data - band_means.unsqueeze(-1),
        lengths=z_batch.lengths,
        sample_rate=z_batch.sample_rate,
    )


def utterance_normalize(z_batch: AudioBatch) -> AudioBatch:
    """Subtract a per-utterance global scalar mean from log-mel features.

    Removes the mean energy level (a crest-factor proxy that leaks absolute
    amplitude information after peak normalisation) while preserving the
    production-level cues needed for intensity prediction:
      - spectral tilt  (relative HF/LF energy ratio — unchanged by a scalar shift)
      - voiced/unvoiced contrast depth  (frame-to-frame differences — unchanged)
      - formant patterns  (spectral shape — unchanged)

    One scalar is computed per utterance — the mean over all valid frames and
    all feature channels — and subtracted from every element.  Padding frames
    are excluded from the mean computation but are also shifted (harmless since
    they are masked out downstream).

    Parameters
    ----------
    z_batch : AudioBatch (B, C, T_frames) log-mel features with right-padding.

    Returns
    -------
    AudioBatch of the same shape with zero global mean per utterance.
    """
    data = z_batch.data                              # (B, C, T_frames)
    mask = z_batch.padding_mask.float()              # (B, T_frames)  1 = valid

    # Sum over valid frames and all channels → one scalar per utterance.
    total_sum = (data * mask.unsqueeze(1)).sum(dim=(1, 2))   # (B,)
    n_valid = mask.sum(-1) * data.shape[1]           # (B,)  valid_frames × C
    mean = total_sum / n_valid.clamp(min=1)          # (B,)

    return AudioBatch(
        data=data - mean.view(-1, 1, 1),
        lengths=z_batch.lengths,
        sample_rate=z_batch.sample_rate,
    )

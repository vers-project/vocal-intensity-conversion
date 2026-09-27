"""Training-time augmentations (batched, GPU-compatible)."""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from vic.data.audio_batch import AudioBatch

if TYPE_CHECKING:
    from vic.core import AudioCodec
    from vic.encoders.mp_senet import MPSENetDenoiser


def add_noise(wav: Tensor, snr_min_db: float, snr_max_db: float) -> Tensor:
    """Add white noise at a per-utterance SNR drawn uniformly from [snr_min_db, snr_max_db].

    Decorrelates the noise floor from the speech level after peak normalisation,
    preventing the model from using the relative noise floor as a proxy for
    absolute vocal intensity.

    SNR is defined as: SNR_dB = 10 * log10(P_signal / P_noise).
    Rearranging: P_noise = P_signal / 10^(SNR_dB / 10).
    The noise scale satisfying this is: scale = sqrt(P_signal / (P_noise_unit * 10^(SNR_dB/10))).

    Parameters
    ----------
    wav        : (B, C, T) waveform batch.
    snr_min_db : lower bound of the SNR range in dB.
    snr_max_db : upper bound of the SNR range in dB.

    Returns
    -------
    (B, C, T) noisy waveform, same dtype and device as input.
    """
    B = wav.shape[0]

    # Per-utterance signal power — keepdim for broadcasting against (B, C, T).
    signal_power = wav.pow(2).mean(dim=(-1, -2), keepdim=True).clamp(min=1e-10)  # (B, 1, 1)

    # Independent SNR target for each utterance in the batch.
    snr_db = torch.empty(B, 1, 1, device=wav.device, dtype=wav.dtype).uniform_(snr_min_db, snr_max_db)

    # Unit white noise — measure actual power rather than assuming exactly 1.0.
    noise = torch.randn_like(wav)
    noise_power = noise.pow(2).mean(dim=(-1, -2), keepdim=True).clamp(min=1e-10)  # (B, 1, 1)

    # Scale noise so that P_signal / P_scaled_noise = 10^(snr_db / 10).
    scale = (signal_power / (noise_power * 10.0 ** (snr_db / 10.0))).sqrt()
    return wav + scale * noise


@torch.no_grad()
def codec_resynthesis(wav: AudioBatch, codec: AudioCodec) -> AudioBatch:
    """Resynthesize waveforms through a frozen codec (encode → RVQ quantize → decode).

    The codec bottleneck strips recording-specific artefacts — noise floor, room
    acoustics, microphone response — while preserving the speech content and the
    vocal effort characteristics encoded in the latent space.  The output is a
    clean, codec-faithful version of the input, free from the recording conditions
    that cause models to overfit to a particular dataset.

    Intended as a training-only waveform augmentation applied before feature
    extraction, analogous to ``add_noise`` but operating via lossy compression
    rather than additive noise.

    Parameters
    ----------
    wav   : AudioBatch (B, 1, T_samples) waveform batch.
    codec : AudioCodec instance (e.g. SpeechTokenizerCodec).

    Returns
    -------
    AudioBatch with resynthesized waveforms at the same sample rate, trimmed to
    the original sample lengths (the codec decode may produce slightly more or
    fewer samples due to strided convolution rounding).
    """
    z = codec.encode(wav)        # (B, D, T_frames)
    resynth = codec.decode(z)    # (B, 1, T_resynth)
    T_orig = wav.data.shape[-1]
    T_resynth = resynth.data.shape[-1]
    return AudioBatch(
        data=resynth.data[..., :T_orig],
        lengths=wav.lengths.clamp(max=T_resynth).clamp(max=T_orig),
        sample_rate=wav.sample_rate,
    )


@torch.no_grad()
def speech_enhancement(wav: AudioBatch, denoiser: "MPSENetDenoiser") -> AudioBatch:
    """Denoise waveforms with a frozen MP-SENet generator.

    Suppresses background noise (HVAC, room noise, mic hiss) while preserving
    speech content and vocal effort characteristics.  Intended as a
    training-only waveform augmentation applied before feature extraction.

    Parameters
    ----------
    wav     : AudioBatch (B, 1, T_samples) waveform batch at 16 kHz.
    denoiser: MPSENetDenoiser instance.

    Returns
    -------
    AudioBatch with background noise suppressed, same shape and sample rate.
    """
    return denoiser(wav)


@torch.no_grad()
def spectral_whitening(wav_batch: AudioBatch, n_fft: int = 1024, hop_length: int = 320) -> AudioBatch:
    """Per-utterance STFT-domain spectral whitening.

    Mirrors ``utterance_normalize_per_band`` but operates on raw waveforms
    before feature extraction.  For each frequency bin f, the time-averaged
    log-magnitude is subtracted from every frame:

        L_norm[f, t] = log|X[f, t]| − mean_t(log|X[f, t]|)

    The original phase is preserved, so iSTFT reconstruction is near-lossless
    (no musical noise, no vocoder artefacts).  Removes the recording-chain
    spectral fingerprint (microphone response, room acoustics) that NAC
    latents cannot abstract away via feature-level normalisation.

    Parameters
    ----------
    wav_batch  : AudioBatch (B, 1, T) waveform batch.
    n_fft      : FFT window size in samples.
    hop_length : hop between STFT frames in samples.

    Returns
    -------
    AudioBatch with the same shape, lengths and sample rate.
    """
    wav = wav_batch.data.squeeze(1)   # (B, T_max)
    T_max = wav.shape[-1]

    window = torch.hann_window(n_fft, device=wav.device, dtype=wav.dtype)

    # Complex STFT: (B, F, T_frames)
    stft = torch.stft(wav, n_fft=n_fft, hop_length=hop_length, window=window,
                      return_complex=True, center=True, pad_mode='reflect')
    T_frames = stft.shape[-1]

    # Valid-frame mask: frame t is centred at sample t * hop_length
    frame_centers = torch.arange(T_frames, device=wav.device) * hop_length   # (T_frames,)
    frame_valid = frame_centers.unsqueeze(0) < wav_batch.lengths.unsqueeze(1) # (B, T_frames)

    # Log-magnitude: (B, F, T_frames)
    log_mag = stft.abs().clamp(min=1e-10).log()

    # Per-frequency temporal mean over valid frames: (B, F)
    n_valid = frame_valid.float().sum(-1).clamp(min=1)              # (B,)
    band_mean = (log_mag * frame_valid.float().unsqueeze(1)).sum(-1) / n_valid.unsqueeze(1)

    # Subtract mean, reconstruct with original phase
    log_mag_norm = log_mag - band_mean.unsqueeze(-1)                # (B, F, T_frames)
    stft_norm = torch.polar(log_mag_norm.exp(), stft.angle())

    wav_norm = torch.istft(stft_norm, n_fft=n_fft, hop_length=hop_length,
                           window=window, length=T_max, center=True)  # (B, T_max)

    return AudioBatch(
        data=wav_norm.unsqueeze(1),
        lengths=wav_batch.lengths,
        sample_rate=wav_batch.sample_rate,
    )


def threshold_spectrum(z_batch: AudioBatch, threshold: float) -> AudioBatch:
    """Zero out encoded feature values below a threshold (spectral hard thresholding).

    Simulates denoising by suppressing low-energy bins, making the model robust
    to varying noise floors in the feature domain.  Intended as a training-time
    augmentation on log-mel or log-linear spectrogram features.

    Parameters
    ----------
    z_batch   : AudioBatch (B, C, T) encoded feature batch.
    threshold : feature values strictly below this are set to zero.

    Returns
    -------
    AudioBatch with the same shape, dtype and device as input.
    """
    return AudioBatch(
        data=z_batch.data * (z_batch.data >= threshold),
        lengths=z_batch.lengths,
        sample_rate=z_batch.sample_rate,
    )


@torch.no_grad()
def random_spectral_coloring(
    wav_batch: AudioBatch,
    n_fft: int = 1024,
    hop_length: int = 320,
    tilt_range_db: tuple[float, float] = (-10.0, 4.0),
    shelf_gain_range_db: tuple[float, float] = (-12.0, 8.0),
    shelf_fc_range_hz: tuple[float, float] = (80.0, 600.0),
    shelf_width_octaves: float = 1 / 3,
    smooth_n_basis: int = 6,
    smooth_sigma0_db: float = 7.0,
    smooth_decay: float = 0.4,
    telephone_prob: float = 0.1,
) -> AudioBatch:
    """Apply a random but physically-motivated spectral coloring to each utterance.

    Simulates the frequency response of diverse recording conditions by compositing
    three independent components in the log-magnitude STFT domain:

        H[f]  =  H_tilt[f]  +  H_shelf[f]  +  H_smooth[f]   (dB)

    1. **Spectral tilt** — the dominant real-world mode, a linear slope in
       log-frequency that models distance effects, air absorption, bright/dark
       microphones and high-pass filtering:

           H_tilt[f] = α · log₂(f / 1 kHz),   α ~ U(tilt_range_db)  [dB/oct]

       The asymmetric default (−10, +4) dB/oct reflects that high-frequency
       roll-off is far more common than boost in uncalibrated recordings.

    2. **Low-frequency shelf** — models proximity effect, bass boost/cut and
       HP filtering below a random corner frequency f_c:

           H_shelf[f] = g · σ((log f_c − log f) / w),
           g ~ U(shelf_gain_range_db),   f_c ~ LogU(shelf_fc_range_hz)

       The sigmoid transition creates a smooth shelf; w = shelf_width_octaves · ln2
       controls how sharply the gain rolls off around f_c.

    3. **Smooth DCT residual** — broad room resonances and microphone irregularities
       represented as a random combination of low-order cosine basis functions over
       the frequency axis (effectively a smooth random EQ curve):

           H_smooth[f] = Σ_{k=1}^{K} a_k · cos(πk · f/F),
           a_k ~ N(0, (σ₀ · e^{−decay·(k−1)})²)

       The exponential decay enforces smoothness; only the first few components
       (k = 1: tilt-like, k = 2: broad arch, …) carry significant variance.

    Optionally, with probability ``telephone_prob``, a hard bandpass mask is applied
    outside [300, 3400] Hz to simulate telephone/VoIP recording conditions.

    The total gain is clamped to ±40 dB before application to prevent extreme
    boosts at DC or near-Nyquist where the tilt formula diverges.  The gain is
    applied multiplicatively to the complex STFT so the original phase is preserved
    (no reconstruction artefacts beyond the intended coloring).

    Parameters
    ----------
    wav_batch           : AudioBatch (B, 1, T_samples).
    n_fft               : STFT window size in samples.
    hop_length          : STFT hop size in samples.
    tilt_range_db       : (min, max) per-octave tilt coefficient α in dB/oct.
    shelf_gain_range_db : (min, max) gain at DC for the low-frequency shelf in dB.
    shelf_fc_range_hz   : (min, max) corner frequency of the shelf in Hz, sampled
                          log-uniformly so that low and high corner freqs are
                          equally probable in octave terms.
    shelf_width_octaves : Width of the shelf transition in octaves.
    smooth_n_basis      : Number K of DCT cosine basis functions.
    smooth_sigma0_db    : Standard deviation of the lowest-order coefficient (dB).
    smooth_decay        : Exponential decay rate for σ_k = σ₀ · exp(−decay·(k−1)).
    telephone_prob      : Probability of applying a hard [300, 3400] Hz bandpass
                          per utterance in the batch.

    Returns
    -------
    AudioBatch with the same shape, lengths and sample_rate as input.
    """
    B = wav_batch.data.shape[0]
    sr = wav_batch.sample_rate
    wav = wav_batch.data.squeeze(1)   # (B, T_max)
    T_max = wav.shape[-1]
    device = wav.device
    dtype = wav.dtype

    window = torch.hann_window(n_fft, device=device, dtype=dtype)

    # Complex STFT: (B, F, T_frames)
    stft = torch.stft(wav, n_fft=n_fft, hop_length=hop_length, window=window,
                      return_complex=True, center=True, pad_mode="reflect")
    F = stft.shape[1]   # n_fft // 2 + 1

    # Frequency axis in Hz: (F,).  Clamp DC bin to the first non-zero frequency
    # so that log-frequency formulas stay finite.
    f_hz = torch.linspace(0, sr / 2, F, device=device, dtype=dtype)
    f_hz_safe = f_hz.clamp(min=float(sr) / n_fft)

    # ------------------------------------------------------------------
    # 1. Spectral tilt: α · log₂(f / 1 kHz)  [dB]
    # ------------------------------------------------------------------
    alpha = torch.empty(B, device=device, dtype=dtype).uniform_(*tilt_range_db)  # (B,)
    log2_f = torch.log2(f_hz_safe / 1000.0)                                       # (F,)
    H_tilt = alpha[:, None] * log2_f[None, :]                                     # (B, F)

    # ------------------------------------------------------------------
    # 2. Low-frequency shelf: g · σ((log f_c − log f) / w)  [dB]
    # ------------------------------------------------------------------
    g = torch.empty(B, device=device, dtype=dtype).uniform_(*shelf_gain_range_db)   # (B,)
    log_fc = torch.empty(B, device=device, dtype=dtype).uniform_(
        math.log(shelf_fc_range_hz[0]),
        math.log(shelf_fc_range_hz[1]),
    )                                                                                 # (B,)
    w = shelf_width_octaves * math.log(2.0)
    shelf_arg = (log_fc[:, None] - f_hz_safe[None, :].log()) / w                    # (B, F)
    H_shelf = g[:, None] * torch.sigmoid(shelf_arg)                                  # (B, F)

    # ------------------------------------------------------------------
    # 3. Smooth DCT residual: Σ a_k · cos(πk · f_norm)  [dB]
    # ------------------------------------------------------------------
    k = torch.arange(1, smooth_n_basis + 1, device=device, dtype=dtype)              # (K,)
    sigma_k = smooth_sigma0_db * torch.exp(-smooth_decay * (k - 1))                  # (K,)
    a_k = torch.randn(B, smooth_n_basis, device=device, dtype=dtype) * sigma_k       # (B, K)
    f_norm = torch.linspace(0.0, 1.0, F, device=device, dtype=dtype)                 # (F,)
    basis = torch.cos(math.pi * k[:, None] * f_norm[None, :])                        # (K, F)
    H_smooth = a_k @ basis                                                            # (B, F)

    # ------------------------------------------------------------------
    # Total coloring → linear gain.  Clamp ±40 dB to prevent extreme
    # boosts at DC where the tilt formula diverges.
    # ------------------------------------------------------------------
    H_db = (H_tilt + H_shelf + H_smooth).clamp(-40.0, 40.0)  # (B, F)
    gain = 10.0 ** (H_db / 20.0)                              # (B, F) amplitude gain

    # ------------------------------------------------------------------
    # 4. Optional telephone bandlimiting: zero gain outside [300, 3400] Hz
    # ------------------------------------------------------------------
    if telephone_prob > 0.0:
        phone_mask = torch.rand(B, device=device, dtype=dtype) < telephone_prob  # (B,)
        if phone_mask.any():
            band = (f_hz >= 300.0) & (f_hz <= 3400.0)                           # (F,)
            # suppression[b, f] = 0 when sample b is telephone and bin f is out-of-band,
            # 1 otherwise.  Written as 1 − phone_mask · ¬band to avoid indexing copies.
            suppression = (
                1.0
                - phone_mask[:, None].to(dtype) * (~band)[None, :].to(dtype)
            )                                                                    # (B, F)
            gain = gain * suppression

    # ------------------------------------------------------------------
    # Apply gain to STFT (phase preserved) and reconstruct waveform
    # ------------------------------------------------------------------
    stft_colored = stft * gain[:, :, None]           # (B, F, T_frames)

    wav_colored = torch.istft(
        stft_colored, n_fft=n_fft, hop_length=hop_length,
        window=window, length=T_max, center=True,
    )                                                # (B, T_max)

    return AudioBatch(
        data=wav_colored.unsqueeze(1),
        lengths=wav_batch.lengths,
        sample_rate=wav_batch.sample_rate,
    )

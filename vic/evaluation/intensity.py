"""P_φ as a ``speech_eval`` metric: the converter's own ruler, applied to its own output.

Every other metric in the suite answers "what did the conversion do to the voice".  This
one answers "did it reach the level it was asked for", which is the question the model was
trained on — and for converted audio there is no ground truth to answer it with, because
the utterance never existed.  P_φ is the instrument that stands in: a frozen predictor
regressing sequence-level dB SPL from whitened latents, and the same one whose reading
``val/rmse_pred_db`` reports during training.  Measuring the rendered audio with it makes
an evaluation number and a training curve comparable.

What it emits
-------------
``leq_db`` — one scalar per utterance, the ``leq_aggregate`` of P_φ's per-frame curve over
the utterance's valid frames, in dB SPL at 1 m.  Not dBFS: the ``level`` metric already
reports the digital amplitude, and the two are different quantities that happen to share a
unit name.  Paired against the real recording at the same target level, its ``.delta`` is
the conversion's intensity error in dB — the headline number.

Why it is constructed rather than named in a config
---------------------------------------------------
Every other metric is built by ``speech_eval.core.build_metrics`` from a ``{"type": ...}``
spec, which is possible because they own their models.  This one borrows the *already
loaded* P_φ and whitening pipeline out of the converter module, so that the reading is
made by the exact checkpoint the run was evaluated against and not by a second copy
configured separately.  Naming it in a config would mean giving it its own checkpoint path,
which is precisely the drift worth avoiding.

``speech_eval`` never imports ``vic``; the dependency runs the other way, so a VIC-specific
metric subclassing ``Metric`` is the sanctioned direction.
"""
from __future__ import annotations

import torch

from speech_eval.core import Features, Metric, Requirements, UtteranceBatch


class IntensityPredictorMetric(Metric):
    """Sequence-level dB SPL from the converter run's own frozen P_φ.

    Parameters
    ----------
    module      : the built :class:`ConverterCGANv2LabelsModule`, already on its device.
                  Only ``estimate_tau_src`` is used, which whitens, re-encodes and
                  aggregates exactly as training did.
    sample_rate : the rate P_φ expects — the codec's, since that is what the converter
                  produced and what the predictor was fitted on.
    name        : column prefix; ``intensity.leq_db`` by default.

    ``level_norm_dbfs=None`` is load-bearing.  P_φ reads a *level*, so normalising its
    input would erase the very quantity it reports.  The whitened front end happens to be
    gain-invariant, which makes the mistake silent rather than obvious — the numbers would
    look entirely reasonable and mean nothing about the converted amplitude.
    """

    metric_type = "vic_intensity"

    def __init__(self, module, sample_rate: int = 16_000, name: str = "intensity"):
        self.requirements = Requirements(
            sample_rate=sample_rate, mono=True, level_norm_dbfs=None
        )
        super().__init__(name)
        self._module = module
        self._device = torch.device("cpu")

    def load(self, device: torch.device) -> None:
        """The module is already built and placed; only its device is recorded here."""
        self._device = device
        self._module.eval()
        self._loaded = True

    @torch.no_grad()
    def compute(self, batch: UtteranceBatch) -> list[Features]:
        # The padded view, because estimate_tau_src aggregates over the padding mask —
        # feeding it unpadded signals one at a time would give the same answer far more
        # slowly, and feeding it padding without a mask would average silence into the Leq.
        audio = batch.audio.to(self._device)
        leq = self._module.estimate_tau_src(audio)
        return [{"leq_db": float(value)} for value in leq]

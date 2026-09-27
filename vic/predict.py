"""Inference API for the trained vocal intensity predictor P_φ.

Quick start
-----------
    from vic.predict import VocalIntensityPredictor
    import torchaudio

    # From a training checkpoint + YAML (internal use)
    pred = VocalIntensityPredictor.from_checkpoint(
        "runs/.../predictor-best.ckpt",
        "configs/paper/train_predictor.yaml",
    )

    # From a self-contained bundle / HF Hub (collaborator use)
    pred = VocalIntensityPredictor.from_pretrained("your-org/vic-predictor")

    wav, sr = torchaudio.load("speech.wav")     # any sample rate, any channels
    result = pred.predict(wav, sr)

    result.leq_db      # (1,) sequence-level equivalent level in dBSPL
    result.frame_db    # (1, T) per-frame dBSPL, one value every 20 ms
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch import Tensor

from vic.checkpoints import load_submodule
from vic.data.audio_batch import AudioBatch
from vic.data.transforms import (
    NORMALIZE_SEQUENCE_PREDICTOR,
    normalize_sequence_from_config,
    prepare_waveform,
)
from vic.features.spl import leq_aggregate
from vic.models.predictor import build_predictor
from vic.training.extraction_pipeline import ExtractionPipeline, build_pipeline


# ---------------------------------------------------------------------------
# Output type
# ---------------------------------------------------------------------------

@dataclass
class PredictionResult:
    """Output of :meth:`VocalIntensityPredictor.predict`.

    Attributes
    ----------
    frame_db     : ``(B, T)`` per-frame intensity in dBSPL, one value every
                   ``frame_duration_ms`` ms (typically 20 ms).
    leq_db       : ``(B,)`` LeqZF sequence-level equivalent level in dBSPL.
    padding_mask : ``(B, T)`` boolean mask, ``True`` where the frame is valid
                   (not right-padding).  Index into ``frame_db`` with this to
                   drop padding values: ``frame_db[i][padding_mask[i]]``.

    All tensors are on CPU regardless of which device was used for inference.
    """

    frame_db: Tensor
    leq_db: Tensor
    padding_mask: Tensor


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class VocalIntensityPredictor:
    """Inference-only wrapper for a trained intensity predictor P_φ.

    Handles the same preprocessing pipeline as training:

        resample → mono-mix → peak-normalize(0.9) → pad-to-multiple(hop_size)

    then runs the frozen :class:`~vic.training.extraction_pipeline.ExtractionPipeline`
    and the predictor head.  The result is identical to what the predictor sees
    during training (without any stochastic augmentations).
    """

    def __init__(
        self,
        pipeline: ExtractionPipeline,
        predictor: nn.Module,
        device: torch.device,
        normalize_sequence: bool = NORMALIZE_SEQUENCE_PREDICTOR,
    ) -> None:
        self.pipeline = pipeline.to(device).eval()
        self.predictor = predictor.to(device).eval()
        self.device = device
        self.normalize_sequence = normalize_sequence

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_checkpoint(
        cls,
        ckpt_path: str | Path,
        config: dict | str | Path,
        device: str | torch.device | None = None,
        ckpt_prefix: str = "predictor",
    ) -> VocalIntensityPredictor:
        """Load from a PredictorModule ``.ckpt`` file and its training YAML.

        Mirrors ``train_predictor.py`` exactly: ``build_pipeline`` →
        ``build_predictor`` → ``load_submodule``.

        Parameters
        ----------
        ckpt_path :
            Path to the ``PredictorModule`` Lightning checkpoint.
        config :
            Path to the training YAML, or an already-loaded config dict.
        device :
            Inference device.  Defaults to CUDA if available, else CPU.
        ckpt_prefix :
            Attribute name the predictor was stored under.  Only needs changing
            for a checkpoint written by some other LightningModule.
        """
        if isinstance(config, (str, Path)):
            config = yaml.safe_load(Path(config).read_text())
        device = _resolve_device(device)

        pipeline = build_pipeline(config)
        predictor = build_predictor(config["model"], pipeline.latent_dim, dropout_override=0.0)
        normalize = normalize_sequence_from_config(config, NORMALIZE_SEQUENCE_PREDICTOR)
        load_submodule(predictor, ckpt_path, prefix=ckpt_prefix)
        return cls(pipeline, predictor, device, normalize_sequence=normalize)

    @classmethod
    def from_pretrained(
        cls,
        path_or_repo_id: str | Path,
        device: str | torch.device | None = None,
    ) -> VocalIntensityPredictor:
        """Load from a self-contained model bundle or a HF Hub repo.

        The bundle is produced by :meth:`save_pretrained` and must contain::

            config.yaml   — extractor + model architecture params
            predictor.pt  — predictor head state dict

        If ``path_or_repo_id`` is not a local directory it is treated as a
        Hugging Face Hub repo ID and downloaded with ``snapshot_download``
        (requires ``pip install huggingface_hub`` and a valid login token).
        """
        path = Path(path_or_repo_id)
        if not path.is_dir():
            from huggingface_hub import snapshot_download
            path = Path(snapshot_download(repo_id=str(path_or_repo_id)))

        config = yaml.safe_load((path / "config.yaml").read_text())

        # Resolve extractor file paths stored relative to the bundle directory.
        ec = config.get("extractor", {})
        for key in ("config_path", "ckpt_path"):
            if key in ec and not Path(ec[key]).is_absolute():
                ec[key] = str(path / ec[key])

        device = _resolve_device(device)
        pipeline = build_pipeline(config)
        predictor = build_predictor(config["model"], pipeline.latent_dim, dropout_override=0.0)
        normalize = normalize_sequence_from_config(config, NORMALIZE_SEQUENCE_PREDICTOR)

        state = torch.load(path / "predictor.pt", map_location="cpu", weights_only=True)
        predictor.load_state_dict(state)
        return cls(pipeline, predictor, device, normalize_sequence=normalize)

    def save_pretrained(self, path: str | Path, config: dict) -> None:
        """Save a self-contained model bundle for :meth:`from_pretrained`.

        Parameters
        ----------
        path :
            Output directory (created if necessary).
        config :
            The same config dict used to build this predictor (from the
            training YAML).  Saved verbatim as ``config.yaml``.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.predictor.state_dict(), path / "predictor.pt")
        (path / "config.yaml").write_text(yaml.dump(config, allow_unicode=True))

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(self, wav: Tensor, sample_rate: int) -> PredictionResult:
        """Predict vocal intensity from one or more audio waveforms.

        Preprocessing is identical to training — ``prepare_waveform`` is the same
        chain ``AudioDataset`` runs — and whether it ends in a peak normalisation
        follows ``self.normalize_sequence``, read from the training config rather
        than assumed.  No stochastic augmentation is applied.

        Parameters
        ----------
        wav :
            Float32 tensor of shape ``(B, C, T)``, ``(C, T)``, or ``(T,)``.
        sample_rate :
            Sample rate of the input waveform in Hz.

        Returns
        -------
        :class:`PredictionResult` — all tensors on CPU.
        """
        # Normalise to (B, C, T)
        if wav.dim() == 1:
            wav = wav.unsqueeze(0).unsqueeze(0)
        elif wav.dim() == 2:
            wav = wav.unsqueeze(0)

        target_sr = self.pipeline.sample_rate
        hop_size = self.pipeline.frame_grid.hop_size

        processed: list[Tensor] = [
            prepare_waveform(
                wav[i], sample_rate, target_sr, hop_size, self.normalize_sequence
            )
            for i in range(wav.shape[0])
        ]

        batch = AudioBatch.from_list(processed, sample_rate=target_sr).to(self.device)
        z = self.pipeline.encode(batch, training=False)

        frame_db: Tensor = self.predictor(z)            # (B, T)
        mask: Tensor = z.padding_mask                   # (B, T) bool
        leq_db: Tensor = leq_aggregate(frame_db, mask)  # (B,)

        return PredictionResult(
            frame_db=frame_db.cpu(),
            leq_db=leq_db.cpu(),
            padding_mask=mask.cpu(),
        )

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def sample_rate(self) -> int:
        """Native sample rate expected by the extractor."""
        return self.pipeline.sample_rate

    @property
    def hop_size(self) -> int:
        """Codec/extractor hop size in samples."""
        return self.pipeline.frame_grid.hop_size

    @property
    def frame_duration_ms(self) -> float:
        """Duration of one output frame in milliseconds (typically 20 ms)."""
        return 1000.0 * self.hop_size / self.sample_rate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_device(device: str | torch.device | None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)

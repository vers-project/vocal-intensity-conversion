"""Intensity conversion inference script.

Converts one or more audio files to a target vocal intensity using a trained
converter checkpoint.  ``--config`` is the converter's own training YAML: the
extractor, converter architecture and label scaler are all read from it, so the
inference path is built the same way ``train_converter_cgan_v2_labels.py`` builds it.

``--target-db`` is τ in **dB SPL at 1 m**, the scale P_φ regresses — not dBFS.

Usage
-----
Single file:
    uv run scripts/convert.py \\
        --checkpoint checkpoints/converter/converter-best.ckpt \\
        --config     configs/converter.yaml \\
        --input      audio/speech.wav \\
        --target-db  70.0 \\
        --output     audio/speech_converted.wav

Batch (glob):
    uv run scripts/convert.py \\
        --checkpoint checkpoints/converter/converter-best.ckpt \\
        --config     configs/converter.yaml \\
        --input      "audio/*.wav" \\
        --target-db  70.0 \\
        --output-dir audio/converted/

Output
------
If --output is given, a single converted file is saved there.
If --output-dir is given, converted files are saved under that directory
with the same filenames as the inputs.
Both options produce a 16-bit PCM WAV at the codec sample rate.
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import torch
import torchaudio
import yaml

from vic.converter_bundle import load_converter_from_ckpt
from vic.core import AudioCodec
from vic.data.audio_batch import AudioBatch
from vic.data.transforms import (
    NORMALIZE_SEQUENCE_CONVERTER,
    load_audio,
    normalize_sequence_from_config,
    prepare_waveform,
)
from vic.training.extraction_pipeline import build_extractor


# ---------------------------------------------------------------------------
# Audio I/O helpers
# ---------------------------------------------------------------------------

def preprocess(
    path: str, target_sr: int, hop_size: int, normalize_sequence: bool
) -> AudioBatch:
    """Load and preprocess a single file into a batch-of-one AudioBatch.

    ``normalize_sequence`` follows the training config and normally resolves to
    False here: the conversion latents are raw NAC, whose job is to carry
    amplitude alongside spectral pattern, so peak-normalising the input would
    strip exactly what the converter reads.  (This function normalised
    unconditionally before, which silently contradicted every converter config.)
    """
    wav, sr = load_audio(path)
    wav = prepare_waveform(wav, sr, target_sr, hop_size, normalize_sequence)
    return AudioBatch.from_list([wav], sample_rate=target_sr)


def save_audio(batch: AudioBatch, path: str | Path, original_length: int) -> None:
    """Save the first item in a batch as a 16-bit PCM WAV, trimmed to original_length."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wav = batch.unbatch()[0]                     # (1, T)
    wav = wav[:, :original_length]
    torchaudio.save(str(path), wav.cpu(), batch.sample_rate, bits_per_sample=16)


# ---------------------------------------------------------------------------
# Core conversion
# ---------------------------------------------------------------------------

@torch.no_grad()
def convert_file(
    input_path: str,
    output_path: str | Path,
    codec: AudioCodec,
    converter: torch.nn.Module,
    target_db: float,
    device: torch.device,
    normalize_sequence: bool,
) -> None:
    wav_batch = preprocess(
        input_path, codec.sample_rate, codec.frame_grid.hop_size, normalize_sequence
    ).to(device)
    original_length = int(wav_batch.lengths[0].item())

    z_real = codec.encode(wav_batch)

    tau = torch.tensor([target_db], device=device)
    z_fake = converter(z_real, tau)

    wav_out = codec.decode(z_fake)
    save_audio(wav_out, output_path, original_length)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def resolve_inputs(input_pattern: str) -> list[str]:
    paths = sorted(glob.glob(input_pattern, recursive=True))
    if not paths:
        raise FileNotFoundError(f"No files matched: {input_pattern!r}")
    return paths


def main():
    parser = argparse.ArgumentParser(description="Vocal intensity conversion inference.")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to a converter training .ckpt file.")
    parser.add_argument("--config", required=True,
                        help="Path to converter YAML config (same one used for training).")
    parser.add_argument("--input", required=True,
                        help="Input audio file or glob pattern (quote globs).")
    parser.add_argument("--target-db", type=float, required=True,
                        help="Target vocal intensity τ in dB SPL at 1 m (e.g. 70.0). "
                             "Not dBFS: this is the scale P_φ regresses and the "
                             "converter was conditioned on, so stay inside the "
                             "training.intensity_range_db of the config.")
    parser.add_argument("--output", default=None,
                        help="Output file path (single-file mode).")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (batch mode).")
    parser.add_argument("--device", default=None,
                        help="Compute device (default: cuda if available, else cpu).")
    args = parser.parse_args()

    if args.output is None and args.output_dir is None:
        parser.error("Provide --output (single file) or --output-dir (batch).")

    cfg = yaml.safe_load(Path(args.config).read_text())
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")

    # The extractor block the converter was trained with — the same key path
    # train_converter_cgan_v2_labels.py reads.  Conversion needs a codec, not merely a
    # feature extractor, because the output has to be decoded back to audio.
    codec = build_extractor(cfg)
    if not hasattr(codec, "decode"):
        parser.error(
            f"extractor.type={cfg['extractor']['type']!r} builds a "
            f"{type(codec).__name__}, which has no decode(); conversion needs a codec."
        )
    codec = codec.to(device).eval()

    converter = load_converter_from_ckpt(args.checkpoint, cfg).to(device)
    if converter.output_proj.out_features != codec.latent_dim:
        parser.error(
            f"The converter expects {converter.output_proj.out_features}-dim latents "
            f"but {type(codec).__name__} produces {codec.latent_dim}."
        )
    normalize = normalize_sequence_from_config(cfg, NORMALIZE_SEQUENCE_CONVERTER)

    input_paths = resolve_inputs(args.input)

    if args.output is not None:
        if len(input_paths) > 1:
            parser.error(f"--output expects a single file but {len(input_paths)} matched.")
        print(f"{input_paths[0]} → {args.output}")
        convert_file(input_paths[0], args.output, codec, converter,
                     args.target_db, device, normalize)
    else:
        out_dir = Path(args.output_dir)
        for src in input_paths:
            dst = out_dir / Path(src).name
            print(f"{src} → {dst}")
            convert_file(src, dst, codec, converter, args.target_db, device, normalize)

    print("Done.")


if __name__ == "__main__":
    main()

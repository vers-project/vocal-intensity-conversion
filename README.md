# Vocal intensity conversion with continuous SPL control via adversarial training

Quentin Le Tellier, Albert Rilliard, Olivier Perrotin, Marc Evrard
Submitted to ICASSP 2027. *Paper link to come.*

**[Audio samples](https://vers-project.github.io/vocal-intensity-conversion/)** ·
**[Model checkpoints](#model-checkpoints)**

## Overview

Vocal intensity conversion renders a recorded utterance at a different vocal intensity,
rather than simply scaling its amplitude, while preserving its content and speaker identity.
This repository converts speech to any target sound pressure level (SPL), given in dB on a
continuous scale. A lightweight converter transforms the features of a frozen speech encoder,
and a pretrained vocoder resynthesizes the speech. The converter is trained as a conditional
cycle-consistent GAN, without parallel data. The same method is applied to three
encoder/decoder pairs: log-mel spectrograms with BigVGAN-v2, WavLM with the kNN-VC HiFi-GAN,
and SpeechTokenizer.

The repository also contains the objective evaluation protocol of the paper. It measures the
achieved change in intensity against the target change on a calibrated corpus.

## Installation

Python 3.12 and [uv](https://docs.astral.sh/uv/) are required.

```bash
git clone https://github.com/vers-project/vocal-intensity-conversion.git
cd vocal-intensity-conversion
uv sync --extra cpu            # or --extra cu128 for CUDA 12.8
```

## Model checkpoints

| model | description | link |
|---|---|---|
| Intensity predictor P_φ | wav2vec 2.0 layer 2 + Transformer head | *to come* |
| Converters C_θ | one per encoder/decoder pair: WavLM (available), log-mel and SpeechTokenizer (to come) | [Hugging Face](https://huggingface.co/vers-project/vocal-intensity-conversion) |

## Converting speech

```bash
# on a CUDA machine, replace --extra cpu with --extra cu128
uv run --extra cpu --extra hub scripts/convert.py \
    --model     vers-project/vocal-intensity-conversion \
    --subfolder converter-wavlm \
    --input     speech.wav \
    --target-db 70.0 \
    --output    speech_70dB.wav
```

`--target-db` is the target intensity in dB SPL at 1 m. The first run downloads the
converter and, for the WavLM converter, WavLM-Large and the kNN-VC HiFi-GAN (about
1.3 GB) into `~/.cache/vic`. To convert with your own training run instead, pass
`--checkpoint /path/to/converter.ckpt --config /path/to/config.yaml` in place of
`--model` and `--subfolder`.

## Reproducing the paper

The evaluation needs extra metric back-ends, and training and evaluation with the
log-mel and WavLM pairs need their vocoders on disk:

```bash
uv sync --extra cpu --extra phonetics --extra asr --extra speaker
uv run --extra cpu --extra vocoder scripts/download_vocoders.py --dest /path/to/vocoders
```

Every experiment is launched from a configuration in `configs/paper/`. Paths starting with
`/path/to/` are placeholders for your own copies of the data, weights and outputs.

```bash
launch_experiment --config configs/paper/<config>.yaml --script scripts/<script>.py
```

| step | configuration | script |
|---|---|---|
| AVID metadata and splits | `prepare_avid_metadata`, `merge_annotations`, `compute_labels`, `make_prediction_metadata` | same names |
| Intensity predictor P_φ | `train_predictor` | `train_predictor.py` |
| Converters (three encoder/decoder pairs) | `train_converter` | `train_converter_cgan_v2_labels.py` |
| Objective evaluation (Table 2) | `evaluate_conversion_{wavlm,mel,speechtokenizer}` | `evaluate_conversion.py` |
| Slopes and preservation measures | `analyze_conversion` | `analyze_conversion.py` |
| Perceptual evaluation stimuli | `convert_selection` | `convert_selection.py` |
| Perceptual evaluation analysis | `analyze_perceptual` | `analyze_perceptual.py`, `perceptual/` |

## Data

The converters are trained and evaluated on
[AVID](https://doi.org/10.5281/zenodo.8331897), the Aalto Vocal Intensity Database
(Alku et al., *Speech Communication*, 2024), distributed under CC BY 4.0.

## Repository layout

```
vic/            library: encoders and vocoders, models, training, evaluation
scripts/        one script per experiment step
configs/paper/  configurations of the paper's experiments
perceptual/     analysis of the perceptual evaluation
tests/          unit tests (uv run --with pytest pytest tests)
```

The shared utilities live in their own repositories and are installed automatically:
[audio_utils](https://github.com/vers-project/audio_utils),
[speech-eval](https://github.com/vers-project/speech-eval),
[slurm_launcher](https://github.com/vers-project/slurm_launcher) and
[speech_tokenizer](https://github.com/vers-project/speech_tokenizer) (a modified
[SpeechTokenizer](https://github.com/ZhangXInFD/SpeechTokenizer)).

## Citation

```bibtex
@misc{letellier2027vocal,
  title  = {Vocal Intensity Conversion with Continuous {SPL} Control via Adversarial Training},
  author = {Le Tellier, Quentin and Rilliard, Albert and Perrotin, Olivier and Evrard, Marc},
  year   = {2026},
  note   = {Submitted to ICASSP 2027}
}
```

## Acknowledgements

This work was partly funded by the French National Research Agency (ANR) through the VERS
project (ANR-23-CE38-0010-01). It was granted access to the HPC resources of IDRIS under the
allocation 2026-AD011016051R1 made by GENCI.

## Licence

MIT. See [`LICENSE`](LICENSE).

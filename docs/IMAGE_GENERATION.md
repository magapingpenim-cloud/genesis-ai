# GENESIS Image Generation — First Working Track

## What this is

The first text-conditioned image diffusion research pipeline. It includes a small U-Net denoiser, a jointly learned UTF-8 byte embedding for captions, a DDPM noise schedule, a JSONL image-caption dataset, a training loop, and a local sampler.

It is **not pretrained** and will produce noise or poor images until trained on a sufficiently large, diverse, correctly licensed image-caption dataset. The tiny byte-level text encoder is a baseline, not a strong semantic language model. Start with 64px images and a small experiment to validate the pipeline.

## Dataset

Create a folder containing images and a JSONL manifest. Each line must contain:
- `image`: path relative to the manifest (or absolute path)
- `caption`: descriptive text

Example:
```json
{"image":"images/000001.jpg","caption":"A red bicycle leaning against a white wall"}
```

Only use images you own or datasets whose license permits the intended training and use. Keep image files outside Git.

## Install

Install a PyTorch build suitable for your hardware from https://pytorch.org/get-started/locally/ then:

```bash
pip install -e ".[dev]"
```

## Train from scratch

```bash
python -m genesis.training.train_image --manifest /path/to/manifest.jsonl --image-size 64 --batch-size 4 --steps 10000
```

For low-memory hardware, try `--image-size 32 --batch-size 1 --base-channels 32`.

## Generate

```bash
python -m genesis.inference.generate_image --checkpoint checkpoints/genesis-image/step-00010000.pt --prompt "A cinematic photograph of a red sports car in Tokyo at night" --size 64 --steps 100
```

The checkpoint's model weights are trained from scratch by this project. The generator runs locally. The current model has no pretrained weights or external inference API.

## Current limitations

- The model works in pixel space and is intentionally small.
- Text encoding is byte-level and learned jointly; prompt understanding will be weak.
- 64px/128px experiments are a starting point, not production quality.
- Better results require more data, training time, model capacity, resolution, and evaluation.
- A high-quality production text-to-image model is a substantial research and compute project.

# GENESIS-LM 0.1 — local training prototype

This folder contains the first trainable, decoder-only Transformer in GENESIS. It predicts the next UTF-8 byte (256-token vocabulary). It is **randomly initialized** and is not an intelligent assistant until trained on a large, clean, licensed corpus. Byte-level tokenization is intentionally simple but inefficient compared with a trained BPE tokenizer.

## Install

Python 3.10+ and PyTorch are required. Install a PyTorch build appropriate for your operating system/GPU from the official PyTorch selector, then:

```bash
python -m pip install torch
```

## Prepare data

Put UTF-8 plain-text files you have permission to use into `data/corpus/`. Keep a separate held-out validation set in mind; this starter currently makes a simple 90/10 split. Do not use private messages, passwords, personal records, or copyrighted datasets without permission.

## Train

From the repository root:

```bash
python genesis_lm/train.py --data data/corpus --steps 5000
```

Start small on CPU if necessary, for example `--width 96 --layers 3 --heads 3 --batch-size 8 --block-size 128`. On a supported NVIDIA GPU, PyTorch will use CUDA automatically. Training time and feasible model size depend on the hardware and corpus.

## Sample

```bash
python genesis_lm/sample.py --checkpoint checkpoints/genesis-lm.pt --prompt "Привет!"
```

## What success means

- The training loss decreases on training data.
- Validation loss also decreases; a widening train/validation gap signals overfitting.
- Samples become more coherent on held-out prompts, not just memorized training passages.
- Save checkpoints and compare experiments. A low loss on a tiny corpus does not mean broad reasoning ability.

## Current limitations

This is a small research baseline, not ChatGPT, not instruction-tuned, not retrieval-grounded, and not connected to the chat server. It will initially generate poor text. Do not present it as a finished assistant. Next work: tokenizer/BPE, robust dataset curation and deduplication, larger context, instruction/conversation fine-tuning, evaluation suite, then carefully integrate inference into the chat backend.

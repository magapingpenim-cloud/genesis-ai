# GENESIS language model — first milestone

The repository now contains a compact decoder-only Transformer implemented with PyTorch. Its weights start randomly initialized: it must be trained on text before it can produce useful continuations.

## Initial hardware profile
- Vocabulary: 256 UTF-8 byte values
- Transformer blocks: 4
- Width: 256
- Attention heads: 4
- Context length: 128
- Default batch size: 4
- CUDA mixed precision is enabled on NVIDIA GPUs

This is a deliberately small baseline for an RTX 3050 Laptop GPU with 4 GB VRAM.

## Prepare data
Create data/corpus.txt locally as UTF-8 text. Use material you have permission to train on. A small corpus is sufficient only for a smoke test; meaningful language ability requires a much larger, varied, high-quality dataset and extensive training.

Training data and checkpoints are excluded from Git because they can be large and may contain copyrighted or private material.

## Train from the repository root (PowerShell)
    .\.venv\Scripts\python.exe -m genesis.train --data data/corpus.txt --steps 2000

## Generate after training
    .\.venv\Scripts\python.exe -m genesis.generate --prompt "Привет, GENESIS" --tokens 100

## Limitations
This is next-byte prediction, not a finished chatbot. It has no pretrained knowledge or reliable instruction following. A short training run will not make it comparable to commercial language models. Future milestones: data validation, tokenizer research, validation loss/perplexity, and dialogue training.

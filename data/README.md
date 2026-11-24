# GENESIS training data

Place UTF-8 training sources in `data/raw/`. The repository includes a small hand-written starter-set builder; it does not download or scrape data.

## Build the starter dialogues

From the repository root:

```powershell
.\.venv\Scripts\python.exe scripts/build_starter_dataset.py
```

This creates `data/raw/starter_dialogues.jsonl` with 80 short examples across basic topics. They are a smoke-test seed set, not a broad or authoritative knowledge base.

## Train

```powershell
.\.venv\Scripts\python.exe -m genesis.train_dialogue --data data/raw/starter_dialogues.jsonl --steps 500 --batch-size 2 --output checkpoints/genesis-starter.pt
```

## Evaluate manually

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_dialogue.py --checkpoint checkpoints/genesis-starter.pt
```

Use only material you wrote, public-domain material, or datasets whose licenses explicitly permit your intended model-training use. Do not include private conversations or personal data without permission. A small training set can be memorized and does not make the model a reliable assistant.

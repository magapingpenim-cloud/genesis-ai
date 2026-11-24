# GENESIS language model — character-level dialogue milestone

GENESIS is an experimental decoder-only Transformer implemented with PyTorch. It is randomly initialized and must be trained locally; this is not a pretrained assistant.

## Why character-level for the first Russian prototype?
The first version predicted raw UTF-8 bytes, which could emit invalid byte sequences and garbled Cyrillic. The current prototype builds a Unicode character vocabulary from the corpus and stores that vocabulary inside each checkpoint. This preserves Russian characters. It is a simple baseline, not a production tokenizer.

## Hardware profile
The default small configuration targets an RTX 3050 Laptop GPU with 4 GB VRAM:
- 4 Transformer blocks, width 256, 4 attention heads
- context length 128 characters
- batch size 2 by default
- CUDA mixed precision on NVIDIA GPUs

## Prepare permitted data
Place UTF-8 .txt/.md files or .jsonl dialogue records in data/raw/. A dialogue JSONL record can look like:
```json
{"messages":[{"role":"user","content":"Привет!"},{"role":"assistant","content":"Привет! Рад тебя видеть."}]}
```
Only use data you own, public-domain material, or datasets whose license permits model training. Do not include private conversations or personal data without permission.

Prepare the corpus:
```powershell
.\.venv\Scripts\python.exe scripts/prepare_corpus.py
```

## Train a new character-level checkpoint
The earlier byte-level checkpoint is incompatible with this vocabulary and must not be reused.
```powershell
.\.venv\Scripts\python.exe -m genesis.train --steps 300 --batch-size 2 --output checkpoints/genesis-char.pt
```

## Generate
```powershell
.\.venv\Scripts\python.exe -m genesis.generate --prompt "Пользователь: Привет!\nGENESIS:" --tokens 100
```

## Limitations
The current objective is next-character prediction over concatenated text, not instruction fine-tuning. The model does not yet understand questions reliably, browse the web, or have pretrained knowledge. A tiny corpus only tests that the pipeline works; useful conversational behavior requires a large, varied, carefully licensed corpus and a separate instruction/dialogue training stage. Character-level models are inefficient for long text; a subword tokenizer is a later milestone.


## Assistant-only dialogue training (new milestone)

The supervised dialogue trainer reads `data/raw/dialogues.jsonl` directly. It builds the same Unicode character vocabulary approach and masks the loss on user-message characters: the model is optimized to predict assistant-response characters only.

Run from the repository root:
```powershell
.\.venv\Scripts\python.exe -m genesis.train_dialogue --steps 300 --batch-size 2 --output checkpoints/genesis-dialogue.pt
```

Generate from a dialogue-style prompt using the matching checkpoint:
```powershell
.\.venv\Scripts\python.exe -m genesis.generate --checkpoint checkpoints/genesis-dialogue.pt --prompt "Пользователь: Привет!\nGENESIS: " --tokens 100 --temperature 0.4
```

Important:
- This creates a separate checkpoint; it does not overwrite `genesis-char.pt`.
- The vocabulary is built from the dialogue file and stored in that checkpoint. If the dialogue dataset changes, retrain and use the newly produced checkpoint.
- The current prototype uses character-level prediction and a small context window. Five examples are only a pipeline smoke test; the model can memorize them and will not generalize reliably.
- The training mask teaches the model to predict assistant text only. It does not by itself create reasoning, factual knowledge, safety behavior, or web search.

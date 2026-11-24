"""Supervised dialogue fine-tuning: calculate loss only on assistant response characters.

Input: data/raw/dialogues.jsonl with {"messages":[{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}.
"""
import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from genesis.model import GenesisLanguageModel, ModelConfig


def load_dialogues(path: Path):
    dialogues = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            print(f"Skip invalid JSON at line {line_no}: {exc}")
            continue
        messages = row.get("messages") if isinstance(row, dict) else None
        if not isinstance(messages, list):
            continue
        turns = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role", "")).lower().strip()
            text = message.get("content")
            if role in {"user", "assistant"} and isinstance(text, str) and text.strip():
                turns.append((role, text.strip()))
        if any(role == "assistant" for role, _ in turns):
            dialogues.append(turns)
    return dialogues


def encode_dialogues(dialogues):
    vocab_chars = set()
    serialized = []
    for turns in dialogues:
        parts = []
        for role, text in turns:
            prefix = "Пользователь: " if role == "user" else "GENESIS: "
            parts.append((prefix, False))
            parts.append((text, role == "assistant"))
            parts.append(("\n", role == "assistant"))
        parts.append(("\n", False))
        serialized.append(parts)
        for value, _ in parts:
            vocab_chars.update(value)
    vocab = ["<unk>"] + sorted(vocab_chars)
    mapping = {char: index for index, char in enumerate(vocab)}
    token_ids, supervised = [], []
    for parts in serialized:
        for value, is_answer in parts:
            for char in value:
                token_ids.append(mapping[char])
                supervised.append(is_answer)
    return torch.tensor(token_ids, dtype=torch.long), torch.tensor(supervised, dtype=torch.bool), vocab


def get_batch(tokens, mask, batch_size, context_length, device):
    max_start = tokens.numel() - context_length - 1
    if max_start < 0:
        raise ValueError("Dialogue dataset is shorter than context length + 1.")
    starts = torch.randint(0, max_start + 1, (batch_size,))
    xs, ys = [], []
    for start in starts.tolist():
        x = tokens[start:start + context_length]
        target = tokens[start + 1:start + context_length + 1].clone()
        target_mask = mask[start + 1:start + context_length + 1]
        target[~target_mask] = -100
        # Retry a few times if a random window contains no assistant answer characters.
        retries = 0
        while not target_mask.any() and retries < 20:
            start = int(torch.randint(0, max_start + 1, (1,)).item())
            x = tokens[start:start + context_length]
            target = tokens[start + 1:start + context_length + 1].clone()
            target_mask = mask[start + 1:start + context_length + 1]
            target[~target_mask] = -100
            retries += 1
        if not target_mask.any():
            raise ValueError("No assistant response characters found in sampled training windows.")
        xs.append(x)
        ys.append(target)
    return torch.stack(xs).to(device), torch.stack(ys).to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw/dialogues.jsonl")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--output", default="checkpoints/genesis-dialogue.pt")
    args = parser.parse_args()

    path = Path(args.data)
    if not path.exists():
        raise SystemExit(f"Dialogue file not found: {path}")
    dialogues = load_dialogues(path)
    if not dialogues:
        raise SystemExit("No valid dialogues found. Check JSONL messages with user/assistant roles.")

    tokens, answer_mask, vocab = encode_dialogues(dialogues)
    if int(answer_mask.sum()) == 0:
        raise SystemExit("No assistant answer text found.")
    if tokens.numel() < args.context_length + 1:
        raise SystemExit(
            f"Only {tokens.numel()} characters found. Add more dialogues or lower --context-length."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = ModelConfig(vocab_size=len(vocab), context_length=args.context_length)
    model = GenesisLanguageModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Dialogues: {len(dialogues)}")
    print(f"Characters: {tokens.numel():,}; assistant target characters: {int(answer_mask.sum()):,}")
    print(f"Character vocabulary: {len(vocab)}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    model.train()
    for step in range(1, args.steps + 1):
        x, targets = get_batch(tokens, answer_mask, args.batch_size, args.context_length, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits, _ = model(x)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-100,
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        if step == 1 or step % 50 == 0:
            print(f"step {step}/{args.steps} assistant_loss={loss.item():.4f}")
        if step % args.save_every == 0 or step == args.steps:
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": config.to_dict(),
                    "vocab": vocab,
                    "step": step,
                    "training_mode": "assistant_only_character_level",
                },
                output,
            )
            print(f"Saved checkpoint: {output}")
    print("Dialogue training complete. Evaluate on unseen prompts; a tiny dataset can be memorized.")


if __name__ == "__main__":
    main()

"""Run a small human-review evaluation set against a GENESIS dialogue checkpoint."""
import argparse
import json
from pathlib import Path

import torch

from genesis.model import GenesisLanguageModel, ModelConfig


QUESTIONS = [
    ("Привет! Как дела?", "Greeting should be natural and relevant."),
    ("Что такое бокс?", "Should define boxing as a rules-based combat sport using punches."),
    ("Для чего нужна видеокарта?", "Should mention graphics and parallel computation."),
    ("Какая столица Франции?", "Expected fact: Paris."),
    ("Что делать, если ты не знаешь ответ?", "Should admit uncertainty rather than invent facts."),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/genesis-dialogue.pt")
    parser.add_argument("--tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.35)
    args = parser.parse_args()
    path = Path(args.checkpoint)
    if not path.exists():
        raise SystemExit(f"Checkpoint not found: {path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    saved = torch.load(path, map_location=device, weights_only=True)
    vocab = saved["vocab"]
    char_to_id = {char: index for index, char in enumerate(vocab)}
    config = ModelConfig(**saved["config"])
    model = GenesisLanguageModel(config).to(device)
    model.load_state_dict(saved["model"])
    model.eval()

    print(f"Checkpoint: {path}; training step: {saved.get('step', 'unknown')}")
    print("Review the answers manually. This script does not claim to score meaning automatically.")
    for question, criterion in QUESTIONS:
        prompt_text = f"Пользователь: {question}\nGENESIS: "
        unknown = char_to_id["<unk>"]
        ids = [char_to_id.get(char, unknown) for char in prompt_text]
        prompt = torch.tensor(ids, dtype=torch.long, device=device)[None, :]
        generated = model.generate(
            prompt,
            max_new_tokens=args.tokens,
            temperature=args.temperature,
            top_k=min(12, len(vocab)),
        )[0].tolist()
        answer = "".join(vocab[token] for token in generated[len(ids):])
        print("\n" + "=" * 60)
        print("QUESTION:", question)
        print("CHECK:", criterion)
        print("GENESIS:", answer.strip())


if __name__ == "__main__":
    main()

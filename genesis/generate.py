"""Generate Unicode text from a trained GENESIS character-level checkpoint."""
import argparse
from pathlib import Path
import torch
from genesis.model import GenesisLanguageModel, ModelConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/genesis-char.pt")
    parser.add_argument("--prompt", default="user: Привет!\nassistant:")
    parser.add_argument("--tokens", type=int, default=120, help="Number of characters to generate")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    path = Path(args.checkpoint)
    if not path.exists():
        raise SystemExit(f"Checkpoint not found: {path}. Train the character model first.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    saved = torch.load(path, map_location=device, weights_only=True)
    vocab = saved["vocab"]
    char_to_id = {char: index for index, char in enumerate(vocab)}
    unknown_id = char_to_id["<unk>"]
    config = ModelConfig(**saved["config"])
    model = GenesisLanguageModel(config).to(device)
    model.load_state_dict(saved["model"])
    model.eval()

    encoded = [char_to_id.get(char, unknown_id) for char in args.prompt]
    prompt = torch.tensor(encoded, dtype=torch.long, device=device)[None, :]
    result = model.generate(
        prompt,
        max_new_tokens=args.tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    decoded = "".join(vocab[index] if index < len(vocab) else "" for index in result[0].tolist())
    print(decoded)


if __name__ == "__main__":
    main()

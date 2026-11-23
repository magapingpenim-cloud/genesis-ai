"""Generate text from a trained GENESIS checkpoint."""
import argparse
from pathlib import Path
import torch
from genesis.model import GenesisLanguageModel, ModelConfig

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/genesis-small.pt")
    parser.add_argument("--prompt", default="GENESIS")
    parser.add_argument("--tokens", type=int, default=100)
    args = parser.parse_args()
    path = Path(args.checkpoint)
    if not path.exists():
        raise SystemExit(f"Checkpoint not found: {path}. Train first.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    saved = torch.load(path, map_location=device, weights_only=True)
    config = ModelConfig(**saved["config"])
    model = GenesisLanguageModel(config).to(device)
    model.load_state_dict(saved["model"])
    prompt = torch.tensor(list(args.prompt.encode("utf-8")), dtype=torch.long, device=device)[None, :]
    result = model.generate(prompt, max_new_tokens=args.tokens)
    print(bytes(result[0].tolist()).decode("utf-8", errors="replace"))

if __name__ == "__main__":
    main()

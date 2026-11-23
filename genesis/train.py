"""Train GENESIS with next-byte prediction on a local UTF-8 corpus."""
import argparse
from pathlib import Path
import torch
from genesis.model import GenesisLanguageModel, ModelConfig

def get_batch(data, batch_size, context_length, device):
    starts = torch.randint(0, data.numel() - context_length - 1, (batch_size,))
    x = torch.stack([data[i:i + context_length] for i in starts])
    y = torch.stack([data[i + 1:i + context_length + 1] for i in starts])
    return x.to(device), y.to(device)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/corpus.txt")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--output", default="checkpoints/genesis-small.pt")
    args = parser.parse_args()

    path = Path(args.data)
    if not path.exists():
        raise SystemExit(f"Corpus not found: {path}. Create data/corpus.txt first.")
    raw = path.read_bytes()
    if len(raw) < args.context_length * 2:
        raise SystemExit("Corpus is too small; add more UTF-8 text.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = ModelConfig(context_length=args.context_length)
    model = GenesisLanguageModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    tokens = torch.tensor(list(raw), dtype=torch.long)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Corpus bytes: {len(raw):,}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    model.train()
    for step in range(1, args.steps + 1):
        x, y = get_batch(tokens, args.batch_size, args.context_length, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            _, loss = model(x, y)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        if step == 1 or step % 50 == 0:
            print(f"step {step}/{args.steps} loss={loss.item():.4f}")
        if step % args.save_every == 0 or step == args.steps:
            torch.save({"model": model.state_dict(), "config": config.to_dict(), "step": step}, output)
            print(f"Saved checkpoint: {output}")
    print("Training complete. This is an experimental byte-level model, not yet a chat-ready assistant.")

if __name__ == "__main__":
    main()

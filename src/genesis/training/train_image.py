"""Train the GENESIS image diffusion baseline from scratch.

Manifest format: one JSON object per line with image and caption fields.
Example: {"image":"images/0001.jpg","caption":"A red bicycle beside a brick wall"}
"""
from __future__ import annotations
import argparse
import random
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from genesis.data.image_caption_dataset import ImageCaptionDataset
from genesis.diffusion.image_scheduler import ImageDDPM
from genesis.models.image_unet import ImageUNet, encode_texts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", default="checkpoints/genesis-image")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.image_size % 4:
        parser.error("--image-size must be divisible by 4")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = ImageCaptionDataset(args.manifest, args.image_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.workers, drop_last=True)
    if len(loader) == 0:
        parser.error("dataset must contain at least one full batch")

    model = ImageUNet(base_channels=args.base_channels).to(device)
    scheduler = ImageDDPM().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    model.train()
    step = 0
    while step < args.steps:
        for images, captions in loader:
            images = images.to(device)
            text_ids = encode_texts(list(captions), device)
            # Classifier-free conditioning dropout teaches an unconditional branch.
            if random.random() < 0.1:
                text_ids = torch.zeros_like(text_ids)
            times = torch.randint(0, scheduler.timesteps, (images.shape[0],), device=device)
            noise = torch.randn_like(images)
            noisy = scheduler.add_noise(images, noise, times)
            predicted = model(noisy, times, text_ids)
            loss = torch.nn.functional.mse_loss(predicted.float(), noise.float())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step += 1
            if step % 50 == 0:
                print(f"step={step}/{args.steps} loss={loss.item():.5f} device={device}")
            if step % args.save_every == 0 or step == args.steps:
                checkpoint = {
                    "step": step,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": vars(args),
                }
                torch.save(checkpoint, output / f"step-{step:08d}.pt")
            if step >= args.steps:
                break


if __name__ == "__main__":
    main()

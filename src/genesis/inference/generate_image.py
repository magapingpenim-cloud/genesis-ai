"""Generate an image from a prompt using a trained GENESIS checkpoint."""
from __future__ import annotations
import argparse
from pathlib import Path
import torch
from PIL import Image
from genesis.diffusion.image_scheduler import ImageDDPM
from genesis.models.image_unet import ImageUNet, encode_texts


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", default="genesis-output.png")
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--guidance", type=float, default=4.0)
    parser.add_argument("--base-channels", type=int, default=64)
    args = parser.parse_args()
    if args.size % 4:
        parser.error("--size must be divisible by 4")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})
    channels = config.get("base_channels", args.base_channels)
    model = ImageUNet(base_channels=channels).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    scheduler = ImageDDPM().to(device)
    prompt_ids = encode_texts([args.prompt], device)
    null_ids = torch.zeros_like(prompt_ids)
    image = torch.randn(1, 3, args.size, args.size, device=device)
    # Use a uniform subset of training timesteps for faster sampling if requested.
    indices = torch.linspace(scheduler.timesteps - 1, 0, args.steps).long().unique_consecutive()
    values = indices.tolist()
    for index, t in enumerate(values):
        previous_t = values[index + 1] if index + 1 < len(values) else -1
        image = scheduler.ddim_step(
            model, image, t, previous_t, prompt_ids, args.guidance, null_ids
        )
    image = (image.clamp(-1, 1) + 1) / 2
    pixels = (image[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype("uint8")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels).save(output)
    print(f"Saved {output.resolve()}")


if __name__ == "__main__":
    main()

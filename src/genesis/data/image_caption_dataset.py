"""Image-caption dataset backed by a JSONL manifest.

Each line: {"image": "relative/or/absolute/path.jpg", "caption": "description"}
"""
from __future__ import annotations
import json
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class ImageCaptionDataset(Dataset):
    def __init__(self, manifest: str | Path, image_size: int = 64):
        self.manifest = Path(manifest).expanduser().resolve()
        self.root = self.manifest.parent
        self.records = []
        with self.manifest.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                item = json.loads(line)
                if "image" not in item or "caption" not in item:
                    raise ValueError(f"manifest line {line_number} needs image and caption")
                self.records.append(item)
        if not self.records:
            raise ValueError("manifest contains no image-caption records")
        self.transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        item = self.records[index]
        path = Path(item["image"])
        if not path.is_absolute():
            path = self.root / path
        with Image.open(path) as image:
            image = image.convert("RGB")
            tensor = self.transform(image)
        return tensor, str(item["caption"])

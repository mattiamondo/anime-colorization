"""Dataset for the Anime Sketch Colorization Pair dataset (Kaggle).

Each raw file is a 1024x512 side-by-side image: left half is the color
target, right half is the sketch. The split into the two halves happens at
load time.

The Kaggle archive ships ``train/`` and ``val/`` folders:

- ``train``: kept unchanged.
- ``val`` / ``test``: ``val`` folder, sorted by filename, first
  half -> val, second half -> test.
"""

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

_SPLITS = ("train", "val", "test")


class AnimeColorizationDataset(Dataset):
    """
    Args:
        root: path to the extracted dataset (data/anime_colorization).
        split: one of "train", "val", "test".
        image_size: working resolution (default 256).
        paired: if False, sketch and color samples are drawn independently
            (via a fixed seeded permutation of the color indices) to simulate
            the unpaired setting for CycleGAN.
    """

    def __init__(self, root: str | Path, split: str = "train",
                 image_size: int = 256, paired: bool = True):
        super().__init__()
        if split not in _SPLITS:
            raise ValueError(f"split must be one of {_SPLITS}, got {split!r}")
        self.root = Path(root)
        self.split = split
        self.image_size = image_size
        self.paired = paired

        if split == "train":
            folder = self.root / "train"
            self.files = sorted(folder.glob("*.png"))
        else:
            folder = self.root / "val"
            files = sorted(folder.glob("*.png"))
            half = len(files) // 2
            self.files = files[:half] if split == "val" else files[half:]

        if not self.files:
            raise FileNotFoundError(f"No images found under {folder}")

        # Fixed permutation for the unpaired setting: color index i is
        # replaced by color_perm[i], decoupling sketches from their targets.
        generator = torch.Generator().manual_seed(0)
        self.color_perm = torch.randperm(len(self.files), generator=generator)

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size),
                              interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),  # [-1, 1]
        ])

    def __len__(self) -> int:
        return len(self.files)

    def _load_halves(self, index: int) -> tuple[Image.Image, Image.Image]:
        """Load one raw file and split it into (sketch, color) PIL halves."""
        image = Image.open(self.files[index]).convert("RGB")
        width, height = image.size
        half = width // 2
        color = image.crop((0, 0, half, height))
        sketch = image.crop((half, 0, width, height))
        return sketch, color

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if self.paired:
            sketch_pil, color_pil = self._load_halves(index)
        else:
            sketch_pil, _ = self._load_halves(index)
            color_index = int(self.color_perm[index])
            _, color_pil = self._load_halves(color_index)

        return {
            "sketch": self.transform(sketch_pil),
            "color": self.transform(color_pil),
        }

"""Dataset for the Anime Sketch Colorization Pair dataset (Kaggle).

Each raw file is a 1024x512 side-by-side image: left half is the color
target, right half is the sketch. The split into the two halves happens at
load time.

The Kaggle archive ships only ``train/`` and ``val/`` folders:

- ``train``: the Kaggle ``train`` folder as-is.
- ``val`` / ``test``: the Kaggle ``val`` folder, sorted by filename, first
  half -> val, second half -> test. Sorting makes the split reproducible
  without any RNG.

Optional colorgram support: if a ``colorgram/`` subdirectory exists under
``root`` (or an explicit ``colorgram_dir`` is provided), each sample also
returns a ``colorgram`` tensor of shape (16, 3) with RGB values in [-1, 1],
representing the 4×4 dominant-color palette of the color target.
"""

import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

_SPLITS = ("train", "val", "test")


class AnimeColorizationDataset(Dataset):
    """Paired sketch/color dataset.

    Args:
        root: path to the extracted Kaggle dataset (data/anime_colorization).
        split: one of "train", "val", "test".
        image_size: working resolution (default 256).
        paired: if False, sketch and color samples are drawn independently
            (via a fixed seeded permutation of the color indices) to simulate
            the unpaired setting for CycleGAN.
    """

    def __init__(self, root: str | Path, split: str = "train",
                 image_size: int = 256, paired: bool = True,
                 colorgram_dir: str | Path | None = None):
        super().__init__()
        if split not in _SPLITS:
            raise ValueError(f"split must be one of {_SPLITS}, got {split!r}")
        self.root = Path(root)
        self.split = split
        self.image_size = image_size
        self.paired = paired

        # Colorgram directory: auto-detect if not specified.
        if colorgram_dir is not None:
            self.colorgram_dir: Path | None = Path(colorgram_dir)
        else:
            candidate = self.root / "colorgram"
            self.colorgram_dir = candidate if candidate.is_dir() else None

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

    def _load_colorgram(self, index: int) -> torch.Tensor | None:
        """Load colorgram for image at ``index``. Returns (16, 3) float tensor
        with values in [-1, 1], or None if no colorgram directory is set."""
        if self.colorgram_dir is None:
            return None
        stem = self.files[index].stem
        json_path = self.colorgram_dir / f"{stem}.json"
        if not json_path.exists():
            return None
        data = json.loads(json_path.read_text())
        colors: list[list[int]] = []
        for row in sorted(data.keys(), key=int):
            for col in sorted(data[row].keys(), key=int):
                colors.append(data[row][col])
        # (16, 3) float in [0, 255] → [-1, 1]
        t = torch.tensor(colors, dtype=torch.float32) / 127.5 - 1.0
        return t

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if self.paired:
            sketch_pil, color_pil = self._load_halves(index)
        else:
            sketch_pil, _ = self._load_halves(index)
            color_index = int(self.color_perm[index])
            _, color_pil = self._load_halves(color_index)

        sample: dict[str, torch.Tensor] = {
            "sketch": self.transform(sketch_pil),
            "color": self.transform(color_pil),
        }
        colorgram = self._load_colorgram(index)
        if colorgram is not None:
            sample["colorgram"] = colorgram
        return sample

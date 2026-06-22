"""Evaluation metrics: PSNR, SSIM, LPIPS, FID.

Inputs are expected in [-1, 1] and remapped internally where needed.
"""

import shutil
from pathlib import Path

import torch
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchvision.utils import save_image
from tqdm import tqdm


def _to_unit_range(x: torch.Tensor) -> torch.Tensor:
    """[-1, 1] -> [0, 1], clamped."""
    return ((x + 1) / 2).clamp(0, 1)


def compute_psnr(prediction: torch.Tensor, target: torch.Tensor) -> float:
    """Peak Signal-to-Noise Ratio, averaged over the batch."""
    metric = PeakSignalNoiseRatio(data_range=1.0).to(prediction.device)
    return metric(_to_unit_range(prediction), _to_unit_range(target)).item()


def compute_ssim(prediction: torch.Tensor, target: torch.Tensor) -> float:
    """Structural Similarity Index, averaged over the batch."""
    metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(prediction.device)
    return metric(_to_unit_range(prediction), _to_unit_range(target)).item()


def compute_lpips(prediction: torch.Tensor, target: torch.Tensor) -> float:
    """Learned Perceptual Image Patch Similarity"""
    metric = LearnedPerceptualImagePatchSimilarity(net_type="alex").to(prediction.device)
    return metric(prediction.clamp(-1, 1), target.clamp(-1, 1)).item()


def compute_fid(generated_dir: str, reference_dir: str,
                device: str = "cuda", batch_size: int = 50) -> float:
    """Frechet Inception Distance"""
    from pytorch_fid import fid_score

    return fid_score.calculate_fid_given_paths(
        [str(generated_dir), str(reference_dir)],
        batch_size=batch_size, device=device, dims=2048)


@torch.no_grad()
def evaluate_model(generator: torch.nn.Module, test_loader, device: str = "cuda",
                   fid_dir: str | Path = "results/fid") -> dict[str, float]:
    """Run a generator over the test set and return {psnr, ssim, lpips, fid}.

    Args:
        generator: nn.Module mapping sketch -> color (e.g. trainer.generator).
        test_loader: dataloader over the test split.
        fid_dir: directory for generated/real PNGs used by FID. Wiped and
            recreated on each call so runs never mix images.
    """
    generator = generator.to(device).eval()

    fid_dir = Path(fid_dir)
    if fid_dir.exists():
        shutil.rmtree(fid_dir)
    generated_dir = fid_dir / "generated"
    reference_dir = fid_dir / "real"
    generated_dir.mkdir(parents=True)
    reference_dir.mkdir(parents=True)

    psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex").to(device)

    index = 0
    for batch in tqdm(test_loader, desc="evaluate"):
        fake_dev = generator(batch["sketch"].to(device))
        real_dev = batch["color"].to(device)
        # Copy data to CPU for the image saving loop
        fake, real = fake_dev.cpu(), real_dev.cpu()
        # Update using _dev versions because metrics are in device
        psnr.update(_to_unit_range(fake_dev), _to_unit_range(real_dev))
        ssim.update(_to_unit_range(fake_dev), _to_unit_range(real_dev))
        lpips.update(fake_dev.clamp(-1, 1), real_dev.clamp(-1, 1))
        for i in range(fake.shape[0]):
            save_image(_to_unit_range(fake[i]), generated_dir / f"{index}.png")
            save_image(_to_unit_range(real[i]), reference_dir / f"{index}.png")
            index += 1

    return {
        "psnr": psnr.compute().item(),
        "ssim": ssim.compute().item(),
        "lpips": lpips.compute().item(),
        "fid": compute_fid(generated_dir, reference_dir, device=device),
    }

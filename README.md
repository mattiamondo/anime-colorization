# Anime Sketch Colorization

Ablation study on **paired vs unpaired supervision** for anime sketch colorization.
Deep Learning exam project — University of Verona, A.Y. 2025-26.

## Research question

Does paired supervision (Pix2Pix) produce measurably better colorization quality
than unpaired supervision (CycleGAN), and how does each loss component contribute
to the final result?

## Dataset

[Anime Sketch Colorization Pair](https://www.kaggle.com/datasets/ktaebum/anime-sketch-colorization-pair)
— ~17,769 paired sketch/color images (512×1024, left = color, right = sketch).

## Model variants

| # | Variant | Key change |
|---|---------|-----------|
| 1 | U-Net + L1 | Baseline — no adversarial loss |
| 2 | Pix2Pix | U-Net + PatchGAN discriminator |
| 3 | Pix2Pix + perceptual | VGG-19 feature-space loss added |
| 4 | CycleGAN (unpaired) | Trained without pairing information |
| 5 | Pix2Pix + global disc. | PatchGAN replaced with global discriminator |
| 6 | Pix2Pix lambda ablation | Rebalanced loss weights (L1=50, GAN=10) |

## Evaluation metrics

FID · SSIM · PSNR · LPIPS

## Repository structure

```
src/          reusable modules (dataset, models, losses, trainers, metrics, viz)
scripts/      headless training scripts (one per variant)
notebooks/    evaluation notebooks (checkpoint loading + metrics + qualitative grids)
report/       technical report (PDF)
```

## Report

See [report/Report_VR538106.pdf](report/Report_VR538106.pdf).

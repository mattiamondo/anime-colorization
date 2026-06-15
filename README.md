# Anime Sketch Colorization — Paired vs Unpaired Supervision

Ablation study on paired (Pix2Pix) vs unpaired (CycleGAN) supervision for anime
sketch colorization. Deep Learning exam project, UniVR A.Y. 2025-26.

## Research question

Does paired supervision produce measurably better colorization quality than
unpaired supervision, and how does each loss component (L1, adversarial,
perceptual) contribute to the final result?

## Model variants

| # | Variant | Supervision | Losses |
|---|---------|-------------|--------|
| 1 | U-Net baseline | paired | L1 |
| 2 | Pix2Pix | paired | L1 + adversarial (PatchGAN) |
| 3 | Pix2Pix + perceptual | paired | L1 + adversarial + VGG-19 perceptual |
| 4 | CycleGAN | **unpaired** | adversarial + cycle-consistency |
| 5 | Conditional VAE | paired | reconstruction + KL |

## Setup

```bash
conda env create -f environment.yml
conda activate anime-colorization
```

Download the [Anime Sketch Colorization Pair](https://www.kaggle.com/datasets/ktaebum/anime-sketch-colorization-pair)
dataset from Kaggle and extract it into `data/anime_colorization/`.

## Repository layout

- `src/` — reusable modules: dataset, models, losses, trainers, metrics, visualization
- `notebooks/` — one notebook per experiment; they import from `src/` and contain
  only configuration, training runs, curves and qualitative grids
- `checkpoints/` — saved weights (gitignored)
- `results/` — quantitative tables and figures

## Evaluation

FID (`pytorch-fid` / torchmetrics), SSIM, PSNR over the test set, plus
qualitative sketch / prediction / ground-truth grids for every variant.

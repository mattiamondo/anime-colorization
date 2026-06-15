# CLAUDE.md — Anime Sketch Colorization

Ablation study on **paired vs unpaired supervision** for anime sketch colorization.
Deep Learning exam project (individual), UniVR.

## Course context

- Course: Machine Learning & Deep Learning / Deep Learning, UniVR A.Y. 2025-26 (Profs. Murino, Beyan, Lucato)
- Deliverables: technical report + GitHub repo + 8-10 min presentation
- Hard requirements: **minimum 5 model variants**; evaluation with standard metrics (course rubric mentions confusion matrix / accuracy / recall / precision — for this generative task the equivalents are FID / SSIM / PSNR, justify this mapping in the report)
- Constraint: ChatGPT must NOT be used to write the report

## Central scientific question

Does paired supervision (Pix2Pix) produce measurably better colorization quality than
unpaired supervision (CycleGAN), and how does each loss component (L1, adversarial,
perceptual) contribute to the final result?

## Dataset

- **Anime Sketch Colorization Pair** (Kaggle), ~17,769 paired images
- Each file is a side-by-side 512×1024 image: **left = color target, right = sketch**
  (verified empirically 2026-06-12: the right half is grayscale; the earlier note
  saying "left = sketch" was wrong)
- The dataset class must split the image into the two halves at load time and accept a
  `split` argument (`train` / `val` / `test`)
- Working resolution: 256×256 for training; upscale to 512×512 only if VRAM allows

## Model variants (progressive ablation — order matters)

1. **U-Net + L1 only** — encoder-decoder with skip connections, pixel-wise L1 loss, no adversarial component (baseline)
2. **Pix2Pix** — U-Net generator + 70×70 PatchGAN discriminator + L1 (primary supervised model)
3. **Pix2Pix + perceptual loss** — VGG-19 feature-space loss (layers `relu_1_2`, `relu_2_2`, `relu_3_3`) supplementing/replacing L1
4. **CycleGAN (unpaired)** — trained WITHOUT pairing information: sketch domain and color domain treated as independent unpaired sets, even though pairs exist. This is intentional, to quantify the value of paired supervision against Pix2Pix
5. **Conditional VAE** — latent-space conditional generation (condition on the sketch via channel concatenation or a separate encoder); explores diversity/stochasticity
6. **(Bonus) Pix2Pix + global discriminator** — replace the 70×70 PatchGAN with a standard global discriminator (single scalar output) to isolate the contribution of patch-level adversarial signal on local texture sharpness; requires a new discriminator class in `src/models/`. Only if time allows after the 5 required variants.

## Evaluation

- **FID** (distribution-level quality) — use `pytorch-fid` or equivalent, computed over the test set
- **SSIM** (structural fidelity)
- **PSNR** (pixel-level fidelity)
- **LPIPS** (paired perceptual distance, AlexNet backbone — added 2026-06-12; covers
  the perceptual-but-per-image quadrant that FID (perceptual, distribution-level)
  and PSNR/SSIM (per-image, non-perceptual) leave open; Zhang et al., CVPR 2018)
- Qualitative grids: sketch / prediction / ground truth side by side, for every variant

## Compute environment

- Primary: RTX 3090 server "pfizer" (24 GB VRAM), accessed via SSH + VSCode Remote
- Backup: Google Colab free tier (Tesla T4)
- Conda environment dedicated to this project (Python 3.10, PyTorch).
  **NEVER run Python scripts outside the project's conda environment.**

## Code conventions

- **Hybrid structure (decided 2026-06-11)**: all reusable code (dataset, models, losses, trainers, metrics, visualization) lives in .py modules under `src/`; notebooks are thin — one per variant — containing only configuration, training runs, loss curves and qualitative grids
- Variants 1-3 share `Pix2PixTrainer`, ablated via loss weights (`lambda_gan=0` → variant 1, `lambda_perceptual=0` → variant 2)
- All code and comments in .py files must be written in **English**
- Every experiment must log train and val loss curves and save the best checkpoint by validation metric
- Seed everything (`torch`, `numpy`, `random`) for reproducibility

## Target repository structure

```
DL_Project/
├── CLAUDE.md
├── README.md
├── environment.yml             # conda env: anime-colorization (py3.10, PyTorch)
├── data/
│   └── anime_colorization/     # raw Kaggle dataset (gitignored)
├── notebooks/                  # thin: config + run + plots, import from src/
│   ├── 00_dataset_exploration.ipynb
│   ├── 01_unet_l1_baseline.ipynb
│   ├── 02_pix2pix.ipynb
│   ├── 03_pix2pix_perceptual.ipynb
│   ├── 04_cyclegan.ipynb
│   └── 05_cvae.ipynb
├── src/
│   ├── data/dataset.py         # AnimeColorizationDataset (split + paired flag)
│   ├── models/                 # unet.py, patchgan.py, cyclegan.py, cvae.py
│   ├── losses/                 # gan.py, perceptual.py (VGG-19)
│   ├── training/               # base.py + one trainer per family
│   └── utils/                  # seed.py, metrics.py, visualization.py
├── checkpoints/                # saved model weights (gitignored)
└── results/
    ├── figures/                # loss curves, qualitative grids
    └── tables/                 # quantitative metrics (FID/SSIM/PSNR)
```

## Key implementation notes

- PatchGAN discriminator: standard Pix2Pix configuration (70×70 receptive field)
- CycleGAN training must not use the paired labels in any way (loss, sampling, or batching)
- The specific choices above (architectures, layers, libraries) are starting points — they may be revised if best practices suggest otherwise, but revisions should be documented here and in the report

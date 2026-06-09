# Project Context for Claude Code

## What this project is

Deep Learning exam project for the Machine Learning & Deep Learning course (A.Y. 2025-26) at the University of Verona (UniVR), taught by Prof. Murino, Prof. Beyan, and Prof. Lucato.

**Task:** Multi-label fashion attribute classification using the Sketchy dataset (Girella et al., ICCV 2025).

**Research question:** Do large-scale pre-trained vision-language representations (CLIP) outperform supervised domain-specific representations (ResNet-50, ViT) on fine-grained fashion attribute recognition? How do both behave under domain shift (real photos → garment sketches)?

---

## Directory structure

```
/media/data/mmondo/mmondo/DL_Project/
├── data/
│   └── sketchy/                        # Sketchy repo (authors' code + dataset)
│       ├── train_sketchy.json          # Final train annotations (362 MB)
│       ├── test_sketchy.json           # Final test annotations (9.7 MB)
│       ├── train/
│       │   ├── images/                 # 512x512 padded train images (one folder per image_id)
│       │   └── partial_sketches/       # Localized garment sketches (one folder per image_id)
│       ├── test/
│       │   ├── images/                 # 512x512 padded test images
│       │   └── partial_sketches/       # Localized garment sketches
│       └── src/sketchy/
│           └── sketchy_dataset.py      # Official PyTorch Dataset class (use this)
├── src/                                # Mattia's code lives here
│   ├── 00_eda.ipynb                    # Exploratory Data Analysis
│   ├── 00b_label_filtering.ipynb       # Label space design → produces valid_labels.json
│   ├── valid_labels.json               # Final label vocabulary (load this, do not hardcode)
│   ├── 01_clip_zeroshot.ipynb
│   ├── 02_clip_linear_probe.ipynb
│   ├── 03_resnet50.ipynb
│   ├── 04_vit_small.ipynb
│   ├── 05_clip_finetuned.ipynb
│   ├── 06_cross_domain.ipynb
│   └── utils.py                        # Shared utilities (dataloader setup, metrics, etc.)
└── results/                            # Saved plots, confusion matrices, metrics
```

---

## Dataset: Sketchy

Built on top of Fashionpedia (ECCV 2020). Each sample is an outfit image with:
- **Hierarchical multi-label attributes** (`top_level` and `sub_level` fields in annotations)
- **Localized garment sketches** (one per annotated garment, in `partial_sketches/`)
- **LLaMA 3.1-generated textual descriptions** (`description` field in annotations)

**Annotation structure** (one annotation = one garment in an image):
```python
{
  'id': 5280,
  'image_id': 16145,
  'category_name': 'shorts',
  'category_id': 7,
  'attribute_ids': [316, 142, 328, ...],
  'top_level': ['normal waist', 'stripe', 'regular (fit)', 'mini (length)', ...],
  'sub_level': None,
  'description': 'Normal waist, mini-length, trunks shorts with a regular fit and stripe design',
  'filename': '5280.jpg',          # sketch filename
  'bbox': [128.0, 0.0, 421.0, 318.0],
  'segmentation': [...],
  'padding': [0, 0, 0, 0],
  'final_height': 512,
  'final_width': 512,
}
```

**Dataset stats:**
- Train: ~45,585 images (shoes removed by default)
- Test: ~1,158 images
- Labels: `top_level` attributes are the multi-label targets

---

## How to load the dataset

```python
import sys
sys.path.append('/media/data/mmondo/mmondo/DL_Project/data/sketchy/src')
from sketchy.sketchy_dataset import SketchyDataset

DATASET_ROOT = '/media/data/mmondo/mmondo/DL_Project/data/sketchy'

dataset = SketchyDataset(
    dataset_root=DATASET_ROOT,
    split='train',           # 'train' or 'test'
    load_img=True,
    load_local_sketch=True,
    load_global_sketch=True, # required if compose_global_sketch=True (default)
)
```

Each `__getitem__` returns a dict with:
- `image` — PIL Image or tensor (512x512)
- `annotations` — list of annotation dicts (variable length per image)
- `local_descriptions` — list of text descriptions (one per garment)
- `global_description` — all descriptions concatenated
- `local_sketches` — list of sketch PIL Images
- `global_sketch` — composed global sketch

---

## Label filtering strategy

All model notebooks use the label vocabulary in `src/valid_labels.json` (produced by `00b_label_filtering.ipynb`).

Two filters applied in sequence:
1. **Remove `nickname` supercategory** — 153 Fashionpedia attributes that are garment sub-types (e.g. "raglan t-shirt", "sailor shirt"). Too fine-grained and rare to be learned from images.
2. **Remove labels with < `MIN_FREQ=50` occurrences in train** — eliminates statistically unusable labels (e.g. entire `neckline type` supercategory: 31 occurrences total).

Result: ~60-80 visually grounded labels covering ~89% of all label occurrences.

This is **label space design**, not feature engineering — images are passed raw to the model.

Load in any notebook with:
```python
import json
with open('/media/data/mmondo/mmondo/DL_Project/src/valid_labels.json') as f:
    valid_labels = json.load(f)
```

---

## Models to implement (in order)

| # | Model | Notes |
|---|---|---|
| 1 | CLIP zero-shot | No training — just prompt engineering per class |
| 2 | CLIP linear probe | CLIP frozen, only a linear head trained |
| 3 | ResNet-50 fine-tuned | `torchvision.models.resnet50(weights='IMAGENET1K_V2')` |
| 4 | ViT-small fine-tuned | `timm.create_model('vit_small_patch16_224', pretrained=True)` |
| 5 | CLIP fine-tuned | Full fine-tuning with fp16 |
| 6 | Cross-domain eval | Best model evaluated on sketches instead of photos |

---

## Evaluation metrics (required by exam)

For every model compute and save:
- **Accuracy** (per-class and overall)
- **Precision** (macro and per-class)
- **Recall** (macro and per-class)
- **F1** (macro)
- **Confusion matrix** (saved to `results/`)

Use `sklearn.metrics` for all of these.

---

## Environment

- **Server:** `pfizer` via SSH, VSCode Remote
- **Conda env:** `sketchy` (Python 3.10) — always activate before running
- **GPU:** available on server (check with `nvidia-smi`)
- **Backup:** Google Colab (free tier, Tesla T4) with dataset on Google Drive at `DL_Project/data/sketchy/`

---

## Key dependencies

```
fashionpedia==1.1
pycocotools==2.0.8
torch==2.5.1
torchvision==0.20.1
timm
transformers  (for CLIP)
scikit-learn  (for metrics)
pillow==11.0.0
tqdm
```

Install with:
```bash
conda activate sketchy
pip install -r /media/data/mmondo/mmondo/DL_Project/data/sketchy/requirements.txt
pip install timm transformers scikit-learn
```

---

## Important notes

- The **labels for classification** are `top_level` attributes in each annotation — these are the multi-label targets.
- Each image can have **multiple annotations** (multiple garments), each with its own label set.
- Shoes are **removed by default** by `SketchyDataset` (`with_shoes=False`).
- The `collate_fn` provided by `SketchyDataset` handles variable-length annotation lists — use it with DataLoader.
- For multi-label classification, use **BCEWithLogitsLoss** as the loss function.
- Sketchy license on HuggingFace is **CC-BY-NC-4.0** (non-commercial).
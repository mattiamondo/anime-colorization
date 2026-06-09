import sys
import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__)))

DATASET_ROOT = '/media/data/mmondo/mmondo/DL_Project/data/sketchy'
VALID_LABELS_PATH = '/media/data/mmondo/mmondo/DL_Project/src/valid_labels.json'


# ---------------------------------------------------------------------------
# Crop utilities
# ---------------------------------------------------------------------------

def bbox_to_512(ann, img_data):
    """Trasforma la bbox dall'originale Fashionpedia allo spazio 512×512.

    Le immagini Fashionpedia vengono processate con ImageOps.contain → 512×512,
    poi paddato con bianco. Le bbox nelle annotazioni sono nello spazio originale;
    questa funzione applica la stessa trasformazione per riportarle a 512×512.
    """
    pad_l, pad_t, pad_r, pad_b = ann['padding']
    orig_w = img_data['width']
    orig_h = img_data['height']

    scale = min(
        (512 - pad_l - pad_r) / orig_w,
        (512 - pad_t - pad_b) / orig_h,
    )

    x, y, w, h = ann['bbox']
    return (x * scale + pad_l, y * scale + pad_t, w * scale, h * scale)


def crop_garment(image, ann, img_data, target_size=224):
    """Croppa il singolo capo dalla foto 512×512, padda a quadrato bianco, ridimensiona.

    Args:
        image:       PIL Image 512×512
        ann:         annotazione del capo (dict con 'bbox' e 'padding')
        img_data:    img_data del sample (dict con 'width' e 'height' originali)
        target_size: dimensione finale del crop quadrato

    Returns:
        PIL Image (target_size × target_size, RGB)
    """
    x, y, w, h = bbox_to_512(ann, img_data)

    x1 = max(0, int(x))
    y1 = max(0, int(y))
    x2 = min(image.width,  int(x + w))
    y2 = min(image.height, int(y + h))

    crop = image.crop((x1, y1, x2, y2))

    side = max(crop.width, crop.height)
    padded = Image.new('RGB', (side, side), (255, 255, 255))
    padded.paste(crop, ((side - crop.width) // 2, (side - crop.height) // 2))

    return padded.resize((target_size, target_size), Image.BILINEAR)


# ---------------------------------------------------------------------------
# GarmentDataset
# ---------------------------------------------------------------------------

class GarmentDataset(Dataset):
    """Wrapper su SketchyDataset che itera per singolo capo invece che per immagine.

    Ogni sample è una singola annotazione (un capo), con il crop della foto
    reale come input visivo e il vettore multi-label degli attributi come target.
    Annotazioni con crop < MIN_CROP_PX in qualche dimensione vengono scartate.

    Args:
        sketchy_dataset: istanza di SketchyDataset (load_img=True richiesto)
        valid_labels:    lista ordinata di label valide (da valid_labels.json)
        target_size:     dimensione del crop quadrato (default 224)
        transform:       torchvision transform applicata al crop PIL prima di restituirlo
    """

    MIN_CROP_PX = 10  # soglia minima in pixel nello spazio 512×512

    def __init__(self, sketchy_dataset, valid_labels, target_size=224, transform=None):
        self.ds           = sketchy_dataset
        self.valid_labels = valid_labels
        self.label_to_idx = {l: i for i, l in enumerate(valid_labels)}
        self.target_size  = target_size
        self.transform    = transform

        self.index  = []
        n_skipped   = 0
        for i in range(len(self.ds)):
            sample   = self.ds[i]
            img_data = sample['img_data']
            for j, ann in enumerate(sample['annotations']):
                x, y, w, h = bbox_to_512(ann, img_data)
                x1, y1 = max(0, x), max(0, y)
                x2, y2 = min(512, x + w), min(512, y + h)
                if (x2 - x1) < self.MIN_CROP_PX or (y2 - y1) < self.MIN_CROP_PX:
                    n_skipped += 1
                    continue
                self.index.append((i, j))

        print(f"GarmentDataset — {len(self.index):,} annotazioni ({n_skipped} tiny scartate)")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        sample_idx, ann_idx = self.index[idx]
        sample   = self.ds[sample_idx]
        ann      = sample['annotations'][ann_idx]
        img_data = sample['img_data']
        image    = sample['image'] if isinstance(sample['image'], Image.Image) \
                   else Image.open(sample['image'])

        crop = crop_garment(image, ann, img_data, target_size=self.target_size)
        if self.transform:
            crop = self.transform(crop)

        label_vec = torch.zeros(len(self.valid_labels))
        for attr in (ann.get('top_level') or []):
            if attr in self.label_to_idx:
                label_vec[self.label_to_idx[attr]] = 1.0

        return {
            'image':    crop,
            'labels':   label_vec,
            'category': ann['category_name'],
            'ann_id':   ann['id'],
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_valid_labels(path=VALID_LABELS_PATH):
    with open(path) as f:
        return json.load(f)


def build_datasets(valid_labels, target_size=224, train_transform=None, val_transform=None):
    """Istanzia SketchyDataset + GarmentDataset per train e test.

    Returns:
        (train_garment_ds, test_garment_ds)
    """
    from sketchy.sketchy_dataset import SketchyDataset  # src/sketchy/sketchy_dataset.py

    train_sketchy = SketchyDataset(
        dataset_root=DATASET_ROOT, split='train',
        load_img=True, load_local_sketch=False, load_global_sketch=False,
    )
    test_sketchy = SketchyDataset(
        dataset_root=DATASET_ROOT, split='test',
        load_img=True, load_local_sketch=False, load_global_sketch=False,
    )

    train_ds = GarmentDataset(train_sketchy, valid_labels,
                               target_size=target_size, transform=train_transform)
    test_ds  = GarmentDataset(test_sketchy,  valid_labels,
                               target_size=target_size, transform=val_transform)

    return train_ds, test_ds

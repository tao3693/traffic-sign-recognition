"""
GTSRB data pipeline.

Key design decisions (explain these in the report, section "Data set"):
  * Validation split is done BY TRACK. Each physical sign was filmed as a track of
    ~30 consecutive frames (file name 000XX_000YY.ppm, XX = track id). A random
    split would put near-identical frames in train and val -> inflated val accuracy.
  * Optional ROI crop using the bounding box given in the annotation CSV.
  * No horizontal flip in augmentation: flipping changes the meaning of some signs
    (e.g. "turn left" <-> "turn right").
  * Class imbalance (~210 to ~2250 images per class) is handled with
    WeightedRandomSampler (weight = 1 / class frequency).
"""
import csv
import os
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import List, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

# Per-channel mean / std of GTSRB training images (RGB, range 0-1)
GTSRB_MEAN = (0.3403, 0.3121, 0.3214)
GTSRB_STD = (0.2724, 0.2608, 0.2669)


@dataclass
class Sample:
    path: str
    roi: Tuple[int, int, int, int]  # x1, y1, x2, y2
    label: int
    track: int = -1


# ---------------------------------------------------------------- annotations
def _read_csv(csv_path: str) -> List[dict]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter=";"))


def _row_to_sample(row: dict, img_dir: str, track: int = -1) -> Sample:
    roi = (int(row["Roi.X1"]), int(row["Roi.Y1"]), int(row["Roi.X2"]), int(row["Roi.Y2"]))
    return Sample(os.path.join(img_dir, row["Filename"]), roi, int(row["ClassId"]), track)


def load_train_samples(data_dir: str) -> List[Sample]:
    """Read every GT-xxxxx.csv under Final_Training/Images/<class>/."""
    root = os.path.join(data_dir, "Final_Training", "Images")
    samples = []
    for class_dir in sorted(os.listdir(root)):
        folder = os.path.join(root, class_dir)
        if not os.path.isdir(folder):
            continue
        for row in _read_csv(os.path.join(folder, f"GT-{class_dir}.csv")):
            track = int(row["Filename"].split("_")[0])
            samples.append(_row_to_sample(row, folder, track))
    return samples


def load_test_samples(data_dir: str) -> List[Sample]:
    """Test images + labels from GT-final_test.csv (from GTSRB_Final_Test_GT.zip)."""
    img_dir = os.path.join(data_dir, "Final_Test", "Images")
    rows = _read_csv(os.path.join(data_dir, "GT-final_test.csv"))
    return [_row_to_sample(r, img_dir) for r in rows]


def split_by_track(samples: List[Sample], val_ratio: float, seed: int):
    """Hold out whole tracks (per class) for validation -> no frame leakage."""
    groups = defaultdict(list)                      # (class, track) -> samples
    for s in samples:
        groups[(s.label, s.track)].append(s)

    tracks_per_class = defaultdict(list)
    for (label, track) in groups:
        tracks_per_class[label].append(track)

    rng = random.Random(seed)
    train, val = [], []
    for label, tracks in tracks_per_class.items():
        tracks = sorted(tracks)
        rng.shuffle(tracks)
        n_val = max(1, round(len(tracks) * val_ratio))
        val_tracks = set(tracks[:n_val])            # slicing the shuffled track list
        for t in tracks:
            (val if t in val_tracks else train).extend(groups[(label, t)])
    return train, val


# ------------------------------------------------------------------- dataset
class GTSRBDataset(Dataset):
    def __init__(self, samples: List[Sample], transform=None, crop_roi: bool = True):
        self.samples = samples
        self.transform = transform
        self.crop_roi = crop_roi

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = Image.open(s.path).convert("RGB")
        if self.crop_roi:
            img = img.crop(s.roi)
        if self.transform:
            img = self.transform(img)
        return img, s.label

    @property
    def labels(self) -> List[int]:
        return [s.label for s in self.samples]


def build_transforms(img_size: int, train: bool, augment: bool = True):
    steps = [transforms.Resize((img_size, img_size))]
    if train and augment:
        steps += [
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        ]
    steps += [transforms.ToTensor(), transforms.Normalize(GTSRB_MEAN, GTSRB_STD)]
    return transforms.Compose(steps)


def make_balanced_sampler(labels: List[int]) -> WeightedRandomSampler:
    """Sampling with probability proportional to 1 / class frequency."""
    counts = Counter(labels)
    weights = torch.tensor([1.0 / counts[y] for y in labels], dtype=torch.double)
    return WeightedRandomSampler(weights, num_samples=len(labels), replacement=True)


def build_dataloaders(cfg: dict):
    """Return train / val / test DataLoaders according to config."""
    all_train = load_train_samples(cfg["data_dir"])
    train_s, val_s = split_by_track(all_train, cfg["val_ratio"], cfg["seed"])

    size, crop = cfg["img_size"], cfg["crop_roi"]
    train_ds = GTSRBDataset(train_s, build_transforms(size, True, cfg["augment"]), crop)
    val_ds = GTSRBDataset(val_s, build_transforms(size, False), crop)

    sampler = make_balanced_sampler(train_ds.labels) if cfg["balanced_sampling"] else None
    common = dict(batch_size=cfg["batch_size"], num_workers=cfg["num_workers"],
                  pin_memory=torch.cuda.is_available(),          # faster CPU->GPU copy
                  persistent_workers=cfg["num_workers"] > 0)
    train_loader = DataLoader(train_ds, sampler=sampler, shuffle=sampler is None, **common)
    val_loader = DataLoader(val_ds, shuffle=False, **common)

    test_loader = None
    if os.path.exists(os.path.join(cfg["data_dir"], "GT-final_test.csv")):
        test_ds = GTSRBDataset(load_test_samples(cfg["data_dir"]), build_transforms(size, False), crop)
        test_loader = DataLoader(test_ds, shuffle=False, **common)
    return train_loader, val_loader, test_loader

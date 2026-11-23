"""
Exploratory data analysis for the report's "Data set" section.

Usage:
    python -m src.eda              figures + statistics
    python -m src.eda --stats      also compute the per-channel mean / std of the
                                   ROI-cropped training images (compare with GTSRB_MEAN/STD
                                   in src/dataset.py)
Outputs (report/figures/):
    class_distribution.png   number of training images per class (shows imbalance)
    samples.png              one example image per class
    image_sizes.png          distribution of original image sizes
    + statistics printed to the console
"""
import argparse
import os
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.dataset import load_train_samples, split_by_track
from src.utils import CLASS_NAMES, NUM_CLASSES, load_config

OUT = os.path.join("report", "figures")


def channel_stats(samples, crop_roi=True):
    """Exact per-channel mean / std over all pixels of the (ROI-cropped) images."""
    total, sq, n = np.zeros(3), np.zeros(3), 0
    for i, s in enumerate(samples):
        img = Image.open(s.path).convert("RGB")
        if crop_roi:
            img = img.crop(s.roi)
        a = np.asarray(img, dtype=np.float64).reshape(-1, 3) / 255.0
        total += a.sum(0)
        sq += (a ** 2).sum(0)
        n += a.shape[0]
        if i % 5000 == 0:
            print(f"  stats: {i}/{len(samples)} images")
    mean = total / n
    std = np.sqrt(sq / n - mean ** 2)
    return mean, std


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats", action="store_true", help="compute mean / std (slow)")
    args = parser.parse_args()
    cfg = load_config()
    os.makedirs(OUT, exist_ok=True)
    samples = load_train_samples(cfg["data_dir"])
    counts = Counter(s.label for s in samples)
    train, val = split_by_track(samples, cfg["val_ratio"], cfg["seed"])

    print(f"Training images: {len(samples)}  | classes: {len(counts)}")
    print(f"Tracks: {len({(s.label, s.track) for s in samples})}")
    print(f"Split by track -> train {len(train)}, val {len(val)}")
    print(f"Min / max per class: {min(counts.values())} / {max(counts.values())}")

    # class distribution
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(range(NUM_CLASSES), [counts[i] for i in range(NUM_CLASSES)], color="#4C72B0")
    ax.set(xlabel="Class id", ylabel="# images", title="GTSRB training set class distribution")
    ax.set_xticks(range(NUM_CLASSES))
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "class_distribution.png"), dpi=150)
    plt.close(fig)

    # one sample per class
    first = {}
    for s in samples:
        first.setdefault(s.label, s)
    fig, axes = plt.subplots(5, 9, figsize=(14, 8.5))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, cls in zip(axes.ravel(), range(NUM_CLASSES)):
        ax.imshow(Image.open(first[cls].path).convert("RGB"))
        ax.set_title(f"{cls}: {CLASS_NAMES[cls][:18]}", fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "samples.png"), dpi=150)
    plt.close(fig)

    # original image sizes (read from ROI-free width via file)
    widths = [Image.open(s.path).size[0] for s in samples[::20]]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(widths, bins=30, color="#55A868")
    ax.set(xlabel="Image width (px)", ylabel="count", title="Original image sizes (5% sample)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "image_sizes.png"), dpi=150)
    plt.close(fig)
    roi_w = [s.roi[2] - s.roi[0] for s in samples]
    print(f"Original width (5% sample): min {min(widths)}, median {int(np.median(widths))}, "
          f"max {max(widths)} px | ROI width: median {int(np.median(roi_w))} px")
    print(f"Figures saved to {OUT}/")

    if args.stats:
        mean, std = channel_stats(train, cfg["crop_roi"])   # training split only
        print("GTSRB_MEAN =", tuple(round(v, 4) for v in mean))
        print("GTSRB_STD  =", tuple(round(v, 4) for v in std))


if __name__ == "__main__":
    main()

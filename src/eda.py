"""
Exploratory data analysis for the report's "Data set" section.

Usage:
    python -m src.eda
Outputs (report/figures/):
    class_distribution.png   number of training images per class (shows imbalance)
    samples.png              one example image per class
    image_sizes.png          distribution of original image sizes
    + statistics printed to the console
"""
import os
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from src.dataset import load_train_samples, split_by_track
from src.utils import CLASS_NAMES, NUM_CLASSES, load_config

OUT = os.path.join("report", "figures")


def main():
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
    print(f"Figures saved to {OUT}/")


if __name__ == "__main__":
    main()

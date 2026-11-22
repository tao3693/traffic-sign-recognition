"""
Final evaluation on the official GTSRB test set (use ONLY after tuning is finished).

Usage:
    python -m src.evaluate --weights weights/latest_cnn.pt
Outputs (results/):
    <name>_test_report.txt      accuracy + per-class precision / recall / F1
    <name>_confusion.png        43x43 confusion matrix
    <name>_errors.png           examples of misclassified signs
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.dataset import GTSRB_MEAN, GTSRB_STD, build_dataloaders
from src.models import build_model
from src.utils import CLASS_NAMES, get_device


def load_checkpoint(path: str, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ckpt["config"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt["config"]


@torch.no_grad()
def predict_loader(model, loader, device):
    preds, labels, images = [], [], []
    for x, y in loader:
        preds.append(model(x.to(device)).argmax(1).cpu())
        labels.append(y)
        images.append(x)
    return torch.cat(preds).numpy(), torch.cat(labels).numpy(), torch.cat(images)


def denormalize(img: torch.Tensor) -> np.ndarray:
    mean, std = torch.tensor(GTSRB_MEAN)[:, None, None], torch.tensor(GTSRB_STD)[:, None, None]
    return (img * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()


def plot_confusion(cm, path):
    fig, ax = plt.subplots(figsize=(12, 10))
    norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    im = ax.imshow(norm, cmap="Blues")
    ax.set(xlabel="Predicted class", ylabel="True class", title="Normalised confusion matrix")
    ax.set_xticks(range(len(cm)))
    ax.set_yticks(range(len(cm)))
    ax.tick_params(labelsize=6)
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_errors(images, preds, labels, path, n=16):
    wrong = np.where(preds != labels)[0][:n]
    if len(wrong) == 0:
        return
    cols = 4
    rows = int(np.ceil(len(wrong) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    for ax in np.array(axes).ravel():
        ax.axis("off")
    for ax, i in zip(np.array(axes).ravel(), wrong):
        ax.imshow(denormalize(images[i]))
        ax.set_title(f"T:{labels[i]} P:{preds[i]}", fontsize=9, color="red")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Evaluate on the GTSRB test set")
    parser.add_argument("--weights", default="weights/latest_cnn.pt")
    args = parser.parse_args()

    device = get_device()
    model, cfg = load_checkpoint(args.weights, device)
    _, _, test_loader = build_dataloaders(cfg)
    if test_loader is None:
        raise FileNotFoundError("GT-final_test.csv not found - run download_data.py first.")

    preds, labels, images = predict_loader(model, test_loader, device)
    acc = accuracy_score(labels, preds)
    report = classification_report(labels, preds, labels=list(range(len(CLASS_NAMES))),
                                   target_names=CLASS_NAMES, digits=4, zero_division=0)
    name = os.path.splitext(os.path.basename(args.weights))[0]
    out = cfg["results_dir"]
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, f"{name}_test_report.txt"), "w", encoding="utf-8") as f:
        f.write(f"Test accuracy: {acc:.4f}\n\n{report}")
    plot_confusion(confusion_matrix(labels, preds, labels=list(range(len(CLASS_NAMES)))),
                   os.path.join(out, f"{name}_confusion.png"))
    plot_errors(images, preds, labels, os.path.join(out, f"{name}_errors.png"))
    print(f"Test accuracy: {acc:.4f}  (report saved in {out}/)")


if __name__ == "__main__":
    main()

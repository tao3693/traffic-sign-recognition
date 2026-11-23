"""
Final evaluation on the official GTSRB test set (use ONLY after tuning is finished).

Usage:
    python -m src.evaluate --weights weights/best_cnn.pt
Outputs (results/):
    <name>_test_report.txt      accuracy, top-3 accuracy, macro-F1, parameters, inference
                                time, most-confused class pairs, per-class P / R / F1
    <name>_confusion.png        43x43 normalised confusion matrix
    <name>_errors.png           examples of misclassified signs (with class names)
    test_results.csv            one summary row per evaluated model (for the report table)
"""
import argparse
import csv
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from src.dataset import build_test_loader, norm_stats
from src.models import build_model
from src.utils import CLASS_NAMES, count_parameters, get_device

N_CLASSES = len(CLASS_NAMES)


def load_checkpoint(path: str, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    cfg["pretrained"] = False           # weights come from the checkpoint, no download needed
    model = build_model(cfg).to(device)
    # older checkpoints stored a duplicate "gradcam_layer.*" copy of layer4 -> drop it
    state = {k: v for k, v in ckpt["state_dict"].items() if not k.startswith("gradcam_layer.")}
    model.load_state_dict(state)
    model.eval()
    return model, cfg


@torch.no_grad()
def predict_loader(model, loader, device, keep_errors=16, topk=3):
    """Return predictions, labels, top-k hit flags, the first `keep_errors` misclassified
    images (only those are kept in memory) and the average inference time per image."""
    preds, labels, topk_hit, errors = [], [], [], []
    seconds, n = 0.0, 0
    for x, y in loader:
        x = x.to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        logits = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        seconds += time.perf_counter() - t0
        n += x.size(0)

        p = logits.argmax(1).cpu()
        top = logits.topk(topk, dim=1).indices.cpu()
        preds.append(p)
        labels.append(y)
        topk_hit.append((top == y[:, None]).any(1))
        for i in torch.nonzero(p != y).flatten().tolist():
            if len(errors) < keep_errors:
                errors.append((x[i].cpu(), y[i].item(), p[i].item()))
    return (torch.cat(preds).numpy(), torch.cat(labels).numpy(),
            torch.cat(topk_hit).numpy(), errors, 1000 * seconds / max(n, 1))


def denormalize(img: torch.Tensor, cfg: dict) -> np.ndarray:
    mean, std = norm_stats(cfg)
    mean, std = torch.tensor(mean)[:, None, None], torch.tensor(std)[:, None, None]
    return (img * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()


def top_confusions(cm, k=10):
    """Most frequent (true, predicted) pairs off the diagonal."""
    off = cm.copy()
    np.fill_diagonal(off, 0)
    idx = np.dstack(np.unravel_index(np.argsort(-off, axis=None), off.shape))[0][:k]
    return [(int(t), int(p), int(off[t, p])) for t, p in idx if off[t, p] > 0]


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


def plot_errors(errors, cfg, path):
    if not errors:
        return
    cols = 4
    rows = int(np.ceil(len(errors) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.4))
    for ax in np.array(axes).ravel():
        ax.axis("off")
    for ax, (img, true, pred) in zip(np.array(axes).ravel(), errors):
        ax.imshow(denormalize(img, cfg))
        ax.set_title(f"T: {CLASS_NAMES[true][:24]}\nP: {CLASS_NAMES[pred][:24]}",
                     fontsize=7, color="red")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def append_row(path, row):
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Evaluate on the GTSRB test set")
    parser.add_argument("--weights", default="weights/best_cnn.pt")
    args = parser.parse_args()

    device = get_device()
    model, cfg = load_checkpoint(args.weights, device)
    test_loader = build_test_loader(cfg)
    if test_loader is None:
        raise FileNotFoundError("GT-final_test.csv not found - run download_data.py first.")

    preds, labels, top3, errors, ms = predict_loader(model, test_loader, device)
    all_cls = list(range(N_CLASSES))
    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, labels=all_cls, average="macro", zero_division=0)
    cm = confusion_matrix(labels, preds, labels=all_cls)
    n_params, _ = count_parameters(model)
    report = classification_report(labels, preds, labels=all_cls, target_names=CLASS_NAMES,
                                   digits=4, zero_division=0)

    name = os.path.splitext(os.path.basename(args.weights))[0]
    out = cfg["results_dir"]
    os.makedirs(out, exist_ok=True)
    lines = [f"Model: {name}  ({cfg['model']}, img_size {cfg['img_size']})",
             f"Test images: {len(labels)}",
             f"Test accuracy:  {acc:.4f}",
             f"Top-3 accuracy: {top3.mean():.4f}",
             f"Macro F1:       {macro_f1:.4f}",
             f"Parameters:     {n_params:,}",
             f"Inference:      {ms:.3f} ms / image on {device} (batch {cfg['batch_size']})",
             "", "Most confused pairs (true -> predicted : count):"]
    lines += [f"  {t:2d} {CLASS_NAMES[t]:40s} -> {p:2d} {CLASS_NAMES[p]:40s}: {c}"
              for t, p, c in top_confusions(cm)]
    with open(os.path.join(out, f"{name}_test_report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n\n" + report)
    plot_confusion(cm, os.path.join(out, f"{name}_confusion.png"))
    plot_errors(errors, cfg, os.path.join(out, f"{name}_errors.png"))
    append_row(os.path.join(out, "test_results.csv"),
               dict(weights=name, model=cfg["model"], img_size=cfg["img_size"],
                    test_acc=round(acc, 4), top3_acc=round(float(top3.mean()), 4),
                    macro_f1=round(macro_f1, 4), params=n_params,
                    ms_per_image=round(ms, 3), device=str(device)))
    print("\n".join(lines[:7]))
    print(f"Report saved in {out}/")


if __name__ == "__main__":
    main()

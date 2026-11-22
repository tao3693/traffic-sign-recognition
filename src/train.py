"""
Training loop with loss tracking, gradient clipping / gradient-norm monitoring,
early stopping and experiment logging.

Usage examples:
    python -m src.train                                   # use config.yaml
    python -m src.train --model resnet18 --lr 0.0001 --img_size 64
    python -m src.train --lr 0.01 --optimizer sgd --tag sgd_test
Outputs:
    weights/<experiment>.pt              best checkpoint (by val accuracy)
    weights/latest_<model>.pt            copy used by main.py
    results/<experiment>_history.csv     per-epoch loss / accuracy / grad norm
    results/<experiment>_curves.png      loss & accuracy curves
    results/experiments.csv              one summary row per run (for the report)
"""
import argparse
import csv
import os
import shutil
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm

from src.dataset import build_dataloaders
from src.models import build_model
from src.utils import add_common_args, ensure_dirs, experiment_name, get_device, load_config, set_seed


def build_optimizer(cfg, params):
    params = [p for p in params if p.requires_grad]
    if cfg["optimizer"] == "sgd":
        return torch.optim.SGD(params, lr=cfg["lr"], momentum=0.9,
                               weight_decay=cfg["weight_decay"])
    return torch.optim.Adam(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])


def train_one_epoch(model, loader, criterion, optimizer, device, grad_clip):
    model.train()
    total_loss, correct, n, grad_norms = 0.0, 0, 0, []
    for x, y in tqdm(loader, desc="train", leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()                                    # compute gradients
        max_norm = grad_clip if grad_clip > 0 else float("inf")
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        grad_norms.append(norm.item())                     # monitor gradient size
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        n += x.size(0)
    return total_loss / n, correct / n, sum(grad_norms) / len(grad_norms)


@torch.no_grad()
def evaluate_loss_acc(model, loader, criterion, device):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += criterion(logits, y).item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        n += x.size(0)
    return total_loss / n, correct / n


def plot_curves(history, path):
    epochs = [h["epoch"] for h in history]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(epochs, [h["train_loss"] for h in history], label="train")
    ax1.plot(epochs, [h["val_loss"] for h in history], label="val")
    ax1.set(title="Loss", xlabel="epoch", ylabel="cross-entropy")
    ax2.plot(epochs, [h["train_acc"] for h in history], label="train")
    ax2.plot(epochs, [h["val_acc"] for h in history], label="val")
    ax2.set(title="Accuracy", xlabel="epoch", ylabel="accuracy")
    for ax in (ax1, ax2):
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_csv(path, rows, append=False):
    exists = os.path.exists(path)
    with open(path, "a" if append else "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if not (append and exists):
            writer.writeheader()
        writer.writerows(rows)


def train(cfg: dict) -> str:
    set_seed(cfg["seed"])
    ensure_dirs(cfg)
    device = get_device()
    name = experiment_name(cfg)
    print(f"Experiment: {name} | device: {device}")

    train_loader, val_loader, _ = build_dataloaders(cfg)
    model = build_model(cfg).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])
    optimizer = build_optimizer(cfg, model.parameters())
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",
                                                           factor=0.5, patience=2)

    ckpt_path = os.path.join(cfg["weights_dir"], f"{name}.pt")
    history, best_acc, best_epoch, bad_epochs = [], 0.0, 0, 0
    start = time.time()
    for epoch in range(1, cfg["epochs"] + 1):
        tr_loss, tr_acc, g_norm = train_one_epoch(model, train_loader, criterion,
                                                  optimizer, device, cfg["grad_clip"])
        va_loss, va_acc = evaluate_loss_acc(model, val_loader, criterion, device)
        scheduler.step(va_loss)
        lr_now = optimizer.param_groups[0]["lr"]
        history.append(dict(epoch=epoch, train_loss=round(tr_loss, 4), train_acc=round(tr_acc, 4),
                            val_loss=round(va_loss, 4), val_acc=round(va_acc, 4),
                            grad_norm=round(g_norm, 4), lr=lr_now))
        print(f"[{epoch:02d}] train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.4f} | grad {g_norm:.2f} | lr {lr_now:.1e}")

        if va_acc > best_acc:
            best_acc, best_epoch, bad_epochs = va_acc, epoch, 0
            torch.save({"model": cfg["model"], "config": cfg,
                        "state_dict": model.state_dict()}, ckpt_path)
        else:
            bad_epochs += 1
            if bad_epochs >= cfg["early_stop_patience"]:
                print("Early stopping.")
                break

    shutil.copy(ckpt_path, os.path.join(cfg["weights_dir"], f"latest_{cfg['model']}.pt"))
    write_csv(os.path.join(cfg["results_dir"], f"{name}_history.csv"), history)
    plot_curves(history, os.path.join(cfg["results_dir"], f"{name}_curves.png"))
    summary = dict(experiment=name, model=cfg["model"], lr=cfg["lr"], batch_size=cfg["batch_size"],
                   optimizer=cfg["optimizer"], img_size=cfg["img_size"], augment=cfg["augment"],
                   balanced_sampling=cfg["balanced_sampling"],
                   label_smoothing=cfg["label_smoothing"],
                   freeze_backbone=cfg.get("freeze_backbone", False),
                   best_val_acc=round(best_acc, 4), best_epoch=best_epoch,
                   train_minutes=round((time.time() - start) / 60, 1))
    write_csv(os.path.join(cfg["results_dir"], "experiments.csv"), [summary], append=True)
    print(f"Best val acc {best_acc:.4f} at epoch {best_epoch}. Saved {ckpt_path}")
    return ckpt_path


def main():
    parser = add_common_args(argparse.ArgumentParser(description="Train a traffic sign classifier"))
    args = vars(parser.parse_args())
    cfg = load_config(args.pop("config"), args)
    train(cfg)


if __name__ == "__main__":
    main()

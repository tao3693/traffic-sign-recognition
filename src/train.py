"""
Training loop with loss tracking, gradient clipping / gradient-norm monitoring,
early stopping and experiment logging.

Usage examples:
    python -m src.train                                   # use config.yaml
    python -m src.train --model resnet18 --lr 0.0001 --img_size 64
    python -m src.train --lr 0.01 --optimizer sgd --tag sgd_test
The experiment name = settings + start time (e.g. cnn_img48_adam_lr0.001_bs64_1124-1530),
so runs on different machines never get the same name.

Outputs:
    weights/<experiment>.pt              best checkpoint of this run (by val accuracy)
    weights/best_<model>.pt              best checkpoint of this model ON THIS MACHINE, among
                                         runs with the same validation split (local helper
                                         for main.py; the team picks the final model by name)
    results/<experiment>_config.yaml     full configuration of the run (reproducibility)
    results/<experiment>_history.csv     per-epoch loss / accuracy / grad norm
    results/<experiment>_curves.png      loss & accuracy curves
    results/runs/<experiment>.json       summary of the run (commit this file)
    results/experiments.csv              table of all runs, regenerated from results/runs/
"""
import argparse
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import yaml
from matplotlib.ticker import MaxNLocator
from tqdm import tqdm

from src.dataset import build_dataloaders
from src.models import build_model, set_train_mode
from src.summary import RUNS, rebuild, save_record, write_rows
from src.utils import (add_common_args, count_parameters, ensure_dirs, environment_info,
                       experiment_name, get_device, load_config, set_seed, unique_name)


def build_optimizer(cfg, params):
    params = [p for p in params if p.requires_grad]
    if cfg["optimizer"] == "sgd":
        return torch.optim.SGD(params, lr=cfg["lr"], momentum=cfg.get("momentum", 0.9),
                               weight_decay=cfg["weight_decay"])
    return torch.optim.Adam(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])


def train_one_epoch(model, loader, criterion, optimizer, device, grad_clip, scaler):
    set_train_mode(model)        # frozen BatchNorm layers stay in eval mode
    use_amp = scaler.is_enabled()
    total_loss, correct, n, grad_norms = 0.0, 0, 0, []
    for x, y in tqdm(loader, desc="train", leave=False):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(x)
            loss = criterion(logits, y)
        scaler.scale(loss).backward()                      # compute gradients
        scaler.unscale_(optimizer)                         # real gradient values before clipping
        max_norm = grad_clip if grad_clip > 0 else float("inf")
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        if torch.isfinite(norm):                           # monitor gradient size; AMP may skip
            grad_norms.append(norm.item())                 # steps with inf/nan gradients
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        n += x.size(0)
    avg_norm = sum(grad_norms) / len(grad_norms) if grad_norms else float("nan")
    return total_loss / n, correct / n, avg_norm


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
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))   # whole epochs only
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def validation_key(cfg: dict) -> dict:
    """Settings that define the validation set. Val accuracies are only comparable when
    these are equal (a different seed / ratio / crop gives a different validation set)."""
    return dict(val_ratio=cfg["val_ratio"], seed=cfg["seed"], crop_roi=cfg["crop_roi"])


def update_best(ckpt_path: str, best_acc: float, cfg: dict) -> bool:
    """Copy this run's checkpoint to best_<model>.pt if it beats the stored one on the
    SAME validation set."""
    best_path = os.path.join(cfg["weights_dir"], f"best_{cfg['model']}.pt")
    if os.path.exists(best_path):
        old = torch.load(best_path, map_location="cpu", weights_only=False)
        if old.get("val_key") != validation_key(cfg):
            print(f"Note: {best_path} was selected on a different validation split "
                  f"({old.get('val_key')}); not compared, not replaced.")
            return False
        if old.get("best_val_acc", 0.0) >= best_acc:
            return False
    torch.save(torch.load(ckpt_path, map_location="cpu", weights_only=False), best_path)
    return True


def train(cfg: dict) -> str:
    set_seed(cfg["seed"], cfg.get("deterministic", True))
    ensure_dirs(cfg)
    device = get_device()
    started = time.strftime("%Y-%m-%d %H:%M")
    name = unique_name(f"{experiment_name(cfg)}_{time.strftime('%m%d-%H%M')}", cfg)
    if device.type == "cuda":
        # cudnn.benchmark (fastest conv algorithms) is set by set_seed: on when
        # deterministic=false in config.yaml, off for reproducible runs
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    use_amp = cfg.get("amp", True) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    print(f"Experiment: {name} | device: {device} | img_size: {cfg['img_size']} | "
          f"mixed precision: {use_amp}")

    train_loader, val_loader = build_dataloaders(cfg)
    model = build_model(cfg).to(device)
    n_params, n_trainable = count_parameters(model)
    print(f"Parameters: {n_params:,} total, {n_trainable:,} trainable")
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])
    optimizer = build_optimizer(cfg, model.parameters())
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",
                                                           factor=0.5, patience=2)

    ckpt_path = os.path.join(cfg["weights_dir"], f"{name}.pt")
    with open(os.path.join(cfg["results_dir"], f"{name}_config.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(dict(cfg, experiment=name), f, sort_keys=False)
    history, best_acc, best_epoch, bad_epochs = [], 0.0, 0, 0
    start = time.time()
    for epoch in range(1, cfg["epochs"] + 1):
        tr_loss, tr_acc, g_norm = train_one_epoch(model, train_loader, criterion,
                                                  optimizer, device, cfg["grad_clip"], scaler)
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
            torch.save({"model": cfg["model"], "config": cfg, "experiment": name,
                        "best_val_acc": best_acc, "epoch": epoch,
                        "val_key": validation_key(cfg),
                        "state_dict": model.state_dict()}, ckpt_path)
        else:
            bad_epochs += 1
            if bad_epochs >= cfg["early_stop_patience"]:
                print("Early stopping.")
                break

    is_best = update_best(ckpt_path, best_acc, cfg)
    write_rows(os.path.join(cfg["results_dir"], f"{name}_history.csv"), history)
    plot_curves(history, os.path.join(cfg["results_dir"], f"{name}_curves.png"))
    summary = dict(experiment=name, started=started, model=cfg["model"], lr=cfg["lr"], batch_size=cfg["batch_size"],
                   optimizer=cfg["optimizer"],
                   momentum=cfg.get("momentum", 0.9) if cfg["optimizer"] == "sgd" else "",
                   weight_decay=cfg["weight_decay"], epochs=cfg["epochs"],
                   early_stop_patience=cfg["early_stop_patience"], grad_clip=cfg["grad_clip"],
                   val_ratio=cfg["val_ratio"], crop_roi=cfg["crop_roi"],
                   img_size=cfg["img_size"], augment=cfg["augment"],
                   balanced_sampling=cfg["balanced_sampling"],
                   label_smoothing=cfg["label_smoothing"],
                   pretrained=cfg.get("pretrained", True) if cfg["model"] == "resnet18" else "",
                   freeze_backbone=cfg.get("freeze_backbone", False),
                   small_input=cfg.get("small_input", False),
                   normalize=cfg.get("normalize", "gtsrb"), dropout=cfg.get("dropout"),
                   amp=use_amp, deterministic=cfg.get("deterministic", True),
                   seed=cfg["seed"], params=n_params, trainable_params=n_trainable,
                   best_val_acc=round(best_acc, 4), best_epoch=best_epoch,
                   epochs_run=len(history), train_minutes=round((time.time() - start) / 60, 1),
                   **environment_info(device))
    save_record(cfg["results_dir"], RUNS, name, summary)
    rebuild(cfg["results_dir"], RUNS)
    print(f"Best val acc {best_acc:.4f} at epoch {best_epoch}. Saved {ckpt_path}")
    if is_best:
        print(f"New overall best for {cfg['model']} -> {cfg['weights_dir']}/best_{cfg['model']}.pt")
    return ckpt_path


def main():
    parser = add_common_args(argparse.ArgumentParser(description="Train a traffic sign classifier"))
    args = vars(parser.parse_args())
    cfg = load_config(args.pop("config"), args)
    train(cfg)


if __name__ == "__main__":
    main()

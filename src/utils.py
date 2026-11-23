"""Shared helpers: config loading, seeding, device selection, class names."""
import argparse
import os
import platform
import random
from typing import Optional

import numpy as np
import torch
import yaml

CLASS_NAMES = [
    "Speed limit (20km/h)", "Speed limit (30km/h)", "Speed limit (50km/h)",
    "Speed limit (60km/h)", "Speed limit (70km/h)", "Speed limit (80km/h)",
    "End of speed limit (80km/h)", "Speed limit (100km/h)", "Speed limit (120km/h)",
    "No passing", "No passing for vehicles over 3.5t", "Right-of-way at next intersection",
    "Priority road", "Yield", "Stop", "No vehicles", "Vehicles over 3.5t prohibited",
    "No entry", "General caution", "Dangerous curve to the left",
    "Dangerous curve to the right", "Double curve", "Bumpy road", "Slippery road",
    "Road narrows on the right", "Road work", "Traffic signals", "Pedestrians",
    "Children crossing", "Bicycles crossing", "Beware of ice/snow",
    "Wild animals crossing", "End of all speed and passing limits", "Turn right ahead",
    "Turn left ahead", "Ahead only", "Go straight or right", "Go straight or left",
    "Keep right", "Keep left", "Roundabout mandatory", "End of no passing",
    "End of no passing by vehicles over 3.5t",
]
NUM_CLASSES = len(CLASS_NAMES)  # 43


def str2bool(s: str) -> bool:
    if s.lower() in ("true", "1", "yes"):
        return True
    if s.lower() in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError(f"expected true/false, got {s!r}")


def load_config(path: str = "config.yaml", overrides: Optional[dict] = None) -> dict:
    """Load YAML config and apply non-None command-line overrides.

    ResNet18 needs larger inputs than the CNN (it down-samples 32x), so when the model is
    resnet18 and --img_size was NOT given explicitly, `resnet_img_size` is used instead.
    """
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    overrides = {k: v for k, v in (overrides or {}).items() if v is not None}
    cfg.update(overrides)
    if cfg["model"] == "resnet18" and "img_size" not in overrides:
        cfg["img_size"] = cfg.get("resnet_img_size", 96)
    return cfg


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """CLI flags that override config.yaml (used by main.py and src/train.py)."""
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--model", choices=["cnn", "resnet18"])
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--optimizer", choices=["adam", "sgd"])
    parser.add_argument("--img_size", type=int)
    parser.add_argument("--label_smoothing", type=float)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--normalize", choices=["gtsrb", "imagenet"])
    parser.add_argument("--freeze_backbone", type=str2bool)
    parser.add_argument("--small_input", type=str2bool)
    parser.add_argument("--pretrained", type=str2bool)
    parser.add_argument("--augment", type=str2bool)
    parser.add_argument("--balanced_sampling", type=str2bool)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--tag", default="", help="name suffix for this experiment")
    return parser


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Make experiments reproducible (also see seed_worker / DataLoader generator)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def seed_worker(worker_id: int) -> None:
    """DataLoader worker_init_fn: give every worker a reproducible numpy/random seed."""
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def environment_info(device: torch.device) -> dict:
    """Software / hardware versions, logged with every experiment for the report."""
    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else device.type
    return dict(python=platform.python_version(), torch=torch.__version__, device=gpu)


def count_parameters(model) -> tuple:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def experiment_name(cfg: dict) -> str:
    """Name that encodes every setting we vary in experiments, so runs never collide.

    Settings at their usual value are left out to keep names short, e.g.
        cnn_img48_adam_lr0.001_bs64
        cnn_img48_adam_lr0.001_bs64_noaug_ls0.1_baseline
        resnet18_img96_adam_lr0.0001_bs64_frozen
    """
    name = (f"{cfg['model']}_img{cfg['img_size']}_{cfg['optimizer']}"
            f"_lr{cfg['lr']}_bs{cfg['batch_size']}")
    if not cfg.get("augment", True):
        name += "_noaug"
    if not cfg.get("balanced_sampling", True):
        name += "_nobal"
    if cfg.get("label_smoothing", 0):
        name += f"_ls{cfg['label_smoothing']}"
    if cfg.get("dropout", 0.3) != 0.3:
        name += f"_do{cfg['dropout']}"
    if cfg.get("normalize", "gtsrb") != "gtsrb":
        name += f"_{cfg['normalize']}norm"
    if cfg.get("seed", 42) != 42:
        name += f"_seed{cfg['seed']}"
    if cfg["model"] == "resnet18":
        if cfg.get("freeze_backbone"):
            name += "_frozen"
        if cfg.get("small_input"):
            name += "_small"
        if not cfg.get("pretrained", True):
            name += "_scratch"
    if cfg.get("tag"):
        name += f"_{cfg['tag']}"
    return name


def unique_name(name: str, weights_dir: str) -> str:
    """Append _run2, _run3 ... if a checkpoint with this name already exists."""
    candidate, i = name, 2
    while os.path.exists(os.path.join(weights_dir, f"{candidate}.pt")):
        candidate, i = f"{name}_run{i}", i + 1
    return candidate


def ensure_dirs(cfg: dict) -> None:
    os.makedirs(cfg["results_dir"], exist_ok=True)
    os.makedirs(cfg["weights_dir"], exist_ok=True)

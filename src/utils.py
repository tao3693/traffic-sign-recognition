"""Shared helpers: config loading, seeding, device selection, class names."""
import argparse
import os
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


def load_config(path: str = "config.yaml", overrides: Optional[dict] = None) -> dict:
    """Load YAML config and apply non-None command-line overrides."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for key, value in (overrides or {}).items():
        if value is not None:
            cfg[key] = value
    return cfg


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """CLI flags that override config.yaml (used by train / evaluate)."""
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--model", choices=["cnn", "resnet18"])
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--optimizer", choices=["adam", "sgd"])
    parser.add_argument("--img_size", type=int)
    parser.add_argument("--label_smoothing", type=float)
    parser.add_argument("--freeze_backbone", type=lambda s: s.lower() == "true")
    parser.add_argument("--augment", type=lambda s: s.lower() == "true")
    parser.add_argument("--balanced_sampling", type=lambda s: s.lower() == "true")
    parser.add_argument("--tag", default="", help="name suffix for this experiment")
    return parser


def set_seed(seed: int) -> None:
    """Make experiments reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def experiment_name(cfg: dict) -> str:
    name = f"{cfg['model']}_lr{cfg['lr']}_bs{cfg['batch_size']}_{cfg['optimizer']}"
    if cfg.get("tag"):
        name += f"_{cfg['tag']}"
    return name


def ensure_dirs(cfg: dict) -> None:
    os.makedirs(cfg["results_dir"], exist_ok=True)
    os.makedirs(cfg["weights_dir"], exist_ok=True)

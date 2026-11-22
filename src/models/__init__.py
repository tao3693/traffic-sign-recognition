from src.models.cnn import TrafficSignCNN
from src.models.resnet import build_resnet18
from src.utils import NUM_CLASSES


def build_model(cfg: dict):
    if cfg["model"] == "cnn":
        return TrafficSignCNN(NUM_CLASSES, cfg.get("dropout", 0.3))
    if cfg["model"] == "resnet18":
        return build_resnet18(NUM_CLASSES, cfg.get("pretrained", True),
                              cfg.get("freeze_backbone", False))
    raise ValueError(f"Unknown model: {cfg['model']}")

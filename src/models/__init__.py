import torch.nn as nn

from src.models.cnn import TrafficSignCNN
from src.models.resnet import build_resnet18
from src.utils import NUM_CLASSES


def build_model(cfg: dict):
    if cfg["model"] == "cnn":
        return TrafficSignCNN(NUM_CLASSES, cfg.get("dropout", 0.3))
    if cfg["model"] == "resnet18":
        return build_resnet18(NUM_CLASSES, cfg.get("pretrained", True),
                              cfg.get("freeze_backbone", False), cfg.get("small_input", False))
    raise ValueError(f"Unknown model: {cfg['model']}")


def set_train_mode(model) -> None:
    """model.train(), but BatchNorm layers whose parameters are frozen stay in eval mode.

    Otherwise a "frozen" backbone would still change: in train mode BatchNorm keeps
    updating its running mean / variance with GTSRB batches even if requires_grad=False.
    """
    model.train()
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            params = list(m.parameters(recurse=False))
            if params and not any(p.requires_grad for p in params):
                m.eval()


def get_gradcam_layer(model, cfg: dict, layer: str = None):
    """Layer whose activations / gradients Grad-CAM uses.

    CNN: output of the last ReLU of the last conv block (12x12 for a 48x48 input).
    ResNet18 down-samples 32x, so layer4 is only 3x3 for a 96x96 input -> heat-map too coarse.
    `auto` therefore picks layer4 when small_input=True (8x down-sampling, 12x12 at 96)
    and layer3 otherwise (16x down-sampling, 6x6 at 96). Override with layer="layer4".
    """
    if cfg["model"] == "cnn":
        return model.gradcam_layer
    layer = layer or cfg.get("gradcam_layer", "auto")
    if layer == "auto":
        layer = "layer4" if cfg.get("small_input", False) else "layer3"
    if layer not in ("layer3", "layer4"):
        raise ValueError(f"gradcam_layer must be auto, layer3 or layer4, got {layer}")
    return getattr(model, layer)[-1]

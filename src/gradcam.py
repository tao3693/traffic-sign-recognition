"""
Grad-CAM: shows which image regions the model relies on, using the GRADIENT of the
predicted class score with respect to a convolutional feature map.

Reference: R. R. Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
via Gradient-based Localization", ICCV 2017. https://arxiv.org/abs/1610.02391
(Implementation written for this project.)

Target layer (see src/models/__init__.py::get_gradcam_layer):
    cnn       last ReLU of the last conv block         -> 12x12 for 48x48 input
    resnet18  layer3 (default) or layer4 (small_input)  -> 6x6 / 12x12 for 96x96 input
The heat-map resolution is input_size / down-sampling factor, which is why we do not use
layer4 of a standard ResNet18 (only 3x3 at 96x96).

Usage:
    python -m src.gradcam --weights weights/best_cnn.pt --image path/to/sign.png
    python -m src.gradcam --weights weights/best_resnet18.pt --image sign.png --layer layer4
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch.nn.functional as F
from PIL import Image

from src.dataset import build_transforms
from src.models import get_gradcam_layer
from src.utils import CLASS_NAMES, get_device


class GradCAM:
    """Use as a context manager so the hooks are always removed:

        with GradCAM(model, layer) as cam:
            heatmap, cls, prob = cam(x)
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        self._handles = [
            target_layer.register_forward_hook(self._save_activation),
            target_layer.register_full_backward_hook(self._save_gradient),
        ]

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.remove()

    def __call__(self, x, class_idx=None):
        self.model.eval()
        x = x.requires_grad_(True)       # needed when the backbone is frozen
        logits = self.model(x)
        if class_idx is None:
            class_idx = logits.argmax(1).item()
        self.model.zero_grad()
        logits[0, class_idx].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)   # global-avg of gradients
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam_size = tuple(cam.shape[2:])
        cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        prob = F.softmax(logits, 1)[0, class_idx].item()
        self.model.zero_grad()
        return cam[0, 0].detach().cpu().numpy(), class_idx, prob, cam_size


def run_gradcam(model, cfg, image_path, out_path="results/gradcam.png", layer=None):
    device = next(model.parameters()).device
    img = Image.open(image_path).convert("RGB")
    tf = build_transforms(cfg["img_size"], train=False, normalize=cfg.get("normalize", "gtsrb"))
    x = tf(img).unsqueeze(0).to(device)
    with GradCAM(model, get_gradcam_layer(model, cfg, layer)) as gc:
        cam, cls, prob, cam_size = gc(x)

    shown = img.resize((cfg["img_size"], cfg["img_size"]))
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))
    axes[0].imshow(shown)
    axes[0].set_title("Input")
    axes[1].imshow(shown)
    axes[1].imshow(cam, cmap="jet", alpha=0.45)
    axes[1].set_title(f"{CLASS_NAMES[cls]} ({prob:.1%})\nCAM {cam_size[0]}x{cam_size[1]}",
                      fontsize=9)
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main():
    from src.evaluate import load_checkpoint
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="weights/best_cnn.pt")
    parser.add_argument("--image", required=True)
    parser.add_argument("--out", default="results/gradcam.png")
    parser.add_argument("--layer", choices=["auto", "layer3", "layer4"],
                        help="resnet18 only: target layer (default from config)")
    args = parser.parse_args()
    model, cfg = load_checkpoint(args.weights, get_device())
    print("Saved", run_gradcam(model, cfg, args.image, args.out, args.layer))


if __name__ == "__main__":
    main()

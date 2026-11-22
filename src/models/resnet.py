"""ResNet18 transfer learning (ImageNet pre-trained)."""
import torch.nn as nn
from torchvision import models


def build_resnet18(num_classes: int = 43, pretrained: bool = True,
                   freeze_backbone: bool = False) -> nn.Module:
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)

    if freeze_backbone:                       # feature extraction: train head only
        for p in model.parameters():
            p.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, num_classes)  # new head is trainable
    model.gradcam_layer = model.layer4[-1]
    return model

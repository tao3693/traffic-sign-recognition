"""ResNet18 transfer learning (ImageNet pre-trained)."""
import torch.nn as nn
from torchvision import models


def build_resnet18(num_classes: int = 43, pretrained: bool = True,
                   freeze_backbone: bool = False, small_input: bool = False) -> nn.Module:
    """
    small_input=True adapts ResNet18 to small images (as in CIFAR-style ResNets):
    the 7x7 stride-2 conv1 is replaced by a 3x3 stride-1 conv and the max-pool is removed,
    so the network down-samples 8x instead of 32x. conv1 and its BatchNorm (bn1) are then
    trained from scratch; all other layers keep their ImageNet weights.

    freeze_backbone=True freezes all pre-trained layers; see set_train_mode() for why their
    BatchNorm layers are also kept in eval mode during training.
    """
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)

    if freeze_backbone:                       # feature extraction: train head only
        for p in model.parameters():
            p.requires_grad = False

    if small_input:                           # new layers are trainable
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.bn1 = nn.BatchNorm2d(64)        # new conv -> new statistics
        model.maxpool = nn.Identity()

    model.fc = nn.Linear(model.fc.in_features, num_classes)  # new head is trainable
    return model

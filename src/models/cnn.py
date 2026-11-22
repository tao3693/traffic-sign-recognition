"""Custom CNN built from scratch (baseline model)."""
import torch.nn as nn


def conv_block(in_ch: int, out_ch: int, dropout: float) -> nn.Sequential:
    """Two 3x3 conv layers + BatchNorm + ReLU, then 2x2 max-pool."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.Dropout(dropout),
    )


class TrafficSignCNN(nn.Module):
    def __init__(self, num_classes: int = 43, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(3, 32, dropout / 2),
            conv_block(32, 64, dropout / 2),
            conv_block(64, 128, dropout),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)          # works for any input size
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))

    @property
    def gradcam_layer(self):
        """Last conv layer, used by Grad-CAM."""
        return self.features[-1][3]

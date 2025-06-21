from __future__ import annotations
from torchvision import models
import torch.nn as nn

__all__ = ["WaveletResNet"]

class WaveletResNet(nn.Module):
    """ResNet-18 + droput head; 3 channels (cA,cH,cV)."""
    def __init__(self, n_cls: int = 2):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        for p in list(backbone.parameters())[:4*2]:
            p.requires_grad = False

        backbone.fc = nn.Sequential(
            nn.Dropout(0.55),
            nn.Linear(backbone.fc.in_features, n_cls)
        )
        self.net = backbone

    def forward(self, x):
        return self.net(x)

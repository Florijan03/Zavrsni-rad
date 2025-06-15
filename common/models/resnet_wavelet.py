from __future__ import annotations
import torch.nn as nn
from torchvision import models

__all__ = ["WaveletResNet"]

class WaveletResNet(nn.Module):
    def __init__(self, n_cls: int = 2):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # --- conv1: 4 channels ---------------------------------
        w = backbone.conv1.weight          # [64,3,7,7]
        new_w = w.mean(dim=1, keepdim=True).repeat(1, 4, 1, 1) / 3.0
        backbone.conv1 = nn.Conv2d(4, 64, 7, 2, 3, bias=False)
        backbone.conv1.weight.data = new_w

        for p in list(backbone.parameters())[:6 * 2]:  
            p.requires_grad = False

        backbone.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(backbone.fc.in_features, n_cls)
        )
        self.net = backbone

    def forward(self, x):
        return self.net(x)
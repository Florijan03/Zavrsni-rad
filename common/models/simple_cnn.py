import torch.nn as nn


class SimpleCNN(nn.Module):
    """
    Simple CNN model for image classification.
    """
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # [B,16,256,256]
            nn.ReLU(),
            nn.MaxPool2d(2),                             # [B,16,128,128]

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                             # [B,32,64,64]

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                             # [B,64,32,32]
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                                # [B,64*32*32]
            nn.Linear(64 * 32 * 32, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        return self.classifier(x)

import torch
import torch.nn as nn


class TinyFusionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        nn.init.zeros_(self.net[4].weight)
        nn.init.constant_(self.net[4].bias, 1.0)

    def forward(self, x):
        return self.net(x)

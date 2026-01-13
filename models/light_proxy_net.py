import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalProxyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(5, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.enc1 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.up2_reduce = nn.Sequential(
            nn.Conv2d(96, 32, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
        )
        self.up1_reduce = nn.Sequential(
            nn.Conv2d(48, 16, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
        )

        self.out_d = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1, stride=1, padding=0),
        )
        self.softplus = nn.Softplus()

        self.conf_head = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, I, D_in, M):
        V = 1.0 - M
        x = torch.cat([I, D_in, V], dim=1)

        f0 = self.stem(x)
        f1 = self.enc1(f0)
        f2 = self.enc2(f1)

        up2 = F.interpolate(f2, size=(48, 72), mode="bilinear", align_corners=False)
        up2 = torch.cat([up2, f1], dim=1)
        up2 = self.up2_reduce(up2)

        up1 = F.interpolate(up2, size=(96, 144), mode="bilinear", align_corners=False)
        up1 = torch.cat([up1, f0], dim=1)
        up1 = self.up1_reduce(up1)

        out = F.interpolate(up1, size=(192, 288), mode="bilinear", align_corners=False)
        out = self.out_d(out)
        D_light = self.softplus(out)

        C_init = self.conf_head(f1)

        return D_light, C_init

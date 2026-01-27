import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        mid = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, reduction=8):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels, reduction=reduction)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = out + self.shortcut(x)
        out = self.relu(out)
        return out


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = torch.cat([x, skip], dim=1)
        return self.fuse(x)


class GlobalProxyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(5, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.layer1 = self._make_layer(32, 32, blocks=2, stride=2)
        self.layer2 = self._make_layer(32, 64, blocks=2, stride=2)
        self.layer3 = self._make_layer(64, 128, blocks=2, stride=2)
        self.layer4 = self._make_layer(128, 256, blocks=2, stride=2)

        self.bridge = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.up4 = UpBlock(128, 128, 64)
        self.up3 = UpBlock(64, 64, 64)
        self.up2 = UpBlock(64, 32, 32)

        self.refine = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.out_d = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
        )
        self.softplus = nn.Softplus()

        self.conf_head = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def _make_layer(self, in_ch, out_ch, blocks, stride):
        layers = [BasicBlock(in_ch, out_ch, stride=stride)]
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_ch, out_ch, stride=1))
        return nn.Sequential(*layers)

    def forward(self, I, D_in, M):
        V = 1.0 - M
        x = torch.cat([I, D_in, V], dim=1)

        f0 = self.stem(x)
        f1 = self.layer1(f0)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)

        neck = self.bridge(f4)

        x = F.interpolate(neck, size=f3.shape[2:], mode="bilinear", align_corners=False)
        x = self.up4(x, f3)

        x = F.interpolate(x, size=f2.shape[2:], mode="bilinear", align_corners=False)
        x = self.up3(x, f2)

        x = F.interpolate(x, size=f1.shape[2:], mode="bilinear", align_corners=False)
        x = self.up2(x, f1)

        x = F.interpolate(x, size=(I.shape[2], I.shape[3]), mode="bilinear", align_corners=False)
        x = self.refine(x)
        D_light = self.softplus(self.out_d(x))

        C_init = self.conf_head(f1)

        return D_light, C_init

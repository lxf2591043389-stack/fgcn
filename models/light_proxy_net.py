import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
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

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + self.shortcut(x)
        out = self.relu(out)
        return out


class MiniASPP(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_pool = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
        )
        self.out_conv = nn.Sequential(
            nn.Conv2d(out_ch * 4, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = F.interpolate(
            self.conv_pool(self.avg_pool(x)),
            size=x.size()[2:],
            mode="bilinear",
            align_corners=False,
        )
        out = torch.cat([x1, x2, x3, x4], dim=1)
        return self.out_conv(out)


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

        self.enc1 = BasicBlock(32, 64, stride=2)
        self.enc2 = BasicBlock(64, 128, stride=2)
        self.enc3 = BasicBlock(128, 256, stride=2)

        self.bridge = MiniASPP(256, 128)

        self.up3_reduce = nn.Sequential(
            nn.Conv2d(128 + 128, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.up2_reduce = nn.Sequential(
            nn.Conv2d(64 + 64, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.up1_reduce = nn.Sequential(
            nn.Conv2d(32 + 32, 32, kernel_size=3, padding=1, bias=False),
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
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, I, D_in, M):
        V = 1.0 - M
        x = torch.cat([I, D_in, V], dim=1)

        f0 = self.stem(x)
        f1 = self.enc1(f0)
        f2 = self.enc2(f1)
        f3 = self.enc3(f2)

        neck = self.bridge(f3)

        up3 = F.interpolate(neck, size=f2.shape[2:], mode="bilinear", align_corners=False)
        up3 = torch.cat([up3, f2], dim=1)
        up3 = self.up3_reduce(up3)

        up2 = F.interpolate(up3, size=f1.shape[2:], mode="bilinear", align_corners=False)
        up2 = torch.cat([up2, f1], dim=1)
        up2 = self.up2_reduce(up2)

        up1 = F.interpolate(up2, size=f0.shape[2:], mode="bilinear", align_corners=False)
        up1 = torch.cat([up1, f0], dim=1)
        up1 = self.up1_reduce(up1)

        out = F.interpolate(
            up1, size=(I.shape[2], I.shape[3]), mode="bilinear", align_corners=False
        )
        D_light = self.softplus(self.out_d(out))

        C_init = self.conf_head(f1)

        return D_light, C_init

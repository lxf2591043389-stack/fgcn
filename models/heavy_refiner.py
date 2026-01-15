import torch
import torch.nn as nn
import torch.nn.functional as F


class FGNC(nn.Module):
    def __init__(
        self,
        in_ch,
        guide_ch=16,
        ks=3,
        alpha=1.0,
        beta=2.0,
        gamma=2.0,
        theta_l=0.2,
        theta_h=0.8,
        s0=4.5,
        eps=1e-6,
    ):
        super().__init__()
        self.in_ch = in_ch
        self.guide_ch = guide_ch
        self.ks = ks
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.theta_l = theta_l
        self.theta_h = theta_h
        self.s0 = s0
        self.eps = eps
        self.log_sigma_f = nn.Parameter(torch.zeros(1))
        self.log_sigma_d = nn.Parameter(torch.zeros(1))

    def forward(self, x, d_light, c, guide):
        b, ch, h, w = x.shape
        ks = self.ks
        pad = ks // 2
        center_idx = (ks * ks) // 2

        x_unfold = F.unfold(x, kernel_size=ks, padding=pad)
        x_unfold = x_unfold.view(b, ch, ks * ks, h, w)

        d_unfold = F.unfold(d_light, kernel_size=ks, padding=pad)
        d_unfold = d_unfold.view(b, 1, ks * ks, h, w)

        c_unfold = F.unfold(c, kernel_size=ks, padding=pad)
        c_unfold = c_unfold.view(b, 1, ks * ks, h, w)

        g_unfold = F.unfold(guide, kernel_size=ks, padding=pad)
        g_unfold = g_unfold.view(b, self.guide_ch, ks * ks, h, w)

        g_center = g_unfold[:, :, center_idx:center_idx + 1, :, :]
        g_diff = (g_unfold - g_center).abs().sum(dim=1)
        sigma_f = torch.exp(self.log_sigma_f) + self.eps
        A_rgb = torch.exp(-g_diff / sigma_f)

        d_center = d_unfold[:, :, center_idx:center_idx + 1, :, :]
        log_d = torch.log(d_unfold + self.eps)
        log_d_center = torch.log(d_center + self.eps)
        d_diff = (log_d - log_d_center).abs()
        sigma_d = torch.exp(self.log_sigma_d) + self.eps
        A_depth = torch.exp(-d_diff / sigma_d)

        C_sat = (c_unfold - self.theta_l) / (self.theta_h - self.theta_l)
        C_sat = torch.clamp(C_sat, 0.0, 1.0)
        A_flux = (C_sat + self.eps) ** self.alpha

        A_rgb = A_rgb.unsqueeze(1)
        A = (A_rgb ** self.beta) * (A_depth ** self.gamma) * A_flux

        sum_A = A.sum(dim=2, keepdim=True)
        w_norm = A / (sum_A + self.eps)
        y = (w_norm * x_unfold).sum(dim=2)

        support = torch.tanh(sum_A.squeeze(2) / self.s0)
        C_new = c + support * (1.0 - c)

        return y, C_new


class FGNCBlock(nn.Module):
    def __init__(
        self,
        ch,
        guide_ch=16,
        ks=3,
        alpha=1.0,
        beta=2.0,
        gamma=2.0,
        theta_l=0.2,
        theta_h=0.8,
        s0=4.5,
    ):
        super().__init__()
        self.fgnc = FGNC(
            ch,
            guide_ch=guide_ch,
            ks=ks,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            theta_l=theta_l,
            theta_h=theta_h,
            s0=s0,
        )
        self.mix = nn.Sequential(
            nn.Conv2d(ch, ch, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, d_light, c, guide):
        x_agg, c_new = self.fgnc(x, d_light, c, guide)
        x_mix = self.mix(x_agg)
        x_out = x + x_mix
        return x_out, c_new


class ResBlock(nn.Module):
    def __init__(self, ch, dilation=1):
        super().__init__()
        padding = dilation
        self.conv1 = nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(x + out)


class HeavyRefineHead(nn.Module):
    def __init__(self, chs=(32, 64, 128), guide_ch=16, ks=3, alpha=1.0, beta=2.0, gamma=2.0, theta_l=0.2, theta_h=0.8, s0=4.5):
        super().__init__()
        ch0, ch1, ch2 = chs

        self.guide_stem = nn.Sequential(
            nn.Conv2d(3, guide_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(guide_ch, guide_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

        self.in_conv = nn.Conv2d(3, ch0, kernel_size=1, stride=1, padding=0)

        self.enc1 = FGNCBlock(ch0, guide_ch=guide_ch, ks=ks, alpha=alpha, beta=beta, gamma=gamma, theta_l=theta_l, theta_h=theta_h, s0=s0)
        self.res1 = ResBlock(ch0, dilation=2)
        self.down1 = nn.Sequential(
            nn.Conv2d(ch0, ch1, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        self.enc2 = FGNCBlock(ch1, guide_ch=guide_ch, ks=ks, alpha=alpha, beta=beta, gamma=gamma, theta_l=theta_l, theta_h=theta_h, s0=s0)
        self.res2 = ResBlock(ch1, dilation=4)
        self.down2 = nn.Sequential(
            nn.Conv2d(ch1, ch2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        self.bottleneck1 = FGNCBlock(ch2, guide_ch=guide_ch, ks=ks, alpha=alpha, beta=beta, gamma=gamma, theta_l=theta_l, theta_h=theta_h, s0=s0)
        self.bottleneck2 = FGNCBlock(ch2, guide_ch=guide_ch, ks=ks, alpha=alpha, beta=beta, gamma=gamma, theta_l=theta_l, theta_h=theta_h, s0=s0)

        self.up2_reduce = nn.Sequential(
            nn.Conv2d(ch2 + ch1, ch1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
        )
        self.dec2 = FGNCBlock(ch1, guide_ch=guide_ch, ks=ks, alpha=alpha, beta=beta, gamma=gamma, theta_l=theta_l, theta_h=theta_h, s0=s0)

        self.up1_reduce = nn.Sequential(
            nn.Conv2d(ch1 + ch0, ch0, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
        )
        self.dec1 = FGNCBlock(ch0, guide_ch=guide_ch, ks=ks, alpha=alpha, beta=beta, gamma=gamma, theta_l=theta_l, theta_h=theta_h, s0=s0)

        self.head = nn.Conv2d(ch0, 2, kernel_size=1, stride=1, padding=0)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        with torch.no_grad():
            self.head.bias[1].fill_(-5.0)

    def forward(self, I_patch, D_in_patch, D_light_patch, C_patch):
        G0 = self.guide_stem(I_patch)
        G16 = F.avg_pool2d(G0, kernel_size=2, stride=2)
        G8 = F.avg_pool2d(G0, kernel_size=4, stride=4)

        x_in = torch.cat([D_in_patch, D_light_patch, C_patch], dim=1)
        x0 = self.in_conv(x_in)

        x1, c1 = self.enc1(x0, D_light_patch, C_patch, G0)
        x1 = self.res1(x1)

        x2_in = self.down1(x1)
        D16 = F.avg_pool2d(D_light_patch, kernel_size=2, stride=2)
        C16 = F.avg_pool2d(c1, kernel_size=2, stride=2)
        x2, c2 = self.enc2(x2_in, D16, C16, G16)
        x2 = self.res2(x2)

        x3_in = self.down2(x2)
        D8 = F.avg_pool2d(D_light_patch, kernel_size=4, stride=4)
        C8 = F.avg_pool2d(c2, kernel_size=2, stride=2)
        x3, c3 = self.bottleneck1(x3_in, D8, C8, G8)
        x3, c3 = self.bottleneck2(x3, D8, c3, G8)

        x_up2 = F.interpolate(x3, size=(16, 16), mode="bilinear", align_corners=False)
        x_up2 = torch.cat([x_up2, x2], dim=1)
        x_up2 = self.up2_reduce(x_up2)
        C_up2 = F.interpolate(C8, size=(16, 16), mode="nearest")
        x_up2, _ = self.dec2(x_up2, D16, C_up2, G16)

        x_up1 = F.interpolate(x_up2, size=(32, 32), mode="bilinear", align_corners=False)
        x_up1 = torch.cat([x_up1, x1], dim=1)
        x_up1 = self.up1_reduce(x_up1)
        C_up1 = F.interpolate(C16, size=(32, 32), mode="nearest")
        x_up1, _ = self.dec1(x_up1, D_light_patch, C_up1, G0)

        out = self.head(x_up1)
        return out

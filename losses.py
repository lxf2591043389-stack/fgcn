import torch
import torch.nn.functional as F


def charbonnier(pred, target, mask=None, eps=1e-3):
    diff = pred - target
    loss = torch.sqrt(diff * diff + eps * eps)
    if mask is not None:
        valid = mask > 0
        if valid.any():
            return loss[valid].mean()
    return loss.mean()


def build_C_gt(D_in, D_gt, M, tau_c=0.05):
    diff = torch.abs(D_in - D_gt)
    C_gt = torch.exp(-diff / tau_c)
    C_gt = C_gt * (1.0 - M.float())
    return C_gt


def bce_loss(pred, target):
    return F.binary_cross_entropy(pred, target)


def edge_aware_smoothness(depth, image, mask=None):
    depth_dx = torch.abs(depth[:, :, :, 1:] - depth[:, :, :, :-1])
    depth_dy = torch.abs(depth[:, :, 1:, :] - depth[:, :, :-1, :])

    image_dx = torch.mean(torch.abs(image[:, :, :, 1:] - image[:, :, :, :-1]), dim=1, keepdim=True)
    image_dy = torch.mean(torch.abs(image[:, :, 1:, :] - image[:, :, :-1, :]), dim=1, keepdim=True)

    weight_x = torch.exp(-image_dx)
    weight_y = torch.exp(-image_dy)

    smooth_x = depth_dx * weight_x
    smooth_y = depth_dy * weight_y

    if mask is not None:
        mask_x = mask[:, :, :, 1:] * mask[:, :, :, :-1]
        mask_y = mask[:, :, 1:, :] * mask[:, :, :-1, :]
        loss = 0.0
        count = 0
        valid_x = mask_x > 0
        if valid_x.any():
            loss += smooth_x[valid_x].mean()
            count += 1
        valid_y = mask_y > 0
        if valid_y.any():
            loss += smooth_y[valid_y].mean()
            count += 1
        if count > 0:
            return loss / count
    return smooth_x.mean() + smooth_y.mean()

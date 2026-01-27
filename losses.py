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


def edge_grad_loss(pred, target, mask=None):
    pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
    diff_x = torch.abs(pred_dx - target_dx)
    diff_y = torch.abs(pred_dy - target_dy)

    if mask is not None:
        mask_x = mask[:, :, :, 1:] * mask[:, :, :, :-1]
        mask_y = mask[:, :, 1:, :] * mask[:, :, :-1, :]
        loss = 0.0
        count = 0
        valid_x = mask_x > 0
        if valid_x.any():
            loss += diff_x[valid_x].mean()
            count += 1
        valid_y = mask_y > 0
        if valid_y.any():
            loss += diff_y[valid_y].mean()
            count += 1
        if count > 0:
            return loss / count

    return diff_x.mean() + diff_y.mean()


def gradient_2d(x):
    dx = x[:, :, :, 1:] - x[:, :, :, :-1]
    dy = x[:, :, 1:, :] - x[:, :, :-1, :]
    return dx, dy


def gradient_consistency_loss(pred, gt, mask, eps=1e-6):
    dxp, dyp = gradient_2d(pred)
    dxg, dyg = gradient_2d(gt)
    mask_x = mask[:, :, :, 1:]
    mask_y = mask[:, :, 1:, :]
    loss_x = (torch.abs(dxp - dxg) * mask_x).sum() / (mask_x.sum() + eps)
    loss_y = (torch.abs(dyp - dyg) * mask_y).sum() / (mask_y.sum() + eps)
    return loss_x + loss_y

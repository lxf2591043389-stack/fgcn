import argparse
import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_factory import build_dataset
import utils
from models.light_proxy_net import GlobalProxyNet
from models.heavy_refiner import HeavyRefineHead
from models.tiny_fusion_head import TinyFusionHead
from test_tofdc_config import add_base_args, add_heavy_args, add_scheduler_args

SAFE_THRESH = 0.6
BUFFER = 0.1
PATCH_SIZE = 32
CONTEXT = 48
PAD_CTX = 8
STRIDE = 32


def build_w_global(mask):
    w = mask.clone()
    prev = mask
    for r, weight in [(2, 0.7), (4, 0.4), (8, 0.2)]:
        dil = F.max_pool2d(prev, kernel_size=2 * r + 1, stride=1, padding=r)
        ring = torch.clamp(dil - prev, 0.0, 1.0)
        w = w + weight * ring
        prev = dil
    return torch.clamp(w, 0.0, 1.0)


def forward_v2(light, heavy, fusion, I, D_in, M):
    D_light, C_init = light(I, D_in, M)
    H, W = I.shape[2], I.shape[3]
    C_full = F.interpolate(C_init, size=(H, W), mode="bilinear", align_corners=False)
    pooled = F.max_pool2d(M, kernel_size=PATCH_SIZE, stride=STRIDE)
    tiles = (pooled > 0).nonzero(as_tuple=False)
    if tiles.numel() == 0:
        return D_light, D_light, C_full

    I_pad = F.pad(I, (PAD_CTX, PAD_CTX, PAD_CTX, PAD_CTX), mode="replicate")
    D_in_pad = F.pad(D_in, (PAD_CTX, PAD_CTX, PAD_CTX, PAD_CTX), mode="replicate")
    D_light_pad = F.pad(D_light, (PAD_CTX, PAD_CTX, PAD_CTX, PAD_CTX), mode="replicate")
    C_pad = F.pad(C_full, (PAD_CTX, PAD_CTX, PAD_CTX, PAD_CTX), mode="replicate")
    M_pad = F.pad(M, (PAD_CTX, PAD_CTX, PAD_CTX, PAD_CTX), mode="replicate")

    I_patches = []
    D_in_patches = []
    D_light_patches = []
    C_patches = []
    M_patches = []
    coords = []
    for t in tiles:
        b = int(t[0])
        i = int(t[2])
        j = int(t[3])
        y0 = i * STRIDE
        x0 = j * STRIDE
        I_patches.append(I_pad[b:b + 1, :, y0:y0 + CONTEXT, x0:x0 + CONTEXT])
        D_in_patches.append(D_in_pad[b:b + 1, :, y0:y0 + CONTEXT, x0:x0 + CONTEXT])
        D_light_patches.append(D_light_pad[b:b + 1, :, y0:y0 + CONTEXT, x0:x0 + CONTEXT])
        C_patches.append(C_pad[b:b + 1, :, y0:y0 + CONTEXT, x0:x0 + CONTEXT])
        M_patches.append(M_pad[b:b + 1, :, y0:y0 + CONTEXT, x0:x0 + CONTEXT])
        coords.append((b, y0, x0))

    I_ctx = torch.cat(I_patches, dim=0)
    D_in_ctx = torch.cat(D_in_patches, dim=0)
    D_light_ctx = torch.cat(D_light_patches, dim=0)
    C_ctx = torch.cat(C_patches, dim=0)
    M_ctx = torch.cat(M_patches, dim=0)

    Dh_ctx = heavy(I_ctx, D_in_ctx, D_light_ctx, C_ctx)
    Dh_core = Dh_ctx[:, :, PAD_CTX:PAD_CTX + PATCH_SIZE, PAD_CTX:PAD_CTX + PATCH_SIZE]
    D_light_core = D_light_ctx[:, :, PAD_CTX:PAD_CTX + PATCH_SIZE, PAD_CTX:PAD_CTX + PATCH_SIZE]
    M_core = M_ctx[:, :, PAD_CTX:PAD_CTX + PATCH_SIZE, PAD_CTX:PAD_CTX + PATCH_SIZE]
    C_core = C_ctx[:, :, PAD_CTX:PAD_CTX + PATCH_SIZE, PAD_CTX:PAD_CTX + PATCH_SIZE]

    w_in = torch.cat([D_light_core, Dh_core, M_core, C_core], dim=1)
    w_core = fusion(w_in)
    D_core = w_core * Dh_core + (1.0 - w_core) * D_light_core

    D_ref = D_light.clone()
    for n, (b, y0, x0) in enumerate(coords):
        D_ref[b:b + 1, :, y0:y0 + PATCH_SIZE, x0:x0 + PATCH_SIZE] = D_core[n:n + 1]

    W_global = build_w_global(M)
    D_final = (1.0 - W_global) * D_light + W_global * D_ref
    return D_final, D_light, C_full


def load_state(model, path, device):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    state = torch.load(path, map_location=device)
    state = utils.remove_moudle(state)
    model.load_state_dict(state, strict=True)


def resolve_light_ckpt(args, result_dir):
    if args.light_ckpt:
        return args.light_ckpt
    cand = os.path.join(result_dir, "light_best_a.pth")
    if os.path.isfile(cand):
        return cand
    return os.path.join(result_dir, "light_best.pth")


def resolve_heavy_ckpt(args, result_dir):
    if args.heavy_ckpt:
        return args.heavy_ckpt
    cand = os.path.join(result_dir, "heavy_best_b.pth")
    if os.path.isfile(cand):
        return cand
    return os.path.join(result_dir, "heavy_best.pth")


def main():
    parser = argparse.ArgumentParser(description="TOFDC test B (heavy refinement)")
    add_base_args(parser)
    add_heavy_args(parser)
    add_scheduler_args(parser)
    parser.add_argument("--fusion_ckpt", default="", type=str)
    args = parser.parse_args()

    result_dir = os.path.join(args.project_root, f"result_{args.datasets}")
    run_id = time.strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(result_dir, f"test_results_b_{run_id}")
    img_save_dir = os.path.join(save_dir, "img")
    os.makedirs(img_save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    light = GlobalProxyNet().to(device)
    heavy = HeavyRefineHead().to(device)
    fusion = TinyFusionHead().to(device)

    light_ckpt = resolve_light_ckpt(args, result_dir)
    heavy_ckpt = resolve_heavy_ckpt(args, result_dir)
    load_state(light, light_ckpt, device)
    load_state(heavy, heavy_ckpt, device)
    fusion_ckpt = args.fusion_ckpt if hasattr(args, "fusion_ckpt") and args.fusion_ckpt else os.path.join(result_dir, "fusion_best_b.pth")
    load_state(fusion, fusion_ckpt, device)

    light.eval()
    heavy.eval()
    fusion.eval()

    dataset = build_dataset(args.datasets, args.split, args.dataset_cfg, args.data_root)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    total_metrics = utils.init_error_metrics()
    total_samples = 0
    total_time = 0.0

    if device.type == "cuda":
        torch.cuda.synchronize()

    with torch.no_grad():
        for idx, batch in tqdm(enumerate(loader), total=len(loader)):
            I = batch["rgb"].to(device)
            D_in = batch["depth"].to(device)
            D_gt = batch["gt"].to(device)
            M = batch["mask"].to(device)

            if args.mask_is_valid:
                M = 1.0 - M
            M = M.float()

            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.time()

            D_pred, _, _ = forward_v2(light, heavy, fusion, I, D_in, M)

            if device.type == "cuda":
                torch.cuda.synchronize()
            total_time += time.time() - start

            metrics = utils.evaluate_error(D_gt, D_pred, M, False)
            for key in total_metrics.keys():
                total_metrics[key] += metrics[key] * I.shape[0]
            total_samples += I.shape[0]

            if args.save_images:
                utils.save_eval_img(
                    img_save_dir,
                    idx,
                    D_in[0].cpu() / args.max_depth_vis,
                    D_gt[0].cpu() / args.max_depth_vis,
                    D_pred[0].cpu() / args.max_depth_vis,
                )

    avg_metrics = {k: total_metrics[k] / max(1, total_samples) for k in total_metrics}
    avg_time = total_time / max(1, total_samples)
    fps = 1.0 / avg_time if avg_time > 0 else 0.0
    summary_path = os.path.join(save_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("=== Test B Results ===\n")
        f.write(f"samples: {total_samples}\n")
        f.write(f"RMSE: {avg_metrics['RMSE']:.4f}\n")
        f.write(f"ABS_REL: {avg_metrics['ABS_REL']:.4f}\n")
        f.write(f"DELTA1.02: {avg_metrics['DELTA1.02']:.4f}\n")
        f.write(f"DELTA1.05: {avg_metrics['DELTA1.05']:.4f}\n")
        f.write(f"DELTA1.10: {avg_metrics['DELTA1.10']:.4f}\n")
        f.write(f"time: {avg_time * 1000:.2f} ms/img\n")
        f.write(f"fps: {fps:.2f}\n")
        f.write(f"save_dir: {save_dir}\n")
        f.write("\n=== Checkpoints ===\n")
        f.write(f"light_ckpt: {light_ckpt}\n")
        f.write(f"heavy_ckpt: {heavy_ckpt}\n")
        f.write(f"fusion_ckpt: {fusion_ckpt}\n")

    print("=== Test B Results ===")
    print(f"samples: {total_samples}")
    print(f"RMSE: {avg_metrics['RMSE']:.4f}")
    print(f"ABS_REL: {avg_metrics['ABS_REL']:.4f}")
    print(f"DELTA1.02: {avg_metrics['DELTA1.02']:.4f}")
    print(f"DELTA1.05: {avg_metrics['DELTA1.05']:.4f}")
    print(f"DELTA1.10: {avg_metrics['DELTA1.10']:.4f}")
    print(f"time: {avg_time * 1000:.2f} ms/img")
    print(f"fps: {fps:.2f}")
    print(f"save_dir: {save_dir}")


if __name__ == "__main__":
    main()

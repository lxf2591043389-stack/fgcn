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
from models.scheduler import tile_scheduler
from test_tofdc_config import add_base_args, add_heavy_args, add_scheduler_args

LOW_CONF_THRESH = 0.3


def crop_tile_patches(I, D_in, D_light, M, C_init, tiles):
    bsz = I.shape[0]
    ksz = tiles.shape[1]
    tiles_list = tiles.detach().cpu().tolist()

    I_patches = []
    D_in_patches = []
    D_light_patches = []
    M_patches = []
    C_patches = []

    for b in range(bsz):
        for k in range(ksz):
            i, j = tiles_list[b][k]
            y0 = int(i) * 32
            x0 = int(j) * 32
            I_patch = I[b:b + 1, :, y0:y0 + 32, x0:x0 + 32]
            D_in_patch = D_in[b:b + 1, :, y0:y0 + 32, x0:x0 + 32]
            D_light_patch = D_light[b:b + 1, :, y0:y0 + 32, x0:x0 + 32]
            M_patch = M[b:b + 1, :, y0:y0 + 32, x0:x0 + 32]

            y1 = int(i) * 8
            x1 = int(j) * 8
            C_tile = C_init[b:b + 1, :, y1:y1 + 8, x1:x1 + 8]
            C_patch = F.interpolate(C_tile, size=(32, 32), mode="bilinear", align_corners=False)

            I_patches.append(I_patch)
            D_in_patches.append(D_in_patch)
            D_light_patches.append(D_light_patch)
            M_patches.append(M_patch)
            C_patches.append(C_patch)

    I_patch = torch.cat(I_patches, dim=0)
    D_in_patch = torch.cat(D_in_patches, dim=0)
    D_light_patch = torch.cat(D_light_patches, dim=0)
    M_patch = torch.cat(M_patches, dim=0)
    C_patch = torch.cat(C_patches, dim=0)
    return I_patch, D_in_patch, D_light_patch, M_patch, C_patch


def scatter_residual(res_patch, tiles, B, H=192, W=288):
    ksz = tiles.shape[1]
    res_full = res_patch.new_zeros((B, 1, H, W))
    res_patch = res_patch.view(B, ksz, 1, 32, 32)
    tiles_list = tiles.detach().cpu().tolist()

    for b in range(B):
        for k in range(ksz):
            i, j = tiles_list[b][k]
            y0 = int(i) * 32
            x0 = int(j) * 32
            res_full[b:b + 1, :, y0:y0 + 32, x0:x0 + 32] += res_patch[b, k]

    return res_full


def compute_update_mask(M_patch, C_patch, thresh=LOW_CONF_THRESH):
    low_confidence_area = (C_patch < thresh).float()
    return torch.max(M_patch, low_confidence_area)


def forward_heavy(light, heavy, I, D_in, M, args):
    D_light, C_init = light(I, D_in, M)
    tiles = tile_scheduler(
        C_init,
        M,
        k_max=args.k_max,
        tau_miss=args.tau_miss,
        dilation_r=args.dilation_r,
        lam=args.lam,
        fill_to_kmax=True,
        adaptive_k=args.adaptive_k,
        risk_top_ratio=args.risk_top_ratio,
        k_min=args.k_min,
    )
    I_patch, D_in_patch, D_light_patch, M_patch, C_patch = crop_tile_patches(
        I, D_in, D_light, M, C_init, tiles
    )
    delta_raw = heavy(I_patch, D_in_patch, D_light_patch, C_patch)
    update_mask = compute_update_mask(M_patch, C_patch)
    res_patch = delta_raw * update_mask
    res_full = scatter_residual(res_patch, tiles, B=I.shape[0], H=I.shape[2], W=I.shape[3])
    D_final = D_light + res_full
    return D_final, D_light, C_init


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
    args = parser.parse_args()

    result_dir = os.path.join(args.project_root, f"result_{args.datasets}")
    run_id = time.strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(result_dir, f"test_results_b_{run_id}")
    img_save_dir = os.path.join(save_dir, "img")
    os.makedirs(img_save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    light = GlobalProxyNet().to(device)
    heavy = HeavyRefineHead().to(device)

    light_ckpt = resolve_light_ckpt(args, result_dir)
    heavy_ckpt = resolve_heavy_ckpt(args, result_dir)
    load_state(light, light_ckpt, device)
    load_state(heavy, heavy_ckpt, device)

    light.eval()
    heavy.eval()

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

            D_pred, _, _ = forward_heavy(light, heavy, I, D_in, M, args)

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

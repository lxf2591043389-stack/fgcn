import argparse
import os

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


def load_state(model, path, device):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    state = torch.load(path, map_location=device)
    state = utils.remove_moudle(state)
    model.load_state_dict(state, strict=True)


def compute_update_mask(M_patch, C_patch, thresh=LOW_CONF_THRESH):
    low_confidence_area = (C_patch < thresh).float()
    return torch.max(M_patch, low_confidence_area)


def main():
    parser = argparse.ArgumentParser(description="Diagnostics for D_light and clamp ratio")
    add_base_args(parser)
    add_heavy_args(parser)
    add_scheduler_args(parser)
    parser.add_argument("--max_batches", default=-1, type=int)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    light = GlobalProxyNet().to(device)
    heavy = HeavyRefineHead().to(device)

    result_dir = os.path.join(args.project_root, f"result_{args.datasets}")
    light_ckpt = args.light_ckpt or os.path.join(result_dir, "light_best.pth")
    heavy_ckpt = args.heavy_ckpt or os.path.join(result_dir, "heavy_best.pth")
    load_state(light, light_ckpt, device)
    load_state(heavy, heavy_ckpt, device)
    light.eval()
    heavy.eval()

    dataset = build_dataset(args.datasets, args.split, args.dataset_cfg, args.data_root)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    diff_list = []
    abs_res_sum = 0.0
    delta_abs_list = []

    with torch.no_grad():
        for idx, batch in tqdm(enumerate(loader), total=len(loader)):
            if args.max_batches > 0 and idx >= args.max_batches:
                break

            I = batch["rgb"].to(device)
            D_in = batch["depth"].to(device)
            D_gt = batch["gt"].to(device)
            M = batch["mask"].to(device)

            if args.mask_is_valid:
                M = 1.0 - M
            M = M.float()

            D_light, C_init = light(I, D_in, M)

            diff = torch.abs(D_light - D_gt)
            hole = M > 0.5
            if hole.any():
                diff_list.append(diff[hole].detach().cpu())

            tiles = tile_scheduler(
                C_init,
                M,
                k_max=args.k_max,
                tau_miss=args.tau_miss,
                dilation_r=args.dilation_r,
                lam=args.lam,
                fill_to_kmax=True,
            )
            I_patch, D_in_patch, D_light_patch, M_patch, C_patch = crop_tile_patches(
                I, D_in, D_light, M, C_init, tiles
            )
            delta_raw = heavy(I_patch, D_in_patch, D_light_patch, C_patch)
            update_mask = compute_update_mask(M_patch, C_patch)
            res_patch = delta_raw * update_mask
            abs_res = torch.abs(res_patch)
            abs_res_sum += float(abs_res.mean().item())
            delta_abs_list.append(torch.abs(delta_raw).detach().cpu())

    if diff_list:
        diffs = torch.cat(diff_list)
        mean_diff = float(diffs.mean().item())
        p95_diff = float(torch.quantile(diffs, 0.95).item())
    else:
        mean_diff = 0.0
        p95_diff = 0.0

    if delta_abs_list:
        delta_abs = torch.cat(delta_abs_list)
        delta_mean = float(delta_abs.mean().item())
        delta_p95 = float(torch.quantile(delta_abs, 0.95).item())
    else:
        delta_mean = 0.0
        delta_p95 = 0.0

    print("=== Diagnostics ===")
    print(f"hole |D_light - D_gt| mean: {mean_diff:.4f} m")
    print(f"hole |D_light - D_gt| p95 : {p95_diff:.4f} m")
    print(f"delta_raw |.| mean: {delta_mean:.4f} m")
    print(f"delta_raw |.| p95 : {delta_p95:.4f} m")
    print(f"res_patch |.| mean (per batch): {abs_res_sum / max(1, idx + 1):.6f}")


if __name__ == "__main__":
    main()

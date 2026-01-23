import argparse
import json
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
from test_tofdc_config import RUN_CONFIG_PATH, add_base_args, add_heavy_args, add_scheduler_args
from config_utils import load_run_config

try:
    from thop import profile
except Exception:
    profile = None

SAFE_THRESH = 0.6
BUFFER = 0.1
PATCH_SIZE = 32
STRIDE = 16


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def crop_tile_patches(I, D_in, D_light, M, C_init, tiles, stride=PATCH_SIZE):
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
            y0 = int(i) * stride
            x0 = int(j) * stride
            if y0 + PATCH_SIZE > I.shape[2] or x0 + PATCH_SIZE > I.shape[3]:
                continue
            I_patch = I[b:b + 1, :, y0:y0 + PATCH_SIZE, x0:x0 + PATCH_SIZE]
            D_in_patch = D_in[b:b + 1, :, y0:y0 + PATCH_SIZE, x0:x0 + PATCH_SIZE]
            D_light_patch = D_light[b:b + 1, :, y0:y0 + PATCH_SIZE, x0:x0 + PATCH_SIZE]
            M_patch = M[b:b + 1, :, y0:y0 + PATCH_SIZE, x0:x0 + PATCH_SIZE]

            c_stride = stride // 4
            c_size = PATCH_SIZE // 4
            y1 = int(i) * c_stride
            x1 = int(j) * c_stride
            C_tile = C_init[b:b + 1, :, y1:y1 + c_size, x1:x1 + c_size]
            C_patch = F.interpolate(C_tile, size=(PATCH_SIZE, PATCH_SIZE), mode="bilinear", align_corners=False)

            I_patches.append(I_patch)
            D_in_patches.append(D_in_patch)
            D_light_patches.append(D_light_patch)
            M_patches.append(M_patch)
            C_patches.append(C_patch)

    if not I_patches:
        return None, None, None, None, None
    I_patch = torch.cat(I_patches, dim=0)
    D_in_patch = torch.cat(D_in_patches, dim=0)
    D_light_patch = torch.cat(D_light_patches, dim=0)
    M_patch = torch.cat(M_patches, dim=0)
    C_patch = torch.cat(C_patches, dim=0)
    return I_patch, D_in_patch, D_light_patch, M_patch, C_patch


def scatter_residual(res_patch, tiles, B, H=192, W=288, stride=PATCH_SIZE):
    ksz = tiles.shape[1]
    res_full = res_patch.new_zeros((B, 1, H, W))
    count = res_patch.new_zeros((B, 1, H, W))
    res_patch = res_patch.view(B, ksz, 1, 32, 32)
    tiles_list = tiles.detach().cpu().tolist()

    for b in range(B):
        for k in range(ksz):
            i, j = tiles_list[b][k]
            y0 = int(i) * stride
            x0 = int(j) * stride
            if y0 + PATCH_SIZE > H or x0 + PATCH_SIZE > W:
                continue
            res_full[b:b + 1, :, y0:y0 + PATCH_SIZE, x0:x0 + PATCH_SIZE] += res_patch[b, k]
            count[b:b + 1, :, y0:y0 + PATCH_SIZE, x0:x0 + PATCH_SIZE] += 1.0

    res_full = res_full / torch.clamp(count, min=1.0)
    return res_full


def compute_update_mask(mask_hole, conf, safe_thresh=SAFE_THRESH, buffer=BUFFER):
    soft_edge = torch.clamp((safe_thresh - conf) / buffer, 0.0, 1.0)
    return torch.max(mask_hole, soft_edge)


def expand_tiles_for_overlap(tiles, H, W, base_stride=PATCH_SIZE, stride=STRIDE):
    bsz = tiles.shape[0]
    tiles_list = tiles.detach().cpu().tolist()
    expanded = []
    max_len = 0
    for b in range(bsz):
        coords = set()
        for i, j in tiles_list[b]:
            y0 = int(i) * base_stride
            x0 = int(j) * base_stride
            for dy in (0, stride):
                for dx in (0, stride):
                    yy = y0 + dy
                    xx = x0 + dx
                    if yy + PATCH_SIZE <= H and xx + PATCH_SIZE <= W:
                        coords.add((yy // stride, xx // stride))
        if not coords:
            coords = {(0, 0)}
        coords_list = sorted(coords)
        expanded.append(coords_list)
        max_len = max(max_len, len(coords_list))

    out = torch.zeros((bsz, max_len, 2), dtype=torch.long, device=tiles.device)
    for b in range(bsz):
        coords_list = expanded[b]
        if len(coords_list) < max_len:
            coords_list = coords_list + [coords_list[-1]] * (max_len - len(coords_list))
        out[b] = torch.tensor(coords_list, dtype=torch.long, device=tiles.device)
    return out


def count_unique_tiles(tiles):
    counts = []
    for b in range(tiles.shape[0]):
        uniq = torch.unique(tiles[b], dim=0)
        counts.append(int(uniq.shape[0]))
    return counts


def forward_heavy(light, heavy, I, D_in, M, args):
    D_light, C_init = light(I, D_in, M)
    tiles = tile_scheduler(
        C_init,
        M,
        k_max=args.k_max,
        tau_miss=args.tau_miss,
        dilation_r=args.dilation_r,
        lam=args.lam,
        fill_to_kmax=False,
        adaptive_k=True,
        risk_top_ratio=0.0 if args.only_miss else args.risk_top_ratio,
        k_min=args.k_min,
    )
    k_counts = count_unique_tiles(tiles)
    tiles = expand_tiles_for_overlap(tiles, H=I.shape[2], W=I.shape[3])
    I_patch, D_in_patch, D_light_patch, M_patch, C_patch = crop_tile_patches(
        I, D_in, D_light, M, C_init, tiles, stride=STRIDE
    )
    if I_patch is None:
        return D_light, D_light, C_init, k_counts
    heavy_out = heavy(I_patch, D_in_patch, D_light_patch, C_patch)
    update_mask = compute_update_mask(M_patch, C_patch)
    delta_raw = heavy_out[:, 0:1, :, :]
    gate_logit = heavy_out[:, 1:2, :, :]
    delta_val = 2.0 * torch.tanh(delta_raw)
    sigma = torch.sigmoid(gate_logit)
    res_patch = update_mask * (sigma * delta_val)
    res_full = scatter_residual(res_patch, tiles, B=I.shape[0], H=I.shape[2], W=I.shape[3], stride=STRIDE)
    D_final = D_light + res_full
    return D_final, D_light, C_init, k_counts


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
    parser = argparse.ArgumentParser(description="TOFDC test total (adaptive refinement)")
    add_base_args(parser)
    add_heavy_args(parser)
    add_scheduler_args(parser)
    parser.add_argument("--only_miss",default="true",action="store_true", help="Use miss_counts only for k_out.")
    args = parser.parse_args()
    args.k_max = 32
    args.risk_top_ratio = 0.02

    run_cfg = load_run_config(RUN_CONFIG_PATH)

    result_dir = os.path.join(args.project_root, f"result_{args.datasets}")
    run_id = time.strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(result_dir, f"test_results_total_{run_id}")
    img_save_dir = os.path.join(save_dir, "img")
    if args.save_images:
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
    k_values = []

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

            D_pred, _, _, k_counts = forward_heavy(light, heavy, I, D_in, M, args)
            k_values.extend(k_counts)

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
    k_min = min(k_values) if k_values else 0
    k_max = max(k_values) if k_values else 0
    k_avg = sum(k_values) / len(k_values) if k_values else 0.0
    light_params = count_params(light)
    heavy_params = count_params(heavy)
    total_params = light_params + heavy_params
    flops_light = None
    flops_heavy = None
    total_flops = None
    if profile is not None:
        flops_light, _ = profile(light, inputs=(I, D_in, M), verbose=False)
        patch_b = max(1, int(round(k_avg))) * args.batch_size
        I_patch = torch.rand(patch_b, 3, 32, 32, device=device)
        D_patch = torch.rand(patch_b, 1, 32, 32, device=device)
        C_patch = torch.rand(patch_b, 1, 32, 32, device=device)
        flops_heavy, _ = profile(heavy, inputs=(I_patch, D_patch, D_patch, C_patch), verbose=False)
        total_flops = flops_light + flops_heavy
    summary_path = os.path.join(save_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("=== Test Total Results ===\n")
        f.write(f"samples: {total_samples}\n")
        f.write(f"RMSE: {avg_metrics['RMSE']:.4f}\n")
        f.write(f"ABS_REL: {avg_metrics['ABS_REL']:.4f}\n")
        f.write(f"DELTA1.02: {avg_metrics['DELTA1.02']:.4f}\n")
        f.write(f"DELTA1.05: {avg_metrics['DELTA1.05']:.4f}\n")
        f.write(f"DELTA1.10: {avg_metrics['DELTA1.10']:.4f}\n")
        f.write(f"time: {avg_time * 1000:.2f} ms/img\n")
        f.write(f"fps: {fps:.2f}\n")
        f.write(f"k_out min/avg/max: {k_min}/{k_avg:.2f}/{k_max}\n")
        f.write(f"only_miss: {args.only_miss}\n")
        f.write(f"params_million: {total_params / 1e6:.3f}\n")
        if total_flops is not None:
            f.write(f"gflops: {total_flops / 1e9:.3f}\n")
        f.write(f"save_dir: {save_dir}\n")
        f.write("\n=== Checkpoints ===\n")
        f.write(f"light_ckpt: {light_ckpt}\n")
        f.write(f"heavy_ckpt: {heavy_ckpt}\n")
        f.write("\n=== Args ===\n")
        f.write(json.dumps(vars(args), sort_keys=True, indent=2, ensure_ascii=True))
        f.write("\n")
        f.write("\n=== Run Config ===\n")
        f.write(json.dumps(run_cfg, sort_keys=True, indent=2, ensure_ascii=True))
        f.write("\n")

    print("=== Test Total Results ===")
    print(f"samples: {total_samples}")
    print(f"RMSE: {avg_metrics['RMSE']:.4f}")
    print(f"ABS_REL: {avg_metrics['ABS_REL']:.4f}")
    print(f"DELTA1.02: {avg_metrics['DELTA1.02']:.4f}")
    print(f"DELTA1.05: {avg_metrics['DELTA1.05']:.4f}")
    print(f"DELTA1.10: {avg_metrics['DELTA1.10']:.4f}")
    print(f"time: {avg_time * 1000:.2f} ms/img")
    print(f"fps: {fps:.2f}")
    print(f"k_out min/avg/max: {k_min}/{k_avg:.2f}/{k_max}")
    print(f"only_miss: {args.only_miss}")
    print(f"params_million: {total_params / 1e6:.3f}")
    if total_flops is not None:
        print(f"gflops: {total_flops / 1e9:.3f}")
    print(f"save_dir: {save_dir}")


if __name__ == "__main__":
    main()

import argparse
import json
import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from config_utils import load_run_config
from dataset_factory import build_dataset, load_dataset_config, resolve_dataset_name
import utils
from models.light_proxy_net import GlobalProxyNet
from models.heavy_refiner import HeavyRefineHead
from test_tofdc_config import RUN_CONFIG_PATH, add_base_args, add_heavy_args, add_scheduler_args

PATCH_SIZE = 32
STRIDE = 32
DELTA_SCALE = 1.0


def scatter_patch_to_full(D_patch, tiles, B, H, W, stride, D_light_fallback):
    ksz = tiles.shape[1]
    full = D_patch.new_zeros((B, 1, H, W))
    count = D_patch.new_zeros((B, 1, H, W))
    D_patch = D_patch.view(B, ksz, 1, PATCH_SIZE, PATCH_SIZE)
    tiles_list = tiles.detach().cpu().tolist()

    for b in range(B):
        for k in range(ksz):
            i, j = tiles_list[b][k]
            y0 = int(i) * stride
            x0 = int(j) * stride
            full[b:b + 1, :, y0:y0 + PATCH_SIZE, x0:x0 + PATCH_SIZE] += D_patch[b, k]
            count[b:b + 1, :, y0:y0 + PATCH_SIZE, x0:x0 + PATCH_SIZE] += 1.0

    out = full / torch.clamp(count, min=1.0)
    out = torch.where(count > 0, out, D_light_fallback)
    return out, count


def forward_v2(light, heavy, I, D_in, M, D_max):
    D_light, C_init = light(I, D_in, M)
    H, W = I.shape[2], I.shape[3]
    D_light = torch.clamp(D_light, 0.0, D_max)
    C_full = F.interpolate(C_init, size=(H, W), mode="bilinear", align_corners=False)

    pooled = F.max_pool2d(M, kernel_size=PATCH_SIZE, stride=STRIDE)
    tiles = (pooled > 0).nonzero(as_tuple=False)
    if tiles.numel() == 0:
        return D_light, D_light, C_full

    B = I.shape[0]
    tiles_list = tiles.detach().cpu().tolist()
    tiles_by_b = [[] for _ in range(B)]
    for t in tiles_list:
        b, i, j = int(t[0]), int(t[2]), int(t[3])
        tiles_by_b[b].append((i, j))
    max_k = max((len(v) for v in tiles_by_b), default=0)
    if max_k == 0:
        return D_light, D_light, C_full

    tiles_tensor = torch.zeros((B, max_k, 2), dtype=torch.long, device=I.device)
    for b in range(B):
        tlist = tiles_by_b[b]
        if not tlist:
            tlist = [(0, 0)]
        if len(tlist) < max_k:
            tlist = tlist + [tlist[-1]] * (max_k - len(tlist))
        tiles_tensor[b] = torch.tensor(tlist, dtype=torch.long, device=I.device)

    I_patches = []
    D_in_patches = []
    D_light_patches = []
    C_patches = []
    M_patches = []
    for b in range(B):
        for i, j in tiles_by_b[b]:
            y0 = i * STRIDE
            x0 = j * STRIDE
            I_patches.append(I[b:b + 1, :, y0:y0 + PATCH_SIZE, x0:x0 + PATCH_SIZE])
            D_in_patches.append(D_in[b:b + 1, :, y0:y0 + PATCH_SIZE, x0:x0 + PATCH_SIZE])
            D_light_patches.append(D_light[b:b + 1, :, y0:y0 + PATCH_SIZE, x0:x0 + PATCH_SIZE])
            C_patches.append(C_full[b:b + 1, :, y0:y0 + PATCH_SIZE, x0:x0 + PATCH_SIZE])
            M_patches.append(M[b:b + 1, :, y0:y0 + PATCH_SIZE, x0:x0 + PATCH_SIZE])

    I_ctx = torch.cat(I_patches, dim=0)
    D_in_ctx = torch.cat(D_in_patches, dim=0)
    D_light_ctx = torch.cat(D_light_patches, dim=0)
    C_ctx = torch.cat(C_patches, dim=0)
    M_ctx = torch.cat(M_patches, dim=0)

    delta_raw = heavy(I_ctx, D_in_ctx, D_light_ctx, C_ctx)
    delta = DELTA_SCALE * torch.tanh(delta_raw)
    Dh = torch.clamp(D_light_ctx + delta, 0.0, D_max)
    hole_patch = (M_ctx > 0.5).float()
    D_final_patch = torch.where(hole_patch > 0, Dh, D_light_ctx)

    D_final_full, count = scatter_patch_to_full(
        D_final_patch, tiles_tensor, B=B, H=H, W=W, stride=STRIDE, D_light_fallback=D_light
    )
    return D_final_full, D_light, C_full


def load_state(model, path, device):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    state = torch.load(path, map_location=device)
    state = utils.remove_moudle(state)
    model.load_state_dict(state, strict=True)


def resolve_light_ckpt(args, result_dir):
    if args.light_ckpt:
        return args.light_ckpt
    if args.light_source == "a":
        cand = os.path.join(result_dir, "light_best_a.pth")
    else:
        cand = os.path.join(result_dir, "light_best_c.pth")
    if os.path.isfile(cand):
        return cand
    return os.path.join(result_dir, "light_best.pth")


def resolve_heavy_ckpt(args, result_dir):
    if args.heavy_ckpt:
        return args.heavy_ckpt
    cand = os.path.join(result_dir, "heavy_best_c.pth")
    if os.path.isfile(cand):
        return cand
    return os.path.join(result_dir, "heavy_best.pth")


def main():
    parser = argparse.ArgumentParser(description="TOFDC test C (V2 hard replace)")
    add_base_args(parser)
    add_heavy_args(parser)
    add_scheduler_args(parser)
    parser.add_argument("--light_source", default="c", choices=["a", "c"])
    args = parser.parse_args()

    run_cfg = load_run_config(RUN_CONFIG_PATH)
    dataset_cfg = {}
    dataset_name = args.datasets
    dataset_root = args.data_root
    try:
        dataset_cfg = load_dataset_config(args.dataset_cfg)
        dataset_name = resolve_dataset_name(args.datasets, dataset_cfg)
        ds_cfg = dataset_cfg.get("datasets", {}).get(dataset_name, {})
        dataset_root = args.data_root if args.data_root else ds_cfg.get("root")
        args.depth_max_m = float(ds_cfg.get("depth_max_m", 5.0))
    except Exception as exc:
        dataset_cfg = {"error": str(exc)}
        args.depth_max_m = 5.0

    result_dir = os.path.join(args.project_root, f"result_{args.datasets}")
    run_id = time.strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(result_dir, f"test_results_c_{run_id}")
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

            D_pred, _, _ = forward_v2(light, heavy, I, D_in, M, D_max=args.depth_max_m)

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
        f.write("=== Test C Results ===\n")
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
        f.write("\n=== Args ===\n")
        f.write(json.dumps(vars(args), sort_keys=True, indent=2, ensure_ascii=True))
        f.write("\n")
        f.write("\n=== Run Config ===\n")
        f.write(json.dumps(run_cfg, sort_keys=True, indent=2, ensure_ascii=True))
        f.write("\n")
        f.write("\n=== Train Config ===\n")
        f.write(json.dumps(run_cfg.get("train", {}), sort_keys=True, indent=2, ensure_ascii=True))
        f.write("\n")
        f.write("\n=== Dataset Config ===\n")
        f.write(f"dataset_name: {dataset_name}\n")
        f.write(f"dataset_root: {dataset_root}\n")
        f.write(f"dataset_cfg_path: {args.dataset_cfg}\n")
        f.write(json.dumps(dataset_cfg, sort_keys=True, indent=2, ensure_ascii=True))
        f.write("\n")

    print("=== Test C Results ===")
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

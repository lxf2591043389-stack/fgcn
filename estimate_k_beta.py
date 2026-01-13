import argparse
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_factory import build_dataset
import utils
from models.light_proxy_net import GlobalProxyNet
from test_tofdc_config import add_base_args, add_scheduler_args


def _dilate_mask(mask, dilation_r):
    out = mask
    for _ in range(dilation_r):
        out = F.max_pool2d(out.float(), kernel_size=3, stride=1, padding=1) > 0
    return out


def parse_betas(text):
    items = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        items.append(float(part))
    return items


def parse_tau_list(text, default_tau):
    if not text:
        return [default_tau]
    items = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        items.append(float(part))
    return items


def resolve_light_ckpt(args, result_dir):
    if args.light_ckpt:
        return args.light_ckpt
    cand = os.path.join(result_dir, "light_best_a.pth")
    if os.path.isfile(cand):
        return cand
    return os.path.join(result_dir, "light_best.pth")


def load_state(model, path, device):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    state = torch.load(path, map_location=device)
    state = utils.remove_moudle(state)
    model.load_state_dict(state, strict=True)


def summarize_array(arr):
    if arr.size == 0:
        return {
            "count": 0,
            "min": 0,
            "p10": 0,
            "p25": 0,
            "p50": 0,
            "p75": 0,
            "p90": 0,
            "max": 0,
            "mean": 0,
        }
    return {
        "count": int(arr.size),
        "min": float(np.min(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
    }


def format_summary(name, stats):
    return (
        f"{name}: count={stats['count']}, min={stats['min']:.2f}, "
        f"p10={stats['p10']:.2f}, p25={stats['p25']:.2f}, p50={stats['p50']:.2f}, "
        f"p75={stats['p75']:.2f}, p90={stats['p90']:.2f}, max={stats['max']:.2f}, "
        f"mean={stats['mean']:.2f}"
    )


def main():
    parser = argparse.ArgumentParser(description="Estimate adaptive K for different betas")
    add_base_args(parser)
    add_scheduler_args(parser)
    parser.add_argument("--low_conf_thresh", default=0.3, type=float)
    parser.add_argument("--betas", default="0.25,0.5,0.75,1.0", type=str)
    parser.add_argument("--tau_list", default="0.01,0.02,0.03,0.05", type=str)
    parser.add_argument("--k_min", default=1, type=int)
    parser.add_argument("--max_samples", default=0, type=int)
    args = parser.parse_args()

    betas = parse_betas(args.betas)
    if not betas:
        raise ValueError("No betas provided")

    tau_list = parse_tau_list(args.tau_list, args.tau_miss)
    if not tau_list:
        raise ValueError("No tau values provided")

    result_dir = os.path.join(args.project_root, f"result_{args.datasets}")
    run_id = time.strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(result_dir, f"k_beta_stats_{run_id}")
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    light = GlobalProxyNet().to(device)
    light_ckpt = resolve_light_ckpt(args, result_dir)
    load_state(light, light_ckpt, device)
    light.eval()

    dataset = build_dataset(args.datasets, args.split, args.dataset_cfg, args.data_root)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    tau_stats = {}
    for tau in tau_list:
        tau_stats[tau] = {
            "miss": [],
            "low_only": [],
            "k": {beta: [] for beta in betas},
        }

    total_seen = 0
    with torch.no_grad():
        for _, batch in tqdm(enumerate(loader), total=len(loader)):
            I = batch["rgb"].to(device)
            D_in = batch["depth"].to(device)
            M = batch["mask"].to(device)

            if args.mask_is_valid:
                M = 1.0 - M
            M = M.float()

            _, C_init = light(I, D_in, M)

            C_tile = F.avg_pool2d(C_init, kernel_size=8, stride=8)
            low_tiles = C_tile < args.low_conf_thresh

            B = C_init.shape[0]
            M_1_4 = F.avg_pool2d(M, kernel_size=4, stride=4)
            hole_ratio = F.avg_pool2d(M_1_4, kernel_size=8, stride=8)

            for tau in tau_list:
                T_miss = hole_ratio > tau
                T_miss_ctx = _dilate_mask(T_miss, args.dilation_r)
                low_only = low_tiles & ~T_miss_ctx

                miss_counts = T_miss_ctx.view(B, -1).sum(dim=1).cpu().tolist()
                low_only_counts = low_only.view(B, -1).sum(dim=1).cpu().tolist()

                for b in range(B):
                    miss = int(miss_counts[b])
                    low = int(low_only_counts[b])
                    tau_stats[tau]["miss"].append(miss)
                    tau_stats[tau]["low_only"].append(low)
                    for beta in betas:
                        k = miss + int(math.ceil(beta * low))
                        k = max(args.k_min, min(args.k_max, k))
                        tau_stats[tau]["k"][beta].append(k)

            total_seen += B
            if args.max_samples > 0 and total_seen >= args.max_samples:
                break

    summary_lines = []
    summary_lines.append(f"samples: {total_seen}")
    summary_lines.append(f"low_conf_thresh: {args.low_conf_thresh}")
    summary_lines.append(f"k_max: {args.k_max}, k_min: {args.k_min}")
    summary_lines.append("tau_list: " + ", ".join([str(v) for v in tau_list]))
    summary_lines.append("")

    for tau in tau_list:
        miss_stats = summarize_array(np.array(tau_stats[tau]["miss"], dtype=np.float32))
        low_stats = summarize_array(np.array(tau_stats[tau]["low_only"], dtype=np.float32))
        summary_lines.append(f"tau_miss={tau}")
        summary_lines.append(format_summary("miss_tiles", miss_stats))
        summary_lines.append(format_summary("low_only_tiles", low_stats))
        summary_lines.append("")

        for beta in betas:
            arr = np.array(tau_stats[tau]["k"][beta], dtype=np.int32)
            stats = summarize_array(arr.astype(np.float32))
            summary_lines.append(f"beta={beta}")
            summary_lines.append(format_summary("K", stats))

            hist = np.bincount(arr, minlength=args.k_max + 1)
            hist_lines = []
            for k in range(len(hist)):
                if hist[k] > 0:
                    hist_lines.append(f"{k}:{int(hist[k])}")
            summary_lines.append("hist: " + " ".join(hist_lines))
            summary_lines.append("")

    summary_path = os.path.join(save_dir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))
        f.write("\n")

    print("\n".join(summary_lines))
    print(f"save_dir: {save_dir}")


if __name__ == "__main__":
    main()

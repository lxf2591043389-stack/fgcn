import argparse
import os
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm

from dataset_factory import build_dataset
import utils
from models.light_proxy_net import GlobalProxyNet
from test_tofdc_config import add_base_args


def save_colormap(tensor, path):
    arr = torch.clamp(tensor, 0.0, 1.0).squeeze(0).cpu().numpy()
    arr = (arr * 255.0).astype(np.uint8)
    colored = cv2.applyColorMap(arr, cv2.COLORMAP_RAINBOW)
    Image.fromarray(colored).save(path)


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


def main():
    parser = argparse.ArgumentParser(description="TOFDC test A (light outputs)")
    add_base_args(parser)
    args = parser.parse_args()

    result_dir = os.path.join(args.project_root, f"result_{args.datasets}")
    run_id = time.strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(result_dir, f"test_results_a_{run_id}")
    img_save_dir = os.path.join(save_dir, "img")
    os.makedirs(img_save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    light = GlobalProxyNet().to(device)

    light_ckpt = resolve_light_ckpt(args, result_dir)
    load_state(light, light_ckpt, device)
    light.eval()

    dataset = build_dataset(args.datasets, args.split, args.dataset_cfg, args.data_root)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    total_time = 0.0
    total_metrics = utils.init_error_metrics()
    total_samples = 0

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

            D_light, C_init = light(I, D_in, M)

            if device.type == "cuda":
                torch.cuda.synchronize()
            total_time += time.time() - start

            metrics = utils.evaluate_error(D_gt, D_light, M, False)
            for key in total_metrics.keys():
                total_metrics[key] += metrics[key] * I.shape[0]
            total_samples += I.shape[0]

            C_up = F.interpolate(C_init, size=(D_light.shape[2], D_light.shape[3]), mode="bilinear", align_corners=False)

            d_path = os.path.join(img_save_dir, f"{idx:04d}_d_light.png")
            c_path = os.path.join(img_save_dir, f"{idx:04d}_c_init.png")
            save_colormap(D_light[0] / args.max_depth_vis, d_path)
            save_colormap(C_up[0], c_path)

    avg_time = total_time / max(1, len(dataset))
    fps = 1.0 / avg_time if avg_time > 0 else 0.0
    avg_metrics = {k: total_metrics[k] / max(1, total_samples) for k in total_metrics}
    summary_path = os.path.join(save_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("=== Test A Results ===\n")
        f.write(f"samples: {len(dataset)}\n")
        f.write(f"RMSE: {avg_metrics['RMSE']:.4f}\n")
        f.write(f"ABS_REL: {avg_metrics['ABS_REL']:.4f}\n")
        f.write(f"DELTA1.02: {avg_metrics['DELTA1.02']:.4f}\n")
        f.write(f"DELTA1.05: {avg_metrics['DELTA1.05']:.4f}\n")
        f.write(f"DELTA1.10: {avg_metrics['DELTA1.10']:.4f}\n")
        f.write(f"time: {avg_time * 1000:.2f} ms/img\n")
        f.write(f"fps: {fps:.2f}\n")
        f.write(f"save_dir: {save_dir}\n")

    print("=== Test A Results ===")
    print(f"samples: {len(dataset)}")
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

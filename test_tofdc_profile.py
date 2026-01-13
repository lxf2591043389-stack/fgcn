import argparse
import os
import time

import torch
import torch.nn.functional as F

from models.light_proxy_net import GlobalProxyNet
from models.heavy_refiner import HeavyRefineHead
from models.scheduler import tile_scheduler
from dataset_factory import build_dataset
import utils
from test_tofdc_config import add_base_args, add_heavy_args, add_scheduler_args, add_stage_args

try:
    from thop import profile
except Exception:
    profile = None

LOW_CONF_THRESH = 0.3

def count_params(model):
    return sum(p.numel() for p in model.parameters())


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


def get_batch(args, device):
    if args.use_dummy:
        I = torch.rand(args.batch_size, 3, 192, 288, device=device)
        D_in = torch.rand(args.batch_size, 1, 192, 288, device=device)
        M = torch.zeros(args.batch_size, 1, 192, 288, device=device)
        return I, D_in, M

    dataset = build_dataset(args.datasets, args.split, args.dataset_cfg, args.data_root)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    batch = next(iter(loader))
    I = batch["rgb"].to(device)
    D_in = batch["depth"].to(device)
    M = batch["mask"].to(device)
    if args.mask_is_valid:
        M = 1.0 - M
    M = M.float()
    return I, D_in, M


def measure_time(fn, device, num_warmup, num_iters):
    if device.type == "cuda":
        torch.cuda.synchronize()
    for _ in range(num_warmup):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_iters):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    total = time.time() - start
    return total / max(1, num_iters)


def main():
    parser = argparse.ArgumentParser(description="TOFDC profile")
    add_base_args(parser)
    add_stage_args(parser)
    add_heavy_args(parser)
    add_scheduler_args(parser)
    parser.add_argument("--num_warmup", default=10, type=int)
    parser.add_argument("--num_iters", default=50, type=int)
    parser.add_argument("--use_dummy", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    light = GlobalProxyNet().to(device)
    heavy = None
    if args.stage in ["B", "C"]:
        heavy = HeavyRefineHead().to(device)

    result_dir = os.path.join(args.project_root, f"result_{args.datasets}")
    light_ckpt = args.light_ckpt or os.path.join(result_dir, "light_best.pth")
    load_state(light, light_ckpt, device)
    if args.stage in ["B", "C"]:
        heavy_ckpt = args.heavy_ckpt or os.path.join(result_dir, "heavy_best.pth")
        load_state(heavy, heavy_ckpt, device)

    light.eval()
    if heavy is not None:
        heavy.eval()

    I, D_in, M = get_batch(args, device)

    print("=== Model ===")
    print(f"light: {light.__class__.__name__}")
    print(light)
    if heavy is not None:
        print(f"heavy: {heavy.__class__.__name__}")
        print(heavy)

    print("=== Params ===")
    light_params = count_params(light)
    heavy_params = count_params(heavy) if heavy is not None else 0
    total_params = light_params + heavy_params
    print(f"light params: {light_params / 1e6:.3f} M")
    if heavy is not None:
        print(f"heavy params: {heavy_params / 1e6:.3f} M")
    print(f"total params: {total_params / 1e6:.3f} M")

    if profile is None:
        print("thop not available, skip FLOPs.")
    else:
        print("=== FLOPs ===")
        flops_light, params_light = profile(light, inputs=(I, D_in, M), verbose=False)
        print(f"light FLOPs: {flops_light / 1e9:.3f} G, params: {params_light / 1e6:.3f} M")
        total_flops = flops_light

        if heavy is not None:
            patch_b = args.batch_size * args.k_max
            I_patch = torch.rand(patch_b, 3, 32, 32, device=device)
            D_patch = torch.rand(patch_b, 1, 32, 32, device=device)
            C_patch = torch.rand(patch_b, 1, 32, 32, device=device)
            flops_heavy, params_heavy = profile(
                heavy, inputs=(I_patch, D_patch, D_patch, C_patch), verbose=False
            )
            print(f"heavy FLOPs (B*K patches): {flops_heavy / 1e9:.3f} G, params: {params_heavy / 1e6:.3f} M")
            total_flops += flops_heavy
        print(f"total FLOPs (light + heavy): {total_flops / 1e9:.3f} G")

    print("=== Inference Time ===")
    with torch.no_grad():
        if args.stage == "A":
            avg_time = measure_time(lambda: light(I, D_in, M), device, args.num_warmup, args.num_iters)
        else:
            avg_time = measure_time(lambda: forward_heavy(light, heavy, I, D_in, M, args), device, args.num_warmup, args.num_iters)
    fps = 1.0 / avg_time if avg_time > 0 else 0.0
    print(f"avg time: {avg_time * 1000:.2f} ms")
    print(f"fps: {fps:.2f}")


if __name__ == "__main__":
    main()

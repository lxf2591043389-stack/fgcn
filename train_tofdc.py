import argparse
import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from config_utils import load_run_config, merge_defaults
from dataset_factory import build_dataset, resolve_dataset_root, load_dataset_config, resolve_dataset_name
import utils
from models.light_proxy_net import GlobalProxyNet
from models.heavy_refiner import HeavyRefineHead
from losses import charbonnier, build_C_gt, bce_loss, edge_aware_smoothness, gradient_consistency_loss

SAFE_THRESH = 0.6
BUFFER = 0.1
GLOBAL_LOSS_WEIGHT = 0.1
SMOOTH_WEIGHT = 0.05
DELTA_SCALE = 1.0
LAMBDA_GRAD = 0.1
PATCH_SIZE = 32
CONTEXT = 48
PAD_CTX = 8
STRIDE = 32
RUN_CONFIG_PATH = "run_config.json"

DEFAULTS = {
    "data_root": "",
    "project_root": "experiments",
    "datasets": "diml",
    "dataset_cfg": "dataset_config.json",
    "batch_size_train": 64,
    "batch_size_eval": 96,
    "batch_size_train_a": 64,
    "batch_size_eval_a": 8,
    "batch_size_train_b": 2,
    "batch_size_eval_b": 1,
    "batch_size_train_c": 2,
    "batch_size_eval_c": 1,
    "num_workers": 0,
    "mask_is_valid": False,
    "stage": "all",
    "epochs_a": 40,
    "epochs_b": 60,
    "epochs_c": 50,
    "k_max": 32,
    "tau_miss": 0.01,
    "dilation_r": 1,
    "lam": 0.7,
    "tau_c": 0.05,
    "theta_l": 0.2,
    "theta_h": 0.8,
    "alpha": 1.5,
    "beta": 2.0,
    "gamma": 2.0,
    "s0": 6.0,
    "delta_max": 0.30,
    "lr_light_a": 1e-3,
    "lr_heavy_b": 1e-4,
    "lr_heavy_c": 1e-5,
    "lr_light_c": 1e-6,
    "lambda_c_a": 0.2,
    "lambda_c_c": 0.03,
    "eta_c": 0.05,
    "weight_decay": 5e-4,
    "grad_clip": 0.5,
    "light_ckpt": "",
    "heavy_ckpt": "",
}

_cfg = load_run_config(RUN_CONFIG_PATH)
merge_defaults(DEFAULTS, _cfg, ["common", "train", "scheduler", "loss"])

if "smooth_weight" in DEFAULTS:
    SMOOTH_WEIGHT = DEFAULTS["smooth_weight"]

def build_w_global(mask):
    w = mask.clone()
    prev = mask
    for r, weight in [(2, 0.7), (4, 0.4), (8, 0.2)]:
        dil = F.max_pool2d(prev, kernel_size=2 * r + 1, stride=1, padding=r)
        ring = torch.clamp(dil - prev, 0.0, 1.0)
        w = w + weight * ring
        prev = dil
    return torch.clamp(w, 0.0, 1.0)


def dilate_mask(mask, r):
    k = 2 * r + 1
    return torch.clamp(F.max_pool2d(mask, kernel_size=k, stride=1, padding=r), 0.0, 1.0)


def set_requires_grad(model, flag):
    if model is None:
        return
    for p in model.parameters():
        p.requires_grad = flag


def load_checkpoint(model, path, device):
    if not path:
        return False
    if not os.path.isfile(path):
        return False
    state = torch.load(path, map_location=device)
    model.load_state_dict(state, strict=True)
    return True


def apply_cfg_overrides(args, cfg):
    if not isinstance(cfg, dict):
        return args
    merged = {}
    for section in ["common", "train", "scheduler", "loss"]:
        values = cfg.get(section, {})
        if isinstance(values, dict):
            merged.update(values)
    for key, value in merged.items():
        if hasattr(args, key):
            setattr(args, key, value)
    return args


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


def forward_v2(light, heavy, I, D_in, M, grad_light, D_max):
    if grad_light:
        D_light, C_init = light(I, D_in, M)
    else:
        with torch.no_grad():
            D_light, C_init = light(I, D_in, M)

    H, W = I.shape[2], I.shape[3]
    D_light = torch.clamp(D_light, 0.0, D_max)
    C_full = F.interpolate(C_init, size=(H, W), mode="bilinear", align_corners=False)
    pooled = F.max_pool2d(M, kernel_size=PATCH_SIZE, stride=STRIDE)
    tiles = (pooled > 0).nonzero(as_tuple=False)
    if tiles.numel() == 0:
        return D_light, D_light, C_full, None, None, None, None

    I_patches = []
    D_in_patches = []
    D_light_patches = []
    C_patches = []
    M_patches = []
    coords = []
    B = I.shape[0]
    tiles_list = tiles.detach().cpu().tolist()
    tiles_by_b = [[] for _ in range(B)]
    for t in tiles_list:
        b, i, j = int(t[0]), int(t[2]), int(t[3])
        tiles_by_b[b].append((i, j))
    max_k = max((len(v) for v in tiles_by_b), default=0)
    if max_k == 0:
        return D_light, D_light, C_full, None, None, None, None

    tiles_tensor = torch.zeros((B, max_k, 2), dtype=torch.long, device=I.device)
    for b in range(B):
        tlist = tiles_by_b[b]
        if not tlist:
            tlist = [(0, 0)]
        if len(tlist) < max_k:
            tlist = tlist + [tlist[-1]] * (max_k - len(tlist))
        tiles_tensor[b] = torch.tensor(tlist, dtype=torch.long, device=I.device)

    for b in range(B):
        for i, j in tiles_by_b[b]:
            y0 = i * STRIDE
            x0 = j * STRIDE
            I_patches.append(I[b:b + 1, :, y0:y0 + PATCH_SIZE, x0:x0 + PATCH_SIZE])
            D_in_patches.append(D_in[b:b + 1, :, y0:y0 + PATCH_SIZE, x0:x0 + PATCH_SIZE])
            D_light_patches.append(D_light[b:b + 1, :, y0:y0 + PATCH_SIZE, x0:x0 + PATCH_SIZE])
            C_patches.append(C_full[b:b + 1, :, y0:y0 + PATCH_SIZE, x0:x0 + PATCH_SIZE])
            M_patches.append(M[b:b + 1, :, y0:y0 + PATCH_SIZE, x0:x0 + PATCH_SIZE])
            coords.append((b, y0, x0))

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
    return D_final_full, D_light, C_full, delta, Dh, M_ctx


def run_epoch(stage, loader, light, heavy, optimizer, args, device, train=True):
    if stage == "A":
        light.train(train)
        if heavy is not None:
            heavy.eval()
        pass
    elif stage == "B":
        light.eval()
        if heavy is not None:
            heavy.train(train)
        pass
    else:
        raise ValueError(f"Unsupported stage: {stage}")

    total_loss = 0.0
    total_step = 0
    error_sum = utils.init_error_metrics()

    tbar = tqdm(loader)
    printed_debug = False
    for batch in tbar:
        I = batch["rgb"].to(device)
        D_in = batch["depth"].to(device)
        D_gt = batch["gt"].to(device)
        M = batch["mask"].to(device)

        if args.mask_is_valid:
            M = 1.0 - M
        M = M.float()

        gt_valid = (D_gt > 1e-6).float()

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            if stage == "A":
                D_light, C_init = light(I, D_in, M)
                L_light = charbonnier(D_light, D_gt, mask=gt_valid)
                C_gt_full = build_C_gt(D_in, D_gt, M, tau_c=args.tau_c)
                C_gt_1_4 = F.avg_pool2d(C_gt_full, kernel_size=4, stride=4)
                L_conf = bce_loss(C_init, C_gt_1_4)
                loss = L_light + args.lambda_c_a * L_conf
                output = D_light
            elif stage == "B":
                D_final, D_light, _, delta, Dh, M_patch = forward_v2(
                    light, heavy, I, D_in, M, grad_light=False, D_max=args.depth_max_m
                )
                if not printed_debug:
                    with torch.no_grad():
                        pooled = F.max_pool2d(M, kernel_size=PATCH_SIZE, stride=STRIDE)
                        tiles_count = int((pooled > 0).sum().item())
                        w_global = build_w_global(M)
                        print(
                            f"[debug][stage B] M mean={M.mean().item():.6f} "
                            f"tiles={tiles_count} W_global mean={w_global.mean().item():.6f}"
                        )
                        head_mean = float(heavy.head.weight.abs().mean().item())
                        print(
                            f"[debug][stage B] heavy.head |W| mean={head_mean:.6f}"
                        )
                        if D_light is not None:
                            dlight_max = float(D_light.max().item())
                            dh_max = float(Dh.max().item()) if Dh is not None else 0.0
                            hole_patch = (M_patch > 0.5) if M_patch is not None else None
                            if hole_patch is not None and hole_patch.any():
                                delta_abs_mean = float(delta.abs()[hole_patch].mean().item())
                            else:
                                delta_abs_mean = float(delta.abs().mean().item())
                            delta_abs_max = float(delta.abs().max().item())
                            print(
                                f"[debug][stage B] D_light max={dlight_max:.6f} "
                                f"Dh max={dh_max:.6f} delta_abs_mean={delta_abs_mean:.6f} "
                                f"delta_abs_max={delta_abs_max:.6f}"
                            )
                eps = 1e-6
                hole_full = (M > 0.5).float()
                ring = torch.clamp(dilate_mask(hole_full, r=3) - hole_full, 0.0, 1.0)
                diff = torch.abs(D_final - D_gt) * hole_full
                L_depth = diff.sum() / (hole_full.sum() + eps)
                L_grad = gradient_consistency_loss(D_final, D_gt, ring, eps=eps)
                if not printed_debug:
                    print(f"[debug][stage B] L_depth={L_depth:.6f} L_grad={L_grad:.6f}")
                    printed_debug = True
                loss = L_depth + LAMBDA_GRAD * L_grad
                output = D_final
            else:
                raise ValueError(f"Unsupported stage: {stage}")

        if train:
            loss.backward()
            if stage == "A":
                torch.nn.utils.clip_grad_norm_(light.parameters(), args.grad_clip)
            elif stage == "B":
                torch.nn.utils.clip_grad_norm_(heavy.parameters(), args.grad_clip)
            else:
                raise ValueError(f"Unsupported stage: {stage}")
            optimizer.step()

        batch_size = I.shape[0]
        total_loss += float(loss.item()) * batch_size
        error_result = utils.evaluate_error(D_gt, output, M, False)
        total_step += batch_size
        error_avg = utils.avg_error(error_sum, error_result, total_step, batch_size)
        tbar.set_description(f"{stage} {'train' if train else 'eval'} loss={total_loss / max(1, total_step):.5f}")

    avg_loss = total_loss / max(1, total_step)
    return avg_loss, error_avg


def train_stage(stage, light, heavy, train_loader, eval_loader, args, device, save_dir, run_id):
    best_rmse = float("inf")
    loss_log = os.path.join(save_dir, f"loss_stage_{stage.lower()}_{run_id}.txt")
    if not os.path.isfile(loss_log):
        with open(loss_log, "w") as f:
            f.write("epoch\ttrain_loss\teval_loss\n")

    scheduler = None
    if stage == "A":
        optimizer = torch.optim.AdamW(light.parameters(), lr=args.lr_light_a, weight_decay=args.weight_decay)
        epochs = args.epochs_a
        ckpt_path = os.path.join(save_dir, "light_best_a.pth")
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-7
        )
    elif stage == "B":
        optimizer = torch.optim.AdamW(
            list(heavy.parameters()),
            lr=args.lr_heavy_b,
            weight_decay=args.weight_decay,
        )
        epochs = args.epochs_b
        ckpt_path = os.path.join(save_dir, "heavy_best_b.pth")
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-6
        )
    else:
        raise ValueError(f"Unsupported stage: {stage}")

    for epoch in range(1, epochs + 1):
        train_loss, train_error = run_epoch(stage, train_loader, light, heavy, optimizer, args, device, train=True)
        eval_loss, eval_error = run_epoch(stage, eval_loader, light, heavy, optimizer, args, device, train=False)

        is_best = eval_error["RMSE"] < best_rmse
        if is_best:
            best_rmse = eval_error["RMSE"]
            if stage == "A":
                torch.save(light.state_dict(), ckpt_path)
                torch.save(light.state_dict(), os.path.join(save_dir, "light_best.pth"))
            elif stage == "B":
                torch.save(heavy.state_dict(), ckpt_path)
                torch.save(heavy.state_dict(), os.path.join(save_dir, "heavy_best.pth"))
            else:
                raise ValueError(f"Unsupported stage: {stage}")

        lr_log = optimizer.param_groups[0]["lr"]
        utils.log_result_lr(save_dir, train_error, epoch, lr_log, False, "train")
        utils.log_result_lr(save_dir, eval_error, epoch, lr_log, is_best, "eval")
        with open(loss_log, "a") as f:
            f.write(f"{epoch}\t{train_loss:.6f}\t{eval_loss:.6f}\n")

        print(
            f"[{stage}] epoch {epoch}/{epochs} train loss {train_loss:.5f} eval loss {eval_loss:.5f} rmse {eval_error['RMSE']:.4f}"
        )
        if scheduler is not None:
            scheduler.step()


def main():
    parser = argparse.ArgumentParser(description="TOFDC training")
    parser.add_argument("--data_root", default=DEFAULTS["data_root"], type=str)
    parser.add_argument("--project_root", default=DEFAULTS["project_root"], type=str)
    parser.add_argument("--datasets", default=DEFAULTS["datasets"], type=str)
    parser.add_argument("--dataset_cfg", default=DEFAULTS["dataset_cfg"], type=str)

    parser.add_argument("--batch_size_train", default=DEFAULTS["batch_size_train"], type=int)
    parser.add_argument("--batch_size_eval", default=DEFAULTS["batch_size_eval"], type=int)
    parser.add_argument("--batch_size_train_a", default=DEFAULTS["batch_size_train_a"], type=int)
    parser.add_argument("--batch_size_eval_a", default=DEFAULTS["batch_size_eval_a"], type=int)
    parser.add_argument("--batch_size_train_b", default=DEFAULTS["batch_size_train_b"], type=int)
    parser.add_argument("--batch_size_eval_b", default=DEFAULTS["batch_size_eval_b"], type=int)
    parser.add_argument("--batch_size_train_c", default=DEFAULTS["batch_size_train_c"], type=int)
    parser.add_argument("--batch_size_eval_c", default=DEFAULTS["batch_size_eval_c"], type=int)
    parser.add_argument("--num_workers", default=DEFAULTS["num_workers"], type=int)
    parser.add_argument("--mask_is_valid", action="store_true", default=DEFAULTS["mask_is_valid"])

    parser.add_argument("--stage", default=DEFAULTS["stage"], choices=["A", "B", "all"])
    parser.add_argument("--epochs_a", default=DEFAULTS["epochs_a"], type=int)
    parser.add_argument("--epochs_b", default=DEFAULTS["epochs_b"], type=int)
    parser.add_argument("--epochs_c", default=DEFAULTS["epochs_c"], type=int)

    parser.add_argument("--k_max", default=DEFAULTS["k_max"], type=int)
    parser.add_argument("--tau_miss", default=DEFAULTS["tau_miss"], type=float)
    parser.add_argument("--dilation_r", default=DEFAULTS["dilation_r"], type=int)
    parser.add_argument("--lam", default=DEFAULTS["lam"], type=float)
    parser.add_argument("--tau_c", default=DEFAULTS["tau_c"], type=float)
    parser.add_argument("--theta_l", default=DEFAULTS["theta_l"], type=float)
    parser.add_argument("--theta_h", default=DEFAULTS["theta_h"], type=float)
    parser.add_argument("--alpha", default=DEFAULTS["alpha"], type=float)
    parser.add_argument("--beta", default=DEFAULTS["beta"], type=float)
    parser.add_argument("--gamma", default=DEFAULTS["gamma"], type=float)
    parser.add_argument("--s0", default=DEFAULTS["s0"], type=float)
    parser.add_argument("--delta_max", default=DEFAULTS["delta_max"], type=float)

    parser.add_argument("--lr_light_a", default=DEFAULTS["lr_light_a"], type=float)
    parser.add_argument("--lr_heavy_b", default=DEFAULTS["lr_heavy_b"], type=float)
    parser.add_argument("--lr_heavy_c", default=DEFAULTS["lr_heavy_c"], type=float)
    parser.add_argument("--lr_light_c", default=DEFAULTS["lr_light_c"], type=float)
    parser.add_argument("--lambda_c_a", default=DEFAULTS["lambda_c_a"], type=float)
    parser.add_argument("--lambda_c_c", default=DEFAULTS["lambda_c_c"], type=float)
    parser.add_argument("--eta_c", default=DEFAULTS["eta_c"], type=float)
    parser.add_argument("--weight_decay", default=DEFAULTS["weight_decay"], type=float)
    parser.add_argument("--grad_clip", default=DEFAULTS["grad_clip"], type=float)

    parser.add_argument("--light_ckpt", default=DEFAULTS["light_ckpt"], type=str)
    parser.add_argument("--heavy_ckpt", default=DEFAULTS["heavy_ckpt"], type=str)

    args = parser.parse_args()
    apply_cfg_overrides(args, _cfg)

    cfg = load_dataset_config(args.dataset_cfg)
    dataset_name = resolve_dataset_name(args.datasets, cfg)
    ds_cfg = cfg.get("datasets", {}).get(dataset_name, {})
    args.depth_max_m = float(ds_cfg.get("depth_max_m", 5.0))

    save_dir = os.path.join(args.project_root, f"result_{args.datasets}")
    utils.log_file_folder_make_lr(save_dir)

    dataset_root = resolve_dataset_root(args.datasets, args.dataset_cfg, args.data_root)
    print(f"==> Loading dataset from: {dataset_root}")
    dataset_train = build_dataset(args.datasets, "train", args.dataset_cfg, args.data_root)
    dataset_test = build_dataset(args.datasets, "test", args.dataset_cfg, args.data_root)

    def make_loaders(batch_train, batch_eval):
        train_loader = DataLoader(
            dataset_train,
            batch_size=batch_train,
            shuffle=True,
            drop_last=True,
            num_workers=args.num_workers,
        )
        eval_loader = DataLoader(
            dataset_test,
            batch_size=batch_eval,
            shuffle=False,
            num_workers=args.num_workers,
        )
        return train_loader, eval_loader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    light = GlobalProxyNet().to(device)
    heavy = None

    run_id = time.strftime("%Y%m%d_%H%M%S")

    if args.stage in ["A", "all"]:
        train_loader, eval_loader = make_loaders(args.batch_size_train_a, args.batch_size_eval_a)
        train_stage("A", light, None, train_loader, eval_loader, args, device, save_dir, run_id)

    if args.stage in ["B", "all"]:
        set_requires_grad(light, False)
        light.eval()

        light_ckpt = args.light_ckpt
        if not light_ckpt:
            light_ckpt = os.path.join(save_dir, "light_best_a.pth")
            if not os.path.isfile(light_ckpt):
                light_ckpt = os.path.join(save_dir, "light_best.pth")
        if not load_checkpoint(light, light_ckpt, device):
            raise FileNotFoundError(f"Light checkpoint not found: {light_ckpt}")

        if heavy is None:
            heavy = HeavyRefineHead(
                alpha=args.alpha,
                beta=args.beta,
                gamma=args.gamma,
                theta_l=args.theta_l,
                theta_h=args.theta_h,
                s0=args.s0,
            ).to(device)
        train_loader, eval_loader = make_loaders(args.batch_size_train_b, args.batch_size_eval_b)
        train_stage("B", light, heavy, train_loader, eval_loader, args, device, save_dir, run_id)

    # Stage C is disabled; use Stage A + B as the final model.


if __name__ == "__main__":
    main()

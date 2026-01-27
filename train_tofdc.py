import argparse
import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from config_utils import load_run_config, merge_defaults
from dataset_factory import build_dataset, resolve_dataset_root
import utils
from models.light_proxy_net import GlobalProxyNet
from models.heavy_refiner import HeavyRefineHead
from models.tiny_fusion_head import TinyFusionHead
from losses import charbonnier, build_C_gt, bce_loss, edge_aware_smoothness

SAFE_THRESH = 0.6
BUFFER = 0.1
GLOBAL_LOSS_WEIGHT = 0.1
SMOOTH_WEIGHT = 0.05
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
    "fusion_ckpt": "",
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
    return F.max_pool2d(mask, kernel_size=2 * r + 1, stride=1, padding=r)


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


def forward_v2(light, heavy, fusion, I, D_in, M, grad_light):
    if grad_light:
        D_light, C_init = light(I, D_in, M)
    else:
        with torch.no_grad():
            D_light, C_init = light(I, D_in, M)

    H, W = I.shape[2], I.shape[3]
    C_full = F.interpolate(C_init, size=(H, W), mode="bilinear", align_corners=False)
    pooled = F.max_pool2d(M, kernel_size=PATCH_SIZE, stride=STRIDE)
    tiles = (pooled > 0).nonzero(as_tuple=False)
    if tiles.numel() == 0:
        return D_light, D_light, C_full, None, None

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
    return D_final, D_light, C_full, w_core, M_core


def run_epoch(stage, loader, light, heavy, fusion, optimizer, args, device, train=True):
    if stage == "A":
        light.train(train)
        if heavy is not None:
            heavy.eval()
        if fusion is not None:
            fusion.eval()
    elif stage == "B":
        light.eval()
        if heavy is not None:
            heavy.train(train)
        if fusion is not None:
            fusion.train(train)
    else:
        raise ValueError(f"Unsupported stage: {stage}")

    total_loss = 0.0
    total_step = 0
    error_sum = utils.init_error_metrics()

    tbar = tqdm(loader)
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
                D_final, _, _, w_core, M_core = forward_v2(light, heavy, fusion, I, D_in, M, grad_light=False)
                eps = 1e-6
                hole_sum = torch.sum(M) + eps
                L_hole = torch.sum(torch.abs(D_final - D_gt) * M) / hole_sum

                Md = dilate_mask(M, r=8)
                dx_pred = D_final[:, :, :, 1:] - D_final[:, :, :, :-1]
                dx_gt = D_gt[:, :, :, 1:] - D_gt[:, :, :, :-1]
                dy_pred = D_final[:, :, 1:, :] - D_final[:, :, :-1, :]
                dy_gt = D_gt[:, :, 1:, :] - D_gt[:, :, :-1, :]
                Md_x = Md[:, :, :, 1:]
                Md_y = Md[:, :, 1:, :]
                grad_x_sum = torch.sum(Md_x) + eps
                grad_y_sum = torch.sum(Md_y) + eps
                L_grad = (
                    torch.sum(torch.abs(dx_pred - dx_gt) * Md_x) / grad_x_sum
                    + torch.sum(torch.abs(dy_pred - dy_gt) * Md_y) / grad_y_sum
                )
                loss = L_hole + 0.3 * L_grad
                output = D_final
            else:
                raise ValueError(f"Unsupported stage: {stage}")

        if train:
            loss.backward()
            if stage == "A":
                torch.nn.utils.clip_grad_norm_(light.parameters(), args.grad_clip)
            elif stage == "B":
                params = list(heavy.parameters()) + list(fusion.parameters())
                torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
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


def train_stage(stage, light, heavy, fusion, train_loader, eval_loader, args, device, save_dir, run_id):
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
            list(heavy.parameters()) + list(fusion.parameters()),
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
        train_loss, train_error = run_epoch(stage, train_loader, light, heavy, fusion, optimizer, args, device, train=True)
        eval_loss, eval_error = run_epoch(stage, eval_loader, light, heavy, fusion, optimizer, args, device, train=False)

        is_best = eval_error["RMSE"] < best_rmse
        if is_best:
            best_rmse = eval_error["RMSE"]
            if stage == "A":
                torch.save(light.state_dict(), ckpt_path)
                torch.save(light.state_dict(), os.path.join(save_dir, "light_best.pth"))
            elif stage == "B":
                torch.save(heavy.state_dict(), ckpt_path)
                torch.save(heavy.state_dict(), os.path.join(save_dir, "heavy_best.pth"))
                torch.save(fusion.state_dict(), os.path.join(save_dir, "fusion_best_b.pth"))
                torch.save(fusion.state_dict(), os.path.join(save_dir, "fusion_best.pth"))
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
    parser.add_argument("--fusion_ckpt", default=DEFAULTS["fusion_ckpt"], type=str)

    args = parser.parse_args()
    apply_cfg_overrides(args, _cfg)

    save_dir = os.path.join(args.project_root, f"result_{args.datasets}")
    utils.log_file_folder_make_lr(save_dir)

    dataset_root = resolve_dataset_root(args.datasets, args.dataset_cfg, args.data_root)
    print(f"==> Loading dataset from: {dataset_root}")
    dataset_train = build_dataset(args.datasets, "train", args.dataset_cfg, args.data_root)
    dataset_test = build_dataset(args.datasets, "train", args.dataset_cfg, args.data_root)

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
    fusion = None

    run_id = time.strftime("%Y%m%d_%H%M%S")

    if args.stage in ["A", "all"]:
        train_loader, eval_loader = make_loaders(args.batch_size_train_a, args.batch_size_eval_a)
        train_stage("A", light, None, None, train_loader, eval_loader, args, device, save_dir, run_id)

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
        if fusion is None:
            fusion = TinyFusionHead().to(device)
        fusion_ckpt = args.fusion_ckpt
        if fusion_ckpt:
            if not load_checkpoint(fusion, fusion_ckpt, device):
                raise FileNotFoundError(f"Fusion checkpoint not found: {fusion_ckpt}")
        train_loader, eval_loader = make_loaders(args.batch_size_train_b, args.batch_size_eval_b)
        train_stage("B", light, heavy, fusion, train_loader, eval_loader, args, device, save_dir, run_id)

    # Stage C is disabled; use Stage A + B as the final model.


if __name__ == "__main__":
    main()

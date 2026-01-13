import os

from config_utils import load_run_config, merge_defaults

RUN_CONFIG_PATH = "run_config.json"

DEFAULTS = {
    "data_root": "",
    "project_root": "experiments",
    "datasets": "tofdc",
    "dataset_cfg": "dataset_config.json",
    "split": "test",
    "stage": "C",
    "batch_size": 1,
    "num_workers": 0,
    "mask_is_valid": False,
    "light_ckpt": "",
    "heavy_ckpt": "",
    "k_max": 32,
    "tau_miss": 0.01,
    "dilation_r": 1,
    "lam": 0.7,
    "delta_max": 0.30,
    "adaptive_k": False,
    "risk_top_ratio": 0.1,
    "k_min": 4,
    "save_images": True,
    "max_depth_vis": 5.0,
}

_cfg = load_run_config(RUN_CONFIG_PATH)
merge_defaults(DEFAULTS, _cfg, ["common", "test", "scheduler", "test_scheduler"])


def add_base_args(parser):
    parser.add_argument("--data_root", default=DEFAULTS["data_root"], type=str)
    parser.add_argument("--project_root", default=DEFAULTS["project_root"], type=str)
    parser.add_argument("--datasets", default=DEFAULTS["datasets"], type=str)
    parser.add_argument("--dataset_cfg", default=DEFAULTS["dataset_cfg"], type=str)
    parser.add_argument("--split", default=DEFAULTS["split"], type=str)
    parser.add_argument("--batch_size", default=DEFAULTS["batch_size"], type=int)
    parser.add_argument("--num_workers", default=DEFAULTS["num_workers"], type=int)
    parser.add_argument("--mask_is_valid", action="store_true", default=DEFAULTS["mask_is_valid"])
    parser.add_argument("--light_ckpt", default=DEFAULTS["light_ckpt"], type=str)
    parser.add_argument("--save_images", action="store_true", default=DEFAULTS["save_images"])
    parser.add_argument("--max_depth_vis", default=DEFAULTS["max_depth_vis"], type=float)


def add_stage_args(parser):
    parser.add_argument("--stage", default=DEFAULTS["stage"], choices=["A", "B", "C"])


def add_heavy_args(parser):
    parser.add_argument("--heavy_ckpt", default=DEFAULTS["heavy_ckpt"], type=str)


def add_scheduler_args(parser):
    parser.add_argument("--k_max", default=DEFAULTS["k_max"], type=int)
    parser.add_argument("--tau_miss", default=DEFAULTS["tau_miss"], type=float)
    parser.add_argument("--dilation_r", default=DEFAULTS["dilation_r"], type=int)
    parser.add_argument("--lam", default=DEFAULTS["lam"], type=float)
    parser.add_argument("--delta_max", default=DEFAULTS["delta_max"], type=float)
    parser.add_argument("--adaptive_k", action="store_true", default=DEFAULTS["adaptive_k"])
    parser.add_argument("--risk_top_ratio", default=DEFAULTS["risk_top_ratio"], type=float)
    parser.add_argument("--k_min", default=DEFAULTS["k_min"], type=int)

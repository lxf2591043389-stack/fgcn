import json
import os

from datasets_tofdc import TOFDC_Dataset
from dataset_diml import DIML_Dataset


def _resolve_config_path(path):
    if os.path.isabs(path):
        return path
    base_dir = os.path.dirname(__file__)
    return os.path.join(base_dir, path)


def _resolve_dataset_root(root, config_path):
    if not root:
        return root
    if os.path.isabs(root):
        return root
    config_path = _resolve_config_path(config_path)
    base_dir = os.path.dirname(config_path)
    return os.path.normpath(os.path.join(base_dir, root))


def load_dataset_config(path):
    config_path = _resolve_config_path(path)
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_dataset_name(name, cfg):
    if name:
        return name
    return cfg.get("default", "tofdc")


def resolve_dataset_root(name, config_path, override_root=None):
    cfg = load_dataset_config(config_path)
    dataset_name = resolve_dataset_name(name, cfg)
    if dataset_name not in cfg["datasets"]:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    ds_cfg = cfg["datasets"][dataset_name]
    root = override_root if override_root else ds_cfg.get("root")
    root = _resolve_dataset_root(root, config_path)
    if not root:
        raise ValueError(f"Dataset root missing for: {dataset_name}")
    return root


def build_dataset(name, split, config_path, override_root=None):
    cfg = load_dataset_config(config_path)
    dataset_name = resolve_dataset_name(name, cfg)
    if dataset_name not in cfg["datasets"]:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    ds_cfg = cfg["datasets"][dataset_name]
    root = override_root if override_root else ds_cfg.get("root")
    root = _resolve_dataset_root(root, config_path)
    if not root:
        raise ValueError(f"Dataset root missing for: {dataset_name}")

    if ds_cfg["type"] == "tofdc":
        return TOFDC_Dataset(root=root, split=split)
    if ds_cfg["type"] == "diml":
        depth_scale = ds_cfg.get("depth_scale", 1000.0)
        return DIML_Dataset(root=root, split=split, depth_scale=depth_scale)

    raise ValueError(f"Unsupported dataset type: {ds_cfg['type']}")

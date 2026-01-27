import argparse
import os

import numpy as np

from dataset_factory import build_dataset, load_dataset_config


def main():
    parser = argparse.ArgumentParser(description="Check max depth (meters) for a dataset split.")
    parser.add_argument("--datasets", default="diml", type=str)
    parser.add_argument("--dataset_cfg", default="dataset_config.json", type=str)
    parser.add_argument("--data_root", default="", type=str)
    parser.add_argument("--split", default="train", type=str)
    parser.add_argument("--num_samples", default=0, type=int, help="0=all, else sample first N.")
    args = parser.parse_args()

    ds = build_dataset(args.datasets, args.split, args.dataset_cfg, args.data_root)
    total = len(ds) if args.num_samples <= 0 else min(len(ds), args.num_samples)
    max_depth = -np.inf
    max_path = None
    count = 0

    for i in range(total):
        sample = ds[i]
        d = sample["gt"].numpy()
        d = d[d > 0]
        if d.size == 0:
            continue
        local_max = float(d.max())
        if local_max > max_depth:
            max_depth = local_max
            max_path = i
        count += 1

    cfg = load_dataset_config(args.dataset_cfg)
    depth_scale = cfg.get("datasets", {}).get(args.datasets, {}).get("depth_scale", None)
    print(f"dataset: {args.datasets}")
    print(f"split: {args.split}")
    print(f"samples_scanned: {total}")
    print(f"valid_samples: {count}")
    print(f"max_depth_m: {max_depth}")
    print(f"max_index: {max_path}")
    print(f"depth_scale: {depth_scale}")


if __name__ == "__main__":
    main()

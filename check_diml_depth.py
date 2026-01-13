import os
from collections import Counter

import numpy as np
from PIL import Image


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "diml"))
LIST_PATH = os.path.join(ROOT_DIR, "diml_train.txt")
MAX_SAMPLES = 50


def iter_list(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 3:
                continue
            yield parts


def read_depth_stats(path):
    img = Image.open(path)
    mode = img.mode
    arr = np.array(img)
    dtype = arr.dtype
    arr_f = arr.astype(np.float32)

    min_v = float(np.min(arr_f))
    max_v = float(np.max(arr_f))
    p95 = float(np.percentile(arr_f, 95))
    zero_ratio = float(np.mean(arr_f == 0))
    neg_ratio = float(np.mean(arr_f < 0))

    return {
        "mode": mode,
        "dtype": str(dtype),
        "min": min_v,
        "max": max_v,
        "p95": p95,
        "zero_ratio": zero_ratio,
        "neg_ratio": neg_ratio,
    }


def resolve_path(root_dir, rel_path):
    if os.path.isabs(rel_path):
        return os.path.normpath(rel_path)
    base_name = os.path.basename(root_dir)
    if rel_path.startswith(base_name + os.sep) or rel_path.startswith(base_name + "/"):
        return os.path.normpath(os.path.join(os.path.dirname(root_dir), rel_path))
    return os.path.normpath(os.path.join(root_dir, rel_path))


def update_summary(summary, stats):
    summary["count"] += 1
    summary["mode_counts"][stats["mode"]] += 1
    summary["dtype_counts"][stats["dtype"]] += 1
    summary["min_vals"].append(stats["min"])
    summary["max_vals"].append(stats["max"])
    summary["p95_vals"].append(stats["p95"])
    summary["zero_ratios"].append(stats["zero_ratio"])
    summary["neg_ratios"].append(stats["neg_ratio"])


def summarize_block(name, summary):
    if summary["count"] == 0:
        return f"{name}: no samples"

    min_v = float(np.min(summary["min_vals"]))
    max_v = float(np.max(summary["max_vals"]))
    p95_med = float(np.median(summary["p95_vals"]))
    zero_avg = float(np.mean(summary["zero_ratios"]))
    neg_avg = float(np.mean(summary["neg_ratios"]))
    dtype_top = summary["dtype_counts"].most_common(1)[0][0]
    mode_top = summary["mode_counts"].most_common(1)[0][0]

    if p95_med > 20.0:
        unit_guess = "likely millimeters (scale 1000)"
    else:
        unit_guess = "likely meters"

    lines = [
        f"{name}: samples={summary['count']}",
        f"  mode={mode_top}, dtype={dtype_top}",
        f"  min={min_v:.3f}, max={max_v:.3f}, p95(median)={p95_med:.3f}",
        f"  zero_ratio(avg)={zero_avg:.4f}, neg_ratio(avg)={neg_avg:.4f}",
        f"  unit guess: {unit_guess}",
    ]
    return "\n".join(lines)


def main():
    if not os.path.isfile(LIST_PATH):
        raise FileNotFoundError(f"List not found: {LIST_PATH}")

    raw_summary = {
        "count": 0,
        "mode_counts": Counter(),
        "dtype_counts": Counter(),
        "min_vals": [],
        "max_vals": [],
        "p95_vals": [],
        "zero_ratios": [],
        "neg_ratios": [],
    }
    gt_summary = {
        "count": 0,
        "mode_counts": Counter(),
        "dtype_counts": Counter(),
        "min_vals": [],
        "max_vals": [],
        "p95_vals": [],
        "zero_ratios": [],
        "neg_ratios": [],
    }

    missing = 0
    total = 0
    for rgb_rel, depth_rel, gt_rel in iter_list(LIST_PATH):
        if total >= MAX_SAMPLES:
            break
        depth_path = resolve_path(ROOT_DIR, depth_rel)
        gt_path = resolve_path(ROOT_DIR, gt_rel)
        if not os.path.isfile(depth_path) or not os.path.isfile(gt_path):
            missing += 1
            continue
        update_summary(raw_summary, read_depth_stats(depth_path))
        update_summary(gt_summary, read_depth_stats(gt_path))
        total += 1

    print(f"list: {LIST_PATH}")
    print(f"checked: {total}, missing: {missing}")
    print(summarize_block("depth_raw", raw_summary))
    print(summarize_block("depth_filled", gt_summary))


if __name__ == "__main__":
    main()

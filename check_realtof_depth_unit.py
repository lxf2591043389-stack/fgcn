import argparse
import os

import numpy as np


def load_depth(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        return np.load(path).astype(np.float32)
    if ext in {".png", ".jpg", ".jpeg", ".tiff", ".tif"}:
        try:
            import cv2
        except ImportError as exc:
            raise ImportError("cv2 is required to read image depth files") from exc
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(path)
        return img.astype(np.float32)
    raise ValueError(f"Unsupported depth file type: {ext}")


def summarize(arr):
    arr = arr.astype(np.float32)
    stats = {
        "shape": arr.shape,
        "dtype": str(arr.dtype),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p1": float(np.percentile(arr, 1)),
        "p99": float(np.percentile(arr, 99)),
        "nonzero_ratio": float(np.count_nonzero(arr) / arr.size),
    }
    return stats


def main():
    parser = argparse.ArgumentParser(description="Check RealToF depth value scale.")
    parser.add_argument("path", type=str, help="Path to a depth file (.npy/.png/.jpg).")
    args = parser.parse_args()

    depth = load_depth(args.path)
    stats = summarize(depth)
    for k, v in stats.items():
        print(f"{k}: {v}")

    if stats["max"] > 100.0:
        print("hint: values look like millimeters (try depth_scale=1000).")
    elif stats["max"] > 10.0:
        print("hint: values look like centimeters/decimeters (check sensor units).")
    else:
        print("hint: values look like meters.")


if __name__ == "__main__":
    main()

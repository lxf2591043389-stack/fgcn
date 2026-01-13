import argparse
import os

import cv2
import numpy as np

DEFAULT_INPUT_PATH = os.path.join("experiments", "result_tofdc", "test_results_a", "img")
DEFAULT_SAMPLE_INDEX = 0
HEATMAP_COLORMAP = cv2.COLORMAP_TURBO
HEATMAP_SUFFIX = "_heatmap.png"


def build_rainbow_lut():
    ramp = np.arange(256, dtype=np.uint8).reshape(256, 1)
    lut = cv2.applyColorMap(ramp, cv2.COLORMAP_RAINBOW)
    return lut.reshape(256, 3)


def decode_colormap_bgr(img_bgr, lut_bgr):
    flat = img_bgr.reshape(-1, 3).astype(np.int16)
    lut_int = lut_bgr.astype(np.int16)
    diff = flat[:, None, :] - lut_int[None, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    idx = np.argmin(dist2, axis=1).astype(np.uint8)
    return idx.reshape(img_bgr.shape[0], img_bgr.shape[1])


def get_confidence_indices(img):
    if img.ndim == 2:
        gray = img.astype(np.uint8)
        return gray, "grayscale"

    if img.shape[2] == 4:
        img = img[:, :, :3]

    if np.all(img[:, :, 0] == img[:, :, 1]) and np.all(img[:, :, 1] == img[:, :, 2]):
        gray = img[:, :, 0].astype(np.uint8)
        return gray, "grayscale"

    lut = build_rainbow_lut()
    idx = decode_colormap_bgr(img, lut)
    return idx, "colormap_rainbow"


def summarize_from_counts(counts):
    total = int(np.sum(counts))
    if total == 0:
        raise RuntimeError("No pixels found for statistics.")

    cum = np.cumsum(counts)

    def percentile_idx(p):
        target = p / 100.0 * (total - 1)
        return int(np.searchsorted(cum, target, side="left"))

    idx_min = int(np.argmax(counts > 0))
    idx_max = int(255 - np.argmax(counts[::-1] > 0))
    idx_mean = float(np.sum(counts * np.arange(256))) / float(total)

    stats = {
        "min": idx_min / 255.0,
        "p10": percentile_idx(10) / 255.0,
        "p25": percentile_idx(25) / 255.0,
        "p50": percentile_idx(50) / 255.0,
        "p75": percentile_idx(75) / 255.0,
        "p90": percentile_idx(90) / 255.0,
        "max": idx_max / 255.0,
        "mean": idx_mean / 255.0,
    }
    return stats


def format_stats(stats):
    lines = [
        f"min: {stats['min']:.4f}",
        f"p10: {stats['p10']:.4f}",
        f"p25: {stats['p25']:.4f}",
        f"p50: {stats['p50']:.4f}",
        f"p75: {stats['p75']:.4f}",
        f"p90: {stats['p90']:.4f}",
        f"max: {stats['max']:.4f}",
        f"mean: {stats['mean']:.4f}",
    ]
    return "\n".join(lines)


def add_text(img_bgr, lines, x=10, y=24, line_gap=18):
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, line in enumerate(lines):
        y_i = y + i * line_gap
        cv2.putText(img_bgr, line, (x + 1, y_i + 1), font, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(img_bgr, line, (x, y_i), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img_bgr


def build_colorbar(height, width=20, colormap=HEATMAP_COLORMAP):
    bar = np.linspace(255, 0, height, dtype=np.uint8).reshape(height, 1)
    bar = np.repeat(bar, width, axis=1)
    bar_color = cv2.applyColorMap(bar, colormap)
    return bar_color


def make_heatmap(idx, stats, colormap=HEATMAP_COLORMAP):
    heat = cv2.applyColorMap(idx.astype(np.uint8), colormap)
    bar = build_colorbar(heat.shape[0], colormap=colormap)
    gap = np.full((heat.shape[0], 6, 3), 255, dtype=np.uint8)
    combined = np.concatenate([heat, gap, bar], axis=1)
    lines = [
        f"min {stats['min']:.3f}",
        f"p10 {stats['p10']:.3f}",
        f"p50 {stats['p50']:.3f}",
        f"p90 {stats['p90']:.3f}",
        f"max {stats['max']:.3f}",
    ]
    combined = add_text(combined, lines)
    h = combined.shape[0]
    combined = add_text(combined, ["1.0"], x=combined.shape[1] - 45, y=18, line_gap=18)
    combined = add_text(combined, ["0.0"], x=combined.shape[1] - 45, y=h - 10, line_gap=18)
    return combined


def heatmap_output_path(sample_path):
    base = os.path.splitext(os.path.basename(sample_path))[0]
    return os.path.join(os.path.dirname(sample_path), f"{base}{HEATMAP_SUFFIX}")


def find_init_pngs(path):
    if os.path.isfile(path):
        return [path]
    if not os.path.isdir(path):
        return []
    hits = []
    for root, _, files in os.walk(path):
        for name in files:
            lower = name.lower()
            if lower.endswith(".png") and "init" in lower:
                hits.append(os.path.join(root, name))
    return sorted(hits)


def main():
    parser = argparse.ArgumentParser(description="Decode confidence map values from saved image or folder.")
    parser.add_argument("--path", default=DEFAULT_INPUT_PATH, type=str)
    args = parser.parse_args()

    file_list = find_init_pngs(args.path)
    if not file_list:
        raise FileNotFoundError(f"No matching images found: {args.path}")

    counts = np.zeros(256, dtype=np.int64)
    mode_counts = {}
    for path in file_list:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise RuntimeError(f"Failed to read image: {path}")

        idx, mode = get_confidence_indices(img)
        mode_counts[mode] = mode_counts.get(mode, 0) + 1
        counts += np.bincount(idx.reshape(-1), minlength=256)

    stats = summarize_from_counts(counts)

    if "colormap_rainbow" in mode_counts:
        print("colormap: cv2.COLORMAP_RAINBOW (matches save_colormap in test_tofdc_a.py)")
        samples = [0, 64, 128, 192, 255]
        lut = build_rainbow_lut()
        print("sample mapping (value -> BGR):")
        for s in samples:
            bgr = tuple(int(x) for x in lut[s])
            print(f"  {s:3d}/255 -> {bgr}")

    print(f"files: {len(file_list)}")
    print(f"modes: {mode_counts}")
    print("stats:")
    print(format_stats(stats))
    print(f"low confidence approx range: [{stats['min']:.4f}, {stats['p10']:.4f}]")
    print(f"high confidence approx range: [{stats['p90']:.4f}, {stats['max']:.4f}]")

    sample_index = min(DEFAULT_SAMPLE_INDEX, len(file_list) - 1)
    sample_path = file_list[sample_index]
    sample_img = cv2.imread(sample_path, cv2.IMREAD_UNCHANGED)
    if sample_img is None:
        raise RuntimeError(f"Failed to read image: {sample_path}")
    sample_idx, sample_mode = get_confidence_indices(sample_img)
    sample_counts = np.bincount(sample_idx.reshape(-1), minlength=256)
    sample_stats = summarize_from_counts(sample_counts)
    heatmap = make_heatmap(sample_idx, sample_stats)
    heatmap_path = heatmap_output_path(sample_path)
    cv2.imwrite(heatmap_path, heatmap)
    print(f"sample: {sample_path}")
    print(f"sample mode: {sample_mode}")
    print("sample stats:")
    print(format_stats(sample_stats))
    print(f"sample heatmap: {heatmap_path}")


if __name__ == "__main__":
    main()

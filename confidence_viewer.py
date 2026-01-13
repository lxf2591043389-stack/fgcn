import argparse
import os

import cv2
import numpy as np


DEFAULT_IMAGE_PATH = os.path.join("experiments", "result_tofdc", "test_results_a", "img", "0000_c_init.png")
LOW_CONF_PERCENTILE = 30
LOW_CONF_COLOR = (0, 0, 255)
LOW_CONF_ALPHA = 0.5


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


def load_confidence_values(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")

    if img.ndim == 2:
        gray = img.astype(np.uint8)
        vals = gray.astype(np.float32) / 255.0
        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        return vals, vis, "grayscale"

    if img.shape[2] == 4:
        img = img[:, :, :3]

    if np.all(img[:, :, 0] == img[:, :, 1]) and np.all(img[:, :, 1] == img[:, :, 2]):
        gray = img[:, :, 0].astype(np.uint8)
        vals = gray.astype(np.float32) / 255.0
        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        return vals, vis, "grayscale"

    lut = build_rainbow_lut()
    idx = decode_colormap_bgr(img, lut)
    vals = idx.astype(np.float32) / 255.0
    return vals, img, "colormap_rainbow"


def apply_low_conf_overlay(base_bgr, mask, color=LOW_CONF_COLOR, alpha=LOW_CONF_ALPHA):
    overlay = base_bgr.copy().astype(np.float32)
    color_arr = np.array(color, dtype=np.float32).reshape(1, 1, 3)
    overlay[mask] = overlay[mask] * (1.0 - alpha) + color_arr * alpha
    return np.clip(overlay, 0, 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description="View confidence map values by mouse hover.")
    parser.add_argument("--path", default=DEFAULT_IMAGE_PATH, type=str)
    args = parser.parse_args()

    if not os.path.isfile(args.path):
        raise FileNotFoundError(f"Image not found: {args.path}")

    vals, vis, mode = load_confidence_values(args.path)
    h, w = vals.shape

    window_name = "confidence_viewer"
    base = vis.copy()
    thresh = float(np.percentile(vals, LOW_CONF_PERCENTILE))
    low_mask = vals <= thresh
    base = apply_low_conf_overlay(base, low_mask)

    def render_text(x, y):
        overlay = base.copy()
        val = float(vals[y, x])
        flag = "LOW" if val <= thresh else "HIGH"
        text = f"x={x} y={y} val={val:.4f} {flag} p{LOW_CONF_PERCENTILE}={thresh:.4f}"
        cv2.putText(overlay, text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(overlay, text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.circle(overlay, (x, y), 3, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(overlay, (x, y), 5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.imshow(window_name, overlay)

    def on_mouse(event, x, y, _flags, _userdata):
        if x < 0 or y < 0 or x >= w or y >= h:
            return
        if event == cv2.EVENT_MOUSEMOVE:
            render_text(x, y)
        elif event == cv2.EVENT_LBUTTONDOWN:
            val = float(vals[y, x])
            print(f"click: x={x} y={y} val={val:.6f}")

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)
    render_text(0, 0)

    print(f"mode: {mode}")
    print(f"low confidence threshold (p{LOW_CONF_PERCENTILE}): {thresh:.4f}")
    print("Low confidence regions are highlighted in red.")
    print("Hover to see value. Left click prints value. Press q or Esc to exit.")
    while True:
        key = cv2.waitKey(20) & 0xFF
        if key in (27, ord("q")):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

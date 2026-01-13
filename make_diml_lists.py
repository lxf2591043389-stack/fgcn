import argparse
import os
from pathlib import Path


def iter_color_files(split_dir):
    split_path = Path(split_dir)
    hr_dir = split_path / "HR"
    if not hr_dir.is_dir():
        return
    for class_dir in sorted(hr_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        color_dir = class_dir / "color"
        if not color_dir.is_dir():
            continue
        for color_path in sorted(color_dir.glob("*.png")):
            yield class_dir, color_path


def build_triplet(class_dir, color_path):
    name = color_path.name
    if name.endswith("_c.png"):
        base = name[:-6]
    else:
        base = os.path.splitext(name)[0]
    raw_name = f"{base}_depth_raw.png"
    filled_name = f"{base}_depth_filled.png"
    raw_path = class_dir / "depth_raw" / raw_name
    filled_path = class_dir / "depth_filled" / filled_name
    return raw_path, filled_path


def to_rel_path(path, base, style):
    rel_path = os.path.relpath(path, base)
    if style == "posix":
        return Path(rel_path).as_posix()
    return rel_path


def write_split(root, split, out_dir, prefix_root, style):
    abs_root = Path(root).resolve()
    base = abs_root.parent if prefix_root else abs_root
    split_dir = abs_root / split
    out_path = Path(out_dir) / f"diml_{split}.txt"

    total = 0
    missing = 0
    lines = []
    for class_dir, color_path in iter_color_files(split_dir):
        raw_path, filled_path = build_triplet(class_dir, color_path)
        if not raw_path.is_file() or not filled_path.is_file():
            missing += 1
            continue
        rgb_rel = to_rel_path(color_path, base, style)
        raw_rel = to_rel_path(raw_path, base, style)
        filled_rel = to_rel_path(filled_path, base, style)
        lines.append(f"{rgb_rel},{raw_rel},{filled_rel}")
        total += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return out_path, total, missing


def main():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "diml"))
    out_dir = root
    prefix_root = True
    path_style = "native"

    train_path, train_total, train_missing = write_split(
        root, "train", out_dir, prefix_root, path_style
    )
    test_path, test_total, test_missing = write_split(
        root, "test", out_dir, prefix_root, path_style
    )

    print(f"train list: {train_path} (pairs: {train_total}, missing: {train_missing})")
    print(f"test list : {test_path} (pairs: {test_total}, missing: {test_missing})")


if __name__ == "__main__":
    main()

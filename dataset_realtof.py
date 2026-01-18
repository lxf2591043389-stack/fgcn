import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class RealToF_Dataset(Dataset):
    def __init__(self, root, split="train", depth_scale=1000.0, target_size=(192, 288)):
        self.root = root
        self.split = split
        self.depth_scale = depth_scale
        self.target_size = target_size

        txt_file_name = f"realtof_{split}.txt"
        txt_path = os.path.join(root, txt_file_name)
        if not os.path.exists(txt_path):
            raise FileNotFoundError(f"List file not found: {txt_path}")

        self.samples = []
        print(f"Loading {split} list from: {txt_path}")
        with open(txt_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) != 3:
                    continue
                self.samples.append(
                    {"rgb": parts[0], "gt": parts[1], "depth": parts[2]}
                )

        print(f"[{split}] total samples: {len(self.samples)}")

    def _resolve_path(self, path_value):
        path_value = os.path.normpath(path_value)
        if not os.path.isabs(path_value):
            return os.path.normpath(os.path.join(self.root, path_value))
        if os.path.exists(path_value):
            return path_value
        drive, tail = os.path.splitdrive(path_value)
        tail = tail.lstrip("\\/")
        return os.path.normpath(os.path.join(self.root, tail))

    def _load_npy(self, path):
        data = np.load(path)
        return data.astype(np.float32)

    def _rgb_to_tensor(self, rgb_np):
        if rgb_np.ndim == 2:
            rgb_np = np.stack([rgb_np] * 3, axis=-1)
        if rgb_np.ndim != 3:
            raise ValueError(f"RGB array must be 2D or 3D, got {rgb_np.shape}")
        if rgb_np.shape[-1] == 3:
            rgb = torch.from_numpy(rgb_np.astype(np.float32)).permute(2, 0, 1)
        elif rgb_np.shape[0] == 3:
            rgb = torch.from_numpy(rgb_np.astype(np.float32))
        else:
            raise ValueError(f"RGB array must have 3 channels, got {rgb_np.shape}")
        if rgb.max() > 1.0:
            scale = 65535.0 if rgb.max() > 255.0 else 255.0
            rgb = rgb / scale
        return rgb

    def _to_single_channel(self, arr, path):
        if arr.ndim == 2:
            arr = arr[None, ...]
        elif arr.ndim == 3:
            if arr.shape[0] == 1:
                pass
            elif arr.shape[-1] == 1:
                arr = np.transpose(arr, (2, 0, 1))
            else:
                raise ValueError(f"Depth array must be single-channel, got {arr.shape} at {path}")
        else:
            raise ValueError(f"Depth array must be 2D or 3D, got {arr.shape} at {path}")
        return arr

    def _resize_tensor(self, tensor, mode="bilinear"):
        tensor = tensor.unsqueeze(0)
        if mode in {"linear", "bilinear", "bicubic", "trilinear"}:
            resized = F.interpolate(
                tensor, size=self.target_size, mode=mode, align_corners=False
            )
        else:
            resized = F.interpolate(tensor, size=self.target_size, mode=mode)
        return resized.squeeze(0)

    def __getitem__(self, index):
        item = self.samples[index]

        rgb_path = self._resolve_path(item["rgb"])
        gt_path = self._resolve_path(item["gt"])
        depth_path = self._resolve_path(item["depth"])

        rgb_np = self._load_npy(rgb_path)
        gt_np = self._load_npy(gt_path)
        depth_np = self._load_npy(depth_path)

        rgb = self._rgb_to_tensor(rgb_np)
        gt = torch.from_numpy(self._to_single_channel(gt_np, gt_path).astype(np.float32))
        depth = torch.from_numpy(self._to_single_channel(depth_np, depth_path).astype(np.float32))

        rgb = self._resize_tensor(rgb, mode="bilinear")
        gt = self._resize_tensor(gt, mode="nearest")
        depth = self._resize_tensor(depth, mode="nearest")

        if self.depth_scale is not None and self.depth_scale != 1.0:
            gt = gt / self.depth_scale
            depth = depth / self.depth_scale

        valid_mask = (depth > 1e-4).float()
        mask = 1.0 - valid_mask
        depth = depth * valid_mask

        return {"rgb": rgb, "gt": gt, "depth": depth, "mask": mask}

    def __len__(self):
        return len(self.samples)

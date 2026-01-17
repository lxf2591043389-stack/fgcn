import os

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset


def read_depth_img(path):
    img = Image.open(path)
    depth_np = np.array(img, dtype=np.float32)
    return Image.fromarray(depth_np)


class DIML_Dataset(Dataset):
    def __init__(self, root, split="train", depth_scale=1000.0):
        self.root = root
        self.split = split
        self.depth_scale = depth_scale

        txt_file_name = f"diml_{split}.txt"
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
                    {"rgb": parts[0], "depth": parts[2], "gt": parts[1]}
                )

        print(f"[{split}] total samples: {len(self.samples)}")

        self.target_size = (192, 288)
        self.rgb_transform = transforms.Compose(
            [
                transforms.Resize(self.target_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.depth_transform = transforms.Compose(
            [transforms.Resize(self.target_size, interpolation=transforms.InterpolationMode.NEAREST)]
        )

    def _resolve_path(self, path_value):
        path_value = os.path.normpath(path_value)
        if not os.path.isabs(path_value):
            return os.path.normpath(os.path.join(self.root, path_value))
        if os.path.exists(path_value):
            return path_value
        drive, tail = os.path.splitdrive(path_value)
        tail = tail.lstrip("\\/")
        return os.path.normpath(os.path.join(self.root, tail))

    def __getitem__(self, index):
        item = self.samples[index]

        rgb_path = self._resolve_path(item["rgb"])
        depth_path = self._resolve_path(item["depth"])
        gt_path = self._resolve_path(item["gt"])

        rgb_pil = Image.open(rgb_path).convert("RGB")
        rgb = self.rgb_transform(rgb_pil)

        gt_pil = read_depth_img(gt_path)
        input_depth_pil = read_depth_img(depth_path)

        gt_pil = self.depth_transform(gt_pil)
        input_depth_pil = self.depth_transform(input_depth_pil)

        gt = torch.from_numpy(np.array(gt_pil)).float().unsqueeze(0)
        depth = torch.from_numpy(np.array(input_depth_pil)).float().unsqueeze(0)

        if self.depth_scale is not None and self.depth_scale != 1.0:
            gt = gt / self.depth_scale
            depth = depth / self.depth_scale

        valid_mask = (depth > 1e-4).float()
        mask = 1.0 - valid_mask
        depth = depth * valid_mask

        return {"rgb": rgb, "gt": gt, "depth": depth, "mask": mask}

    def __len__(self):
        return len(self.samples)

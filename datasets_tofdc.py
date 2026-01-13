import os
import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

def read_depth_img(path):
    """
    专门读取深度图 (16-bit PNG)，保持原始数值，不进行自动归一化
    返回: numpy array (float32), 单位通常是 mm
    """
    img = Image.open(path)
    # 转换为 numpy 数组，保留原始数值 (例如 1000 代表 1000mm)
    depth_np = np.array(img, dtype=np.float32)
    return Image.fromarray(depth_np)

class TOFDC_Dataset(Dataset):
    def __init__(self, root, split='train'):
        """
        Args:
            root: 数据集根目录 (例如 data/TOFDC)
            split: 'train' 或 'test'
        """
        self.root = root
        self.split = split
        
        # 1. 确定索引文件路径
        txt_file_name = f"TOFDC_{split}.txt"
        txt_path = os.path.join(root, txt_file_name)

        if not os.path.exists(txt_path):
            raise FileNotFoundError(f"找不到索引文件: {txt_path}\n请确认文件名是否为 TOFDC_train.txt / TOFDC_test.txt")

        self.samples = []

        # 2. 解析 txt 文件 (格式: RGB路径, GT路径, Depth路径)
        print(f"正在加载 {split} 数据列表...")
        with open(txt_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if not line: continue
                
                parts = line.split(',')
                if len(parts) == 3:
                    # 去除可能存在的空格
                    self.samples.append({
                        'rgb': parts[0].strip(),
                        'gt': parts[1].strip(),
                        'depth': parts[2].strip()
                    })
        
        print(f"[{split}] 数据加载完毕，共 {len(self.samples)} 组样本。")

        # 3. 定义变换
        # === 核心注意 === 
        # 模型的 DepthDecoder.py 里硬编码了层级结构，导致输入分辨率必须是 192(高) x 288(宽)。
        # 如果不Resize到这个尺寸，运行到 Linear 层时会报错。
        self.target_size = (192, 288) 

        self.rgb_transform = transforms.Compose([
            transforms.Resize(self.target_size),
            transforms.ToTensor(),
            # ImageNet 标准归一化，有助于收敛
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # 深度图只需要 Resize，不需要由 transforms 转 Tensor (我们会手动转以控制数值)
        self.depth_transform = transforms.Compose([
            transforms.Resize(self.target_size, interpolation=transforms.InterpolationMode.NEAREST),
        ])

    def __getitem__(self, index):
        item = self.samples[index]
        
        # 拼接完整路径 (使用 normpath 自动处理 Windows 的反斜杠问题)
        rgb_path = os.path.normpath(os.path.join(self.root, item['rgb']))
        gt_path = os.path.normpath(os.path.join(self.root, item['gt']))
        depth_path = os.path.normpath(os.path.join(self.root, item['depth']))

        # 1. 加载 RGB
        rgb_pil = Image.open(rgb_path).convert('RGB')
        rgb = self.rgb_transform(rgb_pil)

        # 2. 加载 Depth 和 GT (使用自定义读取，防止精度丢失)
        gt_pil = read_depth_img(gt_path)
        input_depth_pil = read_depth_img(depth_path)

        # Resize
        gt_pil = self.depth_transform(gt_pil)
        input_depth_pil = self.depth_transform(input_depth_pil)

        # 转为 Tensor
        gt = torch.from_numpy(np.array(gt_pil)).float()
        depth = torch.from_numpy(np.array(input_depth_pil)).float()

        # 增加 Channel 维度: (H, W) -> (1, H, W)
        gt = gt.unsqueeze(0)
        depth = depth.unsqueeze(0)

        # 3. 数值单位转换 (mm -> m)
        # 假设你的 PNG 里存的是毫米 (例如 2500 代表 2.5米)
        # 如果数据已经是米，请注释掉除以 1000
        gt = gt / 1000.0
        depth = depth / 1000.0

        # 4. 生成 Mask (Mask-adaptive Gated Convolution 需要)
        # 规则：深度值 > 0 的地方是有效的。
        # 原代码逻辑：mask=1 表示缺失/无效，mask=0 表示有效 (或者反之，需对齐原代码逻辑)
        # 检查原代码 datasets.py: mask = (depth <= 0.2) -> 1 (即缺失处为1)
        
        # 这里我们定义：输入深度为0的地方是缺失的
        valid_mask = (depth > 1e-4).float()
        
        # 生成 mask: 1代表缺失部分 (供网络填充), 0代表已有深度
        mask = 1.0 - valid_mask

        # 确保输入深度图在缺失区域确实是0
        depth = depth * valid_mask

        return {'rgb': rgb, 'gt': gt, 'depth': depth, 'mask': mask}

    def __len__(self):
        return len(self.samples)
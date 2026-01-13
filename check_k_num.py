import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# ================= 配置区域 =================
# 索引文件路径
TXT_PATH = os.path.join("data", "TOFDC", "TOFDC_train.txt")

# 数据集根目录 (用于拼接相对路径)
# 拼接逻辑: DATA_ROOT + TOFDC_split/train/...
DATA_ROOT = os.path.join("data", "TOFDC")

# 模型输入尺寸 (必须固定为 192x288 以便分块)
IMG_H, IMG_W = 192, 288
PATCH_SIZE = 32

# 网格计算
GRID_H = IMG_H // PATCH_SIZE  # 6
GRID_W = IMG_W // PATCH_SIZE  # 9
TOTAL_PATCHES = GRID_H * GRID_W # 54

# 坏块判定: 只要 patch 里有一个像素是 0，就算坏块
PIXEL_LOSS_THRESHOLD = 1
# ===========================================

def main():
    print(f"==> 1. 读取索引文件: {TXT_PATH}")
    if not os.path.exists(TXT_PATH):
        print("【错误】找不到 txt 文件")
        return

    depth_files = []
    with open(TXT_PATH, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if not line: continue
            # txt格式: rgb_path, gt_path, depth_path
            parts = line.split(',')
            if len(parts) >= 3:
                # 获取相对路径 (第三部分是深度图)
                rel_path = parts[2].strip()
                # 拼接完整路径
                # normpath 可以自动修正 Windows 下的正反斜杠混合问题
                full_path = os.path.normpath(os.path.join(DATA_ROOT, rel_path))
                depth_files.append(full_path)

    print(f"==> 共找到 {len(depth_files)} 个样本")
    print(f"==> 2. 开始分块扫描 (Grid: {GRID_H}x{GRID_W}={TOTAL_PATCHES}) ...")
    
    bad_patch_counts = []
    
    # 遍历所有图片
    for fpath in tqdm(depth_files):
        if not os.path.exists(fpath):
            # 简单跳过不存在的文件，不中断程序
            continue
            
        # 读取 16-bit 深度图 (flag=-1)
        img = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
        if img is None: continue
        
        # 强制 Resize 到 192x288 (模拟模型输入)
        # 使用最近邻插值 (INTER_NEAREST) 以保持 0 值不被模糊
        if img.shape[:2] != (IMG_H, IMG_W):
            img = cv2.resize(img, (IMG_W, IMG_H), interpolation=cv2.INTER_NEAREST)
            
        # 生成 Mask (1=空洞/无效, 0=有效)
        # 假设 0 是无效值
        loss_mask = (img == 0).astype(np.uint8)
        
        # 统计这张图的坏块数
        n_miss = 0
        
        for r in range(GRID_H):
            for c in range(GRID_W):
                # 切片坐标
                y1, y2 = r * PATCH_SIZE, (r + 1) * PATCH_SIZE
                x1, x2 = c * PATCH_SIZE, (c + 1) * PATCH_SIZE
                
                # 提取 Patch
                patch = loss_mask[y1:y2, x1:x2]
                
                # 判定: 如果丢失像素 >= 阈值，则记为需要处理的坏块
                if np.sum(patch) >= PIXEL_LOSS_THRESHOLD:
                    n_miss += 1
        
        bad_patch_counts.append(n_miss)

    # ================= 3. 统计分析 =================
    if not bad_patch_counts:
        print("未处理任何有效图片。")
        return

    counts = np.array(bad_patch_counts)
    
    p50 = np.percentile(counts, 50)
    p90 = np.percentile(counts, 90)
    p95 = np.percentile(counts, 95)
    p99 = np.percentile(counts, 99)
    max_val = np.max(counts)
    
    print("\n" + "="*40)
    print("           坏块分布统计 (Train Set)         ")
    print("="*40)
    print(f"P50 (中位数) : {int(p50)}")
    print(f"P90          : {int(p90)}")
    print(f"P95 (推荐)   : {int(p95)}")
    print(f"P99 (极端)   : {int(p99)}")
    print(f"Max (全黑)   : {int(max_val)} / {TOTAL_PATCHES}")
    print("-" * 40)
    
    # 决策建议
    def suggest_k(val):
        # 向上取整到 8 的倍数 (对 GPU 友好)
        return int(8 * np.ceil(val / 8))

    k_95 = suggest_k(p95)
    k_99 = suggest_k(p99)
    
    print(f"💡 系统参数 K_max 建议:")
    print(f"   覆盖 95% 场景 -> 设为 {k_95} (原始P95={int(p95)})")
    print(f"   覆盖 99% 场景 -> 设为 {k_99} (原始P99={int(p99)})")
    print("="*40)

    # ================= 4. 画直方图 =================
    plt.figure(figsize=(10, 6))
    # bin设置: 0到55，确保包含所有可能的块数
    plt.hist(counts, bins=range(0, TOTAL_PATCHES + 2), 
             color='#4c72b0', edgecolor='black', alpha=0.7, align='left')
    
    plt.axvline(p95, color='red', linestyle='--', linewidth=2, label=f'P95: {int(p95)}')
    plt.axvline(p99, color='orange', linestyle='--', linewidth=2, label=f'P99: {int(p99)}')
    
    plt.title(f'Distribution of Bad Patches per Image\n(Total Patches: {TOTAL_PATCHES}, Grid: 32x32)', fontsize=14)
    plt.xlabel('Number of Patches with Holes', fontsize=12)
    plt.ylabel('Image Count', fontsize=12)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    save_path = 'train_patch_stats.png'
    plt.savefig(save_path)
    print(f"📊 图表已保存至: {os.path.abspath(save_path)}")
    plt.show()

if __name__ == '__main__':
    main()

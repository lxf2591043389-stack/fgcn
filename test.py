import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from thop import profile
import logging

# 引入你的项目模块
from model.Network import Network
from datasets_tofdc import TOFDC_Dataset
import utils

# ================= ⚙️ 配置区域 (根据你的环境修改) =================
# 1. 模型路径
MODEL_PATH = os.path.join("experiments", "result_tofdc", "best_model.pth")
# 2. 数据集根目录
DATA_ROOT = os.path.join("data", "TOFDC")
# 3. 结果保存目录 (自动在模型所在目录创建 test_results 文件夹)
SAVE_DIR = os.path.join(os.path.dirname(MODEL_PATH), "test_results_v1")

# 4. 图像参数
INPUT_HEIGHT = 192
INPUT_WIDTH = 288
MAX_DEPTH_VIS = 5.0  # 可视化时的最大深度(米)，超过这个距离的颜色会饱和。用于把深度归一化到0-1之间绘图
# =================================================================

def setup_logger(save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    log_format = '%(asctime)s - %(message)s'
    logging.basicConfig(filename=os.path.join(save_dir, 'test_log.txt'), 
                        level=logging.INFO, format=log_format, filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)
    return logging.getLogger('')

def load_model(model, path, logger):
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到模型文件: {path}")
    
    logger.info(f"==> Loading model from: {path}")
    checkpoint = torch.load(path)
    
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    state_dict = utils.remove_moudle(state_dict)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def calculate_flops_params(model, logger):
    logger.info("==> Calculating FLOPs and Params...")
    # 构造虚拟输入 (Batch=1, Channel, Height, Width)
    dummy_rgb = torch.randn(1, 3, INPUT_HEIGHT, INPUT_WIDTH).cuda()
    dummy_depth = torch.randn(1, 1, INPUT_HEIGHT, INPUT_WIDTH).cuda()
    dummy_mask = torch.randn(1, 1, INPUT_HEIGHT, INPUT_WIDTH).cuda()

    try:
        flops, params = profile(model, inputs=(dummy_rgb, dummy_depth, dummy_mask), verbose=False)
        logger.info(f"--------------------------------")
        logger.info(f"Params (参数量): {params / 1e6:.2f} M")
        logger.info(f"FLOPs (计算量) : {flops / 1e9:.2f} G")
        logger.info(f"--------------------------------")
        return params, flops
    except Exception as e:
        logger.warning(f"FLOPs calculation failed: {e}")
        return 0, 0

def test():
    # 1. 设置保存路径和日志
    logger = setup_logger(SAVE_DIR)
    img_save_dir = os.path.join(SAVE_DIR, "img")
    if not os.path.exists(img_save_dir):
        os.makedirs(img_save_dir)

    # 2. 初始化设备和模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Network().to(device)
    
    # 3. 加载权重
    model = load_model(model, MODEL_PATH, logger)

    # 4. 计算 FLOPs 和 参数量
    calculate_flops_params(model, logger)

    # 5. 加载测试集
    # 注意：确保这里使用的是 test split
    test_dataset = TOFDC_Dataset(root=DATA_ROOT, split='test') 
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    logger.info(f"==> Test dataset size: {len(test_dataset)}")

    # 6. 初始化指标统计
    total_metrics = utils.init_error_metrics()
    total_samples = 0
    total_inference_time = 0.0

    logger.info(f"==> Starting inference and saving images to {img_save_dir} ...")
    
    # 预热 GPU
    dummy_input = torch.randn(1, 3, INPUT_HEIGHT, INPUT_WIDTH).to(device)
    dummy_d = torch.randn(1, 1, INPUT_HEIGHT, INPUT_WIDTH).to(device)
    dummy_m = torch.randn(1, 1, INPUT_HEIGHT, INPUT_WIDTH).to(device)
    for _ in range(10):
        _ = model(dummy_input, dummy_d, dummy_m)

    with torch.no_grad():
        for i, data in tqdm(enumerate(test_loader), total=len(test_loader)):
            # 准备数据
            color = data['rgb'].to(device)
            depth = data['depth'].to(device)
            mask = data['mask'].to(device)
            gt = data['gt'].to(device)

            # --- 计时开始 ---
            torch.cuda.synchronize()
            start_time = time.time()
            
            # 推理
            output = model(color, depth, mask)
            
            torch.cuda.synchronize()
            end_time = time.time()
            # --- 计时结束 ---

            total_inference_time += (end_time - start_time)

            # 计算误差
            metrics = utils.evaluate_error(gt, output, mask)
            for key in total_metrics.keys():
                total_metrics[key] += metrics[key] * color.size(0)
            
            total_samples += color.size(0)

            # --- 保存可视化图片 ---
            # utils.save_eval_img 需要输入 0-1 之间的 tensor 才能正确画出彩虹图
            # 我们的数据单位是米，所以要除以 MAX_DEPTH_VIS (比如 5米) 进行归一化
            # 如果深度超过 5米，就会被截断显示为最大颜色
            
            # 使用 try-except 防止保存图片出错中断测试
            try:
                utils.save_eval_img(
                    img_save_dir, 
                    i, 
                    depth[0].cpu() / MAX_DEPTH_VIS, 
                    gt[0].cpu() / MAX_DEPTH_VIS, 
                    output[0].cpu() / MAX_DEPTH_VIS
                )
            except Exception as e:
                print(f"Error saving image {i}: {e}")

    # 7. 计算最终平均值
    avg_metrics = {}
    for key in total_metrics.keys():
        avg_metrics[key] = total_metrics[key] / total_samples

    avg_time_per_img = total_inference_time / total_samples
    fps = 1.0 / avg_time_per_img

    logger.info("\n================ Evaluation Results ================")
    logger.info(f"Model: {os.path.basename(MODEL_PATH)}")
    logger.info(f"Total Samples: {total_samples}")
    logger.info("----------------------------------------------------")
    logger.info(f"RMSE       : {avg_metrics['RMSE']:.4f} m (Lower is better)")
    logger.info(f"MAE (ABS)  : {avg_metrics['ABS_REL']:.4f}")
    logger.info(f"Delta 1.02 : {avg_metrics['DELTA1.02']:.4f} (Higher is better)")
    logger.info(f"Delta 1.05 : {avg_metrics['DELTA1.05']:.4f}")
    logger.info(f"Delta 1.10 : {avg_metrics['DELTA1.10']:.4f}")
    logger.info("----------------------------------------------------")
    logger.info(f"Inference Time: {avg_time_per_img*1000:.2f} ms / image")
    logger.info(f"FPS           : {fps:.2f}")
    logger.info("====================================================")
    
    print(f"测试完成！结果已保存在: {SAVE_DIR}")

if __name__ == '__main__':
    test()

# run_config.json 参数说明

除非使用绝对路径，否则所有路径均相对项目根目录。

## common
- data_root: 数据集根目录覆盖；空字符串表示使用 dataset_config.json
- project_root: 实验输出根目录（权重、日志等）
- datasets: 使用的数据集名称（必须在 dataset_config.json 中存在）
- dataset_cfg: dataset_config.json 的路径
- num_workers: DataLoader 的 worker 数量
- mask_is_valid: 为 true 时反转 mask（把 mask 当成有效区域）
- light_ckpt: 指定 light 权重路径（可选）
- heavy_ckpt: 指定 heavy 权重路径（可选）
- save_images: 测试时是否保存可视化图片
- max_depth_vis: 可视化归一化使用的最大深度

## test
- split: 数据集划分名称，如 "test"
- stage: 测试阶段选择（"A"/"B"/"C"）
- batch_size: 测试 batch size

## scheduler
- k_max: heavy 精修的最大 tile 数
- tau_miss: 空洞比例阈值（用于 tile 选择）
- dilation_r: tile 选择的膨胀半径
- lam: tile 得分混合系数
- delta_max: 可选上限（若在其他地方使用）

## train
- batch_size_train: 训练 batch size
- batch_size_eval: 训练期间评估 batch size
- stage: 训练阶段（"A"/"B"/"C"/"all"）
- epochs_a/b/c: 各阶段训练轮数
- tau_c: 置信度 GT 的尺度系数
- theta_l/theta_h: heavy 中置信度裁剪范围
- alpha/beta/gamma: FGNC 权重指数
- s0: FGNC 支持度缩放系数
- lr_light_a: Stage A 的 light 学习率
- lr_heavy_b: Stage B 的 heavy 学习率
- lr_heavy_c: Stage C 的 heavy 学习率
- lr_light_c: Stage C 的 light 解码器学习率
- lambda_c_a: Stage A 置信度损失权重
- lambda_c_c: Stage C 置信度损失权重
- eta_c: Stage C 的 light 损失权重
- weight_decay: AdamW 权重衰减
- grad_clip: 梯度裁剪阈值

## loss
- smooth_weight: 边缘感知平滑损失权重
- final_weight: Stage C 的 L_final 权重

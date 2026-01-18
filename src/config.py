import os
from datetime import datetime


class Config:
    # --- 1. 路径设置 ---
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(ROOT_DIR, "数据集")
    OUTPUT_DIR = os.path.join(ROOT_DIR, "输出")
    MODEL_DIR = os.path.join(ROOT_DIR, "model")
    LOG_DIR = os.path.join(ROOT_DIR, "logs")

    # 确保目录存在
    for d in [OUTPUT_DIR, MODEL_DIR, LOG_DIR]:
        if not os.path.exists(d): os.makedirs(d)

    # --- 2. 物理/数据参数 ---
    INPUT_LEN = 1201
    WAVELENGTH_RANGE = (600, 1800)  # 起始-终止波长
    CLASS_MAP = {'CV': 0, 'MG': 1, 'MB': 2}
    CLASS_NAMES = ['CV', 'MG', 'MB']

    # --- 3. 训练超参数 ---
    SEED = 42  # 随机种子, 可以随意修改
    BATCH_SIZE = 320
    EPOCHS = 40  # 迭代次数
    BASE_LR = 0.0001  # 基础学习率
    MASK_LR_SCALE = 40.0  # 掩码学习率倍数（掩码用于降低背景噪声的影响）
    L1_LAMBDA = 0.0001  # L1正则化参数(用于掩码稀疏化)

    # --- 4. 流程控制开关 ---
    IS_GENERATE_TRAIN_DATA = True  # 是否重新生成训练数据(原始集 -> 经过基线校准和归一化得到的 训练集、测试集, 原始数据如果有变动则启用)
    IS_RETRAIN = True  # 是否重新训练(每次修改配置或模型后都要重新启用)
    IS_USE_SINGLE_SUBSTANCE = True  # 是否使用纯净物作为训练集一部分(启用能更好学习纯净物特征)

    # --- 5. 增强配置 (x, y, z)表示以 x:y 的比例混合, z个混合样本, 也可以随意增设例如(0.5,0.95,10) ---
    MIX_CONFIG = [
        (1, 9, 10), (9, 1, 10),
        (2, 8, 20), (8, 2, 20),
        (3, 7, 30), (7, 3, 30),
        (4, 6, 30), (6, 4, 30),
        (5, 5, 50)
    ]

    # --- 6. 评估参数 ---
    PRED_THRESHOLD = 0.05  # 阈值, 超过这个值认为存在物质
    TOP_N_PEAKS = 10  # 最后识别的关键特征波长数

    @staticmethod
    def get_run_name():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"run_{timestamp}_ep{Config.EPOCHS}_bs{Config.BATCH_SIZE}"
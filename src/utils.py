import os
import torch
import random
import numpy as np
import logging
import sys
from scipy.interpolate import interp1d
from scipy import sparse
from scipy.sparse.linalg import spsolve


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    if not torch.cuda.is_available():
        print("【警告】未检测到 GPU，将使用 CPU 运行 (速度较慢)。")
        return torch.device("cpu")
    print(f"✅ 使用设备: {torch.cuda.get_device_name(0)}")
    return torch.device("cuda")


def setup_logger(log_dir, log_name):
    logger = logging.getLogger("SERS_Logger")
    logger.setLevel(logging.INFO)
    logger.handlers = []  # 清空旧 Handler

    log_path = os.path.join(log_dir, f"{log_name}.log")

    # File Handler
    fh = logging.FileHandler(log_path, encoding='utf-8')
    fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(fh)

    # Stream Handler
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(sh)

    print(f"📝 日志已初始化: {log_path}")
    return logger


# --- 数学/光谱工具 ---

def get_target_axis(start=600, end=1800, length=1201):
    return np.linspace(start, end, length)


def align_spectra(wavelengths, intensities, target_axis):
    """线性插值对齐波长"""
    f = interp1d(wavelengths, intensities, kind='linear', bounds_error=False, fill_value=0)
    return f(target_axis)


def normalize_spectrum(spec):
    """
    Max 归一化 (保留基线)
    假设输入光谱已经过基线校准 (Baseline Corrected)，基线位于 0 附近。
    此方法只进行缩放，不进行平移，防止因噪声底不同导致的基线漂移。
    """
    mx = spec.max()
    # 防止除以0或处理全0/全负光谱
    if mx > 1e-6:
        return spec / mx
    return np.zeros_like(spec)


def baseline_als(y, lam=1e5, p=0.01, niter=10):
    """
    Asymmetric Least Squares Smoothing (去基线算法)
    参考文献: Eilers, P. H. C. and Boelens, H. F. M. (2005)
    :param y: 原始光谱强度
    :param lam: 平滑参数 (lambda)，越大基线越平滑
    :param p: 不对称参数，越小越贴近底部
    :param niter: 迭代次数
    """
    L = len(y)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
    w = np.ones(L)
    for i in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return y - z  # 返回扣除基线后的光谱

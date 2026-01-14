import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
import os
import glob
from scipy.signal import find_peaks

# 导入项目配置和工具
from config import Config
from utils import get_target_axis, align_spectra

# =================配置区域=================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'Heiti TC']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['svg.fonttype'] = 'none'

# 物质定义
SUBSTANCES = ['CV', 'MG', 'MB']
NAMES = ['Crystal Violet (CV)', 'Malachite Green (MG)', 'Methylene Blue (MB)']
COLORS = ['purple', 'green', 'blue']
KEYWORD_MAP = {'结晶紫': 'CV', 'CV': 'CV', '孔雀石绿': 'MG', 'MG': 'MG', '亚甲基蓝': 'MB', 'MB': 'MB'}


def load_raw_data(substance_key):
    """
    从原始数据文件夹加载指定物质的所有样本。
    逻辑参考 data_manager._process_pure_substances，但不做归一化。
    """
    raw_dir = os.path.join(Config.DATA_DIR, "原始数据", "单物质")
    if not os.path.exists(raw_dir):
        print(f"⚠️ 找不到原始数据目录: {raw_dir}")
        return None, None

    target_axis = get_target_axis(Config.WAVELENGTH_RANGE[0], Config.WAVELENGTH_RANGE[1], Config.INPUT_LEN)
    all_spectra = []

    # 遍历原始数据目录下的文件夹
    for folder in os.listdir(raw_dir):
        folder_path = os.path.join(raw_dir, folder)
        if not os.path.isdir(folder_path): continue

        # 检查文件夹名是否匹配当前物质 (e.g., "CV" in "结晶紫_10-5")
        # 使用 data_manager 中的映射逻辑
        category = next((v for k, v in KEYWORD_MAP.items() if k in folder), None)

        # 只有当文件夹对应的类别就是我们当前要找的 substance_key 时才读取
        if category != substance_key:
            continue

        # 读取该文件夹下的所有 CSV
        files = glob.glob(os.path.join(folder_path, "*.csv"))
        for f in files:
            try:
                df = pd.read_csv(f)
                waves = df.iloc[:, 0].values
                # 遍历文件中的每一列光谱
                for i in range(1, df.shape[1]):
                    raw_intensity = df.iloc[:, i].values
                    # 仅做对齐，不做归一化，保留原始强度
                    aligned_spec = align_spectra(waves, raw_intensity, target_axis)
                    all_spectra.append(aligned_spec)
            except Exception as e:
                print(f"Error reading {f}: {e}")
                continue

    if not all_spectra:
        return None, None

    return target_axis, np.array(all_spectra).T  # Shape: (1201, N_samples)


def plot_origin_data_grid(root_dir):
    print("🚀 开始绘制原始数据分布图...")
    fig, axes = plt.subplots(2, 2, figsize=(18, 12), constrained_layout=True)
    axes = axes.flatten()

    mean_spectra_store = {}

    # --- 循环绘制前三张 (单物质) ---
    for idx, sub_key in enumerate(SUBSTANCES):
        ax = axes[idx]

        # 加载原始数据
        wavelengths, intensities = load_raw_data(sub_key)

        if wavelengths is None:
            ax.text(0.5, 0.5, f"No Raw Data Found: {sub_key}", ha='center', va='center')
            continue

        n_samples = intensities.shape[1]

        # 计算平均光谱
        mean_spec = intensities.mean(axis=1)
        mean_spectra_store[sub_key] = (wavelengths, mean_spec)

        # 1. 寻峰 (用于X轴标注)
        # 注意：原始数据强度可能很大，prominence 需要动态调整或基于归一化后的副本计算
        # 这里为了简单，我们基于 mean_spec 的相对高度来寻峰
        norm_mean = (mean_spec - mean_spec.min()) / (mean_spec.max() - mean_spec.min())
        peaks, _ = find_peaks(norm_mean, distance=30, prominence=0.05)
        peak_wavelengths = wavelengths[peaks]

        # 2. 绘制所有样本
        try:
            cmap = matplotlib.colormaps['coolwarm']
        except AttributeError:
            cmap = plt.get_cmap('coolwarm')

        # 为了避免渲染太慢，如果样本过多，可以设置 alpha 更低
        alpha_val = 0.5 if n_samples < 200 else 0.2

        for sample_i in range(n_samples):
            norm_idx = sample_i / (n_samples - 1) if n_samples > 1 else 0.5
            color = cmap(norm_idx)
            ax.plot(wavelengths, intensities[:, sample_i], color=color, linewidth=0.8, alpha=alpha_val)

        # 3. 设置样式
        ax.set_xticks(peak_wavelengths)
        ax.set_xticklabels([f"{w:.0f}" for w in peak_wavelengths], rotation=90, fontsize=8)

        # 辅助线
        for pw in peak_wavelengths:
            ax.axvline(x=pw, color='gray', linestyle='--', alpha=0.2, linewidth=0.5)

        ax.set_title(f"{NAMES[idx]} (Raw Data, n={n_samples})\nSample Index Gradient: Blue->Red", fontsize=14,
                     fontweight='bold')
        ax.set_ylabel("Intensity (Counts)")  # 原始强度
        ax.set_xlabel(r"Wavelength ($cm^{-1}$ / nm)")
        ax.grid(axis='y', linestyle='--', alpha=0.3)

    # --- 绘制第四张 (合并对比图) ---
    ax_merge = axes[3]
    for i, sub_key in enumerate(SUBSTANCES):
        if sub_key in mean_spectra_store:
            waves, mean_spec = mean_spectra_store[sub_key]

            # 绘制平均光谱
            ax_merge.plot(waves, mean_spec, label=f"{NAMES[i]} (Mean)", color=COLORS[i], linewidth=2)

            # 标记最高峰
            norm_mean = (mean_spec - mean_spec.min()) / (mean_spec.max() - mean_spec.min())
            peaks, _ = find_peaks(norm_mean, distance=50, prominence=0.1)
            if len(peaks) > 0:
                max_peak_idx = peaks[np.argmax(mean_spec[peaks])]
                max_peak_wave = waves[max_peak_idx]
                ax_merge.annotate(f"{max_peak_wave:.0f}",
                                  xy=(max_peak_wave, mean_spec[max_peak_idx]),
                                  xytext=(0, 10), textcoords='offset points',
                                  ha='center', color=COLORS[i], fontsize=8, fontweight='bold')

    ax_merge.set_title("Class Comparison (Raw Mean Spectra)", fontsize=14, fontweight='bold')
    ax_merge.legend()
    ax_merge.set_xlabel(r"Wavelength ($cm^{-1}$)")
    ax_merge.set_ylabel("Intensity (Counts)")
    ax_merge.grid(True, alpha=0.3)

    # --- 保存 ---
    save_dir = os.path.join(root_dir, "输出")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = os.path.join(save_dir, "原始数据分布图_Raw.pdf")
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    print(f"✅ 绘图完成！图片已保存至: {save_path}")


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    plot_origin_data_grid(current_dir)
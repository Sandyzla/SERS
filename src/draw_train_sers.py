import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm
import pandas as pd
import numpy as np
import os
import glob
from scipy.signal import find_peaks

# =================配置区域=================
# 设置字体，防止中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'Heiti TC']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 【新增配置】设置 SVG 字体为无（即保留文本属性），方便后续在 Illustrator/Inkscape 中编辑文字
plt.rcParams['svg.fonttype'] = 'none'

# 物质列表 (文件名关键词)
SUBSTANCES = ['CV', 'MG', 'MB']
# 对应的显示名称
NAMES = ['Crystal Violet (CV)', 'Malachite Green (MG)', 'Methylene Blue (MB)']
# 对应的颜色主调 (用于第四张合并图)
COLORS = ['purple', 'green', 'blue']


def get_data_path(root_dir, substance_key):
    """根据物质关键词寻找对应的csv文件"""
    search_path = os.path.join(root_dir, "数据集", "训练集", "纯净物", f"*{substance_key}*.csv")
    files = glob.glob(search_path)
    if not files:
        return None
    return files[0]  # 返回找到的第一个文件


def plot_spectra_grid(root_dir):
    fig, axes = plt.subplots(2, 2, figsize=(18, 12), constrained_layout=True)
    axes = axes.flatten()  # 展平为 1维数组 [ax0, ax1, ax2, ax3]

    # 存储平均光谱用于最后一张图
    mean_spectra_store = {}

    # --- 循环绘制前三张 (单物质所有样本) ---
    for idx, sub_key in enumerate(SUBSTANCES):
        ax = axes[idx]
        file_path = get_data_path(root_dir, sub_key)

        if not file_path:
            ax.text(0.5, 0.5, f"Missing Data: {sub_key}", ha='center', va='center')
            continue

        # 读取数据
        df = pd.read_csv(file_path)
        wavelengths = df['Wavelength'].values
        # 提取强度数据 (排除第一列Wavelength)
        intensities = df.iloc[:, 1:].values  # shape: (n_wavelengths, n_samples)
        n_samples = intensities.shape[1]

        # 计算平均光谱用于寻峰
        mean_spec = intensities.mean(axis=1)
        mean_spectra_store[sub_key] = (wavelengths, mean_spec)

        # 1. 寻找特征峰 (用于X轴标注)
        peaks, _ = find_peaks(mean_spec, distance=30, prominence=0.05)
        peak_wavelengths = wavelengths[peaks]

        # 2. 绘制所有样本 (使用渐变色)
        try:
            cmap = matplotlib.colormaps['coolwarm']
        except AttributeError:
            cmap = plt.get_cmap('coolwarm')

        for sample_i in range(n_samples):
            norm_idx = sample_i / (n_samples - 1) if n_samples > 1 else 0.5
            color = cmap(norm_idx)
            ax.plot(wavelengths, intensities[:, sample_i], color=color, linewidth=0.8, alpha=0.5)

        # 3. 设置X轴刻度为特征峰位置
        ax.set_xticks(peak_wavelengths)
        ax.set_xticklabels([f"{w:.0f}" for w in peak_wavelengths], rotation=90, fontsize=8)

        # 绘制竖虚线辅助看峰
        for pw in peak_wavelengths:
            ax.axvline(x=pw, color='gray', linestyle='--', alpha=0.2, linewidth=0.5)

        # 标题和标签
        ax.set_title(f"{NAMES[idx]} (n={n_samples})\nSample Index Gradient: Blue->Red", fontsize=14, fontweight='bold')
        ax.set_ylabel("Normalized Intensity")
        ax.set_xlabel(r"Wavelength ($cm^{-1}$ / nm)")

        ax.grid(axis='y', linestyle='--', alpha=0.3)

    # --- 绘制第四张 (合并对比图) ---
    ax_merge = axes[3]
    for i, sub_key in enumerate(SUBSTANCES):
        if sub_key in mean_spectra_store:
            waves, mean_spec = mean_spectra_store[sub_key]

            # 绘制平均光谱
            ax_merge.plot(waves, mean_spec, label=f"{NAMES[i]} (Mean)", color=COLORS[i], linewidth=2)

            # 标记该物质特有的最高峰
            peaks, _ = find_peaks(mean_spec, distance=50, prominence=0.1)
            if len(peaks) > 0:
                max_peak_idx = peaks[np.argmax(mean_spec[peaks])]
                max_peak_wave = waves[max_peak_idx]
                ax_merge.annotate(f"{max_peak_wave:.0f}",
                                  xy=(max_peak_wave, mean_spec[max_peak_idx]),
                                  xytext=(0, 10), textcoords='offset points',
                                  ha='center', color=COLORS[i], fontsize=8, fontweight='bold')

    ax_merge.set_title("Class Comparison (Mean Spectra)", fontsize=14, fontweight='bold')
    ax_merge.legend()
    ax_merge.set_xlabel(r"Wavelength ($cm^{-1}$)")
    ax_merge.set_ylabel("Intensity")
    ax_merge.grid(True, alpha=0.3)
    ax_merge.minorticks_on()

    # --- 保存 ---
    save_dir = os.path.join(root_dir, "输出")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 【修改处】文件名后缀改为 .svg
    save_path = os.path.join(save_dir, "纯净物全样本特征分布图.pdf")

    # 【修改处】format='svg'。注意：SVG不需要dpi参数，但保留bbox_inches='tight'防止标签被截断
    plt.savefig(save_path, format='pdf', bbox_inches='tight')

    print(f"✅ 绘图完成！图片已保存至: {save_path}")


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    plot_spectra_grid(current_dir)
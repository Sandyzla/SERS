import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
from config import Config


class Evaluator:
    def __init__(self, model, device, output_dir=Config.OUTPUT_DIR):
        self.model = model
        self.device = device
        self.output_dir = output_dir
        self.model.eval()

    def run_test_evaluation(self):
        """遍历测试集文件夹，生成Excel报告和统计图"""
        test_dir = os.path.join(Config.DATA_DIR, "测试集")
        if not os.path.exists(test_dir):
            print("❌ 测试集目录不存在。")
            return

        results = []
        folder_stats = {}  # {folder_name: [correct, total]}

        print("\n--- 开始测试集评估 ---")

        for folder in os.listdir(test_dir):
            folder_path = os.path.join(test_dir, folder)
            if not os.path.isdir(folder_path): continue

            # 解析真实标签 (Ground Truth)
            true_label_indices = []
            for name, idx in Config.CLASS_MAP.items():
                if name in folder or (name == 'CV' and '结晶紫' in folder):  # 简单兼容中文
                    true_label_indices.append(idx)

            # 读取数据并预测
            csv_files = glob.glob(os.path.join(folder_path, "*MAPPING*.csv"))
            short_name = folder.split('LASER')[0]
            if short_name not in folder_stats: folder_stats[short_name] = [0, 0]

            for csv_f in csv_files:
                try:
                    df = pd.read_csv(csv_f)
                    specs = df.iloc[:, 1:].values.T
                    names = df.columns[1:]

                    # 批量预测优化速度
                    tensor_x = torch.tensor(specs, dtype=torch.float32).unsqueeze(1).to(self.device)
                    with torch.no_grad():
                        logits = self.model(tensor_x)
                        probs = torch.sigmoid(logits).cpu().numpy()

                    for i, prob in enumerate(probs):
                        pred_indices = np.where(prob > Config.PRED_THRESHOLD)[0]
                        is_correct = set(pred_indices) == set(true_label_indices)

                        # 记录统计
                        folder_stats[short_name][1] += 1
                        if is_correct: folder_stats[short_name][0] += 1

                        results.append({
                            "Folder": short_name,
                            "Sample": names[i],
                            "True": "+".join([Config.CLASS_NAMES[k] for k in true_label_indices]),
                            "Pred": "+".join([Config.CLASS_NAMES[k] for k in pred_indices]),
                            "Right": "Yes" if is_correct else "No",
                            **{f"P({n})": f"{p:.4f}" for n, p in zip(Config.CLASS_NAMES, prob)}
                        })
                except Exception as e:
                    print(f"Error processing {csv_f}: {e}")

        # 保存 Excel
        if results:
            pd.DataFrame(results).to_excel(os.path.join(self.output_dir, "详细测试报告.xlsx"), index=False)
            self._plot_accuracy(folder_stats)
            return folder_stats
        return {}

    def analyze_attention(self):
        """导出注意力掩码权重"""
        weights = self.model.spectral_mask.mask_weights.detach().cpu().numpy().squeeze()
        waves = np.linspace(Config.WAVELENGTH_RANGE[0], Config.WAVELENGTH_RANGE[1], len(weights))

        # 保存 CSV
        sub_dir = os.path.join(self.output_dir, "可解释性")
        if not os.path.exists(sub_dir): os.makedirs(sub_dir)

        pd.DataFrame({'Wavelength': waves, 'Weight': weights}).to_csv(
            os.path.join(sub_dir, "Attention_Weights.csv"), index=False
        )

        # 绘图
        plt.figure(figsize=(10, 4))
        plt.plot(waves, weights, color='purple', alpha=0.8)
        plt.title(f"Spectral Attention Mask (Top {Config.TOP_N_PEAKS} Peaks)")

        # 标出 Top N
        top_indices = np.argsort(weights)[-Config.TOP_N_PEAKS:]
        for idx in top_indices:
            plt.scatter(waves[idx], weights[idx], c='red', s=20)
            plt.text(waves[idx], weights[idx], f"{waves[idx]:.0f}", fontsize=8)

        plt.tight_layout()
        plt.savefig(os.path.join(sub_dir, "Attention_Plot.png"))
        plt.close()
        return waves[top_indices]

    def _plot_accuracy(self, stats):
        if not stats: return
        names = list(stats.keys())
        accs = [v[0] / v[1] if v[1] > 0 else 0 for v in stats.values()]

        plt.figure(figsize=(10, 6))
        plt.bar(names, accs, color='skyblue', edgecolor='black')
        plt.ylim(0, 1.1)
        plt.title("Test Accuracy per Folder")
        plt.xticks(rotation=15)
        for i, v in enumerate(accs):
            plt.text(i, v + 0.02, f"{v:.1%}", ha='center')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "测试集准确率.png"))
        plt.close()
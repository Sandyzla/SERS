import os
import glob
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from config import Config
from utils import normalize_spectrum, align_spectra, get_target_axis
from utils import baseline_als


class SERSDataset(Dataset):
    def __init__(self, x_data, y_data):
        if isinstance(x_data, list):
            x_data = np.array(x_data)
        if isinstance(y_data, list):
            y_data = np.array(y_data)

        self.x = torch.tensor(x_data, dtype=torch.float32).unsqueeze(1)
        self.y = torch.tensor(y_data, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class DataManager:
    """数据处理与加载管理器"""

    def __init__(self):
        self.target_axis = get_target_axis(
            Config.WAVELENGTH_RANGE[0], Config.WAVELENGTH_RANGE[1], Config.INPUT_LEN
        )

    def run_preprocessing_pipeline(self):
        """执行完整的离线数据生成流程"""
        print("\n⚡ 正在执行数据预处理流水线...")
        print("纯净物预处理...")
        self._process_pure_substances()
        print("混合增强...")
        self._generate_mix_augmentation()
        print("混合物预处理...")
        self._process_test_set()
        print("✅ 数据预处理完成。\n")

    def _process_pure_substances(self):
        raw_dir = os.path.join(Config.DATA_DIR, "原始数据", "单物质")
        out_dir = os.path.join(Config.DATA_DIR, "训练集", "纯净物")
        if not os.path.exists(out_dir): os.makedirs(out_dir)

        keyword_map = {'结晶紫': 'CV', 'CV': 'CV', '孔雀石绿': 'MG', 'MG': 'MG', '亚甲基蓝': 'MB', 'MB': 'MB'}

        if not os.path.exists(raw_dir): return

        for folder in os.listdir(raw_dir):
            path = os.path.join(raw_dir, folder)
            if not os.path.isdir(path): continue

            category = next((v for k, v in keyword_map.items() if k in folder), None)
            if not category: continue

            self._convert_and_save_csv(path, out_dir, f"{category}_Normalized.csv")

    def _generate_mix_augmentation(self):
        pure_dir = os.path.join(Config.DATA_DIR, "训练集", "纯净物")
        aug_dir = os.path.join(Config.DATA_DIR, "训练集", "数据增强")
        if not os.path.exists(aug_dir): os.makedirs(aug_dir)

        pools = {k: [] for k in Config.CLASS_NAMES}
        for cat in Config.CLASS_NAMES:
            fpath = os.path.join(pure_dir, f"{cat}_Normalized.csv")
            if os.path.exists(fpath):
                df = pd.read_csv(fpath)
                pools[cat] = df.iloc[:, 1:].values.T

        pairs = [('CV', 'MG'), ('MG', 'MB'), ('MB', 'CV')]
        for na, nb in pairs:
            if len(pools[na]) == 0 or len(pools[nb]) == 0: continue

            pair_folder = os.path.join(aug_dir, f"{na}_{nb}")
            if not os.path.exists(pair_folder): os.makedirs(pair_folder)

            for ra, rb, count in Config.MIX_CONFIG:
                alpha, beta = ra / (ra + rb), rb / (ra + rb)
                mixed_data = []
                for _ in range(count):
                    sa = pools[na][np.random.randint(len(pools[na]))]
                    sb = pools[nb][np.random.randint(len(pools[nb]))]
                    mixed_data.append(normalize_spectrum(alpha * sa + beta * sb))

                self._save_data(mixed_data, os.path.join(pair_folder, f"{ra}_{rb}.csv"), prefix="Mix")

    def _process_test_set(self):
        raw_mix_dir = os.path.join(Config.DATA_DIR, "原始数据", "混合物质")
        out_test_dir = os.path.join(Config.DATA_DIR, "测试集")

        if not os.path.exists(raw_mix_dir): return

        for folder in os.listdir(raw_mix_dir):
            src = os.path.join(raw_mix_dir, folder)
            dst = os.path.join(out_test_dir, folder)
            if os.path.isdir(src):
                if not os.path.exists(dst): os.makedirs(dst)
                self._convert_and_save_csv(src, dst, "MAPPING_Normalized.csv")

    def _convert_and_save_csv(self, src_folder, out_folder, filename):
        files = glob.glob(os.path.join(src_folder, "*.csv"))
        processed_data = []
        col_names = []

        for f in files:
            try:
                df = pd.read_csv(f)
                waves = df.iloc[:, 0].values
                for i in range(1, df.shape[1]):
                    spec = align_spectra(waves, df.iloc[:, i].values, self.target_axis)
                    # 【新增】去基线步骤
                    # 只有当确实需要去除荧光背景时才开启
                    spec = baseline_als(spec)
                    processed_data.append(normalize_spectrum(spec))
                    col_names.append(df.columns[i])
            except Exception:
                continue

        if processed_data:
            self._save_data(processed_data, os.path.join(out_folder, filename), col_names=col_names)

    def _save_data(self, data_list, path, col_names=None, prefix="S"):
        data_arr = np.array(data_list).T
        if col_names is None:
            col_names = [f"{prefix}_{i}" for i in range(len(data_list))]

        final_data = np.column_stack((self.target_axis, data_arr))
        cols = ['Wavelength'] + list(col_names)
        pd.DataFrame(final_data, columns=cols).to_csv(path, index=False)

    def load_train_data(self):
        """加载训练数据并返回 (train_loader, val_loader, stats_dict)"""
        tr_x, tr_y = [], []
        # 用于存储详细统计信息
        stats = {}

        if Config.IS_USE_SINGLE_SUBSTANCE:
            path = os.path.join(Config.DATA_DIR, "训练集", "纯净物")
            pure_stats = self._load_folder(path, tr_x, tr_y, is_mix=False)
            stats.update(pure_stats)

        path = os.path.join(Config.DATA_DIR, "训练集", "数据增强")
        mix_stats = self._load_folder(path, tr_x, tr_y, is_mix=True)
        stats.update(mix_stats)

        full_ds = SERSDataset(tr_x, tr_y)
        train_len = int(0.8 * len(full_ds))
        train_ds, val_ds = random_split(full_ds, [train_len, len(full_ds) - train_len])

        return (DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True),
                DataLoader(val_ds, batch_size=Config.BATCH_SIZE, shuffle=False),
                stats)

    def _load_folder(self, root_path, x_list, y_list, is_mix):
        """
        加载文件夹数据，并返回该路径下各类别的样本数量统计。
        """
        local_stats = {}
        if not os.path.exists(root_path): return local_stats

        for root, _, files in os.walk(root_path):
            for file in files:
                if not file.endswith(".csv"): continue

                label = [0.0, 0.0, 0.0]
                category_key = "Unknown"

                # --- 1. 确定标签和统计类别 ---
                if not is_mix:
                    # 纯净物：从文件名判断 (e.g., CV_Normalized.csv)
                    for k, v in Config.CLASS_MAP.items():
                        if k in file:
                            label[v] = 1.0
                            category_key = f"Pure ({k})"
                            break
                else:
                    # 混合物：从父文件夹名判断 (e.g., CV_MG)
                    try:
                        ratio_parts = file.replace('.csv', '').split('_')
                        ra, rb = float(ratio_parts[0]), float(ratio_parts[1])
                        parent = os.path.basename(root)  # e.g., "CV_MG"
                        na, nb = parent.split('_')
                        label[Config.CLASS_MAP[na]] = ra / (ra + rb)
                        label[Config.CLASS_MAP[nb]] = rb / (ra + rb)
                        category_key = f"Augment ({parent})"
                    except:
                        continue

                # --- 2. 读取数据并计数 ---
                try:
                    df = pd.read_csv(os.path.join(root, file))
                    data = df.iloc[:, 1:].values.T  # 转置，每行一个样本

                    count = len(data)
                    if count > 0:
                        # 累加统计
                        local_stats[category_key] = local_stats.get(category_key, 0) + count

                    for spec in data:
                        x_list.append(spec)
                        y_list.append(label)
                except:
                    pass

        return local_stats

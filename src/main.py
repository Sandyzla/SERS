import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
import numpy as np

from config import Config
from utils import setup_logger, set_seed, get_device
from model import CoreModel
from data_manager import DataManager
from evaluator import Evaluator


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(x)

        cls_loss = criterion(output, y)
        mask_loss = torch.norm(model.spectral_mask.mask_weights, 1)
        loss = cls_loss + Config.L1_LAMBDA * mask_loss

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            total_loss += criterion(output, y).item()
    return total_loss / len(loader)


def main():
    # 1. 初始化
    set_seed(Config.SEED)
    run_name = Config.get_run_name()
    logger = setup_logger(Config.LOG_DIR, run_name)
    device = get_device()

    logger.info("=" * 60)
    logger.info(f">>> 🚀 新的训练实验开始 [{run_name}] <<<")
    logger.info("=" * 60)
    for key in dir(Config):
        if key.isupper():
            val = getattr(Config, key)
            logger.info(f"{key:<25}: {val}")
    logger.info("-" * 60)

    # 2. 数据准备
    dm = DataManager()
    if Config.IS_GENERATE_TRAIN_DATA:
        dm.run_preprocessing_pipeline()

    # 【修改】接收 stats 返回值
    train_loader, val_loader, stats = dm.load_train_data()

    # 【新增】详细的数据集统计日志输出
    logger.info("📊 数据集构成详情 (Total Samples):")

    # 纯净物统计
    pure_count = 0
    logger.info(f"   >>> 🧪 纯净物 (Pure):")
    for k, v in sorted(stats.items()):
        if "Pure" in k:
            logger.info(f"       - {k:<15}: {v} samples")
            pure_count += v

    # 增强数据统计
    mix_count = 0
    logger.info(f"   >>> 🌪️ 数据增强 (Augmented/Mixed):")
    for k, v in sorted(stats.items()):
        if "Augment" in k:
            logger.info(f"       - {k:<15}: {v} samples")
            mix_count += v

    logger.info("-" * 40)
    logger.info(f"   [汇总] Pure: {pure_count} | Augmented: {mix_count} | Total: {pure_count + mix_count}")
    logger.info("-" * 60)

    logger.info(
        f"📊 数据加载完毕 (Train/Val Split): 训练集 {len(train_loader.dataset)} / 验证集 {len(val_loader.dataset)}")

    # 3. 模型构建
    model = CoreModel(input_len=Config.INPUT_LEN, num_classes=3).to(device)
    model_save_path = os.path.join(Config.MODEL_DIR, "sers_model.pth")
    best_model_path = os.path.join(Config.MODEL_DIR, "best_model.pth")

    # 4. 训练流程
    if Config.IS_RETRAIN or not os.path.exists(model_save_path):
        logger.info(f"\n🔥 开始训练 (Epochs={Config.EPOCHS}, Batch={Config.BATCH_SIZE})...")

        mask_ids = list(map(id, model.spectral_mask.parameters()))
        base_params = filter(lambda p: id(p) not in mask_ids, model.parameters())
        optimizer = optim.Adam([
            {'params': base_params, 'lr': Config.BASE_LR},
            {'params': model.spectral_mask.parameters(), 'lr': Config.BASE_LR * Config.MASK_LR_SCALE}
        ])
        criterion = nn.BCEWithLogitsLoss()

        best_loss = float('inf')
        start_time = time.time()

        for epoch in range(Config.EPOCHS):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_loss = validate(model, val_loader, criterion, device)

            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(model.state_dict(), best_model_path)

            logger.info(
                f"Epoch {epoch + 1}/{Config.EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        duration = time.time() - start_time
        mins, secs = divmod(duration, 60)
        logger.info("-" * 60)
        logger.info(f"✅ 训练完成！")
        logger.info(f"⏱️  训练总耗时: {int(mins)}分 {int(secs)}秒 ({duration:.2f} 秒)")

        if os.path.exists(best_model_path):
            model.load_state_dict(torch.load(best_model_path))
            logger.info(f"✅ 成功加载最佳模型权重 (来自: best_model.pth)")

        torch.save(model.state_dict(), model_save_path)
    else:
        logger.info("\n🔄 加载已有模型...")
        model.load_state_dict(torch.load(model_save_path, map_location=device))

    # 5. 评估与报告
    evaluator = Evaluator(model, device)
    stats = evaluator.run_test_evaluation()

    logger.info("\n" + "=" * 60)
    logger.info(">>> 📊 最终测试评估报告 <<<")
    logger.info("-" * 60)
    logger.info(f"{'测试组 (Folder)':<30} | {'准确率'}")

    total_correct = 0
    total_count = 0

    if stats:
        for name, (correct, total) in stats.items():
            acc = correct / total if total > 0 else 0
            logger.info(f"{name:<30} | {acc:.2%}")
            total_correct += correct
            total_count += total

        logger.info("-" * 60)
        global_acc = total_correct / total_count if total_count > 0 else 0
        logger.info(f"【总体准确率】: {global_acc:.2%}")
    else:
        logger.info("⚠️ 测试集为空或未找到相关文件。")

    logger.info("=" * 60)

    # 6. 可解释性分析
    top_peaks = evaluator.analyze_attention()
    peaks_str = ", ".join([f"{p:.0f}" for p in top_peaks])
    logger.info(f"🔍 识别出的 Top-{Config.TOP_N_PEAKS} 关键特征波长: [{peaks_str}]")


if __name__ == "__main__":
    main()
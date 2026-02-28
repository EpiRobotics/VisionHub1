#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
焊点火花检测 - 二分类快速验证 (OK / NG)
使用预训练 ResNet18 做微调，自动从 ImageFolder 的 class_to_idx 推断 OK / NG 的索引。

目录结构示例:
D:/weld_data/
    train/OK/*.jpg
    train/NG/*.jpg
    val/OK/*.jpg
    val/NG/*.jpg

支持从 projects.json 配置文件读取项目配置
用法: python train_weld_cls.py [项目ID]
"""

from pathlib import Path
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models

from config_loader import get_project_config, get_default_project, list_projects


# ====== 默认配置 ======
IMG_SIZE = 224
BATCH_SIZE = 8
NUM_EPOCHS = 40
LEARNING_RATE = 1e-4
# ==================

# 全局变量，由 main() 根据项目配置设置
DATA_ROOT = None
TRAIN_DIR = None
VAL_DIR = None
MODEL_PATH = None


def infer_ok_ng_idx(class_to_idx: dict):
    """根据类名推断 OK / NG 的索引（支持 OK/ok, NG/ng）"""
    lower_map = {name.lower(): idx for name, idx in class_to_idx.items()}
    idx_ok = lower_map.get("ok", None)
    idx_ng = lower_map.get("ng", None)
    if idx_ok is None or idx_ng is None:
        raise ValueError(f"class_to_idx 中找不到 OK/NG 类，请检查目录名: {class_to_idx}")
    return idx_ok, idx_ng


def build_dataloaders():
    # 训练增强：亮度/对比度+轻微旋转+水平翻转（如果左右基本对称）
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.RandomRotation(5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])

    train_ds = datasets.ImageFolder(str(TRAIN_DIR), transform=train_tf)
    val_ds   = datasets.ImageFolder(str(VAL_DIR),   transform=val_tf)

    print("classes:", train_ds.classes)
    print("class_to_idx:", train_ds.class_to_idx)

    idx_ok, idx_ng = infer_ok_ng_idx(train_ds.class_to_idx)

    targets = np.array(train_ds.targets)
    num_ok = (targets == idx_ok).sum()
    num_ng = (targets == idx_ng).sum()
    print(f"Train OK: {num_ok}, NG: {num_ng}")

    # 类别权重：样本少的类别权重大
    num_classes = len(train_ds.classes)
    class_counts = np.bincount(targets, minlength=num_classes).astype(np.float32)
    # 避免除零
    class_weights = class_counts.sum() / (num_classes * (class_counts + 1e-6))
    print("class_counts:", class_counts)
    print("class_weights (按索引顺序):", class_weights)

    # 为每个样本分配权重，用于 WeightedRandomSampler
    sample_weights = class_weights[targets]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, val_loader, class_weights, idx_ok, idx_ng, train_ds.class_to_idx


def build_model(num_classes=2):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备:", device)

    train_loader, val_loader, class_weights, idx_ok, idx_ng, class_to_idx = build_dataloaders()

    model = build_model(num_classes=len(class_to_idx)).to(device)

    # 损失带类别权重（顺序按类别索引）
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_t)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, NUM_EPOCHS + 1):
        # -------- Train --------
        model.train()
        total_loss = 0.0
        total = 0

        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            total += imgs.size(0)

        train_loss = total_loss / total

        # -------- Val --------
        model.eval()
        correct = 0
        total = 0
        tp = fp = fn = 0  # 针对 NG 类的 TP/FP/FN

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                labels = labels.to(device)

                logits = model(imgs)
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)

                correct += (preds == labels).sum().item()
                total += labels.size(0)

                # 统计 NG 类(idx_ng)的指标
                for p, t in zip(preds.cpu().numpy(), labels.cpu().numpy()):
                    if t == idx_ng and p == idx_ng:
                        tp += 1
                    elif t != idx_ng and p == idx_ng:
                        fp += 1
                    elif t == idx_ng and p != idx_ng:
                        fn += 1

        val_acc = correct / total if total > 0 else 0.0
        precision_ng = tp / (tp + fp + 1e-6)
        recall_ng    = tp / (tp + fn + 1e-6)

        print(f"Epoch [{epoch}/{NUM_EPOCHS}] "
              f"TrainLoss={train_loss:.4f} "
              f"ValAcc={val_acc:.4f} "
              f"NG_P={precision_ng:.3f} NG_R={recall_ng:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # 保存最好的模型参数
            best_state = model.state_dict().copy()

    if best_state is not None:
        ckpt = {
            "model_state": best_state,
            "class_to_idx": class_to_idx,
            "idx_ok": idx_ok,
            "idx_ng": idx_ng,
            "img_size": IMG_SIZE,
        }
        torch.save(ckpt, MODEL_PATH)
        print(f"最优模型已保存到: {MODEL_PATH}  (ValAcc={best_val_acc:.4f})")
    else:
        print("没有保存模型（可能没有有效样本？）")


def setup_paths(project_id: str = None):
    """根据项目ID设置路径"""
    global DATA_ROOT, TRAIN_DIR, VAL_DIR, MODEL_PATH
    
    config = get_project_config(project_id)
    
    DATA_ROOT = Path(config.get("data_root", "D:/weld_data"))
    TRAIN_DIR = Path(config.get("train_dir", DATA_ROOT / "train"))
    VAL_DIR = Path(config.get("val_dir", DATA_ROOT / "val"))
    MODEL_PATH = Path(config.get("model_path", DATA_ROOT / "weld_resnet18_cls.pth"))
    
    print(f"项目配置:")
    print(f"  数据根目录: {DATA_ROOT}")
    print(f"  训练目录: {TRAIN_DIR}")
    print(f"  验证目录: {VAL_DIR}")
    print(f"  模型路径: {MODEL_PATH}")


def main():
    parser = argparse.ArgumentParser(description="焊点火花检测 - 训练脚本")
    parser.add_argument("project", nargs="?", default=None,
                        help=f"项目ID (可选，默认使用配置文件中的默认项目)")
    parser.add_argument("--list", action="store_true", help="列出所有可用项目")
    args = parser.parse_args()
    
    if args.list:
        print("可用项目:", list_projects())
        print("默认项目:", get_default_project())
        return
    
    project_id = args.project
    if project_id:
        print(f"使用项目: {project_id}")
    else:
        project_id = get_default_project()
        print(f"使用默认项目: {project_id}")
    
    setup_paths(project_id)
    train()


if __name__ == "__main__":
    main()

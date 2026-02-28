#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用训练好的焊点火花检测模型，把测试目录下的图片
自动分到 OK / NG 两个子文件夹中。

前置条件:
- 训练好的模型权重
- 测试目录下存放待分类图片 (jpg/png/jpeg/bmp)

支持从 projects.json 配置文件读取项目配置
用法: python run_weld_sort.py [项目ID]
"""

from pathlib import Path
import argparse
import shutil

import torch
from torchvision import transforms, models
from PIL import Image

from config_loader import get_project_config, get_default_project, list_projects


# ===== 默认配置 =====
IMG_SIZE   = 224
USE_THRESHOLD = False
NG_THRESH     = 0.5
# =================

# 全局变量，由 setup_paths() 根据项目配置设置
MODEL_PATH = None
SRC_DIR = None
OK_DIR = None
NG_DIR = None


def infer_ok_ng_idx(class_to_idx: dict):
    """根据类名推断 OK / NG 的索引"""
    lower_map = {name.lower(): idx for name, idx in class_to_idx.items()}
    idx_ok = lower_map.get("ok", None)
    idx_ng = lower_map.get("ng", None)
    if idx_ok is None or idx_ng is None:
        raise ValueError(f"class_to_idx 中找不到 OK/NG 类，请检查: {class_to_idx}")
    return idx_ok, idx_ng


def build_model(num_classes=2):
    # 结构要和训练时一致
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, num_classes)
    return model


def load_model(device):
    ckpt = torch.load(MODEL_PATH, map_location=device)

    # 兼容：如果旧模型只保存了 state_dict
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        state_dict = ckpt["model_state"]
        class_to_idx = ckpt.get("class_to_idx", None)
        idx_ok = ckpt.get("idx_ok", None)
        idx_ng = ckpt.get("idx_ng", None)
    else:
        state_dict = ckpt
        class_to_idx = None
        idx_ok = idx_ng = None

    model = build_model(num_classes=state_dict[list(state_dict.keys())[-1]].shape[0]).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    # 如果 ckpt 里有 class_to_idx，就用它推断 OK/NG 索引
    if class_to_idx is not None:
        idx_ok2, idx_ng2 = infer_ok_ng_idx(class_to_idx)
        # 如果 ckpt 里已经有 idx_ok/idx_ng，用它们；否则用推断的
        if idx_ok is None or idx_ng is None:
            idx_ok, idx_ng = idx_ok2, idx_ng2
    else:
        # 兜底：假设 {'NG':0,'OK':1}
        print("[警告] ckpt 中没有 class_to_idx，默认假设 0=NG, 1=OK")
        idx_ng, idx_ok = 0, 1

    return model, idx_ok, idx_ng


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备:", device)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"找不到模型文件: {MODEL_PATH}")

    # 创建输出目录
    OK_DIR.mkdir(exist_ok=True, parents=True)
    NG_DIR.mkdir(exist_ok=True, parents=True)

    tfm = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])

    model, idx_ok, idx_ng = load_model(device)
    print(f"idx_ok={idx_ok}, idx_ng={idx_ng}")

    exts = {".jpg", ".jpeg", ".png", ".bmp"}

    files = []
    for p in SRC_DIR.iterdir():
        if p.is_dir():
            # 跳过已经存在的 OK/NG 目录
            continue
        if p.suffix.lower() in exts:
            files.append(p)

    print(f"待分类图片数量: {len(files)}")

    for img_path in files:
        img = Image.open(img_path).convert("RGB")
        inp = tfm(img).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(inp)
            probs = torch.softmax(logits, dim=1).cpu().squeeze(0)  # [num_classes]

        prob_ok = float(probs[idx_ok])
        prob_ng = float(probs[idx_ng])

        if USE_THRESHOLD:
            # 按 NG 概率阈值判
            if prob_ng > NG_THRESH:
                pred_label = "NG"
            else:
                pred_label = "OK"
        else:
            pred_idx = int(torch.argmax(probs).item())
            pred_label = "OK" if pred_idx == idx_ok else "NG"

        dst_dir = OK_DIR if pred_label == "OK" else NG_DIR
        dst_path = dst_dir / img_path.name

        # 用复制更安全，确认没问题后可以改成 shutil.move
        shutil.copy2(img_path, dst_path)

        print(f"{img_path.name}: prob_OK={prob_ok:.3f}, prob_NG={prob_ng:.3f} -> {pred_label}")

    print("分类完成。结果已复制到:")
    print("  OK:", OK_DIR)
    print("  NG:", NG_DIR)


def setup_paths(project_id: str = None):
    """根据项目ID设置路径"""
    global MODEL_PATH, SRC_DIR, OK_DIR, NG_DIR
    
    config = get_project_config(project_id)
    
    MODEL_PATH = Path(config.get("model_path", "D:/weld_data/weld_resnet18_cls.pth"))
    SRC_DIR = Path(config.get("test_dir", "D:/test"))
    OK_DIR = SRC_DIR / "OK"
    NG_DIR = SRC_DIR / "NG"
    
    print(f"项目配置:")
    print(f"  模型路径: {MODEL_PATH}")
    print(f"  测试目录: {SRC_DIR}")
    print(f"  OK输出: {OK_DIR}")
    print(f"  NG输出: {NG_DIR}")


def run():
    parser = argparse.ArgumentParser(description="焊点火花检测 - 测试/分类脚本")
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
    main()


if __name__ == "__main__":
    run()

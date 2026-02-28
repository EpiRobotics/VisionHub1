#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import json
import numpy as np
import cv2
from tqdm import tqdm
import joblib

import torch
import torch.nn as nn
from torchvision import models, transforms
from sklearn.neighbors import NearestNeighbors

IMG_EXTS = {".jpg",".jpeg",".png",".bmp",".tif",".tiff"}

# 你的 glyph_bank 文件夹名映射规则（保持一致）
SPECIAL_MAP = {
    "/": "slash",
    "\\": "backslash",
    ".": "dot",
    ":": "colon",
    "*": "asterisk",
    "?": "question",
    "\"": "quote",
    "<": "lt",
    ">": "gt",
    "|": "pipe",
    " ": "space",
    "\t": "tab",
}
INVALID_WIN = set('<>:"/\\|?*')

def safe_folder_name(ch: str) -> str:
    if ch in SPECIAL_MAP:
        return SPECIAL_MAP[ch]
    if ch in INVALID_WIN:
        return f"u{ord(ch):04X}"
    if ch.endswith(".") or ch.endswith(" "):
        return f"u{ord(ch):04X}"
    return ch

def imread_any_gray(p: Path):
    data = np.fromfile(str(p), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)

def list_imgs(d: Path):
    return sorted([p for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])

class ResNetFeat(nn.Module):
    """取到 layer2 的特征图（浅层，适合缺笔）"""
    def __init__(self):
        super().__init__()
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)
        self.layer1 = m.layer1
        self.layer2 = m.layer2

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)   # [B,C,H,W]
        return x

def preprocess(gray, size=128):
    # 小图必须放大
    g = cv2.resize(gray, (size,size), interpolation=cv2.INTER_CUBIC)
    rgb = np.stack([g,g,g], axis=-1).astype(np.uint8)
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
    ])
    return tf(rgb).unsqueeze(0)

def extract_patch_embeddings(feat_map: torch.Tensor):
    """
    feat_map: [1,C,H,W] -> [H*W, C]
    """
    fm = feat_map.squeeze(0)          # [C,H,W]
    C,H,W = fm.shape
    emb = fm.permute(1,2,0).reshape(-1, C)  # [N,C]
    return emb

def main():
    import argparse
    ap = argparse.ArgumentParser("Train PatchCore-like models for each glyph class (OK-only)")
    ap.add_argument("--bank_dir", required=True, help=r"D:\pic\glyph_bank")
    ap.add_argument("--out_dir", required=True, help=r"D:\pic\models_patchcore")
    ap.add_argument("--img_size", type=int, default=128)
    ap.add_argument("--max_patches_per_class", type=int, default=30000,
                    help="memory bank 最大patch数（超出会随机采样）")
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--score_mode", choices=["max", "topk"], default="topk",
                    help="字符分数聚合方式：max 或 topk平均")
    ap.add_argument("--topk", type=int, default=10, help="score_mode=topk 时取最大的 topk 距离均值")
    ap.add_argument("--p_thr", type=float, default=0.995, help="阈值分位数（OK分布）")
    ap.add_argument("--min_per_class", type=int, default=10)
    args = ap.parse_args()

    bank_dir = Path(args.bank_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = ResNetFeat().to(device).eval()

    index = {"img_size": args.img_size, "score_mode": args.score_mode, "topk": args.topk,
             "k": args.k, "p_thr": args.p_thr, "classes": []}

    class_dirs = [p for p in bank_dir.iterdir() if p.is_dir()]
    if not class_dirs:
        print("[ERROR] no class folders in", bank_dir); return

    for cls_dir in sorted(class_dirs, key=lambda p: p.name):
        cls = cls_dir.name
        imgs = list_imgs(cls_dir)
        if len(imgs) < args.min_per_class:
            print(f"[WARN] skip {cls}: {len(imgs)} < {args.min_per_class}")
            continue

        all_patches = []
        for p in tqdm(imgs, desc=f"Extract {cls}", leave=False):
            g = imread_any_gray(p)
            if g is None:
                continue
            x = preprocess(g, size=args.img_size).to(device)
            with torch.no_grad():
                fmap = net(x)  # [1,C,H,W]
                emb = extract_patch_embeddings(fmap).cpu().numpy().astype(np.float32)  # [N,C]
            all_patches.append(emb)

        if not all_patches:
            continue

        X = np.vstack(all_patches)  # [N_total, C]

        # 采样压缩 memory bank
        if X.shape[0] > args.max_patches_per_class:
            idxs = np.random.choice(X.shape[0], args.max_patches_per_class, replace=False)
            X = X[idxs]

        # 建 NN
        nn = NearestNeighbors(n_neighbors=args.k, metric="euclidean")
        nn.fit(X)

        # 用 OK 自己的 patch 计算 OK-score 分布（字符级）
        # 为了更贴近推理：每张图计算一个 score，再取分位数
        ok_scores = []
        for emb in all_patches:
            d, _ = nn.kneighbors(emb)  # [Npatch,k]
            d = d.mean(axis=1)         # [Npatch]
            if args.score_mode == "max":
                s = float(d.max())
            else:
                topk = min(args.topk, d.shape[0])
                s = float(np.mean(np.sort(d)[-topk:]))
            ok_scores.append(s)

        thr = float(np.quantile(ok_scores, args.p_thr))

        model = {
            "cls": cls,
            "img_size": int(args.img_size),
            "k": int(args.k),
            "score_mode": args.score_mode,
            "topk": int(args.topk),
            "p_thr": float(args.p_thr),
            "thr": thr,
            "memory_bank": X,          # 存 embedding（float32）
        }
        joblib.dump(model, out_dir / f"{cls}.joblib")
        index["classes"].append({"cls": cls, "thr": thr, "n_ok": len(imgs), "n_patches": int(X.shape[0])})
        print(f"[OK] {cls}: ok_imgs={len(imgs)} memory={X.shape} thr={thr:.4f}")

    (out_dir / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[DONE] models:", out_dir.resolve())
    print("[DONE] index:", (out_dir/"index.json").resolve())

if __name__ == "__main__":
    main()

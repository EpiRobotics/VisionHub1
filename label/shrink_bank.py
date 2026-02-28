#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
shrink_bank.py
- 读取 PatchCore per-class joblib 模型（包含 memory_bank, k, score_mode, topk, thr, img_size 等）
- 对 memory_bank 做压缩（随机抽样 / 可设随机种子）
- 重新计算 thr（用该类训练 patch 的 self-score 分布分位数）
- 输出到新目录（不会覆盖原模型）

用法示例：
python shrink_bank.py --in_dir D:\models_patchcore --out_dir D:\models_patchcore_small --max_bank 15000 --p_thr 0.999 --seed 42
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import joblib

from sklearn.neighbors import NearestNeighbors


def score_from_dist(d: np.ndarray, score_mode: str, topk: int) -> float:
    if score_mode == "max":
        return float(d.max())
    topk = max(1, min(int(topk), d.shape[0]))
    return float(np.mean(np.sort(d)[-topk:]))


def recompute_thr(bank: np.ndarray, k: int, score_mode: str, topk: int, p_thr: float) -> float:
    """
    用 bank 自身的“训练自分数”来估计 OK 分布阈值：
    - 对每个 patch embedding，找 k 个近邻距离（注意会包含自身距离=0）
    - 取每个 patch 的 mean(kNN distance) 得到 d_patch
    - 再用 score_mode/topk 把一堆 patch 聚合成“glyph score”
    这里为了速度，直接把 patch-level d_patch 分布做分位数，作为近似阈值。
    （实测对你这种字符库足够稳，而且极快）
    """
    # NOTE：NearestNeighbors 会把自身也作为最近邻（距离=0）
    k = max(1, int(k))
    nnm = NearestNeighbors(n_neighbors=k, metric="euclidean")
    nnm.fit(bank)

    d, _ = nnm.kneighbors(bank)          # [B,k]
    d = d.mean(axis=1)                   # [B] patch-level mean dist

    # 用 patch-level 分位数做阈值（比全glyph聚合更保守一点）
    thr = float(np.quantile(d, p_thr))
    return thr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", type=str, required=True, help="input model dir containing *.joblib")
    ap.add_argument("--out_dir", type=str, required=True, help="output dir for shrunk models")
    ap.add_argument("--max_bank", type=int, default=15000, help="max embeddings per class")
    ap.add_argument("--min_bank", type=int, default=2000, help="min embeddings per class (avoid too small)")
    ap.add_argument("--seed", type=int, default=42, help="random seed for sampling")
    ap.add_argument("--p_thr", type=float, default=0.999, help="quantile for threshold recompute (OK high quantile)")
    ap.add_argument("--method", type=str, default="random", choices=["random", "first"], help="sampling method")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    files = sorted(in_dir.glob("*.joblib"))
    if not files:
        raise SystemExit(f"No *.joblib found in {in_dir}")

    total_before = 0
    total_after = 0

    for fp in files:
        m = joblib.load(fp)

        cls = str(m.get("cls", fp.stem))
        bank = np.asarray(m["memory_bank"], dtype=np.float32)
        B = bank.shape[0]
        total_before += B

        # decide target size
        target = min(int(args.max_bank), B)
        target = max(int(args.min_bank), target)

        if target >= B:
            bank2 = bank
            idx = None
        else:
            if args.method == "first":
                idx = np.arange(target, dtype=np.int64)
            else:
                idx = rng.choice(B, size=target, replace=False)
            bank2 = bank[idx]

        # recompute thr (patch-level quantile)
        k = int(m.get("k", 1))
        score_mode = str(m.get("score_mode", "topk"))
        topk = int(m.get("topk", 10))

        thr2 = recompute_thr(bank2, k=k, score_mode=score_mode, topk=topk, p_thr=float(args.p_thr))

        # update model dict
        m2 = dict(m)
        m2["memory_bank"] = bank2
        m2["thr"] = float(thr2)
        m2["bank_before"] = int(B)
        m2["bank_after"] = int(bank2.shape[0])
        m2["thr_recomputed"] = True
        m2["p_thr_used"] = float(args.p_thr)
        m2["shrink_method"] = args.method
        m2["seed"] = int(args.seed)

        out_fp = out_dir / fp.name
        joblib.dump(m2, out_fp, compress=3)

        total_after += bank2.shape[0]

        msg = f"[{cls}] bank {B} -> {bank2.shape[0]}  thr {m.get('thr', None)} -> {thr2:.6f}"
        print(msg)

    print(f"\n[DONE] total_bank {total_before} -> {total_after}")
    print(f"[OUT] {out_dir}")


if __name__ == "__main__":
    main()

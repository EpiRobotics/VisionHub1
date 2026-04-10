#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json, csv
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm
import joblib

import torch
import torch.nn as nn
from torchvision import models, transforms
from sklearn.neighbors import NearestNeighbors

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

def clip_box(x0,y0,x1,y1,W,H):
    x0=max(0,min(x0,W)); y0=max(0,min(y0,H))
    x1=max(0,min(x1,W)); y1=max(0,min(y1,H))
    if x1<=x0: x1=min(W,x0+1)
    if y1<=y0: y1=min(H,y0+1)
    return x0,y0,x1,y1

def imread_any_bgr(p: Path):
    data = np.fromfile(str(p), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)

class ResNetFeat(nn.Module):
    def __init__(self, feature_layers: str = "layer2"):
        super().__init__()
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)
        self.layer1 = m.layer1
        self.layer2 = m.layer2
        self.feature_layers = feature_layers
    def forward(self, x):
        if self.feature_layers == "multi":
            return self._forward_multiscale(x)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        return x
    def _forward_multiscale(self, x):
        """Multi-scale: concat layer1 (64-dim, 16x16) + upsampled layer2 (128-dim)."""
        x = self.stem(x)
        feat1 = self.layer1(x)
        feat2 = self.layer2(feat1)
        feat2_up = nn.functional.interpolate(
            feat2, size=feat1.shape[2:], mode="bilinear", align_corners=False,
        )
        return torch.cat([feat1, feat2_up], dim=1)

def preprocess(gray, size=128, clahe_clip=0.0):
    g = cv2.resize(gray, (size,size), interpolation=cv2.INTER_CUBIC)
    if clahe_clip > 0:
        clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(4, 4))
        g = clahe.apply(g)
    rgb = np.stack([g,g,g], axis=-1).astype(np.uint8)
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
    ])
    return tf(rgb).unsqueeze(0)

def extract_patch_embeddings(feat_map: torch.Tensor):
    fm = feat_map.squeeze(0)          # [C,H,W]
    C,H,W = fm.shape
    emb = fm.permute(1,2,0).reshape(-1, C)  # [N,C]
    return emb

def score_patchcore(emb_np: np.ndarray, nn: NearestNeighbors, mode: str, topk: int):
    d, _ = nn.kneighbors(emb_np)     # [N,k]
    d = d.mean(axis=1)              # [N]
    if mode == "max":
        return float(d.max())
    if mode == "relative":
        # Subtract median to remove global shift from thickness/size variations;
        # only local outliers (broken/missing lines) contribute to the score.
        median_d = float(np.median(d))
        residuals = d - median_d
        topk = min(topk, residuals.shape[0])
        return float(np.mean(np.sort(residuals)[-topk:]))
    topk = min(topk, d.shape[0])
    topk_mean = float(np.mean(np.sort(d)[-topk:]))
    if mode == "adaptive":
        max_val = float(d.max())
        return float(np.sqrt(max_val * topk_mean))
    return topk_mean

def main():
    import argparse
    ap = argparse.ArgumentParser("PatchCore predict from JSON items")
    ap.add_argument("--model_dir", required=True, help="models_patchcore dir")
    ap.add_argument("--img_dir", required=True)
    ap.add_argument("--json_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--thr_global", type=float, default=None, help="可选：统一阈值覆盖每类阈值")
    ap.add_argument("--pad", type=int, default=2)
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    out_dir = Path(args.out_dir)
    (out_dir/"overlay").mkdir(parents=True, exist_ok=True)
    (out_dir/"csv").mkdir(parents=True, exist_ok=True)

    # load all class models
    model_files = list(model_dir.glob("*.joblib"))
    cls_models = {}
    nn_cache = {}

    for mf in model_files:
        if mf.name == "index.json":
            continue
        m = joblib.load(mf)
        cls = m["cls"]
        cls_models[cls] = m
        nn = NearestNeighbors(n_neighbors=int(m["k"]), metric="euclidean")
        nn.fit(m["memory_bank"])
        nn_cache[cls] = nn

    if not cls_models:
        print("[ERROR] no class models in", model_dir); return

    # Detect feature_layers from loaded models
    effective_fl = "layer2"
    if cls_models:
        first_m = next(iter(cls_models.values()))
        effective_fl = str(first_m.get("feature_layers", "layer2"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = ResNetFeat(feature_layers=effective_fl).to(device).eval()

    detail = [["image","idx","ch","cls","score","thr_used","decision","x0","y0","x1","y1"]]
    summary = [["image","total","ng","unknown"]]

    json_files = sorted(Path(args.json_dir).glob("*.json"))
    for jp in tqdm(json_files, desc="Predict"):
        data = json.loads(jp.read_text(encoding="utf-8-sig"))
        image_name = data.get("image_name","")
        if not image_name:
            continue
        img_path = Path(args.img_dir)/image_name
        if not img_path.exists():
            img_path = Path(args.img_dir)/Path(image_name).name

        img = imread_any_bgr(img_path)
        if img is None:
            continue
        H,W = img.shape[:2]
        gray_full = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        vis = img.copy()
        items = data.get("items", [])
        ng=0; unk=0

        for it in items:
            idx = it.get("i", it.get("idx", -1))
            ch = str(it.get("ch",""))
            if ch=="":
                continue
            cx=float(it.get("cx",0)); cy=float(it.get("cy",0))
            w=float(it.get("w",0)); h=float(it.get("h",0))
            if w<=0 or h<=0:
                continue

            x0=int(round(cx-w/2))-args.pad
            y0=int(round(cy-h/2))-args.pad
            x1=int(round(cx+w/2))+args.pad
            y1=int(round(cy+h/2))+args.pad
            x0,y0,x1,y1 = clip_box(x0,y0,x1,y1,W,H)

            cls = safe_folder_name(ch)
            if cls not in cls_models:
                unk += 1
                cv2.rectangle(vis,(x0,y0),(x1,y1),(0,255,255),2)
                cv2.putText(vis,f"{ch}:UNK",(x0,max(15,y0-5)),cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,255,255),2)
                detail.append([img_path.name,idx,ch,cls,"","","UNKNOWN",x0,y0,x1,y1])
                continue

            m = cls_models[cls]
            thr = float(args.thr_global) if args.thr_global is not None else float(m["thr"])

            patch = gray_full[y0:y1, x0:x1]
            x = preprocess(patch, size=int(m["img_size"]),
                           clahe_clip=float(m.get("clahe_clip", 0.0))).to(device)
            with torch.no_grad():
                fmap = net(x)
                emb = extract_patch_embeddings(fmap).cpu().numpy().astype(np.float32)

            score = score_patchcore(emb, nn_cache[cls], m["score_mode"], int(m["topk"]))
            decision = "NG" if score > thr else "OK"
            if decision=="NG":
                ng += 1
                color=(0,0,255)
                cv2.rectangle(vis,(x0,y0),(x1,y1),color,2)
                cv2.putText(vis,f"{ch}:{score:.2f}",(x0,max(15,y0-5)),cv2.FONT_HERSHEY_SIMPLEX,0.55,color,2)
            else:
                cv2.rectangle(vis,(x0,y0),(x1,y1),(0,255,0),1)

            detail.append([img_path.name,idx,ch,cls,f"{score:.6f}",f"{thr:.6f}",decision,x0,y0,x1,y1])

        outp = out_dir/"overlay"/f"{Path(img_path.name).stem}_overlay.png"
        cv2.imencode(".png", vis)[1].tofile(str(outp))
        summary.append([img_path.name,len(items),ng,unk])

    with (out_dir/"csv/details.csv").open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(detail)
    with (out_dir/"csv/summary.csv").open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(summary)

    print("[DONE] overlay:", (out_dir/"overlay").resolve())
    print("[DONE] details:", (out_dir/"csv/details.csv").resolve())
    print("[DONE] summary:", (out_dir/"csv/summary.csv").resolve())

if __name__ == "__main__":
    main()

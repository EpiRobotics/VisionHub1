#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
PatchCore (OK-only) anomaly detection with ROI tiling + overlap stitching.

- Train (OK-only): build a memory bank of patch embeddings from OK images.
- Predict: tile ROI -> patch anomaly maps -> stitch by max over overlaps -> heatmap + score.

Default tuned for ROI 2050x350 and defects >= ~40 px (2mm @ 0.05mm/px):
- tile_w=512, tile_h=352 (pad from 350)
- stride_w=384 (overlap 128 px), stride_h=352 (no vertical tiling)

Dependencies:
  pip install torch torchvision pillow numpy tqdm matplotlib
Optional (faster kNN):
  pip install faiss-cpu   (or faiss-gpu if you know what you're doing)

Usage:
  # 1) Train OK-only model
  python patchcore_roi_tiling.py train ^
    --train_dir D:\data\roi_ok_train ^
    --model_out D:\data\patchcore_roi_model.pt ^
    --tile_w 512 --tile_h 352 --stride_w 384 --stride_h 352 ^
    --backbone resnet18 --layers layer2,layer3 ^
    --proj_dim 128 --max_patches_per_tile 256 --memory_size 20000 ^
    --batch_size 16 --device auto --compute_threshold 1

  # 2) Predict / generate heatmaps
  python patchcore_roi_tiling.py predict ^
    --model_path D:\data\patchcore_roi_model.pt ^
    --in_dir D:\data\roi_test ^
    --out_dir D:\data\roi_test_out ^
    --device auto --save_heatmap 1 --save_npy 0
"""

import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms


# ----------------------------- utils -----------------------------

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def list_images(folder: Path) -> List[Path]:
    files = []
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            files.append(p)
    files.sort()
    return files


def compute_positions(length: int, tile: int, stride: int) -> List[int]:
    """Positions ensuring last tile touches the end."""
    if tile >= length:
        return [0]
    xs = list(range(0, length - tile + 1, stride))
    last = length - tile
    if xs[-1] != last:
        xs.append(last)
    return xs


def pad_to_at_least(img: np.ndarray, min_h: int, min_w: int, mode: str = "edge") -> np.ndarray:
    """Pad H/W to at least min_h/min_w."""
    h, w = img.shape[:2]
    pad_h = max(0, min_h - h)
    pad_w = max(0, min_w - w)
    if pad_h == 0 and pad_w == 0:
        return img
    if img.ndim == 2:
        pad_width = ((0, pad_h), (0, pad_w))
    else:
        pad_width = ((0, pad_h), (0, pad_w), (0, 0))
    return np.pad(img, pad_width, mode=mode)


def ensure_rgb(np_img: np.ndarray) -> np.ndarray:
    """Convert grayscale to RGB if needed."""
    if np_img.ndim == 2:
        return np.stack([np_img, np_img, np_img], axis=-1)
    if np_img.shape[2] == 1:
        return np.repeat(np_img, 3, axis=2)
    if np_img.shape[2] >= 3:
        return np_img[:, :, :3]
    raise ValueError(f"Unexpected image shape: {np_img.shape}")


# ----------------------------- tiling dataset -----------------------------

@dataclass
class TileInfo:
    img_path: str
    x: int
    y: int
    orig_w: int
    orig_h: int


class TiledROIDataset(Dataset):
    def __init__(
        self,
        img_paths: List[Path],
        tile_w: int,
        tile_h: int,
        stride_w: int,
        stride_h: int,
        pad_mode: str = "edge",
        transform=None,
        return_info: bool = True,
    ):
        self.img_paths = img_paths
        self.tile_w = tile_w
        self.tile_h = tile_h
        self.stride_w = stride_w
        self.stride_h = stride_h
        self.pad_mode = pad_mode
        self.transform = transform
        self.return_info = return_info

        # Precompute all tiles index (img_idx, x, y)
        self.tiles = []
        # read each image size once
        for i, p in enumerate(self.img_paths):
            with Image.open(p) as im:
                w, h = im.size
            xs = compute_positions(w, tile_w, stride_w)
            ys = compute_positions(h, tile_h, stride_h)
            for y in ys:
                for x in xs:
                    self.tiles.append((i, x, y, w, h))

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        i, x, y, w, h = self.tiles[idx]
        p = self.img_paths[i]
        # load
        with Image.open(p) as im:
            im = im.convert("RGB")
            np_img = np.array(im)

        # pad if needed (for tile_h > h or tile_w > w)
        np_img = pad_to_at_least(np_img, self.tile_h, self.tile_w, mode=self.pad_mode)

        # crop tile
        tile = np_img[y:y + self.tile_h, x:x + self.tile_w, :]
        pil_tile = Image.fromarray(tile)

        if self.transform is not None:
            tile_t = self.transform(pil_tile)
        else:
            tile_t = transforms.ToTensor()(pil_tile)

        if not self.return_info:
            return tile_t

        info = TileInfo(img_path=str(p), x=x, y=y, orig_w=w, orig_h=h)
        return tile_t, info


# ----------------------------- model: resnet features + embedding -----------------------------

class ResNetFeat(nn.Module):
    def __init__(self, backbone_name: str = "resnet18", pretrained: bool = True, layers: List[str] = None):
        super().__init__()
        if layers is None:
            layers = ["layer2", "layer3"]
        self.layers = layers

        # build backbone
        backbone_name = backbone_name.lower()
        if backbone_name == "resnet18":
            try:
                w = models.ResNet18_Weights.DEFAULT if pretrained else None
            except Exception:
                w = None
            self.net = models.resnet18(weights=w)
        elif backbone_name == "resnet50":
            try:
                w = models.ResNet50_Weights.DEFAULT if pretrained else None
            except Exception:
                w = None
            self.net = models.resnet50(weights=w)
        else:
            raise ValueError("backbone must be resnet18 or resnet50")

        # remove classifier
        self.net.fc = nn.Identity()

        # ImageNet norm is expected by pretrained weights
        # (we do normalization in transform)

    def forward(self, x) -> Dict[str, torch.Tensor]:
        # copy of torchvision resnet forward with taps
        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = self.net.relu(x)
        x = self.net.maxpool(x)

        f1 = self.net.layer1(x)
        f2 = self.net.layer2(f1)
        f3 = self.net.layer3(f2)
        f4 = self.net.layer4(f3)

        feats = {
            "layer1": f1,
            "layer2": f2,
            "layer3": f3,
            "layer4": f4,
        }
        return {k: feats[k] for k in self.layers}


class PatchEmbedder(nn.Module):
    """
    Select multiple feature maps -> upsample to highest spatial res among selected -> concat -> (optional) random projection.
    Output: embedding map (B, C_emb, H, W) at base resolution of first layer in `layers` order.
    """
    def __init__(self, backbone: ResNetFeat, layers: List[str], proj_dim: int = 128, seed: int = 42):
        super().__init__()
        self.backbone = backbone
        self.layers = layers
        self.proj_dim = int(proj_dim)
        self.seed = int(seed)

        # will init projection after first forward (need channel dim)
        self.register_buffer("proj_mat", torch.empty(0), persistent=True)

    def _init_proj(self, in_dim: int, device: torch.device):
        if self.proj_dim <= 0 or self.proj_dim >= in_dim:
            self.proj_mat = torch.empty(0, device=device)
            return
        g = torch.Generator(device="cpu")
        g.manual_seed(self.seed)
        # random gaussian projection
        mat = torch.randn(in_dim, self.proj_dim, generator=g, dtype=torch.float32)
        # normalize columns a bit (optional)
        mat = mat / (mat.norm(dim=0, keepdim=True) + 1e-12)
        self.proj_mat = mat.to(device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)  # dict
        # base feature = first layer in list (usually layer2)
        base = feats[self.layers[0]]
        base_h, base_w = base.shape[-2:]

        ups = []
        for k in self.layers:
            f = feats[k]
            if f.shape[-2:] != (base_h, base_w):
                f = F.interpolate(f, size=(base_h, base_w), mode="bilinear", align_corners=False)
            ups.append(f)
        emb = torch.cat(ups, dim=1)  # (B,C,H,W)

        # init projection if needed
        if self.proj_mat.numel() == 0 and self.proj_dim > 0:
            self._init_proj(emb.shape[1], emb.device)

        if self.proj_mat.numel() > 0:
            # (B,C,H,W) -> (B,H,W,C) -> projection -> back
            b, c, h, w = emb.shape
            emb_ = emb.permute(0, 2, 3, 1).contiguous().view(-1, c)  # (B*H*W, C)
            emb_ = emb_ @ self.proj_mat  # (B*H*W, proj_dim)
            emb_ = emb_.view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
            emb = emb_

        # L2 normalize (helps kNN stability)
        emb = F.normalize(emb, p=2, dim=1)
        return emb


# ----------------------------- kNN scorer -----------------------------

class KNNSearcher:
    def __init__(self, memory_bank: np.ndarray, use_faiss: bool = True):
        """
        memory_bank: (M, D) float32
        """
        self.use_faiss = False
        self.index = None
        self.mem_torch = None

        memory_bank = np.ascontiguousarray(memory_bank.astype(np.float32))
        self.memory_bank = memory_bank
        self.dim = memory_bank.shape[1]

        if use_faiss:
            try:
                import faiss  # type: ignore
                self.faiss = faiss
                index = faiss.IndexFlatL2(self.dim)
                index.add(memory_bank)
                self.index = index
                self.use_faiss = True
            except Exception:
                self.use_faiss = False

    def _min_dist_torch(self, queries: torch.Tensor, bank: torch.Tensor, chunk: int = 4096) -> torch.Tensor:
        """
        queries: (N, D) float32 on device
        bank: (M, D) float32 on same device
        return: (N,) float32 L2 distance
        """
        # squared norms
        qn = (queries * queries).sum(dim=1, keepdim=True)  # (N,1)
        min_d2 = None
        m = bank.shape[0]
        for s in range(0, m, chunk):
            b = bank[s:s + chunk]  # (c,D)
            bn = (b * b).sum(dim=1).unsqueeze(0)  # (1,c)
            prod = queries @ b.t()  # (N,c)
            d2 = qn + bn - 2.0 * prod
            cur = d2.min(dim=1).values
            min_d2 = cur if min_d2 is None else torch.minimum(min_d2, cur)
        min_d2 = torch.clamp(min_d2, min=0.0)
        return torch.sqrt(min_d2 + 1e-12)

    def query_min_dist(self, queries: torch.Tensor) -> torch.Tensor:
        """
        queries: (N,D) torch tensor on GPU/CPU
        returns: (N,) torch tensor on same device
        """
        if self.use_faiss:
            q = queries.detach().float().cpu().numpy()
            D, _ = self.index.search(q, 1)  # squared L2
            d = np.sqrt(np.maximum(D[:, 0], 0.0) + 1e-12).astype(np.float32)
            return torch.from_numpy(d).to(device=queries.device)

        # torch fallback
        if self.mem_torch is None or self.mem_torch.device != queries.device:
            self.mem_torch = torch.from_numpy(self.memory_bank).to(device=queries.device)
        return self._min_dist_torch(queries.float(), self.mem_torch.float())


# ----------------------------- train / predict core -----------------------------

def build_transform():
    # ImageNet normalization
    try:
        mean = models.ResNet18_Weights.DEFAULT.transforms().mean
        std = models.ResNet18_Weights.DEFAULT.transforms().std
        mean = list(mean)
        std = list(std)
    except Exception:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]), mean, std


def infer_one_image(
    img_path: Path,
    embedder: PatchEmbedder,
    knn: KNNSearcher,
    tile_w: int, tile_h: int, stride_w: int, stride_h: int,
    device: torch.device,
    pad_mode: str = "edge",
    batch_tiles: int = 8,
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """
    return:
      full_map: (H,W) anomaly heatmap in pixel space
      score: scalar score (q0.999 of heatmap)
      (H,W): original image size
    """
    # load full ROI
    with Image.open(img_path) as im:
        im = im.convert("RGB")
        np_img = np.array(im)
    H0, W0 = np_img.shape[0], np_img.shape[1]

    # pad only if needed
    np_img_pad = pad_to_at_least(np_img, tile_h, tile_w, mode=pad_mode)
    Hp, Wp = np_img_pad.shape[0], np_img_pad.shape[1]

    xs = compute_positions(Wp, tile_w, stride_w)
    ys = compute_positions(Hp, tile_h, stride_h)

    # transform
    tfm, _, _ = build_transform()

    # stitch map (use max over overlaps)
    full_map = np.zeros((Hp, Wp), dtype=np.float32)

    tiles = []
    infos = []
    for y in ys:
        for x in xs:
            crop = np_img_pad[y:y + tile_h, x:x + tile_w, :]
            tiles.append(tfm(Image.fromarray(crop)))
            infos.append((x, y))

    embedder.eval()
    with torch.no_grad():
        for s in range(0, len(tiles), batch_tiles):
            batch = torch.stack(tiles[s:s + batch_tiles], dim=0).to(device, non_blocking=True)
            emb_map = embedder(batch)  # (B,C,Hf,Wf)
            b, c, hf, wf = emb_map.shape
            patches = emb_map.permute(0, 2, 3, 1).contiguous().view(-1, c)  # (B*hf*wf, C)

            dists = knn.query_min_dist(patches)  # (B*hf*wf,)
            dists = dists.view(b, 1, hf, wf)

            # upsample to tile size
            up = F.interpolate(dists, size=(tile_h, tile_w), mode="bilinear", align_corners=False)
            up_np = up.squeeze(1).detach().cpu().numpy()  # (B,tile_h,tile_w)

            for i in range(up_np.shape[0]):
                x, y = infos[s + i]
                tile_map = up_np[i]
                # max stitch
                region = full_map[y:y + tile_h, x:x + tile_w]
                np.maximum(region, tile_map, out=region)
                full_map[y:y + tile_h, x:x + tile_w] = region

    # crop back to original
    full_map = full_map[:H0, :W0]

    # robust scalar score
    score = float(np.quantile(full_map, 0.999))
    return full_map, score, (H0, W0)


def save_heatmap_png(out_png: Path, heatmap: np.ndarray, title: str = "", vmin=None, vmax=None):
    import matplotlib.pyplot as plt
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.imshow(heatmap, cmap="jet", vmin=vmin, vmax=vmax)
    plt.colorbar()
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png.as_posix(), dpi=200)
    plt.close()

def save_u16_and_mask(out_base: Path, heatmap: np.ndarray, vmin: float, vmax: float,
                      thr: float, mask_thr_scale: float = 0.99, dilate_px: int = 1, close_px: int = 0):
    """
    mask_thr_scale:
      - 1.00 = 用原 threshold
      - 0.85 = 用 0.85*threshold（更容易把细划痕连成片）
    dilate_px:
      - 膨胀半径（像素），建议 6~15
    close_px:
      - 闭运算半径（先膨胀再腐蚀），用于把断裂的小段连起来；0 表示不做
    """
    out_base.parent.mkdir(parents=True, exist_ok=True)

    # clip + normalize to [0,1]
    hm = np.clip(heatmap, vmin, vmax)
    hm01 = (hm - vmin) / (vmax - vmin + 1e-12)

    # 16-bit score map (统一尺度)
    u16 = (hm01 * 65535.0).astype(np.uint16)
    Image.fromarray(u16, mode="I;16").save(out_base.with_suffix(".u16.png"))

    # --- mask threshold (use a lower threshold for segmentation than decision threshold) ---
    thr_mask = float(thr) * float(mask_thr_scale)
    thr01 = (thr_mask - vmin) / (vmax - vmin + 1e-12)
    mask = (hm01 >= thr01).astype(np.uint8) * 255  # 0/255

    # --- morphology to enlarge thin scratches ---
    # We implement simple dilation/erosion with max/min filters using numpy sliding window.
    def dilate(binary255: np.ndarray, r: int) -> np.ndarray:
        if r <= 0:
            return binary255
        b = (binary255 > 0).astype(np.uint8)
        H, W = b.shape
        out = np.zeros_like(b)
        # max filter
        for dy in range(-r, r + 1):
            y0 = max(0, dy)
            y1 = min(H, H + dy)
            sy0 = max(0, -dy)
            sy1 = sy0 + (y1 - y0)
            for dx in range(-r, r + 1):
                x0 = max(0, dx)
                x1 = min(W, W + dx)
                sx0 = max(0, -dx)
                sx1 = sx0 + (x1 - x0)
                out[y0:y1, x0:x1] = np.maximum(out[y0:y1, x0:x1], b[sy0:sy1, sx0:sx1])
        return (out * 255).astype(np.uint8)

    def erode(binary255: np.ndarray, r: int) -> np.ndarray:
        if r <= 0:
            return binary255
        b = (binary255 > 0).astype(np.uint8)
        H, W = b.shape
        out = np.ones_like(b)
        # min filter
        for dy in range(-r, r + 1):
            y0 = max(0, dy)
            y1 = min(H, H + dy)
            sy0 = max(0, -dy)
            sy1 = sy0 + (y1 - y0)
            for dx in range(-r, r + 1):
                x0 = max(0, dx)
                x1 = min(W, W + dx)
                sx0 = max(0, -dx)
                sx1 = sx0 + (x1 - x0)
                out[y0:y1, x0:x1] = np.minimum(out[y0:y1, x0:x1], b[sy0:sy1, sx0:sx1])
        return (out * 255).astype(np.uint8)

    # dilate
    mask2 = dilate(mask, int(dilate_px))

    # optional close: dilate then erode (fills small gaps)
    if close_px and close_px > 0:
        mask2 = erode(mask2, int(close_px))

    Image.fromarray(mask2, mode="L").save(out_base.with_suffix(".mask.png"))

def train_patchcore(args):
    set_seed(args.seed)

    train_dir = Path(args.train_dir)
    img_paths = list_images(train_dir)
    if not img_paths:
        raise RuntimeError(f"No images found in: {train_dir}")

    # device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # transform
    tfm, mean, std = build_transform()

    # dataset: tiled
    ds = TiledROIDataset(
        img_paths=img_paths,
        tile_w=args.tile_w,
        tile_h=args.tile_h,
        stride_w=args.stride_w,
        stride_h=args.stride_h,
        pad_mode=args.pad_mode,
        transform=tfm,
        return_info=False,
    )
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    # model
    layers = [s.strip() for s in args.layers.split(",") if s.strip()]
    backbone = ResNetFeat(args.backbone, pretrained=bool(args.pretrained), layers=layers)
    embedder = PatchEmbedder(backbone, layers=layers, proj_dim=args.proj_dim, seed=args.seed).to(device)
    embedder.eval()

    # collect sampled patch embeddings
    sampled_feats = []
    total_tiles = len(ds)
    pbar = tqdm(dl, total=math.ceil(total_tiles / args.batch_size), desc="Extract & sample OK patches")

    with torch.no_grad():
        for batch in pbar:
            batch = batch.to(device, non_blocking=True)
            emb_map = embedder(batch)  # (B,C,Hf,Wf)
            b, c, hf, wf = emb_map.shape
            emb_flat = emb_map.permute(0, 2, 3, 1).contiguous().view(b, -1, c)  # (B,HW,C)
            hw = emb_flat.shape[1]
            k = min(args.max_patches_per_tile, hw)

            # sample k patches per tile
            for i in range(b):
                idx = torch.randperm(hw, device=emb_flat.device)[:k]
                feats = emb_flat[i, idx, :]  # (k,C)
                sampled_feats.append(feats.detach().cpu().half())  # store fp16 on CPU to save RAM

    feats_all = torch.cat(sampled_feats, dim=0)  # (N,C) fp16 cpu
    N, C = feats_all.shape
    print(f"[INFO] sampled patches total: {N}, dim: {C}")

    # memory bank sampling
    M = min(args.memory_size, N)
    sel = torch.randperm(N)[:M]
    memory_bank = feats_all[sel].float().numpy().astype(np.float32)  # (M,C) float32

    print(f"[INFO] memory bank size: {M}, dim: {C}")

    # (optional) compute OK threshold by running prediction on training images
    threshold = None
    if args.compute_threshold:
        print("[INFO] computing auto threshold from OK set (this will run inference on train images)...")
        knn = KNNSearcher(memory_bank, use_faiss=bool(args.use_faiss))
        scores = []
        for p in tqdm(img_paths, desc="Infer OK scores"):
            _, s, _ = infer_one_image(
                img_path=p,
                embedder=embedder,
                knn=knn,
                tile_w=args.tile_w, tile_h=args.tile_h,
                stride_w=args.stride_w, stride_h=args.stride_h,
                device=device,
                pad_mode=args.pad_mode,
                batch_tiles=args.infer_batch_tiles,
            )
            scores.append(s)
        scores = np.array(scores, dtype=np.float32)
        # quantile-based threshold (OK-only)
        q = float(np.quantile(scores, args.ok_quantile))
        threshold = q * float(args.thr_scale)
        print(f"[INFO] OK scores quantile={args.ok_quantile}: {q:.6f}, threshold={threshold:.6f}")

    # save model package
    model_out = Path(args.model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)

    pack = {
        "meta": {
            "backbone": args.backbone,
            "layers": layers,
            "proj_dim": args.proj_dim,
            "tile_w": args.tile_w,
            "tile_h": args.tile_h,
            "stride_w": args.stride_w,
            "stride_h": args.stride_h,
            "pad_mode": args.pad_mode,
            "mean": mean,
            "std": std,
            "pixel_mm": args.pixel_mm,
            "defect_mm_min": args.defect_mm_min,
            "defect_px_min": float(args.defect_mm_min / args.pixel_mm),
            "threshold": threshold,
            "ok_quantile": args.ok_quantile,
            "thr_scale": args.thr_scale,
            "created_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "state_dict": embedder.state_dict(),
        "memory_bank": memory_bank,  # numpy float32
    }

    torch.save(pack, model_out.as_posix())
    print(f"[DONE] saved model: {model_out}")


def predict_patchcore(args):
    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # load model pack
    pack = torch.load(args.model_path, map_location="cpu", weights_only=False)
    meta = pack["meta"]
    memory_bank = pack["memory_bank"].astype(np.float32)

    # rebuild embedder
    layers = meta["layers"]
    backbone = ResNetFeat(meta["backbone"], pretrained=False, layers=layers)  # weights not needed for inference load
    embedder = PatchEmbedder(backbone, layers=layers, proj_dim=int(meta["proj_dim"]), seed=42)

    # --- FIX: init proj_mat shape before loading state_dict (PyTorch strict load) ---
    sd = pack["state_dict"]
    if "proj_mat" in sd:
         # make proj_mat the same shape as checkpoint so strict load works
         embedder.proj_mat = sd["proj_mat"].detach().clone()
        # ------------------------------------------------------------------------------

    embedder.load_state_dict(sd, strict=True)
    embedder = embedder.to(device).eval()

    # kNN index
    knn = KNNSearcher(memory_bank, use_faiss=bool(args.use_faiss))

    # threshold
    threshold = meta.get("threshold", None)
    if args.threshold is not None:
        threshold = float(args.threshold)
    # 建议固定色标：0 到 threshold*1.2
    vmin = 0.0
    vmax = float(threshold * 1.2) if threshold is not None else None
    img_paths = list_images(in_dir)
    if not img_paths:
        raise RuntimeError(f"No images found in: {in_dir}")

    results = []
    for p in tqdm(img_paths, desc="Predict"):
        stem = p.stem  # <- 必须先定义

        heatmap, score, (H0, W0) = infer_one_image(
            img_path=p,
            embedder=embedder,
            knn=knn,
            tile_w=int(meta["tile_w"]), tile_h=int(meta["tile_h"]),
            stride_w=int(meta["stride_w"]), stride_h=int(meta["stride_h"]),
            device=device,
            pad_mode=meta.get("pad_mode", "edge"),
            batch_tiles=args.infer_batch_tiles,
        )

        pred = "NG" if (threshold is not None and score >= threshold) else "OK"
        results.append({
            "file": p.name,
            "score": float(score),
            "threshold": None if threshold is None else float(threshold),
            "pred": pred,
            "size": [int(W0), int(H0)],
        })

        # ---- export unified-scale u16 + mask (best for vision software) ----
        if threshold is not None:
            vmin = 0.0
            vmax = float(threshold * 1.2)
            save_u16_and_mask(out_dir / f"{stem}_score", heatmap, vmin=vmin, vmax=vmax, thr=float(threshold))
        # -------------------------------------------------------------------

        if args.save_heatmap:
            # 统一色标（如果没有 threshold，就让 matplotlib 自己缩放）
            save_heatmap_png(
                out_dir / f"{stem}_heatmap.png",
                heatmap,
                title=f"{p.name} score={score:.4f} pred={pred}",
                vmin=(0.0 if threshold is not None else None),
                vmax=(float(threshold * 1.2) if threshold is not None else None),
            )

        if args.save_npy:
            np.save(out_dir / f"{stem}_heatmap.npy", heatmap)

    # save summary json
    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "results": results}, f, ensure_ascii=False, indent=2)

    print(f"[DONE] results saved: {out_dir / 'results.json'}")


# ----------------------------- CLI -----------------------------

def build_argparser():
    ap = argparse.ArgumentParser("PatchCore ROI Tiling")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # train
    tr = sub.add_parser("train", help="Train OK-only PatchCore memory bank")
    tr.add_argument("--train_dir", required=True, help="OK images folder")
    tr.add_argument("--model_out", required=True, help="output .pt path")

    tr.add_argument("--tile_w", type=int, default=512)
    tr.add_argument("--tile_h", type=int, default=352)
    tr.add_argument("--stride_w", type=int, default=384, help="overlap = tile_w - stride_w (default overlap=128px)")
    tr.add_argument("--stride_h", type=int, default=352)
    tr.add_argument("--pad_mode", type=str, default="edge", choices=["edge", "reflect", "constant"])

    tr.add_argument("--backbone", type=str, default="resnet18", choices=["resnet18", "resnet50"])
    tr.add_argument("--layers", type=str, default="layer2,layer3", help="e.g. layer2,layer3")
    tr.add_argument("--pretrained", type=int, default=1, help="1: use ImageNet pretrained weights if available")
    tr.add_argument("--proj_dim", type=int, default=128, help="random projection dim (0 disables). Suggest 128/256")

    tr.add_argument("--max_patches_per_tile", type=int, default=256, help="sampled patches per tile for memory bank build")
    tr.add_argument("--memory_size", type=int, default=20000, help="final memory bank size")

    tr.add_argument("--batch_size", type=int, default=16)
    tr.add_argument("--num_workers", type=int, default=2)
    tr.add_argument("--device", type=str, default="auto", help="auto/cuda/cpu")
    tr.add_argument("--seed", type=int, default=42)

    tr.add_argument("--use_faiss", type=int, default=1, help="try use faiss for threshold inference speed")
    tr.add_argument("--infer_batch_tiles", type=int, default=8, help="tiles per batch during inference")

    tr.add_argument("--compute_threshold", type=int, default=1, help="auto threshold from OK set (slower)")
    tr.add_argument("--ok_quantile", type=float, default=0.999, help="quantile of OK scores to set threshold")
    tr.add_argument("--thr_scale", type=float, default=1.10, help="scale factor on quantile to reduce false NG")

    # for record
    tr.add_argument("--pixel_mm", type=float, default=0.05, help="mm per pixel")
    tr.add_argument("--defect_mm_min", type=float, default=2.0, help="defect size threshold in mm")

    # predict
    pr = sub.add_parser("predict", help="Predict heatmaps / scores")
    pr.add_argument("--model_path", required=True, help=".pt model path from train")
    pr.add_argument("--in_dir", required=True, help="input images folder")
    pr.add_argument("--out_dir", required=True, help="output folder")

    pr.add_argument("--device", type=str, default="auto")
    pr.add_argument("--use_faiss", type=int, default=1)
    pr.add_argument("--infer_batch_tiles", type=int, default=8)

    pr.add_argument("--threshold", type=float, default=None, help="override threshold (else use saved)")
    pr.add_argument("--save_heatmap", type=int, default=1)
    pr.add_argument("--save_npy", type=int, default=0)

    return ap


def main():
    ap = build_argparser()
    args = ap.parse_args()

    if args.cmd == "train":
        train_patchcore(args)
    elif args.cmd == "predict":
        predict_patchcore(args)
    else:
        raise ValueError(args.cmd)


if __name__ == "__main__":
    main()
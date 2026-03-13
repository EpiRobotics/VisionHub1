"""PatchCore core algorithm implementation.

Adapted from example/patchcore_roi_tiling.py.
Provides:
- ResNetFeat: ResNet backbone feature extractor with layer taps
- PatchEmbedder: Multi-layer feature concatenation + random projection
- KNNSearcher: kNN with faiss or torch fallback
- Tiling utilities (compute_positions, pad_to_at_least)
- infer_one_image: full tiled inference pipeline
- save_heatmap_png, save_u16_and_mask: artifact export
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

# ----------------------------- constants / utils -----------------------------

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def list_images(folder: Path) -> list[Path]:
    files = [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]
    files.sort()
    return files


def compute_positions(length: int, tile: int, stride: int) -> list[int]:
    """Tile positions ensuring last tile touches the end."""
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
    """Dataset that tiles images for PatchCore training."""

    def __init__(
        self,
        img_paths: list[Path],
        tile_w: int,
        tile_h: int,
        stride_w: int,
        stride_h: int,
        pad_mode: str = "edge",
        transform: Any = None,
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

        # Precompute all tiles index (img_idx, x, y, w, h)
        self.tiles: list[tuple[int, int, int, int, int]] = []
        for i, p in enumerate(self.img_paths):
            with Image.open(p) as im:
                w, h = im.size
            xs = compute_positions(w, tile_w, stride_w)
            ys = compute_positions(h, tile_h, stride_h)
            for y in ys:
                for x in xs:
                    self.tiles.append((i, x, y, w, h))

    def __len__(self) -> int:
        return len(self.tiles)

    def __getitem__(self, idx: int) -> Any:
        i, x, y, w, h = self.tiles[idx]
        p = self.img_paths[i]
        with Image.open(p) as im:
            im = im.convert("RGB")
            np_img = np.array(im)

        np_img = pad_to_at_least(np_img, self.tile_h, self.tile_w, mode=self.pad_mode)
        tile = np_img[y : y + self.tile_h, x : x + self.tile_w, :]
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
    """ResNet backbone with intermediate layer feature extraction."""

    def __init__(self, backbone_name: str = "resnet18", pretrained: bool = True, layers: list[str] | None = None):
        super().__init__()
        if layers is None:
            layers = ["layer2", "layer3"]
        self.layers = layers

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

        self.net.fc = nn.Identity()

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = self.net.relu(x)
        x = self.net.maxpool(x)

        f1 = self.net.layer1(x)
        f2 = self.net.layer2(f1)
        f3 = self.net.layer3(f2)
        f4 = self.net.layer4(f3)

        feats = {"layer1": f1, "layer2": f2, "layer3": f3, "layer4": f4}
        return {k: feats[k] for k in self.layers}


class PatchEmbedder(nn.Module):
    """
    Multi-layer feature concatenation + optional random projection.
    Output: embedding map (B, C_emb, H, W).
    """

    def __init__(self, backbone: ResNetFeat, layers: list[str], proj_dim: int = 128, seed: int = 42):
        super().__init__()
        self.backbone = backbone
        self.layers = layers
        self.proj_dim = int(proj_dim)
        self.seed = int(seed)
        self.register_buffer("proj_mat", torch.empty(0), persistent=True)

    def _init_proj(self, in_dim: int, device: torch.device) -> None:
        if self.proj_dim <= 0 or self.proj_dim >= in_dim:
            self.proj_mat = torch.empty(0, device=device)
            return
        g = torch.Generator(device="cpu")
        g.manual_seed(self.seed)
        mat = torch.randn(in_dim, self.proj_dim, generator=g, dtype=torch.float32)
        mat = mat / (mat.norm(dim=0, keepdim=True) + 1e-12)
        self.proj_mat = mat.to(device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        base = feats[self.layers[0]]
        base_h, base_w = base.shape[-2:]

        ups = []
        for k in self.layers:
            f = feats[k]
            if f.shape[-2:] != (base_h, base_w):
                f = F.interpolate(f, size=(base_h, base_w), mode="bilinear", align_corners=False)
            ups.append(f)
        emb = torch.cat(ups, dim=1)

        if self.proj_mat.numel() == 0 and self.proj_dim > 0:
            self._init_proj(emb.shape[1], emb.device)

        if self.proj_mat.numel() > 0:
            b, c, h, w = emb.shape
            emb_ = emb.permute(0, 2, 3, 1).contiguous().view(-1, c)
            emb_ = emb_ @ self.proj_mat
            emb_ = emb_.view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
            emb = emb_

        emb = F.normalize(emb, p=2, dim=1)
        return emb


# ----------------------------- kNN scorer -----------------------------


class KNNSearcher:
    """kNN searcher with faiss or torch fallback."""

    def __init__(self, memory_bank: np.ndarray, use_faiss: bool = True):
        self.use_faiss = False
        self.index: Any = None
        self.mem_torch: torch.Tensor | None = None

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
        qn = (queries * queries).sum(dim=1, keepdim=True)
        min_d2: torch.Tensor | None = None
        m = bank.shape[0]
        for s in range(0, m, chunk):
            b = bank[s : s + chunk]
            bn = (b * b).sum(dim=1).unsqueeze(0)
            prod = queries @ b.t()
            d2 = qn + bn - 2.0 * prod
            cur = d2.min(dim=1).values
            min_d2 = cur if min_d2 is None else torch.minimum(min_d2, cur)
        assert min_d2 is not None
        min_d2 = torch.clamp(min_d2, min=0.0)
        return torch.sqrt(min_d2 + 1e-12)

    def query_min_dist(self, queries: torch.Tensor) -> torch.Tensor:
        if self.use_faiss:
            q = queries.detach().float().cpu().numpy()
            d_sq, _ = self.index.search(q, 1)
            d = np.sqrt(np.maximum(d_sq[:, 0], 0.0) + 1e-12).astype(np.float32)
            return torch.from_numpy(d).to(device=queries.device)

        if self.mem_torch is None or self.mem_torch.device != queries.device:
            self.mem_torch = torch.from_numpy(self.memory_bank).to(device=queries.device)
        return self._min_dist_torch(queries.float(), self.mem_torch.float())


# ----------------------------- inference core -----------------------------


def build_transform() -> tuple[Any, list[float], list[float]]:
    """Build ImageNet normalization transform."""
    try:
        mean = list(models.ResNet18_Weights.DEFAULT.transforms().mean)
        std = list(models.ResNet18_Weights.DEFAULT.transforms().std)
    except Exception:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    return (
        transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]),
        mean,
        std,
    )


def infer_one_image(
    img_path: Path,
    embedder: PatchEmbedder,
    knn: KNNSearcher,
    tile_w: int,
    tile_h: int,
    stride_w: int,
    stride_h: int,
    device: torch.device,
    pad_mode: str = "edge",
    batch_tiles: int = 8,
) -> tuple[np.ndarray, float, tuple[int, int]]:
    """
    Full tiled inference on a single image.

    Returns:
        full_map: (H, W) anomaly heatmap
        score: scalar score (q0.999 of heatmap)
        (H, W): original image size
    """
    with Image.open(img_path) as im:
        im = im.convert("RGB")
        np_img = np.array(im)
    h0, w0 = np_img.shape[0], np_img.shape[1]

    np_img_pad = pad_to_at_least(np_img, tile_h, tile_w, mode=pad_mode)
    hp, wp = np_img_pad.shape[0], np_img_pad.shape[1]

    xs = compute_positions(wp, tile_w, stride_w)
    ys = compute_positions(hp, tile_h, stride_h)

    tfm, _, _ = build_transform()

    full_map = np.zeros((hp, wp), dtype=np.float32)

    tiles = []
    infos = []
    for y in ys:
        for x in xs:
            crop = np_img_pad[y : y + tile_h, x : x + tile_w, :]
            tiles.append(tfm(Image.fromarray(crop)))
            infos.append((x, y))

    embedder.eval()
    with torch.no_grad():
        for s in range(0, len(tiles), batch_tiles):
            batch = torch.stack(tiles[s : s + batch_tiles], dim=0).to(device, non_blocking=True)
            emb_map = embedder(batch)
            b, c, hf, wf = emb_map.shape
            patches = emb_map.permute(0, 2, 3, 1).contiguous().view(-1, c)

            dists = knn.query_min_dist(patches)
            dists = dists.view(b, 1, hf, wf)

            up = F.interpolate(dists, size=(tile_h, tile_w), mode="bilinear", align_corners=False)
            up_np = up.squeeze(1).detach().cpu().numpy()

            for i in range(up_np.shape[0]):
                x, y = infos[s + i]
                tile_map = up_np[i]
                region = full_map[y : y + tile_h, x : x + tile_w]
                np.maximum(region, tile_map, out=region)
                full_map[y : y + tile_h, x : x + tile_w] = region

    full_map = full_map[:h0, :w0]
    score = float(np.quantile(full_map, 0.999))
    return full_map, score, (h0, w0)


# ----------------------------- artifact saving -----------------------------


def save_heatmap_png(
    out_png: Path,
    heatmap: np.ndarray,
    title: str = "",
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    """Save heatmap as a colored PNG using matplotlib."""
    import matplotlib

    matplotlib.use("Agg")
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


def save_overlay_png(
    out_png: Path,
    img_path: Path,
    heatmap: np.ndarray,
    vmin: float = 0.0,
    vmax: float = 1.0,
    alpha: float = 0.4,
) -> None:
    """Save an overlay of heatmap on the original image (matplotlib version)."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import cm

    out_png.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(img_path) as im:
        im = im.convert("RGB")
        np_img = np.array(im)

    hm_clipped = np.clip(heatmap, vmin, vmax)
    hm_norm = (hm_clipped - vmin) / (vmax - vmin + 1e-12)
    hm_color = (cm.jet(hm_norm)[:, :, :3] * 255).astype(np.uint8)

    blended = (np_img.astype(np.float32) * (1 - alpha) + hm_color.astype(np.float32) * alpha).astype(np.uint8)
    Image.fromarray(blended).save(str(out_png))


def save_overlay_cv2(
    out_path: Path,
    img_path: Path,
    heatmap: np.ndarray,
    vmin: float = 0.0,
    vmax: float = 1.0,
    alpha: float = 0.4,
    score: float | None = None,
    threshold: float | None = None,
    pred: str | None = None,
) -> None:
    """Save overlay using OpenCV (fast, no matplotlib dependency).

    Uses cv2.applyColorMap(COLORMAP_JET) instead of matplotlib.cm.jet.
    Typically 10-50x faster than the matplotlib version.

    If score/threshold/pred are provided, draws OK/NG text and score on the image.
    """
    import cv2

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Read original image
    with Image.open(img_path) as im:
        im = im.convert("RGB")
        np_img = np.array(im)

    # Normalize heatmap to 0-255
    hm_clipped = np.clip(heatmap, vmin, vmax)
    hm_norm = ((hm_clipped - vmin) / (vmax - vmin + 1e-12) * 255).astype(np.uint8)

    # Apply JET colormap (OpenCV uses BGR)
    hm_color_bgr = cv2.applyColorMap(hm_norm, cv2.COLORMAP_JET)
    hm_color_rgb = cv2.cvtColor(hm_color_bgr, cv2.COLOR_BGR2RGB)

    # Blend
    blended = (np_img.astype(np.float32) * (1 - alpha) + hm_color_rgb.astype(np.float32) * alpha).astype(np.uint8)

    # Draw OK/NG text and score on the image
    if pred is not None and score is not None and threshold is not None:
        blended_bgr = cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)
        img_h, img_w = blended_bgr.shape[:2]

        # Scale font size based on image dimensions
        scale_ref = min(img_w, img_h)
        font_scale_label = max(0.8, scale_ref / 600.0)
        font_scale_score = max(0.5, scale_ref / 900.0)
        thickness_label = max(2, int(scale_ref / 300))
        thickness_score = max(1, int(scale_ref / 500))

        # Colors: green for OK, red for NG
        color = (0, 200, 0) if pred == "OK" else (0, 0, 255)

        # Draw label (OK/NG) at top-left
        label_text = pred
        cv2.putText(
            blended_bgr, label_text, (10, int(40 * font_scale_label)),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale_label, color,
            thickness_label, cv2.LINE_AA,
        )

        # Draw score info below the label
        score_text = f"Score: {score:.4f} / Thr: {threshold:.4f}"
        y_offset = int(40 * font_scale_label) + int(35 * font_scale_score)
        cv2.putText(
            blended_bgr, score_text, (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale_score, (255, 255, 255),
            thickness_score, cv2.LINE_AA,
        )

        blended = cv2.cvtColor(blended_bgr, cv2.COLOR_BGR2RGB)

    # Save as JPEG for speed (much faster than PNG for large images)
    suffix = out_path.suffix.lower()
    if suffix in (".jpg", ".jpeg"):
        cv2.imwrite(str(out_path), cv2.cvtColor(blended, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
    else:
        Image.fromarray(blended).save(str(out_path))


def save_u16_and_mask(
    out_base: Path,
    heatmap: np.ndarray,
    vmin: float,
    vmax: float,
    thr: float,
    mask_thr_scale: float = 0.85,
    dilate_px: int = 10,
    close_px: int = 6,
) -> tuple[str, str]:
    """
    Save 16-bit score map and binary mask.

    Returns:
        (u16_path, mask_path)
    """
    out_base.parent.mkdir(parents=True, exist_ok=True)

    hm = np.clip(heatmap, vmin, vmax)
    hm01 = (hm - vmin) / (vmax - vmin + 1e-12)

    # 16-bit score map
    u16 = (hm01 * 65535.0).astype(np.uint16)
    u16_path = str(out_base.with_suffix(".u16.png"))
    Image.fromarray(u16, mode="I;16").save(u16_path)

    # mask threshold
    thr_mask = float(thr) * float(mask_thr_scale)
    thr01 = (thr_mask - vmin) / (vmax - vmin + 1e-12)
    mask = (hm01 >= thr01).astype(np.uint8) * 255

    # morphology
    mask = _dilate(mask, int(dilate_px))
    if close_px and close_px > 0:
        mask = _dilate(mask, int(close_px))
        mask = _erode(mask, int(close_px))

    mask_path = str(out_base.with_suffix(".mask.png"))
    Image.fromarray(mask, mode="L").save(mask_path)

    return u16_path, mask_path


def _dilate(binary255: np.ndarray, r: int) -> np.ndarray:
    if r <= 0:
        return binary255
    b = (binary255 > 0).astype(np.uint8)
    h, w = b.shape
    out = np.zeros_like(b)
    for dy in range(-r, r + 1):
        y0 = max(0, dy)
        y1 = min(h, h + dy)
        sy0 = max(0, -dy)
        sy1 = sy0 + (y1 - y0)
        for dx in range(-r, r + 1):
            x0 = max(0, dx)
            x1 = min(w, w + dx)
            sx0 = max(0, -dx)
            sx1 = sx0 + (x1 - x0)
            out[y0:y1, x0:x1] = np.maximum(out[y0:y1, x0:x1], b[sy0:sy1, sx0:sx1])
    return (out * 255).astype(np.uint8)


def _erode(binary255: np.ndarray, r: int) -> np.ndarray:
    if r <= 0:
        return binary255
    b = (binary255 > 0).astype(np.uint8)
    h, w = b.shape
    out = np.ones_like(b)
    for dy in range(-r, r + 1):
        y0 = max(0, dy)
        y1 = min(h, h + dy)
        sy0 = max(0, -dy)
        sy1 = sy0 + (y1 - y0)
        for dx in range(-r, r + 1):
            x0 = max(0, dx)
            x1 = min(w, w + dx)
            sx0 = max(0, -dx)
            sx1 = sx0 + (x1 - x0)
            out[y0:y1, x0:x1] = np.minimum(out[y0:y1, x0:x1], b[sy0:sy1, sx0:sx1])
    return (out * 255).astype(np.uint8)


# ----------------------------- connected components (regions) -----------------------------


def extract_regions(
    mask: np.ndarray,
    heatmap: np.ndarray,
    min_area_px: int = 80,
    max_regions: int = 10,
) -> list[dict[str, Any]]:
    """
    Extract connected component regions from a binary mask.

    Returns list of dicts: {x, y, w, h, score, area_px}
    """
    if mask.max() == 0:
        return []

    binary = (mask > 0).astype(np.uint8)
    h, w = binary.shape

    # Simple flood-fill connected components
    visited = np.zeros_like(binary, dtype=bool)
    regions: list[dict[str, Any]] = []

    for row in range(h):
        for col in range(w):
            if binary[row, col] == 0 or visited[row, col]:
                continue

            # BFS flood fill
            component_pixels: list[tuple[int, int]] = []
            stack = [(row, col)]
            visited[row, col] = True

            while stack:
                r, c = stack.pop()
                component_pixels.append((r, c))

                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and binary[nr, nc]:
                        visited[nr, nc] = True
                        stack.append((nr, nc))

            area = len(component_pixels)
            if area < min_area_px:
                continue

            rows = [p[0] for p in component_pixels]
            cols = [p[1] for p in component_pixels]
            min_r, max_r = min(rows), max(rows)
            min_c, max_c = min(cols), max(cols)

            # Region score = max heatmap value in this component
            region_score = float(max(heatmap[r, c] for r, c in component_pixels))

            regions.append({
                "x": int(min_c),
                "y": int(min_r),
                "w": int(max_c - min_c + 1),
                "h": int(max_r - min_r + 1),
                "score": round(region_score, 4),
                "area_px": int(area),
            })

    # Sort by score descending, limit count
    regions.sort(key=lambda r: r["score"], reverse=True)
    return regions[:max_regions]


# ----------------------------- training core -----------------------------


def train_memory_bank(
    img_paths: list[Path],
    embedder: PatchEmbedder,
    tile_w: int,
    tile_h: int,
    stride_w: int,
    stride_h: int,
    device: torch.device,
    max_patches_per_tile: int = 256,
    memory_size: int = 20000,
    batch_size: int = 16,
    num_workers: int = 2,
    pad_mode: str = "edge",
    progress_cb: Any = None,
) -> np.ndarray:
    """
    Build memory bank from OK images.

    Returns:
        memory_bank: (M, D) float32 numpy array
    """
    tfm, _, _ = build_transform()

    ds = TiledROIDataset(
        img_paths=img_paths,
        tile_w=tile_w,
        tile_h=tile_h,
        stride_w=stride_w,
        stride_h=stride_h,
        pad_mode=pad_mode,
        transform=tfm,
        return_info=False,
    )
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    sampled_feats = []
    total_batches = math.ceil(len(ds) / batch_size)
    batch_idx = 0

    embedder.eval()
    with torch.no_grad():
        for batch in dl:
            batch = batch.to(device, non_blocking=True)
            emb_map = embedder(batch)
            b, c, hf, wf = emb_map.shape
            emb_flat = emb_map.permute(0, 2, 3, 1).contiguous().view(b, -1, c)
            hw = emb_flat.shape[1]
            k = min(max_patches_per_tile, hw)

            for i in range(b):
                idx = torch.randperm(hw, device=emb_flat.device)[:k]
                feats = emb_flat[i, idx, :]
                sampled_feats.append(feats.detach().cpu().half())

            batch_idx += 1
            if progress_cb:
                pct = batch_idx / total_batches * 70  # 70% for extraction
                progress_cb(pct, f"Extracting patches: batch {batch_idx}/{total_batches}")

    feats_all = torch.cat(sampled_feats, dim=0)
    n, c_dim = feats_all.shape

    if progress_cb:
        progress_cb(75, f"Sampled {n} patches, dim={c_dim}. Subsampling memory bank...")

    m = min(memory_size, n)
    sel = torch.randperm(n)[:m]
    memory_bank = feats_all[sel].float().numpy().astype(np.float32)

    if progress_cb:
        progress_cb(80, f"Memory bank size: {m}, dim: {c_dim}")

    return memory_bank


def compute_threshold_from_ok(
    img_paths: list[Path],
    embedder: PatchEmbedder,
    knn: KNNSearcher,
    tile_w: int,
    tile_h: int,
    stride_w: int,
    stride_h: int,
    device: torch.device,
    pad_mode: str = "edge",
    batch_tiles: int = 8,
    ok_quantile: float = 0.999,
    thr_scale: float = 1.10,
    progress_cb: Any = None,
) -> float:
    """Compute auto threshold from OK images."""
    scores = []
    total = len(img_paths)

    for i, p in enumerate(img_paths):
        _, s, _ = infer_one_image(
            img_path=p,
            embedder=embedder,
            knn=knn,
            tile_w=tile_w,
            tile_h=tile_h,
            stride_w=stride_w,
            stride_h=stride_h,
            device=device,
            pad_mode=pad_mode,
            batch_tiles=batch_tiles,
        )
        scores.append(s)

        if progress_cb:
            pct = 80 + (i + 1) / total * 15  # 80-95%
            progress_cb(pct, f"Computing threshold: image {i + 1}/{total}")

    scores_arr = np.array(scores, dtype=np.float32)
    q = float(np.quantile(scores_arr, ok_quantile))
    threshold = q * thr_scale

    if progress_cb:
        progress_cb(95, f"OK q{ok_quantile}={q:.6f}, threshold={threshold:.6f}")

    return threshold

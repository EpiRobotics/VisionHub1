"""PatchCore Strip V1 core algorithm implementation.

Strip-based PatchCore for detecting misalignment in narrow groove structures
(e.g., motor iron core lamination grooves).

Key difference from patchcore_core.py (tiling):
- Tiling: splits both H and W, reassembles into one global heatmap, single score
- Strip: slices along the long axis into independent strips, each strip gets its
  own score, any-strip-NG triggers overall NG. This prevents small defects from
  being diluted by a large normal area.

Reuses ResNetFeat, PatchEmbedder, KNNSearcher, build_transform from patchcore_core.
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from app.plugins.patchcore_core import (
    KNNSearcher,
    PatchEmbedder,
    build_transform,
)

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


def compute_strip_positions(length: int, strip_size: int, overlap: int) -> list[int]:
    """Compute strip start positions along the long axis.

    Ensures the last strip touches the end of the image, similar to tile positions.

    Args:
        length: Total length along the long axis (pixels).
        strip_size: Size of each strip (pixels).
        overlap: Overlap between adjacent strips (pixels).

    Returns:
        List of start positions for each strip.
    """
    if strip_size >= length:
        return [0]
    stride = strip_size - overlap
    if stride <= 0:
        stride = 1
    positions = list(range(0, length - strip_size + 1, stride))
    last = length - strip_size
    if positions[-1] != last:
        positions.append(last)
    return positions


def detect_long_axis(h: int, w: int) -> str:
    """Detect which axis is the long axis.

    Returns:
        "vertical" if height >= width, "horizontal" otherwise.
    """
    return "vertical" if h >= w else "horizontal"


def slice_image_into_strips(
    np_img: np.ndarray,
    strip_size: int,
    overlap: int,
    axis: str | None = None,
) -> list[tuple[np.ndarray, int, int, int, int]]:
    """Slice image into strips along the long axis.

    Args:
        np_img: (H, W, 3) numpy array.
        strip_size: Size of each strip along the slicing axis.
        overlap: Overlap between adjacent strips.
        axis: "vertical" or "horizontal". Auto-detected if None.

    Returns:
        List of (strip_img, x0, y0, x1, y1) tuples where coordinates
        are in the original image space.
    """
    h, w = np_img.shape[:2]
    if axis is None:
        axis = detect_long_axis(h, w)

    strips: list[tuple[np.ndarray, int, int, int, int]] = []

    if axis == "vertical":
        # Slice along height
        positions = compute_strip_positions(h, strip_size, overlap)
        for y in positions:
            y_end = min(y + strip_size, h)
            strip = np_img[y:y_end, :, :]
            strips.append((strip, 0, y, w, y_end))
    else:
        # Slice along width
        positions = compute_strip_positions(w, strip_size, overlap)
        for x in positions:
            x_end = min(x + strip_size, w)
            strip = np_img[:, x:x_end, :]
            strips.append((strip, x, 0, x_end, h))

    return strips


# ----------------------------- strip dataset for training -----------------------------


class StripDataset(Dataset):
    """Dataset that slices images into strips for PatchCore training.

    Each strip is resized to (tile_h, tile_w) before feature extraction.
    """

    def __init__(
        self,
        img_paths: list[Path],
        strip_size: int,
        strip_overlap: int,
        tile_w: int,
        tile_h: int,
        axis: str | None = None,
        transform: Any = None,
    ):
        self.img_paths = img_paths
        self.strip_size = strip_size
        self.strip_overlap = strip_overlap
        self.tile_w = tile_w
        self.tile_h = tile_h
        self.axis = axis
        self.transform = transform

        # Precompute all strip indices
        self.strips: list[tuple[int, int, int, int, int, int]] = []
        for i, p in enumerate(self.img_paths):
            with Image.open(p) as im:
                w, h = im.size
            det_axis = axis if axis else detect_long_axis(h, w)
            if det_axis == "vertical":
                positions = compute_strip_positions(h, strip_size, strip_overlap)
                for si, y in enumerate(positions):
                    y_end = min(y + strip_size, h)
                    self.strips.append((i, si, 0, y, w, y_end))
            else:
                positions = compute_strip_positions(w, strip_size, strip_overlap)
                for si, x in enumerate(positions):
                    x_end = min(x + strip_size, w)
                    self.strips.append((i, si, x, 0, x_end, h))

    def __len__(self) -> int:
        return len(self.strips)

    def __getitem__(self, idx: int) -> Any:
        img_idx, strip_idx, x0, y0, x1, y1 = self.strips[idx]
        p = self.img_paths[img_idx]
        with Image.open(p) as im:
            im = im.convert("RGB")
            np_img = np.array(im)

        # Crop strip
        strip = np_img[y0:y1, x0:x1, :]

        # Resize to tile size for feature extraction
        pil_strip = Image.fromarray(strip).resize(
            (self.tile_w, self.tile_h), Image.BILINEAR
        )

        if self.transform is not None:
            strip_t = self.transform(pil_strip)
        else:
            strip_t = transforms.ToTensor()(pil_strip)

        return strip_t


# ----------------------------- inference core -----------------------------


def infer_one_strip(
    strip_img: np.ndarray,
    embedder: PatchEmbedder,
    knn: KNNSearcher,
    tile_w: int,
    tile_h: int,
    device: torch.device,
    score_mode: str = "max",
    score_quantile: float = 0.999,
) -> tuple[np.ndarray, float]:
    """Run PatchCore inference on a single strip.

    The strip is resized to (tile_h, tile_w) for feature extraction,
    then the anomaly map is resized back to the original strip size.

    Args:
        strip_img: (H, W, 3) numpy array of the strip.
        embedder: PatchCore feature embedder.
        knn: kNN searcher with memory bank.
        tile_w: Width to resize strip to for inference.
        tile_h: Height to resize strip to for inference.
        device: torch device.
        score_mode: "max" or "quantile".
        score_quantile: Quantile value when score_mode="quantile".

    Returns:
        heatmap: (H, W) anomaly heatmap at original strip resolution.
        score: Scalar anomaly score for this strip.
    """
    orig_h, orig_w = strip_img.shape[:2]

    # Resize strip for inference
    pil_strip = Image.fromarray(strip_img).resize(
        (tile_w, tile_h), Image.BILINEAR
    )

    tfm, _, _ = build_transform()  # lightweight; cached internally
    inp = tfm(pil_strip).unsqueeze(0).to(device, non_blocking=True)

    embedder.eval()
    with torch.no_grad():
        emb_map = embedder(inp)
        b, c, hf, wf = emb_map.shape
        patches = emb_map.permute(0, 2, 3, 1).contiguous().view(-1, c)

        dists = knn.query_min_dist(patches)
        dists = dists.view(1, 1, hf, wf)

        # Upsample to tile size
        up = F.interpolate(dists, size=(tile_h, tile_w), mode="bilinear", align_corners=False)
        heatmap_tile = up.squeeze().detach().cpu().numpy()

    # Resize heatmap back to original strip size
    heatmap_pil = Image.fromarray(heatmap_tile).resize(
        (orig_w, orig_h), Image.BILINEAR
    )
    heatmap = np.array(heatmap_pil, dtype=np.float32)

    # Compute strip score
    if score_mode == "max":
        score = float(np.max(heatmap))
    else:
        score = float(np.quantile(heatmap, score_quantile))

    return heatmap, score


def infer_strips(
    img_path: Path,
    embedder: PatchEmbedder,
    knn: KNNSearcher,
    tile_w: int,
    tile_h: int,
    strip_size: int,
    strip_overlap: int,
    device: torch.device,
    score_mode: str = "max",
    score_quantile: float = 0.999,
    axis: str | None = None,
) -> tuple[np.ndarray, float, list[dict[str, Any]], tuple[int, int]]:
    """Run strip-based PatchCore inference on a single image.

    Slices the image into strips along the long axis, runs PatchCore
    independently on each strip, and aggregates results.

    Args:
        img_path: Path to the input image.
        embedder: PatchCore feature embedder.
        knn: kNN searcher with memory bank.
        tile_w: Width to resize each strip to for inference.
        tile_h: Height to resize each strip to for inference.
        strip_size: Size of each strip along the long axis.
        strip_overlap: Overlap between adjacent strips.
        device: torch device.
        score_mode: "max" or "quantile".
        score_quantile: Quantile value when score_mode="quantile".
        axis: "vertical" or "horizontal". Auto-detected if None.

    Returns:
        full_heatmap: (H, W) anomaly heatmap (max-stitched from strips).
        overall_score: Max score across all strips.
        strip_results: Per-strip result dicts with score, bbox, pred.
        (H, W): Original image size.
    """
    with Image.open(img_path) as im:
        im = im.convert("RGB")
        np_img = np.array(im)
    h0, w0 = np_img.shape[:2]

    strips = slice_image_into_strips(np_img, strip_size, strip_overlap, axis)

    full_heatmap = np.zeros((h0, w0), dtype=np.float32)
    strip_results: list[dict[str, Any]] = []
    overall_score = 0.0

    for i, (strip_img, x0, y0, x1, y1) in enumerate(strips):
        heatmap, score = infer_one_strip(
            strip_img=strip_img,
            embedder=embedder,
            knn=knn,
            tile_w=tile_w,
            tile_h=tile_h,
            device=device,
            score_mode=score_mode,
            score_quantile=score_quantile,
        )

        # Stitch heatmap (max merge for overlapping regions)
        region = full_heatmap[y0:y1, x0:x1]
        hm_h, hm_w = heatmap.shape[:2]
        reg_h, reg_w = region.shape[:2]
        # Handle potential size mismatch from rounding
        min_h = min(hm_h, reg_h)
        min_w = min(hm_w, reg_w)
        np.maximum(region[:min_h, :min_w], heatmap[:min_h, :min_w], out=region[:min_h, :min_w])
        full_heatmap[y0:y0 + min_h, x0:x0 + min_w] = region[:min_h, :min_w]

        strip_results.append({
            "strip_idx": i,
            "x": x0,
            "y": y0,
            "w": x1 - x0,
            "h": y1 - y0,
            "score": round(float(score), 6),
        })

        if score > overall_score:
            overall_score = score

    return full_heatmap, overall_score, strip_results, (h0, w0)


# ----------------------------- overlay with strip markers -----------------------------


def save_strip_overlay_cv2(
    out_path: Path,
    img_path: Path,
    heatmap: np.ndarray,
    strip_results: list[dict[str, Any]],
    vmin: float = 0.0,
    vmax: float = 1.0,
    alpha: float = 0.4,
    score: float | None = None,
    threshold: float | None = None,
    pred: str | None = None,
) -> None:
    """Save overlay with heatmap and per-strip NG markers.

    Draws strip boundaries and marks NG strips with red rectangles.
    """
    import cv2

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(img_path) as im:
        im = im.convert("RGB")
        np_img = np.array(im)

    # Normalize heatmap to 0-255
    hm_clipped = np.clip(heatmap, vmin, vmax)
    hm_norm = ((hm_clipped - vmin) / (vmax - vmin + 1e-12) * 255).astype(np.uint8)

    # Apply JET colormap
    hm_color_bgr = cv2.applyColorMap(hm_norm, cv2.COLORMAP_JET)
    hm_color_rgb = cv2.cvtColor(hm_color_bgr, cv2.COLOR_BGR2RGB)

    # Blend
    blended = (
        np_img.astype(np.float32) * (1 - alpha)
        + hm_color_rgb.astype(np.float32) * alpha
    ).astype(np.uint8)

    blended_bgr = cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)
    img_h, img_w = blended_bgr.shape[:2]

    # Draw strip boundaries and NG markers
    if threshold is not None:
        for sr in strip_results:
            x, y, w, h = sr["x"], sr["y"], sr["w"], sr["h"]
            s = sr["score"]
            is_ng = s >= threshold

            # Draw strip boundary
            color = (0, 0, 255) if is_ng else (0, 180, 0)  # Red for NG, green for OK
            thickness = 2 if is_ng else 1
            cv2.rectangle(blended_bgr, (x, y), (x + w - 1, y + h - 1), color, thickness)

            # Draw score label for NG strips
            if is_ng:
                label = f"NG {s:.4f}"
                font_scale = max(0.35, min(img_w, img_h) / 1500.0)
                font_thickness = max(1, int(min(img_w, img_h) / 800))
                cv2.putText(
                    blended_bgr, label, (x + 2, y + int(20 * font_scale) + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255),
                    font_thickness, cv2.LINE_AA,
                )

    # Draw overall OK/NG label and score
    if pred is not None and score is not None and threshold is not None:
        scale_ref = min(img_w, img_h)
        font_scale_label = max(0.8, scale_ref / 600.0)
        font_scale_score = max(0.5, scale_ref / 900.0)
        thickness_label = max(2, int(scale_ref / 300))
        thickness_score = max(1, int(scale_ref / 500))

        color = (0, 200, 0) if pred == "OK" else (0, 0, 255)

        cv2.putText(
            blended_bgr, pred, (10, int(40 * font_scale_label)),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale_label, color,
            thickness_label, cv2.LINE_AA,
        )

        # Count NG strips
        ng_count = sum(1 for sr in strip_results if sr["score"] >= threshold)
        score_text = f"Score: {score:.4f} / Thr: {threshold:.4f} | NG strips: {ng_count}/{len(strip_results)}"
        y_offset = int(40 * font_scale_label) + int(35 * font_scale_score)
        cv2.putText(
            blended_bgr, score_text, (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale_score, (255, 255, 255),
            thickness_score, cv2.LINE_AA,
        )

    # Save
    suffix = out_path.suffix.lower()
    if suffix in (".jpg", ".jpeg"):
        cv2.imwrite(str(out_path), blended_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    else:
        cv2.imwrite(str(out_path), blended_bgr)


# ----------------------------- training core -----------------------------


def train_strip_memory_bank(
    img_paths: list[Path],
    embedder: PatchEmbedder,
    strip_size: int,
    strip_overlap: int,
    tile_w: int,
    tile_h: int,
    device: torch.device,
    axis: str | None = None,
    max_patches_per_strip: int = 256,
    memory_size: int = 20000,
    batch_size: int = 16,
    num_workers: int = 2,
    progress_cb: Any = None,
) -> np.ndarray:
    """Build memory bank from OK image strips.

    Slices OK images into strips, extracts PatchCore embeddings from each
    strip, and builds a memory bank.

    Returns:
        memory_bank: (M, D) float32 numpy array.
    """
    tfm, _, _ = build_transform()

    ds = StripDataset(
        img_paths=img_paths,
        strip_size=strip_size,
        strip_overlap=strip_overlap,
        tile_w=tile_w,
        tile_h=tile_h,
        axis=axis,
        transform=tfm,
    )

    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    sampled_feats: list[torch.Tensor] = []
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
            k = min(max_patches_per_strip, hw)

            for i in range(b):
                idx = torch.randperm(hw, device=emb_flat.device)[:k]
                feats = emb_flat[i, idx, :]
                sampled_feats.append(feats.detach().cpu().half())

            batch_idx += 1
            if progress_cb:
                pct = batch_idx / total_batches * 70
                progress_cb(pct, f"Extracting strip patches: batch {batch_idx}/{total_batches}")

    feats_all = torch.cat(sampled_feats, dim=0)
    n, c_dim = feats_all.shape

    if progress_cb:
        progress_cb(75, f"Sampled {n} patches from {len(ds)} strips, dim={c_dim}. Subsampling...")

    m = min(memory_size, n)
    sel = torch.randperm(n)[:m]
    memory_bank = feats_all[sel].float().numpy().astype(np.float32)

    if progress_cb:
        progress_cb(80, f"Memory bank size: {m}, dim: {c_dim}")

    return memory_bank


def compute_strip_threshold_from_ok(
    img_paths: list[Path],
    embedder: PatchEmbedder,
    knn: KNNSearcher,
    tile_w: int,
    tile_h: int,
    strip_size: int,
    strip_overlap: int,
    device: torch.device,
    score_mode: str = "max",
    score_quantile: float = 0.999,
    axis: str | None = None,
    ok_quantile: float = 0.999,
    thr_scale: float = 1.10,
    progress_cb: Any = None,
) -> float:
    """Compute auto threshold from OK images using strip-based scoring.

    For each OK image, computes the max strip score. The threshold is set
    based on the distribution of these max strip scores.
    """
    max_strip_scores: list[float] = []
    total = len(img_paths)

    for i, p in enumerate(img_paths):
        _, overall_score, _, _ = infer_strips(
            img_path=p,
            embedder=embedder,
            knn=knn,
            tile_w=tile_w,
            tile_h=tile_h,
            strip_size=strip_size,
            strip_overlap=strip_overlap,
            device=device,
            score_mode=score_mode,
            score_quantile=score_quantile,
            axis=axis,
        )
        max_strip_scores.append(overall_score)

        if progress_cb:
            pct = 80 + (i + 1) / total * 15
            progress_cb(pct, f"Computing strip threshold: image {i + 1}/{total}")

    scores_arr = np.array(max_strip_scores, dtype=np.float32)
    q = float(np.quantile(scores_arr, ok_quantile))
    threshold = q * thr_scale

    if progress_cb:
        progress_cb(
            95,
            f"OK max-strip-score q{ok_quantile}={q:.6f}, threshold={threshold:.6f}",
        )

    return threshold

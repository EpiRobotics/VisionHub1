"""Siamese Compare V1 core algorithm implementation.

Adjacent-tooth contrastive comparison for detecting alignment deviations
in periodic structures (e.g., motor iron core lamination grooves).

Key idea:
- Slice image into tooth-sized strips along the long axis
- Form adjacent pairs: (strip_i, strip_{i+1})
- Extract feature embeddings from each strip via a shared backbone
- Compute distance between adjacent pair embeddings
- High distance = misalignment between those two teeth
- Training: only OK samples needed; learn the "normal" distance distribution
  and set threshold from it

Reuses slicing utilities from patchcore_strip_core and backbone components
from patchcore_core.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from app.plugins.patchcore_core import ResNetFeat, build_transform
from app.plugins.patchcore_strip_core import slice_image_into_strips

logger = logging.getLogger(__name__)

# ----------------------------- constants / utils -----------------------------

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------------------- embedding head -----------------------------


class EmbeddingHead(nn.Module):
    """Global-average-pool + FC projection head on top of backbone features.

    Takes the multi-layer feature dict from ResNetFeat, concatenates the
    selected layers (upsampled to the same spatial size), applies global
    average pooling, then projects to a compact embedding vector.
    """

    def __init__(self, backbone: ResNetFeat, layers: list[str], embed_dim: int = 256, concat_dim: int = 0):
        super().__init__()
        self.backbone = backbone
        self.layers = layers
        self.embed_dim = embed_dim

        # Projection: eagerly init if concat_dim is known (loading),
        # otherwise lazily init on first forward pass (training).
        self._proj: nn.Linear | None = None
        self._concat_dim: int = concat_dim
        if concat_dim > 0:
            self._proj = nn.Linear(concat_dim, embed_dim, bias=False)

    def _lazy_init_proj(self, concat_dim: int, device: torch.device) -> None:
        self._concat_dim = concat_dim
        self._proj = nn.Linear(concat_dim, self.embed_dim, bias=False).to(device)
        # Xavier init for stable distances
        nn.init.xavier_uniform_(self._proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: image tensor → L2-normalised embedding vector.

        Args:
            x: (B, 3, H, W) input images.

        Returns:
            (B, embed_dim) L2-normalised embedding vectors.
        """
        feats = self.backbone(x)
        base = feats[self.layers[0]]
        base_h, base_w = base.shape[-2:]

        ups = []
        for k in self.layers:
            f = feats[k]
            if f.shape[-2:] != (base_h, base_w):
                f = F.interpolate(f, size=(base_h, base_w), mode="bilinear", align_corners=False)
            ups.append(f)
        cat = torch.cat(ups, dim=1)  # (B, C_cat, H_feat, W_feat)

        # Lazy init projection
        if self._proj is None:
            self._lazy_init_proj(cat.shape[1], cat.device)

        # Global average pool → (B, C_cat)
        gap = cat.mean(dim=[2, 3])

        # Project → (B, embed_dim)
        assert self._proj is not None
        emb = self._proj(gap)

        # L2 normalise
        emb = F.normalize(emb, p=2, dim=1)
        return emb

    def forward_featuremap(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning spatial feature maps (before GAP).

        Args:
            x: (B, 3, H, W) input images.

        Returns:
            (B, C_cat, H_feat, W_feat) concatenated multi-layer feature maps.
        """
        feats = self.backbone(x)
        base = feats[self.layers[0]]
        base_h, base_w = base.shape[-2:]

        ups = []
        for k in self.layers:
            f = feats[k]
            if f.shape[-2:] != (base_h, base_w):
                f = F.interpolate(f, size=(base_h, base_w), mode="bilinear", align_corners=False)
            ups.append(f)
        cat = torch.cat(ups, dim=1)  # (B, C_cat, H_feat, W_feat)

        # Lazy init projection (needed for consistent state)
        if self._proj is None:
            self._lazy_init_proj(cat.shape[1], cat.device)

        return cat


# ----------------------------- pair distance -----------------------------


def compute_pair_distances(
    embeddings: list[torch.Tensor],
    metric: str = "l2",
) -> list[float]:
    """Compute distances between consecutive embeddings.

    Args:
        embeddings: List of (embed_dim,) tensors, one per strip.
        metric: "l2" or "cosine".

    Returns:
        List of distances, length = len(embeddings) - 1.
    """
    if len(embeddings) < 2:
        return []
    distances: list[float] = []
    for i in range(len(embeddings) - 1):
        e1 = embeddings[i]
        e2 = embeddings[i + 1]
        if metric == "cosine":
            # cosine distance = 1 - cosine_similarity
            cos_sim = F.cosine_similarity(e1.unsqueeze(0), e2.unsqueeze(0))
            d = 1.0 - cos_sim.item()
        else:
            d = float(torch.dist(e1, e2, p=2).item())
        distances.append(d)
    return distances


# ----------------------------- inference core -----------------------------


def embed_strips(
    np_img: np.ndarray,
    model: EmbeddingHead,
    strip_size: int,
    strip_overlap: int,
    tile_w: int,
    tile_h: int,
    device: torch.device,
    axis: str | None = None,
) -> tuple[list[torch.Tensor], list[tuple[int, int, int, int]]]:
    """Slice image and compute embeddings for each strip.

    Args:
        np_img: (H, W, 3) numpy array.
        model: EmbeddingHead model.
        strip_size: Size of each strip along the long axis.
        strip_overlap: Overlap between adjacent strips.
        tile_w: Width to resize each strip to for the backbone.
        tile_h: Height to resize each strip to for the backbone.
        device: torch device.
        axis: "vertical" or "horizontal". Auto-detected if None.

    Returns:
        embeddings: List of (embed_dim,) tensors.
        bboxes: List of (x0, y0, x1, y1) for each strip.
    """
    strips = slice_image_into_strips(np_img, strip_size, strip_overlap, axis)
    tfm, _, _ = build_transform()

    embeddings: list[torch.Tensor] = []
    bboxes: list[tuple[int, int, int, int]] = []

    model.eval()
    with torch.no_grad():
        for strip_img, x0, y0, x1, y1 in strips:
            pil_strip = Image.fromarray(strip_img).resize(
                (tile_w, tile_h), Image.BILINEAR
            )
            inp = tfm(pil_strip).unsqueeze(0).to(device, non_blocking=True)
            emb = model(inp).squeeze(0)  # (embed_dim,)
            embeddings.append(emb)
            bboxes.append((x0, y0, x1, y1))

    return embeddings, bboxes


def embed_strips_with_featuremaps(
    np_img: np.ndarray,
    model: EmbeddingHead,
    strip_size: int,
    strip_overlap: int,
    tile_w: int,
    tile_h: int,
    device: torch.device,
    axis: str | None = None,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[tuple[int, int, int, int]]]:
    """Slice image and compute both embeddings and spatial feature maps.

    Returns:
        embeddings: List of (embed_dim,) tensors.
        featuremaps: List of (C_cat, H_feat, W_feat) tensors.
        bboxes: List of (x0, y0, x1, y1) for each strip.
    """
    strips = slice_image_into_strips(np_img, strip_size, strip_overlap, axis)
    tfm, _, _ = build_transform()

    embeddings: list[torch.Tensor] = []
    featuremaps: list[torch.Tensor] = []
    bboxes: list[tuple[int, int, int, int]] = []

    model.eval()
    with torch.no_grad():
        for strip_img, x0, y0, x1, y1 in strips:
            pil_strip = Image.fromarray(strip_img).resize(
                (tile_w, tile_h), Image.BILINEAR
            )
            inp = tfm(pil_strip).unsqueeze(0).to(device, non_blocking=True)
            emb = model(inp).squeeze(0)  # (embed_dim,)
            fmap = model.forward_featuremap(inp).squeeze(0)  # (C, Hf, Wf)
            embeddings.append(emb)
            featuremaps.append(fmap)
            bboxes.append((x0, y0, x1, y1))

    return embeddings, featuremaps, bboxes


def compute_diff_heatmap(
    featuremaps: list[torch.Tensor],
    bboxes: list[tuple[int, int, int, int]],
    img_h: int,
    img_w: int,
) -> np.ndarray:
    """Compute a spatial diff heatmap from adjacent-pair feature maps.

    For each adjacent pair, computes per-spatial-location L2 distance
    between their feature maps, then stitches into a full-image heatmap.

    Args:
        featuremaps: List of (C, Hf, Wf) tensors, one per strip.
        bboxes: List of (x0, y0, x1, y1) for each strip.
        img_h: Original image height.
        img_w: Original image width.

    Returns:
        heatmap: (img_h, img_w) float32 array, 0-1 normalised.
    """
    heatmap = np.zeros((img_h, img_w), dtype=np.float32)
    count = np.zeros((img_h, img_w), dtype=np.float32)

    for i in range(len(featuremaps) - 1):
        f1 = featuremaps[i]   # (C, Hf, Wf)
        f2 = featuremaps[i + 1]
        # Per-spatial-location L2 distance
        diff = torch.sqrt(((f1 - f2) ** 2).sum(dim=0) + 1e-8)  # (Hf, Wf)
        diff_np = diff.cpu().numpy()

        # Map back to image coordinates – cover the overlap region
        # between strip_i and strip_{i+1}
        x0_a, y0_a, x1_a, y1_a = bboxes[i]
        x0_b, y0_b, x1_b, y1_b = bboxes[i + 1]

        # Region covering both strips
        rx0 = min(x0_a, x0_b)
        ry0 = min(y0_a, y0_b)
        rx1 = max(x1_a, x1_b)
        ry1 = max(y1_a, y1_b)
        rh = ry1 - ry0
        rw = rx1 - rx0

        if rh <= 0 or rw <= 0:
            continue

        # Resize diff map to the region size
        from PIL import Image as _PILImage
        diff_resized = np.array(
            _PILImage.fromarray(diff_np).resize((rw, rh), _PILImage.BILINEAR)
        )
        # Accumulate into heatmap (handles overlapping regions)
        heatmap[ry0:ry1, rx0:rx1] += diff_resized
        count[ry0:ry1, rx0:rx1] += 1.0

    # Average overlapping regions
    mask = count > 0
    heatmap[mask] /= count[mask]

    # Normalise to 0-1
    hmax = heatmap.max()
    if hmax > 0:
        heatmap /= hmax

    return heatmap


def infer_adjacent_pairs(
    img_path: Path,
    model: EmbeddingHead,
    strip_size: int,
    strip_overlap: int,
    tile_w: int,
    tile_h: int,
    device: torch.device,
    metric: str = "l2",
    axis: str | None = None,
) -> tuple[float, list[dict[str, Any]], tuple[int, int], np.ndarray]:
    """Run adjacent-pair comparison inference on a single image.

    Slices the image into strips, embeds each strip, computes distance
    between adjacent pairs, and returns per-pair results plus a spatial
    diff heatmap.

    Args:
        img_path: Path to the input image.
        model: EmbeddingHead model.
        strip_size: Size of each strip along the long axis.
        strip_overlap: Overlap between adjacent strips.
        tile_w: Width to resize each strip to.
        tile_h: Height to resize each strip to.
        device: torch device.
        metric: "l2" or "cosine".
        axis: "vertical" or "horizontal". Auto-detected if None.

    Returns:
        max_distance: Maximum distance across all adjacent pairs.
        pair_results: Per-pair result dicts.
        (H, W): Original image size.
        diff_heatmap: (H, W) float32 array, 0-1 normalised spatial diff map.
    """
    with Image.open(img_path) as im:
        im = im.convert("RGB")
        np_img = np.array(im)
    h0, w0 = np_img.shape[:2]

    embeddings, featuremaps, bboxes = embed_strips_with_featuremaps(
        np_img=np_img,
        model=model,
        strip_size=strip_size,
        strip_overlap=strip_overlap,
        tile_w=tile_w,
        tile_h=tile_h,
        device=device,
        axis=axis,
    )

    distances = compute_pair_distances(embeddings, metric=metric)
    diff_heatmap = compute_diff_heatmap(featuremaps, bboxes, h0, w0)

    pair_results: list[dict[str, Any]] = []
    for i, dist in enumerate(distances):
        x0_a, y0_a, x1_a, y1_a = bboxes[i]
        x0_b, y0_b, x1_b, y1_b = bboxes[i + 1]
        pair_results.append({
            "pair_idx": i,
            "strip_a": i,
            "strip_b": i + 1,
            "distance": round(float(dist), 6),
            "bbox_a": {"x": x0_a, "y": y0_a, "w": x1_a - x0_a, "h": y1_a - y0_a},
            "bbox_b": {"x": x0_b, "y": y0_b, "w": x1_b - x0_b, "h": y1_b - y0_b},
        })

    max_distance = max(distances) if distances else 0.0
    return max_distance, pair_results, (h0, w0), diff_heatmap


# ----------------------------- overlay -----------------------------


def save_siamese_overlay_cv2(
    out_path: Path,
    img_path: Path,
    pair_results: list[dict[str, Any]],
    threshold: float,
    max_distance: float,
    pred: str,
    diff_heatmap: np.ndarray | None = None,
    alpha: float = 0.3,
    heatmap_alpha: float = 0.45,
) -> None:
    """Save overlay with diff heatmap, per-pair distance labels, and NG markers.

    When *diff_heatmap* is provided, a JET colourmap overlay is blended onto
    the original image so the user can see *where* adjacent strips differ.
    Every pair also gets a distance label (green for OK, red for NG).
    """
    import cv2

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(img_path) as im:
        im = im.convert("RGB")
        np_img = np.array(im)

    img_h, img_w = np_img.shape[:2]
    canvas_bgr = cv2.cvtColor(np_img.copy(), cv2.COLOR_RGB2BGR)

    # --- Blend diff heatmap ---
    if diff_heatmap is not None and diff_heatmap.max() > 0:
        hmap_u8 = (np.clip(diff_heatmap, 0, 1) * 255).astype(np.uint8)
        hmap_color = cv2.applyColorMap(hmap_u8, cv2.COLORMAP_JET)
        # Only blend where heatmap has signal (avoid colouring zero-regions)
        mask = diff_heatmap > 0.01
        blended = canvas_bgr.copy()
        blended[mask] = cv2.addWeighted(
            canvas_bgr, 1 - heatmap_alpha, hmap_color, heatmap_alpha, 0
        )[mask]
        canvas_bgr = blended

    # --- Per-pair annotations ---
    font_scale = max(0.3, min(img_w, img_h) / 1800.0)
    font_thickness = max(1, int(min(img_w, img_h) / 900))

    for pr in pair_results:
        dist = pr["distance"]
        is_ng = dist >= threshold
        ba = pr["bbox_a"]
        bb = pr["bbox_b"]

        # Draw strip boundaries
        box_color = (0, 0, 255) if is_ng else (0, 180, 0)
        box_thick = 2 if is_ng else 1

        cv2.rectangle(
            canvas_bgr,
            (ba["x"], ba["y"]),
            (ba["x"] + ba["w"] - 1, ba["y"] + ba["h"] - 1),
            box_color, box_thick,
        )
        cv2.rectangle(
            canvas_bgr,
            (bb["x"], bb["y"]),
            (bb["x"] + bb["w"] - 1, bb["y"] + bb["h"] - 1),
            box_color, box_thick,
        )

        # Distance label for EVERY pair (green OK / red NG)
        label = f"d={dist:.4f}"
        if is_ng:
            label = f"NG {label}"
        label_color = (0, 0, 255) if is_ng else (0, 220, 0)

        # Place label at the boundary between strip A and B
        if ba["y"] != bb["y"]:
            # Vertical slicing: boundary is horizontal
            lx = ba["x"] + 2
            ly = max(min(ba["y"] + ba["h"], bb["y"]) - 3, int(16 * font_scale) + 2)
        else:
            # Horizontal slicing: boundary is vertical
            lx = min(ba["x"] + ba["w"], bb["x"]) + 2
            ly = ba["y"] + int(ba["h"] / 2)

        cv2.putText(
            canvas_bgr, label, (lx, ly),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, label_color,
            font_thickness, cv2.LINE_AA,
        )

        # Semi-transparent red fill for NG boundary zones
        if is_ng:
            bx0 = min(ba["x"], bb["x"])
            by0 = min(ba["y"] + ba["h"], bb["y"])
            bx1 = max(ba["x"] + ba["w"], bb["x"] + bb["w"])
            by1 = max(ba["y"] + ba["h"], bb["y"])
            if ba["y"] == bb["y"]:
                bx0 = min(ba["x"] + ba["w"], bb["x"])
                by0 = min(ba["y"], bb["y"])
                bx1 = max(ba["x"] + ba["w"], bb["x"])
                by1 = max(ba["y"] + ba["h"], bb["y"] + bb["h"])
            overlay = canvas_bgr.copy()
            cv2.rectangle(overlay, (bx0, by0), (bx1, by1), (0, 0, 255), cv2.FILLED)
            cv2.addWeighted(overlay, alpha, canvas_bgr, 1 - alpha, 0, canvas_bgr)

    # --- Global header ---
    scale_ref = min(img_w, img_h)
    fs_label = max(0.8, scale_ref / 600.0)
    th_label = max(2, int(scale_ref / 300))
    fs_score = max(0.5, scale_ref / 900.0)
    th_score = max(1, int(scale_ref / 500))

    label_color = (0, 200, 0) if pred == "OK" else (0, 0, 255)
    cv2.putText(
        canvas_bgr, pred, (10, int(40 * fs_label)),
        cv2.FONT_HERSHEY_SIMPLEX, fs_label, label_color,
        th_label, cv2.LINE_AA,
    )

    ng_count = sum(1 for pr in pair_results if pr["distance"] >= threshold)
    score_text = (
        f"MaxDist: {max_distance:.4f} / Thr: {threshold:.4f} | "
        f"NG pairs: {ng_count}/{len(pair_results)}"
    )
    y_offset = int(40 * fs_label) + int(35 * fs_score)
    cv2.putText(
        canvas_bgr, score_text, (10, y_offset),
        cv2.FONT_HERSHEY_SIMPLEX, fs_score, (255, 255, 255),
        th_score, cv2.LINE_AA,
    )

    suffix = out_path.suffix.lower()
    if suffix in (".jpg", ".jpeg"):
        cv2.imwrite(str(out_path), canvas_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    else:
        cv2.imwrite(str(out_path), canvas_bgr)


# ----------------------------- training core -----------------------------


def train_siamese_model(
    img_paths: list[Path],
    backbone_name: str,
    layers: list[str],
    embed_dim: int,
    strip_size: int,
    strip_overlap: int,
    tile_w: int,
    tile_h: int,
    device: torch.device,
    axis: str | None = None,
    fine_tune_epochs: int = 0,
    fine_tune_lr: float = 1e-4,
    fine_tune_margin: float = 0.5,
    progress_cb: Any = None,
) -> tuple[EmbeddingHead, dict[str, Any]]:
    """Build and optionally fine-tune the Siamese embedding model.

    Phase 1 (always): Build pretrained backbone + projection head.
    Phase 2 (if fine_tune_epochs > 0): Fine-tune with contrastive loss on
    OK adjacent pairs to tighten the embedding space.

    Args:
        img_paths: List of OK image paths.
        backbone_name: Backbone name (resnet18 / resnet50).
        layers: Backbone layers to extract features from.
        embed_dim: Embedding dimension.
        strip_size: Strip size for slicing.
        strip_overlap: Strip overlap.
        tile_w: Tile width for resizing strips.
        tile_h: Tile height for resizing strips.
        device: torch device.
        axis: Slicing axis. None for auto-detect.
        fine_tune_epochs: Number of contrastive fine-tuning epochs (0 = skip).
        fine_tune_lr: Learning rate for fine-tuning.
        fine_tune_margin: Margin for contrastive loss.
        progress_cb: Progress callback.

    Returns:
        model: Trained EmbeddingHead.
        stats: Training statistics dict.
    """
    # Build backbone + head
    backbone = ResNetFeat(backbone_name, pretrained=True, layers=layers)
    model = EmbeddingHead(backbone, layers=layers, embed_dim=embed_dim).to(device)
    model.eval()

    if progress_cb:
        progress_cb(15.0, f"Built backbone: {backbone_name}, layers: {layers}, embed_dim: {embed_dim}")

    # Warm up projection by running one forward pass
    tfm, _, _ = build_transform()
    dummy = torch.randn(1, 3, tile_h, tile_w, device=device)
    with torch.no_grad():
        model(dummy)

    if progress_cb:
        progress_cb(20.0, "Backbone initialised, extracting OK pair distances...")

    # Phase 2: optional contrastive fine-tuning
    if fine_tune_epochs > 0:
        model = _fine_tune_contrastive(
            model=model,
            img_paths=img_paths,
            strip_size=strip_size,
            strip_overlap=strip_overlap,
            tile_w=tile_w,
            tile_h=tile_h,
            device=device,
            axis=axis,
            epochs=fine_tune_epochs,
            lr=fine_tune_lr,
            margin=fine_tune_margin,
            progress_cb=progress_cb,
        )

    stats = {
        "backbone": backbone_name,
        "layers": layers,
        "embed_dim": embed_dim,
        "fine_tune_epochs": fine_tune_epochs,
    }

    return model, stats


def _fine_tune_contrastive(
    model: EmbeddingHead,
    img_paths: list[Path],
    strip_size: int,
    strip_overlap: int,
    tile_w: int,
    tile_h: int,
    device: torch.device,
    axis: str | None,
    epochs: int,
    lr: float,
    margin: float,
    progress_cb: Any = None,
) -> EmbeddingHead:
    """Fine-tune with contrastive loss on OK adjacent pairs.

    For OK pairs (label=1, i.e. similar), the loss is: d^2
    This pushes adjacent OK strips closer in embedding space, making
    misaligned pairs (which would be farther) easier to detect.
    """
    tfm, _, _ = build_transform()

    # Collect all adjacent pairs from OK images
    pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
    for img_path in img_paths:
        with Image.open(img_path) as im:
            im = im.convert("RGB")
            np_img = np.array(im)

        strips = slice_image_into_strips(np_img, strip_size, strip_overlap, axis)
        for i in range(len(strips) - 1):
            strip_a = strips[i][0]
            strip_b = strips[i + 1][0]
            pil_a = Image.fromarray(strip_a).resize((tile_w, tile_h), Image.BILINEAR)
            pil_b = Image.fromarray(strip_b).resize((tile_w, tile_h), Image.BILINEAR)
            pairs.append((tfm(pil_a), tfm(pil_b)))

    if not pairs:
        logger.warning("No adjacent pairs found for fine-tuning, skipping.")
        return model

    if progress_cb:
        progress_cb(25.0, f"Fine-tuning on {len(pairs)} OK adjacent pairs for {epochs} epochs...")

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        random.shuffle(pairs)
        total_loss = 0.0
        for pa, pb in pairs:
            pa = pa.unsqueeze(0).to(device)
            pb = pb.unsqueeze(0).to(device)
            emb_a = model(pa)
            emb_b = model(pb)
            # Contrastive loss for positive pairs: minimize distance
            dist = F.pairwise_distance(emb_a, emb_b, p=2)
            loss = dist.pow(2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(len(pairs), 1)
        if progress_cb:
            pct = 25.0 + (epoch + 1) / epochs * 25.0  # 25% → 50%
            progress_cb(pct, f"Fine-tune epoch {epoch + 1}/{epochs}, avg_loss={avg_loss:.6f}")

    model.eval()
    return model


def compute_ok_pair_distances(
    img_paths: list[Path],
    model: EmbeddingHead,
    strip_size: int,
    strip_overlap: int,
    tile_w: int,
    tile_h: int,
    device: torch.device,
    metric: str = "l2",
    axis: str | None = None,
    progress_cb: Any = None,
) -> list[float]:
    """Compute all adjacent-pair distances from OK images.

    Returns:
        List of distances from all OK adjacent pairs.
    """
    all_distances: list[float] = []

    model.eval()
    for idx, img_path in enumerate(img_paths):
        with Image.open(img_path) as im:
            im = im.convert("RGB")
            np_img = np.array(im)

        embeddings, _ = embed_strips(
            np_img=np_img,
            model=model,
            strip_size=strip_size,
            strip_overlap=strip_overlap,
            tile_w=tile_w,
            tile_h=tile_h,
            device=device,
            axis=axis,
        )

        distances = compute_pair_distances(embeddings, metric=metric)
        all_distances.extend(distances)

        if progress_cb and len(img_paths) > 1:
            pct = 55.0 + (idx + 1) / len(img_paths) * 30.0  # 55% → 85%
            progress_cb(pct, f"OK pair distances: {idx + 1}/{len(img_paths)} images, {len(all_distances)} pairs")

    return all_distances


def compute_threshold_from_ok_distances(
    ok_distances: list[float],
    ok_quantile: float = 0.999,
    thr_scale: float = 1.2,
) -> float:
    """Compute threshold from OK pair distance distribution.

    threshold = quantile(ok_distances, ok_quantile) * thr_scale

    Args:
        ok_distances: List of distances from OK adjacent pairs.
        ok_quantile: Quantile of OK distance distribution.
        thr_scale: Scale factor applied to the quantile.

    Returns:
        Computed threshold.
    """
    if not ok_distances:
        return 0.5
    arr = np.array(ok_distances, dtype=np.float32)
    q = float(np.quantile(arr, ok_quantile))
    threshold = q * thr_scale
    logger.info(
        "OK pair distances: n=%d, mean=%.6f, std=%.6f, q%.3f=%.6f, threshold=%.6f",
        len(arr), float(np.mean(arr)), float(np.std(arr)), ok_quantile, q, threshold,
    )
    return threshold

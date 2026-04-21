"""Glyph PatchCore core algorithm module.

Per-character anomaly detection using PatchCore with per-class memory banks.
Adapted from label/glyph_watch_service.py and label/train_glyph_patchcore.py.

Key differences from the tiling PatchCore (patchcore_core.py):
- Detection unit: individual character glyphs (cropped via JSON annotations)
- Model format: one .joblib file per character class (sklearn NearestNeighbors)
- Inference input: image + JSON annotation (character positions)
- kNN: GPU-accelerated torch kNN or sklearn CPU fallback
- Output: per-character OK/NG decisions, overlay image
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Image extensions supported
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# Windows folder name mapping for special characters
SPECIAL_MAP = {
    "/": "slash",
    "\\": "backslash",
    ".": "dot",
    ":": "colon",
    "*": "asterisk",
    "?": "question",
    '"': "quote",
    "<": "lt",
    ">": "gt",
    "|": "pipe",
    " ": "space",
    "\t": "tab",
}
INVALID_WIN = set('<>:"/\\|?*')


def safe_folder_name(ch: str) -> str:
    """Map a character to a safe folder/file name."""
    if ch in SPECIAL_MAP:
        return SPECIAL_MAP[ch]
    if ch in INVALID_WIN:
        return f"u{ord(ch):04X}"
    if ch.endswith(".") or ch.endswith(" "):
        return f"u{ord(ch):04X}"
    return ch


def clip_box(
    x0: int, y0: int, x1: int, y1: int, w_img: int, h_img: int
) -> tuple[int, int, int, int]:
    """Clip bounding box to image dimensions."""
    x0 = max(0, min(int(x0), w_img))
    y0 = max(0, min(int(y0), h_img))
    x1 = max(0, min(int(x1), w_img))
    y1 = max(0, min(int(y1), h_img))
    if x1 <= x0:
        x1 = min(w_img, x0 + 1)
    if y1 <= y0:
        y1 = min(h_img, y0 + 1)
    return x0, y0, x1, y1


def imread_any_bgr(p: Path) -> np.ndarray | None:
    """Read image file with any encoding (handles unicode paths on Windows)."""
    data = np.fromfile(str(p), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imread_any_gray(p: Path) -> np.ndarray | None:
    """Read image as grayscale."""
    data = np.fromfile(str(p), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)


def imwrite_any(p: Path, bgr: np.ndarray) -> None:
    """Write image handling unicode paths."""
    p.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(p.suffix.lower() if p.suffix else ".png", bgr)
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    buf.tofile(str(p))


def list_images(directory: Path) -> list[Path]:
    """List image files in a directory."""
    return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS)


# ---------------------------------------------------------------------------
# ResNet feature extractor (layer2 only, shallow - good for glyph defects)
# ---------------------------------------------------------------------------

_torch_available = False
_torch = None
_nn = None
_models = None
_transforms = None


def _ensure_torch() -> None:
    """Lazy import torch to avoid import errors when not installed."""
    global _torch_available, _torch, _nn, _models, _transforms
    if _torch is not None:
        return
    try:
        import torch
        import torch.nn as nn_module
        from torchvision import models as models_module
        from torchvision import transforms as transforms_module

        _torch = torch
        _nn = nn_module
        _models = models_module
        _transforms = transforms_module
        _torch_available = True
    except ImportError as e:
        raise ImportError("torch/torchvision required for glyph_patchcore") from e


class ResNetFeatGlyph:
    """ResNet18 feature extractor for glyph defects.

    Supports two modes:
    - single_scale (default): layer2 only (128-dim, 1/8 resolution)
      Good for general character anomaly detection.
    - multi_scale: layer1 + layer2 concatenated (192-dim, 1/4 resolution)
      Provides 4x finer spatial detail, making small defects (missing corners,
      thin cracks) affect more patches proportionally. Recommended for
      detecting subtle per-character defects.
    """

    def __init__(self, multi_scale: bool = False) -> None:
        _ensure_torch()
        assert _torch is not None and _nn is not None and _models is not None
        m = _models.resnet18(weights=_models.ResNet18_Weights.IMAGENET1K_V1)
        self.stem = _nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)
        self.layer1 = m.layer1
        self.layer2 = m.layer2
        self.multi_scale = multi_scale
        # Build as nn.Module for .to() / .eval()
        self._net = _nn.Sequential(self.stem, self.layer1, self.layer2)

    def to(self, device: Any) -> "ResNetFeatGlyph":
        self._net = self._net.to(device)
        self.stem = self._net[0]
        self.layer1 = self._net[1]
        self.layer2 = self._net[2]
        return self

    def eval(self) -> "ResNetFeatGlyph":
        self._net.eval()
        return self

    def __call__(self, x: Any) -> Any:
        if self.multi_scale:
            assert _torch is not None
            h = self.stem(x)
            f1 = self.layer1(h)    # [B, 64, H/4, W/4]
            f2 = self.layer2(f1)   # [B, 128, H/8, W/8]
            # Upsample layer2 to match layer1 spatial resolution
            f2_up = _torch.nn.functional.interpolate(
                f2, size=f1.shape[2:], mode="bilinear", align_corners=False,
            )
            return _torch.cat([f1, f2_up], dim=1)  # [B, 192, H/4, W/4]
        return self._net(x)


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

_glyph_tf: Any = None


def _get_glyph_transform() -> Any:
    """Get the standard glyph preprocessing transform."""
    global _glyph_tf
    if _glyph_tf is None:
        _ensure_torch()
        assert _transforms is not None
        _glyph_tf = _transforms.Compose([
            _transforms.ToTensor(),
            _transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
    return _glyph_tf


def _apply_edge_preprocessing(gray: np.ndarray) -> np.ndarray:
    """Convert grayscale image to Sobel edge magnitude map.

    This transforms the image so the CNN sees *boundary/shape* information
    rather than *fill/intensity* information.  Key properties:

    - **Thickness-invariant**: A thick "A" and a thin "A" have the same
      edge topology (edges at stroke boundaries), so their CNN features
      are close → low anomaly score.
    - **Defect-sensitive**: A "2" missing a corner has *missing edges* in
      that region → very different edge pattern → high anomaly score.

    Pipeline: Gaussian blur → Sobel X/Y → magnitude → normalize to 0-255.
    """
    # Light blur to suppress noise while preserving edges
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    # Sobel gradients (float64 to avoid overflow)
    sx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(sx * sx + sy * sy)
    # Normalize to 0-255
    mag_max = mag.max()
    if mag_max > 0:
        mag = (mag / mag_max * 255.0)
    return mag.astype(np.uint8)


def preprocess_gray_to_tensor(
    gray: np.ndarray,
    size: int,
    use_clahe: bool = False,
    clahe_clip: float = 3.0,
    clahe_grid: int = 4,
    use_edge: bool = False,
) -> Any:
    """Preprocess a grayscale glyph crop to a [1,3,S,S] tensor.

    Args:
        gray: Input grayscale image crop.
        size: Target square size.
        use_clahe: Apply CLAHE (Contrast Limited Adaptive Histogram
            Equalization) before feature extraction.  CLAHE normalizes
            local contrast, reducing sensitivity to global stroke
            thickness / intensity variations while enhancing local
            structural defects.
        clahe_clip: CLAHE clip limit (higher = more contrast enhancement).
        clahe_grid: CLAHE tile grid size.
        use_edge: Convert to Sobel edge map before CNN.  This makes the
            model focus on character *boundaries/shape* rather than
            fill/intensity, dramatically improving separation between
            structural defects (missing corner) and cosmetic variations
            (stroke thickness).  Recommended for glyph defect detection.
    """
    _ensure_torch()
    g = cv2.resize(gray, (size, size), interpolation=cv2.INTER_CUBIC)
    if use_clahe:
        clahe = cv2.createCLAHE(
            clipLimit=clahe_clip, tileGridSize=(clahe_grid, clahe_grid),
        )
        g = clahe.apply(g)
    if use_edge:
        g = _apply_edge_preprocessing(g)
    rgb = np.stack([g, g, g], axis=-1).astype(np.uint8)
    return _get_glyph_transform()(rgb).unsqueeze(0)


def extract_patch_embeddings_batch(fm: Any) -> Any:
    """Extract patch embeddings from feature map batch.

    Args:
        fm: [B, C, H, W] feature map tensor

    Returns:
        [B, N, C] patch embedding tensor where N = H*W
    """
    B, C, H, W = fm.shape
    return fm.permute(0, 2, 3, 1).reshape(B, H * W, C)


# ---------------------------------------------------------------------------
# GPU kNN
# ---------------------------------------------------------------------------


def knn_mean_distance_torch(
    emb_t: Any,       # [M, C] on cuda/float16
    bank_t: Any,      # [B, C] on cuda/float16
    bank_norm: Any,    # [B] cuda/float32
    k: int,
    block: int,
) -> Any:
    """Compute mean kNN distance on GPU using blocked matrix multiplication.

    Avoids transferring data to CPU for large memory banks.
    """
    _ensure_torch()
    assert _torch is not None
    emb_norm = (_torch.float32 and (emb_t.to(_torch.float32) * emb_t.to(_torch.float32)).sum(dim=1))
    M = emb_t.shape[0]
    k = max(1, int(k))
    best = _torch.full((M, k), float("inf"), device=emb_t.device, dtype=_torch.float32)

    B = bank_t.shape[0]
    for s in range(0, B, block):
        e = min(B, s + block)
        bt = bank_t[s:e]
        bn = bank_norm[s:e]
        dot = emb_t.to(_torch.float32) @ bt.to(_torch.float32).T
        dist2 = emb_norm[:, None] + bn[None, :] - 2.0 * dot
        dist2 = _torch.clamp(dist2, min=0.0)

        kk = min(k, dist2.shape[1])
        blk_best, _ = _torch.topk(dist2, k=kk, largest=False, dim=1)
        cat = _torch.cat([best, blk_best], dim=1)
        best, _ = _torch.topk(cat, k=k, largest=False, dim=1)

    return _torch.sqrt(best).mean(dim=1)  # [M]


def score_from_patch_distances(
    d_patch: Any,
    score_mode: str,
    topk: int,
    percentile: float = 99.0,
    hybrid_alpha: float = 0.5,
) -> float:
    """Compute glyph-level score from patch distances.

    Args:
        d_patch: [N] tensor of patch distances
        score_mode: Scoring strategy:
            - "max": Maximum patch distance (most sensitive to single outlier)
            - "topk": Mean of top-K patch distances (default)
            - "percentile": Use a high percentile of patch distances
            - "hybrid": Blend of max and topk: alpha*max + (1-alpha)*topk_mean
            - "contrast": top-K mean / median ratio.  Measures how much the
              worst patches stand out from the typical patch.  A localized
              defect (missing corner) produces a few very-high-distance
              patches while the median stays low → high ratio.  A global
              variation (stroke thickness) raises *all* patches uniformly
              → ratio stays close to 1.  This naturally suppresses false
              positives from thickness/size variation.
            - "spatial": topk_mean × (1 + locality).  Weights score by
              spatial concentration of top-K patches.  Clustered anomalies
              (defect in one corner) get up to 2× amplification vs spread
              anomalies (uniform thickness change).
        topk: Number of top distances to average (for topk/hybrid/contrast)
        percentile: Percentile to use (for percentile mode, e.g. 99.0)
        hybrid_alpha: Weight for max in hybrid mode (0..1)
    """
    _ensure_torch()
    assert _torch is not None
    n = d_patch.numel()

    if score_mode == "max":
        return float(d_patch.max().item())

    if score_mode == "percentile":
        # Use percentile: e.g. 99th percentile of N patches
        k = max(1, int(n * (1.0 - percentile / 100.0)))
        k = max(1, min(k, n))
        v = _torch.topk(d_patch, k=k, largest=True).values
        return float(v.mean().item())

    if score_mode == "hybrid":
        # Blend max and topk_mean for sensitivity to both isolated and
        # distributed defects
        max_val = float(d_patch.max().item())
        tk = max(1, min(int(topk), n))
        topk_val = float(_torch.topk(d_patch, k=tk, largest=True).values.mean().item())
        alpha = max(0.0, min(1.0, hybrid_alpha))
        return alpha * max_val + (1.0 - alpha) * topk_val

    if score_mode == "contrast":
        # Ratio of top-K mean to median.
        # Localized defect → high ratio (outlier patches >> median)
        # Global variation  → ratio ≈ 1  (all patches equally elevated)
        tk = max(1, min(int(topk), n))
        topk_mean = float(_torch.topk(d_patch, k=tk, largest=True).values.mean().item())
        median_val = float(_torch.median(d_patch).item())
        eps = 1e-6
        return topk_mean / (median_val + eps)

    if score_mode == "spatial":
        # Spatial-concentration-weighted scoring.
        #
        # Idea: a LOCALIZED defect (missing corner) produces high-distance
        # patches that are CLUSTERED in one area of the feature map.
        # A GLOBAL variation (thickness) produces high-distance patches
        # that are SPREAD across the whole feature map.
        #
        # 1. Find the top-K most anomalous patches and their 2D positions.
        # 2. Measure spatial dispersion (mean distance from centroid).
        # 3. Compute locality = 1 − (dispersion / max_possible_dispersion).
        #    Clustered → locality ≈ 1; spread → locality ≈ 0.
        # 4. Final score = topk_mean × (1 + locality).
        #    Concentrated anomalies get up to 2× amplification.
        tk = max(1, min(int(topk), n))
        values, indices = _torch.topk(d_patch, k=tk, largest=True)
        topk_mean = float(values.mean().item())

        # Infer 2D spatial dimensions (feature map is always square)
        h = int(round(n ** 0.5))
        w = h if h * h == n else n // h
        rows = (indices // w).float()
        cols = (indices % w).float()

        # Spatial dispersion: RMS distance from centroid of top-K
        cr = rows.mean()
        cc = cols.mean()
        dispersion = (((rows - cr) ** 2 + (cols - cc) ** 2).mean()).sqrt()

        # Max possible dispersion ≈ half the diagonal
        max_disp = ((h / 2.0) ** 2 + (w / 2.0) ** 2) ** 0.5
        locality = 1.0 - min(float(dispersion.item()) / (max_disp + 1e-6), 1.0)

        return topk_mean * (1.0 + locality)

    # Default: topk
    topk = max(1, min(int(topk), n))
    v = _torch.topk(d_patch, k=topk, largest=True).values
    return float(v.mean().item())


# ---------------------------------------------------------------------------
# Class model data
# ---------------------------------------------------------------------------


@dataclass
class GlyphClassModel:
    """Holds loaded data for a single glyph class."""

    cls: str
    img_size: int
    k: int
    score_mode: str
    topk: int
    thr: float
    bank_t: Any          # torch.Tensor on device (fp16 for GPU)
    bank_norm: Any        # torch.Tensor [B] float32
    nn_cpu: Any = None    # sklearn NearestNeighbors (CPU fallback)
    multi_scale: bool = False   # layer1+layer2 multi-scale features
    use_clahe: bool = False     # CLAHE preprocessing applied during training
    clahe_clip: float = 3.0     # CLAHE clip limit
    clahe_grid: int = 4         # CLAHE tile grid size
    score_percentile: float = 99.0   # for percentile score_mode
    hybrid_alpha: float = 0.5        # for hybrid score_mode
    use_edge: bool = False      # Sobel edge preprocessing


# ---------------------------------------------------------------------------
# GlyphPatchCoreEngine - the main inference/training engine
# ---------------------------------------------------------------------------


class GlyphPatchCoreEngine:
    """Engine for glyph-level PatchCore anomaly detection.

    Loads per-class .joblib models and performs batched CNN + kNN inference.
    Supports GPU acceleration with FP16 and GPU-based kNN.
    """

    def __init__(
        self,
        model_dir: str,
        device: str = "auto",
        use_fp16: bool = True,
        use_gpu_knn: bool = True,
        cnn_batch: int = 64,
        knn_bank_block: int = 20000,
    ) -> None:
        _ensure_torch()
        assert _torch is not None

        if device == "auto":
            self.device = "cuda" if _torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.fp16 = use_fp16 and self.device == "cuda"
        self.use_gpu_knn = use_gpu_knn and self.device == "cuda"
        self.cnn_batch = cnn_batch
        self.knn_bank_block = knn_bank_block

        if self.device == "cuda":
            _torch.backends.cudnn.benchmark = True

        self.cls_models: dict[str, GlyphClassModel] = {}

        # Load class models first (to read multi_scale flag from saved models)
        self._load_models(Path(model_dir))

        # Determine feature extraction mode from loaded models
        self.multi_scale = False
        self.use_clahe = False
        self.clahe_clip = 3.0
        self.clahe_grid = 4
        self.use_edge = False
        if self.cls_models:
            sample = next(iter(self.cls_models.values()))
            self.multi_scale = sample.multi_scale
            self.use_clahe = sample.use_clahe
            self.clahe_clip = sample.clahe_clip
            self.clahe_grid = sample.clahe_grid
            self.use_edge = sample.use_edge

        # Build feature extractor matching the model's training mode
        self.net = ResNetFeatGlyph(multi_scale=self.multi_scale).to(self.device).eval()
        self._warmup()

    def _load_models(self, model_dir: Path) -> None:
        """Load all .joblib class models from the model directory."""
        import joblib

        _ensure_torch()
        assert _torch is not None

        files = sorted(model_dir.glob("*.joblib"))
        if not files:
            raise RuntimeError(f"No *.joblib model files found in {model_dir}")

        total_bank = 0
        for fp in files:
            m = joblib.load(fp)
            cls = str(m["cls"])
            bank = np.asarray(m["memory_bank"], dtype=np.float32)
            total_bank += bank.shape[0]

            img_size = int(m.get("img_size", 128))
            k = int(m.get("k", 1))
            score_mode = str(m.get("score_mode", "topk"))
            topk = int(m.get("topk", 10))
            thr = float(m.get("thr", 1e9))
            multi_scale = bool(m.get("multi_scale", False))
            use_clahe = bool(m.get("use_clahe", False))
            clahe_clip = float(m.get("clahe_clip", 3.0))
            clahe_grid = int(m.get("clahe_grid", 4))
            score_percentile = float(m.get("score_percentile", 99.0))
            hybrid_alpha = float(m.get("hybrid_alpha", 0.5))
            use_edge = bool(m.get("use_edge", False))

            # Prepare tensors for GPU kNN
            dev = _torch.device("cuda") if self.use_gpu_knn else _torch.device("cpu")
            dtype = _torch.float16 if self.use_gpu_knn else _torch.float32
            bank_t = _torch.from_numpy(bank).to(dev, dtype=dtype)
            bank_norm = (bank_t.to(_torch.float32) * bank_t.to(_torch.float32)).sum(dim=1)

            # Also build sklearn NN for CPU fallback
            nn_cpu = None
            if not self.use_gpu_knn:
                from sklearn.neighbors import NearestNeighbors
                nn_cpu = NearestNeighbors(n_neighbors=k, metric="euclidean")
                nn_cpu.fit(bank)

            self.cls_models[cls] = GlyphClassModel(
                cls=cls, img_size=img_size, k=k, score_mode=score_mode,
                topk=topk, thr=thr, bank_t=bank_t, bank_norm=bank_norm,
                nn_cpu=nn_cpu,
                multi_scale=multi_scale, use_clahe=use_clahe,
                clahe_clip=clahe_clip, clahe_grid=clahe_grid,
                score_percentile=score_percentile, hybrid_alpha=hybrid_alpha,
                use_edge=use_edge,
            )

        logger.info(
            "Glyph models loaded: %d classes from %s, total_bank=%d, "
            "device=%s, fp16=%s, gpu_knn=%s, multi_scale=%s, clahe=%s, edge=%s",
            len(self.cls_models), model_dir, total_bank,
            self.device, self.fp16, self.use_gpu_knn,
            self.cls_models and next(iter(self.cls_models.values())).multi_scale,
            self.cls_models and next(iter(self.cls_models.values())).use_clahe,
            self.cls_models and next(iter(self.cls_models.values())).use_edge,
        )

    def _warmup(self) -> None:
        """Run a dummy forward pass to warm up CUDA."""
        _ensure_torch()
        assert _torch is not None
        dummy = np.zeros((64, 64), dtype=np.uint8)
        x = preprocess_gray_to_tensor(dummy, 128).to(self.device)
        with _torch.no_grad():
            if self.fp16:
                with _torch.autocast(device_type="cuda", dtype=_torch.float16):
                    _ = self.net(x)
            else:
                _ = self.net(x)
        if self.device == "cuda":
            _torch.cuda.synchronize()
        logger.info("Glyph PatchCore engine warmup done")

    def unload(self) -> None:
        """Release GPU memory."""
        _ensure_torch()
        assert _torch is not None
        del self.net
        self.cls_models.clear()
        if _torch.cuda.is_available():
            _torch.cuda.empty_cache()

    def predict(
        self,
        image_path: str,
        json_path: str,
        output_overlay: str | None = None,
        pad: int = 2,
        thr_global: float | None = None,
    ) -> dict[str, Any]:
        """Run glyph-level inference on a single image + JSON annotation pair.

        Args:
            image_path: Path to the input image.
            json_path: Path to the JSON annotation file with character positions.
            output_overlay: Optional path to save overlay visualization.
            pad: Padding pixels around each glyph bounding box.
            thr_global: Optional global threshold override for all classes.

        Returns:
            dict with keys: pred, score, regions, artifacts, timing_ms,
            glyph_total, ng_count, unk_count
        """
        _ensure_torch()
        assert _torch is not None

        t0 = time.perf_counter()

        # Read image
        bgr = imread_any_bgr(Path(image_path))
        if bgr is None:
            raise RuntimeError(f"Failed to read image: {image_path}")
        h_img, w_img = bgr.shape[:2]
        gray_full = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # Read JSON annotation
        json_data = json.loads(Path(json_path).read_text(encoding="utf-8-sig"))
        items = json_data.get("items", [])
        if not isinstance(items, list):
            items = []

        # Parse glyph items
        glyphs: list[dict[str, Any]] = []
        vis = bgr.copy() if output_overlay else None
        unk_count = 0

        for it in items:
            ch = str(it.get("ch", ""))
            if not ch:
                continue
            cx = float(it.get("cx", 0))
            cy = float(it.get("cy", 0))
            ww = float(it.get("w", 0))
            hh = float(it.get("h", 0))
            if ww <= 0 or hh <= 0:
                continue

            x0 = int(round(cx - ww / 2)) - pad
            y0 = int(round(cy - hh / 2)) - pad
            x1 = int(round(cx + ww / 2)) + pad
            y1 = int(round(cy + hh / 2)) + pad
            x0, y0, x1, y1 = clip_box(x0, y0, x1, y1, w_img, h_img)

            cls = safe_folder_name(ch)
            if cls not in self.cls_models:
                unk_count += 1
                if vis is not None:
                    cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 255), 2)
                    cv2.putText(
                        vis, f"{ch}:UNK", (x0, max(15, y0 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2,
                    )
                continue

            cm = self.cls_models[cls]
            thr = thr_global if thr_global is not None else cm.thr
            patch = gray_full[y0:y1, x0:x1]

            glyphs.append({
                "ch": ch, "cls": cls, "cm": cm, "thr": thr,
                "box": (x0, y0, x1, y1), "patch": patch,
                "idx": it.get("i", it.get("idx", -1)),
                "emb_t": None,
            })

        if not glyphs:
            if output_overlay and vis is not None:
                imwrite_any(Path(output_overlay), vis)
            return {
                "pred": "OK",
                "score": 0.0,
                "regions": [],
                "artifacts": {"overlay": output_overlay or ""},
                "timing_ms": {"cnn": 0.0, "knn": 0.0, "total": 0.0},
                "glyph_total": 0,
                "ng_count": 0,
                "unk_count": unk_count,
            }

        # ---- CNN batch inference ----
        if self.device == "cuda":
            _torch.cuda.synchronize()
        t_cnn0 = time.perf_counter()

        # Group by img_size for batching
        groups: dict[int, list[int]] = {}
        for i, g in enumerate(glyphs):
            groups.setdefault(int(g["cm"].img_size), []).append(i)

        with _torch.no_grad():
            for sz, idxs in groups.items():
                for s in range(0, len(idxs), self.cnn_batch):
                    batch_ids = idxs[s:s + self.cnn_batch]
                    xs = [
                        preprocess_gray_to_tensor(
                            glyphs[gi]["patch"], sz,
                            use_clahe=self.use_clahe,
                            clahe_clip=self.clahe_clip,
                            clahe_grid=self.clahe_grid,
                            use_edge=self.use_edge,
                        )
                        for gi in batch_ids
                    ]
                    x = _torch.cat(xs, dim=0).to(self.device, non_blocking=False)

                    if self.fp16:
                        with _torch.autocast(device_type="cuda", dtype=_torch.float16):
                            fm = self.net(x)
                        emb = extract_patch_embeddings_batch(fm).float()
                    else:
                        fm = self.net(x)
                        emb = extract_patch_embeddings_batch(fm)

                    for bi, gi in enumerate(batch_ids):
                        glyphs[gi]["emb_t"] = emb[bi].contiguous()

        if self.device == "cuda":
            _torch.cuda.synchronize()
        t_cnn1 = time.perf_counter()
        cnn_ms = (t_cnn1 - t_cnn0) * 1000.0

        # ---- kNN scoring ----
        t_knn0 = time.perf_counter()

        ng_count = 0
        max_score = 0.0
        regions: list[dict[str, Any]] = []

        if self.use_gpu_knn:
            # GPU kNN: group by class, batch all embeddings
            cls_map: dict[str, list[int]] = {}
            for i, g in enumerate(glyphs):
                cls_map.setdefault(g["cls"], []).append(i)

            for cls, idxs in cls_map.items():
                cm: GlyphClassModel = glyphs[idxs[0]]["cm"]
                emb_list = [
                    glyphs[i]["emb_t"].reshape(-1, glyphs[i]["emb_t"].shape[-1])
                    for i in idxs
                ]
                big_t = _torch.cat(emb_list, dim=0).to("cuda", dtype=_torch.float16)

                d = knn_mean_distance_torch(
                    emb_t=big_t, bank_t=cm.bank_t, bank_norm=cm.bank_norm,
                    k=cm.k, block=self.knn_bank_block,
                )

                n_patches = glyphs[idxs[0]]["emb_t"].shape[0]
                for k_i, gi in enumerate(idxs):
                    dd = d[k_i * n_patches:(k_i + 1) * n_patches]
                    score = score_from_patch_distances(
                        dd, cm.score_mode, cm.topk,
                        percentile=cm.score_percentile,
                        hybrid_alpha=cm.hybrid_alpha,
                    )
                    self._record_glyph_result(
                        glyphs[gi], score, vis, regions, thr_global,
                    )
                    if score > max_score:
                        max_score = score
                    if score > float(glyphs[gi]["thr"]):
                        ng_count += 1
        else:
            # CPU kNN fallback
            for g in glyphs:
                cm = g["cm"]
                emb_np = g["emb_t"].cpu().numpy().astype(np.float32)
                assert cm.nn_cpu is not None
                d_arr, _ = cm.nn_cpu.kneighbors(emb_np)
                d_mean = d_arr.mean(axis=1)
                d_t = _torch.from_numpy(d_mean)
                score = score_from_patch_distances(
                    d_t, cm.score_mode, cm.topk,
                    percentile=cm.score_percentile,
                    hybrid_alpha=cm.hybrid_alpha,
                )
                self._record_glyph_result(g, score, vis, regions, thr_global)
                if score > max_score:
                    max_score = score
                if score > float(g["thr"]):
                    ng_count += 1

        if self.device == "cuda":
            _torch.cuda.synchronize()
        t_knn1 = time.perf_counter()
        knn_ms = (t_knn1 - t_knn0) * 1000.0

        # ---- Save overlay ----
        artifacts: dict[str, str] = {}
        if output_overlay and vis is not None:
            imwrite_any(Path(output_overlay), vis)
            artifacts["overlay"] = output_overlay

        t_total = (time.perf_counter() - t0) * 1000.0

        overall_pred = "NG" if ng_count > 0 else "OK"

        return {
            "pred": overall_pred,
            "score": round(float(max_score), 6),
            "threshold": round(float(thr_global or 0.0), 6) if thr_global else 0.0,
            "regions": regions,
            "artifacts": artifacts,
            "timing_ms": {
                "cnn": round(cnn_ms, 2),
                "knn": round(knn_ms, 2),
                "total": round(t_total, 2),
            },
            "glyph_total": len(glyphs),
            "ng_count": ng_count,
            "unk_count": unk_count,
        }

    def _record_glyph_result(
        self,
        glyph: dict[str, Any],
        score: float,
        vis: np.ndarray | None,
        regions: list[dict[str, Any]],
        thr_global: float | None,
    ) -> None:
        """Record a single glyph's result into regions list and draw on overlay."""
        thr = thr_global if thr_global is not None else float(glyph["thr"])
        is_ng = score > thr
        x0, y0, x1, y1 = glyph["box"]
        ch = glyph["ch"]

        decision = "NG" if is_ng else "OK"
        regions.append({
            "ch": ch,
            "cls": glyph["cls"],
            "idx": glyph.get("idx", -1),
            "x": x0,
            "y": y0,
            "w": x1 - x0,
            "h": y1 - y0,
            "score": round(score, 6),
            "threshold": round(thr, 6),
            "decision": decision,
            "area_px": (x1 - x0) * (y1 - y0),
        })

        if vis is not None:
            if is_ng:
                cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 0, 255), 2)
                cv2.putText(
                    vis, f"{ch}:{score:.2f}", (x0, max(15, y0 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2,
                )
            else:
                cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 0), 1)


# ---------------------------------------------------------------------------
# Training functions
# ---------------------------------------------------------------------------


def crop_glyphs_from_json(
    json_path: Path,
    img_dir: Path,
    out_dir: Path,
    pad: int = 2,
    ext: str = ".jpg",
) -> int:
    """Crop individual glyph images from a JSON annotation file.

    Args:
        json_path: Path to JSON annotation file.
        img_dir: Directory containing the source images.
        out_dir: Output directory for glyph crops (organized by character class).
        pad: Padding pixels around each bounding box.
        ext: Output image extension.

    Returns:
        Number of crops saved.
    """
    from PIL import Image as PILImage

    data = json.loads(json_path.read_text(encoding="utf-8-sig"))
    image_name = data.get("image_name", "")
    if not image_name:
        return 0

    img_path = img_dir / image_name
    if not img_path.exists():
        img_path = img_dir / Path(image_name).name
    if not img_path.exists():
        logger.warning("Image not found for %s: %s", json_path.name, img_path)
        return 0

    items = data.get("items", [])
    if not items:
        return 0

    img = PILImage.open(img_path).convert("RGB")
    w_img, h_img = img.size

    saved = 0
    for it in items:
        ch = str(it.get("ch", ""))
        if not ch:
            continue
        cx = float(it.get("cx", 0))
        cy = float(it.get("cy", 0))
        w = float(it.get("w", 0))
        h = float(it.get("h", 0))
        if w <= 0 or h <= 0:
            continue

        x0 = int(round(cx - w / 2.0)) - pad
        y0 = int(round(cy - h / 2.0)) - pad
        x1 = int(round(cx + w / 2.0)) + pad
        y1 = int(round(cy + h / 2.0)) + pad
        x0, y0, x1, y1 = clip_box(x0, y0, x1, y1, w_img, h_img)

        crop = img.crop((x0, y0, x1, y1))
        folder = out_dir / safe_folder_name(ch)
        folder.mkdir(parents=True, exist_ok=True)

        # Auto-increment filename
        max_n = 0
        for fp in folder.glob(f"*{ext}"):
            if fp.stem.isdigit():
                n = int(fp.stem)
                if n > max_n:
                    max_n = n
        out_file = folder / f"{max_n + 1:04d}{ext}"
        crop.save(out_file, quality=95)
        saved += 1

    return saved


def train_glyph_patchcore(
    bank_dir: str,
    out_model_dir: str,
    img_size: int = 128,
    max_patches_per_class: int = 30000,
    k: int = 1,
    score_mode: str = "topk",
    topk: int = 10,
    p_thr: float = 0.995,
    min_per_class: int = 10,
    multi_scale: bool = False,
    use_clahe: bool = False,
    clahe_clip: float = 3.0,
    clahe_grid: int = 4,
    score_percentile: float = 99.0,
    hybrid_alpha: float = 0.5,
    use_edge: bool = False,
    progress_cb: Any = None,
) -> dict[str, Any]:
    """Train PatchCore models for each glyph class.

    Args:
        bank_dir: Path to glyph bank directory (one subdirectory per class).
        out_model_dir: Output directory for trained models (.joblib files).
        img_size: Input image size for the CNN.
        max_patches_per_class: Maximum patches in memory bank per class.
        k: Number of nearest neighbors.
        score_mode: Score aggregation mode ("max", "topk", "percentile",
            "hybrid").
        topk: Number of top distances to average when score_mode="topk".
        p_thr: Quantile for threshold computation from OK score distribution.
        min_per_class: Minimum images per class to train.
        multi_scale: Use layer1+layer2 multi-scale features (192-dim, 1/4
            resolution) instead of layer2 only (128-dim, 1/8 resolution).
            Multi-scale provides 4x finer spatial detail for detecting small
            defects like missing corners or thin cracks.
        use_clahe: Apply CLAHE preprocessing to normalize local contrast,
            reducing sensitivity to stroke thickness / intensity variations.
        clahe_clip: CLAHE clip limit.
        clahe_grid: CLAHE tile grid size.
        score_percentile: Percentile for "percentile" score_mode.
        hybrid_alpha: Weight for max in "hybrid" score_mode (0..1).
        use_edge: Apply Sobel edge preprocessing before CNN feature
            extraction.  Converts glyph images to edge maps so the model
            compares character *boundaries/shape* instead of fill/intensity.
            This dramatically improves separation between structural defects
            (missing corners) and cosmetic variations (stroke thickness).
        progress_cb: Optional callback(progress_pct, message).

    Returns:
        Training report dict.
    """
    _ensure_torch()
    assert _torch is not None

    import joblib
    from sklearn.neighbors import NearestNeighbors

    bank_path = Path(bank_dir)
    out_path = Path(out_model_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    device = "cuda" if _torch.cuda.is_available() else "cpu"
    net = ResNetFeatGlyph(multi_scale=multi_scale).to(device).eval()

    class_dirs = [p for p in bank_path.iterdir() if p.is_dir()]
    if not class_dirs:
        raise RuntimeError(f"No class folders found in {bank_dir}")

    index: dict[str, Any] = {
        "img_size": img_size,
        "score_mode": score_mode,
        "topk": topk,
        "k": k,
        "p_thr": p_thr,
        "multi_scale": multi_scale,
        "use_clahe": use_clahe,
        "clahe_clip": clahe_clip,
        "clahe_grid": clahe_grid,
        "score_percentile": score_percentile,
        "hybrid_alpha": hybrid_alpha,
        "use_edge": use_edge,
        "classes": [],
    }

    total_classes = len(class_dirs)
    trained_classes = 0

    for ci, cls_dir in enumerate(sorted(class_dirs, key=lambda p: p.name)):
        cls = cls_dir.name
        imgs = list_images(cls_dir)

        if len(imgs) < min_per_class:
            logger.warning("Skip class %s: %d images < min %d", cls, len(imgs), min_per_class)
            continue

        if progress_cb:
            pct = 10.0 + 80.0 * ci / total_classes
            progress_cb(pct, f"Training class {cls} ({len(imgs)} images)...")

        # Extract patch embeddings
        all_patches: list[np.ndarray] = []
        for p in imgs:
            g = imread_any_gray(p)
            if g is None:
                continue
            x = preprocess_gray_to_tensor(
                g, size=img_size,
                use_clahe=use_clahe, clahe_clip=clahe_clip, clahe_grid=clahe_grid,
                use_edge=use_edge,
            ).to(device)
            with _torch.no_grad():
                fmap = net(x)
                emb = extract_patch_embeddings_batch(fmap)
                emb_np = emb.squeeze(0).cpu().numpy().astype(np.float32)
            all_patches.append(emb_np)

        if not all_patches:
            continue

        X = np.vstack(all_patches)

        # Subsample if too large
        if X.shape[0] > max_patches_per_class:
            idxs = np.random.choice(X.shape[0], max_patches_per_class, replace=False)
            X = X[idxs]

        # Build kNN
        nn_model = NearestNeighbors(n_neighbors=k, metric="euclidean")
        nn_model.fit(X)

        # Compute threshold from OK score distribution using same scoring mode
        ok_scores: list[float] = []
        for emb_np in all_patches:
            d_arr, _ = nn_model.kneighbors(emb_np)
            d_mean = d_arr.mean(axis=1)
            d_t = _torch.from_numpy(d_mean)
            s = score_from_patch_distances(
                d_t, score_mode, topk,
                percentile=score_percentile, hybrid_alpha=hybrid_alpha,
            )
            ok_scores.append(s)

        thr = float(np.quantile(ok_scores, p_thr))

        model_data = {
            "cls": cls,
            "img_size": int(img_size),
            "k": int(k),
            "score_mode": score_mode,
            "topk": int(topk),
            "p_thr": float(p_thr),
            "thr": thr,
            "memory_bank": X,
            "multi_scale": multi_scale,
            "use_clahe": use_clahe,
            "clahe_clip": clahe_clip,
            "clahe_grid": clahe_grid,
            "score_percentile": score_percentile,
            "hybrid_alpha": hybrid_alpha,
            "use_edge": use_edge,
        }
        joblib.dump(model_data, out_path / f"{cls}.joblib")

        index["classes"].append({
            "cls": cls,
            "thr": thr,
            "n_ok": len(imgs),
            "n_patches": int(X.shape[0]),
        })
        trained_classes += 1
        logger.info("Class %s: ok_imgs=%d, memory=%s, thr=%.4f", cls, len(imgs), X.shape, thr)

    # Save index
    (out_path / "index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8",
    )

    if progress_cb:
        progress_cb(100.0, f"Training complete: {trained_classes} classes")

    return {
        "trained_classes": trained_classes,
        "total_classes": total_classes,
        "index": index,
    }

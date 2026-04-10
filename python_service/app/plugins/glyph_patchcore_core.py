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
    """ResNet18 feature extractor up to layer2 (shallow, suitable for glyph defects).

    Supports two modes:
    - "layer2" (default): Uses only layer2 (128-dim features at ~8x8 resolution)
    - "multi" (layer1+layer2): Concatenates layer1 (64-dim, ~16x16) with
      upsampled layer2 (128-dim) to produce 192-dim features at ~16x16 resolution.
      This gives 4x more patches and higher spatial sensitivity for small defects.
    """

    def __init__(self, feature_layers: str = "layer2") -> None:
        _ensure_torch()
        assert _torch is not None and _nn is not None and _models is not None
        m = _models.resnet18(weights=_models.ResNet18_Weights.IMAGENET1K_V1)
        self.stem = _nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)
        self.layer1 = m.layer1
        self.layer2 = m.layer2
        self.feature_layers = feature_layers
        # Build as nn.Module for .to() / .eval() / state_dict
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
        if self.feature_layers == "multi":
            return self._forward_multiscale(x)
        return self._net(x)

    def _forward_multiscale(self, x: Any) -> Any:
        """Multi-scale forward: concatenate layer1 + upsampled layer2.

        Returns [B, 192, H1, W1] where H1/W1 is layer1's spatial resolution.
        This provides 4x more patches than layer2-only, each covering a smaller
        area of the original image, making small defects more detectable.
        """
        assert _torch is not None and _nn is not None
        x = self.stem(x)
        feat1 = self.layer1(x)    # [B, 64, H1, W1]  e.g. 16x16 for 128 input
        feat2 = self.layer2(feat1) # [B, 128, H2, W2] e.g. 8x8 for 128 input
        # Upsample layer2 to match layer1's spatial resolution
        feat2_up = _nn.functional.interpolate(
            feat2, size=feat1.shape[2:], mode="bilinear", align_corners=False,
        )
        return _torch.cat([feat1, feat2_up], dim=1)  # [B, 192, H1, W1]


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


def morph_augment_gray(
    gray: np.ndarray,
    rng: np.random.RandomState | None = None,
) -> list[np.ndarray]:
    """Generate morphological augmentations of a grayscale glyph image.

    Creates variants with different stroke widths and slight size changes
    to make the memory bank robust to normal font thickness/size variations.
    This way, during inference, thickness/size changes produce low distances
    while structural defects (broken/missing lines) still produce high distances.

    Returns a list of augmented grayscale images (does NOT include original).
    """
    if rng is None:
        rng = np.random.RandomState()

    h, w = gray.shape[:2]
    augmented: list[np.ndarray] = []

    # --- Erosion variants (simulate thinner strokes) ---
    for ksize in (2, 3):
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        augmented.append(cv2.erode(gray, kern, iterations=1))

    # --- Dilation variants (simulate thicker strokes) ---
    for ksize in (2, 3):
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        augmented.append(cv2.dilate(gray, kern, iterations=1))

    # --- Scale variants (simulate size changes) ---
    for scale in (0.90, 0.95, 1.05, 1.10):
        new_h, new_w = int(h * scale), int(w * scale)
        if new_h < 4 or new_w < 4:
            continue
        scaled = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        # Center-crop or pad back to original size
        canvas = np.full((h, w), gray.mean(), dtype=np.uint8)
        sy = max(0, (h - new_h) // 2)
        sx = max(0, (w - new_w) // 2)
        cy = max(0, (new_h - h) // 2)
        cx = max(0, (new_w - w) // 2)
        ph = min(new_h - cy, h - sy)
        pw = min(new_w - cx, w - sx)
        canvas[sy:sy + ph, sx:sx + pw] = scaled[cy:cy + ph, cx:cx + pw]
        augmented.append(canvas)

    return augmented


def preprocess_gray_to_tensor(
    gray: np.ndarray, size: int, clahe_clip: float = 0.0,
) -> Any:
    """Preprocess a grayscale glyph crop to a [1,3,S,S] tensor.

    Args:
        gray: Grayscale glyph image (uint8).
        size: Target size for resizing.
        clahe_clip: CLAHE clipLimit for local contrast enhancement.
            0.0 means disabled (default, backward compatible).
            Recommended range 1.0-3.0 for small-defect enhancement.
    """
    _ensure_torch()
    g = cv2.resize(gray, (size, size), interpolation=cv2.INTER_CUBIC)
    if clahe_clip > 0:
        clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(4, 4))
        g = clahe.apply(g)
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


def score_from_patch_distances(d_patch: Any, score_mode: str, topk: int) -> float:
    """Compute glyph-level score from patch distances.

    Args:
        d_patch: [N] tensor of patch distances
        score_mode: Aggregation mode:
            - "max": maximum patch distance (most sensitive, may be noisy)
            - "topk": mean of top-k largest distances (default, robust)
            - "adaptive": geometric mean of max and topk-mean, i.e.
              sqrt(max * topk_mean). This amplifies the anomaly signal
              from the worst patch while retaining topk robustness.
              Especially effective for small defects where only 1-2
              patches are anomalous.
            - "relative": subtract median distance then score on top-k of
              the residuals. This makes the score invariant to global
              appearance shifts (e.g. font thickness/size changes that
              affect all patches equally) while remaining sensitive to
              local structural defects (broken/missing lines) that only
              affect a few patches.
        topk: number of top distances to average (for "topk", "adaptive", "relative")
    """
    _ensure_torch()
    assert _torch is not None
    if score_mode == "max":
        return float(d_patch.max().item())
    if score_mode == "relative":
        # Subtract median to remove global shift from thickness/size variations
        median_d = float(d_patch.median().item())
        residuals = d_patch - median_d
        topk = max(1, min(int(topk), residuals.numel()))
        v = _torch.topk(residuals, k=topk, largest=True).values
        return float(v.mean().item())
    topk = max(1, min(int(topk), d_patch.numel()))
    v = _torch.topk(d_patch, k=topk, largest=True).values
    topk_mean = float(v.mean().item())
    if score_mode == "adaptive":
        max_val = float(d_patch.max().item())
        # Geometric mean: amplifies max anomaly while keeping topk robustness
        return float(np.sqrt(max_val * topk_mean))
    return topk_mean


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
    feature_layers: str = "layer2"  # "layer2" or "multi" (layer1+layer2)
    clahe_clip: float = 0.0         # CLAHE clipLimit (0 = disabled)


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
        feature_layers: str = "layer2",
        clahe_clip: float = 0.0,
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
        self.feature_layers = feature_layers
        self.clahe_clip = clahe_clip

        if self.device == "cuda":
            _torch.backends.cudnn.benchmark = True

        # Build feature extractor (detect multi-scale from model files)
        self.cls_models: dict[str, GlyphClassModel] = {}

        # Load class models first to detect feature_layers from saved models
        self._load_models(Path(model_dir))

        # Determine effective feature_layers: model-saved value takes priority
        effective_layers = self._detect_feature_layers()
        self.net = ResNetFeatGlyph(feature_layers=effective_layers).to(self.device).eval()
        self._warmup()

    def _detect_feature_layers(self) -> str:
        """Detect feature_layers from loaded models or use engine default."""
        if self.cls_models:
            first_model = next(iter(self.cls_models.values()))
            return first_model.feature_layers
        return self.feature_layers

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
            # Read feature_layers and clahe_clip from model if saved during training
            model_feature_layers = str(m.get("feature_layers", self.feature_layers))
            model_clahe_clip = float(m.get("clahe_clip", self.clahe_clip))

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
                feature_layers=model_feature_layers,
                clahe_clip=model_clahe_clip,
            )

        logger.info(
            "Glyph models loaded: %d classes from %s, total_bank=%d, "
            "device=%s, fp16=%s, gpu_knn=%s, feature_layers=%s, clahe_clip=%.1f",
            len(self.cls_models), model_dir, total_bank,
            self.device, self.fp16, self.use_gpu_knn,
            self._detect_feature_layers(), self.clahe_clip,
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
                            clahe_clip=glyphs[gi]["cm"].clahe_clip,
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
                    score = score_from_patch_distances(dd, cm.score_mode, cm.topk)
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
                if cm.score_mode == "max":
                    score = float(d_mean.max())
                else:
                    topk_val = min(cm.topk, d_mean.shape[0])
                    score = float(np.mean(np.sort(d_mean)[-topk_val:]))
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


def _compute_ok_score(d_mean: np.ndarray, score_mode: str, topk: int) -> float:
    """Compute a single glyph-level OK score from patch distances (numpy).

    Supports the same modes as score_from_patch_distances but on numpy arrays,
    used during training threshold computation.
    """
    if score_mode == "max":
        return float(d_mean.max())
    if score_mode == "relative":
        median_d = float(np.median(d_mean))
        residuals = d_mean - median_d
        tk = min(topk, residuals.shape[0])
        return float(np.mean(np.sort(residuals)[-tk:]))
    tk = min(topk, d_mean.shape[0])
    topk_mean = float(np.mean(np.sort(d_mean)[-tk:]))
    if score_mode == "adaptive":
        max_val = float(d_mean.max())
        return float(np.sqrt(max_val * topk_mean))
    return topk_mean


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
    progress_cb: Any = None,
    feature_layers: str = "layer2",
    clahe_clip: float = 0.0,
    morph_aug: bool = False,
) -> dict[str, Any]:
    """Train PatchCore models for each glyph class.

    Args:
        bank_dir: Path to glyph bank directory (one subdirectory per class).
        out_model_dir: Output directory for trained models (.joblib files).
        img_size: Input image size for the CNN.
        max_patches_per_class: Maximum patches in memory bank per class.
        k: Number of nearest neighbors.
        score_mode: Score aggregation mode ("max", "topk", "adaptive", or "relative").
        topk: Number of top distances to average.
        p_thr: Quantile for threshold computation from OK score distribution.
        min_per_class: Minimum images per class to train.
        progress_cb: Optional callback(progress_pct, message).
        feature_layers: Feature extraction mode:
            - "layer2" (default): ResNet18 layer2 only (128-dim, 8x8 for 128 input)
            - "multi": layer1+layer2 concatenated (192-dim, 16x16 for 128 input)
              Gives 4x more patches for better small-defect detection.
        clahe_clip: CLAHE clipLimit for local contrast enhancement (0 = disabled).
            Recommended 1.0-3.0 for small-defect enhancement.
        morph_aug: Enable morphological augmentation during training.
            When True, each OK training image is augmented with erosion,
            dilation, and scaling variants (8 extra per image) to make the
            memory bank robust to normal font thickness/size variations.
            This prevents false positives from slightly thicker/thinner/
            larger/smaller characters while maintaining sensitivity to
            structural defects (broken/missing lines).

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
    net = ResNetFeatGlyph(feature_layers=feature_layers).to(device).eval()

    class_dirs = [p for p in bank_path.iterdir() if p.is_dir()]
    if not class_dirs:
        raise RuntimeError(f"No class folders found in {bank_dir}")

    index: dict[str, Any] = {
        "img_size": img_size,
        "score_mode": score_mode,
        "topk": topk,
        "k": k,
        "p_thr": p_thr,
        "feature_layers": feature_layers,
        "clahe_clip": clahe_clip,
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
        all_patches: list[np.ndarray] = []   # original only (for threshold)
        aug_patches: list[np.ndarray] = []   # augmented (for memory bank only)
        for p in imgs:
            g = imread_any_gray(p)
            if g is None:
                continue
            x = preprocess_gray_to_tensor(
                g, size=img_size, clahe_clip=clahe_clip,
            ).to(device)
            with _torch.no_grad():
                fmap = net(x)
                emb = extract_patch_embeddings_batch(fmap)
                emb_np = emb.squeeze(0).cpu().numpy().astype(np.float32)
            all_patches.append(emb_np)

            # Morphological augmentation: add eroded/dilated/scaled variants
            if morph_aug:
                aug_imgs = morph_augment_gray(g)
                for ag in aug_imgs:
                    ax = preprocess_gray_to_tensor(
                        ag, size=img_size, clahe_clip=clahe_clip,
                    ).to(device)
                    with _torch.no_grad():
                        afmap = net(ax)
                        aemb = extract_patch_embeddings_batch(afmap)
                        aemb_np = aemb.squeeze(0).cpu().numpy().astype(np.float32)
                    aug_patches.append(aemb_np)

        if not all_patches:
            continue

        # Memory bank: original + augmented patches
        bank_patches = all_patches + aug_patches
        X = np.vstack(bank_patches)

        # Subsample if too large
        if X.shape[0] > max_patches_per_class:
            idxs = np.random.choice(X.shape[0], max_patches_per_class, replace=False)
            X = X[idxs]

        # Build kNN
        nn_model = NearestNeighbors(n_neighbors=k, metric="euclidean")
        nn_model.fit(X)

        # Compute threshold from OK score distribution
        ok_scores: list[float] = []
        for emb_np in all_patches:
            d_arr, _ = nn_model.kneighbors(emb_np)
            d_mean = d_arr.mean(axis=1)
            s = _compute_ok_score(d_mean, score_mode, topk)
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
            "feature_layers": feature_layers,
            "clahe_clip": float(clahe_clip),
            "morph_aug": bool(morph_aug),
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

"""Glyph Structural Matching -- matchTemplate alignment + local SSIM scoring.

A fundamentally different approach from PatchCore: instead of CNN features
+ kNN distance, this algorithm compares characters directly using
template matching for alignment and SSIM for defect scoring.

Core idea (v5.1 -- refined training, April 2026)
-------------------------------------------------
1. **Training**: For each character class, CLAHE-normalise all OK samples,
   resize, compute initial mean via centroid alignment, then **iteratively
   refine** by re-aligning all images to the mean via matchTemplate
   (2 iterations).  This produces a much sharper mean template with
   lower per-pixel variance -- critical for detecting subtle defects.

2. **Inference**: CLAHE + resize to *larger canvas* (no centroid alignment),
   then cv2.matchTemplate to find the best position within the search
   region.  Extract aligned region and compute **local SSIM** map.
   Score = 1 - percentile(SSIM in foreground, 1)  -- i.e. the worst
   local structural similarity in the foreground region.

Why v5.1 refined training matters
---------------------------------
v5 used centroid alignment during training.  This introduces position
jitter (~2-5 px) between images because centroid computation depends on
binarisation quality, which varies with contrast.  The jitter inflates
the mean-template blur and the OK score distribution, making subtle
defects undetectable (their signal falls within normal variation).

By re-aligning training images to the mean via matchTemplate (the same
alignment used at inference), the mean becomes sharper, the OK score
distribution tightens, and the threshold can catch subtle defects.

Validated on 881 OK '2' images + 625 OK 'A' images + real label crops:
- Large defect (missing corner): detected at p95/p97
- Subtle defect (tiny notch): detected at p95/p97
- Zero false positives at p97 threshold on label crops

No CNN, no kNN, no feature embeddings.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.plugins.glyph_patchcore_core import (
    clip_box,
    imread_any_bgr,
    imread_any_gray,
    imwrite_any,
    list_images,
    safe_folder_name,
)

logger = logging.getLogger(__name__)

# Search padding for matchTemplate alignment during inference.
# The test image is resized to (img_size + 2*SEARCH_PAD) so that
# matchTemplate can slide the template around to find the best position.
SEARCH_PAD = 10


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------

def _normalise_contrast(gray: np.ndarray) -> np.ndarray:
    """Apply CLAHE to normalise contrast across different printing
    conditions (varying ink density, background brightness)."""
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
    return clahe.apply(gray)


def _binarise(gray: np.ndarray) -> np.ndarray:
    """Otsu binarisation with automatic foreground detection.

    After Otsu, ensures foreground (character strokes) = 255 and
    background = 0.  Characters are typically darker than background,
    so if majority of pixels are 255 after Otsu, we invert.
    """
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Ensure character strokes are white (255).
    # If more than half the pixels are white, background is white -> invert.
    if np.count_nonzero(bw) > bw.size * 0.5:
        bw = 255 - bw
    return bw


def _resize_pad(gray: np.ndarray, size: int) -> np.ndarray:
    """Resize keeping aspect ratio, then centre-pad to size x size."""
    h, w = gray.shape[:2]
    scale = size / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size), dtype=np.uint8)
    y0 = (size - nh) // 2
    x0 = (size - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


def _centroid_shift_params(binary_01: np.ndarray) -> tuple[int, int]:
    """Compute (dx, dy) to shift foreground centre of mass to canvas centre."""
    h, w = binary_01.shape
    ys, xs = np.where(binary_01 > 0.5)
    if len(ys) == 0:
        return 0, 0
    cy_fg = ys.mean()
    cx_fg = xs.mean()
    dy = int(round(h / 2.0 - cy_fg))
    dx = int(round(w / 2.0 - cx_fg))
    return dx, dy


def _apply_shift(
    img: np.ndarray, dx: int, dy: int, *,
    interp: int = cv2.INTER_LINEAR,
    border: int = cv2.BORDER_REPLICATE,
    border_value: float = 0,
) -> np.ndarray:
    """Translate *img* by (dx, dy) pixels."""
    if abs(dx) < 1 and abs(dy) < 1:
        return img
    h, w = img.shape[:2]
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(
        img, M, (w, h),
        flags=interp,
        borderMode=border,
        borderValue=border_value,
    )


def _prepare_binary(gray: np.ndarray, img_size: int) -> np.ndarray:
    """Preprocessing pipeline for alignment reference.

    1. CLAHE contrast normalisation
    2. Resize + pad to square
    3. Otsu binarise (with auto foreground detection)
    4. Centroid alignment

    Returns 0/1 float32 array of shape (img_size, img_size).
    """
    normed = _normalise_contrast(gray)
    resized = _resize_pad(normed, img_size)
    bw = _binarise(resized)
    binary_01 = (bw / 255.0).astype(np.float32)
    dx, dy = _centroid_shift_params(binary_01)
    aligned = _apply_shift(
        binary_01, dx, dy,
        interp=cv2.INTER_NEAREST,
        border=cv2.BORDER_CONSTANT,
        border_value=0,
    )
    return aligned


def _prepare_grayscale(gray: np.ndarray, img_size: int) -> np.ndarray:
    """Full preprocessing pipeline returning *grayscale* (not binary).

    Used during TRAINING (binary centroid alignment is OK here because
    training images are more consistent with each other).

    1. CLAHE contrast normalisation
    2. Resize + pad to square
    3. Centroid alignment (shift computed from binary, applied to grayscale)

    Returns float32 array of shape (img_size, img_size) in [0, 255].
    """
    normed = _normalise_contrast(gray)
    resized = _resize_pad(normed, img_size)

    # Compute centroid shift from binarised version (robust to contrast)
    bw = _binarise(resized)
    binary_01 = (bw / 255.0).astype(np.float32)
    dx, dy = _centroid_shift_params(binary_01)

    # Apply the SAME shift to the grayscale image
    gray_f = resized.astype(np.float32)
    aligned = _apply_shift(
        gray_f, dx, dy,
        interp=cv2.INTER_LINEAR,
        border=cv2.BORDER_REPLICATE,
    )
    return aligned


def _prepare_no_align(gray: np.ndarray, img_size: int) -> np.ndarray:
    """CLAHE + resize only, NO alignment.

    Used during INFERENCE -- alignment is done separately via
    phase correlation against the mean template.

    Returns float32 array of shape (img_size, img_size) in [0, 255].
    """
    normed = _normalise_contrast(gray)
    resized = _resize_pad(normed, img_size)
    return resized.astype(np.float32)


def _align_to_template(
    test_gray: np.ndarray,
    mean_template: np.ndarray,
    max_shift_frac: float = 0.20,
) -> np.ndarray:
    """Align *test_gray* to *mean_template* using phase correlation.

    This is FAR more robust than binary-centroid alignment because it
    works directly on grayscale pixel values -- no Otsu binarisation
    needed, so contrast variation does not corrupt the alignment.

    Args:
        test_gray: (H, W) float32 [0-255]
        mean_template: (H, W) float32 [0-255]
        max_shift_frac: reject shifts larger than this fraction of image
            size (alignment probably failed).

    Returns:
        Aligned test image, same shape and dtype.
    """
    h, w = test_gray.shape
    max_shift = int(max(h, w) * max_shift_frac)

    (dx, dy), _response = cv2.phaseCorrelate(
        mean_template.astype(np.float64),
        test_gray.astype(np.float64),
    )

    # Reject unreasonable shifts (alignment failed)
    if abs(dx) > max_shift or abs(dy) > max_shift:
        return test_gray

    return _apply_shift(
        test_gray, int(round(dx)), int(round(dy)),
        interp=cv2.INTER_LINEAR,
        border=cv2.BORDER_REPLICATE,
    )


# ---------------------------------------------------------------------------
# Model data
# ---------------------------------------------------------------------------

@dataclass
class StructuralClassModel:
    """Per-class structural model data (v4 -- binary deficit + edge NCC)."""
    cls: str = ""
    img_size: int = 128
    mean_template: np.ndarray = field(default_factory=lambda: np.zeros((1, 1), dtype=np.float32))
    std_template: np.ndarray = field(default_factory=lambda: np.zeros((1, 1), dtype=np.float32))
    foreground_mask: np.ndarray = field(default_factory=lambda: np.zeros((1, 1), dtype=np.float32))
    # Binary version of mean template (for deficit computation)
    mean_binary: np.ndarray = field(default_factory=lambda: np.zeros((1, 1), dtype=np.uint8))
    thr: float = 0.0
    n_ok: int = 0


# ---------------------------------------------------------------------------
# Scoring: matchTemplate alignment + local SSIM
# ---------------------------------------------------------------------------

def _prepare_for_matching(
    gray: np.ndarray, canvas_size: int,
) -> np.ndarray:
    """CLAHE + resize+pad to *canvas_size*.  NO centroid alignment.

    Used during inference -- alignment is done by matchTemplate.

    Returns float32 array of shape (canvas_size, canvas_size) in [0, 255].
    """
    normed = _normalise_contrast(gray)
    resized = _resize_pad(normed, canvas_size)
    return resized.astype(np.float32)


def _tmpl_align(
    raw_gray: np.ndarray,
    mean_template: np.ndarray,
    img_size: int = 128,
    search_pad: int = SEARCH_PAD,
) -> tuple[np.ndarray, float]:
    """Align a raw crop to *mean_template* using cv2.matchTemplate.

    1. CLAHE + resize to (img_size + 2*search_pad) -- larger canvas.
    2. Slide the mean_template (img_size x img_size) over the padded
       test image using TM_CCOEFF_NORMED.
    3. Extract the best-matching (img_size x img_size) region.

    Returns (aligned_region, match_value).
    """
    canvas = img_size + 2 * search_pad
    test_lg = _prepare_for_matching(raw_gray, canvas)
    tmpl = mean_template.astype(np.float32)
    res = cv2.matchTemplate(test_lg, tmpl, cv2.TM_CCOEFF_NORMED)
    _, mx_val, _, (mx, my) = cv2.minMaxLoc(res)
    aligned = test_lg[my:my + img_size, mx:mx + img_size]
    return aligned, float(mx_val)


def _score_ssim_worst(
    aligned: np.ndarray,
    mean_template: np.ndarray,
    foreground_mask: np.ndarray,
    ssim_percentile: float = 1.0,
) -> float:
    """Compute local SSIM defect score.

    Computes a pixel-wise SSIM map between *aligned* test image and
    *mean_template*, then returns ``1 - percentile(SSIM, ssim_percentile)``
    within the foreground region.

    Using the 1st percentile (default) captures the *worst* local
    region -- exactly what we need to detect localised defects while
    ignoring global variation.

    SSIM is inherently normalised for local luminance and contrast,
    making it robust to brightness/thickness changes.

    Args:
        aligned: (H, W) float32, matchTemplate-aligned test image [0-255]
        mean_template: (H, W) float32, mean of OK grayscale images [0-255]
        foreground_mask: (H, W) float32 in [0,1], foreground region
        ssim_percentile: percentile of foreground SSIM to use (default 1.0)

    Returns:
        score >= 0 -- higher means more defective.
    """
    test_u8 = np.clip(aligned, 0, 255).astype(np.uint8)
    mean_u8 = np.clip(mean_template, 0, 255).astype(np.uint8)

    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    t64 = test_u8.astype(np.float64)
    m64 = mean_u8.astype(np.float64)

    ksize = (11, 11)
    sigma = 1.5
    mu_t = cv2.GaussianBlur(t64, ksize, sigma)
    mu_m = cv2.GaussianBlur(m64, ksize, sigma)

    sig_t2 = cv2.GaussianBlur(t64 ** 2, ksize, sigma) - mu_t ** 2
    sig_m2 = cv2.GaussianBlur(m64 ** 2, ksize, sigma) - mu_m ** 2
    sig_tm = cv2.GaussianBlur(t64 * m64, ksize, sigma) - mu_t * mu_m

    ssim_map = (
        (2.0 * mu_t * mu_m + c1) * (2.0 * sig_tm + c2)
    ) / (
        (mu_t ** 2 + mu_m ** 2 + c1) * (sig_t2 + sig_m2 + c2)
    )

    fg_bool = foreground_mask > 0.3
    if fg_bool.sum() < 10:
        return 0.0

    fg_ssim = ssim_map[fg_bool]
    worst = float(np.percentile(fg_ssim, ssim_percentile))
    return 1.0 - worst


def structural_defect_score(
    test_gray_raw: np.ndarray,
    mean_template: np.ndarray,
    foreground_mask: np.ndarray,
    mean_binary: np.ndarray,  # kept for API compat, unused in v5
    img_size: int = 128,
    search_pad: int = SEARCH_PAD,
) -> float:
    """Compute structural defect score (v5 -- matchTemplate + SSIM).

    Full pipeline: matchTemplate alignment then local SSIM scoring.

    Args:
        test_gray_raw: raw grayscale crop (any size)
        mean_template: (img_size, img_size) float32, mean of OK images
        foreground_mask: (img_size, img_size) float32 in [0,1]
        mean_binary: unused (kept for backward API compatibility)
        img_size: template size (default 128)
        search_pad: pixels of search margin (default SEARCH_PAD)

    Returns:
        score >= 0 -- higher means more defective.
    """
    aligned, _match_val = _tmpl_align(
        test_gray_raw, mean_template, img_size, search_pad,
    )
    return _score_ssim_worst(aligned, mean_template, foreground_mask)


# ---------------------------------------------------------------------------
# Inference engine
# ---------------------------------------------------------------------------

class GlyphStructuralEngine:
    """Inference engine for structural template matching.

    Has the same predict() interface as GlyphPatchCoreEngine so it can
    be used as a drop-in replacement in the plugin.
    """

    def __init__(self, model_dir: str) -> None:
        self.model_dir = model_dir
        self.cls_models: dict[str, StructuralClassModel] = {}
        self._load_models()

    def _load_models(self) -> None:
        """Load all structural models from model_dir/*.joblib."""
        import joblib

        model_path = Path(self.model_dir)
        for jf in sorted(model_path.glob("*.joblib")):
            try:
                data = joblib.load(jf)
                # Only load structural models
                if data.get("algo_method") != "structural":
                    continue
                cls = data["cls"]
                # Build mean_binary from mean template if not stored
                if "mean_binary" in data:
                    mean_binary = data["mean_binary"]
                else:
                    mt = np.clip(data["mean_template"], 0, 255).astype(np.uint8)
                    _, mean_binary = cv2.threshold(
                        mt, 0, 255,
                        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
                    )
                m = StructuralClassModel(
                    cls=cls,
                    img_size=int(data.get("img_size", 128)),
                    mean_template=data["mean_template"],
                    std_template=data["std_template"],
                    foreground_mask=data["foreground_mask"],
                    mean_binary=mean_binary,
                    thr=float(data.get("thr", 0.0)),
                    n_ok=int(data.get("n_ok", 0)),
                )
                self.cls_models[cls] = m
                logger.info(
                    "Loaded structural model: class=%s, thr=%.6f, n_ok=%d",
                    cls, m.thr, m.n_ok,
                )
            except Exception:
                logger.exception("Failed to load structural model: %s", jf)

    def unload(self) -> None:
        """Release model resources."""
        self.cls_models.clear()

    def predict(
        self,
        image_path: str,
        json_path: str,
        output_overlay: str | None = None,
        pad: int = 2,
        thr_global: float | None = None,
    ) -> dict[str, Any]:
        """Run structural matching inference on a single image.

        Same interface as GlyphPatchCoreEngine.predict().
        """
        t0 = time.perf_counter()

        # Read image
        img_bgr = imread_any_bgr(image_path)
        if img_bgr is None:
            return {
                "score": 0.0, "threshold": 0.0, "pred": "OK",
                "regions": [], "artifacts": {},
                "timing_ms": {}, "glyph_total": 0,
                "ng_count": 0, "unk_count": 0,
            }

        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        ih, iw = img_gray.shape[:2]

        # Read JSON annotation (utf-8-sig handles BOM if present)
        with open(json_path, "r", encoding="utf-8-sig") as f:
            ann = json.load(f)
        items = ann.get("items", [])

        overlay = img_bgr.copy() if output_overlay else None
        regions: list[dict[str, Any]] = []
        all_scores: list[float] = []
        ng_count = 0
        unk_count = 0
        max_score = 0.0
        max_thr = 0.0

        for item in items:
            ch = str(item.get("ch", ""))
            cx = int(item.get("cx", 0))
            cy = int(item.get("cy", 0))
            w = int(item.get("w", 0))
            h = int(item.get("h", 0))

            # Convert center+size to corner coords, then clip
            bx0 = cx - w // 2 - pad
            by0 = cy - h // 2 - pad
            bx1 = cx + w // 2 + pad
            by1 = cy + h // 2 + pad
            x1, y1, x2, y2 = clip_box(bx0, by0, bx1, by1, iw, ih)
            crop_gray = img_gray[y1:y2, x1:x2]

            if crop_gray.size == 0:
                continue

            cls_safe = safe_folder_name(ch)
            model = self.cls_models.get(cls_safe)

            if model is None:
                # Unknown class
                unk_count += 1
                regions.append({
                    "ch": ch, "cx": cx, "cy": cy, "w": w, "h": h,
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "score": 0.0, "threshold": 0.0, "pred": "UNK",
                })
                if overlay is not None:
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (200, 200, 0), 1)
                    cv2.putText(overlay, f"{ch}?", (x1, y1 - 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 0), 1)
                continue

            # Score using matchTemplate alignment + local SSIM (v5)
            score = structural_defect_score(
                crop_gray,
                model.mean_template,
                model.foreground_mask,
                model.mean_binary,
                img_size=model.img_size,
            )

            thr = thr_global if thr_global is not None else model.thr
            pred = "NG" if score > thr else "OK"
            if pred == "NG":
                ng_count += 1

            all_scores.append(score)
            max_score = max(max_score, score)
            max_thr = max(max_thr, thr)

            regions.append({
                "ch": ch, "cx": cx, "cy": cy, "w": w, "h": h,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "score": round(score, 6), "threshold": round(thr, 6),
                "pred": pred,
            })

            # Draw overlay
            if overlay is not None:
                color = (0, 0, 255) if pred == "NG" else (0, 200, 0)
                thickness = 2 if pred == "NG" else 1
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness)
                label = f"{ch}:{score:.4f}"
                cv2.putText(overlay, label, (x1, y1 - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

        # Save overlay
        artifacts: dict[str, str] = {}
        if overlay is not None and output_overlay:
            imwrite_any(Path(output_overlay), overlay)
            artifacts["overlay"] = output_overlay

        t_total = (time.perf_counter() - t0) * 1000

        overall_pred = "NG" if ng_count > 0 else "OK"
        return {
            "score": round(max_score, 6),
            "threshold": round(max_thr, 6),
            "pred": overall_pred,
            "regions": regions,
            "artifacts": artifacts,
            "timing_ms": {"infer": round(t_total, 2)},
            "glyph_total": len(items),
            "ng_count": ng_count,
            "unk_count": unk_count,
        }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_glyph_structural(
    bank_dir: str,
    out_model_dir: str,
    img_size: int = 128,
    essential_thr: float = 0.5,
    score_percentile: float = 95.0,
    p_thr: float = 0.97,
    min_per_class: int = 10,
    progress_cb: Any = None,
) -> dict[str, Any]:
    """Train structural template models (v5.1 -- refined training).

    For each character class:
    1. Load all OK images, CLAHE normalise, resize, centroid-align.
    2. Compute initial pixel-wise mean template.
    3. **Iterative refinement** (2 rounds): re-align every training
       image to the current mean via matchTemplate, then recompute
       the mean.  This eliminates centroid-alignment jitter and
       produces a sharper mean with tighter OK score distribution.
    4. Compute foreground mask from binary probability map.
    5. Score each OK sample using the *inference* pipeline
       (matchTemplate alignment + SSIM) -> set threshold at p_thr.
    6. Save .joblib.

    Args:
        bank_dir: Glyph bank directory (one subdirectory per class).
        out_model_dir: Output directory for model files.
        img_size: Resize each glyph crop to this square size.
        essential_thr: Probability threshold for foreground mask (0..1).
            Pixels with binary probability > essential_thr are considered
            part of the character.  Default 0.5.
        score_percentile: (Kept for API compatibility, not used in v5.1.)
        p_thr: Quantile of OK score distribution for threshold.
            Default 0.97 (more sensitive to subtle defects).
        min_per_class: Minimum OK images per class to train.
        progress_cb: Optional callback(progress_pct, message).

    Returns:
        Training report dict.
    """
    import joblib

    bank_path = Path(bank_dir)
    out_path = Path(out_model_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    class_dirs = [p for p in bank_path.iterdir() if p.is_dir()]
    if not class_dirs:
        raise RuntimeError(f"No class folders found in {bank_dir}")

    index: dict[str, Any] = {
        "algo_method": "structural",
        "model_version": "v5.1_refined",
        "img_size": img_size,
        "essential_thr": essential_thr,
        "p_thr": p_thr,
        "classes": [],
    }

    total_classes = len(class_dirs)
    trained_classes = 0

    for ci, cls_dir in enumerate(sorted(class_dirs, key=lambda p: p.name)):
        cls = cls_dir.name
        imgs = list_images(cls_dir)

        if len(imgs) < min_per_class:
            logger.warning(
                "Skip class %s: %d images < min %d", cls, len(imgs), min_per_class
            )
            continue

        if progress_cb:
            pct = 10.0 + 80.0 * ci / total_classes
            progress_cb(pct, f"Training class {cls} ({len(imgs)} images)...")

        # Step 1: Load all raw images once (avoids re-reading from disk)
        raw_images: list[np.ndarray] = []
        for p in imgs:
            g = imread_any_gray(p)
            if g is not None:
                raw_images.append(g)

        if len(raw_images) < min_per_class:
            logger.warning(
                "Skip class %s: only %d valid images", cls, len(raw_images)
            )
            continue

        # Step 2a: Initial mean via centroid alignment
        grays = [_prepare_grayscale(g, img_size) for g in raw_images]
        binaries = [_prepare_binary(g, img_size) for g in raw_images]

        stack = np.stack(grays, axis=0)  # (N, H, W) float32
        mean_template = np.mean(stack, axis=0).astype(np.float32)

        # Step 2b: Iterative matchTemplate refinement (2 rounds)
        # Re-align each training image to the current mean using
        # matchTemplate (same alignment used at inference).  This
        # eliminates centroid-alignment jitter and produces a sharper
        # mean template with a tighter OK score distribution.
        n_refine_iters = 2
        for refine_iter in range(n_refine_iters):
            refined: list[np.ndarray] = []
            for g in raw_images:
                aligned, _mv = _tmpl_align(
                    g, mean_template, img_size, SEARCH_PAD,
                )
                refined.append(aligned)
            stack = np.stack(refined, axis=0)
            mean_template = np.mean(stack, axis=0).astype(np.float32)
            logger.info(
                "Class %s: refinement iter %d/%d, mean_std=%.2f",
                cls, refine_iter + 1, n_refine_iters,
                float(stack.std(0).mean()),
            )

        std_template = np.std(stack, axis=0).astype(np.float32)

        # Step 3: Compute foreground mask from binary probability map
        prob_map = np.mean(binaries, axis=0).astype(np.float32)
        foreground_mask = (prob_map > essential_thr).astype(np.float32)
        fg_pixel_count = int(foreground_mask.sum())

        # Step 4: Binarise mean template for backward compat
        mean_u8 = np.clip(mean_template, 0, 255).astype(np.uint8)
        _, mean_binary = cv2.threshold(
            mean_u8, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
        )

        logger.info(
            "Class %s: templates built from %d images (refined), "
            "foreground pixels=%d, mean_std=%.2f",
            cls, len(raw_images), fg_pixel_count,
            float(std_template[foreground_mask > 0].mean()) if fg_pixel_count > 0 else 0.0,
        )

        # Step 5: Compute OK defect scores for threshold calibration.
        # Score each image through the full inference pipeline
        # (matchTemplate + SSIM) against the REFINED mean template.
        ok_scores: list[float] = []
        for g in raw_images:
            s = structural_defect_score(
                g, mean_template, foreground_mask, mean_binary,
                img_size=img_size,
            )
            ok_scores.append(s)

        thr = float(np.quantile(ok_scores, p_thr))

        # Score statistics for logging
        mean_score = float(np.mean(ok_scores))
        std_score = float(np.std(ok_scores))
        max_score = float(np.max(ok_scores))

        logger.info(
            "Class %s: n_ok=%d, thr=%.6f, ok_mean=%.6f, ok_std=%.6f, "
            "ok_max=%.6f, fg_pixels=%d",
            cls, len(grays), thr, mean_score, std_score, max_score,
            fg_pixel_count,
        )

        # Step 6: Save model
        model_data = {
            "algo_method": "structural",
            "model_version": "v5.1_refined",
            "cls": cls,
            "img_size": img_size,
            "mean_template": mean_template,
            "std_template": std_template,
            "foreground_mask": foreground_mask,
            "mean_binary": mean_binary,
            "essential_thr": essential_thr,
            "thr": thr,
            "p_thr": p_thr,
            "n_ok": len(grays),
            "ok_score_mean": mean_score,
            "ok_score_std": std_score,
            "ok_score_max": max_score,
        }
        joblib.dump(model_data, out_path / f"{cls}.joblib")

        index["classes"].append({
            "cls": cls,
            "thr": thr,
            "n_ok": len(grays),
            "ok_score_mean": round(mean_score, 6),
            "ok_score_std": round(std_score, 6),
            "ok_score_max": round(max_score, 6),
            "fg_pixels": fg_pixel_count,
        })
        trained_classes += 1

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

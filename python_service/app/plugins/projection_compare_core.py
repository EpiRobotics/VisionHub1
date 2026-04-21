"""Projection Compare V1 core algorithm implementation.

Column-projection-based adjacent-strip comparison for detecting alignment
deviations in periodic structures (e.g., motor iron core lamination grooves).

Key idea:
- Slice image into tooth-sized strips along the long axis
- For each strip, compute a 1-D brightness profile by averaging pixel
  intensities perpendicular to the slicing axis (column projection)
- Compare adjacent strips' projection curves; a shift in peak/valley
  positions indicates physical misalignment
- No neural network needed -- pure signal processing

Also includes metal-mask-based boundary detection (V3):
- Build a metalness score map from local texture/edge energy
- Threshold + morphological cleanup to get metal mask
- Keep only border-connected metal regions (left/right)
- Extract inner boundaries xL/xR from metal masks
- Build low-frequency reference boundaries
- Compute residuals and classify anomaly events (intrude / missing_metal)

Reuses slicing utilities from patchcore_strip_core.
"""

from __future__ import annotations

import logging
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter

from app.plugins.patchcore_strip_core import (
    detect_long_axis,
    slice_image_into_strips,
)

logger = logging.getLogger(__name__)

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


# ---------------------------------------------------------------------------
# Projection helpers
# ---------------------------------------------------------------------------


def strip_to_gray(strip_img: np.ndarray) -> np.ndarray:
    """Convert strip image to float32 grayscale in [0, 1].

    Args:
        strip_img: (H, W, 3) uint8 RGB array.

    Returns:
        (H, W) float32 grayscale array.
    """
    # Standard luminance weights
    r, g, b = strip_img[..., 0], strip_img[..., 1], strip_img[..., 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray.astype(np.float32) / 255.0


def _gaussian_kernel_1d(size: int, sigma: float | None = None) -> np.ndarray:
    """Create a 1-D Gaussian kernel."""
    if sigma is None:
        sigma = 0.3 * ((size - 1) * 0.5 - 1) + 0.8
    x = np.arange(size, dtype=np.float32) - (size - 1) / 2.0
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    return kernel / kernel.sum()


def _auto_binary_threshold(gray: np.ndarray) -> float:
    """Compute a binary threshold to separate dark groove from bright iron.

    Uses Otsu-style between-class variance maximisation on the image
    histogram.  This is colour-agnostic: any non-black pixel (regardless
    of heat-treatment hue) ends up above the threshold.
    """
    # Build 256-bin histogram
    hist = np.bincount(gray.ravel().astype(np.int32), minlength=256).astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 128.0

    sum_all = np.dot(np.arange(256, dtype=np.float64), hist)
    sum_bg = 0.0
    w_bg = 0.0
    best_thr = 0.0
    best_var = -1.0

    for t in range(256):
        w_bg += hist[t]
        if w_bg == 0:
            continue
        w_fg = total - w_bg
        if w_fg == 0:
            break
        sum_bg += t * hist[t]
        mean_bg = sum_bg / w_bg
        mean_fg = (sum_all - sum_bg) / w_fg
        var_between = w_bg * w_fg * (mean_bg - mean_fg) ** 2
        if var_between > best_var:
            best_var = var_between
            best_thr = float(t)

    return best_thr


def _local_std(gray: np.ndarray, win: int = 5) -> np.ndarray:
    """Compute local standard deviation using a uniform window.

    Pure-NumPy implementation (no scipy dependency).  Uses cumulative-sum
    based box filtering for O(1)-per-pixel cost.

    Args:
        gray: 2-D float-convertible array.
        win: Window size (must be odd).

    Returns:
        2-D float32 array of local standard deviations.
    """
    g = gray.astype(np.float64)

    # Pad with reflect so output has same size
    pad = win // 2
    gp = np.pad(g, pad, mode="reflect")

    # Integral image approach: box-filter via cumsum
    def _box_filter(arr: np.ndarray) -> np.ndarray:
        # cumsum along rows then cols, then extract window sums
        cs = np.cumsum(arr, axis=0)
        cs = np.insert(cs, 0, 0, axis=0)  # prepend zero row
        row_sum = cs[win:, :] - cs[:-win, :]
        cs2 = np.cumsum(row_sum, axis=1)
        cs2 = np.insert(cs2, 0, 0, axis=1)  # prepend zero col
        return cs2[:, win:] - cs2[:, :-win]

    n = float(win * win)
    s1 = _box_filter(gp) / n          # local mean
    s2 = _box_filter(gp ** 2) / n     # local mean of squares
    var = np.clip(s2 - s1 ** 2, 0, None)
    return np.sqrt(var).astype(np.float32)


def _row_normalize(source: np.ndarray, proj_axis: int) -> np.ndarray:
    """Per-row (or per-column) median subtraction.

    Removes slow brightness / colour drift along the projection axis by
    subtracting each row's median.  This suppresses heat-treatment colour
    bands that change gradually along the strip height while preserving
    the original value scale (no MAD scaling — that was found to amplify
    noise in near-uniform rows).

    For proj_axis=0 (vertical slicing): each **row** is normalised.
    For proj_axis=1 (horizontal slicing): each **column** is normalised.
    """
    out = source.copy()
    if proj_axis == 0:
        # Subtract each row's median independently
        for y in range(out.shape[0]):
            out[y, :] = out[y, :] - np.median(out[y, :])
    else:
        # Subtract each column's median independently
        for x in range(out.shape[1]):
            out[:, x] = out[:, x] - np.median(out[:, x])
    return out


def _compute_edge_band_weight(
    source: np.ndarray,
    proj_axis: int,
    band_width: int = 8,
    bg_weight: float = 0.1,
) -> np.ndarray:
    """Build a 1-D weight profile that emphasises groove-edge columns.

    Steps:
    1. Average ``|source|`` along *proj_axis* → 1-D edge-strength profile.
    2. Find prominent peaks (= groove edge positions).
    3. Return weight vector: 1.0 within ±*band_width* of each peak,
       *bg_weight* elsewhere, with smooth cosine transitions.

    The weight has the same length as the profile dimension (width for
    vertical slicing, height for horizontal slicing).
    """
    raw = np.abs(source).mean(axis=proj_axis).astype(np.float32)
    n = len(raw)
    if n < 3:
        return np.ones(n, dtype=np.float32)

    # Smooth the reference profile lightly to stabilise peak detection
    k = min(5, n)
    if k >= 3:
        kern = _gaussian_kernel_1d(k if k % 2 == 1 else k + 1)
        raw_s = np.convolve(raw, kern, mode="same")
    else:
        raw_s = raw

    pmax = raw_s.max()
    if pmax < 1e-8:
        return np.ones(n, dtype=np.float32)

    norm = raw_s / pmax

    # Find peaks: local maxima above 30 % of global max
    min_dist = max(3, band_width // 2)
    peak_indices: list[int] = []
    for i in range(1, n - 1):
        if norm[i] >= norm[i - 1] and norm[i] >= norm[i + 1] and norm[i] > 0.3:
            peak_indices.append(i)

    # Non-maximum suppression
    if peak_indices:
        kept: list[int] = [peak_indices[0]]
        for p in peak_indices[1:]:
            if p - kept[-1] >= min_dist:
                kept.append(p)
            elif norm[p] > norm[kept[-1]]:
                kept[-1] = p
        peak_indices = kept

    if not peak_indices:
        return np.ones(n, dtype=np.float32)

    # Create weight with cosine ramp at edges of each band
    weight = np.full(n, bg_weight, dtype=np.float32)
    ramp = max(2, band_width // 3)  # transition zone
    for p in peak_indices:
        lo = max(0, p - band_width)
        hi = min(n, p + band_width + 1)
        weight[lo:hi] = 1.0
        # Smooth ramp on left side
        ramp_lo = max(0, lo - ramp)
        for r in range(ramp_lo, lo):
            t = (r - ramp_lo) / float(ramp)
            weight[r] = max(weight[r], bg_weight + (1.0 - bg_weight) * t)
        # Smooth ramp on right side
        ramp_hi = min(n, hi + ramp)
        for r in range(hi, ramp_hi):
            t = 1.0 - (r - hi) / float(ramp)
            weight[r] = max(weight[r], bg_weight + (1.0 - bg_weight) * t)

    return weight


# ---------------------------------------------------------------------------
# V2 Boundary-trace: detect left groove edge xL(y) via slot_score +
# center-outward scan + tooth-tip envelope.  Left-only — right boundary
# is unreliable on heat-treated images due to low contrast.
# ---------------------------------------------------------------------------


def _slot_score_image(
    rgb_img: np.ndarray,
    local_std_win: int = 5,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Compute per-pixel slot likelihood and find groove center.

    Uses max(R,G,B) as brightness (better separation for coloured metal)
    with per-row percentile normalisation to remove slow brightness drift.

    slot_score = 0.7 * dark_score + 0.3 * texture_score
      - dark_score = 1 - normalised brightness (groove is dark)
      - texture_score = 1 - normalised local_std (groove is smooth)

    Args:
        rgb_img: (H, W, 3) uint8 RGB image.
        local_std_win: Window size for local standard deviation.

    Returns:
        (slot_score, In_normalised, groove_center_x)
    """
    # Use max-channel for better colour vs groove separation
    max_ch = np.max(rgb_img.astype(np.float32), axis=2)
    h, w = max_ch.shape

    # Per-row percentile normalisation
    p10 = np.percentile(max_ch, 10, axis=1, keepdims=True)
    p90 = np.percentile(max_ch, 90, axis=1, keepdims=True)
    In = np.clip((max_ch - p10) / (p90 - p10 + 1e-6), 0.0, 1.0)

    # Dark score
    dark_score = 1.0 - In

    # Texture score via local standard deviation
    lstd = _local_std(In, win=local_std_win)
    lstd_max = np.percentile(lstd, 99) + 1e-6
    texture_score = 1.0 - np.clip(lstd / lstd_max, 0.0, 1.0)

    slot = (0.7 * dark_score + 0.3 * texture_score).astype(np.float32)

    # Groove center: column with highest median slot_score
    h_prof = np.median(In, axis=0)
    kern = _gaussian_kernel_1d(min(21, w // 4 * 2 + 1))
    h_smooth = np.convolve(h_prof, kern, mode="same")
    groove_center = int(np.argmin(h_smooth))

    return slot, In, groove_center


def _detect_left_boundary(
    slot_score: np.ndarray,
    groove_center: int,
    block_h: int = 8,
    slot_thr: float = 0.55,
    envelope_win: int = 9,
    median_win: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Detect the left groove boundary from slot_score image.

    For each y-block, scans from groove_center leftward until
    slot_score drops below threshold — this is where the dark+smooth
    groove ends and bright+textured metal begins.

    Then applies tooth-tip envelope extraction (sliding max) and
    median smoothing to get a stable boundary curve.

    Args:
        slot_score: 2-D float32 slot likelihood image.
        groove_center: X-pixel of groove centre.
        block_h: Height of each y-block for averaging.
        slot_thr: Slot-score threshold (pixels below this = metal).
        envelope_win: Sliding-max window for tooth-tip envelope.
        median_win: Median filter window for final smoothing.

    Returns:
        (xL, y_centers) — 1-D int arrays, one entry per block.
    """
    from scipy.ndimage import maximum_filter1d, median_filter

    h, w = slot_score.shape
    n_blocks = h // block_h

    # Block-average the slot_score for stability
    ss_blocks = (
        slot_score[: n_blocks * block_h, :]
        .reshape(n_blocks, block_h, w)
        .mean(axis=1)
    )

    # Per-block: scan from groove_center leftward.
    # Require *min_run* consecutive pixels below threshold to count
    # as a real groove→metal transition.  This avoids stopping at
    # brief slot_score dips caused by lighting artefacts at image
    # edges (e.g. bright background patches that mimic metal).
    min_run = 3
    xL_raw = np.zeros(n_blocks, dtype=np.int32)
    for b in range(n_blocks):
        row_ss = ss_blocks[b, :]
        xl = 0
        run = 0
        for x in range(groove_center, -1, -1):
            if row_ss[x] < slot_thr:
                run += 1
                if run >= min_run:
                    xl = x + min_run
                    break
            else:
                run = 0
        xL_raw[b] = xl

    # Interpolate blocks where the scan failed (xL_raw=0 means scan
    # reached column 0 without finding a sustained metal region).
    valid = xL_raw > 0
    if not np.all(valid) and np.sum(valid) >= 2:
        idx_valid = np.where(valid)[0]
        idx_invalid = np.where(~valid)[0]
        xL_raw[idx_invalid] = np.interp(
            idx_invalid,
            idx_valid,
            xL_raw[idx_valid],
        ).astype(np.int32)
    elif np.sum(valid) < 2:
        xL_raw[:] = groove_center // 2

    # Tooth-tip envelope: take the boundary closest to centre
    # (= largest xL) within a sliding window.
    # At tooth positions xL is large (tooth pushes inward);
    # at gap positions xL is small (gap extends outward).
    # Sliding MAX extracts the tooth-tip envelope.
    xL_env = maximum_filter1d(xL_raw, size=envelope_win)

    # Median smooth to remove residual jitter
    xL_smooth = median_filter(
        xL_env.astype(np.float64), size=median_win,
    ).astype(np.int32)

    y_centers = np.arange(n_blocks) * block_h + block_h // 2
    return xL_smooth, y_centers


def _fit_robust_reference(
    y_centers: np.ndarray,
    boundary: np.ndarray,
    deg: int = 3,
    n_iter: int = 3,
    residual_clip: float = 3.0,
) -> np.ndarray:
    """Fit a low-frequency robust reference curve to boundary positions.

    Uses iteratively-reweighted least squares (Huber-like): fit a low-order
    polynomial, identify outlier points (residual > clip * MAD), down-weight
    them, and refit.  This prevents real defects from pulling the reference
    curve toward them.

    The reference must be **low-frequency** — it represents the overall
    groove posture, not local detail.  A too-flexible fit would absorb
    defects and reduce δ.

    Args:
        y_centers: 1-D array of y-positions for each block.
        boundary: 1-D int array of boundary x-positions.
        deg: Polynomial degree (1-3 recommended).
        n_iter: Number of reweighting iterations.
        residual_clip: Outlier threshold in MAD units.

    Returns:
        1-D float array of reference x-positions (same length as boundary).
    """
    from numpy.polynomial import polynomial as P

    y = y_centers.astype(np.float64)
    x = boundary.astype(np.float64)
    weights = np.ones(len(x), dtype=np.float64)

    for _ in range(n_iter):
        # Weighted polynomial fit
        coeff = P.polyfit(y, x, deg, w=weights)
        x_ref = P.polyval(y, coeff)
        residual = x - x_ref
        mad = np.median(np.abs(residual)) + 1e-8
        # Down-weight outliers
        for k in range(len(residual)):
            if abs(residual[k]) > residual_clip * mad:
                weights[k] *= 0.1
            else:
                weights[k] = min(weights[k] * 1.5, 1.0)

    # Final fit with updated weights
    coeff = P.polyfit(y, x, deg, w=weights)
    return P.polyval(y, coeff).astype(np.float64)


def find_groove_boundaries(
    rgb_img: np.ndarray,
    block_h: int = 8,
    slot_thr: float = 0.55,
    envelope_win: int = 9,
    ref_deg: int = 3,
) -> dict[str, Any]:
    """Detect left groove boundary via slot_score region approach.

    Left-boundary only — the right boundary is unreliable on heat-treated
    images where dark metal has brightness close to the groove.

    Pipeline:
    1. Compute slot_score image (dark + low-texture likelihood)
    2. Find groove centre from horizontal brightness profile
    3. Per-block: scan from centre leftward until slot_score < threshold
    4. Tooth-tip envelope extraction (sliding max) + median smooth
    5. Robust polynomial reference fit (IRLS)
    6. Compute residuals dL = xL - xL_ref

    Performance: <20 ms on 250×1800 images (pure NumPy, no loops over pixels).

    Args:
        rgb_img: (H, W, 3) uint8 RGB image.
        block_h: Height of each y-block for averaging.
        slot_thr: Slot-score threshold (below = metal, above = groove).
        envelope_win: Sliding-max window for tooth-tip envelope.
        ref_deg: Polynomial degree for reference curve fitting.

    Returns:
        Dict with keys: xL, xL_ref, dL, y_centers, groove_center,
        n_blocks, block_h.
    """
    # Step 1-2: slot_score + groove centre
    slot_score, In, groove_center = _slot_score_image(rgb_img)

    # Step 3-4: left boundary detection
    xL, y_centers = _detect_left_boundary(
        slot_score, groove_center,
        block_h=block_h, slot_thr=slot_thr,
        envelope_win=envelope_win,
    )

    # Step 5: robust reference fitting
    xL_ref = _fit_robust_reference(y_centers, xL, deg=ref_deg)

    # Step 6: residuals
    dL = xL.astype(np.float64) - xL_ref

    return {
        "xL": xL,
        "xL_ref": xL_ref,
        "dL": dL,
        "y_centers": y_centers,
        "groove_center": groove_center,
        "n_blocks": len(xL),
        "block_h": block_h,
    }


def _boundary_scores_per_strip(
    boundary_result: dict[str, Any],
    strip_bboxes: list[tuple[int, int, int, int]],
    slicing_axis: str,
    skew_weight: float = 2.0,
) -> list[dict[str, float]]:
    """Compute bump_score and skew_score per strip from left-boundary residuals.

    For each strip, extracts the dL residuals within its y-range and computes:
    - bump_score = max(|dL|) — largest local deviation (pixels)
      - dL > 0 means tooth protrudes into groove
      - dL < 0 means tooth recedes
    - skew_score = |slope(dL)| * strip_height — total tilt across strip (pixels)

    Args:
        boundary_result: Output of find_groove_boundaries().
        strip_bboxes: List of (x0, y0, x1, y1) for each strip.
        slicing_axis: "vertical" or "horizontal".
        skew_weight: Multiplier for skew_score in combined score.

    Returns:
        List of dicts with bump_score, skew_score per strip.
    """
    dL = boundary_result["dL"]
    y_centers = boundary_result["y_centers"]

    scores: list[dict[str, float]] = []

    for x0, y0, x1, y1 in strip_bboxes:
        if slicing_axis == "vertical":
            strip_y0, strip_y1 = y0, y1
        else:
            strip_y0, strip_y1 = x0, x1

        # Find blocks within this strip's y-range
        mask = (y_centers >= strip_y0) & (y_centers < strip_y1)
        if not np.any(mask):
            scores.append({"bump_score": 0.0, "skew_score": 0.0})
            continue

        strip_dL = dL[mask]

        # bump_score: maximum absolute left-boundary deviation
        bump = float(np.max(np.abs(strip_dL))) if len(strip_dL) > 0 else 0.0

        # skew_score: tilt within strip (px across strip height)
        skew = 0.0
        if len(strip_dL) >= 2:
            strip_y = y_centers[mask]
            y_range = strip_y[-1] - strip_y[0]
            if y_range > 0:
                slope = float(np.polyfit(strip_y, strip_dL, 1)[0])
                skew = abs(slope * y_range)

        scores.append({
            "bump_score": round(bump, 4),
            "skew_score": round(skew, 4),
        })

    return scores


def infer_boundary_compare(
    img_path: Path,
    strip_size: int,
    strip_overlap: int,
    axis: str | None = None,
    skip_edge_strips: int = 0,
    boundary_block_h: int = 8,
    boundary_slot_thr: float = 0.55,
    boundary_envelope_win: int = 9,
    boundary_ref_deg: int = 3,
    boundary_skew_weight: float = 2.0,
) -> tuple[float, list[dict[str, Any]], tuple[int, int], list[np.ndarray], list[np.ndarray]]:
    """Run left-boundary-only comparison on a single image.

    Detects the left groove edge xL(y) via slot_score region approach,
    fits a smooth reference, and scores each strip by the left-boundary
    residual dL.  Right boundary is not used (unreliable on heat-treated
    images).

    Score = pixels of left-boundary deviation.  Colour/texture variations
    do NOT affect the score.

    Returns same format as infer_projection_pairs for compatibility:
        max_distance, pair_results, (H, W), profiles, diff_curves
    """
    with Image.open(img_path) as im:
        im = im.convert("RGB")
        np_img = np.array(im)

    h0, w0 = np_img.shape[:2]
    resolved_axis = axis if axis else detect_long_axis(h0, w0)

    # Run boundary detection on full RGB image
    boundary = find_groove_boundaries(
        np_img,
        block_h=boundary_block_h,
        slot_thr=boundary_slot_thr,
        envelope_win=boundary_envelope_win,
        ref_deg=boundary_ref_deg,
    )

    # Slice into strips (same as V1) for per-strip scoring
    strips = slice_image_into_strips(np_img, strip_size, strip_overlap, axis)
    if skip_edge_strips > 0 and len(strips) > 2 * skip_edge_strips:
        strips = strips[skip_edge_strips:-skip_edge_strips]

    bboxes = [(x0, y0, x1, y1) for _, x0, y0, x1, y1 in strips]

    # Compute per-strip boundary scores (left-only)
    strip_scores = _boundary_scores_per_strip(
        boundary, bboxes, resolved_axis,
        skew_weight=boundary_skew_weight,
    )

    # Build pair_results compatible with existing overlay format
    pair_results: list[dict[str, Any]] = []
    distances: list[float] = []
    diff_curves: list[np.ndarray] = []

    # Pseudo-profiles from |dL| for visualisation
    profiles: list[np.ndarray] = []
    for _i, (x0, y0, x1, y1) in enumerate(bboxes):
        if resolved_axis == "vertical":
            strip_y0, strip_y1 = y0, y1
        else:
            strip_y0, strip_y1 = x0, x1
        mask = (boundary["y_centers"] >= strip_y0) & (boundary["y_centers"] < strip_y1)
        if np.any(mask):
            prof = np.abs(boundary["dL"][mask])
        else:
            prof = np.zeros(1, dtype=np.float32)
        profiles.append(prof.astype(np.float32))

    for i in range(len(bboxes) - 1):
        # Pair score = max of both strips' bump/skew
        score_a = max(
            strip_scores[i]["bump_score"],
            strip_scores[i]["skew_score"] * boundary_skew_weight,
        )
        score_b = max(
            strip_scores[i + 1]["bump_score"],
            strip_scores[i + 1]["skew_score"] * boundary_skew_weight,
        )
        pair_dist = max(score_a, score_b)
        distances.append(pair_dist)

        x0_a, y0_a, x1_a, y1_a = bboxes[i]
        x0_b, y0_b, x1_b, y1_b = bboxes[i + 1]

        pair_results.append({
            "pair_idx": i,
            "strip_a": i,
            "strip_b": i + 1,
            "distance": round(float(pair_dist), 6),
            "seg_distances": [
                round(strip_scores[i]["bump_score"], 6),
                round(strip_scores[i]["skew_score"], 6),
                round(strip_scores[i + 1]["bump_score"], 6),
                round(strip_scores[i + 1]["skew_score"], 6),
            ],
            "skew_score": round(float(max(
                strip_scores[i]["skew_score"],
                strip_scores[i + 1]["skew_score"],
            )), 6),
            "bump_score": round(float(max(
                strip_scores[i]["bump_score"],
                strip_scores[i + 1]["bump_score"],
            )), 6),
            "bbox_a": {"x": x0_a, "y": y0_a, "w": x1_a - x0_a, "h": y1_a - y0_a},
            "bbox_b": {"x": x0_b, "y": y0_b, "w": x1_b - x0_b, "h": y1_b - y0_b},
        })

        # Diff curve: |dL| in the pair's y-range
        if resolved_axis == "vertical":
            y_lo = min(y0_a, y0_b)
            y_hi = max(y1_a, y1_b)
        else:
            y_lo = min(x0_a, x0_b)
            y_hi = max(x1_a, x1_b)
        mask = (boundary["y_centers"] >= y_lo) & (boundary["y_centers"] < y_hi)
        if np.any(mask):
            dc = np.abs(boundary["dL"][mask]).astype(np.float32)
        else:
            dc = np.zeros(1, dtype=np.float32)
        diff_curves.append(dc)

    max_distance = max(distances) if distances else 0.0
    return max_distance, pair_results, (h0, w0), profiles, diff_curves


def compute_ok_boundary_distances(
    img_paths: list[Path],
    strip_size: int,
    strip_overlap: int,
    axis: str | None = None,
    skip_edge_strips: int = 0,
    boundary_block_h: int = 8,
    boundary_slot_thr: float = 0.55,
    boundary_envelope_win: int = 9,
    boundary_ref_deg: int = 3,
    boundary_skew_weight: float = 2.0,
    progress_cb: Any = None,
) -> list[float]:
    """Compute all left-boundary scores from OK images for threshold estimation.

    Returns:
        List of per-pair boundary scores from all OK images.
    """
    all_distances: list[float] = []

    for idx, img_path in enumerate(img_paths):
        _max_dist, pair_results, *_ = infer_boundary_compare(
            img_path=img_path,
            strip_size=strip_size,
            strip_overlap=strip_overlap,
            axis=axis,
            skip_edge_strips=skip_edge_strips,
            boundary_block_h=boundary_block_h,
            boundary_slot_thr=boundary_slot_thr,
            boundary_envelope_win=boundary_envelope_win,
            boundary_ref_deg=boundary_ref_deg,
            boundary_skew_weight=boundary_skew_weight,
        )
        for pr in pair_results:
            all_distances.append(pr["distance"])

        if progress_cb and len(img_paths) > 1:
            pct = 30.0 + (idx + 1) / len(img_paths) * 50.0
            progress_cb(
                pct,
                f"OK boundary distances: {idx + 1}/{len(img_paths)} images, "
                f"{len(all_distances)} pairs",
            )

    return all_distances


def column_projection(
    strip_img: np.ndarray,
    slicing_axis: str,
    smooth_kernel: int = 5,
    trim_pct: float = 0.05,
    smooth_type: str = "gaussian",
    projection_type: str = "mean",
    binary_threshold: int = 50,
    edge_band_width: int = 0,
    row_normalize: bool = False,
) -> np.ndarray:
    """Compute 1-D brightness profile perpendicular to the slicing axis.

    For vertical slicing (strips stacked top-to-bottom), projects along
    the height (rows) → result has length = strip width.
    For horizontal slicing, projects along the width (columns) → length = strip height.

    Args:
        strip_img: (H, W, 3) uint8 RGB array.
        slicing_axis: "vertical" or "horizontal".
        smooth_kernel: Smoothing kernel size (must be odd, 0 to skip).
        trim_pct: Fraction of extreme pixels to discard from each end
            before averaging (trimmed mean).  0 = plain mean.
            E.g. 0.05 removes brightest/darkest 5 % of rows/cols.
        smooth_type: "gaussian" or "boxcar".
        projection_type: Projection mode.
            "mean": brightness average (original).
            "gradient": Sobel edge-magnitude average (sensitive to edges).
            "binary_edge": binarise (dark=0, non-dark=1) then gradient.
            "texture": local-std-dev texture map, then gradient.
                Teeth have high texture, groove is smooth → texture
                boundary = groove edge.  Immune to colour AND shadows.
                RECOMMENDED for heat-treated parts with uneven groove
                backgrounds.
        binary_threshold: Fixed grayscale threshold for binary_edge mode.
            Pixels <= threshold → 0 (groove/dark), > threshold → 1 (metal).
            Set to 0 to use automatic Otsu threshold instead.
        edge_band_width: Half-width (pixels) of the groove-edge band to
            emphasise.  0 = disabled (uniform weighting, V1 behaviour).
            When > 0, auto-detects groove edge positions from the
            gradient peak profile and weights edge regions ×1.0 while
            suppressing interior/groove-centre regions to ×0.1.
            Recommended: 6–10 for typical strip widths.
        row_normalize: If True, apply per-row median/MAD normalisation
            before projection.  Removes slow colour / brightness drift
            along the strip height (heat-treatment bands, uneven
            illumination).  Recommended when colour variation causes
            false positives.

    Returns:
        1-D float32 array — the brightness profile.
    """
    gray = strip_to_gray(strip_img)

    if slicing_axis == "vertical":
        # Project along rows → profile has length = width
        proj_axis = 0  # average over rows
        grad_axis = 1  # gradient along width (perpendicular to projection)
    else:
        # Project along columns → profile has length = height
        proj_axis = 1  # average over columns
        grad_axis = 0  # gradient along height

    # Choose source data based on projection type
    if projection_type == "texture":
        # Local standard deviation → texture map.
        # Teeth = high texture (rough metal), groove = low texture (smooth).
        # Completely independent of colour and absolute brightness.
        texture_map = _local_std(gray, win=5)
        # Compute gradient on texture map → peaks at groove boundary
        grad = np.gradient(texture_map, axis=grad_axis)
        source = np.abs(grad).astype(np.float32)
    elif projection_type == "binary_edge":
        # Binarise: dark (groove) → 0, non-dark (iron) → 1
        # Then compute gradient of binary → only groove boundaries respond.
        # Completely immune to heat-treatment colour variation because any
        # non-black pixel is treated identically.
        if binary_threshold > 0:
            thr_val = float(binary_threshold)
        else:
            thr_val = _auto_binary_threshold(gray)
        binary = (gray > thr_val).astype(np.float32)
        n_metal = int(binary.sum())
        n_total = binary.size
        logger.debug(
            "binary_edge: thr=%.1f, metal_pct=%.1f%%, shape=%s",
            thr_val, 100.0 * n_metal / max(n_total, 1), gray.shape,
        )
        grad = np.gradient(binary, axis=grad_axis)
        source = np.abs(grad).astype(np.float32)
    elif projection_type == "gradient":
        # Sobel-like gradient along the perpendicular axis
        # This emphasises edge positions rather than bulk brightness
        grad = np.gradient(gray, axis=grad_axis)
        source = np.abs(grad).astype(np.float32)
    else:
        source = gray

    # --- V1.5 enhancement ② : per-row normalisation ---
    # Removes slow colour / brightness drift along the strip height.
    if row_normalize:
        source = _row_normalize(source, proj_axis)
        # After normalisation values can be negative; take abs so that
        # the subsequent mean/trimmed-mean stays meaningful.
        source = np.abs(source)

    # --- V1.5 enhancement ① : edge-band weighting ---
    # Suppress interior / groove-centre regions; emphasise groove edges.
    if edge_band_width > 0:
        edge_w = _compute_edge_band_weight(
            source, proj_axis, band_width=edge_band_width,
        )
        if proj_axis == 0:
            source = source * edge_w[np.newaxis, :]   # broadcast (H,) * (1, W)
        else:
            source = source * edge_w[:, np.newaxis]   # broadcast (H, 1) * (H, W)

    # For binary_edge the source is already clean (only 0/1 gradient);
    # trimmed-mean would discard the sparse edge pixels as "outliers",
    # producing an all-zero profile.  Always use plain mean for it.
    effective_trim = 0.0 if projection_type == "binary_edge" else trim_pct

    if effective_trim > 0:
        # Trimmed mean: sort along projection axis, discard extremes
        n = source.shape[proj_axis]
        lo = max(1, int(n * effective_trim))
        hi = n - lo
        sorted_vals = np.sort(source, axis=proj_axis)
        if proj_axis == 0:
            profile = sorted_vals[lo:hi, :].mean(axis=0)
        else:
            profile = sorted_vals[:, lo:hi].mean(axis=1)
    else:
        profile = source.mean(axis=proj_axis)

    # Smooth
    if smooth_kernel > 1:
        if smooth_type == "gaussian":
            kernel = _gaussian_kernel_1d(smooth_kernel)
        else:
            kernel = np.ones(smooth_kernel, dtype=np.float32) / smooth_kernel
        profile = np.convolve(profile, kernel, mode="same")

    return profile.astype(np.float32)


def normalise_profile(profile: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-variance normalisation of a 1-D profile.

    Returns zeros if std is near-zero (flat region).
    """
    std = profile.std()
    if std < 1e-8:
        return np.zeros_like(profile)
    return (profile - profile.mean()) / std


# ---------------------------------------------------------------------------
# Distance metrics between adjacent profiles
# ---------------------------------------------------------------------------


def _resample_to_same_length(
    p1: np.ndarray, p2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample two profiles to the same length if they differ."""
    if len(p1) != len(p2):
        target_len = max(len(p1), len(p2))
        p1 = np.interp(np.linspace(0, 1, target_len), np.linspace(0, 1, len(p1)), p1)
        p2 = np.interp(np.linspace(0, 1, target_len), np.linspace(0, 1, len(p2)), p2)
    return p1, p2


def _align_profiles(
    p1: np.ndarray,
    p2: np.ndarray,
    max_shift: int = 0,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Align p2 to p1 via cross-correlation, then trim edges.

    Args:
        p1, p2: normalised profiles of the same length.
        max_shift: maximum allowed shift in pixels.  0 = no alignment.

    Returns:
        (p1_trimmed, p2_shifted_trimmed, shift) where shift is the
        applied offset (positive = p2 shifted right).
    """
    if max_shift <= 0 or len(p1) < 3:
        return p1, p2, 0

    n = len(p1)
    corr = np.correlate(p1, p2, mode="full")
    # corr has length 2*n-1, peak at n-1 means 0 shift
    center = n - 1
    lo = max(0, center - max_shift)
    hi = min(len(corr), center + max_shift + 1)
    best_idx = lo + int(np.argmax(corr[lo:hi]))
    shift = best_idx - center  # positive = p2 needs to shift right

    if shift == 0:
        return p1, p2, 0

    # Apply shift by trimming both arrays
    abs_s = abs(shift)
    if shift > 0:
        # p2 shifted right → trim p2 start, p1 end
        p1_out = p1[:n - abs_s]
        p2_out = p2[abs_s:]
    else:
        # p2 shifted left → trim p1 start, p2 end
        p1_out = p1[abs_s:]
        p2_out = p2[:n - abs_s]

    return p1_out, p2_out, shift


def profile_l2_distance(
    p1: np.ndarray,
    p2: np.ndarray,
    align_max_shift: int = 0,
) -> float:
    """L2 distance between two profiles (after optional alignment).

    Args:
        p1, p2: raw profiles.
        align_max_shift: max pixels to shift for alignment (0 = off).
    """
    p1n = normalise_profile(p1)
    p2n = normalise_profile(p2)
    p1n, p2n = _resample_to_same_length(p1n, p2n)

    if align_max_shift > 0:
        p1n, p2n, _ = _align_profiles(p1n, p2n, align_max_shift)

    return float(np.sqrt(np.mean((p1n - p2n) ** 2)))


def profile_correlation_distance(
    p1: np.ndarray,
    p2: np.ndarray,
    align_max_shift: int = 0,
) -> float:
    """1 - Pearson correlation between normalised profiles.

    Returns value in [0, 2]; 0 = identical, 2 = inverted.
    """
    p1n = normalise_profile(p1)
    p2n = normalise_profile(p2)
    p1n, p2n = _resample_to_same_length(p1n, p2n)

    if align_max_shift > 0:
        p1n, p2n, _ = _align_profiles(p1n, p2n, align_max_shift)

    dot = np.dot(p1n, p2n)
    norm = np.sqrt(np.dot(p1n, p1n) * np.dot(p2n, p2n))
    if norm < 1e-12:
        return 0.0
    corr = dot / norm
    return float(1.0 - corr)


def _find_peaks(
    profile: np.ndarray,
    min_height: float = 0.0,
    min_distance: int = 3,
) -> np.ndarray:
    """Find local maxima in a 1-D array (pure NumPy, no scipy).

    A sample is a peak if it is strictly greater than both its immediate
    neighbours and its value >= *min_height*.  When multiple peaks are
    closer than *min_distance*, keep the tallest.

    Args:
        profile: 1-D float array.
        min_height: Minimum absolute value to qualify as a peak.
        min_distance: Minimum index distance between retained peaks.

    Returns:
        Sorted integer array of peak indices.
    """
    if len(profile) < 3:
        return np.array([], dtype=np.intp)

    # Local-max condition: strictly greater than both neighbours
    left = profile[:-2]
    center = profile[1:-1]
    right = profile[2:]
    is_peak = (center > left) & (center > right)
    candidates = np.nonzero(is_peak)[0] + 1  # +1 to recover original index

    # Height filter
    if min_height > 0:
        candidates = candidates[profile[candidates] >= min_height]

    if len(candidates) == 0:
        return np.array([], dtype=np.intp)

    # Enforce minimum distance: greedily keep tallest first
    order = np.argsort(-profile[candidates])  # descending by height
    keep: list[int] = []
    suppressed = np.zeros(len(profile), dtype=bool)
    for idx in candidates[order]:
        if suppressed[idx]:
            continue
        keep.append(int(idx))
        lo = max(0, idx - min_distance)
        hi = min(len(profile), idx + min_distance + 1)
        suppressed[lo:hi] = True

    return np.sort(np.array(keep, dtype=np.intp))


def _match_peaks(
    peaks_a: np.ndarray,
    peaks_b: np.ndarray,
    max_gap: int = 15,
) -> list[tuple[int, int]]:
    """Greedily match peaks between two profiles by proximity.

    For each peak in *peaks_a*, find the closest unmatched peak in *peaks_b*
    within *max_gap* pixels.  Returns list of (idx_a, idx_b) pairs.
    """
    if len(peaks_a) == 0 or len(peaks_b) == 0:
        return []

    used_b = set[int]()
    pairs: list[tuple[int, int]] = []

    for pa in peaks_a:
        dists = np.abs(peaks_b.astype(np.int64) - int(pa))
        order = np.argsort(dists)
        for oi in order:
            if int(dists[oi]) > max_gap:
                break
            if oi not in used_b:
                pairs.append((int(pa), int(peaks_b[oi])))
                used_b.add(oi)
                break

    return pairs


def profile_edge_shift(
    p1: np.ndarray,
    p2: np.ndarray,
    align_max_shift: int = 0,  # unused, kept for API consistency
    min_peak_height_pct: float = 0.3,
    min_peak_distance: int = 3,
    max_match_gap: int = 15,
) -> float:
    """Max groove-edge shift (pixels) between two projection profiles.

    Finds prominent peaks in each *raw* (un-normalised) profile — these
    correspond to groove boundaries.  Matches peaks by proximity between
    the two profiles.  Returns the **maximum absolute position difference**
    among matched peaks, in pixels.

    If no peaks can be matched the function falls back to
    :func:`profile_peak_shift` (whole-profile cross-correlation).

    The score has direct physical meaning: 0 = edges perfectly aligned,
    N = the worst edge shifted by N pixels.
    """
    p1r, p2r = _resample_to_same_length(p1, p2)

    # Height threshold: fraction of max peak height in each profile
    thr1 = float(np.max(np.abs(p1r))) * min_peak_height_pct
    thr2 = float(np.max(np.abs(p2r))) * min_peak_height_pct
    min_h = min(thr1, thr2)

    peaks1 = _find_peaks(p1r, min_height=min_h, min_distance=min_peak_distance)
    peaks2 = _find_peaks(p2r, min_height=min_h, min_distance=min_peak_distance)

    matched = _match_peaks(peaks1, peaks2, max_gap=max_match_gap)

    if not matched:
        # Fallback: whole-profile cross-correlation shift
        return profile_peak_shift(p1, p2, align_max_shift)

    shifts = [abs(a - b) for a, b in matched]
    return float(max(shifts))


def profile_peak_shift(
    p1: np.ndarray,
    p2: np.ndarray,
    align_max_shift: int = 0,  # unused, kept for API consistency
) -> float:
    """Cross-correlation based peak shift (in pixels) between two profiles.

    Uses normalised cross-correlation to find the offset that maximises
    alignment.  Returns the absolute shift in pixels.

    Also works as a distance: 0 = perfectly aligned, larger = more shift.
    """
    p1n = normalise_profile(p1)
    p2n = normalise_profile(p2)
    p1n, p2n = _resample_to_same_length(p1n, p2n)

    n = len(p1n)
    if n == 0:
        return 0.0

    # Full cross-correlation
    corr = np.correlate(p1n, p2n, mode="full")
    # corr has length 2*n - 1, peak at index n-1 means zero shift
    peak_idx = int(np.argmax(corr))
    shift = abs(peak_idx - (n - 1))
    return float(shift)


def compute_pair_distance(
    p1: np.ndarray,
    p2: np.ndarray,
    metric: str = "l2",
    align_max_shift: int = 0,
) -> float:
    """Compute distance between two projection profiles.

    Args:
        p1, p2: 1-D float32 profiles.
        metric: "l2", "correlation", "peak_shift", or "edge_shift".
        align_max_shift: max pixels to pre-align before distance (0 = off).

    Returns:
        Scalar distance.
    """
    if metric == "correlation":
        return profile_correlation_distance(p1, p2, align_max_shift)
    if metric == "peak_shift":
        return profile_peak_shift(p1, p2, align_max_shift)
    if metric == "edge_shift":
        return profile_edge_shift(p1, p2, align_max_shift)
    return profile_l2_distance(p1, p2, align_max_shift)


# ---------------------------------------------------------------------------
# Per-position diff curve for heatmap-like visualisation
# ---------------------------------------------------------------------------


def compute_profile_diff_curve(
    p1: np.ndarray,
    p2: np.ndarray,
    align_max_shift: int = 0,
) -> np.ndarray:
    """Point-wise absolute difference between two normalised profiles.

    Returns a 1-D curve of the same length showing *where* the profiles
    differ most — analogous to a 1-D "heatmap".
    """
    p1n = normalise_profile(p1)
    p2n = normalise_profile(p2)
    p1n, p2n = _resample_to_same_length(p1n, p2n)

    if align_max_shift > 0:
        p1n, p2n, _ = _align_profiles(p1n, p2n, align_max_shift)

    return np.abs(p1n - p2n).astype(np.float32)


# ---------------------------------------------------------------------------
# Sub-segment splitting for V1.5 skew detection
# ---------------------------------------------------------------------------


def _split_strip_into_segments(
    strip_img: np.ndarray,
    slicing_axis: str,
    n_segments: int,
) -> list[np.ndarray]:
    """Split a strip image into *n_segments* sub-strips along the slicing axis.

    For vertical slicing (strips stacked top-to-bottom), the strip is split
    along its height (rows).  For horizontal slicing, along its width (cols).

    Returns:
        List of sub-strip images (same dtype as input).
    """
    if n_segments <= 1:
        return [strip_img]

    if slicing_axis == "vertical":
        total = strip_img.shape[0]  # height
        seg_size = total // n_segments
        segments = []
        for s in range(n_segments):
            start = s * seg_size
            end = start + seg_size if s < n_segments - 1 else total
            segments.append(strip_img[start:end])
    else:
        total = strip_img.shape[1]  # width
        seg_size = total // n_segments
        segments = []
        for s in range(n_segments):
            start = s * seg_size
            end = start + seg_size if s < n_segments - 1 else total
            segments.append(strip_img[:, start:end])

    return segments


# ---------------------------------------------------------------------------
# Full-image inference
# ---------------------------------------------------------------------------


def infer_projection_pairs(
    img_path: Path,
    strip_size: int,
    strip_overlap: int,
    metric: str = "l2",
    axis: str | None = None,
    smooth_kernel: int = 5,
    trim_pct: float = 0.05,
    smooth_type: str = "gaussian",
    align_max_shift: int = 0,
    skip_edge_strips: int = 0,
    projection_type: str = "mean",
    binary_threshold: int = 50,
    n_segments: int = 1,
    edge_band_width: int = 0,
    row_normalize: bool = False,
) -> tuple[float, list[dict[str, Any]], tuple[int, int], list[np.ndarray], list[np.ndarray]]:
    """Run projection-based adjacent-pair comparison on a single image.

    V1.5 extension: when *n_segments* > 1, each strip is further split into
    sub-segments along the slicing axis.  The pair distance becomes the **max**
    across all sub-segment distances, which makes the algorithm sensitive to
    **skew** (angular tilt) in addition to offset.

    With n_segments=1 this is identical to V1 behaviour.

    Args:
        img_path: Path to image file.
        strip_size: Strip size along the slicing axis.
        strip_overlap: Overlap between adjacent strips.
        metric: Distance metric ("l2", "correlation", "peak_shift").
        axis: Slicing axis ("vertical", "horizontal", or None=auto).
        smooth_kernel: Smoothing kernel size.
        trim_pct: Fraction of extreme pixels to discard (trimmed mean).
        smooth_type: "gaussian" or "boxcar".
        align_max_shift: Max pixels to pre-align profiles (0=off).
        skip_edge_strips: Skip first/last N strips (boundary artifacts).
        projection_type: "mean", "gradient", or "binary_edge".
        binary_threshold: Fixed threshold for binary_edge (0=auto Otsu).
        n_segments: Number of sub-segments per strip (1=V1, 2=V1.5 top/bottom).

    Returns:
        max_distance: Maximum distance across all adjacent pairs.
        pair_results: Per-pair result dicts.
        (H, W): Original image size.
        profiles: List of 1-D profiles, one per strip (full-strip profile).
        diff_curves: List of 1-D diff curves, one per adjacent pair.
    """
    with Image.open(img_path) as im:
        im = im.convert("RGB")
        np_img = np.array(im)

    h0, w0 = np_img.shape[:2]
    resolved_axis = axis if axis else detect_long_axis(h0, w0)

    strips = slice_image_into_strips(np_img, strip_size, strip_overlap, axis)

    # Skip edge strips if requested
    if skip_edge_strips > 0 and len(strips) > 2 * skip_edge_strips:
        strips = strips[skip_edge_strips:-skip_edge_strips]

    proj_kwargs = dict(
        slicing_axis=resolved_axis,
        smooth_kernel=smooth_kernel,
        trim_pct=trim_pct,
        smooth_type=smooth_type,
        projection_type=projection_type,
        binary_threshold=binary_threshold,
        edge_band_width=edge_band_width,
        row_normalize=row_normalize,
    )

    # Compute full-strip profiles (always, for diff heatmap / overlay)
    profiles: list[np.ndarray] = []
    bboxes: list[tuple[int, int, int, int]] = []
    # Per-strip segment profiles: seg_profiles[strip_idx][seg_idx]
    seg_profiles: list[list[np.ndarray]] = []

    for strip_img, x0, y0, x1, y1 in strips:
        prof = column_projection(strip_img, **proj_kwargs)
        profiles.append(prof)
        bboxes.append((x0, y0, x1, y1))

        if n_segments > 1:
            sub_strips = _split_strip_into_segments(
                strip_img, resolved_axis, n_segments,
            )
            seg_profiles.append([
                column_projection(sub, **proj_kwargs) for sub in sub_strips
            ])
        else:
            seg_profiles.append([prof])

    # Compute pairwise distances and diff curves
    distances: list[float] = []
    seg_distances_list: list[list[float]] = []
    diff_curves: list[np.ndarray] = []

    for i in range(len(profiles) - 1):
        # Per-segment distances
        seg_dists: list[float] = []
        for s in range(len(seg_profiles[i])):
            if s < len(seg_profiles[i + 1]):
                sd = compute_pair_distance(
                    seg_profiles[i][s], seg_profiles[i + 1][s],
                    metric, align_max_shift,
                )
                seg_dists.append(sd)

        if seg_dists:
            pair_dist = max(seg_dists)  # max across segments
        else:
            pair_dist = compute_pair_distance(
                profiles[i], profiles[i + 1], metric, align_max_shift,
            )
            seg_dists = [pair_dist]

        distances.append(pair_dist)
        seg_distances_list.append(seg_dists)

        dc = compute_profile_diff_curve(
            profiles[i], profiles[i + 1], align_max_shift,
        )
        diff_curves.append(dc)

    # Build pair results
    pair_results: list[dict[str, Any]] = []
    for i, dist in enumerate(distances):
        x0_a, y0_a, x1_a, y1_a = bboxes[i]
        x0_b, y0_b, x1_b, y1_b = bboxes[i + 1]
        seg_dists = seg_distances_list[i] if i < len(seg_distances_list) else [dist]
        skew_score = max(seg_dists) - min(seg_dists) if len(seg_dists) > 1 else 0.0
        pair_results.append({
            "pair_idx": i,
            "strip_a": i,
            "strip_b": i + 1,
            "distance": round(float(dist), 6),
            "seg_distances": [round(float(sd), 6) for sd in seg_dists],
            "skew_score": round(float(skew_score), 6),
            "bbox_a": {"x": x0_a, "y": y0_a, "w": x1_a - x0_a, "h": y1_a - y0_a},
            "bbox_b": {"x": x0_b, "y": y0_b, "w": x1_b - x0_b, "h": y1_b - y0_b},
        })

    max_distance = max(distances) if distances else 0.0
    return max_distance, pair_results, (h0, w0), profiles, diff_curves


# ---------------------------------------------------------------------------
# Diff heatmap from diff curves
# ---------------------------------------------------------------------------


def build_diff_heatmap(
    diff_curves: list[np.ndarray],
    pair_results: list[dict[str, Any]],
    img_h: int,
    img_w: int,
    slicing_axis: str,
    distance_weighted: bool = True,
) -> np.ndarray:
    """Stitch per-pair 1-D diff curves into a 2-D heatmap in image space.

    Each diff curve is broadcast along the slicing axis to cover the
    boundary zone between the two strips.

    When ``distance_weighted`` is True (default), each pair's diff curve
    is scaled by its pair distance so that low-distance pairs (normal)
    contribute very little colour while high-distance pairs dominate.
    This prevents false-positive hot spots on OK images.

    Returns:
        heatmap: (img_h, img_w) float32 array, 0-1 normalised.
    """
    heatmap = np.zeros((img_h, img_w), dtype=np.float32)
    count = np.zeros((img_h, img_w), dtype=np.float32)

    # Compute distance-based weights
    all_dists = [pr["distance"] for pr in pair_results]
    max_dist = max(all_dists) if all_dists else 1.0
    if max_dist < 1e-12:
        max_dist = 1.0

    for dc, pr in zip(diff_curves, pair_results):
        ba = pr["bbox_a"]
        bb = pr["bbox_b"]

        # Scale diff curve by pair distance (normalised to [0, 1])
        if distance_weighted:
            weight = pr["distance"] / max_dist
            # Apply power curve to suppress low-distance pairs further
            weight = weight ** 2
            dc_scaled = dc * weight
        else:
            dc_scaled = dc

        # Region covering both strips
        rx0 = min(ba["x"], bb["x"])
        ry0 = min(ba["y"], bb["y"])
        rx1 = max(ba["x"] + ba["w"], bb["x"] + bb["w"])
        ry1 = max(ba["y"] + ba["h"], bb["y"] + bb["h"])
        rh = ry1 - ry0
        rw = rx1 - rx0

        if rh <= 0 or rw <= 0:
            continue

        if slicing_axis == "vertical":
            dc_resized = np.interp(
                np.linspace(0, 1, rw), np.linspace(0, 1, len(dc_scaled)), dc_scaled,
            )
            patch = np.tile(dc_resized, (rh, 1))
        else:
            dc_resized = np.interp(
                np.linspace(0, 1, rh), np.linspace(0, 1, len(dc_scaled)), dc_scaled,
            )
            patch = np.tile(dc_resized[:, None], (1, rw))

        heatmap[ry0:ry1, rx0:rx1] += patch
        count[ry0:ry1, rx0:rx1] += 1.0

    mask = count > 0
    heatmap[mask] /= count[mask]

    hmax = heatmap.max()
    if hmax > 0:
        heatmap /= hmax

    return heatmap


# ---------------------------------------------------------------------------
# Overlay drawing
# ---------------------------------------------------------------------------


def save_projection_overlay_cv2(
    out_path: Path,
    img_path: Path,
    pair_results: list[dict[str, Any]],
    threshold: float,
    max_distance: float,
    pred: str,
    profiles: list[np.ndarray],
    diff_curves: list[np.ndarray],
    slicing_axis: str,
    diff_heatmap: np.ndarray | None = None,
    alpha: float = 0.3,
    heatmap_alpha: float = 0.45,
) -> None:
    """Save overlay image with diff heatmap, per-pair distances, and profile curves.

    Draws:
    - JET heatmap overlay showing where adjacent strips differ
    - Strip boundaries (green OK / red NG)
    - Per-pair distance labels
    - Small profile curve plots in margin area
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
        hmap_mask = diff_heatmap > 0.01
        blended = canvas_bgr.copy()
        blended[hmap_mask] = cv2.addWeighted(
            canvas_bgr, 1 - heatmap_alpha, hmap_color, heatmap_alpha, 0
        )[hmap_mask]
        canvas_bgr = blended

    # --- Per-pair annotations ---
    font_scale = max(0.3, min(img_w, img_h) / 1800.0)
    font_thickness = max(1, int(min(img_w, img_h) / 900))

    for pr in pair_results:
        dist = pr["distance"]
        is_ng = dist >= threshold
        ba = pr["bbox_a"]
        bb = pr["bbox_b"]

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

        # Distance label
        seg_dists = pr.get("seg_distances", [dist])
        skew = pr.get("skew_score", 0.0)
        if len(seg_dists) > 1:
            seg_str = "/".join(f"{sd:.3f}" for sd in seg_dists)
            label = f"d={dist:.4f} [{seg_str}]"
            if skew > 0.001:
                label += f" skew={skew:.3f}"
        else:
            label = f"d={dist:.4f}"
        if is_ng:
            label = f"NG {label}"
        label_color = (0, 0, 255) if is_ng else (0, 220, 0)

        if ba["y"] != bb["y"]:
            lx = ba["x"] + 2
            ly = max(min(ba["y"] + ba["h"], bb["y"]) - 3, int(16 * font_scale) + 2)
        else:
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

    header_color = (0, 200, 0) if pred == "OK" else (0, 0, 255)
    cv2.putText(
        canvas_bgr, pred, (10, int(40 * fs_label)),
        cv2.FONT_HERSHEY_SIMPLEX, fs_label, header_color,
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

    # --- Draw small profile curve comparison for each NG pair ---
    for pr in pair_results:
        if pr["distance"] < threshold:
            continue
        idx_a = pr["strip_a"]
        idx_b = pr["strip_b"]
        if idx_a >= len(profiles) or idx_b >= len(profiles):
            continue
        pa = normalise_profile(profiles[idx_a])
        pb = normalise_profile(profiles[idx_b])
        # Draw in the bbox_a region's top-right corner
        ba = pr["bbox_a"]
        _draw_mini_profile_pair(canvas_bgr, pa, pb, ba, font_scale)

    suffix = out_path.suffix.lower()
    if suffix in (".jpg", ".jpeg"):
        cv2.imwrite(str(out_path), canvas_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    else:
        cv2.imwrite(str(out_path), canvas_bgr)


def _draw_mini_profile_pair(
    canvas_bgr: np.ndarray,
    pa: np.ndarray,
    pb: np.ndarray,
    bbox: dict[str, int],
    font_scale: float,
) -> None:
    """Draw a small profile comparison chart inside a strip bbox."""
    import cv2

    # Chart dimensions - fit inside the bbox
    chart_w = min(bbox["w"] - 4, 120)
    chart_h = min(bbox["h"] - 4, 40)
    if chart_w < 20 or chart_h < 10:
        return

    cx0 = bbox["x"] + bbox["w"] - chart_w - 2
    cy0 = bbox["y"] + 2

    # Draw semi-transparent black background
    overlay = canvas_bgr.copy()
    cv2.rectangle(
        overlay, (cx0, cy0), (cx0 + chart_w, cy0 + chart_h),
        (0, 0, 0), cv2.FILLED,
    )
    cv2.addWeighted(overlay, 0.6, canvas_bgr, 0.4, 0, canvas_bgr)

    # Resample profiles to chart width
    xs = np.linspace(0, 1, chart_w)
    ya = np.interp(xs, np.linspace(0, 1, len(pa)), pa)
    yb = np.interp(xs, np.linspace(0, 1, len(pb)), pb)

    # Scale to chart height
    all_vals = np.concatenate([ya, yb])
    vmin, vmax = all_vals.min(), all_vals.max()
    if vmax - vmin < 1e-6:
        return
    ya_scaled = ((ya - vmin) / (vmax - vmin) * (chart_h - 2)).astype(int)
    yb_scaled = ((yb - vmin) / (vmax - vmin) * (chart_h - 2)).astype(int)

    # Draw curves
    for i in range(len(xs) - 1):
        pt1_a = (cx0 + i, cy0 + chart_h - 1 - int(ya_scaled[i]))
        pt2_a = (cx0 + i + 1, cy0 + chart_h - 1 - int(ya_scaled[i + 1]))
        cv2.line(canvas_bgr, pt1_a, pt2_a, (0, 255, 0), 1, cv2.LINE_AA)

        pt1_b = (cx0 + i, cy0 + chart_h - 1 - int(yb_scaled[i]))
        pt2_b = (cx0 + i + 1, cy0 + chart_h - 1 - int(yb_scaled[i + 1]))
        cv2.line(canvas_bgr, pt1_b, pt2_b, (0, 0, 255), 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Training: compute OK distance distribution and threshold
# ---------------------------------------------------------------------------


def compute_ok_pair_distances(
    img_paths: list[Path],
    strip_size: int,
    strip_overlap: int,
    metric: str = "l2",
    axis: str | None = None,
    smooth_kernel: int = 5,
    trim_pct: float = 0.05,
    smooth_type: str = "gaussian",
    align_max_shift: int = 0,
    skip_edge_strips: int = 0,
    projection_type: str = "mean",
    binary_threshold: int = 50,
    n_segments: int = 1,
    edge_band_width: int = 0,
    row_normalize: bool = False,
    progress_cb: Any = None,
) -> list[float]:
    """Compute all adjacent-pair projection distances from OK images.

    When *n_segments* > 1, each strip is split into sub-segments and the
    per-segment distances are included in the returned pool (the pair
    distance is the max across segments, same as inference).

    Returns:
        List of distances from all OK adjacent pairs.
    """
    all_distances: list[float] = []

    proj_kwargs = dict(
        smooth_kernel=smooth_kernel,
        trim_pct=trim_pct,
        smooth_type=smooth_type,
        projection_type=projection_type,
        binary_threshold=binary_threshold,
        edge_band_width=edge_band_width,
        row_normalize=row_normalize,
    )

    for idx, img_path in enumerate(img_paths):
        with Image.open(img_path) as im:
            im = im.convert("RGB")
            np_img = np.array(im)

        h0, w0 = np_img.shape[:2]
        resolved_axis = axis if axis else detect_long_axis(h0, w0)

        strips = slice_image_into_strips(np_img, strip_size, strip_overlap, axis)

        if skip_edge_strips > 0 and len(strips) > 2 * skip_edge_strips:
            strips = strips[skip_edge_strips:-skip_edge_strips]

        # Build per-strip segment profiles
        strip_seg_profiles: list[list[np.ndarray]] = []
        for s_img, *_ in strips:
            if n_segments > 1:
                sub_strips = _split_strip_into_segments(
                    s_img, resolved_axis, n_segments,
                )
                strip_seg_profiles.append([
                    column_projection(sub, slicing_axis=resolved_axis, **proj_kwargs)
                    for sub in sub_strips
                ])
            else:
                strip_seg_profiles.append([
                    column_projection(
                        s_img, slicing_axis=resolved_axis, **proj_kwargs,
                    )
                ])

        for i in range(len(strip_seg_profiles) - 1):
            seg_dists: list[float] = []
            for s in range(len(strip_seg_profiles[i])):
                if s < len(strip_seg_profiles[i + 1]):
                    sd = compute_pair_distance(
                        strip_seg_profiles[i][s],
                        strip_seg_profiles[i + 1][s],
                        metric, align_max_shift,
                    )
                    seg_dists.append(sd)
            pair_dist = max(seg_dists) if seg_dists else 0.0
            all_distances.append(pair_dist)

        if progress_cb and len(img_paths) > 1:
            pct = 30.0 + (idx + 1) / len(img_paths) * 50.0
            progress_cb(
                pct,
                f"OK projection distances: {idx + 1}/{len(img_paths)} images, "
                f"{len(all_distances)} pairs",
            )

    return all_distances


def compute_threshold_from_ok_distances(
    ok_distances: list[float],
    ok_quantile: float = 0.999,
    thr_scale: float = 1.5,
) -> float:
    """Compute threshold from OK pair distance distribution.

    threshold = quantile(ok_distances, ok_quantile) * thr_scale
    """
    if not ok_distances:
        return 0.5
    arr = np.array(ok_distances, dtype=np.float32)
    q = float(np.quantile(arr, ok_quantile))
    threshold = q * thr_scale
    logger.info(
        "OK projection distances: n=%d, mean=%.6f, std=%.6f, q%.3f=%.6f, threshold=%.6f",
        len(arr), float(np.mean(arr)), float(np.std(arr)), ok_quantile, q, threshold,
    )
    return threshold


# ---------------------------------------------------------------------------
# V3 Metal-mask-based boundary detection (final)
# ---------------------------------------------------------------------------
#
# Pipeline:
# 1) Direction-sensitive metal score (std + lap + |Gy| - penalty for |Gx|-|Gy|)
# 2) Per-side adaptive hysteresis thresholds from image statistics
# 3) Border-connected hysteresis masks (left / right)
# 4) Morphology scaled by estimated tooth pitch
# 5) Inner boundary extraction + smoothing (scaled by pitch)
# 6) Low-frequency trend + tooth phase template for scoring reference
# 7) Residual events: intrude / missing metal
# 8) Sigma-based event classification with pitch-scaled height thresholds
# ---------------------------------------------------------------------------


@dataclass
class MetalMaskParams:
    """Parameters for metal-mask boundary detection (final version).

    Key differences from V3-draft:
    - Direction-sensitive score: |Gy| replaces |Gx|, penalty suppresses
      vertical paint reflections.
    - Adaptive hysteresis thresholds per side (no fixed metal_thr).
    - All window sizes scale with auto-estimated tooth pitch.
    - Tooth phase template removes normal waviness before scoring.
    - Event threshold is sigma-based (robust MAD), not fixed pixel value.
    """

    # Preprocessing
    blur_ksize: int = 3
    blur_sigma: float = 0.6

    # Metalness features
    std_kernel: int = 5
    lap_kernel: int = 3
    sobel_kernel: int = 3
    norm_p_low: float = 1.0
    norm_p_high: float = 99.0

    # Directional metal score weights
    w_std: float = 0.42
    w_lap: float = 0.33
    w_gy: float = 0.25
    w_penalty: float = 0.18  # penalty for vertical reflection: max(0, gx-gy)

    # Adaptive threshold sampling
    seed_border_w_ratio: float = 0.03
    groove_w_ratio: float = 0.10
    left_alpha_hi: float = 0.55
    left_alpha_lo: float = 0.35
    right_alpha_hi: float = 0.62
    right_alpha_lo: float = 0.40

    # Pitch estimation
    pitch_min: int = 25
    pitch_max: int = 70

    # Morphology scaled by pitch P
    morph_close_h_ratio: float = 0.25
    morph_open_h_ratio: float = 0.08

    # Boundary smoothing scaled by pitch P
    boundary_median_ratio: float = 0.12
    boundary_sg_ratio: float = 0.80

    # Low-frequency trend reference
    ref_trend_ratio: float = 3.0  # trend window ~ 3*P

    # Tooth template for scoring ONLY
    template_smooth: int = 7
    template_rows_start_frac: float = 0.35  # use lower 65% rows to build template
    template_dev_sigma: float = 2.5         # only inlier residuals contribute

    # Event extraction
    event_sigma_mult: float = 3.0
    event_min_px: float = 2.0
    min_run_ratio: float = 0.25
    gap_merge_ratio: float = 0.10

    # NG / WARN decision
    ng_peak_thr: float = 4.0
    ng_height_ratio: float = 0.30
    ng_area_thr: float = 35.0
    ng_area_height_ratio: float = 0.25
    warn_peak_thr: float = 2.5
    warn_height_ratio: float = 0.20

    # Per-side ignore zones (fraction of image height to skip at top)
    ignore_top_left: float = 0.0    # 0 = full height; 0.20 = skip top 20%
    ignore_top_right: float = 0.20  # skip top 20% on right side

    # Right-side event type filter
    ignore_missing_right: bool = True  # True = only detect intrude on right side

    # Dipole scoring — detect paired protrusion+recession on same tooth
    dipole_enabled: bool = True       # use dipole as primary score
    dipole_amp_weight: float = 1.0    # weight for amplitude (peak displacement) component
    dipole_area_weight: float = 0.03  # weight for area component (normalised by window length)
    dipole_dist_ratio: float = 0.6    # max distance between +/- peaks (fraction of pitch)
    dipole_ng_thr: float = 5.5        # dipole score >= this → NG
    dipole_warn_thr: float = 4.0      # dipole score >= this → WARN

    # Adjacent tooth-peak difference — simplest possible metric.
    # Finds per-tooth boundary peak x-positions, fits a straight line,
    # then measures residual (deviation from line) per tooth.
    # Max |residual| > threshold → NG.
    # When enabled, overrides dipole scoring for judgment.
    adj_diff_enabled: bool = False     # enable adjacent-peak-diff judgment
    adj_diff_ng_thr: float = 5.0      # max residual (px) >= this → NG
    adj_diff_warn_thr: float = 3.0    # max residual (px) >= this → WARN
    adj_diff_overlay: bool = True      # draw peak dots, fitted line & residual labels
    right_fit_offset: float = 160.0   # right fitted line = left fitted line + this offset (px)
    area_depth_thr: float = 15.0      # only count area where |residual| > this threshold (px)


@dataclass
class MetalEvent:
    """A detected anomaly event on the metal boundary."""

    side: str         # 'left' or 'right'
    kind: str         # 'intrude' or 'missing_metal'
    y0: int
    y1: int
    peak_px: float
    height_px: int
    area: float
    level: str        # 'NG' / 'WARN' / 'OK'


def _mm_odd(v: float, min_odd: int = 3) -> int:
    """Round to nearest odd integer >= min_odd."""
    v = max(int(round(v)), min_odd)
    return v if (v % 2 == 1) else (v + 1)


def _robust_norm(
    x: np.ndarray,
    p_low: float,
    p_high: float,
    stride: int = 4,
) -> np.ndarray:
    """Percentile-based normalisation to [0, 1].

    Uses strided subsampling + np.partition for speed.
    stride=4 gives ~3× speedup with negligible accuracy loss.
    """
    flat = x.ravel()
    n = flat.size
    if n == 0:
        return x.astype(np.float32)
    sample = flat[::stride] if stride > 1 else flat
    ns = sample.size
    k_lo = max(0, min(int(p_low / 100.0 * (ns - 1)), ns - 1))
    k_hi = max(0, min(int(p_high / 100.0 * (ns - 1)), ns - 1))
    if k_lo == k_hi:
        lo = hi = float(np.partition(sample, k_lo)[k_lo])
    else:
        part = np.partition(sample, [k_lo, k_hi])
        lo = float(part[k_lo])
        hi = float(part[k_hi])
    y = (x - lo) / (hi - lo + 1e-6)
    return np.clip(y, 0.0, 1.0).astype(np.float32)


def _robust_sigma(x: np.ndarray) -> float:
    """MAD-based robust estimate of standard deviation."""
    x = np.asarray(x, np.float32)
    if x.size == 0:
        return 1e-6
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return float(1.4826 * mad + 1e-6)


def _fill_nan_by_interp(a: np.ndarray) -> np.ndarray:
    """Fill NaN values by linear interpolation."""
    x = np.arange(len(a), dtype=np.float32)
    good = np.isfinite(a)
    out = a.copy().astype(np.float32)
    if good.sum() == 0:
        return np.full_like(out, np.nan, dtype=np.float32)
    if good.sum() == 1:
        out[:] = out[good][0]
        return out
    out[~good] = np.interp(x[~good], x[good], out[good]).astype(np.float32)
    return out


def _smooth_1d(
    a: np.ndarray,
    med_win: int,
    sg_win: int,
    sg_poly: int = 3,
) -> np.ndarray:
    """Median filter + Savitzky-Golay smooth for 1-D signal."""
    med_win = _mm_odd(med_win)
    max_odd = len(a) - 1 if len(a) % 2 == 0 else len(a)
    sg_win = _mm_odd(min(sg_win, max_odd))
    if sg_win <= sg_poly:
        sg_win = _mm_odd(sg_poly + 3)
    b = median_filter(a.astype(np.float32), size=med_win, mode="nearest")
    c = savgol_filter(b, sg_win, sg_poly, mode="interp").astype(np.float32)
    return c


def build_metal_score(
    img_bgr: np.ndarray,
    p: MetalMaskParams,
    gray: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute direction-sensitive metalness score.

    score = w_std*std + w_lap*lap + w_gy*|Gy| - w_penalty*max(0, |Gx|-|Gy|)

    The penalty term suppresses vertical paint reflections inside the groove
    that produce strong |Gx| but weak |Gy|.

    Args:
        img_bgr: BGR image (used only if *gray* is not provided).
        p: Parameters.
        gray: Pre-computed float32 gray image in [0, 1] with blur applied.
              Pass this to avoid redundant gray conversion + blur.

    Returns:
        (score, n_std, n_lap, n_gx, n_gy) — all float32 arrays of shape (H, W).
    """
    if gray is None:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        gray = cv2.GaussianBlur(gray, (p.blur_ksize, p.blur_ksize), p.blur_sigma)

    # cv2.blur is ~5× faster than scipy uniform_filter for box averaging
    k = (p.std_kernel, p.std_kernel)
    mean = cv2.blur(gray, k)
    mean_sq = cv2.blur(gray * gray, k)
    local_std = np.sqrt(np.maximum(mean_sq - mean * mean, 0.0)).astype(np.float32)

    abs_lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=p.lap_kernel))
    abs_gx = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=p.sobel_kernel))
    abs_gy = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=p.sobel_kernel))

    n_std = _robust_norm(local_std, p.norm_p_low, p.norm_p_high)
    n_lap = _robust_norm(abs_lap, p.norm_p_low, p.norm_p_high)
    n_gx = _robust_norm(abs_gx, p.norm_p_low, p.norm_p_high)
    n_gy = _robust_norm(abs_gy, p.norm_p_low, p.norm_p_high)

    score = (
        p.w_std * n_std
        + p.w_lap * n_lap
        + p.w_gy * n_gy
        - p.w_penalty * np.maximum(0.0, n_gx - n_gy)
    )
    score = np.clip(score, 0.0, 1.0).astype(np.float32)
    return score, n_std, n_lap, n_gx, n_gy


def _compute_side_thresholds(
    score: np.ndarray,
    p: MetalMaskParams,
) -> dict[str, Any]:
    """Compute adaptive hysteresis thresholds per side from image statistics.

    Samples groove-centre (low metal score), left-border, right-border regions
    and interpolates between them to get per-side hi/lo thresholds.
    """
    h, w = score.shape
    seed_w = max(6, int(round(w * p.seed_border_w_ratio)))
    groove_w = max(14, int(round(w * p.groove_w_ratio)))
    cx = w // 2

    sg = score[:, max(0, cx - groove_w // 2): min(w, cx + groove_w // 2)]
    sl = score[:, :seed_w]
    sr = score[:, w - seed_w:]

    g95 = float(np.percentile(sg, 95))
    l50 = float(np.percentile(sl, 50))
    r50 = float(np.percentile(sr, 50))

    return {
        "seed_w": seed_w,
        "g95": g95,
        "l50": l50,
        "r50": r50,
        "thrL_hi": g95 + p.left_alpha_hi * (l50 - g95),
        "thrL_lo": g95 + p.left_alpha_lo * (l50 - g95),
        "thrR_hi": g95 + p.right_alpha_hi * (r50 - g95),
        "thrR_lo": g95 + p.right_alpha_lo * (r50 - g95),
    }


def _keep_border_connected(
    mask: np.ndarray,
    side: str,
    border_w: int = 8,
) -> np.ndarray:
    """Keep only connected components that touch the specified border."""
    num, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    h, w = mask.shape
    border = labels[:, :border_w] if side == "left" else labels[:, w - border_w:]
    ids = np.unique(border)
    ids = ids[ids != 0]
    if ids.size == 0:
        return np.zeros(mask.shape, dtype=bool)
    # Vectorised: build a lookup table instead of looping over label IDs
    keep = np.zeros(num, dtype=bool)
    keep[ids] = True
    return keep[labels]


def _morph_bool(
    mask: np.ndarray,
    close_kernel: tuple[int, int],
    open_kernel: tuple[int, int],
) -> np.ndarray:
    """Apply morphological close + open on a boolean mask."""
    m = mask.astype(np.uint8)
    kc = cv2.getStructuringElement(cv2.MORPH_RECT, close_kernel)
    ko = cv2.getStructuringElement(cv2.MORPH_RECT, open_kernel)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kc)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, ko)
    return m.astype(bool)


def _hysteresis_side_mask(
    score: np.ndarray,
    thr_hi: float,
    thr_lo: float,
    side: str,
    border_w: int = 8,
) -> np.ndarray:
    """Build a side mask using hysteresis thresholding.

    Seeds are border-connected components above thr_hi.
    Then flood-fill into connected components of thr_lo.
    """
    hi = score >= thr_hi
    lo = score >= thr_lo

    seed = _keep_border_connected(hi, side=side, border_w=border_w)

    num, labels = cv2.connectedComponents(lo.astype(np.uint8), connectivity=8)

    seed_ids = np.unique(labels[seed])
    seed_ids = seed_ids[seed_ids != 0]
    if seed_ids.size == 0:
        return np.zeros(score.shape, dtype=bool)
    # Vectorised lookup table
    keep = np.zeros(num, dtype=bool)
    keep[seed_ids] = True
    return keep[labels]


def _extract_metal_inner_boundary(
    mask: np.ndarray,
    side: str,
) -> np.ndarray:
    """Extract innermost metal pixel per row (vectorised).

    For left side: rightmost metal pixel (largest x).
    For right side: leftmost metal pixel (smallest x).
    """
    h, w = mask.shape
    has_any = mask.any(axis=1)  # (h,)
    col_idx = np.arange(w, dtype=np.float32)
    if side == "left":
        # For each row, find the maximum column index where mask is True
        weighted = np.where(mask, col_idx[np.newaxis, :], -1.0)
        vals = weighted.max(axis=1)
    else:
        # For each row, find the minimum column index where mask is True
        weighted = np.where(mask, col_idx[np.newaxis, :], float(w + 1))
        vals = weighted.min(axis=1)
    x = np.where(has_any, vals, np.nan).astype(np.float32)
    return x


def _estimate_pitch_from_gray(
    gray: np.ndarray,
    p: MetalMaskParams,
) -> int:
    """Estimate tooth pitch from autocorrelation of left-edge gradient."""
    h, w = gray.shape
    side_w = max(12, int(round(w * 0.10)))
    roi = gray[:, :side_w]
    gx = cv2.Sobel(roi, cv2.CV_32F, 1, 0, ksize=3)
    s = np.mean(np.abs(gx), axis=1)
    s = s - np.mean(s)
    ac = np.correlate(s, s, mode="full")
    ac = ac[len(ac) // 2:]
    lag_min = p.pitch_min
    lag_max = min(p.pitch_max, len(ac) - 1)
    return int(lag_min + np.argmax(ac[lag_min: lag_max + 1]))


def _estimate_pitch_from_signal(
    sig: np.ndarray,
    p: MetalMaskParams,
    fallback_pitch: int,
) -> int:
    """Estimate tooth pitch from autocorrelation of a 1-D boundary signal."""
    s = sig - np.mean(sig)
    if np.allclose(s, 0):
        return fallback_pitch
    ac = np.correlate(s, s, mode="full")
    ac = ac[len(ac) // 2:]
    lag_min = p.pitch_min
    lag_max = min(p.pitch_max, len(ac) - 1)
    return int(lag_min + np.argmax(ac[lag_min: lag_max + 1]))


def _build_phase_template(
    resid: np.ndarray,
    pitch: int,
    rows_mask: np.ndarray,
    smooth_win: int = 7,
) -> np.ndarray:
    """Build a tooth-phase template from inlier residuals.

    The template captures the normal periodic tooth waviness so it can be
    subtracted before scoring.  Only inlier rows contribute.
    """
    phase = np.arange(len(resid)) % pitch
    temp = np.zeros(pitch, np.float32)
    for ph in range(pitch):
        vals = resid[(phase == ph) & rows_mask]
        if vals.size == 0:
            vals = resid[phase == ph]
        temp[ph] = np.median(vals) if vals.size else 0.0

    # Circular smooth
    w = _mm_odd(smooth_win)
    pad = w // 2
    ext = np.r_[temp[-pad:], temp, temp[:pad]]
    sm = savgol_filter(ext, w, 2, mode="interp")
    temp = sm[pad: pad + pitch].astype(np.float32)
    temp -= temp.mean()
    return temp


def _merge_runs(
    binary: np.ndarray,
    gap_merge: int,
) -> list[tuple[int, int]]:
    """Find and merge continuous True runs in a boolean array."""
    ys = np.flatnonzero(binary)
    if ys.size == 0:
        return []
    runs: list[tuple[int, int]] = []
    s = int(ys[0])
    prev = int(ys[0])
    for y in ys[1:]:
        y = int(y)
        if y - prev <= gap_merge + 1:
            prev = y
        else:
            runs.append((s, prev))
            s = y
            prev = y
    runs.append((s, prev))
    return runs


def _compute_adj_peak_diffs(
    xL: np.ndarray,
    pitch: int,
    ignore_top_frac: float = 0.0,
    *,
    external_fit: tuple[float, float] | None = None,
    area_depth_thr: float = 15.0,
) -> dict[str, Any]:
    """Compute tooth-peak residuals from a fitted straight line.

    For each tooth-sized window, find the peak (rightmost point = tooth tip).
    Fit a straight line through all peaks, then compute per-peak residual
    (signed deviation from line).  Also compute adjacent residual diffs.

    Args:
        xL: 1-D smoothed boundary x-positions.
        pitch: Estimated tooth pitch in pixels.
        ignore_top_frac: Fraction of image height to skip at top.
        external_fit: If provided, use this (slope, intercept) instead of
            fitting from the peaks.  Used for right side = left fit + offset.
        area_depth_thr: Only count area where |residual| exceeds this (px).

    Returns:
        Dict with keys:
          peaks: list of (y_center, x_peak) for each tooth
          residuals: list of (y_center, residual_px) signed deviation from line
          adj_diffs: list of (y1, y2, diff_px) for each adjacent pair
          max_adj_diff: maximum adjacent difference in pixels
          max_residual: maximum absolute residual from fitted line
          fit_slope: slope of the fitted line (x per y pixel)
          fit_intercept: intercept of the fitted line
    """
    H = len(xL)
    half_p = max(1, pitch // 2)
    start_y = int(H * ignore_top_frac)

    # Find tooth centers at regular pitch intervals
    centers = list(range(half_p, H - half_p, pitch))

    peaks: list[tuple[int, float]] = []  # (y_actual_peak, x_peak)
    for cy in centers:
        if cy < start_y:
            continue
        y0 = max(0, cy - half_p)
        y1 = min(H, cy + half_p)
        seg = xL[y0:y1]
        if len(seg) == 0:
            continue
        # Tooth tip = rightmost boundary point (max x for left side)
        peak_x = float(np.max(seg))
        # Use actual y-position of the peak, not the window center
        peak_y = y0 + int(np.argmax(seg))
        peaks.append((peak_y, peak_x))

    # Fit straight line through peaks: x = slope * y + intercept
    fit_slope = 0.0
    fit_intercept = 0.0
    residuals: list[tuple[int, float]] = []  # (y_center, residual_px)
    if external_fit is not None:
        # Use externally provided fit (e.g. left side fit shifted for right side)
        fit_slope, fit_intercept = external_fit
        if peaks:
            ys_arr = np.array([p[0] for p in peaks], dtype=float)
            xs_arr = np.array([p[1] for p in peaks], dtype=float)
            x_fit = fit_slope * ys_arr + fit_intercept
            for i, (cy, _px) in enumerate(peaks):
                residuals.append((cy, float(xs_arr[i] - x_fit[i])))
    elif len(peaks) >= 2:
        ys_arr = np.array([p[0] for p in peaks], dtype=float)
        xs_arr = np.array([p[1] for p in peaks], dtype=float)
        coeffs = np.polyfit(ys_arr, xs_arr, 1)
        fit_slope = float(coeffs[0])
        fit_intercept = float(coeffs[1])
        x_fit = np.polyval(coeffs, ys_arr)
        for i, (cy, _px) in enumerate(peaks):
            residuals.append((cy, float(xs_arr[i] - x_fit[i])))
    else:
        for cy, px in peaks:
            residuals.append((cy, 0.0))

    # Compute adjacent differences (raw peak-to-peak)
    adj_diffs: list[tuple[int, int, float]] = []
    for i in range(len(peaks) - 1):
        cy1, x1 = peaks[i]
        cy2, x2 = peaks[i + 1]
        diff = abs(x1 - x2)
        adj_diffs.append((cy1, cy2, diff))

    max_adj_diff = max((d[2] for d in adj_diffs), default=0.0)
    max_residual = max((abs(r[1]) for r in residuals), default=0.0)
    max_idx = -1
    if adj_diffs:
        max_idx = max(range(len(adj_diffs)), key=lambda i: adj_diffs[i][2])

    # ---- Full-curve excursion areas ----
    # For each contiguous region where the boundary deviates to one side
    # of the fitted line (positive or negative), compute the enclosed area
    # (integral of |residual| over that run).  This captures both depth
    # AND width: a narrow deep spike has small area while a wide bulge
    # has large area.
    max_pos_residual = 0.0   # max positive-excursion area
    max_neg_residual = 0.0   # max negative-excursion area (stored as negative)
    max_pos_y = -1           # y of peak within max positive excursion
    max_neg_y = -1           # y of peak within max negative excursion

    if fit_slope != 0.0 or fit_intercept != 0.0:
        ys_full = np.arange(start_y, H, dtype=float)
        x_fit_full = fit_slope * ys_full + fit_intercept
        residual_full = xL[start_y:H].astype(float) - x_fit_full
        n = len(residual_full)

        # Thresholded area: only count pixels where |residual| > depth_thr.
        # This filters out long shallow deviations (normal tooth waviness)
        # and only measures the area of genuinely deep excursions.
        depth_thr = area_depth_thr

        # Positive side: residual > +depth_thr
        pos_mask = residual_full > depth_thr
        pos_excess = np.where(pos_mask, residual_full - depth_thr, 0.0)
        # Negative side: residual < -depth_thr
        neg_mask = residual_full < -depth_thr
        neg_excess = np.where(neg_mask, -residual_full - depth_thr, 0.0)

        # Find contiguous runs and pick the one with largest area
        best_pos_area = 0.0
        best_pos_peak_y = -1
        best_neg_area = 0.0
        best_neg_peak_y = -1

        # --- Positive excursion runs ---
        i = 0
        while i < n:
            if pos_mask[i]:
                area = 0.0
                peak_val = 0.0
                peak_idx = i
                while i < n and pos_mask[i]:
                    area += pos_excess[i]
                    if residual_full[i] > peak_val:
                        peak_val = residual_full[i]
                        peak_idx = i
                    i += 1
                if area > best_pos_area:
                    best_pos_area = area
                    best_pos_peak_y = peak_idx
            else:
                i += 1

        # --- Negative excursion runs ---
        i = 0
        while i < n:
            if neg_mask[i]:
                area = 0.0
                peak_val = 0.0
                peak_idx = i
                while i < n and neg_mask[i]:
                    area += neg_excess[i]
                    if -residual_full[i] > peak_val:
                        peak_val = -residual_full[i]
                        peak_idx = i
                    i += 1
                if area > best_neg_area:
                    best_neg_area = area
                    best_neg_peak_y = peak_idx
            else:
                i += 1

        max_pos_residual = best_pos_area
        max_neg_residual = -best_neg_area  # store as negative
        max_pos_y = int(start_y + best_pos_peak_y) if best_pos_peak_y >= 0 else -1
        max_neg_y = int(start_y + best_neg_peak_y) if best_neg_peak_y >= 0 else -1

    return {
        "peaks": peaks,
        "residuals": residuals,
        "adj_diffs": adj_diffs,
        "max_adj_diff": float(max_adj_diff),
        "max_residual": float(max_residual),
        "max_pos_residual": float(max_pos_residual),
        "max_neg_residual": float(max_neg_residual),
        "max_pos_y": max_pos_y,
        "max_neg_y": max_neg_y,
        "max_adj_pair_idx": max_idx,
        "fit_slope": fit_slope,
        "fit_intercept": fit_intercept,
    }


def _compute_dipole_scores(
    signed_resid: np.ndarray,
    pitch: int,
    p: MetalMaskParams,
) -> list[dict[str, Any]]:
    """Compute per-tooth dipole scores from signed residual.

    A dipole is a paired protrusion+recession within the same tooth window.
    Score = amp_weight * min(A_pos, A_neg)
           + area_weight * min(S_pos, S_neg) / window_rows,
    only when the positive and negative peaks are within dist_ratio * pitch.
    Area is normalised by window length so long-but-small offsets don't dominate.

    Args:
        signed_resid: 1-D signed residual (positive = intrude, negative = missing).
        pitch: Estimated tooth pitch in pixels.
        p: MetalMaskParams with dipole thresholds.

    Returns:
        List of dicts with keys: cy, y0, y1, dipole_score, A_pos, A_neg,
        S_pos, S_neg, peak_dist, level.
    """
    pos = np.maximum(signed_resid, 0.0)
    neg = np.maximum(-signed_resid, 0.0)

    H = len(signed_resid)
    half_p = max(1, pitch // 2)
    centers = list(range(half_p, H - half_p, pitch))

    results: list[dict[str, Any]] = []
    for cy in centers:
        y0 = max(0, cy - half_p)
        y1 = min(H, cy + half_p)

        p_seg = pos[y0:y1]
        n_seg = neg[y0:y1]
        if len(p_seg) == 0:
            continue

        a_pos = float(np.max(p_seg))
        a_neg = float(np.max(n_seg))
        s_pos = float(np.sum(p_seg))
        s_neg = float(np.sum(n_seg))

        iy_pos = int(np.argmax(p_seg))
        iy_neg = int(np.argmax(n_seg))
        peak_dist = abs(iy_pos - iy_neg)

        if peak_dist <= p.dipole_dist_ratio * pitch:
            win_len = max(1, y1 - y0)
            sc = (p.dipole_amp_weight * min(a_pos, a_neg)
                  + p.dipole_area_weight * min(s_pos, s_neg) / win_len)
        else:
            sc = 0.0

        level = "OK"
        if sc >= p.dipole_ng_thr:
            level = "NG"
        elif sc >= p.dipole_warn_thr:
            level = "WARN"

        results.append({
            "cy": cy, "y0": y0, "y1": y1,
            "dipole_score": float(sc),
            "A_pos": a_pos, "A_neg": a_neg,
            "S_pos": s_pos, "S_neg": s_neg,
            "peak_dist": peak_dist,
            "level": level,
        })
    return results


def _classify_metal_events(
    residual: np.ndarray,
    side: str,
    kind: str,
    pitch: int,
    p: MetalMaskParams,
) -> tuple[list[MetalEvent], float]:
    """Classify residual signal into anomaly events.

    Uses adaptive sigma-based threshold and pitch-scaled height thresholds.

    Returns:
        (events, threshold) — list of events and the threshold used.
    """
    thr = max(p.event_min_px, p.event_sigma_mult * _robust_sigma(residual))
    min_run = max(6, int(round(p.min_run_ratio * pitch)))
    gap_merge = max(3, int(round(p.gap_merge_ratio * pitch)))

    runs = _merge_runs(residual > thr, gap_merge)
    events: list[MetalEvent] = []

    for y0, y1 in runs:
        height = y1 - y0 + 1
        if height < min_run:
            continue
        seg = residual[y0: y1 + 1]
        peak = float(np.max(seg))
        area = float(np.sum(seg))
        level = "OK"

        if peak >= p.ng_peak_thr and height >= max(8, int(round(p.ng_height_ratio * pitch))):
            level = "NG"
        elif area >= p.ng_area_thr and height >= max(8, int(round(p.ng_area_height_ratio * pitch))):
            level = "NG"
        elif peak >= p.warn_peak_thr and height >= max(6, int(round(p.warn_height_ratio * pitch))):
            level = "WARN"

        events.append(MetalEvent(
            side=side, kind=kind, y0=y0, y1=y1,
            peak_px=peak, height_px=height, area=area, level=level,
        ))
    return events, thr


def infer_metal_mask(
    img_path: Path,
    params: MetalMaskParams | None = None,
) -> dict[str, Any]:
    """Run metal-mask boundary detection on a single image.

    Full pipeline: directional metal score → adaptive hysteresis masks →
    pitch estimation → morphology → boundaries → trend + tooth template →
    residuals → sigma-based event classification → judgment.

    Args:
        img_path: Path to input image.
        params: Metal mask parameters (uses defaults if None).

    Returns:
        Dict with keys: result, score, events, stats, params,
        xL, xR, xL_score_ref, xR_score_ref, left_mask, right_mask,
        score_map, img_shape.
    """
    import time as _time
    p = params or MetalMaskParams()
    _t = {}  # per-step timing (ms)

    _t0 = _time.perf_counter()
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")
    _t["imread"] = (_time.perf_counter() - _t0) * 1000

    h, w = img_bgr.shape[:2]

    _t0 = _time.perf_counter()
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    gray = cv2.GaussianBlur(gray, (p.blur_ksize, p.blur_ksize), p.blur_sigma)
    _t["gray_blur"] = (_time.perf_counter() - _t0) * 1000

    # 1) Directional metal score (pass pre-computed gray to avoid duplicate work)
    _t0 = _time.perf_counter()
    score, _n_std, _n_lap, _n_gx, _n_gy = build_metal_score(img_bgr, p, gray=gray)
    _t["build_metal_score"] = (_time.perf_counter() - _t0) * 1000

    # 2) Adaptive thresholds
    _t0 = _time.perf_counter()
    thr_info = _compute_side_thresholds(score, p)
    pitch_gray = _estimate_pitch_from_gray(gray, p)
    _t["thresholds_pitch"] = (_time.perf_counter() - _t0) * 1000

    # 3) Side-connected hysteresis masks
    _t0 = _time.perf_counter()
    left_mask = _hysteresis_side_mask(
        score, thr_info["thrL_hi"], thr_info["thrL_lo"],
        side="left", border_w=thr_info["seed_w"],
    )
    right_mask = _hysteresis_side_mask(
        score, thr_info["thrR_hi"], thr_info["thrR_lo"],
        side="right", border_w=thr_info["seed_w"],
    )
    _t["hysteresis_masks"] = (_time.perf_counter() - _t0) * 1000

    # 4) Morphology scaled by pitch
    _t0 = _time.perf_counter()
    close_kernel = (3, _mm_odd(p.morph_close_h_ratio * pitch_gray))
    open_kernel = (3, _mm_odd(p.morph_open_h_ratio * pitch_gray))
    left_mask = _morph_bool(left_mask, close_kernel, open_kernel)
    right_mask = _morph_bool(right_mask, close_kernel, open_kernel)
    _t["morphology"] = (_time.perf_counter() - _t0) * 1000

    # 5) Raw boundaries
    _t0 = _time.perf_counter()
    xL_raw = _extract_metal_inner_boundary(left_mask, "left")
    xR_raw = _extract_metal_inner_boundary(right_mask, "right")
    _t["extract_boundary"] = (_time.perf_counter() - _t0) * 1000

    # 6) Smoothed boundaries (scaled by pitch)
    _t0 = _time.perf_counter()
    pitch_L = _estimate_pitch_from_signal(_fill_nan_by_interp(xL_raw), p, pitch_gray)
    pitch_R = _estimate_pitch_from_signal(_fill_nan_by_interp(xR_raw), p, pitch_gray)

    xL = _smooth_1d(
        _fill_nan_by_interp(xL_raw),
        _mm_odd(p.boundary_median_ratio * pitch_L),
        _mm_odd(p.boundary_sg_ratio * pitch_L),
        3,
    )
    xR = _smooth_1d(
        _fill_nan_by_interp(xR_raw),
        _mm_odd(p.boundary_median_ratio * pitch_R),
        _mm_odd(p.boundary_sg_ratio * pitch_R),
        3,
    )
    _t["smooth_boundary"] = (_time.perf_counter() - _t0) * 1000

    # ---- Fast path: when adj_diff_enabled, skip template / dipole / events ----
    dipole_pitch_L = p.pitch_min
    dipole_pitch_R = p.pitch_min

    if p.adj_diff_enabled:
        # 10b) Adjacent tooth-peak difference only (area-based metric)
        _t0 = _time.perf_counter()
        adj_diff_L = _compute_adj_peak_diffs(
            xL, dipole_pitch_L, p.ignore_top_left,
            area_depth_thr=p.area_depth_thr,
        )
        L_slope = adj_diff_L["fit_slope"]
        L_intercept = adj_diff_L["fit_intercept"]
        r_external: tuple[float, float] | None = None
        if L_slope != 0.0 or L_intercept != 0.0:
            r_external = (L_slope, L_intercept + p.right_fit_offset)
        adj_diff_R = _compute_adj_peak_diffs(
            xR, dipole_pitch_R, p.ignore_top_right,
            external_fit=r_external,
            area_depth_thr=p.area_depth_thr,
        )
        _t["adj_peak_diffs"] = (_time.perf_counter() - _t0) * 1000

        _t["total"] = sum(_t.values())

        max_area = max(
            abs(adj_diff_L["max_pos_residual"]),
            abs(adj_diff_L["max_neg_residual"]),
            abs(adj_diff_R["max_pos_residual"]),
            abs(adj_diff_R["max_neg_residual"]),
        )
        if max_area >= p.adj_diff_ng_thr:
            result = "NG"
        elif max_area >= p.adj_diff_warn_thr:
            result = "WARN"
        else:
            result = "OK"
        final_score = max_area

        # Dummy scoring refs = boundaries themselves (only used for overlay)
        xL_score_ref = xL.copy()
        xR_score_ref = xR.copy()

        return {
            "result": result,
            "score": float(final_score),
            "events": [],
            "dipoles_L": [],
            "dipoles_R": [],
            "stats": {
                "pitch_L": int(pitch_L),
                "pitch_R": int(pitch_R),
                "thrL_hi": float(thr_info["thrL_hi"]),
                "thrL_lo": float(thr_info["thrL_lo"]),
                "thrR_hi": float(thr_info["thrR_hi"]),
                "thrR_lo": float(thr_info["thrR_lo"]),
                "left_mask_ratio": float(left_mask.mean()),
                "right_mask_ratio": float(right_mask.mean()),
                "adj_diff_L": adj_diff_L,
                "adj_diff_R": adj_diff_R,
                "max_adj_diff": float(max(adj_diff_L["max_adj_diff"], adj_diff_R["max_adj_diff"])),
                "max_residual": float(max(adj_diff_L["max_residual"], adj_diff_R["max_residual"])),
                "L_max_pos_residual": float(adj_diff_L["max_pos_residual"]),
                "L_max_neg_residual": float(adj_diff_L["max_neg_residual"]),
                "R_max_pos_residual": float(adj_diff_R["max_pos_residual"]),
                "R_max_neg_residual": float(adj_diff_R["max_neg_residual"]),
            },
            "params": asdict(p),
            "xL": xL,
            "xR": xR,
            "xL_score_ref": xL_score_ref,
            "xR_score_ref": xR_score_ref,
            "left_mask": left_mask,
            "right_mask": right_mask,
            "score_map": score,
            "img_shape": (h, w),
            "timing_ms": _t,
        }

    # ---- Full path: template + dipole + events (legacy / dipole modes) ----
    _t0 = _time.perf_counter()

    # 7) Low-frequency trends
    trend_L = savgol_filter(
        xL, _mm_odd(p.ref_trend_ratio * pitch_L), 2, mode="interp",
    ).astype(np.float32)
    trend_R = savgol_filter(
        xR, _mm_odd(p.ref_trend_ratio * pitch_R), 2, mode="interp",
    ).astype(np.float32)

    # 8) Tooth template for scoring ONLY
    resid_L = xL - trend_L
    resid_R = xR - trend_R

    rows_L = np.arange(len(xL)) >= int(len(xL) * p.template_rows_start_frac)
    rows_R = np.arange(len(xR)) >= int(len(xR) * p.template_rows_start_frac)

    sig_L = _robust_sigma(resid_L[rows_L])
    sig_R = _robust_sigma(resid_R[rows_R])
    rows_L = rows_L & (np.abs(resid_L) <= p.template_dev_sigma * sig_L)
    rows_R = rows_R & (np.abs(resid_R) <= p.template_dev_sigma * sig_R)

    temp_L = _build_phase_template(resid_L, pitch_L, rows_L, smooth_win=p.template_smooth)
    temp_R = _build_phase_template(resid_R, pitch_R, rows_R, smooth_win=p.template_smooth)

    # Scoring references (boundaries themselves are untouched)
    xL_score_ref = trend_L + temp_L[np.arange(len(xL)) % pitch_L]
    xR_score_ref = trend_R + temp_R[np.arange(len(xR)) % pitch_R]

    # 9) Abnormal residuals relative to scoring references
    intrude_L = np.maximum(0.0, xL - xL_score_ref)
    missing_L = np.maximum(0.0, xL_score_ref - xL)
    intrude_R = np.maximum(0.0, xR_score_ref - xR)
    missing_R = np.maximum(0.0, xR - xR_score_ref)

    # 9b) Per-side ignore zones — zero out residuals in ignored rows
    if p.ignore_top_left > 0:
        cut_L = int(h * p.ignore_top_left)
        intrude_L[:cut_L] = 0.0
        missing_L[:cut_L] = 0.0
    if p.ignore_top_right > 0:
        cut_R = int(h * p.ignore_top_right)
        intrude_R[:cut_R] = 0.0
        missing_R[:cut_R] = 0.0

    # 9c) Right-side: only intrude (toward groove), skip missing (away)
    if p.ignore_missing_right:
        missing_R[:] = 0.0

    # 10) Dipole scoring — signed residual per side
    signed_L = (xL - xL_score_ref).copy()
    signed_R = (xR_score_ref - xR).copy()

    if p.ignore_top_left > 0:
        cut_L = int(h * p.ignore_top_left)
        signed_L[:cut_L] = 0.0
    if p.ignore_top_right > 0:
        cut_R = int(h * p.ignore_top_right)
        signed_R[:cut_R] = 0.0

    dipoles_L = _compute_dipole_scores(signed_L, dipole_pitch_L, p)
    if p.ignore_missing_right:
        dipoles_R: list[dict[str, Any]] = []
    else:
        dipoles_R = _compute_dipole_scores(signed_R, dipole_pitch_R, p)

    dipole_max_L = max((d["dipole_score"] for d in dipoles_L), default=0.0)
    dipole_max_R = max((d["dipole_score"] for d in dipoles_R), default=0.0)
    dipole_score = max(dipole_max_L, dipole_max_R)

    # 10b) Adjacent tooth-peak difference
    adj_diff_L = _compute_adj_peak_diffs(
        xL, dipole_pitch_L, p.ignore_top_left,
        area_depth_thr=p.area_depth_thr,
    )
    L_slope = adj_diff_L["fit_slope"]
    L_intercept = adj_diff_L["fit_intercept"]
    r_external_full: tuple[float, float] | None = None
    if L_slope != 0.0 or L_intercept != 0.0:
        r_external_full = (L_slope, L_intercept + p.right_fit_offset)
    adj_diff_R = _compute_adj_peak_diffs(
        xR, dipole_pitch_R, p.ignore_top_right,
        external_fit=r_external_full,
        area_depth_thr=p.area_depth_thr,
    )
    max_adj_diff = max(adj_diff_L["max_adj_diff"], adj_diff_R["max_adj_diff"])
    max_residual = max(adj_diff_L["max_residual"], adj_diff_R["max_residual"])

    # 11) Legacy event-based scoring
    events: list[MetalEvent] = []
    ev, thr_intr_L = _classify_metal_events(intrude_L, "left", "intrude", pitch_L, p)
    events += ev
    ev, thr_miss_L = _classify_metal_events(missing_L, "left", "missing_metal", pitch_L, p)
    events += ev
    ev, thr_intr_R = _classify_metal_events(intrude_R, "right", "intrude", pitch_R, p)
    events += ev
    ev, thr_miss_R = _classify_metal_events(missing_R, "right", "missing_metal", pitch_R, p)
    events += ev

    max_peak = max((ev.peak_px for ev in events), default=0.0)

    # 12) Overall judgment
    if p.dipole_enabled:
        if dipole_score >= p.dipole_ng_thr:
            result = "NG"
        elif dipole_score >= p.dipole_warn_thr:
            result = "WARN"
        else:
            result = "OK"
        final_score = dipole_score
    else:
        result = "OK"
        if any(ev.level == "NG" for ev in events):
            result = "NG"
        elif any(ev.level == "WARN" for ev in events):
            result = "WARN"
        final_score = max_peak

    # Best dipole tooth per side (for overlay / diagnostics)
    best_dipole_L = max(dipoles_L, key=lambda d: d["dipole_score"], default=None)
    best_dipole_R = max(dipoles_R, key=lambda d: d["dipole_score"], default=None)

    _t["template_dipole_events"] = (_time.perf_counter() - _t0) * 1000
    _t["total"] = sum(_t.values())

    return {
        "result": result,
        "score": float(final_score),
        "events": [asdict(ev) for ev in events],
        "dipoles_L": dipoles_L,
        "dipoles_R": dipoles_R,
        "stats": {
            "pitch_L": int(pitch_L),
            "pitch_R": int(pitch_R),
            "thrL_hi": float(thr_info["thrL_hi"]),
            "thrL_lo": float(thr_info["thrL_lo"]),
            "thrR_hi": float(thr_info["thrR_hi"]),
            "thrR_lo": float(thr_info["thrR_lo"]),
            "left_mask_ratio": float(left_mask.mean()),
            "right_mask_ratio": float(right_mask.mean()),
            "intrude_L_max": float(np.max(intrude_L)),
            "missing_L_max": float(np.max(missing_L)),
            "intrude_R_max": float(np.max(intrude_R)),
            "missing_R_max": float(np.max(missing_R)),
            "events_total": len(events),
            "events_NG": int(sum(ev.level == "NG" for ev in events)),
            "events_WARN": int(sum(ev.level == "WARN" for ev in events)),
            "thr_intrude_L": float(thr_intr_L),
            "thr_missing_L": float(thr_miss_L),
            "thr_intrude_R": float(thr_intr_R),
            "thr_missing_R": float(thr_miss_R),
            "dipole_score": float(dipole_score),
            "dipole_max_L": float(dipole_max_L),
            "dipole_max_R": float(dipole_max_R),
            "dipole_best_L": best_dipole_L,
            "dipole_best_R": best_dipole_R,
            "max_peak_legacy": float(max_peak),
            "adj_diff_L": adj_diff_L,
            "adj_diff_R": adj_diff_R,
            "max_adj_diff": float(max_adj_diff),
            "max_residual": float(max_residual),
            "L_max_pos_residual": float(adj_diff_L["max_pos_residual"]),
            "L_max_neg_residual": float(adj_diff_L["max_neg_residual"]),
            "R_max_pos_residual": float(adj_diff_R["max_pos_residual"]),
            "R_max_neg_residual": float(adj_diff_R["max_neg_residual"]),
        },
        "params": asdict(p),
        "xL": xL,
        "xR": xR,
        "xL_score_ref": xL_score_ref,
        "xR_score_ref": xR_score_ref,
        "left_mask": left_mask,
        "right_mask": right_mask,
        "score_map": score,
        "img_shape": (h, w),
        "timing_ms": _t,
    }


def save_metal_mask_overlay(
    out_path: Path,
    img_path: Path,
    result: dict[str, Any],
) -> None:
    """Save a visual overlay showing metal mask, boundaries, and events.

    Drawing order: scoring references first (orange/green), then boundaries
    (red/cyan) on top, then event markers (magenta/yellow circles).
    """
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        return
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    out = rgb.copy()

    left_mask = result["left_mask"]
    right_mask = result["right_mask"]
    xL = result["xL"]
    xR = result["xR"]
    xL_score_ref = result["xL_score_ref"]
    xR_score_ref = result["xR_score_ref"]
    events = result["events"]

    h, w = np.asarray(left_mask).shape

    # Green overlay on metal regions
    green = np.zeros_like(out)
    green[..., 1] = 255
    mask_all = ((np.asarray(left_mask) > 0) | (np.asarray(right_mask) > 0)).astype(np.uint8)
    out = np.where(
        mask_all[..., None].astype(bool),
        (0.72 * out + 0.28 * green).astype(np.uint8),
        out,
    )

    # Draw scoring references first, then boundaries on top
    for y in range(h):
        xi = int(round(float(xL_score_ref[y])))
        if 0 <= xi < w:
            out[y, max(0, xi - 1): min(w, xi + 2)] = (255, 165, 0)  # orange
        xi = int(round(float(xR_score_ref[y])))
        if 0 <= xi < w:
            out[y, max(0, xi - 1): min(w, xi + 2)] = (0, 255, 0)    # green

    for y in range(h):
        xi = int(round(float(xL[y])))
        if 0 <= xi < w:
            out[y, max(0, xi - 1): min(w, xi + 2)] = (255, 0, 0)    # red
        xi = int(round(float(xR[y])))
        if 0 <= xi < w:
            out[y, max(0, xi - 1): min(w, xi + 2)] = (0, 255, 255)  # cyan

    # Draw event markers
    for ev in events:
        yc = (ev["y0"] + ev["y1"]) // 2
        if ev["side"] == "left":
            xc = int(round(float(xL[yc])))
            color = (255, 0, 255)   # magenta
        else:
            xc = int(round(float(xR[yc])))
            color = (255, 255, 0)   # yellow
        cv2.circle(out, (xc, yc), 4, color, -1)

    # Draw fitted line and only the 4 max-residual labels
    params = result.get("params", {})
    overlay_enabled = params.get("adj_diff_overlay", True)
    stats = result.get("stats", {})
    if overlay_enabled:
        for side_key, dot_color, txt_color, line_color, is_left in [
            ("adj_diff_L", (255, 0, 255), (255, 255, 0), (0, 255, 0), True),
            ("adj_diff_R", (255, 255, 0), (0, 255, 255), (0, 200, 0), False),
        ]:
            ad = stats.get(side_key)
            if ad is None:
                continue
            peaks = ad.get("peaks", [])
            slope = ad.get("fit_slope", 0.0)
            intercept = ad.get("fit_intercept", 0.0)
            max_pos_y = ad.get("max_pos_y", -1)
            max_neg_y = ad.get("max_neg_y", -1)

            # Draw fitted straight line
            if len(peaks) >= 2:
                y0_line = peaks[0][0]
                y1_line = peaks[-1][0]
                x0_line = int(round(slope * y0_line + intercept))
                x1_line = int(round(slope * y1_line + intercept))
                cv2.line(out, (x0_line, y0_line), (x1_line, y1_line),
                         line_color, 2)

            # Draw tooth-peak dots only (no labels)
            for cy, peak_x in peaks:
                xi = int(round(peak_x))
                if 0 <= xi < w:
                    cv2.circle(out, (xi, cy), 4, dot_color, -1)

            # Draw labels only at the 4 extreme positions
            boundary = result["xL"] if is_left else result["xR"]
            for ext_y, res_key, marker_color in [
                (max_pos_y, "max_pos_residual", (0, 255, 0)),    # green for max positive
                (max_neg_y, "max_neg_residual", (255, 0, 0)),    # red for max negative
            ]:
                if ext_y < 0 or ext_y >= h:
                    continue
                res_val = ad.get(res_key, 0.0)
                bx = int(round(float(boundary[ext_y])))
                fx = int(round(slope * ext_y + intercept))
                # Large dot at extreme position
                if 0 <= bx < w:
                    cv2.circle(out, (bx, ext_y), 7, marker_color, -1)
                # Connecting line from boundary to fitted line
                if 0 <= bx < w and 0 <= fx < w:
                    cv2.line(out, (bx, ext_y), (fx, ext_y), marker_color, 2)
                # Label (area value)
                label = f"{res_val:+.0f}"
                if is_left:
                    txt_x = max(bx, fx) + 8
                else:
                    txt_x = max(0, min(bx, fx) - 70)
                cv2.putText(
                    out, label, (txt_x, ext_y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, marker_color, 2,
                    cv2.LINE_AA,
                )

    out_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(out_path), out_bgr)


def compute_ok_metal_mask_scores(
    img_paths: list[Path],
    params: MetalMaskParams | None = None,
    progress_cb: Any = None,
) -> list[float]:
    """Compute max-peak scores from OK images for threshold estimation."""
    all_scores: list[float] = []
    for idx, img_path in enumerate(img_paths):
        result = infer_metal_mask(img_path, params)
        all_scores.append(result["score"])

        if progress_cb and len(img_paths) > 1:
            pct = 30.0 + (idx + 1) / len(img_paths) * 50.0
            progress_cb(
                pct,
                f"OK metal mask scores: {idx + 1}/{len(img_paths)} images",
            )

    return all_scores

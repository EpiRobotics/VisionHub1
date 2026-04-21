"""Glyph Synthetic-Defect + U-Net Anomaly Detection.

A fundamentally different approach from PatchCore / structural template matching.
Instead of comparing the test crop to a single mean template, this algorithm
trains a small U-Net to LOCALISE synthetic defects injected into OK samples.
At inference time the U-Net directly outputs a pixel-level defect heatmap.

Why synthetic defects work where template matching fails
--------------------------------------------------------
Template matching + SSIM scores are dominated by the OK-sample variation
(ink density, stroke thickness, slight position jitter).  A subtle defect
(e.g. a tiny notch on '2') produces a signal *within* that natural
variation, so no global threshold can separate them cleanly.

The synthetic-defect approach side-steps this problem:
* The network is trained to predict WHERE a defect is, not whether the
  image as a whole is different.
* Natural OK variation is seen many times in training (every aligned OK
  sample is a "no-defect" negative example), so the network learns to
  ignore it.
* Synthetic defects cover all the visually-plausible structural failures
  (cutout on foreground = missing stroke / notch, extra blob = ink
  splatter, region blur = faded print, patch-paste = shape inconsistency).
* At inference, the network output is a CONCENTRATED heatmap over the
  defect pixels -- even a 3x3 defect region produces a strong localised
  response, while natural variation produces a low diffuse response.

Architecture
------------
* Input: 2-channel grayscale 128x128 [test_image, class_mean_template].
  Providing the mean template explicitly turns the task into "what is
  different from expected" -- much more sample-efficient than asking the
  network to memorise each class implicitly.
* Model: Small U-Net (enc 16-32-64, bottom 128, dec 64-32-16) ~ 500k
  parameters per class.  Trains fast on CPU, very fast on GPU.
* Output: 1-channel sigmoid heatmap 128x128 (defect probability).
* Loss: BCE + Dice on the synthetic defect mask.
* Scoring: top-1% of heatmap pixels within the class foreground mask.

Same predict() interface as GlyphPatchCoreEngine / GlyphStructuralEngine
so it plugs into the existing inference pipeline unchanged.
"""

from __future__ import annotations

import json
import logging
import random
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
from app.plugins.glyph_structural_core import (
    SEARCH_PAD,
    _normalise_contrast,
    _prepare_binary,
    _prepare_grayscale,
    _resize_pad,
    _tmpl_align,
)

logger = logging.getLogger(__name__)

# Lazy torch imports (matches resnet_classify_core.py convention)
_torch: Any = None
_nn: Any = None
_F: Any = None


def _ensure_torch() -> None:
    """Lazy-import torch modules."""
    global _torch, _nn, _F
    if _torch is not None:
        return
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _torch = torch
    _nn = nn
    _F = F


# ---------------------------------------------------------------------------
# U-Net architecture
# ---------------------------------------------------------------------------

def _build_unet(in_channels: int = 2, base_ch: int = 16) -> Any:
    """Construct a small U-Net for defect localisation.

    Encoder: base_ch -> 2*base_ch -> 4*base_ch, bottom: 8*base_ch
    Decoder mirrors the encoder with skip connections.
    Output: 1-channel logits (no sigmoid -- applied in loss / inference).
    """
    _ensure_torch()
    nn = _nn
    F = _F
    torch = _torch

    def conv_block(ic: int, oc: int) -> Any:
        return nn.Sequential(
            nn.Conv2d(ic, oc, 3, padding=1),
            nn.BatchNorm2d(oc),
            nn.ReLU(inplace=True),
            nn.Conv2d(oc, oc, 3, padding=1),
            nn.BatchNorm2d(oc),
            nn.ReLU(inplace=True),
        )

    class SimpleUNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            c1 = base_ch
            c2 = base_ch * 2
            c3 = base_ch * 4
            c4 = base_ch * 8
            self.enc1 = conv_block(in_channels, c1)
            self.enc2 = conv_block(c1, c2)
            self.enc3 = conv_block(c2, c3)
            self.bottom = conv_block(c3, c4)
            self.dec3 = conv_block(c4 + c3, c3)
            self.dec2 = conv_block(c3 + c2, c2)
            self.dec1 = conv_block(c2 + c1, c1)
            self.out_conv = nn.Conv2d(c1, 1, 1)

        def forward(self, x: Any) -> Any:
            e1 = self.enc1(x)
            e2 = self.enc2(F.max_pool2d(e1, 2))
            e3 = self.enc3(F.max_pool2d(e2, 2))
            b = self.bottom(F.max_pool2d(e3, 2))
            d3 = self.dec3(torch.cat(
                [F.interpolate(b, scale_factor=2, mode="nearest"), e3], 1))
            d2 = self.dec2(torch.cat(
                [F.interpolate(d3, scale_factor=2, mode="nearest"), e2], 1))
            d1 = self.dec1(torch.cat(
                [F.interpolate(d2, scale_factor=2, mode="nearest"), e1], 1))
            return self.out_conv(d1)  # logits

    return SimpleUNet()


# ---------------------------------------------------------------------------
# Synthetic defect generators
# ---------------------------------------------------------------------------

def _synth_cutout_fg(
    img: np.ndarray,
    fg_mask: np.ndarray,
    bg_value: float,
    size_range: tuple[int, int] = (3, 14),
) -> tuple[np.ndarray, np.ndarray]:
    """Erase a small region on the foreground -- simulates missing stroke / notch.

    This is the MOST IMPORTANT defect class for glyph inspection.  A
    "2 missing a tiny corner" produces exactly this pattern: foreground
    pixels replaced by background colour over a small connected region.
    """
    h, w = img.shape
    fg_pix = np.argwhere(fg_mask > 0.3)
    if len(fg_pix) == 0:
        return img.copy(), np.zeros_like(img, dtype=np.float32)
    y, x = fg_pix[random.randrange(len(fg_pix))]
    size = random.randint(size_range[0], size_range[1])
    mask = np.zeros((h, w), dtype=np.float32)
    # Shape: mix of rect, ellipse, and irregular
    shape = random.random()
    if shape < 0.33:
        y0 = max(0, y - size // 2); y1 = min(h, y + size // 2 + 1)
        x0 = max(0, x - size // 2); x1 = min(w, x + size // 2 + 1)
        mask[y0:y1, x0:x1] = 1.0
    elif shape < 0.66:
        rx = max(1, size // 2 + random.randint(-1, 2))
        ry = max(1, size // 2 + random.randint(-1, 2))
        cv2.ellipse(mask, (int(x), int(y)), (rx, ry),
                    random.randint(0, 180), 0, 360, 1.0, -1)
    else:
        # Irregular polygon
        n_pts = random.randint(4, 7)
        pts = []
        for i in range(n_pts):
            ang = 2 * np.pi * i / n_pts + random.uniform(-0.3, 0.3)
            r = size * random.uniform(0.3, 0.7)
            pts.append((int(x + r * np.cos(ang)), int(y + r * np.sin(ang))))
        cv2.fillPoly(mask, [np.array(pts, dtype=np.int32)], 1.0)
    # Apply: set masked region to background value (with slight noise)
    noise = np.random.normal(0, 3, img.shape).astype(np.float32)
    augmented = img.astype(np.float32).copy()
    target = bg_value + noise
    augmented = np.where(mask > 0.5, target, augmented)
    return np.clip(augmented, 0, 255), mask


def _synth_blob_add(
    img: np.ndarray,
    fg_mask: np.ndarray,
    fg_value: float,
    size_range: tuple[int, int] = (2, 8),
) -> tuple[np.ndarray, np.ndarray]:
    """Add a small dark blob near (but not inside) the foreground --
    simulates ink splatter / extra dot."""
    h, w = img.shape
    # Dilate fg_mask to get "near-foreground" region
    k = max(3, min(h, w) // 20)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    near_fg = cv2.dilate((fg_mask > 0.3).astype(np.uint8), kernel) & \
              (~(fg_mask > 0.3).astype(bool)).astype(np.uint8)
    candidates = np.argwhere(near_fg > 0)
    if len(candidates) == 0:
        # Fall back to any background pixel
        candidates = np.argwhere(fg_mask < 0.3)
        if len(candidates) == 0:
            return img.copy(), np.zeros_like(img, dtype=np.float32)
    y, x = candidates[random.randrange(len(candidates))]
    size = random.randint(size_range[0], size_range[1])
    mask = np.zeros((h, w), dtype=np.float32)
    cv2.circle(mask, (int(x), int(y)), size, 1.0, -1)
    augmented = img.astype(np.float32).copy()
    noise = np.random.normal(0, 3, img.shape).astype(np.float32)
    augmented = np.where(mask > 0.5, fg_value + noise, augmented)
    return np.clip(augmented, 0, 255), mask


def _synth_patch_paste(
    img: np.ndarray,
    fg_mask: np.ndarray,
    size_range: tuple[int, int] = (6, 18),
) -> tuple[np.ndarray, np.ndarray]:
    """Copy a patch from one foreground location to another -- simulates
    structural inconsistency / wrong-looking stroke."""
    h, w = img.shape
    fg_pix = np.argwhere(fg_mask > 0.3)
    if len(fg_pix) < 2:
        return img.copy(), np.zeros_like(img, dtype=np.float32)
    src_y, src_x = fg_pix[random.randrange(len(fg_pix))]
    dst_y, dst_x = fg_pix[random.randrange(len(fg_pix))]
    if abs(src_y - dst_y) + abs(src_x - dst_x) < 4:
        # Pick a different one
        dst_y, dst_x = fg_pix[random.randrange(len(fg_pix))]
    size = random.randint(size_range[0], size_range[1])
    half = size // 2
    # Extract source patch
    sy0 = max(0, src_y - half); sy1 = min(h, src_y + half)
    sx0 = max(0, src_x - half); sx1 = min(w, src_x + half)
    patch = img[sy0:sy1, sx0:sx1].copy()
    ph, pw = patch.shape
    if ph < 3 or pw < 3:
        return img.copy(), np.zeros_like(img, dtype=np.float32)
    # Paste at destination
    dy0 = max(0, dst_y - ph // 2); dy1 = min(h, dy0 + ph)
    dx0 = max(0, dst_x - pw // 2); dx1 = min(w, dx0 + pw)
    ph2 = dy1 - dy0; pw2 = dx1 - dx0
    if ph2 < 3 or pw2 < 3:
        return img.copy(), np.zeros_like(img, dtype=np.float32)
    mask = np.zeros((h, w), dtype=np.float32)
    mask[dy0:dy1, dx0:dx1] = 1.0
    augmented = img.astype(np.float32).copy()
    augmented[dy0:dy1, dx0:dx1] = patch[:ph2, :pw2]
    return np.clip(augmented, 0, 255), mask


def _synth_region_blur(
    img: np.ndarray,
    fg_mask: np.ndarray,
    size_range: tuple[int, int] = (8, 20),
) -> tuple[np.ndarray, np.ndarray]:
    """Blur a small region around the foreground -- simulates faded print
    / partially smudged character."""
    h, w = img.shape
    fg_pix = np.argwhere(fg_mask > 0.3)
    if len(fg_pix) == 0:
        return img.copy(), np.zeros_like(img, dtype=np.float32)
    y, x = fg_pix[random.randrange(len(fg_pix))]
    size = random.randint(size_range[0], size_range[1])
    mask = np.zeros((h, w), dtype=np.float32)
    cv2.circle(mask, (int(x), int(y)), size, 1.0, -1)
    # Soften mask edges so the blur transition is smooth
    mask = cv2.GaussianBlur(mask, (0, 0), size / 4.0)
    blurred = cv2.GaussianBlur(img.astype(np.float32),
                               (0, 0), random.uniform(2.0, 4.0))
    augmented = img.astype(np.float32) * (1 - mask) + blurred * mask
    # Binarise mask back for loss
    loss_mask = (mask > 0.3).astype(np.float32)
    return np.clip(augmented, 0, 255), loss_mask


def _synth_fade(
    img: np.ndarray,
    fg_mask: np.ndarray,
    bg_value: float,
    size_range: tuple[int, int] = (10, 26),
) -> tuple[np.ndarray, np.ndarray]:
    """Gradual fade: alpha-blend a soft foreground region toward background.

    Real-world fade defects (e.g. top of "2" that partially evaporated)
    leave residual faint ink rather than pure background.  ``cutout_fg``
    replaces pixels with solid bg -- too crisp.  ``fade`` instead mixes
    fg and bg via a soft alpha to produce a gradient like real print fade.

    The centre of the fade is biased toward the *edge* of the fg mask
    (outline pixels), which is where real fade most commonly occurs.
    """
    h, w = img.shape
    fg_bin = (fg_mask > 0.3).astype(np.uint8)
    fg_pix = np.argwhere(fg_bin > 0)
    if len(fg_pix) == 0:
        return img.copy(), np.zeros_like(img, dtype=np.float32)
    # Edge pixels = dilate - erode intersected with fg (outline of glyph)
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    edge = cv2.dilate(fg_bin, kern) - cv2.erode(fg_bin, kern)
    edge_pix = np.argwhere((edge > 0) & (fg_bin > 0))
    if len(edge_pix) >= 5 and random.random() < 0.7:
        y, x = edge_pix[random.randrange(len(edge_pix))]
    else:
        y, x = fg_pix[random.randrange(len(fg_pix))]
    size = random.randint(size_range[0], size_range[1])
    alpha = np.zeros((h, w), dtype=np.float32)
    cv2.circle(alpha, (int(x), int(y)), size, 1.0, -1)
    alpha = cv2.GaussianBlur(alpha, (0, 0), size / 2.5)
    alpha = np.clip(alpha, 0.0, 1.0) * random.uniform(0.55, 0.95)
    augmented = img.astype(np.float32) * (1 - alpha) + float(bg_value) * alpha
    # Loss target: only count fg pixels that actually got faded
    loss_mask = ((alpha > 0.3) & (fg_mask > 0.3)).astype(np.float32)
    return np.clip(augmented, 0, 255), loss_mask


def _apply_random_defect(
    img: np.ndarray,
    fg_mask: np.ndarray,
    bg_value: float,
    fg_value: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply one of the synthetic defect generators at random.

    Weighted toward cutout_fg + fade because "missing stroke / faded
    corner" is the most common and most important defect class to detect.
    """
    r = random.random()
    if r < 0.30:
        return _synth_cutout_fg(img, fg_mask, bg_value)
    elif r < 0.55:
        return _synth_fade(img, fg_mask, bg_value)
    elif r < 0.75:
        return _synth_blob_add(img, fg_mask, fg_value)
    elif r < 0.90:
        return _synth_patch_paste(img, fg_mask)
    else:
        return _synth_region_blur(img, fg_mask)


# ---------------------------------------------------------------------------
# Model data
# ---------------------------------------------------------------------------

# Dilation radius applied to the foreground mask during scoring.
# matchTemplate alignment can leave residual offsets of up to ~3-5 px,
# so scoring over a slightly dilated fg mask is much more robust.
SCORE_FG_DILATE: int = 9


def _dilate_fg(foreground_mask: np.ndarray, ksize: int = SCORE_FG_DILATE) -> np.ndarray:
    """Return bool mask = foreground_mask dilated by ``ksize``."""
    fg_u8 = (foreground_mask > 0.5).astype(np.uint8)
    k = max(1, int(ksize))
    kernel = np.ones((k, k), dtype=np.uint8)
    return cv2.dilate(fg_u8, kernel) > 0


# ---------------------------------------------------------------------------
# Binary / distance-transform "shape gate"
#
# The U-Net heatmap is sensitive to *gray-level* deviations from the mean
# template -- ink density, stroke thickness, sub-pixel blur -- which means
# some OK crops that look visually identical to other OKs can still produce
# a strong heatmap response (``"1"`` class 0449 / 0410 / 0267 / 0225 /
# 0347 are examples where the heatmap said NG 0.94-0.99 but the crops are
# clearly OK).  A PoC run on the actual OK bank confirmed that when
# scoring these crops at the *binary shape* level (Otsu binarise + per-
# pixel distance-transform diff against the template binary), they are
# indistinguishable from the clean reference crops (all ``dt99`` in the
# 2-5 range vs bank p99 of 6).  Real "missing stroke / fade" defects,
# in contrast, produce a clear distance-transform difference spike
# (class "2" B.jpg i9: ``dt99 = 16`` vs cleanest ``7.59`` -- 2x margin).
#
# We therefore run a second, independent scorer on top of the U-Net:
#
#   1. At train time: binarise the mean template (Otsu), score every OK
#      bank sample's Otsu binary against the template via distance-
#      transform diff, store the bank's p_thr quantile as ``dt_thr``
#      (scaled by a safety factor, with a lower floor).  Also persist
#      ``tmpl_bw`` (template binary) and a DT-based suspects list so
#      the user can review contaminated crops.
#   2. At inference time: compute the same dt99 score on the test crop's
#      Otsu binary and flag NG only if *both* U-Net heatmap and DT
#      shape score exceed their thresholds (AND gate).
#
# Backward compatibility: models trained before this change have neither
# ``tmpl_bw`` nor ``dt_thr`` persisted.  Those models fall back to the
# pure U-Net scorer (AND gate bypassed) so existing deployments don't
# change behaviour silently.

# Safety multiplier applied to the DT p_thr quantile when setting
# ``dt_thr``.  1.3 gives ~30 % head-room above the worst clean bank
# sample while still staying well below typical real-defect scores.
DT_THR_SAFETY: float = 1.3
# Lower bound on ``dt_thr`` in raw distance-transform units (pixels).
# Prevents the threshold from collapsing to ~0 on classes whose bank
# is unusually tight (e.g. ``"1"``) and generating spurious NGs for
# sub-pixel alignment jitter.  10 px corresponds to a defect region
# roughly the size of a stroke segment -- smaller shape drift than
# that is not reliably distinguishable from alignment error.
DT_THR_FLOOR: float = 10.0


def _binarize_otsu(u8: np.ndarray) -> np.ndarray:
    """Return bool foreground (dark ink on light background) via Otsu."""
    if u8.dtype != np.uint8:
        u8 = np.clip(u8, 0, 255).astype(np.uint8)
    _, bw = cv2.threshold(u8, 0, 255,
                          cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return bw.astype(bool)


def _dt_diff_p99(
    test_bw: np.ndarray,
    tmpl_bw: np.ndarray,
    valid_mask: np.ndarray,
    top_percentile: float = 99.0,
) -> float:
    """Symmetric per-pixel distance-transform diff between two binaries.

    ``DT_fg(b)`` is the L2 distance from each pixel to the nearest
    foreground pixel in binary ``b``.  A defect that breaks the glyph
    shape (missing stroke, fade, extra blob) causes a localised spike
    in ``|DT_fg(test) - DT_fg(tmpl)|`` at the defect location.  We
    take the ``top_percentile`` of this diff *inside the dilated
    foreground mask* to remain robust to residual matchTemplate
    alignment offsets (~3-5 px).
    """
    if test_bw.shape != tmpl_bw.shape:
        return 0.0
    dt_t = cv2.distanceTransform(
        (~test_bw).astype(np.uint8), cv2.DIST_L2, 5,
    )
    dt_m = cv2.distanceTransform(
        (~tmpl_bw).astype(np.uint8), cv2.DIST_L2, 5,
    )
    diff = np.abs(dt_t - dt_m)
    d = diff[valid_mask]
    if d.size == 0:
        return 0.0
    return float(np.percentile(d, top_percentile))


@dataclass
class SynthDefectClassModel:
    """Per-class U-Net model + alignment data."""
    cls: str = ""
    img_size: int = 128
    mean_template: np.ndarray = field(default_factory=lambda: np.zeros((1, 1), dtype=np.float32))
    foreground_mask: np.ndarray = field(default_factory=lambda: np.zeros((1, 1), dtype=np.float32))
    fg_score_mask: np.ndarray = field(default_factory=lambda: np.zeros((1, 1), dtype=bool))
    bg_value: float = 255.0
    fg_value: float = 0.0
    state_dict: dict[str, Any] = field(default_factory=dict)
    thr: float = 0.0
    n_ok: int = 0
    base_ch: int = 16
    # Binary / distance-transform "shape gate".  ``tmpl_bw`` is the Otsu
    # binarisation of the mean template; ``dt_thr`` is the threshold
    # for the DT p99 score.  When either is missing (old models),
    # ``dt_thr`` stays 0.0 and the predict() path falls back to pure
    # U-Net scoring (AND gate bypassed).
    tmpl_bw: np.ndarray = field(default_factory=lambda: np.zeros((1, 1), dtype=bool))
    dt_thr: float = 0.0
    # Runtime-only (not serialised)
    model: Any = None
    device: str = "cpu"


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def _bce_dice_loss(logits: Any, target: Any) -> Any:
    """BCE + soft-Dice loss.  target: 0/1 mask, logits: raw network output."""
    F = _F
    bce = F.binary_cross_entropy_with_logits(logits, target)
    probs = _torch.sigmoid(logits)
    smooth = 1.0
    inter = (probs * target).sum(dim=(1, 2, 3))
    union = probs.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = 1 - (2 * inter + smooth) / (union + smooth)
    return bce + dice.mean()


# ---------------------------------------------------------------------------
# Training (per-class)
# ---------------------------------------------------------------------------

def _augment_ok(
    img: np.ndarray,
    mask: np.ndarray | None,
    bg_value: float,
    max_rot_deg: float,
    max_scale: float,
    max_shift_px: int,
    gauss_sigma: float,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Apply small geometric + noise augmentation to an OK image.

    Teaches the U-Net that mild natural variation (rotation, scale,
    sub-pixel shift after matchTemplate alignment, sensor noise) is OK
    and should NOT be flagged as a defect.  Without this, the model
    over-reacts to any OK sample whose pose / lighting happens to
    differ from the mean template -- even when the crop is visually
    indistinguishable from other OK crops (e.g. "1" 0449.jpg in the
    user-reported dataset).

    The same transform is applied to ``mask`` (the defect target) if
    provided, so defect locations remain aligned after augmentation.
    """
    h, w = img.shape[:2]
    rot = random.uniform(-max_rot_deg, max_rot_deg)
    scale = 1.0 + random.uniform(-max_scale, max_scale)
    tx = random.uniform(-max_shift_px, max_shift_px)
    ty = random.uniform(-max_shift_px, max_shift_px)
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), rot, scale)
    M[0, 2] += tx
    M[1, 2] += ty
    img_aug = cv2.warpAffine(
        img.astype(np.float32), M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=float(bg_value),
    )
    mask_aug = None
    if mask is not None:
        mask_aug = cv2.warpAffine(
            mask.astype(np.float32), M, (w, h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )
    if gauss_sigma > 0:
        noise = np.random.normal(0.0, gauss_sigma, img_aug.shape).astype(
            np.float32,
        )
        img_aug = img_aug + noise
    return img_aug, mask_aug


def _train_single_class(
    cls: str,
    aligned_grays: list[np.ndarray],
    mean_template: np.ndarray,
    foreground_mask: np.ndarray,
    bg_value: float,
    fg_value: float,
    img_size: int,
    device: str = "cpu",
    epochs: int = 40,
    batch_size: int = 16,
    lr: float = 2e-3,
    clean_frac: float = 0.2,
    base_ch: int = 16,
    # Default augmentation is OFF so recall on subtle real defects is
    # preserved.  Opt-in via kwargs when the user's OK bank contains
    # lots of natural pose / exposure variation on top of clean glyphs.
    aug_rot_deg: float = 0.0,
    aug_scale: float = 0.0,
    aug_shift_px: int = 0,
    aug_noise_sigma: float = 0.0,
    aug_intensity_jitter: float = 10.0,
) -> tuple[Any, list[float]]:
    """Train a U-Net for one character class using synthetic defects.

    Args:
        cls: Class label (for logging).
        aligned_grays: List of (img_size, img_size) float32 [0-255] aligned OK images.
        mean_template: (img_size, img_size) float32 [0-255] mean of aligned OKs.
        foreground_mask: (img_size, img_size) float32 in {0, 1}.
        bg_value, fg_value: scalar intensity values used when synthesising defects.
        epochs: Number of training epochs.
        batch_size: Batch size.
        lr: Learning rate.
        clean_frac: Fraction of each batch that is "no defect" (empty target mask).
            Acts as a negative example so the network doesn't over-predict.
            Bumped to 0.3 (from 0.2) to reduce false positives on OK samples
            whose natural variation resembles a subtle synthetic defect.
        aug_rot_deg: Max absolute rotation in degrees applied to each OK
            crop (and its target mask when a synth defect has been
            painted on it) per batch.  Teaches the network small pose
            variation != defect.
        aug_scale: Max relative scale (e.g. 0.03 = +/-3%).
        aug_shift_px: Max translation in pixels.
        aug_noise_sigma: Std-dev of Gaussian noise added per pixel
            (in 0-255 space).  Simulates sensor / print texture and
            prevents the network from memorising per-pixel micro-
            features that are actually OK variation.
        aug_intensity_jitter: Max absolute additive intensity shift
            applied globally (simulates exposure / ink variation).

    Returns:
        (trained_model, loss_history)
    """
    _ensure_torch()
    torch = _torch

    model = _build_unet(in_channels=2, base_ch=base_ch).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)

    # Pre-normalise mean template and images to [0, 1] for the network
    tmpl_t = torch.from_numpy(
        (mean_template / 255.0).astype(np.float32)
    ).to(device)
    fg_t = torch.from_numpy(foreground_mask.astype(np.float32)).to(device)

    aligned_np = np.stack(aligned_grays, axis=0)  # (N, H, W)

    history: list[float] = []
    n = len(aligned_grays)
    model.train()
    for ep in range(epochs):
        # Shuffle indices
        idx = np.random.permutation(n)
        ep_loss = 0.0
        n_batches = 0
        for b0 in range(0, n, batch_size):
            batch_idx = idx[b0:b0 + batch_size]
            bs = len(batch_idx)
            inputs = np.zeros((bs, 2, img_size, img_size), dtype=np.float32)
            targets = np.zeros((bs, 1, img_size, img_size), dtype=np.float32)
            for i, ii in enumerate(batch_idx):
                base = aligned_np[ii]
                if random.random() < clean_frac:
                    # Clean example: no defect
                    aug = base.astype(np.float32).copy()
                    mask = np.zeros_like(base, dtype=np.float32)
                else:
                    aug, mask = _apply_random_defect(
                        base, foreground_mask, bg_value, fg_value,
                    )
                # Geometric / noise augmentation -- ALWAYS on, whether
                # clean or defect example, so rotation/scale/noise are
                # trained as NON-defect variation.
                aug, mask = _augment_ok(
                    aug.astype(np.float32), mask,
                    bg_value=bg_value,
                    max_rot_deg=aug_rot_deg,
                    max_scale=aug_scale,
                    max_shift_px=aug_shift_px,
                    gauss_sigma=aug_noise_sigma,
                )
                # Global intensity jitter (simulate exposure / ink variation)
                jitter = random.uniform(-aug_intensity_jitter, aug_intensity_jitter)
                aug = np.clip(aug + jitter, 0, 255)
                inputs[i, 0] = aug / 255.0
                inputs[i, 1] = mean_template / 255.0
                targets[i, 0] = mask if mask is not None else 0.0

            x = torch.from_numpy(inputs).to(device)
            y = torch.from_numpy(targets).to(device)
            logits = model(x)
            loss = _bce_dice_loss(logits, y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            ep_loss += float(loss.item())
            n_batches += 1

        scheduler.step()
        avg_loss = ep_loss / max(1, n_batches)
        history.append(avg_loss)
        if (ep + 1) % 10 == 0 or ep == 0:
            logger.info(
                "Class %s: epoch %d/%d loss=%.4f",
                cls, ep + 1, epochs, avg_loss,
            )

    # Free-form checks on the foreground mask during inference happen
    # outside this function.
    return model, history


# Morphological opening kernel size applied to the heatmap before
# reading the percentile.  A previous iteration defaulted this to 5
# in an attempt to suppress single-pixel noise on thin-stroke classes
# like "1", but that also crushed small real defects (e.g. the top-
# fade on "2" B.jpg i9 dropped 0.986 -> 0.593).  We now default to 1
# (no-op) and rely instead on (a) geometric + Gaussian-noise training
# augmentation to teach natural OK variation, and (b) manually
# reviewing the ``<cls>_suspects.txt`` list emitted at training time
# to remove contaminated OK crops.  Leave this at 1 unless you are
# certain your class has dense single-pixel noise -- and confirm
# recall is preserved after increasing it.
SCORE_OPENING_K: int = 1


def _score_heat_inplace(heat: np.ndarray, opening_k: int) -> np.ndarray:
    """Return heatmap after a morphological open of size ``opening_k``.

    If ``opening_k <= 1`` the heatmap is returned unchanged.
    """
    if opening_k <= 1:
        return heat
    kern = np.ones((int(opening_k), int(opening_k)), dtype=np.uint8).astype(np.float32)
    return cv2.morphologyEx(heat, cv2.MORPH_OPEN, kern)


def _score_batch(
    model: Any,
    aligned_imgs: list[np.ndarray],
    mean_template: np.ndarray,
    foreground_mask: np.ndarray,
    img_size: int,
    device: str,
    batch_size: int = 32,
    top_percentile: float = 99.0,
    fg_score_mask: np.ndarray | None = None,
    opening_k: int = SCORE_OPENING_K,
) -> list[float]:
    """Run trained model on aligned images, return per-image scores.

    Score = ``percentile(open(heatmap), top_percentile)`` within the
    *dilated* foreground mask.  Dilation absorbs small alignment
    jitter from ``_tmpl_align`` (~3-5 px).  Morphological opening on
    the heatmap suppresses isolated single-pixel responses that a
    natural stroke / edge variation can produce on classes like "1".
    """
    _ensure_torch()
    torch = _torch
    model.eval()
    tmpl_np = (mean_template / 255.0).astype(np.float32)
    if fg_score_mask is None:
        fg_score_mask = _dilate_fg(foreground_mask)
    if fg_score_mask.sum() < 10:
        fg_score_mask = np.ones_like(foreground_mask, dtype=bool)
    scores: list[float] = []
    with torch.no_grad():
        for b0 in range(0, len(aligned_imgs), batch_size):
            batch = aligned_imgs[b0:b0 + batch_size]
            bs = len(batch)
            x = np.zeros((bs, 2, img_size, img_size), dtype=np.float32)
            for i, g in enumerate(batch):
                x[i, 0] = g / 255.0
                x[i, 1] = tmpl_np
            x_t = torch.from_numpy(x).to(device)
            logits = model(x_t)
            probs = torch.sigmoid(logits).cpu().numpy()  # (bs, 1, H, W)
            for i in range(bs):
                heat = _score_heat_inplace(probs[i, 0], opening_k)
                fg_vals = heat[fg_score_mask]
                s = float(np.percentile(fg_vals, top_percentile))
                scores.append(s)
    return scores


# ---------------------------------------------------------------------------
# Inference engine
# ---------------------------------------------------------------------------

class GlyphSynthDefectEngine:
    """Inference engine for synthetic-defect U-Net models.

    Same predict() interface as GlyphPatchCoreEngine / GlyphStructuralEngine.
    """

    def __init__(self, model_dir: str, device: str = "cpu") -> None:
        _ensure_torch()
        self.model_dir = model_dir
        self.device = device
        self.cls_models: dict[str, SynthDefectClassModel] = {}
        self._load_models()

    def _load_models(self) -> None:
        """Load all .joblib files in model_dir that belong to synthdefect."""
        import joblib
        torch = _torch

        model_path = Path(self.model_dir)
        for jf in sorted(model_path.glob("*.joblib")):
            try:
                data = joblib.load(jf)
                if data.get("algo_method") != "synthdefect":
                    continue
                cls = data["cls"]
                base_ch = int(data.get("base_ch", 16))
                model = _build_unet(in_channels=2, base_ch=base_ch).to(self.device)
                model.load_state_dict(data["state_dict"])
                model.eval()
                foreground_mask = data["foreground_mask"]
                fg_score_mask = data.get("fg_score_mask")
                if fg_score_mask is None:
                    fg_score_mask = _dilate_fg(foreground_mask)
                # DT shape-gate artefacts (may be absent on old models)
                tmpl_bw = data.get("tmpl_bw")
                if tmpl_bw is None:
                    tmpl_bw = np.zeros_like(foreground_mask, dtype=bool)
                else:
                    tmpl_bw = np.asarray(tmpl_bw).astype(bool)
                dt_thr = float(data.get("dt_thr", 0.0))
                m = SynthDefectClassModel(
                    cls=cls,
                    img_size=int(data.get("img_size", 128)),
                    mean_template=data["mean_template"],
                    foreground_mask=foreground_mask,
                    fg_score_mask=fg_score_mask,
                    bg_value=float(data.get("bg_value", 255.0)),
                    fg_value=float(data.get("fg_value", 0.0)),
                    state_dict=data["state_dict"],
                    thr=float(data.get("thr", 0.0)),
                    n_ok=int(data.get("n_ok", 0)),
                    base_ch=base_ch,
                    tmpl_bw=tmpl_bw,
                    dt_thr=dt_thr,
                    model=model,
                    device=self.device,
                )
                self.cls_models[cls] = m
                logger.info(
                    "Loaded synthdefect model: class=%s, thr=%.6f, "
                    "dt_thr=%.3f, n_ok=%d",
                    cls, m.thr, m.dt_thr, m.n_ok,
                )
            except Exception:
                logger.exception("Failed to load synthdefect model: %s", jf)

    def unload(self) -> None:
        """Release model resources."""
        self.cls_models.clear()

    def _score_single(
        self,
        model: SynthDefectClassModel,
        crop_gray: np.ndarray,
    ) -> tuple[float, float, np.ndarray]:
        """Compute U-Net heatmap score, DT shape score, and heatmap.

        Returns ``(unet_score, dt_score, heatmap)``.  ``dt_score`` is
        ``0.0`` when the loaded model has no persisted ``tmpl_bw``
        (old models) -- callers interpret ``dt_score=0.0`` combined
        with ``dt_thr=0.0`` as "DT gate disabled, fall back to the
        U-Net score".
        """
        torch = _torch
        aligned, _mv = _tmpl_align(
            crop_gray, model.mean_template, model.img_size, SEARCH_PAD,
        )
        tmpl_np = (model.mean_template / 255.0).astype(np.float32)
        x = np.zeros((1, 2, model.img_size, model.img_size), dtype=np.float32)
        x[0, 0] = aligned / 255.0
        x[0, 1] = tmpl_np
        with torch.no_grad():
            x_t = torch.from_numpy(x).to(model.device)
            logits = model.model(x_t)
            heat = torch.sigmoid(logits).cpu().numpy()[0, 0]
        # Score = top 1% of heatmap within dilated foreground region.
        # Dilation absorbs matchTemplate alignment jitter (~3-5 px),
        # which is critical for detecting defects right at glyph edges.
        fg_mask = model.fg_score_mask
        if fg_mask is None or fg_mask.sum() < 10:
            fg_mask = _dilate_fg(model.foreground_mask)
        if fg_mask.sum() < 10:
            fg_mask = np.ones_like(model.foreground_mask, dtype=bool)
        heat_s = _score_heat_inplace(heat, SCORE_OPENING_K)
        fg_vals = heat_s[fg_mask]
        unet_score = float(np.percentile(fg_vals, 99.0))

        # DT shape-gate score: computed only when the model has a
        # persisted tmpl_bw AND a non-zero dt_thr.  Otherwise return
        # 0.0 so the predict() path knows to skip the AND gate.
        #
        # The DT valid mask is built by dilating ``tmpl_bw`` directly
        # (NOT by reusing ``fg_score_mask``, which is derived from the
        # unaligned ``_prepare_binary`` probability map and is much
        # tighter -- it focuses on the few pixels that are foreground
        # in 50%+ of unaligned samples, which over-weights the glyph
        # core and inflates DT p99 for visually OK crops that happen
        # to deviate there).  Dilating ``tmpl_bw`` gives a mask that
        # matches the union of aligned-Otsu foreground plus a
        # matchTemplate-jitter safety halo, which is the behaviour
        # the PoC validated (bank p99 dt=6 for '1', defect dt=16 for
        # '2').
        dt_score = 0.0
        if (
            model.tmpl_bw is not None
            and model.tmpl_bw.shape == aligned.shape
            and bool(model.tmpl_bw.any())
        ):
            test_bw = _binarize_otsu(aligned.astype(np.uint8))
            dt_valid = np.asarray(
                _dilate_fg(model.tmpl_bw.astype(np.float32)),
            ).astype(bool)
            dt_score = _dt_diff_p99(test_bw, model.tmpl_bw, dt_valid)
        return unet_score, dt_score, heat

    def predict(
        self,
        image_path: str,
        json_path: str,
        output_overlay: str | None = None,
        pad: int = 2,
        thr_global: float | None = None,
    ) -> dict[str, Any]:
        """Run synthdefect inference on a single image."""
        t0 = time.perf_counter()

        img_bgr = imread_any_bgr(Path(image_path))
        if img_bgr is None:
            return {
                "score": 0.0, "threshold": 0.0, "pred": "OK",
                "regions": [], "artifacts": {},
                "timing_ms": {}, "glyph_total": 0,
                "ng_count": 0, "unk_count": 0,
            }

        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        ih, iw = img_gray.shape[:2]

        with open(json_path, "r", encoding="utf-8-sig") as f:
            ann = json.load(f)
        items = ann.get("items", [])

        overlay = img_bgr.copy() if output_overlay else None
        regions: list[dict[str, Any]] = []
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

            unet_score, dt_score, _heat = self._score_single(model, crop_gray)

            thr = thr_global if thr_global is not None else model.thr
            # AND gate: NG requires *both* the U-Net heatmap and the
            # DT shape score to exceed their thresholds.  When the
            # loaded model has no DT metadata persisted (``dt_thr==0``)
            # the AND gate is bypassed and the behaviour matches the
            # pre-DT U-Net-only path.
            unet_ng = unet_score > thr
            if model.dt_thr > 0.0:
                dt_ng = dt_score > model.dt_thr
                pred = "NG" if (unet_ng and dt_ng) else "OK"
            else:
                pred = "NG" if unet_ng else "OK"
            if pred == "NG":
                ng_count += 1

            max_score = max(max_score, unet_score)
            max_thr = max(max_thr, thr)

            regions.append({
                "ch": ch, "cx": cx, "cy": cy, "w": w, "h": h,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "score": round(unet_score, 6), "threshold": round(thr, 6),
                "dt_score": round(dt_score, 4),
                "dt_threshold": round(model.dt_thr, 4),
                "pred": pred,
            })

            if overlay is not None:
                color = (0, 0, 255) if pred == "NG" else (0, 200, 0)
                thickness = 2 if pred == "NG" else 1
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness)
                if model.dt_thr > 0.0:
                    label = f"{ch}:{unet_score:.3f}/{dt_score:.1f}"
                else:
                    label = f"{ch}:{unet_score:.4f}"
                cv2.putText(overlay, label, (x1, y1 - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

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
# Training entry point (per-class)
# ---------------------------------------------------------------------------

def train_glyph_synthdefect(
    bank_dir: str,
    out_model_dir: str,
    img_size: int = 128,
    essential_thr: float = 0.5,
    epochs: int = 25,
    batch_size: int = 16,
    lr: float = 2e-3,
    p_thr: float = 0.97,
    min_per_class: int = 10,
    device: str = "auto",
    progress_cb: Any = None,
    val_frac: float = 0.15,
    base_ch: int = 16,
    thr_safety: float = 1.15,
    thr_floor: float = 0.30,
    suspect_top_k: int = 10,
    clean_frac: float = 0.2,
    dt_thr_safety: float = DT_THR_SAFETY,
    dt_thr_floor: float = DT_THR_FLOOR,
    dt_p_thr: float = 0.90,
) -> dict[str, Any]:
    """Train synthetic-defect U-Net models per character class.

    Training pipeline per class:
    1. Load all OK images, CLAHE normalise, resize.
    2. Build initial mean via centroid alignment + 2 rounds of
       matchTemplate refinement (reuses v5.1's refined-training recipe).
    3. Randomly hold out ``val_frac`` of images as a **validation split**
       for threshold calibration -- scoring on the training split itself
       yields over-optimistic / inflated scores and an unusable threshold.
    4. Synthesise defects on-the-fly during training; teach the U-Net
       to predict WHERE the defect is.
    5. After training, score the held-out validation split and set the
       threshold at ``p_thr`` quantile (multiplied by ``thr_safety``).
    6. Save state_dict + mean template + foreground mask in a .joblib.

    Args:
        bank_dir: Glyph bank directory with one subdirectory per class.
        out_model_dir: Output directory for model files.
        img_size: Resize each glyph crop to this square size.
        essential_thr: Probability threshold for foreground mask (0..1).
        epochs: Training epochs per class.
        batch_size: Training batch size.
        lr: Initial learning rate.
        p_thr: Quantile of OK score distribution for threshold.
        min_per_class: Minimum OK images per class to train.
        device: "cuda", "cpu", or "auto".
        progress_cb: Optional callback(progress_pct, message).
        val_frac: Fraction of OK images held out of training for
            threshold calibration (default 0.15).
        base_ch: Base channel count of the U-Net (controls capacity).
        thr_safety: Multiplicative safety margin above the val quantile
            when setting the final threshold (e.g. 1.15 = +15%).  The
            final threshold is ``max(val p_thr * thr_safety, thr_floor)``.
            A previous version also clamped to ``val_max * 1.02`` but
            that forces the threshold to saturate (near 1.0) whenever
            one or two OK crops in the val split look defect-like,
            which makes real defects undetectable.  Tightening OK data
            quality is now flagged via ``suspect_top_k`` below.
        thr_floor: Minimum allowed threshold (default 0.3).  Prevents
            the threshold from collapsing toward zero on classes where
            almost every OK val sample scores near zero (otherwise a
            single marginal OK sample could flip to NG).
        suspect_top_k: After training each class, log the ``k`` highest-
            scoring OK training samples.  These are the most likely
            contamination / outlier crops -- reviewing and removing
            them from the OK bank usually fixes the last few FPs.

    Returns:
        Training report dict.
    """
    _ensure_torch()
    torch = _torch
    import joblib

    # Resolve device
    if device == "auto":
        eff_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        eff_device = device

    bank_path = Path(bank_dir)
    out_path = Path(out_model_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    class_dirs = [p for p in bank_path.iterdir() if p.is_dir()]
    if not class_dirs:
        raise RuntimeError(f"No class folders found in {bank_dir}")

    index: dict[str, Any] = {
        "algo_method": "synthdefect",
        "model_version": "v2_unet_fade",
        "img_size": img_size,
        "essential_thr": essential_thr,
        "p_thr": p_thr,
        "epochs": epochs,
        "base_ch": base_ch,
        "val_frac": val_frac,
        "thr_safety": thr_safety,
        "thr_floor": thr_floor,
        "device": eff_device,
        "classes": [],
    }

    total_classes = len(class_dirs)
    trained_classes = 0

    for ci, cls_dir in enumerate(sorted(class_dirs, key=lambda p: p.name)):
        cls = cls_dir.name
        imgs = list_images(cls_dir)

        if len(imgs) < min_per_class:
            logger.warning(
                "Skip class %s: %d images < min %d",
                cls, len(imgs), min_per_class,
            )
            continue

        if progress_cb:
            pct = 5.0 + 90.0 * ci / total_classes
            progress_cb(pct, f"[{cls}] preparing ({len(imgs)} OK images)")

        # Step 1: Load raw images
        raw_images: list[np.ndarray] = []
        for p in imgs:
            g = imread_any_gray(p)
            if g is not None:
                raw_images.append(g)
        if len(raw_images) < min_per_class:
            logger.warning("Skip class %s: only %d valid images", cls, len(raw_images))
            continue

        # Step 2a: initial mean via centroid alignment
        grays = [_prepare_grayscale(g, img_size) for g in raw_images]
        binaries = [_prepare_binary(g, img_size) for g in raw_images]
        stack = np.stack(grays, axis=0).astype(np.float32)
        mean_template = stack.mean(axis=0).astype(np.float32)

        # Step 2b: refine alignment via matchTemplate (2 rounds)
        for _ in range(2):
            refined = []
            for g in raw_images:
                aligned, _mv = _tmpl_align(
                    g, mean_template, img_size, SEARCH_PAD,
                )
                refined.append(aligned.astype(np.float32))
            stack = np.stack(refined, axis=0)
            mean_template = stack.mean(axis=0).astype(np.float32)
        aligned_grays_all: list[np.ndarray] = [s for s in stack]

        # Step 2c: train / val split (deterministic per class)
        n_total = len(aligned_grays_all)
        n_val = max(
            min_per_class // 2, int(round(n_total * max(0.0, val_frac))),
        )
        n_val = min(n_val, max(0, n_total - min_per_class))
        split_rng = random.Random(hash(f"synthdefect-split-{cls}") & 0xFFFFFFFF)
        perm = list(range(n_total))
        split_rng.shuffle(perm)
        val_idx = set(perm[:n_val])
        aligned_train = [
            aligned_grays_all[i] for i in range(n_total) if i not in val_idx
        ]
        aligned_val = [aligned_grays_all[i] for i in sorted(val_idx)]

        # Step 3: foreground mask + bg/fg intensity values
        prob_map = np.mean(binaries, axis=0).astype(np.float32)
        foreground_mask = (prob_map > essential_thr).astype(np.float32)
        fg_pixel_count = int(foreground_mask.sum())
        if fg_pixel_count < 50:
            # Fallback: use a looser threshold
            foreground_mask = (prob_map > 0.3).astype(np.float32)
            fg_pixel_count = int(foreground_mask.sum())
        fg_bool = foreground_mask > 0.5
        bg_bool = ~fg_bool
        bg_value = float(mean_template[bg_bool].mean()) if bg_bool.any() else 255.0
        fg_value = float(mean_template[fg_bool].mean()) if fg_bool.any() else 0.0
        fg_score_mask = _dilate_fg(foreground_mask)

        if progress_cb:
            pct = 5.0 + 90.0 * (ci + 0.3) / total_classes
            progress_cb(
                pct,
                f"[{cls}] training U-Net ({epochs} epochs, {eff_device}, "
                f"train={len(aligned_train)} val={len(aligned_val)})",
            )

        # Step 4: train U-Net on the training split only
        t_train_start = time.perf_counter()
        model, loss_hist = _train_single_class(
            cls=cls,
            aligned_grays=aligned_train,
            mean_template=mean_template,
            foreground_mask=foreground_mask,
            bg_value=bg_value,
            fg_value=fg_value,
            img_size=img_size,
            device=eff_device,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            base_ch=base_ch,
            clean_frac=clean_frac,
        )
        t_train_s = time.perf_counter() - t_train_start

        # Step 5: score train/val and set threshold from the VAL split
        train_scores = _score_batch(
            model=model,
            aligned_imgs=aligned_train,
            mean_template=mean_template,
            foreground_mask=foreground_mask,
            img_size=img_size,
            device=eff_device,
            batch_size=32,
            top_percentile=99.0,
            fg_score_mask=fg_score_mask,
        )
        if len(aligned_val) > 0:
            val_scores = _score_batch(
                model=model,
                aligned_imgs=aligned_val,
                mean_template=mean_template,
                foreground_mask=foreground_mask,
                img_size=img_size,
                device=eff_device,
                batch_size=32,
                top_percentile=99.0,
                fg_score_mask=fg_score_mask,
            )
        else:
            val_scores = list(train_scores)

        val_p = float(np.quantile(val_scores, p_thr))
        val_max = float(np.max(val_scores))
        # Threshold = max(safety * p_thr-quantile, thr_floor).  We used
        # to clamp thr up to ``val_max * 1.02`` too, but that forces
        # thr to saturate near 1.0 whenever a single val sample scores
        # high (happens on classes with narrow strokes + natural shape
        # variation like "1").  Instead we apply a morphological open
        # in _score_batch to suppress single-pixel noise, and report
        # the top-k highest OK training scores so the user can review
        # suspected contamination or outlier crops.
        thr = max(val_p * float(thr_safety), float(thr_floor))
        thr = float(thr)
        mean_score = float(np.mean(val_scores))
        std_score = float(np.std(val_scores))

        # Shared bookkeeping for both the U-Net and DT suspects lists.
        train_imgs_paths = [p for p in imgs if imread_any_gray(p) is not None]
        top_k = max(0, int(suspect_top_k))

        # Step 5b: DT shape-gate statistics + threshold.  Binarise the
        # mean template (Otsu) and score every OK bank sample's Otsu
        # binary against it via dt99 distance-transform diff.  The
        # threshold follows the same pattern as the U-Net one:
        #   dt_thr = max(bank_dt_p99 * dt_thr_safety, dt_thr_floor).
        # A DT-based suspects list is also emitted so the user can
        # review edge-contaminated OK crops (neighbour-character
        # bleed) which typically dominate the DT tail.
        tmpl_bw = _binarize_otsu(mean_template.astype(np.uint8))
        # IMPORTANT: the DT valid mask must match the one used at
        # inference (see GlyphSynthDefectEngine._score_single).  Do NOT
        # reuse ``fg_score_mask`` here -- that mask is derived from the
        # unaligned ``_prepare_binary`` probability map and is much
        # tighter than the aligned-Otsu foreground, causing dt_thr to
        # be inflated by visually-OK crops that differ only in a few
        # core-stroke pixels.  Dilating ``tmpl_bw`` produces the same
        # mask the PoC validated the detection margins on.
        dt_valid_mask = np.asarray(
            _dilate_fg(tmpl_bw.astype(np.float32)),
        ).astype(bool)
        dt_scores_all: list[float] = []
        for g in aligned_grays_all:
            bw = _binarize_otsu(g.astype(np.uint8))
            dt_scores_all.append(
                _dt_diff_p99(bw, tmpl_bw, dt_valid_mask),
            )
        if dt_scores_all:
            dt_arr = np.asarray(dt_scores_all, dtype=np.float32)
            # ``dt_p_thr`` defaults to 0.90 (not ``p_thr``=0.97 used for
            # the U-Net).  The DT bank distribution is frequently
            # bimodal: 90% of OK crops cluster tightly (e.g. dt<=7),
            # while 5-10% are edge-contamination / neighbour-character
            # bleed outliers that shoot up into the 15-25 range.  A
            # p97 threshold swallows those outliers and inflates
            # ``dt_thr`` well above real defect scores (e.g. "2" i9
            # dt=16 with p97-based ``dt_thr``=28).  p90 sits below
            # the bimodal gap and stays robust to contamination while
            # leaving natural variation headroom via ``dt_thr_safety``.
            dt_bank_p = float(np.quantile(dt_arr, dt_p_thr))
            dt_bank_p97 = float(np.quantile(dt_arr, 0.97))
            dt_bank_max = float(dt_arr.max())
            dt_bank_mean = float(dt_arr.mean())
        else:
            dt_bank_p = 0.0
            dt_bank_p97 = 0.0
            dt_bank_max = 0.0
            dt_bank_mean = 0.0
        dt_thr = max(dt_bank_p * float(dt_thr_safety), float(dt_thr_floor))
        dt_thr = float(dt_thr)
        # If the p97/p90 ratio is large the bank is bimodal.  This is
        # NOT necessarily contamination -- some glyph classes (e.g.
        # "2") legitimately have a long tail of OK crops that deviate
        # at the binary-shape level (varying stroke curvature / serif
        # angles).  The AND gate handles this: those OK crops have
        # low U-Net scores, so combining "U-Net NG AND DT NG" keeps
        # them OK.  The message is informational -- inspect
        # {cls}_dt_suspects.txt if you're unsure whether the tail is
        # natural variation or real edge-contamination samples that
        # slipped in.
        if dt_bank_p > 0 and dt_bank_p97 / max(dt_bank_p, 1e-6) > 2.0:
            logger.info(
                "Class %s: DT bank is bimodal "
                "(p%.0f=%.2f, p97=%.2f, max=%.2f, %d/%d crops above "
                "%.1f).  p%.0f is used for dt_thr to stay robust; "
                "AND gate with U-Net keeps natural-variation tail OK."
                "  Inspect {cls}_dt_suspects.txt if unsure.",
                cls, dt_p_thr * 100, dt_bank_p, dt_bank_p97, dt_bank_max,
                int((dt_arr > dt_bank_p * 1.5).sum()), len(dt_arr),
                dt_bank_p * 1.5, dt_p_thr * 100,
            )

        # DT-based suspects: top-k highest-scoring OK bank samples by
        # DT shape diff.  Complements the U-Net suspects list above:
        # U-Net suspects point at gray-level outliers (ink / blur),
        # DT suspects point at shape / edge-contamination outliers.
        dt_suspects: list[dict[str, Any]] = []
        if top_k > 0 and dt_scores_all:
            scored_dt: list[tuple[float, str]] = []
            for local_i, sc in enumerate(dt_scores_all):
                if local_i < len(train_imgs_paths):
                    scored_dt.append((
                        float(sc), str(train_imgs_paths[local_i]),
                    ))
            scored_dt.sort(key=lambda t: t[0], reverse=True)
            for sc, path in scored_dt[:top_k]:
                dt_suspects.append({
                    "score": round(sc, 4), "path": path,
                })

        # Identify suspect OK training samples (top-k highest scores).
        # Returned to the caller + written to <cls>_suspects.txt so the
        # user can review and clean the OK bank iteratively.
        train_imgs_paths = [p for p in imgs if imread_any_gray(p) is not None]
        # ``train_imgs_paths`` lines up with ``raw_images`` which lines up with
        # ``aligned_grays_all`` by index.  Train / val split was built from a
        # shuffled permutation; recover the original index for each split-entry.
        train_orig_idx = [i for i in range(n_total) if i not in val_idx]
        val_orig_idx = sorted(val_idx)
        top_k = max(0, int(suspect_top_k))
        suspects: list[dict[str, Any]] = []
        if top_k > 0 and train_scores:
            # combine train + val with scores and origin paths
            scored: list[tuple[float, str, str]] = []
            for local_i, score in enumerate(train_scores):
                orig = train_orig_idx[local_i]
                if orig < len(train_imgs_paths):
                    scored.append((
                        float(score), "train",
                        str(train_imgs_paths[orig]),
                    ))
            for local_i, score in enumerate(val_scores):
                orig = val_orig_idx[local_i]
                if orig < len(train_imgs_paths):
                    scored.append((
                        float(score), "val",
                        str(train_imgs_paths[orig]),
                    ))
            scored.sort(key=lambda t: t[0], reverse=True)
            for sc, split, path in scored[:top_k]:
                suspects.append({
                    "score": round(sc, 6), "split": split, "path": path,
                })

        logger.info(
            "Class %s: trained in %.1fs, final_loss=%.4f, "
            "n_train=%d n_val=%d, thr=%.6f "
            "(val_p%.0f=%.6f val_max=%.6f thr_floor=%.3f), "
            "train_p99=%.6f, fg_pixels=%d, "
            "dt_thr=%.3f (bank_p%.0f=%.3f bank_max=%.3f dt_floor=%.2f)",
            cls, t_train_s, loss_hist[-1] if loss_hist else 0.0,
            len(aligned_train), len(aligned_val), thr, p_thr * 100, val_p,
            val_max, thr_floor,
            float(np.quantile(train_scores, 0.99)) if train_scores else 0.0,
            fg_pixel_count,
            dt_thr, p_thr * 100, dt_bank_p, dt_bank_max, dt_thr_floor,
        )
        if val_max > thr:
            # At least one val sample would be flagged as NG.  This
            # usually means those specific OK crops are contaminated
            # (neighbour-character at crop edge, misaligned bbox, or
            # a real but unlabelled defect mixed into the OK bank).
            # List the top suspects so the user can review.
            logger.warning(
                "Class %s: %d/%d OK val crops score above threshold "
                "%.4f (val_max=%.4f).  Top suspects:",
                cls,
                int(sum(1 for s in val_scores if s > thr)),
                len(val_scores), thr, val_max,
            )
            for s in suspects[:min(5, top_k)]:
                logger.warning(
                    "    [%s] %.4f  %s",
                    s["split"], s["score"], s["path"],
                )
        if suspects:
            suspect_path = out_path / f"{cls}_suspects.txt"
            try:
                suspect_path.write_text(
                    "# Top {} OK training crops by defect score (highest "
                    "first).  Review these images -- they are the most "
                    "likely contamination / outlier crops in the OK bank.\n"
                    "# Format: score\tsplit\tpath\n".format(top_k) +
                    "\n".join(
                        f"{s['score']:.6f}\t{s['split']}\t{s['path']}"
                        for s in suspects
                    ) + "\n",
                    encoding="utf-8",
                )
            except Exception as exc:  # pragma: no cover - filesystem fallback
                logger.warning(
                    "Failed to write suspects file for class %s: %s",
                    cls, exc,
                )

        if dt_suspects:
            dt_suspect_path = out_path / f"{cls}_dt_suspects.txt"
            try:
                dt_suspect_path.write_text(
                    "# Top {} OK bank crops by DT shape-diff score "
                    "(highest first).  Typically highlights edge-\n"
                    "# contaminated crops (neighbour-character bleed) or "
                    "poorly-aligned samples.  Reviewing\n"
                    "# and deleting the worst ones tightens dt_thr and "
                    "improves defect recall.\n"
                    "# Format: dt99\tpath\n".format(top_k) +
                    "\n".join(
                        f"{s['score']:.4f}\t{s['path']}"
                        for s in dt_suspects
                    ) + "\n",
                    encoding="utf-8",
                )
            except Exception as exc:  # pragma: no cover - filesystem fallback
                logger.warning(
                    "Failed to write DT suspects file for class %s: %s",
                    cls, exc,
                )

        # Step 6: save model
        state_dict_cpu = {
            k: v.detach().cpu() for k, v in model.state_dict().items()
        }
        model_data = {
            "algo_method": "synthdefect",
            "model_version": "v3_dt_gate",
            "cls": cls,
            "img_size": img_size,
            "mean_template": mean_template,
            "foreground_mask": foreground_mask,
            "fg_score_mask": fg_score_mask,
            "bg_value": bg_value,
            "fg_value": fg_value,
            "state_dict": state_dict_cpu,
            "thr": thr,
            "p_thr": p_thr,
            "thr_safety": thr_safety,
            "n_ok": len(aligned_grays_all),
            "n_train": len(aligned_train),
            "n_val": len(aligned_val),
            "val_score_p_thr": val_p,
            "val_score_max": val_max,
            "val_score_mean": mean_score,
            "val_score_std": std_score,
            "thr_floor": float(thr_floor),
            "suspects": suspects,
            "train_score_p99": (
                float(np.quantile(train_scores, 0.99)) if train_scores else 0.0
            ),
            # DT shape-gate artefacts -- added in model_version v3_dt_gate.
            # Old models (v2_unet_fade) have none of these, and loaders
            # should default ``dt_thr`` to 0.0 so the AND gate is bypassed.
            "tmpl_bw": tmpl_bw,
            "dt_thr": dt_thr,
            "dt_thr_safety": float(dt_thr_safety),
            "dt_thr_floor": float(dt_thr_floor),
            "dt_bank_p_thr": dt_bank_p,
            "dt_bank_max": dt_bank_max,
            "dt_bank_mean": dt_bank_mean,
            "dt_suspects": dt_suspects,
            "base_ch": base_ch,
            "essential_thr": essential_thr,
            "epochs": epochs,
            "final_loss": loss_hist[-1] if loss_hist else 0.0,
        }
        joblib.dump(model_data, out_path / f"{cls}.joblib")

        index["classes"].append({
            "cls": cls,
            "thr": thr,
            "dt_thr": round(dt_thr, 4),
            "n_train": len(aligned_train),
            "n_val": len(aligned_val),
            "val_score_p_thr": round(val_p, 6),
            "val_score_max": round(val_max, 6),
            "val_score_mean": round(mean_score, 6),
            "val_score_std": round(std_score, 6),
            "dt_bank_p_thr": round(dt_bank_p, 4),
            "dt_bank_max": round(dt_bank_max, 4),
            "dt_bank_mean": round(dt_bank_mean, 4),
            "fg_pixels": fg_pixel_count,
            "train_time_s": round(t_train_s, 1),
            "n_val_above_thr": int(
                sum(1 for s in val_scores if s > thr)
            ),
            "suspects": suspects[:5],
            "dt_suspects": dt_suspects[:5],
        })
        trained_classes += 1

        # Free GPU memory between classes
        del model
        if eff_device == "cuda":
            torch.cuda.empty_cache()

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

"""Panel Segmentation core algorithm module.

Lightweight semantic segmentation for separating panel (board) regions
from background (conveyor rollers, gaps, table surface) in industrial
inspection images.

Architecture: MobileNetV2 encoder + lightweight U-Net decoder.
Input: 256x256 (resized from original), grayscale or RGB.
Output: binary mask (panel=1, background=0) at original resolution.

Training: binary cross-entropy with optional dice loss on labelled masks.
Inference: forward pass + resize + threshold -> binary mask.
"""

from __future__ import annotations

import logging
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

logger = logging.getLogger(__name__)

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# ---------------------------------------------------------------------------
# Lazy torch imports
# ---------------------------------------------------------------------------

_torch: Any = None
_nn: Any = None
_F: Any = None
_transforms: Any = None
_models: Any = None


def _ensure_torch() -> None:
    """Lazy-import torch and torchvision modules."""
    global _torch, _nn, _F, _transforms, _models
    if _torch is not None:
        return
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision.models as models
    import torchvision.transforms as transforms

    _torch = torch
    _nn = nn
    _F = F
    _transforms = transforms
    _models = models


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def list_images(folder: Path) -> list[Path]:
    """List all image files in a folder recursively."""
    files = [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]
    files.sort()
    return files


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    _ensure_torch()
    _torch.manual_seed(seed)
    _torch.cuda.manual_seed_all(seed)


def imread_any(path: Path) -> np.ndarray | None:
    """Read image as BGR numpy array, handling unicode paths."""
    buf = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return img


def imwrite_any(path: Path, img: np.ndarray) -> bool:
    """Write image handling unicode paths."""
    ext = path.suffix or ".png"
    ok, buf = cv2.imencode(ext, img)
    if ok:
        buf.tofile(str(path))
    return ok


# ---------------------------------------------------------------------------
# Model: MobileNetV2 encoder + U-Net decoder
# ---------------------------------------------------------------------------


class UNetDecoder(_nn.Module if _nn is not None else object):
    """Lightweight U-Net decoder with skip connections."""

    pass  # Will be properly defined when torch is loaded


def _build_unet_decoder_class() -> type:
    """Build the UNetDecoder class after torch is imported."""
    _ensure_torch()

    class _DecoderBlock(_nn.Module):
        """Single decoder block: upsample + concat skip + conv + conv."""

        def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
            super().__init__()
            self.up = _nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            self.conv1 = _nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1, bias=False)
            self.bn1 = _nn.BatchNorm2d(out_ch)
            self.relu = _nn.ReLU(inplace=True)
            self.conv2 = _nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
            self.bn2 = _nn.BatchNorm2d(out_ch)

        def forward(self, x: Any, skip: Any | None = None) -> Any:
            x = self.up(x)
            if skip is not None:
                # Handle size mismatch from odd dimensions
                if x.shape[2:] != skip.shape[2:]:
                    x = _F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
                x = _torch.cat([x, skip], dim=1)
            x = self.relu(self.bn1(self.conv1(x)))
            x = self.relu(self.bn2(self.conv2(x)))
            return x

    class _UNetDecoder(_nn.Module):
        """Lightweight U-Net decoder for MobileNetV2 encoder.

        MobileNetV2 feature channels at different stages (for 256x256 input):
        - features[0:2]   -> 16ch,  128x128  (stride 2)
        - features[2:4]   -> 24ch,  64x64    (stride 4)
        - features[4:7]   -> 32ch,  32x32    (stride 8)
        - features[7:14]  -> 96ch,  16x16    (stride 16)
        - features[14:18] -> 320ch, 8x8      (stride 32)
        """

        def __init__(self) -> None:
            super().__init__()
            # Decoder blocks (bottom-up)
            # 320ch@8x8 -> up + cat 96ch -> 256ch@16x16
            self.dec4 = _DecoderBlock(320, 96, 256)
            # 256ch@16x16 -> up + cat 32ch -> 128ch@32x32
            self.dec3 = _DecoderBlock(256, 32, 128)
            # 128ch@32x32 -> up + cat 24ch -> 64ch@64x64
            self.dec2 = _DecoderBlock(128, 24, 64)
            # 64ch@64x64 -> up + cat 16ch -> 32ch@128x128
            self.dec1 = _DecoderBlock(64, 16, 32)
            # 32ch@128x128 -> up -> 16ch@256x256
            self.dec0 = _DecoderBlock(32, 0, 16)
            # Final 1x1 conv to single channel
            self.final = _nn.Conv2d(16, 1, 1)

        def forward(
            self, f_s2: Any, f_s4: Any, f_s8: Any, f_s16: Any, f_s32: Any
        ) -> Any:
            """Forward pass through decoder.

            Args:
                f_s2: stride-2 features (16ch)
                f_s4: stride-4 features (24ch)
                f_s8: stride-8 features (32ch)
                f_s16: stride-16 features (96ch)
                f_s32: stride-32 features (320ch) - bottleneck
            """
            x = self.dec4(f_s32, f_s16)   # 8->16
            x = self.dec3(x, f_s8)        # 16->32
            x = self.dec2(x, f_s4)        # 32->64
            x = self.dec1(x, f_s2)        # 64->128
            x = self.dec0(x)              # 128->256
            x = self.final(x)
            return x

    return _UNetDecoder


class MobileNetV2UNet:
    """MobileNetV2 encoder + U-Net decoder for binary segmentation.

    This is a factory / wrapper; the actual nn.Module is created inside
    build_model() after torch is imported.
    """

    @staticmethod
    def build_model(pretrained: bool = True) -> Any:
        """Build and return the segmentation model (nn.Module).

        Returns a nn.Module with:
            - encoder: MobileNetV2 features (frozen or trainable)
            - decoder: lightweight U-Net decoder
            - forward(x) -> logits (B, 1, H, W)
        """
        _ensure_torch()

        encoder = _models.mobilenet_v2(
            weights=_models.MobileNet_V2_Weights.DEFAULT if pretrained else None,
        ).features

        UNetDec = _build_unet_decoder_class()
        decoder = UNetDec()

        # Wrap into a single module
        class _SegModel(_nn.Module):
            def __init__(self, enc: Any, dec: Any) -> None:
                super().__init__()
                self.encoder = enc
                self.decoder = dec

                # MobileNetV2 feature extraction indices
                # features[0:2]   -> stride 2,  16ch
                # features[2:4]   -> stride 4,  24ch
                # features[4:7]   -> stride 8,  32ch
                # features[7:14]  -> stride 16, 96ch
                # features[14:18] -> stride 32, 320ch
                self._slices = [
                    (0, 2),    # s2
                    (2, 4),    # s4
                    (4, 7),    # s8
                    (7, 14),   # s16
                    (14, 18),  # s32
                ]

            def forward(self, x: Any) -> Any:
                feats = []
                for start, end in self._slices:
                    for i in range(start, end):
                        x = self.encoder[i](x)
                    feats.append(x)

                logits = self.decoder(feats[0], feats[1], feats[2], feats[3], feats[4])
                return logits

        model = _SegModel(encoder, decoder)
        return model


# ---------------------------------------------------------------------------
# Dataset for training
# ---------------------------------------------------------------------------


@dataclass
class SegSample:
    """A single segmentation training sample."""
    image_path: str
    mask_path: str


class PanelSegDataset:
    """Dataset for panel segmentation training.

    Defined at module level so multiprocessing DataLoader can pickle it.
    Inherits from torch.utils.data.Dataset at runtime (lazy import).

    Expects:
        image_dir/   -> input images (*.bmp, *.jpg, *.png, etc.)
        mask_dir/    -> binary mask images (same name, white=panel, black=bg)

    Images and masks are matched by filename stem.
    """

    # Will be set to True after __init_subclass__ / first instantiation
    _torch_base_patched: bool = False

    def __init__(
        self,
        image_dir: Path,
        mask_dir: Path,
        input_size: int = 256,
        augment: bool = True,
    ) -> None:
        _ensure_torch()
        self.input_size = input_size
        self.augment = augment

        # Match images to masks by stem
        img_files = {p.stem: p for p in list_images(image_dir)}
        mask_files = {p.stem: p for p in list_images(mask_dir)}

        self.samples: list[SegSample] = []
        for stem in sorted(img_files.keys()):
            if stem in mask_files:
                self.samples.append(SegSample(
                    image_path=str(img_files[stem]),
                    mask_path=str(mask_files[stem]),
                ))

        if not self.samples:
            logger.warning(
                "No matching image-mask pairs found in %s and %s",
                image_dir, mask_dir,
            )

        # Transforms
        self.img_transform = _transforms.Compose([
            _transforms.ToTensor(),
            _transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[Any, Any]:
        sample = self.samples[idx]

        # Read image
        img_bgr = imread_any(Path(sample.image_path))
        if img_bgr is None:
            raise RuntimeError(f"Failed to read image: {sample.image_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Read mask (grayscale)
        mask = cv2.imdecode(
            np.fromfile(sample.mask_path, dtype=np.uint8),
            cv2.IMREAD_GRAYSCALE,
        )
        if mask is None:
            raise RuntimeError(f"Failed to read mask: {sample.mask_path}")

        # Resize
        img_rgb = cv2.resize(img_rgb, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.input_size, self.input_size), interpolation=cv2.INTER_NEAREST)

        # Augmentation
        if self.augment:
            # Random horizontal flip
            if random.random() > 0.5:
                img_rgb = np.fliplr(img_rgb).copy()
                mask = np.fliplr(mask).copy()
            # Random vertical flip
            if random.random() > 0.5:
                img_rgb = np.flipud(img_rgb).copy()
                mask = np.flipud(mask).copy()
            # Random 90-degree rotation
            k = random.randint(0, 3)
            if k > 0:
                img_rgb = np.rot90(img_rgb, k).copy()
                mask = np.rot90(mask, k).copy()
            # Random brightness/contrast
            if random.random() > 0.5:
                alpha = random.uniform(0.8, 1.2)  # contrast
                beta = random.uniform(-20, 20)     # brightness
                img_rgb = np.clip(img_rgb.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

        # Convert to tensors
        img_tensor = self.img_transform(img_rgb)

        # Mask: binarize (>127 = panel = 1.0) and convert to float tensor
        mask_binary = (mask > 127).astype(np.float32)
        mask_tensor = _torch.from_numpy(mask_binary).unsqueeze(0)  # (1, H, W)

        return img_tensor, mask_tensor


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------


def dice_loss(pred: Any, target: Any, smooth: float = 1.0) -> Any:
    """Dice loss for binary segmentation.

    Args:
        pred: sigmoid probabilities (B, 1, H, W)
        target: binary masks (B, 1, H, W)
    """
    _ensure_torch()
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    return 1.0 - (2.0 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)


def combined_loss(logits: Any, target: Any, dice_weight: float = 0.5) -> Any:
    """Combined BCE + Dice loss.

    Args:
        logits: raw model output (B, 1, H, W), before sigmoid
        target: binary masks (B, 1, H, W)
        dice_weight: weight for dice loss (bce_weight = 1 - dice_weight)
    """
    _ensure_torch()
    bce = _F.binary_cross_entropy_with_logits(logits, target)
    probs = _torch.sigmoid(logits)
    dl = dice_loss(probs, target)
    return (1.0 - dice_weight) * bce + dice_weight * dl


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


@dataclass
class SegTrainConfig:
    """Training hyperparameters for panel segmentation."""
    input_size: int = 256
    batch_size: int = 8
    num_epochs: int = 50
    learning_rate: float = 1e-4
    dice_weight: float = 0.5
    freeze_encoder_epochs: int = 5
    num_workers: int = 0 if sys.platform == "win32" else 2
    augment: bool = True


@dataclass
class SegTrainResult:
    """Training result summary."""
    model_path: str = ""
    best_val_iou: float = 0.0
    best_epoch: int = 0
    num_epochs: int = 0
    train_samples: int = 0
    val_samples: int = 0
    duration_s: float = 0.0


def compute_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Compute IoU between two binary masks."""
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    if union == 0:
        return 1.0
    return float(intersection / union)


def train_panel_seg(
    train_image_dir: Path,
    train_mask_dir: Path,
    out_model_path: str,
    cfg: SegTrainConfig,
    val_image_dir: Path | None = None,
    val_mask_dir: Path | None = None,
    progress_cb: Callable[[float, str], None] | None = None,
) -> SegTrainResult:
    """Train the panel segmentation model.

    Args:
        train_image_dir: Directory with training images.
        train_mask_dir: Directory with training masks (same filenames).
        out_model_path: Path to save the best model (.pth).
        cfg: Training configuration.
        val_image_dir: Optional validation image directory.
        val_mask_dir: Optional validation mask directory.
        progress_cb: Optional progress callback.

    Returns:
        Training result with metrics.
    """
    _ensure_torch()
    from torch.utils.data import DataLoader

    t_start = time.perf_counter()

    # Resolve device
    device = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")

    # Build datasets
    train_ds = PanelSegDataset(
        image_dir=train_image_dir,
        mask_dir=train_mask_dir,
        input_size=cfg.input_size,
        augment=cfg.augment,
    )

    if len(train_ds) == 0:
        raise RuntimeError(
            f"No training samples found. Check that image and mask files "
            f"in {train_image_dir} and {train_mask_dir} have matching filenames."
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    val_ds = None
    val_loader = None
    if val_image_dir and val_mask_dir and val_image_dir.exists() and val_mask_dir.exists():
        val_ds = PanelSegDataset(
            image_dir=val_image_dir,
            mask_dir=val_mask_dir,
            input_size=cfg.input_size,
            augment=False,
        )
        if len(val_ds) > 0:
            val_loader = DataLoader(
                val_ds,
                batch_size=cfg.batch_size,
                shuffle=False,
                num_workers=cfg.num_workers,
                pin_memory=True,
            )

    # Build model
    model = MobileNetV2UNet.build_model(pretrained=True)
    model = model.to(device)

    # Optimizer
    optimizer = _torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    scheduler = _torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.num_epochs)

    # Training loop
    best_metric = 0.0
    best_epoch = 0

    if progress_cb:
        progress_cb(0.0, f"Training started: {len(train_ds)} samples, device={device}")

    for epoch in range(cfg.num_epochs):
        # Freeze encoder for first few epochs
        freeze_encoder = epoch < cfg.freeze_encoder_epochs
        for param in model.encoder.parameters():
            param.requires_grad = not freeze_encoder

        # Train
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for imgs, masks in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            logits = model(imgs)
            # Ensure logits match mask size
            if logits.shape[2:] != masks.shape[2:]:
                logits = _F.interpolate(logits, size=masks.shape[2:], mode="bilinear", align_corners=False)

            loss = combined_loss(logits, masks, dice_weight=cfg.dice_weight)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(num_batches, 1)

        # Validate
        val_iou = 0.0
        if val_loader is not None:
            model.eval()
            total_iou = 0.0
            val_count = 0
            with _torch.no_grad():
                for imgs, masks in val_loader:
                    imgs = imgs.to(device, non_blocking=True)
                    masks = masks.to(device, non_blocking=True)
                    logits = model(imgs)
                    if logits.shape[2:] != masks.shape[2:]:
                        logits = _F.interpolate(logits, size=masks.shape[2:], mode="bilinear", align_corners=False)
                    probs = _torch.sigmoid(logits)
                    pred_masks = (probs > 0.5).float()

                    for b in range(pred_masks.shape[0]):
                        pred_np = pred_masks[b, 0].cpu().numpy().astype(bool)
                        gt_np = masks[b, 0].cpu().numpy().astype(bool)
                        total_iou += compute_iou(pred_np, gt_np)
                        val_count += 1

            val_iou = total_iou / max(val_count, 1)
            metric = val_iou
        else:
            # Use negative loss as metric when no validation set
            metric = -avg_loss

        # Save best model
        if metric > best_metric or epoch == 0:
            best_metric = metric
            best_epoch = epoch
            Path(out_model_path).parent.mkdir(parents=True, exist_ok=True)
            _torch.save({
                "model_state": model.state_dict(),
                "input_size": cfg.input_size,
                "epoch": epoch,
                "best_metric": best_metric,
            }, out_model_path)

        # Progress
        if progress_cb:
            pct = (epoch + 1) / cfg.num_epochs * 100
            msg = f"Epoch {epoch + 1}/{cfg.num_epochs} loss={avg_loss:.4f}"
            if val_loader is not None:
                msg += f" val_iou={val_iou:.4f}"
            msg += f" best_epoch={best_epoch + 1}"
            progress_cb(pct, msg)

    duration_s = time.perf_counter() - t_start

    return SegTrainResult(
        model_path=out_model_path,
        best_val_iou=best_metric if val_loader is not None else 0.0,
        best_epoch=best_epoch,
        num_epochs=cfg.num_epochs,
        train_samples=len(train_ds),
        val_samples=len(val_ds) if val_ds else 0,
        duration_s=round(duration_s, 2),
    )


# ---------------------------------------------------------------------------
# Inference / prediction
# ---------------------------------------------------------------------------


def load_seg_model(model_path: str, device: str = "cpu") -> tuple[Any, int]:
    """Load a trained panel segmentation model.

    Returns:
        (model, input_size)
    """
    _ensure_torch()
    ckpt = _torch.load(model_path, map_location=device, weights_only=False)

    input_size = ckpt.get("input_size", 256)
    model = MobileNetV2UNet.build_model(pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)
    model.eval()

    return model, input_size


def predict_panel_mask(
    model: Any,
    image_path: str,
    input_size: int = 256,
    device: str = "cpu",
    threshold: float = 0.5,
    morph_close_ksize: int = 15,
    morph_open_ksize: int = 5,
) -> np.ndarray:
    """Run panel segmentation on a single image.

    Args:
        model: Loaded segmentation model.
        image_path: Path to the input image.
        input_size: Model input resolution.
        device: torch device string.
        threshold: Binarization threshold for the probability map.
        morph_close_ksize: Kernel size for morphological closing (fill holes).
        morph_open_ksize: Kernel size for morphological opening (remove noise).

    Returns:
        Binary mask (H, W) uint8, 255=panel, 0=background, at original resolution.
    """
    _ensure_torch()

    # Read image
    img_bgr = imread_any(Path(image_path))
    if img_bgr is None:
        raise RuntimeError(f"Failed to read image: {image_path}")
    orig_h, orig_w = img_bgr.shape[:2]

    # Preprocess
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (input_size, input_size), interpolation=cv2.INTER_LINEAR)

    transform = _transforms.Compose([
        _transforms.ToTensor(),
        _transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    inp = transform(img_resized).unsqueeze(0).to(device)

    # Inference
    with _torch.no_grad():
        logits = model(inp)
        probs = _torch.sigmoid(logits)

    # Convert to numpy
    prob_map = probs[0, 0].cpu().numpy()  # (input_size, input_size)

    # Resize to original resolution
    prob_map_full = cv2.resize(prob_map, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

    # Binarize
    mask = (prob_map_full > threshold).astype(np.uint8) * 255

    # Morphological post-processing
    if morph_close_ksize > 0:
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_close_ksize, morph_close_ksize))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

    if morph_open_ksize > 0:
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_open_ksize, morph_open_ksize))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)

    return mask


def save_overlay(
    out_path: Path,
    image_path: str,
    mask: np.ndarray,
    alpha: float = 0.3,
) -> None:
    """Save a visualization overlay of the panel mask on the original image.

    Panel region is tinted green, background region is tinted red.

    Args:
        out_path: Output file path.
        image_path: Original image path.
        mask: Binary mask (H, W), 255=panel, 0=background.
        alpha: Overlay transparency.
    """
    img_bgr = imread_any(Path(image_path))
    if img_bgr is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    overlay = img_bgr.copy()
    # Green for panel
    panel_region = mask > 127
    overlay[panel_region] = (
        overlay[panel_region].astype(np.float32) * (1 - alpha)
        + np.array([0, 255, 0], dtype=np.float32) * alpha
    ).astype(np.uint8)

    # Red for background
    bg_region = ~panel_region
    overlay[bg_region] = (
        overlay[bg_region].astype(np.float32) * (1 - alpha)
        + np.array([0, 0, 255], dtype=np.float32) * alpha
    ).astype(np.uint8)

    # Draw contours of the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 255), 2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    imwrite_any(out_path, overlay)


def compute_panel_ratio(mask: np.ndarray) -> float:
    """Compute the ratio of panel pixels in the mask.

    Args:
        mask: Binary mask (H, W), 255=panel, 0=background.

    Returns:
        Ratio of panel pixels (0.0 to 1.0).
    """
    total = mask.shape[0] * mask.shape[1]
    panel = np.count_nonzero(mask > 127)
    return float(panel / total) if total > 0 else 0.0

"""ResNet18 binary classification core algorithm module.

Supervised OK/NG classification using ResNet18 fine-tuning.
Adapted from ttr/train_weld_cls.py, ttr/run_weld_sort.py, ttr/server.py.

Key differences from glyph PatchCore (glyph_patchcore_core.py):
- Supervised binary classification (OK/NG) instead of unsupervised anomaly detection
- Training requires both OK and NG labelled samples (ImageFolder structure)
- Single .pth model file (not per-class .joblib files)
- Inference: softmax probabilities instead of kNN distance scores

Inference input: image + JSON annotation (region positions) -> per-region OK/NG
Training input: ImageFolder (train/OK/*.jpg, train/NG/*.jpg) or JSON-based crop
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

logger = logging.getLogger(__name__)

# Image extensions supported
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# Lazy torch imports
_torch: Any = None
_nn: Any = None
_transforms: Any = None
_models: Any = None
_datasets: Any = None


def _ensure_torch() -> None:
    """Lazy-import torch and torchvision modules."""
    global _torch, _nn, _transforms, _models, _datasets
    if _torch is not None:
        return
    import torch
    import torch.nn as nn
    import torchvision.datasets as datasets
    import torchvision.models as models
    import torchvision.transforms as transforms

    _torch = torch
    _nn = nn
    _transforms = transforms
    _models = models
    _datasets = datasets


def clip_box(x0: int, y0: int, x1: int, y1: int, w: int, h: int) -> tuple[int, int, int, int]:
    """Clip bounding box to image boundaries."""
    return max(0, x0), max(0, y0), min(w, x1), min(h, y1)


def imread_any_bgr(path: Path) -> np.ndarray | None:
    """Read image as BGR numpy array, handling unicode paths."""
    buf = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def imwrite_any(path: Path, img: np.ndarray) -> bool:
    """Write image handling unicode paths."""
    ext = path.suffix or ".png"
    ok, buf = cv2.imencode(ext, img)
    if ok:
        buf.tofile(str(path))
    return ok


# ---------------------------------------------------------------------------
# Model building
# ---------------------------------------------------------------------------


def build_resnet18(num_classes: int = 2, pretrained: bool = True) -> Any:
    """Build ResNet18 classifier with custom output layer."""
    _ensure_torch()
    if pretrained:
        model = _models.resnet18(weights=_models.ResNet18_Weights.DEFAULT)
    else:
        model = _models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = _nn.Linear(in_features, num_classes)
    return model


def infer_ok_ng_idx(class_to_idx: dict[str, int]) -> tuple[int, int]:
    """Infer OK/NG class indices from class_to_idx mapping."""
    lower_map = {name.lower(): idx for name, idx in class_to_idx.items()}
    idx_ok = lower_map.get("ok")
    idx_ng = lower_map.get("ng")
    if idx_ok is None or idx_ng is None:
        raise ValueError(f"Cannot find OK/NG classes in class_to_idx: {class_to_idx}")
    return idx_ok, idx_ng


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------


@dataclass
class ResNetCheckpoint:
    """Loaded ResNet classification checkpoint."""

    model: Any  # torch.nn.Module
    idx_ok: int
    idx_ng: int
    img_size: int
    class_to_idx: dict[str, int]
    device: str = "cpu"


def load_checkpoint(model_path: str, device: str = "cpu") -> ResNetCheckpoint:
    """Load a trained ResNet18 checkpoint.

    Supports two formats:
    - Dict with 'model_state', 'class_to_idx', 'idx_ok', 'idx_ng', 'img_size'
    - Raw state_dict (fallback: assumes NG=0, OK=1)
    """
    _ensure_torch()
    ckpt = _torch.load(model_path, map_location=device, weights_only=False)

    if isinstance(ckpt, dict) and "model_state" in ckpt:
        state_dict = ckpt["model_state"]
        class_to_idx = ckpt.get("class_to_idx", {"NG": 0, "OK": 1})
        idx_ok = ckpt.get("idx_ok")
        idx_ng = ckpt.get("idx_ng")
        img_size = ckpt.get("img_size", 224)
    else:
        state_dict = ckpt
        class_to_idx = {"NG": 0, "OK": 1}
        idx_ok = None
        idx_ng = None
        img_size = 224

    # Infer num_classes from last FC layer
    last_key = list(state_dict.keys())[-1]
    num_classes = state_dict[last_key].shape[0]

    model = build_resnet18(num_classes=num_classes, pretrained=False)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    # Infer OK/NG indices
    if idx_ok is None or idx_ng is None:
        try:
            idx_ok, idx_ng = infer_ok_ng_idx(class_to_idx)
        except ValueError:
            logger.warning("Cannot infer OK/NG from class_to_idx, using default NG=0, OK=1")
            idx_ng, idx_ok = 0, 1

    return ResNetCheckpoint(
        model=model,
        idx_ok=idx_ok,
        idx_ng=idx_ng,
        img_size=img_size,
        class_to_idx=class_to_idx,
        device=device,
    )


# ---------------------------------------------------------------------------
# Single-image classification
# ---------------------------------------------------------------------------


def classify_single(
    model: Any,
    img_bgr: np.ndarray,
    idx_ok: int,
    idx_ng: int,
    img_size: int,
    device: str,
    ng_threshold: float | None = None,
) -> dict[str, Any]:
    """Classify a single BGR image crop.

    Returns:
        dict with keys: pred, prob_ok, prob_ng, score
    """
    _ensure_torch()

    from PIL import Image as PILImage

    # Convert BGR to RGB PIL
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = PILImage.fromarray(rgb)

    tfm = _transforms.Compose([
        _transforms.Resize((img_size, img_size)),
        _transforms.ToTensor(),
    ])
    inp = tfm(pil_img).unsqueeze(0).to(device)

    with _torch.no_grad():
        logits = model(inp)
        probs = _torch.softmax(logits, dim=1).cpu().squeeze(0)

    prob_ok = float(probs[idx_ok])
    prob_ng = float(probs[idx_ng])

    if ng_threshold is not None:
        pred = "NG" if prob_ng > ng_threshold else "OK"
    else:
        pred_idx = int(_torch.argmax(probs).item())
        pred = "OK" if pred_idx == idx_ok else "NG"

    return {
        "pred": pred,
        "prob_ok": round(prob_ok, 6),
        "prob_ng": round(prob_ng, 6),
        "score": round(prob_ng, 6),  # Use NG probability as anomaly score
    }


# ---------------------------------------------------------------------------
# JSON-based crop + classify (unified with glyph approach)
# ---------------------------------------------------------------------------


def predict_from_json(
    model: Any,
    idx_ok: int,
    idx_ng: int,
    img_size: int,
    device: str,
    image_path: str,
    json_path: str,
    output_overlay: str | None = None,
    pad: int = 2,
    ng_threshold: float | None = None,
    batch_size: int = 16,
) -> dict[str, Any]:
    """Run per-region classification on an image using JSON annotations.

    Reads the big image, crops each region from JSON coordinates,
    classifies each crop with ResNet18, returns per-region results.

    Args:
        model: Loaded ResNet18 model.
        idx_ok: OK class index.
        idx_ng: NG class index.
        img_size: Input size for the model.
        device: torch device string.
        image_path: Path to the input image.
        json_path: Path to JSON annotation with region coordinates.
        output_overlay: Optional path to save overlay visualization.
        pad: Padding pixels around each bounding box.
        ng_threshold: NG probability threshold (None = argmax mode).
        batch_size: Batch size for inference.

    Returns:
        dict with keys: pred, score, regions, artifacts, timing_ms,
        region_total, ng_count, unk_count
    """
    _ensure_torch()

    from PIL import Image as PILImage

    t0 = time.perf_counter()

    # Read image
    bgr = imread_any_bgr(Path(image_path))
    if bgr is None:
        raise RuntimeError(f"Failed to read image: {image_path}")
    h_img, w_img = bgr.shape[:2]

    # Read JSON annotation
    json_data = json.loads(Path(json_path).read_text(encoding="utf-8-sig"))
    items = json_data.get("items", [])
    if not isinstance(items, list):
        items = []

    # Parse region items and crop
    crops: list[dict[str, Any]] = []
    vis = bgr.copy() if output_overlay else None

    for it in items:
        ch = str(it.get("ch", it.get("label", it.get("name", ""))))
        if not ch:
            ch = f"region_{len(crops)}"

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

        crop_bgr = bgr[y0:y1, x0:x1]
        if crop_bgr.size == 0:
            continue

        crops.append({
            "ch": ch,
            "box": (x0, y0, x1, y1),
            "crop": crop_bgr,
            "idx": it.get("i", it.get("idx", len(crops))),
        })

    if not crops:
        if output_overlay and vis is not None:
            imwrite_any(Path(output_overlay), vis)
        return {
            "pred": "OK",
            "score": 0.0,
            "regions": [],
            "artifacts": {"overlay": output_overlay or ""},
            "timing_ms": {"infer": 0.0, "total": 0.0},
            "region_total": 0,
            "ng_count": 0,
        }

    # ---- Batch inference ----
    t_infer0 = time.perf_counter()

    tfm = _transforms.Compose([
        _transforms.Resize((img_size, img_size)),
        _transforms.ToTensor(),
    ])

    regions: list[dict[str, Any]] = []
    ng_count = 0
    max_ng_prob = 0.0

    for batch_start in range(0, len(crops), batch_size):
        batch = crops[batch_start:batch_start + batch_size]
        tensors = []
        for c in batch:
            rgb = cv2.cvtColor(c["crop"], cv2.COLOR_BGR2RGB)
            pil_img = PILImage.fromarray(rgb)
            tensors.append(tfm(pil_img))

        inp = _torch.stack(tensors, dim=0).to(device)

        with _torch.no_grad():
            logits = model(inp)
            probs = _torch.softmax(logits, dim=1).cpu()

        for bi, c in enumerate(batch):
            prob_ok = float(probs[bi][idx_ok])
            prob_ng = float(probs[bi][idx_ng])

            if ng_threshold is not None:
                pred = "NG" if prob_ng > ng_threshold else "OK"
            else:
                pred_idx = int(_torch.argmax(probs[bi]).item())
                pred = "OK" if pred_idx == idx_ok else "NG"

            x0, y0, x1, y1 = c["box"]
            is_ng = pred == "NG"

            regions.append({
                "ch": c["ch"],
                "idx": c["idx"],
                "x": x0,
                "y": y0,
                "w": x1 - x0,
                "h": y1 - y0,
                "prob_ok": round(prob_ok, 6),
                "prob_ng": round(prob_ng, 6),
                "score": round(prob_ng, 6),
                "decision": pred,
            })

            if is_ng:
                ng_count += 1
            if prob_ng > max_ng_prob:
                max_ng_prob = prob_ng

            # Draw overlay
            if vis is not None:
                color = (0, 0, 255) if is_ng else (0, 255, 0)
                cv2.rectangle(vis, (x0, y0), (x1, y1), color, 2)
                label_text = f"{c['ch']}:{pred} {prob_ng:.2f}"
                cv2.putText(
                    vis, label_text, (x0, max(15, y0 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2,
                )

    if device == "cuda":
        _torch.cuda.synchronize()
    t_infer1 = time.perf_counter()
    infer_ms = (t_infer1 - t_infer0) * 1000.0

    # ---- Save overlay ----
    artifacts: dict[str, str] = {}
    if output_overlay and vis is not None:
        imwrite_any(Path(output_overlay), vis)
        artifacts["overlay"] = output_overlay

    t_total = (time.perf_counter() - t0) * 1000.0
    overall_pred = "NG" if ng_count > 0 else "OK"

    return {
        "pred": overall_pred,
        "score": round(float(max_ng_prob), 6),
        "regions": regions,
        "artifacts": artifacts,
        "timing_ms": {
            "infer": round(infer_ms, 2),
            "total": round(t_total, 2),
        },
        "region_total": len(crops),
        "ng_count": ng_count,
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    """Training hyperparameters."""

    img_size: int = 224
    batch_size: int = 8
    num_epochs: int = 40
    learning_rate: float = 1e-4
    brightness: float = 0.3
    contrast: float = 0.3
    rotation: int = 5
    horizontal_flip: bool = True
    use_threshold: bool = False
    ng_threshold: float = 0.5


@dataclass
class TrainResult:
    """Training result summary."""

    model_path: str = ""
    best_val_acc: float = 0.0
    best_epoch: int = 0
    num_epochs: int = 0
    class_to_idx: dict[str, int] = field(default_factory=dict)
    idx_ok: int = 0
    idx_ng: int = 0
    train_ok: int = 0
    train_ng: int = 0
    val_ok: int = 0
    val_ng: int = 0
    precision_ng: float = 0.0
    recall_ng: float = 0.0


def crop_regions_from_json_for_training(
    json_dir: Path,
    img_dir: Path,
    out_dir: Path,
    label_key: str = "label",
    pad: int = 2,
) -> dict[str, int]:
    """Crop regions from big images using JSON annotations for training.

    Each JSON item must have a 'label' field (or the key specified by label_key)
    with value "OK" or "NG" to sort into the correct training folder.

    Args:
        json_dir: Directory with JSON annotation files.
        img_dir: Directory with source images.
        out_dir: Output directory (creates OK/ and NG/ subdirectories).
        label_key: JSON item key for OK/NG label.
        pad: Padding pixels around bounding box.

    Returns:
        dict with counts: {"ok": N, "ng": N}
    """
    from PIL import Image as PILImage

    ok_dir = out_dir / "OK"
    ng_dir = out_dir / "NG"
    ok_dir.mkdir(parents=True, exist_ok=True)
    ng_dir.mkdir(parents=True, exist_ok=True)

    counts = {"ok": 0, "ng": 0}
    global_idx = 0

    for jp in sorted(json_dir.glob("*.json")):
        data = json.loads(jp.read_text(encoding="utf-8-sig"))
        image_name = data.get("image_name", "")
        if not image_name:
            continue

        img_path = img_dir / image_name
        if not img_path.exists():
            img_path = img_dir / Path(image_name).name
        if not img_path.exists():
            logger.warning("Image not found for %s: %s", jp.name, img_path)
            continue

        items = data.get("items", [])
        if not items:
            continue

        img = PILImage.open(img_path).convert("RGB")
        w_img, h_img = img.size

        for it in items:
            cx = float(it.get("cx", 0))
            cy = float(it.get("cy", 0))
            ww = float(it.get("w", 0))
            hh = float(it.get("h", 0))
            if ww <= 0 or hh <= 0:
                continue

            label = str(it.get(label_key, "")).upper()
            if label not in ("OK", "NG"):
                continue

            x0 = int(round(cx - ww / 2)) - pad
            y0 = int(round(cy - hh / 2)) - pad
            x1 = int(round(cx + ww / 2)) + pad
            y1 = int(round(cy + hh / 2)) + pad
            x0, y0, x1, y1 = clip_box(x0, y0, x1, y1, w_img, h_img)

            crop = img.crop((x0, y0, x1, y1))
            dst = ok_dir if label == "OK" else ng_dir
            out_file = dst / f"{global_idx:06d}.jpg"
            crop.save(out_file, quality=95)
            global_idx += 1

            if label == "OK":
                counts["ok"] += 1
            else:
                counts["ng"] += 1

    return counts


def train_resnet_classify(
    train_dir: str,
    val_dir: str | None,
    out_model_path: str,
    cfg: TrainConfig | None = None,
    progress_cb: Any = None,
) -> TrainResult:
    """Train ResNet18 binary classifier.

    Args:
        train_dir: Training data directory (ImageFolder: OK/, NG/ subdirs).
        val_dir: Validation data directory (same structure). If None, uses train.
        out_model_path: Path to save the trained .pth checkpoint.
        cfg: Training hyperparameters.
        progress_cb: Optional callback(progress_pct, message).

    Returns:
        TrainResult with metrics.
    """
    _ensure_torch()

    if cfg is None:
        cfg = TrainConfig()

    device_str = "cuda" if _torch.cuda.is_available() else "cpu"
    device = _torch.device(device_str)
    logger.info("Training ResNet18 classifier on device: %s", device)

    if progress_cb:
        progress_cb(2.0, f"Preparing data loaders (device={device_str})...")

    # ---- Data augmentation ----
    train_tf = _transforms.Compose([
        _transforms.Resize((cfg.img_size, cfg.img_size)),
        _transforms.ColorJitter(brightness=cfg.brightness, contrast=cfg.contrast),
        _transforms.RandomRotation(cfg.rotation),
        _transforms.RandomHorizontalFlip() if cfg.horizontal_flip else _transforms.Lambda(lambda x: x),
        _transforms.ToTensor(),
    ])

    val_tf = _transforms.Compose([
        _transforms.Resize((cfg.img_size, cfg.img_size)),
        _transforms.ToTensor(),
    ])

    # ---- Datasets ----
    train_ds = _datasets.ImageFolder(str(train_dir), transform=train_tf)

    effective_val_dir = val_dir if val_dir and Path(val_dir).exists() else train_dir
    val_ds = _datasets.ImageFolder(str(effective_val_dir), transform=val_tf)

    class_to_idx = train_ds.class_to_idx
    idx_ok, idx_ng = infer_ok_ng_idx(class_to_idx)
    logger.info("Classes: %s, idx_ok=%d, idx_ng=%d", class_to_idx, idx_ok, idx_ng)

    targets = np.array(train_ds.targets)
    num_ok = int((targets == idx_ok).sum())
    num_ng = int((targets == idx_ng).sum())
    logger.info("Train OK: %d, NG: %d", num_ok, num_ng)

    if num_ok == 0 and num_ng == 0:
        raise RuntimeError("No training images found. Check OK/ and NG/ subdirectories.")

    # ---- Weighted sampling ----
    num_classes = len(train_ds.classes)
    class_counts = np.bincount(targets, minlength=num_classes).astype(np.float32)
    class_weights = class_counts.sum() / (num_classes * (class_counts + 1e-6))
    sample_weights = class_weights[targets]

    from torch.utils.data import DataLoader, WeightedRandomSampler

    sampler = WeightedRandomSampler(
        weights=sample_weights.tolist(),
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, sampler=sampler, num_workers=0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0,
    )

    if progress_cb:
        progress_cb(5.0, f"Building model (OK={num_ok}, NG={num_ng})...")

    # ---- Model ----
    model = build_resnet18(num_classes=num_classes, pretrained=True).to(device)

    class_weights_t = _torch.tensor(class_weights, dtype=_torch.float32).to(device)
    criterion = _nn.CrossEntropyLoss(weight=class_weights_t)
    optimizer = _torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    best_val_acc = 0.0
    best_state = None
    best_epoch = 0
    best_precision_ng = 0.0
    best_recall_ng = 0.0

    for epoch in range(1, cfg.num_epochs + 1):
        # ---- Train ----
        model.train()
        total_loss = 0.0
        total = 0

        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            total += imgs.size(0)

        train_loss = total_loss / max(total, 1)

        # ---- Val ----
        model.eval()
        correct = 0
        val_total = 0
        tp = fp = fn = 0

        with _torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                labels = labels.to(device)

                logits = model(imgs)
                preds = _torch.argmax(_torch.softmax(logits, dim=1), dim=1)

                correct += (preds == labels).sum().item()
                val_total += labels.size(0)

                for p, t in zip(preds.cpu().numpy(), labels.cpu().numpy()):
                    if t == idx_ng and p == idx_ng:
                        tp += 1
                    elif t != idx_ng and p == idx_ng:
                        fp += 1
                    elif t == idx_ng and p != idx_ng:
                        fn += 1

        val_acc = correct / max(val_total, 1)
        precision_ng = tp / max(tp + fp, 1e-6)
        recall_ng = tp / max(tp + fn, 1e-6)

        if progress_cb:
            pct = 5.0 + 90.0 * epoch / cfg.num_epochs
            progress_cb(
                pct,
                f"Epoch {epoch}/{cfg.num_epochs} "
                f"loss={train_loss:.4f} acc={val_acc:.4f} "
                f"NG_P={precision_ng:.3f} NG_R={recall_ng:.3f}",
            )

        logger.info(
            "Epoch [%d/%d] loss=%.4f acc=%.4f NG_P=%.3f NG_R=%.3f",
            epoch, cfg.num_epochs, train_loss, val_acc, precision_ng, recall_ng,
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_precision_ng = precision_ng
            best_recall_ng = recall_ng
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    # ---- Save ----
    if best_state is not None:
        ckpt = {
            "model_state": best_state,
            "class_to_idx": class_to_idx,
            "idx_ok": idx_ok,
            "idx_ng": idx_ng,
            "img_size": cfg.img_size,
        }
        Path(out_model_path).parent.mkdir(parents=True, exist_ok=True)
        _torch.save(ckpt, out_model_path)
        logger.info("Best model saved to %s (val_acc=%.4f, epoch=%d)", out_model_path, best_val_acc, best_epoch)
    else:
        raise RuntimeError("No valid model was trained (no samples?)")

    # Count val samples
    val_targets = np.array(val_ds.targets)
    val_ok = int((val_targets == idx_ok).sum())
    val_ng = int((val_targets == idx_ng).sum())

    if progress_cb:
        progress_cb(100.0, f"Training complete: acc={best_val_acc:.4f} at epoch {best_epoch}")

    return TrainResult(
        model_path=out_model_path,
        best_val_acc=best_val_acc,
        best_epoch=best_epoch,
        num_epochs=cfg.num_epochs,
        class_to_idx=class_to_idx,
        idx_ok=idx_ok,
        idx_ng=idx_ng,
        train_ok=num_ok,
        train_ng=num_ng,
        val_ok=val_ok,
        val_ng=val_ng,
        precision_ng=best_precision_ng,
        recall_ng=best_recall_ng,
    )

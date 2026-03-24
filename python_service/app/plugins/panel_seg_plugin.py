"""Panel Segmentation V1 plugin - lightweight binary segmentation (panel vs background).

Wraps the MobileNetV2-UNet segmentation model into the VisionHub plugin interface.
Used as a preprocessing step to mask out non-panel regions before defect detection.

Model format:
    models/<version>/
        model.pth           # MobileNetV2-UNet checkpoint
        meta.json           # plugin metadata

TCP INFER protocol:
    {
        "cmd": "INFER",
        "job_id": "001",
        "image_path": "E:\\images\\test.bmp"
    }

Training dataset structure:
    datasets/
        images/             # Input images (*.bmp, *.jpg, *.png)
        masks/              # Binary masks (white=panel, black=background)
        val_images/         # Optional: validation images
        val_masks/          # Optional: validation masks
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from app.plugins.base import AlgoPluginBase
from app.plugins.panel_seg_core import (
    SegTrainConfig,
    SegTrainResult,
    compute_panel_ratio,
    load_seg_model,
    predict_panel_mask,
    save_overlay,
    train_panel_seg,
)
from app.plugins.registry import PluginRegistry

logger = logging.getLogger(__name__)


@PluginRegistry.register
class PanelSegV1Plugin(AlgoPluginBase):
    """
    Panel region segmentation using MobileNetV2-UNet.

    Produces a binary mask separating panel (board) from background
    (conveyor rollers, gaps, table surface). Designed to be used as
    a preprocessing step for defect detection pipelines.

    Features:
    - MobileNetV2 encoder (ImageNet pretrained) + lightweight U-Net decoder
    - Fast inference: <5ms GPU, <30ms CPU at 256x256 input
    - Morphological post-processing for clean mask edges
    - Overlay visualization (green=panel, red=background)
    - Panel coverage ratio output
    """

    name: str = "panel_seg_v1"

    def __init__(self) -> None:
        self._model: Any = None
        self._input_size: int = 256
        self._device_str: str = "cpu"
        self._config: dict[str, Any] = {}
        self._loaded: bool = False
        self._model_version: str = ""
        self._threshold: float = 0.5

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def model_version(self) -> str:
        return self._model_version

    def load(self, model_dir: str, device: str, config: dict[str, Any]) -> None:
        """Load a trained panel segmentation model."""
        self._config = config
        model_path = Path(model_dir)

        # Find .pth file
        pth_file = model_path / "model.pth"
        if not pth_file.exists():
            pth_files = list(model_path.glob("*.pth"))
            if not pth_files:
                raise FileNotFoundError(f"No .pth model file found in: {model_dir}")
            pth_file = pth_files[0]

        effective_device = device
        if device == "auto":
            import torch
            effective_device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info("Loading panel segmentation model from %s ...", pth_file)
        self._model, self._input_size = load_seg_model(str(pth_file), device=effective_device)
        self._device_str = effective_device

        # Read threshold from config
        post_cfg = config.get("postprocess", {}).get("decision", {})
        self._threshold = float(post_cfg.get("threshold", 0.5))

        # Read version from meta.json or directory name
        meta_json = model_path / "meta.json"
        if meta_json.exists():
            meta = json.loads(meta_json.read_text(encoding="utf-8"))
            self._model_version = meta.get("version", model_path.name)
        else:
            self._model_version = model_path.name

        self._loaded = True
        logger.info(
            "Panel segmentation model loaded: version=%s, device=%s, input_size=%d, threshold=%.2f",
            self._model_version, effective_device, self._input_size, self._threshold,
        )

    def unload(self) -> None:
        """Release model resources and GPU memory."""
        if self._model is not None:
            del self._model
            self._model = None
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
        self._loaded = False
        self._model_version = ""
        self._config = {}
        logger.info("Panel segmentation model unloaded.")

    def infer(self, image_path: str, config: dict[str, Any]) -> dict[str, Any]:
        """Run panel segmentation on a single image.

        Returns a binary mask and panel coverage ratio.
        If output_dir is configured, saves the mask and overlay images.
        """
        if not self._loaded or self._model is None:
            raise RuntimeError("Panel segmentation model not loaded")

        job_id = config.get("_job_id", "unknown")
        output_dir = config.get("_output_dir", "")

        # Read post-processing params from config
        post_cfg = config.get("postprocess", {})
        decision_cfg = post_cfg.get("decision", {})
        export_cfg = post_cfg.get("export", {})

        threshold = float(decision_cfg.get("threshold", self._threshold))
        morph_close = int(export_cfg.get("morph_close_ksize", 15))
        morph_open = int(export_cfg.get("morph_open_ksize", 5))
        min_panel_ratio = float(decision_cfg.get("min_panel_ratio", 0.0))

        t_infer_start = time.perf_counter()

        # Run segmentation
        mask = predict_panel_mask(
            model=self._model,
            image_path=image_path,
            input_size=self._input_size,
            device=self._device_str,
            threshold=threshold,
            morph_close_ksize=morph_close,
            morph_open_ksize=morph_open,
        )

        t_infer_end = time.perf_counter()
        infer_ms = (t_infer_end - t_infer_start) * 1000

        # Compute panel ratio
        panel_ratio = compute_panel_ratio(mask)

        # Decision: if panel ratio is below minimum, consider it as "no panel" (NG)
        if min_panel_ratio > 0 and panel_ratio < min_panel_ratio:
            pred = "NG"
            score = 1.0 - panel_ratio
        else:
            pred = "OK"
            score = panel_ratio

        # Save artifacts
        t_save_start = time.perf_counter()
        artifacts: dict[str, str] = {}

        if output_dir:
            today = datetime.now().strftime("%Y-%m-%d")
            artifacts_dir = Path(output_dir) / today / "artifacts"
            artifacts_dir.mkdir(parents=True, exist_ok=True)

            # Save binary mask
            if export_cfg.get("save_mask", True):
                mask_path = str(artifacts_dir / f"{job_id}_panel_mask.png")
                import cv2
                cv2.imwrite(mask_path, mask)
                artifacts["mask"] = mask_path

            # Save overlay
            if export_cfg.get("save_overlay", True):
                overlay_path = str(artifacts_dir / f"{job_id}_panel_overlay.jpg")
                save_overlay(
                    out_path=Path(overlay_path),
                    image_path=image_path,
                    mask=mask,
                    alpha=float(export_cfg.get("overlay_alpha", 0.3)),
                )
                artifacts["overlay"] = overlay_path

        # Fixed overlay output path (like other plugins)
        overlay_output_path = config.get("_overlay_output_path", "")
        if overlay_output_path:
            try:
                save_overlay(
                    out_path=Path(overlay_output_path),
                    image_path=image_path,
                    mask=mask,
                    alpha=float(export_cfg.get("overlay_alpha", 0.3)),
                )
                artifacts["overlay"] = overlay_output_path
            except Exception:
                logger.warning(
                    "Failed to save overlay to %s for job %s",
                    overlay_output_path, job_id, exc_info=True,
                )

        t_save_end = time.perf_counter()
        save_ms = (t_save_end - t_save_start) * 1000

        return {
            "score": round(float(score), 6),
            "threshold": round(float(threshold), 6),
            "pred": pred,
            "regions": [],
            "artifacts": artifacts,
            "timing_ms": {
                "infer": round(infer_ms, 2),
                "post": 0.0,
                "save": round(save_ms, 2),
            },
            "model_version": self._model_version,
            "panel_ratio": round(panel_ratio, 4),
        }

    def train(
        self,
        dataset_dir: str,
        out_model_dir: str,
        config: dict[str, Any],
        progress_cb: Callable[[float, str], None] | None = None,
    ) -> dict[str, Any]:
        """Train panel segmentation model.

        Dataset structure:
            dataset_dir/
                images/         # Input images
                masks/          # Binary masks (white=panel, black=background)
                val_images/     # Optional validation images
                val_masks/      # Optional validation masks
        """
        t_start = time.perf_counter()
        out_path = Path(out_model_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        if progress_cb:
            progress_cb(0.0, "Starting panel segmentation training...")

        ds_path = Path(dataset_dir)
        train_cfg = config.get("_train", {})

        # Locate directories
        image_dir = ds_path / "images"
        mask_dir = ds_path / "masks"
        val_image_dir = ds_path / "val_images"
        val_mask_dir = ds_path / "val_masks"

        if not image_dir.exists():
            raise RuntimeError(f"Training image directory not found: {image_dir}")
        if not mask_dir.exists():
            raise RuntimeError(f"Training mask directory not found: {mask_dir}")

        # Build config
        seg_cfg = SegTrainConfig(
            input_size=int(train_cfg.get("input_size", 256)),
            batch_size=int(train_cfg.get("batch_size", 8)),
            num_epochs=int(train_cfg.get("num_epochs", 50)),
            learning_rate=float(train_cfg.get("learning_rate", 1e-4)),
            dice_weight=float(train_cfg.get("dice_weight", 0.5)),
            freeze_encoder_epochs=int(train_cfg.get("freeze_encoder_epochs", 5)),
            num_workers=int(train_cfg.get("num_workers", 2)),
            augment=bool(train_cfg.get("augment", True)),
        )

        model_pth = str(out_path / "model.pth")

        # Wrap progress_cb to offset percentages
        def train_progress(pct: float, msg: str) -> None:
            if progress_cb:
                adjusted = 5.0 + pct * 0.9
                progress_cb(adjusted, msg)

        if progress_cb:
            progress_cb(2.0, f"Found images in {image_dir}, masks in {mask_dir}")

        result: SegTrainResult = train_panel_seg(
            train_image_dir=image_dir,
            train_mask_dir=mask_dir,
            out_model_path=model_pth,
            cfg=seg_cfg,
            val_image_dir=val_image_dir if val_image_dir.exists() else None,
            val_mask_dir=val_mask_dir if val_mask_dir.exists() else None,
            progress_cb=train_progress,
        )

        # Save meta.json
        version = time.strftime("%Y%m%d_%H%M%S")
        meta = {
            "algo": self.name,
            "version": version,
            "created_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "input_size": seg_cfg.input_size,
            "best_val_iou": result.best_val_iou,
            "best_epoch": result.best_epoch,
            "train_samples": result.train_samples,
            "val_samples": result.val_samples,
            "config": {
                "input_size": seg_cfg.input_size,
                "batch_size": seg_cfg.batch_size,
                "num_epochs": seg_cfg.num_epochs,
                "learning_rate": seg_cfg.learning_rate,
                "dice_weight": seg_cfg.dice_weight,
                "freeze_encoder_epochs": seg_cfg.freeze_encoder_epochs,
            },
        }
        with open(out_path / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        duration_s = time.perf_counter() - t_start

        if progress_cb:
            progress_cb(
                100.0,
                f"Training complete: iou={result.best_val_iou:.4f} "
                f"best_epoch={result.best_epoch + 1} "
                f"in {duration_s:.1f}s",
            )

        return {
            "model_version": version,
            "metrics": {
                "best_val_iou": result.best_val_iou,
                "best_epoch": result.best_epoch,
                "train_samples": result.train_samples,
                "val_samples": result.val_samples,
            },
            "threshold": 0.5,
            "duration_s": round(duration_s, 2),
        }

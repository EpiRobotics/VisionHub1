"""ResNet Classify V1 plugin - supervised binary classification (OK/NG).

Wraps the ResNet18 classification engine into the VisionHub plugin interface.
Supports JSON-based region cropping for inference (unified with glyph approach).

Model format:
    models/<version>/
        model.pth           # ResNet18 checkpoint (.pth)
        meta.json            # plugin metadata

TCP INFER protocol (same as glyph_patchcore_v1):
    {
        "cmd": "INFER",
        "job_id": "001",
        "image_path": "E:\\images\\test.jpg",
        "options": {
            "json_path": "E:\\labels\\test.json",
            "ng_threshold": 0.5
        }
    }

Training dataset structure:
    datasets/
        train/
            OK/*.jpg         # OK sample images
            NG/*.jpg         # NG sample images
        val/                 # Optional validation set (same structure)
            OK/*.jpg
            NG/*.jpg
    OR (JSON-based crop mode):
    datasets/
        ok/                  # OK source images (big images)
        json/                # JSON annotations with per-region OK/NG labels
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from app.plugins.base import AlgoPluginBase
from app.plugins.registry import PluginRegistry
from app.plugins.resnet_classify_core import (
    TrainConfig,
    crop_regions_from_json_for_training,
    load_checkpoint,
    predict_from_json,
    train_resnet_classify,
)

logger = logging.getLogger(__name__)


@PluginRegistry.register
class ResNetClassifyV1Plugin(AlgoPluginBase):
    """
    Supervised binary classification (OK/NG) using ResNet18.

    Supports JSON-based region cropping for both training and inference,
    unified with the glyph PatchCore approach.

    Features:
    - ResNet18 fine-tuning with weighted sampling and data augmentation
    - JSON-based region cropping (external software sends coordinates)
    - Batch inference with per-region OK/NG decisions
    - Overlay visualization with per-region annotations
    - Configurable NG probability threshold
    """

    name: str = "resnet_classify_v1"

    def __init__(self) -> None:
        self._model: Any = None  # torch.nn.Module
        self._idx_ok: int = 1
        self._idx_ng: int = 0
        self._img_size: int = 224
        self._class_to_idx: dict[str, int] = {}
        self._device_str: str = "cpu"
        self._config: dict[str, Any] = {}
        self._loaded: bool = False
        self._model_version: str = ""

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def model_version(self) -> str:
        return self._model_version

    def load(self, model_dir: str, device: str, config: dict[str, Any]) -> None:
        """Load a trained ResNet18 checkpoint from model_dir/model.pth."""
        self._config = config
        model_path = Path(model_dir)

        # Find .pth file
        pth_file = model_path / "model.pth"
        if not pth_file.exists():
            # Try any .pth file in the directory
            pth_files = list(model_path.glob("*.pth"))
            if not pth_files:
                raise FileNotFoundError(f"No .pth model file found in: {model_dir}")
            pth_file = pth_files[0]

        effective_device = device
        if device == "auto":
            import torch
            effective_device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info("Loading ResNet18 classifier from %s ...", pth_file)
        ckpt = load_checkpoint(str(pth_file), device=effective_device)

        self._model = ckpt.model
        self._idx_ok = ckpt.idx_ok
        self._idx_ng = ckpt.idx_ng
        self._img_size = ckpt.img_size
        self._class_to_idx = ckpt.class_to_idx
        self._device_str = effective_device

        # Read version from meta.json or directory name
        meta_json = model_path / "meta.json"
        if meta_json.exists():
            meta = json.loads(meta_json.read_text(encoding="utf-8"))
            self._model_version = meta.get("version", model_path.name)
        else:
            self._model_version = model_path.name

        self._loaded = True
        logger.info(
            "ResNet classifier loaded: version=%s, device=%s, img_size=%d, classes=%s",
            self._model_version, effective_device, self._img_size, self._class_to_idx,
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
        logger.info("ResNet classifier model unloaded.")

    def infer(self, image_path: str, config: dict[str, Any]) -> dict[str, Any]:
        """Run per-region classification on a single image.

        Requires config["_options"]["json_path"] for JSON-based region cropping.
        """
        if not self._loaded or self._model is None:
            raise RuntimeError("ResNet classifier model not loaded")

        options = config.get("_options", {})
        json_path = options.get("json_path", "")
        if not json_path:
            raise RuntimeError(
                "Missing json_path in options. "
                "ResNet classify requires a JSON annotation file with region coordinates. "
                'Send: {"cmd":"INFER",...,"options":{"json_path":"E:\\\\labels\\\\test.json"}}'
            )

        job_id = config.get("_job_id", "unknown")
        output_dir = config.get("_output_dir", "")

        # Determine overlay output path
        overlay_path: str | None = None
        if output_dir:
            today = datetime.now().strftime("%Y-%m-%d")
            artifacts_dir = Path(output_dir) / today / "artifacts"
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            overlay_path = str(artifacts_dir / f"{job_id}_overlay.png")

        # NG threshold from options or config
        ng_threshold: float | None = None
        if "ng_threshold" in options:
            ng_threshold = float(options["ng_threshold"])
        elif config.get("postprocess", {}).get("decision", {}).get("ng_threshold"):
            ng_threshold = float(config["postprocess"]["decision"]["ng_threshold"])

        # Pad from config
        pad = int(config.get("_train", {}).get("pad", 2))

        # Batch size from config
        batch_size = int(config.get("infer", {}).get("batch_size", 16))

        t_infer_start = time.perf_counter()

        result = predict_from_json(
            model=self._model,
            idx_ok=self._idx_ok,
            idx_ng=self._idx_ng,
            img_size=self._img_size,
            device=self._device_str,
            image_path=image_path,
            json_path=json_path,
            output_overlay=overlay_path,
            pad=pad,
            ng_threshold=ng_threshold,
            batch_size=batch_size,
        )

        t_infer_end = time.perf_counter()
        infer_ms = (t_infer_end - t_infer_start) * 1000

        # Map to standard plugin result format
        artifacts = result.get("artifacts", {})

        return {
            "score": result.get("score", 0.0),
            "threshold": ng_threshold or 0.5,
            "pred": result.get("pred", "OK"),
            "regions": result.get("regions", []),
            "artifacts": artifacts,
            "timing_ms": {
                "infer": round(infer_ms, 2),
                "post": 0.0,
                "save": 0.0,
            },
            "model_version": self._model_version,
            "region_total": result.get("region_total", 0),
            "ng_count": result.get("ng_count", 0),
        }

    def train(
        self,
        dataset_dir: str,
        out_model_dir: str,
        config: dict[str, Any],
        progress_cb: Callable[[float, str], None] | None = None,
    ) -> dict[str, Any]:
        """Train ResNet18 binary classifier.

        Supports two training modes:
        1. ImageFolder mode: dataset_dir/train/OK/*.jpg + dataset_dir/train/NG/*.jpg
        2. JSON crop mode: dataset_dir/ok/ (images) + dataset_dir/json/ (annotations)

        For JSON crop mode, JSON items must have a 'label' field with "OK" or "NG".
        """
        t_start = time.perf_counter()
        out_path = Path(out_model_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        if progress_cb:
            progress_cb(0.0, "Starting ResNet18 classification training...")

        train_cfg = config.get("_train", {})
        ds_path = Path(dataset_dir)

        # ---- Determine training mode ----
        train_dir = ds_path / "train"
        val_dir = ds_path / "val"
        json_dir = ds_path / "json"

        if json_dir.exists() and list(json_dir.glob("*.json")):
            # JSON crop mode: crop regions from big images
            if progress_cb:
                progress_cb(2.0, "JSON crop mode: cropping regions from big images...")

            ok_img_dir = ds_path / "ok"
            if not ok_img_dir.exists():
                ok_img_dir = ds_path

            crop_out = out_path / "_crop_temp"
            train_crop_dir = crop_out / "train"

            label_key = str(train_cfg.get("label_key", "label"))
            pad = int(train_cfg.get("pad", 2))

            counts = crop_regions_from_json_for_training(
                json_dir=json_dir,
                img_dir=ok_img_dir,
                out_dir=train_crop_dir,
                label_key=label_key,
                pad=pad,
            )

            if counts["ok"] == 0 and counts["ng"] == 0:
                raise RuntimeError(
                    "No crops produced from JSON. Check that JSON items have "
                    f"'{label_key}' field with 'OK' or 'NG' values, "
                    "and images exist."
                )

            if progress_cb:
                progress_cb(5.0, f"Cropped: OK={counts['ok']}, NG={counts['ng']}")

            train_dir = train_crop_dir
            val_dir_path: str | None = None  # Use train as val

        elif train_dir.exists():
            # ImageFolder mode
            if progress_cb:
                progress_cb(2.0, "ImageFolder mode: using train/OK/ and train/NG/...")
            val_dir_path = str(val_dir) if val_dir.exists() else None

        else:
            # Fallback: dataset_dir itself may be an ImageFolder (OK/, NG/ subdirs)
            ok_check = ds_path / "OK"
            ng_check = ds_path / "NG"
            if ok_check.exists() or ng_check.exists():
                train_dir = ds_path
                val_dir_path = None
                if progress_cb:
                    progress_cb(2.0, "Using dataset_dir as ImageFolder (OK/, NG/)...")
            else:
                raise RuntimeError(
                    f"Cannot determine training mode. Expected one of:\n"
                    f"  1. {ds_path}/train/OK/ + {ds_path}/train/NG/\n"
                    f"  2. {ds_path}/json/ (JSON annotations) + {ds_path}/ok/ (images)\n"
                    f"  3. {ds_path}/OK/ + {ds_path}/NG/"
                )

        # ---- Build TrainConfig ----
        tcfg = TrainConfig(
            img_size=int(train_cfg.get("img_size", 224)),
            batch_size=int(train_cfg.get("batch_size", 8)),
            num_epochs=int(train_cfg.get("num_epochs", 40)),
            learning_rate=float(train_cfg.get("learning_rate", 1e-4)),
            brightness=float(train_cfg.get("brightness", 0.3)),
            contrast=float(train_cfg.get("contrast", 0.3)),
            rotation=int(train_cfg.get("rotation", 5)),
            horizontal_flip=bool(train_cfg.get("horizontal_flip", True)),
            ng_threshold=float(train_cfg.get("ng_threshold", 0.5)),
        )

        # ---- Train ----
        model_pth = str(out_path / "model.pth")

        # Wrap progress_cb to offset percentages
        def train_progress(pct: float, msg: str) -> None:
            if progress_cb:
                # Map 0-100 from train to 5-95 overall
                adjusted = 5.0 + pct * 0.9
                progress_cb(adjusted, msg)

        if "val_dir_path" not in dir():
            val_dir_path = str(val_dir) if val_dir.exists() else None

        result = train_resnet_classify(
            train_dir=str(train_dir),
            val_dir=val_dir_path,
            out_model_path=model_pth,
            cfg=tcfg,
            progress_cb=train_progress,
        )

        # ---- Save meta.json ----
        version = time.strftime("%Y%m%d_%H%M%S")
        meta = {
            "algo": self.name,
            "version": version,
            "created_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "class_to_idx": result.class_to_idx,
            "idx_ok": result.idx_ok,
            "idx_ng": result.idx_ng,
            "best_val_acc": result.best_val_acc,
            "best_epoch": result.best_epoch,
            "precision_ng": result.precision_ng,
            "recall_ng": result.recall_ng,
            "train_ok": result.train_ok,
            "train_ng": result.train_ng,
            "config": {
                "img_size": tcfg.img_size,
                "batch_size": tcfg.batch_size,
                "num_epochs": tcfg.num_epochs,
                "learning_rate": tcfg.learning_rate,
            },
        }
        with open(out_path / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        # ---- Cleanup temp crop dir ----
        crop_temp = out_path / "_crop_temp"
        if crop_temp.exists():
            import shutil
            try:
                shutil.rmtree(crop_temp)
            except Exception:
                logger.warning("Failed to cleanup temp crop dir: %s", crop_temp)

        duration_s = time.perf_counter() - t_start

        if progress_cb:
            progress_cb(
                100.0,
                f"Training complete: acc={result.best_val_acc:.4f} "
                f"NG_P={result.precision_ng:.3f} NG_R={result.recall_ng:.3f} "
                f"in {duration_s:.1f}s",
            )

        return {
            "model_version": version,
            "metrics": {
                "best_val_acc": result.best_val_acc,
                "best_epoch": result.best_epoch,
                "precision_ng": result.precision_ng,
                "recall_ng": result.recall_ng,
                "train_ok": result.train_ok,
                "train_ng": result.train_ng,
                "val_ok": result.val_ok,
                "val_ng": result.val_ng,
            },
            "threshold": tcfg.ng_threshold,
            "duration_s": round(duration_s, 2),
        }

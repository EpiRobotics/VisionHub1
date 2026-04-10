"""Glyph PatchCore V1 plugin - per-character anomaly detection.

Wraps the glyph PatchCore engine into the VisionHub plugin interface.
Each character class gets its own memory bank (.joblib file), and inference
is driven by JSON annotations that specify character positions.

Model format:
    models/<version>/
        <cls>.joblib        # per-character class model (memory_bank + threshold)
        index.json          # class index with metadata
        meta.json           # plugin metadata

TCP INFER protocol extension:
    The "options" field must include "json_path" pointing to the JSON annotation:
    {
        "cmd": "INFER",
        "job_id": "001",
        "image_path": "E:\\images\\test.jpg",
        "options": {
            "json_path": "E:\\labels\\test.json",
            "thr_global": 2.05
        }
    }
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from app.plugins.base import AlgoPluginBase
from app.plugins.glyph_patchcore_core import (
    GlyphPatchCoreEngine,
    crop_glyphs_from_json,
    train_glyph_patchcore,
)
from app.plugins.registry import PluginRegistry

logger = logging.getLogger(__name__)


@PluginRegistry.register
class GlyphPatchCoreV1Plugin(AlgoPluginBase):
    """
    Per-character glyph anomaly detection using PatchCore.

    Each character class has its own memory bank. Inference requires
    both an image and a JSON annotation file specifying character positions.

    Supports:
    - Per-class .joblib models with memory banks
    - Batched CNN inference (configurable batch size)
    - GPU-accelerated kNN with FP16 (or sklearn CPU fallback)
    - Overlay visualization with per-character OK/NG annotations
    - Auto-threshold from OK score distribution per class
    """

    name: str = "glyph_patchcore_v1"

    def __init__(self) -> None:
        self._engine: GlyphPatchCoreEngine | None = None
        self._model_dir: str = ""
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
        """Load all glyph class models from model_dir/*.joblib."""
        self._model_dir = model_dir
        self._device_str = device
        self._config = config

        model_path = Path(model_dir)

        # Check for .joblib files
        joblib_files = list(model_path.glob("*.joblib"))
        if not joblib_files:
            raise FileNotFoundError(f"No .joblib model files found in: {model_dir}")

        logger.info("Loading glyph PatchCore models from %s (%d class files)...",
                     model_dir, len(joblib_files))

        # Engine config from pipeline infer section
        infer_cfg = config.get("infer", {})

        effective_device = device
        if device == "auto":
            import torch
            effective_device = "cuda" if torch.cuda.is_available() else "cpu"

        self._engine = GlyphPatchCoreEngine(
            model_dir=model_dir,
            device=effective_device,
            use_fp16=bool(infer_cfg.get("use_fp16", True)),
            use_gpu_knn=bool(infer_cfg.get("use_gpu_knn", True)),
            cnn_batch=int(infer_cfg.get("cnn_batch", 64)),
            knn_bank_block=int(infer_cfg.get("knn_bank_block", 20000)),
            feature_layers=str(infer_cfg.get("feature_layers", "layer2")),
            clahe_clip=float(infer_cfg.get("clahe_clip", 0.0)),
        )

        # Read version from meta.json or directory name
        meta_json = model_path / "meta.json"
        if meta_json.exists():
            meta = json.loads(meta_json.read_text(encoding="utf-8"))
            self._model_version = meta.get("version", model_path.name)
        else:
            self._model_version = model_path.name

        self._loaded = True
        logger.info(
            "Glyph PatchCore loaded: version=%s, device=%s, classes=%d",
            self._model_version, effective_device, len(self._engine.cls_models),
        )

    def unload(self) -> None:
        """Release model resources and GPU memory."""
        if self._engine is not None:
            self._engine.unload()
            self._engine = None
        self._loaded = False
        self._model_version = ""
        self._config = {}
        logger.info("Glyph PatchCore model unloaded.")

    def infer(self, image_path: str, config: dict[str, Any]) -> dict[str, Any]:
        """Run glyph-level inference on a single image.

        Requires config["_options"]["json_path"] to be set with the path
        to the JSON annotation file containing character positions.
        """
        if not self._loaded or self._engine is None:
            raise RuntimeError("Glyph PatchCore model not loaded")

        options = config.get("_options", {})
        json_path = options.get("json_path", "")
        if not json_path:
            raise RuntimeError(
                "Missing json_path in options. "
                "Glyph PatchCore requires a JSON annotation file. "
                'Send: {"cmd":"INFER",...,"options":{"json_path":"E:\\\\labels\\\\test.json"}}'
            )

        job_id = config.get("_job_id", "unknown")

        # Determine overlay output path
        # Priority: io.overlay_output_path (fixed file, overwritten each time) > per-job artifacts
        overlay_path: str | None = None
        fixed_overlay = config.get("_overlay_output_path", "")
        if fixed_overlay:
            # Fixed overlay path: always write to the same file (e.g. D:\results\output.jpg)
            # Another vision software monitors this file to check OK/NG by red bounding boxes
            overlay_dir = Path(fixed_overlay).parent
            overlay_dir.mkdir(parents=True, exist_ok=True)
            overlay_path = fixed_overlay
        else:
            output_dir = config.get("_output_dir", "")
            if output_dir:
                today = datetime.now().strftime("%Y-%m-%d")
                artifacts_dir = Path(output_dir) / today / "artifacts"
                artifacts_dir.mkdir(parents=True, exist_ok=True)
                overlay_path = str(artifacts_dir / f"{job_id}_overlay.png")

        # Global threshold override from options or config
        thr_global: float | None = None
        if "thr_global" in options:
            thr_global = float(options["thr_global"])
        elif config.get("postprocess", {}).get("decision", {}).get("thr_global"):
            thr_global = float(config["postprocess"]["decision"]["thr_global"])

        # Pad from config
        pad = int(config.get("_train", {}).get("pad", 2))

        t_infer_start = time.perf_counter()

        result = self._engine.predict(
            image_path=image_path,
            json_path=json_path,
            output_overlay=overlay_path,
            pad=pad,
            thr_global=thr_global,
        )

        t_infer_end = time.perf_counter()
        infer_ms = (t_infer_end - t_infer_start) * 1000

        # Map to standard plugin result format
        artifacts = result.get("artifacts", {})
        timing = result.get("timing_ms", {})

        return {
            "score": result.get("score", 0.0),
            "threshold": result.get("threshold", 0.0),
            "pred": result.get("pred", "OK"),
            "regions": result.get("regions", []),
            "artifacts": artifacts,
            "timing_ms": {
                "infer": round(infer_ms, 2),
                "post": 0.0,
                "save": 0.0,
                "cnn": timing.get("cnn", 0.0),
                "knn": timing.get("knn", 0.0),
            },
            "model_version": self._model_version,
            "glyph_total": result.get("glyph_total", 0),
            "ng_count": result.get("ng_count", 0),
            "unk_count": result.get("unk_count", 0),
        }

    def train(
        self,
        dataset_dir: str,
        out_model_dir: str,
        config: dict[str, Any],
        progress_cb: Callable[[float, str], None] | None = None,
    ) -> dict[str, Any]:
        """Train glyph PatchCore models from OK samples + JSON annotations.

        Training pipeline:
        1. Crop glyphs from JSON annotations → glyph_bank/<class>/
        2. Train per-class PatchCore memory banks
        3. Save .joblib files + index.json + meta.json

        Dataset directory structure:
            dataset_dir/
                ok/           # OK sample images
                json/         # JSON annotation files (matching image names)
        """
        t_start = time.perf_counter()
        out_path = Path(out_model_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        if progress_cb:
            progress_cb(0.0, "Starting glyph PatchCore training...")

        train_cfg = config.get("_train", {})

        # --- Step 1: Crop glyphs from JSON annotations ---
        if progress_cb:
            progress_cb(2.0, "Step 1: Cropping glyphs from JSON annotations...")

        ds_path = Path(dataset_dir)
        ok_dir = ds_path / "ok"
        json_dir = ds_path / "json"

        if not ok_dir.exists():
            # Fallback: images might be directly in dataset_dir
            ok_dir = ds_path
        if not json_dir.exists():
            raise RuntimeError(
                f"JSON annotation directory not found: {json_dir}. "
                "Expected dataset_dir/json/ with annotation JSON files."
            )

        json_files = sorted(json_dir.glob("*.json"))
        if not json_files:
            raise RuntimeError(f"No JSON annotation files found in: {json_dir}")

        # Create temporary glyph bank directory
        glyph_bank_dir = out_path / "_glyph_bank_temp"
        if glyph_bank_dir.exists():
            shutil.rmtree(glyph_bank_dir)
        glyph_bank_dir.mkdir(parents=True)

        pad = int(train_cfg.get("pad", 2))
        total_crops = 0
        for ji, jp in enumerate(json_files):
            if progress_cb and ji % 10 == 0:
                pct = 2.0 + 8.0 * ji / len(json_files)
                progress_cb(pct, f"Cropping glyphs: {ji}/{len(json_files)} JSONs...")
            total_crops += crop_glyphs_from_json(
                json_path=jp, img_dir=ok_dir, out_dir=glyph_bank_dir,
                pad=pad, ext=".jpg",
            )

        if total_crops == 0:
            raise RuntimeError(
                "No glyph crops produced. Check that JSON files have 'items' with "
                "'ch', 'cx', 'cy', 'w', 'h' fields, and images exist in ok/ directory."
            )

        if progress_cb:
            progress_cb(10.0, f"Cropped {total_crops} glyph images")

        # --- Step 2: Train per-class PatchCore models ---
        report = train_glyph_patchcore(
            bank_dir=str(glyph_bank_dir),
            out_model_dir=str(out_path),
            img_size=int(train_cfg.get("img_size", 128)),
            max_patches_per_class=int(train_cfg.get("max_patches_per_class", 30000)),
            k=int(train_cfg.get("k", 1)),
            score_mode=str(train_cfg.get("score_mode", "topk")),
            topk=int(train_cfg.get("topk", 10)),
            p_thr=float(train_cfg.get("p_thr", 0.995)),
            min_per_class=int(train_cfg.get("min_per_class", 10)),
            progress_cb=progress_cb,
            feature_layers=str(train_cfg.get("feature_layers", "layer2")),
            clahe_clip=float(train_cfg.get("clahe_clip", 0.0)),
            morph_aug=bool(train_cfg.get("morph_aug", False)),
        )

        # --- Step 3: Save meta.json ---
        version = time.strftime("%Y%m%d_%H%M%S")
        meta = {
            "algo": self.name,
            "version": version,
            "created_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "trained_classes": report["trained_classes"],
            "total_crops": total_crops,
            "config": {
                "img_size": int(train_cfg.get("img_size", 128)),
                "k": int(train_cfg.get("k", 1)),
                "score_mode": str(train_cfg.get("score_mode", "topk")),
                "topk": int(train_cfg.get("topk", 10)),
                "p_thr": float(train_cfg.get("p_thr", 0.995)),
                "max_patches_per_class": int(train_cfg.get("max_patches_per_class", 30000)),
                "pad": pad,
                "feature_layers": str(train_cfg.get("feature_layers", "layer2")),
                "clahe_clip": float(train_cfg.get("clahe_clip", 0.0)),
                "morph_aug": bool(train_cfg.get("morph_aug", False)),
            },
        }
        with open(out_path / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        # --- Cleanup temp glyph bank ---
        try:
            shutil.rmtree(glyph_bank_dir)
        except Exception:
            logger.warning("Failed to cleanup temp glyph bank: %s", glyph_bank_dir)

        duration_s = time.perf_counter() - t_start

        if progress_cb:
            progress_cb(100.0, f"Training complete: {report['trained_classes']} classes in {duration_s:.1f}s")

        return {
            "model_version": version,
            "metrics": {
                "trained_classes": report["trained_classes"],
                "total_classes": report["total_classes"],
                "total_crops": total_crops,
                "n_json_files": len(json_files),
            },
            "threshold": 0.0,  # per-class thresholds in individual .joblib files
            "duration_s": round(duration_s, 2),
        }

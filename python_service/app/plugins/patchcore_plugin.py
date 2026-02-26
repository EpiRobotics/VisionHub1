"""PatchCore Tiling V1 plugin stub.

The actual deep learning implementation will be provided separately.
This stub defines the interface and placeholder logic so the service
skeleton can run end-to-end.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Callable

from app.plugins.base import AlgoPluginBase
from app.plugins.registry import PluginRegistry

logger = logging.getLogger(__name__)


@PluginRegistry.register
class PatchCoreTilingV1Plugin(AlgoPluginBase):
    """
    PatchCore with tiling strategy for large-image surface inspection.

    This is a stub implementation. The real algorithm code (memory bank
    construction, kNN scoring, tiling logic) will be injected later.
    """

    name: str = "patchcore_tiling_v1"

    def __init__(self) -> None:
        self._model_dir: str = ""
        self._device: str = "cpu"
        self._config: dict[str, Any] = {}
        self._loaded: bool = False
        self._model_version: str = ""
        self._threshold: float = 0.5

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self, model_dir: str, device: str, config: dict[str, Any]) -> None:
        """Load PatchCore model from model_dir."""
        self._model_dir = model_dir
        self._device = device
        self._config = config

        model_path = Path(model_dir)
        meta_path = model_path / "meta.json"

        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            self._model_version = meta.get("version", model_path.name)
            self._threshold = meta.get("threshold", 0.5)
        else:
            self._model_version = model_path.name
            self._threshold = 0.5

        # TODO: Load actual model (memory bank, backbone, etc.)
        # e.g. self._memory_bank = torch.load(model_path / "model.pt")

        self._loaded = True
        logger.info(
            "PatchCore model loaded: dir=%s, version=%s, device=%s",
            model_dir,
            self._model_version,
            device,
        )

    def unload(self) -> None:
        """Release model resources."""
        # TODO: Release GPU memory
        # e.g. del self._memory_bank; torch.cuda.empty_cache()
        self._loaded = False
        self._model_version = ""
        logger.info("PatchCore model unloaded.")

    def infer(self, image_path: str, config: dict[str, Any]) -> dict[str, Any]:
        """
        Run PatchCore inference on a single image.

        Stub: returns a dummy OK result. Replace with actual implementation.
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded")

        t_infer_start = time.perf_counter()

        # ------------------------------------------------------------------
        # TODO: Replace with actual PatchCore tiling inference
        # 1. Read image
        # 2. Apply ROI / preprocessing
        # 3. Tile image
        # 4. Batch embed tiles through backbone
        # 5. kNN score against memory bank
        # 6. Stitch heatmap
        # 7. Apply postprocessing (quantile, threshold, mask, regions)
        # ------------------------------------------------------------------

        score = 0.0
        pred = "OK" if score < self._threshold else "NG"

        t_infer_end = time.perf_counter()
        infer_ms = (t_infer_end - t_infer_start) * 1000

        return {
            "score": score,
            "threshold": self._threshold,
            "pred": pred,
            "regions": [],
            "artifacts": {},
            "timing_ms": {
                "infer": round(infer_ms, 2),
                "post": 0.0,
                "save": 0.0,
            },
            "model_version": self._model_version,
        }

    def train(
        self,
        dataset_dir: str,
        out_model_dir: str,
        config: dict[str, Any],
        progress_cb: Callable[[float, str], None] | None = None,
    ) -> dict[str, Any]:
        """
        Train PatchCore model from OK samples.

        Stub: creates placeholder model files. Replace with actual implementation.
        """
        t_start = time.perf_counter()

        out_path = Path(out_model_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        if progress_cb:
            progress_cb(0.0, "Starting PatchCore training...")

        # ------------------------------------------------------------------
        # TODO: Replace with actual PatchCore training
        # 1. Load OK images from dataset_dir/ok/
        # 2. Extract features via backbone
        # 3. Build memory bank (coreset subsampling)
        # 4. Compute threshold from training distribution
        # 5. Save model.pt and meta.json
        # ------------------------------------------------------------------

        version = time.strftime("%Y%m%d_%H%M%S")
        threshold = 0.5

        # Write placeholder meta.json
        meta = {
            "algo": self.name,
            "version": version,
            "threshold": threshold,
            "tile_config": config.get("infer", {}).get("tile", {}),
        }
        with open(out_path / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        # Write placeholder model file
        (out_path / "model.pt").write_text("placeholder")

        if progress_cb:
            progress_cb(100.0, "Training complete.")

        duration = time.perf_counter() - t_start

        logger.info("PatchCore training complete: version=%s, dir=%s", version, out_model_dir)

        return {
            "model_version": version,
            "metrics": {"n_samples": 0, "memory_bank_size": 0},
            "threshold": threshold,
            "duration_s": round(duration, 2),
        }

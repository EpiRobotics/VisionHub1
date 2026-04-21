"""Siamese Compare V1 plugin - adjacent-tooth contrastive comparison.

Wraps the Siamese adjacent-pair comparison algorithm into the VisionHub
plugin interface.  Designed for detecting alignment deviations in periodic
structures such as motor iron core lamination grooves.

Key idea:
- Slice image into tooth-sized strips along the long axis
- Embed each strip via a shared ResNet backbone + projection head
- Compute distance between adjacent pairs
- Any pair with distance > threshold → NG (misalignment detected)

Training requires only OK samples (unsupervised).

Model format (.pt saved by torch.save):
    {
        "meta": { backbone, layers, embed_dim, strip/tile params, threshold, ... },
        "state_dict": EmbeddingHead state_dict,
    }

TCP INFER protocol (same as other plugins):
    {
        "cmd": "INFER",
        "job_id": "001",
        "image_path": "E:\\images\\test.bmp"
    }

Training dataset structure:
    datasets/
        ok/                 # OK images only (no NG needed)
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from app.plugins.base import AlgoPluginBase
from app.plugins.patchcore_core import ResNetFeat, build_transform
from app.plugins.patchcore_strip_core import list_images, set_seed
from app.plugins.registry import PluginRegistry
from app.plugins.siamese_compare_core import (
    EmbeddingHead,
    compute_ok_pair_distances,
    compute_threshold_from_ok_distances,
    infer_adjacent_pairs,
    save_siamese_overlay_cv2,
    train_siamese_model,
)

logger = logging.getLogger(__name__)


@PluginRegistry.register
class SiameseCompareV1Plugin(AlgoPluginBase):
    """
    Adjacent-tooth Siamese comparison for periodic structure inspection.

    Slices images into tooth-sized strips along the long axis, embeds each
    strip through a shared backbone, and compares adjacent pairs by distance.
    High distance between adjacent strips indicates misalignment.

    Supports:
    - Auto long-axis detection (vertical / horizontal)
    - Configurable strip size and overlap
    - L2 or cosine distance metrics
    - Optional contrastive fine-tuning on OK pairs
    - Per-pair NG localisation on overlay
    - ResNet18/50 backbone
    """

    name: str = "siamese_compare_v1"

    def __init__(self) -> None:
        self._model_dir: str = ""
        self._device_str: str = "cpu"
        self._device: torch.device = torch.device("cpu")
        self._config: dict[str, Any] = {}
        self._loaded: bool = False
        self._model_version: str = ""
        self._threshold: float = 0.5

        # Core model (populated on load)
        self._model: EmbeddingHead | None = None
        self._meta: dict[str, Any] = {}

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def model_version(self) -> str:
        return self._model_version

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(self, model_dir: str, device: str, config: dict[str, Any]) -> None:
        """Load Siamese Compare model (backbone + projection head)."""
        self._model_dir = model_dir
        self._device_str = device
        self._device = torch.device(
            device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._config = config

        model_path = Path(model_dir)
        pt_file = model_path / "model.pt"

        if not pt_file.exists():
            raise FileNotFoundError(f"Model file not found: {pt_file}")

        logger.info("Loading Siamese Compare model from %s ...", model_dir)

        pack = torch.load(str(pt_file), map_location="cpu", weights_only=False)
        meta = pack["meta"]
        self._meta = meta

        # Rebuild model from meta
        layers = meta["layers"]
        backbone = ResNetFeat(meta["backbone"], pretrained=False, layers=layers)
        # Pass concat_dim so _proj is eagerly created before load_state_dict
        concat_dim = int(meta.get("concat_dim", 0))
        model = EmbeddingHead(
            backbone, layers=layers, embed_dim=int(meta["embed_dim"]),
            concat_dim=concat_dim,
        )

        # Load state dict
        model.load_state_dict(pack["state_dict"], strict=(concat_dim > 0))
        model = model.to(self._device).eval()
        self._model = model

        # Threshold & version
        self._threshold = meta.get("threshold", 0.5) or 0.5
        self._model_version = meta.get("version", model_path.name) or model_path.name

        # Write meta.json if not present
        meta_json_path = model_path / "meta.json"
        if not meta_json_path.exists():
            with open(meta_json_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "algo": self.name,
                        "version": self._model_version,
                        "threshold": self._threshold,
                        "backbone": meta.get("backbone", "resnet18"),
                        "layers": layers,
                        "embed_dim": meta.get("embed_dim", 256),
                        "strip_size": meta.get("strip_size", 256),
                        "strip_overlap": meta.get("strip_overlap", 64),
                        "strip_axis": meta.get("strip_axis", "auto"),
                        "metric": meta.get("metric", "l2"),
                    },
                    f,
                    indent=2,
                )

        self._loaded = True
        logger.info(
            "Siamese Compare model loaded: version=%s, device=%s, "
            "threshold=%.6f, strip_size=%s, embed_dim=%s, metric=%s",
            self._model_version,
            self._device,
            self._threshold,
            meta.get("strip_size"),
            meta.get("embed_dim"),
            meta.get("metric"),
        )

    # ------------------------------------------------------------------
    # Unload
    # ------------------------------------------------------------------

    def unload(self) -> None:
        """Release model resources and GPU memory."""
        if self._model is not None:
            del self._model
            self._model = None
        self._meta = {}
        self._loaded = False
        self._model_version = ""

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Siamese Compare model unloaded.")

    # ------------------------------------------------------------------
    # Infer
    # ------------------------------------------------------------------

    def infer(self, image_path: str, config: dict[str, Any]) -> dict[str, Any]:
        """Run adjacent-pair Siamese comparison on a single image.

        Strip / tile parameters come from saved model meta.
        Post-processing parameters come from project config.
        """
        if not self._loaded or self._model is None:
            raise RuntimeError("Model not loaded")

        meta = self._meta
        job_id = config.get("_job_id", "unknown")
        output_dir = config.get("_output_dir", "")

        # Strip parameters from model meta
        strip_size = int(meta.get("strip_size", 256))
        strip_overlap = int(meta.get("strip_overlap", 64))
        strip_axis_str = meta.get("strip_axis", "auto")
        strip_axis = None if strip_axis_str == "auto" else strip_axis_str

        # Tile parameters from model meta
        tile_w = int(meta.get("tile_w", 512))
        tile_h = int(meta.get("tile_h", 352))

        # Distance metric
        metric = str(meta.get("metric", "l2"))

        # --- Inference ---
        t_infer_start = time.perf_counter()

        max_distance, pair_results, (h0, w0), diff_heatmap = infer_adjacent_pairs(
            img_path=Path(image_path),
            model=self._model,
            strip_size=strip_size,
            strip_overlap=strip_overlap,
            tile_w=tile_w,
            tile_h=tile_h,
            device=self._device,
            metric=metric,
            axis=strip_axis,
        )

        t_infer_end = time.perf_counter()
        infer_ms = (t_infer_end - t_infer_start) * 1000

        # --- Decision ---
        decision_cfg = config.get("postprocess", {}).get("decision", {})
        thr_global = decision_cfg.get("thr_global")
        if thr_global is not None:
            threshold = float(thr_global)
        else:
            threshold = self._threshold

        # Any-pair-NG decision
        ng_pairs = [pr for pr in pair_results if pr["distance"] >= threshold]
        pred = "NG" if len(ng_pairs) > 0 else "OK"

        # --- Postprocessing & save artifacts ---
        t_post_start = time.perf_counter()

        postprocess = config.get("postprocess", {})
        export_cfg = postprocess.get("export", {})

        artifacts: dict[str, str] = {}
        regions: list[dict[str, Any]] = []

        # Build regions from NG pairs
        for pr in ng_pairs:
            ba = pr["bbox_a"]
            bb = pr["bbox_b"]
            # Region covering both strips in the NG pair
            rx = min(ba["x"], bb["x"])
            ry = min(ba["y"], bb["y"])
            rw = max(ba["x"] + ba["w"], bb["x"] + bb["w"]) - rx
            rh = max(ba["y"] + ba["h"], bb["y"] + bb["h"]) - ry
            regions.append({
                "x": rx, "y": ry, "w": rw, "h": rh,
                "score": pr["distance"],
                "pair_idx": pr["pair_idx"],
            })

        # Fixed overlay output path
        overlay_output_path = config.get("_overlay_output_path", "")

        if overlay_output_path:
            try:
                save_siamese_overlay_cv2(
                    out_path=Path(overlay_output_path),
                    img_path=Path(image_path),
                    pair_results=pair_results,
                    threshold=threshold,
                    max_distance=max_distance,
                    pred=pred,
                    diff_heatmap=diff_heatmap,
                )
                artifacts["overlay"] = overlay_output_path
            except Exception:
                logger.warning(
                    "Failed to save siamese overlay to %s for job %s",
                    overlay_output_path, job_id, exc_info=True,
                )

        if output_dir:
            from datetime import datetime

            today = datetime.now().strftime("%Y-%m-%d")
            artifacts_dir = Path(output_dir) / today / "artifacts"
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            base_name = str(job_id)

            # Per-job overlay
            if not overlay_output_path and export_cfg.get("save_overlay", False):
                try:
                    overlay_file = str(artifacts_dir / f"{base_name}_overlay.jpg")
                    save_siamese_overlay_cv2(
                        out_path=Path(overlay_file),
                        img_path=Path(image_path),
                        pair_results=pair_results,
                        threshold=threshold,
                        max_distance=max_distance,
                        pred=pred,
                        diff_heatmap=diff_heatmap,
                    )
                    artifacts["overlay"] = overlay_file
                except Exception:
                    logger.warning(
                        "Failed to save siamese overlay for job %s",
                        job_id, exc_info=True,
                    )

        t_post_end = time.perf_counter()
        post_ms = (t_post_end - t_post_start) * 1000

        return {
            "score": round(float(max_distance), 6),
            "threshold": round(float(threshold), 6),
            "pred": pred,
            "regions": regions,
            "pair_results": pair_results,
            "artifacts": artifacts,
            "timing_ms": {
                "infer": round(infer_ms, 2),
                "post": round(post_ms, 2),
            },
            "model_version": self._model_version,
        }

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------

    def train(
        self,
        dataset_dir: str,
        out_model_dir: str,
        config: dict[str, Any],
        progress_cb: Callable[[float, str], None] | None = None,
    ) -> dict[str, Any]:
        """Train Siamese Compare model from OK samples.

        Steps:
        1. Load OK images from dataset_dir/ok/
        2. Build backbone + projection head
        3. (Optional) fine-tune with contrastive loss on OK adjacent pairs
        4. Compute OK pair distance distribution
        5. Set threshold from distribution
        6. Save model.pt and meta.json
        """
        t_start = time.perf_counter()
        set_seed(42)

        out_path = Path(out_model_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        if progress_cb:
            progress_cb(0.0, "Starting Siamese Compare training...")

        # --- Resolve device ---
        device_str = config.get("infer", {}).get("device", "cuda")
        if device_str == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device_str)

        # --- Find OK images ---
        ok_dir = Path(dataset_dir) / "ok"
        if not ok_dir.exists():
            ok_dir = Path(dataset_dir)
        img_paths = list_images(ok_dir)
        if not img_paths:
            raise RuntimeError(f"No OK images found in: {ok_dir}")

        if progress_cb:
            progress_cb(5.0, f"Found {len(img_paths)} OK images")

        # --- Training parameters ---
        infer_cfg = config.get("infer", {})

        # Strip parameters
        strip_cfg = infer_cfg.get("strip", {})
        strip_size = int(strip_cfg.get("strip_size", 256))
        strip_overlap = int(strip_cfg.get("strip_overlap", 64))
        strip_axis_str = str(strip_cfg.get("strip_axis", "auto"))
        strip_axis = None if strip_axis_str == "auto" else strip_axis_str

        # Tile parameters
        tile_cfg = infer_cfg.get("tile", {})
        tile_w = int(tile_cfg.get("tile_w", 512))
        tile_h = int(tile_cfg.get("tile_h", 352))

        # Siamese-specific training params
        train_cfg = config.get("_train", {})
        backbone_name = str(train_cfg.get("backbone", "resnet18"))
        layers_str = str(train_cfg.get("layers", "layer2,layer3"))
        layers = [s.strip() for s in layers_str.split(",") if s.strip()]
        embed_dim = int(train_cfg.get("embed_dim", 256))
        metric = str(train_cfg.get("metric", "l2"))
        fine_tune_epochs = int(train_cfg.get("fine_tune_epochs", 0))
        fine_tune_lr = float(train_cfg.get("fine_tune_lr", 1e-4))
        fine_tune_margin = float(train_cfg.get("fine_tune_margin", 0.5))
        compute_threshold = bool(train_cfg.get("compute_threshold", True))
        ok_quantile = float(train_cfg.get("ok_quantile", 0.999))
        thr_scale = float(train_cfg.get("thr_scale", 1.2))

        # --- Build & optionally fine-tune model ---
        if progress_cb:
            progress_cb(10.0, f"Building model: {backbone_name}, embed_dim={embed_dim}")

        model, train_stats = train_siamese_model(
            img_paths=img_paths,
            backbone_name=backbone_name,
            layers=layers,
            embed_dim=embed_dim,
            strip_size=strip_size,
            strip_overlap=strip_overlap,
            tile_w=tile_w,
            tile_h=tile_h,
            device=device,
            axis=strip_axis,
            fine_tune_epochs=fine_tune_epochs,
            fine_tune_lr=fine_tune_lr,
            fine_tune_margin=fine_tune_margin,
            progress_cb=progress_cb,
        )

        # --- Compute threshold from OK pair distances ---
        threshold = None
        ok_distances: list[float] = []
        if compute_threshold:
            if progress_cb:
                progress_cb(55.0, "Computing OK pair distances for threshold...")

            ok_distances = compute_ok_pair_distances(
                img_paths=img_paths,
                model=model,
                strip_size=strip_size,
                strip_overlap=strip_overlap,
                tile_w=tile_w,
                tile_h=tile_h,
                device=device,
                metric=metric,
                axis=strip_axis,
                progress_cb=progress_cb,
            )

            if progress_cb:
                progress_cb(88.0, "Computing threshold from OK distance distribution...")

            threshold = compute_threshold_from_ok_distances(
                ok_distances=ok_distances,
                ok_quantile=ok_quantile,
                thr_scale=thr_scale,
            )

        # --- Save model ---
        if progress_cb:
            progress_cb(95.0, "Saving model...")

        _, mean, std = build_transform()
        version = time.strftime("%Y%m%d_%H%M%S")

        meta = {
            "backbone": backbone_name,
            "layers": layers,
            "embed_dim": embed_dim,
            "concat_dim": model._concat_dim,
            "tile_w": tile_w,
            "tile_h": tile_h,
            "strip_size": strip_size,
            "strip_overlap": strip_overlap,
            "strip_axis": strip_axis_str,
            "metric": metric,
            "mean": mean,
            "std": std,
            "threshold": threshold,
            "ok_quantile": ok_quantile,
            "thr_scale": thr_scale,
            "fine_tune_epochs": fine_tune_epochs,
            "version": version,
            "created_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "n_ok_images": len(img_paths),
            "n_ok_pairs": len(ok_distances),
        }

        pack = {
            "meta": meta,
            "state_dict": model.state_dict(),
        }

        model_pt_path = out_path / "model.pt"
        torch.save(pack, str(model_pt_path))

        # Also save meta.json for UI
        with open(out_path / "meta.json", "w", encoding="utf-8") as f:
            json.dump({**meta, "algo": self.name}, f, indent=2, ensure_ascii=False)

        if progress_cb:
            progress_cb(100.0, "Training complete!")

        duration = time.perf_counter() - t_start

        dist_stats: dict[str, float] = {}
        if ok_distances:
            arr = np.array(ok_distances, dtype=np.float32)
            dist_stats = {
                "mean": round(float(np.mean(arr)), 6),
                "std": round(float(np.std(arr)), 6),
                "min": round(float(np.min(arr)), 6),
                "max": round(float(np.max(arr)), 6),
                "q999": round(float(np.quantile(arr, 0.999)), 6),
            }

        logger.info(
            "Siamese Compare training complete: version=%s, "
            "threshold=%s, n_pairs=%d, duration=%.1fs",
            version,
            threshold,
            len(ok_distances),
            duration,
        )

        return {
            "model_version": version,
            "metrics": {
                "n_ok_images": len(img_paths),
                "n_ok_pairs": len(ok_distances),
                "embed_dim": embed_dim,
                "strip_size": strip_size,
                "strip_overlap": strip_overlap,
                "ok_distance_stats": dist_stats,
            },
            "threshold": threshold,
            "duration_s": round(duration, 2),
        }

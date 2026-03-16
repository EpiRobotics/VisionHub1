"""PatchCore Tiling V1 plugin - full implementation.

Wraps the core PatchCore algorithm (ResNet backbone, patch embeddings,
kNN scoring, tiled ROI inference) into the VisionHub plugin interface.

Model format (.pt saved by torch.save):
    {
        "meta": { backbone, layers, proj_dim, tile params, threshold, ... },
        "state_dict": embedder state_dict,
        "memory_bank": (M, D) float32 numpy array,
    }
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from PIL import Image

from app.plugins.base import AlgoPluginBase
from app.plugins.patchcore_core import (
    KNNSearcher,
    PatchEmbedder,
    ResNetFeat,
    build_transform,
    compute_threshold_from_ok,
    extract_regions,
    infer_one_image,
    list_images,
    save_heatmap_png,
    save_overlay_cv2,
    save_u16_and_mask,
    set_seed,
    train_memory_bank,
)
from app.plugins.registry import PluginRegistry

logger = logging.getLogger(__name__)


@PluginRegistry.register
class PatchCoreTilingV1Plugin(AlgoPluginBase):
    """
    PatchCore with tiling strategy for large-image surface inspection.

    Supports:
    - ResNet18/50 backbone with configurable layer taps
    - Random projection for embedding dimensionality reduction
    - Tiled inference with overlap stitching (max)
    - kNN scoring with faiss or torch fallback
    - Heatmap, u16 score map, mask, overlay, regions export
    """

    name: str = "patchcore_tiling_v1"

    def __init__(self) -> None:
        self._model_dir: str = ""
        self._device_str: str = "cpu"
        self._device: torch.device = torch.device("cpu")
        self._config: dict[str, Any] = {}
        self._loaded: bool = False
        self._model_version: str = ""
        self._threshold: float = 0.5

        # Core algorithm components (populated on load)
        self._embedder: PatchEmbedder | None = None
        self._knn: KNNSearcher | None = None
        self._meta: dict[str, Any] = {}

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def model_version(self) -> str:
        return self._model_version

    def load(self, model_dir: str, device: str, config: dict[str, Any]) -> None:
        """Load PatchCore model (backbone + memory bank + kNN index)."""
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

        logger.info("Loading PatchCore model from %s ...", model_dir)

        # Load the model package
        pack = torch.load(str(pt_file), map_location="cpu", weights_only=False)
        meta = pack["meta"]
        self._meta = meta
        memory_bank = pack["memory_bank"].astype(np.float32)

        # Rebuild embedder from saved weights
        layers = meta["layers"]
        backbone = ResNetFeat(meta["backbone"], pretrained=False, layers=layers)
        embedder = PatchEmbedder(
            backbone, layers=layers, proj_dim=int(meta["proj_dim"]), seed=42
        )

        # Fix: init proj_mat shape before loading state_dict (strict load)
        sd = pack["state_dict"]
        if "proj_mat" in sd:
            embedder.proj_mat = sd["proj_mat"].detach().clone()

        embedder.load_state_dict(sd, strict=True)
        embedder = embedder.to(self._device).eval()
        self._embedder = embedder

        # Build kNN index
        use_faiss = config.get("_use_faiss", True)
        self._knn = KNNSearcher(memory_bank, use_faiss=use_faiss)

        # Threshold & version
        self._threshold = meta.get("threshold", 0.5) or 0.5
        self._model_version = meta.get("version", model_path.name) or model_path.name

        # Write meta.json for UI consumption if not present
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
                        "proj_dim": meta.get("proj_dim", 128),
                        "tile_w": meta.get("tile_w", 512),
                        "tile_h": meta.get("tile_h", 352),
                        "stride_w": meta.get("stride_w", 384),
                        "stride_h": meta.get("stride_h", 352),
                        "memory_bank_size": memory_bank.shape[0],
                        "memory_bank_dim": memory_bank.shape[1],
                    },
                    f,
                    indent=2,
                )

        self._loaded = True
        logger.info(
            "PatchCore model loaded: version=%s, device=%s, bank=%s, threshold=%.6f",
            self._model_version,
            self._device,
            memory_bank.shape,
            self._threshold,
        )

    def unload(self) -> None:
        """Release model resources and GPU memory."""
        if self._embedder is not None:
            del self._embedder
            self._embedder = None
        if self._knn is not None:
            del self._knn
            self._knn = None
        self._meta = {}
        self._loaded = False
        self._model_version = ""

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("PatchCore model unloaded.")

    def infer(self, image_path: str, config: dict[str, Any]) -> dict[str, Any]:
        """
        Run PatchCore tiled inference on a single image.

        Tile parameters come from the saved model meta (consistent with
        training). Post-processing parameters come from project config.
        """
        if not self._loaded or self._embedder is None or self._knn is None:
            raise RuntimeError("Model not loaded")

        meta = self._meta
        job_id = config.get("_job_id", "unknown")
        output_dir = config.get("_output_dir", "")

        # Tile parameters from model meta
        tile_w = int(meta.get("tile_w", 512))
        tile_h = int(meta.get("tile_h", 352))
        stride_w = int(meta.get("stride_w", 384))
        stride_h = int(meta.get("stride_h", 352))
        pad_mode = meta.get("pad_mode", "edge")

        # Batch size from project config
        infer_cfg = config.get("infer", {})
        batch_tiles = int(infer_cfg.get("batch_tiles", 8))

        # --- Inference ---
        t_infer_start = time.perf_counter()

        heatmap, score, (h0, w0) = infer_one_image(
            img_path=Path(image_path),
            embedder=self._embedder,
            knn=self._knn,
            tile_w=tile_w,
            tile_h=tile_h,
            stride_w=stride_w,
            stride_h=stride_h,
            device=self._device,
            pad_mode=pad_mode,
            batch_tiles=batch_tiles,
        )

        t_infer_end = time.perf_counter()
        infer_ms = (t_infer_end - t_infer_start) * 1000

        # --- Decision ---
        # Use thr_global from project config (set via UI) if available,
        # otherwise fall back to the threshold stored in the model file.
        decision_cfg = config.get("postprocess", {}).get("decision", {})
        thr_global = decision_cfg.get("thr_global")
        if thr_global is not None:
            threshold = float(thr_global)
        else:
            threshold = self._threshold
        pred = "NG" if score >= threshold else "OK"

        # --- Postprocessing ---
        t_post_start = time.perf_counter()

        postprocess = config.get("postprocess", {})
        export_cfg = postprocess.get("export", {})

        # Compute vmin/vmax for heatmap scaling
        vmin = float(export_cfg.get("heatmap_vmin", 0.0))
        vmax_mode = export_cfg.get("heatmap_vmax_mode", "thr_scale")
        if vmax_mode == "thr_scale":
            vmax_scale = float(export_cfg.get("heatmap_vmax_scale", 1.2))
            vmax = float(threshold * vmax_scale)
        elif vmax_mode == "fixed":
            vmax = float(export_cfg.get("heatmap_vmax", 1.0))
        else:
            vmax = float(np.max(heatmap)) if heatmap.max() > 0 else 1.0

        # Mask / morphology params
        mask_thr_scale = float(export_cfg.get("mask_thr_scale", 0.85))
        dilate_px = int(export_cfg.get("dilate_px", 10))
        close_px = int(export_cfg.get("close_px", 6))

        t_post_end = time.perf_counter()
        post_ms = (t_post_end - t_post_start) * 1000

        # --- Save artifacts ---
        t_save_start = time.perf_counter()
        artifacts: dict[str, str] = {}
        regions: list[dict[str, Any]] = []

        # Fixed overlay output path (overwrite each time, like glyph plugin)
        overlay_output_path = config.get("_overlay_output_path", "")

        if overlay_output_path:
            # Save overlay to fixed path (fast, cv2-based)
            try:
                save_overlay_cv2(
                    out_path=Path(overlay_output_path),
                    img_path=Path(image_path),
                    heatmap=heatmap,
                    vmin=vmin,
                    vmax=vmax,
                    score=score,
                    threshold=threshold,
                    pred=pred,
                )
                artifacts["overlay"] = overlay_output_path
            except Exception:
                logger.warning("Failed to save overlay to %s for job %s", overlay_output_path, job_id, exc_info=True)

        if output_dir:
            today = datetime.now().strftime("%Y-%m-%d")
            artifacts_dir = Path(output_dir) / today / "artifacts"
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            base_name = str(job_id)

            # U16 + Mask (default off for speed)
            if export_cfg.get("save_u16", False) or export_cfg.get("save_mask", False):
                u16_path, mask_path = save_u16_and_mask(
                    out_base=artifacts_dir / f"{base_name}_score",
                    heatmap=heatmap,
                    vmin=vmin,
                    vmax=vmax,
                    thr=threshold,
                    mask_thr_scale=mask_thr_scale,
                    dilate_px=dilate_px,
                    close_px=close_px,
                )
                if export_cfg.get("save_u16", False):
                    artifacts["u16"] = u16_path
                if export_cfg.get("save_mask", False):
                    artifacts["mask"] = mask_path

                    # Extract connected-component regions from mask
                    regions_cfg = export_cfg.get("regions", {})
                    if regions_cfg.get("enabled", True):
                        mask_arr = np.array(Image.open(mask_path).convert("L"))
                        regions = extract_regions(
                            mask=mask_arr,
                            heatmap=heatmap,
                            min_area_px=int(regions_cfg.get("min_area_px", 80)),
                            max_regions=int(regions_cfg.get("max_regions", 10)),
                        )

            # Heatmap PNG (default off — matplotlib is very slow ~1-2s)
            if export_cfg.get("save_heatmap_png", False):
                heatmap_file = str(artifacts_dir / f"{base_name}_heatmap.png")
                save_heatmap_png(
                    out_png=Path(heatmap_file),
                    heatmap=heatmap,
                    title=f"{Path(image_path).name} score={score:.4f} pred={pred}",
                    vmin=vmin,
                    vmax=vmax,
                )
                artifacts["heatmap"] = heatmap_file

            # Per-job overlay (only if no fixed overlay path, default off)
            if not overlay_output_path and export_cfg.get("save_overlay", False):
                try:
                    overlay_file = str(artifacts_dir / f"{base_name}_overlay.jpg")
                    save_overlay_cv2(
                        out_path=Path(overlay_file),
                        img_path=Path(image_path),
                        heatmap=heatmap,
                        vmin=vmin,
                        vmax=vmax,
                        score=score,
                        threshold=threshold,
                        pred=pred,
                    )
                    artifacts["overlay"] = overlay_file
                except Exception:
                    logger.warning("Failed to save overlay for job %s", job_id, exc_info=True)

        t_save_end = time.perf_counter()
        save_ms = (t_save_end - t_save_start) * 1000

        return {
            "score": round(float(score), 6),
            "threshold": round(float(threshold), 6),
            "pred": pred,
            "regions": regions,
            "artifacts": artifacts,
            "timing_ms": {
                "infer": round(infer_ms, 2),
                "post": round(post_ms, 2),
                "save": round(save_ms, 2),
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

        Steps:
        1. Load OK images from dataset_dir/ok/
        2. Build backbone + embedder
        3. Extract patch embeddings from tiled OK images
        4. Subsample to build memory bank
        5. Optionally compute auto-threshold from OK scores
        6. Save model.pt and meta.json
        """
        t_start = time.perf_counter()
        set_seed(42)

        out_path = Path(out_model_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        if progress_cb:
            progress_cb(0.0, "Starting PatchCore training...")

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
        tile_cfg = infer_cfg.get("tile", {})
        tile_w = int(tile_cfg.get("tile_w", 512))
        tile_h = int(tile_cfg.get("tile_h", 352))
        stride_w = int(tile_cfg.get("stride_w", 384))
        stride_h = int(tile_cfg.get("stride_h", 352))

        # PatchCore-specific training params
        train_cfg = config.get("_train", {})
        backbone_name = str(train_cfg.get("backbone", "resnet18"))
        layers_str = str(train_cfg.get("layers", "layer2,layer3"))
        layers = [s.strip() for s in layers_str.split(",") if s.strip()]
        proj_dim = int(train_cfg.get("proj_dim", 128))
        max_patches_per_tile = int(train_cfg.get("max_patches_per_tile", 256))
        memory_size = int(train_cfg.get("memory_size", 20000))
        batch_size = int(train_cfg.get("batch_size", 16))
        num_workers = int(train_cfg.get("num_workers", 2))
        pad_mode = str(train_cfg.get("pad_mode", "edge"))
        compute_threshold = bool(train_cfg.get("compute_threshold", True))
        ok_quantile = float(train_cfg.get("ok_quantile", 0.999))
        thr_scale = float(train_cfg.get("thr_scale", 1.10))
        use_faiss = bool(train_cfg.get("use_faiss", True))
        infer_batch_tiles = int(infer_cfg.get("batch_tiles", 8))

        # --- Build model ---
        if progress_cb:
            progress_cb(10.0, f"Building backbone: {backbone_name}, layers: {layers}")

        backbone = ResNetFeat(backbone_name, pretrained=True, layers=layers)
        embedder = PatchEmbedder(
            backbone, layers=layers, proj_dim=proj_dim, seed=42
        ).to(device)
        embedder.eval()

        # --- Build memory bank ---
        if progress_cb:
            progress_cb(15.0, "Extracting patch embeddings from OK tiles...")

        memory_bank = train_memory_bank(
            img_paths=img_paths,
            embedder=embedder,
            tile_w=tile_w,
            tile_h=tile_h,
            stride_w=stride_w,
            stride_h=stride_h,
            device=device,
            max_patches_per_tile=max_patches_per_tile,
            memory_size=memory_size,
            batch_size=batch_size,
            num_workers=num_workers,
            pad_mode=pad_mode,
            progress_cb=progress_cb,
        )

        # --- Compute threshold ---
        threshold = None
        if compute_threshold:
            if progress_cb:
                progress_cb(80.0, "Computing auto threshold from OK set...")

            knn = KNNSearcher(memory_bank, use_faiss=use_faiss)
            threshold = compute_threshold_from_ok(
                img_paths=img_paths,
                embedder=embedder,
                knn=knn,
                tile_w=tile_w,
                tile_h=tile_h,
                stride_w=stride_w,
                stride_h=stride_h,
                device=device,
                pad_mode=pad_mode,
                batch_tiles=infer_batch_tiles,
                ok_quantile=ok_quantile,
                thr_scale=thr_scale,
                progress_cb=progress_cb,
            )

        # --- Save model ---
        if progress_cb:
            progress_cb(96.0, "Saving model...")

        _, mean, std = build_transform()
        version = time.strftime("%Y%m%d_%H%M%S")

        meta = {
            "backbone": backbone_name,
            "layers": layers,
            "proj_dim": proj_dim,
            "tile_w": tile_w,
            "tile_h": tile_h,
            "stride_w": stride_w,
            "stride_h": stride_h,
            "pad_mode": pad_mode,
            "mean": mean,
            "std": std,
            "threshold": threshold,
            "ok_quantile": ok_quantile,
            "thr_scale": thr_scale,
            "version": version,
            "created_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "n_ok_images": len(img_paths),
            "memory_bank_size": memory_bank.shape[0],
            "memory_bank_dim": memory_bank.shape[1],
        }

        pack = {
            "meta": meta,
            "state_dict": embedder.state_dict(),
            "memory_bank": memory_bank,
        }

        model_pt_path = out_path / "model.pt"
        torch.save(pack, str(model_pt_path))

        # Also save meta.json for UI consumption
        with open(out_path / "meta.json", "w", encoding="utf-8") as f:
            json.dump({**meta, "algo": self.name}, f, indent=2, ensure_ascii=False)

        if progress_cb:
            progress_cb(100.0, "Training complete!")

        duration = time.perf_counter() - t_start

        logger.info(
            "PatchCore training complete: version=%s, bank=%s, threshold=%s, duration=%.1fs",
            version,
            memory_bank.shape,
            threshold,
            duration,
        )

        return {
            "model_version": version,
            "metrics": {
                "n_ok_images": len(img_paths),
                "memory_bank_size": memory_bank.shape[0],
                "memory_bank_dim": memory_bank.shape[1],
            },
            "threshold": threshold,
            "duration_s": round(duration, 2),
        }

"""Projection Compare V1 — VisionHub plugin wrapper.

Column-projection-based adjacent-strip comparison for detecting alignment
deviations in periodic structures.  No neural network required — pure signal
processing on 1-D brightness profiles.

Usage in project.yaml:
    pipeline:
        algo: "projection_compare_v1"
        infer:
            strip:
                strip_size: 100
                strip_overlap: 50
                strip_axis: "auto"
        _train:
            metric: "l2"              # "l2" / "correlation" / "peak_shift" / "edge_shift"
            smooth_kernel: 5          # Gaussian smoothing kernel (odd int, 0=off)
            smooth_type: "gaussian"   # "gaussian" or "boxcar"
            trim_pct: 0.05            # Trimmed mean: discard top/bottom 5%
            align_max_shift: 0        # Pre-align profiles up to Npx (0=off)
            skip_edge_strips: 0       # Skip first/last N strips (0=off)
            projection_type: "texture"    # "mean" / "gradient" / "binary_edge" / "texture"
            binary_threshold: 50          # gray threshold for binary_edge only (0=auto Otsu)
            n_segments: 1             # V1.5: 1=full-strip, 2=top/bottom split for skew
            edge_band_width: 0        # V1.5: 0=off, 8=emphasise groove edges only
            row_normalize: false      # V1.5: per-row normalise to suppress colour drift
            ok_quantile: 0.999
            thr_scale: 1.5

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

from app.plugins.base import AlgoPluginBase
from app.plugins.patchcore_strip_core import (
    detect_long_axis,
    list_images,
    set_seed,
)
from app.plugins.projection_compare_core import (
    MetalMaskParams,
    build_diff_heatmap,
    compute_ok_boundary_distances,
    compute_ok_metal_mask_scores,
    compute_ok_pair_distances,
    compute_threshold_from_ok_distances,
    infer_boundary_compare,
    infer_metal_mask,
    infer_projection_pairs,
    save_metal_mask_overlay,
    save_projection_overlay_cv2,
)
from app.plugins.registry import PluginRegistry

logger = logging.getLogger(__name__)


@PluginRegistry.register
class ProjectionCompareV1Plugin(AlgoPluginBase):
    """
    Column-projection-based adjacent-strip comparison plugin.

    Slices images into tooth-sized strips along the long axis, computes
    a 1-D brightness profile for each strip by averaging perpendicular
    to the slicing axis, and compares adjacent profiles.  A shift in
    peak/valley positions indicates physical misalignment.

    No neural network — fast and lightweight.

    Supports:
    - Auto long-axis detection (vertical / horizontal)
    - Configurable strip size and overlap
    - L2 / correlation / peak-shift / edge-shift / boundary distance metrics
    - Metal-mask boundary detection (V3) for geometric anomaly detection
    - Configurable smoothing kernel
    - Diff heatmap overlay
    """

    name: str = "projection_compare_v1"

    def __init__(self) -> None:
        self._model_dir: str = ""
        self._device_str: str = "cpu"
        self._config: dict[str, Any] = {}
        self._loaded: bool = False
        self._model_version: str = ""
        self._threshold: float = 0.5

        # Parameters (populated on load)
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
        """Load saved projection parameters and threshold."""
        self._model_dir = model_dir
        self._device_str = device
        self._config = config

        model_path = Path(model_dir)
        meta_file = model_path / "meta.json"

        if not meta_file.exists():
            raise FileNotFoundError(f"Meta file not found: {meta_file}")

        logger.info("Loading Projection Compare model from %s ...", model_dir)

        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)

        self._meta = meta
        self._threshold = meta.get("threshold", 0.5) or 0.5
        self._model_version = meta.get("version", model_path.name) or model_path.name

        self._loaded = True
        logger.info(
            "Projection Compare model loaded: version=%s, threshold=%.6f, "
            "metric=%s, strip_size=%s, smooth_kernel=%s",
            self._model_version,
            self._threshold,
            meta.get("metric"),
            meta.get("strip_size"),
            meta.get("smooth_kernel"),
        )

    # ------------------------------------------------------------------
    # Unload
    # ------------------------------------------------------------------

    def unload(self) -> None:
        """Release resources."""
        self._meta = {}
        self._loaded = False
        self._model_version = ""
        logger.info("Projection Compare model unloaded.")

    # ------------------------------------------------------------------
    # Infer
    # ------------------------------------------------------------------

    def infer(self, image_path: str, config: dict[str, Any]) -> dict[str, Any]:
        """Run projection-based adjacent-pair comparison on a single image.

        Strip parameters come from saved model meta.
        Post-processing parameters come from project config.
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded")

        meta = self._meta
        job_id = config.get("_job_id", "unknown")
        output_dir = config.get("_output_dir", "")

        # Strip parameters from model meta
        strip_size = int(meta.get("strip_size", 256))
        strip_overlap = int(meta.get("strip_overlap", 64))
        strip_axis_str = meta.get("strip_axis", "auto")
        strip_axis = None if strip_axis_str == "auto" else strip_axis_str

        # Projection parameters from model meta
        metric = str(meta.get("metric", "l2"))
        smooth_kernel = int(meta.get("smooth_kernel", 5))
        trim_pct = float(meta.get("trim_pct", 0.05))
        smooth_type = str(meta.get("smooth_type", "gaussian"))
        align_max_shift = int(meta.get("align_max_shift", 0))
        skip_edge_strips = int(meta.get("skip_edge_strips", 0))
        projection_type = str(meta.get("projection_type", "mean"))
        binary_threshold = int(meta.get("binary_threshold", 50))
        n_segments = int(meta.get("n_segments", 1))
        edge_band_width = int(meta.get("edge_band_width", 0))
        row_normalize = bool(meta.get("row_normalize", False))

        # V2 boundary-trace parameters (left-only slot_score approach)
        boundary_block_h = int(meta.get("boundary_block_h", 8))
        boundary_slot_thr = float(meta.get("boundary_slot_thr", 0.55))
        boundary_envelope_win = int(meta.get("boundary_envelope_win", 9))
        boundary_ref_deg = int(meta.get("boundary_ref_deg", 3))
        boundary_skew_weight = float(meta.get("boundary_skew_weight", 2.0))

        # --- Inference ---
        t_infer_start = time.perf_counter()

        if metric == "metal_mask":
            # V3 metal-mask boundary detection — completely different path
            mm_params = self._build_metal_mask_params(meta)
            mm_result = infer_metal_mask(Path(image_path), mm_params)

            t_infer_end = time.perf_counter()
            infer_ms = (t_infer_end - t_infer_start) * 1000

            # Decision: use the event-based result directly
            pred = mm_result["result"]
            max_distance = mm_result["score"]

            # Build regions from NG/WARN events
            t_post_start = time.perf_counter()
            artifacts: dict[str, str] = {}
            regions: list[dict[str, Any]] = []
            h_img, w_img = mm_result["img_shape"]

            for ev in mm_result["events"]:
                if ev["level"] in ("NG", "WARN"):
                    regions.append({
                        "x": 0, "y": ev["y0"],
                        "w": w_img, "h": ev["y1"] - ev["y0"] + 1,
                        "score": ev["peak_px"],
                        "side": ev["side"],
                        "kind": ev["kind"],
                        "level": ev["level"],
                    })

            # Save overlay
            overlay_output_path = config.get("_overlay_output_path", "")
            if overlay_output_path:
                try:
                    save_metal_mask_overlay(
                        Path(overlay_output_path), Path(image_path), mm_result,
                    )
                    artifacts["overlay"] = overlay_output_path
                except Exception:
                    logger.warning(
                        "Failed to save metal mask overlay to %s for job %s",
                        overlay_output_path, job_id, exc_info=True,
                    )

            if output_dir:
                from datetime import datetime

                today = datetime.now().strftime("%Y-%m-%d")
                artifacts_dir = Path(output_dir) / today / "artifacts"
                artifacts_dir.mkdir(parents=True, exist_ok=True)
                base_name = str(job_id)

                postprocess = config.get("postprocess", {})
                export_cfg = postprocess.get("export", {})
                if not overlay_output_path and export_cfg.get("save_overlay", False):
                    try:
                        overlay_file = str(
                            artifacts_dir / f"{base_name}_overlay.jpg",
                        )
                        save_metal_mask_overlay(
                            Path(overlay_file), Path(image_path), mm_result,
                        )
                        artifacts["overlay"] = overlay_file
                    except Exception:
                        logger.warning(
                            "Failed to save metal mask overlay for job %s",
                            job_id, exc_info=True,
                        )

            t_post_end = time.perf_counter()
            post_ms = (t_post_end - t_post_start) * 1000

            # Extract signed residual extremes for easy TCP readout
            mm_stats = mm_result["stats"]
            residual_summary = {
                "L_max_pos_residual": round(float(mm_stats.get("L_max_pos_residual", 0.0)), 2),
                "L_max_neg_residual": round(float(mm_stats.get("L_max_neg_residual", 0.0)), 2),
                "R_max_pos_residual": round(float(mm_stats.get("R_max_pos_residual", 0.0)), 2),
                "R_max_neg_residual": round(float(mm_stats.get("R_max_neg_residual", 0.0)), 2),
            }

            # Merge core timing detail into plugin timing
            core_timing = mm_result.get("timing_ms", {})
            merged_timing: dict[str, Any] = {
                "infer": round(infer_ms, 2),
                "post": round(post_ms, 2),
            }
            for k, v in core_timing.items():
                merged_timing[k] = round(v, 2)

            return {
                "score": round(float(max_distance), 6),
                "threshold": round(float(self._threshold), 6),
                "pred": pred,
                "regions": regions,
                "events": mm_result["events"],
                "stats": mm_result["stats"],
                "residual_summary": residual_summary,
                "artifacts": artifacts,
                "timing_ms": merged_timing,
                "model_version": self._model_version,
            }

        if metric == "boundary":
            max_distance, pair_results, (h0, w0), profiles, diff_curves = (
                infer_boundary_compare(
                    img_path=Path(image_path),
                    strip_size=strip_size,
                    strip_overlap=strip_overlap,
                    axis=strip_axis,
                    skip_edge_strips=skip_edge_strips,
                    boundary_block_h=boundary_block_h,
                    boundary_slot_thr=boundary_slot_thr,
                    boundary_envelope_win=boundary_envelope_win,
                    boundary_ref_deg=boundary_ref_deg,
                    boundary_skew_weight=boundary_skew_weight,
                )
            )
        else:
            max_distance, pair_results, (h0, w0), profiles, diff_curves = (
                infer_projection_pairs(
                    img_path=Path(image_path),
                    strip_size=strip_size,
                    strip_overlap=strip_overlap,
                    metric=metric,
                    axis=strip_axis,
                    smooth_kernel=smooth_kernel,
                    trim_pct=trim_pct,
                    smooth_type=smooth_type,
                    align_max_shift=align_max_shift,
                    skip_edge_strips=skip_edge_strips,
                    projection_type=projection_type,
                    binary_threshold=binary_threshold,
                    n_segments=n_segments,
                    edge_band_width=edge_band_width,
                    row_normalize=row_normalize,
                )
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
            rx = min(ba["x"], bb["x"])
            ry = min(ba["y"], bb["y"])
            rw = max(ba["x"] + ba["w"], bb["x"] + bb["w"]) - rx
            rh = max(ba["y"] + ba["h"], bb["y"] + bb["h"]) - ry
            regions.append({
                "x": rx, "y": ry, "w": rw, "h": rh,
                "score": pr["distance"],
                "pair_idx": pr["pair_idx"],
            })

        # Build diff heatmap
        resolved_axis = strip_axis if strip_axis else detect_long_axis(h0, w0)
        diff_heatmap = build_diff_heatmap(
            diff_curves, pair_results, h0, w0, resolved_axis,
        )

        # Fixed overlay output path
        overlay_output_path = config.get("_overlay_output_path", "")

        if overlay_output_path:
            try:
                save_projection_overlay_cv2(
                    out_path=Path(overlay_output_path),
                    img_path=Path(image_path),
                    pair_results=pair_results,
                    threshold=threshold,
                    max_distance=max_distance,
                    pred=pred,
                    profiles=profiles,
                    diff_curves=diff_curves,
                    slicing_axis=resolved_axis,
                    diff_heatmap=diff_heatmap,
                )
                artifacts["overlay"] = overlay_output_path
            except Exception:
                logger.warning(
                    "Failed to save projection overlay to %s for job %s",
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
                    save_projection_overlay_cv2(
                        out_path=Path(overlay_file),
                        img_path=Path(image_path),
                        pair_results=pair_results,
                        threshold=threshold,
                        max_distance=max_distance,
                        pred=pred,
                        profiles=profiles,
                        diff_curves=diff_curves,
                        slicing_axis=resolved_axis,
                        diff_heatmap=diff_heatmap,
                    )
                    artifacts["overlay"] = overlay_file
                except Exception:
                    logger.warning(
                        "Failed to save projection overlay for job %s",
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
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_metal_mask_params(meta: dict[str, Any]) -> MetalMaskParams:
        """Build MetalMaskParams from saved model meta dict."""
        mm_cfg = meta.get("metal_mask", {})
        return MetalMaskParams(
            blur_ksize=int(mm_cfg.get("blur_ksize", 3)),
            blur_sigma=float(mm_cfg.get("blur_sigma", 0.6)),
            std_kernel=int(mm_cfg.get("std_kernel", 5)),
            lap_kernel=int(mm_cfg.get("lap_kernel", 3)),
            sobel_kernel=int(mm_cfg.get("sobel_kernel", 3)),
            norm_p_low=float(mm_cfg.get("norm_p_low", 1.0)),
            norm_p_high=float(mm_cfg.get("norm_p_high", 99.0)),
            w_std=float(mm_cfg.get("w_std", 0.42)),
            w_lap=float(mm_cfg.get("w_lap", 0.33)),
            w_gy=float(mm_cfg.get("w_gy", 0.25)),
            w_penalty=float(mm_cfg.get("w_penalty", 0.18)),
            seed_border_w_ratio=float(mm_cfg.get("seed_border_w_ratio", 0.03)),
            groove_w_ratio=float(mm_cfg.get("groove_w_ratio", 0.10)),
            left_alpha_hi=float(mm_cfg.get("left_alpha_hi", 0.55)),
            left_alpha_lo=float(mm_cfg.get("left_alpha_lo", 0.35)),
            right_alpha_hi=float(mm_cfg.get("right_alpha_hi", 0.62)),
            right_alpha_lo=float(mm_cfg.get("right_alpha_lo", 0.40)),
            pitch_min=int(mm_cfg.get("pitch_min", 25)),
            pitch_max=int(mm_cfg.get("pitch_max", 70)),
            morph_close_h_ratio=float(mm_cfg.get("morph_close_h_ratio", 0.25)),
            morph_open_h_ratio=float(mm_cfg.get("morph_open_h_ratio", 0.08)),
            boundary_median_ratio=float(mm_cfg.get("boundary_median_ratio", 0.12)),
            boundary_sg_ratio=float(mm_cfg.get("boundary_sg_ratio", 0.80)),
            ref_trend_ratio=float(mm_cfg.get("ref_trend_ratio", 3.0)),
            template_smooth=int(mm_cfg.get("template_smooth", 7)),
            template_rows_start_frac=float(mm_cfg.get("template_rows_start_frac", 0.35)),
            template_dev_sigma=float(mm_cfg.get("template_dev_sigma", 2.5)),
            event_sigma_mult=float(mm_cfg.get("event_sigma_mult", 3.0)),
            event_min_px=float(mm_cfg.get("event_min_px", 2.0)),
            min_run_ratio=float(mm_cfg.get("min_run_ratio", 0.25)),
            gap_merge_ratio=float(mm_cfg.get("gap_merge_ratio", 0.10)),
            ng_peak_thr=float(mm_cfg.get("ng_peak_thr", 4.0)),
            ng_height_ratio=float(mm_cfg.get("ng_height_ratio", 0.30)),
            ng_area_thr=float(mm_cfg.get("ng_area_thr", 35.0)),
            ng_area_height_ratio=float(mm_cfg.get("ng_area_height_ratio", 0.25)),
            warn_peak_thr=float(mm_cfg.get("warn_peak_thr", 2.5)),
            warn_height_ratio=float(mm_cfg.get("warn_height_ratio", 0.20)),
            ignore_top_left=float(mm_cfg.get("ignore_top_left", 0.0)),
            ignore_top_right=float(mm_cfg.get("ignore_top_right", 0.20)),
            ignore_missing_right=bool(mm_cfg.get("ignore_missing_right", True)),
            dipole_enabled=bool(mm_cfg.get("dipole_enabled", True)),
            dipole_area_weight=float(mm_cfg.get("dipole_area_weight", 0.03)),
            dipole_dist_ratio=float(mm_cfg.get("dipole_dist_ratio", 0.6)),
            dipole_ng_thr=float(mm_cfg.get("dipole_ng_thr", 5.5)),
            dipole_warn_thr=float(mm_cfg.get("dipole_warn_thr", 4.0)),
            adj_diff_enabled=bool(mm_cfg.get("adj_diff_enabled", False)),
            adj_diff_ng_thr=float(mm_cfg.get("adj_diff_ng_thr", 5.0)),
            adj_diff_warn_thr=float(mm_cfg.get("adj_diff_warn_thr", 3.0)),
            adj_diff_overlay=bool(mm_cfg.get("adj_diff_overlay", True)),
            right_fit_offset=float(mm_cfg.get("right_fit_offset", 160.0)),
            area_depth_thr=float(mm_cfg.get("area_depth_thr", 15.0)),
        )

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
        """Train projection compare model from OK samples.

        Steps:
        1. Load OK images from dataset_dir/ok/
        2. Compute OK pair distance distribution (column projections)
        3. Set threshold from distribution
        4. Save meta.json with parameters + threshold
        """
        t_start = time.perf_counter()
        set_seed(42)

        out_path = Path(out_model_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        if progress_cb:
            progress_cb(0.0, "Starting Projection Compare training...")

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

        # Projection-specific training params
        train_cfg = config.get("_train", {})
        metric = str(train_cfg.get("metric", "l2"))
        smooth_kernel = int(train_cfg.get("smooth_kernel", 5))
        trim_pct = float(train_cfg.get("trim_pct", 0.05))
        smooth_type = str(train_cfg.get("smooth_type", "gaussian"))
        align_max_shift = int(train_cfg.get("align_max_shift", 0))
        skip_edge_strips = int(train_cfg.get("skip_edge_strips", 0))
        projection_type = str(train_cfg.get("projection_type", "mean"))
        binary_threshold = int(train_cfg.get("binary_threshold", 50))
        n_segments = int(train_cfg.get("n_segments", 1))
        edge_band_width = int(train_cfg.get("edge_band_width", 0))
        row_normalize = bool(train_cfg.get("row_normalize", False))
        ok_quantile = float(train_cfg.get("ok_quantile", 0.999))
        thr_scale = float(train_cfg.get("thr_scale", 1.5))

        # V2 boundary-trace parameters (left-only slot_score approach)
        boundary_block_h = int(train_cfg.get("boundary_block_h", 8))
        boundary_slot_thr = float(train_cfg.get("boundary_slot_thr", 0.55))
        boundary_envelope_win = int(train_cfg.get("boundary_envelope_win", 9))
        boundary_ref_deg = int(train_cfg.get("boundary_ref_deg", 3))
        boundary_skew_weight = float(train_cfg.get("boundary_skew_weight", 2.0))

        # --- Compute OK pair distances ---
        if progress_cb:
            progress_cb(10.0, "Computing OK pair projection distances...")

        if metric == "metal_mask":
            # V3 metal-mask mode: build params from _train config
            mm_train_cfg = train_cfg.get("metal_mask", {})
            mm_params = MetalMaskParams(
                blur_ksize=int(mm_train_cfg.get("blur_ksize", 3)),
                blur_sigma=float(mm_train_cfg.get("blur_sigma", 0.6)),
                std_kernel=int(mm_train_cfg.get("std_kernel", 5)),
                lap_kernel=int(mm_train_cfg.get("lap_kernel", 3)),
                sobel_kernel=int(mm_train_cfg.get("sobel_kernel", 3)),
                norm_p_low=float(mm_train_cfg.get("norm_p_low", 1.0)),
                norm_p_high=float(mm_train_cfg.get("norm_p_high", 99.0)),
                w_std=float(mm_train_cfg.get("w_std", 0.42)),
                w_lap=float(mm_train_cfg.get("w_lap", 0.33)),
                w_gy=float(mm_train_cfg.get("w_gy", 0.25)),
                w_penalty=float(mm_train_cfg.get("w_penalty", 0.18)),
                seed_border_w_ratio=float(mm_train_cfg.get("seed_border_w_ratio", 0.03)),
                groove_w_ratio=float(mm_train_cfg.get("groove_w_ratio", 0.10)),
                left_alpha_hi=float(mm_train_cfg.get("left_alpha_hi", 0.55)),
                left_alpha_lo=float(mm_train_cfg.get("left_alpha_lo", 0.35)),
                right_alpha_hi=float(mm_train_cfg.get("right_alpha_hi", 0.62)),
                right_alpha_lo=float(mm_train_cfg.get("right_alpha_lo", 0.40)),
                pitch_min=int(mm_train_cfg.get("pitch_min", 25)),
                pitch_max=int(mm_train_cfg.get("pitch_max", 70)),
                morph_close_h_ratio=float(mm_train_cfg.get("morph_close_h_ratio", 0.25)),
                morph_open_h_ratio=float(mm_train_cfg.get("morph_open_h_ratio", 0.08)),
                boundary_median_ratio=float(mm_train_cfg.get("boundary_median_ratio", 0.12)),
                boundary_sg_ratio=float(mm_train_cfg.get("boundary_sg_ratio", 0.80)),
                ref_trend_ratio=float(mm_train_cfg.get("ref_trend_ratio", 3.0)),
                template_smooth=int(mm_train_cfg.get("template_smooth", 7)),
                template_rows_start_frac=float(mm_train_cfg.get("template_rows_start_frac", 0.35)),
                template_dev_sigma=float(mm_train_cfg.get("template_dev_sigma", 2.5)),
                event_sigma_mult=float(mm_train_cfg.get("event_sigma_mult", 3.0)),
                event_min_px=float(mm_train_cfg.get("event_min_px", 2.0)),
                min_run_ratio=float(mm_train_cfg.get("min_run_ratio", 0.25)),
                gap_merge_ratio=float(mm_train_cfg.get("gap_merge_ratio", 0.10)),
                ng_peak_thr=float(mm_train_cfg.get("ng_peak_thr", 4.0)),
                ng_height_ratio=float(mm_train_cfg.get("ng_height_ratio", 0.30)),
                ng_area_thr=float(mm_train_cfg.get("ng_area_thr", 35.0)),
                ng_area_height_ratio=float(mm_train_cfg.get("ng_area_height_ratio", 0.25)),
                warn_peak_thr=float(mm_train_cfg.get("warn_peak_thr", 2.5)),
                warn_height_ratio=float(mm_train_cfg.get("warn_height_ratio", 0.20)),
                ignore_top_left=float(mm_train_cfg.get("ignore_top_left", 0.0)),
                ignore_top_right=float(mm_train_cfg.get("ignore_top_right", 0.20)),
                ignore_missing_right=bool(mm_train_cfg.get("ignore_missing_right", True)),
                dipole_enabled=bool(mm_train_cfg.get("dipole_enabled", True)),
                dipole_area_weight=float(mm_train_cfg.get("dipole_area_weight", 0.03)),
                dipole_dist_ratio=float(mm_train_cfg.get("dipole_dist_ratio", 0.6)),
                dipole_ng_thr=float(mm_train_cfg.get("dipole_ng_thr", 5.5)),
                dipole_warn_thr=float(mm_train_cfg.get("dipole_warn_thr", 4.0)),
            )
            ok_distances = compute_ok_metal_mask_scores(
                img_paths=img_paths,
                params=mm_params,
                progress_cb=progress_cb,
            )
        elif metric == "boundary":
            ok_distances = compute_ok_boundary_distances(
                img_paths=img_paths,
                strip_size=strip_size,
                strip_overlap=strip_overlap,
                axis=strip_axis,
                skip_edge_strips=skip_edge_strips,
                boundary_block_h=boundary_block_h,
                boundary_slot_thr=boundary_slot_thr,
                boundary_envelope_win=boundary_envelope_win,
                boundary_ref_deg=boundary_ref_deg,
                boundary_skew_weight=boundary_skew_weight,
                progress_cb=progress_cb,
            )
        else:
            ok_distances = compute_ok_pair_distances(
                img_paths=img_paths,
                strip_size=strip_size,
                strip_overlap=strip_overlap,
                metric=metric,
                axis=strip_axis,
                smooth_kernel=smooth_kernel,
                trim_pct=trim_pct,
                smooth_type=smooth_type,
                align_max_shift=align_max_shift,
                skip_edge_strips=skip_edge_strips,
                projection_type=projection_type,
                binary_threshold=binary_threshold,
                n_segments=n_segments,
                edge_band_width=edge_band_width,
                row_normalize=row_normalize,
                progress_cb=progress_cb,
            )

        if progress_cb:
            progress_cb(85.0, "Computing threshold from OK distance distribution...")

        threshold = compute_threshold_from_ok_distances(
            ok_distances=ok_distances,
            ok_quantile=ok_quantile,
            thr_scale=thr_scale,
        )

        # --- Save meta.json ---
        if progress_cb:
            progress_cb(95.0, "Saving model...")

        version = time.strftime("%Y%m%d_%H%M%S")

        meta: dict[str, Any] = {
            "algo": self.name,
            "strip_size": strip_size,
            "strip_overlap": strip_overlap,
            "strip_axis": strip_axis_str,
            "metric": metric,
            "smooth_kernel": smooth_kernel,
            "trim_pct": trim_pct,
            "smooth_type": smooth_type,
            "align_max_shift": align_max_shift,
            "skip_edge_strips": skip_edge_strips,
            "projection_type": projection_type,
            "binary_threshold": binary_threshold,
            "n_segments": n_segments,
            "edge_band_width": edge_band_width,
            "row_normalize": row_normalize,
            "boundary_block_h": boundary_block_h,
            "boundary_slot_thr": boundary_slot_thr,
            "boundary_envelope_win": boundary_envelope_win,
            "boundary_ref_deg": boundary_ref_deg,
            "boundary_skew_weight": boundary_skew_weight,
            "threshold": threshold,
            "ok_quantile": ok_quantile,
            "thr_scale": thr_scale,
            "version": version,
            "created_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "n_ok_images": len(img_paths),
            "n_ok_pairs": len(ok_distances),
        }

        # Save metal_mask params in meta for inference reload
        if metric == "metal_mask":
            from dataclasses import asdict as _asdict
            meta["metal_mask"] = _asdict(mm_params)

        with open(out_path / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

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
            "Projection Compare training complete: version=%s, "
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
                "metric": metric,
                "smooth_kernel": smooth_kernel,
                "strip_size": strip_size,
                "strip_overlap": strip_overlap,
                "ok_distance_stats": dist_stats,
            },
            "threshold": threshold,
            "duration_s": round(duration, 2),
        }

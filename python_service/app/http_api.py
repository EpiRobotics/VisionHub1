"""FastAPI HTTP Control API for VisionHub service.

Provides endpoints for the C# UI to manage projects, trigger training,
run test inference, and monitor status.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Request/Response models
# ---------------------------------------------------------------------------

class EnableRequest(BaseModel):
    enabled: bool = True


class TrainRequest(BaseModel):
    auto_activate: bool = True


class InferRequest(BaseModel):
    image_path: str
    job_id: str = ""
    options: dict[str, Any] = Field(default_factory=dict)


class SetModelRequest(BaseModel):
    version: str


class LabelCropRequest(BaseModel):
    """Request to crop glyphs from JSON annotations (Step 1 of label training)."""
    image_dir: str
    json_dir: str
    output_dir: str
    pad: int = 2


class LabelTrainRequest(BaseModel):
    """Request to train glyph PatchCore models (Step 2 of label training)."""
    bank_dir: str
    output_model_dir: str
    project_id: str = ""
    auto_activate: bool = True
    img_size: int = 128
    max_patches_per_class: int = 30000
    k: int = 1
    score_mode: str = "topk"
    topk: int = 10
    p_thr: float = 0.995


class ProjectSummary(BaseModel):
    project_id: str
    display_name: str = ""
    enabled: bool = True
    tcp_port: int = 0
    algo: str = ""
    model_loaded: bool = False
    model_version: str = ""
    tcp_running: bool = False
    stats: dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    ok: bool = True
    service: str = "visionhub"
    version: str = "0.1.0"
    uptime_s: float = 0.0
    projects_count: int = 0
    gpu_workers: int = 1
    timestamp: str = ""


# ---------------------------------------------------------------------------
# Helper: persist project config to YAML
# ---------------------------------------------------------------------------

def _save_project_yaml(state: Any) -> None:
    """Write the current ProjectConfig back to project.yaml so settings persist."""
    yaml_path = Path(state.project_dir) / "project.yaml"
    data = state.config.model_dump()
    # Remove internal fields that start with '_' (like _train)
    # but keep them if they exist in the original file
    try:
        if yaml_path.exists():
            with open(yaml_path, "r", encoding="utf-8") as f:
                original = yaml.safe_load(f) or {}
            # Preserve _train and other underscore keys from original
            for key, val in original.items():
                if key.startswith("_") and key not in data:
                    data[key] = val
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        logger.info("Saved project config to %s", yaml_path)
    except Exception:
        logger.exception("Failed to save project config to %s", yaml_path)


# ---------------------------------------------------------------------------
# API factory
# ---------------------------------------------------------------------------

def create_api(app_state: Any) -> FastAPI:
    """
    Create the FastAPI application with all endpoints.

    Args:
        app_state: The main ServiceApp instance that provides access to
                   ProjectManager, Scheduler, TrainManager, etc.
    """
    api = FastAPI(
        title="VisionHub AI Service",
        description="Industrial Visual Inspection Unified Platform - Control API",
        version="0.1.0",
    )

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    @api.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        uptime = time.monotonic() - app_state.start_time
        return HealthResponse(
            ok=True,
            service="visionhub",
            version="0.1.0",
            uptime_s=round(uptime, 2),
            projects_count=len(app_state.project_manager.projects),
            gpu_workers=app_state.service_config.gpu.workers,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        )

    # ------------------------------------------------------------------
    # Projects
    # ------------------------------------------------------------------

    @api.get("/projects", response_model=list[ProjectSummary])
    async def list_projects() -> list[dict[str, Any]]:
        return app_state.project_manager.list_projects()

    @api.get("/projects/{project_id}")
    async def get_project(project_id: str) -> dict[str, Any]:
        state = app_state.project_manager.get_project(project_id)
        if state is None:
            raise HTTPException(status_code=404, detail=f"Project not found: {project_id}")
        return {
            "project_id": state.project_id,
            "display_name": state.config.display_name,
            "enabled": state.enabled,
            "tcp_port": state.config.io.tcp_port,
            "algo": state.config.pipeline.algo,
            "model_loaded": state.is_model_loaded,
            "model_version": state.model_version,
            "tcp_running": state.tcp_running,
            "config": state.config.model_dump(),
            "stats": state.stats.to_dict(),
        }

    @api.post("/projects/reload")
    async def reload_all_projects() -> dict[str, Any]:
        loaded = app_state.project_manager.reload_all()
        # Re-setup TCP servers and queues for any new/changed projects
        await app_state.setup_projects()
        return {"ok": True, "loaded_projects": loaded}

    @api.post("/projects/{project_id}/reload")
    async def reload_project(project_id: str) -> dict[str, Any]:
        success = app_state.project_manager.reload_project(project_id)
        if not success:
            raise HTTPException(status_code=404, detail=f"Project not found or reload failed: {project_id}")
        return {"ok": True, "project_id": project_id}

    @api.post("/projects/{project_id}/enable")
    async def enable_project(project_id: str, req: EnableRequest) -> dict[str, Any]:
        success = app_state.project_manager.enable_project(project_id, req.enabled)
        if not success:
            raise HTTPException(status_code=404, detail=f"Project not found: {project_id}")

        state = app_state.project_manager.get_project(project_id)
        if state is None:
            raise HTTPException(status_code=404, detail=f"Project not found: {project_id}")

        if req.enabled:
            # Start TCP server and load model if not already
            await app_state.start_project(project_id)
        else:
            # Stop TCP server
            await app_state.stop_project(project_id)

        return {"ok": True, "project_id": project_id, "enabled": req.enabled}

    # ------------------------------------------------------------------
    # Model management
    # ------------------------------------------------------------------

    @api.post("/projects/{project_id}/load_model")
    async def load_model(project_id: str) -> dict[str, Any]:
        success = app_state.project_manager.load_model(project_id)
        if not success:
            raise HTTPException(status_code=500, detail=f"Failed to load model for project: {project_id}")
        state = app_state.project_manager.get_project(project_id)
        return {
            "ok": True,
            "project_id": project_id,
            "model_version": state.model_version if state else "",
        }

    @api.post("/projects/{project_id}/set_model")
    async def set_model(project_id: str, req: SetModelRequest) -> dict[str, Any]:
        success = app_state.project_manager.switch_model(project_id, req.version)
        if not success:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to switch model for project {project_id} to version {req.version}",
            )
        return {"ok": True, "project_id": project_id, "version": req.version}

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    @api.post("/projects/{project_id}/train")
    async def start_training(project_id: str, req: TrainRequest) -> dict[str, Any]:
        state = app_state.project_manager.get_project(project_id)
        if state is None:
            raise HTTPException(status_code=404, detail=f"Project not found: {project_id}")

        dataset_dir = str(Path(state.project_dir) / "datasets")
        if not Path(dataset_dir).exists():
            raise HTTPException(status_code=400, detail=f"Dataset directory not found: {dataset_dir}")

        try:
            job = app_state.train_manager.start_training(
                project_id=project_id,
                dataset_dir=dataset_dir,
                project_dir=state.project_dir,
                algo_name=state.config.pipeline.algo,
                pipeline_config=state.config.pipeline.model_dump(),
                auto_activate=req.auto_activate,
            )
            return {"ok": True, "train_job_id": job.train_job_id, "project_id": project_id}
        except RuntimeError as e:
            raise HTTPException(status_code=409, detail=str(e))

    @api.get("/train/{train_job_id}")
    async def get_train_status(train_job_id: str) -> dict[str, Any]:
        job = app_state.train_manager.get_job(train_job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Training job not found: {train_job_id}")
        return job.to_dict()

    # ------------------------------------------------------------------
    # Test inference (via HTTP, for UI testing)
    # ------------------------------------------------------------------

    @api.post("/projects/{project_id}/infer")
    async def test_infer(project_id: str, req: InferRequest) -> dict[str, Any]:
        state = app_state.project_manager.get_project(project_id)
        if state is None:
            raise HTTPException(status_code=404, detail=f"Project not found: {project_id}")
        if not state.enabled:
            raise HTTPException(status_code=400, detail=f"Project is disabled: {project_id}")

        job_id = req.job_id or f"http_{int(time.time()*1000)}"

        # Persist last test image path so UI can pre-fill it on restart
        if req.image_path and req.image_path != state.config.io.test_image_path:
            state.config.io.test_image_path = req.image_path
            _save_project_yaml(state)

        try:
            result = await app_state.run_inference(project_id, job_id, req.image_path, req.options)
            return result.to_dict()
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    @api.get("/projects/{project_id}/stats")
    async def get_project_stats(project_id: str) -> dict[str, Any]:
        state = app_state.project_manager.get_project(project_id)
        if state is None:
            raise HTTPException(status_code=404, detail=f"Project not found: {project_id}")

        queue = app_state.gpu_pool.get_queue(project_id)
        queue_info = {"queue_size": queue.size, "max_size": queue.max_size} if queue else {}

        return {
            "project_id": project_id,
            "stats": state.stats.to_dict(),
            "queue": queue_info,
            "model_loaded": state.is_model_loaded,
            "model_version": state.model_version,
        }

    @api.get("/queues")
    async def get_all_queues() -> dict[str, Any]:
        return app_state.gpu_pool.get_all_queue_stats()

    # ------------------------------------------------------------------
    # Project Logs
    # ------------------------------------------------------------------

    @api.get("/projects/{project_id}/logs")
    async def get_project_logs(project_id: str, since: int = 0) -> dict[str, Any]:
        state = app_state.project_manager.get_project(project_id)
        if state is None:
            raise HTTPException(status_code=404, detail=f"Project not found: {project_id}")
        entries, next_index = state.log_buffer.get_entries(since)
        return {"entries": entries, "next_index": next_index}

    @api.post("/projects/{project_id}/logs/clear")
    async def clear_project_logs(project_id: str) -> dict[str, Any]:
        state = app_state.project_manager.get_project(project_id)
        if state is None:
            raise HTTPException(status_code=404, detail=f"Project not found: {project_id}")
        state.log_buffer.clear()
        return {"ok": True}

    # ------------------------------------------------------------------
    # Project Runtime Config
    # ------------------------------------------------------------------

    @api.post("/projects/{project_id}/set_overlay_path")
    async def set_overlay_path(project_id: str, overlay_path: str = "") -> dict[str, Any]:
        """Set the fixed overlay output path for a project at runtime.

        When set, every inference will write overlay to this exact file
        (overwriting each time). Another vision software can monitor
        this file to check OK/NG by red bounding boxes.
        Set to empty string to revert to per-job artifact mode.
        Saves to project.yaml so it persists across restarts.
        """
        state = app_state.project_manager.get_project(project_id)
        if state is None:
            raise HTTPException(status_code=404, detail=f"Project not found: {project_id}")
        state.config.io.overlay_output_path = overlay_path
        _save_project_yaml(state)
        return {
            "ok": True,
            "project_id": project_id,
            "overlay_output_path": overlay_path,
        }

    @api.post("/projects/{project_id}/set_threshold")
    async def set_threshold(project_id: str, thr_global: float | None = None) -> dict[str, Any]:
        """Set the global NG threshold for a project at runtime.

        For glyph_patchcore_v1: overrides per-class thresholds.
        A glyph with score >= thr_global is judged NG.
        Set to null (omit parameter) to revert to per-class trained thresholds.
        Saves to project.yaml so it persists across restarts.
        """
        state = app_state.project_manager.get_project(project_id)
        if state is None:
            raise HTTPException(status_code=404, detail=f"Project not found: {project_id}")

        # Store in the pipeline postprocess.decision config so it persists for each inference
        state.config.pipeline.postprocess.decision.thr_global = thr_global
        _save_project_yaml(state)

        return {
            "ok": True,
            "project_id": project_id,
            "thr_global": thr_global,
        }

    # ------------------------------------------------------------------
    # Label Training Workflow
    # ------------------------------------------------------------------

    # In-memory state for label training jobs
    _label_train_jobs: dict[str, dict[str, Any]] = {}

    @api.post("/label/crop")
    async def label_crop_glyphs(req: LabelCropRequest) -> dict[str, Any]:
        """Step 1: Crop glyphs from JSON annotations into glyph_bank structure.

        Reads images from image_dir, JSON annotations from json_dir,
        and saves per-character crops to output_dir/<ch>/*.jpg.
        """
        from app.plugins.glyph_patchcore_core import crop_glyphs_from_json

        img_dir = Path(req.image_dir)
        json_dir = Path(req.json_dir)
        out_dir = Path(req.output_dir)

        if not img_dir.exists():
            raise HTTPException(status_code=400, detail=f"Image directory not found: {req.image_dir}")
        if not json_dir.exists():
            raise HTTPException(status_code=400, detail=f"JSON directory not found: {req.json_dir}")

        json_files = sorted(json_dir.glob("*.json"))
        if not json_files:
            raise HTTPException(status_code=400, detail=f"No JSON files found in: {req.json_dir}")

        out_dir.mkdir(parents=True, exist_ok=True)

        total_crops = 0
        processed_files = 0
        errors: list[str] = []

        for jf in json_files:
            try:
                n = crop_glyphs_from_json(
                    json_path=jf,
                    img_dir=img_dir,
                    out_dir=out_dir,
                    pad=req.pad,
                )
                total_crops += n
                processed_files += 1
            except Exception as e:
                errors.append(f"{jf.name}: {e}")

        # Scan output directory for class summary
        class_summary: list[dict[str, Any]] = []
        if out_dir.exists():
            for cls_dir in sorted(out_dir.iterdir()):
                if cls_dir.is_dir():
                    img_count = len([f for f in cls_dir.iterdir()
                                     if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}])
                    class_summary.append({"class": cls_dir.name, "count": img_count})

        return {
            "ok": True,
            "total_crops": total_crops,
            "processed_files": processed_files,
            "total_json_files": len(json_files),
            "classes": class_summary,
            "errors": errors[:20],  # Limit error list
        }

    @api.post("/label/scan_bank")
    async def label_scan_bank(bank_dir: str = "") -> dict[str, Any]:
        """Scan a glyph_bank directory and return class statistics."""
        bp = Path(bank_dir)
        if not bp.exists():
            raise HTTPException(status_code=400, detail=f"Bank directory not found: {bank_dir}")

        classes: list[dict[str, Any]] = []
        total_images = 0
        for cls_dir in sorted(bp.iterdir()):
            if cls_dir.is_dir():
                img_count = len([f for f in cls_dir.iterdir()
                                 if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}])
                classes.append({"class": cls_dir.name, "count": img_count})
                total_images += img_count

        return {
            "ok": True,
            "bank_dir": bank_dir,
            "total_classes": len(classes),
            "total_images": total_images,
            "classes": classes,
        }

    @api.post("/label/train")
    async def label_start_training(req: LabelTrainRequest) -> dict[str, Any]:
        """Step 2: Train glyph PatchCore models from glyph_bank.

        Starts training in a background thread and returns a job_id
        for polling progress via GET /label/train/{job_id}.
        """
        bp = Path(req.bank_dir)
        if not bp.exists():
            raise HTTPException(status_code=400, detail=f"Bank directory not found: {req.bank_dir}")

        out_path = Path(req.output_model_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        job_id = f"label_train_{int(time.time())}"

        job_state: dict[str, Any] = {
            "job_id": job_id,
            "status": "running",
            "progress": 0.0,
            "message": "Starting...",
            "log_lines": [],
            "result": None,
            "error": None,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "completed_at": "",
        }
        _label_train_jobs[job_id] = job_state

        def _run() -> None:
            from app.plugins.glyph_patchcore_core import train_glyph_patchcore

            try:
                def progress_cb(pct: float, msg: str) -> None:
                    job_state["progress"] = pct
                    job_state["message"] = msg
                    ts = time.strftime("%H:%M:%S")
                    job_state["log_lines"].append(f"[{ts}] [{pct:.1f}%] {msg}")

                result = train_glyph_patchcore(
                    bank_dir=req.bank_dir,
                    out_model_dir=req.output_model_dir,
                    img_size=req.img_size,
                    max_patches_per_class=req.max_patches_per_class,
                    k=req.k,
                    score_mode=req.score_mode,
                    topk=req.topk,
                    p_thr=req.p_thr,
                    progress_cb=progress_cb,
                )

                job_state["status"] = "completed"
                job_state["progress"] = 100.0
                job_state["message"] = f"Training complete: {result.get('trained_classes', 0)} classes"
                job_state["result"] = result
                job_state["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")

                # Auto-activate model for the specified project
                if req.auto_activate and req.project_id:
                    state = app_state.project_manager.get_project(req.project_id)
                    if state:
                        import json as json_mod
                        active_path = Path(state.project_dir) / "active_model.json"
                        version = out_path.name
                        with open(active_path, "w", encoding="utf-8") as f:
                            json_mod.dump(
                                {"version": version, "model_dir": str(out_path),
                                 "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S")},
                                f, indent=2,
                            )
                        ts = time.strftime("%H:%M:%S")
                        job_state["log_lines"].append(
                            f"[{ts}] Auto-activated model '{version}' for project '{req.project_id}'")

            except Exception as e:
                job_state["status"] = "failed"
                job_state["error"] = str(e)
                job_state["message"] = f"Training failed: {e}"
                job_state["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
                logger.exception("Label training job %s failed", job_id)

        thread = threading.Thread(target=_run, daemon=True, name=f"label-train-{job_id}")
        thread.start()

        return {"ok": True, "job_id": job_id}

    @api.get("/label/train/{job_id}")
    async def label_get_train_status(job_id: str) -> dict[str, Any]:
        """Poll label training job progress."""
        job = _label_train_jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Label training job not found: {job_id}")
        return {
            "job_id": job["job_id"],
            "status": job["status"],
            "progress": job["progress"],
            "message": job["message"],
            "log_lines": job["log_lines"][-100:],
            "result": job["result"],
            "error": job["error"],
            "started_at": job["started_at"],
            "completed_at": job["completed_at"],
        }

    return api

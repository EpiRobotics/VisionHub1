"""FastAPI HTTP Control API for VisionHub service.

Provides endpoints for the C# UI to manage projects, trigger training,
run test inference, and monitor status.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

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

    return api

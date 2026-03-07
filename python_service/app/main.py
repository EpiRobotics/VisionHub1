"""VisionHub AI Service - Main entry point.

Wires together all components:
- ServiceConfig loading
- ProjectManager
- GpuWorkerPool (scheduler)
- TCP per-project servers
- FastAPI HTTP control API
- TrainManager
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import uvicorn

# Ensure plugins are registered at import time
import app.plugins.glyph_patchcore_plugin  # noqa: F401
import app.plugins.patchcore_plugin  # noqa: F401
import app.plugins.resnet_classify_plugin  # noqa: F401
from app.config import ServiceConfig, load_service_config
from app.http_api import create_api
from app.project_manager import ProjectManager
from app.result_schema import (
    ErrorCode,
    InferResult,
    TimingInfo,
    make_error_result,
)
from app.scheduler import GpuWorkerPool, Job
from app.tcp_server import TcpServerManager
from app.train_manager import TrainManager
from app.utils.file_utils import wait_for_file_ready

logger = logging.getLogger("visionhub")


class ServiceApp:
    """
    Main application class that owns and coordinates all components.

    This is passed to the FastAPI factory so HTTP handlers can access
    the project manager, scheduler, etc.
    """

    def __init__(self, service_config: ServiceConfig):
        self.service_config = service_config
        self.start_time = time.monotonic()

        # Core components
        self.project_manager = ProjectManager(service_config)
        self.gpu_pool = GpuWorkerPool(service_config, infer_callback=self._execute_infer_job)
        self.tcp_manager = TcpServerManager()
        self.train_manager = TrainManager()

        # FastAPI app
        self.api = create_api(self)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def startup(self) -> None:
        """Initialize and start all services."""
        logger.info("Starting VisionHub AI Service...")
        logger.info("DATA_ROOT: %s", self.service_config.data_root)

        # Ensure directories
        data_root = Path(self.service_config.data_root)
        (data_root / "service" / "logs").mkdir(parents=True, exist_ok=True)
        (data_root / "projects").mkdir(parents=True, exist_ok=True)

        # Scan projects
        self.project_manager.scan_projects()

        # Start GPU workers
        await self.gpu_pool.start()

        # Setup each enabled project
        await self.setup_projects()

        logger.info(
            "VisionHub service ready. Projects: %d, GPU workers: %d, HTTP: %s:%d",
            len(self.project_manager.projects),
            self.service_config.gpu.workers,
            self.service_config.http.host,
            self.service_config.http.port,
        )

    async def shutdown(self) -> None:
        """Gracefully shut down all services."""
        logger.info("Shutting down VisionHub AI Service...")
        await self.tcp_manager.stop_all()
        await self.gpu_pool.stop()
        # Unload all models
        for pid in list(self.project_manager.projects.keys()):
            self.project_manager.unload_model(pid)
        logger.info("VisionHub service stopped.")

    async def setup_projects(self) -> None:
        """Setup TCP servers and queues for all enabled projects."""
        for pid, state in self.project_manager.projects.items():
            if state.enabled:
                await self.start_project(pid)

    async def start_project(self, project_id: str) -> None:
        """Start TCP server and register queue for a project."""
        state = self.project_manager.get_project(project_id)
        if state is None:
            return

        # Register queue
        await self.gpu_pool.register_project(project_id)

        # Try to load model (non-fatal if fails)
        if not state.is_model_loaded:
            self.project_manager.load_model(project_id)

        # Start TCP server
        try:
            await self.tcp_manager.start_server(
                project_id=project_id,
                host=state.config.io.bind_host,
                port=state.config.io.tcp_port,
                on_infer=self.run_inference,
                on_status=self._get_project_status,
                on_set_model=self._set_active_model,
                log_buffer=state.log_buffer,
            )
            state.tcp_running = True
        except Exception:
            logger.exception("Failed to start TCP server for project: %s", project_id)
            state.tcp_running = False

    async def stop_project(self, project_id: str) -> None:
        """Stop TCP server for a project."""
        state = self.project_manager.get_project(project_id)
        if state is not None:
            state.tcp_running = False
        await self.tcp_manager.stop_server(project_id)
        await self.gpu_pool.unregister_project(project_id)

    # ------------------------------------------------------------------
    # Inference pipeline
    # ------------------------------------------------------------------

    async def run_inference(
        self,
        project_id: str,
        job_id: str,
        image_path: str,
        options: dict[str, Any] | None = None,
    ) -> InferResult:
        """
        Submit an inference job and wait for the result.

        This is called by both TCP and HTTP handlers.
        """
        state = self.project_manager.get_project(project_id)
        if state is None:
            return make_error_result(job_id, project_id, ErrorCode.PROJECT_NOT_FOUND, "Project not found")
        if not state.enabled:
            state.log_buffer.append("ERROR", "SERVICE", f"Inference rejected: project disabled (job={job_id})")
            return make_error_result(job_id, project_id, ErrorCode.PROJECT_DISABLED, "Project is disabled")
        if not state.is_model_loaded:
            state.log_buffer.append("ERROR", "SERVICE", f"Inference rejected: model not loaded (job={job_id})")
            return make_error_result(job_id, project_id, ErrorCode.MODEL_NOT_LOADED, "Model not loaded")

        state.log_buffer.append("INFO", "SERVICE", f"Inference submitted: job={job_id} image={image_path}")

        job = Job(
            job_id=job_id,
            project_id=project_id,
            job_type="infer",
            payload={
                "image_path": image_path,
                "options": options or {},
            },
        )

        try:
            future = await self.gpu_pool.submit_job(job)
            result_dict = await future
            result = InferResult(**result_dict)
        except RuntimeError as e:
            if "BUSY_QUEUE_FULL" in str(e):
                result = make_error_result(job_id, project_id, ErrorCode.BUSY_QUEUE_FULL, "Queue is full")
            else:
                result = make_error_result(job_id, project_id, ErrorCode.INFER_FAILED, str(e))
        except TimeoutError:
            result = make_error_result(job_id, project_id, ErrorCode.TIMEOUT, "Inference timed out")
        except Exception as e:
            result = make_error_result(job_id, project_id, ErrorCode.INFER_FAILED, str(e))

        # Log result
        state.jobs_logger.log(result)

        # Update stats
        state.stats.record(result.pred, result.timing_ms.total)

        return result

    async def _execute_infer_job(self, job: Job) -> dict[str, Any]:
        """
        Execute a single inference job on GPU.

        Called by the GpuWorkerPool worker.
        """
        project_id = job.project_id
        image_path = job.payload["image_path"]
        options = job.payload.get("options", {})

        state = self.project_manager.get_project(project_id)
        if state is None:
            raise RuntimeError(f"Project not found: {project_id}")
        if state.plugin is None:
            raise RuntimeError(f"Model not loaded for project: {project_id}")

        t_total_start = time.perf_counter()
        timing = TimingInfo()

        # 1. Wait for file to be ready
        t0 = time.perf_counter()
        file_cfg = state.config.image.file_ready
        ready = await wait_for_file_ready(
            image_path,
            mode=file_cfg.mode,
            stable_checks=file_cfg.stable_checks,
            stable_interval_ms=file_cfg.stable_interval_ms,
            timeout_ms=file_cfg.timeout_ms,
        )
        timing.wait_file = round((time.perf_counter() - t0) * 1000, 2)

        if not ready:
            if not Path(image_path).exists():
                raise RuntimeError(f"FILE_NOT_FOUND: {image_path}")
            raise RuntimeError(f"FILE_TIMEOUT: image not ready in {file_cfg.timeout_ms}ms")

        # 2. Run plugin inference (includes read, infer, post, save)
        config_dict = state.config.pipeline.model_dump()
        config_dict["_options"] = options
        config_dict["_output_dir"] = state.config.io.output_dir or str(Path(state.project_dir) / "runs")
        config_dict["_job_id"] = job.job_id
        config_dict["_overlay_output_path"] = state.config.io.overlay_output_path or ""

        plugin_result = state.plugin.infer(image_path, config_dict)

        # Merge timing
        plugin_timing = plugin_result.get("timing_ms", {})
        timing.infer = plugin_timing.get("infer", 0.0)
        timing.post = plugin_timing.get("post", 0.0)
        timing.save = plugin_timing.get("save", 0.0)
        timing.read = plugin_timing.get("read", 0.0)
        timing.total = round((time.perf_counter() - t_total_start) * 1000, 2)

        return {
            "job_id": job.job_id,
            "project_id": project_id,
            "ok": plugin_result.get("pred", "OK") == "OK",
            "pred": plugin_result.get("pred", "OK"),
            "score": plugin_result.get("score", 0.0),
            "threshold": plugin_result.get("threshold", 0.0),
            "timing_ms": timing.model_dump(),
            "artifacts": plugin_result.get("artifacts", {}),
            "regions": plugin_result.get("regions", []),
            "model_version": plugin_result.get("model_version", ""),
            "error": None,
        }

    async def _get_project_status(self, project_id: str) -> dict[str, Any]:
        """Get project status for TCP STATUS command."""
        state = self.project_manager.get_project(project_id)
        if state is None:
            return {"error": "Project not found"}

        queue = self.gpu_pool.get_queue(project_id)
        queue_info = {"queue_size": queue.size, "max_size": queue.max_size} if queue else {}

        return {
            "project_id": project_id,
            "enabled": state.enabled,
            "model_loaded": state.is_model_loaded,
            "model_version": state.model_version,
            "queue": queue_info,
            "stats": state.stats.to_dict(),
        }

    async def _set_active_model(self, project_id: str, version: str) -> bool:
        """Handle TCP SET_ACTIVE_MODEL command."""
        return self.project_manager.switch_model(project_id, version)


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(config: ServiceConfig) -> None:
    """Configure logging for the service."""
    log_level = getattr(logging, config.logging.level.upper(), logging.INFO)
    log_dir = config.logging.dir or str(Path(config.data_root) / "service" / "logs")
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level)

    # File handler
    log_file = Path(log_dir) / "service.log"
    file_handler = logging.FileHandler(str(log_file), encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(log_level)

    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    # Suppress noisy loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

_app_instance: ServiceApp | None = None


def get_app() -> ServiceApp:
    """Get or create the global ServiceApp instance."""
    global _app_instance
    if _app_instance is None:
        config_path = os.environ.get(
            "VISIONHUB_CONFIG",
            str(Path(os.environ.get("VISIONHUB_DATA_ROOT", "E:\\AIInspect")) / "service" / "service_config.yaml"),
        )
        config = load_service_config(config_path)
        setup_logging(config)
        _app_instance = ServiceApp(config)
    return _app_instance


# FastAPI app instance (for uvicorn)
app = get_app().api


@app.on_event("startup")
async def on_startup() -> None:
    service_app = get_app()
    await service_app.startup()


@app.on_event("shutdown")
async def on_shutdown() -> None:
    service_app = get_app()
    await service_app.shutdown()


def main() -> None:
    """CLI entry point."""
    service_app = get_app()
    config = service_app.service_config

    uvicorn.run(
        "app.main:app",
        host=config.http.host,
        port=config.http.port,
        log_level="info",
        reload=False,
    )


if __name__ == "__main__":
    main()

"""Training job manager for VisionHub service.

Manages background training jobs:
- Start training in a background thread/task
- Track progress, logs, and status
- Update active_model.json on completion
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from app.plugins.registry import PluginRegistry

logger = logging.getLogger(__name__)


class TrainStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TrainJob:
    """Represents a training job."""

    train_job_id: str
    project_id: str
    status: TrainStatus = TrainStatus.PENDING
    progress: float = 0.0
    message: str = ""
    started_at: str = ""
    completed_at: str = ""
    model_version: str = ""
    out_model_dir: str = ""
    log_lines: list[str] = field(default_factory=list)
    result: dict[str, Any] | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "train_job_id": self.train_job_id,
            "project_id": self.project_id,
            "status": self.status.value,
            "progress": self.progress,
            "message": self.message,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "model_version": self.model_version,
            "out_model_dir": self.out_model_dir,
            "log_lines": self.log_lines[-100:],  # Last 100 lines
            "result": self.result,
            "error": self.error,
        }


class TrainManager:
    """Manages training jobs across all projects."""

    def __init__(self) -> None:
        self._jobs: dict[str, TrainJob] = {}
        self._active_by_project: dict[str, str] = {}  # project_id -> train_job_id
        self._lock = threading.Lock()

    def get_job(self, train_job_id: str) -> TrainJob | None:
        return self._jobs.get(train_job_id)

    def get_active_job(self, project_id: str) -> TrainJob | None:
        job_id = self._active_by_project.get(project_id)
        if job_id:
            return self._jobs.get(job_id)
        return None

    def start_training(
        self,
        project_id: str,
        dataset_dir: str,
        project_dir: str,
        algo_name: str,
        pipeline_config: dict[str, Any],
        auto_activate: bool = True,
    ) -> TrainJob:
        """
        Start a training job in a background thread.

        Returns the TrainJob immediately (status=PENDING).
        """
        # Check if project already has active training
        active = self.get_active_job(project_id)
        if active and active.status in (TrainStatus.PENDING, TrainStatus.RUNNING):
            raise RuntimeError(f"Project {project_id} already has active training: {active.train_job_id}")

        version = time.strftime("%Y%m%d_%H%M%S")
        train_job_id = f"train_{project_id}_{version}"
        out_model_dir = str(Path(project_dir) / "models" / version)

        job = TrainJob(
            train_job_id=train_job_id,
            project_id=project_id,
            model_version=version,
            out_model_dir=out_model_dir,
        )

        with self._lock:
            self._jobs[train_job_id] = job
            self._active_by_project[project_id] = train_job_id

        # Start background thread
        thread = threading.Thread(
            target=self._run_training,
            args=(job, dataset_dir, out_model_dir, algo_name, pipeline_config, project_dir, auto_activate),
            daemon=True,
            name=f"train-{project_id}",
        )
        thread.start()

        logger.info("Started training job: %s (project=%s)", train_job_id, project_id)
        return job

    def _run_training(
        self,
        job: TrainJob,
        dataset_dir: str,
        out_model_dir: str,
        algo_name: str,
        pipeline_config: dict[str, Any],
        project_dir: str,
        auto_activate: bool,
    ) -> None:
        """Background thread that runs the actual training."""
        job.status = TrainStatus.RUNNING
        job.started_at = time.strftime("%Y-%m-%dT%H:%M:%S")
        job.message = "Training started"
        self._log(job, f"Training started for project {job.project_id}")
        self._log(job, f"Algorithm: {algo_name}")
        self._log(job, f"Dataset: {dataset_dir}")
        self._log(job, f"Output: {out_model_dir}")

        try:
            plugin = PluginRegistry.create_instance(algo_name)
            if plugin is None:
                raise RuntimeError(f"Unknown algorithm: {algo_name}")

            def progress_cb(pct: float, msg: str) -> None:
                job.progress = pct
                job.message = msg
                self._log(job, f"[{pct:.1f}%] {msg}")

            result = plugin.train(
                dataset_dir=dataset_dir,
                out_model_dir=out_model_dir,
                config=pipeline_config,
                progress_cb=progress_cb,
            )

            job.result = result
            job.status = TrainStatus.COMPLETED
            job.completed_at = time.strftime("%Y-%m-%dT%H:%M:%S")
            job.progress = 100.0
            job.message = "Training completed successfully"
            self._log(job, "Training completed successfully")

            # Write train log to file
            self._save_log(job, out_model_dir)

            # Auto-activate new model
            if auto_activate:
                active_path = Path(project_dir) / "active_model.json"
                with open(active_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {"version": job.model_version, "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S")},
                        f,
                        indent=2,
                    )
                self._log(job, f"Auto-activated model version: {job.model_version}")

        except Exception as e:
            job.status = TrainStatus.FAILED
            job.error = str(e)
            job.completed_at = time.strftime("%Y-%m-%dT%H:%M:%S")
            job.message = f"Training failed: {e}"
            self._log(job, f"Training FAILED: {e}")
            logger.exception("Training job %s failed", job.train_job_id)
            # Still save log on failure
            try:
                self._save_log(job, out_model_dir)
            except Exception:
                pass

    def _log(self, job: TrainJob, message: str) -> None:
        """Append a log line to the job."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {message}"
        job.log_lines.append(line)

    def _save_log(self, job: TrainJob, out_model_dir: str) -> None:
        """Save training log to file."""
        log_path = Path(out_model_dir) / "train.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("\n".join(job.log_lines))

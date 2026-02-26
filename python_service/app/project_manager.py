"""Project manager for VisionHub service.

Responsible for:
- Scanning and loading all project configurations from DATA_ROOT/projects/
- Managing project lifecycle (enable/disable, reload)
- Model loading/unloading with LRU cache
- Providing project state to other components
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

from app.config import ProjectConfig, ServiceConfig, load_project_config
from app.plugins.base import AlgoPluginBase
from app.plugins.registry import PluginRegistry
from app.result_schema import JobsLogger

logger = logging.getLogger(__name__)


class ProjectState:
    """Runtime state for a single project."""

    def __init__(self, config: ProjectConfig, project_dir: str):
        self.config: ProjectConfig = config
        self.project_dir: str = project_dir
        self.plugin: AlgoPluginBase | None = None
        self.model_version: str = ""
        self.model_dir: str = ""
        self.jobs_logger: JobsLogger = JobsLogger(config.io.output_dir or str(Path(project_dir) / "runs"))
        self.enabled: bool = config.enabled
        self.tcp_running: bool = False
        self.stats: ProjectStats = ProjectStats()
        self._lock = threading.Lock()

    @property
    def project_id(self) -> str:
        return self.config.project_id

    @property
    def is_model_loaded(self) -> bool:
        return self.plugin is not None and self.plugin.is_loaded

    def get_active_model_dir(self) -> str | None:
        """Read active_model.json and return the model directory path."""
        active_path = Path(self.project_dir) / "active_model.json"
        if not active_path.exists():
            # Try to find latest model in models/
            models_dir = Path(self.project_dir) / "models"
            if models_dir.exists():
                versions = sorted(models_dir.iterdir(), reverse=True)
                for v in versions:
                    if v.is_dir() and (v / "meta.json").exists():
                        return str(v)
            return None

        with open(active_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        version = data.get("version", "")
        if not version:
            return None
        model_dir = Path(self.project_dir) / "models" / version
        if model_dir.exists():
            return str(model_dir)
        return None

    def set_active_model(self, version: str) -> bool:
        """Update active_model.json to point to a new version."""
        model_dir = Path(self.project_dir) / "models" / version
        if not model_dir.exists():
            return False
        active_path = Path(self.project_dir) / "active_model.json"
        with open(active_path, "w", encoding="utf-8") as f:
            json.dump({"version": version, "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S")}, f, indent=2)
        return True


class ProjectStats:
    """Runtime statistics for a project."""

    def __init__(self) -> None:
        self.total_jobs: int = 0
        self.ok_count: int = 0
        self.ng_count: int = 0
        self.error_count: int = 0
        self.avg_infer_ms: float = 0.0
        self.last_result_time: str = ""
        self._lock = threading.Lock()

    def record(self, pred: str, infer_ms: float) -> None:
        with self._lock:
            self.total_jobs += 1
            if pred == "OK":
                self.ok_count += 1
            elif pred == "NG":
                self.ng_count += 1
            else:
                self.error_count += 1
            # Running average
            if self.total_jobs == 1:
                self.avg_infer_ms = infer_ms
            else:
                self.avg_infer_ms = self.avg_infer_ms * 0.9 + infer_ms * 0.1
            self.last_result_time = time.strftime("%Y-%m-%dT%H:%M:%S")

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "total_jobs": self.total_jobs,
                "ok_count": self.ok_count,
                "ng_count": self.ng_count,
                "error_count": self.error_count,
                "avg_infer_ms": round(self.avg_infer_ms, 2),
                "last_result_time": self.last_result_time,
            }


class ModelCache:
    """LRU cache for loaded models across projects."""

    def __init__(self, max_size: int = 5):
        self._max_size = max_size
        self._cache: OrderedDict[str, ProjectState] = OrderedDict()
        self._lock = threading.Lock()

    def touch(self, project_id: str, state: ProjectState) -> None:
        """Mark a project as recently used."""
        with self._lock:
            if project_id in self._cache:
                self._cache.move_to_end(project_id)
            else:
                self._cache[project_id] = state
            self._evict_if_needed()

    def remove(self, project_id: str) -> None:
        """Remove a project from cache."""
        with self._lock:
            self._cache.pop(project_id, None)

    def _evict_if_needed(self) -> None:
        """Evict least recently used models if over capacity."""
        while len(self._cache) > self._max_size:
            evicted_id, evicted_state = self._cache.popitem(last=False)
            if evicted_state.plugin is not None:
                try:
                    evicted_state.plugin.unload()
                    evicted_state.plugin = None
                    logger.info("LRU evicted model for project: %s", evicted_id)
                except Exception:
                    logger.exception("Error unloading model for project: %s", evicted_id)


class ProjectManager:
    """Manages all projects: loading, lifecycle, model cache."""

    def __init__(self, service_config: ServiceConfig):
        self._service_config = service_config
        self._data_root = service_config.data_root
        self._projects: dict[str, ProjectState] = {}
        self._model_cache = ModelCache(max_size=service_config.gpu.max_loaded_models)
        self._lock = threading.Lock()

    @property
    def data_root(self) -> str:
        return self._data_root

    @property
    def projects(self) -> dict[str, ProjectState]:
        return self._projects

    def scan_projects(self) -> list[str]:
        """Scan DATA_ROOT/projects/ and load all project configs."""
        projects_dir = Path(self._data_root) / "projects"
        if not projects_dir.exists():
            logger.warning("Projects directory does not exist: %s", projects_dir)
            return []

        loaded: list[str] = []
        for project_dir in sorted(projects_dir.iterdir()):
            if not project_dir.is_dir():
                continue
            config_path = project_dir / "project.yaml"
            if not config_path.exists():
                continue
            try:
                config = load_project_config(config_path, self._data_root)
                state = ProjectState(config, str(project_dir))
                with self._lock:
                    self._projects[config.project_id] = state
                loaded.append(config.project_id)
                logger.info("Loaded project: %s (%s)", config.project_id, config.display_name)
            except Exception:
                logger.exception("Failed to load project from: %s", project_dir)

        logger.info("Scanned %d projects: %s", len(loaded), loaded)
        return loaded

    def reload_project(self, project_id: str) -> bool:
        """Reload a single project's configuration from disk."""
        state = self._projects.get(project_id)
        if state is None:
            return False
        config_path = Path(state.project_dir) / "project.yaml"
        if not config_path.exists():
            return False
        try:
            new_config = load_project_config(config_path, self._data_root)
            state.config = new_config
            state.enabled = new_config.enabled
            # Update jobs logger output dir
            output_dir = new_config.io.output_dir or str(Path(state.project_dir) / "runs")
            state.jobs_logger = JobsLogger(output_dir)
            logger.info("Reloaded project config: %s", project_id)
            return True
        except Exception:
            logger.exception("Failed to reload project: %s", project_id)
            return False

    def reload_all(self) -> list[str]:
        """Reload all existing projects and scan for new ones."""
        # Reload existing
        for pid in list(self._projects.keys()):
            self.reload_project(pid)
        # Scan for new
        return self.scan_projects()

    def get_project(self, project_id: str) -> ProjectState | None:
        return self._projects.get(project_id)

    def list_projects(self) -> list[dict[str, Any]]:
        """Return summary info for all projects."""
        result = []
        for pid, state in self._projects.items():
            result.append({
                "project_id": pid,
                "display_name": state.config.display_name,
                "enabled": state.enabled,
                "tcp_port": state.config.io.tcp_port,
                "algo": state.config.pipeline.algo,
                "model_loaded": state.is_model_loaded,
                "model_version": state.model_version,
                "tcp_running": state.tcp_running,
                "stats": state.stats.to_dict(),
            })
        return result

    def enable_project(self, project_id: str, enabled: bool) -> bool:
        """Enable or disable a project."""
        state = self._projects.get(project_id)
        if state is None:
            return False
        state.enabled = enabled
        logger.info("Project %s %s", project_id, "enabled" if enabled else "disabled")
        return True

    def load_model(self, project_id: str) -> bool:
        """Load the active model for a project."""
        state = self._projects.get(project_id)
        if state is None:
            return False

        model_dir = state.get_active_model_dir()
        if model_dir is None:
            logger.warning("No active model found for project: %s", project_id)
            return False

        algo_name = state.config.pipeline.algo
        plugin = PluginRegistry.create_instance(algo_name)
        if plugin is None:
            logger.error("Unknown algorithm plugin: %s", algo_name)
            return False

        try:
            device = state.config.pipeline.infer.device
            config_dict = state.config.pipeline.model_dump()
            plugin.load(model_dir, device, config_dict)

            # Hot swap: only replace after successful load
            old_plugin = state.plugin
            state.plugin = plugin
            state.model_version = plugin._model_version if hasattr(plugin, "_model_version") else ""
            state.model_dir = model_dir

            # Update LRU cache
            self._model_cache.touch(project_id, state)

            # Unload old plugin
            if old_plugin is not None:
                try:
                    old_plugin.unload()
                except Exception:
                    logger.exception("Error unloading old model for project: %s", project_id)

            logger.info("Model loaded for project %s: %s", project_id, model_dir)
            return True
        except Exception:
            logger.exception("Failed to load model for project: %s", project_id)
            return False

    def unload_model(self, project_id: str) -> bool:
        """Unload the model for a project."""
        state = self._projects.get(project_id)
        if state is None:
            return False
        if state.plugin is not None:
            try:
                state.plugin.unload()
            except Exception:
                logger.exception("Error unloading model for project: %s", project_id)
            state.plugin = None
            state.model_version = ""
            self._model_cache.remove(project_id)
        return True

    def switch_model(self, project_id: str, version: str) -> bool:
        """Switch to a different model version for a project."""
        state = self._projects.get(project_id)
        if state is None:
            return False
        if not state.set_active_model(version):
            return False
        return self.load_model(project_id)

"""Configuration models for VisionHub service."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Service-level config
# ---------------------------------------------------------------------------

class GpuConfig(BaseModel):
    device: str = "cuda"
    workers: int = 1
    max_loaded_models: int = 5


class SchedulerConfig(BaseModel):
    max_queue_per_project: int = 20
    job_timeout_ms: int = 10000
    poll_interval_ms: int = 10


class HttpConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8100


class LoggingConfig(BaseModel):
    level: str = "INFO"
    dir: str = ""
    rotate: str = "daily"
    max_days: int = 30


class ServiceConfig(BaseModel):
    data_root: str = "E:\\AIInspect"
    gpu: GpuConfig = Field(default_factory=GpuConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    http: HttpConfig = Field(default_factory=HttpConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


# ---------------------------------------------------------------------------
# Project-level config
# ---------------------------------------------------------------------------

class FileReadyConfig(BaseModel):
    mode: str = "size_stable"
    stable_checks: int = 3
    stable_interval_ms: int = 50
    timeout_ms: int = 1500


class RoiConfig(BaseModel):
    enabled: bool = False
    x: int = 0
    y: int = 0
    w: int = 0
    h: int = 0


class ImageConfig(BaseModel):
    file_ready: FileReadyConfig = Field(default_factory=FileReadyConfig)
    color: str = "rgb"
    rotate: int = 0
    roi: RoiConfig = Field(default_factory=RoiConfig)


class IoConfig(BaseModel):
    tcp_port: int = 9100
    bind_host: str = "0.0.0.0"
    allow_ips: list[str] = Field(default_factory=lambda: ["127.0.0.1/32"])
    output_dir: str = ""


class TileConfig(BaseModel):
    tile_w: int = 512
    tile_h: int = 352
    stride_w: int = 384
    stride_h: int = 352


class InferConfig(BaseModel):
    device: str = "cuda"
    tile: TileConfig = Field(default_factory=TileConfig)
    batch_tiles: int = 8


class DecisionConfig(BaseModel):
    score_method: str = "q999"
    threshold: str = "from_model"


class RegionsExportConfig(BaseModel):
    enabled: bool = True
    min_area_px: int = 80
    max_regions: int = 10
    output_format: str = "json"


class ExportConfig(BaseModel):
    save_heatmap_png: bool = True
    heatmap_vmin: float = 0.0
    heatmap_vmax_mode: str = "thr_scale"
    heatmap_vmax: float = 1.0
    heatmap_vmax_scale: float = 1.2
    save_u16: bool = True
    save_mask: bool = True
    mask_thr_scale: float = 0.85
    dilate_px: int = 10
    close_px: int = 6
    regions: RegionsExportConfig = Field(default_factory=RegionsExportConfig)


class PostprocessConfig(BaseModel):
    decision: DecisionConfig = Field(default_factory=DecisionConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)


class PreprocessConfig(BaseModel):
    normalize: str = "imagenet"
    align: dict[str, Any] = Field(default_factory=lambda: {"enabled": False})


class ModelRefConfig(BaseModel):
    active: str = ""


class PipelineConfig(BaseModel):
    algo: str = "patchcore_tiling_v1"
    model: ModelRefConfig = Field(default_factory=ModelRefConfig)
    preprocess: PreprocessConfig = Field(default_factory=PreprocessConfig)
    infer: InferConfig = Field(default_factory=InferConfig)
    postprocess: PostprocessConfig = Field(default_factory=PostprocessConfig)


class ProjectConfig(BaseModel):
    project_id: str
    display_name: str = ""
    enabled: bool = True
    io: IoConfig = Field(default_factory=IoConfig)
    image: ImageConfig = Field(default_factory=ImageConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _resolve_vars(data: dict[str, Any], data_root: str, project_dir: str = "") -> dict[str, Any]:
    """Recursively resolve ${DATA_ROOT} and ${PROJECT_DIR} in string values."""
    resolved: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, str):
            value = value.replace("${DATA_ROOT}", data_root)
            value = value.replace("${PROJECT_DIR}", project_dir)
        elif isinstance(value, dict):
            value = _resolve_vars(value, data_root, project_dir)
        elif isinstance(value, list):
            value = [
                item.replace("${DATA_ROOT}", data_root).replace("${PROJECT_DIR}", project_dir)
                if isinstance(item, str)
                else item
                for item in value
            ]
        resolved[key] = value
    return resolved


def load_service_config(path: str | Path) -> ServiceConfig:
    """Load service_config.yaml and return a ServiceConfig model."""
    path = Path(path)
    if not path.exists():
        return ServiceConfig()
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    data_root = raw.get("data_root", "E:\\AIInspect")
    raw = _resolve_vars(raw, data_root)
    return ServiceConfig(**raw)


def load_project_config(path: str | Path, data_root: str) -> ProjectConfig:
    """Load a single project.yaml and return a ProjectConfig model."""
    path = Path(path)
    project_dir = str(path.parent)
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    raw = _resolve_vars(raw, data_root, project_dir)
    return ProjectConfig(**raw)


def get_data_root(service_config: ServiceConfig | None = None) -> str:
    """Return the resolved DATA_ROOT path."""
    if service_config:
        return service_config.data_root
    return os.environ.get("VISIONHUB_DATA_ROOT", "E:\\AIInspect")

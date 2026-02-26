"""Unified result data structures and error codes for VisionHub."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Error codes
# ---------------------------------------------------------------------------

class ErrorCode:
    PROJECT_DISABLED = "PROJECT_DISABLED"
    PROJECT_NOT_FOUND = "PROJECT_NOT_FOUND"
    FILE_NOT_FOUND = "FILE_NOT_FOUND"
    FILE_TIMEOUT = "FILE_TIMEOUT"
    BUSY_QUEUE_FULL = "BUSY_QUEUE_FULL"
    MODEL_NOT_LOADED = "MODEL_NOT_LOADED"
    INFER_FAILED = "INFER_FAILED"
    TRAIN_FAILED = "TRAIN_FAILED"
    TIMEOUT = "TIMEOUT"
    INVALID_CMD = "INVALID_CMD"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class ErrorInfo(BaseModel):
    code: str
    message: str


# ---------------------------------------------------------------------------
# Region
# ---------------------------------------------------------------------------

class Region(BaseModel):
    x: int = 0
    y: int = 0
    w: int = 0
    h: int = 0
    score: float = 0.0
    area_px: int = 0


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

class TimingInfo(BaseModel):
    wait_file: float = 0.0
    read: float = 0.0
    infer: float = 0.0
    post: float = 0.0
    save: float = 0.0
    total: float = 0.0


# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------

class Artifacts(BaseModel):
    u16: str = ""
    mask: str = ""
    heatmap: str = ""
    overlay: str = ""


# ---------------------------------------------------------------------------
# InferResult - the unified result schema
# ---------------------------------------------------------------------------

class InferResult(BaseModel):
    job_id: str
    project_id: str
    ok: bool = True
    pred: str = "OK"
    score: float = 0.0
    threshold: float = 0.0
    timing_ms: TimingInfo = Field(default_factory=TimingInfo)
    artifacts: Artifacts = Field(default_factory=Artifacts)
    regions: list[Region] = Field(default_factory=list)
    model_version: str = ""
    error: ErrorInfo | None = None

    def to_json_line(self) -> str:
        """Serialize to a single JSON line (for jobs.jsonl and TCP response)."""
        return self.model_dump_json()

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return self.model_dump()


# ---------------------------------------------------------------------------
# Error result helper
# ---------------------------------------------------------------------------

def make_error_result(
    job_id: str,
    project_id: str,
    code: str,
    message: str,
) -> InferResult:
    """Create an error InferResult."""
    return InferResult(
        job_id=job_id,
        project_id=project_id,
        ok=False,
        pred="ERROR",
        error=ErrorInfo(code=code, message=message),
    )


# ---------------------------------------------------------------------------
# TCP response helpers
# ---------------------------------------------------------------------------

class TcpResponse(BaseModel):
    """Generic TCP response wrapper for non-infer commands."""
    ok: bool = True
    cmd: str = ""
    data: dict[str, Any] = Field(default_factory=dict)
    error: ErrorInfo | None = None


def make_tcp_error(cmd: str, code: str, message: str) -> TcpResponse:
    return TcpResponse(ok=False, cmd=cmd, error=ErrorInfo(code=code, message=message))


# ---------------------------------------------------------------------------
# Jobs logger
# ---------------------------------------------------------------------------

class JobsLogger:
    """Append inference results to a daily jobs.jsonl file."""

    def __init__(self, output_dir: str):
        self._output_dir = output_dir

    def log(self, result: InferResult) -> None:
        """Append one result to today's jobs.jsonl."""
        today = datetime.now().strftime("%Y-%m-%d")
        day_dir = Path(self._output_dir) / today
        day_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path = day_dir / "jobs.jsonl"
        line = result.to_json_line()
        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def get_artifacts_dir(self, date_str: str | None = None) -> Path:
        """Return the artifacts directory for a given date (default: today)."""
        if date_str is None:
            date_str = datetime.now().strftime("%Y-%m-%d")
        artifacts_dir = Path(self._output_dir) / date_str / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        return artifacts_dir

"""File utility functions for VisionHub service."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from pathlib import Path

logger = logging.getLogger(__name__)


async def wait_for_file_ready(
    file_path: str,
    mode: str = "size_stable",
    stable_checks: int = 3,
    stable_interval_ms: int = 50,
    timeout_ms: int = 1500,
) -> bool:
    """
    Wait until an image file is fully written and ready to read.

    Modes:
        - size_stable: Check file size is stable across N consecutive checks.
        - open_retry: Try to open the file exclusively, retry until success.

    Returns:
        True if file is ready, False if timeout.
    """
    deadline = time.monotonic() + timeout_ms / 1000.0
    interval_s = stable_interval_ms / 1000.0

    if mode == "size_stable":
        last_size = -1
        stable_count = 0
        while time.monotonic() < deadline:
            try:
                size = os.path.getsize(file_path)
            except OSError:
                await asyncio.sleep(interval_s)
                continue

            if size > 0 and size == last_size:
                stable_count += 1
                if stable_count >= stable_checks:
                    return True
            else:
                stable_count = 0
            last_size = size
            await asyncio.sleep(interval_s)

    elif mode == "open_retry":
        while time.monotonic() < deadline:
            try:
                with open(file_path, "rb") as f:
                    # Try reading a byte to verify
                    f.read(1)
                return True
            except (OSError, PermissionError):
                await asyncio.sleep(interval_s)

    return False


def ensure_dir(path: str | Path) -> Path:
    """Create directory if it doesn't exist and return the Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

"""Logging utilities for VisionHub service."""

from __future__ import annotations

import logging


def get_project_logger(project_id: str) -> logging.Logger:
    """Get a logger specific to a project."""
    return logging.getLogger(f"visionhub.project.{project_id}")


def get_train_logger(project_id: str) -> logging.Logger:
    """Get a logger specific to training for a project."""
    return logging.getLogger(f"visionhub.train.{project_id}")

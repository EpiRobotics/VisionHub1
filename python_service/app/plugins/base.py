"""Algorithm plugin base interface for VisionHub."""

from __future__ import annotations

import abc
from typing import Any, Callable


class AlgoPluginBase(abc.ABC):
    """
    All algorithm plugins must inherit from this base class.

    Each project gets its own plugin instance + loaded model.
    Plugins must be safe for concurrent read (infer), but train is exclusive.
    """

    name: str = "base"

    @abc.abstractmethod
    def load(self, model_dir: str, device: str, config: dict[str, Any]) -> None:
        """
        Load model weights and prepare for inference.

        Args:
            model_dir: Path to the model version directory containing model files.
            device: Device string, e.g. "cuda" or "cpu".
            config: The full pipeline config dict from project.yaml.
        """

    @abc.abstractmethod
    def unload(self) -> None:
        """Release model resources and GPU memory."""

    @abc.abstractmethod
    def infer(self, image_path: str, config: dict[str, Any]) -> dict[str, Any]:
        """
        Run inference on a single image.

        Args:
            image_path: Absolute path to the input image.
            config: The full pipeline config dict from project.yaml.

        Returns:
            A dict matching the InferResult schema fields:
            {
                "score": float,
                "threshold": float,
                "pred": "OK" | "NG",
                "regions": [...],
                "artifacts": {"u16": path, "mask": path, "heatmap": path, "overlay": path},
                "timing_ms": {"infer": ..., "post": ..., "save": ...},
                "model_version": str,
            }
        """

    @abc.abstractmethod
    def train(
        self,
        dataset_dir: str,
        out_model_dir: str,
        config: dict[str, Any],
        progress_cb: Callable[[float, str], None] | None = None,
    ) -> dict[str, Any]:
        """
        Train a model from dataset.

        Args:
            dataset_dir: Path to datasets/ directory (containing ok/, ng/, etc.).
            out_model_dir: Path where trained model files should be written.
            config: The full pipeline config dict from project.yaml.
            progress_cb: Optional callback(progress_pct, message) for progress reporting.

        Returns:
            A dict with training report info:
            {
                "model_version": str,
                "metrics": {...},
                "threshold": float,
                "duration_s": float,
            }
        """

    @property
    def is_loaded(self) -> bool:
        """Override to return True when model is ready for inference."""
        return False

"""Tennis trajectory analysis detection engine package."""

from .modes import DetectionMode, ProcessingMode, get_mode_metadata, get_processing_config
from .model_manager import ModelManager, ModelInfo
from .detection_engine import (
    DetectionEngine, 
    FrameProcessor, 
    TrackedObject,
    ExponentialMovingAverage,
    KalmanSmoother,
    main as cli_main
)

__all__ = [
    "DetectionMode",
    "ProcessingMode", 
    "get_mode_metadata",
    "get_processing_config",
    "ModelManager",
    "ModelInfo",
    "DetectionEngine",
    "FrameProcessor", 
    "TrackedObject",
    "ExponentialMovingAverage",
    "KalmanSmoother",
    "cli_main"
]

__version__ = "1.0.0"
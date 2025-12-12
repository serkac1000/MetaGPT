"""Detection modes and processing configuration for tennis trajectory analysis."""

from enum import Enum
from typing import Dict, List, Any
from dataclasses import dataclass


class DetectionMode(Enum):
    """Available detection modes for tennis trajectory analysis."""
    BALL = "ball"
    RACKET = "racket" 
    PLAYER = "player"


class ProcessingMode(Enum):
    """Processing modes for real-time vs batch processing."""
    REAL_TIME = "real-time"
    BATCH = "batch"


@dataclass
class ModeMetadata:
    """Metadata for each detection mode including class IDs and heuristics."""
    class_ids: List[int]
    confidence_threshold: float
    nms_threshold: float
    object_heights: Dict[str, tuple]  # Expected height ranges in pixels
    tracking_config: Dict[str, Any]
    smoothing_factor: float
    frame_skip: int


# Predefined metadata for each detection mode
DETECTION_MODE_CONFIGS: Dict[DetectionMode, ModeMetadata] = {
    DetectionMode.BALL: ModeMetadata(
        class_ids=[32],  # COCO class ID for sports ball (tennis ball)
        confidence_threshold=0.6,
        nms_threshold=0.4,
        object_heights={"min": 5, "max": 25},  # Expected ball size in pixels
        tracking_config={
            "max_age": 30,          # Maximum frames to keep track
            "min_hits": 3,          # Minimum hits to confirm track
            "iou_threshold": 0.3,   # IOU threshold for track assignment
            "inertia": 0.7          # Kalman filter inertia
        },
        smoothing_factor=0.8,  # Higher smoothing for trajectory stability
        frame_skip=1  # Process every frame for ball detection
    ),
    
    DetectionMode.RACKET: ModeMetadata(
        class_ids=[38, 39],  # Tennis racket (custom class) or sports equipment
        confidence_threshold=0.5,
        nms_threshold=0.5,
        object_heights={"min": 40, "max": 200},  # Racket size range
        tracking_config={
            "max_age": 15,
            "min_hits": 2,
            "iou_threshold": 0.4,
            "inertia": 0.8
        },
        smoothing_factor=0.6,
        frame_skip=2  # Can skip frames for racket detection
    ),
    
    DetectionMode.PLAYER: ModeMetadata(
        class_ids=[0],  # COCO person class
        confidence_threshold=0.7,
        nms_threshold=0.4,
        object_heights={"min": 100, "max": 400},  # Player height range
        tracking_config={
            "max_age": 10,
            "min_hits": 1,
            "iou_threshold": 0.5,
            "inertia": 0.9
        },
        smoothing_factor=0.7,
        frame_skip=1  # Process every frame for player tracking
    )
}


def get_mode_metadata(mode: DetectionMode) -> ModeMetadata:
    """Get metadata configuration for a specific detection mode."""
    return DETECTION_MODE_CONFIGS.get(mode)


def get_available_modes() -> List[DetectionMode]:
    """Get list of all available detection modes."""
    return list(DetectionMode)


# Processing mode configurations
PROCESSING_MODE_CONFIGS: Dict[ProcessingMode, Dict[str, Any]] = {
    ProcessingMode.REAL_TIME: {
        "image_size": 640,      # Smaller for speed
        "batch_size": 1,        # Process one frame at a time
        "max_detections": 50,   # Limit detections for speed
        "device": "auto",       # Use available device (GPU preferred)
        "stream_buffer": 8      # Internal frame buffer
    },
    
    ProcessingMode.BATCH: {
        "image_size": 1280,     # Larger for accuracy
        "batch_size": 4,        # Process multiple frames together
        "max_detections": 100,  # More detections allowed
        "device": "auto",       # Use available device
        "stream_buffer": 32     # Larger frame buffer
    }
}


def get_processing_config(mode: ProcessingMode) -> Dict[str, Any]:
    """Get processing configuration for a specific processing mode."""
    return PROCESSING_MODE_CONFIGS.get(mode)
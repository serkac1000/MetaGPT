"""Core detection engine for tennis trajectory analysis using YOLOv8n."""

import time
import logging
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
from collections import deque

# Optional dependencies
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    np = None
    NUMPY_AVAILABLE = False

from .modes import DetectionMode, ProcessingMode, get_mode_metadata, get_processing_config
from .model_manager import ModelManager


@dataclass
class TrackedObject:
    """Structured result for a tracked object."""
    track_id: str
    class_id: int
    class_name: str
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    centroid: Tuple[int, int]
    confidence: float
    frame_number: int
    timestamp: float
    smoothed_centroid: Optional[Tuple[int, int]] = None
    velocity: Optional[Tuple[float, float]] = None  # dx/dt, dy/dt
    
    @property
    def center_x(self) -> int:
        return self.centroid[0]
    
    @property 
    def center_y(self) -> int:
        return self.centroid[1]
    
    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]
    
    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]
    
    @property
    def area(self) -> int:
        return self.width * self.height


class ExponentialMovingAverage:
    """Simple EMA smoother for trajectory stabilization."""
    
    def __init__(self, alpha: float = 0.8):
        self.alpha = alpha
        self._prev_value: Optional[Tuple[float, float]] = None
    
    def update(self, value: Tuple[float, float]) -> Tuple[float, float]:
        """Update EMA with new value."""
        if self._prev_value is None:
            self._prev_value = value
            return value
        
        smoothed_x = self.alpha * value[0] + (1 - self.alpha) * self._prev_value[0]
        smoothed_y = self.alpha * value[1] + (1 - self.alpha) * self._prev_value[1]
        
        smoothed_value = (smoothed_x, smoothed_y)
        self._prev_value = smoothed_value
        
        return smoothed_value
    
    def reset(self):
        """Reset the EMA state."""
        self._prev_value = None


class KalmanSmoother:
    """Simple Kalman filter for trajectory smoothing."""
    
    def __init__(self, process_noise: float = 0.1, measurement_noise: float = 1.0):
        if not NUMPY_AVAILABLE:
            raise ImportError("numpy is required for KalmanSmoother")
            
        # State: [x, y, dx, dy]
        self.state = np.zeros(4)
        self.covariance = np.eye(4) * 100
        
        # Noise parameters
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        
        # Transition matrix
        self.F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # Measurement matrix
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        # Noise matrices
        self.Q = np.eye(4) * process_noise
        self.R = np.eye(2) * measurement_noise
    
    def update(self, measurement: Tuple[float, float]) -> Tuple[float, float]:
        """Update Kalman filter with new measurement."""
        x = np.array([[measurement[0]], [measurement[1]]])
        
        # Prediction
        self.state = self.F @ self.state
        self.covariance = self.F @ self.covariance @ self.F.T + self.Q
        
        # Update
        y = x - self.H @ self.state
        S = self.H @ self.covariance @ self.H.T + self.R
        K = self.covariance @ self.H.T @ np.linalg.inv(S)
        
        self.state = self.state + K @ y
        self.covariance = (np.eye(4) - K @ self.H) @ self.covariance
        
        return (float(self.state[0]), float(self.state[1]))


class FrameProcessor:
    """Processes individual frames for object detection and tracking."""
    
    def __init__(self, model_manager: ModelManager):
        """
        Initialize the frame processor.
        
        Args:
            model_manager: ModelManager instance for accessing YOLO models
        """
        self.model_manager = model_manager
        self.logger = logging.getLogger(__name__)
        
        # Smoothing instances per tracking ID
        self._smoothers: Dict[str, Any] = {}
        
        # Track history for velocity calculation
        self._track_history: Dict[str, deque] = {}
        self._max_history = 5
        
        # Current frame number
        self._frame_count = 0
    
    def _get_smoother(self, track_id: str, smoothing_method: str = "ema") -> Any:
        """Get or create smoother for track."""
        if track_id not in self._smoothers:
            if smoothing_method == "kalman":
                if not NUMPY_AVAILABLE:
                    self.logger.warning("numpy not available, falling back to EMA smoothing")
                    smoothing_method = "ema"
                else:
                    self._smoothers[track_id] = KalmanSmoother()
                    
            if smoothing_method == "ema":
                self._smoothers[track_id] = ExponentialMovingAverage(alpha=0.8)
        
        return self._smoothers[track_id]
    
    def _get_track_history(self, track_id: str) -> deque:
        """Get or create track history for velocity calculation."""
        if track_id not in self._track_history:
            self._track_history[track_id] = deque(maxlen=self._max_history)
        return self._track_history[track_id]
    
    def process_frame(
        self,
        frame: Any,
        mode: DetectionMode,
        processing_mode: ProcessingMode = ProcessingMode.REAL_TIME,
        frame_skip: Optional[int] = None
    ) -> List[TrackedObject]:
        """
        Process a frame for object detection and tracking.
        
        Args:
            frame: Input frame (numpy array or image path)
            mode: Detection mode to use
            processing_mode: Processing mode (real-time vs batch)
            frame_skip: Frame skip rate (override mode default)
            
        Returns:
            List of TrackedObject instances
        """
        if frame_skip is None:
            mode_config = get_mode_metadata(mode)
            frame_skip = mode_config.frame_skip if mode_config else 1
        
        # Skip frame if needed
        if self._frame_count % frame_skip != 0:
            self._frame_count += 1
            return []
        
        # Get model instance
        model = self.model_manager.get_model_instance(mode)
        if model is None:
            self.logger.warning(f"No active model for mode {mode.value}")
            return []
        
        # Run inference
        try:
            results = model(frame, verbose=False)
            
            if not results or len(results) == 0:
                return []
            
            result = results[0]
            if not hasattr(result, 'boxes') or result.boxes is None:
                return []
            
            boxes = result.boxes
            if len(boxes) == 0:
                return []
            
            # Get metadata for current mode
            mode_config = get_mode_metadata(mode)
            if not mode_config:
                return []
            
            tracked_objects = []
            current_time = time.time()
            
            for i, box in enumerate(boxes):
                # Extract detection data
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                confidence = float(box.conf.item())
                class_id = int(box.cls.item())
                
                # Filter by confidence threshold
                if confidence < mode_config.confidence_threshold:
                    continue
                
                # Calculate centroid
                centroid_x = (x1 + x2) // 2
                centroid_y = (y1 + y2) // 2
                
                # Create track ID
                track_id = f"{mode.value}_{class_id}_{i}"
                
                # Apply smoothing
                smoother = self._get_smoother(track_id)
                smoothed_centroid = smoother.update((centroid_x, centroid_y))
                
                # Calculate velocity
                track_history = self._get_track_history(track_id)
                track_history.append((smoothed_centroid[0], smoothed_centroid[1], current_time))
                
                velocity = None
                if len(track_history) >= 2:
                    prev_x, prev_y, prev_time = track_history[-2]
                    curr_x, curr_y, curr_time = track_history[-1]
                    
                    if curr_time > prev_time:
                        dt = curr_time - prev_time
                        dx = curr_x - prev_x
                        dy = curr_y - prev_y
                        velocity = (dx / dt, dy / dt)
                
                # Create tracked object
                tracked_obj = TrackedObject(
                    track_id=track_id,
                    class_id=class_id,
                    class_name=self._get_class_name(class_id, mode),
                    bbox=(x1, y1, x2, y2),
                    centroid=(centroid_x, centroid_y),
                    confidence=confidence,
                    frame_number=self._frame_count,
                    timestamp=current_time,
                    smoothed_centroid=smoothed_centroid,
                    velocity=velocity
                )
                
                tracked_objects.append(tracked_obj)
            
            self._frame_count += 1
            return tracked_objects
            
        except Exception as e:
            self.logger.error(f"Error processing frame: {e}")
            self._frame_count += 1
            return []
    
    def _get_class_name(self, class_id: int, mode: DetectionMode) -> str:
        """Get class name for class ID."""
        # This would typically come from the model metadata
        # For now, return generic names based on mode and class ID
        if mode == DetectionMode.BALL:
            return "tennis_ball"
        elif mode == DetectionMode.RACKET:
            return "racket"
        elif mode == DetectionMode.PLAYER:
            return "person"
        else:
            return f"object_{class_id}"
    
    def reset(self):
        """Reset the processor state."""
        self._smoothers.clear()
        self._track_history.clear()
        self._frame_count = 0


class DetectionEngine:
    """Main detection engine orchestrating frame processing."""
    
    def __init__(
        self,
        models_dir: str = "models",
        persist_path: str = "model_registry.json"
    ):
        """
        Initialize the detection engine.
        
        Args:
            models_dir: Directory for model storage
            persist_path: Path to persist model registry
        """
        self.model_manager = ModelManager(models_dir, persist_path)
        self.processor = FrameProcessor(self.model_manager)
        self.logger = logging.getLogger(__name__)
        
        # Current configuration
        self._current_detection_mode = DetectionMode.BALL
        self._current_processing_mode = ProcessingMode.REAL_TIME
        self._use_tracking = True
        self._smoothing_method = "ema"
        
        # Performance metrics
        self._processing_times = deque(maxlen=100)
        self._total_processed = 0
    
    def set_detection_mode(self, mode: DetectionMode) -> bool:
        """
        Set the active detection mode.
        
        Args:
            mode: Detection mode to use
            
        Returns:
            True if successful
        """
        active_model = self.model_manager.get_active_model(mode)
        if active_model is None:
            self.logger.warning(f"No active model set for mode {mode.value}")
            return False
        
        self._current_detection_mode = mode
        self.processor.reset()  # Reset tracking state when switching modes
        
        self.logger.info(f"Switched to detection mode: {mode.value}")
        return True
    
    def set_processing_mode(self, mode: ProcessingMode) -> bool:
        """
        Set the processing mode.
        
        Args:
            mode: Processing mode to use
            
        Returns:
            True if successful
        """
        self._current_processing_mode = mode
        self.logger.info(f"Switched to processing mode: {mode.value}")
        return True
    
    def get_available_modes(self) -> List[DetectionMode]:
        """Get list of available detection modes."""
        return [mode for mode in DetectionMode if self.model_manager.get_active_model(mode) is not None]
    
    def process(
        self,
        frame: Any,
        mode: Optional[DetectionMode] = None,
        processing_mode: Optional[ProcessingMode] = None,
        **kwargs
    ) -> List[TrackedObject]:
        """
        Process a frame with the current or specified mode.
        
        Args:
            frame: Input frame
            mode: Optional override for detection mode
            processing_mode: Optional override for processing mode
            **kwargs: Additional arguments passed to frame processor
            
        Returns:
            List of TrackedObject instances
        """
        start_time = time.time()
        
        # Use current mode if not specified
        detection_mode = mode if mode is not None else self._current_detection_mode
        proc_mode = processing_mode if processing_mode is not None else self._current_processing_mode
        
        # Process frame
        results = self.processor.process_frame(frame, detection_mode, proc_mode, **kwargs)
        
        # Track performance
        processing_time = time.time() - start_time
        self._processing_times.append(processing_time)
        self._total_processed += 1
        
        return results
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        if self._processing_times:
            avg_time = sum(self._processing_times) / len(self._processing_times)
            min_time = min(self._processing_times)
            max_time = max(self._processing_times)
            fps = 1.0 / avg_time if avg_time > 0 else 0.0
        else:
            avg_time = min_time = max_time = fps = 0.0
        
        return {
            "average_processing_time": avg_time,
            "min_processing_time": min_time,
            "max_processing_time": max_time,
            "fps": fps,
            "total_frames_processed": self._total_processed,
            "current_detection_mode": self._current_detection_mode.value,
            "current_processing_mode": self._current_processing_mode.value
        }
    
    def configure_detection(
        self,
        image_size: Optional[int] = None,
        confidence_threshold: Optional[float] = None,
        nms_threshold: Optional[float] = None,
        frame_skip: Optional[int] = None,
        smoothing_method: str = "ema"
    ) -> bool:
        """
        Configure detection parameters.
        
        Args:
            image_size: Input image size for model
            confidence_threshold: Confidence threshold for detections
            nms_threshold: NMS threshold for detections
            frame_skip: Frame skip rate
            smoothing_method: Smoothing method ("ema" or "kalman")
            
        Returns:
            True if successful
        """
        # This would update configuration parameters
        # For now, just log the configuration
        config = {}
        if image_size is not None:
            config["image_size"] = image_size
        if confidence_threshold is not None:
            config["confidence_threshold"] = confidence_threshold
        if nms_threshold is not None:
            config["nms_threshold"] = nms_threshold
        if frame_skip is not None:
            config["frame_skip"] = frame_skip
        
        config["smoothing_method"] = smoothing_method
        
        self.logger.info(f"Updated detection configuration: {config}")
        self._smoothing_method = smoothing_method
        
        return True
    
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the detection engine."""
        return {
            "model_manager_status": self.model_manager.get_status(),
            "performance_stats": self.get_performance_stats(),
            "current_configuration": {
                "detection_mode": self._current_detection_mode.value,
                "processing_mode": self._current_processing_mode.value,
                "smoothing_method": self._smoothing_method
            },
            "available_modes": [mode.value for mode in self.get_available_modes()]
        }


# CLI interface for testing
def main():
    """CLI interface for testing the detection engine."""
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="Tennis Detection Engine CLI")
    parser.add_argument("--mode", choices=[mode.value for mode in DetectionMode], 
                       default="ball", help="Detection mode")
    parser.add_argument("--image", help="Path to test image")
    parser.add_argument("--list-models", action="store_true", help="List available models")
    parser.add_argument("--status", action="store_true", help="Show engine status")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create detection engine
    engine = DetectionEngine()
    
    if args.list_models:
        print("Available models:")
        for mode in DetectionMode:
            models = engine.model_manager.list_models(mode)
            print(f"\n{mode.value}:")
            for model in models:
                status = "ACTIVE" if model.is_active else "available"
                print(f"  - {model.name} ({status}, {model.size/1024/1024:.1f}MB)")
    
    elif args.status:
        status = engine.get_status()
        print("Detection Engine Status:")
        print(json.dumps(status, indent=2))
    
    elif args.image:
        # Load and process test image
        try:
            import cv2
            import json
            
            frame = cv2.imread(args.image)
            if frame is None:
                print(f"Failed to load image: {args.image}")
                sys.exit(1)
            
            # Process frame
            mode = DetectionMode(args.mode)
            results = engine.process(frame, mode)
            
            print(f"Processed {len(results)} objects:")
            for obj in results:
                print(f"  - {obj.class_name} (ID: {obj.track_id})")
                print(f"    Confidence: {obj.confidence:.3f}")
                print(f"    BBox: {obj.bbox}")
                print(f"    Centroid: {obj.centroid}")
                if obj.velocity:
                    print(f"    Velocity: {obj.velocity}")
                print()
        
        except ImportError:
            print("cv2 not available. Install with: pip install opencv-python")
        except Exception as e:
            print(f"Error processing image: {e}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
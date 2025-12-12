#!/usr/bin/env python3
"""CLI stub for testing the tennis trajectory detection engine.

This provides a simple command-line interface to test detection functionality
without requiring the full GUI application.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

# Add the core module to the path
sys.path.insert(0, str(Path(__file__).parent))

from core import DetectionMode, ProcessingMode, DetectionEngine


def setup_logging(level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def list_available_models(engine: DetectionEngine) -> None:
    """List all available models by mode."""
    print("=== Available Models ===")
    
    for mode in DetectionMode:
        models = engine.model_manager.list_models(mode)
        active_model = engine.model_manager.get_active_model(mode)
        
        print(f"\n{mode.value.upper()}:")
        if not models:
            print("  No models registered")
            continue
            
        for model in models:
            status = "ACTIVE" if model.name == active_model else "available"
            print(f"  - {model.name} ({status}, {model.size/1024/1024:.1f}MB)")
            if model.validation_score:
                print(f"    Validation score: {model.validation_score:.3f}")


def add_sample_model(engine: DetectionEngine, mode: DetectionMode, model_path: Optional[str] = None) -> None:
    """Add a sample model for testing (creates a placeholder if no real model provided)."""
    
    if model_path and Path(model_path).exists():
        # Use the provided model
        model_name = f"yolov8n_{mode.value}_custom"
    else:
        # Create a placeholder model for testing
        model_name = f"yolov8n_{mode.value}_placeholder"
        model_path = create_placeholder_model(model_name)
    
    # Register the model
    if engine.model_manager.register_model(model_name, model_path, mode):
        engine.model_manager.set_active_model(mode, model_name)
        print(f"Registered and activated model: {model_name} for mode {mode.value}")
    else:
        print(f"Failed to register model: {model_name}")


def create_placeholder_model(model_name: str) -> str:
    """Create a placeholder model file for testing when real YOLO models aren't available."""
    
    # Create a minimal mock model structure
    placeholder_path = Path(f"{model_name}.pt")
    
    # Write minimal content (this won't be a real YOLO model, but allows testing the pipeline)
    with open(placeholder_path, 'wb') as f:
        # Write a simple header that identifies this as a placeholder
        header = b"TENNIS_YOLOv8_PLACEHOLDER\x00"
        f.write(header)
        # Pad to reasonable size
        padding = b'\x00' * (1024 * 1024)  # 1MB placeholder
        f.write(padding)
    
    return str(placeholder_path)


def process_test_image(
    engine: DetectionEngine, 
    image_path: str, 
    mode: DetectionMode,
    confidence_threshold: Optional[float] = None
) -> None:
    """Process a test image and display results."""
    
    try:
        # Import OpenCV for image loading
        import cv2
        import numpy as np
    except ImportError:
        print("OpenCV not available. Install with: pip install opencv-python")
        return
    
    # Load image
    if not Path(image_path).exists():
        print(f"Image not found: {image_path}")
        return
    
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Failed to load image: {image_path}")
        return
    
    print(f"Processing image: {image_path}")
    print(f"Frame shape: {frame.shape}")
    print(f"Detection mode: {mode.value}")
    
    # Configure detection if threshold provided
    if confidence_threshold is not None:
        engine.configure_detection(confidence_threshold=confidence_threshold)
    
    # Process frame
    results = engine.process(frame, mode)
    
    print(f"\n=== Detection Results ===")
    print(f"Processed {len(results)} objects:")
    
    if not results:
        print("No objects detected")
        return
    
    for i, obj in enumerate(results, 1):
        print(f"\nObject {i}:")
        print(f"  Class: {obj.class_name} (ID: {obj.track_id})")
        print(f"  Confidence: {obj.confidence:.3f}")
        print(f"  Bounding Box: {obj.bbox}")
        print(f"  Centroid: {obj.centroid}")
        print(f"  Smoothed Centroid: {obj.smoothed_centroid}")
        
        if obj.velocity:
            print(f"  Velocity: {obj.velocity[0]:.2f}, {obj.velocity[1]:.2f} pixels/sec")
        
        # Size information
        print(f"  Dimensions: {obj.width}x{obj.height} pixels")
        print(f"  Area: {obj.area} pixels²")


def process_video_sample(engine: DetectionEngine, video_path: str, mode: DetectionMode, num_frames: int = 10) -> None:
    """Process a sample of frames from a video."""
    
    try:
        import cv2
    except ImportError:
        print("OpenCV not available. Install with: pip install opencv-python")
        return
    
    if not Path(video_path).exists():
        print(f"Video not found: {video_path}")
        return
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return
    
    print(f"Processing video: {video_path}")
    print(f"Processing {num_frames} frames...")
    
    frame_count = 0
    successful_detections = 0
    
    while frame_count < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        results = engine.process(frame, mode)
        
        if results:
            successful_detections += 1
            
        frame_count += 1
    
    cap.release()
    
    print(f"\n=== Video Processing Results ===")
    print(f"Processed {frame_count} frames")
    print(f"Frames with detections: {successful_detections}")
    print(f"Detection rate: {successful_detections/frame_count*100:.1f}%")


def show_performance_stats(engine: DetectionEngine) -> None:
    """Display performance statistics."""
    
    stats = engine.get_performance_stats()
    
    print("=== Performance Statistics ===")
    print(f"Average processing time: {stats['average_processing_time']*1000:.1f} ms")
    print(f"FPS: {stats['fps']:.1f}")
    print(f"Total frames processed: {stats['total_frames_processed']}")
    print(f"Current mode: {stats['current_detection_mode']}")
    print(f"Processing mode: {stats['current_processing_mode']}")


def create_test_data() -> None:
    """Create test data directories and sample files."""
    
    # Create models directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Create a simple test image (if OpenCV is available)
    try:
        import cv2
        import numpy as np
        
        # Create a simple test image
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Add some "tennis ball" like circles for testing
        cv2.circle(test_image, (320, 240), 15, (0, 255, 255), -1)  # Tennis ball
        cv2.circle(test_image, (200, 300), 20, (100, 50, 200), 3)  # Racket area
        cv2.rectangle(test_image, (400, 100), (500, 400), (200, 100, 50), 2)  # Player area
        
        test_image_path = models_dir / "test_image.jpg"
        cv2.imwrite(str(test_image_path), test_image)
        
        print(f"Created test image: {test_image_path}")
        
    except ImportError:
        print("OpenCV not available - skipping test image creation")


def main():
    """Main CLI entry point."""
    
    parser = argparse.ArgumentParser(
        description="Tennis Trajectory Detection Engine CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --list-models
  %(prog)s --add-model ball --yolo-model tennis_ball.pt
  %(prog)s --test-image test_image.jpg --mode ball --confidence 0.7
  %(prog)s --test-video tennis.mp4 --mode ball --frames 50
  %(prog)s --status
        """
    )
    
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    parser.add_argument("--models-dir", default="models", help="Models directory")
    
    # Model management
    parser.add_argument("--list-models", action="store_true", help="List all available models")
    parser.add_argument("--add-model", choices=[mode.value for mode in DetectionMode], 
                       help="Add a model for specified mode")
    parser.add_argument("--yolo-model", help="Path to YOLO model file (.pt)")
    parser.add_argument("--set-active", nargs=2, metavar=('MODE', 'MODEL_NAME'), 
                       help="Set active model for a mode")
    
    # Testing
    parser.add_argument("--test-image", help="Path to test image")
    parser.add_argument("--test-video", help="Path to test video")
    parser.add_argument("--frames", type=int, default=10, help="Number of frames to process from video")
    
    # Configuration
    parser.add_argument("--mode", choices=[mode.value for mode in DetectionMode], 
                       default="ball", help="Detection mode")
    parser.add_argument("--confidence", type=float, help="Confidence threshold")
    parser.add_argument("--processing-mode", choices=[mode.value for mode in ProcessingMode],
                       default="real-time", help="Processing mode")
    
    # Status and utilities
    parser.add_argument("--status", action="store_true", help="Show engine status")
    parser.add_argument("--performance", action="store_true", help="Show performance statistics")
    parser.add_argument("--create-test-data", action="store_true", help="Create test data")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(log_level)
    
    # Create engine
    engine = DetectionEngine(models_dir=args.models_dir)
    
    # Handle commands
    if args.create_test_data:
        create_test_data()
        return
    
    if args.list_models:
        list_available_models(engine)
        return
    
    if args.status:
        status = engine.get_status()
        print("=== Engine Status ===")
        print(json.dumps(status, indent=2))
        return
    
    if args.performance:
        show_performance_stats(engine)
        return
    
    if args.add_model:
        mode = DetectionMode(args.add_model)
        add_sample_model(engine, mode, args.yolo_model)
        return
    
    if args.set_active:
        mode_name, model_name = args.set_active
        try:
            mode = DetectionMode(mode_name)
            if engine.model_manager.set_active_model(mode, model_name):
                print(f"Set active model '{model_name}' for mode {mode_name}")
            else:
                print(f"Failed to set active model")
        except ValueError as e:
            print(f"Error: {e}")
        return
    
    # Test processing
    if args.test_image:
        mode = DetectionMode(args.mode)
        process_test_image(engine, args.test_image, mode, args.confidence)
        return
    
    if args.test_video:
        mode = DetectionMode(args.mode)
        process_video_sample(engine, args.test_video, mode, args.frames)
        return
    
    # Default: show help
    parser.print_help()


if __name__ == "__main__":
    main()
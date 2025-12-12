# Tennis Trajectory Detection Engine

A YOLOv8n-based detection engine for tennis trajectory analysis, providing real-time and batch processing capabilities for detecting tennis balls, rackets, and players in video streams.

## Features

- **Multiple Detection Modes**: Ball, racket, and player detection
- **Flexible Processing**: Real-time and batch processing modes
- **Model Management**: Support for multiple YOLOv8n models with lazy loading
- **Trajectory Smoothing**: Kalman and EMA filtering for stable trajectories
- **Performance Tracking**: Built-in performance monitoring and statistics
- **CLI Interface**: Command-line tool for testing and validation
- **GUI Interface**: Full graphical interface with model setup popup
- **Batch File Startup**: Easy Windows startup with `start.bat`
- **Extensible**: Easy to add new detection modes and custom models

## Architecture

The detection engine consists of three main components:

1. **modes.py**: Defines detection modes, processing configurations, and metadata
2. **model_manager.py**: Manages multiple YOLOv8n models with persistence
3. **detection_engine.py**: Core detection engine with frame processing and tracking

## Installation

### Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Core Dependencies

- `ultralytics>=8.0.0` - YOLOv8 implementation
- `opencv-python>=4.8.0` - Computer vision operations
- `torch>=2.0.0` - Deep learning framework
- `numpy>=1.24.0` - Numerical operations

### Optional Dependencies

For enhanced functionality:
- `scipy>=1.10.0` - Advanced signal processing
- `matplotlib>=3.7.0` - Visualization
- `scikit-image>=0.21.0` - Advanced image processing

## Quick Start

### Option 1: Windows Batch File (Recommended)

**Double-click `start.bat`** for an easy-to-use menu interface:
- GUI interface with model setup popup
- CLI interface
- Demo mode
- Test runner
- Status checker
- Dependency installer

### Option 2: GUI Interface

```bash
# Launch the full GUI application
python detection_gui.py

# Or use the popup-only interface
python gui_popup.py
```

### Option 3: CLI Interface

#### 1. List Available Models

```bash
python cli.py --list-models
```

#### 2. Add a Detection Model

```bash
# Add a tennis ball detection model
python cli.py --add-model ball --yolo-model tennis_ball.pt

# Add a racket detection model
python cli.py --add-model racket --yolo-model racket_detector.pt

# Add a player detection model
python cli.py --add-model player --yolo-model person_detector.pt
```

#### 3. Test Detection on Image

```bash
# Process an image for ball detection
python cli.py --test-image tennis_frame.jpg --mode ball --confidence 0.7

# Process with different thresholds
python cli.py --test-image tennis_frame.jpg --mode player --confidence 0.8
```

#### 4. Test Detection on Video

```bash
# Process first 50 frames of a tennis video
python cli.py --test-video tennis_match.mp4 --mode ball --frames 50
```

#### 5. Check Engine Status

```bash
# Show current engine status
python cli.py --status

# Show performance statistics
python cli.py --performance
```

### Option 4: Demo Mode

```bash
# Run comprehensive demonstration
python demo.py
```

## Programming Interface

### Basic Usage

```python
from core import DetectionEngine, DetectionMode, ProcessingMode

# Create detection engine
engine = DetectionEngine(models_dir="models")

# Set active models for each mode
engine.model_manager.register_model("ball_v1", "tennis_ball.pt", DetectionMode.BALL)
engine.model_manager.set_active_model(DetectionMode.BALL, "ball_v1")

# Process a frame
results = engine.process(frame, DetectionMode.BALL)

# Access results
for obj in results:
    print(f"Detected {obj.class_name} at {obj.centroid}")
    print(f"Confidence: {obj.confidence:.3f}")
    print(f"Velocity: {obj.velocity}")
```

### Advanced Configuration

```python
# Configure detection parameters
engine.configure_detection(
    image_size=1280,
    confidence_threshold=0.8,
    nms_threshold=0.4,
    smoothing_method="kalman"
)

# Switch modes
engine.set_detection_mode(DetectionMode.PLAYER)
engine.set_processing_mode(ProcessingMode.BATCH)

# Get performance statistics
stats = engine.get_performance_stats()
print(f"FPS: {stats['fps']:.1f}")
```

### Model Management

```python
from core import ModelManager, DetectionMode

# Create model manager
manager = ModelManager(models_dir="models")

# Register multiple models
manager.register_model("fast_ball", "yolov8n_ball_fast.pt", DetectionMode.BALL)
manager.register_model("accurate_ball", "yolov8n_ball_accurate.pt", DetectionMode.BALL)

# List available models
models = manager.list_models(DetectionMode.BALL)
for model in models:
    print(f"{model.name}: {model.size/1024/1024:.1f}MB, Score: {model.validation_score}")

# Set active model
manager.set_active_model(DetectionMode.BALL, "fast_ball")

# Get loaded model instance
yolo_model = manager.get_model_instance(DetectionMode.BALL)
```

## Detection Modes

### Ball Detection
- **Target**: Tennis balls
- **Classes**: Sports ball (COCO class 32)
- **Optimized for**: High-speed trajectory tracking
- **Smoothing**: Heavy EMA smoothing for trajectory stability

### Racket Detection
- **Target**: Tennis rackets
- **Classes**: Sports equipment (custom classes 38, 39)
- **Optimized for**: Equipment tracking and player analysis
- **Smoothing**: Moderate smoothing for position stability

### Player Detection
- **Target**: Tennis players
- **Classes**: Person (COCO class 0)
- **Optimized for**: Player position and movement tracking
- **Smoothing**: Moderate smoothing with velocity tracking

## Processing Modes

### Real-Time Processing
- **Image Size**: 640x640
- **Batch Size**: 1 (single frame)
- **Buffer**: 8 frames
- **Optimized for**: Live video streams

### Batch Processing
- **Image Size**: 1280x1280
- **Batch Size**: 4 frames
- **Buffer**: 32 frames
- **Optimized for**: Video analysis

## Result Structure

Each detection returns a `TrackedObject` with:

```python
@dataclass
class TrackedObject:
    track_id: str              # Unique tracking ID
    class_id: int              # YOLO class ID
    class_name: str            # Human-readable class name
    bbox: Tuple[int, int, int, int]  # Bounding box (x1, y1, x2, y2)
    centroid: Tuple[int, int]  # Object center
    confidence: float          # Detection confidence
    frame_number: int          # Frame index
    timestamp: float           # Processing timestamp
    smoothed_centroid: Optional[Tuple[int, int]]  # Smoothed position
    velocity: Optional[Tuple[float, float]]       # Pixel velocity (dx/dt, dy/dt)
```

## Performance Tuning

### Confidence Thresholds
- **Ball Detection**: 0.6 (optimized for small, fast-moving objects)
- **Racket Detection**: 0.5 (allow some false positives)
- **Player Detection**: 0.7 (prioritize accuracy for larger objects)

### Frame Skipping
- **Ball**: Process every frame (frame_skip=1)
- **Racket**: Process every 2nd frame (frame_skip=2)
- **Player**: Process every frame (frame_skip=1)

### Smoothing Factors
- **EMA Alpha**: 0.8 (high smoothing for stability)
- **Kalman Process Noise**: 0.1
- **Kalman Measurement Noise**: 1.0

## Testing

Run the test suite:

```bash
python test_detection_engine.py
```

Create test data:

```bash
python cli.py --create-test-data
```

## Configuration Files

### Model Registry
The engine maintains a JSON registry at `model_registry.json`:
```json
{
  "models": {
    "ball_v1": {
      "name": "ball_v1",
      "file_path": "models/ball_v1.pt",
      "mode": "ball",
      "device": "cuda",
      "size": 12345678,
      "validation_score": 0.892
    }
  },
  "active_models": {
    "ball": "ball_v1",
    "racket": null,
    "player": null
  }
}
```

## Troubleshooting

### Common Issues

1. **"ultralytics package not installed"**
   ```bash
   pip install ultralytics
   ```

2. **"No active model for mode"**
   - Register a model first: `--add-model ball --yolo-model model.pt`
   - Set as active: `--set-active ball model_name`

3. **CUDA out of memory**
   - Use CPU device: register model with `device="cpu"`
   - Reduce image size: configure with `image_size=640`

4. **Low detection confidence**
   - Lower confidence threshold: `--confidence 0.4`
   - Check model quality and image resolution

### Debug Mode

Enable verbose logging:
```bash
python cli.py --verbose --test-image image.jpg --mode ball
```

## Development

### Adding New Detection Modes

1. Add mode to `DetectionMode` enum in `modes.py`
2. Define metadata in `DETECTION_MODE_CONFIGS`
3. Update `get_available_modes()` function

### Adding Custom Smoothing

1. Implement smoother class (extend `ExponentialMovingAverage`)
2. Add to `_get_smoother()` method in `FrameProcessor`
3. Update configuration options

## License

This project is part of the MetaGPT framework and follows the same licensing terms.
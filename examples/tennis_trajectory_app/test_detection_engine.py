"""Unit tests for the tennis trajectory detection engine."""

import unittest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import json

# Add the core module to the path for testing
import sys
sys.path.insert(0, str(Path(__file__).parent / "core"))

from core import (
    DetectionMode, ProcessingMode, 
    ModelManager, ModelInfo,
    DetectionEngine, FrameProcessor, TrackedObject,
    ExponentialMovingAverage, KalmanSmoother,
    get_mode_metadata, get_processing_config
)


class TestModes(unittest.TestCase):
    """Test the modes configuration system."""
    
    def test_detection_mode_enum(self):
        """Test DetectionMode enum values."""
        self.assertEqual(DetectionMode.BALL.value, "ball")
        self.assertEqual(DetectionMode.RACKET.value, "racket")
        self.assertEqual(DetectionMode.PLAYER.value, "player")
    
    def test_processing_mode_enum(self):
        """Test ProcessingMode enum values."""
        self.assertEqual(ProcessingMode.REAL_TIME.value, "real-time")
        self.assertEqual(ProcessingMode.BATCH.value, "batch")
    
    def test_mode_metadata(self):
        """Test mode metadata retrieval."""
        ball_metadata = get_mode_metadata(DetectionMode.BALL)
        self.assertIsNotNone(ball_metadata)
        self.assertIn(32, ball_metadata.class_ids)  # Sports ball class ID
        self.assertGreater(ball_metadata.confidence_threshold, 0)
        
        player_metadata = get_mode_metadata(DetectionMode.PLAYER)
        self.assertIsNotNone(player_metadata)
        self.assertIn(0, player_metadata.class_ids)  # Person class ID
    
    def test_processing_config(self):
        """Test processing mode configurations."""
        realtime_config = get_processing_config(ProcessingMode.REAL_TIME)
        self.assertIsNotNone(realtime_config)
        self.assertEqual(realtime_config["batch_size"], 1)
        self.assertEqual(realtime_config["image_size"], 640)
        
        batch_config = get_processing_config(ProcessingMode.BATCH)
        self.assertIsNotNone(batch_config)
        self.assertGreater(batch_config["batch_size"], 1)
        self.assertGreater(batch_config["image_size"], realtime_config["image_size"])


class TestModelManager(unittest.TestCase):
    """Test the model management system."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.models_dir = Path(self.temp_dir) / "models"
        self.registry_path = Path(self.temp_dir) / "registry.json"
        
        # Create model manager
        self.manager = ModelManager(
            models_dir=str(self.models_dir),
            persist_path=str(self.registry_path)
        )
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def create_mock_model(self, name: str, size: int = 1024*1024) -> str:
        """Create a mock model file."""
        model_path = self.models_dir / f"{name}.pt"
        model_path.parent.mkdir(exist_ok=True)
        
        with open(model_path, 'wb') as f:
            f.write(b'MOCK_YOLO_MODEL' + b'\x00' * (size - len(b'MOCK_YOLO_MODEL')))
        
        return str(model_path)
    
    def test_model_registration(self):
        """Test model registration functionality."""
        model_path = self.create_mock_model("test_ball_model")
        
        # Register a model
        success = self.manager.register_model(
            name="test_ball",
            file_path=model_path,
            mode=DetectionMode.BALL,
            device="cpu"
        )
        
        self.assertTrue(success)
        self.assertIn("test_ball", self.manager._models)
        
        model_info = self.manager.get_model_info("test_ball")
        self.assertIsNotNone(model_info)
        self.assertEqual(model_info.mode, DetectionMode.BALL)
        self.assertEqual(model_info.device, "cpu")
        self.assertGreater(model_info.size, 0)
    
    def test_model_activation(self):
        """Test model activation functionality."""
        model_path = self.create_mock_model("test_racket")
        
        # Register and activate model
        self.manager.register_model("test_racket", model_path, DetectionMode.RACKET)
        success = self.manager.set_active_model(DetectionMode.RACKET, "test_racket")
        
        self.assertTrue(success)
        self.assertEqual(self.manager.get_active_model(DetectionMode.RACKET), "test_racket")
    
    def test_model_listing(self):
        """Test model listing functionality."""
        # Register models for different modes
        ball_path = self.create_mock_model("ball_v1")
        racket_path = self.create_mock_model("racket_v1")
        player_path = self.create_mock_model("player_v1")
        
        self.manager.register_model("ball_v1", ball_path, DetectionMode.BALL)
        self.manager.register_model("racket_v1", racket_path, DetectionMode.RACKET)
        self.manager.register_model("player_v1", player_path, DetectionMode.PLAYER)
        
        # List all models
        all_models = self.manager.list_models()
        self.assertEqual(len(all_models), 3)
        
        # List by mode
        ball_models = self.manager.list_models(DetectionMode.BALL)
        self.assertEqual(len(ball_models), 1)
        self.assertEqual(ball_models[0].name, "ball_v1")
    
    def test_registry_persistence(self):
        """Test that model registry is persisted."""
        model_path = self.create_mock_model("persistent_model")
        
        # Register and activate model
        self.manager.register_model("persistent_model", model_path, DetectionMode.BALL)
        self.manager.set_active_model(DetectionMode.BALL, "persistent_model")
        
        # Create new manager (should load from registry)
        new_manager = ModelManager(
            models_dir=str(self.models_dir),
            persist_path=str(self.registry_path)
        )
        
        self.assertIn("persistent_model", new_manager._models)
        self.assertEqual(
            new_manager.get_active_model(DetectionMode.BALL), 
            "persistent_model"
        )
    
    def test_status_reporting(self):
        """Test status reporting functionality."""
        model_path = self.create_mock_model("status_model")
        self.manager.register_model("status_model", model_path, DetectionMode.BALL)
        
        status = self.manager.get_status()
        
        self.assertEqual(status["total_models"], 1)
        self.assertIsInstance(status["active_models"], dict)
        self.assertIsInstance(status["available_devices"], dict)


class TestSmoothingClasses(unittest.TestCase):
    """Test smoothing and filtering classes."""
    
    def test_exponential_moving_average(self):
        """Test EMA smoothing."""
        ema = ExponentialMovingAverage(alpha=0.8)
        
        # First value should be returned as-is
        smoothed1 = ema.update((10, 20))
        self.assertEqual(smoothed1, (10, 20))
        
        # Second value should be smoothed
        smoothed2 = ema.update((20, 30))
        # With alpha=0.8: 0.8*20 + 0.2*10 = 18 for x, 0.8*30 + 0.2*20 = 28 for y
        expected_x = 0.8 * 20 + 0.2 * 10
        expected_y = 0.8 * 30 + 0.2 * 20
        self.assertAlmostEqual(smoothed2[0], expected_x, places=5)
        self.assertAlmostEqual(smoothed2[1], expected_y, places=5)
        
        # Test reset
        ema.reset()
        smoothed3 = ema.update((50, 60))
        self.assertEqual(smoothed3, (50, 60))
    
    def test_kalman_smoother(self):
        """Test Kalman filter smoothing."""
        try:
            kalman = KalmanSmoother(process_noise=0.1, measurement_noise=1.0)
            
            # First measurement
            smoothed1 = kalman.update((10, 20))
            self.assertEqual(len(smoothed1), 2)
            
            # Second measurement (should be closer to measurement than prediction)
            smoothed2 = kalman.update((15, 25))
            self.assertNotEqual(smoothed1, smoothed2)
        except ImportError:
            self.skipTest("numpy not available for KalmanSmoother test")


class TestTrackedObject(unittest.TestCase):
    """Test TrackedObject dataclass."""
    
    def test_basic_properties(self):
        """Test TrackedObject basic properties and calculations."""
        obj = TrackedObject(
            track_id="test_1",
            class_id=32,
            class_name="tennis_ball",
            bbox=(100, 100, 150, 150),
            centroid=(125, 125),
            confidence=0.95,
            frame_number=1,
            timestamp=1234567890.0
        )
        
        self.assertEqual(obj.center_x, 125)
        self.assertEqual(obj.center_y, 125)
        self.assertEqual(obj.width, 50)
        self.assertEqual(obj.height, 50)
        self.assertEqual(obj.area, 2500)
        self.assertEqual(obj.track_id, "test_1")
        self.assertEqual(obj.class_name, "tennis_ball")
    
    def test_smoothed_properties(self):
        """Test TrackedObject with smoothing."""
        obj = TrackedObject(
            track_id="test_smooth",
            class_id=0,
            class_name="person",
            bbox=(200, 100, 280, 350),
            centroid=(240, 225),
            confidence=0.85,
            frame_number=5,
            timestamp=1234567895.0,
            smoothed_centroid=(242, 228),
            velocity=(2.5, 3.0)
        )
        
        self.assertEqual(obj.smoothed_centroid, (242, 228))
        self.assertEqual(obj.velocity, (2.5, 3.0))
        self.assertEqual(obj.center_x, 240)


class TestDetectionEngine(unittest.TestCase):
    """Test the main DetectionEngine class."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.models_dir = Path(self.temp_dir) / "models"
        
        self.engine = DetectionEngine(
            models_dir=str(self.models_dir),
            persist_path=str(Path(self.temp_dir) / "engine_registry.json")
        )
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_engine_initialization(self):
        """Test DetectionEngine initialization."""
        self.assertIsNotNone(self.engine.model_manager)
        self.assertIsNotNone(self.engine.processor)
        self.assertEqual(self.engine._current_detection_mode, DetectionMode.BALL)
        self.assertEqual(self.engine._current_processing_mode, ProcessingMode.REAL_TIME)
    
    def test_mode_configuration(self):
        """Test mode switching and configuration."""
        # Test detection mode setting (should fail without active model)
        result = self.engine.set_detection_mode(DetectionMode.RACKET)
        self.assertFalse(result)  # No active model
        
        # Test processing mode setting
        result = self.engine.set_processing_mode(ProcessingMode.BATCH)
        self.assertTrue(result)
        self.assertEqual(self.engine._current_processing_mode, ProcessingMode.BATCH)
    
    def test_process_with_mock_frame(self):
        """Test frame processing with mock data."""
        # Create a mock frame
        mock_frame = MagicMock()
        
        # Mock model
        mock_model = MagicMock()
        mock_results = [MagicMock()]
        mock_results[0].boxes = MagicMock()
        mock_results[0].boxes.xyxy = [[[100, 100, 150, 150]]]
        mock_results[0].boxes.conf = [0.95]
        mock_results[0].boxes.cls = [32]
        
        mock_model.return_value = mock_results
        
        # Mock model manager to return our mock model
        self.engine.model_manager.get_active_model = Mock(return_value="mock_model")
        self.engine.model_manager._models["mock_model"] = Mock()
        self.engine.model_manager._models["mock_model"].file_path = "mock.pt"
        self.engine.model_manager._loaded_models["mock_model"] = mock_model
        
        # Process frame
        results = self.engine.process(mock_frame, DetectionMode.BALL)
        
        # Should return empty list since we have mocking setup
        self.assertIsInstance(results, list)
    
    def test_performance_stats(self):
        """Test performance statistics collection."""
        stats = self.engine.get_performance_stats()
        
        self.assertIn("average_processing_time", stats)
        self.assertIn("fps", stats)
        self.assertIn("total_frames_processed", stats)
        self.assertIn("current_detection_mode", stats)
        self.assertEqual(stats["current_detection_mode"], "ball")
    
    def test_status_reporting(self):
        """Test status reporting."""
        status = self.engine.get_status()
        
        self.assertIn("model_manager_status", status)
        self.assertIn("performance_stats", status)
        self.assertIn("current_configuration", status)
        self.assertIn("available_modes", status)
    
    def test_configuration(self):
        """Test detection configuration."""
        result = self.engine.configure_detection(
            image_size=1280,
            confidence_threshold=0.8,
            smoothing_method="kalman"
        )
        
        self.assertTrue(result)


class TestFrameProcessor(unittest.TestCase):
    """Test FrameProcessor class."""
    
    def setUp(self):
        """Set up test environment."""
        temp_dir = tempfile.mkdtemp()
        models_dir = Path(temp_dir) / "models"
        self.processor = FrameProcessor(
            model_manager=ModelManager(models_dir=str(models_dir))
        )
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(Path(self.processor.model_manager.models_dir).parent)
    
    def test_smoother_creation(self):
        """Test smoother creation and management."""
        smoother = self.processor._get_smoother("track_1", "ema")
        self.assertIsInstance(smoother, ExponentialMovingAverage)
        
        smoother2 = self.processor._get_smoother("track_1", "ema")
        self.assertIs(smoother, smoother2)  # Same instance
    
    def test_track_history(self):
        """Test track history management."""
        history = self.processor._get_track_history("track_1")
        self.assertIsInstance(history, type(self.processor._track_history["track_1"]))
        
        # Add some values
        history.append((10, 20, 1.0))
        history.append((15, 25, 2.0))
        
        self.assertEqual(len(history), 2)
    
    def test_reset(self):
        """Test processor reset."""
        # Add some state
        self.processor._get_smoother("track_1")
        self.processor._get_track_history("track_1")
        self.processor._frame_count = 10
        
        # Reset
        self.processor.reset()
        
        self.assertEqual(len(self.processor._smoothers), 0)
        self.assertEqual(len(self.processor._track_history), 0)
        self.assertEqual(self.processor._frame_count, 0)


def create_test_suite():
    """Create and return test suite."""
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestModes,
        TestModelManager,
        TestSmoothingClasses,
        TestTrackedObject,
        TestDetectionEngine,
        TestFrameProcessor
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    return suite


if __name__ == "__main__":
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    test_suite = create_test_suite()
    
    result = runner.run(test_suite)
    
    # Exit with appropriate code
    exit(0 if result.wasSuccessful() else 1)
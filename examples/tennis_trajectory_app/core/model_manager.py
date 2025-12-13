"""Model management system for YOLOv8n detection models."""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Any, List, Union
from dataclasses import dataclass, asdict
from threading import Lock

# Optional dependencies
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

from .modes import DetectionMode, ProcessingMode, get_mode_metadata, get_processing_config


@dataclass
class ModelInfo:
    """Information about a registered detection model."""
    name: str
    file_path: str
    mode: DetectionMode
    device: str
    size: int
    created_at: str
    is_active: bool = False
    validation_score: Optional[float] = None
    metadata: Dict[str, Any] = None


class ModelManager:
    """Manages multiple YOLOv8n models for different detection modes."""
    
    def __init__(self, models_dir: str = "models", persist_path: str = "model_registry.json"):
        """
        Initialize the model manager.
        
        Args:
            models_dir: Directory to store model files
            persist_path: Path to persist model registry
        """
        self.models_dir = Path(models_dir)
        self.persist_path = Path(persist_path)
        self.models_dir.mkdir(exist_ok=True)
        
        # Model registry and active selections
        self._models: Dict[str, ModelInfo] = {}
        self._active_models: Dict[DetectionMode, Optional[str]] = {
            mode: None for mode in DetectionMode
        }
        
        # Loaded model instances (lazy loading)
        self._loaded_models: Dict[str, Any] = {}
        self._model_lock = Lock()
        
        # YOLO imports (lazy)
        self._YOLO = None
        
        # Logging (must be set before _load_registry)
        self.logger = logging.getLogger(__name__)
        
        # Load existing registry if available
        self._load_registry()
    
    def _load_yolo(self):
        """Lazy import YOLO to avoid dependency issues."""
        if self._YOLO is None:
            try:
                from ultralytics import YOLO as YOLOClass
                self._YOLO = YOLOClass
            except ImportError:
                raise ImportError("ultralytics package not installed. Install with: pip install ultralytics")
    
    def _load_registry(self):
        """Load model registry from persistent storage."""
        if self.persist_path.exists():
            try:
                with open(self.persist_path, 'r') as f:
                    data = json.load(f)
                    
                # Reconstruct ModelInfo objects
                for model_name, model_data in data.get('models', {}).items():
                    # Convert mode string back to DetectionMode enum
                    mode = DetectionMode(model_data['mode'])
                    
                    # Create ModelInfo object (excluding the string mode, using enum instead)
                    model_info = ModelInfo(
                        name=model_data['name'],
                        file_path=model_data['file_path'],
                        mode=mode,
                        device=model_data['device'],
                        size=model_data['size'],
                        created_at=model_data['created_at'],
                        is_active=model_data.get('is_active', False),
                        validation_score=model_data.get('validation_score'),
                        metadata=model_data.get('metadata')
                    )
                    self._models[model_name] = model_info
                
                # Restore active model selections
                for mode_name, model_name in data.get('active_models', {}).items():
                    if model_name:  # Skip None values
                        mode = DetectionMode(mode_name)
                        self._active_models[mode] = model_name
                    
                self.logger.info(f"Loaded {len(self._models)} models from registry")
                
            except Exception as e:
                self.logger.error(f"Failed to load registry: {e}")
    
    def _save_registry(self):
        """Save model registry to persistent storage."""
        try:
            # Convert models to serializable format
            models_data = {}
            for name, info in self._models.items():
                info_dict = asdict(info)
                # Convert DetectionMode enum to string
                info_dict['mode'] = info.mode.value
                models_data[name] = info_dict
            
            data = {
                'models': models_data,
                'active_models': {mode.value: model for mode, model in self._active_models.items()}
            }
            
            with open(self.persist_path, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to save registry: {e}")
    
    def register_model(
        self, 
        name: str, 
        file_path: Union[str, Path], 
        mode: DetectionMode,
        device: str = "auto",
        validation_score: Optional[float] = None
    ) -> bool:
        """
        Register a new model for a specific detection mode.
        
        Args:
            name: Unique name for the model
            file_path: Path to the .pt model file
            mode: Detection mode this model is for
            device: Device to load the model on
            validation_score: Optional validation score
            
        Returns:
            True if registration successful
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"Model file not found: {file_path}")
            
            # Copy model to models directory if it's not already there
            if file_path.parent != self.models_dir:
                import shutil
                dest_path = self.models_dir / f"{name}.pt"
                shutil.copy2(file_path, dest_path)
                file_path = dest_path
            
            model_info = ModelInfo(
                name=name,
                file_path=str(file_path),
                mode=mode,
                device=device,
                size=file_path.stat().st_size,
                created_at=str(Path(file_path).stat().st_mtime),
                validation_score=validation_score
            )
            
            self._models[name] = model_info
            self._save_registry()
            
            self.logger.info(f"Registered model '{name}' for mode {mode.value}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register model '{name}': {e}")
            return False
    
    def set_active_model(self, mode: DetectionMode, model_name: str) -> bool:
        """
        Set the active model for a specific detection mode.
        
        Args:
            mode: Detection mode
            model_name: Name of the model to activate
            
        Returns:
            True if successful
        """
        if model_name not in self._models:
            raise ValueError(f"Model '{model_name}' not found")
        
        if self._models[model_name].mode != mode:
            raise ValueError(f"Model '{model_name}' is for mode {self._models[model_name].mode.value}, not {mode.value}")
        
        self._active_models[mode] = model_name
        self._save_registry()
        
        # Unload previous model
        self._unload_model_if_loaded(mode)
        
        self.logger.info(f"Set active model '{model_name}' for mode {mode.value}")
        return True
    
    def get_active_model(self, mode: DetectionMode) -> Optional[str]:
        """Get the currently active model name for a mode."""
        return self._active_models[mode]
    
    def get_model_instance(self, mode: DetectionMode) -> Optional[Any]:
        """
        Get a loaded model instance for the active model of a mode.
        Lazily loads the model if not already loaded.
        
        Args:
            mode: Detection mode
            
        Returns:
            YOLO model instance or None if no active model
        """
        model_name = self._active_models[mode]
        if not model_name:
            return None
        
        with self._model_lock:
            if model_name in self._loaded_models:
                return self._loaded_models[model_name]
            
            # Load model
            model_info = self._models[model_name]
            
            # Determine device
            device = self._get_effective_device(model_info.device)
            
            try:
                self._load_yolo()
                model = self._YOLO(model_info.file_path)
                model.to(device)
                
                self._loaded_models[model_name] = model
                self.logger.info(f"Loaded model '{model_name}' on device {device}")
                
                return model
                
            except Exception as e:
                self.logger.error(f"Failed to load model '{model_name}': {e}")
                return None
    
    def _get_effective_device(self, device: str) -> str:
        """Determine the effective device to use."""
        if device == "auto":
            return "cuda" if (TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu"
        return device
    
    def _unload_model_if_loaded(self, mode: DetectionMode):
        """Unload the currently loaded model for a mode."""
        model_name = self._active_models[mode]
        if model_name and model_name in self._loaded_models:
            # For YOLO models, we can just remove the reference
            # The garbage collector will clean up the model
            del self._loaded_models[model_name]
            self.logger.info(f"Unloaded model '{model_name}' for mode {mode.value}")
    
    def unload_all_models(self):
        """Unload all loaded models to free memory."""
        with self._model_lock:
            self._loaded_models.clear()
            self.logger.info("Unloaded all models")
    
    def list_models(self, mode: Optional[DetectionMode] = None) -> List[ModelInfo]:
        """List all registered models, optionally filtered by mode."""
        models = list(self._models.values())
        if mode:
            models = [m for m in models if m.mode == mode]
        return sorted(models, key=lambda x: x.name)
    
    def get_model_info(self, name: str) -> Optional[ModelInfo]:
        """Get information about a specific model."""
        return self._models.get(name)
    
    def remove_model(self, name: str) -> bool:
        """Remove a model from the registry."""
        if name not in self._models:
            return False
        
        model_info = self._models[name]
        mode = model_info.mode
        
        # If it's the active model, unset it
        if self._active_models[mode] == name:
            self._active_models[mode] = None
        
        # Remove from loaded models if present
        self._unload_model_if_loaded(mode)
        
        # Remove from registry
        del self._models[name]
        
        # Remove file
        try:
            model_path = Path(model_info.file_path)
            if model_path.exists():
                model_path.unlink()
        except Exception as e:
            self.logger.warning(f"Failed to remove model file: {e}")
        
        self._save_registry()
        self.logger.info(f"Removed model '{name}'")
        return True
    
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of all models."""
        return {
            "total_models": len(self._models),
            "active_models": {
                mode.value: model for mode, model in self._active_models.items()
            },
            "loaded_models": list(self._loaded_models.keys()),
            "available_devices": {
                "cuda": TORCH_AVAILABLE and torch.cuda.is_available(),
                "device_count": torch.cuda.device_count() if (TORCH_AVAILABLE and torch.cuda.is_available()) else 0
            },
            "dependencies_available": {
                "torch": TORCH_AVAILABLE,
                "ultralytics": self._YOLO is not None
            }
        }
    
    def validate_model(self, name: str, test_images: List[Any]) -> Optional[float]:
        """
        Validate a model on test images.
        
        Args:
            name: Model name
            test_images: List of test images to validate on
            
        Returns:
            Average confidence score or None if validation fails
        """
        if name not in self._models:
            raise ValueError(f"Model '{name}' not found")
        
        model = None
        try:
            # Temporarily load model for validation
            model_info = self._models[name]
            device = self._get_effective_device(model_info.device)
            
            self._load_yolo()
            model = self._YOLO(model_info.file_path)
            model.to(device)
            
            # Run validation
            scores = []
            for image in test_images:
                results = model(image)
                if results and len(results) > 0:
                    # Get average confidence of detections
                    if hasattr(results[0], 'boxes') and results[0].boxes is not None:
                        confidences = results[0].boxes.conf
                        if len(confidences) > 0:
                            scores.append(float(confidences.mean().item()))
            
            avg_score = sum(scores) / len(scores) if scores else 0.0
            
            # Update model info
            model_info.validation_score = avg_score
            self._save_registry()
            
            self.logger.info(f"Validated model '{name}' with average score: {avg_score:.3f}")
            return avg_score
            
        except Exception as e:
            self.logger.error(f"Failed to validate model '{name}': {e}")
            return None
        
        finally:
            # Cleanup
            del model
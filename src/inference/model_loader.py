"""
Model Loader Module

This module handles loading and managing YOLO models with caching,
validation. and device management.

Main Classes:
  - ModelLoader: Loads and manages YOLO models with caching and validation.
  - ModelInfo: Data class for storing model metadata.
"""

from dataclasses import dataclass
from os import path
from pathlib import Path
from typing import Any, List, Optional, Dict
from ultralytics import YOLO
import torch
from src.config.logger import LoggerClass

@dataclass
class ModelInfo:
    """Stores metadata about a loaded model"""

    model_path: str
    model_name: str
    device: str
    input_size: int # e.g., 640
    num_classes: int
    class_names: Dict[int, str]

    def to_dict(self) -> Dict:
        """Convert ModelInfo to dictionary"""
        return {
            "model_path": self.model_path,
            "model_name": self.model_name,
            "device": self.device,
            "input_size": self.input_size,
            "num_classes": self.num_classes,
            "class_names": self.class_names,
        }
    
class ModelLoader:
    """
    Singleton class for loading and managing YOLO models.

    Features:
      - Model caching to avoid reloading
      - Device management (CPU/CUDA)
      - Model Validation
      - Automatic device selection

    Examples
      >>> loader = ModelLoader()
      >>> model, info = loader.load_model('yolov11n.pt', device='cuda')
    """

    _instance = None
    _cache: Dict[str, tuple] = {} # Cache: {model_path: (model, ModelInfo)}

    def __new__(cls):
        """Singleton pattern implementation"""
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize ModelLoader"""
        if not hasattr(self, '_initialized'):
            self._initialized = True
            LoggerClass.debug("ModelLoader initialized")
    
    @staticmethod
    def get_available_device() -> str:
        """
        Detect and return the best avaiable device

        Returns:
            str: Device string ('cuda' or 'cpu')
        """
        if torch.cuda.is_available():
            device = 'cuda'
            gpu_name = torch.cuda.get_device_name(0)
            LoggerClass.info(f"CUDA available: {gpu_name}")
        else:
            device = 'cpu'
            LoggerClass.info("CUDA not available, using CPU")
      
        return device
      
    @staticmethod
    def validate_model_path(model_path: Path) -> None:
        """
        Validate that model file exists and jas correct extension

        Args:
            model_path (Path): Path to model file
            
        Raises:
            FileNotFoundError: If model file doesn't exist
            ValueError: If model file has wrong extension
        """
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        valid_extensions = ['.pt', '.pth', '.onnx', '.engine']
        if model_path.suffix not in valid_extensions:
            raise ValueError(
                f"Invalid model extension: {model_path.suffix}. "
                f"Expected one of {valid_extensions}"
            )

    def load_model(
        self, 
        model_path: str, 
        device: Optional[str] = None,
        force_reload: bool = False
    ) -> tuple[YOLO, ModelInfo]:
        """
        Load YOLO model with caching and validation

        Args:
            model_path (str): Path to the model file
            device (Optional[str]): Device to load the model on ('cpu' or 'cuda').
                                    If None, auto-detects best available device.
            force_reload (bool): If True, forces reloading the model even if cached.

        Returns:
            tuple[YOLO, ModelInfo]: Loaded YOLO model and its metadata
        """
        model_path_obj = Path(model_path).resolve()
        cache_key = str(model_path)

        if not force_reload and cache_key in self._cache:
            LoggerClass.info(f"Loading model from cache: {model_path}")
            return self._cache[cache_key]
        
        self.validate_model_path(model_path_obj)

        if device is None:
            device = self.get_available_device()
        
        device = self._validate_device(device)

        try:
            LoggerClass.info(f"Loading model from: {model_path_obj.name}")
            
            model = YOLO(str(model_path_obj))

            if device != 'cpu':
                model.to(device)
            
            info = self._extract_model_info(model, model_path_obj, device)
            self._cache[cache_key] = (model, info)

            LoggerClass.info(f"Model loaded successfully")
            LoggerClass.debug(f"Model Info: {info.to_dict()}")

            return model, info
        
        except Exception as e:
            LoggerClass.error(f"Error loading model {model_path_obj}: {e}")
            raise RuntimeError(f"Failed to load model: {e}")

    def _validate_device(self, device: str) -> str:
        """
        Validate the specified device

        Args:
            device (str): Device string ('cpu' or 'cuda')

        Returns:
            str: Validated device string

        Raises:
            ValueError: If device is invalid or CUDA is not available
        """
        device = device.lower()
        if device == 'cuda' and not torch.cuda.is_available():
            raise ValueError("CUDA requested but not available")
        elif device not in ['cpu', 'cuda']:
            raise ValueError(f"Invalid device specified: {device}")
        
        return device
    
    def _extract_model_info(
            self,
            model: YOLO,
            model_path: Path,
            device: str
    ) -> ModelInfo:
        """
        Extract metadata from loaded model

        Args:
            model (YOLO): Loaded YOLO model
            model_path (Path): Path to the model file
            device (str): Device model is loaded one

        Returns:
            ModelInfo: Model metadata
        """
        model_name = model_path.stem

        try:
            num_classes = len(model.names)
            class_names = {i: name for i, name in model.names.items()}

            try:
                input_size = model.overrides.get('imgsz', 640)
                if isinstance(input_size, (list, tuple)):
                    input_size = input_size[0]
            except Exception:
                input_size = 640
        except Exception as e:
            LoggerClass.warning(f"Failed to extract model info: {e}")
            num_classes = 0
            class_names = {}
            input_size = 640
        
        return ModelInfo(
            model_path=str(model_path),
            model_name=model_name,
            device=device,
            input_size=input_size,
            num_classes=num_classes,
            class_names=class_names
        )

    def clear_cache(self) -> None:
        """Clear the model cache"""
        self._cache.clear()
        LoggerClass.info("Model cache cleared")

    def get_cached_models(self) -> List[str]:
        """
        Get information about cached models

        Returns:
            List[str]: Cached models info
        """
        return list(self._cache.keys())

    def is_model_cached(self, model_path: str) -> bool:
        """
        Check if a model is cached

        Args:
            model_path (str): Path to the model file
        
        Returns:
            bool: True if model is cached, False otherwise
        """
        model_path = str(Path(model_path).resolve())
        return model_path in self._cache
    
    @staticmethod
    def get_device_info() -> Dict[str, Any]:
        """
        Get detailed information about available devices

        Returns:
            Dict: Device information
        """
        info = {
            "cuda_available": torch.cuda.is_available(),
            "torch_version": torch.__version__,
        }

        if info['cuda_available']:
            info["cuda_devices"] = [
                {
                    "id": i,
                    "name": torch.cuda.get_device_name(i),
                    "memory_total": torch.cuda.get_device_properties(i).total_memory,
                    "memory_allocated": torch.cuda.memory_allocated(i)
                }
                for i in range(torch.cuda.device_count())
            ]

        return info


# Convenience function
def load_model(
    model_path: str,
    device: Optional[str] = None,
    force_reload: bool = False
) -> tuple[YOLO, ModelInfo]:
    """
    Convenience function to load a YOLO model
    
    Args:
        model_path (str): Path to model file
        device (Optional[str], optional): Device to use. Defaults to None (auto-detect).
        force_reload (bool, optional): Force reload model. Defaults to False.
    
    Returns:
        tuple[YOLO, ModelInfo]: Loaded model and metadata
    """
    loader = ModelLoader()
    return loader.load_model(model_path, device, force_reload)
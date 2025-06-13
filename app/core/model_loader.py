"""
Model loader untuk YOLO face dan body detection models
"""

import logging
import asyncio
from typing import Optional, Dict, Any
from pathlib import Path

import torch
from ultralytics import YOLO

from app.config import settings, ModelConfig, PerformanceConfig

logger = logging.getLogger(__name__)


class ModelLoader:
    """Model loader dan manager untuk YOLO models"""
    
    def __init__(self):
        self.face_model: Optional[YOLO] = None
        self.body_model: Optional[YOLO] = None
        self.models_loaded = False
        
        # Set torch settings
        torch.set_num_threads(PerformanceConfig.TORCH_THREADS)
        
    async def load_models(self) -> None:
        """Load semua models secara async"""
        logger.info("Loading YOLO models...")
        
        try:
            # Load models in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            
            # Load face model
            face_model_path = Path(ModelConfig.FACE_MODEL_CONFIG["path"])
            if face_model_path.exists():
                logger.info(f"Loading face model from {face_model_path}")
                self.face_model = await loop.run_in_executor(
                    None, self._load_yolo_model, str(face_model_path)
                )
                logger.info("Face model loaded successfully")
            else:
                logger.warning(f"Face model not found at {face_model_path}")
            
            # Load body model
            body_model_path = Path(ModelConfig.BODY_MODEL_CONFIG["path"])
            if body_model_path.exists():
                logger.info(f"Loading body model from {body_model_path}")
                self.body_model = await loop.run_in_executor(
                    None, self._load_yolo_model, str(body_model_path)
                )
                logger.info("Body model loaded successfully")
            else:
                logger.warning(f"Body model not found at {body_model_path}")
            
            # Warm up models
            await self._warmup_models()
            
            self.models_loaded = True
            logger.info("All models loaded and warmed up successfully")
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise
    
    def _load_yolo_model(self, model_path: str) -> YOLO:
        """Load single YOLO model"""
        try:
            model = YOLO(model_path)
            
            # Move to device
            if settings.use_gpu and torch.cuda.is_available():
                model.to(settings.device)
                logger.info(f"Model moved to {settings.device}")
            
            # Enable half precision if supported
            if PerformanceConfig.HALF_PRECISION and settings.use_gpu:
                try:
                    model.half()
                    logger.info("Half precision enabled")
                except Exception as e:
                    logger.warning(f"Failed to enable half precision: {e}")
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
            raise
    
    async def _warmup_models(self) -> None:
        """Warm up models dengan dummy inference"""
        logger.info("Warming up models...")
        
        try:
            # Create dummy image
            dummy_image = torch.zeros(
                (3, 640, 640), 
                dtype=torch.float32,
                device=settings.device
            )
            
            if PerformanceConfig.HALF_PRECISION and settings.use_gpu:
                dummy_image = dummy_image.half()
            
            # Warm up face model
            if self.face_model:
                try:
                    _ = self.face_model(dummy_image, verbose=False)
                    logger.info("Face model warmed up")
                except Exception as e:
                    logger.warning(f"Face model warmup failed: {e}")
            
            # Warm up body model
            if self.body_model:
                try:
                    _ = self.body_model(dummy_image, verbose=False)
                    logger.info("Body model warmed up")
                except Exception as e:
                    logger.warning(f"Body model warmup failed: {e}")
                    
        except Exception as e:
            logger.warning(f"Model warmup failed: {e}")
    
    def get_face_model(self) -> Optional[YOLO]:
        """Get face detection model"""
        return self.face_model
    
    def get_body_model(self) -> Optional[YOLO]:
        """Get body detection model"""
        return self.body_model
    
    def is_loaded(self) -> bool:
        """Check if models are loaded"""
        return self.models_loaded
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models"""
        return {
            "face_model": {
                "loaded": self.face_model is not None,
                "path": ModelConfig.FACE_MODEL_CONFIG["path"],
                "device": settings.device,
                "half_precision": PerformanceConfig.HALF_PRECISION and settings.use_gpu
            },
            "body_model": {
                "loaded": self.body_model is not None,
                "path": ModelConfig.BODY_MODEL_CONFIG["path"],
                "device": settings.device,
                "half_precision": PerformanceConfig.HALF_PRECISION and settings.use_gpu
            },
            "gpu_available": torch.cuda.is_available(),
            "device": settings.device
        }
    
    def cleanup(self) -> None:
        """Cleanup models dan free memory"""
        logger.info("Cleaning up models...")
        
        if self.face_model:
            del self.face_model
            self.face_model = None
        
        if self.body_model:
            del self.body_model
            self.body_model = None
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.models_loaded = False
        logger.info("Models cleaned up") 
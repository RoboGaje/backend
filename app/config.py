"""
Configuration settings untuk backend application
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings
import torch


class Settings(BaseSettings):
    """Application settings"""
    
    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    
    # Model Paths - sama seperti inference script
    face_model_path: str = "./models/YOLO12n_finetuned/weights/best.pt"
    body_model_path: str = "./models/YOLO12n_pretrained/yolo12n.pt"
    
    # Detection Settings - sama seperti inference script default
    confidence_threshold: float = 0.5
    face_confidence_threshold: float = 0.5  # Default sama seperti inference script
    body_confidence_threshold: float = 0.5  # Default sama seperti inference script
    crowd_threshold: int = 10
    max_frame_size: int = 1024
    
    # Performance Settings
    use_gpu: bool = torch.cuda.is_available()
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    tensorrt_enabled: bool = False  # Support TensorRT seperti inference script
    max_concurrent_connections: int = 50
    
    # Frame Processing
    max_fps: int = 10
    frame_buffer_size: int = 5
    
    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = "./logs/app.log"
    
    # CORS
    allowed_origins: list = ["*"]
    
    # Database (optional)
    database_url: Optional[str] = None
    
    # Redis (optional)
    redis_url: Optional[str] = None
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Create global settings instance
settings = Settings()


# Model configuration - sama seperti inference script
class ModelConfig:
    """Model-specific configuration"""
    
    FACE_MODEL_CONFIG = {
        "path": settings.face_model_path,
        "confidence": settings.face_confidence_threshold,
        "device": settings.device,
        "classes": [0, 1, 2, 3, 4],  # All face classes: dimas, fabian, people face, sendy, syahrul
        "class_names": ['dimas', 'fabian', 'people face', 'sendy', 'syahrul'],
        "input_size": (640, 640)
    }
    
    BODY_MODEL_CONFIG = {
        "path": settings.body_model_path,
        "confidence": settings.body_confidence_threshold,
        "device": settings.device,
        "classes": [0],  # Person class in COCO - sama seperti inference script
        "input_size": (640, 640)
    }


# Crowd analysis configuration - sama seperti inference script
class CrowdConfig:
    """Crowd analysis configuration"""
    
    CROWD_LEVELS = {
        "empty": {"min": 0, "max": 0, "color": "#6B7280"},
        "low": {"min": 1, "max": 2, "color": "#10B981"},
        "medium": {"min": 3, "max": 5, "color": "#F59E0B"},
        "high": {"min": 6, "max": 10, "color": "#F97316"},
        "very_high": {"min": 11, "max": float('inf'), "color": "#EF4444"}
    }
    
    ALERT_THRESHOLD = settings.crowd_threshold
    
    @staticmethod
    def get_crowd_level(count: int) -> str:
        """Get crowd level based on count"""
        for level, config in CrowdConfig.CROWD_LEVELS.items():
            if config["min"] <= count <= config["max"]:
                return level
        return "very_high"
    
    @staticmethod
    def should_alert(count: int) -> bool:
        """Check if count should trigger alert"""
        return count > CrowdConfig.ALERT_THRESHOLD


# WebSocket configuration
class WebSocketConfig:
    """WebSocket configuration"""
    
    MAX_CONNECTIONS = settings.max_concurrent_connections
    HEARTBEAT_INTERVAL = 30  # seconds
    CONNECTION_TIMEOUT = 60  # seconds
    MAX_MESSAGE_SIZE = 10 * 1024 * 1024  # 10MB


# Performance configuration
class PerformanceConfig:
    """Performance optimization configuration"""
    
    # Frame processing
    MAX_FRAME_SIZE = settings.max_frame_size
    JPEG_QUALITY = 85
    
    # Model optimization
    HALF_PRECISION = settings.use_gpu  # Use FP16 on GPU
    BATCH_SIZE = 1  # For real-time processing
    
    # Memory management
    TORCH_THREADS = 4
    OMP_NUM_THREADS = 4 
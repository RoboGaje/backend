"""
Logging configuration
"""

import logging
import logging.handlers
import os
from pathlib import Path
from app.config import settings


def setup_logging():
    """Setup logging configuration"""
    
    # Create logs directory if it doesn't exist
    if settings.log_file:
        log_dir = Path(settings.log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),  # Console output
        ]
    )
    
    # Add file handler if log file is specified
    if settings.log_file:
        file_handler = logging.handlers.RotatingFileHandler(
            settings.log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        logging.getLogger().addHandler(file_handler)
    
    # Set specific logger levels
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("fastapi").setLevel(logging.INFO)
    logging.getLogger("ultralytics").setLevel(logging.WARNING)  # Reduce YOLO logging
    
    logger = logging.getLogger(__name__)
    logger.info("Logging configured successfully") 
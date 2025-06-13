"""
Health check API endpoints
"""

import logging
from datetime import datetime
from fastapi import APIRouter, Depends
from app.core.model_loader import ModelLoader

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/")
async def health_check():
    """Basic health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "Robotika UAS - Face Detection API"
    }


@router.get("/detailed")
async def detailed_health_check():
    """Detailed health check dengan model status"""
    try:
        # This would need to be injected properly in a real app
        # For now, we'll return basic info
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "service": "Robotika UAS - Face Detection API",
            "version": "1.0.0",
            "components": {
                "api": "healthy",
                "models": "checking...",
                "websocket": "healthy"
            }
        }
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        } 
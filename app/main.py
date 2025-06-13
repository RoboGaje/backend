"""
FastAPI Backend untuk Realtime Face Detection & Crowd Analysis
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import socketio

from app.config import settings
from app.core.model_loader import ModelLoader
from app.core.websocket_manager import WebSocketManager
from app.api import health, detection
from app.services.detection_service import DetectionService
from app.utils.logging import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Global variables
model_loader: ModelLoader = None
detection_service: DetectionService = None
websocket_manager: WebSocketManager = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    global model_loader, detection_service, websocket_manager
    
    logger.info("Starting application...")
    
    try:
        # Initialize model loader
        model_loader = ModelLoader()
        await model_loader.load_models()
        
        # Initialize detection service
        detection_service = DetectionService(model_loader)
        
        # Initialize WebSocket manager
        websocket_manager = WebSocketManager()
        
        # Store in app state
        app.state.model_loader = model_loader
        app.state.detection_service = detection_service
        app.state.websocket_manager = websocket_manager
        
        logger.info("Application started successfully")
        
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        raise
    
    yield
    
    # Cleanup
    logger.info("Shutting down application...")
    if websocket_manager:
        await websocket_manager.disconnect_all()


# Create FastAPI app
app = FastAPI(
    title="Robotika UAS - Face Detection API",
    description="Realtime Face Detection & Crowd Analysis API",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000", 
        "http://10.148.0.5:3000",
        "*"  # Allow all origins for development
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Include routers
app.include_router(health.router, prefix="/health", tags=["health"])
app.include_router(detection.router, prefix="/api", tags=["detection"])


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint untuk real-time detection"""
    client_id = await app.state.websocket_manager.connect(websocket)
    logger.info(f"Client {client_id} connected")
    
    try:
        while True:
            # Receive message from client
            logger.debug(f"Waiting for message from client {client_id}...")
            data = await websocket.receive_json()
            logger.info(f"📨 Received message from client {client_id}: event={data.get('event')}")
            
            if data.get("event") == "process_frame":
                # Process frame
                frame_data = data.get("data", {})
                logger.info(f"🎬 Processing frame for client {client_id}: timestamp={frame_data.get('timestamp')}, settings={frame_data.get('settings')}")
                
                # Log frame data info
                frame = frame_data.get("frame", "")
                if frame:
                    logger.info(f"📸 Frame data received: size={len(frame)} chars, type={type(frame)}, preview={frame[:50]}...")
                else:
                    logger.warning(f"⚠️ No frame data received from client {client_id}")
                
                try:
                    # Run detection
                    logger.info(f"🔍 Starting detection processing for client {client_id}...")
                    result = await app.state.detection_service.process_frame(
                        frame_data.get("frame"),
                        frame_data.get("settings", {})
                    )
                    logger.info(f"✅ Detection completed for client {client_id}: faces={len(result.get('faces', []))}, bodies={len(result.get('bodies', []))}")
                    
                    # Send result back to client
                    await websocket.send_json({
                        "event": "detection_result",
                        "data": result
                    })
                    logger.info(f"📤 Detection result sent to client {client_id}")
                    
                except Exception as e:
                    logger.error(f"❌ Detection error for client {client_id}: {e}")
                    logger.exception("Full error traceback:")
                    await websocket.send_json({
                        "event": "error",
                        "data": {
                            "code": "PROCESSING_ERROR",
                            "message": "Failed to process frame",
                            "details": str(e)
                        }
                    })
            else:
                logger.warning(f"⚠️ Unknown event from client {client_id}: {data.get('event')}")
            
    except WebSocketDisconnect:
        logger.info(f"Client {client_id} disconnected")
    except Exception as e:
        logger.error(f"WebSocket error for client {client_id}: {e}")
        logger.exception("Full WebSocket error traceback:")
    finally:
        await app.state.websocket_manager.disconnect(client_id)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Robotika UAS - Face Detection API",
        "version": "1.0.0",
        "status": "running"
    }


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Global exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc) if settings.debug else "Something went wrong"
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="info"
    ) 
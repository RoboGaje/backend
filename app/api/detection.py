"""
Detection API endpoints
"""

import logging
import asyncio
import tempfile
import os
from typing import Dict, Any
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
import cv2
import numpy as np

from app.services.detection_service import DetectionService
from app.core.model_loader import ModelLoader

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/models/info")
async def get_models_info():
    """Get information about loaded models"""
    try:
        from app.main import app
        model_loader: ModelLoader = app.state.model_loader
        return model_loader.get_model_info()
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/statistics")
async def get_statistics():
    """Get detection service statistics"""
    try:
        from app.main import app
        detection_service: DetectionService = app.state.detection_service
        return detection_service.get_statistics()
    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/process-video")
async def process_video(
    background_tasks: BackgroundTasks,
    video_file: UploadFile = File(...),
    face_confidence: float = 0.2,
    body_confidence: float = 0.5
):
    """Process uploaded video and return annotated result"""
    
    if not video_file.content_type.startswith('video/'):
        raise HTTPException(status_code=400, detail="File must be a video")
    
    # Create temporary files
    temp_input = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    
    try:
        # Save uploaded video
        content = await video_file.read()
        temp_input.write(content)
        temp_input.close()
        
        logger.info(f"Processing video: {video_file.filename}")
        
        # Get detection service
        from app.main import app
        detection_service: DetectionService = app.state.detection_service
        
        # Process video
        result = await process_video_file(
            detection_service,
            temp_input.name,
            temp_output.name,
            face_confidence,
            body_confidence
        )
        
        # Schedule cleanup
        background_tasks.add_task(cleanup_temp_files, temp_input.name, temp_output.name)
        
        logger.info(f"Video processing completed: {result['stats']}")
        
        # Return processed video
        return FileResponse(
            temp_output.name,
            media_type='video/mp4',
            filename=f"processed_{video_file.filename}",
            headers={
                "X-Processing-Stats": str(result['stats'])
            }
        )
        
    except Exception as e:
        # Cleanup on error
        cleanup_temp_files(temp_input.name, temp_output.name)
        logger.error(f"Error processing video: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def process_video_file(
    detection_service: DetectionService,
    input_path: str,
    output_path: str,
    face_confidence: float,
    body_confidence: float
) -> Dict[str, Any]:
    """Process video file frame by frame"""
    
    cap = cv2.VideoCapture(input_path)
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    stats = {
        'total_frames': total_frames,
        'processed_frames': 0,
        'total_faces': 0,
        'total_bodies': 0,
        'processing_time': 0
    }
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process every 5th frame for performance
            if frame_count % 5 == 0:
                # Convert frame to base64 for detection service
                _, buffer = cv2.imencode('.jpg', frame)
                import base64
                frame_base64 = base64.b64encode(buffer).decode('utf-8')
                
                # Run detection with silent_mode=True for batch processing
                settings = {
                    'face_confidence_threshold': face_confidence,
                    'body_confidence_threshold': body_confidence,
                    'crowd_threshold': 10
                }
                
                try:
                    result = await detection_service.process_frame(frame_base64, settings, silent_mode=True)
                    
                    # Draw detections on frame
                    annotated_frame = draw_detections_on_frame(
                        frame, 
                        result['faces'], 
                        result['bodies']
                    )
                    
                    # Update stats
                    stats['total_faces'] += len(result['faces'])
                    stats['total_bodies'] += len(result['bodies'])
                    stats['processing_time'] += result['processing_time']
                    
                except Exception:
                    # Silent error handling - just use original frame
                    annotated_frame = frame
            else:
                annotated_frame = frame
            
            # Write frame
            out.write(annotated_frame)
            stats['processed_frames'] = frame_count
    
    finally:
        cap.release()
        out.release()
    
    return {'stats': stats}


def draw_detections_on_frame(frame: np.ndarray, faces: list, bodies: list) -> np.ndarray:
    """Draw detection boxes on frame"""
    annotated = frame.copy()
    
    # Draw faces (blue)
    for face in faces:
        x1, y1, x2, y2 = face['bbox']
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        # Label
        label = f"{face.get('class_name', 'face')}: {face['confidence']:.2f}"
        cv2.putText(annotated, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    # Draw bodies (green)
    for body in bodies:
        x1, y1, x2, y2 = body['bbox']
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Label
        label = f"body: {body['confidence']:.2f}"
        cv2.putText(annotated, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return annotated


def cleanup_temp_files(*file_paths):
    """Clean up temporary files"""
    for file_path in file_paths:
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
        except Exception:
            pass  # Silent cleanup 
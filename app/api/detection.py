"""
Detection API endpoints
"""

import logging
import asyncio
import tempfile
import os
from typing import Dict, Any
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File, BackgroundTasks, Query
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
    face_confidence: float = 0.5,
    body_confidence: float = 0.5,
    process_all_frames: bool = Query(True, description="Process every frame (True) or skip frames for performance (False)")
):
    """Process uploaded video and return annotated result - memproses setiap frame atau skip untuk performa"""
    
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
        
        logger.info(f"Processing video: {video_file.filename}, process_all_frames={process_all_frames}")
        
        # Get detection service
        from app.main import app
        detection_service: DetectionService = app.state.detection_service
        
        # Process video
        result = await process_video_file(
            detection_service,
            temp_input.name,
            temp_output.name,
            face_confidence,
            body_confidence,
            process_all_frames
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
    body_confidence: float,
    process_all_frames: bool = True
) -> Dict[str, Any]:
    """Process video file frame by frame - sama seperti inference script"""
    
    cap = cv2.VideoCapture(input_path)
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    logger.info(f"🎬 Video properties: {total_frames} frames, {fps} FPS, {width}x{height}")
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    stats = {
        'total_frames': total_frames,
        'processed_frames': 0,
        'detection_processed_frames': 0,  # Frames yang benar-benar dideteksi
        'total_faces': 0,
        'total_bodies': 0,
        'processing_time': 0,
        'avg_faces_per_frame': 0,
        'avg_bodies_per_frame': 0,
        'max_faces_in_frame': 0,
        'max_bodies_in_frame': 0
    }
    
    frame_count = 0
    detection_count = 0
    
    try:
        logger.info(f"🚀 Starting video processing - process_all_frames={process_all_frames}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Decide whether to process this frame
            should_process = process_all_frames or (frame_count % 5 == 0)
            
            if should_process:
                detection_count += 1
                
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
                    
                    # Draw detections on frame - sama seperti inference script
                    annotated_frame = draw_detections_on_frame_like_inference(
                        frame, 
                        result['faces'], 
                        result['bodies'],
                        result.get('density_info', {})
                    )
                    
                    # Update stats - sama seperti inference script
                    num_faces = len(result['faces'])
                    num_bodies = len(result['bodies'])
                    
                    stats['total_faces'] += num_faces
                    stats['total_bodies'] += num_bodies
                    stats['processing_time'] += result['processing_time']
                    stats['max_faces_in_frame'] = max(stats['max_faces_in_frame'], num_faces)
                    stats['max_bodies_in_frame'] = max(stats['max_bodies_in_frame'], num_bodies)
                    
                    if frame_count % 30 == 0:  # Progress update setiap 30 frame
                        progress = (frame_count / total_frames) * 100
                        logger.info(f"📊 Progress: {progress:.1f}% | Frame {frame_count}/{total_frames} | Faces: {num_faces}, Bodies: {num_bodies}")
                    
                except Exception as e:
                    # Silent error handling - just use original frame
                    logger.warning(f"Detection error on frame {frame_count}: {e}")
                    annotated_frame = frame
            else:
                # Use original frame without detection
                annotated_frame = frame
            
            # Write frame
            out.write(annotated_frame)
            stats['processed_frames'] = frame_count
            stats['detection_processed_frames'] = detection_count
    
    finally:
        cap.release()
        out.release()
        
        # Calculate final averages - sama seperti inference script
        if stats['detection_processed_frames'] > 0:
            stats['avg_faces_per_frame'] = round(stats['total_faces'] / stats['detection_processed_frames'], 2)
            stats['avg_bodies_per_frame'] = round(stats['total_bodies'] / stats['detection_processed_frames'], 2)
        
        logger.info(f"✅ Video processing completed:")
        logger.info(f"   📊 Total frames: {stats['processed_frames']}")
        logger.info(f"   🔍 Detection processed: {stats['detection_processed_frames']}")
        logger.info(f"   👤 Total faces: {stats['total_faces']}")
        logger.info(f"   🚶 Total bodies: {stats['total_bodies']}")
        logger.info(f"   ⏱️ Processing time: {stats['processing_time']:.2f}s")
    
    return {'stats': stats}


def draw_detections_on_frame_like_inference(frame: np.ndarray, faces: list, bodies: list, density_info: dict = None) -> np.ndarray:
    """Draw detection boxes on frame - sama seperti inference script"""
    annotated = frame.copy()
    
    # Draw faces (blue) - sama seperti inference script
    for face in faces:
        x1, y1, x2, y2 = face['bbox']
        conf = face['confidence']
        class_name = face.get('class', 'face')
        
        # Face bounding box (Blue)
        color = (255, 0, 0)  # BGR format
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        
        # Face label
        label = f"Face: {class_name} ({conf:.2f})"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(annotated, (x1, y1-label_size[1]-10), (x1+label_size[0], y1), color, -1)
        cv2.putText(annotated, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # Draw bodies (green) - sama seperti inference script
    for body in bodies:
        x1, y1, x2, y2 = body['bbox']
        conf = body['confidence']
        
        # Body bounding box (Green)
        color = (0, 255, 0)  # BGR format
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        
        # Body label
        label = f"Person ({conf:.2f})"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(annotated, (x1, y2), (x1+label_size[0], y2+label_size[1]+10), color, -1)
        cv2.putText(annotated, label, (x1, y2+label_size[1]+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # Draw statistics panel - sama seperti inference script
    if density_info:
        draw_stats_panel_like_inference(annotated, faces, bodies, density_info)
    
    return annotated


def draw_stats_panel_like_inference(image: np.ndarray, faces: list, bodies: list, density_info: dict):
    """Draw statistics panel on image - sama seperti inference script"""
    h, w = image.shape[:2]
    panel_height = 140
    panel_width = 320
    
    # Create semi-transparent panel
    overlay = image.copy()
    cv2.rectangle(overlay, (10, 10), (panel_width, panel_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
    
    # Add text
    y_offset = 30
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    color = (255, 255, 255)
    thickness = 1
    
    texts = [
        f"Faces: {len(faces)} | Bodies: {len(bodies)}",
        f"Crowd Level: {density_info.get('crowd_level', 'Unknown')} ({density_info.get('crowd_intensity', 0)}%)",
        f"People Density: {density_info.get('people_density', 0):.1f}/Mpx",
        f"Area Coverage: {density_info.get('area_coverage', 0):.1f}%",
        f"Face-to-Body Ratio: {density_info.get('face_body_ratio', 0):.2f}"
    ]
    
    for i, text in enumerate(texts):
        cv2.putText(image, text, (20, y_offset + i*18), font, font_scale, color, thickness)


def cleanup_temp_files(*file_paths):
    """Clean up temporary files"""
    for file_path in file_paths:
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
        except Exception as e:
            logger.warning(f"Failed to cleanup {file_path}: {e}") 
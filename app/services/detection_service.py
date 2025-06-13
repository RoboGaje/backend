"""
Detection service untuk memproses frame dengan YOLO models
Dimodifikasi agar sama persis dengan face_body_detection_inference.py
"""

import logging
import asyncio
import time
import base64
from typing import Dict, List, Any, Optional
from io import BytesIO

import cv2
import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO

from app.core.model_loader import ModelLoader
from app.config import settings, CrowdConfig, PerformanceConfig, ModelConfig
from app.services.crowd_analysis import CrowdAnalysisService
from app.utils.image_utils import ImageProcessor

logger = logging.getLogger(__name__)


class DetectionService:
    """Service untuk menjalankan detection pada frame - sama seperti inference script"""
    
    def __init__(self, model_loader: ModelLoader):
        self.model_loader = model_loader
        self.crowd_analyzer = CrowdAnalysisService()
        self.image_processor = ImageProcessor()
        
        # Class IDs - sama seperti inference script
        self.person_class_id = 0  # Index untuk 'person' di COCO
        
        # Enhanced statistics tracking - sama seperti inference script
        self.stats = {
            'total_frames': 0,
            'total_faces': 0,
            'total_bodies': 0,
            'avg_faces_per_frame': 0,
            'avg_bodies_per_frame': 0,
            'max_faces_in_frame': 0,
            'max_bodies_in_frame': 0,
            'total_processing_time': 0.0,
            'avg_processing_time': 0.0,
            'density_analysis': [],
            'face_class_distribution': {},
            'confidence_stats': {
                'face_confidences': [],
                'body_confidences': []
            }
        }
        
    async def process_frame(self, frame_base64: str, settings_override: Dict = None, silent_mode: bool = False) -> Dict[str, Any]:
        """Process single frame dan return detection results - sama seperti inference script"""
        start_time = time.time()
        
        try:
            if not silent_mode:
                logger.info(f"🎬 Starting frame processing...")
            
            # Decode frame
            if not silent_mode:
                logger.info(f"📸 Decoding frame...")
            frame = await self._decode_frame(frame_base64)
            if frame is None:
                raise ValueError("Failed to decode frame")
            
            if not silent_mode:
                logger.info(f"✅ Frame decoded successfully: shape={frame.shape}")
            
            # Get detection settings
            detection_settings = self._get_detection_settings(settings_override)
            if not silent_mode:
                logger.info(f"⚙️ Detection settings: {detection_settings}")
            
            # Run detections - sama seperti inference script
            if not silent_mode:
                logger.info(f"👤 Starting face detection...")
            face_results = await self._detect_faces_like_inference(frame, detection_settings, silent_mode)
            if not silent_mode:
                logger.info(f"✅ Face detection completed: {len(face_results)} faces found")
            
            if not silent_mode:
                logger.info(f"🚶 Starting body detection...")
            body_results = await self._detect_bodies_like_inference(frame, detection_settings, silent_mode)
            if not silent_mode:
                logger.info(f"✅ Body detection completed: {len(body_results)} bodies found")
            
            # Enhanced density analysis - sama seperti inference script
            if not silent_mode:
                logger.info(f"📊 Analyzing crowd density...")
            density_info = self._calculate_density_like_inference(face_results, body_results, frame.shape)
            
            # Legacy crowd analysis for compatibility
            crowd_analysis = self.crowd_analyzer.analyze_crowd(
                len(face_results), 
                len(body_results)
            )
            
            # Merge enhanced density info with crowd analysis
            crowd_analysis.update(density_info)
            
            if not silent_mode:
                logger.info(f"✅ Crowd analysis completed: {crowd_analysis}")
            
            # Generate alerts if needed
            alerts = self._generate_alerts(crowd_analysis)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Update enhanced statistics - sama seperti inference script
            self._update_stats_like_inference(face_results, body_results, density_info, processing_time)
            
            # Prepare result
            result = {
                "faces": face_results,
                "bodies": body_results,
                "crowd_analysis": crowd_analysis,
                "density_info": density_info,
                "alerts": alerts,
                "processing_time": round(processing_time, 3),
                "timestamp": int(time.time() * 1000),
                "frame_info": {
                    "width": frame.shape[1],
                    "height": frame.shape[0],
                    "channels": frame.shape[2] if len(frame.shape) > 2 else 1
                },
                "statistics": self._get_current_stats()
            }
            
            if not silent_mode:
                logger.info(f"🎉 Frame processed successfully in {processing_time:.3f}s - Faces: {len(face_results)}, Bodies: {len(body_results)}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error processing frame: {e}")
            if not silent_mode:
                logger.exception("Full error traceback:")
            raise
    
    async def _detect_faces_like_inference(self, frame: np.ndarray, detection_settings: Dict, silent_mode: bool = False) -> List[Dict]:
        """Detect faces using fine-tuned YOLO model - sama seperti inference script"""
        if not silent_mode:
            logger.info(f"👤 Getting face model...")
        face_model = self.model_loader.get_face_model()
        if not face_model:
            if not silent_mode:
                logger.warning("❌ Face model not available")
            return []
        
        try:
            # Run inference - sama persis seperti inference script
            if not silent_mode:
                logger.info(f"🔍 Running face inference with confidence={detection_settings['face_confidence_threshold']}")
            
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None, 
                self._run_face_inference_like_script, 
                face_model, 
                frame, 
                detection_settings["face_confidence_threshold"],
                silent_mode
            )
            
            # Process results - sama seperti inference script
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        conf = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        class_name = result.names[class_id]
                        
                        detections.append({
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': float(conf),
                            'class': class_name,
                            'class_id': class_id,
                            'type': 'face',
                            'class_name': class_name
                        })
            
            if not silent_mode:
                logger.info(f"✅ Face detection final result: {len(detections)} faces")
            return detections
            
        except Exception as e:
            logger.error(f"❌ Face detection error: {e}")
            if not silent_mode:
                logger.exception("Full face detection error traceback:")
            return []
    
    async def _detect_bodies_like_inference(self, frame: np.ndarray, detection_settings: Dict, silent_mode: bool = False) -> List[Dict]:
        """Detect bodies using pre-trained YOLO model - sama seperti inference script"""
        if not silent_mode:
            logger.info(f"🚶 Getting body model...")
        body_model = self.model_loader.get_body_model()
        if not body_model:
            if not silent_mode:
                logger.warning("❌ Body model not available")
            return []
        
        try:
            # Run inference - sama persis seperti inference script
            if not silent_mode:
                logger.info(f"🔍 Running body inference with confidence={detection_settings['body_confidence_threshold']}")
            
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None, 
                self._run_body_inference_like_script, 
                body_model, 
                frame, 
                detection_settings["body_confidence_threshold"],
                silent_mode
            )
            
            # Process results - sama seperti inference script
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        conf = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        if class_id == self.person_class_id:
                            detections.append({
                                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                'confidence': float(conf),
                                'class': 'person',
                                'class_id': class_id,
                                'type': 'body',
                                'class_name': 'person'
                            })
            
            if not silent_mode:
                logger.info(f"✅ Body detection final result: {len(detections)} bodies")
            return detections
            
        except Exception as e:
            logger.error(f"❌ Body detection error: {e}")
            if not silent_mode:
                logger.exception("Full body detection error traceback:")
            return []
    
    def _run_face_inference_like_script(self, model, frame: np.ndarray, confidence_threshold: float, silent_mode: bool = False):
        """Run face inference exactly like inference script"""
        try:
            if not silent_mode:
                logger.info(f"🤖 Running face YOLO inference: frame_shape={frame.shape}, conf={confidence_threshold}")
            
            # Sama persis seperti inference script
            results = model(frame, conf=confidence_threshold, verbose=False)
            
            if not silent_mode:
                logger.info(f"✅ Face YOLO inference completed: {len(results) if results else 0} results")
            
            return results
        except Exception as e:
            logger.error(f"❌ Face YOLO inference error: {e}")
            if not silent_mode:
                logger.exception("Full face YOLO inference error traceback:")
            return []
    
    def _run_body_inference_like_script(self, model, frame: np.ndarray, confidence_threshold: float, silent_mode: bool = False):
        """Run body inference exactly like inference script"""
        try:
            if not silent_mode:
                logger.info(f"🤖 Running body YOLO inference: frame_shape={frame.shape}, conf={confidence_threshold}")
            
            # Sama persis seperti inference script - dengan classes=[0] untuk person only
            results = model(frame, conf=confidence_threshold, verbose=False, classes=[0])
            
            if not silent_mode:
                logger.info(f"✅ Body YOLO inference completed: {len(results) if results else 0} results")
            
            return results
        except Exception as e:
            logger.error(f"❌ Body YOLO inference error: {e}")
            if not silent_mode:
                logger.exception("Full body YOLO inference error traceback:")
            return []
    
    def _calculate_density_like_inference(self, faces: List[Dict], bodies: List[Dict], image_shape: tuple) -> Dict[str, Any]:
        """Calculate crowd density analysis exactly like inference script"""
        h, w = image_shape[:2]
        total_area = h * w
        
        # Jumlah orang (berdasarkan body detection) - sama seperti inference script
        num_people = len(bodies)
        
        # Simple density: jumlah orang per area - sama seperti inference script
        people_density = num_people / (total_area / 1000000)  # orang per juta pixel
        
        # Crowd level berdasarkan jumlah orang - sama seperti inference script
        if num_people == 0:
            crowd_level = "Empty"
            crowd_intensity = 0
        elif num_people <= 2:
            crowd_level = "Low"
            crowd_intensity = 25
        elif num_people <= 5:
            crowd_level = "Medium"
            crowd_intensity = 50
        elif num_people <= 10:
            crowd_level = "High"
            crowd_intensity = 75
        else:
            crowd_level = "Very High"
            crowd_intensity = 100
        
        # Face-to-body ratio - sama seperti inference script
        face_body_ratio = len(faces) / num_people if num_people > 0 else 0
        
        # Area coverage - sama seperti inference script
        area_coverage = self._calculate_area_coverage_like_inference(bodies, total_area)
        
        # Face class distribution - sama seperti inference script
        face_class_dist = {}
        for face in faces:
            class_name = face.get('class', 'unknown')
            face_class_dist[class_name] = face_class_dist.get(class_name, 0) + 1
        
        return {
            'people_count': num_people,
            'face_count': len(faces),
            'people_density': people_density,
            'crowd_level': crowd_level,
            'crowd_intensity': crowd_intensity,
            'face_body_ratio': face_body_ratio,
            'area_coverage': area_coverage,
            'face_class_distribution': face_class_dist
        }
    
    def _calculate_area_coverage_like_inference(self, bodies: List[Dict], total_area: int) -> float:
        """Calculate percentage of area covered by people - sama seperti inference script"""
        if len(bodies) == 0:
            return 0.0
        
        covered_area = 0
        for body in bodies:
            x1, y1, x2, y2 = body['bbox']
            covered_area += (x2 - x1) * (y2 - y1)
        
        return (covered_area / total_area) * 100
    
    def _update_stats_like_inference(self, faces: List[Dict], bodies: List[Dict], density_info: Dict, processing_time: float):
        """Update running statistics - sama seperti inference script"""
        self.stats['total_frames'] += 1
        self.stats['total_faces'] += len(faces)
        self.stats['total_bodies'] += len(bodies)
        
        self.stats['max_faces_in_frame'] = max(self.stats['max_faces_in_frame'], len(faces))
        self.stats['max_bodies_in_frame'] = max(self.stats['max_bodies_in_frame'], len(bodies))
        
        self.stats['density_analysis'].append(density_info)
        self.stats['total_processing_time'] += processing_time
        
        # Update averages
        if self.stats['total_frames'] > 0:
            self.stats['avg_faces_per_frame'] = self.stats['total_faces'] / self.stats['total_frames']
            self.stats['avg_bodies_per_frame'] = self.stats['total_bodies'] / self.stats['total_frames']
            self.stats['avg_processing_time'] = self.stats['total_processing_time'] / self.stats['total_frames']
        
        # Face class distribution
        for face in faces:
            class_name = face.get('class', 'unknown')
            self.stats['face_class_distribution'][class_name] = self.stats['face_class_distribution'].get(class_name, 0) + 1
        
        # Confidence statistics
        for face in faces:
            self.stats['confidence_stats']['face_confidences'].append(face['confidence'])
        for body in bodies:
            self.stats['confidence_stats']['body_confidences'].append(body['confidence'])
    
    def _get_current_stats(self) -> Dict[str, Any]:
        """Get current statistics - sama seperti inference script"""
        face_confs = self.stats['confidence_stats']['face_confidences']
        body_confs = self.stats['confidence_stats']['body_confidences']
        
        # Calculate confidence stats
        face_conf_stats = None
        if face_confs:
            face_conf_stats = {
                'avg': round(sum(face_confs) / len(face_confs), 3),
                'min': round(min(face_confs), 3),
                'max': round(max(face_confs), 3)
            }
        
        body_conf_stats = None
        if body_confs:
            body_conf_stats = {
                'avg': round(sum(body_confs) / len(body_confs), 3),
                'min': round(min(body_confs), 3),
                'max': round(max(body_confs), 3)
            }
        
        # Crowd level distribution
        crowd_levels = [d['crowd_level'] for d in self.stats['density_analysis']]
        from collections import Counter
        crowd_dist = Counter(crowd_levels)
        
        return {
            'total_frames': self.stats['total_frames'],
            'total_faces': self.stats['total_faces'],
            'total_bodies': self.stats['total_bodies'],
            'avg_faces_per_frame': round(self.stats['avg_faces_per_frame'], 2),
            'avg_bodies_per_frame': round(self.stats['avg_bodies_per_frame'], 2),
            'max_faces_in_frame': self.stats['max_faces_in_frame'],
            'max_bodies_in_frame': self.stats['max_bodies_in_frame'],
            'avg_processing_time': round(self.stats['avg_processing_time'], 3),
            'face_class_distribution': self.stats['face_class_distribution'].copy(),
            'face_confidence_stats': face_conf_stats,
            'body_confidence_stats': body_conf_stats,
            'recent_crowd_level_distribution': dict(crowd_dist)
        }

    async def _decode_frame(self, frame_base64: str) -> Optional[np.ndarray]:
        """Decode base64 frame to numpy array"""
        try:
            # Remove data URL prefix if present
            if frame_base64.startswith('data:'):
                frame_base64 = frame_base64.split(',')[1]
            
            # Decode base64
            frame_bytes = base64.b64decode(frame_base64)
            
            # Convert to numpy array
            nparr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                logger.error("Failed to decode frame from base64")
                return None
            
            return frame
            
        except Exception as e:
            logger.error(f"Error decoding frame: {e}")
            return None
    
    def _get_detection_settings(self, settings_override: Dict = None) -> Dict:
        """Get detection settings with optional override"""
        default_settings = {
            "face_confidence_threshold": settings.face_confidence_threshold,
            "body_confidence_threshold": settings.body_confidence_threshold,
            "crowd_threshold": settings.crowd_threshold
        }
        
        if settings_override:
            # Map frontend settings to backend settings
            if "confidence_threshold" in settings_override:
                default_settings["face_confidence_threshold"] = settings_override["confidence_threshold"]
                default_settings["body_confidence_threshold"] = settings_override["confidence_threshold"]
            
            if "face_confidence_threshold" in settings_override:
                default_settings["face_confidence_threshold"] = settings_override["face_confidence_threshold"]
            
            if "body_confidence_threshold" in settings_override:
                default_settings["body_confidence_threshold"] = settings_override["body_confidence_threshold"]
            
            if "crowd_threshold" in settings_override:
                default_settings["crowd_threshold"] = settings_override["crowd_threshold"]
        
        return default_settings

    def _generate_alerts(self, crowd_analysis: Dict) -> List[Dict]:
        """Generate alerts based on crowd analysis"""
        alerts = []
        
        # Check crowd threshold
        total_people = crowd_analysis.get("total_people", 0)
        if total_people > settings.crowd_threshold:
            alerts.append({
                "type": "crowd_alert",
                "level": "warning",
                "message": f"High crowd density detected: {total_people} people",
                "crowd_level": crowd_analysis.get("crowd_level", "unknown"),
                "timestamp": int(time.time() * 1000)
            })
        
        return alerts
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current statistics"""
        return self._get_current_stats()
    
    def reset_statistics(self) -> None:
        """Reset all statistics"""
        self.stats = {
            'total_frames': 0,
            'total_faces': 0,
            'total_bodies': 0,
            'avg_faces_per_frame': 0,
            'avg_bodies_per_frame': 0,
            'max_faces_in_frame': 0,
            'max_bodies_in_frame': 0,
            'total_processing_time': 0.0,
            'avg_processing_time': 0.0,
            'density_analysis': [],
            'face_class_distribution': {},
            'confidence_stats': {
                'face_confidences': [],
                'body_confidences': []
            }
        }
        logger.info("Statistics reset")
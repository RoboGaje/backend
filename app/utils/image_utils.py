"""
Image processing utilities
"""

import logging
import cv2
import numpy as np
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


class ImageProcessor:
    """Utility class untuk image processing"""
    
    def __init__(self):
        pass
    
    def resize_frame(self, frame: np.ndarray, max_size: int) -> np.ndarray:
        """Resize frame jika terlalu besar, maintain aspect ratio"""
        try:
            height, width = frame.shape[:2]
            
            # Check if resize needed
            if max(height, width) <= max_size:
                return frame
            
            # Calculate new dimensions
            if width > height:
                new_width = max_size
                new_height = int(height * (max_size / width))
            else:
                new_height = max_size
                new_width = int(width * (max_size / height))
            
            # Resize frame
            resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
            logger.debug(f"Frame resized from {width}x{height} to {new_width}x{new_height}")
            return resized_frame
            
        except Exception as e:
            logger.error(f"Error resizing frame: {e}")
            return frame
    
    def normalize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Normalize frame untuk model input"""
        try:
            # Convert to float32 and normalize to [0, 1]
            normalized = frame.astype(np.float32) / 255.0
            return normalized
        except Exception as e:
            logger.error(f"Error normalizing frame: {e}")
            return frame
    
    def preprocess_for_yolo(self, frame: np.ndarray, target_size: Tuple[int, int] = (640, 640)) -> np.ndarray:
        """Preprocess frame untuk YOLO input"""
        try:
            # Resize to target size
            resized = cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)
            
            # Convert BGR to RGB if needed
            if len(resized.shape) == 3 and resized.shape[2] == 3:
                # Assume input is RGB, YOLO expects RGB
                pass
            
            return resized
            
        except Exception as e:
            logger.error(f"Error preprocessing frame for YOLO: {e}")
            return frame
    
    def enhance_frame(self, frame: np.ndarray) -> np.ndarray:
        """Enhance frame quality untuk better detection"""
        try:
            # Convert to LAB color space
            lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            
            # Convert back to RGB
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            
            return enhanced
            
        except Exception as e:
            logger.warning(f"Error enhancing frame: {e}")
            return frame
    
    def crop_region(self, frame: np.ndarray, bbox: list, padding: int = 10) -> Optional[np.ndarray]:
        """Crop region dari frame berdasarkan bounding box"""
        try:
            x1, y1, x2, y2 = bbox
            height, width = frame.shape[:2]
            
            # Add padding
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(width, x2 + padding)
            y2 = min(height, y2 + padding)
            
            # Crop region
            cropped = frame[y1:y2, x1:x2]
            
            return cropped if cropped.size > 0 else None
            
        except Exception as e:
            logger.error(f"Error cropping region: {e}")
            return None
    
    def draw_bounding_boxes(self, frame: np.ndarray, detections: list, colors: dict = None) -> np.ndarray:
        """Draw bounding boxes pada frame"""
        try:
            result_frame = frame.copy()
            
            # Default colors
            if colors is None:
                colors = {
                    "face": (0, 0, 255),  # Red
                    "body": (0, 255, 0)   # Green
                }
            
            for detection in detections:
                bbox = detection.get("bbox", [])
                detection_type = detection.get("type", "unknown")
                confidence = detection.get("confidence", 0.0)
                
                if len(bbox) == 4:
                    x1, y1, x2, y2 = bbox
                    color = colors.get(detection_type, (255, 255, 255))
                    
                    # Draw rectangle
                    cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw label
                    label = f"{detection_type}: {confidence:.2f}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                    
                    # Background for text
                    cv2.rectangle(
                        result_frame, 
                        (x1, y1 - label_size[1] - 10), 
                        (x1 + label_size[0], y1), 
                        color, 
                        -1
                    )
                    
                    # Text
                    cv2.putText(
                        result_frame, 
                        label, 
                        (x1, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, 
                        (255, 255, 255), 
                        1
                    )
            
            return result_frame
            
        except Exception as e:
            logger.error(f"Error drawing bounding boxes: {e}")
            return frame
    
    def calculate_iou(self, box1: list, box2: list) -> float:
        """Calculate Intersection over Union (IoU) antara dua bounding boxes"""
        try:
            x1_1, y1_1, x2_1, y2_1 = box1
            x1_2, y1_2, x2_2, y2_2 = box2
            
            # Calculate intersection area
            x1_i = max(x1_1, x1_2)
            y1_i = max(y1_1, y1_2)
            x2_i = min(x2_1, x2_2)
            y2_i = min(y2_1, y2_2)
            
            if x2_i <= x1_i or y2_i <= y1_i:
                return 0.0
            
            intersection_area = (x2_i - x1_i) * (y2_i - y1_i)
            
            # Calculate union area
            area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
            area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
            union_area = area1 + area2 - intersection_area
            
            # Calculate IoU
            iou = intersection_area / union_area if union_area > 0 else 0.0
            
            return iou
            
        except Exception as e:
            logger.error(f"Error calculating IoU: {e}")
            return 0.0 
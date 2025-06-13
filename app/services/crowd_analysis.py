"""
Crowd analysis service untuk menganalisis kepadatan
"""

import logging
from typing import Dict, Any
from app.config import CrowdConfig

logger = logging.getLogger(__name__)


class CrowdAnalysisService:
    """Service untuk analisis crowd density"""
    
    def __init__(self):
        self.crowd_config = CrowdConfig()
    
    def analyze_crowd(self, face_count: int, body_count: int) -> Dict[str, Any]:
        """Analyze crowd berdasarkan face dan body count"""
        
        # Use maximum count as total people estimate
        total_people = max(face_count, body_count)
        
        # Get crowd level
        crowd_level = self.crowd_config.get_crowd_level(total_people)
        
        # Calculate density score (0.0 to 1.0)
        density_score = self._calculate_density_score(total_people)
        
        # Get crowd level info
        level_info = self.crowd_config.CROWD_LEVELS.get(crowd_level, {})
        
        # Calculate face-to-body ratio
        face_to_body_ratio = self._calculate_ratio(face_count, body_count)
        
        return {
            "face_count": face_count,
            "body_count": body_count,
            "total_people": total_people,
            "crowd_level": crowd_level,
            "density_score": round(density_score, 3),
            "color": level_info.get("color", "#6B7280"),
            "face_to_body_ratio": round(face_to_body_ratio, 3),
            "should_alert": self.crowd_config.should_alert(total_people),
            "level_info": {
                "min": level_info.get("min", 0),
                "max": level_info.get("max", 0),
                "description": self._get_level_description(crowd_level)
            }
        }
    
    def _calculate_density_score(self, total_people: int) -> float:
        """Calculate density score dari 0.0 (empty) to 1.0 (very crowded)"""
        if total_people == 0:
            return 0.0
        elif total_people <= 2:
            return 0.2
        elif total_people <= 5:
            return 0.4
        elif total_people <= 10:
            return 0.7
        else:
            # Cap at 1.0 for very high density
            return min(1.0, 0.7 + (total_people - 10) * 0.03)
    
    def _calculate_ratio(self, face_count: int, body_count: int) -> float:
        """Calculate face-to-body ratio"""
        if body_count == 0:
            return 1.0 if face_count > 0 else 0.0
        return face_count / body_count
    
    def _get_level_description(self, crowd_level: str) -> str:
        """Get description untuk crowd level"""
        descriptions = {
            "empty": "Tidak ada orang terdeteksi",
            "low": "Kepadatan rendah - beberapa orang",
            "medium": "Kepadatan sedang - cukup ramai",
            "high": "Kepadatan tinggi - sangat ramai",
            "very_high": "Kepadatan sangat tinggi - overcrowded"
        }
        return descriptions.get(crowd_level, "Unknown crowd level")
    
    def get_crowd_statistics(self, analysis_history: list) -> Dict[str, Any]:
        """Get statistics dari history analysis"""
        if not analysis_history:
            return {
                "total_analyses": 0,
                "average_people": 0,
                "max_people": 0,
                "most_common_level": "empty"
            }
        
        total_people_counts = [a.get("total_people", 0) for a in analysis_history]
        crowd_levels = [a.get("crowd_level", "empty") for a in analysis_history]
        
        # Count level occurrences
        level_counts = {}
        for level in crowd_levels:
            level_counts[level] = level_counts.get(level, 0) + 1
        
        # Find most common level
        most_common_level = max(level_counts, key=level_counts.get) if level_counts else "empty"
        
        return {
            "total_analyses": len(analysis_history),
            "average_people": round(sum(total_people_counts) / len(total_people_counts), 1),
            "max_people": max(total_people_counts),
            "min_people": min(total_people_counts),
            "most_common_level": most_common_level,
            "level_distribution": level_counts
        } 
"""
Health Score Calculator - Computes overall health score based on predictions
"""
import numpy as np
from typing import Dict, List

class HealthScoreCalculator:
    def __init__(self):
        self.disease_weights = {
            'diabetes': 0.15,
            'heart': 0.20,
            'parkinsons': 0.10,
            'breast_cancer': 0.15,
            'stroke': 0.15,
            'kidney': 0.15,
            'hypertension': 0.10
        }
    
    def calculate_score(self, predictions: Dict[str, Dict]) -> Dict[str, any]:
        """
        Calculate overall health score based on all predictions
        
        Args:
            predictions: Dictionary of predictions for each disease
        
        Returns:
            Health score and category
        """
        total_risk = 0
        total_weight = 0
        
        for disease, result in predictions.items():
            if 'error' not in result and result.get('probability'):
                weight = self.disease_weights.get(disease, 0.2)
                # If prediction is positive, risk is probability, else inverse
                if result['prediction'] == 1:
                    risk = result['probability']
                else:
                    risk = 100 - result['probability']
                
                total_risk += risk * weight
                total_weight += weight
        
        if total_weight > 0:
            health_score = 100 - (total_risk / total_weight)
        else:
            health_score = 75  # Default score
        
        # Determine health category
        if health_score >= 80:
            category = "Excellent"
            color = "#28a745"
            icon = "🌟"
        elif health_score >= 60:
            category = "Good"
            color = "#17a2b8"
            icon = "👍"
        elif health_score >= 40:
            category = "Fair"
            color = "#ffc107"
            icon = "⚠️"
        else:
            category = "Needs Attention"
            color = "#dc3545"
            icon = "❗"
        
        return {
            'score': round(health_score, 1),
            'category': category,
            'color': color,
            'icon': icon,
            'risk_level': self._get_risk_level(health_score)
        }
    
    def _get_risk_level(self, score: float) -> str:
        """Get risk level based on health score"""
        if score >= 70:
            return "Low Risk"
        elif score >= 50:
            return "Moderate Risk"
        else:
            return "High Risk"
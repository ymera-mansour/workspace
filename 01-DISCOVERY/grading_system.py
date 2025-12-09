"""Grading System - Assigns A+ to F grades based on performance"""
from typing import Dict, Any, List

class GradingSystem:
    """Grades model performance on 0-10 scale, assigns A+ to F"""
    
    def __init__(self):
        self.grades = {}
        
    def calculate_grade(self, score: float) -> str:
        """Convert 0-10 score to letter grade"""
        if score >= 9.5: return "A+"
        elif score >= 9.0: return "A"
        elif score >= 8.5: return "B+"
        elif score >= 8.0: return "B"
        elif score >= 7.5: return "C+"
        elif score >= 7.0: return "C"
        elif score >= 6.0: return "D"
        else: return "F"
        
    def grade_model(self, model: str, criteria: Dict[str, float]) -> Dict[str, Any]:
        """Grade a model based on 5 criteria"""
        # Accuracy 35%, Completeness 25%, Quality 25%, Efficiency 10%, Reliability 5%
        score = (
            criteria.get("accuracy", 0) * 0.35 +
            criteria.get("completeness", 0) * 0.25 +
            criteria.get("quality", 0) * 0.25 +
            criteria.get("efficiency", 0) * 0.10 +
            criteria.get("reliability", 0) * 0.05
        )
        grade = self.calculate_grade(score)
        self.grades[model] = {"score": score, "grade": grade, "criteria": criteria}
        return self.grades[model]
        
    def get_leaderboard(self) -> List[Dict[str, Any]]:
        """Get sorted leaderboard"""
        return sorted(
            [{"model": k, **v} for k, v in self.grades.items()],
            key=lambda x: x["score"],
            reverse=True
        )

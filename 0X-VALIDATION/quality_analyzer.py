"""
Quality Analyzer for Phase X
Analyzes quality of phase outputs with 5 criteria
"""

from typing import Dict, List, Any, Tuple
import time


class QualityAnalyzer:
    """Analyzes quality with 5 criteria"""
    
    def __init__(self):
        self.criteria_weights = {
            'accuracy': 0.35,
            'completeness': 0.25,
            'quality': 0.25,
            'efficiency': 0.10,
            'reliability': 0.05
        }
        self.scores = {}
        self.grade = ''
        
    def analyze(self, outcomes: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze quality with 5 criteria"""
        print(f"\n{'=' * 60}")
        print("PHASE X: Quality Analysis")
        print(f"{'=' * 60}")
        
        # 1. Accuracy (35%)
        accuracy = self._evaluate_accuracy(outcomes)
        
        # 2. Completeness (25%)
        completeness = self._evaluate_completeness(outcomes)
        
        # 3. Quality (25%)
        quality = self._evaluate_quality(outcomes)
        
        # 4. Efficiency (10%)
        efficiency = self._evaluate_efficiency(outcomes)
        
        # 5. Reliability (5%)
        reliability = self._evaluate_reliability(outcomes)
        
        # Calculate weighted score
        weighted_score = (
            accuracy * self.criteria_weights['accuracy'] +
            completeness * self.criteria_weights['completeness'] +
            quality * self.criteria_weights['quality'] +
            efficiency * self.criteria_weights['efficiency'] +
            reliability * self.criteria_weights['reliability']
        )
        
        # Assign grade
        grade = self._assign_grade(weighted_score)
        
        return {
            'weighted_score': weighted_score,
            'grade': grade,
            'criteria': {
                'accuracy': accuracy,
                'completeness': completeness,
                'quality': quality,
                'efficiency': efficiency,
                'reliability': reliability
            },
            'passed': weighted_score >= 7.0
        }
    
    def _evaluate_accuracy(self, outcomes: Dict[str, Any]) -> float:
        """Evaluate accuracy (35%)"""
        print("\n[1/5] Evaluating Accuracy (35%)...")
        
        # Check for errors in outcomes
        errors = outcomes.get('errors', [])
        total_items = outcomes.get('total_items', 1)
        
        if total_items == 0:
            score = 8.0
        else:
            error_rate = len(errors) / total_items
            score = max(0, 10.0 - (error_rate * 10.0))
        
        print(f"  Errors: {len(errors)}")
        print(f"  Score: {score:.1f}/10.0 (Weight: 35%)")
        
        return score
    
    def _evaluate_completeness(self, outcomes: Dict[str, Any]) -> float:
        """Evaluate completeness (25%)"""
        print("\n[2/5] Evaluating Completeness (25%)...")
        
        # Use outcome validator's completeness score
        completeness = outcomes.get('completeness', 8.0)
        
        print(f"  Score: {completeness:.1f}/10.0 (Weight: 25%)")
        
        return completeness
    
    def _evaluate_quality(self, outcomes: Dict[str, Any]) -> float:
        """Evaluate quality (25%)"""
        print("\n[3/5] Evaluating Quality (25%)...")
        
        # Check quality metrics if available
        quality_metrics = outcomes.get('quality_metrics', {})
        
        if not quality_metrics:
            score = 7.5  # Default
        else:
            # Average quality metrics
            scores = [v for v in quality_metrics.values() if isinstance(v, (int, float))]
            score = sum(scores) / len(scores) if scores else 7.5
        
        print(f"  Score: {score:.1f}/10.0 (Weight: 25%)")
        
        return score
    
    def _evaluate_efficiency(self, outcomes: Dict[str, Any]) -> float:
        """Evaluate efficiency (10%)"""
        print("\n[4/5] Evaluating Efficiency (10%)...")
        
        # Check execution time
        execution_time = outcomes.get('execution_time', 0)
        expected_time = outcomes.get('expected_time', execution_time * 1.5)
        
        if expected_time == 0:
            score = 8.0
        else:
            time_ratio = execution_time / expected_time
            if time_ratio <= 0.8:
                score = 10.0
            elif time_ratio <= 1.0:
                score = 9.0
            elif time_ratio <= 1.2:
                score = 8.0
            elif time_ratio <= 1.5:
                score = 7.0
            else:
                score = 6.0
        
        print(f"  Execution time: {execution_time:.1f}s")
        print(f"  Expected time: {expected_time:.1f}s")
        print(f"  Score: {score:.1f}/10.0 (Weight: 10%)")
        
        return score
    
    def _evaluate_reliability(self, outcomes: Dict[str, Any]) -> float:
        """Evaluate reliability (5%)"""
        print("\n[5/5] Evaluating Reliability (5%)...")
        
        # Check consistency and stability
        warnings = outcomes.get('warnings', [])
        
        if len(warnings) == 0:
            score = 10.0
        elif len(warnings) <= 2:
            score = 9.0
        elif len(warnings) <= 5:
            score = 8.0
        else:
            score = 7.0
        
        print(f"  Warnings: {len(warnings)}")
        print(f"  Score: {score:.1f}/10.0 (Weight: 5%)")
        
        return score
    
    def _assign_grade(self, score: float) -> str:
        """Assign letter grade"""
        if score >= 9.5:
            return 'A+'
        elif score >= 9.0:
            return 'A'
        elif score >= 8.5:
            return 'B+'
        elif score >= 8.0:
            return 'B'
        elif score >= 7.5:
            return 'C+'
        elif score >= 7.0:
            return 'C'
        elif score >= 6.0:
            return 'D'
        else:
            return 'F'


if __name__ == "__main__":
    analyzer = QualityAnalyzer()
    
    # Test with sample data
    test_outcomes = {
        'errors': [],
        'total_items': 100,
        'completeness': 9.0,
        'quality_metrics': {'code_quality': 8.5, 'test_coverage': 9.0},
        'execution_time': 120,
        'expected_time': 150,
        'warnings': ['minor warning']
    }
    
    result = analyzer.analyze(test_outcomes)
    print(f"\n\nFinal Score: {result['weighted_score']:.2f}/10.0")
    print(f"Grade: {result['grade']}")
    print(f"Passed: {result['passed']}")

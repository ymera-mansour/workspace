"""
Decision Engine for Phase X
Makes proceed/block/retry decisions
"""

from typing import Dict, List, Any, Tuple
from enum import Enum


class Decision(Enum):
    """Decision types"""
    PROCEED = "PROCEED"
    BLOCK = "BLOCK"
    RETRY = "RETRY"


class DecisionEngine:
    """Makes proceed/block/retry decisions"""
    
    def __init__(self):
        self.decision_history = []
        self.thresholds = {
            'quality_min': 7.0,
            'alignment_min': 7.0,
            'completeness_min': 7.0
        }
        
    def decide(self, 
               quality_score: float,
               alignment_score: float,
               validation_results: Dict[str, Any],
               approval_status: str = None) -> Tuple[Decision, str]:
        """Make proceed/block/retry decision"""
        print(f"\n{'=' * 60}")
        print("PHASE X: Decision Engine")
        print(f"{'=' * 60}")
        
        # Evaluate criteria
        print(f"\nEvaluating decision criteria...")
        print(f"  Quality Score: {quality_score:.1f}/10.0 (min: {self.thresholds['quality_min']})")
        print(f"  Alignment Score: {alignment_score:.1f}/10.0 (min: {self.thresholds['alignment_min']})")
        
        # Check validation results
        completeness = validation_results.get('completeness', 0.0)
        print(f"  Completeness: {completeness:.1f}/10.0 (min: {self.thresholds['completeness_min']})")
        
        # Decision logic
        decision, explanation = self._evaluate_decision(
            quality_score,
            alignment_score,
            completeness,
            validation_results,
            approval_status
        )
        
        # Record decision
        self._record_decision(decision, explanation, {
            'quality_score': quality_score,
            'alignment_score': alignment_score,
            'completeness': completeness
        })
        
        # Display decision
        self._display_decision(decision, explanation)
        
        return decision, explanation
    
    def _evaluate_decision(self,
                          quality_score: float,
                          alignment_score: float,
                          completeness: float,
                          validation_results: Dict[str, Any],
                          approval_status: str) -> Tuple[Decision, str]:
        """Evaluate and make decision"""
        
        # Critical errors = BLOCK
        if validation_results.get('errors', []):
            critical_errors = [e for e in validation_results['errors'] if 'critical' in e.lower()]
            if critical_errors:
                return Decision.BLOCK, f"Critical errors detected: {len(critical_errors)}"
        
        # Check if human rejected
        if approval_status == 'REJECTED':
            return Decision.BLOCK, "Human reviewer rejected the results"
        
        # All scores above threshold = PROCEED
        if (quality_score >= self.thresholds['quality_min'] and
            alignment_score >= self.thresholds['alignment_min'] and
            completeness >= self.thresholds['completeness_min']):
            return Decision.PROCEED, "All quality criteria met"
        
        # Quality too low = RETRY
        if quality_score < self.thresholds['quality_min'] - 1.0:
            return Decision.RETRY, f"Quality score too low: {quality_score:.1f}/10.0"
        
        # Alignment issues = BLOCK
        if alignment_score < self.thresholds['alignment_min'] - 1.0:
            return Decision.BLOCK, f"Poor alignment with goals: {alignment_score:.1f}/10.0"
        
        # Borderline cases
        if (quality_score >= self.thresholds['quality_min'] - 0.5 and
            alignment_score >= self.thresholds['alignment_min']):
            # Check if human approved
            if approval_status == 'APPROVED':
                return Decision.PROCEED, "Human approved despite borderline scores"
            else:
                return Decision.RETRY, "Quality borderline - retry recommended"
        
        # Default to RETRY for marginal cases
        return Decision.RETRY, "Scores below threshold - retry recommended"
    
    def _record_decision(self, decision: Decision, explanation: str, scores: Dict[str, float]):
        """Record decision in history"""
        from datetime import datetime
        
        record = {
            'timestamp': datetime.now().isoformat(),
            'decision': decision.value,
            'explanation': explanation,
            'scores': scores
        }
        
        self.decision_history.append(record)
    
    def _display_decision(self, decision: Decision, explanation: str):
        """Display decision"""
        print(f"\n{'=' * 60}")
        
        if decision == Decision.PROCEED:
            print("✅ DECISION: PROCEED")
            print("   Continue to next phase")
        elif decision == Decision.RETRY:
            print("⚠️  DECISION: RETRY")
            print("   Re-execute previous phase with improvements")
        else:  # BLOCK
            print("❌ DECISION: BLOCK")
            print("   Workflow blocked - manual intervention required")
        
        print(f"\nExplanation: {explanation}")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    engine = DecisionEngine()
    
    # Test case 1: Good scores - should PROCEED
    print("\n\nTest 1: Good Scores")
    decision, explanation = engine.decide(
        quality_score=8.5,
        alignment_score=8.0,
        validation_results={'completeness': 8.5, 'errors': []},
        approval_status='APPROVED'
    )
    print(f"Result: {decision.value} - {explanation}")
    
    # Test case 2: Low quality - should RETRY
    print("\n\nTest 2: Low Quality")
    decision, explanation = engine.decide(
        quality_score=6.2,
        alignment_score=8.0,
        validation_results={'completeness': 7.5, 'errors': []},
        approval_status=None
    )
    print(f"Result: {decision.value} - {explanation}")
    
    # Test case 3: Critical errors - should BLOCK
    print("\n\nTest 3: Critical Errors")
    decision, explanation = engine.decide(
        quality_score=7.5,
        alignment_score=7.5,
        validation_results={'completeness': 8.0, 'errors': ['Critical: Data corruption']},
        approval_status=None
    )
    print(f"Result: {decision.value} - {explanation}")

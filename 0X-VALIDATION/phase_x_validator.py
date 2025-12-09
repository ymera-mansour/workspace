"""
Phase X Validator
=================

Inter-phase validation and planning system that runs BETWEEN all phases.
Validates outcomes, analyzes quality, updates plans dynamically, and generates
additional tasks as needed.

Validation Criteria:
- Accuracy (35%): Correctness of output
- Completeness (25%): Coverage of requirements
- Quality (25%): Code/analysis quality
- Efficiency (10%): Time and resource usage
- Reliability (5%): Consistency across runs
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class PhaseXValidator:
    """Phase X inter-phase validation and planning"""
    
    def __init__(self, config: Dict[str, Any], state_manager: Any):
        """
        Initialize Phase X validator
        
        Args:
            config: Platform configuration
            state_manager: State management system
        """
        self.config = config
        self.state_manager = state_manager
        
        # Validation thresholds
        self.quality_threshold = config.get("phase_x", {}).get("quality_threshold", 7.0)
        self.min_accuracy = config.get("phase_x", {}).get("min_accuracy", 7.0)
        self.min_completeness = config.get("phase_x", {}).get("min_completeness", 7.0)
        
        # Weights for overall score
        self.weights = {
            "accuracy": 0.35,
            "completeness": 0.25,
            "quality": 0.25,
            "efficiency": 0.10,
            "reliability": 0.05
        }
        
        logger.info("Phase X validator initialized")
    
    async def validate(
        self,
        completed_phase: str,
        next_phase: str,
        phase_result: Dict[str, Any],
        all_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run Phase X validation
        
        Args:
            completed_phase: Name of completed phase
            next_phase: Name of next phase
            phase_result: Results from completed phase
            all_results: All workflow results so far
        
        Returns:
            dict: Validation results with decision
        """
        validation_start = datetime.now()
        
        logger.info(f"🔍 Running Phase X validation...")
        logger.info(f"   Completed: {completed_phase}")
        logger.info(f"   Next: {next_phase}")
        
        try:
            # Step 1: Validate phase outcomes
            outcome_validation = await self.validate_outcomes(
                phase_result, completed_phase
            )
            
            # Step 2: Analyze quality across all criteria
            quality_analysis = await self.analyze_quality(
                phase_result, all_results
            )
            
            # Step 3: Check alignment with goals
            alignment_check = await self.check_alignment(
                phase_result, all_results, next_phase
            )
            
            # Step 4: Update plan if needed
            plan_updates = await self.update_plan(
                quality_analysis, alignment_check, next_phase
            )
            
            # Step 5: Generate additional tasks if needed
            additional_tasks = await self.generate_additional_tasks(
                quality_analysis, alignment_check, completed_phase
            )
            
            # Step 6: Make decision
            decision = await self.make_decision(
                outcome_validation,
                quality_analysis,
                alignment_check,
                additional_tasks
            )
            
            validation_result = {
                "validation_id": f"phase_x_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "completed_phase": completed_phase,
                "next_phase": next_phase,
                "timestamp": datetime.now().isoformat(),
                "duration": str(datetime.now() - validation_start),
                
                "outcome_valid": outcome_validation["valid"],
                "outcome_issues": outcome_validation.get("issues", []),
                
                "quality_score": quality_analysis["overall_score"],
                "quality_grade": quality_analysis["grade"],
                "quality_breakdown": quality_analysis["breakdown"],
                
                "alignment_score": alignment_check["score"],
                "alignment_issues": alignment_check.get("issues", []),
                
                "plan_updates": plan_updates,
                "additional_tasks": additional_tasks,
                
                "approved_to_proceed": decision["approved"],
                "decision_reason": decision["reason"],
                "requires_human_approval": decision.get("requires_human", False),
                
                "recommendations": decision.get("recommendations", [])
            }
            
            # Log summary
            self._log_validation_summary(validation_result)
            
            # Save validation results
            await self.state_manager.save_state({
                "phase_x_validation": validation_result
            }, checkpoint=False)
            
            return validation_result
            
        except Exception as e:
            logger.error(f"❌ Phase X validation failed: {e}")
            return {
                "error": str(e),
                "approved_to_proceed": False,
                "requires_human_approval": True
            }
    
    async def validate_outcomes(
        self,
        phase_result: Dict[str, Any],
        phase_name: str
    ) -> Dict[str, Any]:
        """Validate phase outcomes"""
        issues = []
        
        # Check if phase completed
        if phase_result.get("status") != "completed":
            issues.append({
                "severity": "critical",
                "message": f"Phase did not complete successfully: {phase_result.get('status')}"
            })
        
        # Check for artifacts
        artifacts = phase_result.get("artifacts", [])
        if not artifacts and "Discovery" in phase_name:
            issues.append({
                "severity": "warning",
                "message": "No artifacts produced by discovery phase"
            })
        
        # Check metrics
        metrics = phase_result.get("metrics", {})
        if metrics.get("files_processed", 0) == 0 and "Discovery" in phase_name:
            issues.append({
                "severity": "error",
                "message": "No files were processed"
            })
        
        valid = len([i for i in issues if i["severity"] == "critical"]) == 0
        
        return {
            "valid": valid,
            "issues": issues,
            "artifacts_count": len(artifacts),
            "metrics": metrics
        }
    
    async def analyze_quality(
        self,
        phase_result: Dict[str, Any],
        all_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze quality across all criteria"""
        
        # Get or estimate scores for each criterion
        accuracy = phase_result.get("accuracy_score", 8.0)
        completeness = phase_result.get("completeness_score", 8.0)
        quality = phase_result.get("quality_score", 8.5)
        efficiency = phase_result.get("efficiency_score", 8.0)
        reliability = phase_result.get("reliability_score", 8.5)
        
        # Calculate weighted overall score
        overall_score = (
            accuracy * self.weights["accuracy"] +
            completeness * self.weights["completeness"] +
            quality * self.weights["quality"] +
            efficiency * self.weights["efficiency"] +
            reliability * self.weights["reliability"]
        )
        
        # Assign grade
        grade = self._calculate_grade(overall_score)
        
        return {
            "overall_score": round(overall_score, 2),
            "grade": grade,
            "breakdown": {
                "accuracy": {"score": accuracy, "weight": self.weights["accuracy"]},
                "completeness": {"score": completeness, "weight": self.weights["completeness"]},
                "quality": {"score": quality, "weight": self.weights["quality"]},
                "efficiency": {"score": efficiency, "weight": self.weights["efficiency"]},
                "reliability": {"score": reliability, "weight": self.weights["reliability"]}
            }
        }
    
    async def check_alignment(
        self,
        phase_result: Dict[str, Any],
        all_results: Dict[str, Any],
        next_phase: str
    ) -> Dict[str, Any]:
        """Check alignment with overall goals"""
        issues = []
        
        # Check if results align with expected outcomes
        # (This is a simplified version - would be more sophisticated in production)
        
        alignment_score = 8.5  # Mock score
        
        return {
            "score": alignment_score,
            "issues": issues,
            "aligned": alignment_score >= 7.0
        }
    
    async def update_plan(
        self,
        quality_analysis: Dict[str, Any],
        alignment_check: Dict[str, Any],
        next_phase: str
    ) -> List[Dict[str, Any]]:
        """Update plan based on validation results"""
        updates = []
        
        # If quality is low, suggest improvements
        if quality_analysis["overall_score"] < self.quality_threshold:
            updates.append({
                "type": "quality_improvement",
                "phase": next_phase,
                "description": "Increase validation depth due to lower quality in previous phase",
                "priority": "high"
            })
        
        # If alignment issues, adjust strategy
        if not alignment_check.get("aligned", True):
            updates.append({
                "type": "strategy_adjustment",
                "phase": next_phase,
                "description": "Adjust strategy to better align with goals",
                "priority": "medium"
            })
        
        return updates
    
    async def generate_additional_tasks(
        self,
        quality_analysis: Dict[str, Any],
        alignment_check: Dict[str, Any],
        completed_phase: str
    ) -> List[Dict[str, Any]]:
        """Generate additional tasks if needed"""
        tasks = []
        
        # If quality is below threshold, generate remediation tasks
        if quality_analysis["overall_score"] < self.quality_threshold:
            tasks.append({
                "task_id": f"remediate_{completed_phase}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "type": "remediation",
                "phase": completed_phase,
                "description": f"Re-run {completed_phase} with improved parameters",
                "priority": "high",
                "estimated_duration": "1-2 hours"
            })
        
        # Check specific quality issues
        breakdown = quality_analysis.get("breakdown", {})
        
        if breakdown.get("completeness", {}).get("score", 10) < 7.0:
            tasks.append({
                "task_id": f"complete_{completed_phase}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "type": "completion",
                "phase": completed_phase,
                "description": "Complete missing aspects of the phase",
                "priority": "high"
            })
        
        return tasks
    
    async def make_decision(
        self,
        outcome_validation: Dict[str, Any],
        quality_analysis: Dict[str, Any],
        alignment_check: Dict[str, Any],
        additional_tasks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Make decision on whether to proceed"""
        
        # Critical issues block progression
        critical_issues = [
            i for i in outcome_validation.get("issues", [])
            if i.get("severity") == "critical"
        ]
        
        if critical_issues:
            return {
                "approved": False,
                "reason": f"Critical issues found: {len(critical_issues)}",
                "requires_human": True,
                "recommendations": ["Fix critical issues before proceeding"]
            }
        
        # Low quality requires human approval
        if quality_analysis["overall_score"] < self.quality_threshold:
            return {
                "approved": False,
                "reason": f"Quality score {quality_analysis['overall_score']:.1f} below threshold {self.quality_threshold}",
                "requires_human": True,
                "recommendations": [
                    f"Improve quality to at least {self.quality_threshold}",
                    "Consider re-running previous phase with adjusted parameters"
                ]
            }
        
        # Alignment issues need attention
        if not alignment_check.get("aligned", True):
            return {
                "approved": True,
                "reason": "Proceeding with alignment adjustments",
                "requires_human": False,
                "recommendations": [
                    "Monitor alignment in next phase",
                    "Apply plan updates"
                ]
            }
        
        # All good - proceed
        return {
            "approved": True,
            "reason": f"Quality score {quality_analysis['overall_score']:.1f}, all validations passed",
            "requires_human": False,
            "recommendations": ["Continue to next phase as planned"]
        }
    
    def _calculate_grade(self, score: float) -> str:
        """Calculate letter grade from score"""
        if score >= 9.5:
            return "A+"
        elif score >= 9.0:
            return "A"
        elif score >= 8.5:
            return "B+"
        elif score >= 8.0:
            return "B"
        elif score >= 7.5:
            return "C+"
        elif score >= 7.0:
            return "C"
        elif score >= 6.0:
            return "D"
        else:
            return "F"
    
    def _log_validation_summary(self, result: Dict[str, Any]):
        """Log validation summary"""
        logger.info(f"\n{'='*60}")
        logger.info("📊 Phase X Validation Summary")
        logger.info(f"{'='*60}")
        logger.info(f"Outcome Valid: {'✅' if result['outcome_valid'] else '❌'}")
        logger.info(f"Quality Score: {result['quality_score']:.1f}/10.0 (Grade: {result['quality_grade']})")
        logger.info(f"Alignment Score: {result['alignment_score']:.1f}/10.0")
        logger.info(f"Plan Updates: {len(result['plan_updates'])}")
        logger.info(f"Additional Tasks: {len(result['additional_tasks'])}")
        logger.info(f"Decision: {'✅ PROCEED' if result['approved_to_proceed'] else '❌ BLOCKED'}")
        if result.get("requires_human_approval"):
            logger.info("⚠️  Requires human approval")
        logger.info(f"{'='*60}\n")

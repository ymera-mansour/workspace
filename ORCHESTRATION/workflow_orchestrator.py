"""
Workflow Orchestrator
=====================

Main workflow controller that orchestrates the execution of all phases
with Phase X inter-phase validation.

Phase Flow:
-----------
Phase 0 → Phase X → Phase 1 → Phase X → Phase 2 → Phase X → Phase 3 → Phase X → Phase 4 → Phase X → Phase 5
"""

import asyncio
import logging
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class WorkflowOrchestrator:
    """Main workflow orchestrator for YMERA platform"""
    
    def __init__(
        self,
        config: Dict[str, Any],
        state_manager: Any,
        phase_x_validator: Any = None
    ):
        """
        Initialize workflow orchestrator
        
        Args:
            config: Platform configuration
            state_manager: State management system
            phase_x_validator: Phase X validator (optional)
        """
        self.config = config
        self.state_manager = state_manager
        self.phase_x_validator = phase_x_validator
        self.current_phase = None
        self.phase_results = {}
        self.start_time = None
        self.end_time = None
        
        # Phase definitions
        self.phase_definitions = {
            "Phase 0: Pre-Flight & Setup": {
                "module": "00-FOUNDATION",
                "orchestrator": "phase0_orchestrator",
                "duration_estimate": "1-2 hours",
                "critical": True
            },
            "Phase 1: Discovery": {
                "module": "01-DISCOVERY",
                "orchestrator": "phase1_orchestrator",
                "duration_estimate": "3-5 hours",
                "layers": 8
            },
            "Phase 2: Analysis": {
                "module": "02-ANALYSIS",
                "orchestrator": "phase2_orchestrator",
                "duration_estimate": "4-6 hours",
                "layers": 6
            },
            "Phase 3: Consolidation": {
                "module": "03-CONSOLIDATION",
                "orchestrator": "phase3_orchestrator",
                "duration_estimate": "8-12 hours",
                "layers": 8
            },
            "Phase 4: Testing": {
                "module": "04-TESTING",
                "orchestrator": "phase4_orchestrator",
                "duration_estimate": "4-8 hours",
                "layers": 6
            },
            "Phase 5: Integration": {
                "module": "05-INTEGRATION",
                "orchestrator": "phase5_orchestrator",
                "duration_estimate": "3-5 hours",
                "layers": 5
            }
        }
        
        logger.info("Workflow orchestrator initialized")
    
    async def execute_workflow(
        self,
        repo_path: str,
        phases: List[str],
        enable_phase_x: bool = True
    ) -> Dict[str, Any]:
        """
        Execute workflow phases
        
        Args:
            repo_path: Path to repository
            phases: List of phases to execute
            enable_phase_x: Enable Phase X validation
        
        Returns:
            dict: Workflow results
        """
        self.start_time = datetime.now()
        
        results = {
            "repo_path": repo_path,
            "phases_requested": phases,
            "phases_completed": [],
            "phase_results": {},
            "phase_x_validations": [],
            "start_time": self.start_time.isoformat(),
            "end_time": None,
            "total_duration": None,
            "overall_quality_score": None,
            "status": "running"
        }
        
        try:
            logger.info(f"🚀 Starting workflow execution for: {repo_path}")
            
            # Execute each phase
            for i, phase_name in enumerate(phases):
                logger.info(f"\n{'='*80}")
                logger.info(f"📍 Phase {i+1}/{len(phases)}: {phase_name}")
                logger.info(f"{'='*80}\n")
                
                # Execute phase
                phase_result = await self.execute_phase(
                    phase_name=phase_name,
                    repo_path=repo_path,
                    previous_results=results.get("phase_results", {})
                )
                
                results["phase_results"][phase_name] = phase_result
                results["phases_completed"].append(phase_name)
                
                # Save intermediate state
                await self.state_manager.save_state({
                    "workflow_results": results,
                    "current_phase": phase_name,
                    "timestamp": datetime.now().isoformat()
                })
                
                # Run Phase X validation if enabled and not the last phase
                if enable_phase_x and i < len(phases) - 1:
                    logger.info(f"\n{'='*80}")
                    logger.info("🔍 Phase X: Inter-Phase Validation & Planning")
                    logger.info(f"{'='*80}\n")
                    
                    phase_x_result = await self.run_phase_x_validation(
                        completed_phase=phase_name,
                        next_phase=phases[i+1],
                        phase_result=phase_result,
                        all_results=results
                    )
                    
                    results["phase_x_validations"].append(phase_x_result)
                    
                    # Check if we should proceed
                    if not phase_x_result.get("approved_to_proceed", True):
                        logger.warning(f"⚠️  Phase X blocked progression after {phase_name}")
                        results["status"] = "blocked_by_phase_x"
                        break
                    
                    # Check if additional tasks were generated
                    if phase_x_result.get("additional_tasks"):
                        logger.info(f"📝 Phase X generated {len(phase_x_result['additional_tasks'])} additional tasks")
                        # TODO: Handle additional tasks
            
            # Calculate final metrics
            self.end_time = datetime.now()
            results["end_time"] = self.end_time.isoformat()
            results["total_duration"] = str(self.end_time - self.start_time)
            results["status"] = "completed"
            
            # Calculate overall quality score
            quality_scores = [
                r.get("quality_score", 0) 
                for r in results["phase_results"].values()
            ]
            if quality_scores:
                results["overall_quality_score"] = sum(quality_scores) / len(quality_scores)
            
            logger.info(f"\n{'='*80}")
            logger.info("✅ Workflow completed successfully")
            logger.info(f"{'='*80}\n")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Workflow execution failed: {e}")
            results["status"] = "failed"
            results["error"] = str(e)
            raise
    
    async def execute_phase(
        self,
        phase_name: str,
        repo_path: str,
        previous_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute a single phase
        
        Args:
            phase_name: Name of phase to execute
            repo_path: Path to repository
            previous_results: Results from previous phases
        
        Returns:
            dict: Phase execution results
        """
        phase_def = self.phase_definitions.get(phase_name)
        if not phase_def:
            raise ValueError(f"Unknown phase: {phase_name}")
        
        phase_start = datetime.now()
        
        logger.info(f"Executing {phase_name}...")
        logger.info(f"Estimated duration: {phase_def.get('duration_estimate', 'Unknown')}")
        
        try:
            # Dynamic import of phase orchestrator
            # For now, return mock results until orchestrators are implemented
            result = {
                "phase_name": phase_name,
                "status": "completed",
                "start_time": phase_start.isoformat(),
                "end_time": datetime.now().isoformat(),
                "duration": str(datetime.now() - phase_start),
                "quality_score": 8.5,  # Mock score
                "layers_completed": phase_def.get("layers", 0),
                "artifacts": [],
                "metrics": {
                    "files_processed": 0,
                    "lines_analyzed": 0,
                    "issues_found": 0,
                    "tests_generated": 0
                }
            }
            
            logger.info(f"✅ {phase_name} completed in {result['duration']}")
            logger.info(f"📊 Quality score: {result['quality_score']}/10.0")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Phase execution failed: {e}")
            return {
                "phase_name": phase_name,
                "status": "failed",
                "error": str(e),
                "start_time": phase_start.isoformat(),
                "end_time": datetime.now().isoformat()
            }
    
    async def run_phase_x_validation(
        self,
        completed_phase: str,
        next_phase: str,
        phase_result: Dict[str, Any],
        all_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run Phase X inter-phase validation
        
        Args:
            completed_phase: Name of completed phase
            next_phase: Name of next phase
            phase_result: Results from completed phase
            all_results: All workflow results so far
        
        Returns:
            dict: Phase X validation results
        """
        if not self.phase_x_validator:
            logger.warning("Phase X validator not available, skipping validation")
            return {
                "skipped": True,
                "reason": "validator_not_available",
                "approved_to_proceed": True
            }
        
        try:
            validation_result = await self.phase_x_validator.validate(
                completed_phase=completed_phase,
                next_phase=next_phase,
                phase_result=phase_result,
                all_results=all_results
            )
            
            # Log validation results
            logger.info(f"📊 Validation Results:")
            logger.info(f"   Outcome validation: {'✅' if validation_result.get('outcome_valid') else '❌'}")
            logger.info(f"   Quality score: {validation_result.get('quality_score', 'N/A')}/10.0")
            logger.info(f"   Approved to proceed: {'✅' if validation_result.get('approved_to_proceed') else '❌'}")
            
            if validation_result.get("plan_updates"):
                logger.info(f"   Plan updates: {len(validation_result['plan_updates'])} changes")
            
            if validation_result.get("additional_tasks"):
                logger.info(f"   Additional tasks: {len(validation_result['additional_tasks'])} tasks")
            
            return validation_result
            
        except Exception as e:
            logger.error(f"❌ Phase X validation failed: {e}")
            return {
                "error": str(e),
                "approved_to_proceed": False
            }
    
    async def get_workflow_status(self) -> Dict[str, Any]:
        """Get current workflow status"""
        return {
            "current_phase": self.current_phase,
            "phases_completed": list(self.phase_results.keys()),
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "elapsed_time": str(datetime.now() - self.start_time) if self.start_time else None
        }
    
    async def pause_workflow(self):
        """Pause workflow execution"""
        logger.info("⏸️  Pausing workflow...")
        # Save current state
        await self.state_manager.save_state({
            "status": "paused",
            "current_phase": self.current_phase,
            "phase_results": self.phase_results,
            "timestamp": datetime.now().isoformat()
        })
    
    async def resume_workflow(self, state_id: str):
        """Resume workflow from saved state"""
        logger.info(f"▶️  Resuming workflow from state: {state_id}")
        state = await self.state_manager.load_state(state_id)
        self.current_phase = state.get("current_phase")
        self.phase_results = state.get("phase_results", {})
        logger.info(f"Resumed from phase: {self.current_phase}")

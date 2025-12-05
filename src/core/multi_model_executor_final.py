# YMERA Refactoring Project
# Phase: 2E Enhanced | Agent: qoder | Created: 2024-12-05
# Multi-Model Task Execution Engine - COMPLETE

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import asyncio
import logging
from enum import Enum
import time

logger = logging.getLogger(__name__)

@dataclass
class PhaseResult:
    """Result from a task phase"""
    phase: str
    model_used: str
    provider_used: str
    success: bool
    result: Any
    execution_time: float
    tokens_used: int
    error: Optional[str] = None

@dataclass
class MultiModelResult:
    """Complete result from multi-model execution"""
    strategy_type: str
    total_phases: int
    successful_phases: int
    phase_results: List[PhaseResult]
    final_result: Any
    total_execution_time: float
    total_tokens_used: int
    total_cost: float
    models_used: List[str]

class MultiModelExecutor:
    """
    Executes tasks using multiple models across different phases
    
    Workflow:
    1. PLANNING phase (fast model) - Understand task
    2. RESEARCH phase (reasoning model) - Gather information
    3. GENERATION phase (specialized model) - Create solution
    4. REVIEW phase (quality model) - Check quality
    5. REFINEMENT phase (specialized model) - Improve
    6. VALIDATION phase (accuracy model) - Final check
    """
    
    def __init__(self):
        from .ai_orchestrator import get_orchestrator
        from .agent_model_matcher import get_agent_model_matcher
        
        self.orchestrator = get_orchestrator()
        self.matcher = get_agent_model_matcher()
    
    async def execute_with_multi_model(
        self,
        agent_name: str,
        task_description: str,
        task_parameters: Dict[str, Any],
        enable_phases: Optional[List[str]] = None
    ) -> MultiModelResult:
        """
        Execute task using multiple models
        
        Args:
            agent_name: Name of the agent
            task_description: Description of the task
            task_parameters: Task parameters
            enable_phases: Which phases to enable (None = all applicable)
        
        Returns:
            MultiModelResult with results from all phases
        """
        
        logger.info(f"Starting multi-model execution for {agent_name}")
        
        # Get model strategy
        strategy = await self.matcher.match_agent_to_models(
            agent_name,
            task_description,
            task_parameters
        )
        
        if strategy["strategy_type"] == "single_model":
            # Use single model
            return await self._execute_single_model(strategy, task_description, task_parameters)
        
        # Multi-model execution
        phase_results = []
        total_time = 0
        total_tokens = 0
        total_cost = 0
        models_used = set()
        
        # Context accumulator for passing data between phases
        context = {
            "task_description": task_description,
            "task_parameters": task_parameters,
            "phase_outputs": {}
        }
        
        for phase_assignment in strategy["phases"]:
            phase_name = phase_assignment["phase"]
            
            # Skip if not enabled
            if enable_phases and phase_name not in enable_phases:
                continue
            
            logger.info(f"Executing phase: {phase_name}")
            
            # Execute phase
            phase_result = await self._execute_phase(
                phase_assignment,
                context
            )
            
            phase_results.append(phase_result)
            total_time += phase_result.execution_time
            total_tokens += phase_result.tokens_used
            models_used.add(f"{phase_result.provider_used}:{phase_result.model_used}")
            
            # Add phase output to context
            if phase_result.success:
                context["phase_outputs"][phase_name] = phase_result.result
            else:
                logger.warning(f"Phase {phase_name} failed: {phase_result.error}")
                # Decide whether to continue or abort
                if phase_name in ["planning", "generation"]:
                    # Critical phases - abort
                    break
        
        # Combine results
        final_result = await self._combine_phase_results(phase_results, strategy)
        
        return MultiModelResult(
            strategy_type="multi_model",
            total_phases=len(strategy["phases"]),
            successful_phases=sum(1 for r in phase_results if r.success),
            phase_results=phase_results,
            final_result=final_result,
            total_execution_time=total_time,
            total_tokens_used=total_tokens,
            total_cost=total_cost,
            models_used=list(models_used)
        )
    
    async def _execute_phase(
        self,
        phase_assignment: Dict[str, Any],
        context: Dict[str, Any]
    ) -> PhaseResult:
        """Execute a single phase"""
        
        from .ai_orchestrator import AIRequest, TaskType, TaskComplexity
        
        phase_name = phase_assignment["phase"]
        primary_model = phase_assignment["primary_model"]
        
        start_time = time.time()
        
        # Build phase-specific prompt
        prompt = self._build_phase_prompt(phase_name, context)
        
        try:
            # Create request
            request = AIRequest(
                task_id=f"{context['task_description'][:20]}_{phase_name}",
                task_type=TaskType.TEXT_GENERATION,
                complexity=TaskComplexity.MEDIUM,
                prompt=prompt,
                model_hint=primary_model["model"]
            )
            
            # Execute with orchestrator
            result = await self.orchestrator.execute(request)
            
            execution_time = time.time() - start_time
            
            if result.success:
                return PhaseResult(
                    phase=phase_name,
                    model_used=result.model_used,
                    provider_used=result.provider_used,
                    success=True,
                    result=result.response,
                    execution_time=execution_time,
                    tokens_used=result.tokens_used
                )
            else:
                return PhaseResult(
                    phase=phase_name,
                    model_used=result.model_used or "unknown",
                    provider_used=result.provider_used or "unknown",
                    success=False,
                    result=None,
                    execution_time=execution_time,
                    tokens_used=0,
                    error=result.error
                )
                
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Phase {phase_name} execution failed: {e}")
            return PhaseResult(
                phase=phase_name,
                model_used="unknown",
                provider_used="unknown",
                success=False,
                result=None,
                execution_time=execution_time,
                tokens_used=0,
                error=str(e)
            )
    
    def _build_phase_prompt(
        self,
        phase_name: str,
        context: Dict[str, Any]
    ) -> str:
        """Build phase-specific prompt with context"""
        
        task_description = context["task_description"]
        task_parameters = context["task_parameters"]
        phase_outputs = context["phase_outputs"]
        
        # Base prompt
        prompt_parts = [
            f"Task: {task_description}",
            f"\nParameters: {task_parameters}",
        ]
        
        # Add previous phase outputs as context
        if phase_outputs:
            prompt_parts.append("\n\nPrevious Phase Results:")
            for prev_phase, output in phase_outputs.items():
                prompt_parts.append(f"\n{prev_phase.upper()}: {output}")
        
        # Phase-specific instructions
        phase_instructions = {
            "planning": """
\n\nPHASE: PLANNING
Your role: Understand the task and create a high-level plan.
Output: A clear plan with steps to accomplish the task.
""",
            "research": """
\n\nPHASE: RESEARCH
Your role: Gather necessary information and context.
Output: Relevant information, references, and insights.
""",
            "generation": """
\n\nPHASE: GENERATION
Your role: Create the primary solution/output.
Output: The main deliverable (code, content, analysis, etc.).
""",
            "review": """
\n\nPHASE: REVIEW
Your role: Review the generated output for quality and correctness.
Output: Assessment with identified issues and suggestions.
""",
            "refinement": """
\n\nPHASE: REFINEMENT
Your role: Improve the output based on review feedback.
Output: Enhanced version of the deliverable.
""",
            "validation": """
\n\nPHASE: VALIDATION
Your role: Final validation of correctness and completeness.
Output: Validation report and final approval.
"""
        }
        
        prompt_parts.append(phase_instructions.get(phase_name, ""))
        
        return "".join(prompt_parts)
    
    async def _execute_single_model(
        self,
        strategy: Dict[str, Any],
        task_description: str,
        task_parameters: Dict[str, Any]
    ) -> MultiModelResult:
        """Execute with single model"""
        
        from .ai_orchestrator import AIRequest, TaskType, TaskComplexity
        
        model = strategy["model"]
        start_time = time.time()
        
        try:
            # Build prompt
            prompt = f"""Task: {task_description}

Parameters: {task_parameters}

Please complete this task comprehensively."""
            
            # Create request
            request = AIRequest(
                task_id=task_description[:20],
                task_type=TaskType.TEXT_GENERATION,
                complexity=TaskComplexity.MEDIUM,
                prompt=prompt,
                model_hint=model["model"]
            )
            
            # Execute
            result = await self.orchestrator.execute(request)
            
            execution_time = time.time() - start_time
            
            phase_result = PhaseResult(
                phase="all",
                model_used=result.model_used,
                provider_used=result.provider_used,
                success=result.success,
                result=result.response,
                execution_time=execution_time,
                tokens_used=result.tokens_used,
                error=result.error if not result.success else None
            )
            
            return MultiModelResult(
                strategy_type="single_model",
                total_phases=1,
                successful_phases=1 if result.success else 0,
                phase_results=[phase_result],
                final_result=result.response if result.success else None,
                total_execution_time=execution_time,
                total_tokens_used=result.tokens_used,
                total_cost=0.0,  # Cost calculation to be implemented
                models_used=[f"{result.provider_used}:{result.model_used}"]
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Single model execution failed: {e}")
            
            phase_result = PhaseResult(
                phase="all",
                model_used="unknown",
                provider_used="unknown",
                success=False,
                result=None,
                execution_time=execution_time,
                tokens_used=0,
                error=str(e)
            )
            
            return MultiModelResult(
                strategy_type="single_model",
                total_phases=1,
                successful_phases=0,
                phase_results=[phase_result],
                final_result=None,
                total_execution_time=execution_time,
                total_tokens_used=0,
                total_cost=0.0,
                models_used=[]
            )
    
    async def _combine_phase_results(
        self,
        phase_results: List[PhaseResult],
        strategy: Dict[str, Any]
    ) -> Any:
        """
        Combine results from multiple phases into final output
        
        Strategy:
        1. If all phases successful, use refinement/generation result
        2. If some phases failed, use best available result
        3. Include metadata about the process
        """
        
        if not phase_results:
            return None
        
        # Find the most important successful result
        priority_phases = ["refinement", "generation", "validation", "review", "planning"]
        
        for priority_phase in priority_phases:
            for result in phase_results:
                if result.phase == priority_phase and result.success:
                    return {
                        "final_output": result.result,
                        "phase": result.phase,
                        "model_used": f"{result.provider_used}:{result.model_used}",
                        "all_phases": [
                            {
                                "phase": r.phase,
                                "success": r.success,
                                "model": f"{r.provider_used}:{r.model_used}"
                            }
                            for r in phase_results
                        ]
                    }
        
        # If no successful results found, return error summary
        return {
            "final_output": None,
            "error": "All phases failed",
            "all_phases": [
                {
                    "phase": r.phase,
                    "success": r.success,
                    "error": r.error
                }
                for r in phase_results
            ]
        }


# Singleton
_executor_instance = None

def get_multi_model_executor() -> MultiModelExecutor:
    """Get singleton executor instance"""
    global _executor_instance
    if _executor_instance is None:
        _executor_instance = MultiModelExecutor()
    return _executor_instance

# YMERA Refactoring Project
# Phase: 2E Enhanced | Agent: qoder | Created: 2024-12-05
# Complete Workflow Validation, Task Distribution & Quality System

from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import asyncio
import logging
import json
import uuid
from collections import defaultdict

logger = logging.getLogger(__name__)

# ============================================================================
# ENUMS & DATA CLASSES
# ============================================================================

class WorkflowStatus(Enum):
    """Workflow execution status"""
    PENDING = "pending"
    VALIDATING = "validating"
    DISTRIBUTING = "distributing"
    EXECUTING = "executing"
    REVIEWING = "reviewing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class TaskPriority(Enum):
    """Task priority levels"""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4

class QualityLevel(Enum):
    """Quality assessment levels"""
    EXCELLENT = "excellent"      # 90-100%
    GOOD = "good"               # 75-89%
    ACCEPTABLE = "acceptable"   # 60-74%
    POOR = "poor"              # 40-59%
    UNACCEPTABLE = "unacceptable"  # 0-39%

@dataclass
class Task:
    """Individual task in workflow"""
    task_id: str
    workflow_id: str
    agent_name: str
    task_type: str
    description: str
    parameters: Dict[str, Any]
    dependencies: List[str] = field(default_factory=list)
    priority: TaskPriority = TaskPriority.MEDIUM
    status: str = "pending"
    assigned_model: Optional[str] = None
    result: Any = None
    quality_score: float = 0.0
    execution_time: float = 0.0
    retry_count: int = 0
    max_retries: int = 3
    error: Optional[str] = None

@dataclass
class WorkflowValidation:
    """Workflow validation result"""
    valid: bool
    issues: List[str]
    warnings: List[str]
    estimated_time: float
    estimated_cost: float
    risk_level: str

@dataclass
class QualityMetrics:
    """Quality assessment metrics"""
    accuracy: float         # 0-1: Correctness
    completeness: float     # 0-1: All requirements met
    consistency: float      # 0-1: Internal consistency
    efficiency: float       # 0-1: Resource usage
    maintainability: float  # 0-1: Code/output quality
    overall_score: float    # 0-1: Weighted average

@dataclass
class TaskOutcome:
    """Complete task outcome with validation"""
    task_id: str
    success: bool
    result: Any
    quality_metrics: QualityMetrics
    execution_time: float
    model_used: str
    validation_passed: bool
    validation_details: Dict[str, Any]
    security_passed: bool
    security_details: Dict[str, Any]

# ============================================================================
# WORKFLOW VALIDATOR
# ============================================================================

class WorkflowValidator:
    """
    Validates workflows before execution
    
    Checks:
    - Task dependencies are valid (no cycles)
    - Required resources available
    - Agent capabilities match tasks
    - Estimated feasibility
    """
    
    def __init__(self):
        self.validation_rules = self._initialize_validation_rules()
    
    def _initialize_validation_rules(self) -> Dict[str, Callable]:
        """Initialize validation rules"""
        return {
            "no_cycles": self._check_no_cycles,
            "valid_dependencies": self._check_valid_dependencies,
            "agent_capabilities": self._check_agent_capabilities,
            "resource_availability": self._check_resource_availability,
            "time_feasibility": self._check_time_feasibility,
            "cost_feasibility": self._check_cost_feasibility
        }
    
    async def validate_workflow(
        self,
        workflow_id: str,
        tasks: List[Task]
    ) -> WorkflowValidation:
        """
        Validate entire workflow before execution
        
        Returns:
            WorkflowValidation with issues, warnings, and estimates
        """
        issues = []
        warnings = []
        
        # Run all validation rules
        for rule_name, rule_func in self.validation_rules.items():
            try:
                result = await rule_func(tasks)
                if not result["valid"]:
                    issues.extend(result.get("issues", []))
                warnings.extend(result.get("warnings", []))
            except Exception as e:
                issues.append(f"Validation rule '{rule_name}' failed: {e}")
        
        # Calculate estimates
        estimated_time = self._estimate_execution_time(tasks)
        estimated_cost = self._estimate_execution_cost(tasks)
        
        # Determine risk level
        risk_level = self._assess_risk_level(tasks, issues, warnings)
        
        valid = len(issues) == 0
        
        return WorkflowValidation(
            valid=valid,
            issues=issues,
            warnings=warnings,
            estimated_time=estimated_time,
            estimated_cost=estimated_cost,
            risk_level=risk_level
        )
    
    async def _check_no_cycles(self, tasks: List[Task]) -> Dict[str, Any]:
        """Check for dependency cycles"""
        task_map = {task.task_id: task for task in tasks}
        visited = set()
        rec_stack = set()
        
        def has_cycle(task_id: str) -> bool:
            visited.add(task_id)
            rec_stack.add(task_id)
            
            task = task_map.get(task_id)
            if task:
                for dep_id in task.dependencies:
                    if dep_id not in visited:
                        if has_cycle(dep_id):
                            return True
                    elif dep_id in rec_stack:
                        return True
            
            rec_stack.remove(task_id)
            return False
        
        for task in tasks:
            if task.task_id not in visited:
                if has_cycle(task.task_id):
                    return {
                        "valid": False,
                        "issues": [f"Circular dependency detected involving task {task.task_id}"]
                    }
        
        return {"valid": True, "issues": [], "warnings": []}
    
    async def _check_valid_dependencies(self, tasks: List[Task]) -> Dict[str, Any]:
        """Check all dependencies reference valid tasks"""
        task_ids = {task.task_id for task in tasks}
        issues = []
        
        for task in tasks:
            for dep_id in task.dependencies:
                if dep_id not in task_ids:
                    issues.append(f"Task {task.task_id} depends on non-existent task {dep_id}")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": []
        }
    
    async def _check_agent_capabilities(self, tasks: List[Task]) -> Dict[str, Any]:
        """Check agents can handle assigned tasks"""
        from agents.registry import AgentRegistry
        
        warnings = []
        available_agents = AgentRegistry.list_agents()
        
        for task in tasks:
            if task.agent_name not in available_agents:
                warnings.append(f"Agent '{task.agent_name}' not found for task {task.task_id}")
        
        return {
            "valid": True,
            "issues": [],
            "warnings": warnings
        }
    
    async def _check_resource_availability(self, tasks: List[Task]) -> Dict[str, Any]:
        """Check required resources are available"""
        warnings = []
        
        # Check API keys
        from core_services.ai_mcp.enhanced_model_selector import get_enhanced_selector
        selector = get_enhanced_selector()
        stats = selector.get_provider_stats()
        
        for provider, provider_stats in stats.items():
            if provider_stats["available_keys"] == 0:
                warnings.append(f"No available API keys for provider '{provider}'")
        
        return {
            "valid": True,
            "issues": [],
            "warnings": warnings
        }
    
    async def _check_time_feasibility(self, tasks: List[Task]) -> Dict[str, Any]:
        """Check if workflow can complete in reasonable time"""
        estimated_time = self._estimate_execution_time(tasks)
        warnings = []
        
        if estimated_time > 3600:  # 1 hour
            warnings.append(f"Workflow may take over 1 hour (estimated: {estimated_time/60:.1f} minutes)")
        
        return {
            "valid": True,
            "issues": [],
            "warnings": warnings
        }
    
    async def _check_cost_feasibility(self, tasks: List[Task]) -> Dict[str, Any]:
        """Check estimated cost is acceptable"""
        estimated_cost = self._estimate_execution_cost(tasks)
        warnings = []
        
        if estimated_cost > 0:
            warnings.append(f"Workflow may incur costs: ${estimated_cost:.2f}")
        
        return {
            "valid": True,
            "issues": [],
            "warnings": warnings
        }
    
    def _estimate_execution_time(self, tasks: List[Task]) -> float:
        """Estimate total execution time"""
        # Build dependency graph
        task_map = {task.task_id: task for task in tasks}
        
        # Calculate longest path (critical path)
        def get_task_time(task_id: str, memo: Dict[str, float]) -> float:
            if task_id in memo:
                return memo[task_id]
            
            task = task_map.get(task_id)
            if not task:
                return 0.0
            
            # Base time estimate by task type
            base_time = {
                "code_generation": 10.0,
                "code_review": 5.0,
                "database_operations": 3.0,
                "data_analysis": 15.0,
                "documentation": 8.0,
                "general": 5.0
            }.get(task.task_type, 5.0)
            
            # Add max dependency time
            dep_time = max(
                (get_task_time(dep_id, memo) for dep_id in task.dependencies),
                default=0.0
            )
            
            total_time = base_time + dep_time
            memo[task_id] = total_time
            return total_time
        
        memo = {}
        return max(
            (get_task_time(task.task_id, memo) for task in tasks),
            default=0.0
        )
    
    def _estimate_execution_cost(self, tasks: List[Task]) -> float:
        """Estimate total execution cost"""
        # Assuming free tier, cost is 0
        # But could calculate paid tier usage
        return 0.0
    
    def _assess_risk_level(
        self,
        tasks: List[Task],
        issues: List[str],
        warnings: List[str]
    ) -> str:
        """Assess overall risk level"""
        if issues:
            return "high"
        
        risk_score = 0
        
        # Complex workflows are riskier
        if len(tasks) > 20:
            risk_score += 1
        
        # Many warnings increase risk
        if len(warnings) > 5:
            risk_score += 1
        
        # Critical priority tasks are riskier
        critical_tasks = sum(1 for t in tasks if t.priority == TaskPriority.CRITICAL)
        if critical_tasks > 0:
            risk_score += 1
        
        if risk_score >= 2:
            return "medium"
        elif risk_score >= 1:
            return "low"
        else:
            return "minimal"

# ============================================================================
# TASK DISTRIBUTOR
# ============================================================================

class TaskDistributor:
    """
    Distributes tasks to optimal agents and models
    
    Features:
    - Priority-based scheduling
    - Load balancing across models
    - Dependency resolution
    - Parallel execution optimization
    """
    
    def __init__(self):
        from core_services.ai_mcp.agent_model_matcher import get_agent_model_matcher
        self.matcher = get_agent_model_matcher()
        self.active_tasks: Dict[str, Task] = {}
        self.task_queue: List[Task] = []
    
    async def distribute_workflow(
        self,
        workflow_id: str,
        tasks: List[Task]
    ) -> Dict[str, Any]:
        """
        Distribute workflow tasks for execution
        
        Returns:
            Distribution plan with execution order and model assignments
        """
        # Build dependency graph
        task_map = {task.task_id: task for task in tasks}
        
        # Topological sort for execution order
        execution_order = self._topological_sort(tasks)
        
        # Identify parallel execution opportunities
        parallel_groups = self._identify_parallel_groups(tasks, task_map)
        
        # Assign models to tasks
        model_assignments = {}
        for task in tasks:
            assignment = await self._assign_model_to_task(task)
            model_assignments[task.task_id] = assignment
        
        return {
            "workflow_id": workflow_id,
            "execution_order": execution_order,
            "parallel_groups": parallel_groups,
            "model_assignments": model_assignments,
            "estimated_parallel_speedup": self._calculate_speedup(parallel_groups)
        }
    
    def _topological_sort(self, tasks: List[Task]) -> List[str]:
        """Sort tasks by dependencies (topological order)"""
        task_map = {task.task_id: task for task in tasks}
        in_degree = {task.task_id: len(task.dependencies) for task in tasks}
        
        # Start with tasks that have no dependencies
        queue = [task_id for task_id, degree in in_degree.items() if degree == 0]
        result = []
        
        while queue:
            # Sort by priority
            queue.sort(key=lambda tid: task_map[tid].priority.value)
            task_id = queue.pop(0)
            result.append(task_id)
            
            # Reduce in-degree for dependent tasks
            for task in tasks:
                if task_id in task.dependencies:
                    in_degree[task.task_id] -= 1
                    if in_degree[task.task_id] == 0:
                        queue.append(task.task_id)
        
        return result
    
    def _identify_parallel_groups(
        self,
        tasks: List[Task],
        task_map: Dict[str, Task]
    ) -> List[List[str]]:
        """Identify tasks that can run in parallel"""
        execution_order = self._topological_sort(tasks)
        parallel_groups = []
        
        completed = set()
        
        for task_id in execution_order:
            task = task_map[task_id]
            
            # Check if all dependencies are completed
            if all(dep in completed for dep in task.dependencies):
                # Find existing group or create new one
                added = False
                for group in parallel_groups:
                    # Can add to this group if no dependencies on group members
                    group_tasks = [task_map[tid] for tid in group]
                    if not any(
                        gt.task_id in task.dependencies or task.task_id in gt.dependencies
                        for gt in group_tasks
                    ):
                        group.append(task_id)
                        added = True
                        break
                
                if not added:
                    parallel_groups.append([task_id])
                
                completed.add(task_id)
        
        return parallel_groups
    
    async def _assign_model_to_task(self, task: Task) -> Dict[str, Any]:
        """Assign optimal model to a task"""
        assignment = await self.matcher.match_agent_to_models(
            task.agent_name,
            task.description,
            task.parameters
        )
        
        return {
            "task_id": task.task_id,
            "strategy": assignment["strategy_type"],
            "models": assignment.get("phases") or [assignment.get("model")]
        }
    
    def _calculate_speedup(self, parallel_groups: List[List[str]]) -> float:
        """Calculate potential speedup from parallelization"""
        total_tasks = sum(len(group) for group in parallel_groups)
        sequential_time = total_tasks
        parallel_time = len(parallel_groups)
        
        return sequential_time / parallel_time if parallel_time > 0 else 1.0

# ============================================================================
# QUALITY ASSESSOR
# ============================================================================

class QualityAssessor:
    """
    Assesses quality of task outputs
    
    Evaluates:
    - Accuracy (correctness)
    - Completeness (all requirements met)
    - Consistency (internal logic)
    - Efficiency (resource usage)
    - Maintainability (code quality)
    """
    
    def __init__(self):
        self.quality_benchmarks = self._initialize_benchmarks()
        self.assessment_history: List[QualityMetrics] = []
    
    def _initialize_benchmarks(self) -> Dict[str, Dict[str, float]]:
        """Initialize quality benchmarks by task type"""
        return {
            "code_generation": {
                "min_accuracy": 0.8,
                "min_completeness": 0.9,
                "min_consistency": 0.85,
                "min_efficiency": 0.7,
                "min_maintainability": 0.8
            },
            "code_review": {
                "min_accuracy": 0.9,
                "min_completeness": 0.95,
                "min_consistency": 0.9,
                "min_efficiency": 0.8,
                "min_maintainability": 0.85
            },
            "data_analysis": {
                "min_accuracy": 0.9,
                "min_completeness": 0.85,
                "min_consistency": 0.9,
                "min_efficiency": 0.75,
                "min_maintainability": 0.7
            },
            "general": {
                "min_accuracy": 0.75,
                "min_completeness": 0.8,
                "min_consistency": 0.75,
                "min_efficiency": 0.7,
                "min_maintainability": 0.7
            }
        }
    
    async def assess_quality(
        self,
        task: Task,
        result: Any
    ) -> QualityMetrics:
        """
        Assess quality of task output
        
        Returns:
            QualityMetrics with scores for each dimension
        """
        task_type = task.task_type
        
        # Get appropriate assessors
        accuracy = await self._assess_accuracy(task, result)
        completeness = await self._assess_completeness(task, result)
        consistency = await self._assess_consistency(task, result)
        efficiency = await self._assess_efficiency(task, result)
        maintainability = await self._assess_maintainability(task, result)
        
        # Calculate weighted overall score
        weights = {
            "accuracy": 0.3,
            "completeness": 0.25,
            "consistency": 0.2,
            "efficiency": 0.15,
            "maintainability": 0.1
        }
        
        overall_score = (
            accuracy * weights["accuracy"] +
            completeness * weights["completeness"] +
            consistency * weights["consistency"] +
            efficiency * weights["efficiency"] +
            maintainability * weights["maintainability"]
        )
        
        metrics = QualityMetrics(
            accuracy=accuracy,
            completeness=completeness,
            consistency=consistency,
            efficiency=efficiency,
            maintainability=maintainability,
            overall_score=overall_score
        )
        
        # Record for benchmarking
        self.assessment_history.append(metrics)
        
        return metrics
    
    async def _assess_accuracy(self, task: Task, result: Any) -> float:
        """Assess accuracy/correctness"""
        if not result:
            return 0.0
        
        score = 0.8  # Base score
        
        # Code generation specific checks
        if task.task_type == "code_generation" and isinstance(result, str):
            # Check for syntax errors indicators
            error_indicators = ["error", "exception", "failed", "invalid"]
            if any(indicator in result.lower() for indicator in error_indicators):
                score -= 0.2
            
            # Check for completeness indicators
            if "def " in result or "class " in result or "function " in result:
                score += 0.1
        
        return min(1.0, max(0.0, score))
    
    async def _assess_completeness(self, task: Task, result: Any) -> float:
        """Assess if all requirements are met"""
        if not result:
            return 0.0
        
        score = 0.75  # Base score
        
        # Check against task parameters
        required_elements = task.parameters.get("required_elements", [])
        if required_elements and isinstance(result, str):
            found = sum(1 for elem in required_elements if elem.lower() in result.lower())
            score = found / len(required_elements) if required_elements else 0.75
        
        return min(1.0, score)
    
    async def _assess_consistency(self, task: Task, result: Any) -> float:
        """Assess internal consistency"""
        if not result:
            return 0.0
        
        # Basic consistency check
        score = 0.8
        
        if isinstance(result, str):
            # Check for contradictions (simple heuristic)
            lines = result.split('\n')
            if len(lines) > 10:
                # Penalize if too many TODO or FIXME comments
                todo_count = sum(1 for line in lines if 'TODO' in line or 'FIXME' in line)
                if todo_count > len(lines) * 0.1:
                    score -= 0.2
        
        return min(1.0, max(0.0, score))
    
    async def _assess_efficiency(self, task: Task, result: Any) -> float:
        """Assess resource efficiency"""
        # Based on execution time
        expected_time = {
            "code_generation": 10.0,
            "code_review": 5.0,
            "data_analysis": 15.0,
            "general": 5.0
        }.get(task.task_type, 5.0)
        
        if task.execution_time > 0:
            efficiency = expected_time / task.execution_time
            return min(1.0, efficiency)
        
        return 0.7  # Default
    
    async def _assess_maintainability(self, task: Task, result: Any) -> float:
        """Assess code/output maintainability"""
        if not result or not isinstance(result, str):
            return 0.7
        
        score = 0.7
        
        # Check for documentation
        if "\"\"\"" in result or "'''" in result or "#" in result:
            score += 0.1
        
        # Check for good structure
        if "\n\n" in result:  # Paragraphs/sections
            score += 0.1
        
        # Check length (not too long, not too short)
        length = len(result)
        if 100 < length < 10000:
            score += 0.1
        
        return min(1.0, score)
    
    def get_quality_level(self, score: float) -> QualityLevel:
        """Convert quality score to level"""
        if score >= 0.9:
            return QualityLevel.EXCELLENT
        elif score >= 0.75:
            return QualityLevel.GOOD
        elif score >= 0.6:
            return QualityLevel.ACCEPTABLE
        elif score >= 0.4:
            return QualityLevel.POOR
        else:
            return QualityLevel.UNACCEPTABLE
    
    def meets_benchmark(
        self,
        metrics: QualityMetrics,
        task_type: str
    ) -> Dict[str, Any]:
        """Check if quality meets benchmark"""
        benchmarks = self.quality_benchmarks.get(task_type, self.quality_benchmarks["general"])
        
        results = {
            "meets_benchmark": True,
            "failures": []
        }
        
        checks = {
            "accuracy": (metrics.accuracy, benchmarks["min_accuracy"]),
            "completeness": (metrics.completeness, benchmarks["min_completeness"]),
            "consistency": (metrics.consistency, benchmarks["min_consistency"]),
            "efficiency": (metrics.efficiency, benchmarks["min_efficiency"]),
            "maintainability": (metrics.maintainability, benchmarks["min_maintainability"])
        }
        
        for dimension, (actual, required) in checks.items():
            if actual < required:
                results["meets_benchmark"] = False
                results["failures"].append({
                    "dimension": dimension,
                    "actual": actual,
                    "required": required,
                    "gap": required - actual
                })
        
        return results

# ============================================================================
# OUTCOME VALIDATOR
# ============================================================================

class OutcomeValidator:
    """
    Validates task outcomes
    
    Performs:
    - Result validation
    - Quality assessment
    - Security check
    - Benchmark comparison
    """
    
    def __init__(self):
        from core_services.ai_mcp.security.security_manager import get_security_manager
        self.quality_assessor = QualityAssessor()
        self.security_manager = get_security_manager()
    
    async def validate_outcome(
        self,
        task: Task,
        result: Any,
        model_used: str,
        execution_time: float
    ) -> TaskOutcome:
        """
        Comprehensive outcome validation
        
        Returns:
            TaskOutcome with all validation results
        """
        # Quality assessment
        quality_metrics = await self.quality_assessor.assess_quality(task, result)
        
        # Quality validation
        quality_check = self.quality_assessor.meets_benchmark(
            quality_metrics,
            task.task_type
        )
        
        # Security validation
        security_check = await self._validate_security(result)
        
        # Format validation
        format_check = await self._validate_format(task, result)
        
        # Combine validations
        validation_passed = (
            quality_check["meets_benchmark"] and
            format_check["valid"]
        )
        
        security_passed = security_check["safe"]
        
        success = validation_passed and security_passed
        
        return TaskOutcome(
            task_id=task.task_id,
            success=success,
            result=result,
            quality_metrics=quality_metrics,
            execution_time=execution_time,
            model_used=model_used,
            validation_passed=validation_passed,
            validation_details={
                "quality_check": quality_check,
                "format_check": format_check
            },
            security_passed=security_passed,
            security_details=security_check
        )
    
    async def _validate_security(self, result: Any) -> Dict[str, Any]:
        """Validate result security"""
        if not result:
            return {"safe": True, "issues": []}
        
        result_str = str(result)
        validation = self.security_manager.validate_output(result_str)
        
        return {
            "safe": validation["valid"],
            "issues": validation["issues"],
            "severity": validation["severity"]
        }
    
    async def _validate_format(self, task: Task, result: Any) -> Dict[str, Any]:
        """Validate result format"""
        expected_format = task.parameters.get("expected_format", "text")
        
        if expected_format == "code":
            if not isinstance(result, str):
                return {"valid": False, "reason": "Expected string result"}
            
            # Check for code indicators
            has_code = any(indicator in result for indicator in ["def ", "class ", "function ", "import ", "from "])
            if not has_code:
                return {"valid": False, "reason": "No code detected in result"}
        
        elif expected_format == "json":
            if isinstance(result, str):
                try:
                    json.loads(result)
                except:
                    return {"valid": False, "reason": "Invalid JSON format"}
        
        return {"valid": True, "reason": "Format validation passed"}

# ============================================================================
# CONTINUOUS MONITOR
# ============================================================================

class ContinuousMonitor:
    """
    Continuously monitors quality and security
    
    Features:
    - Real-time quality tracking
    - Security event monitoring
    - Anomaly detection
    - Automated alerts
    - Trend analysis
    """
    
    def __init__(self):
        self.quality_history: List[Dict[str, Any]] = []
        self.security_events: List[Dict[str, Any]] = []
        self.alert_thresholds = self._initialize_thresholds()
        self.is_monitoring = False
        self._monitor_task = None
    
    def _initialize_thresholds(self) -> Dict[str, float]:
        """Initialize alert thresholds"""
        return {
            "min_quality_score": 0.75,
            "max_error_rate": 0.1,
            "max_security_events_per_hour": 10,
            "min_success_rate": 0.9
        }
    
    async def start_monitoring(self):
        """Start continuous monitoring"""
        if self.is_monitoring:
            logger.warning("Monitor already running")
            return
        
        self.is_monitoring = True
        self._monitor_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Continuous monitoring started")
    
    async def stop_monitoring(self):
        """Stop continuous monitoring"""
        self.is_monitoring = False
        if self._monitor_task:
            self._monitor_task.cancel()
        logger.info("Continuous monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                # Check quality trends
                await self._check_quality_trends()
                
                # Check security
                await self._check_security_status()
                
                # Check performance
                await self._check_performance()
                
                # Generate alerts if needed
                await self._generate_alerts()
                
                # Wait before next check
                await asyncio.sleep(60)  # Check every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
    
    async def _check_quality_trends(self):
        """Check quality score trends"""
        if len(self.quality_history) < 10:
            return
        
        recent = self.quality_history[-10:]
        avg_quality = sum(item["quality_score"] for item in recent) / len(recent)
        
        if avg_quality < self.alert_thresholds["min_quality_score"]:
            await self._raise_alert(
                "quality_degradation",
                f"Average quality score ({avg_quality:.2f}) below threshold ({self
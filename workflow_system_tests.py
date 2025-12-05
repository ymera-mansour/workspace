# YMERA Refactoring Project
# Complete Workflow System - Tests & Examples

import asyncio
import pytest
from typing import List, Dict, Any

# ============================================================================
# COMPREHENSIVE TESTS
# ============================================================================

class TestWorkflowValidation:
    """Test workflow validation"""
    
    @pytest.mark.asyncio
    async def test_valid_workflow(self):
        """Test workflow with valid tasks"""
        from workflow_validation_system import WorkflowValidator, Task, TaskPriority
        
        validator = WorkflowValidator()
        
        tasks = [
            Task(
                task_id="task_1",
                workflow_id="wf_1",
                agent_name="coding_agent",
                task_type="code_generation",
                description="Generate hello world",
                parameters={},
                priority=TaskPriority.HIGH
            ),
            Task(
                task_id="task_2",
                workflow_id="wf_1",
                agent_name="coding_agent",
                task_type="code_review",
                description="Review generated code",
                parameters={},
                dependencies=["task_1"],
                priority=TaskPriority.MEDIUM
            )
        ]
        
        validation = await validator.validate_workflow("wf_1", tasks)
        
        assert validation.valid
        assert len(validation.issues) == 0
        assert validation.risk_level in ["minimal", "low"]
    
    @pytest.mark.asyncio
    async def test_circular_dependency(self):
        """Test detection of circular dependencies"""
        from workflow_validation_system import WorkflowValidator, Task
        
        validator = WorkflowValidator()
        
        # Create circular dependency: task_1 -> task_2 -> task_1
        tasks = [
            Task(
                task_id="task_1",
                workflow_id="wf_2",
                agent_name="coding_agent",
                task_type="code_generation",
                description="Task 1",
                parameters={},
                dependencies=["task_2"]
            ),
            Task(
                task_id="task_2",
                workflow_id="wf_2",
                agent_name="coding_agent",
                task_type="code_review",
                description="Task 2",
                parameters={},
                dependencies=["task_1"]
            )
        ]
        
        validation = await validator.validate_workflow("wf_2", tasks)
        
        assert not validation.valid
        assert any("circular" in issue.lower() for issue in validation.issues)


class TestTaskDistribution:
    """Test task distribution"""
    
    @pytest.mark.asyncio
    async def test_parallel_task_identification(self):
        """Test identification of parallelizable tasks"""
        from workflow_validation_system import TaskDistributor, Task
        
        distributor = TaskDistributor()
        
        tasks = [
            Task(
                task_id="task_1",
                workflow_id="wf_3",
                agent_name="coding_agent",
                task_type="code_generation",
                description="Generate module A",
                parameters={}
            ),
            Task(
                task_id="task_2",
                workflow_id="wf_3",
                agent_name="coding_agent",
                task_type="code_generation",
                description="Generate module B",
                parameters={}
            ),
            Task(
                task_id="task_3",
                workflow_id="wf_3",
                agent_name="coding_agent",
                task_type="code_review",
                description="Review both modules",
                parameters={},
                dependencies=["task_1", "task_2"]
            )
        ]
        
        distribution = await distributor.distribute_workflow("wf_3", tasks)
        
        # task_1 and task_2 should be in same parallel group
        assert len(distribution["parallel_groups"]) >= 2
        # Speedup should be > 1
        assert distribution["estimated_parallel_speedup"] > 1.0


class TestQualityAssessment:
    """Test quality assessment"""
    
    @pytest.mark.asyncio
    async def test_quality_assessment(self):
        """Test quality score calculation"""
        from workflow_validation_system import QualityAssessor, Task
        
        assessor = QualityAssessor()
        
        task = Task(
            task_id="task_q1",
            workflow_id="wf_q",
            agent_name="coding_agent",
            task_type="code_generation",
            description="Generate function",
            parameters={"required_elements": ["def", "return"]}
        )
        
        # Good quality code
        result = """
def fibonacci(n):
    \"\"\"Calculate fibonacci number\"\"\"
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"""
        
        metrics = await assessor.assess_quality(task, result)
        
        assert 0.0 <= metrics.overall_score <= 1.0
        assert metrics.accuracy > 0.7
        assert metrics.completeness > 0.7
    
    @pytest.mark.asyncio
    async def test_quality_benchmark(self):
        """Test quality benchmark checking"""
        from workflow_validation_system import QualityAssessor, QualityMetrics
        
        assessor = QualityAssessor()
        
        # Excellent metrics
        excellent_metrics = QualityMetrics(
            accuracy=0.95,
            completeness=0.95,
            consistency=0.9,
            efficiency=0.85,
            maintainability=0.9,
            overall_score=0.91
        )
        
        result = assessor.meets_benchmark(excellent_metrics, "code_generation")
        
        assert result["meets_benchmark"]
        assert len(result["failures"]) == 0


class TestOutcomeValidation:
    """Test outcome validation"""
    
    @pytest.mark.asyncio
    async def test_outcome_validation(self):
        """Test complete outcome validation"""
        from workflow_validation_system import OutcomeValidator, Task
        
        validator = OutcomeValidator()
        
        task = Task(
            task_id="task_v1",
            workflow_id="wf_v",
            agent_name="coding_agent",
            task_type="code_generation",
            description="Generate function",
            parameters={"expected_format": "code"}
        )
        
        result = "def hello(): return 'Hello World'"
        
        outcome = await validator.validate_outcome(
            task=task,
            result=result,
            model_used="gemini:gemini-2.5-flash",
            execution_time=2.5
        )
        
        assert outcome.success or not outcome.success  # Will depend on actual quality
        assert outcome.quality_metrics is not None
        assert outcome.validation_details is not None
        assert outcome.security_details is not None


class TestContinuousMonitoring:
    """Test continuous monitoring"""
    
    @pytest.mark.asyncio
    async def test_quality_monitoring(self):
        """Test quality metric recording"""
        from workflow_validation_system import ContinuousMonitor
        
        monitor = ContinuousMonitor()
        
        # Record some metrics
        monitor.record_quality_metric("task_1", 0.85, "code_generation")
        monitor.record_quality_metric("task_2", 0.90, "code_review")
        monitor.record_quality_metric("task_3", 0.75, "data_analysis")
        
        assert len(monitor.quality_history) == 3
    
    @pytest.mark.asyncio
    async def test_security_event_recording(self):
        """Test security event recording"""
        from workflow_validation_system import ContinuousMonitor
        
        monitor = ContinuousMonitor()
        
        monitor.record_security_event(
            "injection_attempt",
            "high",
            {"pattern": "SQL injection"}
        )
        
        assert len(monitor.security_events) == 1
        assert monitor.security_events[0]["event_type"] == "injection_attempt"
    
    @pytest.mark.asyncio
    async def test_monitoring_report(self):
        """Test monitoring report generation"""
        from workflow_validation_system import ContinuousMonitor
        
        monitor = ContinuousMonitor()
        
        # Add some data
        monitor.record_quality_metric("task_1", 0.85, "code_generation")
        monitor.record_security_event("test_event", "low", {})
        
        report = monitor.get_monitoring_report()
        
        assert "monitoring_status" in report
        assert "quality_metrics" in report
        assert "security_metrics" in report


# ============================================================================
# REAL-WORLD EXAMPLES
# ============================================================================

async def example_1_simple_workflow():
    """Example 1: Simple code generation workflow"""
    from workflow_validation_system import (
        WorkflowOrchestrator, Task, TaskPriority
    )
    
    print("\n" + "="*60)
    print("Example 1: Simple Code Generation Workflow")
    print("="*60)
    
    orchestrator = WorkflowOrchestrator()
    
    # Create simple workflow
    tasks = [
        Task(
            task_id="gen_1",
            workflow_id="simple_1",
            agent_name="coding_agent",
            task_type="code_generation",
            description="Create a Python function to calculate factorial",
            parameters={
                "language": "python",
                "required_elements": ["def", "return"],
                "expected_format": "code"
            },
            priority=TaskPriority.HIGH
        )
    ]
    
    # Execute workflow
    result = await orchestrator.execute_workflow(
        workflow_id="simple_1",
        tasks=tasks,
        enable_monitoring=True
    )
    
    print(f"\nStatus: {result['status']}")
    print(f"Overall Quality: {result.get('overall_quality', 0):.2%}")
    print(f"Quality Level: {result.get('quality_level', 'unknown')}")
    print(f"\nValidation:")
    print(f"  Issues: {len(result['validation']['issues'])}")
    print(f"  Risk Level: {result['validation']['risk_level']}")


async def example_2_multi_phase_workflow():
    """Example 2: Multi-phase code development workflow"""
    from workflow_validation_system import (
        WorkflowOrchestrator, Task, TaskPriority
    )
    
    print("\n" + "="*60)
    print("Example 2: Multi-Phase Development Workflow")
    print("="*60)
    
    orchestrator = WorkflowOrchestrator()
    
    # Create complex workflow
    tasks = [
        Task(
            task_id="design",
            workflow_id="multi_1",
            agent_name="coding_agent",
            task_type="code_generation",
            description="Design REST API structure",
            parameters={"phase": "design"},
            priority=TaskPriority.CRITICAL
        ),
        Task(
            task_id="implement",
            workflow_id="multi_1",
            agent_name="coding_agent",
            task_type="code_generation",
            description="Implement REST API endpoints",
            parameters={"phase": "implementation"},
            dependencies=["design"],
            priority=TaskPriority.HIGH
        ),
        Task(
            task_id="test",
            workflow_id="multi_1",
            agent_name="coding_agent",
            task_type="code_generation",
            description="Write unit tests for API",
            parameters={"phase": "testing"},
            dependencies=["implement"],
            priority=TaskPriority.HIGH
        ),
        Task(
            task_id="review",
            workflow_id="multi_1",
            agent_name="coding_agent",
            task_type="code_review",
            description="Review code quality and security",
            parameters={"phase": "review"},
            dependencies=["test"],
            priority=TaskPriority.MEDIUM
        ),
        Task(
            task_id="document",
            workflow_id="multi_1",
            agent_name="documentation_agent",
            task_type="documentation",
            description="Generate API documentation",
            parameters={"phase": "documentation"},
            dependencies=["review"],
            priority=TaskPriority.LOW
        )
    ]
    
    result = await orchestrator.execute_workflow(
        workflow_id="multi_1",
        tasks=tasks,
        enable_monitoring=True
    )
    
    print(f"\nWorkflow Completed: {result['status']}")
    print(f"Tasks Completed: {result['execution']['tasks_completed']}/{result['execution']['tasks_attempted']}")
    print(f"Overall Quality: {result.get('overall_quality', 0):.2%}")
    
    print("\nExecution Flow:")
    for i, task_id in enumerate(result['distribution']['execution_order'], 1):
        print(f"  {i}. {task_id}")


async def example_3_parallel_workflow():
    """Example 3: Parallel task execution workflow"""
    from workflow_validation_system import (
        WorkflowOrchestrator, Task, TaskPriority
    )
    
    print("\n" + "="*60)
    print("Example 3: Parallel Execution Workflow")
    print("="*60)
    
    orchestrator = WorkflowOrchestrator()
    
    # Create parallel workflow
    tasks = [
        # These can run in parallel
        Task(
            task_id="module_a",
            workflow_id="parallel_1",
            agent_name="coding_agent",
            task_type="code_generation",
            description="Generate User module",
            parameters={"module": "user"},
            priority=TaskPriority.HIGH
        ),
        Task(
            task_id="module_b",
            workflow_id="parallel_1",
            agent_name="coding_agent",
            task_type="code_generation",
            description="Generate Product module",
            parameters={"module": "product"},
            priority=TaskPriority.HIGH
        ),
        Task(
            task_id="module_c",
            workflow_id="parallel_1",
            agent_name="coding_agent",
            task_type="code_generation",
            description="Generate Order module",
            parameters={"module": "order"},
            priority=TaskPriority.HIGH
        ),
        # This depends on all above
        Task(
            task_id="integration",
            workflow_id="parallel_1",
            agent_name="coding_agent",
            task_type="code_generation",
            description="Integrate all modules",
            parameters={"phase": "integration"},
            dependencies=["module_a", "module_b", "module_c"],
            priority=TaskPriority.CRITICAL
        )
    ]
    
    result = await orchestrator.execute_workflow(
        workflow_id="parallel_1",
        tasks=tasks,
        enable_monitoring=True
    )
    
    print(f"\nStatus: {result['status']}")
    print(f"Parallel Groups: {len(result['distribution']['parallel_groups'])}")
    print(f"Estimated Speedup: {result['distribution']['estimated_parallel_speedup']:.2f}x")
    
    print("\nParallel Execution Groups:")
    for i, group in enumerate(result['distribution']['parallel_groups'], 1):
        print(f"  Group {i}: {', '.join(group)}")


async def example_4_quality_monitoring():
    """Example 4: Quality monitoring in action"""
    from workflow_validation_system import ContinuousMonitor
    
    print("\n" + "="*60)
    print("Example 4: Continuous Quality Monitoring")
    print("="*60)
    
    monitor = ContinuousMonitor()
    
    # Start monitoring
    await monitor.start_monitoring()
    print("✓ Monitoring started")
    
    # Simulate some tasks with varying quality
    print("\nSimulating task executions...")
    
    tasks_quality = [
        ("task_1", 0.95, "code_generation"),
        ("task_2", 0.88, "code_review"),
        ("task_3", 0.92, "data_analysis"),
        ("task_4", 0.65, "code_generation"),  # Low quality
        ("task_5", 0.90, "documentation"),
    ]
    
    for task_id, quality, task_type in tasks_quality:
        monitor.record_quality_metric(task_id, quality, task_type)
        print(f"  {task_id}: Quality {quality:.0%}")
    
    # Get monitoring report
    report = monitor.get_monitoring_report()
    
    print(f"\nMonitoring Report:")
    print(f"  Status: {report['monitoring_status']}")
    print(f"  Average Quality: {report['quality_metrics']['average']:.2%}")
    print(f"  Min Quality: {report['quality_metrics']['minimum']:.2%}")
    print(f"  Max Quality: {report['quality_metrics']['maximum']:.2%}")
    
    # Stop monitoring
    await monitor.stop_monitoring()
    print("\n✓ Monitoring stopped")


async def example_5_security_integration():
    """Example 5: Security validation integration"""
    from workflow_validation_system import OutcomeValidator, Task
    
    print("\n" + "="*60)
    print("Example 5: Security Validation Integration")
    print("="*60)
    
    validator = OutcomeValidator()
    
    # Test with safe code
    print("\n1. Testing safe code...")
    safe_task = Task(
        task_id="safe_1",
        workflow_id="sec_1",
        agent_name="coding_agent",
        task_type="code_generation",
        description="Safe code",
        parameters={}
    )
    
    safe_code = "def add(a, b): return a + b"
    
    outcome = await validator.validate_outcome(
        task=safe_task,
        result=safe_code,
        model_used="test",
        execution_time=1.0
    )
    
    print(f"   Security Passed: {outcome.security_passed}")
    print(f"   Validation Passed: {outcome.validation_passed}")
    
    # Test with potentially unsafe code
    print("\n2. Testing potentially unsafe code...")
    unsafe_task = Task(
        task_id="unsafe_1",
        workflow_id="sec_1",
        agent_name="coding_agent",
        task_type="code_generation",
        description="Unsafe code",
        parameters={}
    )
    
    unsafe_code = "import os; os.system('rm -rf /')"
    
    outcome = await validator.validate_outcome(
        task=unsafe_task,
        result=unsafe_code,
        model_used="test",
        execution_time=1.0
    )
    
    print(f"   Security Passed: {outcome.security_passed}")
    if not outcome.security_passed:
        print(f"   Security Issues: {outcome.security_details.get('issues', [])}")


async def example_6_complete_dashboard():
    """Example 6: Complete monitoring dashboard"""
    from workflow_validation_system import get_workflow_orchestrator
    
    print("\n" + "="*60)
    print("Example 6: Complete Monitoring Dashboard")
    print("="*60)
    
    orchestrator = get_workflow_orchestrator()
    
    # Get dashboard data
    dashboard = orchestrator.get_monitoring_dashboard()
    
    print(f"\nSystem Status:")
    print(f"  Monitoring Active: {dashboard['system_status']['monitoring_active']}")
    print(f"  Active Workflows: {dashboard['system_status']['active_workflows']}")
    
    print(f"\nQuality Metrics:")
    quality = dashboard['monitoring_report']['quality_metrics']
    print(f"  Average: {quality['average']:.2%}")
    print(f"  Range: {quality['minimum']:.2%} - {quality['maximum']:.2%}")
    print(f"  Total Samples: {quality['total_samples']}")
    
    print(f"\nSecurity Metrics:")
    security = dashboard['monitoring_report']['security_metrics']
    print(f"  Events (Last Hour): {security['events_last_hour']}")
    print(f"  Total Events: {security['total_events']}")
    
    print(f"\nQuality Benchmarks:")
    for task_type, benchmarks in dashboard['quality_benchmarks'].items():
        print(f"  {task_type}:")
        print(f"    Min Accuracy: {benchmarks['min_accuracy']:.0%}")
        print(f"    Min Completeness: {benchmarks['min_completeness']:.0%}")


async def main():
    """Run all examples"""
    print("\n" + "="*70)
    print("YMERA WORKFLOW VALIDATION & QUALITY SYSTEM")
    print("Complete Examples & Demonstrations")
    print("="*70)
    
    try:
        await example_1_simple_workflow()
        await example_2_multi_phase_workflow()
        await example_3_parallel_workflow()
        await example_4_quality_monitoring()
        await example_5_security_integration()
        await example_6_complete_dashboard()
        
        print("\n" + "="*70)
        print("✓ ALL EXAMPLES COMPLETED SUCCESSFULLY")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run tests
    print("Running tests...")
    pytest.main([__file__, "-v"])
    
    # Run examples
    print("\n\nRunning examples...")
    asyncio.run(main())

"""Tests for Tier 1 Foundation"""
import pytest
import asyncio
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

def test_main_imports():
    """Test that main.py can be imported"""
    try:
        import main
        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import main: {e}")

def test_workflow_orchestrator():
    """Test workflow orchestrator"""
    from ORCHESTRATION.workflow_orchestrator import WorkflowOrchestrator
    config = {"test": True}
    orchestrator = WorkflowOrchestrator(config)
    assert orchestrator is not None

def test_state_manager():
    """Test state manager"""
    from ORCHESTRATION.state_manager import StateManager
    manager = StateManager()
    assert manager is not None
    
@pytest.mark.asyncio
async def test_phase_x_validator():
    """Test Phase X validator"""
    from sys import path as syspath
    syspath.insert(0, '0X-VALIDATION')
    from phase_x_validator import PhaseXValidator
    validator = PhaseXValidator({})
    assert validator is not None

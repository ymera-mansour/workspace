"""Tests for Tier 3 Phase 1 Discovery"""
import pytest
import asyncio

@pytest.mark.asyncio
async def test_phase1_orchestrator():
    """Test Phase 1 orchestrator"""
    import sys
    sys.path.insert(0, '01-DISCOVERY')
    from phase1_orchestrator import Phase1Orchestrator
    orchestrator = Phase1Orchestrator({})
    assert orchestrator is not None

def test_silent_monitor():
    """Test silent monitor"""
    import sys
    sys.path.insert(0, '01-DISCOVERY')
    from silent_monitor import SilentMonitor
    monitor = SilentMonitor()
    assert monitor is not None

def test_grading_system():
    """Test grading system"""
    import sys
    sys.path.insert(0, '01-DISCOVERY')
    from grading_system import GradingSystem
    grader = GradingSystem()
    grade = grader.calculate_grade(9.5)
    assert grade == "A+"

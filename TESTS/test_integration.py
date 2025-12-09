"""Integration tests for complete workflow"""
import pytest
import asyncio

@pytest.mark.asyncio
async def test_complete_workflow_structure():
    """Test that all phase orchestrators exist"""
    import os
    assert os.path.exists('01-DISCOVERY/phase1_orchestrator.py')
    assert os.path.exists('02-ANALYSIS/phase2_orchestrator.py')
    assert os.path.exists('03-CONSOLIDATION/phase3_orchestrator.py')
    assert os.path.exists('04-TESTING/phase4_orchestrator.py')
    assert os.path.exists('05-INTEGRATION/phase5_orchestrator.py')

def test_all_validators_exist():
    """Test that all validator directories exist"""
    import os
    assert os.path.exists('01-DISCOVERY/validators')
    assert os.path.exists('02-ANALYSIS/validators')
    assert os.path.exists('03-CONSOLIDATION/validators')
    assert os.path.exists('04-TESTING/validators')
    assert os.path.exists('05-INTEGRATION/validators')

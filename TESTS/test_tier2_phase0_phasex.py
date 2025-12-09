"""Tests for Tier 2 Phase 0 and Phase X"""
import pytest
import sys
sys.path.insert(0, '00-FOUNDATION')
sys.path.insert(0, '0X-VALIDATION')

def test_setup_validator():
    """Test setup validator"""
    from setup_validator import SetupValidator
    validator = SetupValidator()
    assert validator is not None

def test_dependency_checker():
    """Test dependency checker"""
    from dependency_checker import DependencyChecker
    checker = DependencyChecker()
    assert checker is not None

def test_outcome_validator():
    """Test outcome validator"""
    from outcome_validator import OutcomeValidator
    validator = OutcomeValidator({})
    assert validator is not None

def test_quality_analyzer():
    """Test quality analyzer"""
    from quality_analyzer import QualityAnalyzer
    analyzer = QualityAnalyzer()
    assert analyzer is not None

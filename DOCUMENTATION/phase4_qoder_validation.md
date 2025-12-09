========================================
PHASE 4 - QODER: VALIDATION & TESTING
========================================

=== YOUR IDENTITY ===
Your name: QODER
Your role: Quality assurance and validation engineer
Your phase: 4
Your workspace: C:\Users\Mohamed Mansour\Desktop\YmeraRefactor\

=== CONTEXT FROM PREVIOUS PHASES ===
✅ Phase 1A: shared\ library created (You!)
✅ Phase 1B: Master test plan documented (Gemini)
✅ Phase 1C: Test framework installed (Claude)
✅ Phase 2: core_services\ created (You!)
✅ Phase 3A: agents\ refactored (You!)
✅ Phase 3B: Tests implemented (You!)
✅ Phase 3C: Documentation created (Gemini)

Current complete structure:
```
YmeraRefactor\
├── shared\           ✅ Complete
├── core_services\    ✅ Complete
├── agents\           ✅ Complete
├── tests\            ✅ Complete (24 tests implemented)
└── docs\             ✅ Complete
```

=== YOUR MISSION ===
Validate the entire system by:
1. Running all tests and ensuring they pass
2. Debugging and fixing any failures
3. Performance testing and optimization
4. Integration verification
5. Code quality checks
6. Security audits
7. Final system validation

This is the **FINAL VALIDATION PHASE** before production readiness.

=== STEP-BY-STEP INSTRUCTIONS ===

## STEP 1: ENVIRONMENT SETUP VERIFICATION (10 minutes)

### 1.1 Verify Python Environment
```bash
python --version
# Should be Python 3.9+

python -c "import sys; print(sys.version)"
```

### 1.2 Verify All Dependencies Installed
```bash
# Install all requirements
pip install -r shared/requirements.txt
pip install -r core_services/requirements.txt
pip install -r agents/requirements.txt
pip install -r tests/requirements.txt

# Verify installations
python -c "import pytest; print('pytest:', pytest.__version__)"
python -c "import asyncio; print('asyncio: OK')"
python -c "import aiohttp; print('aiohttp: OK')"
```

### 1.3 Verify Directory Structure
```bash
# Check all directories exist
python -c "
import os
from pathlib import Path

dirs = [
    'shared', 'shared/config', 'shared/database',
    'core_services', 'core_services/agent_manager', 
    'core_services/engines', 'core_services/ai_mcp',
    'agents', 'agents/base', 'agents/coding',
    'tests', 'tests/unit', 'tests/integration', 'tests/e2e',
    'docs'
]

for d in dirs:
    assert Path(d).exists(), f'Missing: {d}'
    
print('✅ All directories exist')
"
```

### 1.4 Create Validation Report Directory
```bash
mkdir -p _reports/validation
```

## STEP 2: RUN FULL TEST SUITE (20 minutes)

### 2.1 Run Unit Tests First
```bash
# Run unit tests with verbose output
python -m pytest tests/unit/ -v --tb=short

# Expected output:
# tests/unit/test_shared_config.py::test_load_environment_variables PASSED
# tests/unit/test_shared_config.py::test_get_config_with_defaults PASSED
# ... (15 total)
# =============== 15 passed in X.XXs ===============
```

**If any fail:**
- Read error messages carefully
- Fix the underlying code
- Re-run failed tests: `pytest tests/unit/test_file.py::test_name -v`

### 2.2 Run Integration Tests
```bash
# Run integration tests
python -m pytest tests/integration/ -v --tb=short

# Expected: 4 passed
```

**If any fail:**
- Check if services are properly initialized
- Verify imports are correct
- Check if engines are available

### 2.3 Run E2E Tests
```bash
# Run E2E tests (these may be slower)
python -m pytest tests/e2e/ -v --tb=short -m e2e

# Expected: 5 passed
```

**If any fail:**
- Check full integration chain
- Verify AI providers are configured (if testing AI features)
- Check database connections

### 2.4 Run Complete Test Suite with Coverage
```bash
# Run all tests with coverage report
python -m pytest tests/ -v \
    --cov=shared \
    --cov=core_services \
    --cov=agents \
    --cov-report=html \
    --cov-report=term-missing

# Save output
python -m pytest tests/ -v --cov=shared --cov=core_services --cov=agents --cov-report=html > _reports/validation/test_results.txt 2>&1
```

**Target Metrics:**
- ✅ All 24 tests passing
- ✅ Coverage > 80%
- ✅ No critical errors
- ✅ Execution time < 30 seconds

### 2.5 Generate Test Report
```bash
# Create test summary
python -m pytest tests/ --html=_reports/validation/test_report.html --self-contained-html

# Create JSON report
python -m pytest tests/ --json-report --json-report-file=_reports/validation/test_report.json
```

## STEP 3: DEBUG FAILING TESTS (Variable time)

For each failing test:

### 3.1 Identify the Failure
```bash
# Run specific failing test with maximum verbosity
pytest tests/path/to/test.py::test_name -vvv --tb=long
```

### 3.2 Common Issues and Fixes

#### Issue: Import Errors
```python
# Error: ModuleNotFoundError: No module named 'shared'

# Fix: Add to conftest.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
```

#### Issue: Async Errors
```python
# Error: RuntimeError: Event loop is closed

# Fix: Use pytest-asyncio correctly
@pytest.mark.asyncio
async def test_something():
    result = await async_function()
    assert result is not None
```

#### Issue: Database Connection Errors
```python
# Error: Connection refused

# Fix: Use in-memory database for tests
db = DatabaseManager(db_type="sqlite", db_path=":memory:")
```

#### Issue: Timeout Errors
```python
# Error: asyncio.TimeoutError

# Fix: Increase timeout or optimize code
config = AgentConfig(
    name="test",
    version="1.0",
    description="test",
    capabilities=[],
    timeout=600  # Increase from 300 to 600
)
```

### 3.3 Document Fixes
Create: `_reports/validation/fixes_applied.md`
```markdown
# Test Fixes Applied

## Issue 1: Import Error in test_shared_config.py
**Error**: ModuleNotFoundError: No module named 'shared'
**Fix**: Updated conftest.py to add shared/ to sys.path
**Status**: ✅ FIXED

## Issue 2: [Description]
**Error**: [Error message]
**Fix**: [What was done]
**Status**: [✅ FIXED | ⏳ IN PROGRESS | ❌ BLOCKED]
```

## STEP 4: PERFORMANCE TESTING (15 minutes)

### 4.1 Create Performance Test Script

**File: _reports/validation/performance_test.py**
```python
# YMERA Refactoring Project
# Phase: 4 | Agent: qoder | Created: 2024-11-30
# Performance testing script

import asyncio
import time
from typing import List, Dict
import statistics

from agents.coding.coding_agent import CodingAgent
from agents.base.base_agent import AgentRequest

async def measure_execution_time(agent, request):
    """Measure single execution time"""
    start = time.time()
    response = await agent.execute(request)
    end = time.time()
    return end - start, response.status

async def run_performance_tests():
    """Run performance benchmarks"""
    results = {
        "single_execution": [],
        "concurrent_execution": [],
        "sequential_execution": []
    }
    
    # Test 1: Single execution performance
    print("Test 1: Single Execution Performance")
    agent = CodingAgent()
    await agent.initialize()
    
    for i in range(5):
        request = AgentRequest(
            task_id=f"perf_single_{i}",
            task_type="code_generation",
            parameters={
                "task_type": "code_generation",
                "language": "python",
                "prompt": "Create a simple hello world function"
            }
        )
        exec_time, status = await measure_execution_time(agent, request)
        results["single_execution"].append(exec_time)
        print(f"  Run {i+1}: {exec_time:.3f}s - {status}")
    
    await agent.shutdown()
    
    # Test 2: Concurrent execution
    print("\nTest 2: Concurrent Execution (3 agents)")
    start = time.time()
    
    agents = [CodingAgent() for _ in range(3)]
    await asyncio.gather(*[agent.initialize() for agent in agents])
    
    requests = [
        AgentRequest(
            task_id=f"perf_concurrent_{i}",
            task_type="code_generation",
            parameters={
                "task_type": "code_generation",
                "language": "python",
                "prompt": f"Create function {i}"
            }
        )
        for i in range(3)
    ]
    
    responses = await asyncio.gather(
        *[agents[i].execute(requests[i]) for i in range(3)]
    )
    
    end = time.time()
    concurrent_time = end - start
    results["concurrent_execution"].append(concurrent_time)
    print(f"  Total time: {concurrent_time:.3f}s")
    print(f"  Avg per agent: {concurrent_time/3:.3f}s")
    
    await asyncio.gather(*[agent.shutdown() for agent in agents])
    
    # Test 3: Sequential execution
    print("\nTest 3: Sequential Execution (3 requests)")
    agent = CodingAgent()
    await agent.initialize()
    
    start = time.time()
    for i in range(3):
        request = AgentRequest(
            task_id=f"perf_seq_{i}",
            task_type="code_generation",
            parameters={
                "task_type": "code_generation",
                "language": "python",
                "prompt": f"Create function {i}"
            }
        )
        await agent.execute(request)
    end = time.time()
    
    sequential_time = end - start
    results["sequential_execution"].append(sequential_time)
    print(f"  Total time: {sequential_time:.3f}s")
    print(f"  Avg per request: {sequential_time/3:.3f}s")
    
    await agent.shutdown()
    
    # Print summary
    print("\n" + "="*50)
    print("PERFORMANCE SUMMARY")
    print("="*50)
    
    single_avg = statistics.mean(results["single_execution"])
    single_min = min(results["single_execution"])
    single_max = max(results["single_execution"])
    
    print(f"\nSingle Execution:")
    print(f"  Average: {single_avg:.3f}s")
    print(f"  Min: {single_min:.3f}s")
    print(f"  Max: {single_max:.3f}s")
    
    print(f"\nConcurrent (3 agents): {concurrent_time:.3f}s")
    print(f"Sequential (3 requests): {sequential_time:.3f}s")
    print(f"Speedup: {sequential_time/concurrent_time:.2f}x")
    
    # Performance targets
    print("\n" + "="*50)
    print("PERFORMANCE TARGETS")
    print("="*50)
    targets = {
        "Single execution": (single_avg, 5.0, "< 5s"),
        "Concurrent speedup": (sequential_time/concurrent_time, 2.0, "> 2x"),
    }
    
    all_passed = True
    for name, (actual, target, description) in targets.items():
        passed = actual <= target if "execution" in name else actual >= target
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{name}: {actual:.2f} (target: {description}) - {status}")
        if not passed:
            all_passed = False
    
    return all_passed

if __name__ == "__main__":
    result = asyncio.run(run_performance_tests())
    exit(0 if result else 1)
```

### 4.2 Run Performance Tests
```bash
cd C:\Users\Mohamed Mansour\Desktop\YmeraRefactor
python _reports/validation/performance_test.py > _reports/validation/performance_results.txt 2>&1
```

**Analyze Results:**
- Single execution should be < 5 seconds
- Concurrent execution should show speedup
- No memory leaks
- Reasonable resource usage

## STEP 5: INTEGRATION VERIFICATION (15 minutes)

### 5.1 Verify Component Integration

**File: _reports/validation/integration_check.py**
```python
# YMERA Refactoring Project
# Phase: 4 | Agent: qoder | Created: 2024-11-30
# Integration verification script

import asyncio
from pathlib import Path

async def verify_integrations():
    """Verify all components integrate correctly"""
    results = {}
    
    # 1. Verify shared → core_services integration
    print("1. Testing shared → core_services integration...")
    try:
        from core_services.agent_manager.manager import AgentManager
        from shared.config.environment import get_config
        
        config_value = get_config("TEST_VALUE", default="test")
        results["shared_to_core"] = "✅ PASS"
        print("   ✅ core_services can import from shared")
    except Exception as e:
        results["shared_to_core"] = f"❌ FAIL: {e}"
        print(f"   ❌ FAIL: {e}")
    
    # 2. Verify core_services → agents integration
    print("2. Testing core_services → agents integration...")
    try:
        from agents.coding.coding_agent import CodingAgent
        from core_services.engines.engine_factory import EngineFactory
        
        agent = CodingAgent()
        results["core_to_agents"] = "✅ PASS"
        print("   ✅ agents can import from core_services")
    except Exception as e:
        results["core_to_agents"] = f"❌ FAIL: {e}"
        print(f"   ❌ FAIL: {e}")
    
    # 3. Verify tests can import everything
    print("3. Testing test imports...")
    try:
        from tests.conftest import mock_config, mock_agent_manager
        results["test_imports"] = "✅ PASS"
        print("   ✅ tests can import fixtures")
    except Exception as e:
        results["test_imports"] = f"❌ FAIL: {e}"
        print(f"   ❌ FAIL: {e}")
    
    # 4. Verify full integration chain
    print("4. Testing full integration chain...")
    try:
        from agents.coding.coding_agent import CodingAgent
        from agents.base.base_agent import AgentRequest
        
        agent = CodingAgent()
        await agent.initialize()
        
        request = AgentRequest(
            task_id="integration_test",
            task_type="code_generation",
            parameters={
                "task_type": "code_generation",
                "language": "python",
                "prompt": "test"
            }
        )
        
        response = await agent.execute(request)
        
        if response.status in ["success", "error"]:
            results["full_chain"] = "✅ PASS"
            print("   ✅ Full integration chain works")
        else:
            results["full_chain"] = f"❌ FAIL: Unexpected status {response.status}"
            print(f"   ❌ FAIL: Unexpected status")
        
        await agent.shutdown()
    except Exception as e:
        results["full_chain"] = f"❌ FAIL: {e}"
        print(f"   ❌ FAIL: {e}")
    
    # Summary
    print("\n" + "="*50)
    print("INTEGRATION VERIFICATION SUMMARY")
    print("="*50)
    passed = sum(1 for v in results.values() if "✅" in v)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    for test, result in results.items():
        print(f"  {test}: {result}")
    
    return passed == total

if __name__ == "__main__":
    result = asyncio.run(verify_integrations())
    exit(0 if result else 1)
```

### 5.2 Run Integration Verification
```bash
python _reports/validation/integration_check.py
```

## STEP 6: CODE QUALITY CHECKS (15 minutes)

### 6.1 Run Linting
```bash
# Check code style with flake8 (if installed)
python -m flake8 shared/ core_services/ agents/ --max-line-length=100 --exclude=__pycache__,*.pyc > _reports/validation/flake8_results.txt 2>&1 || true

# Check with pylint (if installed)
python -m pylint shared/ core_services/ agents/ --exit-zero > _reports/validation/pylint_results.txt 2>&1 || true
```

### 6.2 Type Checking
```bash
# Run mypy type checking (if installed)
python -m mypy shared/ core_services/ agents/ --ignore-missing-imports > _reports/validation/mypy_results.txt 2>&1 || true
```

### 6.3 Security Audit
```bash
# Check for security issues with bandit (if installed)
python -m bandit -r shared/ core_services/ agents/ -f txt -o _reports/validation/security_audit.txt || true
```

## STEP 7: FINAL VALIDATION CHECKLIST (10 minutes)

Create comprehensive validation report:

**File: _reports/validation/final_validation_checklist.md**
```markdown
# Final Validation Checklist
Phase: 4 | Agent: qoder | Created: 2024-11-30

## Test Suite Results
- [ ] All 24 tests passing
- [ ] Test coverage > 80%
- [ ] No critical test failures
- [ ] Test execution time < 30s

## Performance Benchmarks
- [ ] Single agent execution < 5s
- [ ] Concurrent execution shows speedup > 2x
- [ ] No memory leaks detected
- [ ] Resource usage reasonable

## Integration Verification
- [ ] shared → core_services integration works
- [ ] core_services → agents integration works
- [ ] Tests can import all modules
- [ ] Full integration chain functional

## Code Quality
- [ ] No critical linting errors
- [ ] Type hints present and valid
- [ ] No security vulnerabilities
- [ ] Code follows PEP 8 standards

## Component Verification
- [ ] shared/config working correctly
- [ ] shared/database working correctly
- [ ] All engines operational
- [ ] All agents functional
- [ ] AI-MCP integration working

## Documentation
- [ ] API documentation accurate
- [ ] User guides complete
- [ ] Code examples working
- [ ] Architecture docs reflect reality

## Deployment Readiness
- [ ] All dependencies documented
- [ ] Configuration templates provided
- [ ] Security guidelines followed
- [ ] Monitoring points identified

## Issues Found
[List any issues discovered during validation]

## Fixes Applied
[List all fixes applied]

## Known Limitations
[List any known limitations or technical debt]

## Recommendations
[List recommendations for future improvements]
```

### 7.1 Fill Out Checklist
Go through each item and verify:
- ✅ = Complete and passing
- ⚠️ = Complete with issues (document issues)
- ❌ = Not complete (must fix)

## STEP 8: CREATE PHASE 4 COMPLETION REPORT (15 minutes)

**File: _reports/qoder/phase4_qoder_YYYYMMDD_HHMMSS.md**
```markdown
# Qoder Phase 4 Completion Report
Phase: 4 | Agent: qoder | Created: [TIMESTAMP]

## Executive Summary
Phase 4 validation and testing has been completed. The YMERA v2.0 refactored system has been thoroughly tested and validated.

**Overall Status**: [✅ READY FOR PRODUCTION | ⚠️ READY WITH ISSUES | ❌ NOT READY]

## Test Results

### Test Suite Execution
- **Total Tests**: 24
- **Passed**: [X]
- **Failed**: [Y]
- **Skipped**: [Z]
- **Success Rate**: [X/24 * 100]%
- **Execution Time**: [T] seconds
- **Code Coverage**: [C]%

### Test Breakdown
**Unit Tests (15)**:
- shared/config tests: [X/5] passed
- shared/database tests: [X/10] passed

**Integration Tests (4)**:
- All integration scenarios: [X/4] passed

**E2E Tests (5)**:
- End-to-end workflows: [X/5] passed

### Coverage Report
- **shared/**: [X]%
- **core_services/**: [Y]%
- **agents/**: [Z]%
- **Overall**: [A]%

## Performance Results

### Single Agent Execution
- Average: [X.XX]s
- Min: [X.XX]s
- Max: [X.XX]s
- Target: < 5s
- Status: [✅ PASS | ❌ FAIL]

### Concurrent Execution
- 3 agents concurrent: [X.XX]s
- 3 agents sequential: [Y.XX]s
- Speedup: [Z.ZZ]x
- Target: > 2x speedup
- Status: [✅ PASS | ❌ FAIL]

### Resource Usage
- Memory: [X]MB average
- CPU: [Y]% average
- Status: [✅ ACCEPTABLE | ⚠️ HIGH | ❌ EXCESSIVE]

## Integration Verification

| Integration Point | Status | Notes |
|-------------------|--------|-------|
| shared → core_services | [✅/❌] | [notes] |
| core_services → agents | [✅/❌] | [notes] |
| Tests → All modules | [✅/❌] | [notes] |
| Full integration chain | [✅/❌] | [notes] |

## Code Quality

### Linting Results
- **flake8**: [X] issues found ([Y] critical)
- **pylint**: Score [X.XX]/10
- **Status**: [✅ ACCEPTABLE | ⚠️ NEEDS WORK | ❌ CRITICAL ISSUES]

### Type Checking
- **mypy**: [X] errors, [Y] warnings
- **Status**: [✅ PASS | ⚠️ MINOR ISSUES | ❌ CRITICAL ERRORS]

### Security Audit
- **bandit**: [X] issues ([Y] high severity)
- **Status**: [✅ SECURE | ⚠️ MINOR RISKS | ❌ CRITICAL VULNERABILITIES]

## Issues Found and Fixed

### Critical Issues
1. [Issue description]
   - **Impact**: [description]
   - **Fix Applied**: [description]
   - **Status**: [✅ FIXED | ⏳ IN PROGRESS]

### Minor Issues
1. [Issue description]
   - **Impact**: [description]
   - **Fix Applied**: [description]
   - **Status**: [✅ FIXED | 📝 DOCUMENTED]

## Known Limitations

1. [Limitation 1]
   - **Impact**: [description]
   - **Workaround**: [if available]
   - **Future Fix**: [planned resolution]

## Recommendations

### Immediate Actions
- [ ] [Action item 1]
- [ ] [Action item 2]

### Future Improvements
- [ ] [Improvement 1]
- [ ] [Improvement 2]

### Technical Debt
- [ ] [Debt item 1]
- [ ] [Debt item 2]

## Deployment Readiness Assessment

| Category | Status | Score | Notes |
|----------|--------|-------|-------|
| Functionality | [✅/⚠️/❌] | [X/10] | [notes] |
| Performance | [✅/⚠️/❌] | [X/10] | [notes] |
| Security | [✅/⚠️/❌] | [X/10] | [notes] |
| Documentation | [✅/⚠️/❌] | [X/10] | [notes] |
| Testing | [✅/⚠️/❌] | [X/10] | [notes] |
| **Overall** | [✅/⚠️/❌] | [X/10] | [notes] |

## Files Generated

- test_results.txt - Complete test output
- test_report.html - HTML test report
- test_report.json - JSON test results
- performance_results.txt - Performance benchmarks
- integration_check.py - Integration verification script
- flake8_results.txt - Linting results
- pylint_results.txt - Code quality analysis
- mypy_results.txt - Type checking results
- security_audit.txt - Security scan results
- final_validation_checklist.md - Validation checklist
- fixes_applied.md - Documentation of fixes

## For Next Phase (Phase 5)

Phase 5 (Gemini) will create the final project report including:
- Complete project summary
- Architecture documentation
- Migration guide from v1.0 to v2.0
- Lessons learned
- Future roadmap

All validation is complete and system is [READY/NOT READY] for Phase 5.

## Sign-off

**Phase 4 Status**: [✅ COMPLETE | ⚠️ COMPLETE WITH ISSUES | ❌ INCOMPLETE]

**Validated by**: QODER
**Date**: [YYYY-MM-DD HH:MM:SS]
**Ready for Phase 5**: [YES/NO]

---

## Appendices

### Appendix A: Complete Test Output
[Attach or reference test_results.txt]

### Appendix B: Performance Metrics
[Attach or reference performance_results.txt]

### Appendix C: Issues Log
[Detailed log of all issues encountered and resolutions]
```

## STEP 9: PREPARE FOR PHASE 5 (5 minutes)

Create handoff document for Gemini:

**File: _reports/validation/phase5_handoff.md**
```markdown
# Phase 5 Handoff Document

## System Status
- **Phase 4 Complete**: [YES/NO]
- **All Tests Passing**: [YES/NO]
- **Production Ready**: [YES/NO]

## What Gemini Needs to Know

### System Statistics
- Total lines of code: [X]
- Test coverage: [Y]%
- Number of components: [Z]
- Performance benchmarks: [list key metrics]

### Architecture Decisions
- [Decision 1]
- [Decision 2]

### Migration Notes
- What changed from v1.0
- Breaking changes
- Migration steps

### Lessons Learned
- [Lesson 1]
- [Lesson 2]

### Future Recommendations
- [Recommendation 1]
- [Recommendation 2]

Gemini should use this information to create the final Phase 5 report.
```

=== CRITICAL REQUIREMENTS ===

1. **ALL TESTS MUST PASS** - No exceptions
2. **FIX ALL FAILURES** - Debug until green
3. **DOCUMENT EVERYTHING** - Every issue, every fix
4. **PERFORMANCE MATTERS** - Meet all benchmarks
5. **INTEGRATION VERIFIED** - All components work together
6. **CODE QUALITY** - Meet quality standards
7. **SECURITY CHECKED** - No critical vulnerabilities
8. **PRODUCTION READY** - System must be deployable

=== SUCCESS CRITERIA ===

Phase 4 is complete when:
1. ✅ All 24 tests passing
2. ✅ Code coverage > 80%
3. ✅ Performance benchmarks met
4. ✅ All integrations verified
5. ✅ Code quality acceptable
6. ✅ Security audit clean
7. ✅ Documentation validated
8. ✅ Completion report saved
9. ✅ System is production-ready

=== ESTIMATED TIME ===
Total: ~2 hours
- Environment setup: 10 min
- Test execution: 20 min
- Debug failures: Variable (0-60 min)
- Performance testing: 15 min
- Integration verification: 15 min
- Code quality: 15 min
- Validation checklist: 10 min
- Completion report: 15 min
- Phase 5 prep: 5 min

=== IF TESTS FAIL ===

**DO NOT PROCEED TO PHASE 5 UNTIL:**
- All critical tests pass
- All critical issues fixed
- System is functional

Document any known issues that are non-blocking.

========================================
END OF PHASE 4 - QODER PROMPT
========================================
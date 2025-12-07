# File Sharing Checklist for Three-CLI Implementation

## Quick Reference: What to Share with Each CLI

---

## Phase 3B: Agent-Model Integration (Week 1-2)

### Day 1: Pre-Planning

**Share with Gemini CLI:**
```
☐ docs/prompts/CUSTOMIZED_INTEGRATION_FOR_ORGANIZED_SYSTEM.md
☐ docs/prompts/GEMINI_CLI_TASKS.md
☐ docs/prompts/QODERCLI_QUALITY_CONTROL_SYSTEM.md
```

**Command:**
```bash
gemini plan --input "docs/prompts/CUSTOMIZED_INTEGRATION_FOR_ORGANIZED_SYSTEM.md" \
           --section "Phase 3B: Agent-Model Integration" \
           --output "plans/phase_3b_detailed_plan.md"
```

**Wait for:** ⏸️  `plans/phase_3b_detailed_plan.md` created

---

### Day 2-3: File Verification

**Share with QoderCLI:**
```
☐ docs/prompts/QODERCLI_QUALITY_CONTROL_SYSTEM.md
☐ plans/phase_3b_detailed_plan.md
```

**Command:**
```bash
qoder implement "Create file collection tool at scripts/collect_files.py"
```

**Then run:**
```bash
python scripts/collect_files.py --verify-all --phase 3B --target organized_system/
```

**Wait for:** ⏸️  `reports/file_verification_phase3b.json` with status "READY"

---

### Day 4-7: Implementation

**Share with QoderCLI:**
```
☐ docs/prompts/CUSTOMIZED_INTEGRATION_FOR_ORGANIZED_SYSTEM.md
☐ plans/phase_3b_detailed_plan.md
☐ reports/file_verification_phase3b.json
```

**Command:**
```bash
qoder implement "Phase 3B: Create AgentModelConnector in organized_system/src/core_services/integration/"
```

**Wait for:** ⏸️  New files created:
```
☐ organized_system/src/core_services/integration/agent_model_connector.py
☐ organized_system/src/core_services/integration/multi_agent_orchestrator.py
☐ organized_system/src/core_services/integration/existing_agent_wrapper.py
```

---

### Day 8-9: Code Review

**Share with Gemini CLI:**
```
☐ organized_system/src/core_services/integration/*.py (all new files)
☐ docs/prompts/GEMINI_CLI_TASKS.md
☐ docs/prompts/QODERCLI_QUALITY_CONTROL_SYSTEM.md
```

**Command:**
```bash
gemini review --files "organized_system/src/core_services/integration/**/*.py" \
              --check-all --strict \
              --output "reviews/phase_3b_review.json"
```

**Wait for:** ⏸️  `reviews/phase_3b_review.json` created

**Check:** Review issues list, note any high severity issues

---

### Day 10-11: Fix Issues (If Needed)

**Share with Copilot CLI:**
```
☐ organized_system/src/core_services/integration/*.py (files with issues)
☐ reviews/phase_3b_review.json
☐ docs/prompts/COPILOT_CLI_TASKS.md
```

**Command:**
```bash
copilot fix --issues reviews/phase_3b_review.json \
            --auto-apply \
            --output "reports/phase_3b_fixes.json"
```

**Wait for:** ⏸️  `reports/phase_3b_fixes.json` showing all issues resolved

---

### Day 12: Validation

**Share with QoderCLI (for tool creation):**
```
☐ docs/prompts/QODERCLI_QUALITY_CONTROL_SYSTEM.md
```

**Command:**
```bash
qoder implement "Create validation runner at scripts/run_validations.py"
```

**Then run:**
```bash
python scripts/run_validations.py --phase 3B --all
```

**Wait for:** ⏸️  `reports/phase_3b_validation.json` with:
```
☐ "overall_status": "PASSED"
☐ "can_proceed_to_next_phase": true
☐ All benchmarks met
```

**⚠️  CRITICAL: DO NOT PROCEED if status is "FAILED"**

---

### Day 13: Archive

**Share with QoderCLI (for tool creation):**
```
☐ docs/prompts/QODERCLI_QUALITY_CONTROL_SYSTEM.md
```

**Command:**
```bash
qoder implement "Create archive manager at scripts/archive_phase.py"
```

**Then run:**
```bash
python scripts/archive_phase.py --phase 3B --validate
```

**Wait for:** ⏸️  `archive/phase_3b_integration_YYYYMMDD_HHMMSS/` created and validated

---

## Phase 3C: Multi-Agent Orchestration (Week 3)

### Day 1: Pre-Planning

**Share with Gemini CLI:**
```
☐ docs/prompts/CUSTOMIZED_INTEGRATION_FOR_ORGANIZED_SYSTEM.md
☐ docs/prompts/GEMINI_CLI_TASKS.md
☐ docs/prompts/QODERCLI_QUALITY_CONTROL_SYSTEM.md
```

**Command:**
```bash
gemini plan --section "Phase 3C" --output "plans/phase_3c_detailed_plan.md"
```

---

### Day 2-3: File Verification

**Command:**
```bash
python scripts/collect_files.py --verify-all --phase 3C --target organized_system/
```

---

### Day 4-7: Implementation

**Share with QoderCLI:**
```
☐ docs/prompts/CUSTOMIZED_INTEGRATION_FOR_ORGANIZED_SYSTEM.md
☐ plans/phase_3c_detailed_plan.md
☐ reports/file_verification_phase3c.json
☐ archive/phase_3b_integration_YYYYMMDD_HHMMSS/ (previous work)
```

**Command:**
```bash
qoder implement "Phase 3C: Extend AIOrchestrator with 5 coordination types"
```

---

### Continue same pattern: Review → Fix → Validate → Archive

---

## Phase 4: Collective Learning (Week 4-5)

### Pre-Planning

**Share with Gemini CLI:**
```
☐ docs/prompts/PHASE_4_7_ADVANCED_SYSTEMS.md (Phase 4 section)
☐ docs/prompts/GEMINI_CLI_TASKS.md
☐ docs/prompts/QODERCLI_QUALITY_CONTROL_SYSTEM.md
```

### Implementation

**Share with QoderCLI:**
```
☐ docs/prompts/PHASE_4_7_ADVANCED_SYSTEMS.md (Phase 4 section)
☐ plans/phase_4_detailed_plan.md
☐ reports/file_verification_phase4.json
☐ archive/phase_3c_*/ (previous work)
```

---

## Phase 5: Multi-Agent Training (Week 6-7)

### Pre-Planning

**Share with Gemini CLI:**
```
☐ docs/prompts/PHASE_4_7_ADVANCED_SYSTEMS.md (Phase 5 section)
☐ docs/prompts/GEMINI_CLI_TASKS.md
☐ docs/prompts/QODERCLI_QUALITY_CONTROL_SYSTEM.md
```

### Implementation

**Share with QoderCLI:**
```
☐ docs/prompts/PHASE_4_7_ADVANCED_SYSTEMS.md (Phase 5 section)
☐ plans/phase_5_detailed_plan.md
☐ reports/file_verification_phase5.json
☐ archive/phase_4_*/ (previous work)
```

---

## Phase 6: Enhanced Security (Week 8)

### Pre-Planning

**Share with Gemini CLI:**
```
☐ docs/prompts/PHASE_4_7_ADVANCED_SYSTEMS.md (Phase 6 section)
☐ docs/prompts/GEMINI_CLI_TASKS.md
☐ docs/prompts/QODERCLI_QUALITY_CONTROL_SYSTEM.md
```

### Implementation

**Share with QoderCLI:**
```
☐ docs/prompts/PHASE_4_7_ADVANCED_SYSTEMS.md (Phase 6 section)
☐ plans/phase_6_detailed_plan.md
☐ reports/file_verification_phase6.json
☐ archive/phase_5_*/ (previous work)
```

---

## Phase 7: Frontend Integration (Week 9-10)

### Pre-Planning

**Share with Gemini CLI:**
```
☐ docs/prompts/CUSTOMIZED_INTEGRATION_FOR_ORGANIZED_SYSTEM.md (Phase 7)
☐ docs/prompts/GEMINI_CLI_TASKS.md
☐ docs/prompts/QODERCLI_QUALITY_CONTROL_SYSTEM.md
```

### Implementation

**Share with QoderCLI:**
```
☐ docs/prompts/CUSTOMIZED_INTEGRATION_FOR_ORGANIZED_SYSTEM.md (Phase 7)
☐ plans/phase_7_detailed_plan.md
☐ reports/file_verification_phase7.json
☐ archive/phase_6_*/ (previous work)
```

---

## Phase 8: Production Deployment (Week 11-12)

### Pre-Planning

**Share with Gemini CLI:**
```
☐ docs/prompts/CUSTOMIZED_INTEGRATION_FOR_ORGANIZED_SYSTEM.md (Phase 8)
☐ docs/prompts/GEMINI_CLI_TASKS.md
☐ docs/prompts/QODERCLI_QUALITY_CONTROL_SYSTEM.md
```

### Implementation

**Share with QoderCLI:**
```
☐ docs/prompts/CUSTOMIZED_INTEGRATION_FOR_ORGANIZED_SYSTEM.md (Phase 8)
☐ plans/phase_8_detailed_plan.md
☐ reports/file_verification_phase8.json
☐ archive/phase_7_*/ (previous work)
```

---

## General Pattern (Every Phase)

### 1. Pre-Planning (Gemini CLI)
```
☐ Primary guide file for phase
☐ GEMINI_CLI_TASKS.md
☐ QODERCLI_QUALITY_CONTROL_SYSTEM.md
```

### 2. File Verification (Tool)
```
☐ Run: python scripts/collect_files.py --phase X
```

### 3. Implementation (QoderCLI)
```
☐ Primary guide file for phase
☐ plans/phase_X_detailed_plan.md
☐ reports/file_verification_phase_X.json
☐ archive/previous_phase_*/ (if not first phase)
```

### 4. Code Review (Gemini CLI)
```
☐ New source files from implementation
☐ GEMINI_CLI_TASKS.md
☐ QODERCLI_QUALITY_CONTROL_SYSTEM.md
```

### 5. Fix Issues (Copilot CLI) - If needed
```
☐ Source files with issues
☐ reviews/phase_X_review.json
☐ COPILOT_CLI_TASKS.md
```

### 6. Validation (Tool)
```
☐ Run: python scripts/run_validations.py --phase X --all
☐ Check: "overall_status": "PASSED"
```

### 7. Archive (Tool)
```
☐ Run: python scripts/archive_phase.py --phase X --validate
```

---

## Master File Reference

### Always Keep These Handy:

**For QoderCLI:**
- `CUSTOMIZED_INTEGRATION_FOR_ORGANIZED_SYSTEM.md` (PRIMARY - your 40+ agents)
- `PHASE_4_7_ADVANCED_SYSTEMS.md` (Phases 4-7 details)
- `QODERCLI_QUALITY_CONTROL_SYSTEM.md` (validation specs)

**For Gemini CLI:**
- `GEMINI_CLI_TASKS.md` (your responsibilities)
- `QODERCLI_QUALITY_CONTROL_SYSTEM.md` (quality benchmarks)

**For Copilot CLI:**
- `COPILOT_CLI_TASKS.md` (your responsibilities)

**For Reference:**
- `QODERCLI_QUICK_REFERENCE.md` (quick commands)
- `INTEGRATION_QUICK_START.md` (timeline reference)

---

## Critical Checkpoints

### After Each Step, Verify:

**✅ After Pre-Planning:**
```
☐ Plan file exists
☐ Plan makes sense
☐ File locations identified
```

**✅ After File Verification:**
```
☐ Verification report exists
☐ Status is "READY"
☐ No missing files
```

**✅ After Implementation:**
```
☐ All expected files created
☐ Files not empty
☐ Basic syntax check passes
```

**✅ After Code Review:**
```
☐ Review report exists
☐ Issues identified
☐ Severity ratings noted
```

**✅ After Fixes:**
```
☐ Fix report shows all issues resolved
☐ Code changes applied
```

**✅ After Validation (CRITICAL):**
```
☐ Overall status is "PASSED"
☐ All 5 validation types passed
☐ Coverage ≥ 80%
☐ Performance targets met
☐ No critical vulnerabilities
☐ can_proceed_to_next_phase is true
```

**✅ After Archive:**
```
☐ Archive directory exists
☐ Checksums validated
☐ Re-run tests from archive passed
```

---

## Emergency: If Something Fails

### If Validation Fails:
```
1. Check reports/phase_X_validation.json for failed checks
2. Fix issues based on failure type:
   - Low coverage → Use Copilot CLI to generate more tests
   - Performance → Use Copilot CLI to optimize code
   - Security → Use Copilot CLI to fix vulnerabilities
3. Re-run validation
4. Repeat until PASSED
```

### If Implementation Crashes:
```
1. Run: python scripts/analyze_logs.py --failures
2. Check last successful checkpoint
3. Run: python scripts/resume_execution.py
4. Continue from checkpoint
```

### If Archive Fails:
```
1. Check error message
2. Fix issues (usually checksum mismatches)
3. Run: python scripts/archive_phase.py --force-recreate
```

---

## Quick Command Reference

```bash
# Pre-planning
gemini plan --section "Phase X" --output plans/phase_X.md

# File verification
python scripts/collect_files.py --verify-all --phase X

# Implementation
qoder implement "Phase X: [description]" --validate-real-world

# Code review
gemini review --files "src/**/*.py" --check-all --strict

# Fix issues
copilot fix --issues reviews/phase_X_review.json --auto-apply

# Validation
python scripts/run_validations.py --phase X --all

# Archive
python scripts/archive_phase.py --phase X --validate

# Check if can proceed
python scripts/can_proceed_check.py --phase X

# Resume after crash
python scripts/resume_execution.py --from-log logs/qodercli_execution.log

# Analyze failures
python scripts/analyze_logs.py --from 2024-12-05 --failures
```

---

**Print this checklist and check off items as you complete them!**

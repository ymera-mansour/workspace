# Phase 3B Prompt Usage Guide
## How to Use Gemini CLI and QoderCLI Prompts

---

## Overview

Phase 3B has two main components:
1. **Original Goal**: Implement 24 test stubs (unit, integration, E2E)
2. **New Goal**: Integrate Gemini optimization system into YMERA platform

Two prompts have been created to handle this:

---

## Prompt 1: Gemini CLI (Strategic Planning)

**File**: `GEMINI_CLI_PHASE3B_PLANNING_PROMPT.md`

**Purpose**: Create comprehensive implementation plan for Phase 3B

**When to Use**: At the START of Phase 3B, before any implementation

**How to Use**:
```bash
# Feed to Gemini CLI
gemini < GEMINI_CLI_PHASE3B_PLANNING_PROMPT.md > phase3b_detailed_plan.md

# Or if using Gemini API directly:
cat GEMINI_CLI_PHASE3B_PLANNING_PROMPT.md | gemini-cli --output phase3b_detailed_plan.md
```

**What It Does**:
- Analyzes Phase 3B requirements
- Designs integration architecture  
- Creates 18-task work breakdown
- Defines 3-week timeline
- Identifies risks and mitigations
- Generates QoderCLI execution sequence
- Defines success criteria

**Output**: 15-20 page detailed plan with:
- Executive summary
- Work breakdown structure
- Architecture diagrams
- Risk management plan
- 9 QoderCLI prompts to execute sequentially
- Validation procedures

---

## Prompt 2: QoderCLI (Implementation)

**File**: `QODERCLI_GEMINI_INTEGRATION_PROMPT.md`

**Purpose**: Step-by-step implementation instructions

**When to Use**: AFTER Gemini CLI creates the detailed plan

**How to Use**:
```bash
# Feed to QoderCLI
qodercli < QODERCLI_GEMINI_INTEGRATION_PROMPT.md

# Or execute step-by-step
qodercli --step 1  # Environment setup
qodercli --step 2  # Install dependencies
# ... continue through all 10 steps
```

**What It Does**:
- Sets up environment (API keys, .env file)
- Installs dependencies
- Copies Gemini optimization files
- Integrates with agent_manager
- Configures all 40+ agents
- Updates individual agents
- Sets up training loop
- Creates monitoring dashboard
- Implements integration tests
- Updates documentation

**Steps** (10 total):
1. Environment Setup (15 min)
2. Install Dependencies (10 min)
3. Copy Optimization Files (10 min)
4. Integrate Agent Manager (30 min)
5. Configure Agent Preferences (20 min)
6. Update Individual Agents (40 min)
7. Implement Training Loop (30 min)
8. Add Monitoring Dashboard (20 min)
9. Testing Integration (30 min)
10. Documentation Update (15 min)

**Total Time**: ~4 hours for basic integration

---

## Recommended Workflow

### Step 1: Strategic Planning (Gemini CLI)
```bash
# Have Gemini create the detailed plan
gemini < GEMINI_CLI_PHASE3B_PLANNING_PROMPT.md > phase3b_plan.md

# Review the plan
cat phase3b_plan.md

# Share with team for approval
```

**Deliverable**: Comprehensive 15-20 page implementation plan

### Step 2: Implementation (QoderCLI)
```bash
# Execute the plan step-by-step
qodercli < QODERCLI_GEMINI_INTEGRATION_PROMPT.md

# Or use the specific prompts from Gemini's plan
# Gemini will have generated 9 specific QoderCLI prompts
```

**Deliverable**: Fully integrated Gemini optimization system

### Step 3: Validation
```bash
# Run all tests
pytest tests/ -v

# Show monitoring dashboard
python -m gemini_optimization.monitoring show_dashboard

# Verify training data
python -c "from gemini_optimization.training_loop import training_loop; print(training_loop.get_statistics())"
```

**Deliverable**: Validated, working system

---

## Model Selection Strategy

Both prompts include guidance on which Gemini model to use:

### By Agent Type

**Gemini 2.0 Flash** (10 RPM, 1,500 RPD):
- coding_agent (simple tasks)
- documentation_agent
- web_scraping_agent
- notification_agent

**Gemini 1.5 Flash** (15 RPM, 1,500 RPD):
- database_agent
- api_agent
- testing_agent
- monitoring_agent

**Gemini 1.5 Pro** (2 RPM, 50 RPD) ⚠️ LIMITED:
- architecture_agent
- security_agent
- refactoring_agent
- optimization_agent

**Gemini 1.5 Flash-8B** (15 RPM, 4,000 RPD):
- validation_agent
- formatting_agent
- linting_agent
- utility_agent

### By Task Complexity

| Complexity | Model | Example |
|------------|-------|---------|
| Simple | Flash-8B or 2.0 Flash | Single function, < 50 lines |
| Moderate | 1.5 Flash or 2.0 Flash | Standard feature, 50-200 lines |
| Complex | 1.5 Pro | Architecture, > 200 lines |
| Critical | 1.5 Pro | Production, security-sensitive |

---

## Timeline

**Week 1**: Tests + Foundation
- Days 1-2: Unit tests implementation
- Day 3: Environment + file setup
- Days 4-5: Integration tests + agent manager

**Week 2**: Gemini Integration
- Days 1-2: Agent configuration (40+ agents)
- Days 3-4: Individual agent updates
- Day 5: Training + monitoring setup

**Week 3**: Validation
- Day 1: E2E tests
- Days 2-3: Integration testing
- Day 4: Documentation
- Day 5: Final validation

---

## Expected Results

### After Gemini CLI Planning:
- ✅ Detailed 15-20 page implementation plan
- ✅ Clear task breakdown with estimates
- ✅ Risk mitigation strategies
- ✅ 9 executable QoderCLI prompts
- ✅ Success criteria defined

### After QoderCLI Implementation:
- ✅ 24 test stubs → full implementations
- ✅ All tests passing (30 total tests)
- ✅ Gemini optimization integrated
- ✅ 40+ agents configured
- ✅ Multi-org key rotation working
- ✅ Agent training system active
- ✅ Monitoring dashboard operational
- ✅ 60-75% reduction in API calls
- ✅ 100% free tier compliance

---

## Troubleshooting

### Issue: Gemini CLI not producing plan
**Solution**: Check prompt formatting, ensure using latest Gemini model

### Issue: QoderCLI steps failing
**Solution**: Follow prerequisites in each step, check error messages

### Issue: API keys not working
**Solution**: Verify .env file, check key validity at https://aistudio.google.com/

### Issue: Tests not passing
**Solution**: Run tests individually to isolate issues:
```bash
pytest tests/unit/test_shared_config.py -v
pytest tests/integration/test_gemini_optimization.py -v
```

---

## Support Resources

**Planning Phase**:
- Prompt: `GEMINI_CLI_PHASE3B_PLANNING_PROMPT.md`
- Context: Previous phase files (phase3a, phase3b)

**Implementation Phase**:
- Prompt: `QODERCLI_GEMINI_INTEGRATION_PROMPT.md`
- Guides: All GEMINI_*_GUIDE.md files
- Reference: `QUICK_REFERENCE.md`

**Documentation**:
- Setup: `GEMINI_ADVANCED_SETUP_GUIDE.md`
- Quick start: `IMPLEMENTATION_GUIDE.md`
- Complete review: `GEMINI_GOOGLE_PRODUCTS_OPTIMIZATION_REVIEW.md`

---

## Quick Commands

```bash
# Generate plan
gemini < GEMINI_CLI_PHASE3B_PLANNING_PROMPT.md > plan.md

# Execute plan
qodercli < QODERCLI_GEMINI_INTEGRATION_PROMPT.md

# Run tests
pytest tests/ -v

# Show dashboard
python -m gemini_optimization.monitoring show_dashboard

# Check training
python -c "from gemini_optimization.training_loop import training_loop; print(training_loop.get_statistics())"

# View metrics
cat gemini_metrics.json | jq
```

---

## Summary

1. **Gemini CLI**: Creates strategic plan (use at start)
2. **QoderCLI**: Executes implementation (use after plan)
3. **Timeline**: 3 weeks total
4. **Result**: Fully integrated, optimized Gemini system

Both prompts work together to deliver Phase 3B successfully.

---

Last Updated: December 6, 2024  
Version: 1.0

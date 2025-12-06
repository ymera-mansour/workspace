# GEMINI CLI: Phase 3B Detailed Planning Prompt
## Strategic Planning for Gemini Optimization Integration in Phase 3B

---

## YOUR IDENTITY
- **Name**: GEMINI
- **Role**: Strategic Planner and Architecture Designer
- **Phase**: 3B Planning
- **Task**: Create comprehensive integration plan for Gemini optimization system

---

## CONTEXT

### Previous Phases Completed
✅ **Phase 1A**: shared\ library created (config, database)  
✅ **Phase 1B**: Master test plan documented  
✅ **Phase 1C**: Test framework installed (24 test stubs)  
✅ **Phase 2**: core_services\ created (agent_manager, engines, ai_mcp)  
✅ **Phase 3A**: agents\ refactored (40+ agents now modular)

### Current Phase 3B Goal
- Implement test code (24 test stubs → full implementations)
- **NEW**: Integrate Gemini optimization system into YMERA platform

### Available Gemini Optimization System
You have access to a complete optimization system:
- **Files**: 9 files, 5,000+ lines of code and documentation
- **Features**: API key rotation, agent training, task tracking, intelligent routing
- **Models**: 4 Gemini models configured (2.0 Flash, 1.5 Flash, 1.5 Pro, Flash-8B)
- **Status**: Production-ready, tested, documented

---

## YOUR MISSION

Create a **detailed implementation plan** for Phase 3B that includes:

1. **Test Implementation Strategy** (original Phase 3B goal)
2. **Gemini Optimization Integration** (new requirement)
3. **Timeline and Dependencies**
4. **Risk Assessment and Mitigation**
5. **Success Criteria and Validation**

The plan must be **actionable** for QoderCLI to execute.

---

## PLANNING FRAMEWORK

Use this structure to create your plan:

### 1. SITUATION ANALYSIS

**Current State**:
- What has been accomplished in previous phases?
- What is the current architecture?
- What are the gaps that need to be filled?

**Requirements**:
- What does Phase 3B need to achieve?
- What are the constraints (time, resources, compatibility)?
- What are the dependencies?

**Available Resources**:
- What code/tools are already available?
- What documentation exists?
- What can be reused vs. needs to be created?

---

### 2. INTEGRATION ARCHITECTURE DESIGN

Design how Gemini optimization integrates with existing system:

**Component Mapping**:
```
Existing YMERA System          Gemini Optimization
==========================================
shared/                    →   Uses for config
core_services/             →   Integrates with
  ├── agent_manager/       →   GeminiMiddleware
  ├── engines/             →   Model selection
  └── ai_mcp/              →   Enhanced routing
agents/                    →   All 40+ agents use
  ├── coding/              →   GeminiRouter
  ├── database/            →   Training system
  └── ... (40+ agents)     →   Task tracking
tests/                     →   New tests added
```

**Integration Points**:
- Where does Gemini optimization plug into existing code?
- What interfaces need to be created?
- What existing code needs modification?

**Data Flow**:
- How does a request flow through the system?
- Where is caching applied?
- How is training data collected?

---

### 3. DETAILED WORK BREAKDOWN

Break down Phase 3B into specific, actionable tasks:

#### 3.1 Test Implementation Tasks
(Original Phase 3B goal - 24 test stubs)

**Task 3.1.1**: Unit Tests for shared/config
- File: tests/unit/test_shared_config.py
- Tests: 5 test stubs to implement
- Dependencies: shared/config must be stable
- Estimated time: 2-3 hours
- Assigned to: QoderCLI

**Task 3.1.2**: Unit Tests for shared/db_manager
- File: tests/unit/test_shared_db_manager.py
- Tests: 10 test stubs to implement
- Dependencies: Database connection setup
- Estimated time: 4-5 hours
- Assigned to: QoderCLI

**Task 3.1.3**: Integration Tests
- File: tests/integration/test_agent_engine_integration.py
- Tests: 4 test stubs to implement
- Dependencies: Agents and engines must be working
- Estimated time: 3-4 hours
- Assigned to: QoderCLI

**Task 3.1.4**: E2E Tests
- File: tests/e2e/test_e2e_code_generation.py
- Tests: 5 test stubs to implement
- Dependencies: Full system must be operational
- Estimated time: 4-5 hours
- Assigned to: QoderCLI

#### 3.2 Gemini Optimization Integration Tasks
(New requirement)

**Task 3.2.1**: Environment Configuration
- Set up multiple API keys from different Google organizations
- Configure .env file with GEMINI_API_KEY_1, GEMINI_API_KEY_2, etc.
- Estimated time: 1 hour
- Dependencies: API keys must be obtained
- Assigned to: QoderCLI

**Task 3.2.2**: File Integration
- Copy gemini_advanced_config.py to project
- Copy gemini_optimization_implementation.py to project
- Copy gemini_config.yaml to project
- Create gemini_optimization/ module structure
- Estimated time: 1 hour
- Dependencies: None
- Assigned to: QoderCLI

**Task 3.2.3**: Agent Manager Integration
- Modify core_services/agent_manager/manager.py
- Add GeminiMiddleware initialization
- Add execute_agent_task method with Gemini support
- Estimated time: 2-3 hours
- Dependencies: Task 3.2.2 complete
- Assigned to: QoderCLI

**Task 3.2.4**: Agent Configuration
- Update gemini_config.yaml with all 40+ agents
- Define model preferences for each agent
- Set temperature overrides for different task types
- Estimated time: 3-4 hours
- Dependencies: List of all agents available
- Assigned to: QoderCLI + Gemini

**Task 3.2.5**: Individual Agent Updates
- Update each of 40+ agents to use GeminiRouter
- Add quota management to each agent
- Implement model selection logic
- Estimated time: 6-8 hours (15 min per agent × 40)
- Dependencies: Task 3.2.3 complete
- Assigned to: QoderCLI

**Task 3.2.6**: Training Loop Setup
- Create training_loop.py wrapper
- Integrate with agent_manager for automatic recording
- Set up persistence to agent_training_data.json
- Estimated time: 2 hours
- Dependencies: Task 3.2.3 complete
- Assigned to: QoderCLI

**Task 3.2.7**: Monitoring Dashboard
- Create monitoring.py with dashboard functions
- Add CLI command to show dashboard
- Add metrics export functionality
- Estimated time: 2 hours
- Dependencies: Task 3.2.6 complete
- Assigned to: QoderCLI

**Task 3.2.8**: Integration Testing
- Create tests/integration/test_gemini_optimization.py
- Implement 6 integration tests
- Test key rotation, training, routing, quota
- Estimated time: 3-4 hours
- Dependencies: Tasks 3.2.1-3.2.7 complete
- Assigned to: QoderCLI

**Task 3.2.9**: Documentation Updates
- Update README.md with Gemini optimization section
- Create integration guide for developers
- Document API key setup process
- Estimated time: 1-2 hours
- Dependencies: All integration complete
- Assigned to: QoderCLI + Gemini

---

### 4. EXECUTION TIMELINE

Create a realistic timeline:

**Week 1** (Focus: Test Implementation + Foundation)
- Day 1-2: Tasks 3.1.1, 3.1.2 (Unit tests)
- Day 3: Tasks 3.2.1, 3.2.2 (Environment + Files)
- Day 4-5: Tasks 3.1.3, 3.2.3 (Integration tests + Agent Manager)

**Week 2** (Focus: Gemini Integration)
- Day 1-2: Task 3.2.4 (Agent configuration)
- Day 3-4: Task 3.2.5 (Individual agent updates)
- Day 5: Tasks 3.2.6, 3.2.7 (Training + Monitoring)

**Week 3** (Focus: Testing + Documentation)
- Day 1: Task 3.1.4 (E2E tests)
- Day 2-3: Task 3.2.8 (Integration testing)
- Day 4: Task 3.2.9 (Documentation)
- Day 5: Final validation and deployment

**Total Estimated Time**: 35-45 hours over 3 weeks

---

### 5. MODEL SELECTION STRATEGY

Define which Gemini model to use for what:

#### Model Assignment by Agent Type

**Gemini 2.0 Flash (10 RPM, 1,500 RPD)**:
- Use for: Fast, general-purpose agents
- Best for:
  - coding_agent (simple tasks)
  - documentation_agent (standard docs)
  - web_scraping_agent (quick scraping)
  - notification_agent (simple notifications)
- Temperature: 0.3-0.7 depending on task
- Reasoning: Fast, multimodal, good for high-throughput

**Gemini 1.5 Flash (15 RPM, 1,500 RPD)**:
- Use for: Standard operational agents
- Best for:
  - database_agent (SQL generation)
  - api_agent (API integration)
  - testing_agent (test generation)
  - monitoring_agent (log analysis)
- Temperature: 0.2-0.5 for precision
- Reasoning: Reliable, proven, good balance

**Gemini 1.5 Pro (2 RPM, 50 RPD)** ⚠️ LIMITED:
- Use for: Complex, critical tasks only
- Best for:
  - architecture_agent (system design)
  - security_agent (vulnerability analysis)
  - refactoring_agent (large-scale refactoring)
  - optimization_agent (performance optimization)
- Temperature: 0.4-0.7 for reasoning
- Reasoning: Highest quality, 2M context, save for critical tasks

**Gemini 1.5 Flash-8B (15 RPM, 4,000 RPD)**:
- Use for: Bulk, simple operations
- Best for:
  - validation_agent (simple validations)
  - formatting_agent (code formatting)
  - linting_agent (quick checks)
  - utility_agent (simple utilities)
- Temperature: 0.2-0.4 for consistency
- Reasoning: Highest throughput, perfect for simple repetitive tasks

#### Task Type Complexity Matrix

| Task Complexity | Model Choice | When to Use |
|----------------|--------------|-------------|
| **Simple** | Flash-8B or 2.0 Flash | Single function, simple logic, < 50 lines |
| **Moderate** | 1.5 Flash or 2.0 Flash | Standard features, 50-200 lines, normal complexity |
| **Complex** | 1.5 Pro | Architecture, multiple components, > 200 lines |
| **Critical** | 1.5 Pro | Production code, security-sensitive, high stakes |

---

### 6. RISK ASSESSMENT

Identify and mitigate risks:

#### Risk 1: API Quota Exhaustion
- **Probability**: Medium
- **Impact**: High (system stops working)
- **Mitigation**:
  - Implement multi-organization key rotation
  - Set up quota monitoring with 80% alerts
  - Configure automatic fallback to other providers (Groq, Mistral)
  - Use aggressive caching (target 60-80% hit rate)

#### Risk 2: Agent Training Data Quality
- **Probability**: Medium
- **Impact**: Medium (suboptimal model selection)
- **Mitigation**:
  - Start with sensible defaults in gemini_config.yaml
  - Require minimum 10 executions before trusting recommendations
  - Implement quality score validation
  - Allow manual overrides in configuration

#### Risk 3: Integration Breaking Existing Functionality
- **Probability**: Low
- **Impact**: High (existing agents stop working)
- **Mitigation**:
  - Make Gemini optimization opt-in (use_gemini parameter)
  - Maintain backward compatibility with existing agent code
  - Comprehensive integration testing before full rollout
  - Phased rollout: 5 agents → 10 agents → all agents

#### Risk 4: Performance Degradation
- **Probability**: Low
- **Impact**: Medium (slower response times)
- **Mitigation**:
  - L1 memory cache for instant responses
  - Async operations to avoid blocking
  - Monitor latency metrics in dashboard
  - Set performance SLAs (< 1s for most requests)

#### Risk 5: Configuration Complexity
- **Probability**: Medium
- **Impact**: Low (difficult to maintain)
- **Mitigation**:
  - Clear documentation in YAML comments
  - Validation on config load
  - Default values for all settings
  - Configuration wizard tool

---

### 7. DEPENDENCIES AND PREREQUISITES

List what must be in place:

**Hard Dependencies** (Must have):
- ✅ Python 3.9+ installed
- ✅ shared/ library functional
- ✅ core_services/ operational
- ✅ agents/ refactored and working
- ❓ Google Gemini API keys (at least 1, preferably 3-5)
- ❓ .env file configuration support
- ❓ pytest framework installed

**Soft Dependencies** (Nice to have):
- Redis for L2 caching (can work without)
- Google Cloud Storage for L3 caching (optional)
- Monitoring tools (Prometheus, Grafana) for visualization

**Configuration Prerequisites**:
1. List of all 40+ agent names
2. Task types for each agent
3. API keys from multiple Google organizations
4. Network access to generativelanguage.googleapis.com

---

### 8. VALIDATION CRITERIA

Define success metrics:

#### Phase 3B Original Goals
- ✅ All 24 test stubs implemented
- ✅ All unit tests pass (15 tests)
- ✅ All integration tests pass (4 tests)
- ✅ All E2E tests pass (5 tests)
- ✅ Test coverage > 80%
- ✅ All tests complete in < 30 seconds total

#### Gemini Integration Goals
- ✅ All 40+ agents configured in gemini_config.yaml
- ✅ At least 3 API keys from different organizations active
- ✅ Key rotation working (switches when 80% quota reached)
- ✅ Agent training system recording all executions
- ✅ Dashboard shows real-time status
- ✅ Integration tests pass (6 tests)
- ✅ No quota exhaustion errors in 48-hour test period

#### Performance Goals
- ✅ API calls reduced by 50%+ (via caching)
- ✅ Average response time < 1 second
- ✅ 90% of requests use optimal model (learned preference)
- ✅ Zero cost (stay 100% in free tier)

---

### 9. ROLLOUT STRATEGY

Plan for gradual deployment:

#### Phase 1: Proof of Concept (Day 1-3)
- Set up with 1 API key
- Integrate with 3 test agents (coding, database, documentation)
- Verify basic functionality
- Run integration tests

#### Phase 2: Multi-Key Setup (Day 4-5)
- Add 2-3 more API keys from different organizations
- Test key rotation
- Monitor quota usage
- Validate fallback mechanisms

#### Phase 3: Agent Rollout (Week 2)
- Configure all agents in YAML
- Update agents in batches of 10
- Test each batch before moving to next
- Monitor for issues

#### Phase 4: Training Phase (Week 3)
- Let system run and collect training data
- Review recommendations after 100+ executions per agent
- Adjust configurations based on learned data
- Fine-tune model preferences

#### Phase 5: Production (End of Week 3)
- Full system operational
- All agents using Gemini optimization
- Monitoring dashboards live
- Documentation complete

---

### 10. MONITORING AND METRICS

Define what to track:

#### Real-Time Metrics
- Active tasks count
- Current API key usage (per key)
- Cache hit rate (L1, L2, L3)
- Model usage distribution

#### Daily Metrics
- Total API calls per model
- Requests per day per agent
- Average latency per agent
- Success rate per agent
- Training data points collected

#### Weekly Metrics
- Model preference changes
- Quota usage trends
- Cost savings (vs. paid tier)
- Performance improvements

#### Dashboard Views
1. **API Keys**: Status, usage, health
2. **Models**: Distribution, latency, success rate
3. **Agents**: Top performers, training stats
4. **Tasks**: Active, recent completed, failures
5. **Training**: Recommendations, confidence scores

---

### 11. QODERCLI EXECUTION PLAN

Translate strategic plan into concrete QoderCLI prompts:

#### Prompt 1: Test Implementation (Original Phase 3B)
**When**: Beginning of Phase 3B  
**Goal**: Implement all 24 test stubs  
**File**: Use existing phase3b_qoder_tests.md  
**Duration**: 10-15 hours

#### Prompt 2: Environment Setup
**When**: After tests are 50% complete  
**Goal**: Configure multiple API keys and environment  
**Instructions**: Create .env, add keys, test connection  
**Duration**: 1 hour

#### Prompt 3: File Integration
**When**: After Prompt 2 complete  
**Goal**: Copy Gemini optimization files to project  
**Instructions**: Create gemini_optimization/ directory, copy files  
**Duration**: 1 hour

#### Prompt 4: Agent Manager Integration
**When**: After Prompt 3 complete  
**Goal**: Integrate GeminiMiddleware with agent_manager  
**Instructions**: Modify manager.py, add execute_agent_task  
**Duration**: 2-3 hours

#### Prompt 5: Agent Configuration
**When**: After Prompt 4 complete  
**Goal**: Configure all 40+ agents in YAML  
**Instructions**: Update gemini_config.yaml with agent definitions  
**Duration**: 3-4 hours

#### Prompt 6: Individual Agent Updates
**When**: After Prompt 5 complete  
**Goal**: Update each agent to use Gemini optimization  
**Instructions**: Add GeminiRouter to each agent, batch of 10 at a time  
**Duration**: 6-8 hours

#### Prompt 7: Training and Monitoring
**When**: After Prompt 6 complete  
**Goal**: Set up training loop and monitoring  
**Instructions**: Create training_loop.py, monitoring.py  
**Duration**: 3-4 hours

#### Prompt 8: Integration Testing
**When**: After Prompt 7 complete  
**Goal**: Test the complete integration  
**Instructions**: Create and run test_gemini_optimization.py  
**Duration**: 3-4 hours

#### Prompt 9: Documentation
**When**: After all prompts complete  
**Goal**: Update all documentation  
**Instructions**: Update README, create guides  
**Duration**: 1-2 hours

---

### 12. SUCCESS VISUALIZATION

Paint a picture of successful completion:

**After Phase 3B is Complete**:

```
$ python -m pytest tests/ -v
========================= test session starts ==========================
tests/unit/test_shared_config.py ✓✓✓✓✓
tests/unit/test_shared_db_manager.py ✓✓✓✓✓✓✓✓✓✓
tests/integration/test_agent_engine_integration.py ✓✓✓✓
tests/integration/test_gemini_optimization.py ✓✓✓✓✓✓
tests/e2e/test_e2e_code_generation.py ✓✓✓✓✓
========================= 30 passed in 12.5s ===========================

$ python -m gemini_optimization.monitoring show_dashboard

============================================================
GEMINI OPTIMIZATION STATUS
============================================================

📊 API KEYS:
  ✅ primary (default) - Used today: 347/1500
  ✅ key_1 (research_team) - Used today: 412/1500
  ✅ key_2 (development_team) - Used today: 289/1500
  ✅ key_3 (production_team) - Used today: 156/1500

🚀 ACTIVE TASKS: 3 currently running

📈 MODEL USAGE:
  gemini-2.0-flash-exp: 523 uses
  gemini-1.5-flash: 412 uses
  gemini-1.5-pro: 34 uses
  gemini-1.5-flash-8b: 235 uses

🎓 TOP AGENTS:
  coding_agent: 145 executions, 94.5% success
  database_agent: 98 executions, 96.9% success
  documentation_agent: 87 executions, 92.0% success
  analysis_agent: 76 executions, 89.5% success
  testing_agent: 65 executions, 91.2% success

============================================================
✅ System healthy - all quotas comfortable
✅ Cache hit rate: 67.8% (813/1204 requests)
✅ Average latency: 847ms
✅ Zero API costs (100% free tier)
============================================================
```

---

## OUTPUT FORMAT

Your plan should be delivered as:

1. **Executive Summary** (1 page)
   - High-level overview
   - Key milestones
   - Timeline
   - Success criteria

2. **Detailed Work Breakdown** (5-10 pages)
   - All tasks listed above
   - Dependencies mapped
   - Time estimates
   - Assigned to QoderCLI

3. **Integration Architecture** (2-3 pages)
   - System diagrams
   - Component interactions
   - Data flows

4. **Risk Management Plan** (2 pages)
   - All risks identified
   - Mitigation strategies
   - Contingency plans

5. **QoderCLI Prompt Sequence** (3-4 pages)
   - 9 specific prompts
   - Execution order
   - Verification steps

---

## DELIVERABLE

Create a comprehensive document that QoderCLI can follow step-by-step to:
1. Complete original Phase 3B test implementation
2. Integrate Gemini optimization system
3. Configure all 40+ agents
4. Set up monitoring and training
5. Validate everything works

The plan should be so detailed that QoderCLI can execute it without ambiguity.

---

**End of Gemini CLI Planning Prompt**

**Your Task**: Create the comprehensive Phase 3B implementation plan following the framework above.

**Format**: Markdown document, 15-20 pages, with clear sections and actionable items.

**Timeline**: Plan should span 3 weeks with daily breakdown.

**Audience**: QoderCLI (will execute), Gemini (will refine), Human reviewer (will approve).

---

Last Updated: December 6, 2024  
Status: Ready for Gemini CLI execution

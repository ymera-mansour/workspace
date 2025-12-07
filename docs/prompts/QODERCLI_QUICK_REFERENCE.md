# QoderCLI Quick Reference - YMERA Platform

## 🚀 Quick Start Commands

### Phase-by-Phase Implementation

```bash
# Phase 1: Foundation (Week 1)
qoder implement "Create YMERA platform foundation with project structure, configuration management, and logging"

# Phase 2: AI Providers (Week 2)
qoder implement "Integrate all AI providers (Groq, Gemini, OpenRouter, Mistral, Claude, OpenAI) with unified interface"

# Phase 3: Model Selection (Week 3)
qoder implement "Build intelligent model selection system with cost optimization and performance tracking"

# Phase 4: Workflow Engine (Week 4)
qoder implement "Create 6-phase workflow execution engine with context passing and quality validation"

# Phase 5: MCP Integration (Week 5)
qoder implement "Integrate Model Context Protocol tools (Brave Search, GitHub, PostgreSQL, Puppeteer, Slack)"

# Phase 6: Self-Healing (Week 6)
qoder implement "Implement self-healing system with circuit breakers, retry logic, and fallback chains"

# Phase 7: AI/ML Learning (Week 7)
qoder implement "Build AI/ML learning system that tracks executions and improves model selection over time"

# Phase 8: Agent Training (Week 8)
qoder implement "Create agent training system with training data generation and evaluation"

# Phase 9: Agents & Engines (Week 9)
qoder implement "Create specialized agents (Coding, Analysis, Research, Documentation) and execution engines"

# Phase 10: Security (Week 10)
qoder implement "Implement complete security system with JWT auth, RBAC, encryption, and audit logging"

# Phase 11: API & Web (Week 11)
qoder implement "Build REST API with FastAPI and web interface with dashboard"

# Phase 12: Monitoring (Week 12)
qoder implement "Add monitoring with Prometheus, Grafana dashboards, and alerting"

# Phase 13: Deployment (Week 13)
qoder implement "Create production deployment with Docker, scripts, and CI/CD pipeline"
```

## 📁 File Placement Reference

### After Each Phase, Files Go Here:

**Phase 1 - Foundation:**
```
src/core/config.py
src/core/logging_config.py
src/utils/helpers.py
tests/conftest.py
.env.example (add variables)
```

**Phase 2 - AI Providers:**
```
src/core/providers/base_provider.py
src/core/providers/groq_provider.py
src/core/providers/gemini_provider.py
src/core/providers/openrouter_provider.py
src/core/providers/mistral_provider.py
src/core/providers/anthropic_provider.py
src/core/providers/openai_provider.py
src/core/providers/provider_factory.py
tests/unit/test_providers.py
```

**Phase 3 - Model Selection:**
```
src/core/model_registry.py
src/core/model_selector.py
src/core/cost_calculator.py
src/core/performance_tracker.py
tests/unit/test_model_selector.py
```

**Phase 4 - Workflow Engine:**
```
src/core/workflow_types.py
src/core/workflow_engine.py
src/core/phase_executor.py
src/core/context_manager.py
src/core/quality_validator.py
tests/integration/test_workflow.py
```

**Phase 5 - MCP Integration:**
```
src/mcp/mcp_client.py
src/mcp/tool_registry.py
src/mcp/brave_search.py
src/mcp/github_integration.py
src/mcp/filesystem.py
tests/integration/test_mcp.py
```

**Phase 6 - Self-Healing:**
```
src/core/circuit_breaker.py
src/core/retry_handler.py
src/core/fallback_manager.py
src/core/health_monitor.py
tests/unit/test_resilience.py
```

**Phase 7 - AI/ML Learning:**
```
src/ml/learning_engine.py
src/ml/execution_tracker.py
src/ml/pattern_recognizer.py
src/ml/strategy_recommender.py
src/ml/model_evaluator.py
tests/unit/test_learning_system.py
```

**Phase 8 - Agent Training:**
```
src/agents/base_agent.py
src/agents/training_manager.py
src/agents/training_data_generator.py
src/agents/agent_evaluator.py
src/agents/fine_tuner.py
tests/integration/test_agent_training.py
```

**Phase 9 - Agents & Engines:**
```
src/agents/coding_agent.py
src/agents/analysis_agent.py
src/agents/research_agent.py
src/agents/documentation_agent.py
src/engines/code_engine.py
src/engines/database_engine.py
src/engines/search_engine.py
tests/integration/test_agents.py
```

**Phase 10 - Security:**
```
src/core/auth_manager.py
src/core/rbac.py
src/core/encryption.py
src/core/rate_limiter.py
src/core/audit_logger.py
src/core/security_middleware.py
tests/unit/test_security.py
```

**Phase 11 - API & Web:**
```
src/core/api.py
src/core/endpoints/completions.py
src/core/endpoints/models.py
src/core/endpoints/agents.py
src/web/static/index.html
src/web/static/dashboard.html
tests/e2e/test_api.py
```

**Phase 12 - Monitoring:**
```
src/core/metrics_collector.py
src/core/prometheus_exporter.py
deployment/prometheus.yml
deployment/grafana-dashboards/main.json
tests/unit/test_monitoring.py
```

**Phase 13 - Deployment:**
```
Dockerfile
docker-compose.yml
scripts/quick_start.sh
scripts/quick_start.ps1
deployment/nginx.conf
.env.example (finalize)
requirements.txt (finalize)
```

## 🔧 Models & MCPs to Include in Prompts

### Free AI Models (Include in Phase 2):

**Groq (FREE)**:
- llama-3.1-8b-instant
- llama-3.1-70b-versatile  
- llama-3.3-70b-versatile
- mixtral-8x7b-32768
- gemma2-9b-it

**Gemini (FREE TIER)**:
- gemini-1.5-flash (15 RPM)
- gemini-1.5-pro (2 RPM)

**OpenRouter (40+ FREE MODELS)**:
- amazon/nova-2-lite-v1:free
- amazon/nova-micro-v1:free
- meta-llama/llama-3.2-1b-instruct:free
- meta-llama/llama-3.2-3b-instruct:free
- meta-llama/llama-3.1-8b-instruct:free
- mistralai/mistral-7b-instruct:free
- mistralai/mixtral-8x7b-instruct:free
- google/gemma-2-9b-it:free
- google/gemma-7b-it:free
- microsoft/phi-3-mini-128k-instruct:free
- microsoft/phi-3-medium-128k-instruct:free
- qwen/qwen-2-7b-instruct:free
- qwen/qwen-2.5-7b-instruct:free
- deepseek/deepseek-coder-6.7b-instruct:free
- nousresearch/hermes-3-llama-3.1-405b:free
- liquid/lfm-40b:free

### MCP Servers (Include in Phase 5):

**Official MCPs**:
1. Brave Search - `@modelcontextprotocol/server-brave-search`
2. Filesystem - `@modelcontextprotocol/server-filesystem`
3. GitHub - `@modelcontextprotocol/server-github`
4. PostgreSQL - `@modelcontextprotocol/server-postgres`
5. Puppeteer - `@modelcontextprotocol/server-puppeteer`

**Community MCPs**:
6. Slack - `@modelcontextprotocol/server-slack`
7. Memory - `@modelcontextprotocol/server-memory`
8. Time - `@modelcontextprotocol/server-time`
9. Google Drive - `@modelcontextprotocol/server-gdrive`
10. Sequential Thinking - `@modelcontextprotocol/server-sequential-thinking`

## 📦 Essential Packages by Phase

### Phase 1:
```
python-dotenv
pydantic
pydantic-settings
```

### Phase 2:
```
groq
google-generativeai
mistralai
anthropic
openai
cohere
ai21
huggingface-hub
aiohttp
```

### Phase 3:
```
numpy
pandas
scikit-learn
```

### Phase 4:
```
asyncio
```

### Phase 5:
```
# Node.js packages (install with npm)
@modelcontextprotocol/server-brave-search
@modelcontextprotocol/server-filesystem
@modelcontextprotocol/server-github
```

### Phase 6:
```
tenacity
```

### Phase 7-8:
```
sqlalchemy
asyncpg
psycopg2-binary
sentence-transformers
```

### Phase 9:
```
ast
subprocess
bandit
radon
```

### Phase 10:
```
pyjwt
passlib[bcrypt]
cryptography
```

### Phase 11:
```
fastapi
uvicorn[standard]
python-multipart
```

### Phase 12:
```
prometheus-client
opentelemetry-api
```

### Phase 13:
```
docker
gunicorn
```

## 🧪 Testing Commands

```bash
# After each phase
pytest tests/unit/ -v
pytest tests/integration/ -v
pytest --cov=src tests/

# Specific phase tests
pytest tests/unit/test_providers.py -v
pytest tests/integration/test_workflow.py -v
pytest tests/integration/test_agents.py -v
```

## 📊 Validation Checklist

After each phase:
- [ ] All files created in correct locations
- [ ] All imports working
- [ ] Tests passing (>80% coverage)
- [ ] No syntax errors
- [ ] Documentation updated
- [ ] Integration with previous phases verified

## 🎯 Key Configuration Variables

Add to .env.example:

```bash
# AI Providers
GROQ_API_KEY=
GEMINI_API_KEY=
OPENROUTER_API_KEY=
MISTRAL_API_KEY=
ANTHROPIC_API_KEY=
OPENAI_API_KEY=

# MCP Tools
BRAVE_API_KEY=
GITHUB_TOKEN=

# Database
POSTGRES_CONNECTION_STRING=
REDIS_URL=

# Security
JWT_SECRET_KEY=
ADMIN_API_KEY=

# Server
HOST=0.0.0.0
PORT=8000

# Cost Control
MONTHLY_BUDGET=10.00

# Learning System
ENABLE_LEARNING=true
ENABLE_TRAINING=true
```

## 📈 Progress Tracking

Track your completion (update as you go):
- [ ] Week 1: Foundation
- [ ] Week 2: AI Providers
- [ ] Week 3: Model Selection
- [ ] Week 4: Workflow Engine
- [ ] Week 5: MCP Integration
- [ ] Week 6: Self-Healing
- [ ] Week 7: AI/ML Learning
- [ ] Week 8: Agent Training
- [ ] Week 9: Agents & Engines
- [ ] Week 10: Security
- [ ] Week 11: API & Web
- [ ] Week 12: Monitoring
- [ ] Week 13: Deployment

## 🚀 Final Deployment

```bash
# Build and deploy
docker-compose build
docker-compose up -d

# Verify
curl http://localhost:8000/health

# Access from iPhone
curl http://YOUR_PC_IP:8000/health
```

## 📞 Help

- Full Guide: `docs/prompts/QODERCLI_MASTER_IMPLEMENTATION_GUIDE.md`
- Phases 9-13: `docs/prompts/QODERCLI_PHASES_9-13.md`
- Architecture: `docs/architecture/`
- User Guides: `docs/guides/`

# YMERA Platform

Complete AI-powered workspace consolidation platform with multi-layer workflow system.

## Features

- ✅ **33-Layer Multi-Phase Workflow**: Progressive complexity from simple to expert
- ✅ **53 AI Models**: 51 FREE models across 9 providers
- ✅ **Phase X Validation**: Inter-phase quality control with A+ to F grading
- ✅ **Silent Monitoring**: Zero-interference performance tracking
- ✅ **6 Optimization Features**: Vector DB, streaming, caching, circuit breaker, analytics, batch processing
- ✅ **REST API**: Complete API for workflow execution
- ✅ **Docker Deployment**: Multi-service containerized deployment
- ✅ **Cost**: $0/month base system

## Quick Start

```bash
# 1. Clone and setup
git clone <repo>
cd workspace
cp 00-FOUNDATION/.env.template .env
# Edit .env with your API keys

# 2. Install
pip install -r 00-FOUNDATION/requirements.txt

# 3. Run
python main.py --repo-path /path/to/your/repo --phases all

# 4. Or use Docker
docker-compose up -d
```

## Architecture

```
Phase 0: Pre-Flight & Setup
    ↓
Phase X: Inter-Phase Validation
    ↓
Phase 1: Discovery (5 layers + 3 validators)
    ↓
Phase X: Inter-Phase Validation
    ↓
Phase 2: Analysis (4 layers + 2 validators)
    ↓
Phase X: Inter-Phase Validation
    ↓
Phase 3: Consolidation (5 layers + 3 validators)
    ↓
Phase X: Inter-Phase Validation
    ↓
Phase 4: Testing (4 layers + 2 validators)
    ↓
Phase X: Inter-Phase Validation
    ↓
Phase 5: Integration (3 layers + 2 validators)
```

## API Usage

```bash
# Start API server
uvicorn api:app --host 0.0.0.0 --port 8000

# Execute workflow
curl -X POST http://localhost:8000/api/v1/workflow/execute \
  -H "Content-Type: application/json" \
  -d '{"repo_path": "/path/to/repo", "phases": ["all"]}'

# Check status
curl http://localhost:8000/api/v1/workflow/{workflow_id}/status

# Get leaderboard
curl http://localhost:8000/api/v1/models/leaderboard
```

## Testing

```bash
# Run all tests
pytest TESTS/ -v

# Run with coverage
pytest TESTS/ --cov=. --cov-report=html
```

## Documentation

- [Complete Workflow Analysis](DOCUMENTATION/WORKFLOW_COMPLETE_ANALYSIS_AND_REORGANIZATION.md)
- [Implementation Tools](DOCUMENTATION/IMPLEMENTATION_TOOLS_FOR_MODELS.md)
- [Phase 1 Layers](DOCUMENTATION/PHASE1_DISCOVERY_LAYERS.md)
- [ML/Learning System](DOCUMENTATION/ML_LEARNING_SYSTEM_COMPREHENSIVE.md)
- [AI Models Review](DOCUMENTATION/AI_MODELS_AND_MCP_SYSTEMS_REVIEW.md)

## License

MIT License - See LICENSE file for details

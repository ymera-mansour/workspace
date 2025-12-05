# Gemini CLI Integration Guide

## Overview

Gemini CLI specializes in:
- **Pre-implementation planning**: Detailed requirement analysis and planning
- **Code review**: Quality, security, and best practices verification
- **Documentation generation**: Comprehensive docs from code
- **Architecture validation**: Design patterns and structure verification
- **Test case generation**: Extensive test scenario creation
- **Performance analysis**: Bottleneck identification and optimization suggestions

## Role in Three-CLI Workflow

```
QoderCLI (Implementation) ←→ Gemini CLI (Planning/Review) ←→ Copilot CLI (Assistance/Fixes)
```

**Gemini CLI** acts as the **planner and reviewer**, ensuring quality before and after implementation.

## Phase-by-Phase Tasks

### Phase 3B: Agent Integration

#### Pre-Implementation Planning

```bash
gemini plan \
  --task "Analyze requirements for Phase 3B: Agent-Model Integration" \
  --context "40+ existing agents in organized_system/src/agents/" \
  --output "plans/phase_3b_plan.md"
```

**Expected Output**:
```markdown
# Phase 3B Implementation Plan

## Agents Identified
- coding, code_review, testing (code-related)
- analysis, analytics, reporting (data-related)
- documentation, documentation_v2 (docs-related)
... (all 40+ agents)

## Recommended Architecture
1. AgentModelConnector
   - Auto-detect agent type from name
   - Map to optimal free models
   - Provision MCP tools

2. MultiAgentOrchestrator
   - 5 coordination types
   - Result aggregation
   - Resource management

3. ExistingAgentWrapper
   - Wrap without modifications
   - Add model integration
   - Track for learning

## Model Assignments
- Coding agents → DeepSeek Coder (free), Llama 70B
- Analysis agents → Llama 3.3 70B, Gemini Flash
... (all mappings)

## Implementation Steps
1. Create integration/ directory
2. Implement AgentModelConnector
3. Implement MultiAgentOrchestrator
4. Implement ExistingAgentWrapper
5. Write unit tests (target: 80%+ coverage)
6. Write integration tests
7. Run real-world validation
```

#### Post-Implementation Review

```bash
gemini review \
  --files "organized_system/src/core_services/integration/**/*.py" \
  --check quality,security,best-practices,performance \
  --output "reviews/phase_3b_review.md"
```

**Expected Output**:
```markdown
# Phase 3B Code Review

## Files Reviewed
- agent_model_connector.py (450 lines)
- multi_agent_orchestrator.py (380 lines)
- existing_agent_wrapper.py (220 lines)

## Quality Assessment

### Strengths
✓ Clean separation of concerns
✓ Type hints throughout
✓ Comprehensive docstrings
✓ Error handling present

### Issues Found

#### Critical (Must Fix)
❌ agent_model_connector.py:125 - SQL injection risk in query
   ```python
   # Bad:
   query = f"SELECT * FROM agents WHERE name = '{agent_name}'"
   
   # Fix:
   query = "SELECT * FROM agents WHERE name = %s"
   cursor.execute(query, (agent_name,))
   ```

#### High Priority
⚠ multi_agent_orchestrator.py:87 - Memory leak in parallel execution
   - Resource pool not properly closed
   - Add context manager

#### Medium Priority
⚠ existing_agent_wrapper.py:45 - Missing input validation
   - Add Pydantic model validation

### Security Scan
- 1 critical vulnerability (SQL injection)
- 2 medium issues (input validation)
- Recommendation: Fix before proceeding

### Performance Analysis
- AgentModelConnector.recommend_models(): 85ms (target: <100ms) ✓
- MultiAgentOrchestrator.execute_parallel(): 1.2s (acceptable)
- No major bottlenecks detected

### Best Practices
✓ Follows Python PEP 8
✓ Async/await used appropriately
✓ Logging implemented
⚠ Consider adding circuit breakers for external calls

## Recommendations
1. Fix SQL injection (CRITICAL)
2. Add input validation with Pydantic
3. Implement resource cleanup (context managers)
4. Add circuit breakers for resilience

## Overall Score: 7.5/10
Fix critical issues before validation.
```

#### Architecture Validation

```bash
gemini validate-architecture \
  --source "organized_system/src/core_services/integration/" \
  --patterns "singleton,factory,wrapper,orchestrator" \
  --output "validation/architecture_validation.md"
```

**Expected Output**:
```markdown
# Architecture Validation: Phase 3B

## Design Patterns Used
✓ Singleton: AgentModelConnector (appropriate)
✓ Factory: Model recommendation factory (good)
✓ Wrapper: ExistingAgentWrapper (clean)
✓ Orchestrator: MultiAgentOrchestrator (well-structured)

## Structure Analysis
✓ Modular design - easy to extend
✓ Low coupling between components
✓ High cohesion within modules
✓ Dependency injection used

## Extensibility
✓ Easy to add new agent types
✓ Easy to add new models
✓ Easy to add coordination types
✓ Configuration-driven

## Recommendations
- Add interface/protocol definitions for better typing
- Consider plugin architecture for coordination types
- Document extension points

## Overall: Excellent architecture ✓
```

### Phase 4: Collective Learning System

#### Pre-Implementation Planning

```bash
gemini plan \
  --task "Design Collective Learning System for 40+ agents" \
  --requirements "Learn patterns from all agents, auto-apply insights" \
  --output "plans/phase_4_plan.md"
```

**Expected Output**:
```markdown
# Phase 4: Collective Learning System Plan

## Objectives
- Track all agent executions
- Analyze patterns across agents
- Learn optimal model selections
- Auto-apply insights

## Components

### 1. CollectiveLearningEngine
- Store executions in PostgreSQL
- Use embeddings for similarity search
- Recognize 5 pattern types
- Generate insights

### 2. AgentPerformanceAnalyzer
- Track metrics per agent
- Identify trends
- Common error patterns
- Improvement tracking

### 3. PatternRecognizer
- Model selection patterns
- Tool usage patterns
- Collaboration patterns
- Error recovery patterns
- Task decomposition patterns

## Data Model
```sql
CREATE TABLE agent_executions (
  id SERIAL PRIMARY KEY,
  agent_name VARCHAR(100),
  task TEXT,
  model_used VARCHAR(100),
  tools_used TEXT[],
  success BOOLEAN,
  duration_ms INTEGER,
  quality_score FLOAT,
  timestamp TIMESTAMP
);

CREATE TABLE learned_insights (
  id SERIAL PRIMARY KEY,
  pattern_type VARCHAR(50),
  insight TEXT,
  confidence FLOAT,
  applied_count INTEGER,
  success_rate FLOAT,
  created_at TIMESTAMP
);
```

## Learning Algorithms
1. Frequency analysis (which models work best)
2. Correlation analysis (tool combinations)
3. Sequence mining (agent collaboration patterns)
4. Anomaly detection (error patterns)

## Auto-Application Strategy
1. Threshold: Apply insights with >0.8 confidence
2. A/B testing: Test before full rollout
3. Rollback: Revert if performance drops
4. Monitoring: Track impact of applied insights

## Implementation Timeline
- Week 1: Database schema, execution tracking
- Week 2: Pattern recognition algorithms
- Week 3: Insight generation
- Week 4: Auto-application and monitoring
```

#### Test Case Generation

```bash
gemini generate-tests \
  --source "organized_system/src/core_services/learning/collective_learning_engine.py" \
  --scenarios unit,integration,performance \
  --output "tests/test_collective_learning.py"
```

**Expected Output**:
```python
"""
Generated Test Cases for Collective Learning Engine
Coverage: Unit + Integration + Performance
"""

import pytest
import asyncio
from datetime import datetime, timedelta

from organized_system.src.core_services.learning import CollectiveLearningEngine

class TestCollectiveLearningEngineUnit:
    """Unit tests for individual methods"""
    
    @pytest.fixture
    def learning_engine(self):
        return CollectiveLearningEngine()
    
    def test_track_execution_success(self, learning_engine):
        """Test successful execution tracking"""
        execution = {
            'agent_name': 'coding',
            'task': 'Write fibonacci function',
            'model_used': 'llama-3.1-70b',
            'tools_used': ['github', 'filesystem'],
            'success': True,
            'duration_ms': 1250,
            'quality_score': 0.92
        }
        
        result = learning_engine.track_execution(execution)
        
        assert result['success'] == True
        assert result['execution_id'] is not None
    
    def test_track_execution_validation(self, learning_engine):
        """Test input validation"""
        invalid_execution = {
            'agent_name': '',  # Invalid: empty name
            'task': 'Test'
        }
        
        with pytest.raises(ValueError):
            learning_engine.track_execution(invalid_execution)
    
    def test_recognize_pattern_model_selection(self, learning_engine):
        """Test model selection pattern recognition"""
        # Add 100 successful executions with llama-70b
        for i in range(100):
            learning_engine.track_execution({
                'agent_name': 'coding',
                'model_used': 'llama-3.1-70b',
                'success': True,
                'quality_score': 0.90 + (i % 10) * 0.01
            })
        
        # Add 20 failures with different model
        for i in range(20):
            learning_engine.track_execution({
                'agent_name': 'coding',
                'model_used': 'different-model',
                'success': False,
                'quality_score': 0.60
            })
        
        patterns = learning_engine.recognize_patterns('model_selection')
        
        assert len(patterns) > 0
        best_pattern = patterns[0]
        assert best_pattern['model'] == 'llama-3.1-70b'
        assert best_pattern['success_rate'] > 0.80

class TestCollectiveLearningEngineIntegration:
    """Integration tests with database"""
    
    @pytest.fixture
    async def engine_with_db(self):
        """Setup engine with test database"""
        engine = CollectiveLearningEngine(db_url="postgresql://test:test@localhost/test_learning")
        await engine.initialize()
        yield engine
        await engine.cleanup()
    
    @pytest.mark.asyncio
    async def test_full_learning_cycle(self, engine_with_db):
        """Test complete learning cycle"""
        # 1. Track executions
        for i in range(50):
            await engine_with_db.track_execution({
                'agent_name': 'analysis',
                'task': f'Analyze dataset {i}',
                'model_used': 'llama-3.3-70b',
                'tools_used': ['database', 'filesystem'],
                'success': True,
                'duration_ms': 2000,
                'quality_score': 0.88
            })
        
        # 2. Recognize patterns
        patterns = await engine_with_db.recognize_all_patterns()
        assert len(patterns) > 0
        
        # 3. Generate insights
        insights = await engine_with_db.generate_insights(patterns)
        assert len(insights) > 0
        
        # 4. Auto-apply insights
        applied = await engine_with_db.apply_insights(insights, threshold=0.8)
        assert applied > 0

class TestCollectiveLearningEnginePerformance:
    """Performance tests"""
    
    @pytest.mark.asyncio
    async def test_bulk_tracking_performance(self):
        """Test bulk execution tracking"""
        engine = CollectiveLearningEngine()
        
        executions = [
            {'agent_name': f'agent_{i}', 'success': True}
            for i in range(1000)
        ]
        
        start = datetime.now()
        await engine.track_executions_bulk(executions)
        duration = (datetime.now() - start).total_seconds()
        
        # Should complete 1000 inserts in < 5 seconds
        assert duration < 5.0
    
    @pytest.mark.asyncio
    async def test_pattern_recognition_performance(self):
        """Test pattern recognition speed"""
        engine = CollectiveLearningEngine()
        
        # Add 10K executions
        for i in range(10000):
            await engine.track_execution({
                'agent_name': f'agent_{i % 40}',
                'success': True
            })
        
        start = datetime.now()
        patterns = await engine.recognize_all_patterns()
        duration = (datetime.now() - start).total_seconds()
        
        # Should analyze 10K executions in < 10 seconds
        assert duration < 10.0
        assert len(patterns) > 0
```

### Phase 5: Multi-Agent Training

#### Documentation Generation

```bash
gemini document \
  --source "organized_system/src/core_services/training/multi_agent_training_system.py" \
  --format markdown \
  --include-examples \
  --output "docs/training_system_api.md"
```

**Expected Output**:
```markdown
# Multi-Agent Training System API Documentation

## Overview

The Multi-Agent Training System provides batch and continuous training capabilities for all 40+ agents in the YMERA platform.

## Classes

### MultiAgentTrainingSystem

Main orchestrator for agent training.

```python
class MultiAgentTrainingSystem:
    """
    Multi-agent training orchestrator.
    
    Manages training for multiple agents simultaneously with:
    - Batch training (100+ examples per agent)
    - Continuous learning from production
    - Before/after evaluation
    - Resource management (10 agents in parallel)
    """
    
    def __init__(self, 
                 max_parallel: int = 10,
                 db_connection: str = None):
        """
        Initialize training system.
        
        Args:
            max_parallel: Maximum agents to train simultaneously
            db_connection: PostgreSQL connection string
        """
```

#### Methods

##### train_agents_batch()

Train multiple agents using batch training.

```python
async def train_agents_batch(
    self,
    agent_names: List[str],
    training_data: Dict[str, List[TrainingExample]],
    evaluation_set: Dict[str, List[TrainingExample]]
) -> TrainingResults:
    """
    Train multiple agents in batch mode.
    
    Args:
        agent_names: List of agent names to train
        training_data: Training examples per agent
        evaluation_set: Evaluation examples per agent
    
    Returns:
        TrainingResults with before/after metrics
    
    Example:
        >>> training_system = MultiAgentTrainingSystem()
        >>> results = await training_system.train_agents_batch(
        ...     agent_names=['coding', 'testing'],
        ...     training_data={
        ...         'coding': [example1, example2, ...],
        ...         'testing': [example1, example2, ...]
        ...     },
        ...     evaluation_set={
        ...         'coding': [eval1, eval2, ...],
        ...         'testing': [eval1, eval2, ...]
        ...     }
        ... )
        >>> print(f"Coding agent improved: {results['coding'].improvement}%")
    """
```

## Usage Examples

### Basic Training

```python
from organized_system.src.core_services.training import MultiAgentTrainingSystem

# Initialize
training_system = MultiAgentTrainingSystem(max_parallel=10)

# Train single agent
results = await training_system.train_agent(
    agent_name='coding',
    training_examples=coding_training_data,
    evaluation_examples=coding_eval_data
)

print(f"Before: {results.baseline_metrics}")
print(f"After: {results.post_training_metrics}")
print(f"Improvement: {results.improvement_percentage}%")
```

### Batch Training All Agents

```python
# Get all agent names
agent_names = get_all_agent_names()  # Returns 40+ agents

# Generate training data
training_data = {}
evaluation_data = {}

for agent_name in agent_names:
    training_data[agent_name] = generate_training_examples(agent_name, count=100)
    evaluation_data[agent_name] = generate_evaluation_examples(agent_name, count=20)

# Train all agents (10 at a time)
results = await training_system.train_agents_batch(
    agent_names=agent_names,
    training_data=training_data,
    evaluation_set=evaluation_data
)

# Print results
for agent_name, agent_results in results.items():
    print(f"{agent_name}: {agent_results.improvement_percentage}% improvement")
```

### Continuous Learning

```python
# Enable continuous learning from production
training_system.enable_continuous_learning(
    check_interval_hours=24,
    min_examples=50,
    auto_retrain=True
)

# System will automatically:
# 1. Collect production executions daily
# 2. Generate training examples from successes
# 3. Retrain agents when enough examples collected
# 4. Evaluate and rollback if performance drops
```

## Performance Characteristics

- **Training Speed**: 100 examples in ~5 minutes per agent
- **Parallel Capacity**: 10 agents simultaneously
- **Memory Usage**: ~200MB per agent during training
- **Evaluation Speed**: 20 examples in ~30 seconds

## Best Practices

1. **Batch Size**: Use 100-200 examples for initial training
2. **Evaluation Set**: Use 20-30 examples for evaluation
3. **Parallel Training**: Train 10 agents at a time for optimal resource usage
4. **Continuous Learning**: Check daily, retrain when 50+ new examples
5. **Monitoring**: Track improvement metrics over time

## Error Handling

All methods raise descriptive exceptions:
- `InsufficientTrainingDataError`: Not enough examples
- `TrainingFailedError`: Training process failed
- `EvaluationFailedError`: Evaluation failed
- `ResourceExhaustedError`: Too many parallel trainings

```python
try:
    results = await training_system.train_agent('coding', examples, eval_set)
except InsufficientTrainingDataError as e:
    print(f"Need more examples: {e}")
except TrainingFailedError as e:
    print(f"Training failed: {e}")
    # Rollback to previous version
    await training_system.rollback_agent('coding')
```
```

## Quality Improvement Cycle

### The Review-Fix Loop

```
┌────────────────┐
│ QoderCLI       │
│ Implements     │
└───────┬────────┘
        │
        ↓
┌────────────────┐
│ Gemini CLI     │
│ Reviews        │
└───────┬────────┘
        │
        ↓
   Issues Found?
        │
    Yes │         No
        ↓          ↓
┌────────────────┐  ┌────────────────┐
│ Copilot CLI    │  │ Validation     │
│ Fixes Issues   │  │ Tests          │
└───────┬────────┘  └────────────────┘
        │
        ↓
┌────────────────┐
│ Gemini CLI     │
│ Re-reviews     │
└───────┬────────┘
        │
        ↓
    Loop until
    all pass
```

## Commands Reference

### Pre-Implementation

```bash
# Analyze requirements
gemini analyze --task "Phase 4 requirements" --context organized_system

# Create implementation plan
gemini plan --phase 4 --output plans/phase_4.md

# Generate architecture diagram
gemini diagram --source organized_system/src --output diagrams/architecture.png
```

### Code Review

```bash
# Quick review
gemini review --files src/**/*.py

# Comprehensive review
gemini review --files src/**/*.py --check all --strict --output reviews/full_review.md

# Security-focused review
gemini review --files src/**/*.py --security-only --output reviews/security_review.md

# Performance-focused review
gemini review --files src/**/*.py --performance-only --suggest-optimizations
```

### Documentation

```bash
# Generate API docs
gemini document --source src/ --format markdown --output docs/api.md

# Generate user guide
gemini guide --source src/ --audience users --output docs/user_guide.md

# Generate developer docs
gemini guide --source src/ --audience developers --include-examples --output docs/dev_guide.md
```

### Testing

```bash
# Generate unit tests
gemini generate-tests --source src/module.py --type unit --coverage-target 90

# Generate integration tests
gemini generate-tests --source src/ --type integration --scenarios 10

# Generate performance tests
gemini generate-tests --source src/ --type performance --benchmark
```

### Architecture

```bash
# Validate architecture
gemini validate-architecture --source src/ --patterns SOLID,DRY,KISS

# Suggest improvements
gemini improve-architecture --source src/ --output recommendations.md

# Check dependencies
gemini analyze-dependencies --source src/ --detect-circular --detect-unused
```

## Integration with Validation System

Gemini CLI is automatically invoked during validation:

```python
# In validation script
class ValidationRunner:
    async def run_validation(self, phase: str):
        # 1. Run tests
        test_results = await self.run_tests()
        
        # 2. Gemini CLI review
        review_results = subprocess.run([
            'gemini', 'review',
            '--files', f'organized_system/src/phase_{phase}/**/*.py',
            '--check', 'all',
            '--json'
        ], capture_output=True)
        
        review = json.loads(review_results.stdout)
        
        # 3. Fail if critical issues found
        if review['critical_issues'] > 0:
            raise ValidationError(f"Critical issues found: {review['critical_issues']}")
        
        # 4. Continue with other validations
        ...
```

## Summary

Gemini CLI provides:

✅ **Pre-Planning**: Detailed requirement analysis and implementation plans  
✅ **Code Review**: Quality, security, best practices, performance checks  
✅ **Documentation**: Auto-generated comprehensive docs  
✅ **Architecture Validation**: Design pattern verification  
✅ **Test Generation**: Unit, integration, performance test creation  
✅ **Quality Enforcement**: Ensures high standards before proceeding

**Gemini CLI is the quality guardian of the implementation process!** 🛡️

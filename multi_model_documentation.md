# YMERA Multi-Model Execution System

## Overview

The YMERA Multi-Model Execution System intelligently matches AI models to tasks and executes complex workflows using multiple specialized models across different phases.

## Architecture

### Components

1. **AgentModelMatcher** (`agent_model_matcher.py`)
   - Profiles for all agents (coding, database, analysis, etc.)
   - Capabilities for all AI models (Gemini, Groq, Codestral, etc.)
   - Intelligent matching algorithm
   - Multi-model strategy creation

2. **MultiModelExecutor** (`multi_model_executor.py`)
   - Phase-based execution engine
   - Context passing between phases
   - Result aggregation
   - Error handling and fallbacks

### Execution Phases

1. **PLANNING** - Fast model understands task and creates plan
2. **RESEARCH** - Reasoning model gathers information
3. **GENERATION** - Specialized model creates solution
4. **REVIEW** - Quality model checks output
5. **REFINEMENT** - Specialized model improves result
6. **VALIDATION** - Accuracy model performs final check

## Model Capabilities

### Speed-Optimized Models
- **Groq Llama 3.3 70B**: Ultra-fast (10/10), excellent reasoning
- **Groq Llama 3.1 8B**: Instant (10/10), simple tasks only
- **Gemini 2.5 Flash**: Fast (9/10), balanced quality

### Code-Specialized Models
- **Codestral**: Code generation (10/10), refactoring expert
- **DeepSeek Coder**: Code understanding (10/10), high quality
- **Gemini 2.5 Pro**: Multi-modal (8/10), comprehensive

### Quality-Focused Models
- **Gemini 2.5 Pro**: Highest quality (10/10), reasoning (10/10)
- **Azure GPT-4o**: Premium quality (10/10), multi-modal
- **AI21 Jamba**: Reasoning (10/10), accuracy (10/10)

### Balanced Models
- **Mistral Small**: Efficient (8/8/8), general purpose
- **DeepSeek Chat**: Conversation (8/8), reasoning
- **Cohere Command R+**: Enterprise (9/9), RAG-optimized

## Usage Examples

### Basic Usage

```python
from core_services.ai_mcp.multi_model_executor import get_multi_model_executor

executor = get_multi_model_executor()

result = await executor.execute_with_multi_model(
    agent_name="coding_agent",
    task_description="Create a FastAPI endpoint for user management",
    task_parameters={
        "language": "python",
        "framework": "fastapi"
    }
)

print(f"Strategy: {result.strategy_type}")
print(f"Success: {result.successful_phases}/{result.total_phases}")
print(f"Output: {result.final_result}")
```

### Custom Phase Selection

```python
# Only run specific phases
result = await executor.execute_with_multi_model(
    agent_name="coding_agent",
    task_description="Quick code snippet",
    task_parameters={"language": "python"},
    enable_phases=["planning", "generation"]  # Skip review/refinement
)
```

### Model Strategy Inspection

```python
from core_services.ai_mcp.agent_model_matcher import get_agent_model_matcher

matcher = get_agent_model_matcher()

strategy = await matcher.match_agent_to_models(
    agent_name="coding_agent",
    task_description="Complex microservices architecture",
    task_parameters={"complexity": "high"}
)

# Inspect which models will be used for each phase
for phase in strategy['phases']:
    print(f"{phase['phase']}: {phase['primary_model']['model']}")
    print(f"Reason: {phase['reason']}")
```

## Agent Profiles

### Coding Agent
- **Task Type**: Code generation, refactoring
- **Preferred Models**: Codestral, DeepSeek Coder, Gemini 2.5 Pro
- **Phases**: Planning → Generation → Review → Refinement
- **Requirements**: High accuracy, creativity

### Database Agent
- **Task Type**: SQL generation, data operations
- **Preferred Models**: Codestral, Gemini Flash, DeepSeek Coder
- **Phases**: Planning → Generation → Validation
- **Requirements**: High accuracy, speed

### Analysis Agent
- **Task Type**: Data analysis, insights
- **Preferred Models**: Jamba, Gemini Pro, GPT-4o
- **Phases**: Research → Generation → Review → Validation
- **Requirements**: High accuracy, reasoning

### Web Scraping Agent
- **Task Type**: Web extraction, parsing
- **Preferred Models**: Gemini Flash, Llama 3.3 70B
- **Phases**: Planning → Generation → Validation
- **Requirements**: Speed, accuracy

### Documentation Agent
- **Task Type**: Documentation generation
- **Preferred Models**: Gemini Pro, GPT-4o, Claude 3.5
- **Phases**: Research → Generation → Review → Refinement
- **Requirements**: Creativity, accuracy

## Result Structure

### MultiModelResult

```python
@dataclass
class MultiModelResult:
    strategy_type: str              # "multi_model" or "single_model"
    total_phases: int               # Number of phases executed
    successful_phases: int          # Number of successful phases
    phase_results: List[PhaseResult]  # Results from each phase
    final_result: Any               # Final combined output
    total_execution_time: float     # Total time in seconds
    total_tokens_used: int          # Total tokens consumed
    total_cost: float               # Estimated cost
    models_used: List[str]          # List of models used
```

### PhaseResult

```python
@dataclass
class PhaseResult:
    phase: str                      # Phase name
    model_used: str                 # Model identifier
    provider_used: str              # Provider name
    success: bool                   # Success status
    result: Any                     # Phase output
    execution_time: float           # Time in seconds
    tokens_used: int                # Tokens consumed
    error: Optional[str]            # Error message if failed
```

## Advanced Features

### Adaptive Model Selection

The system adapts model selection based on:

1. **Task Complexity**: Detected from keywords
   - "simple" → Fast models
   - "complex" → High-quality models
   
2. **Speed Requirements**: Detected from keywords
   - "fast", "urgent" → Speed-optimized models
   
3. **Creativity Requirements**: Detected from keywords
   - "creative", "innovative" → Creative models
   
4. **Performance History**: Learns from past executions

### Context Passing

Each phase receives context from previous phases:

```python
context = {
    "task_description": "...",
    "task_parameters": {...},
    "phase_outputs": {
        "planning": "...",      # Plan from planning phase
        "generation": "...",    # Output from generation phase
        "review": "..."         # Review from review phase
    }
}
```

### Error Handling

The system handles failures gracefully:

1. **Critical Phase Failure**: Aborts execution
   - Planning phase failure → Stop
   - Generation phase failure → Stop

2. **Non-Critical Phase Failure**: Continues execution
   - Review phase failure → Skip to refinement
   - Refinement phase failure → Use generation output

3. **Fallback Models**: Automatically tries fallback models on failure

## Performance Optimization

### Parallel Execution

When possible, non-dependent phases run in parallel:

```python
# Research and generation can run in parallel if independent
await asyncio.gather(
    execute_phase("research", context),
    execute_phase("generation", context)
)
```

### Caching

Results are cached to avoid redundant executions:

```python
# If same task requested again, return cached result
cache_key = f"{agent_name}:{task_description}:{task_parameters}"
```

### Token Optimization

The system minimizes token usage:

1. Concise prompts for planning phase
2. Full context only when necessary
3. Streaming for long outputs

## Testing

### Unit Tests

```python
import pytest
from core_services.ai_mcp.agent_model_matcher import AgentModelMatcher

@pytest.mark.asyncio
async def test_model_matching():
    matcher = AgentModelMatcher()
    
    strategy = await matcher.match_agent_to_models(
        agent_name="coding_agent",
        task_description="Simple function",
        task_parameters={}
    )
    
    assert strategy["strategy_type"] in ["single_model", "multi_model"]
    assert "phases" in strategy or "model" in strategy
```

### Integration Tests

```python
@pytest.mark.asyncio
async def test_multi_model_execution():
    executor = get_multi_model_executor()
    
    result = await executor.execute_with_multi_model(
        agent_name="coding_agent",
        task_description="Create hello world function",
        task_parameters={"language": "python"}
    )
    
    assert result.successful_phases > 0
    assert result.final_result is not None
```

## Configuration

### Environment Variables

```bash
# Enable/disable multi-model execution
ENABLE_MULTI_MODEL=true

# Default strategy
DEFAULT_STRATEGY=multi_model  # or single_model

# Performance tuning
MAX_PARALLEL_PHASES=2
PHASE_TIMEOUT=300
```

### Agent Configuration

```python
# Override agent profile
custom_profile = AgentTaskProfile(
    agent_name="custom_agent",
    task_type="custom",
    preferred_models=["gemini-2.5-flash"],
    phases_needed=[TaskPhase.GENERATION],
    can_use_multiple_models=False
)

matcher.agent_profiles["custom_agent"] = custom_profile
```

## Troubleshooting

### Common Issues

1. **"No profile for agent"**
   - Agent name not registered
   - Add profile or use default

2. **"Phase execution timeout"**
   - Task too complex for model
   - Increase timeout or simplify task

3. **"All phases failed"**
   - Check API keys are valid
   - Verify network connectivity
   - Review error logs

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Detailed logs for model selection and execution
```

## Best Practices

1. **Task Description**: Be specific and clear
2. **Phase Selection**: Use only necessary phases
3. **Error Handling**: Always check result.success
4. **Performance**: Profile for optimization
5. **Cost**: Monitor token usage

## Future Enhancements

- [ ] Learning from execution history
- [ ] Dynamic phase ordering
- [ ] Cost optimization algorithms
- [ ] Custom scoring functions
- [ ] A/B testing of strategies
- [ ] Real-time model availability checking
- [ ] Automatic fallback chains
- [ ] Performance prediction

## Support

For issues or questions:
- GitHub Issues: [link]
- Documentation: [link]
- Examples: See `multi_model_integration_example.py`

# Copilot CLI Integration Guide

## Overview

Copilot CLI specializes in:
- **Code Completion**: Assist with complex implementations
- **Refactoring**: Improve code quality and maintainability
- **Bug Fixing**: Identify and fix defects
- **Test Writing**: Create comprehensive test suites
- **Security Hardening**: Add security controls
- **Performance Optimization**: Optimize slow code paths

## Role in Three-CLI Workflow

```
QoderCLI (Implementation) ← Copilot CLI (Assistance) → Gemini CLI (Review)
```

**Copilot CLI** acts as the **assistant and fixer**, helping with implementation and resolving issues.

## Core Capabilities

### 1. Code Completion

Help complete complex implementations:

```bash
# Complete a function
copilot complete \
  --file "organized_system/src/core_services/integration/agent_model_connector.py" \
  --function "recommend_models" \
  --context "Function should analyze agent name and return optimal free models"
```

**Example:**

Input:
```python
def recommend_models(self, agent_name: str) -> List[ModelConfig]:
    """Recommend optimal models for agent"""
    # TODO: Implement
```

Copilot Output:
```python
def recommend_models(self, agent_name: str) -> List[ModelConfig]:
    """Recommend optimal models for agent"""
    agent_type = self._detect_agent_type(agent_name)
    
    # Model recommendations based on agent type
    model_map = {
        'coding': [
            ModelConfig('deepseek-coder', provider='openrouter', cost=0.0),
            ModelConfig('llama-3.1-70b', provider='groq', cost=0.0),
            ModelConfig('codestral', provider='mistral', cost=0.001)
        ],
        'analysis': [
            ModelConfig('llama-3.3-70b', provider='groq', cost=0.0),
            ModelConfig('gemini-flash', provider='google', cost=0.0),
            ModelConfig('claude-sonnet', provider='anthropic', cost=0.003)
        ],
        'documentation': [
            ModelConfig('llama-3.1-70b', provider='groq', cost=0.0),
            ModelConfig('gemini-flash', provider='google', cost=0.0)
        ]
    }
    
    # Return models for agent type (fallback to general if not found)
    return model_map.get(agent_type, model_map['coding'])
```

### 2. Refactoring

Improve code quality:

```bash
# Refactor for better performance
copilot refactor \
  --file "organized_system/src/core_services/integration/multi_agent_orchestrator.py" \
  --goal "improve-performance" \
  --metrics "response-time,memory-usage"
```

**Before:**
```python
async def execute_parallel(self, agents: List[Tuple[str, str]]) -> List[Dict]:
    """Execute agents in parallel"""
    results = []
    for agent_name, task in agents:
        result = await self._execute_agent(agent_name, task)
        results.append(result)
    return results
```

**After (Copilot Refactored):**
```python
async def execute_parallel(self, agents: List[Tuple[str, str]]) -> List[Dict]:
    """Execute agents in parallel (optimized)"""
    # Use asyncio.gather for true parallelism
    tasks = [
        self._execute_agent(agent_name, task)
        for agent_name, task in agents
    ]
    
    # Execute all tasks concurrently with timeout
    try:
        results = await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True),
            timeout=30.0
        )
    except asyncio.TimeoutError:
        # Handle timeout gracefully
        results = [{'error': 'timeout'} for _ in agents]
    
    return [r for r in results if not isinstance(r, Exception)]
```

### 3. Bug Fixing

Identify and fix bugs:

```bash
# Fix a specific bug
copilot fix \
  --file "organized_system/src/core_services/integration/existing_agent_wrapper.py" \
  --issue "Memory leak when wrapping agents" \
  --analyze
```

**Issue Identified:**
```python
class ExistingAgentWrapper:
    def __init__(self):
        self.wrapped_agents = []  # Memory leak: never cleaned up
    
    def wrap_agent(self, agent):
        wrapped = AgentProxy(agent)
        self.wrapped_agents.append(wrapped)  # Keeps growing
        return wrapped
```

**Copilot Fix:**
```python
class ExistingAgentWrapper:
    def __init__(self):
        self.wrapped_agents = weakref.WeakValueDictionary()  # Use weak references
    
    def wrap_agent(self, agent):
        agent_id = id(agent)
        
        # Reuse existing wrapper if available
        if agent_id in self.wrapped_agents:
            return self.wrapped_agents[agent_id]
        
        wrapped = AgentProxy(agent)
        self.wrapped_agents[agent_id] = wrapped
        return wrapped
    
    def cleanup_unused(self):
        """Manually trigger cleanup of unused wrappers"""
        # Weak references auto-cleanup, but can force gc
        import gc
        gc.collect()
```

### 4. Test Writing

Generate comprehensive tests:

```bash
# Generate unit tests
copilot test \
  --source "organized_system/src/core_services/integration/agent_model_connector.py" \
  --output "organized_system/tests/unit/test_agent_model_connector.py" \
  --coverage-target 90 \
  --include-edge-cases
```

**Generated Tests:**
```python
import pytest
from unittest.mock import Mock, patch

from organized_system.src.core_services.integration import AgentModelConnector

class TestAgentModelConnector:
    """Comprehensive tests for AgentModelConnector"""
    
    @pytest.fixture
    def connector(self):
        return AgentModelConnector()
    
    # Happy path tests
    def test_recommend_models_coding_agent(self, connector):
        """Test model recommendation for coding agent"""
        models = connector.recommend_models('coding')
        
        assert len(models) > 0
        assert models[0].provider in ['openrouter', 'groq', 'mistral']
        assert models[0].cost == 0.0  # First choice should be free
    
    # Edge cases
    def test_recommend_models_unknown_agent(self, connector):
        """Test recommendation for unknown agent type"""
        models = connector.recommend_models('unknown_agent_xyz')
        
        assert len(models) > 0  # Should fallback to default
    
    def test_recommend_models_empty_name(self, connector):
        """Test with empty agent name"""
        with pytest.raises(ValueError, match="Agent name cannot be empty"):
            connector.recommend_models('')
    
    def test_recommend_models_none_name(self, connector):
        """Test with None agent name"""
        with pytest.raises(TypeError):
            connector.recommend_models(None)
    
    def test_recommend_models_special_characters(self, connector):
        """Test agent name with special characters"""
        models = connector.recommend_models('coding-agent-v2')
        assert len(models) > 0
    
    # Performance tests
    def test_recommend_models_performance(self, connector):
        """Test recommendation performance"""
        import time
        
        start = time.time()
        for _ in range(100):
            connector.recommend_models('coding')
        duration = time.time() - start
        
        # Should complete 100 calls in < 100ms
        assert duration < 0.1
    
    # Integration tests
    @patch('organized_system.src.core_services.integration.ModelRegistry')
    def test_recommend_models_with_registry(self, mock_registry, connector):
        """Test integration with ModelRegistry"""
        mock_registry.get_available_models.return_value = [
            {'name': 'llama-3.1-70b', 'provider': 'groq'}
        ]
        
        models = connector.recommend_models('coding')
        assert mock_registry.get_available_models.called
```

### 5. Security Hardening

Add security controls:

```bash
# Harden security
copilot secure \
  --file "organized_system/src/core_services/integration/agent_model_connector.py" \
  --checks "injection,validation,authentication" \
  --output-diff
```

**Before:**
```python
def execute_query(self, query: str):
    """Execute database query"""
    return self.db.execute(f"SELECT * FROM agents WHERE {query}")
```

**After (Copilot Secured):**
```python
def execute_query(self, query: str):
    """Execute database query (secured)"""
    # Input validation
    if not query or len(query) > 1000:
        raise ValueError("Invalid query length")
    
    # Parameterized query to prevent SQL injection
    allowed_fields = ['name', 'type', 'status']
    parsed_query = self._parse_safe_query(query, allowed_fields)
    
    # Use parameterized query
    return self.db.execute(
        "SELECT * FROM agents WHERE name = %s AND type = %s",
        (parsed_query['name'], parsed_query['type'])
    )

def _parse_safe_query(self, query: str, allowed_fields: List[str]) -> Dict:
    """Parse query safely"""
    # Only allow whitelisted fields
    parts = query.split(' AND ')
    parsed = {}
    
    for part in parts:
        field, value = part.split('=')
        field = field.strip()
        
        if field not in allowed_fields:
            raise SecurityError(f"Field '{field}' not allowed")
        
        parsed[field] = value.strip().strip("'\"")
    
    return parsed
```

### 6. Performance Optimization

Optimize slow code:

```bash
# Optimize performance
copilot optimize \
  --file "organized_system/src/core_services/learning/collective_learning_engine.py" \
  --target "recognize_patterns" \
  --goal "10x-faster"
```

**Before:**
```python
def recognize_patterns(self) -> List[Pattern]:
    """Recognize patterns from executions"""
    executions = self.get_all_executions()  # 10K records
    
    patterns = []
    for i, exec1 in enumerate(executions):
        for exec2 in executions[i+1:]:
            if self._is_similar(exec1, exec2):
                patterns.append(Pattern(exec1, exec2))
    
    return patterns  # O(n²) - very slow!
```

**After (Copilot Optimized):**
```python
def recognize_patterns(self) -> List[Pattern]:
    """Recognize patterns from executions (optimized)"""
    executions = self.get_all_executions()
    
    # Use embeddings for fast similarity search
    embeddings = self._compute_embeddings_batch(executions)
    
    # Build index for O(log n) search
    from annoy import AnnoyIndex
    index = AnnoyIndex(len(embeddings[0]), 'angular')
    
    for i, embedding in enumerate(embeddings):
        index.add_item(i, embedding)
    
    index.build(10)  # 10 trees
    
    # Find similar pairs efficiently
    patterns = []
    seen = set()
    
    for i, embedding in enumerate(embeddings):
        # Get 5 nearest neighbors
        neighbors = index.get_nns_by_vector(embedding, 5)
        
        for neighbor_idx in neighbors:
            if neighbor_idx != i:
                pair = tuple(sorted([i, neighbor_idx]))
                if pair not in seen:
                    seen.add(pair)
                    patterns.append(
                        Pattern(executions[i], executions[neighbor_idx])
                    )
    
    return patterns  # O(n log n) - 10x faster!
```

## Phase-by-Phase Assistance

### Phase 3B: Agent Integration

#### Complete Complex Logic

```bash
copilot complete \
  --file "multi_agent_orchestrator.py" \
  --function "execute_collaborative" \
  --context "Agents negotiate via round-robin for max 3 rounds"
```

#### Generate Tests

```bash
copilot test \
  --source "agent_model_connector.py" \
  --coverage-target 85 \
  --include-performance-tests
```

#### Fix Issues from Gemini Review

```bash
# After Gemini CLI identifies issues
copilot fix \
  --file "agent_model_connector.py" \
  --issues "issues_from_gemini.json" \
  --auto-apply
```

### Phase 4: Collective Learning

#### Optimize Database Queries

```bash
copilot optimize \
  --file "collective_learning_engine.py" \
  --target "track_execution" \
  --database-queries
```

#### Add Caching

```bash
copilot enhance \
  --file "pattern_recognizer.py" \
  --add-feature "caching" \
  --cache-strategy "redis"
```

### Phase 5: Multi-Agent Training

#### Parallelize Training

```bash
copilot refactor \
  --file "multi_agent_training_system.py" \
  --function "train_agents_batch" \
  --goal "parallelize" \
  --max-workers 10
```

#### Add Progress Tracking

```bash
copilot enhance \
  --file "training_manager.py" \
  --add-feature "progress-tracking" \
  --with-websockets
```

### Phase 6: Security

#### Implement JWT Authentication

```bash
copilot implement \
  --file "agent_security_manager.py" \
  --feature "jwt-authentication" \
  --with-refresh-tokens
```

#### Add Rate Limiting

```bash
copilot implement \
  --file "rate_limiter.py" \
  --feature "token-bucket-rate-limiting" \
  --backend "redis"
```

## Integration with Validation System

Copilot CLI is invoked when issues are found:

```python
class IssueResolver:
    async def resolve_issues(self, issues: List[Issue]):
        """Automatically resolve issues using Copilot CLI"""
        
        for issue in issues:
            if issue.severity == 'critical':
                # Use Copilot to fix
                result = subprocess.run([
                    'copilot', 'fix',
                    '--file', issue.file,
                    '--line', str(issue.line),
                    '--issue', issue.description,
                    '--auto-apply'
                ], capture_output=True)
                
                if result.returncode == 0:
                    print(f"✓ Fixed: {issue.description}")
                else:
                    print(f"✗ Failed to fix: {issue.description}")
```

## Commands Reference

### Code Completion

```bash
# Complete function
copilot complete --file src/module.py --function func_name

# Complete class
copilot complete --file src/module.py --class ClassName

# Complete with context
copilot complete --file src/module.py --function func_name --context "Should handle async operations"
```

### Refactoring

```bash
# General refactoring
copilot refactor --file src/module.py --goal "improve-readability"

# Performance refactoring
copilot refactor --file src/module.py --goal "improve-performance"

# Extract method
copilot refactor --file src/module.py --extract-method --lines 10-50

# Rename
copilot refactor --file src/module.py --rename old_name new_name
```

### Bug Fixing

```bash
# Fix specific bug
copilot fix --file src/module.py --issue "Memory leak in function X"

# Auto-fix all issues
copilot fix --file src/module.py --auto-fix-all

# Fix with analysis
copilot fix --file src/module.py --issue "Bug description" --analyze --suggest-tests
```

### Test Writing

```bash
# Generate unit tests
copilot test --source src/module.py --type unit --coverage-target 90

# Generate integration tests
copilot test --source src/ --type integration --scenarios 5

# Generate with mocks
copilot test --source src/module.py --use-mocks --mock-external-calls
```

### Security

```bash
# Security scan
copilot secure --file src/module.py --scan

# Fix security issues
copilot secure --file src/module.py --fix-all

# Add authentication
copilot secure --file src/api.py --add-auth jwt

# Add input validation
copilot secure --file src/module.py --add-validation pydantic
```

### Performance

```bash
# Profile code
copilot optimize --file src/module.py --profile

# Optimize function
copilot optimize --file src/module.py --function slow_function --goal "10x-faster"

# Add caching
copilot optimize --file src/module.py --add-cache redis

# Parallelize
copilot optimize --file src/module.py --parallelize --max-workers 10
```

## Best Practices

### 1. Iterative Improvement

```bash
# First pass: Basic implementation
copilot complete --file module.py --function process_data

# Second pass: Add error handling
copilot enhance --file module.py --add-error-handling

# Third pass: Optimize
copilot optimize --file module.py --function process_data

# Fourth pass: Add tests
copilot test --source module.py --coverage-target 90
```

### 2. Review Before Apply

```bash
# Generate diff first
copilot fix --file module.py --issue "Bug" --dry-run --output diff.patch

# Review diff
cat diff.patch

# Apply if good
copilot fix --file module.py --issue "Bug" --auto-apply
```

### 3. Combine with Gemini Reviews

```bash
# 1. Implement with QoderCLI
qoder implement "Phase 3B integration"

# 2. Review with Gemini
gemini review --files src/**/*.py --output issues.json

# 3. Fix with Copilot
copilot fix --issues issues.json --auto-apply

# 4. Re-review
gemini review --files src/**/*.py --verify-fixes
```

## Example: Complete Fix Cycle

### Issue Identified by Gemini

```json
{
  "file": "agent_model_connector.py",
  "line": 125,
  "severity": "critical",
  "type": "security",
  "issue": "SQL injection vulnerability",
  "code": "query = f\"SELECT * FROM agents WHERE name = '{agent_name}'\""
}
```

### Fix with Copilot

```bash
copilot fix \
  --file "agent_model_connector.py" \
  --line 125 \
  --issue "SQL injection vulnerability" \
  --auto-apply
```

### Result

**Before:**
```python
def get_agent(self, agent_name: str):
    query = f"SELECT * FROM agents WHERE name = '{agent_name}'"
    return self.db.execute(query)
```

**After:**
```python
def get_agent(self, agent_name: str):
    # Input validation
    if not agent_name or not agent_name.isalnum():
        raise ValueError("Invalid agent name")
    
    # Parameterized query
    query = "SELECT * FROM agents WHERE name = %s"
    return self.db.execute(query, (agent_name,))
```

### Verify with Tests

```bash
# Copilot generates security test
copilot test \
  --source "agent_model_connector.py" \
  --function "get_agent" \
  --security-tests

# Generated test:
def test_get_agent_sql_injection_protection():
    connector = AgentModelConnector()
    
    # Try SQL injection
    with pytest.raises(ValueError):
        connector.get_agent("'; DROP TABLE agents; --")
```

## Summary

Copilot CLI provides:

✅ **Code Completion**: Assist with complex implementations  
✅ **Refactoring**: Improve code quality and performance  
✅ **Bug Fixing**: Identify and fix defects automatically  
✅ **Test Generation**: Comprehensive test suites  
✅ **Security Hardening**: Add security controls  
✅ **Performance Optimization**: Speed up slow code  

**Copilot CLI is the implementation assistant and problem solver!** 🔧

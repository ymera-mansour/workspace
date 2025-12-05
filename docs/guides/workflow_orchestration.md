# Workflow Orchestration & Task Execution Guide

## 🎯 Overview

This guide explains how the YMERA platform orchestrates complex workflows using multiple AI models, MCP tools, and intelligent task distribution.

## Table of Contents

1. [Workflow Concepts](#workflow-concepts)
2. [Task Execution Phases](#task-execution-phases)
3. [Model Selection Strategy](#model-selection-strategy)
4. [Natural Language Processing](#natural-language-processing)
5. [Quality Benchmarking](#quality-benchmarking)
6. [Self-Healing & Error Recovery](#self-healing--error-recovery)
7. [Examples](#examples)

---

## Workflow Concepts

### Workflow Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    USER INPUT                                │
│  "Create a REST API with authentication and deploy it"      │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│               INTENT ANALYZER (NLP)                          │
│  • Extracts task requirements                                │
│  • Identifies complexity level                               │
│  • Determines required capabilities                          │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│            WORKFLOW PLANNER                                  │
│  • Breaks down into sub-tasks                                │
│  • Identifies dependencies                                   │
│  • Selects execution strategy                                │
│  • Assigns models to phases                                  │
└──────────────────────┬──────────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┬──────────────┐
        │              │              │              │
        ▼              ▼              ▼              ▼
   ┌────────┐    ┌────────┐    ┌────────┐    ┌────────┐
   │ Phase  │    │ Phase  │    │ Phase  │    │ Phase  │
   │   1    │ -> │   2    │ -> │   3    │ -> │   4    │
   └────────┘    └────────┘    └────────┘    └────────┘
        │              │              │              │
        └──────────────┴──────────────┴──────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│            QUALITY VALIDATOR                                 │
│  • Checks output quality                                     │
│  • Runs validation tests                                     │
│  • Triggers refinement if needed                             │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                 FINAL RESULT                                 │
│  • Validated output                                          │
│  • Execution metrics                                         │
│  • Cost breakdown                                            │
└─────────────────────────────────────────────────────────────┘
```

### Execution Strategies

#### 1. **Single Model Strategy**
- Use one model for entire workflow
- Best for: Simple tasks, speed-critical operations
- Example: "Summarize this text"

#### 2. **Multi-Model Strategy**
- Different models for different phases
- Best for: Complex tasks, quality-critical operations
- Example: "Research topic → Generate code → Review → Test"

#### 3. **Parallel Execution**
- Multiple models work simultaneously
- Best for: High-confidence tasks, A/B testing
- Example: Generate 3 versions and pick best

#### 4. **Cascade Strategy**
- Sequential refinement through multiple models
- Best for: Quality improvement, iterative tasks
- Example: Draft → Refine → Polish → Validate

---

## Task Execution Phases

### Phase 1: Planning (Fast Model)

**Purpose**: Understand task and create execution plan

**Best Models**:
- Groq Llama 3.1 8B Instant (fastest)
- Gemini 1.5 Flash (balanced)
- Mistral Small (efficient)

**Activities**:
```python
phase_result = {
    "understanding": "Task requires API creation with auth",
    "subtasks": [
        "Design API endpoints",
        "Implement authentication",
        "Create database schema",
        "Write tests",
        "Deploy to server"
    ],
    "dependencies": {
        "authentication": ["database_schema"],
        "tests": ["api_endpoints", "authentication"]
    },
    "complexity": "medium",
    "estimated_time": "30 minutes"
}
```

### Phase 2: Research (Reasoning Model)

**Purpose**: Gather information and best practices

**Best Models**:
- Llama 3.3 70B (best reasoning)
- Gemini 1.5 Pro (long context)
- AI21 Jamba (accuracy)

**Activities**:
```python
research_result = {
    "best_practices": [
        "Use JWT for authentication",
        "Implement rate limiting",
        "Use bcrypt for password hashing"
    ],
    "libraries": ["fastapi", "pyjwt", "passlib"],
    "security_concerns": [
        "SQL injection prevention",
        "XSS protection",
        "CSRF tokens"
    ],
    "examples": ["link1", "link2"]
}
```

### Phase 3: Generation (Specialized Model)

**Purpose**: Create the actual implementation

**Best Models**:
- Codestral (code generation)
- DeepSeek Coder (code quality)
- Gemini 1.5 Pro (comprehensive)

**Activities**:
```python
generation_result = {
    "code": "# FastAPI implementation...",
    "files": {
        "main.py": "...",
        "auth.py": "...",
        "models.py": "...",
        "tests.py": "..."
    },
    "documentation": "API usage guide"
}
```

### Phase 4: Review (Quality Model)

**Purpose**: Check for issues and improvements

**Best Models**:
- Claude 3.5 Sonnet (best reviewer)
- Gemini 1.5 Pro (comprehensive)
- GPT-4o (detail-oriented)

**Activities**:
```python
review_result = {
    "issues_found": [
        {
            "type": "security",
            "severity": "high",
            "description": "Missing input validation",
            "suggestion": "Add pydantic models"
        }
    ],
    "code_quality": 8.5,
    "improvements": ["Add error handling", "Improve comments"]
}
```

### Phase 5: Refinement (Specialized Model)

**Purpose**: Apply improvements from review

**Best Models**:
- Codestral (code refinement)
- DeepSeek Coder (optimization)
- Same model as Phase 3 for consistency

**Activities**:
```python
refinement_result = {
    "changes_made": [
        "Added input validation",
        "Improved error handling",
        "Enhanced documentation"
    ],
    "code": "# Improved implementation...",
    "quality_improvement": 1.2  # 20% better
}
```

### Phase 6: Validation (Accuracy Model)

**Purpose**: Final quality check and testing

**Best Models**:
- AI21 Jamba (accuracy)
- Gemini 1.5 Pro (comprehensive)
- Claude 3.5 Sonnet (thorough)

**Activities**:
```python
validation_result = {
    "passed": True,
    "test_results": {
        "unit_tests": "passed",
        "integration_tests": "passed",
        "security_scan": "passed"
    },
    "performance": {
        "response_time": "< 100ms",
        "throughput": "1000 req/s"
    },
    "final_score": 9.2
}
```

---

## Model Selection Strategy

### Automatic Model Selection

```python
class ModelSelector:
    """Intelligent model selection based on task characteristics"""
    
    def select_model(self, task_description: str, phase: str) -> str:
        # Analyze task
        complexity = self.analyze_complexity(task_description)
        urgency = self.detect_urgency(task_description)
        creativity = self.needs_creativity(task_description)
        
        if phase == "planning":
            return self.select_fast_model()
        
        elif phase == "research":
            if "complex" in task_description.lower():
                return "llama-3.3-70b-versatile"  # Best reasoning
            return "gemini-1.5-flash"  # Fast research
        
        elif phase == "generation":
            if "code" in task_description.lower():
                return "codestral"  # Code specialist
            elif creativity > 0.7:
                return "gemini-1.5-pro"  # Creative tasks
            return "llama-3.1-70b-versatile"  # General
        
        elif phase == "review":
            return "claude-3.5-sonnet"  # Best reviewer
        
        elif phase == "refinement":
            # Use same model as generation for consistency
            return self.previous_generation_model
        
        elif phase == "validation":
            return "ai21/jamba-1.5-large"  # Accuracy
        
        return self.default_model
    
    def analyze_complexity(self, description: str) -> float:
        """Return complexity score 0-1"""
        indicators = {
            "simple": -0.3,
            "basic": -0.2,
            "complex": 0.4,
            "advanced": 0.5,
            "enterprise": 0.6
        }
        
        score = 0.5  # baseline
        for keyword, weight in indicators.items():
            if keyword in description.lower():
                score += weight
        
        return max(0.0, min(1.0, score))
```

### Multi-Model Selection

```python
class MultiModelStrategy:
    """Strategy for using multiple models"""
    
    def create_strategy(self, task: Task) -> Dict:
        if task.complexity < 0.3:
            # Simple task - single fast model
            return {
                "strategy": "single_model",
                "model": "groq/llama-3.1-8b-instant"
            }
        
        elif task.complexity < 0.6:
            # Medium task - multi-model with essential phases
            return {
                "strategy": "multi_model",
                "phases": {
                    "planning": "groq/llama-3.1-8b-instant",
                    "generation": "codestral" if task.is_code else "gemini-1.5-flash",
                    "validation": "gemini-1.5-flash"
                }
            }
        
        else:
            # Complex task - full multi-model pipeline
            return {
                "strategy": "multi_model",
                "phases": {
                    "planning": "gemini-1.5-flash",
                    "research": "llama-3.3-70b-versatile",
                    "generation": "codestral" if task.is_code else "gemini-1.5-pro",
                    "review": "claude-3.5-sonnet",
                    "refinement": "codestral" if task.is_code else "gemini-1.5-pro",
                    "validation": "ai21/jamba-1.5-large"
                }
            }
```

---

## Natural Language Processing

### Intent Extraction

```python
class IntentAnalyzer:
    """Extract intent from natural language"""
    
    async def analyze(self, user_input: str) -> Dict:
        # Use fast model for intent classification
        prompt = f"""
        Analyze this request and extract:
        1. Main task type (code, data, research, etc.)
        2. Complexity level (simple/medium/complex)
        3. Required capabilities
        4. Urgency indicators
        
        Request: {user_input}
        
        Respond in JSON format.
        """
        
        result = await self.fast_model.complete(prompt)
        return json.loads(result)
```

### Task Decomposition

```python
class TaskDecomposer:
    """Break complex tasks into subtasks"""
    
    async def decompose(self, task: str) -> List[SubTask]:
        prompt = f"""
        Break this task into subtasks with dependencies:
        
        Task: {task}
        
        Format:
        1. Subtask name - Dependencies: [list] - Estimated time
        2. ...
        """
        
        result = await self.model.complete(prompt)
        return self.parse_subtasks(result)
```

---

## Quality Benchmarking

### Automated Quality Metrics

```python
class QualityBenchmark:
    """Measure output quality"""
    
    def evaluate(self, output: Any, task_type: str) -> QualityScore:
        scores = {}
        
        # Code quality (for code tasks)
        if task_type == "code":
            scores["syntax"] = self.check_syntax(output)
            scores["complexity"] = self.analyze_complexity(output)
            scores["coverage"] = self.test_coverage(output)
            scores["security"] = self.security_scan(output)
        
        # Content quality (for text tasks)
        elif task_type == "text":
            scores["coherence"] = self.check_coherence(output)
            scores["relevance"] = self.check_relevance(output)
            scores["completeness"] = self.check_completeness(output)
        
        # Overall score
        overall = sum(scores.values()) / len(scores)
        
        return QualityScore(
            overall=overall,
            breakdown=scores,
            passed=overall >= 0.7
        )
```

### Validation Tests

```python
class ValidationSuite:
    """Run validation tests"""
    
    async def validate(self, output: Any, requirements: List[str]) -> ValidationResult:
        results = []
        
        for requirement in requirements:
            result = await self.test_requirement(output, requirement)
            results.append(result)
        
        return ValidationResult(
            passed=all(r.passed for r in results),
            tests=results,
            coverage=len([r for r in results if r.passed]) / len(results)
        )
```

---

## Self-Healing & Error Recovery

### Automatic Retry Strategy

```python
class SelfHealingExecutor:
    """Executor with automatic error recovery"""
    
    async def execute_with_healing(self, task: Task) -> Result:
        max_retries = 3
        fallback_models = [
            "primary_model",
            "fallback_model_1",
            "fallback_model_2"
        ]
        
        for attempt in range(max_retries):
            try:
                model = fallback_models[attempt]
                result = await self.execute(task, model)
                
                # Validate result
                if self.quality_check(result):
                    return result
                
                # If quality is poor, try refinement
                if attempt < max_retries - 1:
                    logger.info(f"Quality low, trying refinement with {fallback_models[attempt + 1]}")
                    continue
                
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed: {e}")
                
                if attempt < max_retries - 1:
                    # Try fallback model
                    continue
                else:
                    # All retries exhausted
                    raise
        
        raise Exception("All retry attempts failed")
```

### Circuit Breaker Pattern

```python
class CircuitBreaker:
    """Prevent cascading failures"""
    
    def __init__(self, failure_threshold=5, timeout=60):
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.last_failure_time = None
    
    async def execute(self, func, *args, **kwargs):
        if self.state == "OPEN":
            if time.time() - self.last_failure_time >= self.timeout:
                self.state = "HALF_OPEN"
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = await func(*args, **kwargs)
            
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0
            
            return result
            
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
            
            raise
```

---

## Examples

### Example 1: Simple Task (Single Model)

```python
# User input
task = "Summarize this article about AI"

# Automatic selection
strategy = {
    "strategy": "single_model",
    "model": "groq/llama-3.1-8b-instant",  # Fastest
    "estimated_cost": "$0.00",
    "estimated_time": "2 seconds"
}

# Execution
result = await executor.execute(task, strategy)
```

### Example 2: Medium Task (Multi-Model)

```python
# User input
task = "Create a Python function to parse CSV files with error handling"

# Automatic selection
strategy = {
    "strategy": "multi_model",
    "phases": {
        "planning": {
            "model": "gemini-1.5-flash",
            "purpose": "Understand requirements"
        },
        "generation": {
            "model": "codestral",
            "purpose": "Write code"
        },
        "validation": {
            "model": "deepseek-coder",
            "purpose": "Check quality"
        }
    },
    "estimated_cost": "$0.01",
    "estimated_time": "15 seconds"
}

# Execution with context passing
result = await executor.execute_multi_phase(task, strategy)
```

### Example 3: Complex Task (Full Pipeline)

```python
# User input
task = """
Create a microservices architecture for an e-commerce platform with:
- User authentication service
- Product catalog service
- Order management service
- Payment processing service
- Notification service

Include API documentation, database schemas, and deployment configuration.
"""

# Automatic selection
strategy = {
    "strategy": "multi_model",
    "phases": {
        "planning": {
            "model": "gemini-1.5-flash",
            "purpose": "Create architecture plan"
        },
        "research": {
            "model": "llama-3.3-70b-versatile",
            "purpose": "Best practices for microservices"
        },
        "generation": {
            "model": "codestral",
            "purpose": "Generate code for all services",
            "parallel": True,  # Generate services in parallel
            "subtasks": [
                "auth_service",
                "catalog_service",
                "order_service",
                "payment_service",
                "notification_service"
            ]
        },
        "review": {
            "model": "claude-3.5-sonnet",
            "purpose": "Review architecture and code"
        },
        "refinement": {
            "model": "codestral",
            "purpose": "Apply improvements"
        },
        "validation": {
            "model": "gemini-1.5-pro",
            "purpose": "Final validation and documentation"
        }
    },
    "estimated_cost": "$0.50",
    "estimated_time": "5 minutes"
}

# Execution with quality monitoring
result = await executor.execute_with_monitoring(task, strategy)
```

---

## Advanced Features

### Learning System

```python
class LearningSystem:
    """Learn from execution history"""
    
    def record_execution(self, task: Task, strategy: Dict, result: Result):
        """Record execution for learning"""
        self.history.append({
            "task_type": task.type,
            "complexity": task.complexity,
            "strategy": strategy,
            "models_used": result.models_used,
            "quality_score": result.quality_score,
            "execution_time": result.execution_time,
            "cost": result.cost,
            "success": result.success
        })
    
    def recommend_strategy(self, task: Task) -> Dict:
        """Recommend strategy based on history"""
        similar_tasks = self.find_similar_tasks(task)
        
        if not similar_tasks:
            return self.default_strategy(task)
        
        # Find best performing strategy
        best = max(similar_tasks, key=lambda x: x["quality_score"])
        return best["strategy"]
```

### Cost Optimization

```python
class CostOptimizer:
    """Optimize for cost while maintaining quality"""
    
    def optimize_strategy(self, task: Task, budget: float) -> Dict:
        """Create cost-optimized strategy"""
        
        if budget <= 0.01:
            # Ultra-low budget - use free models only
            return {
                "strategy": "single_model",
                "model": "groq/llama-3.1-70b-versatile"  # Free & good
            }
        
        elif budget <= 0.10:
            # Low budget - balance quality and cost
            return {
                "strategy": "multi_model",
                "phases": {
                    "planning": "groq/llama-3.1-8b-instant",  # Free
                    "generation": "gemini-1.5-flash",  # Cheap
                    "validation": "groq/llama-3.1-70b-versatile"  # Free
                }
            }
        
        else:
            # Normal budget - optimize for quality
            return self.quality_optimized_strategy(task)
```

---

## Next Steps

1. [Model Selection Guide](./model_selection.md)
2. [MCP Tools Integration](./mcp_complete_guide.md)
3. [Security Best Practices](./security.md)
4. [Performance Optimization](../architecture/performance_benchmarking.md)

---

**Updated**: December 2024  
**Maintainer**: YMERA Team

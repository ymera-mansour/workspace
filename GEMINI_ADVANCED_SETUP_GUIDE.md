# Gemini Advanced Setup Guide
## Complete Configuration for Endpoints, Key Rotation, Agent Training, and Task Tracking

**Created in response to**: User request for detailed configuration with endpoints, middleware, key rotation for multiple organizations, and agent training system.

---

## Overview

This guide covers the advanced Gemini integration features:
1. **Model Endpoints Configuration** - Complete endpoint details for all 4 Gemini models
2. **API Key Rotation** - Automatic rotation between multiple keys from different Google organizations
3. **Agent Training System** - Learning system for 40+ agents to optimize model selection
4. **Task-Model Tracking** - Real-time tracking of which model handles which task
5. **Middleware Integration** - Complete integration layer connecting all components

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Model Endpoints Configuration](#model-endpoints-configuration)
3. [API Key Rotation Setup](#api-key-rotation-setup)
4. [Agent Training System](#agent-training-system)
5. [Task-Model Tracking](#task-model-tracking)
6. [Middleware Integration](#middleware-integration)
7. [Configuration Files](#configuration-files)
8. [Examples](#examples)

---

## Quick Start

### 1. Set Up Environment Variables

Create a `.env` file with multiple API keys:

```bash
# Primary API key
GEMINI_API_KEY=your_primary_key_here

# Secondary keys from different Google organizations
GEMINI_API_KEY_1=your_org1_key_here
GEMINI_ORG_1=research_team

GEMINI_API_KEY_2=your_org2_key_here
GEMINI_ORG_2=development_team

GEMINI_API_KEY_3=your_org3_key_here
GEMINI_ORG_3=production_team

# Backup key
GEMINI_API_KEY_BACKUP=your_backup_key_here
```

### 2. Install Dependencies

```bash
pip install google-generativeai asyncio pyyaml
```

### 3. Initialize the System

```python
from gemini_advanced_config import GeminiMiddleware

# Initialize with all components
middleware = GeminiMiddleware()

# Execute a request
result = await middleware.execute_request(
    agent_name="coding_agent",
    task_type="code_generation",
    task_description="Create a REST API endpoint"
)

print(f"Model used: {result['model_used']}")
print(f"API key used: {result['api_key_used']}")
print(f"Success: {result['success']}")
```

---

## Model Endpoints Configuration

### Complete Endpoint Details

Each Gemini model has specific endpoints and capabilities:

#### Gemini 2.0 Flash (Experimental)

```python
{
    "model_id": "gemini-2.0-flash-exp",
    "endpoint": "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent",
    "streaming_endpoint": "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:streamGenerateContent",
    "api_version": "v1beta",
    
    # Capabilities
    "supports_streaming": True,
    "supports_multimodal": True,  # Images, video, audio
    "supports_function_calling": True,
    "supports_grounding": True,  # Google Search integration
    
    # Rate Limits (Free Tier)
    "rpm_limit": 10,    # Requests per minute
    "rpd_limit": 1500,  # Requests per day
    
    # Token Limits
    "max_input_tokens": 1_000_000,
    "max_output_tokens": 8192,
    
    # Configuration
    "temperature_range": (0.0, 2.0),
    "default_temperature": 0.7,
    "top_p": 0.95,
    "top_k": 40
}
```

**Best For:**
- Fast, general-purpose tasks
- Multimodal analysis (images, diagrams)
- Real-time streaming applications
- High-throughput scenarios

#### Gemini 1.5 Flash

```python
{
    "model_id": "gemini-1.5-flash",
    "endpoint": "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent",
    "streaming_endpoint": "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:streamGenerateContent",
    
    "rpm_limit": 15,
    "rpd_limit": 1500,
    "max_input_tokens": 1_000_000,
    "max_output_tokens": 8192,
}
```

**Best For:**
- Standard code generation
- Documentation
- Data analysis
- Reliable, consistent quality

#### Gemini 1.5 Pro

```python
{
    "model_id": "gemini-1.5-pro",
    "endpoint": "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent",
    
    "rpm_limit": 2,     # Lower rate limit
    "rpd_limit": 50,    # Lower daily limit
    "max_input_tokens": 2_000_000,  # Largest context window
    "max_output_tokens": 8192,
}
```

**Best For:**
- Complex reasoning tasks
- Large context (up to 2M tokens)
- Critical, high-quality tasks
- Architectural reviews

#### Gemini 1.5 Flash 8B

```python
{
    "model_id": "gemini-1.5-flash-8b",
    "endpoint": "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-8b:generateContent",
    
    "rpm_limit": 15,
    "rpd_limit": 4000,  # Highest daily limit!
    "max_input_tokens": 1_000_000,
    "supports_multimodal": False,
}
```

**Best For:**
- Bulk operations
- Simple, repetitive tasks
- High-throughput processing
- Quick validations

### Making API Calls

Example of direct API call with endpoint:

```python
import google.generativeai as genai

# Configure API key
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Select model
model = genai.GenerativeModel('gemini-2.0-flash-exp')

# Generate content
response = model.generate_content(
    "Explain quantum computing",
    generation_config={
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 2048,
    }
)

print(response.text)
```

---

## API Key Rotation Setup

### Why Key Rotation?

When you have multiple Google API keys (from different organizations or projects), rotation helps:
- **Distribute load** across multiple quotas
- **Avoid hitting limits** on any single key
- **Increase reliability** with automatic failover
- **Maximize throughput** by using all available quotas

### Configuration

The `APIKeyRotationManager` handles automatic rotation:

```python
from gemini_advanced_config import APIKeyRotationManager

# Initialize
key_manager = APIKeyRotationManager()

# Keys are automatically loaded from environment variables:
# GEMINI_API_KEY (primary)
# GEMINI_API_KEY_1, GEMINI_API_KEY_2, ... (secondary)
# GEMINI_ORG_1, GEMINI_ORG_2, ... (organization names)

# Or manually add keys
key_manager.add_key(
    key_id="org_research",
    api_key="your_research_team_key",
    organization="Research Team"
)

key_manager.add_key(
    key_id="org_production",
    api_key="your_production_key",
    organization="Production"
)
```

### Rotation Strategies

#### 1. Round Robin (Default)

Cycles through keys in order:

```python
key_manager.rotation_strategy = "round_robin"

# Calls will use: key1 → key2 → key3 → key1 → key2 → ...
```

#### 2. Least Used

Selects the key with lowest current usage:

```python
key_manager.rotation_strategy = "least_used"

# Always picks the key with the most remaining quota
```

#### 3. Weighted

Intelligently weights selection based on remaining quota:

```python
key_manager.rotation_strategy = "weighted"

# Keys with more quota are more likely to be selected
# Automatically adapts as quotas are consumed
```

### Automatic Rotation on Limit

```python
# Check if rotation is needed
new_key_id = await key_manager.check_and_rotate_if_needed(
    key_id="current_key",
    model_id="gemini-2.0-flash-exp"
)

if new_key_id:
    print(f"Rotated to: {new_key_id}")
```

### Key Status Monitoring

```python
# Get status of all keys
status = key_manager.get_status()

for key_id, info in status.items():
    print(f"\n{key_id} ({info['organization']}):")
    print(f"  RPM used: {info['rpm_used']}")
    print(f"  RPD used: {info['rpd_used']}")
    print(f"  Active: {info['is_active']}")
    print(f"  Failures: {info['failure_count']}")
```

### Environment Setup for Multiple Keys

```bash
# .env file structure
# ==================

# Primary key (required)
GEMINI_API_KEY=AIzaSy...

# Organization 1 keys
GEMINI_API_KEY_1=AIzaSy...
GEMINI_ORG_1=research_team

# Organization 2 keys
GEMINI_API_KEY_2=AIzaSy...
GEMINI_ORG_2=development_team

# Organization 3 keys
GEMINI_API_KEY_3=AIzaSy...
GEMINI_ORG_3=qa_team

# Backup key
GEMINI_API_KEY_BACKUP=AIzaSy...
```

---

## Agent Training System

### Overview

The `AgentTrainingSystem` learns which models work best for each of your 40+ agents over time.

### How It Works

1. **Tracks Performance**: Records success rate, latency, quality for each agent-model-task combination
2. **Learns Preferences**: Automatically identifies best models for each agent and task type
3. **Optimizes Selection**: Uses historical data to recommend optimal models

### Setup

```python
from gemini_advanced_config import AgentTrainingSystem

# Initialize (loads historical data if available)
training = AgentTrainingSystem(storage_path="agent_training_data.json")

# After each execution, record results
training.record_execution(
    agent_name="coding_agent",
    model_id="gemini-2.0-flash-exp",
    task_type="code_generation",
    success=True,
    latency_ms=1250.5,
    tokens_used=500,
    quality_score=0.92,  # Optional: from validation
    cost=0.0
)
```

### Getting Recommendations

```python
# Get recommended model for an agent and task
recommended_model = training.get_recommended_model(
    agent_name="coding_agent",
    task_type="code_generation",
    fallback="gemini-1.5-flash"
)

print(f"Recommended: {recommended_model}")
```

### Viewing Statistics

```python
# Get statistics for a specific agent
stats = training.get_agent_statistics("coding_agent")

print(f"Agent: {stats['agent_name']}")
print(f"Total executions: {stats['total_executions']}")
print(f"Success rate: {stats['success_rate']:.2%}")
print(f"Preferred models: {stats['preferred_models']}")
print(f"Task types: {stats['task_types']}")
```

### All Agents Summary

```python
# Get summary for all 40+ agents
all_agents = training.get_all_agents_summary()

for agent_stat in all_agents:
    print(f"\n{agent_stat['agent_name']}:")
    print(f"  Executions: {agent_stat['total_executions']}")
    print(f"  Success Rate: {agent_stat['success_rate']:.2%}")
    print(f"  Preferred Models:")
    for task, model in agent_stat['preferred_models'].items():
        print(f"    {task}: {model}")
```

### Training Data Persistence

Training data is automatically saved to `agent_training_data.json`:

```json
{
  "metrics": {
    "coding_agent:gemini-2.0-flash-exp:code_generation": {
      "success_count": 145,
      "failure_count": 5,
      "total_latency_ms": 182500.5,
      "total_tokens_used": 72500,
      "quality_scores": [0.92, 0.88, 0.95, ...],
      "last_updated": "2024-12-06T12:30:00"
    }
  },
  "preferences": {
    "coding_agent": {
      "code_generation": "gemini-2.0-flash-exp",
      "code_review": "gemini-1.5-pro"
    }
  }
}
```

---

## Task-Model Tracking

### Overview

The `TaskModelTracker` provides real-time visibility into:
- Which models are handling which tasks
- Active tasks currently running
- Historical task execution data
- Model usage patterns

### Tracking Tasks

```python
from gemini_advanced_config import TaskModelTracker

tracker = TaskModelTracker()

# Start tracking a task
tracker.start_task(
    task_id="task_12345",
    agent_name="coding_agent",
    task_type="code_generation",
    model_id="gemini-2.0-flash-exp",
    description="Create REST API endpoint for users"
)

# ... task executes ...

# Complete the task
tracker.complete_task(
    task_id="task_12345",
    success=True,
    result="API endpoint created successfully"
)
```

### Viewing Active Tasks

```python
# Get all currently running tasks
active_tasks = tracker.get_active_tasks()

for task in active_tasks:
    print(f"\nTask: {task['task_id']}")
    print(f"  Agent: {task['agent_name']}")
    print(f"  Model: {task['model_id']}")
    print(f"  Status: {task['status']}")
    print(f"  Running for: {(datetime.now() - task['start_time']).seconds}s")
```

### Model Usage Summary

```python
# Get summary of which models are being used
usage = tracker.get_model_usage_summary()

for model_id, stats in usage.items():
    print(f"\n{model_id}:")
    print(f"  Total uses: {stats['total_uses']}")
    print(f"  Active tasks: {stats['active_tasks']}")
    print(f"  Task breakdown:")
    for task_type, count in stats['task_breakdown'].items():
        print(f"    {task_type}: {count}")
```

### Recent Task History

```python
# Get last 50 completed tasks
recent = tracker.get_recent_tasks(limit=50)

for task in recent:
    status_icon = "✅" if task['success'] else "❌"
    print(f"{status_icon} {task['task_id']}: {task['model_id']} "
          f"({task['duration_ms']:.0f}ms)")
```

---

## Middleware Integration

### Complete Integration

The `GeminiMiddleware` class integrates all components:

```python
from gemini_advanced_config import GeminiMiddleware

# Initialize (automatically starts all components)
middleware = GeminiMiddleware()

# The middleware handles:
# 1. Key rotation
# 2. Model selection from training
# 3. Task tracking
# 4. Performance recording
```

### Execute Requests

```python
# Execute with full integration
result = await middleware.execute_request(
    agent_name="coding_agent",
    task_type="code_generation",
    task_description="Create FastAPI user authentication endpoint"
)

# Result includes:
{
    "success": True,
    "task_id": "task_1234567890",
    "model_used": "gemini-2.0-flash-exp",
    "api_key_used": "key_1",
    "latency_ms": 1250.5
}
```

### Dashboard Data

```python
# Get comprehensive dashboard
dashboard = middleware.get_dashboard_data()

print(f"Timestamp: {dashboard['timestamp']}")
print(f"\nAPI Keys Status:")
for key_id, status in dashboard['api_keys'].items():
    print(f"  {key_id}: {status['rpd_used']} requests today")

print(f"\nActive Tasks: {len(dashboard['active_tasks'])}")

print(f"\nModel Usage:")
for model, usage in dashboard['model_usage'].items():
    print(f"  {model}: {usage['total_uses']} uses")

print(f"\nAgent Statistics:")
for agent in dashboard['agent_statistics']:
    print(f"  {agent['agent_name']}: "
          f"{agent['total_executions']} executions, "
          f"{agent['success_rate']:.2%} success")
```

---

## Configuration Files

### Using YAML Configuration

Load configuration from `gemini_config.yaml`:

```python
import yaml

# Load configuration
with open('gemini_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Access model endpoints
models = config['models']
for model_id, model_config in models.items():
    print(f"{model_id}: {model_config['endpoint']}")

# Access agent configurations
agents = config['agents']
for agent_id, agent_config in agents.items():
    print(f"{agent_id}: {agent_config['display_name']}")
```

### Configuration Structure

The `gemini_config.yaml` file includes:

1. **API Keys**: Multiple keys from different organizations
2. **Model Endpoints**: Complete endpoint configuration for all models
3. **Agent Configuration**: 40+ agent definitions with model preferences
4. **Task-Model Mapping**: Default model assignments for task types
5. **Training Configuration**: Learning system parameters
6. **Monitoring Configuration**: Tracking and alerting settings

---

## Examples

### Example 1: Simple Request with Key Rotation

```python
from gemini_advanced_config import GeminiMiddleware

async def main():
    middleware = GeminiMiddleware()
    
    # Execute request - automatically uses best key and model
    result = await middleware.execute_request(
        agent_name="coding_agent",
        task_type="code_generation",
        task_description="Create a Python class for user management"
    )
    
    print(f"✅ Success: {result['success']}")
    print(f"📊 Model: {result['model_used']}")
    print(f"🔑 Key: {result['api_key_used']}")
    print(f"⏱️  Latency: {result['latency_ms']:.0f}ms")

asyncio.run(main())
```

### Example 2: Training Multiple Agents

```python
from gemini_advanced_config import AgentTrainingSystem

# Initialize training
training = AgentTrainingSystem()

# Simulate training for multiple agents
agents = ["coding_agent", "database_agent", "analysis_agent"]
models = ["gemini-2.0-flash-exp", "gemini-1.5-flash", "gemini-1.5-pro"]
tasks = ["generation", "review", "optimization"]

for agent in agents:
    for model in models:
        for task in tasks:
            # Record successful execution
            training.record_execution(
                agent_name=agent,
                model_id=model,
                task_type=task,
                success=True,
                latency_ms=random.uniform(500, 2000),
                tokens_used=random.randint(100, 1000),
                quality_score=random.uniform(0.7, 0.95)
            )

# Get recommendations
for agent in agents:
    for task in tasks:
        model = training.get_recommended_model(agent, task)
        print(f"{agent} + {task} → {model}")
```

### Example 3: Monitor All Components

```python
from gemini_advanced_config import GeminiMiddleware

async def monitor_system():
    middleware = GeminiMiddleware()
    
    # Execute some tasks
    for i in range(10):
        await middleware.execute_request(
            agent_name=f"agent_{i % 3}",
            task_type="generation",
            task_description=f"Task {i}"
        )
    
    # Get comprehensive status
    dashboard = middleware.get_dashboard_data()
    
    print("\n=== System Status ===")
    print(f"Time: {dashboard['timestamp']}")
    
    print("\n=== API Keys ===")
    for key_id, status in dashboard['api_keys'].items():
        print(f"{key_id}:")
        print(f"  Organization: {status['organization']}")
        print(f"  Used today: {status['rpd_used']}")
        print(f"  Active: {status['is_active']}")
    
    print("\n=== Active Tasks ===")
    print(f"Count: {len(dashboard['active_tasks'])}")
    
    print("\n=== Model Usage ===")
    for model, usage in dashboard['model_usage'].items():
        print(f"{model}: {usage['total_uses']} uses")
    
    print("\n=== Top Agents ===")
    for agent in dashboard['agent_statistics'][:5]:
        print(f"{agent['agent_name']}: {agent['total_executions']} executions")

asyncio.run(monitor_system())
```

---

## Advanced Topics

### Custom Agent Configuration

Add your own agents to the system:

```python
# In gemini_config.yaml, add:
agents:
  custom_ml_agent:
    display_name: "Machine Learning Agent"
    description: "ML model training and optimization"
    task_types:
      - "model_training"
      - "hyperparameter_tuning"
      - "model_evaluation"
    model_preferences:
      simple: "gemini-1.5-flash"
      moderate: "gemini-1.5-pro"
      complex: "gemini-1.5-pro"
    temperature_overrides:
      model_training: 0.5
```

### Custom Rotation Strategy

Implement your own rotation strategy:

```python
class CustomKeyRotation(APIKeyRotationManager):
    def custom_selection(self, keys, time_of_day):
        """Use different keys based on time of day"""
        hour = datetime.now().hour
        if hour < 12:
            return keys[0]  # Morning key
        elif hour < 18:
            return keys[1]  # Afternoon key
        else:
            return keys[2]  # Evening key
```

---

## Troubleshooting

### Issue: Keys Not Rotating

**Check:**
```python
status = key_manager.get_status()
print(f"Total keys loaded: {len(status)}")
for key_id, info in status.items():
    print(f"{key_id}: Active={info['is_active']}")
```

**Solution**: Ensure environment variables are set correctly

### Issue: Training Data Not Persisting

**Check:**
```python
import os
print(f"Training file exists: {os.path.exists('agent_training_data.json')}")
```

**Solution**: Ensure write permissions and call `training._save_training_data()`

### Issue: Tasks Not Being Tracked

**Check:**
```python
active = tracker.get_active_tasks()
print(f"Active tasks: {len(active)}")
```

**Solution**: Ensure `start_task()` is called before `complete_task()`

---

## Summary

This advanced setup provides:

✅ **Complete endpoint configuration** for all 4 Gemini models  
✅ **Automatic key rotation** across multiple Google organizations  
✅ **Intelligent agent training** for 40+ agents  
✅ **Real-time task tracking** and monitoring  
✅ **Integrated middleware** connecting all components  
✅ **Production-ready** with persistence and error handling  

**Files Created:**
- `gemini_advanced_config.py` - Complete implementation (830 lines)
- `gemini_config.yaml` - Configuration file (400+ lines)
- `GEMINI_ADVANCED_SETUP_GUIDE.md` - This guide

**Ready to use** - All components work together seamlessly!

---

**Last Updated**: December 6, 2024  
**Version**: 1.0

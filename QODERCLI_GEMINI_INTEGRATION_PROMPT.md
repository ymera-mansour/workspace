# QODERCLI: Gemini Optimization Integration Prompt
## Phase 3B Implementation

---

## YOUR IDENTITY
- **Name**: QODERCLI
- **Role**: Implementation and Integration Specialist
- **Phase**: 3B - Testing and Integration
- **Task**: Integrate Gemini optimization system into YMERA platform

---

## CONTEXT

You have access to a complete Gemini optimization system with:
1. **Advanced Configuration** (`gemini_advanced_config.py`) - 870 lines
2. **Basic Optimization** (`gemini_optimization_implementation.py`) - 572 lines
3. **Configuration Template** (`gemini_config.yaml`) - 430 lines
4. **Setup Guides** - Multiple comprehensive documentation files

The system provides:
- ✅ API key rotation for multiple Google organizations
- ✅ Agent training system for 40+ agents
- ✅ Task-model tracking and monitoring
- ✅ Intelligent model routing based on complexity
- ✅ Complete endpoint configuration for all 4 Gemini models

---

## YOUR MISSION

Integrate the Gemini optimization system into the existing YMERA multi-agent platform, ensuring:
1. All 40+ agents can use the optimized Gemini integration
2. API keys from multiple organizations rotate automatically
3. The system learns which models work best for each agent
4. Complete monitoring and tracking is operational

---

## IMPLEMENTATION STEPS

### STEP 1: Environment Setup (15 minutes)

**Task**: Configure environment variables for multiple API keys

**Action Required**:
```bash
# Create or update .env file in project root
cd /path/to/YmeraRefactor/

# Add the following variables:
cat >> .env << 'EOF'

# === GEMINI OPTIMIZATION CONFIGURATION ===

# Primary Gemini API key
GEMINI_API_KEY=your_primary_key_here

# Secondary keys from different Google organizations
GEMINI_API_KEY_1=your_org1_key_here
GEMINI_ORG_1=research_team

GEMINI_API_KEY_2=your_org2_key_here
GEMINI_ORG_2=development_team

GEMINI_API_KEY_3=your_org3_key_here
GEMINI_ORG_3=production_team

# Optional: Backup key
GEMINI_API_KEY_BACKUP=your_backup_key_here

# Caching configuration (optional)
REDIS_URL=redis://localhost:6379

# Google Cloud Storage for persistent cache (optional)
GOOGLE_CLOUD_PROJECT=your_project_id
GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json

EOF
```

**Verification**:
```python
import os
from dotenv import load_dotenv

load_dotenv()
assert os.getenv("GEMINI_API_KEY"), "Primary key not set"
print("✅ Environment configured")
```

---

### STEP 2: Install Dependencies (10 minutes)

**Task**: Install required Python packages

**Action Required**:
```bash
cd /path/to/YmeraRefactor/

# Install core dependencies
pip install google-generativeai>=0.3.0
pip install asyncio
pip install pyyaml
pip install python-dotenv

# Optional: For caching
pip install redis
pip install aioredis

# Optional: For monitoring
pip install prometheus-client
```

**Verification**:
```bash
python -c "import google.generativeai; print('✅ google-generativeai installed')"
python -c "import yaml; print('✅ pyyaml installed')"
```

---

### STEP 3: Copy Optimization Files (10 minutes)

**Task**: Copy the Gemini optimization files to your project

**Action Required**:
```bash
cd /path/to/YmeraRefactor/

# Create gemini optimization directory
mkdir -p gemini_optimization/

# Copy core files
cp /path/to/workspace/gemini_advanced_config.py gemini_optimization/
cp /path/to/workspace/gemini_optimization_implementation.py gemini_optimization/
cp /path/to/workspace/gemini_config.yaml gemini_optimization/

# Create __init__.py
cat > gemini_optimization/__init__.py << 'EOF'
"""
Gemini Optimization Module
Provides intelligent routing, key rotation, and agent training for Gemini models
"""

from .gemini_advanced_config import (
    GeminiMiddleware,
    APIKeyRotationManager,
    AgentTrainingSystem,
    TaskModelTracker,
    GEMINI_ENDPOINTS
)

from .gemini_optimization_implementation import (
    GeminiRouter,
    GeminiQuotaManager,
    GeminiCacheManager,
    ContextWindowOptimizer,
    GeminiUsageMonitor
)

__all__ = [
    'GeminiMiddleware',
    'APIKeyRotationManager',
    'AgentTrainingSystem',
    'TaskModelTracker',
    'GeminiRouter',
    'GeminiQuotaManager',
    'GeminiCacheManager',
    'ContextWindowOptimizer',
    'GeminiUsageMonitor',
    'GEMINI_ENDPOINTS'
]
EOF
```

**Directory Structure After**:
```
YmeraRefactor/
├── shared/
├── core_services/
├── agents/
├── gemini_optimization/  ← NEW
│   ├── __init__.py
│   ├── gemini_advanced_config.py
│   ├── gemini_optimization_implementation.py
│   └── gemini_config.yaml
└── tests/
```

---

### STEP 4: Integrate with Agent Manager (30 minutes)

**Task**: Update agent_manager to use Gemini middleware

**File**: `core_services/agent_manager/manager.py`

**Action Required**:
```python
# Add at the top of the file
from gemini_optimization import GeminiMiddleware, GeminiRouter
import asyncio

class AgentManager:
    def __init__(self):
        # ... existing initialization ...
        
        # Initialize Gemini optimization
        self.gemini_middleware = GeminiMiddleware()
        print("✅ Gemini optimization initialized")
    
    async def execute_agent_task(
        self,
        agent_name: str,
        task_type: str,
        task_description: str,
        use_gemini: bool = True
    ):
        """
        Execute an agent task with Gemini optimization
        
        Args:
            agent_name: Name of the agent (e.g., "coding_agent")
            task_type: Type of task (e.g., "code_generation")
            task_description: Description of the task
            use_gemini: Whether to use Gemini optimization
        """
        if use_gemini:
            # Use optimized Gemini system
            result = await self.gemini_middleware.execute_request(
                agent_name=agent_name,
                task_type=task_type,
                task_description=task_description
            )
            
            return {
                "success": result["success"],
                "model_used": result["model_used"],
                "api_key_used": result["api_key_used"],
                "latency_ms": result["latency_ms"],
                "task_id": result["task_id"]
            }
        else:
            # Use existing agent system
            # ... your existing code ...
            pass
    
    def get_gemini_dashboard(self):
        """Get Gemini optimization dashboard data"""
        return self.gemini_middleware.get_dashboard_data()
```

**Verification**:
```python
# Test the integration
async def test_integration():
    manager = AgentManager()
    
    result = await manager.execute_agent_task(
        agent_name="coding_agent",
        task_type="code_generation",
        task_description="Create a Python function to calculate fibonacci"
    )
    
    print(f"✅ Task executed")
    print(f"  Model: {result['model_used']}")
    print(f"  API Key: {result['api_key_used']}")
    print(f"  Latency: {result['latency_ms']}ms")

# Run test
asyncio.run(test_integration())
```

---

### STEP 5: Configure Agent Preferences (20 minutes)

**Task**: Update gemini_config.yaml with your actual agents

**File**: `gemini_optimization/gemini_config.yaml`

**Action Required**:

Replace the agent section with your actual 40+ agents. For each agent:

```yaml
agents:
  # Example: Your coding agent
  coding_agent:
    display_name: "Coding Agent"
    description: "Handles code generation and refactoring"
    task_types:
      - "code_generation"
      - "code_review"
      - "refactoring"
      - "bug_fixing"
    model_preferences:
      simple: "gemini-2.0-flash-exp"      # For simple tasks
      moderate: "gemini-1.5-flash"         # For standard tasks
      complex: "gemini-1.5-pro"            # For complex tasks
    temperature_overrides:
      code_generation: 0.3                 # Lower temp for code
      code_review: 0.4
  
  # Add all your other agents following the same pattern
  database_agent:
    display_name: "Database Agent"
    description: "SQL generation and database operations"
    task_types:
      - "sql_generation"
      - "query_optimization"
      - "schema_design"
    model_preferences:
      simple: "gemini-1.5-flash"
      moderate: "gemini-1.5-flash"
      complex: "gemini-1.5-pro"
    temperature_overrides:
      sql_generation: 0.1                  # Very low for SQL
  
  # ... Add all 40+ agents here ...
```

**Get Your Agent List**:
```bash
# Find all your agents
cd /path/to/YmeraRefactor/agents/
find . -name "*_agent.py" | sed 's/.*\///' | sed 's/_agent.py//'
```

---

### STEP 6: Add Model Selection Logic to Each Agent (40 minutes)

**Task**: Update each agent to use intelligent model selection

**For Each Agent File** (e.g., `agents/coding/coding_agent.py`):

```python
# Add at the top
from gemini_optimization import GeminiRouter, GeminiQuotaManager

class CodingAgent:
    def __init__(self):
        # ... existing initialization ...
        
        # Initialize Gemini components
        self.quota_manager = GeminiQuotaManager()
        self.router = GeminiRouter(self.quota_manager)
    
    async def execute_task(self, task_description: str, complexity: str = None):
        """
        Execute a coding task with optimal model selection
        
        Args:
            task_description: Description of the coding task
            complexity: Optional complexity override ("simple", "moderate", "complex")
        """
        # Get recommended model
        if complexity:
            # Use specified complexity
            model = self.router.ROUTING_RULES["coding_agent"][complexity]
        else:
            # Auto-detect complexity
            model = await self.router.route_request(
                agent_type="coding_agent",
                task_description=task_description
            )
        
        # Check quota
        can_proceed = await self.quota_manager.can_make_request(model)
        if not can_proceed:
            # Get fallback model
            model = await self.router._get_fallback("coding_agent", model)
            print(f"⚠️  Switched to fallback model: {model}")
        
        # Execute with selected model
        print(f"✅ Using model: {model}")
        
        # Here you would make the actual Gemini API call
        # ... your existing code ...
        
        # Record the request
        await self.quota_manager.record_request(model)
        
        return result
```

**Apply to All Agents**:
```bash
# Script to update all agents (customize as needed)
for agent_file in agents/*/*.py; do
    if [[ $agent_file == *"_agent.py" ]]; then
        echo "Updating $agent_file"
        # Add imports and integration code
    fi
done
```

---

### STEP 7: Implement Training Loop (30 minutes)

**Task**: Set up automatic learning from agent executions

**File**: Create `gemini_optimization/training_loop.py`

```python
"""
Training loop for Gemini agent optimization
Records all agent executions for continuous learning
"""

from gemini_optimization import AgentTrainingSystem
import asyncio
import logging

logger = logging.getLogger(__name__)

class GeminiTrainingLoop:
    """Automatic training system for all agents"""
    
    def __init__(self):
        self.training = AgentTrainingSystem()
        self.execution_count = 0
    
    async def record_execution(
        self,
        agent_name: str,
        model_id: str,
        task_type: str,
        success: bool,
        start_time: float,
        end_time: float,
        tokens_used: int,
        quality_score: float = None
    ):
        """Record an agent execution for training"""
        latency_ms = (end_time - start_time) * 1000
        
        self.training.record_execution(
            agent_name=agent_name,
            model_id=model_id,
            task_type=task_type,
            success=success,
            latency_ms=latency_ms,
            tokens_used=tokens_used,
            quality_score=quality_score or 0.8,  # Default score
            cost=0.0  # Free tier
        )
        
        self.execution_count += 1
        
        if self.execution_count % 10 == 0:
            logger.info(f"✅ Recorded {self.execution_count} executions")
    
    def get_recommendation(self, agent_name: str, task_type: str) -> str:
        """Get recommended model for an agent"""
        return self.training.get_recommended_model(
            agent_name=agent_name,
            task_type=task_type,
            fallback="gemini-1.5-flash"
        )
    
    def get_statistics(self, agent_name: str = None):
        """Get training statistics"""
        if agent_name:
            return self.training.get_agent_statistics(agent_name)
        else:
            return self.training.get_all_agents_summary()

# Global training loop instance
training_loop = GeminiTrainingLoop()
```

**Integrate with Agent Manager**:
```python
# In core_services/agent_manager/manager.py
from gemini_optimization.training_loop import training_loop
import time

class AgentManager:
    async def execute_agent_task(self, agent_name, task_type, task_description):
        start_time = time.time()
        
        # Execute task
        result = await self.gemini_middleware.execute_request(...)
        
        end_time = time.time()
        
        # Record for training
        await training_loop.record_execution(
            agent_name=agent_name,
            model_id=result["model_used"],
            task_type=task_type,
            success=result["success"],
            start_time=start_time,
            end_time=end_time,
            tokens_used=result.get("tokens_used", 100)
        )
        
        return result
```

---

### STEP 8: Add Monitoring Dashboard (20 minutes)

**Task**: Create monitoring endpoints for the optimization system

**File**: Create `gemini_optimization/monitoring.py`

```python
"""
Monitoring and dashboard for Gemini optimization
"""

from gemini_optimization import GeminiMiddleware
from gemini_optimization.training_loop import training_loop
import json

class GeminiMonitor:
    """Monitoring dashboard for Gemini optimization"""
    
    def __init__(self, middleware: GeminiMiddleware):
        self.middleware = middleware
    
    def get_full_status(self) -> dict:
        """Get complete system status"""
        dashboard = self.middleware.get_dashboard_data()
        
        # Add training statistics
        dashboard["training_stats"] = training_loop.get_statistics()
        
        return dashboard
    
    def print_status(self):
        """Print human-readable status"""
        status = self.get_full_status()
        
        print("\n" + "="*60)
        print("GEMINI OPTIMIZATION STATUS")
        print("="*60)
        
        print("\n📊 API KEYS:")
        for key_id, info in status["api_keys"].items():
            active = "✅" if info["is_active"] else "❌"
            print(f"  {active} {key_id} ({info['organization']})")
            print(f"     Used today: {info['rpd_used']}")
        
        print("\n🚀 ACTIVE TASKS:")
        print(f"  Currently running: {len(status['active_tasks'])}")
        
        print("\n📈 MODEL USAGE:")
        for model, stats in status["model_usage"].items():
            print(f"  {model}: {stats['total_uses']} uses")
        
        print("\n🎓 TOP AGENTS:")
        for agent in status["training_stats"][:5]:
            print(f"  {agent['agent_name']}:")
            print(f"    Executions: {agent['total_executions']}")
            print(f"    Success: {agent['success_rate']:.1%}")
        
        print("\n" + "="*60)
    
    def export_metrics(self, filename: str = "gemini_metrics.json"):
        """Export metrics to file"""
        status = self.get_full_status()
        with open(filename, 'w') as f:
            json.dump(status, f, indent=2)
        print(f"✅ Metrics exported to {filename}")

# Usage in your API or CLI
def show_dashboard():
    from core_services.agent_manager import manager
    monitor = GeminiMonitor(manager.gemini_middleware)
    monitor.print_status()
```

---

### STEP 9: Testing Integration (30 minutes)

**Task**: Create tests for the Gemini integration

**File**: Create `tests/integration/test_gemini_optimization.py`

```python
"""
Integration tests for Gemini optimization system
Phase: 3B | Agent: qoder
"""

import pytest
import asyncio
from gemini_optimization import GeminiMiddleware, APIKeyRotationManager
from gemini_optimization.training_loop import training_loop

@pytest.mark.integration
@pytest.mark.asyncio
async def test_gemini_middleware_initialization():
    """Test that Gemini middleware initializes correctly"""
    middleware = GeminiMiddleware()
    
    # Check components are initialized
    assert middleware.key_manager is not None
    assert middleware.training_system is not None
    assert middleware.task_tracker is not None
    
    print("✅ Middleware initialized")

@pytest.mark.integration
@pytest.mark.asyncio
async def test_api_key_rotation():
    """Test that API key rotation works"""
    key_manager = APIKeyRotationManager()
    
    # Should have at least primary key
    assert len(key_manager.keys) >= 1
    
    # Test getting next key
    key_info = key_manager.get_next_key("gemini-2.0-flash-exp")
    assert key_info is not None
    assert key_info.api_key != ""
    
    print(f"✅ Key rotation works, {len(key_manager.keys)} keys loaded")

@pytest.mark.integration
@pytest.mark.asyncio
async def test_agent_execution_with_training():
    """Test that agent execution records training data"""
    middleware = GeminiMiddleware()
    
    # Execute a task
    result = await middleware.execute_request(
        agent_name="test_agent",
        task_type="test_task",
        task_description="Test task description"
    )
    
    # Check result
    assert "model_used" in result
    assert "task_id" in result
    
    # Check training was recorded
    stats = training_loop.get_statistics("test_agent")
    assert stats is not None
    
    print("✅ Training loop working")

@pytest.mark.integration
def test_model_selection_routing():
    """Test that model routing works correctly"""
    from gemini_optimization import GeminiRouter, GeminiQuotaManager
    
    quota = GeminiQuotaManager()
    router = GeminiRouter(quota)
    
    # Test simple task
    complexity = router.detect_complexity("Write a simple hello world function")
    assert complexity == "simple"
    
    # Test complex task
    complexity = router.detect_complexity(
        "Design a comprehensive microservices architecture with event sourcing"
    )
    assert complexity == "complex"
    
    print("✅ Model routing works")

@pytest.mark.integration
def test_quota_tracking():
    """Test that quota tracking works"""
    from gemini_optimization import GeminiQuotaManager
    
    quota = GeminiQuotaManager()
    
    # Record some requests
    asyncio.run(quota.record_request("gemini-2.0-flash-exp"))
    asyncio.run(quota.record_request("gemini-2.0-flash-exp"))
    
    # Check status
    status = quota.get_quota_status()
    assert "gemini-2.0-flash-exp" in status
    assert status["gemini-2.0-flash-exp"]["used_today"] >= 2
    
    print("✅ Quota tracking works")

@pytest.mark.integration
def test_configuration_loading():
    """Test that configuration loads correctly"""
    import yaml
    
    with open("gemini_optimization/gemini_config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # Check models
    assert "models" in config
    assert "gemini-2.0-flash-exp" in config["models"]
    
    # Check agents
    assert "agents" in config
    assert len(config["agents"]) > 0
    
    print(f"✅ Configuration loaded: {len(config['agents'])} agents")

# Run all tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
```

**Run Tests**:
```bash
cd /path/to/YmeraRefactor/
pytest tests/integration/test_gemini_optimization.py -v
```

---

### STEP 10: Documentation Update (15 minutes)

**Task**: Update project documentation

**File**: Update `README.md`

```markdown
# YMERA Multi-Agent Platform

## Gemini Optimization Integration ✨ NEW

The platform now includes comprehensive Gemini optimization:

### Features
- ✅ **Multi-Organization Key Rotation**: Automatically rotate between API keys from different Google organizations
- ✅ **Agent Training System**: Learns which models work best for each of 40+ agents
- ✅ **Task Tracking**: Real-time visibility into which model handles which task
- ✅ **Intelligent Routing**: Automatically selects optimal model based on task complexity
- ✅ **Complete Monitoring**: Dashboard showing all system status

### Quick Start

1. **Configure API Keys**:
   ```bash
   # Add to .env
   GEMINI_API_KEY=your_key
   GEMINI_API_KEY_1=org1_key
   GEMINI_ORG_1=research_team
   ```

2. **Use in Your Agents**:
   ```python
   from gemini_optimization import GeminiMiddleware
   
   middleware = GeminiMiddleware()
   result = await middleware.execute_request(
       agent_name="coding_agent",
       task_type="code_generation",
       task_description="Create a REST API"
   )
   ```

3. **Monitor Status**:
   ```python
   from gemini_optimization.monitoring import show_dashboard
   show_dashboard()
   ```

### Documentation
- Full setup: `gemini_optimization/GEMINI_ADVANCED_SETUP_GUIDE.md`
- Quick reference: `gemini_optimization/QUICK_REFERENCE.md`
```

---

## VERIFICATION CHECKLIST

After completing all steps, verify:

- [ ] Environment variables are set (Step 1)
- [ ] All dependencies installed (Step 2)
- [ ] Files copied to project (Step 3)
- [ ] Agent manager integrated (Step 4)
- [ ] Agent preferences configured (Step 5)
- [ ] All 40+ agents updated (Step 6)
- [ ] Training loop active (Step 7)
- [ ] Monitoring dashboard works (Step 8)
- [ ] All tests pass (Step 9)
- [ ] Documentation updated (Step 10)

**Final Verification Command**:
```bash
cd /path/to/YmeraRefactor/

# Run all tests
pytest tests/integration/test_gemini_optimization.py -v

# Show dashboard
python -c "from gemini_optimization.monitoring import show_dashboard; show_dashboard()"

# Check training data
python -c "from gemini_optimization.training_loop import training_loop; print(training_loop.get_statistics())"
```

---

## EXPECTED RESULTS

After successful integration:

✅ **API Key Rotation Working**:
- Multiple keys from different organizations active
- Automatic rotation when limits approached
- No quota exhaustion errors

✅ **Agent Training Active**:
- System learns from every execution
- Optimal models recommended for each agent
- Training data persists to disk

✅ **Task Tracking Operational**:
- Real-time visibility into active tasks
- Historical execution data available
- Model usage statistics updated

✅ **Performance Improved**:
- 60-75% fewer API calls (via caching)
- 50-66% faster responses (optimal model selection)
- 100% free tier compliance

---

## TROUBLESHOOTING

### Issue: API Keys Not Loading
```python
# Check environment
import os
print(os.getenv("GEMINI_API_KEY"))

# Check key manager
from gemini_optimization import APIKeyRotationManager
mgr = APIKeyRotationManager()
print(f"Loaded {len(mgr.keys)} keys")
```

### Issue: Tests Failing
```bash
# Run with verbose output
pytest tests/integration/test_gemini_optimization.py -v -s

# Check specific test
pytest tests/integration/test_gemini_optimization.py::test_gemini_middleware_initialization -v
```

### Issue: No Training Data
```python
# Force save training data
from gemini_optimization.training_loop import training_loop
training_loop.training._save_training_data()

# Check if file exists
import os
print(os.path.exists("agent_training_data.json"))
```

---

## SUCCESS CRITERIA

Your integration is complete when:

1. ✅ All tests pass
2. ✅ Dashboard shows active keys and models
3. ✅ Agent executions are being tracked
4. ✅ Training data is accumulating
5. ✅ No quota errors in production
6. ✅ Response times improved

---

## NEXT STEPS

After integration:
1. Monitor for 48 hours
2. Review training recommendations
3. Optimize agent configurations based on learned data
4. Scale to all 40+ agents
5. Set up automated monitoring alerts

---

**End of QoderCLI Integration Prompt**

Status: Ready for Phase 3B Implementation
Last Updated: December 6, 2024

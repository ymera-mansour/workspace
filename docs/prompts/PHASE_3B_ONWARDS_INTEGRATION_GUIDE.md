# YMERA Platform - Integration Guide (Phase 3B Onwards)

## 🎯 Starting Point

**What's Already Complete** (Phase 1-3A):
- ✅ Foundation & shared library (config, database, utils)
- ✅ Core services (agent_manager, engines, ai_mcp)
- ✅ **40+ Agents already implemented** in agents/ directory
- ✅ Base agent classes and interfaces
- ✅ Test framework (24 tests)
- ✅ AI provider integrations (Mistral, Gemini, Groq, HuggingFace)

**What We're Building Now** (Phase 3B-7):
- 🔄 Integrate existing 40+ agents with models, MCPs, and tools
- 🔄 Advanced workflow orchestration for multi-agent coordination
- 🔄 AI/ML learning and training systems
- 🔄 Security, authentication, and authorization
- 🔄 Frontend integration
- 🔄 Production deployment

---

## 📋 Phase 3B: Advanced Multi-Agent Workflow Integration

### Goal
Connect existing 40+ agents to models, MCPs, and tools with intelligent workflow orchestration.

### QoderCLI Command
```bash
qoder implement "Integrate existing 40+ agents with AI models (Groq, Gemini, OpenRouter), MCP tools, and create advanced workflow orchestration for multi-agent collaboration"
```

### Detailed Implementation

#### Step 1: Create Agent-Model Integration Layer

**File**: `src/core/agent_model_connector.py`

```python
"""
Agent-Model Integration Layer
Connects existing agents to AI models dynamically
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import asyncio

@dataclass
class AgentCapabilityProfile:
    """Profile of agent's capabilities and model requirements"""
    agent_name: str
    agent_type: str  # coding, analysis, research, documentation, database, etc.
    capabilities: List[str]
    preferred_models: List[str]  # Ordered by preference
    fallback_models: List[str]
    required_tools: List[str]  # MCP tools needed
    context_length_needed: int
    avg_tokens_per_task: int
    specializations: List[str]  # code, reasoning, creative, multimodal

class AgentModelConnector:
    """
    Dynamically connects agents to appropriate models
    
    For existing 40+ agents:
    - Analyzes agent capabilities
    - Selects optimal models
    - Manages MCP tool access
    - Handles fallback chains
    """
    
    def __init__(
        self,
        model_selector,  # From Phase 3
        provider_factory,  # From Phase 2
        mcp_registry,  # MCP tool registry
        agent_registry  # Existing agent registry
    ):
        self.model_selector = model_selector
        self.provider_factory = provider_factory
        self.mcp_registry = mcp_registry
        self.agent_registry = agent_registry
        self.agent_profiles = self._build_agent_profiles()
    
    def _build_agent_profiles(self) -> Dict[str, AgentCapabilityProfile]:
        """
        Build capability profiles for all 40+ existing agents
        Analyze each agent to determine best models and tools
        """
        
        profiles = {}
        
        # Get all registered agents
        agents = self.agent_registry.get_all_agents()
        
        for agent in agents:
            profile = self._analyze_agent_capabilities(agent)
            profiles[agent.name] = profile
        
        return profiles
    
    def _analyze_agent_capabilities(self, agent) -> AgentCapabilityProfile:
        """Analyze single agent to build capability profile"""
        
        # Detect agent type from class name or metadata
        agent_type = self._detect_agent_type(agent)
        
        # Map agent type to optimal models
        model_recommendations = self._recommend_models_for_type(agent_type)
        
        # Determine required MCP tools
        required_tools = self._determine_required_tools(agent, agent_type)
        
        # Estimate token requirements
        token_estimates = self._estimate_token_requirements(agent_type)
        
        return AgentCapabilityProfile(
            agent_name=agent.name,
            agent_type=agent_type,
            capabilities=self._extract_capabilities(agent),
            preferred_models=model_recommendations["preferred"],
            fallback_models=model_recommendations["fallback"],
            required_tools=required_tools,
            context_length_needed=token_estimates["context"],
            avg_tokens_per_task=token_estimates["avg"],
            specializations=self._detect_specializations(agent, agent_type)
        )
    
    def _detect_agent_type(self, agent) -> str:
        """Detect agent type from name or class"""
        
        agent_name = agent.name.lower()
        
        if "cod" in agent_name or "dev" in agent_name:
            return "coding"
        elif "analy" in agent_name or "data" in agent_name:
            return "analysis"
        elif "research" in agent_name or "search" in agent_name:
            return "research"
        elif "doc" in agent_name or "writer" in agent_name:
            return "documentation"
        elif "database" in agent_name or "sql" in agent_name:
            return "database"
        elif "web" in agent_name or "scrape" in agent_name:
            return "web_scraping"
        elif "test" in agent_name or "qa" in agent_name:
            return "testing"
        elif "secur" in agent_name or "audit" in agent_name:
            return "security"
        elif "deploy" in agent_name or "devops" in agent_name:
            return "devops"
        elif "monitor" in agent_name or "observ" in agent_name:
            return "monitoring"
        else:
            return "general"
    
    def _recommend_models_for_type(self, agent_type: str) -> Dict[str, List[str]]:
        """
        Recommend models based on agent type
        Uses FREE models as much as possible
        """
        
        recommendations = {
            "coding": {
                "preferred": [
                    "openrouter/deepseek/deepseek-coder-6.7b-instruct:free",  # FREE code specialist
                    "groq/llama-3.1-70b-versatile",  # FREE, very capable
                    "mistral/codestral",  # PAID, best for code
                ],
                "fallback": [
                    "groq/llama-3.1-8b-instant",  # FREE, ultra fast
                    "openrouter/meta-llama/llama-3.1-8b-instruct:free",
                ]
            },
            "analysis": {
                "preferred": [
                    "groq/llama-3.3-70b-versatile",  # FREE, excellent reasoning
                    "gemini/gemini-1.5-flash",  # FREE tier (15 RPM), fast
                    "anthropic/claude-3.5-sonnet",  # PAID, best quality
                ],
                "fallback": [
                    "groq/mixtral-8x7b-32768",  # FREE
                    "openrouter/mistralai/mistral-7b-instruct:free",
                ]
            },
            "research": {
                "preferred": [
                    "gemini/gemini-1.5-pro",  # FREE tier (2 RPM), 2M context
                    "groq/llama-3.3-70b-versatile",  # FREE
                    "openrouter/nousresearch/hermes-3-llama-3.1-405b:free",  # FREE
                ],
                "fallback": [
                    "gemini/gemini-1.5-flash",
                    "groq/llama-3.1-70b-versatile",
                ]
            },
            "documentation": {
                "preferred": [
                    "groq/llama-3.1-70b-versatile",  # FREE, good at writing
                    "gemini/gemini-1.5-flash",  # FREE tier, fast
                    "anthropic/claude-3.5-sonnet",  # PAID, excellent writer
                ],
                "fallback": [
                    "openrouter/meta-llama/llama-3.1-8b-instruct:free",
                    "groq/llama-3.1-8b-instant",
                ]
            },
            "database": {
                "preferred": [
                    "groq/llama-3.1-70b-versatile",  # FREE, good at SQL
                    "openrouter/meta-llama/llama-3.1-8b-instruct:free",
                    "mistral/mistral-small-latest",  # PAID, efficient
                ],
                "fallback": [
                    "groq/llama-3.1-8b-instant",
                    "gemini/gemini-1.5-flash",
                ]
            },
            "web_scraping": {
                "preferred": [
                    "groq/llama-3.1-8b-instant",  # FREE, fast for extraction
                    "gemini/gemini-1.5-flash",  # FREE tier, multimodal
                ],
                "fallback": [
                    "openrouter/meta-llama/llama-3.1-8b-instruct:free",
                ]
            },
            "testing": {
                "preferred": [
                    "groq/llama-3.1-70b-versatile",  # FREE, good at test generation
                    "openrouter/deepseek/deepseek-coder-6.7b-instruct:free",
                ],
                "fallback": [
                    "groq/llama-3.1-8b-instant",
                ]
            },
            "security": {
                "preferred": [
                    "groq/llama-3.3-70b-versatile",  # FREE, good reasoning
                    "anthropic/claude-3.5-sonnet",  # PAID, excellent at security
                ],
                "fallback": [
                    "groq/llama-3.1-70b-versatile",
                ]
            },
            "general": {
                "preferred": [
                    "groq/llama-3.1-70b-versatile",  # FREE, versatile
                    "gemini/gemini-1.5-flash",  # FREE tier
                ],
                "fallback": [
                    "groq/llama-3.1-8b-instant",
                    "openrouter/meta-llama/llama-3.1-8b-instruct:free",
                ]
            }
        }
        
        return recommendations.get(agent_type, recommendations["general"])
    
    def _determine_required_tools(self, agent, agent_type: str) -> List[str]:
        """Determine which MCP tools this agent needs"""
        
        tool_mapping = {
            "coding": ["github", "filesystem"],
            "analysis": ["database", "filesystem"],
            "research": ["brave_search", "web", "sequential_thinking"],
            "documentation": ["github", "filesystem"],
            "database": ["postgres", "filesystem"],
            "web_scraping": ["puppeteer", "brave_search"],
            "testing": ["github", "filesystem"],
            "security": ["github", "filesystem", "audit"],
            "devops": ["github", "filesystem", "docker"],
            "monitoring": ["prometheus", "time"],
            "general": ["filesystem"]
        }
        
        return tool_mapping.get(agent_type, ["filesystem"])
    
    async def connect_agent_to_models(
        self,
        agent_name: str,
        task: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Connect agent to appropriate models for task
        
        Returns connection info with models and tools
        """
        
        profile = self.agent_profiles.get(agent_name)
        if not profile:
            raise ValueError(f"Unknown agent: {agent_name}")
        
        # Select best model for this specific task
        model = await self._select_model_for_task(profile, task)
        
        # Get MCP tools
        tools = await self._provision_tools(profile.required_tools)
        
        # Create provider instance
        provider = self.provider_factory.create_provider(
            provider_name=model.provider,
            api_key=self._get_api_key(model.provider)
        )
        
        return {
            "agent": agent_name,
            "model": model,
            "provider": provider,
            "tools": tools,
            "profile": profile
        }
    
    async def _select_model_for_task(
        self,
        profile: AgentCapabilityProfile,
        task: Dict[str, Any]
    ) -> Any:
        """Select best model from agent's preferred list"""
        
        # Try preferred models first
        for model_id in profile.preferred_models:
            try:
                model = self.model_selector.registry.get_model(model_id)
                
                # Check if model can handle task
                if self._can_model_handle_task(model, task, profile):
                    return model
                    
            except Exception:
                continue
        
        # Fall back to fallback models
        for model_id in profile.fallback_models:
            try:
                model = self.model_selector.registry.get_model(model_id)
                if self._can_model_handle_task(model, task, profile):
                    return model
            except Exception:
                continue
        
        # Last resort: default model
        return self.model_selector._get_default_model()
    
    async def _provision_tools(self, tool_names: List[str]) -> Dict[str, Any]:
        """Provision MCP tools for agent"""
        
        tools = {}
        
        for tool_name in tool_names:
            try:
                tool = await self.mcp_registry.get_tool(tool_name)
                tools[tool_name] = tool
            except Exception as e:
                print(f"Warning: Could not provision tool {tool_name}: {e}")
        
        return tools
```

#### Step 2: Create Advanced Multi-Agent Workflow Orchestrator

**File**: `src/core/multi_agent_orchestrator.py`

```python
"""
Advanced Multi-Agent Workflow Orchestrator
Coordinates 40+ agents with intelligent task routing
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
import asyncio

class TaskPriority(Enum):
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4

class AgentCommunicationType(Enum):
    SEQUENTIAL = "sequential"  # One after another
    PARALLEL = "parallel"  # All at once
    HIERARCHICAL = "hierarchical"  # Lead agent coordinates
    COLLABORATIVE = "collaborative"  # Agents negotiate
    COMPETITIVE = "competitive"  # Best result wins

@dataclass
class MultiAgentTask:
    """Task that requires multiple agents"""
    task_id: str
    description: str
    required_agents: List[str]  # Agents that must participate
    optional_agents: List[str]  # Agents that could help
    coordination_type: AgentCommunicationType
    priority: TaskPriority
    max_duration_seconds: int
    dependencies: Dict[str, List[str]]  # Agent dependencies
    shared_context: Dict[str, Any]

@dataclass
class AgentExecution:
    """Result of single agent execution"""
    agent_name: str
    success: bool
    result: Any
    execution_time: float
    model_used: str
    tools_used: List[str]
    cost: float
    quality_score: float
    error: Optional[str] = None

class MultiAgentOrchestrator:
    """
    Orchestrate complex workflows across 40+ agents
    
    Features:
    - Intelligent agent selection
    - Task decomposition
    - Parallel execution
    - Result aggregation
    - Conflict resolution
    - Resource management
    """
    
    def __init__(
        self,
        agent_registry,  # All 40+ agents
        agent_model_connector,  # Connect agents to models
        workflow_engine,  # Phase 4 workflow engine
        learning_engine  # Phase 7 learning system
    ):
        self.agent_registry = agent_registry
        self.connector = agent_model_connector
        self.workflow_engine = workflow_engine
        self.learning_engine = learning_engine
        self.execution_history = []
    
    async def execute_multi_agent_task(
        self,
        task: MultiAgentTask
    ) -> Dict[str, Any]:
        """
        Execute task across multiple agents
        
        Handles all coordination types:
        - Sequential: Agent A → Agent B → Agent C
        - Parallel: Agent A, B, C simultaneously
        - Hierarchical: Lead agent orchestrates others
        - Collaborative: Agents communicate and coordinate
        - Competitive: Multiple agents try, best wins
        """
        
        # Step 1: Validate and prepare agents
        agents = await self._prepare_agents(task)
        
        # Step 2: Execute based on coordination type
        if task.coordination_type == AgentCommunicationType.SEQUENTIAL:
            results = await self._execute_sequential(task, agents)
        elif task.coordination_type == AgentCommunicationType.PARALLEL:
            results = await self._execute_parallel(task, agents)
        elif task.coordination_type == AgentCommunicationType.HIERARCHICAL:
            results = await self._execute_hierarchical(task, agents)
        elif task.coordination_type == AgentCommunicationType.COLLABORATIVE:
            results = await self._execute_collaborative(task, agents)
        elif task.coordination_type == AgentCommunicationType.COMPETITIVE:
            results = await self._execute_competitive(task, agents)
        
        # Step 3: Aggregate results
        final_result = await self._aggregate_results(results, task)
        
        # Step 4: Learn from execution
        await self._record_for_learning(task, results, final_result)
        
        return final_result
    
    async def _execute_sequential(
        self,
        task: MultiAgentTask,
        agents: List[Any]
    ) -> List[AgentExecution]:
        """Execute agents one after another"""
        
        results = []
        context = task.shared_context.copy()
        
        # Respect dependencies
        execution_order = self._resolve_dependencies(
            agents,
            task.dependencies
        )
        
        for agent in execution_order:
            try:
                # Connect agent to models and tools
                connection = await self.connector.connect_agent_to_models(
                    agent_name=agent.name,
                    task={"description": task.description, "context": context}
                )
                
                # Execute agent
                result = await self._execute_single_agent(
                    agent=agent,
                    connection=connection,
                    context=context
                )
                
                results.append(result)
                
                # Update context for next agent
                context[f"{agent.name}_result"] = result.result
                
            except Exception as e:
                results.append(AgentExecution(
                    agent_name=agent.name,
                    success=False,
                    result=None,
                    execution_time=0,
                    model_used="none",
                    tools_used=[],
                    cost=0,
                    quality_score=0,
                    error=str(e)
                ))
        
        return results
    
    async def _execute_parallel(
        self,
        task: MultiAgentTask,
        agents: List[Any]
    ) -> List[AgentExecution]:
        """Execute all agents simultaneously"""
        
        # Create tasks for all agents
        agent_tasks = []
        
        for agent in agents:
            agent_task = self._create_agent_task(agent, task)
            agent_tasks.append(agent_task)
        
        # Execute all in parallel
        results = await asyncio.gather(*agent_tasks, return_exceptions=True)
        
        # Process results
        processed = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed.append(AgentExecution(
                    agent_name=agents[i].name,
                    success=False,
                    result=None,
                    execution_time=0,
                    model_used="none",
                    tools_used=[],
                    cost=0,
                    quality_score=0,
                    error=str(result)
                ))
            else:
                processed.append(result)
        
        return processed
    
    async def _execute_hierarchical(
        self,
        task: MultiAgentTask,
        agents: List[Any]
    ) -> List[AgentExecution]:
        """Lead agent orchestrates others"""
        
        # Select lead agent (first in required list or most capable)
        lead_agent = await self._select_lead_agent(agents, task)
        worker_agents = [a for a in agents if a != lead_agent]
        
        # Lead agent creates subtasks
        subtasks = await self._lead_agent_plan(lead_agent, task, worker_agents)
        
        # Execute subtasks with workers
        worker_results = []
        for subtask in subtasks:
            assigned_agent = subtask["agent"]
            result = await self._execute_single_agent_task(assigned_agent, subtask)
            worker_results.append(result)
        
        # Lead agent aggregates
        final = await self._lead_agent_aggregate(lead_agent, worker_results, task)
        
        return [final] + worker_results
    
    async def _execute_collaborative(
        self,
        task: MultiAgentTask,
        agents: List[Any]
    ) -> List[AgentExecution]:
        """Agents communicate and coordinate"""
        
        results = []
        shared_workspace = task.shared_context.copy()
        
        # Round-robin collaboration
        max_rounds = 3
        for round_num in range(max_rounds):
            round_results = []
            
            for agent in agents:
                # Agent sees all previous work
                result = await self._execute_collaborative_round(
                    agent=agent,
                    shared_workspace=shared_workspace,
                    round_num=round_num,
                    task=task
                )
                
                round_results.append(result)
                
                # Update shared workspace
                shared_workspace[f"{agent.name}_round_{round_num}"] = result.result
            
            results.extend(round_results)
            
            # Check if consensus reached
            if await self._check_consensus(round_results):
                break
        
        return results
    
    async def _execute_competitive(
        self,
        task: MultiAgentTask,
        agents: List[Any]
    ) -> List[AgentExecution]:
        """Multiple agents try, best result wins"""
        
        # Execute all agents in parallel
        results = await self._execute_parallel(task, agents)
        
        # Evaluate and rank results
        ranked = await self._rank_agent_results(results, task)
        
        # Mark winner
        if ranked:
            ranked[0].result["is_winner"] = True
        
        return ranked
    
    async def _aggregate_results(
        self,
        results: List[AgentExecution],
        task: MultiAgentTask
    ) -> Dict[str, Any]:
        """Aggregate results from multiple agents"""
        
        successful = [r for r in results if r.success]
        
        if task.coordination_type == AgentCommunicationType.COMPETITIVE:
            # Return best result
            if successful:
                best = max(successful, key=lambda r: r.quality_score)
                return {
                    "type": "competitive",
                    "winner": best.agent_name,
                    "result": best.result,
                    "all_attempts": len(results),
                    "successful_attempts": len(successful)
                }
        
        # For other types, aggregate all results
        return {
            "type": task.coordination_type.value,
            "results": {r.agent_name: r.result for r in successful},
            "execution_summary": {
                "total_agents": len(results),
                "successful": len(successful),
                "total_cost": sum(r.cost for r in results),
                "total_time": sum(r.execution_time for r in results),
                "avg_quality": sum(r.quality_score for r in successful) / len(successful) if successful else 0
            }
        }
```

#### Step 3: Integration with Existing Agents

**File**: `src/core/existing_agent_wrapper.py`

```python
"""
Wrapper to integrate existing 40+ agents seamlessly
"""

class ExistingAgentWrapper:
    """
    Wrap existing agents to work with new systems:
    - Model integration
    - MCP tool access
    - Workflow orchestration
    - Learning system
    """
    
    def __init__(self, existing_agent, connector, learning_engine):
        self.agent = existing_agent
        self.connector = connector
        self.learning = learning_engine
        self.execution_count = 0
    
    async def execute_with_integration(
        self,
        task: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute existing agent with full integration"""
        
        # Step 1: Connect to optimal model
        connection = await self.connector.connect_agent_to_models(
            agent_name=self.agent.name,
            task=task
        )
        
        # Step 2: Provide MCP tools
        self.agent.tools = connection["tools"]
        
        # Step 3: Execute agent's original logic
        result = await self.agent.execute(task)
        
        # Step 4: Track for learning
        await self.learning.record_agent_execution(
            agent_name=self.agent.name,
            task=task,
            result=result,
            model_used=connection["model"].model_id,
            tools_used=list(connection["tools"].keys())
        )
        
        self.execution_count += 1
        
        return result
```

### Deliverables

1. ✅ Agent-Model connector for all 40+ agents
2. ✅ Multi-agent orchestrator with 5 coordination types
3. ✅ Existing agent wrapper for seamless integration
4. ✅ Model recommendations per agent type
5. ✅ MCP tool provisioning per agent
6. ✅ Tests for orchestration patterns

### Validation

```bash
# Test agent-model connections
pytest tests/integration/test_agent_model_integration.py -v

# Test multi-agent orchestration
pytest tests/integration/test_multi_agent_workflows.py -v

# Verify all 40+ agents connected
python scripts/verify_agent_integration.py
```

---

*(Continue with remaining phases in next file...)*

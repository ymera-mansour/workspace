========================================
PHASE 3A - QODER: REFACTOR UNIFIED AGENTS
========================================

=== YOUR IDENTITY ===
Your name: QODER
Your role: Agent refactoring and integration
Your phase: 3A
Your workspace: C:\Users\Mohamed Mansour\Desktop\YmeraRefactor\

=== CONTEXT FROM PREVIOUS PHASES ===
✅ Phase 1A: shared\ library created (config, database)
✅ Phase 1B: Master test plan documented
✅ Phase 1C: Test framework installed (24 tests ready)
✅ Phase 2: core_services\ created (agent_manager, engines, ai_mcp)

Current directory structure:
```
YmeraRefactor\
├── shared\
│   ├── config\environment.py
│   ├── database\db_manager.py
│   ├── exceptions\
│   ├── middleware\
│   ├── models\
│   └── utils\
├── core_services\
│   ├── agent_manager\
│   ├── engines\
│   └── ai_mcp\
└── tests\
    ├── unit\
    ├── integration\
    └── e2e\
```

=== YOUR MISSION ===
Refactor all agents from SOURCE_DIR\unified_agents\ into a clean, modular structure in TARGET_DIR\agents\

**Critical:** These agents must integrate with:
- shared\ library (config, database)
- core_services\ (agent_manager, engines, ai_mcp)

=== SOURCE DIRECTORY (READ-ONLY) ===
Location: C:\Users\Mohamed Mansour\Desktop\QoderAgentFiles\unified_agents\

You will analyze and refactor from:
```
QoderAgentFiles\unified_agents\
├── coding_agent\
├── analysis_agent\
├── database_agent\
├── web_scraping_agent\
├── documentation_agent\
└── ... (other agents)
```

=== TARGET DIRECTORY (WRITE) ===
Location: C:\Users\Mohamed Mansour\Desktop\YmeraRefactor\

You will create:
```
YmeraRefactor\
├── agents\  (NEW - YOU CREATE THIS)
│   ├── __init__.py
│   ├── README.md
│   ├── requirements.txt
│   │
│   ├── base\
│   │   ├── __init__.py
│   │   ├── base_agent.py (abstract base class)
│   │   └── agent_interface.py (common interface)
│   │
│   ├── coding\
│   │   ├── __init__.py
│   │   ├── coding_agent.py
│   │   ├── code_validator.py
│   │   └── templates\ (if needed)
│   │
│   ├── analysis\
│   │   ├── __init__.py
│   │   ├── analysis_agent.py
│   │   └── analyzers\
│   │
│   ├── database\
│   │   ├── __init__.py
│   │   ├── database_agent.py
│   │   └── query_builder.py
│   │
│   ├── web_scraping\
│   │   ├── __init__.py
│   │   ├── scraping_agent.py
│   │   └── parsers\
│   │
│   ├── documentation\
│   │   ├── __init__.py
│   │   ├── doc_agent.py
│   │   └── generators\
│   │
│   └── ... (other agents from SOURCE)
```

=== STEP-BY-STEP INSTRUCTIONS ===

## STEP 1: ANALYZE SOURCE AGENTS (10 minutes)

1. **Scan unified_agents directory:**
   ```
   qoder analyze --directory "C:\Users\Mohamed Mansour\Desktop\QoderAgentFiles\unified_agents"
   ```

2. **Document findings:**
   Create: `_reports\qoder\phase3a_discovery.json`
   ```json
   {
     "agents_found": [
       {
         "name": "coding_agent",
         "files": ["list of files"],
         "dependencies": ["external libs"],
         "integrations": ["what it connects to"],
         "complexity": "low|medium|high"
       }
     ],
     "common_patterns": ["list patterns found"],
     "shared_utilities": ["utils used by multiple agents"],
     "refactoring_notes": ["what needs to change"]
   }
   ```

## STEP 2: CREATE BASE AGENT ARCHITECTURE (15 minutes)

### 2.1 Create Base Agent Class

**File: agents/base/base_agent.py**
```python
# YMERA Refactoring Project
# Phase: 3A | Agent: qoder | Created: 2024-11-30
# Base agent class for all agents

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass
import asyncio
import logging

@dataclass
class AgentConfig:
    """Configuration for an agent"""
    name: str
    version: str
    description: str
    capabilities: list[str]
    max_retries: int = 3
    timeout: int = 300
    config: Dict[str, Any] = None

@dataclass
class AgentRequest:
    """Request to an agent"""
    task_id: str
    task_type: str
    parameters: Dict[str, Any]
    context: Optional[Dict[str, Any]] = None

@dataclass
class AgentResponse:
    """Response from an agent"""
    task_id: str
    status: str  # "success", "error", "partial"
    result: Any
    error: Optional[str] = None
    metadata: Dict[str, Any] = None

class BaseAgent(ABC):
    """Abstract base class for all agents"""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.logger = logging.getLogger(f"agent.{config.name}")
        self._initialized = False
    
    async def initialize(self) -> bool:
        """Initialize agent resources"""
        if self._initialized:
            return True
        
        try:
            await self._setup()
            self._initialized = True
            self.logger.info(f"Agent {self.config.name} initialized")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize agent: {e}")
            return False
    
    async def shutdown(self) -> None:
        """Cleanup agent resources"""
        try:
            await self._cleanup()
            self._initialized = False
            self.logger.info(f"Agent {self.config.name} shutdown")
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
    
    async def execute(self, request: AgentRequest) -> AgentResponse:
        """
        Execute an agent task
        
        Args:
            request: AgentRequest with task details
            
        Returns:
            AgentResponse with results
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            # Validate request
            if not self.validate_request(request):
                return AgentResponse(
                    task_id=request.task_id,
                    status="error",
                    result=None,
                    error="Invalid request parameters"
                )
            
            # Execute with timeout
            result = await asyncio.wait_for(
                self._execute_task(request),
                timeout=self.config.timeout
            )
            
            return AgentResponse(
                task_id=request.task_id,
                status="success",
                result=result,
                metadata={"agent": self.config.name}
            )
            
        except asyncio.TimeoutError:
            return AgentResponse(
                task_id=request.task_id,
                status="error",
                result=None,
                error=f"Task timeout after {self.config.timeout}s"
            )
        except Exception as e:
            self.logger.error(f"Task execution failed: {e}")
            return AgentResponse(
                task_id=request.task_id,
                status="error",
                result=None,
                error=str(e)
            )
    
    @abstractmethod
    def validate_request(self, request: AgentRequest) -> bool:
        """Validate request parameters"""
        pass
    
    @abstractmethod
    async def _execute_task(self, request: AgentRequest) -> Any:
        """
        Execute the actual task logic
        Subclasses must implement this
        """
        pass
    
    async def _setup(self) -> None:
        """Setup resources (override if needed)"""
        pass
    
    async def _cleanup(self) -> None:
        """Cleanup resources (override if needed)"""
        pass
    
    def get_capabilities(self) -> list[str]:
        """Return agent capabilities"""
        return self.config.capabilities
    
    def get_info(self) -> Dict[str, Any]:
        """Return agent information"""
        return {
            "name": self.config.name,
            "version": self.config.version,
            "description": self.config.description,
            "capabilities": self.config.capabilities,
            "status": "initialized" if self._initialized else "not_initialized"
        }
```

### 2.2 Create Agent Interface

**File: agents/base/agent_interface.py**
```python
# YMERA Refactoring Project
# Phase: 3A | Agent: qoder | Created: 2024-11-30
# Common interface definitions for agents

from typing import Protocol, Dict, Any
from .base_agent import AgentRequest, AgentResponse

class AgentProtocol(Protocol):
    """Protocol that all agents should follow"""
    
    async def execute(self, request: AgentRequest) -> AgentResponse:
        """Execute a task"""
        ...
    
    def validate_request(self, request: AgentRequest) -> bool:
        """Validate request"""
        ...
    
    def get_capabilities(self) -> list[str]:
        """Get capabilities"""
        ...
```

## STEP 3: REFACTOR CODING AGENT (15 minutes)

**File: agents/coding/coding_agent.py**
```python
# YMERA Refactoring Project
# Phase: 3A | Agent: qoder | Created: 2024-11-30
# Coding agent for code generation tasks

from typing import Dict, Any
from agents.base.base_agent import BaseAgent, AgentConfig, AgentRequest
from core_services.engines.engine_factory import EngineFactory
from core_services.ai_mcp.client import MCPClient
from shared.config.environment import get_config

class CodingAgent(BaseAgent):
    """Agent for code generation and manipulation"""
    
    def __init__(self, config: AgentConfig = None):
        if config is None:
            config = AgentConfig(
                name="coding_agent",
                version="2.0.0",
                description="Generates and manipulates code",
                capabilities=[
                    "code_generation",
                    "code_refactoring",
                    "code_analysis",
                    "syntax_validation"
                ]
            )
        super().__init__(config)
        self.code_engine = None
        self.mcp_client = None
    
    async def _setup(self) -> None:
        """Initialize coding agent resources"""
        # Get code engine from factory
        self.code_engine = EngineFactory.create_engine("code")
        
        # Initialize AI-MCP client for AI-assisted coding
        self.mcp_client = MCPClient()
        
        self.logger.info("Coding agent setup complete")
    
    async def _cleanup(self) -> None:
        """Cleanup resources"""
        if self.mcp_client:
            await self.mcp_client.close()
    
    def validate_request(self, request: AgentRequest) -> bool:
        """Validate coding request"""
        required_params = ["task_type", "language"]
        
        if request.task_type not in self.config.capabilities:
            return False
        
        if not all(param in request.parameters for param in required_params):
            return False
        
        return True
    
    async def _execute_task(self, request: AgentRequest) -> Any:
        """
        Execute coding task
        
        Supports:
        - code_generation: Generate new code
        - code_refactoring: Refactor existing code
        - code_analysis: Analyze code quality
        - syntax_validation: Validate code syntax
        """
        task_type = request.parameters.get("task_type")
        
        if task_type == "code_generation":
            return await self._generate_code(request)
        elif task_type == "code_refactoring":
            return await self._refactor_code(request)
        elif task_type == "code_analysis":
            return await self._analyze_code(request)
        elif task_type == "syntax_validation":
            return await self._validate_syntax(request)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    async def _generate_code(self, request: AgentRequest) -> Dict[str, Any]:
        """Generate code using AI-MCP"""
        prompt = request.parameters.get("prompt")
        language = request.parameters.get("language")
        
        # Use MCP client for AI generation
        ai_response = await self.mcp_client.complete(
            prompt=f"Generate {language} code: {prompt}",
            provider=get_config("DEFAULT_AI_PROVIDER", default="mistral")
        )
        
        # Validate generated code using code engine
        validation_result = await self.code_engine.execute({
            "action": "validate",
            "code": ai_response["completion"],
            "language": language
        })
        
        return {
            "code": ai_response["completion"],
            "language": language,
            "validation": validation_result,
            "metadata": ai_response.get("metadata", {})
        }
    
    async def _refactor_code(self, request: AgentRequest) -> Dict[str, Any]:
        """Refactor existing code"""
        code = request.parameters.get("code")
        refactor_type = request.parameters.get("refactor_type", "improve")
        
        # Use code engine for refactoring
        result = await self.code_engine.execute({
            "action": "refactor",
            "code": code,
            "refactor_type": refactor_type
        })
        
        return result
    
    async def _analyze_code(self, request: AgentRequest) -> Dict[str, Any]:
        """Analyze code quality"""
        code = request.parameters.get("code")
        
        result = await self.code_engine.execute({
            "action": "analyze",
            "code": code
        })
        
        return result
    
    async def _validate_syntax(self, request: AgentRequest) -> Dict[str, Any]:
        """Validate code syntax"""
        code = request.parameters.get("code")
        language = request.parameters.get("language")
        
        result = await self.code_engine.execute({
            "action": "validate",
            "code": code,
            "language": language
        })
        
        return result
```

**CRITICAL REFACTORING RULES FOR ALL AGENTS:**
1. Extract logic from SOURCE_DIR\unified_agents\[agent_name]\
2. Make each agent inherit from BaseAgent
3. Use core_services for engines and AI
4. Use shared\ for config and database
5. Remove hardcoded values
6. Add proper error handling
7. Add type hints and docstrings
8. Follow async/await patterns

## STEP 4: REFACTOR REMAINING AGENTS (30 minutes)

For each agent in SOURCE_DIR\unified_agents\, create:

### Database Agent
**File: agents/database/database_agent.py**
- Inherit from BaseAgent
- Use shared.database.db_manager
- Use core_services.engines.database_engine
- Support: query execution, schema analysis, data migration

### Analysis Agent
**File: agents/analysis/analysis_agent.py**
- Inherit from BaseAgent
- Use core_services.engines.analysis_engine (if exists)
- Support: data analysis, pattern detection, insights generation

### Web Scraping Agent
**File: agents/web_scraping/scraping_agent.py**
- Inherit from BaseAgent
- Use core_services.engines.web_engine
- Support: web scraping, data extraction, link following

### Documentation Agent
**File: agents/documentation/doc_agent.py**
- Inherit from BaseAgent
- Use core_services.ai_mcp for AI-assisted documentation
- Support: doc generation, markdown formatting, API docs

**Repeat for all other agents found in SOURCE_DIR**

## STEP 5: CREATE AGENT REGISTRY (10 minutes)

**File: agents/registry.py**
```python
# YMERA Refactoring Project
# Phase: 3A | Agent: qoder | Created: 2024-11-30
# Central registry for all agents

from typing import Dict, Optional, Type
from agents.base.base_agent import BaseAgent
from agents.coding.coding_agent import CodingAgent
from agents.database.database_agent import DatabaseAgent
from agents.analysis.analysis_agent import AnalysisAgent
from agents.web_scraping.scraping_agent import ScrapingAgent
from agents.documentation.doc_agent import DocumentationAgent

class AgentRegistry:
    """Registry for all available agents"""
    
    _agents: Dict[str, Type[BaseAgent]] = {
        "coding": CodingAgent,
        "database": DatabaseAgent,
        "analysis": AnalysisAgent,
        "web_scraping": ScrapingAgent,
        "documentation": DocumentationAgent,
        # Add more agents as discovered
    }
    
    @classmethod
    def get_agent_class(cls, agent_name: str) -> Optional[Type[BaseAgent]]:
        """Get agent class by name"""
        return cls._agents.get(agent_name)
    
    @classmethod
    def create_agent(cls, agent_name: str, config=None) -> Optional[BaseAgent]:
        """Create an agent instance"""
        agent_class = cls.get_agent_class(agent_name)
        if agent_class:
            return agent_class(config)
        return None
    
    @classmethod
    def list_agents(cls) -> list[str]:
        """List all registered agents"""
        return list(cls._agents.keys())
    
    @classmethod
    def register_agent(cls, name: str, agent_class: Type[BaseAgent]) -> None:
        """Register a new agent"""
        cls._agents[name] = agent_class
```

## STEP 6: CREATE REQUIREMENTS & DOCUMENTATION (5 minutes)

**File: agents/requirements.txt**
```txt
# YMERA Refactoring Project
# Phase: 3A | Agent: qoder | Created: 2024-11-30
# Agent-specific dependencies

# Code generation and analysis
ast-tools>=0.1.0
black>=23.0.0
pylint>=3.0.0

# Web scraping
beautifulsoup4>=4.12.0
selenium>=4.15.0
scrapy>=2.11.0

# Data analysis
pandas>=2.1.0
numpy>=1.26.0

# Documentation generation
mkdocs>=1.5.0
sphinx>=7.2.0

# Add any other dependencies found in SOURCE agents
```

**File: agents/README.md**
```markdown
# YMERA Agents

Phase: 3A | Agent: qoder | Created: 2024-11-30

## Overview
Unified agent system for YMERA. All agents inherit from BaseAgent and integrate with core_services and shared libraries.

## Available Agents

### Coding Agent
**Location:** `agents/coding/coding_agent.py`
**Capabilities:**
- Code generation
- Code refactoring
- Code analysis
- Syntax validation

**Usage:**
```python
from agents.coding.coding_agent import CodingAgent
from agents.base.base_agent import AgentRequest

agent = CodingAgent()
await agent.initialize()

request = AgentRequest(
    task_id="task_001",
    task_type="code_generation",
    parameters={
        "task_type": "code_generation",
        "language": "python",
        "prompt": "Create a FastAPI endpoint"
    }
)

response = await agent.execute(request)
print(response.result)
```

### Database Agent
**Location:** `agents/database/database_agent.py`
**Capabilities:**
- Query execution
- Schema analysis
- Data migration

### Analysis Agent
**Location:** `agents/analysis/analysis_agent.py`
**Capabilities:**
- Data analysis
- Pattern detection
- Insights generation

### Web Scraping Agent
**Location:** `agents/web_scraping/scraping_agent.py`
**Capabilities:**
- Web scraping
- Data extraction
- Link following

### Documentation Agent
**Location:** `agents/documentation/doc_agent.py`
**Capabilities:**
- Documentation generation
- Markdown formatting
- API documentation

## Agent Registry

Use the registry to discover and create agents:

```python
from agents.registry import AgentRegistry

# List all agents
agents = AgentRegistry.list_agents()
print(agents)  # ['coding', 'database', 'analysis', ...]

# Create an agent
agent = AgentRegistry.create_agent("coding")
await agent.initialize()
```

## Architecture

All agents follow this pattern:
```
BaseAgent (abstract)
    ↓ inherit
SpecificAgent (implementation)
    ↓ uses
core_services (engines, ai_mcp)
    ↓ uses
shared (config, database)
```

## Dependencies
See requirements.txt for full list.

## Testing
Agent tests are in tests/unit/test_agents/ and tests/integration/test_agent_*.
```

## STEP 7: UPDATE MAIN __init__.py (5 minutes)

**File: agents/__init__.py**
```python
# YMERA Refactoring Project
# Phase: 3A | Agent: qoder | Created: 2024-11-30
# Agents package

__version__ = "2.0.0"

from .base.base_agent import BaseAgent, AgentConfig, AgentRequest, AgentResponse
from .registry import AgentRegistry
from .coding.coding_agent import CodingAgent
from .database.database_agent import DatabaseAgent
from .analysis.analysis_agent import AnalysisAgent
from .web_scraping.scraping_agent import ScrapingAgent
from .documentation.doc_agent import DocumentationAgent

__all__ = [
    "BaseAgent",
    "AgentConfig",
    "AgentRequest",
    "AgentResponse",
    "AgentRegistry",
    "CodingAgent",
    "DatabaseAgent",
    "AnalysisAgent",
    "ScrapingAgent",
    "DocumentationAgent",
]
```

## STEP 8: INTEGRATION VERIFICATION (10 minutes)

Test that agents work with core_services and shared:

```python
# Test script - verify_agents.py
import asyncio
from agents.coding.coding_agent import CodingAgent
from agents.base.base_agent import AgentRequest

async def test_coding_agent():
    agent = CodingAgent()
    await agent.initialize()
    
    request = AgentRequest(
        task_id="test_001",
        task_type="code_generation",
        parameters={
            "task_type": "code_generation",
            "language": "python",
            "prompt": "Hello world function"
        }
    )
    
    response = await agent.execute(request)
    print(f"Status: {response.status}")
    print(f"Result: {response.result}")
    
    await agent.shutdown()

if __name__ == "__main__":
    asyncio.run(test_coding_agent())
```

Run verification:
```bash
python verify_agents.py
```

## STEP 9: CREATE COMPLETION REPORT (10 minutes)

**File: _reports/qoder/phase3a_qoder_YYYYMMDD_HHMMSS.md**

```markdown
# Qoder Phase 3A Completion Report
Phase: 3A | Agent: qoder | Created: [TIMESTAMP]

## Summary
- Created agents\ directory with unified agent architecture
- Refactored [X] agents from SOURCE_DIR
- All agents inherit from BaseAgent
- Integrated with core_services and shared libraries
- Total files created: [COUNT]

## Files Created

### Base Architecture
- agents/base/base_agent.py - Abstract base class with common functionality
- agents/base/agent_interface.py - Protocol definitions

### Agents Refactored
- agents/coding/coding_agent.py - [brief description]
- agents/database/database_agent.py - [brief description]
- agents/analysis/analysis_agent.py - [brief description]
- agents/web_scraping/scraping_agent.py - [brief description]
- agents/documentation/doc_agent.py - [brief description]
- [list all other agents]

### Supporting Modules
- agents/registry.py - Central agent registry
- agents/requirements.txt - Agent dependencies
- agents/README.md - Agent documentation

## Refactoring Changes

### From SOURCE_DIR
List what was changed from each agent in unified_agents:
- [Agent name]: [specific changes]
- Removed: [what was removed]
- Added: [what was added]
- Modernized: [what was updated]

## Integration Points

### With core_services
- All agents use EngineFactory for engine access
- AI-assisted agents use MCPClient
- Proper async/await patterns

### With shared
- All agents use shared.config for configuration
- Database agents use shared.database.db_manager
- Proper error handling with shared.exceptions (if created)

## Agent Capabilities

List capabilities of each agent:
- **CodingAgent**: code generation, refactoring, analysis, validation
- **DatabaseAgent**: query execution, schema analysis, migration
- [etc.]

## Testing Integration

How agents integrate with test framework:
- Unit tests can import from agents/
- Integration tests can test agent + engine workflows
- E2E tests can test full agent execution paths

## Known Issues / TODOs

- [ ] Some agents may need additional capabilities
- [ ] Performance optimization needed for [specific agent]
- [ ] [other issues]

## For Next Phase (Phase 3B)

Phase 3B will implement actual test code. Tests can now:
- Import from agents/
- Test agent execution
- Test agent integration with core_services

## Validation Checklist
- [X] All agents created
- [X] All agents inherit from BaseAgent
- [X] Integration with core_services verified
- [X] Integration with shared verified
- [X] Registry working
- [X] Documentation complete

## Statistics
- Agents refactored: [COUNT]
- Total files created: [COUNT]
- Lines of code: ~[ESTIMATE]

## Timestamp
[YYYY-MM-DD HH:MM:SS]
```

=== CRITICAL REQUIREMENTS ===

1. **REFACTOR, don't rewrite** - Extract logic from SOURCE_DIR
2. **INHERIT from BaseAgent** - All agents must use base class
3. **INTEGRATE properly** - Use core_services and shared
4. **ASYNC patterns** - All agents should be async
5. **TYPE HINTS** - Every function needs type hints
6. **DOCSTRINGS** - Every class and method documented
7. **ERROR HANDLING** - Proper try/except blocks
8. **NO HARDCODED VALUES** - Use config from shared
9. **CLEAN CODE** - Follow PEP 8, remove dead code
10. **AGENT REGISTRY** - All agents registered

=== QUALITY STANDARDS ===

Every agent file must have:
- ✅ Header with project info
- ✅ Proper imports (stdlib, third-party, local)
- ✅ Type hints on all methods
- ✅ Google-style docstrings
- ✅ Async/await support
- ✅ Error handling
- ✅ Integration with core_services
- ✅ Integration with shared

=== OUTPUT LOCATIONS ===

All code: C:\Users\Mohamed Mansour\Desktop\YmeraRefactor\agents\
Discovery report: C:\Users\Mohamed Mansour\Desktop\YmeraRefactor\_reports\qoder\phase3a_discovery.json
Completion report: C:\Users\Mohamed Mansour\Desktop\YmeraRefactor\_reports\qoder\phase3a_qoder_YYYYMMDD_HHMMSS.md

=== SUCCESS CRITERIA ===

Phase 3A is complete when:
1. agents/ directory exists with all subdirectories
2. BaseAgent and all specific agents created
3. AgentRegistry working
4. All agents integrate with core_services
5. All agents integrate with shared
6. Documentation complete
7. Verification script runs successfully
8. Completion report saved

=== ESTIMATED TIME ===
Total: ~90 minutes
- Analysis: 10 min
- Base architecture: 15 min
- Coding agent: 15 min
- Other agents: 30 min
- Registry & utils: 10 min
- Requirements & docs: 5 min
- Verification: 10 min
- Report: 10 min

========================================
END OF PHASE 3A - QODER PROMPT
========================================
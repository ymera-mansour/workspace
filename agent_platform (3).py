# Production Multi-Agent Platform with Real Integrations
# Supports: Mistral, Gemini, Groq, HuggingFace + Redis + MCP

import asyncio
import json
import hashlib
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
from datetime import datetime
import redis.asyncio as redis
from sentence_transformers import SentenceTransformer
import numpy as np

# LLM Provider imports
try:
    from mistralai.async_client import MistralAsyncClient
    from mistralai.models.chat_completion import ChatMessage
except ImportError:
    print("⚠️  pip install mistralai")

try:
    import google.generativeai as genai
except ImportError:
    print("⚠️  pip install google-generativeai")

try:
    from groq import AsyncGroq
except ImportError:
    print("⚠️  pip install groq")

try:
    from huggingface_hub import AsyncInferenceClient
except ImportError:
    print("⚠️  pip install huggingface-hub")

try:
    import aiohttp
except ImportError:
    print("⚠️  pip install aiohttp")


# ============================================================================
# CONFIGURATION MANAGEMENT
# ============================================================================

@dataclass
class ProviderConfig:
    name: str
    api_key: str
    models: List[str]
    cost_per_1k_tokens: float
    max_tokens: int
    rate_limit_rpm: int
    enabled: bool = True

class ConfigManager:
    """Centralized configuration for all providers"""
    
    def __init__(self):
        self.providers = self._load_providers()
        
    def _load_providers(self) -> Dict[str, ProviderConfig]:
        """Load API keys from environment"""
        return {
            "mistral": ProviderConfig(
                name="mistral",
                api_key=os.getenv("MISTRAL_API_KEY", ""),
                models=[
                    "mistral-small-latest",      # Fastest, cheapest
                    "mistral-medium-latest",     # Balanced
                    "mistral-large-latest",      # Most capable
                ],
                cost_per_1k_tokens=0.0002,
                max_tokens=32000,
                rate_limit_rpm=60
            ),
            "groq": ProviderConfig(
                name="groq",
                api_key=os.getenv("GROQ_API_KEY", ""),
                models=[
                    "llama-3.1-8b-instant",      # Ultra fast, free tier
                    "llama-3.1-70b-versatile",   # Best free model
                    "mixtral-8x7b-32768",        # Good for reasoning
                ],
                cost_per_1k_tokens=0.0,  # Free tier
                max_tokens=32768,
                rate_limit_rpm=30
            ),
            "gemini": ProviderConfig(
                name="gemini",
                api_key=os.getenv("GEMINI_API_KEY", ""),
                models=[
                    "gemini-1.5-flash",          # Fast and cheap
                    "gemini-1.5-pro",            # Most capable
                ],
                cost_per_1k_tokens=0.00015,
                max_tokens=1000000,  # 1M context!
                rate_limit_rpm=60
            ),
            "huggingface": ProviderConfig(
                name="huggingface",
                api_key=os.getenv("HF_API_KEY", ""),
                models=[
                    "meta-llama/Meta-Llama-3-8B-Instruct",
                    "mistralai/Mistral-7B-Instruct-v0.2",
                    "microsoft/phi-2",
                ],
                cost_per_1k_tokens=0.0,  # Free inference API
                max_tokens=4096,
                rate_limit_rpm=20
            )
        }
    
    def get_cheapest_model(self, capability: str = "general") -> tuple[str, str]:
        """Return (provider, model) for cheapest option"""
        # Priority: Free (Groq) > Cheap (Mistral/Gemini) > Backup (HF)
        if self.providers["groq"].enabled:
            return ("groq", "llama-3.1-70b-versatile")
        elif self.providers["gemini"].enabled:
            return ("gemini", "gemini-1.5-flash")
        elif self.providers["mistral"].enabled:
            return ("mistral", "mistral-small-latest")
        else:
            return ("huggingface", "meta-llama/Meta-Llama-3-8B-Instruct")
    
    def get_best_model(self, capability: str = "general") -> tuple[str, str]:
        """Return (provider, model) for highest quality"""
        if self.providers["gemini"].enabled:
            return ("gemini", "gemini-1.5-pro")
        elif self.providers["mistral"].enabled:
            return ("mistral", "mistral-large-latest")
        elif self.providers["groq"].enabled:
            return ("groq", "llama-3.1-70b-versatile")
        else:
            return ("huggingface", "meta-llama/Meta-Llama-3-8B-Instruct")


# ============================================================================
# REDIS CACHE IMPLEMENTATION
# ============================================================================

class RedisCache:
    """Production Redis cache with semantic search"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.client: Optional[redis.Redis] = None
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
    async def connect(self):
        """Initialize Redis connection"""
        self.client = await redis.from_url(
            self.redis_url,
            encoding="utf-8",
            decode_responses=True
        )
        print("✓ Redis connected")
    
    async def close(self):
        """Close Redis connection"""
        if self.client:
            await self.client.close()
    
    def _get_exact_key(self, prompt: str) -> str:
        """Generate exact match cache key"""
        return f"cache:exact:{hashlib.md5(prompt.encode()).hexdigest()}"
    
    def _get_semantic_key(self, prompt: str) -> str:
        """Generate semantic cache key (normalized)"""
        normalized = ' '.join(prompt.lower().split())
        return f"cache:semantic:{hashlib.md5(normalized.encode()).hexdigest()}"
    
    async def get(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Retrieve from cache with fallback strategy"""
        if not self.client:
            return None
        
        # Try exact match first
        exact_key = self._get_exact_key(prompt)
        cached = await self.client.get(exact_key)
        
        if cached:
            data = json.loads(cached)
            await self.client.hincrby("cache:stats", "exact_hits", 1)
            return data
        
        # Try semantic match
        semantic_key = self._get_semantic_key(prompt)
        cached = await self.client.get(semantic_key)
        
        if cached:
            data = json.loads(cached)
            # Only return if confidence is high
            if data.get("confidence", 0) > 0.85:
                await self.client.hincrby("cache:stats", "semantic_hits", 1)
                return data
        
        await self.client.hincrby("cache:stats", "misses", 1)
        return None
    
    async def set(self, 
                  prompt: str, 
                  response: str, 
                  metadata: Dict[str, Any],
                  ttl: int = 3600):
        """Store in cache with TTL"""
        if not self.client:
            return
        
        cache_data = {
            "response": response,
            "metadata": metadata,
            "timestamp": datetime.now().isoformat(),
            "confidence": 1.0,
            "hit_count": 0
        }
        
        # Store both exact and semantic
        exact_key = self._get_exact_key(prompt)
        semantic_key = self._get_semantic_key(prompt)
        
        await self.client.setex(
            exact_key,
            ttl,
            json.dumps(cache_data)
        )
        
        await self.client.setex(
            semantic_key,
            ttl,
            json.dumps(cache_data)
        )
    
    async def get_stats(self) -> Dict[str, int]:
        """Get cache performance stats"""
        if not self.client:
            return {}
        
        stats = await self.client.hgetall("cache:stats")
        return {k: int(v) for k, v in stats.items()}
    
    async def rate_limit_check(self, user_id: str, limit: int = 100) -> bool:
        """Redis-based rate limiting"""
        if not self.client:
            return True
        
        key = f"ratelimit:{user_id}"
        current = await self.client.incr(key)
        
        if current == 1:
            await self.client.expire(key, 60)  # 1 minute window
        
        return current <= limit
    
    async def store_conversation(self, user_id: str, message: Dict[str, Any]):
        """Store conversation history"""
        if not self.client:
            return
        
        key = f"conversation:{user_id}"
        await self.client.lpush(key, json.dumps(message))
        await self.client.ltrim(key, 0, 49)  # Keep last 50 messages
        await self.client.expire(key, 86400)  # 24 hour TTL
    
    async def get_conversation(self, user_id: str, limit: int = 10) -> List[Dict]:
        """Retrieve conversation history"""
        if not self.client:
            return []
        
        key = f"conversation:{user_id}"
        messages = await self.client.lrange(key, 0, limit - 1)
        return [json.loads(msg) for msg in messages]


# ============================================================================
# MCP CLIENT IMPLEMENTATION
# ============================================================================

@dataclass
class MCPToolDefinition:
    name: str
    description: str
    parameters: Dict[str, Any]
    server_url: str

class MCPClient:
    """Model Context Protocol client for tool integration"""
    
    def __init__(self):
        self.tools: Dict[str, MCPToolDefinition] = {}
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def initialize(self):
        """Initialize HTTP session"""
        self.session = aiohttp.ClientSession()
        print("✓ MCP Client initialized")
    
    async def close(self):
        """Close HTTP session"""
        if self.session:
            await self.session.close()
    
    def register_tool(self, tool: MCPToolDefinition):
        """Register an MCP tool"""
        self.tools[tool.name] = tool
        print(f"✓ Registered MCP tool: {tool.name}")
    
    async def call_tool(self, 
                       tool_name: str, 
                       parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute MCP tool via JSON-RPC"""
        if tool_name not in self.tools:
            raise ValueError(f"Tool '{tool_name}' not registered")
        
        tool = self.tools[tool_name]
        
        # Build JSON-RPC 2.0 request
        request = {
            "jsonrpc": "2.0",
            "id": hashlib.md5(f"{tool_name}{datetime.now()}".encode()).hexdigest()[:16],
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": parameters
            }
        }
        
        try:
            async with self.session.post(
                tool.server_url,
                json=request,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                result = await response.json()
                
                if "error" in result:
                    raise Exception(f"MCP Error: {result['error']}")
                
                return result.get("result", {})
                
        except asyncio.TimeoutError:
            raise Exception(f"MCP tool '{tool_name}' timed out")
        except Exception as e:
            raise Exception(f"MCP call failed: {str(e)}")
    
    def get_tools_for_llm(self) -> List[Dict[str, Any]]:
        """Format tools for LLM function calling"""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters
                }
            }
            for tool in self.tools.values()
        ]


# ============================================================================
# LLM PROVIDER IMPLEMENTATIONS
# ============================================================================

class LLMProviderManager:
    """Unified interface for all LLM providers"""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.clients = {}
        self.call_counts: Dict[str, int] = {}
        self.total_cost: float = 0.0
        
    async def initialize(self):
        """Initialize all provider clients"""
        cfg = self.config.providers
        
        # Mistral
        if cfg["mistral"].api_key:
            self.clients["mistral"] = MistralAsyncClient(api_key=cfg["mistral"].api_key)
            print("✓ Mistral initialized")
        
        # Groq
        if cfg["groq"].api_key:
            self.clients["groq"] = AsyncGroq(api_key=cfg["groq"].api_key)
            print("✓ Groq initialized")
        
        # Gemini
        if cfg["gemini"].api_key:
            genai.configure(api_key=cfg["gemini"].api_key)
            self.clients["gemini"] = genai
            print("✓ Gemini initialized")
        
        # HuggingFace
        if cfg["huggingface"].api_key:
            self.clients["huggingface"] = AsyncInferenceClient(
                token=cfg["huggingface"].api_key
            )
            print("✓ HuggingFace initialized")
    
    async def complete(self,
                      provider: str,
                      model: str,
                      messages: List[Dict[str, str]],
                      max_tokens: int = 4096,
                      temperature: float = 0.7,
                      tools: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """Unified completion interface"""
        
        if provider not in self.clients:
            raise ValueError(f"Provider '{provider}' not initialized")
        
        start_time = datetime.now()
        
        try:
            if provider == "mistral":
                result = await self._mistral_complete(model, messages, max_tokens, temperature, tools)
            elif provider == "groq":
                result = await self._groq_complete(model, messages, max_tokens, temperature, tools)
            elif provider == "gemini":
                result = await self._gemini_complete(model, messages, max_tokens, temperature)
            elif provider == "huggingface":
                result = await self._huggingface_complete(model, messages, max_tokens, temperature)
            else:
                raise ValueError(f"Unknown provider: {provider}")
            
            # Track usage
            latency_ms = (datetime.now() - start_time).total_seconds() * 1000
            self.call_counts[provider] = self.call_counts.get(provider, 0) + 1
            
            # Estimate cost
            tokens = result.get("usage", {}).get("total_tokens", 1000)
            cost = (tokens / 1000) * self.config.providers[provider].cost_per_1k_tokens
            self.total_cost += cost
            
            result["metadata"] = {
                "provider": provider,
                "model": model,
                "latency_ms": latency_ms,
                "cost": cost
            }
            
            return result
            
        except Exception as e:
            print(f"❌ {provider} error: {str(e)}")
            raise
    
    async def _mistral_complete(self, model, messages, max_tokens, temperature, tools):
        """Mistral implementation"""
        mistral_messages = [
            ChatMessage(role=msg["role"], content=msg["content"])
            for msg in messages
        ]
        
        kwargs = {
            "model": model,
            "messages": mistral_messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        
        response = await self.clients["mistral"].chat(**kwargs)
        
        return {
            "content": response.choices[0].message.content,
            "tool_calls": response.choices[0].message.tool_calls or [],
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
        }
    
    async def _groq_complete(self, model, messages, max_tokens, temperature, tools):
        """Groq implementation"""
        kwargs = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        
        response = await self.clients["groq"].chat.completions.create(**kwargs)
        
        return {
            "content": response.choices[0].message.content,
            "tool_calls": response.choices[0].message.tool_calls or [],
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
        }
    
    async def _gemini_complete(self, model, messages, max_tokens, temperature):
        """Gemini implementation"""
        # Convert messages to Gemini format
        gemini_model = self.clients["gemini"].GenerativeModel(model)
        
        # Combine messages into single prompt (Gemini uses different format)
        prompt = "\n\n".join([
            f"{msg['role'].upper()}: {msg['content']}"
            for msg in messages
        ])
        
        response = await gemini_model.generate_content_async(
            prompt,
            generation_config={
                "max_output_tokens": max_tokens,
                "temperature": temperature
            }
        )
        
        return {
            "content": response.text,
            "tool_calls": [],
            "usage": {
                "prompt_tokens": response.usage_metadata.prompt_token_count,
                "completion_tokens": response.usage_metadata.candidates_token_count,
                "total_tokens": response.usage_metadata.total_token_count
            }
        }
    
    async def _huggingface_complete(self, model, messages, max_tokens, temperature):
        """HuggingFace implementation"""
        # Combine messages for HF
        prompt = "\n".join([
            f"{msg['role']}: {msg['content']}"
            for msg in messages
        ])
        
        response = await self.clients["huggingface"].text_generation(
            prompt,
            model=model,
            max_new_tokens=max_tokens,
            temperature=temperature
        )
        
        return {
            "content": response,
            "tool_calls": [],
            "usage": {
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": len(response.split()),
                "total_tokens": len(prompt.split()) + len(response.split())
            }
        }
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics"""
        return {
            "total_calls": sum(self.call_counts.values()),
            "calls_by_provider": self.call_counts,
            "total_cost": round(self.total_cost, 4),
            "avg_cost_per_call": round(
                self.total_cost / max(sum(self.call_counts.values()), 1),
                4
            )
        }


# ============================================================================
# AGENT DEFINITIONS WITH PROVIDER ROUTING
# ============================================================================

@dataclass
class AgentConfig:
    name: str
    description: str
    system_prompt: str
    keywords: List[str]
    preferred_provider: str  # "cheap", "balanced", "best"
    requires_tools: bool = False
    cost_tier: int = 1

class AgentRegistry:
    """Registry of all available agents"""
    
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.agents: Dict[str, AgentConfig] = {}
        
    def register_default_agents(self):
        """Register production-ready agents"""
        
        agents = [
            AgentConfig(
                name="code_generator",
                description="Generates production-ready code in any language",
                system_prompt="""You are an expert software engineer. Generate clean, 
                efficient, well-documented code. Follow best practices and include error handling.
                Explain your code choices briefly.""",
                keywords=["code", "python", "javascript", "function", "class", "programming"],
                preferred_provider="balanced",
                cost_tier=2
            ),
            
            AgentConfig(
                name="web_researcher",
                description="Searches web and synthesizes information",
                system_prompt="""You are a research assistant. Search the web for accurate,
                current information. Synthesize findings clearly and cite sources.""",
                keywords=["search", "research", "find", "web", "google", "information"],
                preferred_provider="cheap",
                requires_tools=True,
                cost_tier=1
            ),
            
            AgentConfig(
                name="data_analyst",
                description="Analyzes data and creates visualizations",
                system_prompt="""You are a data scientist. Analyze data thoroughly,
                identify patterns, and provide actionable insights. Create clear visualizations.""",
                keywords=["analyze", "data", "statistics", "chart", "graph", "sql"],
                preferred_provider="balanced",
                cost_tier=2
            ),
            
            AgentConfig(
                name="technical_writer",
                description="Creates documentation and technical content",
                system_prompt="""You are a technical writer. Create clear, comprehensive
                documentation. Use proper formatting and examples.""",
                keywords=["documentation", "guide", "tutorial", "explain", "how-to"],
                preferred_provider="cheap",
                cost_tier=1
            ),
            
            AgentConfig(
                name="code_reviewer",
                description="Reviews code for bugs and improvements",
                system_prompt="""You are a senior code reviewer. Identify bugs, security issues,
                performance problems, and suggest improvements. Be constructive.""",
                keywords=["review", "audit", "check", "debug", "improve", "refactor"],
                preferred_provider="best",
                cost_tier=3
            ),
            
            AgentConfig(
                name="creative_writer",
                description="Writes creative content and stories",
                system_prompt="""You are a creative writer. Write engaging, original content
                with vivid descriptions and compelling narratives.""",
                keywords=["write", "story", "creative", "narrative", "content", "blog"],
                preferred_provider="best",
                cost_tier=3
            ),
        ]
        
        for agent in agents:
            self.agents[agent.name] = agent
            
        print(f"✓ Registered {len(agents)} agents")
    
    def get_agent(self, name: str) -> Optional[AgentConfig]:
        """Get agent configuration"""
        return self.agents.get(name)
    
    def get_all_agents(self) -> List[AgentConfig]:
        """Get all registered agents"""
        return list(self.agents.values())
    
    def get_model_for_agent(self, agent_name: str) -> tuple[str, str]:
        """Determine best model for agent"""
        agent = self.agents.get(agent_name)
        if not agent:
            return self.config_manager.get_cheapest_model()
        
        if agent.preferred_provider == "cheap":
            return self.config_manager.get_cheapest_model()
        elif agent.preferred_provider == "best":
            return self.config_manager.get_best_model()
        else:  # balanced
            # Use Groq for balanced (fast + free)
            if self.config_manager.providers["groq"].enabled:
                return ("groq", "llama-3.1-70b-versatile")
            return self.config_manager.get_cheapest_model()


# ============================================================================
# INTEGRATED ORCHESTRATION ENGINE
# ============================================================================

class ProductionOrchestrator:
    """Complete orchestrator with all integrations"""
    
    def __init__(self):
        self.config = ConfigManager()
        self.cache = RedisCache()
        self.mcp = MCPClient()
        self.llm = LLMProviderManager(self.config)
        self.agents = AgentRegistry(self.config)
        
    async def initialize(self):
        """Initialize all components"""
        print("\n🚀 Initializing Production Multi-Agent Platform...\n")
        
        await self.cache.connect()
        await self.mcp.initialize()
        await self.llm.initialize()
        self.agents.register_default_agents()
        
        # Register example MCP tools
        self._register_mcp_tools()
        
        print("\n✅ Platform ready!\n")
    
    def _register_mcp_tools(self):
        """Register MCP tools for agents"""
        
        # Web search tool
        self.mcp.register_tool(MCPToolDefinition(
            name="web_search",
            description="Search the web for current information",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                },
                "required": ["query"]
            },
            server_url="http://localhost:3000/mcp"  # Your MCP server
        ))
        
        # GitHub tool (use your GitHub Pro!)
        self.mcp.register_tool(MCPToolDefinition(
            name="github_search_code",
            description="Search GitHub repositories for code",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "language": {"type": "string"}
                },
                "required": ["query"]
            },
            server_url="http://localhost:3000/mcp"
        ))
    
    async def close(self):
        """Cleanup resources"""
        await self.cache.close()
        await self.mcp.close()
    
    async def process_request(self, 
                             user_id: str, 
                             prompt: str,
                             agent_name: Optional[str] = None) -> Dict[str, Any]:
        """Main entry point for requests"""
        
        # 1. Rate limiting
        if not await self.cache.rate_limit_check(user_id):
            return {
                "error": "Rate limit exceeded",
                "retry_after": 60
            }
        
        # 2. Check cache
        cached = await self.cache.get(prompt)
        if cached:
            return {
                "response": cached["response"],
                "cached": True,
                "metadata": cached["metadata"]
            }
        
        # 3. Select agent (or use specified)
        if not agent_name:
            agent_name = self._route_to_agent(prompt)
        
        agent = self.agents.get_agent(agent_name)
        if not agent:
            return {"error": f"Agent '{agent_name}' not found"}
        
        # 4. Get conversation context
        history = await self.cache.get_conversation(user_id, limit=5)
        
        # 5. Determine provider/model
        provider, model = self.agents.get_model_for_agent(agent_name)
        
        # 6. Prepare messages
        messages = [
            {"role": "system", "content": agent.system_prompt}
        ]
        
        # Add recent history
        for msg in reversed(history):
            messages.append({
                "role": msg.get("role", "user"),
                "content": msg.get("content", "")
            })
        
        messages.append({"role": "user", "content": prompt})
        
        # 7. Get MCP tools if needed
        tools = None
        if agent.requires_tools:
            tools = self.mcp.get_tools_for_llm()
        
        # 8. Execute LLM call with retries
        max_retries = 3
        last_error = None
        
        for attempt in range(max_retries):
            try:
                result = await self.llm.complete(
                    provider=provider,
                    model=model,
                    messages=messages,
                    max_tokens=4096,
                    temperature=0.7,
                    tools=tools
                )
                
                # 9. Handle tool calls if any
                if result.get("tool_calls"):
                    result = await self._handle_tool_calls(result, messages, provider, model)
                
                # 10. Cache the result
                await self.cache.set(
                    prompt,
                    result["content"],
                    result["metadata"],
                    ttl=3600
                )
                
                # 11. Store conversation
                await self.cache.store_conversation(user_id, {
                    "role": "user",
                    "content": prompt
                })
                await self.cache.store_conversation(user_id, {
                    "role": "assistant",
                    "content": result["content"]
                })
                
                return {
                    "response": result["content"],
                    "agent": agent_name,
                    "metadata": result["metadata"],
                    "cached": False
                }
                
            except Exception as e:
                last_error = str(e)
                print(f"⚠️  Attempt {attempt + 1} failed: {last_error}")
                
                if attempt < max_retries - 1:
                    # Try fallback provider
                    provider, model = self._get_fallback_provider(provider)
                    await asyncio.sleep(1)
        
        return {
            "error": "All providers failed",
            "details": last_error
        }
    
    def _route_to_agent(self, prompt: str) -> str:
        """Simple keyword-based routing"""
        prompt_lower = prompt.lower()
        
        best_agent = "technical_writer"  # Default
        best_score = 0
        
        for agent in self.agents.get_all_agents():
            score = sum(1 for kw in agent.keywords if kw in prompt_lower)
            if score > best_score:
                best_score = score
                best_agent = agent.name
        
        return best_agent
    
    def _get_fallback_provider(self, current: str) -> tuple[str, str]:
        """Get fallback provider if current fails"""
        fallback_chain = [
            ("groq", "llama-3.1-70b-versatile"),
            ("gemini", "gemini-1.5-flash"),
            ("mistral", "mistral-small-latest"),
            ("huggingface", "meta-llama/Meta-Llama-3-8B-Instruct")
        ]
        
        # Find current and return next
        for i, (provider, model) in enumerate(fallback_chain):
            if provider == current:
                next_idx = (i + 1) % len(fallback_chain)
                return fallback_chain[next_idx]
        
        return fallback_chain[0]
    
    async def _handle_tool_calls(self, 
                                 result: Dict[str, Any], 
                                 messages: List[Dict],
                                 provider: str,
                                 model: str) -> Dict[str, Any]:
        """Execute tool calls and get final response"""
        
        tool_calls = result.get("tool_calls", [])
        if not tool_calls:
            return result
        
        # Execute each tool
        tool_results = []
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)
            
            try:
                tool_result = await self.mcp.call_tool(tool_name, tool_args)
                tool_results.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(tool_result)
                })
            except Exception as e:
                tool_results.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": f"Error: {str(e)}"
                })
        
        # Add assistant message with tool calls
        messages.append({
            "role": "assistant",
            "content": result["content"],
            "tool_calls": tool_calls
        })
        
        # Add tool results
        messages.extend(tool_results)
        
        # Get final response
        final_result = await self.llm.complete(
            provider=provider,
            model=model,
            messages=messages,
            max_tokens=4096,
            temperature=0.7
        )
        
        return final_result
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        cache_stats = await self.cache.get_stats()
        llm_stats = self.llm.get_usage_stats()
        
        return {
            "status": "operational",
            "providers": {
                name: config.enabled
                for name, config in self.config.providers.items()
            },
            "cache": {
                "total_hits": cache_stats.get("exact_hits", 0) + cache_stats.get("semantic_hits", 0),
                "misses": cache_stats.get("misses", 0),
                "hit_rate": round(
                    (cache_stats.get("exact_hits", 0) + cache_stats.get("semantic_hits", 0)) /
                    max(sum(cache_stats.values()), 1) * 100,
                    2
                )
            },
            "llm_usage": llm_stats,
            "agents": {
                "total": len(self.agents.get_all_agents()),
                "available": [agent.name for agent in self.agents.get_all_agents()]
            },
            "mcp_tools": list(self.mcp.tools.keys())
        }


# ============================================================================
# GITHUB COPILOT INTEGRATION (GitHub Pro Feature)
# ============================================================================

class GitHubCopilotAgent:
    """Enhanced code generation using GitHub Copilot"""
    
    def __init__(self, github_token: str):
        self.github_token = github_token
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def initialize(self):
        """Initialize GitHub session"""
        self.session = aiohttp.ClientSession(
            headers={
                "Authorization": f"Bearer {self.github_token}",
                "Accept": "application/vnd.github+json"
            }
        )
    
    async def close(self):
        if self.session:
            await self.session.close()
    
    async def search_code(self, query: str, language: Optional[str] = None) -> List[Dict]:
        """Search GitHub for code examples"""
        params = {"q": query}
        if language:
            params["q"] += f" language:{language}"
        
        try:
            async with self.session.get(
                "https://api.github.com/search/code",
                params=params
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("items", [])[:5]  # Top 5 results
        except Exception as e:
            print(f"GitHub search error: {e}")
        
        return []
    
    async def get_file_content(self, repo: str, path: str) -> Optional[str]:
        """Get content of a specific file"""
        try:
            async with self.session.get(
                f"https://api.github.com/repos/{repo}/contents/{path}"
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    import base64
                    return base64.b64decode(data["content"]).decode()
        except Exception as e:
            print(f"GitHub file fetch error: {e}")
        
        return None


# ============================================================================
# EXAMPLE MCP SERVER (Node.js)
# ============================================================================

# Save this as mcp-server.js and run with: node mcp-server.js

MCP_SERVER_CODE = '''
// MCP Server Example - Install: npm install express body-parser axios
const express = require('express');
const bodyParser = require('body-parser');
const axios = require('axios');

const app = express();
app.use(bodyParser.json());

// MCP JSON-RPC endpoint
app.post('/mcp', async (req, res) => {
    const { jsonrpc, id, method, params } = req.body;
    
    if (jsonrpc !== '2.0') {
        return res.json({
            jsonrpc: '2.0',
            id,
            error: { code: -32600, message: 'Invalid Request' }
        });
    }
    
    try {
        let result;
        
        switch (params.name) {
            case 'web_search':
                result = await handleWebSearch(params.arguments);
                break;
            
            case 'github_search_code':
                result = await handleGitHubSearch(params.arguments);
                break;
            
            default:
                throw new Error(`Unknown tool: ${params.name}`);
        }
        
        res.json({
            jsonrpc: '2.0',
            id,
            result
        });
        
    } catch (error) {
        res.json({
            jsonrpc: '2.0',
            id,
            error: { code: -32000, message: error.message }
        });
    }
});

async function handleWebSearch(args) {
    // Example using DuckDuckGo instant answer API
    const response = await axios.get('https://api.duckduckgo.com/', {
        params: {
            q: args.query,
            format: 'json',
            no_html: 1
        }
    });
    
    return {
        query: args.query,
        results: [
            {
                title: response.data.Heading || 'Search Results',
                snippet: response.data.AbstractText || 'No results',
                url: response.data.AbstractURL || ''
            }
        ]
    };
}

async function handleGitHubSearch(args) {
    const query = `${args.query}${args.language ? ` language:${args.language}` : ''}`;
    
    const response = await axios.get('https://api.github.com/search/code', {
        params: { q: query },
        headers: {
            'Accept': 'application/vnd.github+json',
            'Authorization': `Bearer ${process.env.GITHUB_TOKEN}`
        }
    });
    
    return {
        total_count: response.data.total_count,
        items: response.data.items.slice(0, 5).map(item => ({
            name: item.name,
            path: item.path,
            repository: item.repository.full_name,
            url: item.html_url
        }))
    };
}

const PORT = 3000;
app.listen(PORT, () => {
    console.log(`✓ MCP Server running on http://localhost:${PORT}/mcp`);
});
'''


# ============================================================================
# ADVANCED FEATURES
# ============================================================================

class StreamingOrchestrator(ProductionOrchestrator):
    """Extended orchestrator with streaming support"""
    
    async def stream_request(self, 
                            user_id: str, 
                            prompt: str,
                            agent_name: Optional[str] = None):
        """Stream responses token by token"""
        
        # Same setup as process_request
        if not await self.cache.rate_limit_check(user_id):
            yield {"error": "Rate limit exceeded"}
            return
        
        cached = await self.cache.get(prompt)
        if cached:
            yield {
                "type": "cached",
                "content": cached["response"],
                "metadata": cached["metadata"]
            }
            return
        
        if not agent_name:
            agent_name = self._route_to_agent(prompt)
        
        agent = self.agents.get_agent(agent_name)
        provider, model = self.agents.get_model_for_agent(agent_name)
        
        # Stream status updates
        yield {"type": "status", "message": f"Routing to {agent_name}..."}
        yield {"type": "status", "message": f"Using {provider} ({model})..."}
        
        history = await self.cache.get_conversation(user_id, limit=5)
        
        messages = [
            {"role": "system", "content": agent.system_prompt}
        ]
        
        for msg in reversed(history):
            messages.append({
                "role": msg.get("role", "user"),
                "content": msg.get("content", "")
            })
        
        messages.append({"role": "user", "content": prompt})
        
        # Streaming implementation varies by provider
        # For now, simulate streaming
        yield {"type": "status", "message": "Generating response..."}
        
        try:
            result = await self.llm.complete(
                provider=provider,
                model=model,
                messages=messages,
                max_tokens=4096,
                temperature=0.7
            )
            
            # Simulate token-by-token streaming
            content = result["content"]
            words = content.split()
            
            for i, word in enumerate(words):
                yield {
                    "type": "content",
                    "delta": word + " ",
                    "progress": (i + 1) / len(words)
                }
                await asyncio.sleep(0.05)  # Simulate streaming delay
            
            yield {
                "type": "complete",
                "metadata": result["metadata"]
            }
            
            # Cache the result
            await self.cache.set(prompt, content, result["metadata"])
            
        except Exception as e:
            yield {"type": "error", "message": str(e)}


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

async def example_basic_usage():
    """Example 1: Basic request handling"""
    
    orchestrator = ProductionOrchestrator()
    await orchestrator.initialize()
    
    try:
        # Example request
        result = await orchestrator.process_request(
            user_id="user123",
            prompt="Write a Python function to validate email addresses using regex"
        )
        
        print("\n" + "="*60)
        print("BASIC REQUEST EXAMPLE")
        print("="*60)
        print(f"\nAgent: {result.get('agent')}")
        print(f"Cached: {result.get('cached', False)}")
        print(f"\nResponse:\n{result.get('response', result.get('error'))}")
        
        if 'metadata' in result:
            print(f"\nMetadata:")
            print(f"  Provider: {result['metadata']['provider']}")
            print(f"  Model: {result['metadata']['model']}")
            print(f"  Latency: {result['metadata']['latency_ms']:.2f}ms")
            print(f"  Cost: ${result['metadata']['cost']:.4f}")
        
    finally:
        await orchestrator.close()


async def example_streaming():
    """Example 2: Streaming responses"""
    
    orchestrator = StreamingOrchestrator()
    await orchestrator.initialize()
    
    try:
        print("\n" + "="*60)
        print("STREAMING REQUEST EXAMPLE")
        print("="*60)
        print()
        
        async for chunk in orchestrator.stream_request(
            user_id="user456",
            prompt="Explain how async/await works in Python"
        ):
            if chunk["type"] == "status":
                print(f"[STATUS] {chunk['message']}")
            elif chunk["type"] == "content":
                print(chunk["delta"], end="", flush=True)
            elif chunk["type"] == "complete":
                print("\n\n[COMPLETE]")
                print(f"Provider: {chunk['metadata']['provider']}")
            elif chunk["type"] == "error":
                print(f"\n[ERROR] {chunk['message']}")
        
    finally:
        await orchestrator.close()


async def example_multi_agent():
    """Example 3: Multiple agents working together"""
    
    orchestrator = ProductionOrchestrator()
    await orchestrator.initialize()
    
    try:
        print("\n" + "="*60)
        print("MULTI-AGENT WORKFLOW EXAMPLE")
        print("="*60)
        
        # Step 1: Generate code
        print("\n[Step 1] Generating code...")
        code_result = await orchestrator.process_request(
            user_id="user789",
            prompt="Write a Python function to sort a list using quicksort",
            agent_name="code_generator"
        )
        
        print(f"✓ Code generated by {code_result['metadata']['model']}")
        
        # Step 2: Review the code
        print("\n[Step 2] Reviewing code...")
        review_result = await orchestrator.process_request(
            user_id="user789",
            prompt=f"Review this code for bugs and improvements:\n\n{code_result['response']}",
            agent_name="code_reviewer"
        )
        
        print(f"✓ Review completed by {review_result['metadata']['model']}")
        
        # Step 3: Document it
        print("\n[Step 3] Creating documentation...")
        doc_result = await orchestrator.process_request(
            user_id="user789",
            prompt=f"Create documentation for this code:\n\n{code_result['response']}",
            agent_name="technical_writer"
        )
        
        print(f"✓ Documentation created by {doc_result['metadata']['model']}")
        
        # Final output
        print("\n" + "="*60)
        print("WORKFLOW COMPLETE")
        print("="*60)
        print(f"\nOriginal Code:\n{code_result['response'][:200]}...")
        print(f"\nCode Review:\n{review_result['response'][:200]}...")
        print(f"\nDocumentation:\n{doc_result['response'][:200]}...")
        
        # Show total cost
        total_cost = (
            code_result['metadata']['cost'] +
            review_result['metadata']['cost'] +
            doc_result['metadata']['cost']
        )
        print(f"\nTotal workflow cost: ${total_cost:.4f}")
        
    finally:
        await orchestrator.close()


async def example_system_status():
    """Example 4: System monitoring"""
    
    orchestrator = ProductionOrchestrator()
    await orchestrator.initialize()
    
    try:
        # Make some requests first
        await orchestrator.process_request(
            user_id="test",
            prompt="Hello world"
        )
        
        await orchestrator.process_request(
            user_id="test",
            prompt="Hello world"  # Should hit cache
        )
        
        # Get status
        status = await orchestrator.get_system_status()
        
        print("\n" + "="*60)
        print("SYSTEM STATUS")
        print("="*60)
        print(json.dumps(status, indent=2))
        
    finally:
        await orchestrator.close()


async def example_github_integration():
    """Example 5: Using GitHub Pro features"""
    
    github_token = os.getenv("GITHUB_TOKEN")
    if not github_token:
        print("⚠️  GITHUB_TOKEN not set, skipping example")
        return
    
    github = GitHubCopilotAgent(github_token)
    await github.initialize()
    
    try:
        print("\n" + "="*60)
        print("GITHUB CODE SEARCH EXAMPLE")
        print("="*60)
        
        # Search for code examples
        results = await github.search_code(
            query="async def process_request",
            language="python"
        )
        
        print(f"\nFound {len(results)} examples:")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result['name']}")
            print(f"   Repo: {result['repository']['full_name']}")
            print(f"   Path: {result['path']}")
            print(f"   URL: {result['html_url']}")
        
        # Get actual code from first result
        if results:
            first = results[0]
            content = await github.get_file_content(
                first['repository']['full_name'],
                first['path']
            )
            
            if content:
                print(f"\n\nSample code from {first['name']}:")
                print("-" * 60)
                print(content[:500] + "...")
        
    finally:
        await github.close()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

async def main():
    """Run all examples"""
    
    print("""
╔══════════════════════════════════════════════════════════════╗
║   Production Multi-Agent Platform with Full Integration      ║
║   Mistral + Gemini + Groq + HuggingFace + Redis + MCP       ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Check required environment variables
    required_vars = [
        "MISTRAL_API_KEY",
        "GEMINI_API_KEY", 
        "GROQ_API_KEY",
        "HF_API_KEY"
    ]
    
    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        print(f"⚠️  Missing environment variables: {', '.join(missing)}")
        print("Set them using: export VAR_NAME=your_key\n")
    
    # Run examples
    try:
        await example_basic_usage()
        await asyncio.sleep(1)
        
        await example_streaming()
        await asyncio.sleep(1)
        
        await example_multi_agent()
        await asyncio.sleep(1)
        
        await example_system_status()
        await asyncio.sleep(1)
        
        await example_github_integration()
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
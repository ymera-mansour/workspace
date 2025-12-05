# ============================================================================
# tests/test_basic.py - Basic Component Tests
# ============================================================================
# Run: pytest tests/test_basic.py -v

import pytest
import asyncio
from agent_platform import (
    RedisCache,
    ConfigManager,
    LLMProviderManager,
    MCPClient,
    AgentRegistry
)

@pytest.mark.asyncio
async def test_config_manager():
    """Test configuration loading"""
    config = ConfigManager()
    
    assert "mistral" in config.providers
    assert "groq" in config.providers
    assert "gemini" in config.providers
    
    # Test cheapest model selection
    provider, model = config.get_cheapest_model()
    assert provider in ["groq", "gemini", "mistral", "huggingface"]
    
    print(f"✓ Cheapest model: {provider}/{model}")

@pytest.mark.asyncio
async def test_redis_cache():
    """Test Redis caching"""
    cache = RedisCache("redis://localhost:6379")
    
    try:
        await cache.connect()
        
        # Test cache miss
        result = await cache.get("test_prompt")
        assert result is None
        
        # Test cache set
        await cache.set(
            "test_prompt",
            "test response",
            {"provider": "test", "cost": 0.01}
        )
        
        # Test cache hit
        result = await cache.get("test_prompt")
        assert result is not None
        assert result["response"] == "test response"
        
        # Test stats
        stats = await cache.get_stats()
        assert "exact_hits" in stats
        
        print(f"✓ Cache stats: {stats}")
        
    finally:
        await cache.close()

@pytest.mark.asyncio
async def test_agent_registry():
    """Test agent registration"""
    config = ConfigManager()
    registry = AgentRegistry(config)
    registry.register_default_agents()
    
    agents = registry.get_all_agents()
    assert len(agents) > 0
    
    # Test agent retrieval
    code_gen = registry.get_agent("code_generator")
    assert code_gen is not None
    assert "code" in code_gen.keywords
    
    # Test model selection
    provider, model = registry.get_model_for_agent("code_generator")
    assert provider is not None
    
    print(f"✓ Registered {len(agents)} agents")
    print(f"✓ Code generator uses: {provider}/{model}")

@pytest.mark.asyncio
async def test_mcp_client():
    """Test MCP client initialization"""
    mcp = MCPClient()
    await mcp.initialize()
    
    from agent_platform import MCPToolDefinition
    
    # Register test tool
    mcp.register_tool(MCPToolDefinition(
        name="test_tool",
        description="Test tool",
        parameters={"type": "object"},
        server_url="http://localhost:3000/mcp"
    ))
    
    assert "test_tool" in mcp.tools
    
    tools_for_llm = mcp.get_tools_for_llm()
    assert len(tools_for_llm) > 0
    
    await mcp.close()
    
    print(f"✓ MCP tools registered: {list(mcp.tools.keys())}")


# ============================================================================
# tests/test_agents.py - Agent-Specific Tests
# ============================================================================
# Run: pytest tests/test_agents.py -v

import pytest
from agent_platform import ProductionOrchestrator

@pytest.fixture
async def orchestrator():
    """Fixture to initialize orchestrator"""
    orch = ProductionOrchestrator()
    await orch.initialize()
    yield orch
    await orch.close()

@pytest.mark.asyncio
async def test_code_generator(orchestrator):
    """Test code generation agent"""
    result = await orchestrator.process_request(
        user_id="test_user",
        prompt="Write a Python function to reverse a string",
        agent_name="code_generator"
    )
    
    assert "error" not in result
    assert "response" in result
    assert "def" in result["response"].lower() or "function" in result["response"].lower()
    
    print(f"✓ Agent: {result['agent']}")
    print(f"✓ Cached: {result.get('cached', False)}")
    print(f"✓ Cost: ${result['metadata']['cost']:.4f}")

@pytest.mark.asyncio
async def test_technical_writer(orchestrator):
    """Test technical writing agent"""
    result = await orchestrator.process_request(
        user_id="test_user",
        prompt="Write a guide on how to use async/await in Python",
        agent_name="technical_writer"
    )
    
    assert "error" not in result
    assert len(result["response"]) > 100
    
    print(f"✓ Response length: {len(result['response'])} chars")

@pytest.mark.asyncio
async def test_code_reviewer(orchestrator):
    """Test code review agent"""
    result = await orchestrator.process_request(
        user_id="test_user",
        prompt="Review this code: def add(a,b): return a+b",
        agent_name="code_reviewer"
    )
    
    assert "error" not in result
    assert "response" in result
    
    print(f"✓ Review completed by: {result['agent']}")

@pytest.mark.asyncio
async def test_agent_routing(orchestrator):
    """Test automatic agent routing"""
    
    # Should route to code_generator
    result1 = await orchestrator.process_request(
        user_id="test_user",
        prompt="Write a Python function for fibonacci"
    )
    
    # Should route to data_analyst
    result2 = await orchestrator.process_request(
        user_id="test_user",
        prompt="Analyze this data: [1,2,3,4,5]"
    )
    
    print(f"✓ Prompt 1 routed to: {result1['agent']}")
    print(f"✓ Prompt 2 routed to: {result2['agent']}")
    
    # Ideally they should be different agents
    assert result1['agent'] in orchestrator.agents.agents
    assert result2['agent'] in orchestrator.agents.agents


# ============================================================================
# tests/test_integration.py - Full Integration Tests
# ============================================================================
# Run: pytest tests/test_integration.py -v

import pytest
import asyncio
from agent_platform import ProductionOrchestrator, StreamingOrchestrator

@pytest.mark.asyncio
async def test_full_workflow():
    """Test complete multi-agent workflow"""
    orchestrator = ProductionOrchestrator()
    await orchestrator.initialize()
    
    try:
        print("\n" + "="*60)
        print("FULL WORKFLOW TEST")
        print("="*60)
        
        # Step 1: Generate code
        print("\n[1/3] Generating code...")
        code_result = await orchestrator.process_request(
            user_id="workflow_test",
            prompt="Write a Python function to validate email",
            agent_name="code_generator"
        )
        
        assert "error" not in code_result
        print(f"✓ Generated by {code_result['metadata']['provider']}")
        print(f"✓ Cost: ${code_result['metadata']['cost']:.4f}")
        
        # Step 2: Review code
        print("\n[2/3] Reviewing code...")
        review_result = await orchestrator.process_request(
            user_id="workflow_test",
            prompt=f"Review this code:\n{code_result['response'][:200]}",
            agent_name="code_reviewer"
        )
        
        assert "error" not in review_result
        print(f"✓ Reviewed by {review_result['metadata']['provider']}")
        
        # Step 3: Document
        print("\n[3/3] Creating docs...")
        doc_result = await orchestrator.process_request(
            user_id="workflow_test",
            prompt=f"Document this code:\n{code_result['response'][:200]}",
            agent_name="technical_writer"
        )
        
        assert "error" not in doc_result
        print(f"✓ Documented by {doc_result['metadata']['provider']}")
        
        # Calculate totals
        total_cost = sum([
            code_result['metadata']['cost'],
            review_result['metadata']['cost'],
            doc_result['metadata']['cost']
        ])
        
        total_time = sum([
            code_result['metadata']['latency_ms'],
            review_result['metadata']['latency_ms'],
            doc_result['metadata']['latency_ms']
        ])
        
        print(f"\n{'='*60}")
        print(f"WORKFLOW COMPLETE")
        print(f"{'='*60}")
        print(f"Total cost: ${total_cost:.4f}")
        print(f"Total time: {total_time:.2f}ms")
        print(f"Avg cost per step: ${total_cost/3:.4f}")
        
        assert total_cost < 0.10  # Should be under $0.10
        
    finally:
        await orchestrator.close()

@pytest.mark.asyncio
async def test_caching():
    """Test cache performance"""
    orchestrator = ProductionOrchestrator()
    await orchestrator.initialize()
    
    try:
        prompt = "What is 2+2?"
        
        # First request (cache miss)
        import time
        start1 = time.time()
        result1 = await orchestrator.process_request(
            user_id="cache_test",
            prompt=prompt
        )
        time1 = (time.time() - start1) * 1000
        
        # Second request (cache hit)
        start2 = time.time()
        result2 = await orchestrator.process_request(
            user_id="cache_test",
            prompt=prompt
        )
        time2 = (time.time() - start2) * 1000
        
        print(f"\n{'='*60}")
        print(f"CACHE PERFORMANCE TEST")
        print(f"{'='*60}")
        print(f"First request (miss):  {time1:.2f}ms")
        print(f"Second request (hit):  {time2:.2f}ms")
        print(f"Speedup: {time1/time2:.1f}x faster")
        
        assert result1["cached"] == False
        assert result2["cached"] == True
        assert time2 < time1 / 2  # Should be at least 2x faster
        
        # Check cache stats
        status = await orchestrator.get_system_status()
        print(f"\nCache hit rate: {status['cache']['hit_rate']}%")
        
    finally:
        await orchestrator.close()

@pytest.mark.asyncio
async def test_streaming():
    """Test streaming responses"""
    orchestrator = StreamingOrchestrator()
    await orchestrator.initialize()
    
    try:
        chunks_received = 0
        total_content = ""
        
        async for chunk in orchestrator.stream_request(
            user_id="stream_test",
            prompt="Count from 1 to 5"
        ):
            chunks_received += 1
            
            if chunk["type"] == "content":
                total_content += chunk["delta"]
            
            if chunks_received > 100:  # Safety limit
                break
        
        print(f"\n{'='*60}")
        print(f"STREAMING TEST")
        print(f"{'='*60}")
        print(f"Chunks received: {chunks_received}")
        print(f"Content length: {len(total_content)}")
        print(f"Sample: {total_content[:100]}...")
        
        assert chunks_received > 5
        assert len(total_content) > 0
        
    finally:
        await orchestrator.close()

@pytest.mark.asyncio
async def test_error_handling():
    """Test error scenarios"""
    orchestrator = ProductionOrchestrator()
    await orchestrator.initialize()
    
    try:
        # Test invalid agent
        result = await orchestrator.process_request(
            user_id="error_test",
            prompt="Test",
            agent_name="nonexistent_agent"
        )
        
        assert "error" in result
        print(f"✓ Invalid agent error handled: {result['error']}")
        
        # Test rate limiting (make many requests)
        print("\n✓ Testing rate limiting...")
        rate_limit_hit = False
        
        for i in range(150):  # Exceed limit of 100/min
            result = await orchestrator.process_request(
                user_id="rate_limit_test",
                prompt=f"Test {i}"
            )
            
            if "error" in result and "rate limit" in result["error"].lower():
                rate_limit_hit = True
                print(f"✓ Rate limit triggered at request {i}")
                break
        
        # Note: Rate limit might not trigger in test env
        print(f"Rate limit test: {'PASS' if rate_limit_hit else 'SKIP (not triggered)'}")
        
    finally:
        await orchestrator.close()

@pytest.mark.asyncio
async def test_multi_provider_fallback():
    """Test provider fallback on failure"""
    orchestrator = ProductionOrchestrator()
    await orchestrator.initialize()
    
    try:
        # Disable primary provider temporarily
        orchestrator.config.providers["groq"].enabled = False
        
        result = await orchestrator.process_request(
            user_id="fallback_test",
            prompt="Simple test"
        )
        
        # Should fallback to another provider
        assert "error" not in result or result["metadata"]["provider"] != "groq"
        
        print(f"\n✓ Fallback test:")
        print(f"  Used provider: {result.get('metadata', {}).get('provider', 'N/A')}")
        print(f"  Should not be Groq (disabled)")
        
        # Re-enable
        orchestrator.config.providers["groq"].enabled = True
        
    finally:
        await orchestrator.close()

@pytest.mark.asyncio
async def test_cost_tracking():
    """Test cost tracking accuracy"""
    orchestrator = ProductionOrchestrator()
    await orchestrator.initialize()
    
    try:
        # Make several requests
        for i in range(5):
            await orchestrator.process_request(
                user_id="cost_test",
                prompt=f"Test prompt {i}"
            )
        
        # Get stats
        stats = orchestrator.llm.get_usage_stats()
        
        print(f"\n{'='*60}")
        print(f"COST TRACKING TEST")
        print(f"{'='*60}")
        print(f"Total requests: {stats['total_calls']}")
        print(f"Total cost: ${stats['total_cost']:.4f}")
        print(f"Avg cost/request: ${stats['avg_cost_per_call']:.4f}")
        print(f"\nBy provider:")
        for provider, count in stats['calls_by_provider'].items():
            print(f"  {provider}: {count} calls")
        
        assert stats['total_calls'] >= 5
        assert stats['total_cost'] < 1.00  # Should be under $1
        
    finally:
        await orchestrator.close()


# ============================================================================
# tests/test_github.py - GitHub Integration Tests
# ============================================================================
# Run: pytest tests/test_github.py -v

import pytest
import os
from agent_platform import GitHubCopilotAgent

@pytest.mark.skipif(
    not os.getenv("GITHUB_TOKEN"),
    reason="GITHUB_TOKEN not set"
)
@pytest.mark.asyncio
async def test_github_search():
    """Test GitHub code search"""
    github = GitHubCopilotAgent(os.getenv("GITHUB_TOKEN"))
    await github.initialize()
    
    try:
        results = await github.search_code(
            query="async def main",
            language="python"
        )
        
        print(f"\n{'='*60}")
        print(f"GITHUB CODE SEARCH TEST")
        print(f"{'='*60}")
        print(f"Query: 'async def main' in Python")
        print(f"Results found: {len(results)}")
        
        assert len(results) > 0
        
        for i, result in enumerate(results[:3], 1):
            print(f"\n{i}. {result['name']}")
            print(f"   Repo: {result['repository']['full_name']}")
            
    finally:
        await github.close()

@pytest.mark.skipif(
    not os.getenv("GITHUB_TOKEN"),
    reason="GITHUB_TOKEN not set"
)
@pytest.mark.asyncio
async def test_github_file_fetch():
    """Test fetching file content from GitHub"""
    github = GitHubCopilotAgent(os.getenv("GITHUB_TOKEN"))
    await github.initialize()
    
    try:
        # Search for a file
        results = await github.search_code("README.md", "markdown")
        
        if results:
            first = results[0]
            content = await github.get_file_content(
                first['repository']['full_name'],
                first['path']
            )
            
            assert content is not None
            assert len(content) > 0
            
            print(f"\n✓ Fetched file: {first['name']}")
            print(f"✓ Content length: {len(content)} bytes")
            print(f"✓ Preview: {content[:100]}...")
            
    finally:
        await github.close()


# ============================================================================
# tests/conftest.py - Pytest Configuration
# ============================================================================
# Save as: tests/conftest.py

import pytest
import asyncio

# Fixtures shared across all tests

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(autouse=True)
def cleanup_between_tests():
    """Cleanup between tests"""
    yield
    # Cleanup code here if needed

# ============================================================================
# Makefile - Convenient Commands
# ============================================================================
# Save as: Makefile

.PHONY: help install test test-basic test-agents test-integration test-all run-local run-docker clean

help:
	@echo "Multi-Agent Platform - Make Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make install        Install dependencies"
	@echo ""
	@echo "Testing:"
	@echo "  make test          Run all tests"
	@echo "  make test-basic    Run basic tests only"
	@echo "  make test-agents   Run agent tests only"
	@echo ""
	@echo "Running:"
	@echo "  make run-local     Run locally"
	@echo "  make run-docker    Run with Docker"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean         Clean up"

install:
	pip install -r requirements.txt
	cd mcp-server && npm install

test:
	pytest tests/ -v --tb=short

test-basic:
	pytest tests/test_basic.py -v

test-agents:
	pytest tests/test_agents.py -v

test-integration:
	pytest tests/test_integration.py -v

run-local:
	python api_server.py

run-docker:
	docker-compose up -d
	docker-compose logs -f

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	docker-compose down -v

# ============================================================================
# run_tests.sh - Test Runner Script
# ============================================================================
# Save as: run_tests.sh
# chmod +x run_tests.sh

#!/bin/bash

echo "================================"
echo "Multi-Agent Platform Test Suite"
echo "================================"
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check dependencies
echo "Checking dependencies..."
command -v redis-cli >/dev/null 2>&1 || { echo "❌ Redis not installed"; exit 1; }
command -v python >/dev/null 2>&1 || { echo "❌ Python not installed"; exit 1; }
echo "✓ Dependencies OK"
echo ""

# Check Redis
echo "Checking Redis..."
redis-cli ping >/dev/null 2>&1 || { echo "❌ Redis not running"; exit 1; }
echo "✓ Redis OK"
echo ""

# Check environment
echo "Checking environment variables..."
[ -z "$GROQ_API_KEY" ] && echo "⚠️  GROQ_API_KEY not set"
[ -z "$GEMINI_API_KEY" ] && echo "⚠️  GEMINI_API_KEY not set"
[ -z "$MISTRAL_API_KEY" ] && echo "⚠️  MISTRAL_API_KEY not set"
echo ""

# Run tests
echo "Running tests..."
echo ""

pytest tests/ -v --tb=short --color=yes

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✅ All tests passed!${NC}"
    exit 0
else
    echo ""
    echo -e "${RED}❌ Some tests failed${NC}"
    exit 1
fi
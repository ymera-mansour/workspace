# ============================================================================
# ADVANCED FEATURES MODULE
# Enhances the base platform with production-grade capabilities
# ============================================================================

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from collections import defaultdict
import hashlib
import numpy as np
from enum import Enum

# ============================================================================
# 1. INTELLIGENT LOAD BALANCER
# ============================================================================

class ProviderHealth(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    DOWN = "down"

@dataclass
class ProviderMetrics:
    success_count: int = 0
    failure_count: int = 0
    total_latency_ms: float = 0.0
    last_success: Optional[datetime] = None
    last_failure: Optional[datetime] = None
    current_health: ProviderHealth = ProviderHealth.HEALTHY
    
    @property
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 1.0
    
    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / max(self.success_count, 1)

class IntelligentLoadBalancer:
    """
    Smart load balancer that routes requests based on:
    - Provider health and success rates
    - Current latency
    - Cost efficiency
    - Rate limit awareness
    """
    
    def __init__(self, config_manager):
        self.config = config_manager
        self.metrics: Dict[str, ProviderMetrics] = defaultdict(ProviderMetrics)
        self.rate_limits: Dict[str, List[float]] = defaultdict(list)
        self.health_check_interval = 60  # seconds
        self.last_health_check = datetime.now()
        
    def select_provider(self, 
                       required_capability: str = "general",
                       prefer_cost: bool = True) -> tuple[str, str]:
        """
        Intelligently select best provider based on multiple factors
        """
        candidates = []
        now = datetime.now()
        
        for provider_name, provider_config in self.config.providers.items():
            if not provider_config.enabled:
                continue
            
            metrics = self.metrics[provider_name]
            
            # Check if provider is rate limited
            if self._is_rate_limited(provider_name, provider_config.rate_limit_rpm):
                continue
            
            # Skip if provider is down
            if metrics.current_health == ProviderHealth.DOWN:
                # Check if enough time has passed for recovery
                if metrics.last_failure:
                    time_since_failure = (now - metrics.last_failure).seconds
                    if time_since_failure < 60:  # Wait 60s before retry
                        continue
            
            # Calculate score based on multiple factors
            score = self._calculate_provider_score(
                provider_name,
                metrics,
                provider_config,
                prefer_cost
            )
            
            candidates.append((provider_name, score, provider_config.models[0]))
        
        if not candidates:
            # Fallback: return any available provider
            for name, config in self.config.providers.items():
                if config.enabled:
                    return (name, config.models[0])
            raise Exception("No providers available")
        
        # Sort by score (higher is better)
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        return (candidates[0][0], candidates[0][2])
    
    def _calculate_provider_score(self, 
                                  provider_name: str,
                                  metrics: ProviderMetrics,
                                  config,
                                  prefer_cost: bool) -> float:
        """
        Score = (success_rate * 0.4) + (latency_score * 0.3) + (cost_score * 0.3)
        """
        # Success rate component (0-1)
        success_score = metrics.success_rate
        
        # Latency component (normalize to 0-1, lower is better)
        avg_latency = metrics.avg_latency_ms
        latency_score = max(0, 1 - (avg_latency / 5000))  # 5s = 0 score
        
        # Cost component (0-1, free = 1.0)
        if config.cost_per_1k_tokens == 0:
            cost_score = 1.0
        else:
            # Normalize cost (assuming max $0.01 per 1k tokens)
            cost_score = max(0, 1 - (config.cost_per_1k_tokens / 0.01))
        
        # Weighted combination
        if prefer_cost:
            return (success_score * 0.3 + latency_score * 0.2 + cost_score * 0.5)
        else:
            return (success_score * 0.4 + latency_score * 0.4 + cost_score * 0.2)
    
    def _is_rate_limited(self, provider_name: str, limit_rpm: int) -> bool:
        """Check if provider is currently rate limited"""
        now = time.time()
        
        # Clean old requests (older than 1 minute)
        self.rate_limits[provider_name] = [
            ts for ts in self.rate_limits[provider_name]
            if now - ts < 60
        ]
        
        return len(self.rate_limits[provider_name]) >= limit_rpm
    
    def record_request(self, provider_name: str, 
                      success: bool, 
                      latency_ms: float):
        """Record request outcome for learning"""
        metrics = self.metrics[provider_name]
        now = datetime.now()
        
        if success:
            metrics.success_count += 1
            metrics.total_latency_ms += latency_ms
            metrics.last_success = now
            
            # Recover health if was degraded
            if metrics.current_health == ProviderHealth.DEGRADED:
                if metrics.success_rate > 0.9:
                    metrics.current_health = ProviderHealth.HEALTHY
        else:
            metrics.failure_count += 1
            metrics.last_failure = now
            
            # Degrade health based on failure rate
            if metrics.success_rate < 0.5:
                metrics.current_health = ProviderHealth.DOWN
            elif metrics.success_rate < 0.8:
                metrics.current_health = ProviderHealth.DEGRADED
        
        # Track rate limit
        self.rate_limits[provider_name].append(time.time())
    
    def get_health_report(self) -> Dict[str, Any]:
        """Generate health report for all providers"""
        report = {}
        
        for provider_name, metrics in self.metrics.items():
            report[provider_name] = {
                "health": metrics.current_health.value,
                "success_rate": round(metrics.success_rate * 100, 2),
                "avg_latency_ms": round(metrics.avg_latency_ms, 2),
                "total_requests": metrics.success_count + metrics.failure_count,
                "last_success": metrics.last_success.isoformat() if metrics.last_success else None
            }
        
        return report


# ============================================================================
# 2. SMART PROMPT OPTIMIZER (DSPy-inspired)
# ============================================================================

@dataclass
class PromptTemplate:
    name: str
    template: str
    success_count: int = 0
    failure_count: int = 0
    avg_quality_score: float = 0.0
    
    @property
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.5

class PromptOptimizer:
    """
    Automatically optimizes prompts based on success rates
    Similar to DSPy but simpler
    """
    
    def __init__(self):
        self.templates: Dict[str, List[PromptTemplate]] = defaultdict(list)
        self.load_default_templates()
        
    def load_default_templates(self):
        """Load default prompt templates for different tasks"""
        
        # Code generation templates
        self.templates["code_generation"] = [
            PromptTemplate(
                name="detailed_step_by_step",
                template="""You are an expert programmer. Generate {language} code for the following task:

Task: {task}

Requirements:
- Write clean, efficient code
- Include error handling
- Add helpful comments
- Follow best practices

Step by step:
1. Understand the requirements
2. Design the solution
3. Implement the code
4. Add error handling

Code:"""
            ),
            PromptTemplate(
                name="concise_direct",
                template="""Generate {language} code for: {task}

Requirements: Clean, efficient, well-commented.

Code:"""
            ),
            PromptTemplate(
                name="test_driven",
                template="""Create {language} code for: {task}

Include:
1. Main implementation
2. Example usage
3. Test cases

Code:"""
            )
        ]
        
        # Data analysis templates
        self.templates["data_analysis"] = [
            PromptTemplate(
                name="structured_analysis",
                template="""Analyze the following data:

{data}

Provide:
1. Summary statistics
2. Key patterns
3. Insights
4. Recommendations

Analysis:"""
            ),
            PromptTemplate(
                name="quick_insights",
                template="""Data: {data}

Quick analysis - what are the top 3 insights?"""
            )
        ]
    
    def get_best_prompt(self, category: str, **kwargs) -> str:
        """
        Select best performing prompt template for category
        """
        templates = self.templates.get(category, [])
        
        if not templates:
            # Fallback to basic template
            return kwargs.get("fallback_prompt", "")
        
        # Select template based on success rate and quality
        best_template = max(
            templates,
            key=lambda t: t.success_rate * 0.6 + t.avg_quality_score * 0.4
        )
        
        # Format template with provided kwargs
        try:
            return best_template.template.format(**kwargs)
        except KeyError as e:
            # Missing required parameter, use first template
            return templates[0].template.format(**kwargs)
    
    def record_outcome(self, 
                      category: str, 
                      template_name: str,
                      success: bool,
                      quality_score: float = 0.5):
        """Record outcome to improve future selections"""
        
        templates = self.templates.get(category, [])
        
        for template in templates:
            if template.name == template_name:
                if success:
                    template.success_count += 1
                else:
                    template.failure_count += 1
                
                # Update average quality score (exponential moving average)
                template.avg_quality_score = (
                    template.avg_quality_score * 0.8 + quality_score * 0.2
                )
                break


# ============================================================================
# 3. CONTEXT-AWARE CONVERSATION MANAGER
# ============================================================================

@dataclass
class ConversationContext:
    user_id: str
    messages: List[Dict[str, str]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    active_topics: List[str] = field(default_factory=list)
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    total_tokens_used: int = 0
    
    def add_message(self, role: str, content: str, tokens: int = 0):
        """Add message to conversation"""
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "tokens": tokens
        })
        self.total_tokens_used += tokens
        
        # Keep only last 20 messages to manage context length
        if len(self.messages) > 20:
            self.messages = self.messages[-20:]
    
    def get_recent_context(self, max_tokens: int = 4000) -> List[Dict[str, str]]:
        """Get recent messages within token budget"""
        result = []
        token_count = 0
        
        for msg in reversed(self.messages):
            msg_tokens = msg.get("tokens", 100)  # Estimate if not tracked
            if token_count + msg_tokens > max_tokens:
                break
            
            result.insert(0, {
                "role": msg["role"],
                "content": msg["content"]
            })
            token_count += msg_tokens
        
        return result
    
    def extract_topics(self) -> List[str]:
        """Extract main topics from conversation"""
        # Simple keyword extraction (in production, use NLP)
        keywords = set()
        
        for msg in self.messages[-5:]:  # Last 5 messages
            content = msg["content"].lower()
            words = content.split()
            
            # Extract potential topics (simple heuristic)
            for word in words:
                if len(word) > 5 and word.isalpha():
                    keywords.add(word)
        
        return list(keywords)[:5]

class ConversationManager:
    """
    Manages conversation state and context across multiple interactions
    """
    
    def __init__(self, redis_cache):
        self.cache = redis_cache
        self.active_conversations: Dict[str, ConversationContext] = {}
        
    async def get_context(self, user_id: str) -> ConversationContext:
        """Get or create conversation context"""
        
        if user_id in self.active_conversations:
            return self.active_conversations[user_id]
        
        # Try to load from Redis
        cached = await self.cache.get_conversation(user_id, limit=20)
        
        context = ConversationContext(user_id=user_id)
        
        if cached:
            context.messages = cached
            context.active_topics = context.extract_topics()
        
        self.active_conversations[user_id] = context
        return context
    
    async def add_interaction(self, 
                            user_id: str,
                            user_message: str,
                            assistant_message: str,
                            tokens_used: int = 0):
        """Add interaction to conversation"""
        
        context = await self.get_context(user_id)
        
        context.add_message("user", user_message, tokens_used // 2)
        context.add_message("assistant", assistant_message, tokens_used // 2)
        
        # Update topics
        context.active_topics = context.extract_topics()
        
        # Persist to Redis
        await self.cache.store_conversation(user_id, {
            "role": "user",
            "content": user_message
        })
        await self.cache.store_conversation(user_id, {
            "role": "assistant",
            "content": assistant_message
        })
    
    def get_context_summary(self, user_id: str) -> Dict[str, Any]:
        """Get summary of conversation context"""
        
        context = self.active_conversations.get(user_id)
        
        if not context:
            return {"status": "no_context"}
        
        return {
            "message_count": len(context.messages),
            "total_tokens": context.total_tokens_used,
            "active_topics": context.active_topics,
            "last_interaction": context.messages[-1]["timestamp"] if context.messages else None
        }


# ============================================================================
# 4. AUTOMATIC QUALITY EVALUATOR
# ============================================================================

class ResponseQualityEvaluator:
    """
    Automatically evaluates response quality to improve agent selection
    """
    
    def __init__(self):
        self.quality_history: Dict[str, List[float]] = defaultdict(list)
        
    def evaluate_response(self, 
                         prompt: str,
                         response: str,
                         agent_name: str) -> float:
        """
        Evaluate response quality (0.0 to 1.0)
        Uses simple heuristics - in production, use LLM-as-judge
        """
        score = 0.0
        
        # Length check (not too short, not too long)
        response_length = len(response)
        if response_length < 50:
            score += 0.2  # Too short
        elif response_length < 200:
            score += 0.6  # Decent length
        elif response_length < 2000:
            score += 1.0  # Good length
        else:
            score += 0.7  # Maybe too long
        
        # Check for code markers if prompt asks for code
        if any(word in prompt.lower() for word in ["code", "function", "program"]):
            if "```" in response or "def " in response or "function" in response:
                score += 1.0
            else:
                score += 0.3
        
        # Check for structure (paragraphs, lists)
        if "\n\n" in response or "\n- " in response or "\n1." in response:
            score += 0.5
        
        # Check for errors/apologies (negative signal)
        if "sorry" in response.lower() or "cannot" in response.lower():
            score -= 0.3
        
        # Normalize to 0-1
        final_score = max(0.0, min(1.0, score / 3.0))
        
        # Track history
        self.quality_history[agent_name].append(final_score)
        
        return final_score
    
    def get_agent_quality_stats(self, agent_name: str) -> Dict[str, float]:
        """Get quality statistics for an agent"""
        
        scores = self.quality_history.get(agent_name, [])
        
        if not scores:
            return {"avg": 0.5, "min": 0.0, "max": 1.0, "count": 0}
        
        return {
            "avg": sum(scores) / len(scores),
            "min": min(scores),
            "max": max(scores),
            "count": len(scores),
            "recent_avg": sum(scores[-10:]) / min(len(scores), 10)
        }


# ============================================================================
# 5. BATCH PROCESSING SYSTEM
# ============================================================================

@dataclass
class BatchRequest:
    id: str
    user_id: str
    prompts: List[str]
    agent_name: Optional[str] = None
    priority: int = 5  # 1=highest, 10=lowest
    created_at: datetime = field(default_factory=datetime.now)
    status: str = "pending"
    results: List[Any] = field(default_factory=list)
    
class BatchProcessor:
    """
    Process multiple requests efficiently in batches
    """
    
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.queue: List[BatchRequest] = []
        self.processing = False
        
    async def submit_batch(self, 
                          user_id: str,
                          prompts: List[str],
                          agent_name: Optional[str] = None,
                          priority: int = 5) -> str:
        """Submit batch request"""
        
        batch_id = hashlib.md5(
            f"{user_id}{time.time()}".encode()
        ).hexdigest()[:16]
        
        batch = BatchRequest(
            id=batch_id,
            user_id=user_id,
            prompts=prompts,
            agent_name=agent_name,
            priority=priority
        )
        
        self.queue.append(batch)
        
        # Sort queue by priority
        self.queue.sort(key=lambda x: x.priority)
        
        # Start processing if not already running
        if not self.processing:
            asyncio.create_task(self._process_queue())
        
        return batch_id
    
    async def _process_queue(self):
        """Process queued batch requests"""
        self.processing = True
        
        while self.queue:
            batch = self.queue.pop(0)
            batch.status = "processing"
            
            # Process prompts in parallel (groups of 5)
            batch_size = 5
            
            for i in range(0, len(batch.prompts), batch_size):
                chunk = batch.prompts[i:i + batch_size]
                
                tasks = [
                    self.orchestrator.process_request(
                        user_id=batch.user_id,
                        prompt=prompt,
                        agent_name=batch.agent_name
                    )
                    for prompt in chunk
                ]
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                batch.results.extend(results)
            
            batch.status = "completed"
        
        self.processing = False
    
    def get_batch_status(self, batch_id: str) -> Optional[Dict[str, Any]]:
        """Get status of batch request"""
        
        for batch in self.queue:
            if batch.id == batch_id:
                return {
                    "id": batch.id,
                    "status": batch.status,
                    "total": len(batch.prompts),
                    "completed": len(batch.results),
                    "progress": len(batch.results) / len(batch.prompts) * 100
                }
        
        return None


# ============================================================================
# 6. ADVANCED CACHING STRATEGIES
# ============================================================================

class SmartCache:
    """
    Advanced caching with predictive pre-warming
    """
    
    def __init__(self, redis_cache):
        self.cache = redis_cache
        self.access_patterns: Dict[str, List[float]] = defaultdict(list)
        self.prediction_accuracy = 0.0
        
    async def get_with_prediction(self, key: str) -> Optional[Any]:
        """Get from cache and predict next likely queries"""
        
        result = await self.cache.get(key)
        
        # Record access time
        self.access_patterns[key].append(time.time())
        
        # Predict next queries
        if len(self.access_patterns[key]) > 3:
            await self._predict_and_prewarm(key)
        
        return result
    
    async def _predict_and_prewarm(self, recent_key: str):
        """Predict likely next queries and pre-warm cache"""
        
        # Simple pattern: if user asks about "Python lists",
        # they might ask about "Python dictionaries" next
        
        # This is a placeholder - in production, use ML model
        related_topics = self._find_related_topics(recent_key)
        
        for topic in related_topics[:3]:
            # Check if already cached
            cached = await self.cache.get(topic)
            if not cached:
                # Could pre-fetch, but skip for now to avoid cost
                pass
    
    def _find_related_topics(self, key: str) -> List[str]:
        """Find related topics (placeholder)"""
        # In production, use embeddings or knowledge graph
        return []


# ============================================================================
# 7. INTEGRATION: ENHANCED ORCHESTRATOR
# ============================================================================

class EnhancedOrchestrator:
    """
    Enhanced orchestrator with all advanced features
    """
    
    def __init__(self, base_orchestrator):
        self.base = base_orchestrator
        
        # Advanced components
        self.load_balancer = IntelligentLoadBalancer(self.base.config)
        self.prompt_optimizer = PromptOptimizer()
        self.conversation_manager = ConversationManager(self.base.cache)
        self.quality_evaluator = ResponseQualityEvaluator()
        self.batch_processor = BatchProcessor(self.base)
        self.smart_cache = SmartCache(self.base.cache)
        
    async def process_request_enhanced(self,
                                      user_id: str,
                                      prompt: str,
                                      agent_name: Optional[str] = None,
                                      optimize_prompt: bool = True,
                                      use_smart_routing: bool = True) -> Dict[str, Any]:
        """
        Enhanced request processing with all optimizations
        """
        
        # 1. Get conversation context
        context = await self.conversation_manager.get_context(user_id)
        
        # 2. Optimize prompt if requested
        if optimize_prompt and agent_name:
            agent = self.base.agents.get_agent(agent_name)
            if agent:
                category = self._get_prompt_category(agent_name)
                optimized = self.prompt_optimizer.get_best_prompt(
                    category,
                    task=prompt,
                    language="Python",
                    fallback_prompt=prompt
                )
                prompt = optimized if optimized else prompt
        
        # 3. Smart provider selection
        if use_smart_routing:
            provider, model = self.load_balancer.select_provider(
                prefer_cost=True
            )
            # Override the base orchestrator's provider selection
            # (This would require modifying the base orchestrator)
        
        # 4. Process with base orchestrator
        start_time = time.time()
        
        try:
            result = await self.base.process_request(
                user_id=user_id,
                prompt=prompt,
                agent_name=agent_name
            )
            
            # Record success
            if "error" not in result:
                latency_ms = (time.time() - start_time) * 1000
                
                provider = result.get("metadata", {}).get("provider", "unknown")
                self.load_balancer.record_request(provider, True, latency_ms)
                
                # Evaluate quality
                quality = self.quality_evaluator.evaluate_response(
                    prompt,
                    result.get("response", ""),
                    result.get("agent", "")
                )
                
                # Update conversation
                await self.conversation_manager.add_interaction(
                    user_id,
                    prompt,
                    result.get("response", ""),
                    result.get("metadata", {}).get("usage", {}).get("total_tokens", 0)
                )
                
                # Add quality score to result
                result["quality_score"] = quality
                
                return result
            else:
                # Record failure
                provider = result.get("metadata", {}).get("provider", "unknown")
                self.load_balancer.record_request(provider, False, 0)
                return result
                
        except Exception as e:
            return {"error": str(e)}
    
    def _get_prompt_category(self, agent_name: str) -> str:
        """Map agent to prompt category"""
        mapping = {
            "code_generator": "code_generation",
            "code_reviewer": "code_generation",
            "data_analyst": "data_analysis",
            "web_researcher": "research"
        }
        return mapping.get(agent_name, "general")
    
    async def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        
        base_stats = await self.base.get_system_status()
        
        return {
            **base_stats,
            "load_balancer": self.load_balancer.get_health_report(),
            "batch_processing": {
                "queue_size": len(self.batch_processor.queue),
                "processing": self.batch_processor.processing
            },
            "conversation_stats": {
                "active_users": len(self.conversation_manager.active_conversations)
            }
        }


# ============================================================================
# 8. USAGE EXAMPLE
# ============================================================================

async def demo_advanced_features():
    """Demonstrate advanced features"""
    
    from agent_platform import ProductionOrchestrator
    
    # Initialize base orchestrator
    base_orch = ProductionOrchestrator()
    await base_orch.initialize()
    
    # Wrap with enhanced orchestrator
    enhanced = EnhancedOrchestrator(base_orch)
    
    print("\n" + "="*60)
    print("ADVANCED FEATURES DEMO")
    print("="*60)
    
    # 1. Test smart routing
    print("\n[1] Testing intelligent load balancing...")
    result1 = await enhanced.process_request_enhanced(
        user_id="demo_user",
        prompt="Write a Python function for binary search",
        use_smart_routing=True
    )
    print(f"✓ Provider selected: {result1.get('metadata', {}).get('provider')}")
    print(f"✓ Quality score: {result1.get('quality_score', 0):.2f}")
    
    # 2. Test batch processing
    print("\n[2] Testing batch processing...")
    batch_id = await enhanced.batch_processor.submit_batch(
        user_id="demo_user",
        prompts=[
            "What is Python?",
            "What is JavaScript?",
            "What is Go?"
        ]
    )
    print(f"✓ Batch submitted: {batch_id}")
    
    # Wait for processing
    await asyncio.sleep(5)
    
    status = enhanced.batch_processor.get_batch_status(batch_id)
    if status:
        print(f"✓ Batch progress: {status['progress']:.1f}%")
    
    # 3. Get comprehensive stats
    print("\n[3] System statistics...")
    stats = await enhanced.get_comprehensive_stats()
    print(json.dumps(stats, indent=2, default=str))
    
    await base_orch.close()

if __name__ == "__main__":
    asyncio.run(demo_advanced_features())
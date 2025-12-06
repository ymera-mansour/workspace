"""
Gemini Optimization Implementation
Practical code examples for optimizing Gemini and Google products usage
"""

import asyncio
import os
import hashlib
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# UPDATED MODEL CONFIGURATION
# ============================================================================

@dataclass
class GeminiModelConfig:
    """Enhanced configuration for Gemini models"""
    model_id: str
    display_name: str
    rpm_limit: int  # Requests per minute
    rpd_limit: int  # Requests per day
    max_context_tokens: int
    cost_per_1k_tokens: float
    speed_rating: int  # 1-10
    quality_rating: int  # 1-10
    best_for: List[str]

# Updated model configurations with Gemini 2.0
GEMINI_MODELS = {
    "gemini-2.0-flash-exp": GeminiModelConfig(
        model_id="gemini-2.0-flash-exp",
        display_name="Gemini 2.0 Flash (Experimental)",
        rpm_limit=10,
        rpd_limit=1500,
        max_context_tokens=1_000_000,
        cost_per_1k_tokens=0.0,  # Free tier
        speed_rating=10,
        quality_rating=9,
        best_for=["fast_tasks", "general", "multimodal", "streaming"]
    ),
    "gemini-1.5-flash": GeminiModelConfig(
        model_id="gemini-1.5-flash",
        display_name="Gemini 1.5 Flash",
        rpm_limit=15,
        rpd_limit=1500,
        max_context_tokens=1_000_000,
        cost_per_1k_tokens=0.0,  # Free tier
        speed_rating=9,
        quality_rating=8,
        best_for=["standard_tasks", "documentation", "code_generation"]
    ),
    "gemini-1.5-pro": GeminiModelConfig(
        model_id="gemini-1.5-pro",
        display_name="Gemini 1.5 Pro",
        rpm_limit=2,
        rpd_limit=50,
        max_context_tokens=2_000_000,
        cost_per_1k_tokens=0.0,  # Free tier
        speed_rating=7,
        quality_rating=10,
        best_for=["complex_reasoning", "large_context", "critical_tasks"]
    ),
    "gemini-1.5-flash-8b": GeminiModelConfig(
        model_id="gemini-1.5-flash-8b",
        display_name="Gemini 1.5 Flash 8B",
        rpm_limit=15,
        rpd_limit=4000,
        max_context_tokens=1_000_000,
        cost_per_1k_tokens=0.0,  # Free tier
        speed_rating=10,
        quality_rating=7,
        best_for=["bulk_operations", "simple_tasks", "high_throughput"]
    )
}

# ============================================================================
# INTELLIGENT ROUTING
# ============================================================================

class GeminiRouter:
    """Smart routing based on task complexity and quotas"""
    
    # Routing rules for different agent types
    ROUTING_RULES = {
        "coding_agent": {
            "simple": "gemini-2.0-flash-exp",
            "moderate": "gemini-1.5-flash",
            "complex": "gemini-1.5-pro"
        },
        "database_agent": {
            "simple": "gemini-1.5-flash",
            "moderate": "gemini-1.5-flash",
            "complex": "gemini-1.5-pro"
        },
        "analysis_agent": {
            "simple": "gemini-1.5-flash-8b",
            "moderate": "gemini-2.0-flash-exp",
            "complex": "gemini-1.5-pro"
        },
        "documentation_agent": {
            "simple": "gemini-2.0-flash-exp",
            "moderate": "gemini-1.5-flash",
            "complex": "gemini-1.5-pro"
        }
    }
    
    def __init__(self, quota_manager):
        self.quota_manager = quota_manager
    
    def detect_complexity(self, task_description: str) -> str:
        """Detect task complexity from description"""
        task_lower = task_description.lower()
        
        # Complex indicators
        complex_keywords = [
            "architecture", "refactor", "comprehensive", "entire",
            "multiple files", "codebase", "complex", "advanced",
            "optimize", "design", "system", "integration"
        ]
        if any(word in task_lower for word in complex_keywords):
            return "complex"
        
        # Simple indicators
        simple_keywords = [
            "simple", "quick", "small", "single", "validate",
            "check", "basic", "trivial", "straightforward"
        ]
        if any(word in task_lower for word in simple_keywords):
            return "simple"
        
        return "moderate"
    
    async def route_request(self, agent_type: str, task_description: str) -> str:
        """Route to optimal model based on complexity and quota"""
        # Get complexity
        complexity = self.detect_complexity(task_description)
        
        # Get recommended model
        rules = self.ROUTING_RULES.get(agent_type, self.ROUTING_RULES["coding_agent"])
        preferred_model = rules.get(complexity, "gemini-1.5-flash")
        
        # Check if quota available
        if await self.quota_manager.can_make_request(preferred_model):
            logger.info(f"Routing to {preferred_model} ({complexity} task)")
            return preferred_model
        
        # Try fallbacks
        fallback_model = await self._get_fallback(agent_type, preferred_model)
        logger.warning(f"Quota exceeded for {preferred_model}, using {fallback_model}")
        return fallback_model
    
    async def _get_fallback(self, agent_type: str, preferred_model: str) -> str:
        """Get fallback model when preferred is unavailable"""
        # Fallback chain
        fallback_chain = [
            "gemini-1.5-flash-8b",  # Highest RPD limit
            "gemini-2.0-flash-exp",  # Fast and capable
            "gemini-1.5-flash",      # Reliable standard
        ]
        
        for model in fallback_chain:
            if model != preferred_model and await self.quota_manager.can_make_request(model):
                return model
        
        # If all Gemini models exhausted, return cheapest available
        return "gemini-1.5-flash-8b"

# ============================================================================
# QUOTA MANAGEMENT
# ============================================================================

class GeminiQuotaManager:
    """Manage quotas to stay within free tier"""
    
    def __init__(self):
        self.usage = {}
        self.daily_reset_time = None
        self._initialize_usage()
    
    def _initialize_usage(self):
        """Initialize usage tracking"""
        for model_id, config in GEMINI_MODELS.items():
            self.usage[model_id] = {
                "requests_today": 0,
                "requests_this_minute": 0,
                "minute_window_start": time.time(),
                "daily_reset": datetime.now().date()
            }
    
    def _reset_minute_counter(self, model_id: str):
        """Reset minute counter if window passed"""
        now = time.time()
        if now - self.usage[model_id]["minute_window_start"] >= 60:
            self.usage[model_id]["requests_this_minute"] = 0
            self.usage[model_id]["minute_window_start"] = now
    
    def _reset_daily_counter(self, model_id: str):
        """Reset daily counter if new day"""
        today = datetime.now().date()
        if self.usage[model_id]["daily_reset"] != today:
            self.usage[model_id]["requests_today"] = 0
            self.usage[model_id]["daily_reset"] = today
    
    async def can_make_request(self, model_id: str) -> bool:
        """Check if request is within limits"""
        if model_id not in GEMINI_MODELS:
            return True  # Unknown model, allow
        
        config = GEMINI_MODELS[model_id]
        self._reset_minute_counter(model_id)
        self._reset_daily_counter(model_id)
        
        # Check RPM
        if self.usage[model_id]["requests_this_minute"] >= config.rpm_limit:
            logger.warning(f"⚠️ RPM limit reached for {model_id}")
            return False
        
        # Check RPD
        if self.usage[model_id]["requests_today"] >= config.rpd_limit:
            logger.warning(f"⚠️ Daily limit reached for {model_id}")
            return False
        
        return True
    
    async def record_request(self, model_id: str):
        """Record a request"""
        if model_id in self.usage:
            self.usage[model_id]["requests_today"] += 1
            self.usage[model_id]["requests_this_minute"] += 1
    
    def get_quota_status(self) -> Dict[str, Any]:
        """Get current quota status for all models"""
        status = {}
        for model_id, config in GEMINI_MODELS.items():
            used_today = self.usage[model_id]["requests_today"]
            remaining = config.rpd_limit - used_today
            percent_used = (used_today / config.rpd_limit) * 100
            
            status[model_id] = {
                "model": config.display_name,
                "used_today": used_today,
                "daily_limit": config.rpd_limit,
                "remaining": remaining,
                "percent_used": round(percent_used, 1),
                "status": self._get_health_status(percent_used)
            }
        
        return status
    
    def _get_health_status(self, percent_used: float) -> str:
        """Determine health status based on usage"""
        if percent_used < 70:
            return "healthy"
        elif percent_used < 90:
            return "warning"
        else:
            return "critical"

# ============================================================================
# MULTI-TIER CACHING
# ============================================================================

class GeminiCacheManager:
    """3-tier caching: Memory -> Redis -> Cloud Storage"""
    
    def __init__(self):
        self.l1_memory = {}  # In-memory cache
        self.l2_redis = None  # Redis client (optional)
        self.l3_cloud = None  # Cloud storage client (optional)
        self.stats = {
            "l1_hits": 0,
            "l2_hits": 0,
            "l3_hits": 0,
            "misses": 0
        }
    
    def _generate_cache_key(self, prompt: str, model: str) -> str:
        """Generate cache key from prompt and model"""
        combined = f"{model}:{prompt}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    async def get(self, prompt: str, model: str) -> Optional[str]:
        """Get cached response"""
        cache_key = self._generate_cache_key(prompt, model)
        
        # L1: Memory (instant)
        if cache_key in self.l1_memory:
            self.stats["l1_hits"] += 1
            logger.debug(f"✅ L1 cache hit for {model}")
            return self.l1_memory[cache_key]
        
        # L2: Redis (if available)
        if self.l2_redis:
            try:
                cached = await self.l2_redis.get(cache_key)
                if cached:
                    self.stats["l2_hits"] += 1
                    self.l1_memory[cache_key] = cached  # Promote to L1
                    logger.debug(f"✅ L2 cache hit for {model}")
                    return cached
            except Exception as e:
                logger.warning(f"Redis error: {e}")
        
        # L3: Cloud Storage (if available)
        if self.l3_cloud:
            try:
                cached = await self.l3_cloud.get(cache_key)
                if cached:
                    self.stats["l3_hits"] += 1
                    self.l1_memory[cache_key] = cached  # Promote to L1
                    if self.l2_redis:
                        await self.l2_redis.set(cache_key, cached, ex=3600)
                    logger.debug(f"✅ L3 cache hit for {model}")
                    return cached
            except Exception as e:
                logger.warning(f"Cloud storage error: {e}")
        
        self.stats["misses"] += 1
        return None
    
    async def set(self, prompt: str, model: str, response: str):
        """Cache response in all tiers"""
        cache_key = self._generate_cache_key(prompt, model)
        
        # L1: Always cache in memory
        self.l1_memory[cache_key] = response
        
        # L2: Cache in Redis (if available)
        if self.l2_redis:
            try:
                await self.l2_redis.set(cache_key, response, ex=3600)  # 1 hour
            except Exception as e:
                logger.warning(f"Redis caching error: {e}")
        
        # L3: Cache in Cloud Storage (if available)
        if self.l3_cloud:
            try:
                await self.l3_cloud.set(cache_key, response, ttl=604800)  # 7 days
            except Exception as e:
                logger.warning(f"Cloud storage caching error: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        total_requests = sum(self.stats.values())
        total_hits = self.stats["l1_hits"] + self.stats["l2_hits"] + self.stats["l3_hits"]
        
        return {
            "total_requests": total_requests,
            "total_hits": total_hits,
            "total_misses": self.stats["misses"],
            "hit_rate_percent": round((total_hits / total_requests * 100), 2) if total_requests > 0 else 0,
            "l1_hits": self.stats["l1_hits"],
            "l2_hits": self.stats["l2_hits"],
            "l3_hits": self.stats["l3_hits"],
            "api_calls_saved": total_hits
        }

# ============================================================================
# CONTEXT WINDOW OPTIMIZER
# ============================================================================

class ContextWindowOptimizer:
    """Optimize usage of Gemini's massive context windows"""
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count (approximate)"""
        # Gemini: ~4 characters per token
        return len(text) // 4
    
    def should_use_long_context_model(
        self,
        content_size_tokens: int,
        task_type: str
    ) -> str:
        """Determine best model for content size"""
        # Very large contexts -> use Pro (2M tokens)
        if content_size_tokens > 500_000:
            return "gemini-1.5-pro"
        
        # Multi-file operations -> use Pro
        if task_type in [
            "codebase_analysis",
            "multi_file_refactor",
            "full_documentation",
            "architectural_review"
        ]:
            return "gemini-1.5-pro"
        
        # Large contexts -> use Flash
        if content_size_tokens > 100_000:
            return "gemini-1.5-flash"
        
        # Standard contexts -> use 2.0 Flash
        return "gemini-2.0-flash-exp"
    
    async def prepare_large_context(
        self,
        files: List[Tuple[str, str]],  # (path, content)
        model_id: str
    ) -> Tuple[str, int]:
        """Combine multiple files into single context"""
        config = GEMINI_MODELS[model_id]
        max_tokens = config.max_context_tokens
        safety_margin = 0.75  # Use 75% to leave room for response
        
        combined_parts = []
        total_tokens = 0
        files_included = 0
        
        for file_path, content in files:
            content_tokens = self.estimate_tokens(content)
            
            if total_tokens + content_tokens < max_tokens * safety_margin:
                combined_parts.append(f"\n\n=== FILE: {file_path} ===\n{content}")
                total_tokens += content_tokens
                files_included += 1
            else:
                logger.warning(
                    f"⚠️ Context limit reached at {files_included} files, "
                    f"skipping {file_path}"
                )
                break
        
        combined_content = "\n".join(combined_parts)
        logger.info(
            f"✅ Prepared large context: {files_included} files, "
            f"~{total_tokens:,} tokens for {model_id}"
        )
        
        return combined_content, total_tokens

# ============================================================================
# USAGE MONITOR
# ============================================================================

class GeminiUsageMonitor:
    """Monitor usage to ensure free tier compliance"""
    
    def __init__(self, quota_manager: GeminiQuotaManager, cache_manager: GeminiCacheManager):
        self.quota_manager = quota_manager
        self.cache_manager = cache_manager
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data"""
        quota_status = self.quota_manager.get_quota_status()
        cache_stats = self.cache_manager.get_stats()
        
        # Calculate totals
        total_requests = sum(
            status["used_today"]
            for status in quota_status.values()
        )
        
        # Determine overall health
        statuses = [status["status"] for status in quota_status.values()]
        overall_health = "healthy"
        if "critical" in statuses:
            overall_health = "critical"
        elif "warning" in statuses:
            overall_health = "warning"
        
        # Generate recommendations
        recommendations = self._generate_recommendations(quota_status, cache_stats)
        
        return {
            "timestamp": datetime.now().isoformat(),
            "overall_health": overall_health,
            "total_requests_today": total_requests,
            "models": quota_status,
            "cache_performance": cache_stats,
            "recommendations": recommendations
        }
    
    def _generate_recommendations(
        self,
        quota_status: Dict,
        cache_stats: Dict
    ) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Check quota health
        for model_id, status in quota_status.items():
            if status["status"] == "critical":
                recommendations.append(
                    f"⚠️ {status['model']} at {status['percent_used']}% - "
                    f"switch to alternative models"
                )
            elif status["status"] == "warning":
                recommendations.append(
                    f"💡 {status['model']} at {status['percent_used']}% - "
                    f"monitor usage closely"
                )
        
        # Check cache performance
        if cache_stats["hit_rate_percent"] < 50:
            recommendations.append(
                f"💡 Cache hit rate at {cache_stats['hit_rate_percent']}% - "
                f"consider improving caching strategy"
            )
        
        # If all healthy
        if not recommendations:
            recommendations.append("✅ All systems healthy")
            recommendations.append(
                f"💰 Saved {cache_stats['api_calls_saved']} API calls via caching"
            )
        
        return recommendations

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

async def example_usage():
    """Example of using the optimization components"""
    
    # Initialize components
    quota_manager = GeminiQuotaManager()
    cache_manager = GeminiCacheManager()
    router = GeminiRouter(quota_manager)
    context_optimizer = ContextWindowOptimizer()
    monitor = GeminiUsageMonitor(quota_manager, cache_manager)
    
    # Example 1: Route a request
    model = await router.route_request(
        agent_type="coding_agent",
        task_description="Create a complex microservices architecture"
    )
    print(f"Selected model: {model}")
    
    # Example 2: Check cache
    cached_response = await cache_manager.get(
        prompt="What is Python?",
        model="gemini-2.0-flash-exp"
    )
    if not cached_response:
        # Make API call (simulated)
        response = "Python is a programming language..."
        await cache_manager.set("What is Python?", "gemini-2.0-flash-exp", response)
        await quota_manager.record_request("gemini-2.0-flash-exp")
    
    # Example 3: Prepare large context
    files = [
        ("app.py", "# Main application\nprint('Hello')"),
        ("utils.py", "# Utilities\ndef helper(): pass"),
    ]
    combined_context, tokens = await context_optimizer.prepare_large_context(
        files,
        "gemini-1.5-pro"
    )
    print(f"Context prepared: {tokens} tokens")
    
    # Example 4: Get dashboard
    dashboard = monitor.get_dashboard_data()
    print("\n=== Dashboard ===")
    print(f"Overall health: {dashboard['overall_health']}")
    print(f"Total requests today: {dashboard['total_requests_today']}")
    print(f"Cache hit rate: {dashboard['cache_performance']['hit_rate_percent']}%")
    print("\nRecommendations:")
    for rec in dashboard['recommendations']:
        print(f"  {rec}")

if __name__ == "__main__":
    # Run example
    asyncio.run(example_usage())

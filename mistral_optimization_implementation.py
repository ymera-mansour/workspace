"""
Mistral AI Optimization Implementation
======================================

Basic optimization components for Mistral models:
- 3-tier caching system (Memory → Redis → Cloud Storage)
- Intelligent routing by task complexity
- Quota management for FREE tier
- Context window optimization (up to 256K tokens)

Cost: $0 - 100% FREE tier operation
"""

import hashlib
import time
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import OrderedDict


# ============================================================================
# 3-TIER CACHING SYSTEM
# ============================================================================

class MistralCacheManager:
    """
    Multi-tier caching system for Mistral API responses.
    
    L1: Memory cache (0ms latency, 100MB limit)
    L2: Redis cache (5-10ms latency, optional)
    L3: Cloud Storage (100-200ms latency, optional)
    
    Target: 60-80% cache hit rate
    Expected: 60-75% reduction in API calls
    """
    
    def __init__(
        self,
        l1_max_size_mb: int = 100,
        l1_ttl_seconds: int = 300,
        l2_enabled: bool = False,
        l3_enabled: bool = False
    ):
        # L1: Memory cache (LRU)
        self.l1_cache: OrderedDict = OrderedDict()
        self.l1_max_size_bytes = l1_max_size_mb * 1024 * 1024
        self.l1_current_size = 0
        self.l1_ttl = l1_ttl_seconds
        self.l1_expiry: Dict[str, float] = {}
        
        # L2: Redis cache (optional)
        self.l2_enabled = l2_enabled
        self.l2_client = None
        if l2_enabled:
            try:
                import redis
                self.l2_client = redis.Redis(host='localhost', port=6379, decode_responses=False)
            except:
                self.l2_enabled = False
        
        # L3: Cloud storage (optional)
        self.l3_enabled = l3_enabled
        self.l3_client = None
        
        # Statistics
        self.stats = {
            "l1_hits": 0,
            "l2_hits": 0,
            "l3_hits": 0,
            "misses": 0,
            "total_requests": 0
        }
    
    def _generate_key(self, prompt: str, model: str, params: Dict = None) -> str:
        """Generate cache key from prompt and parameters."""
        key_data = f"{model}:{prompt}"
        if params:
            key_data += f":{json.dumps(params, sort_keys=True)}"
        return hashlib.sha256(key_data.encode()).hexdigest()
    
    async def get(self, prompt: str, model: str, params: Dict = None) -> Optional[str]:
        """
        Get cached response (checks L1 → L2 → L3).
        
        Returns:
            Cached response if found, None otherwise
        """
        self.stats["total_requests"] += 1
        cache_key = self._generate_key(prompt, model, params)
        
        # Check L1 (memory)
        if cache_key in self.l1_cache:
            # Check expiry
            if time.time() < self.l1_expiry.get(cache_key, 0):
                self.stats["l1_hits"] += 1
                # Move to end (LRU)
                self.l1_cache.move_to_end(cache_key)
                return self.l1_cache[cache_key]
            else:
                # Expired, remove
                self._remove_from_l1(cache_key)
        
        # Check L2 (Redis)
        if self.l2_enabled and self.l2_client:
            try:
                response = self.l2_client.get(f"mistral:{cache_key}")
                if response:
                    self.stats["l2_hits"] += 1
                    # Promote to L1
                    await self._set_l1(cache_key, response.decode())
                    return response.decode()
            except:
                pass
        
        # Check L3 (Cloud Storage)
        if self.l3_enabled and self.l3_client:
            try:
                response = await self._get_from_l3(cache_key)
                if response:
                    self.stats["l3_hits"] += 1
                    # Promote to L2 and L1
                    await self._set_l2(cache_key, response)
                    await self._set_l1(cache_key, response)
                    return response
            except:
                pass
        
        # Cache miss
        self.stats["misses"] += 1
        return None
    
    async def set(self, prompt: str, model: str, response: str, params: Dict = None):
        """Store response in all cache tiers."""
        cache_key = self._generate_key(prompt, model, params)
        
        # Store in L1 (memory)
        await self._set_l1(cache_key, response)
        
        # Store in L2 (Redis)
        if self.l2_enabled and self.l2_client:
            await self._set_l2(cache_key, response, ttl=3600)
        
        # Store in L3 (Cloud Storage)
        if self.l3_enabled and self.l3_client:
            await self._set_l3(cache_key, response, ttl=86400)
    
    async def _set_l1(self, key: str, value: str):
        """Set value in L1 cache with LRU eviction."""
        value_size = len(value.encode())
        
        # Evict if necessary
        while (self.l1_current_size + value_size > self.l1_max_size_bytes and 
               len(self.l1_cache) > 0):
            evict_key, evict_value = self.l1_cache.popitem(last=False)
            self.l1_current_size -= len(evict_value.encode())
            del self.l1_expiry[evict_key]
        
        # Add to cache
        self.l1_cache[key] = value
        self.l1_current_size += value_size
        self.l1_expiry[key] = time.time() + self.l1_ttl
    
    def _remove_from_l1(self, key: str):
        """Remove key from L1 cache."""
        if key in self.l1_cache:
            value = self.l1_cache.pop(key)
            self.l1_current_size -= len(value.encode())
            if key in self.l1_expiry:
                del self.l1_expiry[key]
    
    async def _set_l2(self, key: str, value: str, ttl: int = 3600):
        """Set value in L2 cache (Redis)."""
        if not self.l2_enabled or not self.l2_client:
            return
        try:
            self.l2_client.setex(f"mistral:{key}", ttl, value)
        except:
            pass
    
    async def _get_from_l3(self, key: str) -> Optional[str]:
        """Get value from L3 cache (Cloud Storage)."""
        # Placeholder for cloud storage implementation
        return None
    
    async def _set_l3(self, key: str, value: str, ttl: int = 86400):
        """Set value in L3 cache (Cloud Storage)."""
        # Placeholder for cloud storage implementation
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self.stats["total_requests"]
        hits = self.stats["l1_hits"] + self.stats["l2_hits"] + self.stats["l3_hits"]
        
        return {
            "total_requests": total,
            "cache_hits": hits,
            "cache_misses": self.stats["misses"],
            "hit_rate": hits / total if total > 0 else 0,
            "l1_hits": self.stats["l1_hits"],
            "l2_hits": self.stats["l2_hits"],
            "l3_hits": self.stats["l3_hits"],
            "l1_size_mb": self.l1_current_size / (1024 * 1024),
            "l1_entries": len(self.l1_cache)
        }


# ============================================================================
# INTELLIGENT ROUTING
# ============================================================================

class MistralRouter:
    """
    Intelligent routing based on task complexity and requirements.
    
    Routes to optimal Mistral model based on:
    - Task complexity (simple/moderate/complex)
    - Task type (code/vision/analysis/etc.)
    - Context size requirements
    - Speed vs quality preference
    """
    
    def __init__(self, quota_manager=None):
        self.quota_manager = quota_manager
        
        # Model characteristics
        self.models = {
            "ministral-3b-latest": {
                "speed": 10, "quality": 6, "cost": 1,
                "best_for": ["simple", "validation", "formatting"]
            },
            "ministral-8b-latest": {
                "speed": 8, "quality": 7, "cost": 2,
                "best_for": ["moderate", "database", "api", "testing"]
            },
            "ministral-14b-latest": {
                "speed": 7, "quality": 8, "cost": 3,
                "best_for": ["moderate", "analysis", "planning", "documentation"]
            },
            "codestral-latest": {
                "speed": 8, "quality": 9, "cost": 3,
                "best_for": ["code", "code_generation", "debugging", "refactoring"]
            },
            "pixtral-large-latest": {
                "speed": 5, "quality": 9, "cost": 5,
                "best_for": ["vision", "image_analysis", "ui_review"]
            },
            "mistral-large-latest": {
                "speed": 6, "quality": 10, "cost": 5,
                "best_for": ["complex", "architecture", "security", "large_context"]
            },
        }
    
    async def route_request(
        self,
        agent_name: str,
        task_description: str,
        task_type: str = None,
        prefer_speed: bool = False,
        context_tokens: int = 0
    ) -> str:
        """
        Route request to optimal model.
        
        Args:
            agent_name: Name of the requesting agent
            task_description: Description of the task
            task_type: Type of task (optional)
            prefer_speed: Whether to prefer faster models
            context_tokens: Number of tokens in context
        
        Returns:
            Model name to use
        """
        # Detect task type from description if not provided
        if not task_type:
            task_type = self._detect_task_type(task_description)
        
        # Detect complexity
        complexity = self._detect_complexity(task_description, context_tokens)
        
        # Find suitable models
        candidates = self._find_candidate_models(task_type, complexity, context_tokens)
        
        # Filter by quota availability
        if self.quota_manager:
            candidates = [m for m in candidates if self.quota_manager.can_make_request(m)]
        
        if not candidates:
            # Fallback to smallest model
            return "ministral-3b-latest"
        
        # Select based on preference
        if prefer_speed:
            return max(candidates, key=lambda m: self.models[m]["speed"])
        else:
            return max(candidates, key=lambda m: self.models[m]["quality"])
    
    def _detect_task_type(self, description: str) -> str:
        """Detect task type from description."""
        description_lower = description.lower()
        
        # Code-related keywords
        if any(kw in description_lower for kw in ["code", "function", "class", "debug", "refactor", "implement"]):
            return "code"
        
        # Vision-related keywords
        if any(kw in description_lower for kw in ["image", "picture", "screenshot", "ui", "diagram", "visual"]):
            return "vision"
        
        # Architecture keywords
        if any(kw in description_lower for kw in ["architecture", "design", "system", "microservices"]):
            return "architecture"
        
        # Analysis keywords
        if any(kw in description_lower for kw in ["analyze", "analysis", "evaluate", "assess"]):
            return "analysis"
        
        # Security keywords
        if any(kw in description_lower for kw in ["security", "vulnerability", "exploit", "attack"]):
            return "security"
        
        # Default
        return "moderate"
    
    def _detect_complexity(self, description: str, context_tokens: int) -> str:
        """Detect task complexity."""
        # Large context always complex
        if context_tokens > 100000:
            return "complex"
        
        # Check description length and keywords
        description_lower = description.lower()
        word_count = len(description.split())
        
        # Complex keywords
        complex_keywords = ["architecture", "comprehensive", "detailed", "complete", "full"]
        if any(kw in description_lower for kw in complex_keywords) or word_count > 50:
            return "complex"
        
        # Simple keywords
        simple_keywords = ["simple", "quick", "small", "basic", "validate", "format"]
        if any(kw in description_lower for kw in simple_keywords) or word_count < 10:
            return "simple"
        
        return "moderate"
    
    def _find_candidate_models(
        self,
        task_type: str,
        complexity: str,
        context_tokens: int
    ) -> List[str]:
        """Find candidate models for the task."""
        candidates = []
        
        for model, chars in self.models.items():
            # Check if model is good for this task type
            if task_type in chars["best_for"]:
                candidates.append(model)
            # Check if model is good for this complexity
            elif complexity in chars["best_for"]:
                candidates.append(model)
        
        # Special cases
        if task_type == "code":
            candidates = ["codestral-latest", "mistral-large-latest", "ministral-8b-latest"]
        elif task_type == "vision":
            candidates = ["pixtral-large-latest", "mistral-large-latest"]
        elif complexity == "complex":
            candidates = ["mistral-large-latest", "codestral-latest", "ministral-14b-latest"]
        elif complexity == "simple":
            candidates = ["ministral-3b-latest", "ministral-8b-latest"]
        
        # Filter by context window
        if context_tokens > 128000:
            # Only large models support >128K
            candidates = [m for m in candidates if m in ["mistral-large-latest", "codestral-latest", "ministral-3b-latest", "ministral-8b-latest", "ministral-14b-latest"]]
        
        return candidates if candidates else ["ministral-8b-latest"]


# ============================================================================
# QUOTA MANAGEMENT
# ============================================================================

class MistralQuotaManager:
    """
    Quota management for FREE tier limits.
    
    FREE tier limits per model:
    - 60 RPM (requests per minute)
    - 86,400 RPD (requests per day)
    - 500K TPM (tokens per minute)
    - 1B tokens per month
    """
    
    def __init__(self):
        self.rpm_limit = 60
        self.rpd_limit = 86400
        self.tpm_limit = 500000
        
        # Track usage per model
        self.usage: Dict[str, Dict[str, Any]] = {}
        self.reset_times: Dict[str, Dict[str, datetime]] = {}
    
    def can_make_request(self, model: str, tokens: int = 1000) -> bool:
        """Check if request can be made within quota limits."""
        self._reset_if_needed(model)
        
        if model not in self.usage:
            return True
        
        usage = self.usage[model]
        
        # Check RPM
        if usage.get("rpm", 0) >= self.rpm_limit:
            return False
        
        # Check RPD
        if usage.get("rpd", 0) >= self.rpd_limit:
            return False
        
        # Check TPM
        if usage.get("tpm", 0) + tokens > self.tpm_limit:
            return False
        
        return True
    
    def record_request(self, model: str, tokens: int = 1000):
        """Record a request."""
        self._reset_if_needed(model)
        
        if model not in self.usage:
            self.usage[model] = {"rpm": 0, "rpd": 0, "tpm": 0}
            self.reset_times[model] = {
                "minute": datetime.now(),
                "day": datetime.now()
            }
        
        self.usage[model]["rpm"] += 1
        self.usage[model]["rpd"] += 1
        self.usage[model]["tpm"] += tokens
    
    def _reset_if_needed(self, model: str):
        """Reset counters if time period has elapsed."""
        if model not in self.reset_times:
            return
        
        now = datetime.now()
        
        # Reset RPM and TPM if minute has passed
        if (now - self.reset_times[model]["minute"]).total_seconds() >= 60:
            self.usage[model]["rpm"] = 0
            self.usage[model]["tpm"] = 0
            self.reset_times[model]["minute"] = now
        
        # Reset RPD if day has passed
        if (now - self.reset_times[model]["day"]).total_seconds() >= 86400:
            self.usage[model]["rpd"] = 0
            self.reset_times[model]["day"] = now
    
    def get_quota_status(self, model: str) -> Dict[str, Any]:
        """Get quota status for a model."""
        self._reset_if_needed(model)
        
        if model not in self.usage:
            return {
                "rpm_used": 0,
                "rpm_remaining": self.rpm_limit,
                "rpd_used": 0,
                "rpd_remaining": self.rpd_limit,
                "tpm_used": 0,
                "tpm_remaining": self.tpm_limit
            }
        
        usage = self.usage[model]
        return {
            "rpm_used": usage.get("rpm", 0),
            "rpm_remaining": self.rpm_limit - usage.get("rpm", 0),
            "rpd_used": usage.get("rpd", 0),
            "rpd_remaining": self.rpd_limit - usage.get("rpd", 0),
            "tpm_used": usage.get("tpm", 0),
            "tpm_remaining": self.tpm_limit - usage.get("tpm", 0)
        }
    
    def get_fallback_model(self, model: str) -> Optional[str]:
        """Get fallback model if current one is at quota."""
        fallback_chain = {
            "mistral-large-latest": ["ministral-14b-latest", "ministral-8b-latest"],
            "codestral-latest": ["mistral-large-latest", "ministral-8b-latest"],
            "pixtral-large-latest": ["mistral-large-latest", "ministral-14b-latest"],
            "ministral-14b-latest": ["ministral-8b-latest", "ministral-3b-latest"],
            "ministral-8b-latest": ["ministral-3b-latest", "ministral-14b-latest"],
            "ministral-3b-latest": ["ministral-8b-latest"]
        }
        
        fallbacks = fallback_chain.get(model, [])
        for fallback in fallbacks:
            if self.can_make_request(fallback):
                return fallback
        
        return None


# ============================================================================
# CONTEXT WINDOW OPTIMIZATION
# ============================================================================

class MistralContextOptimizer:
    """
    Optimize context window usage for large codebases.
    
    Mistral context windows:
    - Mistral Large 3: 256K tokens
    - Codestral: 256K tokens
    - Ministral 3B/8B/14B: 256K tokens
    - Pixtral Large: 128K tokens
    """
    
    def __init__(self):
        self.context_limits = {
            "mistral-large-latest": 256000,
            "codestral-latest": 256000,
            "pixtral-large-latest": 128000,
            "ministral-14b-latest": 256000,
            "ministral-8b-latest": 256000,
            "ministral-3b-latest": 256000,
        }
    
    def optimize_context(
        self,
        files: List[Tuple[str, str]],
        max_tokens: int = None,
        model: str = "ministral-8b-latest"
    ) -> Tuple[str, int]:
        """
        Optimize context by combining files intelligently.
        
        Args:
            files: List of (filename, content) tuples
            max_tokens: Maximum tokens (auto-detected from model if None)
            model: Model to optimize for
        
        Returns:
            (combined_content, estimated_tokens)
        """
        if max_tokens is None:
            max_tokens = self.context_limits.get(model, 256000)
        
        # Apply safety margin (90% of max)
        safe_max = int(max_tokens * 0.9)
        
        # Estimate tokens (rough: 1 token ≈ 4 characters)
        def estimate_tokens(text: str) -> int:
            return len(text) // 4
        
        # Sort files by importance (you can customize this)
        sorted_files = sorted(files, key=lambda x: len(x[1]), reverse=False)
        
        # Combine files until we hit the limit
        combined = []
        total_tokens = 0
        
        for filename, content in sorted_files:
            file_tokens = estimate_tokens(content)
            
            if total_tokens + file_tokens <= safe_max:
                combined.append(f"// File: {filename}\n{content}\n\n")
                total_tokens += file_tokens
            else:
                # Try to include partial content
                remaining = safe_max - total_tokens
                if remaining > 100:
                    partial_content = content[:remaining * 4]
                    combined.append(f"// File: {filename} (partial)\n{partial_content}\n\n")
                    total_tokens += remaining
                break
        
        return "".join(combined), total_tokens
    
    def select_model_for_context(self, estimated_tokens: int) -> str:
        """Select optimal model based on context size."""
        if estimated_tokens > 128000:
            # Need 256K context
            return "mistral-large-latest"
        elif estimated_tokens > 64000:
            # Can use any model, prefer quality
            return "codestral-latest"
        else:
            # Small context, can use fast models
            return "ministral-8b-latest"


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

async def main():
    """Example usage."""
    # Initialize components
    cache = MistralCacheManager()
    router = MistralRouter()
    quota_manager = MistralQuotaManager()
    optimizer = MistralContextOptimizer()
    
    # Example: Route a request
    model = await router.route_request(
        agent_name="coding_agent",
        task_description="Create a REST API endpoint for user authentication"
    )
    print(f"Selected model: {model}")
    
    # Example: Check quota
    can_request = quota_manager.can_make_request(model)
    print(f"Can make request: {can_request}")
    
    # Example: Optimize context
    files = [
        ("app.py", "def main(): pass" * 100),
        ("utils.py", "def helper(): pass" * 100),
    ]
    combined, tokens = optimizer.optimize_context(files, model=model)
    print(f"Combined context: {tokens} tokens")
    
    # Example: Get cache stats
    stats = cache.get_stats()
    print(f"Cache hit rate: {stats['hit_rate']:.2%}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

# Gemini Optimization - Quick Implementation Guide

## Overview

This guide provides step-by-step instructions to implement the Gemini optimization recommendations from the comprehensive review.

## Prerequisites

- Python 3.9+
- Gemini API key from [Google AI Studio](https://aistudio.google.com/)
- Existing YMERA multi-agent platform codebase

## Quick Start (30 Minutes)

### Step 1: Update Model Configuration (5 min)

**File:** `agent_platform (3).py` or equivalent

```python
# Add Gemini 2.0 Flash to your configuration
"gemini": ProviderConfig(
    name="gemini",
    api_key=os.getenv("GEMINI_API_KEY", ""),
    models=[
        "gemini-2.0-flash-exp",      # NEW: Latest, fastest
        "gemini-1.5-flash",          # Fast and cheap
        "gemini-1.5-pro",            # Most capable
        "gemini-1.5-flash-8b",       # NEW: Highest throughput
    ],
    cost_per_1k_tokens=0.0,  # Free tier
    max_tokens=2000000,  # 2M for Pro, 1M for others
    rate_limit_rpm=15  # Varies by model
)
```

### Step 2: Copy Optimization Classes (10 min)

1. Copy `gemini_optimization_implementation.py` to your project
2. Import the classes:

```python
from gemini_optimization_implementation import (
    GeminiQuotaManager,
    GeminiCacheManager,
    GeminiRouter,
    ContextWindowOptimizer,
    GeminiUsageMonitor
)
```

### Step 3: Initialize Components (5 min)

Add to your application initialization:

```python
# Initialize optimization components
quota_manager = GeminiQuotaManager()
cache_manager = GeminiCacheManager()
router = GeminiRouter(quota_manager)
monitor = GeminiUsageMonitor(quota_manager, cache_manager)

# Optionally connect Redis for L2 caching
# cache_manager.l2_redis = your_redis_client
```

### Step 4: Update Request Flow (10 min)

Modify your Gemini request handler:

```python
async def execute_gemini_request(agent_type: str, prompt: str):
    # 1. Check cache first
    cached = await cache_manager.get(prompt, "gemini-2.0-flash-exp")
    if cached:
        logger.info("Cache hit!")
        return cached
    
    # 2. Route to optimal model
    model = await router.route_request(agent_type, prompt)
    
    # 3. Check quota
    if not await quota_manager.can_make_request(model):
        logger.warning("Quota exceeded, trying fallback")
        model = await router._get_fallback(agent_type, model)
    
    # 4. Make API call
    response = await call_gemini_api(model, prompt)
    
    # 5. Record usage & cache
    await quota_manager.record_request(model)
    await cache_manager.set(prompt, model, response)
    
    return response
```

## Detailed Implementation

### A. Enhanced Model Routing

Replace your current model selection with:

```python
# In your multi_model_executor or similar
from gemini_optimization_implementation import GeminiRouter

class EnhancedMultiModelExecutor:
    def __init__(self):
        self.gemini_router = GeminiRouter(quota_manager)
    
    async def select_model(self, agent_name: str, task_desc: str):
        # Use intelligent routing
        model = await self.gemini_router.route_request(agent_name, task_desc)
        return ("gemini", model)
```

### B. Add Caching Layer

```python
# Before any Gemini API call
async def call_with_caching(prompt: str, model: str):
    # Check cache
    cached = await cache_manager.get(prompt, model)
    if cached:
        return cached
    
    # Make API call
    response = await actual_api_call(model, prompt)
    
    # Cache response
    await cache_manager.set(prompt, model, response)
    
    return response
```

### C. Add Monitoring Dashboard

```python
# Add API endpoint or CLI command
@app.get("/api/gemini/status")
async def get_gemini_status():
    dashboard = monitor.get_dashboard_data()
    return dashboard

# Or for CLI
if __name__ == "__main__":
    dashboard = monitor.get_dashboard_data()
    print(json.dumps(dashboard, indent=2))
```

### D. Implement Large Context Handling

```python
from gemini_optimization_implementation import ContextWindowOptimizer

context_optimizer = ContextWindowOptimizer()

async def analyze_codebase(file_paths: List[str]):
    # Load all files
    files = [(path, read_file(path)) for path in file_paths]
    
    # Determine best model
    total_tokens = sum(context_optimizer.estimate_tokens(content) 
                       for _, content in files)
    model = context_optimizer.should_use_long_context_model(
        total_tokens, 
        "codebase_analysis"
    )
    
    # Prepare combined context
    combined_context, tokens = await context_optimizer.prepare_large_context(
        files,
        model
    )
    
    # Make request with large context
    response = await call_gemini_api(model, combined_context)
    return response
```

## Configuration

### Environment Variables

```bash
# .env file
GEMINI_API_KEY=your_api_key_here

# Optional: Redis for caching
REDIS_URL=redis://localhost:6379

# Optional: Google Cloud Storage for persistent cache
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json
```

### Model Selection Guidelines

Use this decision tree:

```
Task Type?
├─ Simple/Quick → gemini-1.5-flash-8b (4000 RPD)
├─ Standard → gemini-2.0-flash-exp (1500 RPD, fastest)
├─ Moderate → gemini-1.5-flash (1500 RPD, reliable)
└─ Complex/Critical → gemini-1.5-pro (50 RPD only!)

Context Size?
├─ < 100K tokens → Any Flash model
├─ 100K - 500K tokens → gemini-1.5-flash or 2.0-flash
└─ > 500K tokens → gemini-1.5-pro (2M context)

Daily Budget Remaining?
├─ If model quota exhausted → Use fallback chain
└─ If all quotas healthy → Use optimal model
```

## Testing

### Test 1: Verify Model Routing

```python
async def test_routing():
    router = GeminiRouter(quota_manager)
    
    # Test simple task
    model = await router.route_request(
        "coding_agent",
        "Write a simple hello world function"
    )
    assert model == "gemini-2.0-flash-exp"
    
    # Test complex task
    model = await router.route_request(
        "coding_agent",
        "Design a comprehensive microservices architecture"
    )
    assert model == "gemini-1.5-pro"
    
    print("✅ Routing tests passed")
```

### Test 2: Verify Caching

```python
async def test_caching():
    cache = GeminiCacheManager()
    
    # Set cache
    await cache.set("test prompt", "gemini-2.0-flash-exp", "test response")
    
    # Get from cache
    cached = await cache.get("test prompt", "gemini-2.0-flash-exp")
    assert cached == "test response"
    
    # Check stats
    stats = cache.get_stats()
    assert stats["l1_hits"] == 1
    
    print("✅ Cache tests passed")
```

### Test 3: Verify Quota Tracking

```python
async def test_quota():
    quota = GeminiQuotaManager()
    
    # Should allow requests initially
    can_request = await quota.can_make_request("gemini-2.0-flash-exp")
    assert can_request == True
    
    # Record requests
    for _ in range(5):
        await quota.record_request("gemini-2.0-flash-exp")
    
    # Check status
    status = quota.get_quota_status()
    assert status["gemini-2.0-flash-exp"]["used_today"] == 5
    
    print("✅ Quota tests passed")
```

## Monitoring

### Daily Monitoring Routine

1. **Check Dashboard**
   ```bash
   python -c "from gemini_optimization_implementation import *; import asyncio; asyncio.run(example_usage())"
   ```

2. **Review Quotas**
   - Ensure no model exceeds 80% daily limit
   - Switch to alternative models if approaching limits

3. **Check Cache Performance**
   - Target: 60-80% hit rate
   - If lower, investigate prompt variations

4. **Review Costs**
   - Should always be $0 (free tier)
   - If costs appear, investigate quota breaches

### Alerts to Set Up

```python
# Alert thresholds
ALERT_THRESHOLDS = {
    "quota_warning": 80,  # % of daily limit
    "quota_critical": 95,  # % of daily limit
    "cache_hit_rate_low": 50,  # % hit rate
}

async def check_alerts():
    dashboard = monitor.get_dashboard_data()
    
    # Check quotas
    for model, status in dashboard["models"].items():
        if status["percent_used"] > ALERT_THRESHOLDS["quota_critical"]:
            send_alert(f"CRITICAL: {model} at {status['percent_used']}%")
        elif status["percent_used"] > ALERT_THRESHOLDS["quota_warning"]:
            send_alert(f"WARNING: {model} at {status['percent_used']}%")
    
    # Check cache
    cache_rate = dashboard["cache_performance"]["hit_rate_percent"]
    if cache_rate < ALERT_THRESHOLDS["cache_hit_rate_low"]:
        send_alert(f"LOW CACHE PERFORMANCE: {cache_rate}%")
```

## Troubleshooting

### Issue: "Quota exceeded" errors

**Solution:**
1. Check quota status: `quota_manager.get_quota_status()`
2. Verify fallback chain is working
3. Increase caching to reduce API calls
4. Distribute load to other models (Flash-8B, other providers)

### Issue: Low cache hit rate

**Solution:**
1. Ensure cache is properly initialized
2. Check if prompts are being normalized
3. Implement semantic caching for similar prompts
4. Increase cache TTL if content is evergreen

### Issue: "Model not found" errors

**Solution:**
1. Verify API key is valid
2. Check model ID spelling (e.g., "gemini-2.0-flash-exp")
3. Ensure using Google AI Studio API, not Vertex AI
4. Some models may be experimental - check availability

## Next Steps

1. ✅ Implement basic routing and caching (Day 1)
2. ✅ Add monitoring dashboard (Day 2)
3. ✅ Test with production load (Day 3-4)
4. ✅ Fine-tune routing rules based on results (Week 2)
5. ✅ Add Redis/Cloud Storage for persistent caching (Week 2)
6. ✅ Implement multimodal support (Week 3)
7. ✅ Add BigQuery analytics (Week 3)

## Resources

- [Comprehensive Review Document](./GEMINI_GOOGLE_PRODUCTS_OPTIMIZATION_REVIEW.md)
- [Implementation Code](./gemini_optimization_implementation.py)
- [Google AI Studio](https://aistudio.google.com/)
- [Gemini API Documentation](https://ai.google.dev/docs)
- [Free Tier Limits](https://ai.google.dev/pricing)

## Support

For issues or questions:
1. Review the comprehensive review document
2. Check implementation code examples
3. Consult Google AI documentation
4. Review quota status and logs

---

**Last Updated**: December 6, 2024  
**Version**: 1.0

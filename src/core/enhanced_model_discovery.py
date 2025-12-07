# YMERA Enhanced Model Discovery & Auto-Configuration
# Automatically discovers, tests, and integrates ALL available AI models

import asyncio
import aiohttp
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

@dataclass
class ModelCapabilities:
    """Comprehensive model capability assessment"""
    model_id: str
    provider: str
    
    # Performance metrics
    avg_latency_ms: float = 0.0
    tokens_per_second: float = 0.0
    max_context_length: int = 4096
    
    # Capability scores (0-100)
    code_generation: int = 0
    code_review: int = 0
    reasoning: int = 0
    math: int = 0
    creativity: int = 0
    instruction_following: int = 0
    multilingual: int = 0
    
    # Cost & availability
    cost_per_1k_tokens: float = 0.0
    is_free: bool = True
    rate_limit_rpm: int = 60
    
    # Specializations
    specializations: List[str] = field(default_factory=list)
    best_use_cases: List[str] = field(default_factory=list)
    
    # Quality metrics
    reliability_score: float = 0.0  # 0-1
    success_rate: float = 0.0  # 0-1
    
    # Metadata
    tested_at: str = ""
    version: str = ""

class EnhancedModelDiscovery:
    """
    Advanced model discovery system that:
    1. Discovers ALL models from ALL providers
    2. Tests each model's actual capabilities
    3. Benchmarks performance
    4. Auto-configures optimal usage
    5. Continuously monitors and updates
    """
    
    def __init__(self):
        self.discovered_models: Dict[str, ModelCapabilities] = {}
        self.benchmark_tests = self._initialize_benchmark_tests()
        
    def _initialize_benchmark_tests(self) -> Dict[str, str]:
        """Standardized tests to measure model capabilities"""
        return {
            "code_generation": """Write a Python function that:
1. Takes a list of numbers
2. Returns the sum of even numbers
3. Includes type hints and docstring""",
            
            "code_review": """Review this code and suggest improvements:
def calc(x, y):
    return x + y if x > 0 else x - y""",
            
            "reasoning": """If all A are B, and all B are C, can we conclude all A are C?
Explain step by step.""",
            
            "math": """Solve: If f(x) = 2x + 3, what is f(5)?
Show your work.""",
            
            "creativity": """Write a creative tagline for an AI-powered task automation platform.""",
            
            "instruction_following": """Follow these exact instructions:
1. Say "START"
2. Count to 3
3. Say "END"
Nothing else.""",
            
            "multilingual": """Translate to Spanish: 'The quick brown fox jumps over the lazy dog'"""
        }
    
    async def discover_all_providers(self) -> Dict[str, List[ModelCapabilities]]:
        """Discover models from all providers including OpenRouter"""
        
        providers = {
            "openrouter": self._discover_openrouter,
            "gemini": self._discover_gemini,
            "groq": self._discover_groq,
            "mistral": self._discover_mistral,
            "anthropic": self._discover_anthropic,
            "cohere": self._discover_cohere,
            "together": self._discover_together,
            "replicate": self._discover_replicate,
        }
        
        all_models = {}
        
        for provider_name, discover_func in providers.items():
            try:
                logger.info(f"Discovering models from {provider_name}...")
                models = await discover_func()
                all_models[provider_name] = models
                logger.info(f"✓ Found {len(models)} models from {provider_name}")
            except Exception as e:
                logger.error(f"Failed to discover {provider_name}: {e}")
                all_models[provider_name] = []
        
        return all_models
    
    async def _discover_openrouter(self) -> List[ModelCapabilities]:
        """Discover ALL models from OpenRouter including free ones"""
        
        url = "https://openrouter.ai/api/v1/models"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status != 200:
                    logger.error(f"OpenRouter API returned {response.status}")
                    return []
                
                data = await response.json()
                models = []
                
                for model in data.get("data", []):
                    # Extract pricing info
                    pricing = model.get("pricing", {})
                    prompt_cost = float(pricing.get("prompt", "0"))
                    completion_cost = float(pricing.get("completion", "0"))
                    is_free = (prompt_cost == 0 and completion_cost == 0)
                    
                    # Only include free models or high-quality paid ones
                    if not is_free and not self._is_premium_model(model["id"]):
                        continue
                    
                    capability = ModelCapabilities(
                        model_id=model["id"],
                        provider="openrouter",
                        max_context_length=model.get("context_length", 4096),
                        cost_per_1k_tokens=(prompt_cost + completion_cost) / 2,
                        is_free=is_free,
                        version=model.get("id", "").split(":")[-1]
                    )
                    
                    # Infer capabilities from model name/description
                    self._infer_capabilities(capability, model)
                    
                    models.append(capability)
                
                # Sort by free first, then by context length
                models.sort(key=lambda m: (not m.is_free, -m.max_context_length))
                
                logger.info(f"OpenRouter: {sum(1 for m in models if m.is_free)} free models found")
                return models
    
    def _is_premium_model(self, model_id: str) -> bool:
        """Check if a paid model is worth including"""
        premium_keywords = [
            "gpt-4", "claude-3-opus", "claude-3.5-sonnet",
            "gemini-pro", "command-r-plus"
        ]
        return any(keyword in model_id.lower() for keyword in premium_keywords)
    
    def _infer_capabilities(self, capability: ModelCapabilities, model_data: Dict):
        """Infer model capabilities from metadata"""
        
        model_id = model_data.get("id", "").lower()
        description = model_data.get("description", "").lower()
        name = model_data.get("name", "").lower()
        
        combined_text = f"{model_id} {description} {name}"
        
        # Code capabilities
        if any(term in combined_text for term in ["code", "codestral", "deepseek-coder"]):
            capability.code_generation = 85
            capability.code_review = 85
            capability.specializations.append("code")
            capability.best_use_cases.extend(["code_generation", "code_review"])
        
        # Reasoning
        if any(term in combined_text for term in ["reasoning", "think", "analysis"]):
            capability.reasoning = 80
            capability.math = 75
            capability.specializations.append("reasoning")
            capability.best_use_cases.append("complex_reasoning")
        
        # Creativity
        if any(term in combined_text for term in ["creative", "story", "writing"]):
            capability.creativity = 80
            capability.specializations.append("creative")
            capability.best_use_cases.append("creative_writing")
        
        # Fast/efficient models
        if any(term in combined_text for term in ["fast", "instant", "quick", "lite", "mini"]):
            capability.tokens_per_second = 100.0
            capability.avg_latency_ms = 500
            capability.specializations.append("fast")
            capability.best_use_cases.append("quick_tasks")
        
        # Large context models
        if capability.max_context_length > 32000:
            capability.specializations.append("long_context")
            capability.best_use_cases.append("large_documents")
        
        # Multilingual
        if any(term in combined_text for term in ["multilingual", "translation", "international"]):
            capability.multilingual = 80
            capability.specializations.append("multilingual")
        
        # Set defaults for unscored capabilities
        if capability.code_generation == 0:
            capability.code_generation = 60  # Default moderate capability
        if capability.reasoning == 0:
            capability.reasoning = 65
        if capability.creativity == 0:
            capability.creativity = 60
        if capability.instruction_following == 0:
            capability.instruction_following = 70
    
    async def benchmark_model(
        self,
        capability: ModelCapabilities,
        api_key: Optional[str] = None
    ) -> ModelCapabilities:
        """Benchmark a model's actual performance"""
        
        logger.info(f"Benchmarking {capability.provider}:{capability.model_id}")
        
        try:
            # Test each capability
            for test_name, test_prompt in self.benchmark_tests.items():
                score = await self._run_benchmark_test(
                    capability,
                    test_name,
                    test_prompt,
                    api_key
                )
                
                # Update capability score
                if test_name == "code_generation":
                    capability.code_generation = max(capability.code_generation, score)
                elif test_name == "code_review":
                    capability.code_review = max(capability.code_review, score)
                elif test_name == "reasoning":
                    capability.reasoning = max(capability.reasoning, score)
                elif test_name == "math":
                    capability.math = max(capability.math, score)
                elif test_name == "creativity":
                    capability.creativity = max(capability.creativity, score)
                elif test_name == "instruction_following":
                    capability.instruction_following = max(capability.instruction_following, score)
                elif test_name == "multilingual":
                    capability.multilingual = max(capability.multilingual, score)
            
            # Calculate overall reliability
            capability.reliability_score = (
                capability.code_generation +
                capability.reasoning +
                capability.instruction_following
            ) / 300.0
            
            capability.tested_at = datetime.now().isoformat()
            
        except Exception as e:
            logger.error(f"Benchmark failed for {capability.model_id}: {e}")
            capability.reliability_score = 0.0
        
        return capability
    
    async def _run_benchmark_test(
        self,
        capability: ModelCapabilities,
        test_name: str,
        test_prompt: str,
        api_key: Optional[str]
    ) -> int:
        """Run a single benchmark test and return score (0-100)"""
        
        import time
        
        start_time = time.time()
        
        try:
            # Make API call to model
            response = await self._call_model(
                capability.provider,
                capability.model_id,
                test_prompt,
                api_key
            )
            
            elapsed = time.time() - start_time
            
            # Update latency
            if capability.avg_latency_ms == 0:
                capability.avg_latency_ms = elapsed * 1000
            else:
                capability.avg_latency_ms = (capability.avg_latency_ms + elapsed * 1000) / 2
            
            # Score based on response quality
            score = self._score_response(test_name, test_prompt, response)
            
            return score
            
        except Exception as e:
            logger.debug(f"Test {test_name} failed: {e}")
            return 0
    
    async def _call_model(
        self,
        provider: str,
        model_id: str,
        prompt: str,
        api_key: Optional[str]
    ) -> str:
        """Call a model via its provider API"""
        
        if provider == "openrouter":
            return await self._call_openrouter(model_id, prompt, api_key)
        # Add other providers...
        
        return ""
    
    async def _call_openrouter(
        self,
        model_id: str,
        prompt: str,
        api_key: Optional[str]
    ) -> str:
        """Call OpenRouter API"""
        
        import os
        
        api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        
        if not api_key:
            raise ValueError("OpenRouter API key required")
        
        url = "https://openrouter.ai/api/v1/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 500
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload, timeout=30) as response:
                if response.status != 200:
                    raise Exception(f"API returned {response.status}")
                
                data = await response.json()
                return data["choices"][0]["message"]["content"]
    
    def _score_response(self, test_name: str, prompt: str, response: str) -> int:
        """Score a model's response (0-100)"""
        
        if not response or len(response) < 10:
            return 0
        
        score = 50  # Base score for any response
        
        # Test-specific scoring
        if test_name == "code_generation":
            if "def " in response and "return" in response:
                score += 30
            if '"""' in response or "'''" in response:  # Has docstring
                score += 10
            if "->" in response:  # Has type hints
                score += 10
        
        elif test_name == "instruction_following":
            response_lower = response.lower()
            if "start" in response_lower:
                score += 15
            if "1" in response and "2" in response and "3" in response:
                score += 20
            if "end" in response_lower:
                score += 15
        
        elif test_name == "reasoning":
            if any(word in response.lower() for word in ["yes", "true", "correct"]):
                score += 20
            if any(word in response.lower() for word in ["because", "therefore", "since"]):
                score += 20
        
        elif test_name == "math":
            if "13" in response or "thirteen" in response.lower():
                score += 40
        
        return min(100, score)
    
    async def generate_optimal_config(
        self,
        models: Dict[str, List[ModelCapabilities]]
    ) -> Dict[str, Any]:
        """Generate optimal configuration based on discovered models"""
        
        # Categorize models by capability
        categorized = {
            "code_specialists": [],
            "reasoning_experts": [],
            "fast_models": [],
            "long_context": [],
            "creative_models": [],
            "free_models": [],
            "premium_models": []
        }
        
        for provider, model_list in models.items():
            for model in model_list:
                if model.code_generation >= 80:
                    categorized["code_specialists"].append(model)
                if model.reasoning >= 80:
                    categorized["reasoning_experts"].append(model)
                if model.avg_latency_ms < 1000 and model.avg_latency_ms > 0:
                    categorized["fast_models"].append(model)
                if model.max_context_length > 32000:
                    categorized["long_context"].append(model)
                if model.creativity >= 75:
                    categorized["creative_models"].append(model)
                if model.is_free:
                    categorized["free_models"].append(model)
                else:
                    categorized["premium_models"].append(model)
        
        # Generate routing rules
        routing_config = {
            "task_routing": {
                "code_generation": {
                    "primary": self._get_best_model(categorized["code_specialists"]),
                    "fallbacks": self._get_fallbacks(categorized["code_specialists"], 3)
                },
                "complex_reasoning": {
                    "primary": self._get_best_model(categorized["reasoning_experts"]),
                    "fallbacks": self._get_fallbacks(categorized["reasoning_experts"], 3)
                },
                "quick_tasks": {
                    "primary": self._get_best_model(categorized["fast_models"]),
                    "fallbacks": self._get_fallbacks(categorized["fast_models"], 2)
                },
                "creative_writing": {
                    "primary": self._get_best_model(categorized["creative_models"]),
                    "fallbacks": self._get_fallbacks(categorized["creative_models"], 2)
                }
            },
            "cost_optimization": {
                "prefer_free": True,
                "free_models_count": len(categorized["free_models"]),
                "premium_models_count": len(categorized["premium_models"])
            },
            "statistics": {
                "total_models": sum(len(m) for m in models.values()),
                "free_models": len(categorized["free_models"]),
                "code_specialists": len(categorized["code_specialists"]),
                "reasoning_experts": len(categorized["reasoning_experts"]),
                "fast_models": len(categorized["fast_models"])
            }
        }
        
        return routing_config
    
    def _get_best_model(self, models: List[ModelCapabilities]) -> Optional[str]:
        """Get the best model from a list"""
        if not models:
            return None
        
        # Sort by reliability and free status
        sorted_models = sorted(
            models,
            key=lambda m: (m.is_free, m.reliability_score),
            reverse=True
        )
        
        best = sorted_models[0]
        return f"{best.provider}:{best.model_id}"
    
    def _get_fallbacks(
        self,
        models: List[ModelCapabilities],
        count: int
    ) -> List[str]:
        """Get fallback models"""
        sorted_models = sorted(
            models,
            key=lambda m: (m.is_free, m.reliability_score),
            reverse=True
        )
        
        return [
            f"{m.provider}:{m.model_id}"
            for m in sorted_models[1:count+1]
        ]
    
    # Implement other provider discovery methods...
    async def _discover_gemini(self) -> List[ModelCapabilities]:
        """Discover Gemini models"""
        # Implementation for Gemini
        return []
    
    async def _discover_groq(self) -> List[ModelCapabilities]:
        """Discover Groq models"""
        return []
    
    async def _discover_mistral(self) -> List[ModelCapabilities]:
        """Discover Mistral models"""
        return []
    
    async def _discover_anthropic(self) -> List[ModelCapabilities]:
        """Discover Anthropic models"""
        return []
    
    async def _discover_cohere(self) -> List[ModelCapabilities]:
        """Discover Cohere models"""
        return []
    
    async def _discover_together(self) -> List[ModelCapabilities]:
        """Discover Together AI models"""
        return []
    
    async def _discover_replicate(self) -> List[ModelCapabilities]:
        """Discover Replicate models"""
        return []


# Singleton
_discovery_instance = None

def get_enhanced_discovery() -> EnhancedModelDiscovery:
    global _discovery_instance
    if _discovery_instance is None:
        _discovery_instance = EnhancedModelDiscovery()
    return _discovery_instance


# Usage Example
async def discover_and_configure():
    """Complete discovery and configuration workflow"""
    
    discovery = get_enhanced_discovery()
    
    # Step 1: Discover all models
    print("Discovering models from all providers...")
    all_models = await discovery.discover_all_providers()
    
    # Step 2: Benchmark key models
    print("Benchmarking selected models...")
    for provider, models in all_models.items():
        for model in models[:5]:  # Benchmark top 5 per provider
            await discovery.benchmark_model(model)
    
    # Step 3: Generate optimal configuration
    print("Generating optimal configuration...")
    config = await discovery.generate_optimal_config(all_models)
    
    # Step 4: Save configuration
    import json
    with open("optimal_model_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("✓ Configuration saved to optimal_model_config.json")
    
    return config

if __name__ == "__main__":
    asyncio.run(discover_and_configure())
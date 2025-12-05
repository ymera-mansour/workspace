# YMERA Complete OpenRouter Integration
# Comprehensive integration with ALL free models + premium options

import asyncio
import aiohttp
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import logging
import os

logger = logging.getLogger(__name__)

@dataclass
class OpenRouterModel:
    """OpenRouter model configuration"""
    id: str
    name: str
    context_length: int
    pricing_prompt: float
    pricing_completion: float
    is_free: bool
    top_provider: str
    architecture: Dict[str, Any]
    
    # Inferred capabilities
    specialization: str = "general"
    speed_tier: str = "medium"  # fast, medium, slow
    quality_tier: str = "medium"  # high, medium, low
    
    # Recommended use cases
    best_for: List[str] = None
    
    def __post_init__(self):
        if self.best_for is None:
            self.best_for = []
        
        # Infer specialization
        self._infer_specialization()
    
    def _infer_specialization(self):
        """Infer specialization from model name/architecture"""
        name_lower = self.id.lower()
        
        if "code" in name_lower or "deepseek-coder" in name_lower:
            self.specialization = "code"
            self.best_for = ["code_generation", "code_review"]
        elif "math" in name_lower or "wizardlm" in name_lower:
            self.specialization = "math_reasoning"
            self.best_for = ["math", "reasoning", "analysis"]
        elif "vision" in name_lower or "llava" in name_lower:
            self.specialization = "multimodal"
            self.best_for = ["image_analysis", "visual_qa"]
        elif "chat" in name_lower or "instruct" in name_lower:
            self.specialization = "conversation"
            self.best_for = ["chat", "general_qa"]
        
        # Infer speed from model size/name
        if "mini" in name_lower or "small" in name_lower or "lite" in name_lower:
            self.speed_tier = "fast"
        elif "large" in name_lower or "70b" in name_lower or "405b" in name_lower:
            self.speed_tier = "slow"
        
        # Infer quality
        if "large" in name_lower or self.context_length > 50000:
            self.quality_tier = "high"
        elif "mini" in name_lower or self.context_length < 8000:
            self.quality_tier = "low"


class OpenRouterIntegration:
    """
    Complete OpenRouter integration with:
    - All free models automatically discovered
    - Smart routing based on task requirements
    - Automatic fallback chains
    - Cost optimization
    - Performance tracking
    """
    
    # ALL Known Free Models on OpenRouter (as of Dec 2024)
    KNOWN_FREE_MODELS = [
        # Amazon Nova (NEW!)
        "amazon/nova-2-lite-v1:free",
        "amazon/nova-micro-v1:free",
        
        # Meta Llama
        "meta-llama/llama-3.2-1b-instruct:free",
        "meta-llama/llama-3.2-3b-instruct:free",
        "meta-llama/llama-3.1-8b-instruct:free",
        "meta-llama/llama-3-8b-instruct:free",
        
        # Mistral
        "mistralai/mistral-7b-instruct:free",
        "mistralai/mistral-7b-instruct-v0.3:free",
        "mistralai/mixtral-8x7b-instruct:free",
        
        # Google
        "google/gemma-2-9b-it:free",
        "google/gemma-7b-it:free",
        
        # Microsoft
        "microsoft/phi-3-mini-128k-instruct:free",
        "microsoft/phi-3-medium-128k-instruct:free",
        
        # Qwen
        "qwen/qwen-2-7b-instruct:free",
        "qwen/qwen-2.5-7b-instruct:free",
        
        # HuggingFace
        "huggingfaceh4/zephyr-7b-beta:free",
        
        # OpenChat
        "openchat/openchat-7b:free",
        
        # Nous Research
        "nousresearch/hermes-3-llama-3.1-405b:free",
        
        # Liquid
        "liquid/lfm-40b:free",
        
        # Gryphe
        "gryphe/mythomax-l2-13b:free",
        
        # Toppy
        "undi95/toppy-m-7b:free",
        
        # Code-specific
        "deepseek/deepseek-coder-6.7b-instruct:free",
    ]
    
    def __init__(self):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.base_url = "https://openrouter.ai/api/v1"
        self.models: Dict[str, OpenRouterModel] = {}
        self.routing_cache: Dict[str, str] = {}
    
    async def initialize(self):
        """Initialize by discovering all available models"""
        logger.info("Initializing OpenRouter integration...")
        
        # Fetch latest model list from API
        discovered = await self.discover_models()
        
        # Add known free models that might not be in API
        for model_id in self.KNOWN_FREE_MODELS:
            if model_id not in self.models:
                # Add with default configuration
                self.models[model_id] = OpenRouterModel(
                    id=model_id,
                    name=model_id.split("/")[-1],
                    context_length=8192,
                    pricing_prompt=0.0,
                    pricing_completion=0.0,
                    is_free=True,
                    top_provider="openrouter",
                    architecture={}
                )
        
        logger.info(f"✓ Initialized with {len(self.models)} OpenRouter models")
        logger.info(f"  - Free models: {sum(1 for m in self.models.values() if m.is_free)}")
        
        return self.models
    
    async def discover_models(self) -> Dict[str, OpenRouterModel]:
        """Discover all models from OpenRouter API"""
        
        url = f"{self.base_url}/models"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status != 200:
                        logger.error(f"OpenRouter API returned {response.status}")
                        return {}
                    
                    data = await response.json()
                    
                    for model_data in data.get("data", []):
                        model_id = model_data["id"]
                        pricing = model_data.get("pricing", {})
                        
                        prompt_price = float(pricing.get("prompt", "0"))
                        completion_price = float(pricing.get("completion", "0"))
                        is_free = (prompt_price == 0 and completion_price == 0)
                        
                        model = OpenRouterModel(
                            id=model_id,
                            name=model_data.get("name", model_id),
                            context_length=model_data.get("context_length", 8192),
                            pricing_prompt=prompt_price,
                            pricing_completion=completion_price,
                            is_free=is_free,
                            top_provider=model_data.get("top_provider", {}).get("name", "unknown"),
                            architecture=model_data.get("architecture", {})
                        )
                        
                        self.models[model_id] = model
                    
                    return self.models
        
        except Exception as e:
            logger.error(f"Failed to discover OpenRouter models: {e}")
            return {}
    
    async def select_model_for_task(
        self,
        task_type: str,
        complexity: str = "medium",
        prefer_free: bool = True,
        max_cost_per_1k: float = 0.0
    ) -> Optional[str]:
        """
        Intelligently select the best model for a task
        
        Args:
            task_type: Type of task (code, reasoning, chat, etc.)
            complexity: Task complexity (simple, medium, complex)
            prefer_free: Prefer free models
            max_cost_per_1k: Maximum cost per 1k tokens
        
        Returns:
            Model ID of selected model
        """
        
        # Check cache
        cache_key = f"{task_type}:{complexity}:{prefer_free}"
        if cache_key in self.routing_cache:
            return self.routing_cache[cache_key]
        
        # Filter models
        candidates = []
        
        for model in self.models.values():
            # Cost filter
            if prefer_free and not model.is_free:
                continue
            
            avg_cost = (model.pricing_prompt + model.pricing_completion) / 2
            if avg_cost > max_cost_per_1k:
                continue
            
            # Specialization filter
            if task_type in model.best_for or task_type == model.specialization:
                score = 100
            elif task_type in ["general", "chat"]:
                score = 60
            else:
                score = 40
            
            # Complexity filter
            if complexity == "simple" and model.speed_tier == "fast":
                score += 20
            elif complexity == "complex" and model.quality_tier == "high":
                score += 20
            elif complexity == "medium":
                score += 10
            
            # Context length bonus
            if model.context_length > 32000:
                score += 10
            
            candidates.append((score, model.id))
        
        if not candidates:
            logger.warning("No suitable OpenRouter model found")
            return None
        
        # Select best
        candidates.sort(reverse=True)
        selected = candidates[0][1]
        
        # Cache selection
        self.routing_cache[cache_key] = selected
        
        logger.info(f"Selected {selected} for {task_type} (complexity: {complexity})")
        return selected
    
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        task_type: str = "chat",
        max_tokens: int = 2048,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """
        Make a chat completion request to OpenRouter
        
        Args:
            messages: List of chat messages
            model: Specific model ID (auto-selected if None)
            task_type: Type of task for auto-selection
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional parameters
        
        Returns:
            Generated text response
        """
        
        # Auto-select model if not specified
        if not model:
            model = await self.select_model_for_task(task_type)
            if not model:
                raise ValueError("No suitable model available")
        
        # Prepare request
        url = f"{self.base_url}/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://ymera.ai",  # Optional: Your site URL
            "X-Title": "YMERA AI Platform"  # Optional: Your app name
        }
        
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            **kwargs
        }
        
        # Make request
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload, timeout=60) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"OpenRouter API error {response.status}: {error_text}")
                
                data = await response.json()
                
                # Extract response
                if "choices" in data and len(data["choices"]) > 0:
                    return data["choices"][0]["message"]["content"]
                else:
                    raise Exception("No response from model")
    
    async def get_model_recommendations(self) -> Dict[str, List[str]]:
        """Get model recommendations by category"""
        
        recommendations = {
            "code_generation": [],
            "code_review": [],
            "fast_tasks": [],
            "long_context": [],
            "reasoning": [],
            "creative": [],
            "free_best_quality": []
        }
        
        for model in self.models.values():
            if not model.is_free:
                continue
            
            if model.specialization == "code":
                recommendations["code_generation"].append(model.id)
                recommendations["code_review"].append(model.id)
            
            if model.speed_tier == "fast":
                recommendations["fast_tasks"].append(model.id)
            
            if model.context_length > 32000:
                recommendations["long_context"].append(model.id)
            
            if model.specialization == "math_reasoning":
                recommendations["reasoning"].append(model.id)
            
            if model.quality_tier == "high":
                recommendations["free_best_quality"].append(model.id)
        
        return recommendations
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about available models"""
        
        return {
            "total_models": len(self.models),
            "free_models": sum(1 for m in self.models.values() if m.is_free),
            "paid_models": sum(1 for m in self.models.values() if not m.is_free),
            "specializations": {
                "code": sum(1 for m in self.models.values() if m.specialization == "code"),
                "reasoning": sum(1 for m in self.models.values() if m.specialization == "math_reasoning"),
                "multimodal": sum(1 for m in self.models.values() if m.specialization == "multimodal"),
                "conversation": sum(1 for m in self.models.values() if m.specialization == "conversation"),
            },
            "context_lengths": {
                "small (<16k)": sum(1 for m in self.models.values() if m.context_length < 16000),
                "medium (16k-64k)": sum(1 for m in self.models.values() if 16000 <= m.context_length < 64000),
                "large (64k+)": sum(1 for m in self.models.values() if m.context_length >= 64000),
            }
        }


# Singleton
_openrouter_instance = None

def get_openrouter() -> OpenRouterIntegration:
    global _openrouter_instance
    if _openrouter_instance is None:
        _openrouter_instance = OpenRouterIntegration()
    return _openrouter_instance


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

async def example_usage():
    """Complete usage examples"""
    
    openrouter = get_openrouter()
    await openrouter.initialize()
    
    # Example 1: Auto-select model for code generation
    print("\n=== Example 1: Code Generation ===")
    response = await openrouter.chat_completion(
        messages=[{"role": "user", "content": "Write a Python function to reverse a string"}],
        task_type="code"
    )
    print(response)
    
    # Example 2: Use specific model (Amazon Nova)
    print("\n=== Example 2: Amazon Nova ===")
    response = await openrouter.chat_completion(
        messages=[{"role": "user", "content": "Explain quantum computing in simple terms"}],
        model="amazon/nova-2-lite-v1:free"
    )
    print(response)
    
    # Example 3: Long context task
    print("\n=== Example 3: Long Context ===")
    long_model = await openrouter.select_model_for_task(
        task_type="analysis",
        complexity="complex"
    )
    print(f"Selected: {long_model}")
    
    # Example 4: Get recommendations
    print("\n=== Example 4: Model Recommendations ===")
    recommendations = await openrouter.get_model_recommendations()
    print("Best for code generation:", recommendations["code_generation"][:3])
    print("Best for fast tasks:", recommendations["fast_tasks"][:3])
    
    # Example 5: Statistics
    print("\n=== Example 5: Statistics ===")
    stats = openrouter.get_statistics()
    print(f"Total models: {stats['total_models']}")
    print(f"Free models: {stats['free_models']}")
    print(f"Code specialists: {stats['specializations']['code']}")


if __name__ == "__main__":
    asyncio.run(example_usage())

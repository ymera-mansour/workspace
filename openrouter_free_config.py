"""
OpenRouter FREE Models Configuration System
Implements 8 strategic FREE models for reasoning, testing, security, and code quality.
Integrates with existing multi-provider architecture.

Models included:
- DeepSeek R1: Advanced reasoning
- DeepSeek Chat V3: Large context (163K)
- DeepSeek Coder: Code specialist
- AWS Nova Lite/Micro: Fast processing
- Phi-3 128K: Testing specialist
- Gemma 2 9B: Security specialist
- Llama 3.2 3B: Quick validation

All models are 100% FREE with no credit cards required.
"""

import os
import time
import json
import asyncio
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import hashlib


# ============================================================================
# FREE Model Endpoints Configuration
# ============================================================================

OPENROUTER_FREE_ENDPOINTS = {
    "deepseek/deepseek-r1:free": {
        "endpoint": "https://openrouter.ai/api/v1/chat/completions",
        "context_window": 64000,
        "rpm_limit": 20,
        "rpd_limit": 200,
        "supports_streaming": True,
        "supports_function_calling": False,
        "best_for": ["reasoning", "analysis", "threat_modeling", "security"],
        "cost": "$0 FREE",
        "description": "Best FREE reasoning model, advanced analysis capabilities"
    },
    "deepseek/deepseek-chat-v3-0324:free": {
        "endpoint": "https://openrouter.ai/api/v1/chat/completions",
        "context_window": 163000,  # Largest FREE context
        "rpm_limit": 20,
        "rpd_limit": 200,
        "supports_streaming": True,
        "supports_function_calling": False,
        "best_for": ["complex_tasks", "large_codebase", "architecture"],
        "cost": "$0 FREE",
        "description": "Largest FREE context window (163K tokens)"
    },
    "deepseek/deepseek-coder-6.7b-instruct:free": {
        "endpoint": "https://openrouter.ai/api/v1/chat/completions",
        "context_window": 16000,
        "rpm_limit": 20,
        "rpd_limit": 200,
        "supports_streaming": True,
        "supports_function_calling": False,
        "best_for": ["code_generation", "code_review", "refactoring"],
        "cost": "$0 FREE",
        "description": "FREE code specialist, optimized for programming"
    },
    "aws/amazon-nova-lite-v1:free": {
        "endpoint": "https://openrouter.ai/api/v1/chat/completions",
        "context_window": 300000,  # 300K context
        "rpm_limit": 20,
        "rpd_limit": 200,
        "supports_streaming": True,
        "supports_function_calling": False,
        "best_for": ["fast_processing", "bulk_tasks", "monitoring"],
        "cost": "$0 FREE",
        "description": "Latest AWS model, fast and efficient"
    },
    "aws/amazon-nova-micro-v1:free": {
        "endpoint": "https://openrouter.ai/api/v1/chat/completions",
        "context_window": 128000,
        "rpm_limit": 20,
        "rpd_limit": 200,
        "supports_streaming": True,
        "supports_function_calling": False,
        "best_for": ["ultra_fast", "validation", "quick_checks"],
        "cost": "$0 FREE",
        "description": "Ultra-lightweight AWS model, fastest processing"
    },
    "microsoft/phi-3-mini-128k-instruct:free": {
        "endpoint": "https://openrouter.ai/api/v1/chat/completions",
        "context_window": 128000,
        "rpm_limit": 20,
        "rpd_limit": 200,
        "supports_streaming": True,
        "supports_function_calling": False,
        "best_for": ["testing", "test_generation", "validation", "qa"],
        "cost": "$0 FREE",
        "description": "Dedicated testing specialist, 128K context for large test suites"
    },
    "google/gemma-2-9b-it:free": {
        "endpoint": "https://openrouter.ai/api/v1/chat/completions",
        "context_window": 8192,
        "rpm_limit": 20,
        "rpd_limit": 200,
        "supports_streaming": True,
        "supports_function_calling": False,
        "best_for": ["security", "vulnerability_scan", "threat_analysis", "auth"],
        "cost": "$0 FREE",
        "description": "Security specialist, strong reasoning for threat detection"
    },
    "meta-llama/llama-3.2-3b-instruct:free": {
        "endpoint": "https://openrouter.ai/api/v1/chat/completions",
        "context_window": 8192,
        "rpm_limit": 20,
        "rpd_limit": 200,
        "supports_streaming": True,
        "supports_function_calling": False,
        "best_for": ["quick_validation", "bulk_operations", "simple_tasks"],
        "cost": "$0 FREE",
        "description": "Lightweight model for quick validation tasks"
    }
}


# ============================================================================
# Enums and Data Classes
# ============================================================================

class RotationStrategy(Enum):
    """Key rotation strategies"""
    ROUND_ROBIN = "round_robin"
    LEAST_USED = "least_used"
    WEIGHTED = "weighted"


@dataclass
class APIKey:
    """API key with usage tracking"""
    key: str
    organization: str
    requests_per_minute: int = 0
    requests_per_day: int = 0
    total_requests: int = 0
    last_used: Optional[datetime] = None
    last_rpm_reset: datetime = field(default_factory=datetime.now)
    last_rpd_reset: datetime = field(default_factory=datetime.now)
    health_status: str = "healthy"
    
    def reset_counters(self):
        """Reset RPM/RPD counters if needed"""
        now = datetime.now()
        if (now - self.last_rpm_reset).seconds >= 60:
            self.requests_per_minute = 0
            self.last_rpm_reset = now
        if (now - self.last_rpd_reset).seconds >= 86400:
            self.requests_per_day = 0
            self.last_rpd_reset = now


@dataclass
class TaskRecord:
    """Record of a task execution"""
    task_id: str
    agent_name: str
    task_type: str
    model_used: str
    start_time: datetime
    end_time: Optional[datetime] = None
    success: bool = False
    latency_ms: int = 0
    error: Optional[str] = None


# ============================================================================
# API Key Rotation Manager
# ============================================================================

class APIKeyRotationManager:
    """
    Manages multiple OpenRouter API keys with intelligent rotation.
    Supports unlimited keys from different organizations.
    """
    
    def __init__(self, strategy: RotationStrategy = RotationStrategy.LEAST_USED):
        self.strategy = strategy
        self.keys: List[APIKey] = []
        self.current_index = 0
        self._load_keys_from_env()
    
    def _load_keys_from_env(self):
        """Load all OpenRouter API keys from environment variables"""
        i = 1
        while True:
            key = os.getenv(f"OPENROUTER_API_KEY_{i}")
            org = os.getenv(f"OPENROUTER_ORG_{i}", f"org_{i}")
            
            if not key:
                break
            
            self.keys.append(APIKey(key=key, organization=org))
            i += 1
        
        # Load default key if no numbered keys found
        if not self.keys:
            default_key = os.getenv("OPENROUTER_API_KEY")
            if default_key:
                self.keys.append(APIKey(key=default_key, organization="default"))
    
    def get_next_key(self) -> Optional[APIKey]:
        """Get next API key based on rotation strategy"""
        if not self.keys:
            return None
        
        # Reset counters for all keys
        for key in self.keys:
            key.reset_counters()
        
        if self.strategy == RotationStrategy.ROUND_ROBIN:
            key = self.keys[self.current_index]
            self.current_index = (self.current_index + 1) % len(self.keys)
            return key
        
        elif self.strategy == RotationStrategy.LEAST_USED:
            # Return key with lowest total requests
            return min(self.keys, key=lambda k: k.total_requests)
        
        elif self.strategy == RotationStrategy.WEIGHTED:
            # Return key with most available quota
            available = []
            for key in self.keys:
                rpm_available = 20 - key.requests_per_minute
                rpd_available = 200 - key.requests_per_day
                score = rpm_available + (rpd_available / 10)
                available.append((key, score))
            return max(available, key=lambda x: x[1])[0]
        
        return self.keys[0]
    
    def record_request(self, key: APIKey):
        """Record a request for the given key"""
        key.requests_per_minute += 1
        key.requests_per_day += 1
        key.total_requests += 1
        key.last_used = datetime.now()
    
    def add_key(self, key: str, organization: str):
        """Add a new API key"""
        self.keys.append(APIKey(key=key, organization=organization))
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of all keys"""
        return {
            "total_keys": len(self.keys),
            "strategy": self.strategy.value,
            "keys": [
                {
                    "organization": key.organization,
                    "rpm": key.requests_per_minute,
                    "rpd": key.requests_per_day,
                    "total": key.total_requests,
                    "health": key.health_status
                }
                for key in self.keys
            ]
        }


# ============================================================================
# Agent Training System
# ============================================================================

class AgentTrainingSystem:
    """
    Learns optimal models for each agent based on historical performance.
    Tracks success rate, latency, and quality scores.
    """
    
    def __init__(self, data_file: str = "openrouter_agent_training_data.json"):
        self.data_file = data_file
        self.training_data: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self._load_training_data()
    
    def _load_training_data(self):
        """Load training data from disk"""
        if os.path.exists(self.data_file):
            with open(self.data_file, 'r') as f:
                self.training_data = json.load(f)
    
    def _save_training_data(self):
        """Save training data to disk"""
        with open(self.data_file, 'w') as f:
            json.dump(self.training_data, f, indent=2)
    
    def record_execution(
        self,
        agent_name: str,
        model: str,
        task_type: str,
        success: bool,
        latency_ms: int,
        quality_score: float = 0.0
    ):
        """Record execution results for training"""
        if agent_name not in self.training_data:
            self.training_data[agent_name] = {}
        
        if model not in self.training_data[agent_name]:
            self.training_data[agent_name][model] = {
                "total_executions": 0,
                "successful_executions": 0,
                "total_latency_ms": 0,
                "total_quality_score": 0,
                "task_types": {}
            }
        
        model_data = self.training_data[agent_name][model]
        model_data["total_executions"] += 1
        if success:
            model_data["successful_executions"] += 1
        model_data["total_latency_ms"] += latency_ms
        model_data["total_quality_score"] += quality_score
        
        # Track by task type
        if task_type not in model_data["task_types"]:
            model_data["task_types"][task_type] = 0
        model_data["task_types"][task_type] += 1
        
        self._save_training_data()
    
    def get_recommended_model(
        self,
        agent_name: str,
        task_type: Optional[str] = None
    ) -> Optional[str]:
        """Get recommended model for agent based on training data"""
        if agent_name not in self.training_data:
            return None
        
        agent_data = self.training_data[agent_name]
        if not agent_data:
            return None
        
        # Calculate score for each model
        scores = {}
        for model, data in agent_data.items():
            if data["total_executions"] == 0:
                continue
            
            success_rate = data["successful_executions"] / data["total_executions"]
            avg_latency = data["total_latency_ms"] / data["total_executions"]
            avg_quality = data["total_quality_score"] / data["total_executions"]
            
            # Normalize and combine scores
            latency_score = 1.0 / (1.0 + (avg_latency / 1000.0))
            score = (success_rate * 0.5) + (latency_score * 0.3) + (avg_quality * 0.2)
            
            scores[model] = score
        
        if not scores:
            return None
        
        return max(scores.items(), key=lambda x: x[1])[0]
    
    def get_statistics(self, agent_name: Optional[str] = None) -> Dict[str, Any]:
        """Get training statistics"""
        if agent_name:
            return self.training_data.get(agent_name, {})
        return self.training_data


# ============================================================================
# Task-Model Tracking
# ============================================================================

class TaskModelTracker:
    """
    Tracks which models are handling which tasks in real-time.
    Provides visibility into model usage and performance.
    """
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.active_tasks: Dict[str, TaskRecord] = {}
        self.completed_tasks: List[TaskRecord] = []
    
    def start_task(
        self,
        task_id: str,
        agent_name: str,
        task_type: str,
        model: str
    ) -> TaskRecord:
        """Start tracking a task"""
        record = TaskRecord(
            task_id=task_id,
            agent_name=agent_name,
            task_type=task_type,
            model_used=model,
            start_time=datetime.now()
        )
        self.active_tasks[task_id] = record
        return record
    
    def complete_task(
        self,
        task_id: str,
        success: bool,
        error: Optional[str] = None
    ):
        """Complete a task"""
        if task_id not in self.active_tasks:
            return
        
        record = self.active_tasks.pop(task_id)
        record.end_time = datetime.now()
        record.success = success
        record.error = error
        record.latency_ms = int((record.end_time - record.start_time).total_seconds() * 1000)
        
        self.completed_tasks.append(record)
        
        # Keep only last N tasks
        if len(self.completed_tasks) > self.max_history:
            self.completed_tasks = self.completed_tasks[-self.max_history:]
    
    def get_active_tasks(self) -> List[Dict[str, Any]]:
        """Get all active tasks"""
        return [
            {
                "task_id": record.task_id,
                "agent": record.agent_name,
                "task_type": record.task_type,
                "model": record.model_used,
                "duration_ms": int((datetime.now() - record.start_time).total_seconds() * 1000)
            }
            for record in self.active_tasks.values()
        ]
    
    def get_model_usage(self) -> Dict[str, int]:
        """Get usage count per model"""
        usage = {}
        for record in self.completed_tasks:
            usage[record.model_used] = usage.get(record.model_used, 0) + 1
        return usage
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        total_tasks = len(self.completed_tasks)
        successful_tasks = sum(1 for r in self.completed_tasks if r.success)
        
        return {
            "active_tasks": len(self.active_tasks),
            "total_completed": total_tasks,
            "successful": successful_tasks,
            "success_rate": successful_tasks / total_tasks if total_tasks > 0 else 0,
            "model_usage": self.get_model_usage(),
            "avg_latency_ms": sum(r.latency_ms for r in self.completed_tasks) / total_tasks if total_tasks > 0 else 0
        }


# ============================================================================
# Integrated Middleware
# ============================================================================

class OpenRouterMiddleware:
    """
    Unified middleware that integrates all components:
    - API key rotation
    - Agent training
    - Task tracking
    - Model selection
    """
    
    def __init__(
        self,
        rotation_strategy: RotationStrategy = RotationStrategy.LEAST_USED
    ):
        self.key_manager = APIKeyRotationManager(rotation_strategy)
        self.training_system = AgentTrainingSystem()
        self.task_tracker = TaskModelTracker()
    
    async def execute_request(
        self,
        agent_name: str,
        task_type: str,
        task_description: str,
        preferred_model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute a request with automatic model selection and tracking.
        
        Args:
            agent_name: Name of the agent making the request
            task_type: Type of task (e.g., "reasoning", "code_generation")
            task_description: Description of the task
            preferred_model: Optional preferred model to use
        
        Returns:
            Dict with results and metadata
        """
        # Generate task ID
        task_id = hashlib.md5(
            f"{agent_name}_{task_type}_{time.time()}".encode()
        ).hexdigest()
        
        # Select model
        if preferred_model and preferred_model in OPENROUTER_FREE_ENDPOINTS:
            model = preferred_model
        else:
            # Try to get recommended model from training
            model = self.training_system.get_recommended_model(agent_name, task_type)
            
            # If no recommendation, select based on task type
            if not model:
                model = self._select_model_by_task_type(task_type)
        
        # Get API key
        api_key = self.key_manager.get_next_key()
        if not api_key:
            return {
                "success": False,
                "error": "No API keys available",
                "model_used": None
            }
        
        # Start tracking
        self.task_tracker.start_task(task_id, agent_name, task_type, model)
        
        start_time = time.time()
        
        try:
            # Make request (placeholder - implement actual API call)
            result = await self._make_api_request(
                model=model,
                api_key=api_key.key,
                task_description=task_description
            )
            
            # Calculate metrics
            latency_ms = int((time.time() - start_time) * 1000)
            
            # Record success
            self.key_manager.record_request(api_key)
            self.task_tracker.complete_task(task_id, success=True)
            self.training_system.record_execution(
                agent_name=agent_name,
                model=model,
                task_type=task_type,
                success=True,
                latency_ms=latency_ms,
                quality_score=0.9  # Placeholder
            )
            
            return {
                "success": True,
                "result": result,
                "model_used": model,
                "latency_ms": latency_ms,
                "task_id": task_id
            }
        
        except Exception as e:
            # Record failure
            self.task_tracker.complete_task(task_id, success=False, error=str(e))
            self.training_system.record_execution(
                agent_name=agent_name,
                model=model,
                task_type=task_type,
                success=False,
                latency_ms=int((time.time() - start_time) * 1000),
                quality_score=0.0
            )
            
            return {
                "success": False,
                "error": str(e),
                "model_used": model
            }
    
    def _select_model_by_task_type(self, task_type: str) -> str:
        """Select best model based on task type"""
        task_to_model = {
            "reasoning": "deepseek/deepseek-r1:free",
            "analysis": "deepseek/deepseek-r1:free",
            "security": "google/gemma-2-9b-it:free",
            "vulnerability_scan": "google/gemma-2-9b-it:free",
            "threat_analysis": "google/gemma-2-9b-it:free",
            "code_generation": "deepseek/deepseek-coder-6.7b-instruct:free",
            "code_review": "deepseek/deepseek-coder-6.7b-instruct:free",
            "refactoring": "deepseek/deepseek-coder-6.7b-instruct:free",
            "testing": "microsoft/phi-3-mini-128k-instruct:free",
            "test_generation": "microsoft/phi-3-mini-128k-instruct:free",
            "validation": "microsoft/phi-3-mini-128k-instruct:free",
            "architecture": "deepseek/deepseek-chat-v3-0324:free",
            "complex_tasks": "deepseek/deepseek-chat-v3-0324:free",
            "quick_validation": "meta-llama/llama-3.2-3b-instruct:free",
            "bulk_operations": "aws/amazon-nova-lite-v1:free",
            "fast_processing": "aws/amazon-nova-micro-v1:free"
        }
        
        return task_to_model.get(task_type, "deepseek/deepseek-chat-v3-0324:free")
    
    async def _make_api_request(
        self,
        model: str,
        api_key: str,
        task_description: str
    ) -> str:
        """Make actual API request (placeholder)"""
        # This is a placeholder - implement actual OpenRouter API call
        await asyncio.sleep(0.1)  # Simulate API call
        return f"Response for: {task_description}"
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data"""
        return {
            "api_keys": self.key_manager.get_status(),
            "task_statistics": self.task_tracker.get_statistics(),
            "active_tasks": self.task_tracker.get_active_tasks(),
            "agent_statistics": {
                "total_agents": len(self.training_system.training_data),
                "trained_agents": sum(
                    1 for data in self.training_system.training_data.values()
                    if any(model_data["total_executions"] > 10 for model_data in data.values())
                )
            }
        }


# ============================================================================
# Utility Functions
# ============================================================================

def get_model_for_task(task_type: str) -> str:
    """Get recommended model for a specific task type"""
    middleware = OpenRouterMiddleware()
    return middleware._select_model_by_task_type(task_type)


def get_all_free_models() -> List[str]:
    """Get list of all available FREE models"""
    return list(OPENROUTER_FREE_ENDPOINTS.keys())


def get_model_capabilities(model: str) -> Dict[str, Any]:
    """Get capabilities for a specific model"""
    return OPENROUTER_FREE_ENDPOINTS.get(model, {})


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Example: Initialize middleware
    middleware = OpenRouterMiddleware()
    
    # Example: Execute a security scan
    async def example_security_scan():
        result = await middleware.execute_request(
            agent_name="security_agent",
            task_type="vulnerability_scan",
            task_description="Scan codebase for SQL injection vulnerabilities"
        )
        print(f"Security scan result: {result}")
    
    # Example: Generate tests
    async def example_test_generation():
        result = await middleware.execute_request(
            agent_name="testing_agent",
            task_type="test_generation",
            task_description="Generate unit tests for authentication module"
        )
        print(f"Test generation result: {result}")
    
    # Run examples
    asyncio.run(example_security_scan())
    asyncio.run(example_test_generation())
    
    # Show dashboard
    dashboard = middleware.get_dashboard_data()
    print(json.dumps(dashboard, indent=2))

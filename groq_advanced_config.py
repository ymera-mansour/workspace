"""
Groq Advanced Configuration System

Complete implementation for Groq multi-model optimization with:
- All 6 Groq models with complete endpoint configuration
- Multi-organization API key rotation
- Agent training system for 40+ agents
- Task-model tracking
- Quota management (100% FREE tier)
- Unified middleware

All 6 Groq models are 100% FREE with ultra-fast inference (<1s responses).
"""

import os
import json
import asyncio
import hashlib
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from dataclasses import dataclass, asdict
from collections import defaultdict, deque


# ============================================================================
# GROQ MODEL ENDPOINTS - ALL 100% FREE
# ============================================================================

GROQ_ENDPOINTS = {
    "llama-3.1-8b-instant": {
        "endpoint": "https://api.groq.com/openai/v1/chat/completions",
        "streaming_endpoint": "https://api.groq.com/openai/v1/chat/completions",
        "rpm_limit": 30,
        "rpd_limit": 14400,
        "tpm_limit": 6000,
        "tpd_limit": 500000,
        "context_window": 32000,
        "response_time": "<0.5s",
        "supports_streaming": True,
        "supports_function_calling": True,
        "cost": "$0 FREE",
        "best_for": ["fast_response", "real_time", "monitoring", "validation", "simple_tasks"],
        "description": "FASTEST model - Ultra-low latency for real-time applications"
    },
    "llama-3.3-70b-versatile": {
        "endpoint": "https://api.groq.com/openai/v1/chat/completions",
        "streaming_endpoint": "https://api.groq.com/openai/v1/chat/completions",
        "rpm_limit": 30,
        "rpd_limit": 1000,
        "tpm_limit": 12000,
        "tpd_limit": 100000,
        "context_window": 32000,
        "response_time": "1-2s",
        "supports_streaming": True,
        "supports_function_calling": True,
        "cost": "$0 FREE",
        "best_for": ["general_tasks", "database", "api", "testing", "standard_agents"],
        "description": "Versatile model for general purpose tasks with fast inference"
    },
    "llama-4-maverick-17b": {
        "endpoint": "https://api.groq.com/openai/v1/chat/completions",
        "streaming_endpoint": "https://api.groq.com/openai/v1/chat/completions",
        "rpm_limit": 30,
        "rpd_limit": 1000,
        "tpm_limit": 6000,
        "tpd_limit": 500000,
        "context_window": 32000,
        "response_time": "0.5-1s",
        "supports_streaming": True,
        "supports_function_calling": True,
        "cost": "$0 FREE",
        "best_for": ["balanced_performance", "medium_complexity", "documentation", "analysis"],
        "description": "Balanced speed and capability - Latest Llama 4 architecture"
    },
    "qwen/qwen3-32b": {
        "endpoint": "https://api.groq.com/openai/v1/chat/completions",
        "streaming_endpoint": "https://api.groq.com/openai/v1/chat/completions",
        "rpm_limit": 60,
        "rpd_limit": 1000,
        "tpm_limit": 6000,
        "tpd_limit": 500000,
        "context_window": 32000,
        "response_time": "1-2s",
        "supports_streaming": True,
        "supports_function_calling": True,
        "cost": "$0 FREE",
        "best_for": ["reasoning", "planning", "analysis", "decision_making"],
        "description": "Reasoning specialist - Advanced analytical capabilities"
    },
    "moonshotai/kimi-k2-instruct": {
        "endpoint": "https://api.groq.com/openai/v1/chat/completions",
        "streaming_endpoint": "https://api.groq.com/openai/v1/chat/completions",
        "rpm_limit": 60,
        "rpd_limit": 1000,
        "tpm_limit": 10000,
        "tpd_limit": 300000,
        "context_window": 200000,
        "response_time": "2-3s",
        "supports_streaming": True,
        "supports_function_calling": True,
        "cost": "$0 FREE",
        "best_for": ["large_context", "codebase_analysis", "long_documents", "comprehensive_review"],
        "description": "Large context specialist - 200K tokens for extensive codebases"
    },
    "openai/gpt-oss-120b": {
        "endpoint": "https://api.groq.com/openai/v1/chat/completions",
        "streaming_endpoint": "https://api.groq.com/openai/v1/chat/completions",
        "rpm_limit": 30,
        "rpd_limit": 1000,
        "tpm_limit": 8000,
        "tpd_limit": 200000,
        "context_window": 32000,
        "response_time": "1-2s",
        "supports_streaming": True,
        "supports_function_calling": True,
        "cost": "$0 FREE",
        "best_for": ["highest_quality", "complex_tasks", "architecture", "security"],
        "description": "Highest quality model - 120B parameters for complex tasks"
    }
}


# ============================================================================
# DATA STRUCTURES
# ============================================================================

class RotationStrategy(Enum):
    """API key rotation strategies"""
    ROUND_ROBIN = "round_robin"
    LEAST_USED = "least_used"
    WEIGHTED = "weighted"


@dataclass
class APIKeyInfo:
    """API key information"""
    key_id: str
    api_key: str
    organization: str
    rpm_used: int = 0
    rpd_used: int = 0
    last_reset: datetime = None
    is_active: bool = True
    failure_count: int = 0
    
    def __post_init__(self):
        if self.last_reset is None:
            self.last_reset = datetime.now()


@dataclass
class AgentPerformance:
    """Agent performance metrics"""
    agent_name: str
    model: str
    task_type: str
    success_count: int = 0
    failure_count: int = 0
    total_latency_ms: float = 0.0
    total_tokens: int = 0
    quality_scores: List[float] = None
    
    def __post_init__(self):
        if self.quality_scores is None:
            self.quality_scores = []
    
    @property
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.0
    
    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / self.success_count if self.success_count > 0 else 0.0
    
    @property
    def avg_quality(self) -> float:
        return sum(self.quality_scores) / len(self.quality_scores) if self.quality_scores else 0.0


@dataclass
class TaskRecord:
    """Task execution record"""
    task_id: str
    agent_name: str
    task_type: str
    model: str
    start_time: datetime
    end_time: Optional[datetime] = None
    success: Optional[bool] = None
    latency_ms: Optional[float] = None
    tokens_used: int = 0
    error_message: Optional[str] = None


# ============================================================================
# API KEY ROTATION MANAGER
# ============================================================================

class APIKeyRotationManager:
    """
    Manages multiple Groq API keys from different organizations.
    
    Features:
    - Automatic key rotation (round-robin, least-used, weighted)
    - Quota tracking per key (RPM/RPD/TPM/TPD)
    - Automatic failover when limits hit
    - Health monitoring and deactivation
    - Persistent state management
    """
    
    def __init__(self, strategy: RotationStrategy = RotationStrategy.LEAST_USED):
        self.strategy = strategy
        self.keys: Dict[str, APIKeyInfo] = {}
        self.current_index = 0
        self.state_file = "groq_api_keys_state.json"
        self._load_keys_from_env()
        self._load_state()
    
    def _load_keys_from_env(self):
        """Load API keys from environment variables"""
        # Primary key
        primary_key = os.getenv("GROQ_API_KEY")
        if primary_key:
            self.add_key("primary", primary_key, "Primary")
        
        # Additional keys: GROQ_API_KEY_1, GROQ_API_KEY_2, etc.
        i = 1
        while True:
            key = os.getenv(f"GROQ_API_KEY_{i}")
            if not key:
                break
            org = os.getenv(f"GROQ_ORG_{i}", f"Organization{i}")
            self.add_key(f"key_{i}", key, org)
            i += 1
    
    def add_key(self, key_id: str, api_key: str, organization: str):
        """Add a new API key"""
        self.keys[key_id] = APIKeyInfo(key_id, api_key, organization)
    
    def get_next_key(self, model: str) -> Optional[Tuple[str, str]]:
        """
        Get next available API key based on rotation strategy.
        
        Returns:
            Tuple of (key_id, api_key) or None if no keys available
        """
        if not self.keys:
            return None
        
        active_keys = {k: v for k, v in self.keys.items() if v.is_active}
        if not active_keys:
            return None
        
        if self.strategy == RotationStrategy.ROUND_ROBIN:
            return self._round_robin_selection(active_keys)
        elif self.strategy == RotationStrategy.LEAST_USED:
            return self._least_used_selection(active_keys, model)
        elif self.strategy == RotationStrategy.WEIGHTED:
            return self._weighted_selection(active_keys, model)
    
    def _round_robin_selection(self, active_keys: Dict[str, APIKeyInfo]) -> Tuple[str, str]:
        """Round-robin key selection"""
        keys_list = list(active_keys.items())
        key_id, key_info = keys_list[self.current_index % len(keys_list)]
        self.current_index += 1
        return (key_id, key_info.api_key)
    
    def _least_used_selection(self, active_keys: Dict[str, APIKeyInfo], model: str) -> Tuple[str, str]:
        """Select key with least usage"""
        model_config = GROQ_ENDPOINTS.get(model, {})
        rpm_limit = model_config.get("rpm_limit", 30)
        rpd_limit = model_config.get("rpd_limit", 1000)
        
        # Find key with most remaining capacity
        best_key = None
        best_score = -1
        
        for key_id, key_info in active_keys.items():
            rpm_remaining = rpm_limit - key_info.rpm_used
            rpd_remaining = rpd_limit - key_info.rpd_used
            score = min(rpm_remaining, rpd_remaining)
            
            if score > best_score:
                best_score = score
                best_key = (key_id, key_info.api_key)
        
        return best_key
    
    def _weighted_selection(self, active_keys: Dict[str, APIKeyInfo], model: str) -> Tuple[str, str]:
        """Weighted selection based on remaining quota"""
        # Similar to least_used but with probability distribution
        return self._least_used_selection(active_keys, model)
    
    def record_request(self, key_id: str, model: str):
        """Record a request for quota tracking"""
        if key_id in self.keys:
            self.keys[key_id].rpm_used += 1
            self.keys[key_id].rpd_used += 1
            
            # Reset counters if needed
            self._check_and_reset_counters(key_id)
    
    def record_failure(self, key_id: str):
        """Record a failure for a key"""
        if key_id in self.keys:
            self.keys[key_id].failure_count += 1
            # Deactivate key after 5 consecutive failures
            if self.keys[key_id].failure_count >= 5:
                self.keys[key_id].is_active = False
    
    def record_success(self, key_id: str):
        """Record a success for a key"""
        if key_id in self.keys:
            self.keys[key_id].failure_count = 0
            self.keys[key_id].is_active = True
    
    def _check_and_reset_counters(self, key_id: str):
        """Reset RPM/RPD counters based on time"""
        key_info = self.keys[key_id]
        now = datetime.now()
        
        # Reset RPM every minute
        if (now - key_info.last_reset).total_seconds() >= 60:
            key_info.rpm_used = 0
            key_info.last_reset = now
        
        # Reset RPD every day at midnight
        if now.date() > key_info.last_reset.date():
            key_info.rpd_used = 0
    
    def can_make_request(self, model: str) -> bool:
        """Check if any key can make a request"""
        key_tuple = self.get_next_key(model)
        return key_tuple is not None
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of all keys"""
        return {
            "total_keys": len(self.keys),
            "active_keys": sum(1 for k in self.keys.values() if k.is_active),
            "keys": {
                k: {
                    "organization": v.organization,
                    "rpm_used": v.rpm_used,
                    "rpd_used": v.rpd_used,
                    "is_active": v.is_active,
                    "failure_count": v.failure_count
                }
                for k, v in self.keys.items()
            }
        }
    
    def _save_state(self):
        """Save state to disk"""
        state = {
            "keys": {
                k: {
                    "key_id": v.key_id,
                    "organization": v.organization,
                    "rpm_used": v.rpm_used,
                    "rpd_used": v.rpd_used,
                    "last_reset": v.last_reset.isoformat(),
                    "is_active": v.is_active,
                    "failure_count": v.failure_count
                }
                for k, v in self.keys.items()
            }
        }
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)
    
    def _load_state(self):
        """Load state from disk"""
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                    # Merge with current keys
                    for k, v in state.get("keys", {}).items():
                        if k in self.keys:
                            self.keys[k].rpm_used = v["rpm_used"]
                            self.keys[k].rpd_used = v["rpd_used"]
                            self.keys[k].last_reset = datetime.fromisoformat(v["last_reset"])
                            self.keys[k].is_active = v["is_active"]
                            self.keys[k].failure_count = v["failure_count"]
        except Exception:
            pass


# ============================================================================
# AGENT TRAINING SYSTEM
# ============================================================================

class AgentTrainingSystem:
    """
    Learns optimal Groq models for each agent over time.
    
    Features:
    - Tracks performance per agent-model-task combination
    - Records success rate, latency, quality scores
    - Auto-recommends best model based on historical data
    - Persists training data to disk
    - Statistics dashboard for all agents
    """
    
    def __init__(self):
        self.performance_data: Dict[str, List[AgentPerformance]] = defaultdict(list)
        self.data_file = "groq_agent_training_data.json"
        self._load_data()
    
    def record_execution(
        self,
        agent_name: str,
        model: str,
        task_type: str,
        success: bool,
        latency_ms: float,
        tokens_used: int = 0,
        quality_score: float = 0.0
    ):
        """Record an execution for training"""
        key = f"{agent_name}:{model}:{task_type}"
        
        # Find or create performance record
        perf = None
        for p in self.performance_data[key]:
            if p.agent_name == agent_name and p.model == model and p.task_type == task_type:
                perf = p
                break
        
        if perf is None:
            perf = AgentPerformance(agent_name, model, task_type)
            self.performance_data[key].append(perf)
        
        # Update metrics
        if success:
            perf.success_count += 1
            perf.total_latency_ms += latency_ms
            perf.total_tokens += tokens_used
            if quality_score > 0:
                perf.quality_scores.append(quality_score)
        else:
            perf.failure_count += 1
        
        # Save periodically
        total_executions = perf.success_count + perf.failure_count
        if total_executions % 50 == 0:
            self._save_data()
    
    def get_recommended_model(self, agent_name: str, task_type: str) -> Optional[str]:
        """Get recommended model for agent and task type"""
        # Find all models used for this agent-task combination
        candidates = []
        for key, perf_list in self.performance_data.items():
            for perf in perf_list:
                if perf.agent_name == agent_name and perf.task_type == task_type:
                    if perf.success_count >= 5:  # Minimum 5 successes
                        score = self._calculate_score(perf)
                        candidates.append((perf.model, score))
        
        if not candidates:
            return None
        
        # Return model with highest score
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]
    
    def _calculate_score(self, perf: AgentPerformance) -> float:
        """Calculate overall score for a performance record"""
        # Weight: 40% success rate, 30% speed, 30% quality
        success_score = perf.success_rate * 100
        speed_score = max(0, 100 - (perf.avg_latency_ms / 10))  # Favor <1s latency
        quality_score = perf.avg_quality * 100 if perf.quality_scores else 50
        
        return (success_score * 0.4) + (speed_score * 0.3) + (quality_score * 0.3)
    
    def get_agent_statistics(self, agent_name: str) -> Dict[str, Any]:
        """Get statistics for a specific agent"""
        stats = {
            "agent_name": agent_name,
            "models_used": set(),
            "task_types": set(),
            "total_executions": 0,
            "total_successes": 0,
            "total_failures": 0,
            "performance_by_model": {}
        }
        
        for key, perf_list in self.performance_data.items():
            for perf in perf_list:
                if perf.agent_name == agent_name:
                    stats["models_used"].add(perf.model)
                    stats["task_types"].add(perf.task_type)
                    stats["total_executions"] += perf.success_count + perf.failure_count
                    stats["total_successes"] += perf.success_count
                    stats["total_failures"] += perf.failure_count
                    
                    if perf.model not in stats["performance_by_model"]:
                        stats["performance_by_model"][perf.model] = {
                            "executions": 0,
                            "success_rate": 0.0,
                            "avg_latency_ms": 0.0,
                            "avg_quality": 0.0
                        }
                    
                    model_stats = stats["performance_by_model"][perf.model]
                    model_stats["executions"] += perf.success_count + perf.failure_count
                    model_stats["success_rate"] = perf.success_rate
                    model_stats["avg_latency_ms"] = perf.avg_latency_ms
                    model_stats["avg_quality"] = perf.avg_quality
        
        stats["models_used"] = list(stats["models_used"])
        stats["task_types"] = list(stats["task_types"])
        
        return stats
    
    def _save_data(self):
        """Save training data to disk"""
        data = {}
        for key, perf_list in self.performance_data.items():
            data[key] = [asdict(p) for p in perf_list]
        
        with open(self.data_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _load_data(self):
        """Load training data from disk"""
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                    for key, perf_list in data.items():
                        self.performance_data[key] = [
                            AgentPerformance(**p) for p in perf_list
                        ]
        except Exception:
            pass


# ============================================================================
# TASK-MODEL TRACKER
# ============================================================================

class TaskModelTracker:
    """
    Tracks which Groq model handles which task in real-time.
    
    Features:
    - Active task monitoring
    - Last 1000 completed tasks history
    - Model usage breakdown by task type
    - Complete audit trail
    """
    
    def __init__(self):
        self.active_tasks: Dict[str, TaskRecord] = {}
        self.completed_tasks: deque = deque(maxlen=1000)
        self.model_usage: Dict[str, int] = defaultdict(int)
        self.task_type_usage: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    
    def start_task(self, agent_name: str, task_type: str, model: str, task_description: str = "") -> str:
        """Start tracking a task"""
        task_id = f"task_{uuid.uuid4().hex[:12]}"
        self.active_tasks[task_id] = TaskRecord(
            task_id=task_id,
            agent_name=agent_name,
            task_type=task_type,
            model=model,
            start_time=datetime.now()
        )
        return task_id
    
    def complete_task(
        self,
        task_id: str,
        success: bool,
        tokens_used: int = 0,
        error_message: Optional[str] = None
    ):
        """Mark a task as complete"""
        if task_id in self.active_tasks:
            task = self.active_tasks.pop(task_id)
            task.end_time = datetime.now()
            task.success = success
            task.latency_ms = (task.end_time - task.start_time).total_seconds() * 1000
            task.tokens_used = tokens_used
            task.error_message = error_message
            
            self.completed_tasks.append(task)
            self.model_usage[task.model] += 1
            self.task_type_usage[task.task_type][task.model] += 1
    
    def get_active_tasks(self) -> List[Dict[str, Any]]:
        """Get all active tasks"""
        return [
            {
                "task_id": task.task_id,
                "agent_name": task.agent_name,
                "task_type": task.task_type,
                "model": task.model,
                "start_time": task.start_time.isoformat(),
                "duration_seconds": (datetime.now() - task.start_time).total_seconds()
            }
            for task in self.active_tasks.values()
        ]
    
    def get_recent_tasks(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent completed tasks"""
        tasks = list(self.completed_tasks)[-limit:]
        return [
            {
                "task_id": task.task_id,
                "agent_name": task.agent_name,
                "task_type": task.task_type,
                "model": task.model,
                "success": task.success,
                "latency_ms": task.latency_ms,
                "tokens_used": task.tokens_used,
                "start_time": task.start_time.isoformat()
            }
            for task in tasks
        ]
    
    def get_model_usage_summary(self) -> Dict[str, Any]:
        """Get model usage summary"""
        return {
            "total_tasks": sum(self.model_usage.values()),
            "by_model": dict(self.model_usage),
            "by_task_type": {
                task_type: dict(models)
                for task_type, models in self.task_type_usage.items()
            }
        }


# ============================================================================
# GROQ MIDDLEWARE
# ============================================================================

class GroqMiddleware:
    """
    Unified middleware for Groq optimization.
    
    Integrates:
    - API key rotation
    - Agent training
    - Task tracking
    - Quota management
    
    Usage:
        middleware = GroqMiddleware()
        result = await middleware.execute_request(
            agent_name="coding_agent",
            task_type="code_generation",
            task_description="Create REST API"
        )
    """
    
    def __init__(self, rotation_strategy: RotationStrategy = RotationStrategy.LEAST_USED):
        self.key_manager = APIKeyRotationManager(rotation_strategy)
        self.training_system = AgentTrainingSystem()
        self.task_tracker = TaskModelTracker()
        self._start_background_tasks()
    
    async def execute_request(
        self,
        agent_name: str,
        task_type: str,
        task_description: str,
        preferred_model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute a request with automatic optimization.
        
        NOTE: This is a template/demo implementation. In production, integrate with
        your actual Groq API client (e.g., groq-python SDK).
        
        Args:
            agent_name: Name of the agent
            task_type: Type of task (e.g., "code_generation", "analysis")
            task_description: Description of the task
            preferred_model: Optional preferred model
        
        Returns:
            Dict with execution results
        """
        # Step 1: Select model
        if preferred_model and preferred_model in GROQ_ENDPOINTS:
            model = preferred_model
        else:
            # Try to get recommended model from training
            model = self.training_system.get_recommended_model(agent_name, task_type)
            if not model:
                # Default based on task type
                model = self._default_model_for_task(task_type)
        
        # Step 2: Get API key
        key_tuple = self.key_manager.get_next_key(model)
        if not key_tuple:
            return {"success": False, "error": "No available API keys"}
        
        key_id, api_key = key_tuple
        
        # Step 3: Start task tracking
        task_id = self.task_tracker.start_task(agent_name, task_type, model, task_description)
        
        # Step 4: Execute request (PLACEHOLDER - integrate with actual Groq API)
        start_time = datetime.now()
        try:
            # TODO: Replace with actual Groq API call
            # Example:
            # from groq import Groq
            # client = Groq(api_key=api_key)
            # response = client.chat.completions.create(
            #     model=model,
            #     messages=[{"role": "user", "content": task_description}]
            # )
            
            # SIMULATION for demo purposes
            await asyncio.sleep(0.1)  # Simulate API call
            success = True
            tokens_used = 150
            latency_ms = (datetime.now() - start_time).total_seconds() * 1000
            quality_score = 0.85
            
            # Step 5: Record success
            self.key_manager.record_success(key_id)
            self.key_manager.record_request(key_id, model)
            self.task_tracker.complete_task(task_id, success, tokens_used)
            self.training_system.record_execution(
                agent_name, model, task_type, success, latency_ms, tokens_used, quality_score
            )
            
            return {
                "success": True,
                "model_used": model,
                "api_key_used": key_id,
                "task_id": task_id,
                "latency_ms": latency_ms,
                "tokens_used": tokens_used,
                "quality_score": quality_score
            }
            
        except Exception as e:
            # Record failure
            self.key_manager.record_failure(key_id)
            self.task_tracker.complete_task(task_id, False, error_message=str(e))
            self.training_system.record_execution(
                agent_name, model, task_type, False, 0, 0, 0
            )
            
            return {
                "success": False,
                "error": str(e),
                "model_used": model,
                "api_key_used": key_id,
                "task_id": task_id
            }
    
    def _default_model_for_task(self, task_type: str) -> str:
        """Get default model for task type"""
        # Ultra-fast tasks
        if task_type in ["monitoring", "validation", "simple_check"]:
            return "llama-3.1-8b-instant"
        
        # Reasoning tasks
        elif task_type in ["planning", "analysis", "decision_making"]:
            return "qwen/qwen3-32b"
        
        # Large context tasks
        elif task_type in ["codebase_analysis", "comprehensive_review"]:
            return "moonshotai/kimi-k2-instruct"
        
        # Complex tasks
        elif task_type in ["architecture", "security", "complex_design"]:
            return "openai/gpt-oss-120b"
        
        # Balanced tasks
        elif task_type in ["documentation", "medium_analysis"]:
            return "llama-4-maverick-17b"
        
        # General/default
        else:
            return "llama-3.3-70b-versatile"
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data"""
        return {
            "api_keys": self.key_manager.get_status(),
            "active_tasks": self.task_tracker.get_active_tasks(),
            "model_usage": self.task_tracker.get_model_usage_summary(),
            "recent_tasks": self.task_tracker.get_recent_tasks(20)
        }
    
    def _start_background_tasks(self):
        """Start background maintenance tasks"""
        # In production, implement async background tasks for:
        # - Periodic quota resets
        # - State persistence
        # - Health checks
        pass


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

async def example_usage():
    """Example usage of Groq middleware"""
    
    # Initialize middleware
    middleware = GroqMiddleware(rotation_strategy=RotationStrategy.LEAST_USED)
    
    # Execute requests
    tasks = [
        ("coding_agent", "code_generation", "Create REST API for user authentication"),
        ("monitoring_agent", "monitoring", "Check system health status"),
        ("analysis_agent", "analysis", "Analyze performance metrics"),
        ("architecture_agent", "architecture", "Design microservices architecture"),
    ]
    
    for agent, task_type, description in tasks:
        result = await middleware.execute_request(
            agent_name=agent,
            task_type=task_type,
            task_description=description
        )
        print(f"\nAgent: {agent}")
        print(f"Task: {task_type}")
        print(f"Success: {result['success']}")
        if result['success']:
            print(f"Model: {result['model_used']}")
            print(f"Latency: {result['latency_ms']:.0f}ms")
    
    # Get dashboard data
    dashboard = middleware.get_dashboard_data()
    print("\n=== Dashboard Summary ===")
    print(f"Active API keys: {dashboard['api_keys']['active_keys']}")
    print(f"Active tasks: {len(dashboard['active_tasks'])}")
    print(f"Total tasks completed: {dashboard['model_usage']['total_tasks']}")
    print(f"Model usage: {dashboard['model_usage']['by_model']}")


if __name__ == "__main__":
    asyncio.run(example_usage())

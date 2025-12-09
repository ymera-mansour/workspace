"""
Mistral AI Advanced Configuration System
========================================

Complete endpoint configuration, multi-organization key rotation, agent training,
and task tracking for all 6 Mistral models (100% FREE tier).

Features:
- All 6 Mistral models with complete endpoint URLs
- Multi-organization API key rotation (unlimited keys)
- Agent training system (learns optimal models for 40+ agents)
- Task-model tracking (real-time monitoring)
- Quota management (FREE tier: 60 RPM, 1B tokens/month)
- Intelligent routing by task complexity
- Unified middleware interface

Cost: $0 - 100% FREE tier operation
"""

import os
import asyncio
import time
import json
import hashlib
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, deque


# ============================================================================
# MISTRAL MODEL ENDPOINT CONFIGURATION
# ============================================================================

MISTRAL_ENDPOINTS = {
    "mistral-large-latest": {
        "name": "Mistral Large 3",
        "endpoint": "https://api.mistral.ai/v1/chat/completions",
        "streaming_endpoint": "https://api.mistral.ai/v1/chat/completions",
        "rpm_limit": 60,  # Requests per minute
        "rpd_limit": 86400,  # Requests per day (60 RPM * 1440 min)
        "tpm_limit": 500000,  # Tokens per minute
        "tpd_limit": 1000000000,  # 1 billion tokens per month
        "context_window": 256000,  # 256K tokens
        "parameters": "41B active / 675B total",
        "supports_multimodal": True,
        "supports_function_calling": True,
        "supports_json_mode": True,
        "best_for": ["complex_reasoning", "architecture", "security", "large_context"],
        "cost": "$0 FREE",
        "response_time_avg": "1-2s",
    },
    "codestral-latest": {
        "name": "Codestral 25.01",
        "endpoint": "https://api.mistral.ai/v1/chat/completions",
        "streaming_endpoint": "https://api.mistral.ai/v1/chat/completions",
        "rpm_limit": 60,
        "rpd_limit": 86400,
        "tpm_limit": 500000,
        "tpd_limit": 1000000000,
        "context_window": 256000,
        "parameters": "22B",
        "supports_multimodal": False,
        "supports_function_calling": True,
        "supports_json_mode": True,
        "supports_code_completion": True,
        "languages_supported": 80,
        "best_for": ["code_generation", "code_review", "debugging", "refactoring"],
        "cost": "$0 FREE",
        "response_time_avg": "0.8-1.5s",
    },
    "pixtral-large-latest": {
        "name": "Pixtral Large",
        "endpoint": "https://api.mistral.ai/v1/chat/completions",
        "streaming_endpoint": "https://api.mistral.ai/v1/chat/completions",
        "rpm_limit": 60,
        "rpd_limit": 86400,
        "tpm_limit": 500000,
        "tpd_limit": 1000000000,
        "context_window": 128000,  # 128K tokens
        "parameters": "123B",
        "supports_multimodal": True,
        "supports_vision": True,
        "supports_function_calling": True,
        "supports_json_mode": True,
        "image_formats": ["jpeg", "png", "gif", "webp"],
        "best_for": ["vision", "image_analysis", "ui_review", "diagram_understanding"],
        "cost": "$0 FREE",
        "response_time_avg": "1.5-2.5s",
    },
    "ministral-3b-latest": {
        "name": "Ministral 3B",
        "endpoint": "https://api.mistral.ai/v1/chat/completions",
        "streaming_endpoint": "https://api.mistral.ai/v1/chat/completions",
        "rpm_limit": 60,
        "rpd_limit": 86400,
        "tpm_limit": 500000,
        "tpd_limit": 1000000000,
        "context_window": 256000,
        "parameters": "3B",
        "supports_multimodal": False,
        "supports_function_calling": True,
        "supports_json_mode": True,
        "edge_optimized": True,
        "best_for": ["validation", "formatting", "simple_tasks", "monitoring"],
        "cost": "$0 FREE",
        "response_time_avg": "0.3-0.8s",
    },
    "ministral-8b-latest": {
        "name": "Ministral 8B",
        "endpoint": "https://api.mistral.ai/v1/chat/completions",
        "streaming_endpoint": "https://api.mistral.ai/v1/chat/completions",
        "rpm_limit": 60,
        "rpd_limit": 86400,
        "tpm_limit": 500000,
        "tpd_limit": 1000000000,
        "context_window": 256000,
        "parameters": "8B",
        "supports_multimodal": False,
        "supports_function_calling": True,
        "supports_json_mode": True,
        "edge_optimized": True,
        "best_for": ["database", "api", "testing", "standard_tasks"],
        "cost": "$0 FREE",
        "response_time_avg": "0.5-1s",
    },
    "ministral-14b-latest": {
        "name": "Ministral 14B",
        "endpoint": "https://api.mistral.ai/v1/chat/completions",
        "streaming_endpoint": "https://api.mistral.ai/v1/chat/completions",
        "rpm_limit": 60,
        "rpd_limit": 86400,
        "tpm_limit": 500000,
        "tpd_limit": 1000000000,
        "context_window": 256000,
        "parameters": "14B",
        "supports_multimodal": False,
        "supports_function_calling": True,
        "supports_json_mode": True,
        "edge_optimized": True,
        "best_for": ["analysis", "planning", "documentation", "moderate_tasks"],
        "cost": "$0 FREE",
        "response_time_avg": "0.7-1.2s",
    },
}


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class APIKeyInfo:
    """Information about an API key."""
    key_id: str
    api_key: str
    organization: str
    rpm_count: int = 0
    rpd_count: int = 0
    last_reset_minute: datetime = None
    last_reset_day: datetime = None
    is_active: bool = True
    failure_count: int = 0
    last_used: datetime = None
    
    def __post_init__(self):
        if self.last_reset_minute is None:
            self.last_reset_minute = datetime.now()
        if self.last_reset_day is None:
            self.last_reset_day = datetime.now()


@dataclass
class AgentPerformance:
    """Performance metrics for agent-model combination."""
    agent_name: str
    model_name: str
    task_type: str
    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    total_latency_ms: float = 0
    total_tokens: int = 0
    total_quality_score: float = 0
    last_updated: datetime = None
    
    def __post_init__(self):
        if self.last_updated is None:
            self.last_updated = datetime.now()
    
    @property
    def success_rate(self) -> float:
        if self.total_executions == 0:
            return 0.0
        return self.successful_executions / self.total_executions
    
    @property
    def avg_latency_ms(self) -> float:
        if self.total_executions == 0:
            return 0.0
        return self.total_latency_ms / self.total_executions
    
    @property
    def avg_quality_score(self) -> float:
        if self.total_executions == 0:
            return 0.0
        return self.total_quality_score / self.total_executions


@dataclass
class TaskInfo:
    """Information about a task execution."""
    task_id: str
    agent_name: str
    task_type: str
    model_used: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    success: Optional[bool] = None
    latency_ms: Optional[float] = None
    tokens_used: Optional[int] = None
    error_message: Optional[str] = None


class RotationStrategy(Enum):
    """API key rotation strategies."""
    ROUND_ROBIN = "round_robin"
    LEAST_USED = "least_used"
    WEIGHTED = "weighted"


# ============================================================================
# API KEY ROTATION MANAGER
# ============================================================================

class APIKeyRotationManager:
    """
    Manages API keys from multiple organizations with automatic rotation.
    
    Features:
    - Unlimited keys from different organizations
    - 3 rotation strategies: round-robin, least-used, weighted
    - Automatic failover when limits hit
    - Per-key quota tracking and health monitoring
    - Deterministic state persistence
    """
    
    def __init__(self, strategy: RotationStrategy = RotationStrategy.LEAST_USED):
        self.keys: Dict[str, APIKeyInfo] = {}
        self.strategy = strategy
        self.current_index = 0
        self._load_keys_from_env()
    
    def _load_keys_from_env(self):
        """Load API keys from environment variables."""
        # Primary key
        primary_key = os.getenv("MISTRAL_API_KEY")
        if primary_key:
            self.add_key("primary", primary_key, "Primary Organization")
        
        # Additional keys: MISTRAL_API_KEY_1, MISTRAL_API_KEY_2, etc.
        i = 1
        while True:
            key = os.getenv(f"MISTRAL_API_KEY_{i}")
            if not key:
                break
            org = os.getenv(f"MISTRAL_ORG_{i}", f"Organization {i}")
            self.add_key(f"key_{i}", key, org)
            i += 1
    
    def add_key(self, key_id: str, api_key: str, organization: str):
        """Add a new API key."""
        self.keys[key_id] = APIKeyInfo(
            key_id=key_id,
            api_key=api_key,
            organization=organization
        )
    
    def get_next_key(self, model: str) -> Optional[APIKeyInfo]:
        """Get the next available API key based on rotation strategy."""
        if not self.keys:
            return None
        
        if self.strategy == RotationStrategy.ROUND_ROBIN:
            return self._get_round_robin_key(model)
        elif self.strategy == RotationStrategy.LEAST_USED:
            return self._get_least_used_key(model)
        elif self.strategy == RotationStrategy.WEIGHTED:
            return self._get_weighted_key(model)
        
        return None
    
    def _get_round_robin_key(self, model: str) -> Optional[APIKeyInfo]:
        """Round-robin rotation."""
        active_keys = [k for k in self.keys.values() if k.is_active]
        if not active_keys:
            return None
        
        key = active_keys[self.current_index % len(active_keys)]
        self.current_index += 1
        
        # Reset counters if needed
        self._reset_counters(key, model)
        
        # Check if key can make request
        if self._can_make_request(key, model):
            return key
        
        # Try next key
        return self._get_round_robin_key(model)
    
    def _get_least_used_key(self, model: str) -> Optional[APIKeyInfo]:
        """Get key with lowest usage."""
        active_keys = [k for k in self.keys.values() if k.is_active]
        if not active_keys:
            return None
        
        # Reset counters if needed
        for key in active_keys:
            self._reset_counters(key, model)
        
        # Sort by RPM count (ascending)
        available_keys = [k for k in active_keys if self._can_make_request(k, model)]
        if not available_keys:
            return None
        
        return min(available_keys, key=lambda k: k.rpm_count)
    
    def _get_weighted_key(self, model: str) -> Optional[APIKeyInfo]:
        """Get key based on remaining quota (weighted selection)."""
        active_keys = [k for k in self.keys.values() if k.is_active]
        if not active_keys:
            return None
        
        # Reset counters if needed
        for key in active_keys:
            self._reset_counters(key, model)
        
        # Calculate weights based on remaining quota
        model_config = MISTRAL_ENDPOINTS.get(model, {})
        rpm_limit = model_config.get("rpm_limit", 60)
        
        weights = []
        for key in active_keys:
            if self._can_make_request(key, model):
                remaining = rpm_limit - key.rpm_count
                weights.append((key, remaining))
        
        if not weights:
            return None
        
        # Select key with most remaining quota
        return max(weights, key=lambda x: x[1])[0]
    
    def _can_make_request(self, key: APIKeyInfo, model: str) -> bool:
        """Check if key can make a request."""
        model_config = MISTRAL_ENDPOINTS.get(model, {})
        rpm_limit = model_config.get("rpm_limit", 60)
        rpd_limit = model_config.get("rpd_limit", 86400)
        
        return key.rpm_count < rpm_limit and key.rpd_count < rpd_limit
    
    def _reset_counters(self, key: APIKeyInfo, model: str):
        """Reset counters if time period has elapsed."""
        now = datetime.now()
        
        # Reset RPM if minute has passed
        if (now - key.last_reset_minute).total_seconds() >= 60:
            key.rpm_count = 0
            key.last_reset_minute = now
        
        # Reset RPD if day has passed
        if (now - key.last_reset_day).total_seconds() >= 86400:
            key.rpd_count = 0
            key.last_reset_day = now
    
    def record_request(self, key: APIKeyInfo):
        """Record a successful request."""
        key.rpm_count += 1
        key.rpd_count += 1
        key.last_used = datetime.now()
        
        # Save state every 10 requests for persistence
        if key.rpd_count % 10 == 0:
            self._save_state()
    
    def record_failure(self, key: APIKeyInfo):
        """Record a failed request."""
        key.failure_count += 1
        
        # Deactivate key after 5 consecutive failures
        if key.failure_count >= 5:
            key.is_active = False
    
    def record_success(self, key: APIKeyInfo):
        """Record successful request (reset failure count)."""
        key.failure_count = 0
        if not key.is_active:
            key.is_active = True
    
    def _save_state(self):
        """Save key state to disk for persistence."""
        state = {
            key_id: {
                "rpm_count": key.rpm_count,
                "rpd_count": key.rpd_count,
                "last_reset_minute": key.last_reset_minute.isoformat(),
                "last_reset_day": key.last_reset_day.isoformat(),
                "is_active": key.is_active,
                "failure_count": key.failure_count,
            }
            for key_id, key in self.keys.items()
        }
        
        try:
            with open("mistral_key_state.json", "w") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            print(f"Failed to save key state: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of all keys."""
        return {
            "total_keys": len(self.keys),
            "active_keys": sum(1 for k in self.keys.values() if k.is_active),
            "strategy": self.strategy.value,
            "keys": [
                {
                    "key_id": k.key_id,
                    "organization": k.organization,
                    "rpm_count": k.rpm_count,
                    "rpd_count": k.rpd_count,
                    "is_active": k.is_active,
                    "failure_count": k.failure_count,
                    "last_used": k.last_used.isoformat() if k.last_used else None,
                }
                for k in self.keys.values()
            ]
        }


# ============================================================================
# AGENT TRAINING SYSTEM
# ============================================================================

class AgentTrainingSystem:
    """
    Learns optimal models for 40+ agents over time.
    
    Features:
    - Tracks success rate, latency, quality per agent-model-task
    - Auto-recommends best model based on historical performance
    - Persists training data to disk
    - Statistics dashboard for all agents
    """
    
    def __init__(self, data_file: str = "mistral_agent_training_data.json"):
        self.data_file = data_file
        self.performance: Dict[str, AgentPerformance] = {}
        self._load_data()
    
    def _get_key(self, agent_name: str, model_name: str, task_type: str) -> str:
        """Generate unique key for agent-model-task combination."""
        return f"{agent_name}:{model_name}:{task_type}"
    
    def record_execution(
        self,
        agent_name: str,
        model_name: str,
        task_type: str,
        success: bool,
        latency_ms: float,
        tokens_used: int = 0,
        quality_score: float = 0.5,
    ):
        """Record execution results for training."""
        key = self._get_key(agent_name, model_name, task_type)
        
        if key not in self.performance:
            self.performance[key] = AgentPerformance(
                agent_name=agent_name,
                model_name=model_name,
                task_type=task_type
            )
        
        perf = self.performance[key]
        perf.total_executions += 1
        if success:
            perf.successful_executions += 1
        else:
            perf.failed_executions += 1
        perf.total_latency_ms += latency_ms
        perf.total_tokens += tokens_used
        perf.total_quality_score += quality_score
        perf.last_updated = datetime.now()
        
        # Save data every 50 executions for persistence
        if perf.total_executions % 50 == 0:
            self._save_data()
    
    def get_recommended_model(
        self,
        agent_name: str,
        task_type: str,
        prefer_speed: bool = False
    ) -> Optional[str]:
        """Get recommended model for agent and task type."""
        # Find all performance data for this agent-task combination
        candidates = [
            perf for perf in self.performance.values()
            if perf.agent_name == agent_name and perf.task_type == task_type
            and perf.total_executions >= 10  # Minimum 10 executions
        ]
        
        if not candidates:
            # No training data, use defaults
            return self._get_default_model(task_type)
        
        # Score based on success rate and latency
        def score(perf: AgentPerformance) -> float:
            success_weight = 0.4 if prefer_speed else 0.6
            latency_weight = 0.6 if prefer_speed else 0.4
            
            # Normalize latency (lower is better, assume max 5000ms)
            latency_score = max(0, 1 - (perf.avg_latency_ms / 5000))
            
            return (success_weight * perf.success_rate + 
                    latency_weight * latency_score)
        
        best = max(candidates, key=score)
        return best.model_name
    
    def _get_default_model(self, task_type: str) -> str:
        """Get default model for task type."""
        defaults = {
            "code_generation": "codestral-latest",
            "code_review": "codestral-latest",
            "debugging": "codestral-latest",
            "refactoring": "codestral-latest",
            "architecture": "mistral-large-latest",
            "security": "mistral-large-latest",
            "complex_reasoning": "mistral-large-latest",
            "vision": "pixtral-large-latest",
            "image_analysis": "pixtral-large-latest",
            "ui_review": "pixtral-large-latest",
            "validation": "ministral-3b-latest",
            "formatting": "ministral-3b-latest",
            "monitoring": "ministral-3b-latest",
            "database": "ministral-8b-latest",
            "api": "ministral-8b-latest",
            "testing": "ministral-8b-latest",
            "analysis": "ministral-14b-latest",
            "planning": "ministral-14b-latest",
            "documentation": "ministral-14b-latest",
        }
        return defaults.get(task_type, "ministral-8b-latest")
    
    def get_agent_statistics(self, agent_name: str) -> Dict[str, Any]:
        """Get statistics for a specific agent."""
        agent_perfs = [
            perf for perf in self.performance.values()
            if perf.agent_name == agent_name
        ]
        
        if not agent_perfs:
            return {"agent_name": agent_name, "no_data": True}
        
        total_executions = sum(p.total_executions for p in agent_perfs)
        total_successful = sum(p.successful_executions for p in agent_perfs)
        
        return {
            "agent_name": agent_name,
            "total_executions": total_executions,
            "success_rate": total_successful / total_executions if total_executions > 0 else 0,
            "models_used": len(set(p.model_name for p in agent_perfs)),
            "task_types": len(set(p.task_type for p in agent_perfs)),
            "preferred_model": max(agent_perfs, key=lambda p: p.total_executions).model_name,
        }
    
    def get_all_statistics(self) -> Dict[str, Any]:
        """Get statistics for all agents."""
        agents = set(p.agent_name for p in self.performance.values())
        return {
            "total_agents": len(agents),
            "total_executions": sum(p.total_executions for p in self.performance.values()),
            "agents": [self.get_agent_statistics(agent) for agent in agents]
        }
    
    def _save_data(self):
        """Save training data to disk."""
        data = {
            key: {
                "agent_name": perf.agent_name,
                "model_name": perf.model_name,
                "task_type": perf.task_type,
                "total_executions": perf.total_executions,
                "successful_executions": perf.successful_executions,
                "failed_executions": perf.failed_executions,
                "total_latency_ms": perf.total_latency_ms,
                "total_tokens": perf.total_tokens,
                "total_quality_score": perf.total_quality_score,
                "last_updated": perf.last_updated.isoformat(),
            }
            for key, perf in self.performance.items()
        }
        
        try:
            with open(self.data_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Failed to save training data: {e}")
    
    def _load_data(self):
        """Load training data from disk."""
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, "r") as f:
                    data = json.load(f)
                
                for key, perf_data in data.items():
                    perf = AgentPerformance(
                        agent_name=perf_data["agent_name"],
                        model_name=perf_data["model_name"],
                        task_type=perf_data["task_type"],
                        total_executions=perf_data["total_executions"],
                        successful_executions=perf_data["successful_executions"],
                        failed_executions=perf_data["failed_executions"],
                        total_latency_ms=perf_data["total_latency_ms"],
                        total_tokens=perf_data["total_tokens"],
                        total_quality_score=perf_data["total_quality_score"],
                        last_updated=datetime.fromisoformat(perf_data["last_updated"]),
                    )
                    self.performance[key] = perf
        except Exception as e:
            print(f"Failed to load training data: {e}")


# ============================================================================
# TASK-MODEL TRACKING
# ============================================================================

class TaskModelTracker:
    """
    Real-time tracking of which model handles which task.
    
    Features:
    - Active task monitoring
    - Last 1000 completed tasks history
    - Model usage breakdown by task type
    - Complete audit trail with latency and success tracking
    """
    
    def __init__(self, max_history: int = 1000):
        self.active_tasks: Dict[str, TaskInfo] = {}
        self.completed_tasks: deque = deque(maxlen=max_history)
        self.model_usage: Dict[str, int] = defaultdict(int)
        self.task_type_usage: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    
    def start_task(
        self,
        agent_name: str,
        task_type: str,
        model_name: str,
        task_description: str = ""
    ) -> str:
        """Start tracking a new task."""
        task_id = f"task_{uuid.uuid4().hex[:12]}"
        
        task = TaskInfo(
            task_id=task_id,
            agent_name=agent_name,
            task_type=task_type,
            model_used=model_name,
            started_at=datetime.now()
        )
        
        self.active_tasks[task_id] = task
        return task_id
    
    def complete_task(
        self,
        task_id: str,
        success: bool,
        tokens_used: int = 0,
        error_message: str = None
    ):
        """Mark task as completed."""
        if task_id not in self.active_tasks:
            return
        
        task = self.active_tasks.pop(task_id)
        task.completed_at = datetime.now()
        task.success = success
        task.latency_ms = (task.completed_at - task.started_at).total_seconds() * 1000
        task.tokens_used = tokens_used
        task.error_message = error_message
        
        self.completed_tasks.append(task)
        self.model_usage[task.model_used] += 1
        self.task_type_usage[task.task_type][task.model_used] += 1
    
    def get_active_tasks(self) -> List[Dict[str, Any]]:
        """Get all currently active tasks."""
        return [
            {
                "task_id": task.task_id,
                "agent_name": task.agent_name,
                "task_type": task.task_type,
                "model_used": task.model_used,
                "started_at": task.started_at.isoformat(),
                "duration_seconds": (datetime.now() - task.started_at).total_seconds(),
            }
            for task in self.active_tasks.values()
        ]
    
    def get_recent_tasks(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent completed tasks."""
        tasks = list(self.completed_tasks)[-limit:]
        return [
            {
                "task_id": task.task_id,
                "agent_name": task.agent_name,
                "task_type": task.task_type,
                "model_used": task.model_used,
                "success": task.success,
                "latency_ms": task.latency_ms,
                "tokens_used": task.tokens_used,
                "completed_at": task.completed_at.isoformat() if task.completed_at else None,
            }
            for task in tasks
        ]
    
    def get_model_usage_summary(self) -> Dict[str, Any]:
        """Get summary of model usage."""
        total_tasks = sum(self.model_usage.values())
        
        return {
            "total_tasks": total_tasks,
            "model_breakdown": {
                model: {
                    "count": count,
                    "percentage": (count / total_tasks * 100) if total_tasks > 0 else 0
                }
                for model, count in self.model_usage.items()
            },
            "task_type_breakdown": dict(self.task_type_usage)
        }


# ============================================================================
# UNIFIED MIDDLEWARE
# ============================================================================

class MistralMiddleware:
    """
    Unified interface for all Mistral optimization features.
    
    Features:
    - Automatic key rotation
    - Model selection from training
    - Task tracking
    - Performance recording
    - Background maintenance tasks
    - Comprehensive dashboard
    """
    
    def __init__(
        self,
        rotation_strategy: RotationStrategy = RotationStrategy.LEAST_USED
    ):
        self.key_manager = APIKeyRotationManager(strategy=rotation_strategy)
        self.training = AgentTrainingSystem()
        self.tracker = TaskModelTracker()
        self._start_background_tasks()
    
    def _start_background_tasks(self):
        """Start background maintenance tasks."""
        # In a real implementation, these would run in separate threads/tasks
        # For demo purposes, we'll call them manually when needed
        pass
    
    async def execute_request(
        self,
        agent_name: str,
        task_type: str,
        task_description: str,
        prefer_speed: bool = False,
        model_override: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute a request with full optimization.
        
        NOTE: This is a template/demo method. In production, you would:
        1. Integrate with actual Mistral API client (e.g., `mistralai` package)
        2. Replace the simulation with real API calls
        3. Add proper error handling and retries
        
        Args:
            agent_name: Name of the agent making the request
            task_type: Type of task (code_generation, analysis, etc.)
            task_description: Description of the task
            prefer_speed: Whether to prefer faster models
            model_override: Optional specific model to use
        
        Returns:
            Dict containing response and metadata
        """
        # 1. Get recommended model from training or use override
        if model_override:
            model = model_override
        else:
            model = self.training.get_recommended_model(
                agent_name, task_type, prefer_speed
            )
        
        # 2. Get API key
        key_info = self.key_manager.get_next_key(model)
        if not key_info:
            raise Exception("No available API keys")
        
        # 3. Start task tracking
        task_id = self.tracker.start_task(agent_name, task_type, model, task_description)
        
        # 4. Execute request (THIS IS WHERE YOU'D CALL ACTUAL API)
        start_time = time.time()
        try:
            # TODO: Replace with actual Mistral API call
            # Example:
            # from mistralai.client import MistralClient
            # client = MistralClient(api_key=key_info.api_key)
            # response = client.chat(
            #     model=model,
            #     messages=[{"role": "user", "content": task_description}]
            # )
            
            # For now, simulate a response
            await asyncio.sleep(0.1)  # Simulate API call
            response_text = f"[Demo Response for {model}]"
            tokens_used = 100
            success = True
            error_message = None
            
            # Record success
            self.key_manager.record_success(key_info)
            self.key_manager.record_request(key_info)
            
        except Exception as e:
            success = False
            response_text = None
            tokens_used = 0
            error_message = str(e)
            
            # Record failure
            self.key_manager.record_failure(key_info)
        
        # 5. Calculate metrics
        latency_ms = (time.time() - start_time) * 1000
        quality_score = 0.8 if success else 0.0
        
        # 6. Complete task tracking
        self.tracker.complete_task(task_id, success, tokens_used, error_message)
        
        # 7. Record for training
        self.training.record_execution(
            agent_name, model, task_type, success, latency_ms, tokens_used, quality_score
        )
        
        # 8. Return result
        return {
            "success": success,
            "model_used": model,
            "api_key_used": key_info.key_id,
            "response": response_text,
            "latency_ms": latency_ms,
            "tokens_used": tokens_used,
            "task_id": task_id,
            "error": error_message
        }
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data."""
        return {
            "api_keys": self.key_manager.get_status(),
            "active_tasks": self.tracker.get_active_tasks(),
            "model_usage": self.tracker.get_model_usage_summary(),
            "agent_statistics": self.training.get_all_statistics(),
            "recent_tasks": self.tracker.get_recent_tasks(limit=20)
        }


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

async def main():
    """Example usage of Mistral advanced configuration."""
    
    # Initialize middleware
    middleware = MistralMiddleware(
        rotation_strategy=RotationStrategy.LEAST_USED
    )
    
    # Execute request
    result = await middleware.execute_request(
        agent_name="coding_agent",
        task_type="code_generation",
        task_description="Create a REST API endpoint for user authentication"
    )
    
    print(f"Model used: {result['model_used']}")
    print(f"Success: {result['success']}")
    print(f"Latency: {result['latency_ms']:.2f}ms")
    
    # Get dashboard data
    dashboard = middleware.get_dashboard_data()
    print(f"\nTotal API keys: {dashboard['api_keys']['total_keys']}")
    print(f"Active tasks: {len(dashboard['active_tasks'])}")
    print(f"Total agents: {dashboard['agent_statistics']['total_agents']}")


if __name__ == "__main__":
    asyncio.run(main())

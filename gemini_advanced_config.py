"""
Advanced Gemini Configuration with Endpoints, Key Rotation, and Agent Training
Addresses: Model endpoints, API key rotation, agent training system, task-model tracking
"""

import asyncio
import os
import json
import random
import uuid
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# GEMINI MODEL ENDPOINTS CONFIGURATION
# ============================================================================

@dataclass
class GeminiEndpointConfig:
    """Complete endpoint configuration for each Gemini model"""
    model_id: str
    display_name: str
    endpoint_url: str
    api_version: str
    supports_streaming: bool
    supports_multimodal: bool
    supports_function_calling: bool
    rpm_limit: int
    rpd_limit: int
    max_input_tokens: int
    max_output_tokens: int
    temperature_range: Tuple[float, float]
    default_temperature: float


# Complete endpoint configurations
GEMINI_ENDPOINTS = {
    "gemini-2.0-flash-exp": GeminiEndpointConfig(
        model_id="gemini-2.0-flash-exp",
        display_name="Gemini 2.0 Flash (Experimental)",
        endpoint_url="https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent",
        api_version="v1beta",
        supports_streaming=True,
        supports_multimodal=True,
        supports_function_calling=True,
        rpm_limit=10,
        rpd_limit=1500,
        max_input_tokens=1_000_000,
        max_output_tokens=8192,
        temperature_range=(0.0, 2.0),
        default_temperature=0.7
    ),
    "gemini-1.5-flash": GeminiEndpointConfig(
        model_id="gemini-1.5-flash",
        display_name="Gemini 1.5 Flash",
        endpoint_url="https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent",
        api_version="v1beta",
        supports_streaming=True,
        supports_multimodal=True,
        supports_function_calling=True,
        rpm_limit=15,
        rpd_limit=1500,
        max_input_tokens=1_000_000,
        max_output_tokens=8192,
        temperature_range=(0.0, 2.0),
        default_temperature=0.7
    ),
    "gemini-1.5-pro": GeminiEndpointConfig(
        model_id="gemini-1.5-pro",
        display_name="Gemini 1.5 Pro",
        endpoint_url="https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent",
        api_version="v1beta",
        supports_streaming=True,
        supports_multimodal=True,
        supports_function_calling=True,
        rpm_limit=2,
        rpd_limit=50,
        max_input_tokens=2_000_000,
        max_output_tokens=8192,
        temperature_range=(0.0, 2.0),
        default_temperature=0.7
    ),
    "gemini-1.5-flash-8b": GeminiEndpointConfig(
        model_id="gemini-1.5-flash-8b",
        display_name="Gemini 1.5 Flash 8B",
        endpoint_url="https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-8b:generateContent",
        api_version="v1beta",
        supports_streaming=True,
        supports_multimodal=False,
        supports_function_calling=True,
        rpm_limit=15,
        rpd_limit=4000,
        max_input_tokens=1_000_000,
        max_output_tokens=8192,
        temperature_range=(0.0, 2.0),
        default_temperature=0.7
    ),
}

# ============================================================================
# API KEY ROTATION SYSTEM
# ============================================================================

@dataclass
class APIKeyInfo:
    """Information about an API key"""
    key_id: str
    api_key: str
    organization: str
    rpm_used: int = 0
    rpd_used: int = 0
    last_reset: datetime = field(default_factory=datetime.now)
    is_active: bool = True
    failure_count: int = 0
    last_used: datetime = field(default_factory=datetime.now)


class APIKeyRotationManager:
    """
    Manages multiple Google API keys with intelligent rotation
    Handles quota limits, failure recovery, and load balancing
    """
    
    def __init__(self):
        self.keys: Dict[str, APIKeyInfo] = {}
        self.current_key_index = 0
        self.rotation_strategy = "round_robin"  # Options: round_robin, least_used, weighted
        self._load_keys_from_env()
    
    def _load_keys_from_env(self):
        """Load multiple API keys from environment variables"""
        # Primary key
        primary_key = os.getenv("GEMINI_API_KEY")
        if primary_key:
            self.keys["primary"] = APIKeyInfo(
                key_id="primary",
                api_key=primary_key,
                organization="default"
            )
        
        # Secondary keys (can be from different organizations)
        for i in range(1, 10):  # Support up to 9 additional keys
            key = os.getenv(f"GEMINI_API_KEY_{i}")
            org = os.getenv(f"GEMINI_ORG_{i}", f"org_{i}")
            if key:
                self.keys[f"key_{i}"] = APIKeyInfo(
                    key_id=f"key_{i}",
                    api_key=key,
                    organization=org
                )
        
        logger.info(f"Loaded {len(self.keys)} API keys for rotation")
    
    def add_key(self, key_id: str, api_key: str, organization: str = "custom"):
        """Manually add an API key"""
        self.keys[key_id] = APIKeyInfo(
            key_id=key_id,
            api_key=api_key,
            organization=organization
        )
        logger.info(f"Added API key: {key_id} from organization: {organization}")
    
    def get_next_key(self, model_id: str) -> Optional[APIKeyInfo]:
        """Get next available API key based on rotation strategy"""
        if not self.keys:
            logger.error("No API keys available")
            return None
        
        # Filter active keys
        active_keys = [k for k in self.keys.values() if k.is_active]
        if not active_keys:
            logger.error("No active API keys available")
            return None
        
        # Apply rotation strategy
        if self.rotation_strategy == "round_robin":
            return self._round_robin_selection(active_keys)
        elif self.rotation_strategy == "least_used":
            return self._least_used_selection(active_keys)
        elif self.rotation_strategy == "weighted":
            return self._weighted_selection(active_keys, model_id)
        else:
            return active_keys[0]
    
    def _round_robin_selection(self, keys: List[APIKeyInfo]) -> APIKeyInfo:
        """Simple round-robin selection"""
        key = keys[self.current_key_index % len(keys)]
        self.current_key_index += 1
        return key
    
    def _least_used_selection(self, keys: List[APIKeyInfo]) -> APIKeyInfo:
        """Select key with lowest usage"""
        return min(keys, key=lambda k: k.rpd_used)
    
    def _weighted_selection(self, keys: List[APIKeyInfo], model_id: str) -> APIKeyInfo:
        """Weighted selection based on remaining quota"""
        endpoint = GEMINI_ENDPOINTS.get(model_id)
        if not endpoint:
            return keys[0]
        
        # Calculate weights based on remaining quota
        weights = []
        for key in keys:
            remaining_quota = endpoint.rpd_limit - key.rpd_used
            weight = max(0, remaining_quota)
            weights.append(weight)
        
        # If all weights are 0, fallback to round-robin
        if sum(weights) == 0:
            return self._round_robin_selection(keys)
        
        # Weighted random selection
        return random.choices(keys, weights=weights)[0]
    
    async def record_request(self, key_id: str, model_id: str, success: bool = True):
        """Record API request for quota tracking"""
        if key_id not in self.keys:
            return
        
        key_info = self.keys[key_id]
        key_info.rpm_used += 1
        key_info.rpd_used += 1
        key_info.last_used = datetime.now()
        
        if not success:
            key_info.failure_count += 1
            # Deactivate key after too many failures
            if key_info.failure_count >= 5:
                key_info.is_active = False
                logger.warning(f"Deactivated API key {key_id} due to failures")
        else:
            # Reset failure count on success
            key_info.failure_count = max(0, key_info.failure_count - 1)
        
        # Save after every 10th request for data persistence
        if (key_info.rpm_used + key_info.rpd_used) % 10 == 0:
            self._save_state()
    
    def _save_state(self):
        """Save key manager state to disk"""
        try:
            state = {
                key_id: {
                    "organization": info.organization,
                    "rpm_used": info.rpm_used,
                    "rpd_used": info.rpd_used,
                    "last_reset": info.last_reset.isoformat(),
                    "is_active": info.is_active,
                    "failure_count": info.failure_count
                }
                for key_id, info in self.keys.items()
            }
            with open("api_key_state.json", "w") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save key state: {e}")
    
    async def check_and_rotate_if_needed(self, key_id: str, model_id: str) -> Optional[str]:
        """Check if key needs rotation and return new key if needed"""
        if key_id not in self.keys:
            return None
        
        key_info = self.keys[key_id]
        endpoint = GEMINI_ENDPOINTS.get(model_id)
        
        if not endpoint:
            return None
        
        # Check if approaching limits
        rpm_threshold = endpoint.rpm_limit * 0.9
        rpd_threshold = endpoint.rpd_limit * 0.9
        
        if key_info.rpm_used >= rpm_threshold or key_info.rpd_used >= rpd_threshold:
            logger.info(f"Key {key_id} approaching limits, rotating...")
            next_key = self.get_next_key(model_id)
            if next_key and next_key.key_id != key_id:
                return next_key.key_id
        
        return None
    
    def reset_counters(self):
        """Reset usage counters (call daily/hourly)"""
        now = datetime.now()
        for key_info in self.keys.values():
            # Reset RPM every minute - use total_seconds() for accurate calculation
            elapsed_seconds = (now - key_info.last_reset).total_seconds()
            if elapsed_seconds >= 60:
                key_info.rpm_used = 0
            
            # Reset RPD daily
            if (now - key_info.last_reset).days >= 1:
                key_info.rpd_used = 0
                key_info.last_reset = now
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of all API keys"""
        return {
            key_id: {
                "organization": info.organization,
                "rpm_used": info.rpm_used,
                "rpd_used": info.rpd_used,
                "is_active": info.is_active,
                "failure_count": info.failure_count,
                "last_used": info.last_used.isoformat()
            }
            for key_id, info in self.keys.items()
        }

# ============================================================================
# AGENT TRAINING SYSTEM
# ============================================================================

@dataclass
class AgentPerformanceMetrics:
    """Performance metrics for agent-model combinations"""
    agent_name: str
    model_id: str
    task_type: str
    success_count: int = 0
    failure_count: int = 0
    total_latency_ms: float = 0.0
    total_tokens_used: int = 0
    total_cost: float = 0.0
    quality_scores: List[float] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.now)


class AgentTrainingSystem:
    """
    Training and optimization system for 40+ agents
    Learns which models work best for each agent and task type
    """
    
    def __init__(self, storage_path: str = "agent_training_data.json"):
        self.storage_path = storage_path
        self.metrics: Dict[str, AgentPerformanceMetrics] = {}
        self.agent_model_preferences: Dict[str, Dict[str, str]] = {}
        self._load_training_data()
    
    def _load_training_data(self):
        """Load historical training data"""
        try:
            if os.path.exists(self.storage_path):
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                    # Reconstruct metrics from saved data
                    for key, value in data.get("metrics", {}).items():
                        parts = key.split(":")
                        if len(parts) == 3:
                            agent, model, task = parts
                            self.metrics[key] = AgentPerformanceMetrics(
                                agent_name=agent,
                                model_id=model,
                                task_type=task,
                                **{k: v for k, v in value.items() 
                                   if k not in ["agent_name", "model_id", "task_type"]}
                            )
                    self.agent_model_preferences = data.get("preferences", {})
                logger.info(f"Loaded training data: {len(self.metrics)} metrics")
        except Exception as e:
            logger.warning(f"Could not load training data: {e}")
    
    def _save_training_data(self):
        """Save training data to disk"""
        try:
            data = {
                "metrics": {
                    key: {
                        "success_count": m.success_count,
                        "failure_count": m.failure_count,
                        "total_latency_ms": m.total_latency_ms,
                        "total_tokens_used": m.total_tokens_used,
                        "total_cost": m.total_cost,
                        "quality_scores": m.quality_scores,
                        "last_updated": m.last_updated.isoformat()
                    }
                    for key, m in self.metrics.items()
                },
                "preferences": self.agent_model_preferences,
                "last_saved": datetime.now().isoformat()
            }
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save training data: {e}")
    
    def record_execution(
        self,
        agent_name: str,
        model_id: str,
        task_type: str,
        success: bool,
        latency_ms: float,
        tokens_used: int,
        quality_score: float = 0.0,
        cost: float = 0.0
    ):
        """Record an agent execution for training"""
        key = f"{agent_name}:{model_id}:{task_type}"
        
        if key not in self.metrics:
            self.metrics[key] = AgentPerformanceMetrics(
                agent_name=agent_name,
                model_id=model_id,
                task_type=task_type
            )
        
        metrics = self.metrics[key]
        
        if success:
            metrics.success_count += 1
        else:
            metrics.failure_count += 1
        
        metrics.total_latency_ms += latency_ms
        metrics.total_tokens_used += tokens_used
        metrics.total_cost += cost
        
        if quality_score > 0:
            metrics.quality_scores.append(quality_score)
            # Keep only last 100 scores
            if len(metrics.quality_scores) > 100:
                metrics.quality_scores = metrics.quality_scores[-100:]
        
        metrics.last_updated = datetime.now()
        
        # Update preferences
        self._update_preferences(agent_name, task_type)
        
        # Save deterministically every 50 executions
        total_executions = sum(
            m.success_count + m.failure_count 
            for m in self.metrics.values()
        )
        if total_executions % 50 == 0:
            self._save_training_data()
    
    def _update_preferences(self, agent_name: str, task_type: str):
        """Update model preferences based on performance"""
        # Find best model for this agent and task
        relevant_metrics = [
            m for m in self.metrics.values()
            if m.agent_name == agent_name and m.task_type == task_type
        ]
        
        if not relevant_metrics:
            return
        
        # Score models based on success rate, latency, and quality
        scores = {}
        for m in relevant_metrics:
            total_executions = m.success_count + m.failure_count
            if total_executions == 0:
                continue
            
            success_rate = m.success_count / total_executions
            avg_latency = m.total_latency_ms / total_executions if total_executions > 0 else 9999
            avg_quality = sum(m.quality_scores) / len(m.quality_scores) if m.quality_scores else 0.5
            
            # Composite score (weighted)
            score = (
                success_rate * 0.4 +
                (1.0 - min(avg_latency / 5000, 1.0)) * 0.3 +  # Normalize latency
                avg_quality * 0.3
            )
            scores[m.model_id] = score
        
        # Set preference
        if scores:
            best_model = max(scores.items(), key=lambda x: x[1])[0]
            if agent_name not in self.agent_model_preferences:
                self.agent_model_preferences[agent_name] = {}
            self.agent_model_preferences[agent_name][task_type] = best_model
    
    def get_recommended_model(
        self,
        agent_name: str,
        task_type: str,
        fallback: str = "gemini-1.5-flash"
    ) -> str:
        """Get recommended model based on training data"""
        # Check learned preferences
        if agent_name in self.agent_model_preferences:
            if task_type in self.agent_model_preferences[agent_name]:
                return self.agent_model_preferences[agent_name][task_type]
        
        # Fallback to general preference
        if agent_name in self.agent_model_preferences:
            if "general" in self.agent_model_preferences[agent_name]:
                return self.agent_model_preferences[agent_name]["general"]
        
        return fallback
    
    def get_agent_statistics(self, agent_name: str) -> Dict[str, Any]:
        """Get performance statistics for an agent"""
        agent_metrics = [m for m in self.metrics.values() if m.agent_name == agent_name]
        
        if not agent_metrics:
            return {"agent_name": agent_name, "status": "no_data"}
        
        total_success = sum(m.success_count for m in agent_metrics)
        total_failure = sum(m.failure_count for m in agent_metrics)
        total_executions = total_success + total_failure
        
        return {
            "agent_name": agent_name,
            "total_executions": total_executions,
            "success_rate": total_success / total_executions if total_executions > 0 else 0,
            "preferred_models": self.agent_model_preferences.get(agent_name, {}),
            "task_types": list(set(m.task_type for m in agent_metrics)),
            "total_tokens_used": sum(m.total_tokens_used for m in agent_metrics),
            "total_cost": sum(m.total_cost for m in agent_metrics)
        }
    
    def get_all_agents_summary(self) -> List[Dict[str, Any]]:
        """Get summary for all agents"""
        unique_agents = set(m.agent_name for m in self.metrics.values())
        return [self.get_agent_statistics(agent) for agent in unique_agents]

# ============================================================================
# TASK-MODEL TRACKING SYSTEM
# ============================================================================

class TaskModelTracker:
    """
    Real-time tracking of which model is used for which task
    Provides visibility into model usage patterns
    """
    
    def __init__(self):
        self.active_tasks: Dict[str, Dict[str, Any]] = {}
        self.completed_tasks: List[Dict[str, Any]] = []
        self.model_usage_stats: Dict[str, Dict[str, int]] = {}
    
    def start_task(
        self,
        task_id: str,
        agent_name: str,
        task_type: str,
        model_id: str,
        description: str = ""
    ):
        """Record start of a task"""
        self.active_tasks[task_id] = {
            "task_id": task_id,
            "agent_name": agent_name,
            "task_type": task_type,
            "model_id": model_id,
            "description": description,
            "start_time": datetime.now(),
            "status": "running"
        }
        
        # Update usage stats
        if model_id not in self.model_usage_stats:
            self.model_usage_stats[model_id] = {}
        if task_type not in self.model_usage_stats[model_id]:
            self.model_usage_stats[model_id][task_type] = 0
        self.model_usage_stats[model_id][task_type] += 1
    
    def complete_task(
        self,
        task_id: str,
        success: bool,
        result: Any = None,
        error: str = None
    ):
        """Record completion of a task"""
        if task_id not in self.active_tasks:
            logger.warning(f"Task {task_id} not found in active tasks")
            return
        
        task = self.active_tasks.pop(task_id)
        task["end_time"] = datetime.now()
        task["duration_ms"] = (task["end_time"] - task["start_time"]).total_seconds() * 1000
        task["success"] = success
        task["status"] = "completed" if success else "failed"
        
        if error:
            task["error"] = error
        
        # Keep last 1000 completed tasks
        self.completed_tasks.append(task)
        if len(self.completed_tasks) > 1000:
            self.completed_tasks = self.completed_tasks[-1000:]
    
    def get_active_tasks(self) -> List[Dict[str, Any]]:
        """Get all currently active tasks"""
        return list(self.active_tasks.values())
    
    def get_model_usage_summary(self) -> Dict[str, Any]:
        """Get summary of model usage"""
        summary = {}
        for model_id, task_types in self.model_usage_stats.items():
            total_uses = sum(task_types.values())
            summary[model_id] = {
                "total_uses": total_uses,
                "task_breakdown": task_types,
                "active_tasks": len([t for t in self.active_tasks.values() if t["model_id"] == model_id])
            }
        return summary
    
    def get_recent_tasks(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent completed tasks"""
        return self.completed_tasks[-limit:]
    
    def get_task_by_id(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get specific task by ID"""
        # Check active tasks
        if task_id in self.active_tasks:
            return self.active_tasks[task_id]
        
        # Check completed tasks
        for task in reversed(self.completed_tasks):
            if task["task_id"] == task_id:
                return task
        
        return None

# ============================================================================
# MIDDLEWARE INTEGRATION
# ============================================================================

class GeminiMiddleware:
    """
    Middleware that integrates all components:
    - Endpoint configuration
    - Key rotation
    - Agent training
    - Task tracking
    """
    
    def __init__(self):
        self.endpoints = GEMINI_ENDPOINTS
        self.key_manager = APIKeyRotationManager()
        self.training_system = AgentTrainingSystem()
        self.task_tracker = TaskModelTracker()
        
        # Start background tasks
        self._start_background_tasks()
    
    def _start_background_tasks(self):
        """Start background tasks for maintenance"""
        # Reset counters periodically
        asyncio.create_task(self._reset_counters_periodically())
        # Save training data periodically
        asyncio.create_task(self._save_training_data_periodically())
    
    async def _reset_counters_periodically(self):
        """Reset usage counters every minute"""
        while True:
            await asyncio.sleep(60)
            self.key_manager.reset_counters()
    
    async def _save_training_data_periodically(self):
        """Save training data every 5 minutes"""
        while True:
            await asyncio.sleep(300)
            self.training_system._save_training_data()
    
    async def execute_request(
        self,
        agent_name: str,
        task_type: str,
        task_description: str,
        task_id: str = None
    ) -> Dict[str, Any]:
        """
        Execute a request with full middleware integration
        
        This method:
        1. Gets recommended model from training system
        2. Selects appropriate API key
        3. Tracks the task
        4. Records results for training
        
        NOTE: This is a demonstration/template method.
        Replace the simulation code (lines 690-695) with actual Gemini API calls.
        Example integration:
            import google.generativeai as genai
            genai.configure(api_key=key_info.api_key)
            model = genai.GenerativeModel(recommended_model)
            response = await model.generate_content_async(task_description)
        """
        # Generate unique task ID using UUID
        if not task_id:
            task_id = f"task_{uuid.uuid4().hex[:12]}"
        
        # Get recommended model from training
        recommended_model = self.training_system.get_recommended_model(
            agent_name=agent_name,
            task_type=task_type
        )
        
        # Get API key
        key_info = self.key_manager.get_next_key(recommended_model)
        if not key_info:
            return {
                "success": False,
                "error": "No API keys available"
            }
        
        # Start tracking
        self.task_tracker.start_task(
            task_id=task_id,
            agent_name=agent_name,
            task_type=task_type,
            model_id=recommended_model,
            description=task_description
        )
        
        start_time = datetime.now()
        
        try:
            # DEMO: Replace this section with actual Gemini API call
            # Example implementation:
            # import google.generativeai as genai
            # genai.configure(api_key=key_info.api_key)
            # model = genai.GenerativeModel(recommended_model)
            # response = await model.generate_content_async(
            #     task_description,
            #     generation_config={"temperature": 0.7}
            # )
            # result_text = response.text
            # tokens_used = response.usage_metadata.total_token_count
            
            endpoint = self.endpoints[recommended_model]
            
            # DEMO: Simulate API call (remove in production)
            await asyncio.sleep(0.1)  # Simulate network delay
            tokens_used = 100  # Would come from actual API response
            
            # Record success
            latency_ms = (datetime.now() - start_time).total_seconds() * 1000
            
            # Record for key manager
            await self.key_manager.record_request(
                key_id=key_info.key_id,
                model_id=recommended_model,
                success=True
            )
            
            # Record for training
            self.training_system.record_execution(
                agent_name=agent_name,
                model_id=recommended_model,
                task_type=task_type,
                success=True,
                latency_ms=latency_ms,
                tokens_used=100,  # Would come from actual API response
                quality_score=0.85  # Would come from validation
            )
            
            # Complete tracking
            self.task_tracker.complete_task(
                task_id=task_id,
                success=True,
                result="Task completed successfully"
            )
            
            return {
                "success": True,
                "task_id": task_id,
                "model_used": recommended_model,
                "api_key_used": key_info.key_id,
                "latency_ms": latency_ms
            }
            
        except Exception as e:
            # Record failure
            await self.key_manager.record_request(
                key_id=key_info.key_id,
                model_id=recommended_model,
                success=False
            )
            
            self.task_tracker.complete_task(
                task_id=task_id,
                success=False,
                error=str(e)
            )
            
            return {
                "success": False,
                "task_id": task_id,
                "error": str(e)
            }
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data"""
        return {
            "timestamp": datetime.now().isoformat(),
            "api_keys": self.key_manager.get_status(),
            "active_tasks": self.task_tracker.get_active_tasks(),
            "model_usage": self.task_tracker.get_model_usage_summary(),
            "agent_statistics": self.training_system.get_all_agents_summary(),
            "recent_tasks": self.task_tracker.get_recent_tasks(limit=20)
        }


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

async def example_usage():
    """Example of using the advanced configuration system"""
    
    # Initialize middleware
    middleware = GeminiMiddleware()
    
    # Add additional API keys
    middleware.key_manager.add_key(
        key_id="backup_key",
        api_key="your_backup_key_here",
        organization="backup_org"
    )
    
    # Execute requests for multiple agents
    agents = [
        ("coding_agent", "code_generation"),
        ("database_agent", "sql_generation"),
        ("analysis_agent", "data_analysis"),
    ]
    
    for agent_name, task_type in agents:
        result = await middleware.execute_request(
            agent_name=agent_name,
            task_type=task_type,
            task_description=f"Sample {task_type} task"
        )
        print(f"\n{agent_name} - {task_type}:")
        print(f"  Success: {result['success']}")
        print(f"  Model: {result.get('model_used', 'N/A')}")
        print(f"  API Key: {result.get('api_key_used', 'N/A')}")
    
    # Get dashboard data
    dashboard = middleware.get_dashboard_data()
    print("\n=== Dashboard Summary ===")
    print(f"Active tasks: {len(dashboard['active_tasks'])}")
    print(f"API keys: {len(dashboard['api_keys'])}")
    print(f"Agents tracked: {len(dashboard['agent_statistics'])}")
    
    # Get specific agent statistics
    for agent_stat in dashboard['agent_statistics']:
        print(f"\n{agent_stat['agent_name']}:")
        print(f"  Total executions: {agent_stat['total_executions']}")
        print(f"  Success rate: {agent_stat['success_rate']:.2%}")
        print(f"  Preferred models: {agent_stat['preferred_models']}")


if __name__ == "__main__":
    # Run example
    asyncio.run(example_usage())

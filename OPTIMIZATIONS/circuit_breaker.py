"""
Circuit Breaker Pattern - Error Recovery Enhancement
Prevents cascading failures and enables graceful degradation
"""

import time
import asyncio
from typing import Callable, Any, Optional, Dict
from enum import Enum
from dataclasses import dataclass
import logging


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if recovered


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration"""
    failure_threshold: int = 5  # Failures before opening
    success_threshold: int = 2  # Successes to close from half-open
    timeout_seconds: int = 60  # Time before trying half-open
    expected_exception: type = Exception  # Exception type to catch


class CircuitBreakerOpenError(Exception):
    """Raised when circuit is open"""
    pass


class CircuitBreaker:
    """
    Circuit Breaker implementation for AI model calls
    
    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Too many failures, reject requests immediately
    - HALF_OPEN: Testing recovery, allow limited requests
    
    Benefits:
    - Prevent cascading failures
    - Fast failure when service is down
    - Automatic recovery testing
    - Graceful degradation
    """
    
    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None
    ):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        
        # State management
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.last_state_change = time.time()
        
        # Statistics
        self.stats = {
            'total_calls': 0,
            'successful_calls': 0,
            'failed_calls': 0,
            'rejected_calls': 0,
            'state_changes': 0
        }
        
        # Logger
        self.logger = logging.getLogger(f"CircuitBreaker:{name}")
    
    def _change_state(self, new_state: CircuitState):
        """Change circuit breaker state"""
        old_state = self.state
        self.state = new_state
        self.last_state_change = time.time()
        self.stats['state_changes'] += 1
        
        self.logger.info(f"Circuit {self.name}: {old_state.value} -> {new_state.value}")
    
    def _should_attempt_reset(self) -> bool:
        """Check if should attempt reset from OPEN to HALF_OPEN"""
        if self.state != CircuitState.OPEN:
            return False
        
        if self.last_failure_time is None:
            return False
        
        time_since_failure = time.time() - self.last_failure_time
        return time_since_failure >= self.config.timeout_seconds
    
    def _record_success(self):
        """Record successful call"""
        self.failure_count = 0
        self.stats['successful_calls'] += 1
        
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self._change_state(CircuitState.CLOSED)
                self.success_count = 0
    
    def _record_failure(self):
        """Record failed call"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        self.stats['failed_calls'] += 1
        
        if self.state == CircuitState.HALF_OPEN:
            # Failed during recovery test
            self._change_state(CircuitState.OPEN)
            self.success_count = 0
        
        elif self.state == CircuitState.CLOSED:
            # Check if should open circuit
            if self.failure_count >= self.config.failure_threshold:
                self._change_state(CircuitState.OPEN)
    
    async def call_async(
        self,
        func: Callable,
        *args,
        fallback: Optional[Callable] = None,
        **kwargs
    ) -> Any:
        """
        Execute async function with circuit breaker protection
        
        Args:
            func: Async function to execute
            *args: Function arguments
            fallback: Optional fallback function if circuit is open
            **kwargs: Function keyword arguments
        
        Returns:
            Function result or fallback result
        
        Raises:
            CircuitBreakerOpenError: If circuit is open and no fallback
        """
        self.stats['total_calls'] += 1
        
        # Check if should attempt reset
        if self._should_attempt_reset():
            self._change_state(CircuitState.HALF_OPEN)
        
        # Check current state
        if self.state == CircuitState.OPEN:
            self.stats['rejected_calls'] += 1
            
            if fallback:
                self.logger.warning(f"Circuit {self.name} is OPEN, using fallback")
                return await fallback(*args, **kwargs) if asyncio.iscoroutinefunction(fallback) else fallback(*args, **kwargs)
            
            raise CircuitBreakerOpenError(
                f"Circuit breaker '{self.name}' is OPEN. "
                f"Last failure: {time.time() - self.last_failure_time:.1f}s ago"
            )
        
        # Attempt call
        try:
            result = await func(*args, **kwargs)
            self._record_success()
            return result
        
        except self.config.expected_exception as e:
            self._record_failure()
            self.logger.error(f"Circuit {self.name} call failed: {e}")
            
            # Use fallback if available
            if fallback:
                self.logger.info(f"Using fallback for {self.name}")
                return await fallback(*args, **kwargs) if asyncio.iscoroutinefunction(fallback) else fallback(*args, **kwargs)
            
            raise
    
    def call_sync(
        self,
        func: Callable,
        *args,
        fallback: Optional[Callable] = None,
        **kwargs
    ) -> Any:
        """
        Execute sync function with circuit breaker protection
        
        Args:
            func: Function to execute
            *args: Function arguments
            fallback: Optional fallback function if circuit is open
            **kwargs: Function keyword arguments
        
        Returns:
            Function result or fallback result
        """
        self.stats['total_calls'] += 1
        
        # Check if should attempt reset
        if self._should_attempt_reset():
            self._change_state(CircuitState.HALF_OPEN)
        
        # Check current state
        if self.state == CircuitState.OPEN:
            self.stats['rejected_calls'] += 1
            
            if fallback:
                self.logger.warning(f"Circuit {self.name} is OPEN, using fallback")
                return fallback(*args, **kwargs)
            
            raise CircuitBreakerOpenError(
                f"Circuit breaker '{self.name}' is OPEN. "
                f"Last failure: {time.time() - self.last_failure_time:.1f}s ago"
            )
        
        # Attempt call
        try:
            result = func(*args, **kwargs)
            self._record_success()
            return result
        
        except self.config.expected_exception as e:
            self._record_failure()
            self.logger.error(f"Circuit {self.name} call failed: {e}")
            
            # Use fallback if available
            if fallback:
                self.logger.info(f"Using fallback for {self.name}")
                return fallback(*args, **kwargs)
            
            raise
    
    def get_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state"""
        return {
            'name': self.name,
            'state': self.state.value,
            'failure_count': self.failure_count,
            'success_count': self.success_count,
            'last_failure_time': self.last_failure_time,
            'time_in_current_state': time.time() - self.last_state_change,
            'stats': self.stats
        }
    
    def reset(self):
        """Manually reset circuit breaker to CLOSED state"""
        self._change_state(CircuitState.CLOSED)
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None


class CircuitBreakerManager:
    """Manage multiple circuit breakers for different services"""
    
    def __init__(self):
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
    
    def get_or_create(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None
    ) -> CircuitBreaker:
        """Get existing or create new circuit breaker"""
        if name not in self.circuit_breakers:
            self.circuit_breakers[name] = CircuitBreaker(name, config)
        
        return self.circuit_breakers[name]
    
    def get_all_states(self) -> Dict[str, Dict[str, Any]]:
        """Get states of all circuit breakers"""
        return {
            name: cb.get_state()
            for name, cb in self.circuit_breakers.items()
        }
    
    def reset_all(self):
        """Reset all circuit breakers"""
        for cb in self.circuit_breakers.values():
            cb.reset()


# Example usage with AI models
async def call_ai_model_with_circuit_breaker():
    """Example of using circuit breaker with AI model"""
    from langchain_google_genai import ChatGoogleGenerativeAI
    
    # Create circuit breaker
    cb = CircuitBreaker(
        name="gemini-model",
        config=CircuitBreakerConfig(
            failure_threshold=3,
            success_threshold=2,
            timeout_seconds=30
        )
    )
    
    # Define fallback function
    async def fallback_response(*args, **kwargs):
        return "I'm sorry, the AI service is temporarily unavailable. Please try again later."
    
    # Create model
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    
    # Call with circuit breaker protection
    try:
        response = await cb.call_async(
            model.ainvoke,
            "What is machine learning?",
            fallback=fallback_response
        )
        print(f"Response: {response}")
    
    except CircuitBreakerOpenError as e:
        print(f"Circuit breaker is open: {e}")
    
    # Get circuit state
    state = cb.get_state()
    print(f"\nCircuit Breaker State: {state['state']}")
    print(f"Total Calls: {state['stats']['total_calls']}")
    print(f"Success Rate: {state['stats']['successful_calls'] / max(1, state['stats']['total_calls']) * 100:.1f}%")


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Run example
    asyncio.run(call_ai_model_with_circuit_breaker())

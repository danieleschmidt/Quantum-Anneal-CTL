"""
Circuit breaker implementation for quantum HVAC pipeline protection.
"""

import time
import threading
from typing import Callable, Any, Optional
from enum import Enum
from dataclasses import dataclass


class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Circuit broken, fail fast
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    expected_exception: type = Exception


class CircuitBreakerException(Exception):
    """Exception raised when circuit breaker is open."""
    pass


class CircuitBreaker:
    """
    Circuit breaker pattern implementation to prevent cascading failures
    in quantum HVAC pipeline components.
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: type = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = CircuitState.CLOSED
        self._lock = threading.RLock()
        
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function through circuit breaker.
        Raises CircuitBreakerException if circuit is open.
        """
        with self._lock:
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitState.HALF_OPEN
                else:
                    raise CircuitBreakerException(
                        f"Circuit breaker is OPEN. Last failure: {self.last_failure_time}"
                    )
                    
            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
                
            except self.expected_exception as e:
                self._on_failure()
                raise e
                
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.last_failure_time is None:
            return True
            
        return (time.time() - self.last_failure_time) >= self.recovery_timeout
        
    def _on_success(self):
        """Handle successful call."""
        with self._lock:
            self.failure_count = 0
            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.CLOSED
                
    def _on_failure(self):
        """Handle failed call."""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN
                
    def reset(self):
        """Manually reset the circuit breaker."""
        with self._lock:
            self.failure_count = 0
            self.last_failure_time = None
            self.state = CircuitState.CLOSED
            
    def force_open(self):
        """Manually open the circuit breaker."""
        with self._lock:
            self.state = CircuitState.OPEN
            self.last_failure_time = time.time()
            
    def get_status(self) -> dict:
        """Get current circuit breaker status."""
        with self._lock:
            return {
                "state": self.state.value,
                "failure_count": self.failure_count,
                "failure_threshold": self.failure_threshold,
                "last_failure_time": self.last_failure_time,
                "recovery_timeout": self.recovery_timeout,
                "time_until_retry": (
                    max(0, self.recovery_timeout - (time.time() - self.last_failure_time))
                    if self.last_failure_time else 0
                )
            }
            

class QuantumCircuitBreaker(CircuitBreaker):
    """
    Specialized circuit breaker for quantum annealing operations.
    """
    
    def __init__(
        self,
        failure_threshold: int = 3,  # Lower threshold for quantum operations
        recovery_timeout: float = 30.0,  # Shorter recovery time
        chain_break_threshold: float = 0.1,  # Max allowed chain break rate
        **kwargs
    ):
        super().__init__(failure_threshold, recovery_timeout, **kwargs)
        self.chain_break_threshold = chain_break_threshold
        self.chain_break_count = 0
        self.total_quantum_calls = 0
        
    def call_quantum(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute quantum function with enhanced monitoring.
        Monitors chain breaks and quantum-specific failures.
        """
        with self._lock:
            self.total_quantum_calls += 1
            
            try:
                result = self.call(func, *args, **kwargs)
                
                # Check for chain breaks in quantum result
                if hasattr(result, 'info') and 'chain_break_fraction' in result.info:
                    chain_break_fraction = result.info['chain_break_fraction']
                    if chain_break_fraction > self.chain_break_threshold:
                        self.chain_break_count += 1
                        
                        # Consider high chain break rate as partial failure
                        if self.chain_break_count / self.total_quantum_calls > 0.3:
                            self._on_failure()
                            
                return result
                
            except Exception as e:
                # Enhanced error handling for quantum-specific errors
                if "QPU" in str(e) or "annealing" in str(e).lower():
                    # Quantum-specific failure
                    self.failure_count += 2  # Weight quantum failures more
                else:
                    self.failure_count += 1
                    
                self.last_failure_time = time.time()
                
                if self.failure_count >= self.failure_threshold:
                    self.state = CircuitState.OPEN
                    
                raise e
                
    def get_quantum_status(self) -> dict:
        """Get quantum-specific circuit breaker status."""
        status = self.get_status()
        
        with self._lock:
            status.update({
                "total_quantum_calls": self.total_quantum_calls,
                "chain_break_count": self.chain_break_count,
                "chain_break_rate": (
                    self.chain_break_count / self.total_quantum_calls
                    if self.total_quantum_calls > 0 else 0
                ),
                "chain_break_threshold": self.chain_break_threshold
            })
            
        return status
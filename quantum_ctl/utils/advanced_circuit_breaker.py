"""
Advanced Circuit Breaker System for Quantum HVAC Control.

Multi-level circuit breaker system with adaptive thresholds, quantum-aware
failure detection, and intelligent recovery strategies.

Features:
1. Hierarchical circuit breaker patterns
2. Quantum-specific failure modes detection
3. Adaptive threshold adjustment based on quantum performance
4. Intelligent recovery with exponential backoff
5. Health-based circuit breaker state transitions
"""

from typing import Dict, Any, List, Optional, Callable, Union
import asyncio
import time
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import json
import statistics
import numpy as np


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"        # Normal operation
    OPEN = "open"           # Failing, calls blocked
    HALF_OPEN = "half_open" # Testing recovery
    DEGRADED = "degraded"   # Limited operation


class FailureType(Enum):
    """Types of failures that can trigger circuit breaker."""
    TIMEOUT = "timeout"
    QUANTUM_ERROR = "quantum_error" 
    CHAIN_BREAKS = "chain_breaks"
    SOLVER_ERROR = "solver_error"
    NETWORK_ERROR = "network_error"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    THERMAL_VIOLATION = "thermal_violation"
    SAFETY_VIOLATION = "safety_violation"


@dataclass
class FailureRecord:
    """Record of a failure event."""
    timestamp: datetime
    failure_type: FailureType
    error_message: str
    context: Dict[str, Any] = field(default_factory=dict)
    recovery_time: Optional[float] = None
    quantum_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""
    failure_threshold: int = 5
    recovery_timeout: float = 60.0  # seconds
    success_threshold: int = 3
    max_half_open_calls: int = 3
    
    # Adaptive thresholds
    adaptive_thresholds: bool = True
    min_failure_threshold: int = 2
    max_failure_threshold: int = 20
    
    # Quantum-specific parameters
    chain_break_threshold: float = 0.3
    embedding_quality_threshold: float = 0.5
    quantum_advantage_threshold: float = 0.3
    
    # Recovery parameters
    recovery_timeout_multiplier: float = 1.5
    max_recovery_timeout: float = 3600.0  # 1 hour max


class AdaptiveCircuitBreaker:
    """
    Advanced circuit breaker with quantum-aware failure detection
    and adaptive threshold management.
    """
    
    def __init__(
        self,
        name: str,
        config: CircuitBreakerConfig = None,
        fallback_function: Optional[Callable] = None
    ):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.fallback_function = fallback_function
        
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state_changed_time = datetime.now()
        
        # Failure tracking
        self.failure_history: deque = deque(maxlen=1000)
        self.recent_failures: deque = deque(maxlen=100)
        self.failure_patterns: Dict[FailureType, List[datetime]] = defaultdict(list)
        
        # Performance tracking
        self.response_times: deque = deque(maxlen=100)
        self.quantum_performance: deque = deque(maxlen=100)
        
        # Adaptive components
        self.dynamic_threshold = self.config.failure_threshold
        self.adaptive_recovery_timeout = self.config.recovery_timeout
        
        self.logger = logging.getLogger(f"{__name__}.{name}")
        self._lock = asyncio.Lock()
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        async with self._lock:
            # Check circuit state
            if self.state == CircuitState.OPEN:
                if await self._should_attempt_reset():
                    await self._transition_to_half_open()
                else:
                    raise CircuitBreakerOpenError(f"Circuit breaker {self.name} is OPEN")
            
            elif self.state == CircuitState.HALF_OPEN:
                if self.success_count >= self.config.success_threshold:
                    await self._transition_to_closed()
                elif self.failure_count >= self.config.max_half_open_calls:
                    await self._transition_to_open()
                    raise CircuitBreakerOpenError(f"Circuit breaker {self.name} failed recovery")
        
        # Execute function with monitoring
        start_time = time.time()
        try:
            result = await self._execute_with_monitoring(func, *args, **kwargs)
            
            # Record success
            execution_time = time.time() - start_time
            await self._record_success(execution_time, result)
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            await self._record_failure(e, execution_time, *args, **kwargs)
            
            # Try fallback if available
            if self.fallback_function and self.state in [CircuitState.OPEN, CircuitState.DEGRADED]:
                self.logger.info(f"Executing fallback for {self.name}")
                return await self.fallback_function(*args, **kwargs)
            
            raise
    
    async def _execute_with_monitoring(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with detailed monitoring."""
        # Add timeout monitoring
        timeout = self._calculate_adaptive_timeout()
        
        try:
            result = await asyncio.wait_for(func(*args, **kwargs), timeout=timeout)
            return result
        except asyncio.TimeoutError:
            raise CircuitBreakerTimeoutError(f"Function {func.__name__} timed out after {timeout}s")
    
    def _calculate_adaptive_timeout(self) -> float:
        """Calculate adaptive timeout based on recent performance."""
        if not self.response_times:
            return 30.0  # Default timeout
        
        # Use 95th percentile of recent response times
        recent_times = list(self.response_times)
        p95_time = np.percentile(recent_times, 95)
        
        # Add buffer based on circuit state
        buffer_multiplier = {
            CircuitState.CLOSED: 2.0,
            CircuitState.HALF_OPEN: 1.5,
            CircuitState.DEGRADED: 3.0,
            CircuitState.OPEN: 1.0
        }[self.state]
        
        adaptive_timeout = p95_time * buffer_multiplier
        return min(max(adaptive_timeout, 5.0), 120.0)  # Clamp between 5s and 2min
    
    async def _record_success(self, execution_time: float, result: Any) -> None:
        """Record successful execution."""
        self.success_count += 1
        self.failure_count = max(0, self.failure_count - 1)  # Decay failure count
        self.response_times.append(execution_time)
        
        # Extract quantum performance metrics if available
        quantum_metrics = self._extract_quantum_metrics(result)
        if quantum_metrics:
            self.quantum_performance.append(quantum_metrics)
        
        # Adaptive threshold adjustment
        if self.config.adaptive_thresholds:
            await self._adjust_thresholds_on_success()
        
        # State transition logic
        if self.state == CircuitState.HALF_OPEN and self.success_count >= self.config.success_threshold:
            await self._transition_to_closed()
        elif self.state == CircuitState.DEGRADED:
            await self._check_degraded_recovery()
    
    async def _record_failure(self, error: Exception, execution_time: float, *args, **kwargs) -> None:
        """Record failure and update circuit state."""
        failure_type = self._classify_failure(error)
        
        failure_record = FailureRecord(
            timestamp=datetime.now(),
            failure_type=failure_type,
            error_message=str(error),
            context={'args': str(args), 'kwargs': str(kwargs)},
            quantum_metrics=self._extract_quantum_metrics_from_error(error)
        )
        
        self.failure_history.append(failure_record)
        self.recent_failures.append(failure_record)
        self.failure_patterns[failure_type].append(failure_record.timestamp)
        
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        # Quantum-specific failure handling
        if failure_type in [FailureType.QUANTUM_ERROR, FailureType.CHAIN_BREAKS]:
            await self._handle_quantum_failure(failure_record)
        
        # Adaptive threshold adjustment
        if self.config.adaptive_thresholds:
            await self._adjust_thresholds_on_failure(failure_type)
        
        # State transition logic
        if self.failure_count >= self.dynamic_threshold:
            if self.state == CircuitState.CLOSED:
                await self._transition_to_open()
            elif self.state == CircuitState.HALF_OPEN:
                await self._transition_to_open()
            elif self.state == CircuitState.DEGRADED:
                await self._transition_to_open()
    
    def _classify_failure(self, error: Exception) -> FailureType:
        """Classify the type of failure."""
        error_str = str(error).lower()
        
        if isinstance(error, asyncio.TimeoutError) or 'timeout' in error_str:
            return FailureType.TIMEOUT
        elif 'chain' in error_str and 'break' in error_str:
            return FailureType.CHAIN_BREAKS
        elif 'quantum' in error_str or 'qubo' in error_str or 'annealing' in error_str:
            return FailureType.QUANTUM_ERROR
        elif 'solver' in error_str:
            return FailureType.SOLVER_ERROR
        elif 'network' in error_str or 'connection' in error_str:
            return FailureType.NETWORK_ERROR
        elif 'memory' in error_str or 'resource' in error_str:
            return FailureType.RESOURCE_EXHAUSTION
        elif 'temperature' in error_str or 'thermal' in error_str:
            return FailureType.THERMAL_VIOLATION
        elif 'safety' in error_str or 'violation' in error_str:
            return FailureType.SAFETY_VIOLATION
        else:
            return FailureType.SOLVER_ERROR
    
    def _extract_quantum_metrics(self, result: Any) -> Dict[str, float]:
        """Extract quantum performance metrics from result."""
        metrics = {}
        
        if hasattr(result, 'chain_break_fraction'):
            metrics['chain_break_fraction'] = result.chain_break_fraction
        
        if hasattr(result, 'embedding_stats'):
            stats = result.embedding_stats
            if isinstance(stats, dict):
                metrics.update({
                    'embedding_quality': stats.get('embedding_quality', 0.0),
                    'quantum_advantage': stats.get('quantum_advantage', 0.0)
                })
        
        if hasattr(result, 'performance_metrics'):
            perf = result.performance_metrics
            if hasattr(perf, 'quantum_advantage_score'):
                metrics['quantum_advantage_score'] = perf.quantum_advantage_score
        
        return metrics
    
    def _extract_quantum_metrics_from_error(self, error: Exception) -> Dict[str, float]:
        """Extract quantum metrics from error context if available."""
        # In practice, quantum errors might contain performance data
        return {}
    
    async def _handle_quantum_failure(self, failure_record: FailureRecord) -> None:
        """Handle quantum-specific failures with specialized logic."""
        if failure_record.failure_type == FailureType.CHAIN_BREAKS:
            # High chain breaks indicate embedding issues
            self.logger.warning(f"High chain breaks detected in {self.name}")
            
            # Increase chain strength threshold
            self.config.chain_break_threshold = min(0.5, self.config.chain_break_threshold * 1.2)
            
        elif failure_record.failure_type == FailureType.QUANTUM_ERROR:
            # General quantum solver issues
            self.logger.warning(f"Quantum solver error in {self.name}")
            
            # Consider transitioning to degraded mode with classical fallback
            if self.failure_count >= self.dynamic_threshold // 2:
                await self._transition_to_degraded()
    
    async def _adjust_thresholds_on_success(self) -> None:
        """Adjust thresholds based on successful operations."""
        # Gradually increase threshold on sustained success
        if self.success_count >= self.dynamic_threshold * 2:
            self.dynamic_threshold = min(
                self.config.max_failure_threshold,
                self.dynamic_threshold + 1
            )
            self.success_count = 0  # Reset counter
    
    async def _adjust_thresholds_on_failure(self, failure_type: FailureType) -> None:
        """Adjust thresholds based on failure patterns."""
        # More aggressive threshold reduction for critical failures
        critical_failures = [
            FailureType.SAFETY_VIOLATION,
            FailureType.THERMAL_VIOLATION,
            FailureType.RESOURCE_EXHAUSTION
        ]
        
        if failure_type in critical_failures:
            self.dynamic_threshold = max(
                self.config.min_failure_threshold,
                self.dynamic_threshold - 2
            )
        else:
            # Gradual threshold reduction
            recent_failure_rate = len(self.recent_failures) / max(len(self.failure_history), 1)
            if recent_failure_rate > 0.3:  # 30% recent failure rate
                self.dynamic_threshold = max(
                    self.config.min_failure_threshold,
                    self.dynamic_threshold - 1
                )
    
    async def _should_attempt_reset(self) -> bool:
        """Determine if circuit should attempt reset."""
        if not self.last_failure_time:
            return True
        
        time_since_failure = (datetime.now() - self.last_failure_time).total_seconds()
        return time_since_failure >= self.adaptive_recovery_timeout
    
    async def _transition_to_open(self) -> None:
        """Transition circuit breaker to OPEN state."""
        if self.state != CircuitState.OPEN:
            self.logger.warning(f"Circuit breaker {self.name} transitioning to OPEN")
            self.state = CircuitState.OPEN
            self.state_changed_time = datetime.now()
            
            # Increase recovery timeout for next attempt
            self.adaptive_recovery_timeout = min(
                self.config.max_recovery_timeout,
                self.adaptive_recovery_timeout * self.config.recovery_timeout_multiplier
            )
    
    async def _transition_to_half_open(self) -> None:
        """Transition circuit breaker to HALF_OPEN state."""
        self.logger.info(f"Circuit breaker {self.name} transitioning to HALF_OPEN")
        self.state = CircuitState.HALF_OPEN
        self.state_changed_time = datetime.now()
        self.success_count = 0
        self.failure_count = 0
    
    async def _transition_to_closed(self) -> None:
        """Transition circuit breaker to CLOSED state."""
        self.logger.info(f"Circuit breaker {self.name} transitioning to CLOSED")
        self.state = CircuitState.CLOSED
        self.state_changed_time = datetime.now()
        self.success_count = 0
        self.failure_count = 0
        
        # Reset adaptive parameters on successful recovery
        self.adaptive_recovery_timeout = self.config.recovery_timeout
    
    async def _transition_to_degraded(self) -> None:
        """Transition circuit breaker to DEGRADED state."""
        self.logger.warning(f"Circuit breaker {self.name} transitioning to DEGRADED")
        self.state = CircuitState.DEGRADED
        self.state_changed_time = datetime.now()
        self.failure_count = 0  # Reset to give degraded mode a chance
    
    async def _check_degraded_recovery(self) -> None:
        """Check if circuit can recover from degraded state."""
        if self.success_count >= self.config.success_threshold * 2:
            await self._transition_to_closed()
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive circuit breaker status."""
        time_in_state = (datetime.now() - self.state_changed_time).total_seconds()
        
        # Calculate failure rates
        total_calls = len(self.failure_history) + self.success_count
        failure_rate = len(self.failure_history) / max(total_calls, 1)
        
        recent_calls = len(self.recent_failures) + min(self.success_count, 20)
        recent_failure_rate = len(self.recent_failures) / max(recent_calls, 1)
        
        # Performance metrics
        avg_response_time = statistics.mean(self.response_times) if self.response_times else 0.0
        
        # Quantum metrics
        quantum_health = {}
        if self.quantum_performance:
            recent_quantum = list(self.quantum_performance)[-10:]
            if recent_quantum and len(recent_quantum[0]) > 0:
                for metric in recent_quantum[0].keys():
                    values = [q.get(metric, 0.0) for q in recent_quantum if metric in q]
                    if values:
                        quantum_health[f'avg_{metric}'] = statistics.mean(values)
        
        return {
            'name': self.name,
            'state': self.state.value,
            'time_in_state_seconds': time_in_state,
            'failure_count': self.failure_count,
            'success_count': self.success_count,
            'dynamic_threshold': self.dynamic_threshold,
            'adaptive_recovery_timeout': self.adaptive_recovery_timeout,
            'metrics': {
                'total_calls': total_calls,
                'failure_rate': failure_rate,
                'recent_failure_rate': recent_failure_rate,
                'avg_response_time': avg_response_time,
                'quantum_health': quantum_health
            },
            'failure_patterns': {
                failure_type.value: len(timestamps)
                for failure_type, timestamps in self.failure_patterns.items()
            },
            'config': {
                'failure_threshold': self.config.failure_threshold,
                'recovery_timeout': self.config.recovery_timeout,
                'adaptive_thresholds': self.config.adaptive_thresholds
            }
        }
    
    async def reset(self) -> None:
        """Manually reset circuit breaker."""
        async with self._lock:
            self.logger.info(f"Manually resetting circuit breaker {self.name}")
            await self._transition_to_closed()
    
    async def force_open(self) -> None:
        """Manually force circuit breaker open."""
        async with self._lock:
            self.logger.warning(f"Manually forcing circuit breaker {self.name} to OPEN")
            await self._transition_to_open()


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open."""
    pass


class CircuitBreakerTimeoutError(Exception):
    """Exception raised when operation times out."""
    pass


class HierarchicalCircuitBreakerSystem:
    """
    Hierarchical system of circuit breakers for different system components.
    
    Manages multiple circuit breakers with different policies and provides
    system-wide failure isolation and recovery coordination.
    """
    
    def __init__(self):
        self.circuit_breakers: Dict[str, AdaptiveCircuitBreaker] = {}
        self.circuit_hierarchy: Dict[str, List[str]] = {}  # parent -> children
        self.global_health_score = 1.0
        self.logger = logging.getLogger(__name__)
    
    def create_circuit_breaker(
        self,
        name: str,
        config: CircuitBreakerConfig = None,
        fallback_function: Optional[Callable] = None,
        parent: Optional[str] = None
    ) -> AdaptiveCircuitBreaker:
        """Create and register a new circuit breaker."""
        circuit_breaker = AdaptiveCircuitBreaker(name, config, fallback_function)
        self.circuit_breakers[name] = circuit_breaker
        
        # Set up hierarchy
        if parent:
            if parent not in self.circuit_hierarchy:
                self.circuit_hierarchy[parent] = []
            self.circuit_hierarchy[parent].append(name)
        
        self.logger.info(f"Created circuit breaker: {name}" + (f" (parent: {parent})" if parent else ""))
        return circuit_breaker
    
    async def execute_with_circuit_breaker(
        self,
        circuit_name: str,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Execute function with specified circuit breaker protection."""
        if circuit_name not in self.circuit_breakers:
            raise ValueError(f"Circuit breaker {circuit_name} not found")
        
        circuit_breaker = self.circuit_breakers[circuit_name]
        
        try:
            result = await circuit_breaker.call(func, *args, **kwargs)
            await self._update_global_health()
            return result
            
        except Exception as e:
            await self._handle_circuit_failure(circuit_name, e)
            raise
    
    async def _handle_circuit_failure(self, circuit_name: str, error: Exception) -> None:
        """Handle circuit breaker failure with hierarchy consideration."""
        # Check if failure should cascade to parent circuits
        parent_circuits = [
            parent for parent, children in self.circuit_hierarchy.items()
            if circuit_name in children
        ]
        
        for parent in parent_circuits:
            parent_circuit = self.circuit_breakers.get(parent)
            if parent_circuit:
                # Increase parent circuit failure count for cascading failures
                parent_circuit.failure_count += 0.5  # Partial failure contribution
                
                self.logger.warning(
                    f"Circuit {circuit_name} failure cascading to parent {parent}"
                )
        
        await self._update_global_health()
    
    async def _update_global_health(self) -> None:
        """Update global system health score."""
        if not self.circuit_breakers:
            self.global_health_score = 1.0
            return
        
        health_scores = []
        
        for circuit_breaker in self.circuit_breakers.values():
            # Calculate health score for each circuit
            if circuit_breaker.state == CircuitState.CLOSED:
                health = 1.0
            elif circuit_breaker.state == CircuitState.DEGRADED:
                health = 0.6
            elif circuit_breaker.state == CircuitState.HALF_OPEN:
                health = 0.4
            else:  # OPEN
                health = 0.1
            
            # Adjust based on recent performance
            recent_failure_rate = len(circuit_breaker.recent_failures) / max(
                len(circuit_breaker.recent_failures) + circuit_breaker.success_count, 1
            )
            health *= (1.0 - recent_failure_rate * 0.5)
            
            health_scores.append(health)
        
        # Global health is weighted average (critical circuits weighted more)
        self.global_health_score = statistics.mean(health_scores)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        circuit_statuses = {}
        for name, circuit_breaker in self.circuit_breakers.items():
            circuit_statuses[name] = circuit_breaker.get_status()
        
        # Calculate aggregate metrics
        total_circuits = len(self.circuit_breakers)
        open_circuits = sum(1 for cb in self.circuit_breakers.values() if cb.state == CircuitState.OPEN)
        degraded_circuits = sum(1 for cb in self.circuit_breakers.values() if cb.state == CircuitState.DEGRADED)
        
        return {
            'global_health_score': self.global_health_score,
            'circuit_summary': {
                'total_circuits': total_circuits,
                'open_circuits': open_circuits,
                'degraded_circuits': degraded_circuits,
                'healthy_circuits': total_circuits - open_circuits - degraded_circuits
            },
            'circuit_statuses': circuit_statuses,
            'hierarchy': self.circuit_hierarchy
        }
    
    async def reset_all_circuits(self) -> None:
        """Reset all circuit breakers."""
        self.logger.info("Resetting all circuit breakers")
        
        for circuit_breaker in self.circuit_breakers.values():
            await circuit_breaker.reset()
        
        await self._update_global_health()
    
    async def emergency_shutdown(self) -> None:
        """Emergency shutdown - open all circuit breakers."""
        self.logger.critical("EMERGENCY SHUTDOWN - Opening all circuit breakers")
        
        for circuit_breaker in self.circuit_breakers.values():
            await circuit_breaker.force_open()
        
        await self._update_global_health()


# Global circuit breaker system
_circuit_breaker_system: Optional[HierarchicalCircuitBreakerSystem] = None


def get_circuit_breaker_system() -> HierarchicalCircuitBreakerSystem:
    """Get global circuit breaker system."""
    global _circuit_breaker_system
    if _circuit_breaker_system is None:
        _circuit_breaker_system = HierarchicalCircuitBreakerSystem()
    return _circuit_breaker_system


async def initialize_quantum_circuit_breakers() -> HierarchicalCircuitBreakerSystem:
    """Initialize circuit breakers for quantum HVAC system."""
    system = get_circuit_breaker_system()
    
    # Main system circuit breaker
    system.create_circuit_breaker(
        "main_system",
        CircuitBreakerConfig(
            failure_threshold=10,
            recovery_timeout=300.0,
            adaptive_thresholds=True
        )
    )
    
    # Quantum solver circuit breaker
    quantum_config = CircuitBreakerConfig(
        failure_threshold=5,
        recovery_timeout=120.0,
        chain_break_threshold=0.2,
        embedding_quality_threshold=0.6,
        adaptive_thresholds=True
    )
    
    system.create_circuit_breaker(
        "quantum_solver",
        quantum_config,
        parent="main_system"
    )
    
    # Building control circuit breakers
    building_config = CircuitBreakerConfig(
        failure_threshold=3,
        recovery_timeout=60.0,
        adaptive_thresholds=True
    )
    
    system.create_circuit_breaker(
        "building_control",
        building_config,
        parent="main_system"
    )
    
    # Sensor integration circuit breaker
    sensor_config = CircuitBreakerConfig(
        failure_threshold=8,
        recovery_timeout=30.0,
        adaptive_thresholds=True
    )
    
    system.create_circuit_breaker(
        "sensor_integration",
        sensor_config,
        parent="building_control"
    )
    
    return system
"""
Advanced circuit breaker implementation for quantum annealing systems.

This module provides sophisticated circuit breaking with quantum-specific
failure patterns, adaptive thresholds, and multi-dimensional health metrics.
"""

import time
import asyncio
from typing import Dict, Any, Optional, List, Callable, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
import statistics
from collections import deque, defaultdict

from ..optimization.quantum_solver import QuantumSolution


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"         # Circuit is open, requests fail immediately
    HALF_OPEN = "half_open"  # Testing if service has recovered


@dataclass
class FailureMetrics:
    """Metrics for failure analysis."""
    total_requests: int = 0
    failed_requests: int = 0
    chain_break_failures: int = 0
    timeout_failures: int = 0
    quantum_specific_failures: int = 0
    average_response_time: float = 0.0
    recent_response_times: List[float] = None
    
    def __post_init__(self):
        if self.recent_response_times is None:
            self.recent_response_times = []


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: float = 0.5  # Failure rate threshold (0-1)
    min_requests: int = 10  # Minimum requests before evaluation
    timeout_threshold: float = 30.0  # Request timeout threshold
    recovery_timeout: float = 60.0  # How long to wait before trying half-open
    chain_break_threshold: float = 0.3  # Chain break failure threshold
    response_time_threshold: float = 120.0  # Response time threshold
    quantum_failure_weight: float = 2.0  # Weight for quantum-specific failures


class QuantumCircuitBreaker:
    """
    Advanced circuit breaker for quantum annealing operations.
    
    Monitors quantum solver health using multiple metrics and provides
    intelligent failure detection with quantum-specific patterns.
    """
    
    def __init__(
        self,
        name: str,
        config: CircuitBreakerConfig = None,
        notification_callback: Optional[Callable] = None
    ):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.notification_callback = notification_callback
        
        self.logger = logging.getLogger(__name__)
        
        # Circuit state
        self.state = CircuitState.CLOSED
        self.failure_metrics = FailureMetrics()
        self.last_failure_time = 0
        self.last_success_time = time.time()
        
        # State transition history
        self.state_history: deque = deque(maxlen=100)
        self.state_changed_at = time.time()
        
        # Adaptive thresholds
        self.adaptive_thresholds = {
            'failure_rate': self.config.failure_threshold,
            'response_time': self.config.response_time_threshold,
            'chain_breaks': self.config.chain_break_threshold
        }
        
        # Health scoring system
        self.health_score = 1.0
        self.health_history: deque = deque(maxlen=50)
        
        # Quantum-specific failure patterns
        self.quantum_failure_patterns = {
            'embedding_failures': 0,
            'qpu_unavailable': 0,
            'calibration_errors': 0,
            'excessive_chain_breaks': 0
        }
        
    async def call(
        self,
        operation: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Execute operation through circuit breaker.
        
        Args:
            operation: The operation to execute
            *args: Operation arguments
            **kwargs: Operation keyword arguments
            
        Returns:
            Operation result
            
        Raises:
            CircuitBreakerOpenError: When circuit is open
        """
        
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                await self._transition_to_half_open()
            else:
                raise CircuitBreakerOpenError(
                    f"Circuit breaker {self.name} is OPEN. Last failure: {self.last_failure_time}"
                )
                
        start_time = time.time()
        
        try:
            # Execute the operation
            result = await self._execute_with_timeout(operation, *args, **kwargs)
            
            # Record successful execution
            execution_time = time.time() - start_time
            await self._record_success(execution_time, result)
            
            return result
            
        except Exception as e:
            # Record failure
            execution_time = time.time() - start_time
            await self._record_failure(execution_time, e)
            
            # Re-raise the exception
            raise
            
    async def _execute_with_timeout(
        self,
        operation: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Execute operation with timeout protection."""
        
        try:
            if asyncio.iscoroutinefunction(operation):
                result = await asyncio.wait_for(
                    operation(*args, **kwargs),
                    timeout=self.config.timeout_threshold
                )
            else:
                result = await asyncio.wait_for(
                    asyncio.to_thread(operation, *args, **kwargs),
                    timeout=self.config.timeout_threshold
                )
                
            return result
            
        except asyncio.TimeoutError:
            raise QuantumTimeoutError(f"Operation timed out after {self.config.timeout_threshold}s")
            
    async def _record_success(self, execution_time: float, result: Any) -> None:
        """Record successful execution and update metrics."""
        
        self.failure_metrics.total_requests += 1
        self.last_success_time = time.time()
        
        # Update response time metrics
        self.failure_metrics.recent_response_times.append(execution_time)
        if len(self.failure_metrics.recent_response_times) > 20:
            self.failure_metrics.recent_response_times.pop(0)
            
        self.failure_metrics.average_response_time = statistics.mean(
            self.failure_metrics.recent_response_times
        )
        
        # Analyze result for quantum-specific metrics
        if isinstance(result, QuantumSolution):
            self._analyze_quantum_solution(result)
            
        # Update health score
        self._update_health_score(success=True, execution_time=execution_time)
        
        # Check if we should transition to CLOSED from HALF_OPEN
        if self.state == CircuitState.HALF_OPEN:
            await self._transition_to_closed()
            
        self.logger.debug(f"Circuit {self.name}: Success recorded. Health score: {self.health_score:.3f}")
        
    async def _record_failure(self, execution_time: float, exception: Exception) -> None:
        """Record failure and update metrics."""
        
        self.failure_metrics.total_requests += 1
        self.failure_metrics.failed_requests += 1
        self.last_failure_time = time.time()
        
        # Classify the type of failure
        failure_type = self._classify_failure(exception)
        
        if failure_type == 'timeout':
            self.failure_metrics.timeout_failures += 1
        elif failure_type == 'chain_breaks':
            self.failure_metrics.chain_break_failures += 1
        elif failure_type == 'quantum_specific':
            self.failure_metrics.quantum_specific_failures += 1
            
        # Update quantum-specific failure patterns
        self._update_quantum_failure_patterns(exception)
        
        # Update health score
        self._update_health_score(success=False, execution_time=execution_time)
        
        # Check if circuit should open
        if await self._should_open_circuit():
            await self._transition_to_open()
            
        self.logger.warning(
            f"Circuit {self.name}: Failure recorded. Type: {failure_type}. "
            f"Health score: {self.health_score:.3f}. State: {self.state.value}"
        )
        
    def _classify_failure(self, exception: Exception) -> str:
        """Classify the type of failure."""
        
        exception_str = str(exception).lower()
        
        if 'timeout' in exception_str:
            return 'timeout'
        elif 'chain' in exception_str and 'break' in exception_str:
            return 'chain_breaks'
        elif any(keyword in exception_str for keyword in [
            'embedding', 'qpu', 'quantum', 'annealer', 'solver'
        ]):
            return 'quantum_specific'
        else:
            return 'generic'
            
    def _analyze_quantum_solution(self, solution: QuantumSolution) -> None:
        """Analyze quantum solution for health indicators."""
        
        # Check chain break fraction
        if solution.chain_break_fraction > self.config.chain_break_threshold:
            self.quantum_failure_patterns['excessive_chain_breaks'] += 1
            
        # Check for embedding issues (high chain break rate)
        if solution.chain_break_fraction > 0.5:
            self.quantum_failure_patterns['embedding_failures'] += 1
            
    def _update_quantum_failure_patterns(self, exception: Exception) -> None:
        """Update quantum-specific failure pattern tracking."""
        
        exception_str = str(exception).lower()
        
        if 'embedding' in exception_str:
            self.quantum_failure_patterns['embedding_failures'] += 1
        elif 'qpu' in exception_str and 'unavailable' in exception_str:
            self.quantum_failure_patterns['qpu_unavailable'] += 1
        elif 'calibration' in exception_str:
            self.quantum_failure_patterns['calibration_errors'] += 1
            
    def _update_health_score(self, success: bool, execution_time: float) -> None:
        """Update circuit health score using multiple factors."""
        
        # Base score adjustment
        if success:
            score_delta = 0.1 * (1.0 - self.health_score)  # Faster recovery when health is low
        else:
            score_delta = -0.2  # Penalty for failure
            
        # Adjust based on response time
        if execution_time > self.config.response_time_threshold:
            score_delta -= 0.1  # Penalty for slow response
            
        # Apply quantum-specific adjustments
        recent_quantum_failures = sum(self.quantum_failure_patterns.values())
        if recent_quantum_failures > 5:  # Recent quantum issues
            score_delta -= 0.1
            
        # Update health score
        self.health_score = max(0.0, min(1.0, self.health_score + score_delta))
        
        # Record in history
        self.health_history.append((time.time(), self.health_score))
        
        # Update adaptive thresholds based on health trend
        self._update_adaptive_thresholds()
        
    def _update_adaptive_thresholds(self) -> None:
        """Update adaptive thresholds based on historical performance."""
        
        if len(self.health_history) < 10:
            return
            
        # Analyze health trend
        recent_scores = [score for _, score in list(self.health_history)[-10:]]
        health_trend = statistics.mean(recent_scores)
        
        # Adjust thresholds based on trend
        if health_trend > 0.8:
            # System is healthy, can be more lenient
            self.adaptive_thresholds['failure_rate'] = min(
                self.config.failure_threshold * 1.2,
                0.8
            )
        elif health_trend < 0.3:
            # System is unhealthy, be more strict
            self.adaptive_thresholds['failure_rate'] = max(
                self.config.failure_threshold * 0.8,
                0.1
            )
            
    async def _should_open_circuit(self) -> bool:
        """Determine if circuit should open based on current metrics."""
        
        # Need minimum requests for evaluation
        if self.failure_metrics.total_requests < self.config.min_requests:
            return False
            
        # Calculate current failure rate
        failure_rate = self.failure_metrics.failed_requests / self.failure_metrics.total_requests
        
        # Check failure rate threshold
        if failure_rate > self.adaptive_thresholds['failure_rate']:
            return True
            
        # Check quantum-specific failure patterns
        quantum_failure_rate = (
            self.failure_metrics.quantum_specific_failures / 
            max(self.failure_metrics.total_requests, 1)
        )
        
        if quantum_failure_rate > 0.2:  # High quantum failure rate
            return True
            
        # Check health score
        if self.health_score < 0.2:
            return True
            
        # Check chain break rate
        if (
            self.failure_metrics.chain_break_failures / 
            max(self.failure_metrics.total_requests, 1) > 
            self.adaptive_thresholds['chain_breaks']
        ):
            return True
            
        return False
        
    def _should_attempt_reset(self) -> bool:
        """Check if circuit should attempt reset from OPEN state."""
        
        current_time = time.time()
        time_since_failure = current_time - self.last_failure_time
        
        return time_since_failure >= self.config.recovery_timeout
        
    async def _transition_to_open(self) -> None:
        """Transition circuit to OPEN state."""
        
        old_state = self.state
        self.state = CircuitState.OPEN
        self.state_changed_at = time.time()
        
        self._record_state_transition(old_state, self.state)
        
        self.logger.warning(
            f"Circuit {self.name} OPENED. Failure rate: "
            f"{self.failure_metrics.failed_requests}/{self.failure_metrics.total_requests}"
        )
        
        if self.notification_callback:
            await self.notification_callback(
                'circuit_opened',
                self.name,
                {
                    'failure_rate': self.failure_metrics.failed_requests / self.failure_metrics.total_requests,
                    'health_score': self.health_score,
                    'quantum_failures': self.quantum_failure_patterns
                }
            )
            
    async def _transition_to_half_open(self) -> None:
        """Transition circuit to HALF_OPEN state."""
        
        old_state = self.state
        self.state = CircuitState.HALF_OPEN
        self.state_changed_at = time.time()
        
        self._record_state_transition(old_state, self.state)
        
        self.logger.info(f"Circuit {self.name} transitioned to HALF_OPEN for testing")
        
        if self.notification_callback:
            await self.notification_callback(
                'circuit_half_open',
                self.name,
                {'time_since_failure': time.time() - self.last_failure_time}
            )
            
    async def _transition_to_closed(self) -> None:
        """Transition circuit to CLOSED state."""
        
        old_state = self.state
        self.state = CircuitState.CLOSED
        self.state_changed_at = time.time()
        
        # Reset failure metrics on successful recovery
        self._reset_failure_metrics()
        
        self._record_state_transition(old_state, self.state)
        
        self.logger.info(f"Circuit {self.name} CLOSED - service recovered")
        
        if self.notification_callback:
            await self.notification_callback(
                'circuit_closed',
                self.name,
                {'recovery_time': time.time() - self.last_failure_time}
            )
            
    def _reset_failure_metrics(self) -> None:
        """Reset failure metrics after recovery."""
        
        # Partial reset - keep some history for adaptive behavior
        self.failure_metrics.failed_requests = 0
        self.failure_metrics.total_requests = 0
        self.failure_metrics.chain_break_failures = 0
        self.failure_metrics.timeout_failures = 0
        self.failure_metrics.quantum_specific_failures = 0
        
        # Reset quantum failure patterns
        for pattern in self.quantum_failure_patterns:
            self.quantum_failure_patterns[pattern] = 0
            
    def _record_state_transition(self, from_state: CircuitState, to_state: CircuitState) -> None:
        """Record state transition in history."""
        
        self.state_history.append({
            'timestamp': time.time(),
            'from_state': from_state.value,
            'to_state': to_state.value,
            'health_score': self.health_score,
            'failure_rate': (
                self.failure_metrics.failed_requests / 
                max(self.failure_metrics.total_requests, 1)
            )
        })
        
    def get_status(self) -> Dict[str, Any]:
        """Get current circuit breaker status."""
        
        current_time = time.time()
        
        return {
            'name': self.name,
            'state': self.state.value,
            'health_score': self.health_score,
            'time_in_current_state': current_time - self.state_changed_at,
            'failure_metrics': {
                'total_requests': self.failure_metrics.total_requests,
                'failed_requests': self.failure_metrics.failed_requests,
                'failure_rate': (
                    self.failure_metrics.failed_requests / 
                    max(self.failure_metrics.total_requests, 1)
                ),
                'average_response_time': self.failure_metrics.average_response_time,
                'timeout_failures': self.failure_metrics.timeout_failures,
                'chain_break_failures': self.failure_metrics.chain_break_failures,
                'quantum_specific_failures': self.failure_metrics.quantum_specific_failures
            },
            'quantum_failure_patterns': self.quantum_failure_patterns,
            'adaptive_thresholds': self.adaptive_thresholds,
            'last_failure_time': self.last_failure_time,
            'last_success_time': self.last_success_time
        }
        
    def get_state_history(self) -> List[Dict[str, Any]]:
        """Get circuit state transition history."""
        
        return list(self.state_history)
        
    async def force_open(self) -> None:
        """Force circuit to open (for testing/emergency)."""
        
        await self._transition_to_open()
        self.logger.warning(f"Circuit {self.name} force opened")
        
    async def force_closed(self) -> None:
        """Force circuit to closed (for testing/recovery)."""
        
        await self._transition_to_closed()
        self.logger.info(f"Circuit {self.name} force closed")


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open."""
    pass


class QuantumTimeoutError(Exception):
    """Exception raised when quantum operation times out."""
    pass


class CircuitBreakerManager:
    """
    Manager for multiple circuit breakers with coordination.
    
    Provides centralized management and coordination between
    multiple circuit breakers in a quantum system.
    """
    
    def __init__(self):
        self.circuit_breakers: Dict[str, QuantumCircuitBreaker] = {}
        self.logger = logging.getLogger(__name__)
        
        # Global health tracking
        self.global_health_score = 1.0
        self.system_degradation_level = 0
        
    def register_circuit_breaker(
        self,
        name: str,
        config: CircuitBreakerConfig = None
    ) -> QuantumCircuitBreaker:
        """Register a new circuit breaker."""
        
        circuit_breaker = QuantumCircuitBreaker(
            name=name,
            config=config,
            notification_callback=self._handle_circuit_notification
        )
        
        self.circuit_breakers[name] = circuit_breaker
        
        self.logger.info(f"Registered circuit breaker: {name}")
        return circuit_breaker
        
    async def _handle_circuit_notification(
        self,
        event: str,
        circuit_name: str,
        details: Dict[str, Any]
    ) -> None:
        """Handle circuit breaker notifications."""
        
        self.logger.info(f"Circuit breaker event: {event} from {circuit_name}")
        
        # Update global health
        await self._update_global_health()
        
        # Coordinate with other circuits if needed
        if event == 'circuit_opened':
            await self._handle_circuit_opened(circuit_name, details)
            
    async def _update_global_health(self) -> None:
        """Update global system health based on all circuits."""
        
        if not self.circuit_breakers:
            self.global_health_score = 1.0
            return
            
        # Calculate weighted average health
        total_weight = 0
        weighted_health = 0
        
        for cb in self.circuit_breakers.values():
            weight = 1.0
            if cb.state == CircuitState.OPEN:
                weight = 0.1  # Very low weight for open circuits
            elif cb.state == CircuitState.HALF_OPEN:
                weight = 0.5  # Reduced weight for half-open
                
            weighted_health += cb.health_score * weight
            total_weight += weight
            
        if total_weight > 0:
            self.global_health_score = weighted_health / total_weight
        else:
            self.global_health_score = 0.0
            
        # Update system degradation level
        if self.global_health_score > 0.8:
            self.system_degradation_level = 0
        elif self.global_health_score > 0.6:
            self.system_degradation_level = 1
        elif self.global_health_score > 0.3:
            self.system_degradation_level = 2
        else:
            self.system_degradation_level = 3
            
    async def _handle_circuit_opened(self, circuit_name: str, details: Dict[str, Any]) -> None:
        """Handle when a circuit breaker opens."""
        
        # Check if multiple circuits are failing
        open_circuits = [
            name for name, cb in self.circuit_breakers.items() 
            if cb.state == CircuitState.OPEN
        ]
        
        if len(open_circuits) > len(self.circuit_breakers) * 0.5:
            self.logger.critical(
                f"System degradation: {len(open_circuits)} circuits open. "
                f"Global health: {self.global_health_score:.3f}"
            )
            
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status."""
        
        circuit_statuses = {}
        for name, cb in self.circuit_breakers.items():
            circuit_statuses[name] = cb.get_status()
            
        return {
            'global_health_score': self.global_health_score,
            'system_degradation_level': self.system_degradation_level,
            'total_circuits': len(self.circuit_breakers),
            'circuits_by_state': {
                'closed': len([cb for cb in self.circuit_breakers.values() if cb.state == CircuitState.CLOSED]),
                'open': len([cb for cb in self.circuit_breakers.values() if cb.state == CircuitState.OPEN]),
                'half_open': len([cb for cb in self.circuit_breakers.values() if cb.state == CircuitState.HALF_OPEN])
            },
            'circuit_details': circuit_statuses
        }
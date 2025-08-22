"""
Intelligent recovery system for quantum HVAC control failures.
Multi-level recovery strategies with adaptive learning.
"""

import time
import logging
import numpy as np
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import asyncio
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class RecoveryStrategy(Enum):
    """Recovery strategy types."""
    IMMEDIATE_RETRY = "immediate_retry"
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    QUANTUM_FALLBACK = "quantum_fallback"
    CLASSICAL_FALLBACK = "classical_fallback"
    PROBLEM_DECOMPOSITION = "problem_decomposition"
    CACHED_SOLUTION = "cached_solution"
    SAFE_MODE = "safe_mode"
    GRACEFUL_DEGRADATION = "graceful_degradation"


class FailureType(Enum):
    """Types of system failures."""
    QUANTUM_TIMEOUT = "quantum_timeout"
    QUANTUM_ERROR = "quantum_error"
    OPTIMIZATION_FAILED = "optimization_failed"
    VALIDATION_ERROR = "validation_error"
    COMMUNICATION_ERROR = "communication_error"
    RESOURCE_EXHAUSTED = "resource_exhausted"
    UNKNOWN_ERROR = "unknown_error"


@dataclass
class RecoveryAction:
    """Recovery action with metadata."""
    strategy: RecoveryStrategy
    priority: int
    estimated_success_rate: float
    estimated_time: float
    prerequisites: List[str] = field(default_factory=list)
    description: str = ""


@dataclass
class FailureEvent:
    """Failure event record."""
    failure_type: FailureType
    timestamp: float
    context: Dict[str, Any]
    error_message: str
    recovery_attempted: Optional[RecoveryStrategy] = None
    recovery_successful: bool = False
    recovery_time: float = 0.0


class IntelligentRecoveryManager:
    """Intelligent recovery manager with adaptive strategies."""
    
    def __init__(self):
        self.failure_history = deque(maxlen=1000)
        self.recovery_success_rates = defaultdict(lambda: defaultdict(float))
        self.strategy_performance = defaultdict(list)
        self.cached_solutions = {}
        self.safe_mode_active = False
        
    def record_failure(self, failure_type: FailureType, context: Dict[str, Any], 
                      error_message: str) -> FailureEvent:
        """Record a system failure."""
        event = FailureEvent(
            failure_type=failure_type,
            timestamp=time.time(),
            context=context.copy(),
            error_message=error_message
        )
        
        self.failure_history.append(event)
        logger.warning(f"System failure recorded: {failure_type.value} - {error_message}")
        
        return event
    
    def get_recovery_strategies(self, failure_event: FailureEvent) -> List[RecoveryAction]:
        """Get recommended recovery strategies for a failure."""
        strategies = []
        
        failure_type = failure_event.failure_type
        context = failure_event.context
        
        # Strategy selection based on failure type
        if failure_type == FailureType.QUANTUM_TIMEOUT:
            strategies.extend([
                RecoveryAction(
                    strategy=RecoveryStrategy.QUANTUM_FALLBACK,
                    priority=1,
                    estimated_success_rate=self._get_strategy_success_rate(
                        failure_type, RecoveryStrategy.QUANTUM_FALLBACK),
                    estimated_time=2.0,
                    description="Switch to classical quantum solver"
                ),
                RecoveryAction(
                    strategy=RecoveryStrategy.CLASSICAL_FALLBACK,
                    priority=2,
                    estimated_success_rate=self._get_strategy_success_rate(
                        failure_type, RecoveryStrategy.CLASSICAL_FALLBACK),
                    estimated_time=1.0,
                    description="Use classical optimization"
                ),
                RecoveryAction(
                    strategy=RecoveryStrategy.PROBLEM_DECOMPOSITION,
                    priority=3,
                    estimated_success_rate=self._get_strategy_success_rate(
                        failure_type, RecoveryStrategy.PROBLEM_DECOMPOSITION),
                    estimated_time=3.0,
                    description="Break problem into smaller parts"
                )
            ])
        
        elif failure_type == FailureType.QUANTUM_ERROR:
            strategies.extend([
                RecoveryAction(
                    strategy=RecoveryStrategy.EXPONENTIAL_BACKOFF,
                    priority=1,
                    estimated_success_rate=self._get_strategy_success_rate(
                        failure_type, RecoveryStrategy.EXPONENTIAL_BACKOFF),
                    estimated_time=5.0,
                    description="Retry with increasing delays"
                ),
                RecoveryAction(
                    strategy=RecoveryStrategy.CLASSICAL_FALLBACK,
                    priority=2,
                    estimated_success_rate=self._get_strategy_success_rate(
                        failure_type, RecoveryStrategy.CLASSICAL_FALLBACK),
                    estimated_time=1.0,
                    description="Switch to classical solver"
                )
            ])
        
        elif failure_type == FailureType.OPTIMIZATION_FAILED:
            strategies.extend([
                RecoveryAction(
                    strategy=RecoveryStrategy.CACHED_SOLUTION,
                    priority=1,
                    estimated_success_rate=self._get_cached_solution_rate(context),
                    estimated_time=0.1,
                    prerequisites=["similar_cache_available"],
                    description="Use cached similar solution"
                ),
                RecoveryAction(
                    strategy=RecoveryStrategy.PROBLEM_DECOMPOSITION,
                    priority=2,
                    estimated_success_rate=self._get_strategy_success_rate(
                        failure_type, RecoveryStrategy.PROBLEM_DECOMPOSITION),
                    estimated_time=2.0,
                    description="Decompose optimization problem"
                ),
                RecoveryAction(
                    strategy=RecoveryStrategy.GRACEFUL_DEGRADATION,
                    priority=3,
                    estimated_success_rate=0.9,
                    estimated_time=0.5,
                    description="Use simplified control strategy"
                )
            ])
        
        elif failure_type == FailureType.VALIDATION_ERROR:
            strategies.extend([
                RecoveryAction(
                    strategy=RecoveryStrategy.SAFE_MODE,
                    priority=1,
                    estimated_success_rate=0.95,
                    estimated_time=0.2,
                    description="Activate safe operating mode"
                ),
                RecoveryAction(
                    strategy=RecoveryStrategy.GRACEFUL_DEGRADATION,
                    priority=2,
                    estimated_success_rate=0.8,
                    estimated_time=0.5,
                    description="Use conservative controls"
                )
            ])
        
        else:  # Default strategies
            strategies.extend([
                RecoveryAction(
                    strategy=RecoveryStrategy.IMMEDIATE_RETRY,
                    priority=1,
                    estimated_success_rate=self._get_strategy_success_rate(
                        failure_type, RecoveryStrategy.IMMEDIATE_RETRY),
                    estimated_time=0.1,
                    description="Retry immediately once"
                ),
                RecoveryAction(
                    strategy=RecoveryStrategy.SAFE_MODE,
                    priority=2,
                    estimated_success_rate=0.9,
                    estimated_time=0.2,
                    description="Activate safe mode"
                )
            ])
        
        # Sort by priority and success rate
        strategies.sort(key=lambda s: (s.priority, -s.estimated_success_rate))
        
        return strategies
    
    def execute_recovery(self, failure_event: FailureEvent, 
                        recovery_action: RecoveryAction,
                        recovery_context: Dict[str, Any] = None) -> Tuple[bool, Any, str]:
        """
        Execute a recovery strategy.
        
        Returns:
            Tuple of (success, result, message)
        """
        start_time = time.time()
        
        try:
            if recovery_context is None:
                recovery_context = {}
            
            # Execute strategy
            success, result, message = self._execute_strategy(
                recovery_action.strategy, failure_event, recovery_context)
            
            # Record results
            recovery_time = time.time() - start_time
            failure_event.recovery_attempted = recovery_action.strategy
            failure_event.recovery_successful = success
            failure_event.recovery_time = recovery_time
            
            # Update strategy performance
            self._update_strategy_performance(
                failure_event.failure_type, recovery_action.strategy, 
                success, recovery_time)
            
            if success:
                logger.info(f"Recovery successful: {recovery_action.strategy.value} "
                          f"in {recovery_time:.2f}s - {message}")
            else:
                logger.warning(f"Recovery failed: {recovery_action.strategy.value} "
                             f"after {recovery_time:.2f}s - {message}")
            
            return success, result, message
            
        except Exception as e:
            recovery_time = time.time() - start_time
            error_msg = f"Recovery strategy execution failed: {str(e)}"
            logger.error(error_msg)
            
            failure_event.recovery_attempted = recovery_action.strategy
            failure_event.recovery_successful = False
            failure_event.recovery_time = recovery_time
            
            self._update_strategy_performance(
                failure_event.failure_type, recovery_action.strategy, 
                False, recovery_time)
            
            return False, None, error_msg
    
    def _execute_strategy(self, strategy: RecoveryStrategy, 
                         failure_event: FailureEvent,
                         context: Dict[str, Any]) -> Tuple[bool, Any, str]:
        """Execute specific recovery strategy."""
        
        if strategy == RecoveryStrategy.IMMEDIATE_RETRY:
            # Simply return success to trigger retry
            return True, None, "Immediate retry authorized"
        
        elif strategy == RecoveryStrategy.EXPONENTIAL_BACKOFF:
            # Calculate backoff time based on recent failures
            recent_failures = [f for f in self.failure_history 
                             if f.failure_type == failure_event.failure_type
                             and time.time() - f.timestamp < 300]  # 5 minutes
            
            backoff_time = min(2 ** len(recent_failures), 30)  # Max 30 seconds
            time.sleep(backoff_time)
            
            return True, backoff_time, f"Waited {backoff_time}s before retry"
        
        elif strategy == RecoveryStrategy.CLASSICAL_FALLBACK:
            # Signal to use classical solver
            return True, {"solver_type": "classical_fallback"}, "Classical fallback activated"
        
        elif strategy == RecoveryStrategy.QUANTUM_FALLBACK:
            # Signal to use hybrid quantum solver
            return True, {"solver_type": "hybrid"}, "Quantum hybrid fallback activated"
        
        elif strategy == RecoveryStrategy.CACHED_SOLUTION:
            # Try to find cached solution
            cached_key = self._generate_cache_key(failure_event.context)
            if cached_key in self.cached_solutions:
                solution = self.cached_solutions[cached_key]
                return True, solution, "Cached solution retrieved"
            else:
                return False, None, "No suitable cached solution found"
        
        elif strategy == RecoveryStrategy.PROBLEM_DECOMPOSITION:
            # Signal to decompose the problem
            return True, {"decompose": True, "max_subproblems": 4}, "Problem decomposition requested"
        
        elif strategy == RecoveryStrategy.SAFE_MODE:
            # Activate safe mode
            self.safe_mode_active = True
            safe_controls = self._generate_safe_controls(failure_event.context)
            return True, safe_controls, "Safe mode activated"
        
        elif strategy == RecoveryStrategy.GRACEFUL_DEGRADATION:
            # Use simplified control strategy
            simplified_controls = self._generate_simplified_controls(failure_event.context)
            return True, simplified_controls, "Graceful degradation activated"
        
        else:
            return False, None, f"Unknown recovery strategy: {strategy.value}"
    
    def _generate_safe_controls(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate safe control outputs."""
        # Very conservative controls - maintain current state
        current_temp = context.get('current_temperature', 22.0)
        
        return {
            'control_actions': [0.0] * context.get('num_zones', 1),  # No action
            'setpoints': [current_temp] * context.get('num_zones', 1),
            'mode': 'safe',
            'description': 'Safe mode - minimal intervention'
        }
    
    def _generate_simplified_controls(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate simplified control strategy."""
        # Simple thermostat-like control
        current_temp = context.get('current_temperature', 22.0)
        target_temp = context.get('target_temperature', 22.0)
        
        # Simple proportional control
        error = target_temp - current_temp
        control_action = np.clip(error * 0.1, -1.0, 1.0)  # Simple proportional
        
        return {
            'control_actions': [control_action] * context.get('num_zones', 1),
            'setpoints': [target_temp] * context.get('num_zones', 1),
            'mode': 'simplified',
            'description': 'Simplified proportional control'
        }
    
    def _generate_cache_key(self, context: Dict[str, Any]) -> str:
        """Generate cache key for similar problems."""
        # Simple cache key based on context
        key_parts = []
        
        for key in ['num_zones', 'horizon', 'outside_temp', 'occupancy']:
            if key in context:
                key_parts.append(f"{key}:{context[key]}")
        
        return "_".join(key_parts)
    
    def _get_strategy_success_rate(self, failure_type: FailureType, 
                                  strategy: RecoveryStrategy) -> float:
        """Get historical success rate for strategy."""
        if failure_type in self.recovery_success_rates:
            if strategy in self.recovery_success_rates[failure_type]:
                return self.recovery_success_rates[failure_type][strategy]
        
        # Default success rates based on strategy type
        defaults = {
            RecoveryStrategy.IMMEDIATE_RETRY: 0.3,
            RecoveryStrategy.EXPONENTIAL_BACKOFF: 0.6,
            RecoveryStrategy.CLASSICAL_FALLBACK: 0.9,
            RecoveryStrategy.QUANTUM_FALLBACK: 0.7,
            RecoveryStrategy.PROBLEM_DECOMPOSITION: 0.8,
            RecoveryStrategy.CACHED_SOLUTION: 0.95,
            RecoveryStrategy.SAFE_MODE: 0.99,
            RecoveryStrategy.GRACEFUL_DEGRADATION: 0.85
        }
        
        return defaults.get(strategy, 0.5)
    
    def _get_cached_solution_rate(self, context: Dict[str, Any]) -> float:
        """Check if cached solution is available."""
        cache_key = self._generate_cache_key(context)
        return 0.95 if cache_key in self.cached_solutions else 0.0
    
    def _update_strategy_performance(self, failure_type: FailureType,
                                   strategy: RecoveryStrategy,
                                   success: bool, recovery_time: float):
        """Update strategy performance metrics."""
        # Update success rates
        current_rate = self.recovery_success_rates[failure_type][strategy]
        new_rate = current_rate * 0.9 + (1.0 if success else 0.0) * 0.1
        self.recovery_success_rates[failure_type][strategy] = new_rate
        
        # Update performance history
        self.strategy_performance[strategy].append({
            'success': success,
            'time': recovery_time,
            'timestamp': time.time()
        })
    
    def cache_solution(self, context: Dict[str, Any], solution: Any):
        """Cache a successful solution."""
        cache_key = self._generate_cache_key(context)
        self.cached_solutions[cache_key] = {
            'solution': solution,
            'timestamp': time.time(),
            'context': context.copy()
        }
        
        # Limit cache size
        if len(self.cached_solutions) > 100:
            # Remove oldest entries
            oldest_key = min(self.cached_solutions.keys(),
                           key=lambda k: self.cached_solutions[k]['timestamp'])
            del self.cached_solutions[oldest_key]
    
    def get_recovery_analytics(self) -> Dict[str, Any]:
        """Get recovery system analytics."""
        analytics = {
            'total_failures': len(self.failure_history),
            'failure_types': {},
            'recovery_success_rates': {},
            'strategy_performance': {},
            'cache_hit_rate': 0.0,
            'safe_mode_active': self.safe_mode_active
        }
        
        # Failure type distribution
        for event in self.failure_history:
            failure_type = event.failure_type.value
            analytics['failure_types'][failure_type] = analytics['failure_types'].get(failure_type, 0) + 1
        
        # Recovery success rates
        for failure_type, strategies in self.recovery_success_rates.items():
            analytics['recovery_success_rates'][failure_type.value] = dict(strategies)
        
        # Strategy performance
        for strategy, performance in self.strategy_performance.items():
            if performance:
                analytics['strategy_performance'][strategy.value] = {
                    'total_attempts': len(performance),
                    'success_rate': sum(1 for p in performance if p['success']) / len(performance),
                    'avg_time': np.mean([p['time'] for p in performance])
                }
        
        return analytics
    
    def reset_safe_mode(self):
        """Reset safe mode."""
        self.safe_mode_active = False
        logger.info("Safe mode deactivated")


# Global recovery manager
_global_recovery_manager = None

def get_recovery_manager() -> IntelligentRecoveryManager:
    """Get global recovery manager instance."""
    global _global_recovery_manager
    if _global_recovery_manager is None:
        _global_recovery_manager = IntelligentRecoveryManager()
    return _global_recovery_manager


__all__ = [
    'IntelligentRecoveryManager',
    'RecoveryStrategy',
    'FailureType', 
    'RecoveryAction',
    'FailureEvent',
    'get_recovery_manager'
]
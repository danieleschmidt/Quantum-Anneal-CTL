"""
Resilient Error Recovery System for Quantum HVAC Control.

Advanced error recovery system with intelligent failure analysis,
adaptive recovery strategies, and quantum-aware healing mechanisms.

Features:
1. Intelligent error classification and context analysis
2. Multi-strategy recovery with adaptive selection
3. Quantum-specific recovery mechanisms
4. Self-healing system components
5. Predictive failure prevention
"""

from typing import Dict, Any, List, Optional, Callable, Union, Tuple
import asyncio
import time
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import traceback
import json
import pickle
import hashlib
import random
from abc import ABC, abstractmethod


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    CATASTROPHIC = "catastrophic"


class RecoveryStrategy(Enum):
    """Types of recovery strategies."""
    RETRY = "retry"
    FALLBACK = "fallback"
    DEGRADE = "degrade"
    ISOLATE = "isolate"
    RESTART = "restart"
    ESCALATE = "escalate"
    QUANTUM_REHEAL = "quantum_reheal"


class ErrorCategory(Enum):
    """Categories of errors for specialized handling."""
    QUANTUM_SOLVER = "quantum_solver"
    NETWORK_IO = "network_io"
    RESOURCE_LIMIT = "resource_limit"
    DATA_VALIDATION = "data_validation"
    THERMAL_CONTROL = "thermal_control"
    SAFETY_VIOLATION = "safety_violation"
    SYSTEM_INTEGRATION = "system_integration"
    UNKNOWN = "unknown"


@dataclass
class ErrorContext:
    """Comprehensive error context for analysis."""
    error: Exception
    timestamp: datetime
    function_name: str
    module_name: str
    traceback_str: str
    
    # Function context
    args: Tuple = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    
    # System context
    system_state: Dict[str, Any] = field(default_factory=dict)
    quantum_state: Dict[str, Any] = field(default_factory=dict)
    resource_usage: Dict[str, float] = field(default_factory=dict)
    
    # Recovery context
    previous_attempts: int = 0
    recovery_history: List[str] = field(default_factory=list)
    
    @property
    def error_hash(self) -> str:
        """Generate unique hash for error pattern matching."""
        error_signature = f"{type(self.error).__name__}:{self.function_name}:{str(self.error)}"
        return hashlib.md5(error_signature.encode()).hexdigest()[:16]


@dataclass
class RecoveryResult:
    """Result of a recovery attempt."""
    success: bool
    strategy_used: RecoveryStrategy
    execution_time: float
    error_resolved: bool
    fallback_used: bool
    result: Any = None
    new_error: Optional[Exception] = None
    recovery_notes: str = ""


class RecoveryStrategyBase(ABC):
    """Base class for recovery strategies."""
    
    def __init__(self, name: str):
        self.name = name
        self.success_count = 0
        self.failure_count = 0
        self.avg_execution_time = 0.0
        
    @abstractmethod
    async def execute_recovery(
        self,
        error_context: ErrorContext,
        original_function: Callable,
        *args,
        **kwargs
    ) -> RecoveryResult:
        """Execute recovery strategy."""
        pass
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate of this strategy."""
        total = self.success_count + self.failure_count
        return self.success_count / max(total, 1)
    
    def update_metrics(self, result: RecoveryResult) -> None:
        """Update strategy performance metrics."""
        if result.success:
            self.success_count += 1
        else:
            self.failure_count += 1
        
        # Update average execution time
        total_attempts = self.success_count + self.failure_count
        self.avg_execution_time = (
            (self.avg_execution_time * (total_attempts - 1) + result.execution_time) / total_attempts
        )


class RetryStrategy(RecoveryStrategyBase):
    """Retry strategy with exponential backoff."""
    
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 60.0):
        super().__init__("retry")
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
    
    async def execute_recovery(
        self,
        error_context: ErrorContext,
        original_function: Callable,
        *args,
        **kwargs
    ) -> RecoveryResult:
        """Execute retry with exponential backoff."""
        start_time = time.time()
        
        for attempt in range(min(self.max_retries, 5)):  # Cap retries for safety
            if attempt > 0:
                # Exponential backoff with jitter
                delay = min(self.base_delay * (2 ** attempt), self.max_delay)
                jitter = random.uniform(0.1, 0.3) * delay
                await asyncio.sleep(delay + jitter)
            
            try:
                result = await original_function(*args, **kwargs)
                execution_time = time.time() - start_time
                
                return RecoveryResult(
                    success=True,
                    strategy_used=RecoveryStrategy.RETRY,
                    execution_time=execution_time,
                    error_resolved=True,
                    fallback_used=False,
                    result=result,
                    recovery_notes=f"Succeeded on attempt {attempt + 1}"
                )
                
            except Exception as e:
                if attempt == self.max_retries - 1:
                    execution_time = time.time() - start_time
                    return RecoveryResult(
                        success=False,
                        strategy_used=RecoveryStrategy.RETRY,
                        execution_time=execution_time,
                        error_resolved=False,
                        fallback_used=False,
                        new_error=e,
                        recovery_notes=f"Failed after {attempt + 1} attempts"
                    )
        
        execution_time = time.time() - start_time
        return RecoveryResult(
            success=False,
            strategy_used=RecoveryStrategy.RETRY,
            execution_time=execution_time,
            error_resolved=False,
            fallback_used=False,
            recovery_notes="Max retries exceeded"
        )


class FallbackStrategy(RecoveryStrategyBase):
    """Fallback strategy using alternative implementations."""
    
    def __init__(self, fallback_functions: Dict[str, Callable]):
        super().__init__("fallback")
        self.fallback_functions = fallback_functions
    
    async def execute_recovery(
        self,
        error_context: ErrorContext,
        original_function: Callable,
        *args,
        **kwargs
    ) -> RecoveryResult:
        """Execute fallback strategy."""
        start_time = time.time()
        
        # Find appropriate fallback function
        function_name = error_context.function_name
        fallback_func = self.fallback_functions.get(function_name)
        
        if not fallback_func:
            execution_time = time.time() - start_time
            return RecoveryResult(
                success=False,
                strategy_used=RecoveryStrategy.FALLBACK,
                execution_time=execution_time,
                error_resolved=False,
                fallback_used=False,
                recovery_notes="No fallback function available"
            )
        
        try:
            result = await fallback_func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            return RecoveryResult(
                success=True,
                strategy_used=RecoveryStrategy.FALLBACK,
                execution_time=execution_time,
                error_resolved=True,
                fallback_used=True,
                result=result,
                recovery_notes=f"Used fallback function: {fallback_func.__name__}"
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return RecoveryResult(
                success=False,
                strategy_used=RecoveryStrategy.FALLBACK,
                execution_time=execution_time,
                error_resolved=False,
                fallback_used=True,
                new_error=e,
                recovery_notes=f"Fallback function {fallback_func.__name__} also failed"
            )


class QuantumRehealStrategy(RecoveryStrategyBase):
    """Quantum-specific recovery strategy."""
    
    def __init__(self):
        super().__init__("quantum_reheal")
    
    async def execute_recovery(
        self,
        error_context: ErrorContext,
        original_function: Callable,
        *args,
        **kwargs
    ) -> RecoveryResult:
        """Execute quantum-specific recovery."""
        start_time = time.time()
        
        # Check if this is a quantum-related error
        if not self._is_quantum_error(error_context):
            execution_time = time.time() - start_time
            return RecoveryResult(
                success=False,
                strategy_used=RecoveryStrategy.QUANTUM_REHEAL,
                execution_time=execution_time,
                error_resolved=False,
                fallback_used=False,
                recovery_notes="Not a quantum error"
            )
        
        # Apply quantum-specific healing
        try:
            # Reset quantum state
            await self._reset_quantum_state(error_context)
            
            # Adjust quantum parameters
            await self._adjust_quantum_parameters(error_context, kwargs)
            
            # Retry with adjusted parameters
            result = await original_function(*args, **kwargs)
            execution_time = time.time() - start_time
            
            return RecoveryResult(
                success=True,
                strategy_used=RecoveryStrategy.QUANTUM_REHEAL,
                execution_time=execution_time,
                error_resolved=True,
                fallback_used=False,
                result=result,
                recovery_notes="Quantum state reset and parameters adjusted"
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return RecoveryResult(
                success=False,
                strategy_used=RecoveryStrategy.QUANTUM_REHEAL,
                execution_time=execution_time,
                error_resolved=False,
                fallback_used=False,
                new_error=e,
                recovery_notes="Quantum healing failed"
            )
    
    def _is_quantum_error(self, error_context: ErrorContext) -> bool:
        """Check if error is quantum-related."""
        error_str = str(error_context.error).lower()
        quantum_keywords = ['chain', 'embedding', 'qubo', 'annealing', 'quantum', 'dwave']
        return any(keyword in error_str for keyword in quantum_keywords)
    
    async def _reset_quantum_state(self, error_context: ErrorContext) -> None:
        """Reset quantum solver state."""
        # In practice, would reset quantum solver instances
        await asyncio.sleep(0.1)  # Simulate reset time
    
    async def _adjust_quantum_parameters(self, error_context: ErrorContext, kwargs: Dict[str, Any]) -> None:
        """Adjust quantum parameters based on error."""
        # Chain break errors - increase chain strength
        if 'chain' in str(error_context.error).lower():
            if 'chain_strength' in kwargs:
                kwargs['chain_strength'] *= 1.5
        
        # Embedding errors - reduce problem size or adjust embedding
        if 'embedding' in str(error_context.error).lower():
            if 'num_reads' in kwargs:
                kwargs['num_reads'] = max(100, kwargs['num_reads'] // 2)


class DegradeStrategy(RecoveryStrategyBase):
    """Graceful degradation strategy."""
    
    def __init__(self, degraded_functions: Dict[str, Callable]):
        super().__init__("degrade")
        self.degraded_functions = degraded_functions
    
    async def execute_recovery(
        self,
        error_context: ErrorContext,
        original_function: Callable,
        *args,
        **kwargs
    ) -> RecoveryResult:
        """Execute graceful degradation."""
        start_time = time.time()
        
        function_name = error_context.function_name
        degraded_func = self.degraded_functions.get(function_name)
        
        if not degraded_func:
            execution_time = time.time() - start_time
            return RecoveryResult(
                success=False,
                strategy_used=RecoveryStrategy.DEGRADE,
                execution_time=execution_time,
                error_resolved=False,
                fallback_used=False,
                recovery_notes="No degraded function available"
            )
        
        try:
            # Execute with reduced functionality
            result = await degraded_func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            return RecoveryResult(
                success=True,
                strategy_used=RecoveryStrategy.DEGRADE,
                execution_time=execution_time,
                error_resolved=True,
                fallback_used=True,
                result=result,
                recovery_notes=f"Degraded to simplified function: {degraded_func.__name__}"
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return RecoveryResult(
                success=False,
                strategy_used=RecoveryStrategy.DEGRADE,
                execution_time=execution_time,
                error_resolved=False,
                fallback_used=True,
                new_error=e,
                recovery_notes=f"Degraded function {degraded_func.__name__} also failed"
            )


class ErrorAnalyzer:
    """Intelligent error analysis and classification."""
    
    def __init__(self):
        self.error_patterns: Dict[str, List[ErrorContext]] = defaultdict(list)
        self.recovery_effectiveness: Dict[str, Dict[RecoveryStrategy, float]] = defaultdict(dict)
        self.logger = logging.getLogger(__name__)
    
    def analyze_error(self, error_context: ErrorContext) -> Tuple[ErrorSeverity, ErrorCategory, List[RecoveryStrategy]]:
        """Analyze error and recommend recovery strategies."""
        severity = self._classify_severity(error_context)
        category = self._classify_category(error_context)
        strategies = self._recommend_strategies(error_context, severity, category)
        
        # Store error pattern for learning
        self.error_patterns[error_context.error_hash].append(error_context)
        
        return severity, category, strategies
    
    def _classify_severity(self, error_context: ErrorContext) -> ErrorSeverity:
        """Classify error severity."""
        error_str = str(error_context.error).lower()
        
        # Catastrophic errors
        if any(keyword in error_str for keyword in ['safety', 'thermal_runaway', 'system_failure']):
            return ErrorSeverity.CATASTROPHIC
        
        # Critical errors
        if any(keyword in error_str for keyword in ['critical', 'emergency', 'violation']):
            return ErrorSeverity.CRITICAL
        
        # High severity errors
        if any(keyword in error_str for keyword in ['timeout', 'connection', 'resource']):
            return ErrorSeverity.HIGH
        
        # Medium severity errors
        if any(keyword in error_str for keyword in ['quantum', 'solver', 'embedding']):
            return ErrorSeverity.MEDIUM
        
        # Default to low severity
        return ErrorSeverity.LOW
    
    def _classify_category(self, error_context: ErrorContext) -> ErrorCategory:
        """Classify error category."""
        error_str = str(error_context.error).lower()
        module_name = error_context.module_name.lower()
        
        if any(keyword in error_str or keyword in module_name 
               for keyword in ['quantum', 'qubo', 'annealing', 'dwave']):
            return ErrorCategory.QUANTUM_SOLVER
        
        if any(keyword in error_str for keyword in ['network', 'connection', 'timeout', 'io']):
            return ErrorCategory.NETWORK_IO
        
        if any(keyword in error_str for keyword in ['memory', 'resource', 'limit', 'capacity']):
            return ErrorCategory.RESOURCE_LIMIT
        
        if any(keyword in error_str for keyword in ['validation', 'invalid', 'format']):
            return ErrorCategory.DATA_VALIDATION
        
        if any(keyword in error_str for keyword in ['temperature', 'thermal', 'hvac']):
            return ErrorCategory.THERMAL_CONTROL
        
        if any(keyword in error_str for keyword in ['safety', 'violation', 'emergency']):
            return ErrorCategory.SAFETY_VIOLATION
        
        if any(keyword in module_name for keyword in ['integration', 'api', 'bms']):
            return ErrorCategory.SYSTEM_INTEGRATION
        
        return ErrorCategory.UNKNOWN
    
    def _recommend_strategies(
        self, 
        error_context: ErrorContext, 
        severity: ErrorSeverity, 
        category: ErrorCategory
    ) -> List[RecoveryStrategy]:
        """Recommend recovery strategies based on error analysis."""
        strategies = []
        
        # Strategy selection based on severity
        if severity == ErrorSeverity.CATASTROPHIC:
            strategies = [RecoveryStrategy.ISOLATE, RecoveryStrategy.ESCALATE]
        elif severity == ErrorSeverity.CRITICAL:
            strategies = [RecoveryStrategy.FALLBACK, RecoveryStrategy.DEGRADE, RecoveryStrategy.ESCALATE]
        elif severity == ErrorSeverity.HIGH:
            strategies = [RecoveryStrategy.RETRY, RecoveryStrategy.FALLBACK, RecoveryStrategy.DEGRADE]
        else:
            strategies = [RecoveryStrategy.RETRY, RecoveryStrategy.FALLBACK]
        
        # Add category-specific strategies
        if category == ErrorCategory.QUANTUM_SOLVER:
            strategies.insert(0, RecoveryStrategy.QUANTUM_REHEAL)
        
        # Use historical effectiveness
        error_hash = error_context.error_hash
        if error_hash in self.recovery_effectiveness:
            effectiveness = self.recovery_effectiveness[error_hash]
            strategies.sort(key=lambda s: effectiveness.get(s, 0.0), reverse=True)
        
        return strategies[:3]  # Return top 3 strategies
    
    def record_recovery_result(self, error_context: ErrorContext, result: RecoveryResult) -> None:
        """Record recovery result for learning."""
        error_hash = error_context.error_hash
        
        if error_hash not in self.recovery_effectiveness:
            self.recovery_effectiveness[error_hash] = {}
        
        # Update effectiveness score
        current_score = self.recovery_effectiveness[error_hash].get(result.strategy_used, 0.0)
        new_score = 1.0 if result.success else 0.0
        
        # Exponential moving average
        alpha = 0.3
        self.recovery_effectiveness[error_hash][result.strategy_used] = (
            alpha * new_score + (1 - alpha) * current_score
        )


class ResilientErrorRecoverySystem:
    """
    Comprehensive error recovery system with intelligent analysis
    and adaptive recovery strategies.
    """
    
    def __init__(self):
        self.strategies: Dict[RecoveryStrategy, RecoveryStrategyBase] = {}
        self.analyzer = ErrorAnalyzer()
        self.recovery_history: deque = deque(maxlen=1000)
        self.active_recoveries: Dict[str, int] = defaultdict(int)
        self.max_concurrent_recoveries = 10
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize default strategies
        self._initialize_default_strategies()
        
        # Performance tracking
        self.total_recoveries = 0
        self.successful_recoveries = 0
        self.recovery_times: deque = deque(maxlen=100)
    
    def _initialize_default_strategies(self) -> None:
        """Initialize default recovery strategies."""
        self.strategies[RecoveryStrategy.RETRY] = RetryStrategy()
        self.strategies[RecoveryStrategy.FALLBACK] = FallbackStrategy({})
        self.strategies[RecoveryStrategy.QUANTUM_REHEAL] = QuantumRehealStrategy()
        self.strategies[RecoveryStrategy.DEGRADE] = DegradeStrategy({})
    
    def register_fallback_function(self, function_name: str, fallback_func: Callable) -> None:
        """Register fallback function for specific function."""
        if RecoveryStrategy.FALLBACK in self.strategies:
            fallback_strategy = self.strategies[RecoveryStrategy.FALLBACK]
            fallback_strategy.fallback_functions[function_name] = fallback_func
    
    def register_degraded_function(self, function_name: str, degraded_func: Callable) -> None:
        """Register degraded function for specific function."""
        if RecoveryStrategy.DEGRADE not in self.strategies:
            self.strategies[RecoveryStrategy.DEGRADE] = DegradeStrategy({})
        
        degrade_strategy = self.strategies[RecoveryStrategy.DEGRADE]
        degrade_strategy.degraded_functions[function_name] = degraded_func
    
    async def recover_from_error(
        self,
        error: Exception,
        original_function: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Attempt to recover from error using intelligent strategies."""
        # Create error context
        error_context = ErrorContext(
            error=error,
            timestamp=datetime.now(),
            function_name=original_function.__name__,
            module_name=original_function.__module__,
            traceback_str=traceback.format_exc(),
            args=args,
            kwargs=kwargs,
            previous_attempts=self.active_recoveries[original_function.__name__]
        )
        
        # Check concurrent recovery limit
        if self.active_recoveries[original_function.__name__] >= 3:
            self.logger.error(f"Too many concurrent recoveries for {original_function.__name__}")
            raise error
        
        self.active_recoveries[original_function.__name__] += 1
        
        try:
            # Analyze error and get recommended strategies
            severity, category, recommended_strategies = self.analyzer.analyze_error(error_context)
            
            self.logger.warning(
                f"Error recovery initiated for {original_function.__name__}: "
                f"{severity.value} {category.value} - {str(error)}"
            )
            
            # Attempt recovery with each strategy
            for strategy_type in recommended_strategies:
                if strategy_type not in self.strategies:
                    continue
                
                strategy = self.strategies[strategy_type]
                
                try:
                    result = await strategy.execute_recovery(
                        error_context, original_function, *args, **kwargs
                    )
                    
                    # Update strategy metrics
                    strategy.update_metrics(result)
                    
                    # Record result for learning
                    self.analyzer.record_recovery_result(error_context, result)
                    
                    # Update system metrics
                    self.total_recoveries += 1
                    self.recovery_times.append(result.execution_time)
                    
                    if result.success:
                        self.successful_recoveries += 1
                        
                        self.logger.info(
                            f"Recovery successful using {strategy_type.value} "
                            f"for {original_function.__name__} in {result.execution_time:.2f}s"
                        )
                        
                        # Store successful recovery
                        self.recovery_history.append({
                            'timestamp': datetime.now(),
                            'function': original_function.__name__,
                            'error': str(error),
                            'strategy': strategy_type.value,
                            'success': True,
                            'execution_time': result.execution_time
                        })
                        
                        return result.result
                    
                    else:
                        self.logger.warning(
                            f"Recovery strategy {strategy_type.value} failed "
                            f"for {original_function.__name__}: {result.recovery_notes}"
                        )
                
                except Exception as strategy_error:
                    self.logger.error(
                        f"Recovery strategy {strategy_type.value} threw exception: {strategy_error}"
                    )
                    continue
            
            # All recovery strategies failed
            self.logger.error(f"All recovery strategies failed for {original_function.__name__}")
            
            # Store failed recovery
            self.recovery_history.append({
                'timestamp': datetime.now(),
                'function': original_function.__name__,
                'error': str(error),
                'strategy': 'all_failed',
                'success': False,
                'execution_time': 0.0
            })
            
            # Re-raise original error
            raise error
        
        finally:
            self.active_recoveries[original_function.__name__] -= 1
    
    def get_recovery_statistics(self) -> Dict[str, Any]:
        """Get comprehensive recovery statistics."""
        if not self.recovery_times:
            avg_recovery_time = 0.0
        else:
            avg_recovery_time = sum(self.recovery_times) / len(self.recovery_times)
        
        success_rate = self.successful_recoveries / max(self.total_recoveries, 1)
        
        # Strategy effectiveness
        strategy_stats = {}
        for strategy_type, strategy in self.strategies.items():
            strategy_stats[strategy_type.value] = {
                'success_rate': strategy.success_rate,
                'total_attempts': strategy.success_count + strategy.failure_count,
                'avg_execution_time': strategy.avg_execution_time
            }
        
        # Recent recovery trends
        recent_recoveries = list(self.recovery_history)[-50:]  # Last 50 recoveries
        recent_success_rate = sum(1 for r in recent_recoveries if r['success']) / max(len(recent_recoveries), 1)
        
        return {
            'overall_stats': {
                'total_recoveries': self.total_recoveries,
                'successful_recoveries': self.successful_recoveries,
                'success_rate': success_rate,
                'avg_recovery_time': avg_recovery_time,
                'recent_success_rate': recent_success_rate
            },
            'strategy_effectiveness': strategy_stats,
            'active_recoveries': dict(self.active_recoveries),
            'error_patterns': len(self.analyzer.error_patterns),
            'learned_effectiveness_patterns': len(self.analyzer.recovery_effectiveness)
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform system health check."""
        health_status = {
            'recovery_system_healthy': True,
            'issues': [],
            'recommendations': []
        }
        
        # Check recovery success rate
        if self.total_recoveries > 10:
            success_rate = self.successful_recoveries / self.total_recoveries
            if success_rate < 0.7:
                health_status['recovery_system_healthy'] = False
                health_status['issues'].append(f"Low recovery success rate: {success_rate:.2f}")
                health_status['recommendations'].append("Review and improve recovery strategies")
        
        # Check for high concurrent recoveries
        max_concurrent = max(self.active_recoveries.values()) if self.active_recoveries else 0
        if max_concurrent >= 3:
            health_status['issues'].append(f"High concurrent recoveries: {max_concurrent}")
            health_status['recommendations'].append("Investigate root cause of repeated failures")
        
        # Check strategy effectiveness
        for strategy_type, strategy in self.strategies.items():
            if strategy.success_count + strategy.failure_count > 5:
                if strategy.success_rate < 0.5:
                    health_status['issues'].append(f"Poor {strategy_type.value} strategy performance")
                    health_status['recommendations'].append(f"Tune {strategy_type.value} strategy parameters")
        
        return health_status


def resilient_recovery(
    fallback_function: Optional[Callable] = None,
    degraded_function: Optional[Callable] = None,
    max_retries: int = 3
):
    """
    Decorator for automatic error recovery.
    
    Args:
        fallback_function: Alternative function to use if original fails
        degraded_function: Simplified function for graceful degradation
        max_retries: Maximum number of retry attempts
    """
    def decorator(func: Callable) -> Callable:
        async def wrapper(*args, **kwargs):
            recovery_system = get_recovery_system()
            
            # Register functions if provided
            if fallback_function:
                recovery_system.register_fallback_function(func.__name__, fallback_function)
            if degraded_function:
                recovery_system.register_degraded_function(func.__name__, degraded_function)
            
            # Update retry strategy
            if RecoveryStrategy.RETRY in recovery_system.strategies:
                retry_strategy = recovery_system.strategies[RecoveryStrategy.RETRY]
                retry_strategy.max_retries = max_retries
            
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                # Attempt recovery
                return await recovery_system.recover_from_error(e, func, *args, **kwargs)
        
        return wrapper
    return decorator


# Global recovery system
_recovery_system: Optional[ResilientErrorRecoverySystem] = None


def get_recovery_system() -> ResilientErrorRecoverySystem:
    """Get global recovery system instance."""
    global _recovery_system
    if _recovery_system is None:
        _recovery_system = ResilientErrorRecoverySystem()
    return _recovery_system
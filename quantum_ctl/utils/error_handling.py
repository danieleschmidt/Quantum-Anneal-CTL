"""
Comprehensive error handling and recovery mechanisms.
"""

import logging
import traceback
import functools
import asyncio
import numpy as np
from typing import Dict, Any, Optional, Callable, Type, Union
from enum import Enum
import time


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for classification."""
    QUANTUM_SOLVER = "quantum_solver"
    BUILDING_MODEL = "building_model"
    OPTIMIZATION = "optimization"
    COMMUNICATION = "communication"
    VALIDATION = "validation"
    SYSTEM = "system"
    UNKNOWN = "unknown"


class QuantumControlError(Exception):
    """Base exception for quantum control system."""
    
    def __init__(self, message: str, category: ErrorCategory = ErrorCategory.UNKNOWN, 
                 severity: ErrorSeverity = ErrorSeverity.MEDIUM, context: Dict[str, Any] = None):
        super().__init__(message)
        self.message = message
        self.category = category
        self.severity = severity
        self.context = context or {}
        self.timestamp = time.time()


class QuantumSolverError(QuantumControlError):
    """Quantum solver related errors."""
    
    def __init__(self, message: str, solver_type: str = None, **kwargs):
        super().__init__(
            message, 
            category=ErrorCategory.QUANTUM_SOLVER,
            severity=ErrorSeverity.HIGH,
            **kwargs
        )
        self.solver_type = solver_type


class OptimizationError(QuantumControlError):
    """Optimization process errors."""
    
    def __init__(self, message: str, problem_size: int = None, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.OPTIMIZATION, 
            severity=ErrorSeverity.MEDIUM,
            **kwargs
        )
        self.problem_size = problem_size


class ValidationError(QuantumControlError):
    """Data validation errors."""
    
    def __init__(self, message: str, field_name: str = None, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.LOW,
            **kwargs
        )
        self.field_name = field_name


class CommunicationError(QuantumControlError):
    """Communication/network related errors."""
    
    def __init__(self, message: str, endpoint: str = None, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.COMMUNICATION,
            severity=ErrorSeverity.MEDIUM,
            **kwargs
        )
        self.endpoint = endpoint


class ErrorHandler:
    """Centralized error handling and recovery."""
    
    def __init__(self, logger_name: str = __name__):
        self.logger = logging.getLogger(logger_name)
        self.error_counts: Dict[ErrorCategory, int] = {}
        self.recovery_strategies: Dict[ErrorCategory, Callable] = {}
        
        # Register default recovery strategies
        self._register_default_strategies()
    
    def _register_default_strategies(self):
        """Register default error recovery strategies."""
        self.recovery_strategies[ErrorCategory.QUANTUM_SOLVER] = self._recover_quantum_solver
        self.recovery_strategies[ErrorCategory.OPTIMIZATION] = self._recover_optimization
        self.recovery_strategies[ErrorCategory.COMMUNICATION] = self._recover_communication
        
    def register_recovery_strategy(self, category: ErrorCategory, strategy: Callable):
        """Register custom recovery strategy for error category."""
        self.recovery_strategies[category] = strategy
        self.logger.info(f"Registered recovery strategy for {category.value}")
    
    def handle_error(self, error: Exception, context: Dict[str, Any] = None) -> Optional[Any]:
        """Handle error with appropriate recovery strategy."""
        # Convert to QuantumControlError if needed
        if not isinstance(error, QuantumControlError):
            error = QuantumControlError(
                str(error),
                category=self._classify_error(error),
                context=context
            )
        
        # Log error
        self._log_error(error)
        
        # Update error counts
        self.error_counts[error.category] = self.error_counts.get(error.category, 0) + 1
        
        # Attempt recovery
        if error.category in self.recovery_strategies:
            try:
                recovery_result = self.recovery_strategies[error.category](error, context)
                self.logger.info(f"Successfully recovered from {error.category.value} error")
                return recovery_result
            except Exception as recovery_error:
                self.logger.error(f"Recovery failed for {error.category.value}: {recovery_error}")
        
        # Re-raise if no recovery possible
        raise error
    
    def _classify_error(self, error: Exception) -> ErrorCategory:
        """Classify error by type and message."""
        error_str = str(error).lower()
        error_type = type(error).__name__.lower()
        
        if 'quantum' in error_str or 'dwave' in error_str or 'qubo' in error_str:
            return ErrorCategory.QUANTUM_SOLVER
        elif 'optimization' in error_str or 'solver' in error_str:
            return ErrorCategory.OPTIMIZATION
        elif 'building' in error_str or 'thermal' in error_str or 'zone' in error_str:
            return ErrorCategory.BUILDING_MODEL
        elif 'connection' in error_str or 'network' in error_str or 'timeout' in error_str:
            return ErrorCategory.COMMUNICATION
        elif 'validation' in error_str or 'invalid' in error_str:
            return ErrorCategory.VALIDATION
        elif 'system' in error_str or 'memory' in error_str or 'cpu' in error_str:
            return ErrorCategory.SYSTEM
        else:
            return ErrorCategory.UNKNOWN
    
    def _log_error(self, error: QuantumControlError):
        """Log error with appropriate level."""
        log_level = {
            ErrorSeverity.LOW: logging.WARNING,
            ErrorSeverity.MEDIUM: logging.ERROR,
            ErrorSeverity.HIGH: logging.ERROR,
            ErrorSeverity.CRITICAL: logging.CRITICAL
        }.get(error.severity, logging.ERROR)
        
        self.logger.log(
            log_level,
            f"{error.category.value.upper()} ERROR [{error.severity.value}]: {error.message}",
            extra={
                'category': error.category.value,
                'severity': error.severity.value,
                'context': error.context,
                'timestamp': error.timestamp
            }
        )
    
    def _recover_quantum_solver(self, error: QuantumControlError, context: Dict[str, Any]) -> Any:
        """Recovery strategy for quantum solver errors."""
        self.logger.info("Attempting quantum solver recovery...")
        
        # Strategy 1: Fallback to classical solver
        if 'controller' in context:
            controller = context['controller']
            if hasattr(controller, 'quantum_solver'):
                # Switch to classical fallback
                controller.quantum_solver.solver_type = "classical_fallback"
                self.logger.info("Switched to classical fallback solver")
                return "classical_fallback"
        
        # Strategy 2: Reduce problem size
        if 'problem_size' in context:
            reduced_size = min(context['problem_size'] // 2, 100)
            self.logger.info(f"Reducing problem size to {reduced_size}")
            return {'reduced_problem_size': reduced_size}
        
        raise error
    
    def _recover_optimization(self, error: QuantumControlError, context: Dict[str, Any]) -> Any:
        """Recovery strategy for optimization errors."""
        self.logger.info("Attempting optimization recovery...")
        
        # Strategy 1: Use last known good solution
        if 'controller' in context:
            controller = context['controller']
            if hasattr(controller, '_control_history') and controller._control_history:
                last_control = controller._control_history[-1]
                self.logger.info("Using last known good control schedule")
                return last_control
        
        # Strategy 2: Generate safe default control
        if 'building' in context:
            building = context['building']
            n_controls = building.get_control_dimension()
            default_control = np.full(n_controls * 4, 0.5)  # Safe middle values
            self.logger.info("Generated safe default control schedule")
            return default_control
        
        raise error
    
    def _recover_communication(self, error: QuantumControlError, context: Dict[str, Any]) -> Any:
        """Recovery strategy for communication errors."""
        self.logger.info("Attempting communication recovery...")
        
        # Strategy 1: Retry with exponential backoff
        if 'retry_func' in context:
            import time
            for attempt in range(3):
                try:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    return context['retry_func']()
                except Exception:
                    continue
        
        # Strategy 2: Use cached data
        if 'cache' in context and context['cache']:
            self.logger.info("Using cached data due to communication failure")
            return context['cache']
        
        raise error
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics and trends."""
        total_errors = sum(self.error_counts.values())
        
        return {
            'total_errors': total_errors,
            'by_category': dict(self.error_counts),
            'most_common': max(self.error_counts.items(), key=lambda x: x[1]) if self.error_counts else None,
            'recovery_strategies': list(self.recovery_strategies.keys())
        }


def error_handler(category: ErrorCategory = None, severity: ErrorSeverity = ErrorSeverity.MEDIUM):
    """Decorator for automatic error handling."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if not isinstance(e, QuantumControlError):
                    e = QuantumControlError(
                        str(e),
                        category=category or ErrorCategory.UNKNOWN,
                        severity=severity
                    )
                
                # Get error handler from first arg if it's a class instance
                error_handler_instance = None
                if args and hasattr(args[0], '_error_handler'):
                    error_handler_instance = args[0]._error_handler
                else:
                    error_handler_instance = ErrorHandler()
                
                return error_handler_instance.handle_error(e, {'function': func.__name__})
        
        return wrapper
    return decorator


def async_error_handler(category: ErrorCategory = None, severity: ErrorSeverity = ErrorSeverity.MEDIUM):
    """Decorator for automatic async error handling."""
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                if not isinstance(e, QuantumControlError):
                    e = QuantumControlError(
                        str(e),
                        category=category or ErrorCategory.UNKNOWN,
                        severity=severity
                    )
                
                # Get error handler from first arg if it's a class instance
                error_handler_instance = None
                if args and hasattr(args[0], '_error_handler'):
                    error_handler_instance = args[0]._error_handler
                else:
                    error_handler_instance = ErrorHandler()
                
                return error_handler_instance.handle_error(e, {'function': func.__name__})
        
        return wrapper
    return decorator


class ErrorReporter:
    """Error reporting and alerting system."""
    
    def __init__(self, alert_webhook: str = None):
        self.alert_webhook = alert_webhook
        self.logger = logging.getLogger(__name__)
        
    def report_error(self, error: QuantumControlError, additional_context: Dict[str, Any] = None):
        """Report error to external systems."""
        error_report = {
            'timestamp': error.timestamp,
            'category': error.category.value,
            'severity': error.severity.value,
            'message': error.message,
            'context': error.context,
            'traceback': traceback.format_exc(),
            'additional_context': additional_context or {}
        }
        
        # Log locally
        self.logger.error(f"Error Report: {error_report}")
        
        # Send to webhook if configured
        if self.alert_webhook and error.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
            self._send_webhook_alert(error_report)
    
    def _send_webhook_alert(self, error_report: Dict[str, Any]):
        """Send error alert to webhook."""
        try:
            import requests
            import json
            
            payload = {
                'text': f"🚨 Quantum Control Error: {error_report['category']} [{error_report['severity']}]",
                'attachments': [{
                    'color': 'danger',
                    'fields': [
                        {'title': 'Category', 'value': error_report['category'], 'short': True},
                        {'title': 'Severity', 'value': error_report['severity'], 'short': True},
                        {'title': 'Message', 'value': error_report['message'], 'short': False}
                    ]
                }]
            }
            
            response = requests.post(
                self.alert_webhook,
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                self.logger.info("Error alert sent successfully")
            else:
                self.logger.warning(f"Failed to send alert: {response.status_code}")
                
        except Exception as e:
            self.logger.error(f"Failed to send webhook alert: {e}")


# Circuit breaker for critical systems
class CircuitBreaker:
    """Circuit breaker pattern for preventing cascading failures."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'closed'  # closed, open, half-open
        self.logger = logging.getLogger(__name__)
    
    def call(self, func: Callable, *args, **kwargs):
        """Call function through circuit breaker."""
        if self.state == 'open':
            if self._should_attempt_reset():
                self.state = 'half-open'
                self.logger.info("Circuit breaker attempting reset")
            else:
                raise QuantumControlError(
                    "Circuit breaker is open - function calls blocked",
                    category=ErrorCategory.SYSTEM,
                    severity=ErrorSeverity.HIGH
                )
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset."""
        if self.last_failure_time is None:
            return True
        return time.time() - self.last_failure_time > self.recovery_timeout
    
    def _on_success(self):
        """Handle successful function call."""
        if self.state == 'half-open':
            self.state = 'closed'
            self.failure_count = 0
            self.logger.info("Circuit breaker reset to closed state")
    
    def _on_failure(self):
        """Handle failed function call."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = 'open'
            self.logger.warning(f"Circuit breaker opened after {self.failure_count} failures")


class AdvancedErrorHandler(ErrorHandler):
    """Enhanced error handler with monitoring integration."""
    
    def __init__(self, logger_name: str = __name__, monitor=None):
        super().__init__(logger_name)
        self.monitor = monitor
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        
        # Enhanced recovery strategies
        self._register_advanced_strategies()
    
    def _register_advanced_strategies(self):
        """Register advanced error recovery strategies."""
        self.recovery_strategies[ErrorCategory.SYSTEM] = self._recover_system_error
        self.recovery_strategies[ErrorCategory.BUILDING_MODEL] = self._recover_building_model
    
    def handle_error(self, error: Exception, context: Dict[str, Any] = None) -> Optional[Any]:
        """Enhanced error handling with monitoring integration."""
        # Convert to QuantumControlError if needed
        if not isinstance(error, QuantumControlError):
            error = QuantumControlError(
                str(error),
                category=self._classify_error(error),
                context=context
            )
        
        # Record error in monitoring system
        if self.monitor:
            self.monitor.record_error(
                error.category.value,
                error.severity.value,
                error
            )
        
        # Call parent handler
        return super().handle_error(error, context)
    
    def get_circuit_breaker(self, name: str) -> CircuitBreaker:
        """Get or create circuit breaker."""
        if name not in self.circuit_breakers:
            self.circuit_breakers[name] = CircuitBreaker()
        return self.circuit_breakers[name]
    
    def _recover_system_error(self, error: QuantumControlError, context: Dict[str, Any]) -> Any:
        """Recovery strategy for system errors."""
        self.logger.info("Attempting system error recovery...")
        
        # Strategy 1: Clear caches to free memory
        if 'memory' in str(error).lower():
            if 'controller' in context:
                controller = context['controller']
                if hasattr(controller, 'clear_caches'):
                    controller.clear_caches()
                    self.logger.info("Cleared system caches")
                    return "caches_cleared"
        
        # Strategy 2: Reduce system load
        if 'cpu' in str(error).lower() or 'load' in str(error).lower():
            self.logger.info("Reducing system load")
            return {'reduce_load': True, 'priority': 'low'}
        
        raise error
    
    def _recover_building_model(self, error: QuantumControlError, context: Dict[str, Any]) -> Any:
        """Recovery strategy for building model errors."""
        self.logger.info("Attempting building model recovery...")
        
        # Strategy 1: Use simplified model
        if 'building' in context:
            building = context['building']
            if hasattr(building, 'use_simplified_model'):
                building.use_simplified_model(True)
                self.logger.info("Switched to simplified building model")
                return "simplified_model"
        
        # Strategy 2: Reset model parameters
        if 'validation' in str(error).lower():
            self.logger.info("Resetting building model to defaults")
            return {'reset_model': True, 'use_defaults': True}
        
        raise error
    
    def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive error handling status."""
        status = super().get_error_statistics()
        
        # Add circuit breaker status
        circuit_status = {}
        for name, breaker in self.circuit_breakers.items():
            circuit_status[name] = {
                'state': breaker.state,
                'failure_count': breaker.failure_count,
                'last_failure': breaker.last_failure_time
            }
        
        status['circuit_breakers'] = circuit_status
        status['monitoring_integration'] = self.monitor is not None
        
        return status


# Enhanced retry decorator with exponential backoff
def retry_with_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """Decorator for retrying functions with exponential backoff."""
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_retries:
                        break
                    
                    await asyncio.sleep(delay)
                    delay = min(delay * backoff_factor, max_delay)
            
            raise last_exception
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_retries:
                        break
                    
                    time.sleep(delay)
                    delay = min(delay * backoff_factor, max_delay)
            
            raise last_exception
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# Global error handler instance
global_error_handler = AdvancedErrorHandler()
global_error_reporter = ErrorReporter()
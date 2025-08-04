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


# Global error handler instance
global_error_handler = ErrorHandler()
global_error_reporter = ErrorReporter()
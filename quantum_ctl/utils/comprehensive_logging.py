"""
Comprehensive logging system for quantum HVAC control operations.
Provides structured logging, performance tracking, security auditing, and compliance logging.
"""

import logging
import logging.handlers
import json
import time
import asyncio
import threading
import traceback
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
import os

# Try to import optional dependencies
try:
    import structlog
    STRUCTLOG_AVAILABLE = True
except ImportError:
    STRUCTLOG_AVAILABLE = False


class LogLevel(Enum):
    """Enhanced log levels for quantum HVAC operations."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    SECURITY = "SECURITY"
    PERFORMANCE = "PERFORMANCE"
    AUDIT = "AUDIT"
    QUANTUM = "QUANTUM"


class LogCategory(Enum):
    """Categories for different types of logs."""
    SYSTEM = "system"
    OPTIMIZATION = "optimization"
    QUANTUM = "quantum"
    SECURITY = "security"
    PERFORMANCE = "performance"
    AUDIT = "audit"
    BMS = "bms"
    WEATHER = "weather"
    ENERGY = "energy"
    HEALTH = "health"


@dataclass
class LogEntry:
    """Structured log entry for quantum HVAC operations."""
    timestamp: float
    level: LogLevel
    category: LogCategory
    component: str
    message: str
    context: Dict[str, Any]
    correlation_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    building_id: Optional[str] = None
    quantum_job_id: Optional[str] = None
    performance_metrics: Optional[Dict[str, float]] = None
    error_details: Optional[Dict[str, Any]] = None


class QuantumHVACLogger:
    """Advanced logging system for quantum HVAC operations."""
    
    def __init__(
        self,
        log_dir: str = "logs",
        max_file_size: int = 100 * 1024 * 1024,  # 100MB
        backup_count: int = 10,
        structured_logging: bool = True,
        log_to_stdout: bool = True,
        log_level: LogLevel = LogLevel.INFO
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.structured_logging = structured_logging and STRUCTLOG_AVAILABLE
        self.log_level = log_level
        
        # Initialize loggers for different categories
        self.loggers: Dict[LogCategory, logging.Logger] = {}
        self._setup_loggers(max_file_size, backup_count, log_to_stdout)
        
        # Performance tracking
        self.performance_tracker = PerformanceTracker()
        
        # Security audit logger
        self.security_logger = SecurityAuditLogger(self.log_dir)
        
        # Async log queue for high-performance logging
        self.log_queue = asyncio.Queue()
        self.log_processor_task = None
        
        # Context storage for correlation
        self._local = threading.local()
    
    def _setup_loggers(self, max_file_size: int, backup_count: int, log_to_stdout: bool):
        """Set up category-specific loggers."""
        
        # Configure structured logging if available
        if self.structured_logging:
            structlog.configure(
                processors=[
                    structlog.stdlib.filter_by_level,
                    structlog.stdlib.add_logger_name,
                    structlog.stdlib.add_log_level,
                    structlog.stdlib.PositionalArgumentsFormatter(),
                    structlog.processors.TimeStamper(fmt="iso"),
                    structlog.processors.StackInfoRenderer(),
                    structlog.processors.format_exc_info,
                    structlog.processors.UnicodeDecoder(),
                    structlog.processors.JSONRenderer()
                ],
                context_class=dict,
                logger_factory=structlog.stdlib.LoggerFactory(),
                wrapper_class=structlog.stdlib.BoundLogger,
                cache_logger_on_first_use=True,
            )
        
        # Set up category-specific loggers
        for category in LogCategory:
            logger = logging.getLogger(f"quantum_hvac.{category.value}")
            logger.setLevel(getattr(logging, self.log_level.value))
            
            # File handler with rotation
            file_handler = logging.handlers.RotatingFileHandler(
                self.log_dir / f"{category.value}.log",
                maxBytes=max_file_size,
                backupCount=backup_count
            )
            
            # JSON formatter for structured logs
            if self.structured_logging:
                file_handler.setFormatter(JSONLogFormatter())
            else:
                file_handler.setFormatter(logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                ))
            
            logger.addHandler(file_handler)
            
            # Console handler if requested
            if log_to_stdout:
                console_handler = logging.StreamHandler()
                console_handler.setFormatter(logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                ))
                logger.addHandler(console_handler)
            
            self.loggers[category] = logger
    
    def set_context(
        self,
        correlation_id: str = None,
        user_id: str = None,
        session_id: str = None,
        building_id: str = None,
        quantum_job_id: str = None
    ):
        """Set logging context for correlation across operations."""
        if not hasattr(self._local, 'context'):
            self._local.context = {}
        
        if correlation_id:
            self._local.context['correlation_id'] = correlation_id
        if user_id:
            self._local.context['user_id'] = user_id
        if session_id:
            self._local.context['session_id'] = session_id
        if building_id:
            self._local.context['building_id'] = building_id
        if quantum_job_id:
            self._local.context['quantum_job_id'] = quantum_job_id
    
    def get_context(self) -> Dict[str, Any]:
        """Get current logging context."""
        if hasattr(self._local, 'context'):
            return self._local.context.copy()
        return {}
    
    def log(
        self,
        level: LogLevel,
        category: LogCategory,
        component: str,
        message: str,
        **kwargs
    ):
        """Log a message with structured data."""
        
        # Get current context
        context = self.get_context()
        context.update(kwargs.get('context', {}))
        
        # Create log entry
        entry = LogEntry(
            timestamp=time.time(),
            level=level,
            category=category,
            component=component,
            message=message,
            context=context,
            correlation_id=context.get('correlation_id'),
            user_id=context.get('user_id'),
            session_id=context.get('session_id'),
            building_id=context.get('building_id'),
            quantum_job_id=context.get('quantum_job_id'),
            performance_metrics=kwargs.get('performance_metrics'),
            error_details=kwargs.get('error_details')
        )
        
        # Log to appropriate logger
        logger = self.loggers.get(category, self.loggers[LogCategory.SYSTEM])
        log_level = getattr(logging, level.value if level.value != 'QUANTUM' else 'INFO')
        
        if self.structured_logging:
            logger.log(log_level, message, extra=asdict(entry))
        else:
            logger.log(log_level, f"[{component}] {message}", extra={"context": context})
    
    # Convenience methods for different log levels
    
    def debug(self, category: LogCategory, component: str, message: str, **kwargs):
        """Log debug message."""
        self.log(LogLevel.DEBUG, category, component, message, **kwargs)
    
    def info(self, category: LogCategory, component: str, message: str, **kwargs):
        """Log info message."""
        self.log(LogLevel.INFO, category, component, message, **kwargs)
    
    def warning(self, category: LogCategory, component: str, message: str, **kwargs):
        """Log warning message."""
        self.log(LogLevel.WARNING, category, component, message, **kwargs)
    
    def error(self, category: LogCategory, component: str, message: str, **kwargs):
        """Log error message."""
        # Add stack trace if exception context is available
        if 'exception' in kwargs:
            kwargs['error_details'] = {
                'exception_type': type(kwargs['exception']).__name__,
                'exception_message': str(kwargs['exception']),
                'stack_trace': traceback.format_exc()
            }
        
        self.log(LogLevel.ERROR, category, component, message, **kwargs)
    
    def critical(self, category: LogCategory, component: str, message: str, **kwargs):
        """Log critical message."""
        if 'exception' in kwargs:
            kwargs['error_details'] = {
                'exception_type': type(kwargs['exception']).__name__,
                'exception_message': str(kwargs['exception']),
                'stack_trace': traceback.format_exc()
            }
        
        self.log(LogLevel.CRITICAL, category, component, message, **kwargs)
    
    # Specialized logging methods
    
    def quantum_operation(
        self,
        component: str,
        operation: str,
        quantum_params: Dict[str, Any],
        result: Optional[Dict[str, Any]] = None,
        execution_time: Optional[float] = None
    ):
        """Log quantum operation with specialized formatting."""
        context = {
            'operation': operation,
            'quantum_params': quantum_params,
            'result_summary': self._summarize_quantum_result(result) if result else None
        }
        
        performance_metrics = {}
        if execution_time:
            performance_metrics['execution_time'] = execution_time
        
        self.log(
            LogLevel.QUANTUM,
            LogCategory.QUANTUM,
            component,
            f"Quantum operation '{operation}' executed",
            context=context,
            performance_metrics=performance_metrics
        )
    
    def performance_event(
        self,
        component: str,
        event: str,
        metrics: Dict[str, float],
        threshold_violations: Optional[List[str]] = None
    ):
        """Log performance event with metrics."""
        level = LogLevel.WARNING if threshold_violations else LogLevel.PERFORMANCE
        
        context = {
            'event': event,
            'threshold_violations': threshold_violations or []
        }
        
        self.log(
            level,
            LogCategory.PERFORMANCE,
            component,
            f"Performance event: {event}",
            context=context,
            performance_metrics=metrics
        )
    
    def security_event(
        self,
        component: str,
        event_type: str,
        severity: str,
        details: Dict[str, Any],
        user_id: str = None,
        ip_address: str = None
    ):
        """Log security event."""
        context = {
            'event_type': event_type,
            'severity': severity,
            'ip_address': ip_address,
            'details': details
        }
        
        if user_id:
            context['user_id'] = user_id
        
        self.log(
            LogLevel.SECURITY,
            LogCategory.SECURITY,
            component,
            f"Security event: {event_type}",
            context=context
        )
        
        # Also log to security audit system
        self.security_logger.log_security_event(
            event_type, severity, component, details, user_id, ip_address
        )
    
    def audit_event(
        self,
        component: str,
        action: str,
        resource: str,
        user_id: str,
        result: str,
        details: Optional[Dict[str, Any]] = None
    ):
        """Log audit event for compliance."""
        context = {
            'action': action,
            'resource': resource,
            'user_id': user_id,
            'result': result,
            'details': details or {}
        }
        
        self.log(
            LogLevel.AUDIT,
            LogCategory.AUDIT,
            component,
            f"Audit: {user_id} {action} {resource} -> {result}",
            context=context
        )
    
    def _summarize_quantum_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary of quantum operation results for logging."""
        summary = {}
        
        if 'energy' in result:
            summary['energy'] = result['energy']
        if 'num_occurrences' in result:
            summary['num_occurrences'] = result['num_occurrences']
        if 'chain_break_fraction' in result:
            summary['chain_break_fraction'] = result['chain_break_fraction']
        if 'timing' in result:
            summary['timing'] = result['timing']
        
        return summary


class PerformanceTracker:
    """Track performance metrics for logging and monitoring."""
    
    def __init__(self):
        self.active_operations: Dict[str, float] = {}
        self._lock = threading.RLock()
    
    def start_operation(self, operation_id: str) -> str:
        """Start tracking an operation."""
        with self._lock:
            self.active_operations[operation_id] = time.time()
        return operation_id
    
    def end_operation(self, operation_id: str) -> float:
        """End tracking an operation and return duration."""
        with self._lock:
            start_time = self.active_operations.pop(operation_id, None)
            if start_time:
                return time.time() - start_time
            return 0.0
    
    def get_active_operations(self) -> List[str]:
        """Get list of currently active operations."""
        with self._lock:
            return list(self.active_operations.keys())


class SecurityAuditLogger:
    """Specialized logger for security audit events."""
    
    def __init__(self, log_dir: Path):
        self.security_log_path = log_dir / "security_audit.log"
        self.security_logger = logging.getLogger("quantum_hvac.security_audit")
        
        # Set up dedicated security audit file
        handler = logging.handlers.RotatingFileHandler(
            self.security_log_path,
            maxBytes=50 * 1024 * 1024,  # 50MB
            backupCount=20  # Keep more security logs
        )
        
        handler.setFormatter(JSONLogFormatter())
        self.security_logger.addHandler(handler)
        self.security_logger.setLevel(logging.INFO)
    
    def log_security_event(
        self,
        event_type: str,
        severity: str,
        component: str,
        details: Dict[str, Any],
        user_id: str = None,
        ip_address: str = None
    ):
        """Log security audit event."""
        audit_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "severity": severity,
            "component": component,
            "user_id": user_id,
            "ip_address": ip_address,
            "details": details
        }
        
        self.security_logger.info(
            f"SECURITY_EVENT: {event_type}",
            extra={"audit_entry": audit_entry}
        )


class JSONLogFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record):
        """Format log record as JSON."""
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add extra fields if present
        if hasattr(record, 'context'):
            log_entry["context"] = record.context
        
        if hasattr(record, 'performance_metrics'):
            log_entry["performance_metrics"] = record.performance_metrics
        
        if hasattr(record, 'error_details'):
            log_entry["error_details"] = record.error_details
        
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry)


# Global logger instance
_global_logger: Optional[QuantumHVACLogger] = None


def get_logger() -> QuantumHVACLogger:
    """Get or create global logger instance."""
    global _global_logger
    if _global_logger is None:
        _global_logger = QuantumHVACLogger()
    return _global_logger


def configure_logging(
    log_dir: str = "logs",
    log_level: Union[LogLevel, str] = LogLevel.INFO,
    structured_logging: bool = True,
    **kwargs
) -> QuantumHVACLogger:
    """Configure global logging system."""
    global _global_logger
    
    # Handle string log level
    if isinstance(log_level, str):
        log_level = LogLevel(log_level.upper())
    
    _global_logger = QuantumHVACLogger(
        log_dir=log_dir,
        log_level=log_level,
        structured_logging=structured_logging,
        **kwargs
    )
    return _global_logger


# Context manager for operation tracking
class LoggedOperation:
    """Context manager for automatic operation logging."""
    
    def __init__(
        self,
        component: str,
        operation: str,
        category: LogCategory = LogCategory.SYSTEM,
        logger: Optional[QuantumHVACLogger] = None
    ):
        self.component = component
        self.operation = operation
        self.category = category
        self.logger = logger or get_logger()
        self.start_time = None
        self.operation_id = None
    
    def __enter__(self):
        """Start operation tracking."""
        self.start_time = time.time()
        self.operation_id = f"{self.component}_{self.operation}_{int(self.start_time)}"
        
        self.logger.performance_tracker.start_operation(self.operation_id)
        self.logger.debug(
            self.category,
            self.component,
            f"Started operation: {self.operation}"
        )
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """End operation tracking and log results."""
        duration = self.logger.performance_tracker.end_operation(self.operation_id)
        
        if exc_type is None:
            # Successful operation
            self.logger.info(
                self.category,
                self.component,
                f"Completed operation: {self.operation}",
                performance_metrics={"duration": duration}
            )
        else:
            # Failed operation
            self.logger.error(
                self.category,
                self.component,
                f"Failed operation: {self.operation}",
                exception=exc_val,
                performance_metrics={"duration": duration}
            )
        
        return False  # Don't suppress exceptions
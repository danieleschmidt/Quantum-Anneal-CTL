"""Enhanced structured logging with JSON output and correlation IDs."""

import json
import logging
import sys
import traceback
import uuid
from contextlib import contextmanager
from datetime import datetime
from typing import Dict, Any, Optional, List
from contextvars import ContextVar

# Context variable for correlation ID
correlation_id_var: ContextVar[Optional[str]] = ContextVar('correlation_id', default=None)
request_id_var: ContextVar[Optional[str]] = ContextVar('request_id', default=None)
user_id_var: ContextVar[Optional[str]] = ContextVar('user_id', default=None)


class CorrelationFilter(logging.Filter):
    """Add correlation ID and request context to log records."""
    
    def filter(self, record):
        """Add context variables to log record."""
        record.correlation_id = correlation_id_var.get()
        record.request_id = request_id_var.get()
        record.user_id = user_id_var.get()
        return True


class JSONFormatter(logging.Formatter):
    """JSON log formatter with structured fields."""
    
    def __init__(self, service_name: str = "quantum-hvac", version: str = "1.0.0"):
        super().__init__()
        self.service_name = service_name
        self.version = version
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_entry = {
            "timestamp": datetime.utcfromtimestamp(record.created).isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "service": self.service_name,
            "version": self.version,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "thread": record.thread,
            "thread_name": record.threadName,
        }
        
        # Add context information
        if hasattr(record, 'correlation_id') and record.correlation_id:
            log_entry["correlation_id"] = record.correlation_id
        
        if hasattr(record, 'request_id') and record.request_id:
            log_entry["request_id"] = record.request_id
        
        if hasattr(record, 'user_id') and record.user_id:
            log_entry["user_id"] = record.user_id
        
        # Add exception information
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": self.formatException(record.exc_info)
            }
        
        # Add extra fields from record
        for key, value in record.__dict__.items():
            if key not in {
                'name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                'filename', 'module', 'exc_info', 'exc_text', 'stack_info',
                'lineno', 'funcName', 'created', 'msecs', 'relativeCreated',
                'thread', 'threadName', 'processName', 'process', 'message',
                'correlation_id', 'request_id', 'user_id'
            } and not key.startswith('_'):
                log_entry[key] = value
        
        return json.dumps(log_entry, default=str)


class StructuredLogger:
    """Enhanced logger with structured logging capabilities."""
    
    def __init__(self, name: str, level: str = "INFO"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        
        # Remove existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Add JSON formatter
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JSONFormatter())
        handler.addFilter(CorrelationFilter())
        self.logger.addHandler(handler)
        
        # Prevent duplicate logs
        self.logger.propagate = False
    
    def debug(self, message: str, **kwargs):
        """Log debug message with extra fields."""
        self.logger.debug(message, extra=kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message with extra fields."""
        self.logger.info(message, extra=kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message with extra fields."""
        self.logger.warning(message, extra=kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message with extra fields."""
        self.logger.error(message, extra=kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message with extra fields."""
        self.logger.critical(message, extra=kwargs)
    
    def exception(self, message: str, **kwargs):
        """Log exception with traceback."""
        self.logger.exception(message, extra=kwargs)


class MetricsLogger:
    """Logger for metrics and performance data."""
    
    def __init__(self):
        self.logger = StructuredLogger("quantum_ctl.metrics")
    
    def log_optimization_metrics(
        self,
        building_id: str,
        optimization_id: str,
        solver_type: str,
        computation_time_ms: float,
        objective_value: float,
        quantum_metrics: Optional[Dict[str, Any]] = None
    ):
        """Log optimization performance metrics."""
        self.logger.info(
            "Optimization completed",
            metric_type="optimization",
            building_id=building_id,
            optimization_id=optimization_id,
            solver_type=solver_type,
            computation_time_ms=computation_time_ms,
            objective_value=objective_value,
            quantum_metrics=quantum_metrics or {}
        )
    
    def log_energy_metrics(
        self,
        building_id: str,
        energy_kwh: float,
        cost_usd: float,
        period_start: datetime,
        period_end: datetime
    ):
        """Log energy consumption metrics."""
        self.logger.info(
            "Energy consumption recorded",
            metric_type="energy",
            building_id=building_id,
            energy_kwh=energy_kwh,
            cost_usd=cost_usd,
            period_start=period_start.isoformat(),
            period_end=period_end.isoformat()
        )
    
    def log_comfort_metrics(
        self,
        building_id: str,
        comfort_score: float,
        temperature_violations: int,
        humidity_violations: int
    ):
        """Log comfort performance metrics."""
        self.logger.info(
            "Comfort metrics recorded",
            metric_type="comfort",
            building_id=building_id,
            comfort_score=comfort_score,
            temperature_violations=temperature_violations,
            humidity_violations=humidity_violations
        )
    
    def log_system_metrics(
        self,
        cpu_usage_percent: float,
        memory_usage_mb: float,
        active_connections: int,
        response_time_ms: float
    ):
        """Log system performance metrics."""
        self.logger.info(
            "System metrics recorded",
            metric_type="system",
            cpu_usage_percent=cpu_usage_percent,
            memory_usage_mb=memory_usage_mb,
            active_connections=active_connections,
            response_time_ms=response_time_ms
        )


class SecurityLogger:
    """Logger for security events."""
    
    def __init__(self):
        self.logger = StructuredLogger("quantum_ctl.security")
    
    def log_authentication_event(
        self,
        event_type: str,
        username: str,
        success: bool,
        ip_address: str = None,
        user_agent: str = None
    ):
        """Log authentication events."""
        self.logger.info(
            f"Authentication event: {event_type}",
            event_type="authentication",
            auth_event=event_type,
            username=username,
            success=success,
            ip_address=ip_address,
            user_agent=user_agent
        )
    
    def log_authorization_event(
        self,
        event_type: str,
        username: str,
        resource: str,
        action: str,
        allowed: bool,
        reason: str = None
    ):
        """Log authorization events."""
        self.logger.info(
            f"Authorization event: {event_type}",
            event_type="authorization",
            auth_event=event_type,
            username=username,
            resource=resource,
            action=action,
            allowed=allowed,
            reason=reason
        )
    
    def log_security_violation(
        self,
        violation_type: str,
        severity: str,
        description: str,
        username: str = None,
        ip_address: str = None,
        details: Dict[str, Any] = None
    ):
        """Log security violations."""
        self.logger.warning(
            f"Security violation: {violation_type}",
            event_type="security_violation",
            violation_type=violation_type,
            severity=severity,
            description=description,
            username=username,
            ip_address=ip_address,
            details=details or {}
        )


class AuditLogger:
    """Logger for audit trail."""
    
    def __init__(self):
        self.logger = StructuredLogger("quantum_ctl.audit")
    
    def log_building_config_change(
        self,
        building_id: str,
        username: str,
        action: str,
        changes: Dict[str, Any],
        old_values: Dict[str, Any] = None
    ):
        """Log building configuration changes."""
        self.logger.info(
            f"Building configuration {action}",
            event_type="config_change",
            building_id=building_id,
            username=username,
            action=action,
            changes=changes,
            old_values=old_values or {}
        )
    
    def log_control_command(
        self,
        building_id: str,
        username: str,
        command_type: str,
        command_data: Dict[str, Any],
        immediate: bool = False
    ):
        """Log control commands."""
        self.logger.info(
            f"Control command issued: {command_type}",
            event_type="control_command",
            building_id=building_id,
            username=username,
            command_type=command_type,
            command_data=command_data,
            immediate=immediate
        )
    
    def log_optimization_request(
        self,
        building_id: str,
        username: str,
        optimization_id: str,
        parameters: Dict[str, Any]
    ):
        """Log optimization requests."""
        self.logger.info(
            "Optimization requested",
            event_type="optimization_request",
            building_id=building_id,
            username=username,
            optimization_id=optimization_id,
            parameters=parameters
        )


# Context managers for correlation tracking
@contextmanager
def correlation_context(correlation_id: str = None):
    """Context manager for correlation ID tracking."""
    if correlation_id is None:
        correlation_id = str(uuid.uuid4())
    
    token = correlation_id_var.set(correlation_id)
    try:
        yield correlation_id
    finally:
        correlation_id_var.reset(token)


@contextmanager
def request_context(request_id: str = None, user_id: str = None):
    """Context manager for request tracking."""
    if request_id is None:
        request_id = str(uuid.uuid4())
    
    request_token = request_id_var.set(request_id)
    user_token = user_id_var.set(user_id) if user_id else None
    
    try:
        yield request_id
    finally:
        request_id_var.reset(request_token)
        if user_token:
            user_id_var.reset(user_token)


# Global logger instances
main_logger = StructuredLogger("quantum_ctl.main")
metrics_logger = MetricsLogger()
security_logger = SecurityLogger()
audit_logger = AuditLogger()


def setup_logging(level: str = "INFO", service_name: str = "quantum-hvac"):
    """Setup global logging configuration."""
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add structured logging handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter(service_name=service_name))
    handler.addFilter(CorrelationFilter())
    root_logger.addHandler(handler)
    
    # Configure third-party loggers
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("fastapi").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    
    main_logger.info("Structured logging initialized", log_level=level)
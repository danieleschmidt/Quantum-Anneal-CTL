"""
Logging configuration for quantum HVAC control.
"""

import logging
import logging.config
import sys
from typing import Optional, Dict, Any
from pathlib import Path


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
    format_string: Optional[str] = None,
    json_logging: bool = False
) -> None:
    """
    Setup logging configuration.
    
    Args:
        level: Logging level
        log_file: Optional log file path
        format_string: Custom format string
        json_logging: Enable structured JSON logging
    """
    if format_string is None:
        if json_logging:
            format_string = '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s"}'
        else:
            format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Configure root logger
    logging.basicConfig(
        level=level,
        format=format_string,
        handlers=[]
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_formatter = logging.Formatter(format_string)
    console_handler.setFormatter(console_formatter)
    
    # File handler (if specified)
    handlers = [console_handler]
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(format_string)
        file_handler.setFormatter(file_formatter)
        handlers.append(file_handler)
    
    # Apply to root logger
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    for handler in handlers:
        root_logger.addHandler(handler)
    
    # Set specific logger levels
    logging.getLogger('quantum_ctl').setLevel(level)
    logging.getLogger('dwave').setLevel(logging.WARNING)  # Reduce D-Wave SDK verbosity
    logging.getLogger('dimod').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    
    # Create quantum_ctl logger
    logger = logging.getLogger('quantum_ctl')
    logger.info(f"Logging configured at level {logging.getLevelName(level)}")


def get_logging_config(
    level: str = "INFO",
    log_file: Optional[str] = None,
    json_logging: bool = False
) -> Dict[str, Any]:
    """
    Get logging configuration dictionary.
    
    Args:
        level: Logging level as string
        log_file: Optional log file path
        json_logging: Enable JSON format
        
    Returns:
        Logging configuration dictionary
    """
    if json_logging:
        format_string = '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s"}'
    else:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': format_string
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': level,
                'formatter': 'standard',
                'stream': 'ext://sys.stdout'
            }
        },
        'loggers': {
            'quantum_ctl': {
                'level': level,
                'handlers': ['console'],
                'propagate': False
            },
            'dwave': {
                'level': 'WARNING',
                'handlers': ['console'],
                'propagate': False
            },
            'dimod': {
                'level': 'WARNING', 
                'handlers': ['console'],
                'propagate': False
            }
        },
        'root': {
            'level': level,
            'handlers': ['console']
        }
    }
    
    # Add file handler if specified
    if log_file:
        config['handlers']['file'] = {
            'class': 'logging.FileHandler',
            'level': level,
            'formatter': 'standard',
            'filename': log_file
        }
        
        # Add file handler to all loggers
        for logger_config in config['loggers'].values():
            logger_config['handlers'].append('file')
        config['root']['handlers'].append('file')
    
    return config


class QuantumLogger:
    """Custom logger for quantum operations."""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(f"quantum_ctl.{name}")
        self._solve_count = 0
        self._error_count = 0
    
    def log_optimization_start(self, problem_size: int, solver_type: str) -> None:
        """Log start of optimization."""
        self._solve_count += 1
        self.logger.info(
            f"Starting optimization #{self._solve_count} "
            f"(size: {problem_size}, solver: {solver_type})"
        )
    
    def log_optimization_complete(
        self,
        energy: float,
        solve_time: float,
        chain_breaks: float = 0.0
    ) -> None:
        """Log optimization completion."""
        self.logger.info(
            f"Optimization complete: energy={energy:.6f}, "
            f"time={solve_time:.2f}s, chain_breaks={chain_breaks:.3f}"
        )
    
    def log_optimization_error(self, error: Exception) -> None:
        """Log optimization error."""
        self._error_count += 1
        self.logger.error(f"Optimization failed: {error}")
    
    def log_fallback_used(self, reason: str) -> None:
        """Log when fallback solver is used."""
        self.logger.warning(f"Using fallback solver: {reason}")
    
    def log_embedding_stats(self, stats: Dict[str, Any]) -> None:
        """Log embedding statistics."""
        if 'max_chain_length' in stats:
            self.logger.debug(
                f"Embedding: max_chain={stats['max_chain_length']}, "
                f"qubits_used={stats.get('qubits_used', 'unknown')}"
            )
    
    def get_stats(self) -> Dict[str, int]:
        """Get logger statistics."""
        return {
            'solve_count': self._solve_count,
            'error_count': self._error_count,
            'success_rate': (
                (self._solve_count - self._error_count) / self._solve_count
                if self._solve_count > 0 else 0.0
            )
        }


# Enhanced logging features for Generation 2
import json
import time
from datetime import datetime
from typing import Union, List
from collections import deque
import threading
import asyncio
from dataclasses import dataclass, asdict


@dataclass
class StructuredLogEvent:
    """Structured log event for advanced analytics."""
    timestamp: float
    level: str
    logger_name: str
    message: str
    component: str
    operation: Optional[str] = None
    duration_ms: Optional[float] = None
    success: Optional[bool] = None
    error_type: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {k: v for k, v in asdict(self).items() if v is not None}


class StructuredLogger:
    """Enhanced logger with structured output and metrics."""
    
    def __init__(self, component: str, base_logger: Optional[logging.Logger] = None):
        self.component = component
        self.base_logger = base_logger or logging.getLogger(f"quantum_ctl.{component}")
        
        # Event storage for analytics
        self.event_buffer = deque(maxlen=10000)
        self._lock = threading.Lock()
        
        # Metrics tracking
        self.metrics = {
            'total_events': 0,
            'error_count': 0,
            'warning_count': 0,
            'operation_counts': {},
            'operation_durations': {}
        }
    
    def log_event(
        self,
        level: str,
        message: str,
        operation: Optional[str] = None,
        duration_ms: Optional[float] = None,
        success: Optional[bool] = None,
        error_type: Optional[str] = None,
        **metadata
    ) -> None:
        """Log structured event."""
        
        event = StructuredLogEvent(
            timestamp=time.time(),
            level=level.upper(),
            logger_name=self.base_logger.name,
            message=message,
            component=self.component,
            operation=operation,
            duration_ms=duration_ms,
            success=success,
            error_type=error_type,
            metadata=metadata if metadata else None
        )
        
        # Store in buffer
        with self._lock:
            self.event_buffer.append(event)
            self._update_metrics(event)
        
        # Log to standard logger
        log_level = getattr(logging, level.upper(), logging.INFO)
        structured_msg = self._format_structured_message(event)
        self.base_logger.log(log_level, structured_msg)
    
    def _format_structured_message(self, event: StructuredLogEvent) -> str:
        """Format structured log message."""
        base_msg = f"[{event.component}] {event.message}"
        
        if event.operation:
            base_msg += f" | Operation: {event.operation}"
        
        if event.duration_ms is not None:
            base_msg += f" | Duration: {event.duration_ms:.2f}ms"
        
        if event.success is not None:
            base_msg += f" | Success: {event.success}"
        
        if event.error_type:
            base_msg += f" | Error: {event.error_type}"
        
        if event.metadata:
            metadata_str = ", ".join(f"{k}={v}" for k, v in event.metadata.items())
            base_msg += f" | {metadata_str}"
        
        return base_msg
    
    def _update_metrics(self, event: StructuredLogEvent) -> None:
        """Update internal metrics."""
        self.metrics['total_events'] += 1
        
        if event.level == 'ERROR':
            self.metrics['error_count'] += 1
        elif event.level == 'WARNING':
            self.metrics['warning_count'] += 1
        
        if event.operation:
            self.metrics['operation_counts'][event.operation] = (
                self.metrics['operation_counts'].get(event.operation, 0) + 1
            )
            
            if event.duration_ms is not None:
                if event.operation not in self.metrics['operation_durations']:
                    self.metrics['operation_durations'][event.operation] = []
                self.metrics['operation_durations'][event.operation].append(event.duration_ms)
                
                # Keep only last 100 measurements per operation
                if len(self.metrics['operation_durations'][event.operation]) > 100:
                    self.metrics['operation_durations'][event.operation].pop(0)
    
    def info(self, message: str, **kwargs) -> None:
        """Log info level event."""
        self.log_event('INFO', message, **kwargs)
    
    def warning(self, message: str, **kwargs) -> None:
        """Log warning level event."""
        self.log_event('WARNING', message, **kwargs)
    
    def error(self, message: str, **kwargs) -> None:
        """Log error level event."""
        self.log_event('ERROR', message, **kwargs)
    
    def debug(self, message: str, **kwargs) -> None:
        """Log debug level event."""
        self.log_event('DEBUG', message, **kwargs)
    
    def critical(self, message: str, **kwargs) -> None:
        """Log critical level event."""
        self.log_event('CRITICAL', message, **kwargs)
    
    def log_operation_start(self, operation: str, **metadata) -> str:
        """Log start of operation and return correlation ID."""
        correlation_id = f"{operation}_{int(time.time() * 1000)}"
        self.info(f"Starting {operation}", operation=operation, correlation_id=correlation_id, **metadata)
        return correlation_id
    
    def log_operation_complete(
        self,
        operation: str,
        correlation_id: str,
        start_time: float,
        success: bool = True,
        **metadata
    ) -> None:
        """Log operation completion."""
        duration_ms = (time.time() - start_time) * 1000
        
        level = 'INFO' if success else 'ERROR'
        message = f"Completed {operation}" + ("" if success else " with errors")
        
        self.log_event(
            level,
            message,
            operation=operation,
            duration_ms=duration_ms,
            success=success,
            correlation_id=correlation_id,
            **metadata
        )
    
    def get_recent_events(
        self,
        minutes: int = 60,
        level: Optional[str] = None,
        operation: Optional[str] = None
    ) -> List[StructuredLogEvent]:
        """Get recent log events."""
        cutoff_time = time.time() - (minutes * 60)
        
        with self._lock:
            events = [e for e in self.event_buffer if e.timestamp >= cutoff_time]
        
        if level:
            events = [e for e in events if e.level == level.upper()]
        
        if operation:
            events = [e for e in events if e.operation == operation]
        
        return events
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get logging metrics summary."""
        with self._lock:
            operation_stats = {}
            for operation, durations in self.metrics['operation_durations'].items():
                if durations:
                    operation_stats[operation] = {
                        'count': self.metrics['operation_counts'].get(operation, 0),
                        'avg_duration_ms': sum(durations) / len(durations),
                        'min_duration_ms': min(durations),
                        'max_duration_ms': max(durations)
                    }
            
            return {
                'total_events': self.metrics['total_events'],
                'error_count': self.metrics['error_count'],
                'warning_count': self.metrics['warning_count'],
                'error_rate': (
                    self.metrics['error_count'] / self.metrics['total_events']
                    if self.metrics['total_events'] > 0 else 0.0
                ),
                'operation_counts': self.metrics['operation_counts'].copy(),
                'operation_stats': operation_stats
            }


class LogAggregator:
    """Aggregate logs from multiple components for system-wide analysis."""
    
    def __init__(self):
        self.loggers: Dict[str, StructuredLogger] = {}
        self.logger = logging.getLogger("log_aggregator")
    
    def register_logger(self, component: str, structured_logger: StructuredLogger) -> None:
        """Register a structured logger."""
        self.loggers[component] = structured_logger
        self.logger.info(f"Registered logger for component: {component}")
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get system-wide logging metrics."""
        total_events = 0
        total_errors = 0
        total_warnings = 0
        component_metrics = {}
        
        for component, logger in self.loggers.items():
            metrics = logger.get_metrics_summary()
            component_metrics[component] = metrics
            
            total_events += metrics['total_events']
            total_errors += metrics['error_count']
            total_warnings += metrics['warning_count']
        
        return {
            'system_totals': {
                'total_events': total_events,
                'total_errors': total_errors,
                'total_warnings': total_warnings,
                'system_error_rate': total_errors / total_events if total_events > 0 else 0.0
            },
            'component_metrics': component_metrics,
            'timestamp': time.time()
        }
    
    def get_system_events(
        self,
        minutes: int = 60,
        level: Optional[str] = None,
        component: Optional[str] = None
    ) -> List[StructuredLogEvent]:
        """Get events across all components."""
        all_events = []
        
        loggers_to_check = (
            [self.loggers[component]] if component and component in self.loggers
            else self.loggers.values()
        )
        
        for logger in loggers_to_check:
            events = logger.get_recent_events(minutes=minutes, level=level)
            all_events.extend(events)
        
        # Sort by timestamp
        all_events.sort(key=lambda x: x.timestamp, reverse=True)
        
        return all_events
    
    def export_metrics_to_json(self, filepath: str) -> None:
        """Export system metrics to JSON file."""
        metrics = self.get_system_metrics()
        
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        
        self.logger.info(f"Exported metrics to {filepath}")


class HealthCheckLogger(StructuredLogger):
    """Specialized logger for health check operations."""
    
    def __init__(self):
        super().__init__("health_check")
    
    def log_health_check_start(self, component: str, check_type: str) -> str:
        """Log start of health check."""
        return self.log_operation_start(
            "health_check",
            component=component,
            check_type=check_type
        )
    
    def log_health_check_result(
        self,
        component: str,
        check_type: str,
        correlation_id: str,
        start_time: float,
        healthy: bool,
        response_time_ms: float,
        error_message: Optional[str] = None
    ) -> None:
        """Log health check result."""
        metadata = {
            'component': component,
            'check_type': check_type,
            'healthy': healthy,
            'response_time_ms': response_time_ms
        }
        
        if error_message:
            metadata['error_message'] = error_message
        
        self.log_operation_complete(
            "health_check",
            correlation_id,
            start_time,
            success=healthy,
            **metadata
        )
    
    def log_alert_triggered(self, alert_name: str, severity: str, message: str) -> None:
        """Log alert trigger."""
        self.warning(
            f"Alert triggered: {alert_name}",
            operation="alert_trigger",
            alert_name=alert_name,
            severity=severity,
            alert_message=message
        )
    
    def log_alert_resolved(self, alert_name: str) -> None:
        """Log alert resolution."""
        self.info(
            f"Alert resolved: {alert_name}",
            operation="alert_resolve",
            alert_name=alert_name
        )


# Global instances for system-wide logging
_log_aggregator = LogAggregator()
_health_check_logger = HealthCheckLogger()


def get_structured_logger(component: str) -> StructuredLogger:
    """Get or create structured logger for component."""
    return StructuredLogger(component)


def get_log_aggregator() -> LogAggregator:
    """Get global log aggregator."""
    return _log_aggregator


def get_health_check_logger() -> HealthCheckLogger:
    """Get global health check logger.""" 
    return _health_check_logger
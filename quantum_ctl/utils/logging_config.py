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
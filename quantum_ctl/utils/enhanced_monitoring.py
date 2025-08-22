"""
Enhanced monitoring and health checking system for quantum HVAC control.
Real-time system monitoring, performance metrics, and predictive health analysis.
"""

import time
import asyncio
import logging
import numpy as np
import psutil
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from collections import deque, defaultdict
from datetime import datetime, timedelta
import threading
from enum import Enum

logger = logging.getLogger(__name__)


class SystemHealth(Enum):
    """System health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    DEGRADED = "degraded"  
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class HealthMetric:
    """Individual health metric."""
    name: str
    value: float
    status: SystemHealth
    threshold_warning: float
    threshold_critical: float
    unit: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    
    def update(self, value: float):
        """Update metric value and determine status."""
        self.value = value
        self.timestamp = datetime.now()
        
        if value >= self.threshold_critical:
            self.status = SystemHealth.CRITICAL
        elif value >= self.threshold_warning:
            self.status = SystemHealth.WARNING
        else:
            self.status = SystemHealth.HEALTHY


@dataclass
class SystemStatus:
    """Overall system status."""
    health: SystemHealth
    metrics: Dict[str, HealthMetric]
    alerts: List[str] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.now)
    uptime: float = 0.0
    optimization_success_rate: float = 100.0
    average_response_time: float = 0.0


class EnhancedHealthMonitor:
    """Enhanced system health monitoring with predictive analytics."""
    
    def __init__(self, update_interval: float = 30.0):
        self.update_interval = update_interval
        self.start_time = time.time()
        self.metrics: Dict[str, HealthMetric] = {}
        self.metric_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.performance_history = deque(maxlen=1000)
        self.error_counts = defaultdict(int)
        self.optimization_results = deque(maxlen=100)
        
        self.monitoring_active = False
        self.monitor_thread = None
        self.callbacks: List[Callable] = []
        
        self._init_metrics()
    
    def _init_metrics(self):
        """Initialize system metrics."""
        self.metrics = {
            'cpu_usage': HealthMetric(
                name='CPU Usage',
                value=0.0,
                status=SystemHealth.HEALTHY,
                threshold_warning=70.0,
                threshold_critical=90.0,
                unit='%'
            ),
            'memory_usage': HealthMetric(
                name='Memory Usage', 
                value=0.0,
                status=SystemHealth.HEALTHY,
                threshold_warning=80.0,
                threshold_critical=95.0,
                unit='%'
            ),
            'quantum_solver_latency': HealthMetric(
                name='Quantum Solver Latency',
                value=0.0,
                status=SystemHealth.HEALTHY, 
                threshold_warning=5.0,
                threshold_critical=10.0,
                unit='s'
            ),
            'optimization_success_rate': HealthMetric(
                name='Optimization Success Rate',
                value=100.0,
                status=SystemHealth.HEALTHY,
                threshold_warning=90.0,
                threshold_critical=70.0,
                unit='%'
            ),
            'control_loop_frequency': HealthMetric(
                name='Control Loop Frequency',
                value=1.0,
                status=SystemHealth.HEALTHY,
                threshold_warning=0.5,
                threshold_critical=0.1,
                unit='Hz'
            )
        }
    
    def start_monitoring(self):
        """Start continuous monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Enhanced health monitoring started")
    
    def stop_monitoring(self):
        """Stop monitoring."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        logger.info("Enhanced health monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                self._update_system_metrics()
                self._analyze_trends()
                self._check_alerts()
                self._notify_callbacks()
                
                time.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(self.update_interval)
    
    def _update_system_metrics(self):
        """Update system-level metrics."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.metrics['cpu_usage'].update(cpu_percent)
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.metrics['memory_usage'].update(memory.percent)
            
            # Update metric history
            for name, metric in self.metrics.items():
                self.metric_history[name].append({
                    'timestamp': metric.timestamp,
                    'value': metric.value,
                    'status': metric.status
                })
                
        except Exception as e:
            logger.error(f"Failed to update system metrics: {e}")
    
    def _analyze_trends(self):
        """Analyze metric trends for predictive insights."""
        try:
            for metric_name, history in self.metric_history.items():
                if len(history) < 10:  # Need sufficient data
                    continue
                
                # Calculate trend (simple linear regression)
                values = [h['value'] for h in list(history)[-10:]]
                times = list(range(len(values)))
                
                if len(values) > 1:
                    trend = np.polyfit(times, values, 1)[0]  # Slope
                    
                    # Predict next value
                    predicted_next = values[-1] + trend
                    
                    # Check if trend indicates future problems
                    metric = self.metrics[metric_name]
                    if (trend > 0 and predicted_next > metric.threshold_warning and 
                        metric.status == SystemHealth.HEALTHY):
                        logger.warning(f"Trend analysis: {metric_name} trending toward warning level")
                        
        except Exception as e:
            logger.error(f"Trend analysis failed: {e}")
    
    def _check_alerts(self):
        """Check for system alerts."""
        alerts = []
        
        for metric in self.metrics.values():
            if metric.status == SystemHealth.CRITICAL:
                alerts.append(f"CRITICAL: {metric.name} = {metric.value:.1f}{metric.unit}")
            elif metric.status == SystemHealth.WARNING:
                alerts.append(f"WARNING: {metric.name} = {metric.value:.1f}{metric.unit}")
        
        # System-level alerts
        uptime = time.time() - self.start_time
        if len(self.optimization_results) > 0:
            success_rate = sum(1 for r in self.optimization_results if r.get('success', False)) / len(self.optimization_results) * 100
            if success_rate < 70:
                alerts.append(f"CRITICAL: Low optimization success rate {success_rate:.1f}%")
        
        # Log new alerts
        for alert in alerts:
            if alert not in getattr(self, '_last_alerts', []):
                logger.warning(f"Health Alert: {alert}")
        
        self._last_alerts = alerts
    
    def _notify_callbacks(self):
        """Notify registered callbacks of status updates."""
        try:
            status = self.get_system_status()
            for callback in self.callbacks:
                try:
                    callback(status)
                except Exception as e:
                    logger.error(f"Callback notification failed: {e}")
        except Exception as e:
            logger.error(f"Callback notification error: {e}")
    
    def record_optimization_result(self, success: bool, duration: float, 
                                 error: Optional[str] = None):
        """Record optimization attempt result."""
        result = {
            'success': success,
            'duration': duration,
            'timestamp': datetime.now(),
            'error': error
        }
        
        self.optimization_results.append(result)
        
        # Update metrics
        if 'quantum_solver_latency' in self.metrics:
            self.metrics['quantum_solver_latency'].update(duration)
        
        # Update success rate
        if len(self.optimization_results) > 0:
            success_rate = sum(1 for r in self.optimization_results if r['success']) / len(self.optimization_results) * 100
            self.metrics['optimization_success_rate'].update(success_rate)
    
    def record_performance_metric(self, operation: str, duration: float, success: bool = True):
        """Record performance metric."""
        metric = {
            'operation': operation,
            'duration': duration,
            'success': success,
            'timestamp': datetime.now()
        }
        
        self.performance_history.append(metric)
        
        # Update control loop frequency if applicable
        if operation == 'control_loop':
            recent_loops = [m for m in list(self.performance_history)[-10:] 
                           if m['operation'] == 'control_loop']
            if len(recent_loops) > 1:
                time_diff = (recent_loops[-1]['timestamp'] - recent_loops[0]['timestamp']).total_seconds()
                frequency = len(recent_loops) / max(time_diff, 1.0)
                self.metrics['control_loop_frequency'].update(frequency)
    
    def record_error(self, error_type: str, message: str, severity: str = "medium"):
        """Record system error."""
        self.error_counts[error_type] += 1
        logger.error(f"System error [{error_type}]: {message}")
        
        # Update error-related metrics based on frequency
        total_errors = sum(self.error_counts.values())
        if total_errors > 10:  # Threshold for concern
            logger.warning(f"High error count detected: {total_errors} total errors")
    
    def get_system_status(self) -> SystemStatus:
        """Get current system status."""
        # Determine overall health
        statuses = [metric.status for metric in self.metrics.values()]
        
        if SystemHealth.CRITICAL in statuses:
            overall_health = SystemHealth.CRITICAL
        elif SystemHealth.WARNING in statuses:
            overall_health = SystemHealth.WARNING
        elif SystemHealth.DEGRADED in statuses:
            overall_health = SystemHealth.DEGRADED
        else:
            overall_health = SystemHealth.HEALTHY
        
        # Calculate metrics
        uptime = time.time() - self.start_time
        
        # Success rate
        success_rate = 100.0
        if len(self.optimization_results) > 0:
            success_rate = sum(1 for r in self.optimization_results if r['success']) / len(self.optimization_results) * 100
        
        # Average response time
        avg_response_time = 0.0
        if len(self.performance_history) > 0:
            avg_response_time = np.mean([m['duration'] for m in self.performance_history])
        
        return SystemStatus(
            health=overall_health,
            metrics=self.metrics.copy(),
            alerts=getattr(self, '_last_alerts', []),
            last_updated=datetime.now(),
            uptime=uptime,
            optimization_success_rate=success_rate,
            average_response_time=avg_response_time
        )
    
    def add_callback(self, callback: Callable[[SystemStatus], None]):
        """Add status update callback."""
        self.callbacks.append(callback)
    
    def remove_callback(self, callback: Callable):
        """Remove status update callback."""
        if callback in self.callbacks:
            self.callbacks.remove(callback)
    
    def get_metric_history(self, metric_name: str, hours: int = 1) -> List[Dict]:
        """Get historical data for a metric."""
        if metric_name not in self.metric_history:
            return []
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        history = self.metric_history[metric_name]
        
        return [h for h in history if h['timestamp'] >= cutoff_time]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics."""
        if not self.performance_history:
            return {}
        
        operations = defaultdict(list)
        for metric in self.performance_history:
            operations[metric['operation']].append(metric['duration'])
        
        summary = {}
        for operation, durations in operations.items():
            summary[operation] = {
                'count': len(durations),
                'mean': np.mean(durations),
                'median': np.median(durations),
                'min': np.min(durations),
                'max': np.max(durations),
                'std': np.std(durations)
            }
        
        return summary
    
    def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        status = self.get_system_status()
        
        # Additional checks
        health_report = {
            'overall_health': status.health.value,
            'uptime_hours': status.uptime / 3600,
            'success_rate': status.optimization_success_rate,
            'response_time_ms': status.average_response_time * 1000,
            'active_alerts': len(status.alerts),
            'error_counts': dict(self.error_counts),
            'system_load': psutil.getloadavg() if hasattr(psutil, 'getloadavg') else 'unavailable',
            'disk_usage': psutil.disk_usage('/').percent if hasattr(psutil, 'disk_usage') else 'unavailable',
            'timestamp': datetime.now().isoformat()
        }
        
        return health_report


# Global monitor instance
_global_monitor = None

def get_health_monitor() -> EnhancedHealthMonitor:
    """Get global health monitor instance."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = EnhancedHealthMonitor()
        _global_monitor.start_monitoring()
    return _global_monitor


__all__ = [
    'EnhancedHealthMonitor',
    'SystemHealth', 
    'HealthMetric',
    'SystemStatus',
    'get_health_monitor'
]
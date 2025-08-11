"""
Advanced monitoring and observability for Quantum HVAC Control system.

Provides comprehensive system monitoring, alerting, and performance analytics
with integration to Prometheus, Grafana, and external alerting systems.
"""

import asyncio
import time
import psutil
import logging
import json
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque, defaultdict
from enum import Enum
import threading
from functools import wraps
import traceback

try:
    from prometheus_client import Counter, Histogram, Gauge, Summary, Info, start_http_server
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    Counter = Histogram = Gauge = Summary = Info = None

try:
    import aioredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MetricType(Enum):
    """Types of metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class Alert:
    """System alert representation."""
    id: str
    level: AlertLevel
    title: str
    message: str
    source: str
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            'id': self.id,
            'level': self.level.value,
            'title': self.title,
            'message': self.message,
            'source': self.source,
            'timestamp': self.timestamp.isoformat(),
            'tags': self.tags,
            'resolved': self.resolved,
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None
        }


@dataclass
class MetricValue:
    """Metric value with metadata."""
    name: str
    value: Union[int, float]
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)


class PerformanceTracker:
    """Track performance metrics and statistics."""
    
    def __init__(self, max_samples: int = 1000):
        self.max_samples = max_samples
        self._metrics = defaultdict(lambda: deque(maxlen=max_samples))
        self._lock = threading.RLock()
    
    def record(self, metric_name: str, value: Union[int, float], labels: Dict[str, str] = None):
        """Record a metric value."""
        with self._lock:
            metric_value = MetricValue(
                name=metric_name,
                value=value,
                timestamp=datetime.utcnow(),
                labels=labels or {}
            )
            self._metrics[metric_name].append(metric_value)
    
    def get_stats(self, metric_name: str, time_window: Optional[timedelta] = None) -> Dict[str, float]:
        """Get statistics for a metric."""
        with self._lock:
            values = self._metrics[metric_name]
            
            if not values:
                return {}
            
            # Filter by time window if specified
            if time_window:
                cutoff = datetime.utcnow() - time_window
                values = [v for v in values if v.timestamp >= cutoff]
            
            if not values:
                return {}
            
            numeric_values = [v.value for v in values]
            
            return {
                'count': len(numeric_values),
                'min': min(numeric_values),
                'max': max(numeric_values),
                'mean': sum(numeric_values) / len(numeric_values),
                'median': sorted(numeric_values)[len(numeric_values) // 2],
                'p95': sorted(numeric_values)[int(len(numeric_values) * 0.95)],
                'p99': sorted(numeric_values)[int(len(numeric_values) * 0.99)],
                'latest': numeric_values[-1],
                'rate_per_second': len(numeric_values) / (time_window.total_seconds() if time_window else 60)
            }
    
    def get_all_metrics(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get all recorded metrics."""
        with self._lock:
            result = {}
            for name, values in self._metrics.items():
                result[name] = [
                    {
                        'value': v.value,
                        'timestamp': v.timestamp.isoformat(),
                        'labels': v.labels
                    }
                    for v in list(values)  # Create a copy to avoid concurrent modification
                ]
            return result


class SystemMonitor:
    """Monitor system resources and health."""
    
    def __init__(self, check_interval: int = 30):
        self.check_interval = check_interval
        self.logger = logging.getLogger(__name__)
        self._running = False
        self._task = None
        self.performance_tracker = PerformanceTracker()
        
        # Thresholds for alerts
        self.thresholds = {
            'cpu_usage': 80.0,
            'memory_usage': 85.0,
            'disk_usage': 90.0,
            'response_time': 5.0,
            'error_rate': 5.0
        }
        
        # Alert handlers
        self.alert_handlers: List[Callable[[Alert], None]] = []
    
    async def start(self):
        """Start system monitoring."""
        if self._running:
            return
        
        self._running = True
        self._task = asyncio.create_task(self._monitoring_loop())
        self.logger.info("System monitoring started")
    
    async def stop(self):
        """Stop system monitoring."""
        if not self._running:
            return
        
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("System monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while self._running:
            try:
                await self._collect_system_metrics()
                await self._check_thresholds()
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(self.check_interval)
    
    async def _collect_system_metrics(self):
        """Collect system metrics."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.performance_tracker.record('system_cpu_usage', cpu_percent)
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.performance_tracker.record('system_memory_usage', memory.percent)
            self.performance_tracker.record('system_memory_available', memory.available)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            self.performance_tracker.record('system_disk_usage', disk_percent)
            
            # Network I/O
            net_io = psutil.net_io_counters()
            self.performance_tracker.record('network_bytes_sent', net_io.bytes_sent)
            self.performance_tracker.record('network_bytes_recv', net_io.bytes_recv)
            
            # Process-specific metrics
            process = psutil.Process()
            self.performance_tracker.record('process_memory_rss', process.memory_info().rss)
            self.performance_tracker.record('process_cpu_percent', process.cpu_percent())
            self.performance_tracker.record('process_num_threads', process.num_threads())
            
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")
    
    async def _check_thresholds(self):
        """Check if any metrics exceed thresholds."""
        try:
            # Check CPU usage
            cpu_stats = self.performance_tracker.get_stats('system_cpu_usage', timedelta(minutes=5))
            if cpu_stats and cpu_stats.get('mean', 0) > self.thresholds['cpu_usage']:
                await self._create_alert(
                    AlertLevel.WARNING,
                    "High CPU Usage",
                    f"CPU usage is {cpu_stats['mean']:.1f}% (threshold: {self.thresholds['cpu_usage']}%)",
                    "system_monitor"
                )
            
            # Check memory usage
            memory_stats = self.performance_tracker.get_stats('system_memory_usage', timedelta(minutes=5))
            if memory_stats and memory_stats.get('latest', 0) > self.thresholds['memory_usage']:
                await self._create_alert(
                    AlertLevel.WARNING,
                    "High Memory Usage",
                    f"Memory usage is {memory_stats['latest']:.1f}% (threshold: {self.thresholds['memory_usage']}%)",
                    "system_monitor"
                )
            
            # Check disk usage
            disk_stats = self.performance_tracker.get_stats('system_disk_usage', timedelta(minutes=5))
            if disk_stats and disk_stats.get('latest', 0) > self.thresholds['disk_usage']:
                await self._create_alert(
                    AlertLevel.ERROR,
                    "High Disk Usage",
                    f"Disk usage is {disk_stats['latest']:.1f}% (threshold: {self.thresholds['disk_usage']}%)",
                    "system_monitor"
                )
                
        except Exception as e:
            self.logger.error(f"Error checking thresholds: {e}")
    
    async def _create_alert(self, level: AlertLevel, title: str, message: str, source: str):
        """Create and handle an alert."""
        alert = Alert(
            id=f"{source}_{int(time.time())}",
            level=level,
            title=title,
            message=message,
            source=source,
            timestamp=datetime.utcnow()
        )
        
        # Log the alert
        log_level = logging.WARNING if level == AlertLevel.WARNING else logging.ERROR
        self.logger.log(log_level, f"ALERT [{level.value.upper()}] {title}: {message}")
        
        # Call alert handlers
        for handler in self.alert_handlers:
            try:
                await asyncio.get_event_loop().run_in_executor(None, handler, alert)
            except Exception as e:
                self.logger.error(f"Alert handler error: {e}")
    
    def add_alert_handler(self, handler: Callable[[Alert], None]):
        """Add an alert handler."""
        self.alert_handlers.append(handler)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status."""
        try:
            cpu_stats = self.performance_tracker.get_stats('system_cpu_usage', timedelta(minutes=5))
            memory_stats = self.performance_tracker.get_stats('system_memory_usage', timedelta(minutes=5))
            disk_stats = self.performance_tracker.get_stats('system_disk_usage', timedelta(minutes=5))
            
            return {
                'timestamp': datetime.utcnow().isoformat(),
                'status': 'healthy',
                'cpu': {
                    'current': cpu_stats.get('latest', 0) if cpu_stats else 0,
                    'average_5m': cpu_stats.get('mean', 0) if cpu_stats else 0,
                    'threshold': self.thresholds['cpu_usage']
                },
                'memory': {
                    'current': memory_stats.get('latest', 0) if memory_stats else 0,
                    'threshold': self.thresholds['memory_usage']
                },
                'disk': {
                    'current': disk_stats.get('latest', 0) if disk_stats else 0,
                    'threshold': self.thresholds['disk_usage']
                },
                'uptime': time.time() - psutil.boot_time(),
                'monitoring': {
                    'running': self._running,
                    'check_interval': self.check_interval,
                    'metrics_collected': len(self.performance_tracker._metrics)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error getting system status: {e}")
            return {
                'timestamp': datetime.utcnow().isoformat(),
                'status': 'error',
                'error': str(e)
            }


class PrometheusMetrics:
    """Prometheus metrics integration."""
    
    def __init__(self, port: int = 8000, enabled: bool = True):
        self.port = port
        self.enabled = enabled and PROMETHEUS_AVAILABLE
        self.logger = logging.getLogger(__name__)
        
        if not self.enabled:
            if not PROMETHEUS_AVAILABLE:
                self.logger.warning("Prometheus client not available, metrics disabled")
            return
        
        # Define metrics
        self._setup_metrics()
        
        # Start metrics server
        try:
            start_http_server(port)
            self.logger.info(f"Prometheus metrics server started on port {port}")
        except Exception as e:
            self.logger.error(f"Failed to start Prometheus metrics server: {e}")
            self.enabled = False
    
    def _setup_metrics(self):
        """Setup Prometheus metrics."""
        if not self.enabled:
            return
        
        # Request metrics
        self.http_requests_total = Counter(
            'quantum_hvac_http_requests_total',
            'Total HTTP requests',
            ['method', 'endpoint', 'status_code']
        )
        
        self.http_request_duration = Histogram(
            'quantum_hvac_http_request_duration_seconds',
            'HTTP request duration in seconds',
            ['method', 'endpoint']
        )
        
        # System metrics
        self.system_cpu_usage = Gauge(
            'quantum_hvac_system_cpu_usage_percent',
            'System CPU usage percentage'
        )
        
        self.system_memory_usage = Gauge(
            'quantum_hvac_system_memory_usage_percent',
            'System memory usage percentage'
        )
        
        self.system_disk_usage = Gauge(
            'quantum_hvac_system_disk_usage_percent',
            'System disk usage percentage'
        )
        
        # Application metrics
        self.quantum_optimizations_total = Counter(
            'quantum_hvac_quantum_optimizations_total',
            'Total quantum optimizations performed',
            ['solver_type', 'status']
        )
        
        self.quantum_optimization_duration = Histogram(
            'quantum_hvac_quantum_optimization_duration_seconds',
            'Quantum optimization duration in seconds',
            ['solver_type']
        )
        
        self.bms_operations_total = Counter(
            'quantum_hvac_bms_operations_total',
            'Total BMS operations',
            ['protocol', 'operation', 'status']
        )
        
        self.building_zones_total = Gauge(
            'quantum_hvac_building_zones_total',
            'Total number of building zones'
        )
        
        self.active_buildings = Gauge(
            'quantum_hvac_active_buildings',
            'Number of active buildings'
        )
        
        # Error metrics
        self.errors_total = Counter(
            'quantum_hvac_errors_total',
            'Total errors',
            ['component', 'error_type']
        )
        
        # Cache metrics
        self.cache_hits_total = Counter(
            'quantum_hvac_cache_hits_total',
            'Total cache hits',
            ['cache_type']
        )
        
        self.cache_misses_total = Counter(
            'quantum_hvac_cache_misses_total',
            'Total cache misses',
            ['cache_type']
        )
    
    def record_http_request(self, method: str, endpoint: str, status_code: int, duration: float):
        """Record HTTP request metrics."""
        if not self.enabled:
            return
        
        self.http_requests_total.labels(method=method, endpoint=endpoint, status_code=str(status_code)).inc()
        self.http_request_duration.labels(method=method, endpoint=endpoint).observe(duration)
    
    def update_system_metrics(self, cpu: float, memory: float, disk: float):
        """Update system metrics."""
        if not self.enabled:
            return
        
        self.system_cpu_usage.set(cpu)
        self.system_memory_usage.set(memory)
        self.system_disk_usage.set(disk)
    
    def record_quantum_optimization(self, solver_type: str, duration: float, status: str):
        """Record quantum optimization metrics."""
        if not self.enabled:
            return
        
        self.quantum_optimizations_total.labels(solver_type=solver_type, status=status).inc()
        self.quantum_optimization_duration.labels(solver_type=solver_type).observe(duration)
    
    def record_bms_operation(self, protocol: str, operation: str, status: str):
        """Record BMS operation metrics."""
        if not self.enabled:
            return
        
        self.bms_operations_total.labels(protocol=protocol, operation=operation, status=status).inc()
    
    def update_building_metrics(self, zones_count: int, buildings_count: int):
        """Update building metrics."""
        if not self.enabled:
            return
        
        self.building_zones_total.set(zones_count)
        self.active_buildings.set(buildings_count)
    
    def record_error(self, component: str, error_type: str):
        """Record error metrics."""
        if not self.enabled:
            return
        
        self.errors_total.labels(component=component, error_type=error_type).inc()
    
    def record_cache_hit(self, cache_type: str):
        """Record cache hit."""
        if not self.enabled:
            return
        
        self.cache_hits_total.labels(cache_type=cache_type).inc()
    
    def record_cache_miss(self, cache_type: str):
        """Record cache miss."""
        if not self.enabled:
            return
        
        self.cache_misses_total.labels(cache_type=cache_type).inc()


class AdvancedMonitor:
    """Advanced monitoring system combining multiple monitoring components."""
    
    def __init__(
        self,
        prometheus_port: int = 8000,
        system_check_interval: int = 30,
        redis_url: Optional[str] = None
    ):
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.system_monitor = SystemMonitor(check_interval=system_check_interval)
        self.prometheus_metrics = PrometheusMetrics(port=prometheus_port)
        self.performance_tracker = PerformanceTracker()
        
        # Redis connection for distributed metrics (optional)
        self.redis_url = redis_url
        self.redis_client = None
        
        # Alert history
        self.alert_history: List[Alert] = []
        self.max_alert_history = 1000
        
        # Setup alert handlers
        self.system_monitor.add_alert_handler(self._handle_alert)
        
    async def start(self):
        """Start all monitoring components."""
        try:
            # Start system monitor
            await self.system_monitor.start()
            
            # Initialize Redis connection if configured
            if self.redis_url and REDIS_AVAILABLE:
                try:
                    self.redis_client = aioredis.from_url(self.redis_url)
                    await self.redis_client.ping()
                    self.logger.info("Redis connection established for monitoring")
                except Exception as e:
                    self.logger.warning(f"Failed to connect to Redis for monitoring: {e}")
            
            # Start background tasks
            asyncio.create_task(self._metrics_sync_loop())
            
            self.logger.info("Advanced monitoring started")
            
        except Exception as e:
            self.logger.error(f"Failed to start advanced monitoring: {e}")
            raise
    
    async def stop(self):
        """Stop all monitoring components."""
        try:
            await self.system_monitor.stop()
            
            if self.redis_client:
                await self.redis_client.close()
            
            self.logger.info("Advanced monitoring stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping advanced monitoring: {e}")
    
    async def _metrics_sync_loop(self):
        """Sync metrics between components."""
        while True:
            try:
                # Sync system metrics to Prometheus
                system_status = self.system_monitor.get_system_status()
                if system_status.get('status') == 'healthy':
                    cpu = system_status['cpu']['current']
                    memory = system_status['memory']['current']
                    disk = system_status['disk']['current']
                    
                    self.prometheus_metrics.update_system_metrics(cpu, memory, disk)
                
                # Sync to Redis if available
                if self.redis_client:
                    await self._sync_metrics_to_redis()
                
                await asyncio.sleep(30)  # Sync every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Metrics sync error: {e}")
                await asyncio.sleep(30)
    
    async def _sync_metrics_to_redis(self):
        """Sync metrics to Redis for distributed access."""
        try:
            if not self.redis_client:
                return
            
            # Store system status
            system_status = self.system_monitor.get_system_status()
            await self.redis_client.setex(
                'quantum_hvac:monitoring:system_status',
                300,  # 5 minutes TTL
                json.dumps(system_status)
            )
            
            # Store recent alerts
            recent_alerts = [
                alert.to_dict() for alert in self.alert_history[-10:]
            ]
            await self.redis_client.setex(
                'quantum_hvac:monitoring:recent_alerts',
                300,
                json.dumps(recent_alerts)
            )
            
        except Exception as e:
            self.logger.error(f"Redis sync error: {e}")
    
    def _handle_alert(self, alert: Alert):
        """Handle system alerts."""
        # Add to history
        self.alert_history.append(alert)
        
        # Trim history if too long
        if len(self.alert_history) > self.max_alert_history:
            self.alert_history = self.alert_history[-self.max_alert_history:]
        
        # Record in Prometheus
        self.prometheus_metrics.record_error(alert.source, alert.level.value)
    
    def record_quantum_operation(self, solver_type: str, duration: float, success: bool):
        """Record quantum optimization operation."""
        status = "success" if success else "failure"
        self.prometheus_metrics.record_quantum_optimization(solver_type, duration, status)
        self.performance_tracker.record('quantum_optimization_duration', duration, {'solver_type': solver_type})
    
    def record_bms_operation(self, protocol: str, operation: str, success: bool):
        """Record BMS operation."""
        status = "success" if success else "failure"
        self.prometheus_metrics.record_bms_operation(protocol, operation, status)
        self.performance_tracker.record('bms_operation_count', 1, {'protocol': protocol, 'operation': operation})
    
    def record_http_request(self, method: str, endpoint: str, status_code: int, duration: float):
        """Record HTTP request."""
        self.prometheus_metrics.record_http_request(method, endpoint, status_code, duration)
        self.performance_tracker.record('http_request_duration', duration, {'method': method, 'endpoint': endpoint})
    
    def record_error(self, component: str, error_type: str, error: Exception):
        """Record system error."""
        self.prometheus_metrics.record_error(component, error_type)
        self.logger.error(f"Error in {component} ({error_type}): {error}")
        
        # Create alert for critical errors
        asyncio.create_task(self._create_error_alert(component, error_type, error))
    
    async def _create_error_alert(self, component: str, error_type: str, error: Exception):
        """Create alert for errors."""
        level = AlertLevel.ERROR if error_type in ['exception', 'timeout'] else AlertLevel.WARNING
        
        alert = Alert(
            id=f"{component}_{error_type}_{int(time.time())}",
            level=level,
            title=f"Error in {component}",
            message=f"{error_type}: {str(error)}",
            source=component,
            timestamp=datetime.utcnow(),
            tags={'error_type': error_type}
        )
        
        self._handle_alert(alert)
    
    def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        try:
            system_status = self.system_monitor.get_system_status()
            
            # Get recent performance metrics
            recent_metrics = {}
            for metric_name in ['http_request_duration', 'quantum_optimization_duration', 'bms_operation_count']:
                stats = self.performance_tracker.get_stats(metric_name, timedelta(hours=1))
                if stats:
                    recent_metrics[metric_name] = stats
            
            # Get recent alerts
            recent_alerts = [
                alert.to_dict() for alert in self.alert_history[-5:]
            ]
            
            return {
                'timestamp': datetime.utcnow().isoformat(),
                'system': system_status,
                'metrics': {
                    'prometheus_enabled': self.prometheus_metrics.enabled,
                    'redis_connected': self.redis_client is not None,
                    'recent_performance': recent_metrics
                },
                'alerts': {
                    'total_count': len(self.alert_history),
                    'recent': recent_alerts,
                    'unresolved_count': sum(1 for alert in self.alert_history if not alert.resolved)
                },
                'monitoring': {
                    'system_monitor_running': self.system_monitor._running,
                    'components_healthy': True  # Would implement component health checks
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error getting comprehensive status: {e}")
            return {
                'timestamp': datetime.utcnow().isoformat(),
                'status': 'error',
                'error': str(e)
            }


# Decorator for monitoring function performance
def monitor_performance(component: str, operation: str):
    """Decorator to monitor function performance."""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            success = True
            error = None
            
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                error = e
                raise
            finally:
                duration = time.time() - start_time
                
                # Record metrics if monitor is available
                if hasattr(func, '_monitor') and func._monitor:
                    if hasattr(func._monitor, 'record_operation'):
                        func._monitor.record_operation(component, operation, duration, success)
                
                # Log performance
                logger = logging.getLogger(func.__module__)
                if success:
                    logger.debug(f"{component}.{operation} completed in {duration:.3f}s")
                else:
                    logger.error(f"{component}.{operation} failed in {duration:.3f}s: {error}")
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            success = True
            error = None
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                error = e
                raise
            finally:
                duration = time.time() - start_time
                
                # Log performance
                logger = logging.getLogger(func.__module__)
                if success:
                    logger.debug(f"{component}.{operation} completed in {duration:.3f}s")
                else:
                    logger.error(f"{component}.{operation} failed in {duration:.3f}s: {error}")
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# Global monitoring instance
_global_monitor: Optional[AdvancedMonitor] = None


def get_monitor() -> Optional[AdvancedMonitor]:
    """Get global monitoring instance."""
    return _global_monitor


def init_monitoring(
    prometheus_port: int = 8000,
    system_check_interval: int = 30,
    redis_url: Optional[str] = None
) -> AdvancedMonitor:
    """Initialize global monitoring."""
    global _global_monitor
    
    if _global_monitor is None:
        _global_monitor = AdvancedMonitor(
            prometheus_port=prometheus_port,
            system_check_interval=system_check_interval,
            redis_url=redis_url
        )
    
    return _global_monitor
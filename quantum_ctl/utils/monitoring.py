"""
System monitoring and health checks for quantum HVAC control.
"""

import time
import psutil
import logging
import json
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from collections import deque, defaultdict
import threading
import asyncio


@dataclass
class SystemMetrics:
    """System performance metrics."""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    disk_used_percent: float
    optimization_count: int = 0
    avg_optimization_time: float = 0.0
    error_count: int = 0
    last_error: Optional[str] = None


@dataclass
class PerformanceTracker:
    """Tracks optimization performance metrics."""
    solve_times: deque = field(default_factory=lambda: deque(maxlen=100))
    energy_values: deque = field(default_factory=lambda: deque(maxlen=100))
    chain_breaks: deque = field(default_factory=lambda: deque(maxlen=100))
    success_count: int = 0
    failure_count: int = 0
    
    def add_solve(self, solve_time: float, energy: float, chain_breaks: float = 0.0):
        """Add optimization result."""
        self.solve_times.append(solve_time)
        self.energy_values.append(energy)
        self.chain_breaks.append(chain_breaks)
        self.success_count += 1
    
    def add_failure(self):
        """Add optimization failure."""
        self.failure_count += 1
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.0
    
    @property
    def avg_solve_time(self) -> float:
        """Average solve time."""
        return sum(self.solve_times) / len(self.solve_times) if self.solve_times else 0.0
    
    @property
    def avg_energy(self) -> float:
        """Average energy value."""
        return sum(self.energy_values) / len(self.energy_values) if self.energy_values else 0.0
    
    @property
    def avg_chain_breaks(self) -> float:
        """Average chain break fraction."""
        return sum(self.chain_breaks) / len(self.chain_breaks) if self.chain_breaks else 0.0


class HealthMonitor:
    """System health monitoring and alerting."""
    
    def __init__(self, check_interval: int = 60):
        self.check_interval = check_interval
        self.logger = logging.getLogger(__name__)
        
        # Metrics tracking
        self.metrics_history: deque = deque(maxlen=1000)
        self.performance_tracker = PerformanceTracker()
        
        # Health thresholds
        self.cpu_threshold = 90.0      # %
        self.memory_threshold = 85.0   # %
        self.disk_threshold = 90.0     # %
        self.error_rate_threshold = 0.1  # 10%
        
        # Monitoring state
        self._monitoring = False
        self._monitor_thread = None
        self._lock = threading.Lock()
        
        # Alert callbacks
        self._alert_callbacks: List = []
    
    def add_alert_callback(self, callback):
        """Add callback for health alerts."""
        self._alert_callbacks.append(callback)
    
    def start_monitoring(self):
        """Start health monitoring."""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        
        self.logger.info("Health monitoring started")
    
    def stop_monitoring(self):
        """Stop health monitoring."""
        self._monitoring = False
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5.0)
        
        self.logger.info("Health monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self._monitoring:
            try:
                metrics = self._collect_metrics()
                
                with self._lock:
                    self.metrics_history.append(metrics)
                
                # Check health conditions
                self._check_health(metrics)
                
                time.sleep(self.check_interval)
                
            except Exception as e:
                self.logger.error(f"Health monitoring error: {e}")
                time.sleep(self.check_interval)
    
    def _collect_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        # CPU and memory
        cpu_percent = psutil.cpu_percent(interval=1.0)
        memory = psutil.virtual_memory()
        
        # Disk usage
        disk = psutil.disk_usage('/')
        disk_percent = disk.used / disk.total * 100
        
        # Performance metrics
        perf = self.performance_tracker
        
        return SystemMetrics(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_mb=memory.used / 1024 / 1024,
            disk_used_percent=disk_percent,
            optimization_count=perf.success_count,
            avg_optimization_time=perf.avg_solve_time,
            error_count=perf.failure_count
        )
    
    def _check_health(self, metrics: SystemMetrics):
        """Check health conditions and trigger alerts."""
        alerts = []
        
        # CPU check
        if metrics.cpu_percent > self.cpu_threshold:
            alerts.append(f"High CPU usage: {metrics.cpu_percent:.1f}%")
        
        # Memory check
        if metrics.memory_percent > self.memory_threshold:
            alerts.append(f"High memory usage: {metrics.memory_percent:.1f}%")
        
        # Disk check
        if metrics.disk_used_percent > self.disk_threshold:
            alerts.append(f"High disk usage: {metrics.disk_used_percent:.1f}%")
        
        # Error rate check
        if self.performance_tracker.success_rate < (1 - self.error_rate_threshold):
            error_rate = (1 - self.performance_tracker.success_rate) * 100
            alerts.append(f"High error rate: {error_rate:.1f}%")
        
        # Trigger alerts
        for alert in alerts:
            self._trigger_alert("HEALTH_WARNING", alert, metrics)
    
    def _trigger_alert(self, alert_type: str, message: str, metrics: SystemMetrics):
        """Trigger health alert."""
        self.logger.warning(f"{alert_type}: {message}")
        
        alert_data = {
            'type': alert_type,
            'message': message,
            'timestamp': metrics.timestamp,
            'metrics': metrics
        }
        
        # Call alert callbacks
        for callback in self._alert_callbacks:
            try:
                callback(alert_data)
            except Exception as e:
                self.logger.error(f"Alert callback error: {e}")
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status."""
        with self._lock:
            current_metrics = self.metrics_history[-1] if self.metrics_history else None
        
        if not current_metrics:
            return {'status': 'unknown', 'message': 'No metrics available'}
        
        # Determine overall health
        issues = []
        
        if current_metrics.cpu_percent > self.cpu_threshold:
            issues.append('high_cpu')
        if current_metrics.memory_percent > self.memory_threshold:
            issues.append('high_memory')
        if current_metrics.disk_used_percent > self.disk_threshold:
            issues.append('high_disk')
        if self.performance_tracker.success_rate < (1 - self.error_rate_threshold):
            issues.append('high_error_rate')
        
        if not issues:
            status = 'healthy'
        elif len(issues) == 1:
            status = 'warning'
        else:
            status = 'critical'
        
        return {
            'status': status,
            'issues': issues,
            'metrics': current_metrics,
            'performance': {
                'success_rate': self.performance_tracker.success_rate,
                'avg_solve_time': self.performance_tracker.avg_solve_time,
                'avg_energy': self.performance_tracker.avg_energy,
                'total_optimizations': self.performance_tracker.success_count + self.performance_tracker.failure_count
            }
        }
    
    def get_metrics_history(self, hours: int = 24) -> List[SystemMetrics]:
        """Get metrics history for specified hours."""
        cutoff_time = time.time() - (hours * 3600)
        
        with self._lock:
            return [m for m in self.metrics_history if m.timestamp > cutoff_time]
    
    def record_optimization(self, success: bool, solve_time: float = 0.0, 
                          energy: float = 0.0, chain_breaks: float = 0.0):
        """Record optimization result."""
        if success:
            self.performance_tracker.add_solve(solve_time, energy, chain_breaks)
        else:
            self.performance_tracker.add_failure()


class CircuitBreaker:
    """Circuit breaker pattern for fault tolerance."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = 'closed'  # closed, open, half_open
        
        self.logger = logging.getLogger(__name__)
    
    def call(self, func, *args, **kwargs):
        """Call function with circuit breaker protection."""
        if self.state == 'open':
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = 'half_open'
                self.logger.info("Circuit breaker half-open, attempting recovery")
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
            
        except Exception as e:
            self._on_failure()
            raise e
    
    def _on_success(self):
        """Handle successful call."""
        if self.state == 'half_open':
            self.state = 'closed'
            self.failure_count = 0
            self.logger.info("Circuit breaker closed after recovery")
    
    def _on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = 'open'
            self.logger.warning(f"Circuit breaker OPEN after {self.failure_count} failures")
    
    def get_state(self) -> Dict[str, Any]:
        """Get circuit breaker state."""
        return {
            'state': self.state,
            'failure_count': self.failure_count,
            'last_failure_time': self.last_failure_time,
            'time_since_failure': time.time() - self.last_failure_time
        }


class RetryManager:
    """Exponential backoff retry manager."""
    
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 60.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        
        self.logger = logging.getLogger(__name__)
    
    async def retry_async(self, func, *args, **kwargs):
        """Retry async function with exponential backoff."""
        for attempt in range(self.max_retries + 1):
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
                    
            except Exception as e:
                if attempt == self.max_retries:
                    self.logger.error(f"All {self.max_retries + 1} attempts failed: {e}")
                    raise e
                
                delay = min(self.base_delay * (2 ** attempt), self.max_delay)
                self.logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.1f}s")
                
                await asyncio.sleep(delay)
    
    def retry_sync(self, func, *args, **kwargs):
        """Retry sync function with exponential backoff."""
        for attempt in range(self.max_retries + 1):
            try:
                return func(*args, **kwargs)
                
            except Exception as e:
                if attempt == self.max_retries:
                    self.logger.error(f"All {self.max_retries + 1} attempts failed: {e}")
                    raise e
                
                delay = min(self.base_delay * (2 ** attempt), self.max_delay)
                self.logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.1f}s")
                
                time.sleep(delay)


# Enhanced monitoring features for Generation 2
class AdvancedMetricsCollector:
    """Enhanced metrics collector with Prometheus integration."""
    
    def __init__(self):
        self.logger = logging.getLogger("advanced_metrics")
        self.metrics_buffer = deque(maxlen=10000)
        
        # Try to import prometheus_client
        try:
            from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry
            self.prometheus_available = True
            self.registry = CollectorRegistry()
            self._setup_prometheus_metrics()
        except ImportError:
            self.logger.warning("prometheus_client not available, using basic metrics only")
            self.prometheus_available = False
    
    def _setup_prometheus_metrics(self):
        """Setup Prometheus metrics."""
        if not self.prometheus_available:
            return
        
        from prometheus_client import Counter, Histogram, Gauge
        
        # Core quantum control metrics
        self.quantum_operations = Counter(
            'quantum_operations_total', 
            'Total quantum operations performed',
            ['operation_type', 'status'],
            registry=self.registry
        )
        
        self.optimization_duration = Histogram(
            'optimization_duration_seconds',
            'Time spent on optimization',
            ['solver_type'],
            registry=self.registry
        )
        
        self.bms_connectivity = Gauge(
            'bms_connectivity_status',
            'BMS connection status (1=connected, 0=disconnected)',
            ['protocol'],
            registry=self.registry
        )
        
        self.energy_consumption = Gauge(
            'hvac_energy_consumption_kw',
            'Current HVAC energy consumption',
            ['building', 'zone'],
            registry=self.registry
        )
        
        self.temperature_control = Gauge(
            'zone_temperature_celsius',
            'Zone temperature measurements',
            ['building', 'zone'],
            registry=self.registry
        )
    
    def record_quantum_operation(self, operation_type: str, success: bool, duration: float = 0.0):
        """Record quantum operation metrics."""
        status = 'success' if success else 'failure'
        
        if self.prometheus_available:
            self.quantum_operations.labels(operation_type=operation_type, status=status).inc()
            if operation_type in ['optimization', 'embedding']:
                self.optimization_duration.labels(solver_type='quantum').observe(duration)
        
        # Store in buffer for local access
        metric_point = {
            'timestamp': time.time(),
            'type': 'quantum_operation',
            'operation_type': operation_type,
            'success': success,
            'duration': duration
        }
        self.metrics_buffer.append(metric_point)
    
    def record_bms_status(self, protocol: str, connected: bool):
        """Record BMS connectivity status."""
        if self.prometheus_available:
            self.bms_connectivity.labels(protocol=protocol).set(1.0 if connected else 0.0)
        
        self.metrics_buffer.append({
            'timestamp': time.time(),
            'type': 'bms_status',
            'protocol': protocol,
            'connected': connected
        })
    
    def record_building_metrics(self, building_id: str, zone_data: Dict[str, Dict]):
        """Record building temperature and energy metrics."""
        for zone_id, data in zone_data.items():
            temp = data.get('temperature', 0.0)
            power = data.get('power_kw', 0.0)
            
            if self.prometheus_available:
                self.temperature_control.labels(building=building_id, zone=zone_id).set(temp)
                self.energy_consumption.labels(building=building_id, zone=zone_id).set(power)
            
            self.metrics_buffer.append({
                'timestamp': time.time(),
                'type': 'building_metrics',
                'building': building_id,
                'zone': zone_id,
                'temperature': temp,
                'power_kw': power
            })
    
    def get_recent_metrics(self, metric_type: str = None, minutes: int = 60) -> List[Dict]:
        """Get recent metrics, optionally filtered by type."""
        cutoff_time = time.time() - (minutes * 60)
        
        filtered_metrics = [
            m for m in self.metrics_buffer 
            if m['timestamp'] >= cutoff_time
        ]
        
        if metric_type:
            filtered_metrics = [m for m in filtered_metrics if m.get('type') == metric_type]
        
        return filtered_metrics


class AlertManager:
    """Advanced alerting system with multiple notification channels."""
    
    def __init__(self):
        self.logger = logging.getLogger("alert_manager")
        self.active_alerts: Dict[str, Dict] = {}
        self.alert_history: deque = deque(maxlen=1000)
        self.notification_channels: List[Callable] = []
    
    def add_notification_channel(self, channel_func: Callable):
        """Add notification channel (email, Slack, webhook, etc.)."""
        self.notification_channels.append(channel_func)
        self.logger.info(f"Added notification channel: {channel_func.__name__}")
    
    async def trigger_alert(
        self,
        alert_id: str,
        severity: str,
        message: str,
        details: Dict = None
    ):
        """Trigger an alert with notification."""
        alert_data = {
            'id': alert_id,
            'severity': severity,
            'message': message,
            'details': details or {},
            'timestamp': time.time(),
            'status': 'active'
        }
        
        # Store active alert
        self.active_alerts[alert_id] = alert_data
        self.alert_history.append(alert_data.copy())
        
        # Log the alert
        log_level = {
            'info': logging.INFO,
            'warning': logging.WARNING,
            'critical': logging.CRITICAL,
            'error': logging.ERROR
        }.get(severity.lower(), logging.WARNING)
        
        self.logger.log(log_level, f"ALERT [{severity.upper()}] {alert_id}: {message}")
        
        # Send notifications
        for channel in self.notification_channels:
            try:
                if asyncio.iscoroutinefunction(channel):
                    await channel(alert_data)
                else:
                    channel(alert_data)
            except Exception as e:
                self.logger.error(f"Notification channel error: {e}")
    
    async def resolve_alert(self, alert_id: str):
        """Resolve an active alert."""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert['status'] = 'resolved'
            alert['resolved_at'] = time.time()
            
            # Move to history and remove from active
            self.alert_history.append(alert)
            del self.active_alerts[alert_id]
            
            self.logger.info(f"RESOLVED: Alert {alert_id}")
    
    def get_active_alerts(self, severity: str = None) -> List[Dict]:
        """Get currently active alerts."""
        alerts = list(self.active_alerts.values())
        
        if severity:
            alerts = [a for a in alerts if a['severity'].lower() == severity.lower()]
        
        return alerts
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get alert summary statistics."""
        recent_alerts = [
            a for a in self.alert_history 
            if a['timestamp'] > time.time() - 3600  # Last hour
        ]
        
        severity_counts = defaultdict(int)
        for alert in recent_alerts:
            severity_counts[alert['severity']] += 1
        
        return {
            'active_alerts': len(self.active_alerts),
            'alerts_last_hour': len(recent_alerts),
            'severity_breakdown': dict(severity_counts),
            'most_recent_alert': max(recent_alerts, key=lambda x: x['timestamp']) if recent_alerts else None
        }


# Notification channel implementations
async def slack_notification_channel(alert_data: Dict):
    """Send alert to Slack (placeholder implementation)."""
    # In production, this would use Slack API
    print(f"SLACK ALERT: [{alert_data['severity']}] {alert_data['message']}")


async def email_notification_channel(alert_data: Dict):
    """Send alert via email (placeholder implementation)."""
    # In production, this would use SMTP
    print(f"EMAIL ALERT: [{alert_data['severity']}] {alert_data['message']}")


async def webhook_notification_channel(alert_data: Dict):
    """Send alert to webhook endpoint (placeholder implementation)."""
    # In production, this would make HTTP POST request
    print(f"WEBHOOK ALERT: {json.dumps(alert_data, indent=2)}")


# Global enhanced monitoring instances
_global_advanced_metrics = AdvancedMetricsCollector()
_global_alert_manager = AlertManager()


def get_advanced_metrics() -> AdvancedMetricsCollector:
    """Get global advanced metrics collector."""
    return _global_advanced_metrics


def get_alert_manager() -> AlertManager:
    """Get global alert manager."""
    return _global_alert_manager
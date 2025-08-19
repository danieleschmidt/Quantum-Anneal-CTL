"""
Advanced health monitoring system for quantum HVAC operations.
Provides comprehensive system health checks, performance monitoring, and predictive maintenance.
"""

import asyncio
import time
import threading
import psutil
import logging
from typing import Any, Dict, List, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import statistics
import numpy as np
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class ComponentType(Enum):
    """Types of system components."""
    QUANTUM_SOLVER = "quantum_solver"
    OPTIMIZER = "optimizer"
    BMS_CONNECTOR = "bms_connector"
    WEATHER_SERVICE = "weather_service"
    DATABASE = "database"
    API_SERVER = "api_server"
    CONTROL_LOOP = "control_loop"
    SECURITY_MONITOR = "security_monitor"
    SYSTEM = "system"


@dataclass
class HealthMetric:
    """Individual health metric."""
    name: str
    value: float
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None
    unit: Optional[str] = None
    description: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class ComponentHealth:
    """Health status of a system component."""
    component_id: str
    component_type: ComponentType
    status: HealthStatus
    metrics: Dict[str, HealthMetric]
    last_check: float
    uptime: float
    error_count: int = 0
    warning_count: int = 0
    status_history: List[Tuple[float, HealthStatus]] = field(default_factory=list)
    custom_data: Dict[str, Any] = field(default_factory=dict)


class HealthChecker:
    """Base class for component-specific health checkers."""
    
    def __init__(self, component_id: str, component_type: ComponentType):
        self.component_id = component_id
        self.component_type = component_type
    
    async def check_health(self) -> ComponentHealth:
        """Perform health check for this component."""
        raise NotImplementedError("Subclasses must implement check_health")


class SystemHealthChecker(HealthChecker):
    """Health checker for system-level metrics."""
    
    def __init__(self):
        super().__init__("system", ComponentType.SYSTEM)
        self.boot_time = psutil.boot_time()
    
    async def check_health(self) -> ComponentHealth:
        """Check system-level health metrics."""
        metrics = {}
        
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        metrics["cpu_usage"] = HealthMetric(
            name="cpu_usage",
            value=cpu_percent,
            threshold_warning=80.0,
            threshold_critical=95.0,
            unit="percent",
            description="CPU utilization percentage"
        )
        
        # Memory usage
        memory = psutil.virtual_memory()
        metrics["memory_usage"] = HealthMetric(
            name="memory_usage",
            value=memory.percent,
            threshold_warning=80.0,
            threshold_critical=95.0,
            unit="percent",
            description="Memory utilization percentage"
        )
        
        # Disk usage
        disk = psutil.disk_usage('/')
        disk_percent = (disk.used / disk.total) * 100
        metrics["disk_usage"] = HealthMetric(
            name="disk_usage",
            value=disk_percent,
            threshold_warning=80.0,
            threshold_critical=95.0,
            unit="percent",
            description="Disk utilization percentage"
        )
        
        # Load average (on Unix systems)
        try:
            load_avg = psutil.getloadavg()[0]  # 1-minute load average
            cpu_count = psutil.cpu_count()
            load_percent = (load_avg / cpu_count) * 100
            
            metrics["load_average"] = HealthMetric(
                name="load_average",
                value=load_percent,
                threshold_warning=70.0,
                threshold_critical=90.0,
                unit="percent",
                description="System load average (1min)"
            )
        except AttributeError:
            # getloadavg not available on Windows
            pass
        
        # Network connections
        try:
            connections = psutil.net_connections()
            active_connections = len([c for c in connections if c.status == 'ESTABLISHED'])
            
            metrics["active_connections"] = HealthMetric(
                name="active_connections",
                value=active_connections,
                threshold_warning=1000,
                threshold_critical=2000,
                unit="count",
                description="Number of active network connections"
            )
        except psutil.AccessDenied:
            # May require elevated privileges
            pass
        
        # Process count
        process_count = len(psutil.pids())
        metrics["process_count"] = HealthMetric(
            name="process_count",
            value=process_count,
            threshold_warning=500,
            threshold_critical=1000,
            unit="count",
            description="Number of running processes"
        )
        
        # Determine overall health status
        status = self._determine_status(metrics)
        
        return ComponentHealth(
            component_id=self.component_id,
            component_type=self.component_type,
            status=status,
            metrics=metrics,
            last_check=time.time(),
            uptime=time.time() - self.boot_time
        )
    
    def _determine_status(self, metrics: Dict[str, HealthMetric]) -> HealthStatus:
        """Determine overall health status from metrics."""
        critical_violations = 0
        warning_violations = 0
        
        for metric in metrics.values():
            if metric.threshold_critical and metric.value >= metric.threshold_critical:
                critical_violations += 1
            elif metric.threshold_warning and metric.value >= metric.threshold_warning:
                warning_violations += 1
        
        if critical_violations > 0:
            return HealthStatus.CRITICAL
        elif warning_violations >= 2:
            return HealthStatus.DEGRADED
        elif warning_violations > 0:
            return HealthStatus.WARNING
        else:
            return HealthStatus.HEALTHY


class QuantumSolverHealthChecker(HealthChecker):
    """Health checker for quantum solver components."""
    
    def __init__(self, solver_instance=None):
        super().__init__("quantum_solver", ComponentType.QUANTUM_SOLVER)
        self.solver_instance = solver_instance
        self.last_successful_solve = None
        self.solve_count = 0
        self.failure_count = 0
    
    async def check_health(self) -> ComponentHealth:
        """Check quantum solver health."""
        metrics = {}
        
        # Solver availability
        solver_available = await self._check_solver_availability()
        metrics["solver_available"] = HealthMetric(
            name="solver_available",
            value=1.0 if solver_available else 0.0,
            threshold_critical=0.5,
            unit="boolean",
            description="Quantum solver availability"
        )
        
        # Solve success rate
        if self.solve_count > 0:
            success_rate = (self.solve_count - self.failure_count) / self.solve_count
            metrics["success_rate"] = HealthMetric(
                name="success_rate",
                value=success_rate * 100,
                threshold_warning=90.0,
                threshold_critical=70.0,
                unit="percent",
                description="Quantum solve success rate"
            )
        
        # Time since last successful solve
        if self.last_successful_solve:
            time_since_success = time.time() - self.last_successful_solve
            metrics["time_since_success"] = HealthMetric(
                name="time_since_success",
                value=time_since_success,
                threshold_warning=3600,  # 1 hour
                threshold_critical=7200,  # 2 hours
                unit="seconds",
                description="Time since last successful quantum solve"
            )
        
        # Queue length (if available)
        try:
            queue_length = await self._get_solver_queue_length()
            if queue_length is not None:
                metrics["queue_length"] = HealthMetric(
                    name="queue_length",
                    value=queue_length,
                    threshold_warning=10,
                    threshold_critical=25,
                    unit="jobs",
                    description="Quantum solver queue length"
                )
        except Exception as e:
            logger.warning(f"Could not get solver queue length: {e}")
        
        # Determine status
        status = self._determine_status(metrics)
        
        return ComponentHealth(
            component_id=self.component_id,
            component_type=self.component_type,
            status=status,
            metrics=metrics,
            last_check=time.time(),
            uptime=time.time() - (self.last_successful_solve or time.time()),
            custom_data={
                "solve_count": self.solve_count,
                "failure_count": self.failure_count
            }
        )
    
    async def _check_solver_availability(self) -> bool:
        """Check if quantum solver is available."""
        try:
            # Simple connectivity test
            if hasattr(self.solver_instance, 'ping'):
                await self.solver_instance.ping()
                return True
            elif hasattr(self.solver_instance, 'test_connection'):
                return await self.solver_instance.test_connection()
            else:
                # Assume available if we can't test
                return True
        except Exception as e:
            logger.warning(f"Quantum solver availability check failed: {e}")
            return False
    
    async def _get_solver_queue_length(self) -> Optional[int]:
        """Get current solver queue length."""
        try:
            if hasattr(self.solver_instance, 'get_queue_length'):
                return await self.solver_instance.get_queue_length()
        except Exception:
            pass
        return None
    
    def record_solve_attempt(self, success: bool):
        """Record a solve attempt for health tracking."""
        self.solve_count += 1
        if success:
            self.last_successful_solve = time.time()
        else:
            self.failure_count += 1
    
    def _determine_status(self, metrics: Dict[str, HealthMetric]) -> HealthStatus:
        """Determine quantum solver health status."""
        if "solver_available" in metrics and metrics["solver_available"].value < 0.5:
            return HealthStatus.CRITICAL
        
        critical_count = sum(1 for m in metrics.values() 
                           if m.threshold_critical and m.value >= m.threshold_critical)
        warning_count = sum(1 for m in metrics.values() 
                          if m.threshold_warning and m.value >= m.threshold_warning)
        
        if critical_count > 0:
            return HealthStatus.CRITICAL
        elif warning_count > 1:
            return HealthStatus.DEGRADED
        elif warning_count > 0:
            return HealthStatus.WARNING
        else:
            return HealthStatus.HEALTHY


class HealthMonitoringSystem:
    """Comprehensive health monitoring system."""
    
    def __init__(self, check_interval: int = 60):
        self.check_interval = check_interval
        self.health_checkers: Dict[str, HealthChecker] = {}
        self.component_health: Dict[str, ComponentHealth] = {}
        self.health_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.monitoring_task: Optional[asyncio.Task] = None
        self.running = False
        
        # Health change callbacks
        self.status_change_callbacks: List[Callable] = []
        
        # Predictive analytics
        self.trend_analyzer = HealthTrendAnalyzer()
        
        # Add default system health checker
        self.add_health_checker(SystemHealthChecker())
    
    def add_health_checker(self, checker: HealthChecker):
        """Add a health checker for a component."""
        self.health_checkers[checker.component_id] = checker
        logger.info(f"Added health checker for {checker.component_id}")
    
    def remove_health_checker(self, component_id: str):
        """Remove a health checker."""
        if component_id in self.health_checkers:
            del self.health_checkers[component_id]
            if component_id in self.component_health:
                del self.component_health[component_id]
            logger.info(f"Removed health checker for {component_id}")
    
    def add_status_change_callback(self, callback: Callable):
        """Add callback for health status changes."""
        self.status_change_callbacks.append(callback)
    
    async def start_monitoring(self):
        """Start continuous health monitoring."""
        if self.running:
            logger.warning("Health monitoring already running")
            return
        
        self.running = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Started health monitoring system")
    
    async def stop_monitoring(self):
        """Stop health monitoring."""
        self.running = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        logger.info("Stopped health monitoring system")
    
    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                await self._check_all_components()
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health monitoring loop: {e}")
                await asyncio.sleep(5)  # Short delay before retrying
    
    async def _check_all_components(self):
        """Check health of all registered components."""
        check_tasks = []
        
        for component_id, checker in self.health_checkers.items():
            task = asyncio.create_task(self._check_component(component_id, checker))
            check_tasks.append(task)
        
        # Wait for all checks to complete
        if check_tasks:
            results = await asyncio.gather(*check_tasks, return_exceptions=True)
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    component_id = list(self.health_checkers.keys())[i]
                    logger.error(f"Health check failed for {component_id}: {result}")
    
    async def _check_component(self, component_id: str, checker: HealthChecker):
        """Check health of a specific component."""
        try:
            start_time = time.time()
            health = await checker.check_health()
            check_duration = time.time() - start_time
            
            # Add check duration metric
            health.metrics["check_duration"] = HealthMetric(
                name="check_duration",
                value=check_duration,
                threshold_warning=10.0,  # 10 seconds
                threshold_critical=30.0,  # 30 seconds
                unit="seconds",
                description="Health check execution time"
            )
            
            # Check for status changes
            previous_health = self.component_health.get(component_id)
            if previous_health and previous_health.status != health.status:
                await self._handle_status_change(component_id, previous_health.status, health.status)
            
            # Update component health
            self.component_health[component_id] = health
            
            # Add to history
            self.health_history[component_id].append(health)
            
            # Update trend analyzer
            self.trend_analyzer.add_health_data(component_id, health)
            
            logger.debug(f"Health check completed for {component_id}: {health.status.value}")
            
        except Exception as e:
            logger.error(f"Health check error for {component_id}: {e}")
            
            # Create error health status
            error_health = ComponentHealth(
                component_id=component_id,
                component_type=checker.component_type,
                status=HealthStatus.UNKNOWN,
                metrics={"error": HealthMetric("error", 1.0, description=str(e))},
                last_check=time.time(),
                uptime=0.0
            )
            
            self.component_health[component_id] = error_health
    
    async def _handle_status_change(self, component_id: str, old_status: HealthStatus, new_status: HealthStatus):
        """Handle component status changes."""
        logger.info(f"Component {component_id} status changed: {old_status.value} -> {new_status.value}")
        
        # Call registered callbacks
        for callback in self.status_change_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(component_id, old_status, new_status)
                else:
                    callback(component_id, old_status, new_status)
            except Exception as e:
                logger.error(f"Error in status change callback: {e}")
    
    def get_overall_health(self) -> HealthStatus:
        """Get overall system health status."""
        if not self.component_health:
            return HealthStatus.UNKNOWN
        
        statuses = [health.status for health in self.component_health.values()]
        
        # Overall health is the worst component status
        if HealthStatus.CRITICAL in statuses:
            return HealthStatus.CRITICAL
        elif HealthStatus.DEGRADED in statuses:
            return HealthStatus.DEGRADED
        elif HealthStatus.WARNING in statuses:
            return HealthStatus.WARNING
        else:
            return HealthStatus.HEALTHY
    
    def get_component_health(self, component_id: str) -> Optional[ComponentHealth]:
        """Get health status for a specific component."""
        return self.component_health.get(component_id)
    
    def get_all_health(self) -> Dict[str, ComponentHealth]:
        """Get health status for all components."""
        return self.component_health.copy()
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get a summary of system health."""
        summary = {
            "overall_status": self.get_overall_health().value,
            "timestamp": time.time(),
            "component_count": len(self.component_health),
            "components": {}
        }
        
        status_counts = defaultdict(int)
        
        for component_id, health in self.component_health.items():
            status_counts[health.status.value] += 1
            
            summary["components"][component_id] = {
                "status": health.status.value,
                "type": health.component_type.value,
                "uptime": health.uptime,
                "last_check": health.last_check,
                "error_count": health.error_count,
                "warning_count": health.warning_count,
                "critical_metrics": [
                    name for name, metric in health.metrics.items()
                    if metric.threshold_critical and metric.value >= metric.threshold_critical
                ]
            }
        
        summary["status_distribution"] = dict(status_counts)
        
        # Add trend predictions
        predictions = self.trend_analyzer.get_predictions()
        if predictions:
            summary["predictions"] = predictions
        
        return summary


class HealthTrendAnalyzer:
    """Analyze health trends for predictive maintenance."""
    
    def __init__(self, history_length: int = 100):
        self.history_length = history_length
        self.component_trends: Dict[str, Dict[str, deque]] = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=history_length))
        )
    
    def add_health_data(self, component_id: str, health: ComponentHealth):
        """Add health data point for trend analysis."""
        timestamp = health.last_check
        
        # Store metric values with timestamps
        for metric_name, metric in health.metrics.items():
            self.component_trends[component_id][metric_name].append((timestamp, metric.value))
    
    def get_trend(self, component_id: str, metric_name: str, window_minutes: int = 60) -> Optional[float]:
        """Get trend slope for a specific metric."""
        if component_id not in self.component_trends:
            return None
        
        if metric_name not in self.component_trends[component_id]:
            return None
        
        data = self.component_trends[component_id][metric_name]
        if len(data) < 10:  # Need at least 10 data points
            return None
        
        # Filter data to specified window
        current_time = time.time()
        window_start = current_time - (window_minutes * 60)
        
        windowed_data = [(t, v) for t, v in data if t >= window_start]
        if len(windowed_data) < 5:
            return None
        
        # Calculate trend slope using linear regression
        timestamps, values = zip(*windowed_data)
        
        # Normalize timestamps to start from 0
        min_timestamp = min(timestamps)
        normalized_times = [(t - min_timestamp) for t in timestamps]
        
        try:
            # Simple linear regression
            n = len(normalized_times)
            sum_x = sum(normalized_times)
            sum_y = sum(values)
            sum_xy = sum(x * y for x, y in zip(normalized_times, values))
            sum_x2 = sum(x * x for x in normalized_times)
            
            if n * sum_x2 - sum_x * sum_x == 0:
                return 0.0
            
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
            return slope
        except (ValueError, ZeroDivisionError):
            return None
    
    def get_predictions(self, prediction_minutes: int = 30) -> Dict[str, Any]:
        """Get predictive health warnings based on trends."""
        predictions = {}
        
        for component_id in self.component_trends:
            component_predictions = {}
            
            for metric_name in self.component_trends[component_id]:
                trend = self.get_trend(component_id, metric_name, window_minutes=60)
                
                if trend is not None and abs(trend) > 1e-6:  # Significant trend
                    # Get latest value
                    latest_data = self.component_trends[component_id][metric_name]
                    if latest_data:
                        latest_value = latest_data[-1][1]
                        
                        # Predict value after prediction_minutes
                        predicted_value = latest_value + (trend * prediction_minutes * 60)
                        
                        component_predictions[metric_name] = {
                            "current_value": latest_value,
                            "trend_slope": trend,
                            "predicted_value": predicted_value,
                            "trend_direction": "increasing" if trend > 0 else "decreasing"
                        }
            
            if component_predictions:
                predictions[component_id] = component_predictions
        
        return predictions
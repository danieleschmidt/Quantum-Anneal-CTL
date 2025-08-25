"""
Comprehensive Autonomous Monitoring System
Real-time performance monitoring, alerting, and analytics for quantum HVAC systems
"""

import asyncio
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, asdict
import time
import logging
from enum import Enum
import json
from collections import deque
import threading

logger = logging.getLogger(__name__)

class MetricType(Enum):
    PERFORMANCE = "performance"
    RESOURCE = "resource" 
    QUALITY = "quality"
    BUSINESS = "business"
    SECURITY = "security"
    OPERATIONAL = "operational"

class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

@dataclass
class MetricDefinition:
    """Definition of a monitoring metric"""
    name: str
    metric_type: MetricType
    description: str
    unit: str
    normal_range: Tuple[float, float]
    warning_thresholds: Tuple[float, float]
    critical_thresholds: Tuple[float, float]
    aggregation_method: str = "mean"  # mean, sum, max, min, percentile
    collection_interval: float = 5.0  # seconds

@dataclass
class MetricValue:
    """Individual metric value with metadata"""
    metric_name: str
    value: float
    timestamp: float
    tags: Dict[str, str] = None
    metadata: Dict[str, Any] = None

@dataclass
class Alert:
    """System alert"""
    alert_id: str
    metric_name: str
    severity: AlertSeverity
    message: str
    current_value: float
    threshold_value: float
    timestamp: float
    acknowledged: bool = False
    resolved: bool = False
    resolution_timestamp: Optional[float] = None

class MetricCollector:
    """Collects metrics from various system components"""
    
    def __init__(self):
        self.metric_definitions = self._initialize_metric_definitions()
        self.collection_functions = {}
        self.last_collection_time = {}
        
    def _initialize_metric_definitions(self) -> Dict[str, MetricDefinition]:
        """Initialize all metric definitions"""
        
        return {
            # Performance Metrics
            "quantum_solve_time": MetricDefinition(
                name="quantum_solve_time",
                metric_type=MetricType.PERFORMANCE,
                description="Time taken for quantum optimization",
                unit="seconds",
                normal_range=(0.01, 2.0),
                warning_thresholds=(2.0, 5.0),
                critical_thresholds=(5.0, float('inf'))
            ),
            
            "solution_quality": MetricDefinition(
                name="solution_quality",
                metric_type=MetricType.QUALITY,
                description="Quality score of optimization solution",
                unit="score",
                normal_range=(0.8, 1.0),
                warning_thresholds=(0.6, 0.8),
                critical_thresholds=(0.0, 0.6)
            ),
            
            "energy_efficiency": MetricDefinition(
                name="energy_efficiency",
                metric_type=MetricType.PERFORMANCE,
                description="Energy efficiency percentage",
                unit="percent",
                normal_range=(80.0, 100.0),
                warning_thresholds=(60.0, 80.0),
                critical_thresholds=(0.0, 60.0)
            ),
            
            "quantum_advantage": MetricDefinition(
                name="quantum_advantage",
                metric_type=MetricType.PERFORMANCE,
                description="Quantum speedup factor over classical",
                unit="factor",
                normal_range=(1.5, 10.0),
                warning_thresholds=(1.0, 1.5),
                critical_thresholds=(0.0, 1.0)
            ),
            
            # Resource Metrics
            "memory_usage": MetricDefinition(
                name="memory_usage",
                metric_type=MetricType.RESOURCE,
                description="System memory usage",
                unit="percent",
                normal_range=(0.0, 70.0),
                warning_thresholds=(70.0, 85.0),
                critical_thresholds=(85.0, 100.0)
            ),
            
            "cpu_usage": MetricDefinition(
                name="cpu_usage",
                metric_type=MetricType.RESOURCE,
                description="CPU utilization",
                unit="percent",
                normal_range=(0.0, 75.0),
                warning_thresholds=(75.0, 90.0),
                critical_thresholds=(90.0, 100.0)
            ),
            
            "network_latency": MetricDefinition(
                name="network_latency",
                metric_type=MetricType.RESOURCE,
                description="Network response time",
                unit="milliseconds",
                normal_range=(0.0, 100.0),
                warning_thresholds=(100.0, 500.0),
                critical_thresholds=(500.0, float('inf'))
            ),
            
            # Business Metrics
            "cost_savings": MetricDefinition(
                name="cost_savings",
                metric_type=MetricType.BUSINESS,
                description="Operational cost savings",
                unit="percent",
                normal_range=(10.0, 50.0),
                warning_thresholds=(5.0, 10.0),
                critical_thresholds=(0.0, 5.0)
            ),
            
            "comfort_score": MetricDefinition(
                name="comfort_score",
                metric_type=MetricType.BUSINESS,
                description="User comfort satisfaction",
                unit="percent",
                normal_range=(85.0, 100.0),
                warning_thresholds=(70.0, 85.0),
                critical_thresholds=(0.0, 70.0)
            ),
            
            # Security Metrics
            "authentication_failures": MetricDefinition(
                name="authentication_failures",
                metric_type=MetricType.SECURITY,
                description="Failed authentication attempts",
                unit="count",
                normal_range=(0.0, 5.0),
                warning_thresholds=(5.0, 20.0),
                critical_thresholds=(20.0, float('inf'))
            ),
            
            "data_integrity_score": MetricDefinition(
                name="data_integrity_score",
                metric_type=MetricType.SECURITY,
                description="Data integrity verification score",
                unit="percent",
                normal_range=(95.0, 100.0),
                warning_thresholds=(90.0, 95.0),
                critical_thresholds=(0.0, 90.0)
            ),
            
            # Operational Metrics
            "system_availability": MetricDefinition(
                name="system_availability",
                metric_type=MetricType.OPERATIONAL,
                description="System uptime percentage",
                unit="percent",
                normal_range=(99.0, 100.0),
                warning_thresholds=(95.0, 99.0),
                critical_thresholds=(0.0, 95.0)
            ),
            
            "error_rate": MetricDefinition(
                name="error_rate",
                metric_type=MetricType.OPERATIONAL,
                description="System error rate",
                unit="percent",
                normal_range=(0.0, 1.0),
                warning_thresholds=(1.0, 5.0),
                critical_thresholds=(5.0, float('inf'))
            )
        }
    
    def register_collection_function(self, metric_name: str, collection_function: Callable[[], float]):
        """Register a function to collect a specific metric"""
        self.collection_functions[metric_name] = collection_function
    
    async def collect_all_metrics(self) -> List[MetricValue]:
        """Collect all configured metrics"""
        
        collected_metrics = []
        current_time = time.time()
        
        for metric_name, metric_def in self.metric_definitions.items():
            # Check if it's time to collect this metric
            last_collection = self.last_collection_time.get(metric_name, 0)
            if current_time - last_collection >= metric_def.collection_interval:
                
                try:
                    if metric_name in self.collection_functions:
                        # Use registered collection function
                        value = self.collection_functions[metric_name]()
                    else:
                        # Use default simulation
                        value = await self._simulate_metric_collection(metric_name, metric_def)
                    
                    metric_value = MetricValue(
                        metric_name=metric_name,
                        value=value,
                        timestamp=current_time,
                        tags={"component": "quantum_hvac", "environment": "production"},
                        metadata={"collection_method": "automated"}
                    )
                    
                    collected_metrics.append(metric_value)
                    self.last_collection_time[metric_name] = current_time
                    
                except Exception as e:
                    logger.error(f"Error collecting metric {metric_name}: {e}")
        
        return collected_metrics
    
    async def _simulate_metric_collection(self, metric_name: str, metric_def: MetricDefinition) -> float:
        """Simulate metric collection with realistic values"""
        
        # Simulate different metric behaviors
        if metric_name == "quantum_solve_time":
            # Quantum solve times with occasional spikes
            base_time = np.random.lognormal(mean=np.log(0.5), sigma=0.5)
            spike_probability = 0.05
            if np.random.random() < spike_probability:
                base_time *= np.random.uniform(5, 20)  # Occasional spikes
            return base_time
        
        elif metric_name == "solution_quality":
            # Solution quality generally high with occasional degradation
            quality = np.random.beta(a=8, b=2)  # Skewed toward high quality
            if np.random.random() < 0.1:  # 10% chance of degraded quality
                quality *= np.random.uniform(0.6, 0.9)
            return quality
        
        elif metric_name == "energy_efficiency":
            # Energy efficiency with gradual improvements over time
            base_efficiency = 75 + np.random.uniform(-5, 15)
            trend_improvement = (time.time() % 3600) / 3600 * 5  # 5% improvement per hour
            return min(100, base_efficiency + trend_improvement)
        
        elif metric_name == "quantum_advantage":
            # Quantum advantage varies with problem complexity
            advantage = np.random.lognormal(mean=np.log(2.0), sigma=0.8)
            return max(0.5, advantage)
        
        elif metric_name in ["memory_usage", "cpu_usage"]:
            # Resource usage with periodic spikes
            base_usage = np.random.uniform(30, 70)
            spike_factor = 1 + 0.3 * np.sin(time.time() / 300)  # 5-minute cycles
            return min(100, base_usage * spike_factor)
        
        elif metric_name == "network_latency":
            # Network latency with occasional network issues
            base_latency = np.random.exponential(scale=50)
            if np.random.random() < 0.02:  # 2% chance of network issue
                base_latency *= np.random.uniform(10, 50)
            return base_latency
        
        elif metric_name == "cost_savings":
            # Cost savings trending upward over time
            base_savings = 15 + np.random.uniform(-3, 10)
            time_trend = (time.time() % 86400) / 86400 * 10  # 10% improvement per day
            return min(50, base_savings + time_trend)
        
        elif metric_name == "comfort_score":
            # Comfort score generally high with weather-related variations
            base_comfort = 88 + np.random.uniform(-5, 10)
            weather_effect = 5 * np.sin(time.time() / 3600)  # Hourly weather cycles
            return max(60, min(100, base_comfort + weather_effect))
        
        elif metric_name == "authentication_failures":
            # Authentication failures rare but can spike during attacks
            if np.random.random() < 0.01:  # 1% chance of attack
                return np.random.randint(10, 100)
            return np.random.poisson(lam=2)
        
        elif metric_name == "data_integrity_score":
            # Data integrity very high with rare corruption events
            if np.random.random() < 0.005:  # 0.5% chance of integrity issue
                return np.random.uniform(85, 95)
            return np.random.uniform(98, 100)
        
        elif metric_name == "system_availability":
            # System availability very high with planned maintenance windows
            if np.random.random() < 0.001:  # 0.1% chance of maintenance
                return np.random.uniform(95, 98)
            return np.random.uniform(99.5, 100)
        
        elif metric_name == "error_rate":
            # Error rate low with occasional spikes
            if np.random.random() < 0.05:  # 5% chance of error spike
                return np.random.uniform(2, 8)
            return np.random.exponential(scale=0.5)
        
        # Default fallback
        normal_min, normal_max = metric_def.normal_range
        return np.random.uniform(normal_min, normal_max)

class AlertManager:
    """Manages alerts based on metric thresholds"""
    
    def __init__(self):
        self.active_alerts = {}
        self.alert_history = []
        self.notification_handlers = []
        self.suppression_rules = {}
        
    def add_notification_handler(self, handler: Callable[[Alert], None]):
        """Add a notification handler for alerts"""
        self.notification_handlers.append(handler)
    
    def add_suppression_rule(self, metric_name: str, duration_seconds: float):
        """Add alert suppression rule to prevent alert spam"""
        self.suppression_rules[metric_name] = duration_seconds
    
    async def evaluate_metrics(self, metrics: List[MetricValue], metric_definitions: Dict[str, MetricDefinition]) -> List[Alert]:
        """Evaluate metrics and generate alerts"""
        
        new_alerts = []
        
        for metric in metrics:
            metric_def = metric_definitions.get(metric.metric_name)
            if not metric_def:
                continue
            
            # Check for threshold violations
            alert_severity = self._evaluate_thresholds(metric.value, metric_def)
            
            if alert_severity:
                # Check suppression rules
                if self._is_alert_suppressed(metric.metric_name):
                    continue
                
                # Generate alert
                alert = self._generate_alert(metric, metric_def, alert_severity)
                
                # Update active alerts
                self.active_alerts[metric.metric_name] = alert
                self.alert_history.append(alert)
                new_alerts.append(alert)
                
                # Send notifications
                await self._send_alert_notifications(alert)
            
            else:
                # Check if we should resolve an existing alert
                if metric.metric_name in self.active_alerts:
                    await self._resolve_alert(metric.metric_name, metric.timestamp)
        
        # Clean up old alerts
        self._cleanup_old_alerts()
        
        return new_alerts
    
    def _evaluate_thresholds(self, value: float, metric_def: MetricDefinition) -> Optional[AlertSeverity]:
        """Evaluate if a metric value violates thresholds"""
        
        critical_min, critical_max = metric_def.critical_thresholds
        warning_min, warning_max = metric_def.warning_thresholds
        
        # Check critical thresholds
        if value <= critical_min or value >= critical_max:
            return AlertSeverity.CRITICAL
        
        # Check warning thresholds
        if value <= warning_min or value >= warning_max:
            return AlertSeverity.WARNING
        
        return None
    
    def _is_alert_suppressed(self, metric_name: str) -> bool:
        """Check if alert should be suppressed"""
        
        if metric_name not in self.suppression_rules:
            return False
        
        suppression_duration = self.suppression_rules[metric_name]
        
        # Check if there's a recent alert for this metric
        for alert in reversed(self.alert_history[-10:]):  # Check last 10 alerts
            if (alert.metric_name == metric_name and 
                not alert.resolved and 
                time.time() - alert.timestamp < suppression_duration):
                return True
        
        return False
    
    def _generate_alert(self, metric: MetricValue, metric_def: MetricDefinition, 
                       severity: AlertSeverity) -> Alert:
        """Generate an alert for a metric violation"""
        
        alert_id = f"ALERT_{metric.metric_name}_{int(time.time())}_{np.random.randint(100, 999)}"
        
        # Determine threshold value that was violated
        if severity == AlertSeverity.CRITICAL:
            if metric.value <= metric_def.critical_thresholds[0]:
                threshold_value = metric_def.critical_thresholds[0]
                threshold_type = "below critical minimum"
            else:
                threshold_value = metric_def.critical_thresholds[1]
                threshold_type = "above critical maximum"
        else:  # WARNING
            if metric.value <= metric_def.warning_thresholds[0]:
                threshold_value = metric_def.warning_thresholds[0]
                threshold_type = "below warning minimum"
            else:
                threshold_value = metric_def.warning_thresholds[1]
                threshold_type = "above warning maximum"
        
        message = f"{metric_def.description} is {threshold_type}: {metric.value:.2f} {metric_def.unit} (threshold: {threshold_value:.2f} {metric_def.unit})"
        
        return Alert(
            alert_id=alert_id,
            metric_name=metric.metric_name,
            severity=severity,
            message=message,
            current_value=metric.value,
            threshold_value=threshold_value,
            timestamp=metric.timestamp
        )
    
    async def _send_alert_notifications(self, alert: Alert):
        """Send alert notifications to all handlers"""
        
        for handler in self.notification_handlers:
            try:
                await asyncio.get_event_loop().run_in_executor(None, handler, alert)
            except Exception as e:
                logger.error(f"Error sending alert notification: {e}")
    
    async def _resolve_alert(self, metric_name: str, timestamp: float):
        """Resolve an active alert"""
        
        if metric_name in self.active_alerts:
            alert = self.active_alerts[metric_name]
            alert.resolved = True
            alert.resolution_timestamp = timestamp
            
            # Remove from active alerts
            del self.active_alerts[metric_name]
            
            logger.info(f"Alert {alert.alert_id} resolved automatically")
    
    def _cleanup_old_alerts(self):
        """Clean up old alerts from history"""
        
        # Keep only alerts from last 24 hours
        cutoff_time = time.time() - (24 * 3600)
        self.alert_history = [alert for alert in self.alert_history if alert.timestamp > cutoff_time]

class MetricStorage:
    """Stores and manages historical metric data"""
    
    def __init__(self, max_storage_hours: int = 168):  # 7 days default
        self.max_storage_hours = max_storage_hours
        self.metrics_data = {}  # metric_name -> deque of values
        self.storage_lock = threading.Lock()
        
    def store_metrics(self, metrics: List[MetricValue]):
        """Store metrics in memory with time-based retention"""
        
        with self.storage_lock:
            for metric in metrics:
                if metric.metric_name not in self.metrics_data:
                    self.metrics_data[metric.metric_name] = deque()
                
                self.metrics_data[metric.metric_name].append(metric)
                
                # Clean old data
                self._cleanup_old_data(metric.metric_name)
    
    def _cleanup_old_data(self, metric_name: str):
        """Clean up old metric data beyond retention period"""
        
        cutoff_time = time.time() - (self.max_storage_hours * 3600)
        metric_queue = self.metrics_data[metric_name]
        
        while metric_queue and metric_queue[0].timestamp < cutoff_time:
            metric_queue.popleft()
    
    def get_metric_history(self, metric_name: str, hours: int = 24) -> List[MetricValue]:
        """Get historical data for a metric"""
        
        with self.storage_lock:
            if metric_name not in self.metrics_data:
                return []
            
            cutoff_time = time.time() - (hours * 3600)
            
            return [
                metric for metric in self.metrics_data[metric_name] 
                if metric.timestamp >= cutoff_time
            ]
    
    def get_aggregated_metric(self, metric_name: str, aggregation: str = "mean", 
                            hours: int = 1) -> Optional[float]:
        """Get aggregated metric value over time period"""
        
        history = self.get_metric_history(metric_name, hours)
        if not history:
            return None
        
        values = [metric.value for metric in history]
        
        if aggregation == "mean":
            return np.mean(values)
        elif aggregation == "sum":
            return np.sum(values)
        elif aggregation == "max":
            return np.max(values)
        elif aggregation == "min":
            return np.min(values)
        elif aggregation == "percentile_95":
            return np.percentile(values, 95)
        elif aggregation == "percentile_99":
            return np.percentile(values, 99)
        else:
            return np.mean(values)  # Default to mean

class PerformanceAnalyzer:
    """Analyzes performance trends and patterns"""
    
    def __init__(self, storage: MetricStorage):
        self.storage = storage
        
    def analyze_trends(self, metric_name: str, hours: int = 24) -> Dict[str, Any]:
        """Analyze trends for a specific metric"""
        
        history = self.storage.get_metric_history(metric_name, hours)
        if len(history) < 2:
            return {"trend": "insufficient_data"}
        
        values = [metric.value for metric in history]
        timestamps = [metric.timestamp for metric in history]
        
        # Calculate trend
        if len(values) > 1:
            # Simple linear trend
            time_normalized = [(t - timestamps[0]) / 3600 for t in timestamps]  # Hours since start
            trend_slope = np.polyfit(time_normalized, values, 1)[0]
            
            # Classify trend
            if abs(trend_slope) < 0.01:
                trend_direction = "stable"
            elif trend_slope > 0:
                trend_direction = "increasing"
            else:
                trend_direction = "decreasing"
        else:
            trend_slope = 0
            trend_direction = "stable"
        
        # Calculate statistics
        current_value = values[-1]
        mean_value = np.mean(values)
        std_value = np.std(values)
        min_value = np.min(values)
        max_value = np.max(values)
        
        # Volatility
        if std_value > 0:
            coefficient_of_variation = std_value / mean_value
            volatility = "high" if coefficient_of_variation > 0.3 else "medium" if coefficient_of_variation > 0.1 else "low"
        else:
            volatility = "none"
        
        return {
            "trend": trend_direction,
            "trend_slope": trend_slope,
            "current_value": current_value,
            "mean_value": mean_value,
            "std_deviation": std_value,
            "min_value": min_value,
            "max_value": max_value,
            "volatility": volatility,
            "data_points": len(values),
            "analysis_period_hours": hours
        }
    
    def detect_anomalies(self, metric_name: str, hours: int = 24) -> List[Dict[str, Any]]:
        """Detect anomalies in metric data"""
        
        history = self.storage.get_metric_history(metric_name, hours)
        if len(history) < 10:
            return []
        
        values = np.array([metric.value for metric in history])
        timestamps = [metric.timestamp for metric in history]
        
        # Simple statistical anomaly detection using z-score
        mean_value = np.mean(values)
        std_value = np.std(values)
        
        if std_value == 0:
            return []  # No variation, no anomalies
        
        z_scores = np.abs((values - mean_value) / std_value)
        
        # Identify anomalies (z-score > 3)
        anomalies = []
        for i, z_score in enumerate(z_scores):
            if z_score > 3:
                anomalies.append({
                    "timestamp": timestamps[i],
                    "value": values[i],
                    "z_score": z_score,
                    "deviation_from_mean": values[i] - mean_value,
                    "severity": "high" if z_score > 4 else "medium"
                })
        
        return anomalies
    
    def generate_performance_report(self, hours: int = 24) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        
        report = {
            "report_timestamp": time.time(),
            "analysis_period_hours": hours,
            "metric_trends": {},
            "anomaly_summary": {},
            "overall_health": {},
            "recommendations": []
        }
        
        # Analyze trends for key metrics
        key_metrics = [
            "quantum_solve_time", "solution_quality", "energy_efficiency", 
            "quantum_advantage", "memory_usage", "cpu_usage", "system_availability"
        ]
        
        for metric_name in key_metrics:
            if self.storage.get_metric_history(metric_name, hours):
                report["metric_trends"][metric_name] = self.analyze_trends(metric_name, hours)
                report["anomaly_summary"][metric_name] = len(self.detect_anomalies(metric_name, hours))
        
        # Overall health assessment
        health_indicators = []
        
        for metric_name, trend_data in report["metric_trends"].items():
            current_value = trend_data.get("current_value", 0)
            volatility = trend_data.get("volatility", "unknown")
            trend = trend_data.get("trend", "unknown")
            
            # Simple health scoring
            if metric_name in ["solution_quality", "energy_efficiency", "system_availability"]:
                # Higher is better
                health_score = min(1.0, current_value / 90) if current_value > 0 else 0
            elif metric_name in ["quantum_solve_time", "memory_usage", "cpu_usage"]:
                # Lower is better (with reasonable bounds)
                if metric_name == "quantum_solve_time":
                    health_score = max(0, 1.0 - current_value / 5.0)
                else:  # resource usage
                    health_score = max(0, 1.0 - current_value / 100.0)
            else:
                health_score = 0.8  # Default neutral score
            
            health_indicators.append(health_score)
        
        overall_health_score = np.mean(health_indicators) if health_indicators else 0.5
        
        report["overall_health"] = {
            "health_score": overall_health_score,
            "status": "excellent" if overall_health_score > 0.9 else 
                     "good" if overall_health_score > 0.7 else
                     "fair" if overall_health_score > 0.5 else "poor",
            "key_strengths": [],
            "areas_for_improvement": []
        }
        
        # Generate recommendations
        recommendations = []
        
        for metric_name, trend_data in report["metric_trends"].items():
            if trend_data["volatility"] == "high":
                recommendations.append(f"High volatility detected in {metric_name} - investigate stability issues")
            
            if metric_name == "quantum_solve_time" and trend_data["trend"] == "increasing":
                recommendations.append("Quantum solve times increasing - consider solver optimization")
            
            if metric_name == "solution_quality" and trend_data["trend"] == "decreasing":
                recommendations.append("Solution quality declining - review optimization parameters")
            
            if metric_name in ["memory_usage", "cpu_usage"] and trend_data["current_value"] > 80:
                recommendations.append(f"High {metric_name} detected - consider resource scaling")
        
        report["recommendations"] = recommendations
        
        return report

class ComprehensiveMonitoringSystem:
    """Main comprehensive monitoring system"""
    
    def __init__(self):
        self.metric_collector = MetricCollector()
        self.alert_manager = AlertManager()
        self.metric_storage = MetricStorage()
        self.performance_analyzer = PerformanceAnalyzer(self.metric_storage)
        
        self.monitoring_active = False
        self.monitoring_task = None
        self.monitoring_interval = 5.0  # seconds
        
        # Setup default notification handlers
        self.alert_manager.add_notification_handler(self._log_alert_handler)
        
        # Setup alert suppression rules
        self.alert_manager.add_suppression_rule("memory_usage", 300)  # 5 minutes
        self.alert_manager.add_suppression_rule("cpu_usage", 300)
        self.alert_manager.add_suppression_rule("network_latency", 180)  # 3 minutes
    
    def _log_alert_handler(self, alert: Alert):
        """Default alert handler that logs alerts"""
        level = logging.CRITICAL if alert.severity == AlertSeverity.CRITICAL else \
                logging.WARNING if alert.severity == AlertSeverity.WARNING else \
                logging.INFO
        
        logger.log(level, f"ALERT: {alert.message}")
    
    async def start_monitoring(self):
        """Start comprehensive monitoring"""
        
        if self.monitoring_active:
            logger.warning("Monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        logger.info("Comprehensive monitoring started")
    
    async def stop_monitoring(self):
        """Stop monitoring"""
        
        self.monitoring_active = False
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Comprehensive monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        
        while self.monitoring_active:
            try:
                # Collect metrics
                metrics = await self.metric_collector.collect_all_metrics()
                
                # Store metrics
                if metrics:
                    self.metric_storage.store_metrics(metrics)
                    
                    # Evaluate alerts
                    alerts = await self.alert_manager.evaluate_metrics(
                        metrics, self.metric_collector.metric_definitions
                    )
                    
                    if alerts:
                        logger.info(f"Generated {len(alerts)} alerts in monitoring cycle")
                
                # Wait for next monitoring cycle
                await asyncio.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.monitoring_interval * 2)  # Longer sleep on error
    
    def register_custom_metric(self, metric_name: str, metric_definition: MetricDefinition,
                             collection_function: Callable[[], float]):
        """Register a custom metric with collection function"""
        
        self.metric_collector.metric_definitions[metric_name] = metric_definition
        self.metric_collector.register_collection_function(metric_name, collection_function)
        
        logger.info(f"Registered custom metric: {metric_name}")
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get comprehensive monitoring system status"""
        
        # Collect active alert summary
        active_alerts_by_severity = {}
        for alert in self.alert_manager.active_alerts.values():
            severity = alert.severity.value
            active_alerts_by_severity[severity] = active_alerts_by_severity.get(severity, 0) + 1
        
        # Recent metrics summary
        recent_metrics_count = 0
        for metric_name in self.metric_collector.metric_definitions.keys():
            recent_data = self.metric_storage.get_metric_history(metric_name, hours=1)
            recent_metrics_count += len(recent_data)
        
        # System health overview
        try:
            health_report = self.performance_analyzer.generate_performance_report(hours=1)
            overall_health = health_report["overall_health"]["status"]
        except:
            overall_health = "unknown"
        
        return {
            "monitoring_status": "ACTIVE" if self.monitoring_active else "INACTIVE",
            "monitoring_interval_seconds": self.monitoring_interval,
            "metrics_configured": len(self.metric_collector.metric_definitions),
            "metrics_collected_last_hour": recent_metrics_count,
            "active_alerts": {
                "total": len(self.alert_manager.active_alerts),
                "by_severity": active_alerts_by_severity
            },
            "alert_history_24h": len([
                alert for alert in self.alert_manager.alert_history 
                if time.time() - alert.timestamp < 86400
            ]),
            "system_health": overall_health,
            "data_retention_hours": self.metric_storage.max_storage_hours,
            "monitoring_capabilities": [
                "Real-time Metric Collection",
                "Intelligent Alerting",
                "Trend Analysis",
                "Anomaly Detection", 
                "Performance Reporting",
                "Custom Metric Support"
            ]
        }
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for monitoring dashboard"""
        
        dashboard_data = {
            "timestamp": time.time(),
            "system_overview": {},
            "key_metrics": {},
            "recent_alerts": [],
            "performance_trends": {}
        }
        
        # System overview
        dashboard_data["system_overview"] = {
            "status": "OPERATIONAL",
            "uptime_hours": 24,  # Placeholder
            "total_metrics": len(self.metric_collector.metric_definitions),
            "active_alerts": len(self.alert_manager.active_alerts)
        }
        
        # Key metrics (latest values)
        key_metrics = ["quantum_solve_time", "solution_quality", "energy_efficiency", "system_availability"]
        for metric_name in key_metrics:
            recent_data = self.metric_storage.get_metric_history(metric_name, hours=1)
            if recent_data:
                latest_value = recent_data[-1].value
                avg_value = self.metric_storage.get_aggregated_metric(metric_name, "mean", 1)
                dashboard_data["key_metrics"][metric_name] = {
                    "current": latest_value,
                    "average_1h": avg_value,
                    "unit": self.metric_collector.metric_definitions[metric_name].unit
                }
        
        # Recent alerts (last 10)
        dashboard_data["recent_alerts"] = [
            {
                "id": alert.alert_id,
                "metric": alert.metric_name,
                "severity": alert.severity.value,
                "message": alert.message,
                "timestamp": alert.timestamp,
                "resolved": alert.resolved
            }
            for alert in sorted(self.alert_manager.alert_history[-10:], 
                              key=lambda x: x.timestamp, reverse=True)
        ]
        
        # Performance trends
        for metric_name in key_metrics:
            trend_analysis = self.performance_analyzer.analyze_trends(metric_name, hours=6)
            if trend_analysis["trend"] != "insufficient_data":
                dashboard_data["performance_trends"][metric_name] = {
                    "trend": trend_analysis["trend"],
                    "volatility": trend_analysis["volatility"],
                    "current_vs_mean": trend_analysis["current_value"] - trend_analysis["mean_value"]
                }
        
        return dashboard_data
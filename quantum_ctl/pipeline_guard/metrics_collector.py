"""
Advanced metrics collection for quantum HVAC pipeline monitoring.
"""

import time
import asyncio
import json
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from datetime import datetime, timedelta
import threading


@dataclass
class Metric:
    name: str
    value: float
    timestamp: float
    labels: Dict[str, str]
    unit: Optional[str] = None


@dataclass
class Alert:
    id: str
    severity: str  # critical, warning, info
    component: str
    message: str
    timestamp: float
    resolved: bool = False
    resolution_time: Optional[float] = None


class MetricsCollector:
    """
    Advanced metrics collection system for quantum HVAC pipeline components.
    Provides real-time monitoring, alerting, and historical data analysis.
    """
    
    def __init__(self, retention_hours: int = 72):
        self.retention_hours = retention_hours
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.alerts: List[Alert] = []
        self.alert_rules: Dict[str, Callable[[float], bool]] = {}
        self.counters: Dict[str, int] = defaultdict(int)
        self.gauges: Dict[str, float] = {}
        self.histograms: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.RLock()
        
        # Start cleanup task
        self._cleanup_task = None
        self._running = False
        
    def start(self):
        """Start the metrics collector."""
        self._running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
    async def stop(self):
        """Stop the metrics collector."""
        self._running = False
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
                
    async def _cleanup_loop(self):
        """Periodic cleanup of old metrics."""
        while self._running:
            try:
                await asyncio.sleep(3600)  # Cleanup every hour
                self._cleanup_old_metrics()
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Metrics cleanup error: {e}")
                
    def _cleanup_old_metrics(self):
        """Remove metrics older than retention period."""
        cutoff_time = time.time() - (self.retention_hours * 3600)
        
        with self._lock:
            for metric_name, metric_queue in self.metrics.items():
                # Remove old metrics
                while metric_queue and metric_queue[0].timestamp < cutoff_time:
                    metric_queue.popleft()
                    
    def record_metric(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
        unit: Optional[str] = None
    ):
        """Record a metric value."""
        metric = Metric(
            name=name,
            value=value,
            timestamp=time.time(),
            labels=labels or {},
            unit=unit
        )
        
        with self._lock:
            self.metrics[name].append(metric)
            
            # Update gauges
            self.gauges[name] = value
            
            # Check alert rules
            self._check_alerts(name, value)
            
    def increment_counter(self, name: str, value: int = 1, labels: Optional[Dict[str, str]] = None):
        """Increment a counter metric."""
        with self._lock:
            self.counters[name] += value
            self.record_metric(f"{name}_total", self.counters[name], labels, "count")
            
    def record_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Record a value for histogram analysis."""
        with self._lock:
            self.histograms[name].append(value)
            
            # Keep only recent values for histograms
            if len(self.histograms[name]) > 1000:
                self.histograms[name] = self.histograms[name][-1000:]
                
            # Record current value as metric
            self.record_metric(name, value, labels)
            
    def record_quantum_metrics(self, solver_result: Any):
        """Record quantum annealing specific metrics."""
        try:
            if hasattr(solver_result, 'info'):
                info = solver_result.info
                
                # Chain break fraction
                if 'chain_break_fraction' in info:
                    self.record_metric(
                        "quantum_chain_break_fraction",
                        info['chain_break_fraction'],
                        labels={"solver": "quantum"},
                        unit="fraction"
                    )
                    
                # QPU access time
                if 'qpu_access_time' in info:
                    self.record_metric(
                        "quantum_qpu_access_time",
                        info['qpu_access_time'] / 1000,  # Convert to seconds
                        labels={"solver": "quantum"},
                        unit="seconds"
                    )
                    
                # Number of reads
                if 'num_reads' in info:
                    self.record_metric(
                        "quantum_num_reads",
                        info['num_reads'],
                        labels={"solver": "quantum"},
                        unit="count"
                    )
                    
                # Energy metrics
                if hasattr(solver_result, 'first'):
                    energy = solver_result.first.energy
                    self.record_metric(
                        "quantum_solution_energy",
                        energy,
                        labels={"solver": "quantum"},
                        unit="energy"
                    )
                    
        except Exception as e:
            print(f"Error recording quantum metrics: {e}")
            
    def record_hvac_metrics(self, controller_state: Dict[str, Any]):
        """Record HVAC controller specific metrics."""
        try:
            # Temperature metrics
            if 'zone_temperatures' in controller_state:
                for zone, temp in controller_state['zone_temperatures'].items():
                    self.record_metric(
                        "hvac_zone_temperature",
                        temp,
                        labels={"zone": zone},
                        unit="celsius"
                    )
                    
            # Energy consumption
            if 'energy_consumption' in controller_state:
                self.record_metric(
                    "hvac_energy_consumption",
                    controller_state['energy_consumption'],
                    labels={"building": "main"},
                    unit="kwh"
                )
                
            # Control actions
            if 'control_actions' in controller_state:
                for action, value in controller_state['control_actions'].items():
                    self.record_metric(
                        "hvac_control_action",
                        value,
                        labels={"action": action},
                        unit="percent"
                    )
                    
            # Optimization time
            if 'optimization_time' in controller_state:
                self.record_histogram(
                    "hvac_optimization_duration",
                    controller_state['optimization_time'],
                    labels={"controller": "hvac"}
                )
                
        except Exception as e:
            print(f"Error recording HVAC metrics: {e}")
            
    def add_alert_rule(self, metric_name: str, condition: Callable[[float], bool], severity: str = "warning"):
        """Add an alert rule for a metric."""
        self.alert_rules[f"{metric_name}_{severity}"] = {
            "metric": metric_name,
            "condition": condition,
            "severity": severity
        }
        
    def _check_alerts(self, metric_name: str, value: float):
        """Check if any alert rules are triggered."""
        for rule_name, rule in self.alert_rules.items():
            if rule["metric"] == metric_name:
                try:
                    if rule["condition"](value):
                        self._trigger_alert(
                            rule_name,
                            rule["severity"],
                            metric_name,
                            f"Alert triggered for {metric_name}: {value}"
                        )
                except Exception as e:
                    print(f"Error checking alert rule {rule_name}: {e}")
                    
    def _trigger_alert(self, alert_id: str, severity: str, component: str, message: str):
        """Trigger a new alert."""
        # Check if alert already exists and is not resolved
        existing_alert = next(
            (alert for alert in self.alerts 
             if alert.id == alert_id and not alert.resolved),
            None
        )
        
        if existing_alert:
            return  # Alert already active
            
        alert = Alert(
            id=alert_id,
            severity=severity,
            component=component,
            message=message,
            timestamp=time.time()
        )
        
        with self._lock:
            self.alerts.append(alert)
            
        print(f"ALERT [{severity.upper()}] {component}: {message}")
        
    def resolve_alert(self, alert_id: str):
        """Resolve an active alert."""
        with self._lock:
            for alert in self.alerts:
                if alert.id == alert_id and not alert.resolved:
                    alert.resolved = True
                    alert.resolution_time = time.time()
                    print(f"RESOLVED: Alert {alert_id}")
                    return True
        return False
        
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get all active (unresolved) alerts."""
        with self._lock:
            return [
                asdict(alert) for alert in self.alerts
                if not alert.resolved
            ]
            
    def get_metric_summary(self, metric_name: str, hours: int = 1) -> Dict[str, Any]:
        """Get summary statistics for a metric over time period."""
        cutoff_time = time.time() - (hours * 3600)
        
        with self._lock:
            if metric_name not in self.metrics:
                return {"error": f"Metric {metric_name} not found"}
                
            recent_metrics = [
                m for m in self.metrics[metric_name]
                if m.timestamp >= cutoff_time
            ]
            
            if not recent_metrics:
                return {"error": "No recent data"}
                
            values = [m.value for m in recent_metrics]
            
            return {
                "metric": metric_name,
                "time_period_hours": hours,
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "avg": sum(values) / len(values),
                "current": values[-1] if values else None,
                "unit": recent_metrics[-1].unit if recent_metrics else None
            }
            
    def get_histogram_percentiles(self, metric_name: str) -> Dict[str, float]:
        """Get histogram percentiles for a metric."""
        with self._lock:
            if metric_name not in self.histograms:
                return {"error": f"Histogram {metric_name} not found"}
                
            values = sorted(self.histograms[metric_name])
            if not values:
                return {"error": "No data"}
                
            def percentile(data: List[float], p: float) -> float:
                k = (len(data) - 1) * p / 100
                f = int(k)
                c = k - f
                if f == len(data) - 1:
                    return data[f]
                return data[f] * (1 - c) + data[f + 1] * c
                
            return {
                "p50": percentile(values, 50),
                "p90": percentile(values, 90),
                "p95": percentile(values, 95),
                "p99": percentile(values, 99),
                "count": len(values)
            }
            
    def export_metrics(self, format: str = "json") -> str:
        """Export metrics in specified format."""
        if format != "json":
            raise ValueError("Only JSON format supported currently")
            
        export_data = {
            "timestamp": time.time(),
            "retention_hours": self.retention_hours,
            "metrics": {},
            "counters": dict(self.counters),
            "gauges": dict(self.gauges),
            "alerts": [asdict(alert) for alert in self.alerts[-100:]]  # Last 100 alerts
        }
        
        # Export recent metrics (last hour)
        cutoff_time = time.time() - 3600
        
        with self._lock:
            for metric_name, metric_queue in self.metrics.items():
                recent_metrics = [
                    asdict(m) for m in metric_queue
                    if m.timestamp >= cutoff_time
                ]
                if recent_metrics:
                    export_data["metrics"][metric_name] = recent_metrics
                    
        return json.dumps(export_data, indent=2)
        
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data formatted for dashboard display."""
        current_time = time.time()
        
        # Get key metrics summaries
        key_metrics = [
            "quantum_chain_break_fraction",
            "quantum_qpu_access_time", 
            "hvac_optimization_duration",
            "hvac_energy_consumption"
        ]
        
        metric_summaries = {}
        for metric in key_metrics:
            summary = self.get_metric_summary(metric, hours=1)
            if "error" not in summary:
                metric_summaries[metric] = summary
                
        return {
            "timestamp": current_time,
            "active_alerts": self.get_active_alerts(),
            "total_metrics_collected": sum(len(q) for q in self.metrics.values()),
            "metric_summaries": metric_summaries,
            "system_status": "healthy" if not self.get_active_alerts() else "degraded"
        }
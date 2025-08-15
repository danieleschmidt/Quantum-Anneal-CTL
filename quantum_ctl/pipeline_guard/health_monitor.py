"""
Health monitoring system for quantum HVAC pipeline components.
"""

import time
import asyncio
from typing import Dict, List, Callable, Any, Optional
from dataclasses import dataclass, field
from collections import deque


@dataclass
class HealthMetric:
    component: str
    timestamp: float
    healthy: bool
    response_time: float
    error: Optional[str] = None


@dataclass
class ComponentHealth:
    name: str
    current_status: bool = True
    last_check: float = field(default_factory=time.time)
    failure_count: int = 0
    success_count: int = 0
    avg_response_time: float = 0.0
    history: deque = field(default_factory=lambda: deque(maxlen=100))


class HealthMonitor:
    """
    Monitors health of pipeline components and maintains historical data.
    """
    
    def __init__(self, history_size: int = 1000):
        self.components: Dict[str, ComponentHealth] = {}
        self.health_checks: Dict[str, Callable[[], bool]] = {}
        self.history: deque = deque(maxlen=history_size)
        
    def register_component(self, name: str, health_check: Callable[[], bool]):
        """Register a component for health monitoring."""
        self.components[name] = ComponentHealth(name=name)
        self.health_checks[name] = health_check
        
    async def check_component_health(self, name: str) -> HealthMetric:
        """Check health of a specific component."""
        if name not in self.health_checks:
            raise ValueError(f"Component {name} not registered")
            
        start_time = time.time()
        error = None
        healthy = False
        
        try:
            health_check = self.health_checks[name]
            if asyncio.iscoroutinefunction(health_check):
                healthy = await health_check()
            else:
                healthy = health_check()
        except Exception as e:
            error = str(e)
            healthy = False
            
        response_time = time.time() - start_time
        
        # Update component health data
        component = self.components[name]
        component.current_status = healthy
        component.last_check = time.time()
        
        if healthy:
            component.success_count += 1
        else:
            component.failure_count += 1
            
        # Update average response time
        component.avg_response_time = (
            (component.avg_response_time * (component.success_count + component.failure_count - 1) + response_time) /
            (component.success_count + component.failure_count)
        )
        
        metric = HealthMetric(
            component=name,
            timestamp=time.time(),
            healthy=healthy,
            response_time=response_time,
            error=error
        )
        
        # Add to history
        component.history.append(metric)
        self.history.append(metric)
        
        return metric
        
    async def check_all_components(self) -> Dict[str, HealthMetric]:
        """Check health of all registered components."""
        results = {}
        
        # Run health checks concurrently
        tasks = [
            self.check_component_health(name) 
            for name in self.health_checks
        ]
        
        metrics = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, metric in enumerate(metrics):
            component_name = list(self.health_checks.keys())[i]
            if isinstance(metric, Exception):
                results[component_name] = HealthMetric(
                    component=component_name,
                    timestamp=time.time(),
                    healthy=False,
                    response_time=0.0,
                    error=str(metric)
                )
            else:
                results[component_name] = metric
                
        return results
        
    def get_component_status(self, name: str) -> Dict[str, Any]:
        """Get detailed status of a component."""
        if name not in self.components:
            raise ValueError(f"Component {name} not found")
            
        component = self.components[name]
        total_checks = component.success_count + component.failure_count
        
        return {
            "name": name,
            "current_status": component.current_status,
            "last_check": component.last_check,
            "uptime_percentage": (
                (component.success_count / total_checks * 100) 
                if total_checks > 0 else 100.0
            ),
            "failure_count": component.failure_count,
            "success_count": component.success_count,
            "avg_response_time": component.avg_response_time,
            "recent_errors": [
                m.error for m in list(component.history)[-10:] 
                if m.error is not None
            ]
        }
        
    def get_all_component_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all components."""
        return {
            name: self.get_component_status(name)
            for name in self.components
        }
        
    def get_health_trends(self, component: str, hours: int = 24) -> Dict[str, Any]:
        """Get health trends for a component over specified time period."""
        if component not in self.components:
            raise ValueError(f"Component {component} not found")
            
        cutoff_time = time.time() - (hours * 3600)
        comp_history = self.components[component].history
        
        recent_metrics = [
            m for m in comp_history 
            if m.timestamp >= cutoff_time
        ]
        
        if not recent_metrics:
            return {"error": "No recent data available"}
            
        total_checks = len(recent_metrics)
        successful_checks = sum(1 for m in recent_metrics if m.healthy)
        avg_response_time = sum(m.response_time for m in recent_metrics) / total_checks
        
        # Calculate hourly availability
        hourly_stats = {}
        for metric in recent_metrics:
            hour = int(metric.timestamp // 3600)
            if hour not in hourly_stats:
                hourly_stats[hour] = {"total": 0, "successful": 0}
            hourly_stats[hour]["total"] += 1
            if metric.healthy:
                hourly_stats[hour]["successful"] += 1
                
        hourly_availability = [
            {
                "hour": hour,
                "availability": stats["successful"] / stats["total"] * 100
            }
            for hour, stats in sorted(hourly_stats.items())
        ]
        
        return {
            "component": component,
            "time_period_hours": hours,
            "total_checks": total_checks,
            "availability_percentage": successful_checks / total_checks * 100,
            "avg_response_time": avg_response_time,
            "hourly_availability": hourly_availability
        }
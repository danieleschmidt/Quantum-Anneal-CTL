"""
Performance optimization for quantum HVAC pipeline guard.
"""

import asyncio
import time
import threading
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass
from collections import deque, defaultdict
import statistics
import resource
import psutil


@dataclass
class PerformanceMetric:
    name: str
    value: float
    timestamp: float
    component: str
    metadata: Dict[str, Any] = None


class PerformanceOptimizer:
    """
    Advanced performance optimization system for quantum HVAC pipeline guard.
    Implements adaptive scaling, intelligent caching, and resource optimization.
    """
    
    def __init__(self):
        self.metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.performance_baselines: Dict[str, float] = {}
        self.optimization_rules: List[Dict[str, Any]] = []
        self.active_optimizations: Dict[str, Any] = {}
        self.resource_limits: Dict[str, float] = {}
        self.scaling_policies: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()
        
        # Initialize default resource limits
        self._initialize_resource_limits()
        
    def _initialize_resource_limits(self):
        """Initialize default resource limits based on system capacity."""
        try:
            # Get system resources
            memory = psutil.virtual_memory()
            cpu_count = psutil.cpu_count()
            
            # Set conservative limits (70% of available resources)
            self.resource_limits = {
                "max_memory_mb": memory.total / (1024 * 1024) * 0.7,
                "max_cpu_percent": 70.0,
                "max_connections": 1000,
                "max_concurrent_operations": cpu_count * 2,
                "max_cache_size_mb": 100
            }
            
        except Exception:
            # Fallback limits
            self.resource_limits = {
                "max_memory_mb": 1024,
                "max_cpu_percent": 70.0,
                "max_connections": 100,
                "max_concurrent_operations": 4,
                "max_cache_size_mb": 50
            }
            
    def register_scaling_policy(
        self,
        component: str,
        scale_up_threshold: float,
        scale_down_threshold: float,
        max_instances: int = 10,
        min_instances: int = 1,
        cooldown_seconds: int = 300
    ):
        """Register auto-scaling policy for a component."""
        self.scaling_policies[component] = {
            "scale_up_threshold": scale_up_threshold,
            "scale_down_threshold": scale_down_threshold,
            "max_instances": max_instances,
            "min_instances": min_instances,
            "cooldown_seconds": cooldown_seconds,
            "current_instances": min_instances,
            "last_scale_time": 0
        }
        
    def add_optimization_rule(
        self,
        name: str,
        condition: Callable[[Dict[str, float]], bool],
        action: Callable[[], Any],
        priority: int = 1
    ):
        """Add performance optimization rule."""
        rule = {
            "name": name,
            "condition": condition,
            "action": action,
            "priority": priority,
            "last_triggered": 0,
            "trigger_count": 0
        }
        
        self.optimization_rules.append(rule)
        self.optimization_rules.sort(key=lambda x: x["priority"], reverse=True)
        
    def record_performance_metric(
        self,
        name: str,
        value: float,
        component: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Record performance metric for analysis."""
        metric = PerformanceMetric(
            name=name,
            value=value,
            timestamp=time.time(),
            component=component,
            metadata=metadata or {}
        )
        
        with self._lock:
            self.metrics_history[name].append(metric)
            
            # Update baseline if not set
            if name not in self.performance_baselines:
                self._calculate_baseline(name)
                
    def _calculate_baseline(self, metric_name: str):
        """Calculate performance baseline for a metric."""
        if metric_name not in self.metrics_history:
            return
            
        history = list(self.metrics_history[metric_name])
        if len(history) < 10:
            return  # Need more data
            
        # Use recent median as baseline
        recent_values = [m.value for m in history[-50:]]
        baseline = statistics.median(recent_values)
        self.performance_baselines[metric_name] = baseline
        
    async def optimize_performance(self) -> Dict[str, Any]:
        """
        Analyze performance metrics and apply optimizations.
        Returns summary of optimizations applied.
        """
        optimizations_applied = []
        current_metrics = self._get_current_metrics()
        
        # Check optimization rules
        for rule in self.optimization_rules:
            try:
                if rule["condition"](current_metrics):
                    # Check cooldown period
                    if time.time() - rule["last_triggered"] > 60:  # 1 minute cooldown
                        result = await self._execute_optimization(rule)
                        if result:
                            optimizations_applied.append({
                                "rule": rule["name"],
                                "result": result,
                                "timestamp": time.time()
                            })
                            rule["last_triggered"] = time.time()
                            rule["trigger_count"] += 1
                            
            except Exception as e:
                print(f"Error in optimization rule {rule['name']}: {e}")
                
        # Apply auto-scaling
        scaling_results = await self._apply_auto_scaling(current_metrics)
        optimizations_applied.extend(scaling_results)
        
        # Resource optimization
        resource_results = await self._optimize_resources(current_metrics)
        optimizations_applied.extend(resource_results)
        
        return {
            "optimizations_applied": optimizations_applied,
            "current_metrics": current_metrics,
            "timestamp": time.time()
        }
        
    def _get_current_metrics(self) -> Dict[str, float]:
        """Get current performance metrics summary."""
        current_metrics = {}
        
        with self._lock:
            for metric_name, history in self.metrics_history.items():
                if history:
                    # Get recent average (last 5 minutes)
                    cutoff_time = time.time() - 300
                    recent_metrics = [
                        m for m in history
                        if m.timestamp > cutoff_time
                    ]
                    
                    if recent_metrics:
                        avg_value = sum(m.value for m in recent_metrics) / len(recent_metrics)
                        current_metrics[metric_name] = avg_value
                        
        # Add system metrics
        try:
            current_metrics.update({
                "system_cpu_percent": psutil.cpu_percent(interval=0.1),
                "system_memory_percent": psutil.virtual_memory().percent,
                "system_disk_io": psutil.disk_io_counters().read_bytes + psutil.disk_io_counters().write_bytes,
                "system_network_io": psutil.net_io_counters().bytes_sent + psutil.net_io_counters().bytes_recv
            })
        except Exception:
            pass
            
        return current_metrics
        
    async def _execute_optimization(self, rule: Dict[str, Any]) -> Any:
        """Execute an optimization rule action."""
        try:
            action = rule["action"]
            if asyncio.iscoroutinefunction(action):
                return await action()
            else:
                return action()
        except Exception as e:
            print(f"Error executing optimization {rule['name']}: {e}")
            return None
            
    async def _apply_auto_scaling(self, current_metrics: Dict[str, float]) -> List[Dict[str, Any]]:
        """Apply auto-scaling based on current metrics."""
        scaling_results = []
        
        for component, policy in self.scaling_policies.items():
            try:
                # Check if cooldown period has passed
                if time.time() - policy["last_scale_time"] < policy["cooldown_seconds"]:
                    continue
                    
                # Get component-specific metrics
                component_load = self._calculate_component_load(component, current_metrics)
                
                scale_action = None
                
                # Check scale up
                if component_load > policy["scale_up_threshold"]:
                    if policy["current_instances"] < policy["max_instances"]:
                        policy["current_instances"] += 1
                        policy["last_scale_time"] = time.time()
                        scale_action = "scale_up"
                        
                # Check scale down
                elif component_load < policy["scale_down_threshold"]:
                    if policy["current_instances"] > policy["min_instances"]:
                        policy["current_instances"] -= 1
                        policy["last_scale_time"] = time.time()
                        scale_action = "scale_down"
                        
                if scale_action:
                    scaling_results.append({
                        "rule": f"auto_scale_{component}",
                        "result": {
                            "action": scale_action,
                            "new_instances": policy["current_instances"],
                            "load": component_load
                        },
                        "timestamp": time.time()
                    })
                    
            except Exception as e:
                print(f"Error in auto-scaling for {component}: {e}")
                
        return scaling_results
        
    def _calculate_component_load(self, component: str, current_metrics: Dict[str, float]) -> float:
        """Calculate load percentage for a component."""
        # Look for component-specific metrics
        component_metrics = [
            value for name, value in current_metrics.items()
            if component.lower() in name.lower()
        ]
        
        if component_metrics:
            return sum(component_metrics) / len(component_metrics)
            
        # Fallback to system metrics
        cpu_load = current_metrics.get("system_cpu_percent", 0)
        memory_load = current_metrics.get("system_memory_percent", 0)
        
        return (cpu_load + memory_load) / 2
        
    async def _optimize_resources(self, current_metrics: Dict[str, float]) -> List[Dict[str, Any]]:
        """Optimize system resources based on current usage."""
        optimizations = []
        
        # Memory optimization
        memory_percent = current_metrics.get("system_memory_percent", 0)
        if memory_percent > 80:
            result = await self._optimize_memory()
            if result:
                optimizations.append({
                    "rule": "memory_optimization",
                    "result": result,
                    "timestamp": time.time()
                })
                
        # CPU optimization  
        cpu_percent = current_metrics.get("system_cpu_percent", 0)
        if cpu_percent > 85:
            result = await self._optimize_cpu()
            if result:
                optimizations.append({
                    "rule": "cpu_optimization", 
                    "result": result,
                    "timestamp": time.time()
                })
                
        # Cache optimization
        cache_size = current_metrics.get("cache_size_mb", 0)
        if cache_size > self.resource_limits["max_cache_size_mb"]:
            result = await self._optimize_cache()
            if result:
                optimizations.append({
                    "rule": "cache_optimization",
                    "result": result,
                    "timestamp": time.time()
                })
                
        return optimizations
        
    async def _optimize_memory(self) -> Dict[str, Any]:
        """Optimize memory usage."""
        try:
            import gc
            
            # Force garbage collection
            initial_objects = len(gc.get_objects())
            gc.collect()
            final_objects = len(gc.get_objects())
            
            objects_freed = initial_objects - final_objects
            
            return {
                "action": "garbage_collection",
                "objects_freed": objects_freed,
                "memory_freed_estimate": objects_freed * 64  # Rough estimate in bytes
            }
            
        except Exception as e:
            print(f"Memory optimization failed: {e}")
            return None
            
    async def _optimize_cpu(self) -> Dict[str, Any]:
        """Optimize CPU usage."""
        try:
            # Reduce process priority if possible
            current_priority = resource.getpriority(resource.PRIO_PROCESS, 0)
            
            if current_priority < 5:  # Only if not already low priority
                resource.setpriority(resource.PRIO_PROCESS, 0, current_priority + 1)
                
                return {
                    "action": "reduce_priority",
                    "old_priority": current_priority,
                    "new_priority": current_priority + 1
                }
                
        except Exception as e:
            print(f"CPU optimization failed: {e}")
            
        return None
        
    async def _optimize_cache(self) -> Dict[str, Any]:
        """Optimize cache usage."""
        try:
            # This would integrate with the actual cache implementation
            # For now, simulate cache cleanup
            
            return {
                "action": "cache_cleanup",
                "entries_removed": 100,  # Placeholder
                "memory_freed_mb": 10     # Placeholder
            }
            
        except Exception as e:
            print(f"Cache optimization failed: {e}")
            return None
            
    def get_performance_analytics(self) -> Dict[str, Any]:
        """Get performance analytics and insights."""
        current_metrics = self._get_current_metrics()
        
        # Calculate performance trends
        trends = {}
        for metric_name in self.metrics_history:
            trend = self._calculate_trend(metric_name)
            if trend is not None:
                trends[metric_name] = trend
                
        # Resource utilization
        resource_utilization = {}
        for resource_name, limit in self.resource_limits.items():
            current_value = current_metrics.get(resource_name.replace("max_", ""), 0)
            utilization = (current_value / limit) * 100 if limit > 0 else 0
            resource_utilization[resource_name] = {
                "current": current_value,
                "limit": limit,
                "utilization_percent": utilization
            }
            
        # Optimization statistics
        optimization_stats = {
            "total_rules": len(self.optimization_rules),
            "active_optimizations": len(self.active_optimizations),
            "rule_triggers": {
                rule["name"]: rule["trigger_count"]
                for rule in self.optimization_rules
            }
        }
        
        return {
            "current_metrics": current_metrics,
            "performance_trends": trends,
            "resource_utilization": resource_utilization,
            "optimization_statistics": optimization_stats,
            "scaling_policies": {
                component: {
                    "current_instances": policy["current_instances"],
                    "min_instances": policy["min_instances"],
                    "max_instances": policy["max_instances"]
                }
                for component, policy in self.scaling_policies.items()
            },
            "timestamp": time.time()
        }
        
    def _calculate_trend(self, metric_name: str, window_minutes: int = 30) -> Optional[str]:
        """Calculate trend for a metric over specified window."""
        if metric_name not in self.metrics_history:
            return None
            
        cutoff_time = time.time() - (window_minutes * 60)
        recent_metrics = [
            m for m in self.metrics_history[metric_name]
            if m.timestamp > cutoff_time
        ]
        
        if len(recent_metrics) < 5:
            return None
            
        # Simple trend calculation
        values = [m.value for m in recent_metrics]
        first_half = values[:len(values)//2]
        second_half = values[len(values)//2:]
        
        first_avg = sum(first_half) / len(first_half)
        second_avg = sum(second_half) / len(second_half)
        
        change_percent = ((second_avg - first_avg) / first_avg) * 100 if first_avg > 0 else 0
        
        if change_percent > 10:
            return "increasing"
        elif change_percent < -10:
            return "decreasing"
        else:
            return "stable"
            
    def suggest_optimizations(self) -> List[Dict[str, Any]]:
        """Suggest performance optimizations based on analysis."""
        suggestions = []
        current_metrics = self._get_current_metrics()
        
        # Memory suggestions
        memory_percent = current_metrics.get("system_memory_percent", 0)
        if memory_percent > 70:
            suggestions.append({
                "type": "memory",
                "priority": "high" if memory_percent > 85 else "medium",
                "suggestion": "Consider increasing memory limits or implementing memory pooling",
                "current_usage": f"{memory_percent:.1f}%"
            })
            
        # CPU suggestions
        cpu_percent = current_metrics.get("system_cpu_percent", 0)
        if cpu_percent > 70:
            suggestions.append({
                "type": "cpu",
                "priority": "high" if cpu_percent > 85 else "medium", 
                "suggestion": "Consider implementing CPU throttling or load balancing",
                "current_usage": f"{cpu_percent:.1f}%"
            })
            
        # Quantum-specific suggestions
        chain_break_rate = current_metrics.get("quantum_chain_break_fraction", 0)
        if chain_break_rate > 0.1:
            suggestions.append({
                "type": "quantum",
                "priority": "high",
                "suggestion": "High chain break rate detected. Consider adjusting embedding parameters or using hybrid solver",
                "current_rate": f"{chain_break_rate:.3f}"
            })
            
        # HVAC optimization suggestions
        optimization_time = current_metrics.get("hvac_optimization_duration", 0)
        if optimization_time > 30:  # 30 seconds
            suggestions.append({
                "type": "hvac",
                "priority": "medium",
                "suggestion": "HVAC optimization taking too long. Consider problem decomposition or caching",
                "current_time": f"{optimization_time:.2f}s"
            })
            
        return suggestions
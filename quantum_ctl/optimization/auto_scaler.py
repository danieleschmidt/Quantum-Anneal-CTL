"""
Auto-scaling and resource optimization for quantum HVAC control.
"""

import asyncio
import logging
import time
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import psutil

from ..utils.performance import get_resource_manager
from ..utils.monitoring import HealthMonitor


class ScalingDecision(Enum):
    """Scaling decisions."""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    NO_CHANGE = "no_change"
    EMERGENCY_SCALE = "emergency_scale"


@dataclass
class ResourceMetrics:
    """Resource usage metrics."""
    cpu_percent: float
    memory_percent: float
    optimization_queue_length: int
    avg_optimization_time: float
    success_rate: float
    timestamp: float = field(default_factory=time.time)


@dataclass 
class ScalingPolicy:
    """Auto-scaling policy configuration."""
    # CPU thresholds
    cpu_scale_up_threshold: float = 70.0
    cpu_scale_down_threshold: float = 30.0
    
    # Memory thresholds  
    memory_scale_up_threshold: float = 80.0
    memory_scale_down_threshold: float = 40.0
    
    # Queue length thresholds
    queue_scale_up_threshold: int = 10
    queue_scale_down_threshold: int = 2
    
    # Performance thresholds
    optimization_time_threshold: float = 30.0  # seconds
    success_rate_threshold: float = 0.85
    
    # Scaling limits
    min_workers: int = 1
    max_workers: int = 16
    
    # Timing
    evaluation_interval: float = 60.0  # seconds
    cooldown_period: float = 300.0     # 5 minutes


class AutoScaler:
    """Automatic resource scaling and optimization."""
    
    def __init__(self, policy: ScalingPolicy):
        self.policy = policy
        self.logger = logging.getLogger(__name__)
        
        # Scaling state
        self.current_workers = 2
        self.last_scaling_action = 0.0
        self.scaling_history: List[Tuple[float, ScalingDecision, int]] = []
        
        # Metrics tracking
        self.metrics_history: List[ResourceMetrics] = []
        self.max_history_length = 100
        
        # Control
        self._is_running = False
        self._scaling_task: Optional[asyncio.Task] = None
        
        # Resource manager integration
        self.resource_manager = get_resource_manager()
        self.health_monitor = HealthMonitor()
    
    def start(self):
        """Start auto-scaling monitoring."""
        if self._is_running:
            return
        
        self._is_running = True
        self._scaling_task = asyncio.create_task(self._scaling_loop())
        self.logger.info("Started auto-scaling with policy: %s", self.policy)
    
    def stop(self):
        """Stop auto-scaling monitoring."""
        self._is_running = False
        
        if self._scaling_task:
            self._scaling_task.cancel()
            self._scaling_task = None
        
        self.logger.info("Stopped auto-scaling")
    
    async def _scaling_loop(self):
        """Main auto-scaling loop."""
        while self._is_running:
            try:
                # Collect current metrics
                metrics = await self._collect_metrics()
                self.metrics_history.append(metrics)
                
                # Trim history
                if len(self.metrics_history) > self.max_history_length:
                    self.metrics_history.pop(0)
                
                # Evaluate scaling decision
                decision, target_workers = self._evaluate_scaling(metrics)
                
                # Apply scaling decision
                if decision != ScalingDecision.NO_CHANGE:
                    await self._apply_scaling(decision, target_workers, metrics)
                
                # Wait for next evaluation
                await asyncio.sleep(self.policy.evaluation_interval)
                
            except Exception as e:
                self.logger.error(f"Auto-scaling loop error: {e}")
                await asyncio.sleep(30)  # Short delay before retrying
    
    async def _collect_metrics(self) -> ResourceMetrics:
        """Collect current resource and performance metrics."""
        
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=1.0)
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        # Performance metrics
        performance_summary = self.resource_manager.get_performance_summary()
        avg_opt_time = performance_summary.get('performance', {}).get('avg_optimization_time', 0.0)
        
        # Queue metrics (simulated - would come from actual queue)
        queue_length = self._estimate_queue_length()
        
        # Success rate (from health monitor)
        health_status = self.health_monitor.get_health_status()
        success_rate = health_status.get('performance', {}).get('success_rate', 1.0)
        
        return ResourceMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            optimization_queue_length=queue_length,
            avg_optimization_time=avg_opt_time,
            success_rate=success_rate
        )
    
    def _estimate_queue_length(self) -> int:
        """Estimate optimization queue length."""
        # In a real implementation, this would query actual queue
        # For now, estimate based on recent optimization times
        if len(self.metrics_history) < 2:
            return 0
        
        recent_times = [m.avg_optimization_time for m in self.metrics_history[-5:]]
        avg_recent_time = np.mean(recent_times) if recent_times else 0.0
        
        # Simulate queue buildup when optimizations are slow
        if avg_recent_time > 15.0:
            return int(avg_recent_time / 5.0)  # Rough estimate
        
        return 0
    
    def _evaluate_scaling(self, metrics: ResourceMetrics) -> Tuple[ScalingDecision, int]:
        """Evaluate whether scaling is needed."""
        
        # Check cooldown period
        time_since_last_action = time.time() - self.last_scaling_action
        if time_since_last_action < self.policy.cooldown_period:
            return ScalingDecision.NO_CHANGE, self.current_workers
        
        # Collect scaling signals
        scale_up_signals = []
        scale_down_signals = []
        emergency_signals = []
        
        # CPU-based scaling
        if metrics.cpu_percent > self.policy.cpu_scale_up_threshold:
            scale_up_signals.append(f"CPU {metrics.cpu_percent:.1f}% > {self.policy.cpu_scale_up_threshold}%")
        elif metrics.cpu_percent < self.policy.cpu_scale_down_threshold:
            scale_down_signals.append(f"CPU {metrics.cpu_percent:.1f}% < {self.policy.cpu_scale_down_threshold}%")
        
        # Memory-based scaling
        if metrics.memory_percent > self.policy.memory_scale_up_threshold:
            scale_up_signals.append(f"Memory {metrics.memory_percent:.1f}% > {self.policy.memory_scale_up_threshold}%")
        elif metrics.memory_percent < self.policy.memory_scale_down_threshold:
            scale_down_signals.append(f"Memory {metrics.memory_percent:.1f}% < {self.policy.memory_scale_down_threshold}%")
        
        # Queue-based scaling
        if metrics.optimization_queue_length > self.policy.queue_scale_up_threshold:
            scale_up_signals.append(f"Queue length {metrics.optimization_queue_length} > {self.policy.queue_scale_up_threshold}")
        elif metrics.optimization_queue_length < self.policy.queue_scale_down_threshold:
            scale_down_signals.append(f"Queue length {metrics.optimization_queue_length} < {self.policy.queue_scale_down_threshold}")
        
        # Performance-based scaling
        if metrics.avg_optimization_time > self.policy.optimization_time_threshold:
            scale_up_signals.append(f"Opt time {metrics.avg_optimization_time:.1f}s > {self.policy.optimization_time_threshold}s")
        
        if metrics.success_rate < self.policy.success_rate_threshold:
            scale_up_signals.append(f"Success rate {metrics.success_rate:.3f} < {self.policy.success_rate_threshold}")
        
        # Emergency conditions
        if metrics.cpu_percent > 95.0 or metrics.memory_percent > 95.0:
            emergency_signals.append("Critical resource usage")
        
        if metrics.success_rate < 0.5:
            emergency_signals.append("Critical success rate")
        
        # Make scaling decision
        if emergency_signals:
            target = min(self.current_workers * 2, self.policy.max_workers)
            self.logger.warning(f"Emergency scaling triggered: {emergency_signals}")
            return ScalingDecision.EMERGENCY_SCALE, target
        
        elif len(scale_up_signals) >= 2 or (len(scale_up_signals) >= 1 and not scale_down_signals):
            # Scale up if multiple signals or single strong signal
            target = min(self.current_workers + 1, self.policy.max_workers)
            if target > self.current_workers:
                self.logger.info(f"Scale up triggered: {scale_up_signals}")
                return ScalingDecision.SCALE_UP, target
        
        elif len(scale_down_signals) >= 2 and not scale_up_signals:
            # Scale down only if multiple down signals and no up signals
            target = max(self.current_workers - 1, self.policy.min_workers)
            if target < self.current_workers:
                self.logger.info(f"Scale down triggered: {scale_down_signals}")
                return ScalingDecision.SCALE_DOWN, target
        
        return ScalingDecision.NO_CHANGE, self.current_workers
    
    async def _apply_scaling(self, decision: ScalingDecision, target_workers: int, 
                           metrics: ResourceMetrics):
        """Apply scaling decision."""
        
        old_workers = self.current_workers
        self.current_workers = target_workers
        self.last_scaling_action = time.time()
        
        # Record scaling action
        self.scaling_history.append((time.time(), decision, target_workers))
        
        # Apply the scaling (in real implementation, would adjust actual worker pool)
        await self._adjust_worker_pool(target_workers)
        
        # Update resource manager
        if hasattr(self.resource_manager, '_parallel_processor'):
            self.resource_manager._parallel_processor.max_workers = target_workers
        
        # Log scaling action
        action_desc = {
            ScalingDecision.SCALE_UP: "Scaled up",
            ScalingDecision.SCALE_DOWN: "Scaled down", 
            ScalingDecision.EMERGENCY_SCALE: "Emergency scaled"
        }.get(decision, "Adjusted")
        
        self.logger.info(
            f"{action_desc} from {old_workers} to {target_workers} workers "
            f"(CPU: {metrics.cpu_percent:.1f}%, Memory: {metrics.memory_percent:.1f}%, "
            f"Queue: {metrics.optimization_queue_length}, OptTime: {metrics.avg_optimization_time:.1f}s)"
        )
    
    async def _adjust_worker_pool(self, target_workers: int):
        """Adjust actual worker pool size."""
        # In a real implementation, this would:
        # 1. Spawn/terminate worker processes
        # 2. Update load balancer configuration
        # 3. Adjust quantum solver connection pools
        # 4. Update resource allocations
        
        # For demo, we'll simulate the adjustment
        await asyncio.sleep(0.1)  # Simulate adjustment time
        
        # Update any thread/process pools
        if hasattr(self.resource_manager, 'parallel_processor'):
            processor = self.resource_manager.parallel_processor
            if hasattr(processor, 'max_workers'):
                processor.max_workers = target_workers
    
    def get_scaling_status(self) -> Dict[str, Any]:
        """Get current auto-scaling status."""
        recent_metrics = self.metrics_history[-1] if self.metrics_history else None
        
        return {
            'is_active': self._is_running,
            'current_workers': self.current_workers,
            'policy': {
                'cpu_thresholds': [self.policy.cpu_scale_down_threshold, self.policy.cpu_scale_up_threshold],
                'memory_thresholds': [self.policy.memory_scale_down_threshold, self.policy.memory_scale_up_threshold],
                'worker_limits': [self.policy.min_workers, self.policy.max_workers]
            },
            'recent_metrics': {
                'cpu_percent': recent_metrics.cpu_percent if recent_metrics else 0.0,
                'memory_percent': recent_metrics.memory_percent if recent_metrics else 0.0,
                'queue_length': recent_metrics.optimization_queue_length if recent_metrics else 0,
                'optimization_time': recent_metrics.avg_optimization_time if recent_metrics else 0.0,
                'success_rate': recent_metrics.success_rate if recent_metrics else 1.0
            },
            'scaling_history': [
                {
                    'timestamp': ts,
                    'action': decision.value,
                    'target_workers': workers
                }
                for ts, decision, workers in self.scaling_history[-10:]  # Last 10 actions
            ],
            'time_since_last_action': time.time() - self.last_scaling_action,
            'cooldown_remaining': max(0, self.policy.cooldown_period - (time.time() - self.last_scaling_action))
        }
    
    def predict_scaling_need(self, forecast_horizon: int = 300) -> Dict[str, Any]:
        """Predict future scaling needs based on trends."""
        if len(self.metrics_history) < 10:
            return {'prediction': 'insufficient_data'}
        
        # Analyze trends in recent metrics
        recent_metrics = self.metrics_history[-10:]
        
        # CPU trend
        cpu_values = [m.cpu_percent for m in recent_metrics]
        cpu_trend = np.polyfit(range(len(cpu_values)), cpu_values, 1)[0]
        
        # Memory trend  
        memory_values = [m.memory_percent for m in recent_metrics]
        memory_trend = np.polyfit(range(len(memory_values)), memory_values, 1)[0]
        
        # Optimization time trend
        opt_time_values = [m.avg_optimization_time for m in recent_metrics]
        opt_time_trend = np.polyfit(range(len(opt_time_values)), opt_time_values, 1)[0]
        
        # Project forward
        forecast_steps = forecast_horizon // 60  # Assuming 1-minute intervals
        
        projected_cpu = cpu_values[-1] + cpu_trend * forecast_steps
        projected_memory = memory_values[-1] + memory_trend * forecast_steps
        projected_opt_time = opt_time_values[-1] + opt_time_trend * forecast_steps
        
        # Predict scaling need
        will_need_scale_up = (
            projected_cpu > self.policy.cpu_scale_up_threshold or
            projected_memory > self.policy.memory_scale_up_threshold or 
            projected_opt_time > self.policy.optimization_time_threshold
        )
        
        will_need_scale_down = (
            projected_cpu < self.policy.cpu_scale_down_threshold and
            projected_memory < self.policy.memory_scale_down_threshold and
            projected_opt_time < self.policy.optimization_time_threshold / 2
        )
        
        prediction = 'scale_up' if will_need_scale_up else ('scale_down' if will_need_scale_down else 'stable')
        
        return {
            'prediction': prediction,
            'confidence': min(1.0, len(recent_metrics) / 20.0),  # More data = higher confidence
            'forecast_horizon': forecast_horizon,
            'projected_metrics': {
                'cpu_percent': max(0, min(100, projected_cpu)),
                'memory_percent': max(0, min(100, projected_memory)),
                'optimization_time': max(0, projected_opt_time)
            },
            'trends': {
                'cpu_trend': cpu_trend,
                'memory_trend': memory_trend,
                'optimization_time_trend': opt_time_trend
            }
        }


class AdaptiveOptimizationScheduler:
    """Adaptive scheduler for optimization workloads."""
    
    def __init__(self, auto_scaler: AutoScaler):
        self.auto_scaler = auto_scaler
        self.logger = logging.getLogger(__name__)
        
        # Workload state
        self._pending_optimizations: List[Dict] = []
        self._active_optimizations: Dict[str, Dict] = {}
        self._optimization_history: List[Dict] = []
        
    def submit_optimization(self, optimization_request: Dict[str, Any]) -> str:
        """Submit optimization request for scheduling."""
        request_id = f"opt_{int(time.time() * 1000000)}"
        
        optimization_task = {
            'id': request_id,
            'request': optimization_request,
            'submitted_at': time.time(),
            'priority': optimization_request.get('priority', 'normal'),
            'estimated_duration': self._estimate_duration(optimization_request),
            'status': 'pending'
        }
        
        self._pending_optimizations.append(optimization_task)
        self.logger.info(f"Submitted optimization {request_id} (priority: {optimization_task['priority']})")
        
        return request_id
    
    def _estimate_duration(self, request: Dict[str, Any]) -> float:
        """Estimate optimization duration based on problem characteristics."""
        
        # Base duration based on problem size
        horizon = request.get('horizon', 24)
        zones = request.get('zones', 5)
        complexity_factor = request.get('complexity_factor', 1.0)
        
        base_duration = (horizon * zones * complexity_factor) / 100.0
        
        # Adjust based on historical performance
        if self._optimization_history:
            recent_durations = [h.get('actual_duration', base_duration) 
                              for h in self._optimization_history[-10:]]
            avg_recent = np.mean(recent_durations)
            base_duration = 0.7 * base_duration + 0.3 * avg_recent
        
        return max(5.0, min(300.0, base_duration))  # Clamp between 5s and 5min
    
    async def schedule_optimization(self) -> Optional[Dict]:
        """Schedule next optimization based on available resources."""
        if not self._pending_optimizations:
            return None
        
        # Get current resource status
        scaling_status = self.auto_scaler.get_scaling_status()
        current_load = len(self._active_optimizations)
        max_concurrent = scaling_status['current_workers']
        
        if current_load >= max_concurrent:
            self.logger.debug("All workers busy, cannot schedule new optimization")
            return None
        
        # Sort by priority and submission time
        self._pending_optimizations.sort(
            key=lambda x: (
                {'high': 0, 'normal': 1, 'low': 2}.get(x['priority'], 1),
                x['submitted_at']
            )
        )
        
        # Select next optimization
        selected = self._pending_optimizations.pop(0)
        selected['status'] = 'active'
        selected['started_at'] = time.time()
        
        self._active_optimizations[selected['id']] = selected
        
        self.logger.info(f"Scheduled optimization {selected['id']} (estimated duration: {selected['estimated_duration']:.1f}s)")
        
        return selected
    
    def complete_optimization(self, optimization_id: str, result: Dict[str, Any]):
        """Mark optimization as completed."""
        if optimization_id not in self._active_optimizations:
            self.logger.warning(f"Completing unknown optimization {optimization_id}")
            return
        
        optimization = self._active_optimizations.pop(optimization_id)
        
        # Record completion
        completion_record = {
            **optimization,
            'completed_at': time.time(),
            'actual_duration': time.time() - optimization['started_at'],
            'result': result,
            'status': 'completed'
        }
        
        self._optimization_history.append(completion_record)
        
        # Keep history bounded
        if len(self._optimization_history) > 1000:
            self._optimization_history = self._optimization_history[-500:]
        
        self.logger.info(f"Completed optimization {optimization_id} in {completion_record['actual_duration']:.1f}s")
    
    def get_scheduler_status(self) -> Dict[str, Any]:
        """Get current scheduler status."""
        return {
            'pending_count': len(self._pending_optimizations),
            'active_count': len(self._active_optimizations),
            'total_completed': len(self._optimization_history),
            'queue_priorities': {
                priority: len([o for o in self._pending_optimizations if o['priority'] == priority])
                for priority in ['high', 'normal', 'low']
            },
            'average_wait_time': self._calculate_average_wait_time(),
            'average_duration': self._calculate_average_duration()
        }
    
    def _calculate_average_wait_time(self) -> float:
        """Calculate average wait time for completed optimizations."""
        if not self._optimization_history:
            return 0.0
        
        recent_history = self._optimization_history[-50:]  # Last 50 completions
        wait_times = []
        
        for record in recent_history:
            if 'started_at' in record and 'submitted_at' in record:
                wait_time = record['started_at'] - record['submitted_at']
                wait_times.append(wait_time)
        
        return np.mean(wait_times) if wait_times else 0.0
    
    def _calculate_average_duration(self) -> float:
        """Calculate average optimization duration."""
        if not self._optimization_history:
            return 0.0
        
        recent_history = self._optimization_history[-50:]  # Last 50 completions
        durations = [r.get('actual_duration', 0.0) for r in recent_history]
        
        return np.mean(durations) if durations else 0.0


# Global instances
_auto_scaler = None
_scheduler = None


def get_auto_scaler() -> Optional[AutoScaler]:
    """Get global auto-scaler instance."""
    return _auto_scaler


def get_scheduler() -> Optional[AdaptiveOptimizationScheduler]:
    """Get global scheduler instance."""
    return _scheduler


def initialize_auto_scaling(policy: ScalingPolicy = None):
    """Initialize auto-scaling components."""
    global _auto_scaler, _scheduler
    
    if _auto_scaler is None:
        _auto_scaler = AutoScaler(policy or ScalingPolicy())
        _scheduler = AdaptiveOptimizationScheduler(_auto_scaler)
        
        # Start auto-scaling
        _auto_scaler.start()
        
        logger = logging.getLogger(__name__)
        logger.info("Initialized auto-scaling and adaptive scheduling")
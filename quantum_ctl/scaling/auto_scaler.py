"""
Auto-scaling system for quantum annealing workloads.

This module implements intelligent auto-scaling that dynamically adjusts
quantum computing resources based on workload patterns, performance metrics,
and cost optimization objectives.
"""

import asyncio
import time
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import statistics
from collections import deque
import numpy as np


class ScalingDirection(Enum):
    """Scaling direction options."""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    MAINTAIN = "maintain"


class ScalingTrigger(Enum):
    """Scaling trigger types."""
    CPU_UTILIZATION = "cpu_utilization"
    QUEUE_LENGTH = "queue_length"
    RESPONSE_TIME = "response_time"
    THROUGHPUT = "throughput"
    QUANTUM_SOLVER_AVAILABILITY = "quantum_solver_availability"
    COST_OPTIMIZATION = "cost_optimization"


@dataclass
class ResourceMetrics:
    """Current resource utilization metrics."""
    cpu_utilization: float = 0.0
    memory_utilization: float = 0.0
    quantum_solver_utilization: float = 0.0
    queue_length: int = 0
    average_response_time: float = 0.0
    throughput_rps: float = 0.0
    active_connections: int = 0
    error_rate: float = 0.0
    cost_per_hour: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class ScalingPolicy:
    """Auto-scaling policy configuration."""
    min_instances: int = 1
    max_instances: int = 10
    target_cpu_utilization: float = 70.0
    target_response_time: float = 5.0
    scale_up_threshold: float = 80.0
    scale_down_threshold: float = 30.0
    scale_up_cooldown: int = 300  # seconds
    scale_down_cooldown: int = 600  # seconds
    evaluation_window: int = 300  # seconds
    cost_optimization_enabled: bool = True
    quantum_solver_priority: float = 0.8


@dataclass
class ScalingDecision:
    """Result of a scaling evaluation."""
    direction: ScalingDirection
    target_instances: int
    confidence: float
    triggers: List[ScalingTrigger]
    reasoning: str
    estimated_cost_impact: float
    estimated_performance_impact: float


class QuantumAutoScaler:
    """
    Intelligent auto-scaler for quantum annealing workloads.
    
    Monitors system performance and automatically scales quantum solver
    instances based on workload patterns, performance targets, and cost constraints.
    """
    
    def __init__(
        self,
        policy: ScalingPolicy = None,
        cost_optimizer: Optional[Callable] = None,
        notification_callback: Optional[Callable] = None
    ):
        self.policy = policy or ScalingPolicy()
        self.cost_optimizer = cost_optimizer
        self.notification_callback = notification_callback
        
        self.logger = logging.getLogger(__name__)
        
        # Current state
        self.current_instances = self.policy.min_instances
        self.target_instances = self.policy.min_instances
        
        # Metrics history for decision making
        self.metrics_history: deque = deque(maxlen=200)  # Last 200 metrics points
        self.scaling_history: deque = deque(maxlen=50)   # Last 50 scaling decisions
        
        # Cooldown tracking
        self.last_scale_up_time = 0
        self.last_scale_down_time = 0
        
        # Performance prediction models
        self.performance_predictor = None
        self.workload_patterns = {}
        
        # Cost tracking
        self.cost_history: deque = deque(maxlen=100)
        self.cost_savings_achieved = 0.0
        
        # Control loop state
        self._running = False
        self._control_task = None
        
    async def start_auto_scaling(self, metrics_provider: Callable) -> None:
        """
        Start the auto-scaling control loop.
        
        Args:
            metrics_provider: Async function that returns current ResourceMetrics
        """
        
        if self._running:
            self.logger.warning("Auto-scaler already running")
            return
            
        self._running = True
        self._control_task = asyncio.create_task(
            self._auto_scaling_loop(metrics_provider)
        )
        
        self.logger.info(
            f"Auto-scaler started with policy: min={self.policy.min_instances}, "
            f"max={self.policy.max_instances}, target_cpu={self.policy.target_cpu_utilization}%"
        )
        
    async def stop_auto_scaling(self) -> None:
        """Stop the auto-scaling control loop."""
        
        self._running = False
        
        if self._control_task:
            self._control_task.cancel()
            try:
                await self._control_task
            except asyncio.CancelledError:
                pass
                
        self.logger.info("Auto-scaler stopped")
        
    async def _auto_scaling_loop(self, metrics_provider: Callable) -> None:
        """Main auto-scaling control loop."""
        
        while self._running:
            try:
                # Collect current metrics
                current_metrics = await metrics_provider()
                
                # Store metrics history
                self.metrics_history.append(current_metrics)
                
                # Make scaling decision
                decision = await self._evaluate_scaling_decision(current_metrics)
                
                # Execute scaling if needed
                if decision.direction != ScalingDirection.MAINTAIN:
                    await self._execute_scaling_decision(decision)
                    
                # Update workload patterns
                self._update_workload_patterns(current_metrics)
                
                # Wait for next evaluation
                await asyncio.sleep(30)  # Evaluate every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Auto-scaling loop error: {e}")
                await asyncio.sleep(60)  # Back off on error
                
    async def _evaluate_scaling_decision(
        self,
        current_metrics: ResourceMetrics
    ) -> ScalingDecision:
        """
        Evaluate whether scaling is needed based on current metrics.
        
        Args:
            current_metrics: Current system resource metrics
            
        Returns:
            Scaling decision with reasoning
        """
        
        triggers = []
        scale_up_reasons = []
        scale_down_reasons = []
        
        # Evaluate CPU utilization
        if current_metrics.cpu_utilization > self.policy.scale_up_threshold:
            triggers.append(ScalingTrigger.CPU_UTILIZATION)
            scale_up_reasons.append(f"CPU utilization {current_metrics.cpu_utilization:.1f}% > {self.policy.scale_up_threshold}%")
        elif current_metrics.cpu_utilization < self.policy.scale_down_threshold:
            scale_down_reasons.append(f"CPU utilization {current_metrics.cpu_utilization:.1f}% < {self.policy.scale_down_threshold}%")
            
        # Evaluate response time
        if current_metrics.average_response_time > self.policy.target_response_time * 1.5:
            triggers.append(ScalingTrigger.RESPONSE_TIME)
            scale_up_reasons.append(f"Response time {current_metrics.average_response_time:.2f}s > {self.policy.target_response_time * 1.5:.2f}s")
            
        # Evaluate queue length
        if current_metrics.queue_length > self.current_instances * 10:
            triggers.append(ScalingTrigger.QUEUE_LENGTH)
            scale_up_reasons.append(f"Queue length {current_metrics.queue_length} > {self.current_instances * 10}")
            
        # Evaluate quantum solver availability
        if current_metrics.quantum_solver_utilization > 90.0:
            triggers.append(ScalingTrigger.QUANTUM_SOLVER_AVAILABILITY)
            scale_up_reasons.append(f"Quantum solver utilization {current_metrics.quantum_solver_utilization:.1f}% > 90%")
            
        # Evaluate error rate
        if current_metrics.error_rate > 0.05:  # 5% error rate
            scale_up_reasons.append(f"Error rate {current_metrics.error_rate:.3f} > 0.05")
            
        # Cost optimization evaluation
        cost_decision = await self._evaluate_cost_optimization(current_metrics)
        if cost_decision != ScalingDirection.MAINTAIN:
            triggers.append(ScalingTrigger.COST_OPTIMIZATION)
            
        # Predictive scaling based on patterns
        predicted_load = self._predict_future_load()
        if predicted_load > current_metrics.cpu_utilization * 1.3:
            scale_up_reasons.append(f"Predicted load increase: {predicted_load:.1f}%")
            
        # Check cooldown periods
        current_time = time.time()
        scale_up_cooldown_active = (current_time - self.last_scale_up_time) < self.policy.scale_up_cooldown
        scale_down_cooldown_active = (current_time - self.last_scale_down_time) < self.policy.scale_down_cooldown
        
        # Make scaling decision
        if scale_up_reasons and not scale_up_cooldown_active:
            if self.current_instances < self.policy.max_instances:
                target_instances = min(
                    self.current_instances + self._calculate_scale_up_amount(current_metrics),
                    self.policy.max_instances
                )
                
                return ScalingDecision(
                    direction=ScalingDirection.SCALE_UP,
                    target_instances=target_instances,
                    confidence=self._calculate_decision_confidence(scale_up_reasons),
                    triggers=triggers,
                    reasoning="; ".join(scale_up_reasons),
                    estimated_cost_impact=self._estimate_cost_impact(target_instances),
                    estimated_performance_impact=self._estimate_performance_impact(target_instances)
                )
                
        elif scale_down_reasons and not scale_down_cooldown_active:
            if self.current_instances > self.policy.min_instances:
                # Be more conservative with scale down
                target_instances = max(
                    self.current_instances - 1,
                    self.policy.min_instances
                )
                
                return ScalingDecision(
                    direction=ScalingDirection.SCALE_DOWN,
                    target_instances=target_instances,
                    confidence=self._calculate_decision_confidence(scale_down_reasons),
                    triggers=triggers,
                    reasoning="; ".join(scale_down_reasons),
                    estimated_cost_impact=self._estimate_cost_impact(target_instances),
                    estimated_performance_impact=self._estimate_performance_impact(target_instances)
                )
                
        # Default: maintain current scale
        return ScalingDecision(
            direction=ScalingDirection.MAINTAIN,
            target_instances=self.current_instances,
            confidence=1.0,
            triggers=[],
            reasoning="No scaling triggers activated or in cooldown period",
            estimated_cost_impact=0.0,
            estimated_performance_impact=0.0
        )
        
    async def _evaluate_cost_optimization(
        self,
        current_metrics: ResourceMetrics
    ) -> ScalingDirection:
        """Evaluate cost optimization opportunities."""
        
        if not self.policy.cost_optimization_enabled or not self.cost_optimizer:
            return ScalingDirection.MAINTAIN
            
        try:
            # Get cost optimization recommendation
            cost_recommendation = await self.cost_optimizer(
                current_instances=self.current_instances,
                metrics=current_metrics,
                history=list(self.metrics_history)
            )
            
            return cost_recommendation.get('direction', ScalingDirection.MAINTAIN)
            
        except Exception as e:
            self.logger.warning(f"Cost optimization evaluation failed: {e}")
            return ScalingDirection.MAINTAIN
            
    def _calculate_scale_up_amount(self, current_metrics: ResourceMetrics) -> int:
        """Calculate how many instances to add when scaling up."""
        
        # Base scaling amount
        scale_amount = 1
        
        # Increase scale amount based on severity of metrics
        if current_metrics.cpu_utilization > 95.0:
            scale_amount = 2
        elif current_metrics.queue_length > self.current_instances * 20:
            scale_amount = 2
        elif current_metrics.error_rate > 0.1:
            scale_amount = 2
            
        # Consider predicted load
        predicted_load = self._predict_future_load()
        if predicted_load > 90.0:
            scale_amount = max(scale_amount, 2)
            
        return scale_amount
        
    def _calculate_decision_confidence(self, reasons: List[str]) -> float:
        """Calculate confidence level for scaling decision."""
        
        # Base confidence
        confidence = 0.5
        
        # Increase confidence based on number and type of triggers
        confidence += min(len(reasons) * 0.15, 0.4)
        
        # Historical accuracy bonus
        if self.scaling_history:
            recent_decisions = list(self.scaling_history)[-10:]
            successful_decisions = sum(1 for d in recent_decisions if d.get('success', False))
            accuracy = successful_decisions / len(recent_decisions)
            confidence += accuracy * 0.1
            
        return min(confidence, 1.0)
        
    def _predict_future_load(self) -> float:
        """Predict future load based on historical patterns."""
        
        if len(self.metrics_history) < 10:
            return 0.0
            
        # Simple trend analysis
        recent_cpu = [m.cpu_utilization for m in list(self.metrics_history)[-10:]]
        
        if len(recent_cpu) >= 5:
            # Linear trend prediction
            x = np.arange(len(recent_cpu))
            coeffs = np.polyfit(x, recent_cpu, 1)
            
            # Predict next 3 time periods
            future_x = len(recent_cpu) + 3
            predicted = coeffs[0] * future_x + coeffs[1]
            
            return max(0.0, min(100.0, predicted))
            
        return 0.0
        
    def _estimate_cost_impact(self, target_instances: int) -> float:
        """Estimate cost impact of scaling decision."""
        
        if not self.metrics_history:
            return 0.0
            
        # Get current cost per instance
        recent_metrics = self.metrics_history[-1]
        cost_per_instance = recent_metrics.cost_per_hour / max(self.current_instances, 1)
        
        # Calculate cost difference
        current_cost = cost_per_instance * self.current_instances
        target_cost = cost_per_instance * target_instances
        
        return target_cost - current_cost
        
    def _estimate_performance_impact(self, target_instances: int) -> float:
        """Estimate performance impact of scaling decision."""
        
        # Simple performance model based on instance count
        current_capacity = self.current_instances * 100  # 100% per instance
        target_capacity = target_instances * 100
        
        capacity_change = (target_capacity - current_capacity) / current_capacity
        
        # Assume performance scales linearly with capacity (simplified)
        return capacity_change * 100  # Percentage improvement
        
    async def _execute_scaling_decision(self, decision: ScalingDecision) -> None:
        """
        Execute the scaling decision.
        
        Args:
            decision: The scaling decision to execute
        """
        
        self.logger.info(
            f"Executing scaling decision: {decision.direction.value} to {decision.target_instances} instances. "
            f"Reason: {decision.reasoning}"
        )
        
        try:
            # Update target instances
            old_instances = self.current_instances
            self.target_instances = decision.target_instances
            
            # Simulate scaling execution (in production, this would call cloud APIs)
            if decision.direction == ScalingDirection.SCALE_UP:
                await self._scale_up_instances(decision.target_instances - self.current_instances)
                self.last_scale_up_time = time.time()
            elif decision.direction == ScalingDirection.SCALE_DOWN:
                await self._scale_down_instances(self.current_instances - decision.target_instances)
                self.last_scale_down_time = time.time()
                
            self.current_instances = decision.target_instances
            
            # Record scaling event
            scaling_event = {
                'timestamp': time.time(),
                'old_instances': old_instances,
                'new_instances': self.current_instances,
                'decision': decision,
                'success': True
            }
            
            self.scaling_history.append(scaling_event)
            
            # Notify callback
            if self.notification_callback:
                await self.notification_callback('scaling_executed', scaling_event)
                
            # Update cost tracking
            if decision.estimated_cost_impact < 0:  # Cost savings
                self.cost_savings_achieved += abs(decision.estimated_cost_impact)
                
            self.logger.info(
                f"Scaling completed: {old_instances} -> {self.current_instances} instances. "
                f"Cost impact: ${decision.estimated_cost_impact:.2f}/hour"
            )
            
        except Exception as e:
            self.logger.error(f"Scaling execution failed: {e}")
            
            # Record failed scaling event
            scaling_event = {
                'timestamp': time.time(),
                'old_instances': self.current_instances,
                'new_instances': self.current_instances,  # No change
                'decision': decision,
                'success': False,
                'error': str(e)
            }
            
            self.scaling_history.append(scaling_event)
            
    async def _scale_up_instances(self, count: int) -> None:
        """Scale up by adding instances."""
        
        self.logger.info(f"Adding {count} instances")
        
        # Simulate instance startup time
        await asyncio.sleep(2.0)
        
        # In production, this would:
        # 1. Launch new quantum solver instances
        # 2. Wait for health checks to pass
        # 3. Add instances to load balancer
        # 4. Verify scaling success
        
    async def _scale_down_instances(self, count: int) -> None:
        """Scale down by removing instances."""
        
        self.logger.info(f"Removing {count} instances")
        
        # Simulate graceful shutdown time
        await asyncio.sleep(1.0)
        
        # In production, this would:
        # 1. Remove instances from load balancer
        # 2. Wait for active requests to complete
        # 3. Gracefully shutdown quantum solver instances
        # 4. Verify scaling success
        
    def _update_workload_patterns(self, current_metrics: ResourceMetrics) -> None:
        """Update workload patterns for predictive scaling."""
        
        current_hour = time.localtime().tm_hour
        
        if current_hour not in self.workload_patterns:
            self.workload_patterns[current_hour] = []
            
        # Store CPU utilization by hour
        self.workload_patterns[current_hour].append(current_metrics.cpu_utilization)
        
        # Keep only recent data (last 30 days worth)
        if len(self.workload_patterns[current_hour]) > 30:
            self.workload_patterns[current_hour].pop(0)
            
    def get_scaling_status(self) -> Dict[str, Any]:
        """Get current auto-scaling status and metrics."""
        
        recent_metrics = self.metrics_history[-1] if self.metrics_history else None
        
        return {
            'running': self._running,
            'current_instances': self.current_instances,
            'target_instances': self.target_instances,
            'policy': {
                'min_instances': self.policy.min_instances,
                'max_instances': self.policy.max_instances,
                'target_cpu_utilization': self.policy.target_cpu_utilization,
                'scale_up_threshold': self.policy.scale_up_threshold,
                'scale_down_threshold': self.policy.scale_down_threshold
            },
            'current_metrics': recent_metrics.__dict__ if recent_metrics else None,
            'cooldown_status': {
                'scale_up_cooldown_remaining': max(
                    0, self.policy.scale_up_cooldown - (time.time() - self.last_scale_up_time)
                ),
                'scale_down_cooldown_remaining': max(
                    0, self.policy.scale_down_cooldown - (time.time() - self.last_scale_down_time)
                )
            },
            'scaling_history_count': len(self.scaling_history),
            'cost_savings_achieved': self.cost_savings_achieved,
            'workload_patterns': {
                hour: statistics.mean(values) if values else 0.0
                for hour, values in self.workload_patterns.items()
            }
        }
        
    def get_scaling_history(self) -> List[Dict[str, Any]]:
        """Get recent scaling history."""
        
        return [
            {
                'timestamp': event['timestamp'],
                'old_instances': event['old_instances'],
                'new_instances': event['new_instances'],
                'direction': event['decision'].direction.value,
                'reasoning': event['decision'].reasoning,
                'success': event['success'],
                'cost_impact': event['decision'].estimated_cost_impact,
                'performance_impact': event['decision'].estimated_performance_impact
            }
            for event in list(self.scaling_history)
        ]
        
    async def manual_scale(self, target_instances: int, reason: str = "manual_override") -> bool:
        """
        Manually trigger scaling to specific instance count.
        
        Args:
            target_instances: Desired number of instances
            reason: Reason for manual scaling
            
        Returns:
            True if scaling was successful
        """
        
        if not (self.policy.min_instances <= target_instances <= self.policy.max_instances):
            self.logger.error(f"Target instances {target_instances} outside policy bounds")
            return False
            
        # Create manual scaling decision
        decision = ScalingDecision(
            direction=(
                ScalingDirection.SCALE_UP if target_instances > self.current_instances
                else ScalingDirection.SCALE_DOWN if target_instances < self.current_instances
                else ScalingDirection.MAINTAIN
            ),
            target_instances=target_instances,
            confidence=1.0,
            triggers=[],
            reasoning=f"Manual scaling: {reason}",
            estimated_cost_impact=self._estimate_cost_impact(target_instances),
            estimated_performance_impact=self._estimate_performance_impact(target_instances)
        )
        
        if decision.direction != ScalingDirection.MAINTAIN:
            await self._execute_scaling_decision(decision)
            return True
            
        return False
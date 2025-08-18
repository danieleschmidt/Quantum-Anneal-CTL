"""
Performance optimization system for quantum annealing workloads.

This module implements intelligent performance optimization that automatically
tunes system parameters, manages resource allocation, and optimizes quantum
solver configurations for maximum throughput and efficiency.
"""

import asyncio
import time
import statistics
from typing import Dict, Any, List, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import deque, defaultdict
import numpy as np

try:
    from scipy import optimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class OptimizationStrategy(Enum):
    """Performance optimization strategies."""
    THROUGHPUT_FOCUSED = "throughput_focused"
    LATENCY_FOCUSED = "latency_focused"
    RESOURCE_EFFICIENT = "resource_efficient"
    COST_OPTIMIZED = "cost_optimized"
    BALANCED = "balanced"


class OptimizationPhase(Enum):
    """Optimization phases."""
    BASELINE_MEASUREMENT = "baseline_measurement"
    PARAMETER_EXPLORATION = "parameter_exploration"
    FINE_TUNING = "fine_tuning"
    MONITORING = "monitoring"
    ADAPTATION = "adaptation"


@dataclass
class PerformanceMetrics:
    """Performance metrics for optimization."""
    throughput_qps: float = 0.0
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    cpu_utilization: float = 0.0
    memory_utilization: float = 0.0
    error_rate: float = 0.0
    quantum_solver_efficiency: float = 0.0
    cache_hit_rate: float = 0.0
    cost_per_request: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class OptimizationParameters:
    """System parameters for optimization."""
    # Quantum solver parameters
    quantum_num_reads: int = 1000
    quantum_annealing_time: int = 20
    quantum_chain_strength: float = 1.0
    
    # Caching parameters
    cache_size_mb: int = 512
    cache_ttl_seconds: int = 300
    
    # Concurrency parameters
    max_concurrent_requests: int = 100
    thread_pool_size: int = 10
    
    # Load balancing parameters
    load_balancer_algorithm: str = "round_robin"
    health_check_interval: int = 30
    
    # Resource management
    memory_limit_mb: int = 2048
    cpu_limit_percentage: float = 80.0
    
    # Timeout parameters
    request_timeout_seconds: float = 30.0
    quantum_solver_timeout_seconds: float = 120.0


@dataclass
class OptimizationResult:
    """Result of an optimization run."""
    strategy: OptimizationStrategy
    baseline_metrics: PerformanceMetrics
    optimized_metrics: PerformanceMetrics
    optimal_parameters: OptimizationParameters
    improvement_percentage: float
    optimization_time_seconds: float
    confidence_score: float
    recommendations: List[str]


class PerformanceOptimizer:
    """
    Intelligent performance optimizer for quantum HVAC control systems.
    
    Automatically tunes system parameters to maximize performance based on
    workload patterns, resource constraints, and optimization objectives.
    """
    
    def __init__(
        self,
        strategy: OptimizationStrategy = OptimizationStrategy.BALANCED,
        target_metrics: Optional[Dict[str, float]] = None
    ):
        self.strategy = strategy
        self.target_metrics = target_metrics or self._get_default_targets()
        
        self.logger = logging.getLogger(__name__)
        
        # Current state
        self.current_parameters = OptimizationParameters()
        self.optimization_phase = OptimizationPhase.BASELINE_MEASUREMENT
        
        # Performance history
        self.performance_history: deque = deque(maxlen=1000)
        self.optimization_history: List[OptimizationResult] = []
        
        # Parameter exploration state
        self.parameter_space = self._define_parameter_space()
        self.explored_parameters: Dict[str, List[Tuple[Any, float]]] = defaultdict(list)
        
        # Adaptive learning
        self.parameter_sensitivity = {}
        self.workload_patterns = {}
        
        # Control loop
        self._running = False
        self._optimization_task = None
        
    async def start_optimization(
        self,
        metrics_provider: Callable[[], PerformanceMetrics],
        parameter_applier: Callable[[OptimizationParameters], None]
    ) -> None:
        """
        Start continuous performance optimization.
        
        Args:
            metrics_provider: Function to get current performance metrics
            parameter_applier: Function to apply new parameters
        """
        
        if self._running:
            self.logger.warning("Performance optimizer already running")
            return
            
        self._running = True
        self._optimization_task = asyncio.create_task(
            self._optimization_loop(metrics_provider, parameter_applier)
        )
        
        self.logger.info(f"Performance optimizer started with strategy: {self.strategy.value}")
        
    async def stop_optimization(self) -> None:
        """Stop continuous optimization."""
        
        self._running = False
        
        if self._optimization_task:
            self._optimization_task.cancel()
            try:
                await self._optimization_task
            except asyncio.CancelledError:
                pass
                
        self.logger.info("Performance optimizer stopped")
        
    async def _optimization_loop(
        self,
        metrics_provider: Callable[[], PerformanceMetrics],
        parameter_applier: Callable[[OptimizationParameters], None]
    ) -> None:
        """Main optimization control loop."""
        
        optimization_cycle = 0
        
        while self._running:
            try:
                optimization_cycle += 1
                
                self.logger.info(f"Starting optimization cycle {optimization_cycle}, phase: {self.optimization_phase.value}")
                
                if self.optimization_phase == OptimizationPhase.BASELINE_MEASUREMENT:
                    await self._measure_baseline(metrics_provider)
                    
                elif self.optimization_phase == OptimizationPhase.PARAMETER_EXPLORATION:
                    await self._explore_parameters(metrics_provider, parameter_applier)
                    
                elif self.optimization_phase == OptimizationPhase.FINE_TUNING:
                    await self._fine_tune_parameters(metrics_provider, parameter_applier)
                    
                elif self.optimization_phase == OptimizationPhase.MONITORING:
                    await self._monitor_performance(metrics_provider)
                    
                elif self.optimization_phase == OptimizationPhase.ADAPTATION:
                    await self._adapt_to_workload(metrics_provider, parameter_applier)
                    
                # Wait between optimization cycles
                await asyncio.sleep(self._get_cycle_interval())
                
            except Exception as e:
                self.logger.error(f"Optimization loop error: {e}")
                await asyncio.sleep(60)  # Back off on error
                
    async def _measure_baseline(self, metrics_provider: Callable) -> None:
        """Measure baseline performance with current parameters."""
        
        self.logger.info("Measuring baseline performance")
        
        baseline_samples = []
        
        # Collect baseline samples over 5 minutes
        for _ in range(10):
            metrics = await metrics_provider()
            baseline_samples.append(metrics)
            self.performance_history.append(metrics)
            await asyncio.sleep(30)
            
        # Calculate baseline averages
        self.baseline_metrics = self._calculate_average_metrics(baseline_samples)
        
        self.logger.info(
            f"Baseline established - Throughput: {self.baseline_metrics.throughput_qps:.2f} QPS, "
            f"Latency: {self.baseline_metrics.avg_latency_ms:.2f}ms"
        )
        
        # Move to parameter exploration
        self.optimization_phase = OptimizationPhase.PARAMETER_EXPLORATION
        
    async def _explore_parameters(
        self,
        metrics_provider: Callable,
        parameter_applier: Callable
    ) -> None:
        """Explore parameter space to find optimization opportunities."""
        
        self.logger.info("Exploring parameter space")
        
        # Get parameters to explore based on strategy
        parameters_to_explore = self._get_exploration_parameters()
        
        for param_name, param_values in parameters_to_explore.items():
            self.logger.info(f"Exploring parameter: {param_name}")
            
            for value in param_values:
                try:
                    # Apply parameter change
                    old_value = getattr(self.current_parameters, param_name)
                    setattr(self.current_parameters, param_name, value)
                    
                    await parameter_applier(self.current_parameters)
                    
                    # Wait for system to stabilize
                    await asyncio.sleep(60)
                    
                    # Measure performance
                    test_samples = []
                    for _ in range(5):
                        metrics = await metrics_provider()
                        test_samples.append(metrics)
                        await asyncio.sleep(30)
                        
                    avg_metrics = self._calculate_average_metrics(test_samples)
                    
                    # Calculate improvement score
                    improvement_score = self._calculate_improvement_score(avg_metrics)
                    
                    # Store exploration result
                    self.explored_parameters[param_name].append((value, improvement_score))
                    
                    self.logger.info(
                        f"Parameter {param_name}={value}: improvement_score={improvement_score:.3f}"
                    )
                    
                    # Restore original value for next test
                    setattr(self.current_parameters, param_name, old_value)
                    await parameter_applier(self.current_parameters)
                    await asyncio.sleep(30)  # Stabilize
                    
                except Exception as e:
                    self.logger.error(f"Parameter exploration error for {param_name}={value}: {e}")
                    
        # Move to fine-tuning phase
        self.optimization_phase = OptimizationPhase.FINE_TUNING
        
    async def _fine_tune_parameters(
        self,
        metrics_provider: Callable,
        parameter_applier: Callable
    ) -> None:
        """Fine-tune parameters using optimization algorithms."""
        
        self.logger.info("Fine-tuning parameters")
        
        # Get best parameters from exploration
        best_parameters = self._get_best_explored_parameters()
        
        if SCIPY_AVAILABLE:
            # Use scipy optimization
            optimized_parameters = await self._scipy_optimization(
                best_parameters, metrics_provider, parameter_applier
            )
        else:
            # Use grid search optimization
            optimized_parameters = await self._grid_search_optimization(
                best_parameters, metrics_provider, parameter_applier
            )
            
        # Apply final optimized parameters
        self.current_parameters = optimized_parameters
        await parameter_applier(self.current_parameters)
        
        # Measure final performance
        final_samples = []
        for _ in range(10):
            metrics = await metrics_provider()
            final_samples.append(metrics)
            await asyncio.sleep(30)
            
        final_metrics = self._calculate_average_metrics(final_samples)
        
        # Record optimization result
        optimization_result = OptimizationResult(
            strategy=self.strategy,
            baseline_metrics=self.baseline_metrics,
            optimized_metrics=final_metrics,
            optimal_parameters=optimized_parameters,
            improvement_percentage=self._calculate_improvement_percentage(final_metrics),
            optimization_time_seconds=time.time() - self.optimization_start_time,
            confidence_score=self._calculate_confidence_score(final_metrics),
            recommendations=self._generate_recommendations(final_metrics)
        )
        
        self.optimization_history.append(optimization_result)
        
        self.logger.info(
            f"Optimization completed - Improvement: {optimization_result.improvement_percentage:.1f}%"
        )
        
        # Move to monitoring phase
        self.optimization_phase = OptimizationPhase.MONITORING
        
    async def _monitor_performance(self, metrics_provider: Callable) -> None:
        """Monitor performance and detect degradation."""
        
        # Collect performance samples for 30 minutes
        monitoring_samples = []
        
        for _ in range(60):  # 60 samples over 30 minutes
            metrics = await metrics_provider()
            monitoring_samples.append(metrics)
            self.performance_history.append(metrics)
            
            # Check for performance degradation
            if self._detect_performance_degradation(metrics):
                self.logger.warning("Performance degradation detected, moving to adaptation phase")
                self.optimization_phase = OptimizationPhase.ADAPTATION
                return
                
            await asyncio.sleep(30)
            
        # Update workload patterns
        self._update_workload_patterns(monitoring_samples)
        
        # Stay in monitoring phase if performance is stable
        self.logger.info("Performance monitoring completed - system stable")
        
    async def _adapt_to_workload(
        self,
        metrics_provider: Callable,
        parameter_applier: Callable
    ) -> None:
        """Adapt parameters to changing workload patterns."""
        
        self.logger.info("Adapting to workload changes")
        
        # Analyze recent performance trends
        recent_metrics = list(self.performance_history)[-20:]
        
        if len(recent_metrics) < 10:
            self.optimization_phase = OptimizationPhase.MONITORING
            return
            
        # Detect workload pattern changes
        workload_change = self._detect_workload_change(recent_metrics)
        
        if workload_change:
            # Apply adaptive parameter adjustments
            adaptive_parameters = self._calculate_adaptive_parameters(workload_change)
            
            # Test adaptive parameters
            old_parameters = self.current_parameters
            self.current_parameters = adaptive_parameters
            
            await parameter_applier(self.current_parameters)
            await asyncio.sleep(120)  # Allow adaptation time
            
            # Measure adaptation results
            adaptation_samples = []
            for _ in range(5):
                metrics = await metrics_provider()
                adaptation_samples.append(metrics)
                await asyncio.sleep(30)
                
            adaptation_metrics = self._calculate_average_metrics(adaptation_samples)
            
            # Check if adaptation improved performance
            if self._calculate_improvement_score(adaptation_metrics) > 0:
                self.logger.info("Workload adaptation successful")
            else:
                # Revert to previous parameters
                self.current_parameters = old_parameters
                await parameter_applier(self.current_parameters)
                self.logger.info("Workload adaptation reverted - no improvement")
                
        # Return to monitoring
        self.optimization_phase = OptimizationPhase.MONITORING
        
    def _get_default_targets(self) -> Dict[str, float]:
        """Get default performance targets based on strategy."""
        
        if self.strategy == OptimizationStrategy.THROUGHPUT_FOCUSED:
            return {
                'throughput_qps': 1000.0,
                'avg_latency_ms': 100.0,
                'cpu_utilization': 85.0,
                'error_rate': 0.01
            }
        elif self.strategy == OptimizationStrategy.LATENCY_FOCUSED:
            return {
                'throughput_qps': 500.0,
                'avg_latency_ms': 50.0,
                'p95_latency_ms': 100.0,
                'cpu_utilization': 70.0,
                'error_rate': 0.005
            }
        elif self.strategy == OptimizationStrategy.RESOURCE_EFFICIENT:
            return {
                'throughput_qps': 300.0,
                'avg_latency_ms': 200.0,
                'cpu_utilization': 60.0,
                'memory_utilization': 70.0,
                'error_rate': 0.01
            }
        elif self.strategy == OptimizationStrategy.COST_OPTIMIZED:
            return {
                'throughput_qps': 200.0,
                'avg_latency_ms': 500.0,
                'cpu_utilization': 50.0,
                'cost_per_request': 0.001,
                'error_rate': 0.02
            }
        else:  # BALANCED
            return {
                'throughput_qps': 500.0,
                'avg_latency_ms': 100.0,
                'cpu_utilization': 75.0,
                'memory_utilization': 80.0,
                'error_rate': 0.01
            }
            
    def _define_parameter_space(self) -> Dict[str, List[Any]]:
        """Define parameter space for exploration."""
        
        return {
            'quantum_num_reads': [100, 500, 1000, 2000, 5000],
            'quantum_annealing_time': [5, 10, 20, 50, 100],
            'cache_size_mb': [128, 256, 512, 1024, 2048],
            'cache_ttl_seconds': [60, 300, 600, 1800, 3600],
            'max_concurrent_requests': [50, 100, 200, 500, 1000],
            'thread_pool_size': [5, 10, 20, 50, 100],
            'request_timeout_seconds': [10.0, 30.0, 60.0, 120.0, 300.0],
            'quantum_solver_timeout_seconds': [30.0, 60.0, 120.0, 300.0, 600.0]
        }
        
    def _get_exploration_parameters(self) -> Dict[str, List[Any]]:
        """Get parameters to explore based on current strategy."""
        
        all_params = self._define_parameter_space()
        
        if self.strategy == OptimizationStrategy.THROUGHPUT_FOCUSED:
            return {
                'max_concurrent_requests': all_params['max_concurrent_requests'],
                'thread_pool_size': all_params['thread_pool_size'],
                'cache_size_mb': all_params['cache_size_mb']
            }
        elif self.strategy == OptimizationStrategy.LATENCY_FOCUSED:
            return {
                'quantum_annealing_time': all_params['quantum_annealing_time'],
                'request_timeout_seconds': all_params['request_timeout_seconds'],
                'cache_ttl_seconds': all_params['cache_ttl_seconds']
            }
        else:  # Explore all parameters for other strategies
            return all_params
            
    def _calculate_average_metrics(self, samples: List[PerformanceMetrics]) -> PerformanceMetrics:
        """Calculate average metrics from samples."""
        
        if not samples:
            return PerformanceMetrics()
            
        return PerformanceMetrics(
            throughput_qps=statistics.mean(s.throughput_qps for s in samples),
            avg_latency_ms=statistics.mean(s.avg_latency_ms for s in samples),
            p95_latency_ms=statistics.mean(s.p95_latency_ms for s in samples),
            p99_latency_ms=statistics.mean(s.p99_latency_ms for s in samples),
            cpu_utilization=statistics.mean(s.cpu_utilization for s in samples),
            memory_utilization=statistics.mean(s.memory_utilization for s in samples),
            error_rate=statistics.mean(s.error_rate for s in samples),
            quantum_solver_efficiency=statistics.mean(s.quantum_solver_efficiency for s in samples),
            cache_hit_rate=statistics.mean(s.cache_hit_rate for s in samples),
            cost_per_request=statistics.mean(s.cost_per_request for s in samples)
        )
        
    def _calculate_improvement_score(self, metrics: PerformanceMetrics) -> float:
        """Calculate improvement score based on strategy and targets."""
        
        score = 0.0
        
        for metric_name, target_value in self.target_metrics.items():
            if hasattr(metrics, metric_name):
                current_value = getattr(metrics, metric_name)
                baseline_value = getattr(self.baseline_metrics, metric_name, target_value)
                
                if metric_name in ['throughput_qps', 'cache_hit_rate', 'quantum_solver_efficiency']:
                    # Higher is better
                    improvement = (current_value - baseline_value) / max(baseline_value, 0.001)
                else:
                    # Lower is better
                    improvement = (baseline_value - current_value) / max(baseline_value, 0.001)
                    
                score += improvement * self._get_metric_weight(metric_name)
                
        return score
        
    def _get_metric_weight(self, metric_name: str) -> float:
        """Get weight for metric based on optimization strategy."""
        
        weights = {
            OptimizationStrategy.THROUGHPUT_FOCUSED: {
                'throughput_qps': 0.4,
                'avg_latency_ms': 0.2,
                'cpu_utilization': 0.2,
                'error_rate': 0.2
            },
            OptimizationStrategy.LATENCY_FOCUSED: {
                'avg_latency_ms': 0.3,
                'p95_latency_ms': 0.3,
                'p99_latency_ms': 0.2,
                'throughput_qps': 0.2
            },
            OptimizationStrategy.RESOURCE_EFFICIENT: {
                'cpu_utilization': 0.3,
                'memory_utilization': 0.3,
                'throughput_qps': 0.2,
                'avg_latency_ms': 0.2
            },
            OptimizationStrategy.COST_OPTIMIZED: {
                'cost_per_request': 0.4,
                'cpu_utilization': 0.2,
                'throughput_qps': 0.2,
                'error_rate': 0.2
            },
            OptimizationStrategy.BALANCED: {
                'throughput_qps': 0.25,
                'avg_latency_ms': 0.25,
                'cpu_utilization': 0.25,
                'error_rate': 0.25
            }
        }
        
        strategy_weights = weights.get(self.strategy, weights[OptimizationStrategy.BALANCED])
        return strategy_weights.get(metric_name, 0.1)
        
    def get_optimization_status(self) -> Dict[str, Any]:
        """Get current optimization status."""
        
        recent_metrics = self.performance_history[-1] if self.performance_history else None
        
        return {
            'running': self._running,
            'strategy': self.strategy.value,
            'optimization_phase': self.optimization_phase.value,
            'current_parameters': self.current_parameters.__dict__,
            'target_metrics': self.target_metrics,
            'current_metrics': recent_metrics.__dict__ if recent_metrics else None,
            'optimization_history_count': len(self.optimization_history),
            'performance_samples_collected': len(self.performance_history),
            'explored_parameters_count': {
                param: len(values) for param, values in self.explored_parameters.items()
            }
        }
        
    def get_optimization_results(self) -> List[Dict[str, Any]]:
        """Get optimization results history."""
        
        return [
            {
                'strategy': result.strategy.value,
                'improvement_percentage': result.improvement_percentage,
                'optimization_time_seconds': result.optimization_time_seconds,
                'confidence_score': result.confidence_score,
                'recommendations': result.recommendations,
                'baseline_throughput': result.baseline_metrics.throughput_qps,
                'optimized_throughput': result.optimized_metrics.throughput_qps,
                'baseline_latency': result.baseline_metrics.avg_latency_ms,
                'optimized_latency': result.optimized_metrics.avg_latency_ms
            }
            for result in self.optimization_history
        ]
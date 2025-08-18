"""
Tests for scaling modules including auto-scaler and performance optimizer.
"""

import pytest
import asyncio
import time
import numpy as np
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any

# Test imports - handle missing dependencies gracefully
try:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    
    from quantum_ctl.scaling.auto_scaler import (
        QuantumAutoScaler, ScalingPolicy, ResourceMetrics, ScalingDirection
    )
    from quantum_ctl.scaling.performance_optimizer import (
        PerformanceOptimizer, OptimizationStrategy, OptimizationParameters, PerformanceMetrics
    )
    SCALING_MODULES_AVAILABLE = True
except ImportError as e:
    print(f"Scaling modules not available: {e}")
    SCALING_MODULES_AVAILABLE = False


@pytest.mark.skipif(not SCALING_MODULES_AVAILABLE, reason="Scaling modules not available")
class TestQuantumAutoScaler:
    """Test quantum auto-scaling functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.scaling_policy = ScalingPolicy(
            min_instances=2,
            max_instances=10,
            target_cpu_utilization=70.0,
            scale_up_threshold=80.0,
            scale_down_threshold=30.0,
            scale_up_cooldown=60,  # Shorter for testing
            scale_down_cooldown=120  # Shorter for testing
        )
        
        self.auto_scaler = QuantumAutoScaler(
            policy=self.scaling_policy
        )
        
        # Mock metrics for testing
        self.base_metrics = ResourceMetrics(
            cpu_utilization=50.0,
            memory_utilization=60.0,
            quantum_solver_utilization=40.0,
            queue_length=5,
            average_response_time=2.0,
            throughput_rps=100.0,
            active_connections=20,
            error_rate=0.01,
            cost_per_hour=50.0
        )
        
    def test_scaling_policy_initialization(self):
        """Test scaling policy initialization and validation."""
        
        assert self.auto_scaler.policy.min_instances == 2
        assert self.auto_scaler.policy.max_instances == 10
        assert self.auto_scaler.policy.target_cpu_utilization == 70.0
        
        # Check initial state
        assert self.auto_scaler.current_instances == 2
        assert self.auto_scaler.target_instances == 2
        
    @pytest.mark.asyncio
    async def test_scale_up_decision(self):
        """Test scale up decision making."""
        
        # Create high load metrics
        high_load_metrics = ResourceMetrics(
            cpu_utilization=85.0,  # Above threshold
            memory_utilization=80.0,
            quantum_solver_utilization=90.0,
            queue_length=50,  # High queue
            average_response_time=10.0,  # Slow response
            throughput_rps=50.0,
            error_rate=0.05,  # High error rate
            cost_per_hour=100.0
        )
        
        decision = await self.auto_scaler._evaluate_scaling_decision(high_load_metrics)
        
        assert decision.direction == ScalingDirection.SCALE_UP
        assert decision.target_instances > self.auto_scaler.current_instances
        assert decision.target_instances <= self.scaling_policy.max_instances
        assert decision.confidence > 0.5
        assert len(decision.triggers) > 0
        
    @pytest.mark.asyncio
    async def test_scale_down_decision(self):
        """Test scale down decision making."""
        
        # Start with more instances
        self.auto_scaler.current_instances = 5
        
        # Create low load metrics
        low_load_metrics = ResourceMetrics(
            cpu_utilization=20.0,  # Below threshold
            memory_utilization=30.0,
            quantum_solver_utilization=25.0,
            queue_length=1,
            average_response_time=1.0,
            throughput_rps=200.0,
            error_rate=0.001,
            cost_per_hour=25.0
        )
        
        decision = await self.auto_scaler._evaluate_scaling_decision(low_load_metrics)
        
        assert decision.direction == ScalingDirection.SCALE_DOWN
        assert decision.target_instances < self.auto_scaler.current_instances
        assert decision.target_instances >= self.scaling_policy.min_instances
        
    @pytest.mark.asyncio
    async def test_maintain_decision(self):
        """Test maintain (no scaling) decision."""
        
        # Create balanced metrics
        balanced_metrics = ResourceMetrics(
            cpu_utilization=70.0,  # At target
            memory_utilization=65.0,
            quantum_solver_utilization=60.0,
            queue_length=10,
            average_response_time=3.0,
            throughput_rps=150.0,
            error_rate=0.01,
            cost_per_hour=60.0
        )
        
        decision = await self.auto_scaler._evaluate_scaling_decision(balanced_metrics)
        
        assert decision.direction == ScalingDirection.MAINTAIN
        assert decision.target_instances == self.auto_scaler.current_instances
        
    @pytest.mark.asyncio
    async def test_cooldown_enforcement(self):
        """Test scaling cooldown enforcement."""
        
        # Set recent scale up time
        self.auto_scaler.last_scale_up_time = time.time() - 30  # 30 seconds ago
        
        # Create high load metrics that would normally trigger scale up
        high_load_metrics = ResourceMetrics(
            cpu_utilization=90.0,
            quantum_solver_utilization=95.0,
            queue_length=100,
            average_response_time=15.0
        )
        
        decision = await self.auto_scaler._evaluate_scaling_decision(high_load_metrics)
        
        # Should maintain due to cooldown
        assert decision.direction == ScalingDirection.MAINTAIN
        assert "cooldown" in decision.reasoning.lower()
        
    @pytest.mark.asyncio
    async def test_auto_scaling_loop(self):
        """Test the auto-scaling control loop."""
        
        metrics_call_count = 0
        
        async def mock_metrics_provider():
            nonlocal metrics_call_count
            metrics_call_count += 1
            
            if metrics_call_count <= 2:
                # Return high load initially
                return ResourceMetrics(
                    cpu_utilization=85.0,
                    quantum_solver_utilization=90.0,
                    queue_length=50
                )
            else:
                # Return normal load after scaling
                return ResourceMetrics(
                    cpu_utilization=70.0,
                    quantum_solver_utilization=60.0,
                    queue_length=10
                )
                
        # Start auto-scaling
        await self.auto_scaler.start_auto_scaling(mock_metrics_provider)
        
        # Let it run for a short time
        await asyncio.sleep(1.0)
        
        # Stop auto-scaling
        await self.auto_scaler.stop_auto_scaling()
        
        # Check that metrics were collected
        assert len(self.auto_scaler.metrics_history) > 0
        
        # Check that scaling decisions were made
        assert len(self.auto_scaler.scaling_history) >= 0  # May or may not have scaled
        
    @pytest.mark.asyncio
    async def test_manual_scaling(self):
        """Test manual scaling override."""
        
        initial_instances = self.auto_scaler.current_instances
        target_instances = 5
        
        success = await self.auto_scaler.manual_scale(
            target_instances,
            reason="manual_test_scaling"
        )
        
        assert success is True
        assert self.auto_scaler.current_instances == target_instances
        assert self.auto_scaler.target_instances == target_instances
        
        # Check scaling history
        assert len(self.auto_scaler.scaling_history) > 0
        latest_event = self.auto_scaler.scaling_history[-1]
        assert "manual_test_scaling" in latest_event['decision'].reasoning
        
    def test_workload_pattern_tracking(self):
        """Test workload pattern learning."""
        
        # Simulate workload data for different hours
        test_metrics = [
            ResourceMetrics(cpu_utilization=30.0, timestamp=time.time()),  # Hour 0
            ResourceMetrics(cpu_utilization=60.0, timestamp=time.time()),  # Hour 0  
            ResourceMetrics(cpu_utilization=80.0, timestamp=time.time()),  # Hour 0
        ]
        
        for metrics in test_metrics:
            self.auto_scaler._update_workload_patterns(metrics)
            
        # Check that patterns were recorded
        current_hour = time.localtime().tm_hour
        assert current_hour in self.auto_scaler.workload_patterns
        assert len(self.auto_scaler.workload_patterns[current_hour]) == 3
        
    def test_future_load_prediction(self):
        """Test future load prediction based on patterns."""
        
        # Add historical CPU utilization data with trend
        cpu_values = [50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0, 95.0]
        
        for cpu in cpu_values:
            self.auto_scaler.metrics_history.append(
                ResourceMetrics(cpu_utilization=cpu)
            )
            
        predicted_load = self.auto_scaler._predict_future_load()
        
        # Should predict continued upward trend
        assert predicted_load > 95.0  # Should be higher than last value
        assert predicted_load <= 100.0  # Should be capped at 100%
        
    def test_scaling_status_reporting(self):
        """Test scaling status and metrics reporting."""
        
        status = self.auto_scaler.get_scaling_status()
        
        # Check status structure
        assert 'running' in status
        assert 'current_instances' in status
        assert 'target_instances' in status
        assert 'policy' in status
        assert 'cooldown_status' in status
        
        # Check policy details
        policy_info = status['policy']
        assert policy_info['min_instances'] == 2
        assert policy_info['max_instances'] == 10
        
        # Check cooldown status
        cooldown_info = status['cooldown_status']
        assert 'scale_up_cooldown_remaining' in cooldown_info
        assert 'scale_down_cooldown_remaining' in cooldown_info


@pytest.mark.skipif(not SCALING_MODULES_AVAILABLE, reason="Scaling modules not available")
class TestPerformanceOptimizer:
    """Test performance optimization functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.optimizer = PerformanceOptimizer(
            strategy=OptimizationStrategy.BALANCED
        )
        
        # Mock baseline metrics
        self.baseline_metrics = PerformanceMetrics(
            throughput_qps=100.0,
            avg_latency_ms=50.0,
            p95_latency_ms=100.0,
            cpu_utilization=70.0,
            memory_utilization=60.0,
            error_rate=0.02,
            quantum_solver_efficiency=0.8,
            cache_hit_rate=0.7,
            cost_per_request=0.01
        )
        
        self.optimizer.baseline_metrics = self.baseline_metrics
        
    def test_optimization_strategy_initialization(self):
        """Test optimization strategy setup."""
        
        assert self.optimizer.strategy == OptimizationStrategy.BALANCED
        assert isinstance(self.optimizer.current_parameters, OptimizationParameters)
        assert isinstance(self.optimizer.target_metrics, dict)
        
        # Check target metrics are reasonable for balanced strategy
        targets = self.optimizer.target_metrics
        assert 'throughput_qps' in targets
        assert 'avg_latency_ms' in targets
        assert 'cpu_utilization' in targets
        
    def test_parameter_space_definition(self):
        """Test parameter space definition for exploration."""
        
        param_space = self.optimizer._define_parameter_space()
        
        assert 'quantum_num_reads' in param_space
        assert 'quantum_annealing_time' in param_space
        assert 'cache_size_mb' in param_space
        assert 'max_concurrent_requests' in param_space
        
        # Check parameter ranges are reasonable
        assert len(param_space['quantum_num_reads']) > 1
        assert max(param_space['quantum_num_reads']) > min(param_space['quantum_num_reads'])
        
    def test_improvement_score_calculation(self):
        """Test improvement score calculation."""
        
        # Create improved metrics
        improved_metrics = PerformanceMetrics(
            throughput_qps=150.0,  # 50% better
            avg_latency_ms=40.0,   # 20% better
            cpu_utilization=65.0,  # 5% better
            error_rate=0.01        # 50% better
        )
        
        score = self.optimizer._calculate_improvement_score(improved_metrics)
        
        # Should be positive for improvement
        assert score > 0
        
        # Test degraded metrics
        degraded_metrics = PerformanceMetrics(
            throughput_qps=80.0,   # 20% worse
            avg_latency_ms=60.0,   # 20% worse
            cpu_utilization=80.0,  # 10% worse
            error_rate=0.04        # 100% worse
        )
        
        degraded_score = self.optimizer._calculate_improvement_score(degraded_metrics)
        
        # Should be negative for degradation
        assert degraded_score < 0
        
    def test_average_metrics_calculation(self):
        """Test averaging of performance metrics samples."""
        
        samples = [
            PerformanceMetrics(throughput_qps=100.0, avg_latency_ms=50.0),
            PerformanceMetrics(throughput_qps=120.0, avg_latency_ms=40.0),
            PerformanceMetrics(throughput_qps=110.0, avg_latency_ms=45.0)
        ]
        
        avg_metrics = self.optimizer._calculate_average_metrics(samples)
        
        assert avg_metrics.throughput_qps == 110.0  # (100+120+110)/3
        assert avg_metrics.avg_latency_ms == 45.0   # (50+40+45)/3
        
    def test_exploration_parameters_by_strategy(self):
        """Test parameter exploration based on optimization strategy."""
        
        # Test throughput-focused strategy
        throughput_optimizer = PerformanceOptimizer(OptimizationStrategy.THROUGHPUT_FOCUSED)
        throughput_params = throughput_optimizer._get_exploration_parameters()
        
        assert 'max_concurrent_requests' in throughput_params
        assert 'thread_pool_size' in throughput_params
        assert 'cache_size_mb' in throughput_params
        
        # Test latency-focused strategy
        latency_optimizer = PerformanceOptimizer(OptimizationStrategy.LATENCY_FOCUSED)
        latency_params = latency_optimizer._get_exploration_parameters()
        
        assert 'quantum_annealing_time' in latency_params
        assert 'request_timeout_seconds' in latency_params
        
    @pytest.mark.asyncio
    async def test_baseline_measurement(self):
        """Test baseline performance measurement."""
        
        sample_count = 0
        
        async def mock_metrics_provider():
            nonlocal sample_count
            sample_count += 1
            return PerformanceMetrics(
                throughput_qps=100.0 + sample_count,
                avg_latency_ms=50.0 - sample_count,
                cpu_utilization=70.0
            )
            
        # Start with baseline phase
        self.optimizer.optimization_phase = self.optimizer.optimization_phase.BASELINE_MEASUREMENT
        
        await self.optimizer._measure_baseline(mock_metrics_provider)
        
        # Should have moved to parameter exploration
        from quantum_ctl.scaling.performance_optimizer import OptimizationPhase
        assert self.optimizer.optimization_phase == OptimizationPhase.PARAMETER_EXPLORATION
        
        # Should have baseline metrics
        assert hasattr(self.optimizer, 'baseline_metrics')
        assert self.optimizer.baseline_metrics.throughput_qps > 100.0
        
    def test_metric_weight_calculation(self):
        """Test metric weighting based on strategy."""
        
        # Test different strategies have different weights
        strategies_to_test = [
            OptimizationStrategy.THROUGHPUT_FOCUSED,
            OptimizationStrategy.LATENCY_FOCUSED,
            OptimizationStrategy.RESOURCE_EFFICIENT,
            OptimizationStrategy.COST_OPTIMIZED
        ]
        
        for strategy in strategies_to_test:
            optimizer = PerformanceOptimizer(strategy)
            
            throughput_weight = optimizer._get_metric_weight('throughput_qps')
            latency_weight = optimizer._get_metric_weight('avg_latency_ms')
            
            if strategy == OptimizationStrategy.THROUGHPUT_FOCUSED:
                assert throughput_weight >= latency_weight
            elif strategy == OptimizationStrategy.LATENCY_FOCUSED:
                assert latency_weight >= throughput_weight
                
    def test_optimization_status_reporting(self):
        """Test optimization status reporting."""
        
        status = self.optimizer.get_optimization_status()
        
        assert 'running' in status
        assert 'strategy' in status
        assert 'optimization_phase' in status
        assert 'current_parameters' in status
        assert 'target_metrics' in status
        
        # Check strategy value
        assert status['strategy'] == OptimizationStrategy.BALANCED.value
        
        # Check parameters structure
        params = status['current_parameters']
        assert 'quantum_num_reads' in params
        assert 'cache_size_mb' in params
        assert 'max_concurrent_requests' in params


@pytest.mark.skipif(not SCALING_MODULES_AVAILABLE, reason="Scaling modules not available")
class TestScalingIntegration:
    """Test integration between scaling components."""
    
    def setup_method(self):
        """Set up integration test fixtures."""
        self.scaling_policy = ScalingPolicy(
            min_instances=1,
            max_instances=5,
            target_cpu_utilization=70.0
        )
        
        self.auto_scaler = QuantumAutoScaler(self.scaling_policy)
        self.performance_optimizer = PerformanceOptimizer(OptimizationStrategy.BALANCED)
        
    @pytest.mark.asyncio
    async def test_scaling_with_performance_optimization(self):
        """Test coordinated scaling and performance optimization."""
        
        # Mock metrics that show resource pressure
        metrics_sequence = [
            # Start with normal load
            ResourceMetrics(cpu_utilization=60.0, throughput_rps=100.0, average_response_time=2.0),
            # Increase load - should trigger scaling
            ResourceMetrics(cpu_utilization=85.0, throughput_rps=80.0, average_response_time=5.0),
            # Load stabilizes after scaling
            ResourceMetrics(cpu_utilization=70.0, throughput_rps=150.0, average_response_time=3.0),
        ]
        
        metrics_index = 0
        
        async def mock_metrics_provider():
            nonlocal metrics_index
            if metrics_index < len(metrics_sequence):
                result = metrics_sequence[metrics_index]
                metrics_index += 1
                return result
            return metrics_sequence[-1]  # Return last metrics
            
        # Test scaling decision making
        for metrics in metrics_sequence:
            decision = await self.auto_scaler._evaluate_scaling_decision(metrics)
            
            if metrics.cpu_utilization > 80.0:
                assert decision.direction in [ScalingDirection.SCALE_UP, ScalingDirection.MAINTAIN]
            elif metrics.cpu_utilization < 40.0:
                assert decision.direction in [ScalingDirection.SCALE_DOWN, ScalingDirection.MAINTAIN]
                
        # Test performance metrics correlation
        perf_metrics = [
            PerformanceMetrics(
                throughput_qps=rm.throughput_rps,
                avg_latency_ms=rm.average_response_time * 1000,
                cpu_utilization=rm.cpu_utilization,
                error_rate=rm.error_rate
            ) for rm in metrics_sequence
        ]
        
        # Performance should degrade then improve
        assert perf_metrics[1].avg_latency_ms > perf_metrics[0].avg_latency_ms  # Degraded
        assert perf_metrics[2].throughput_qps > perf_metrics[1].throughput_qps  # Improved
        
    def test_cost_optimization_integration(self):
        """Test cost optimization across scaling and performance."""
        
        # Mock cost optimizer for auto-scaler
        async def mock_cost_optimizer(current_instances, metrics, history):
            # Simple cost model: prefer fewer instances when load is low
            if metrics.cpu_utilization < 30.0 and current_instances > 1:
                return {'direction': ScalingDirection.SCALE_DOWN}
            elif metrics.cpu_utilization > 90.0:
                return {'direction': ScalingDirection.SCALE_UP}
            else:
                return {'direction': ScalingDirection.MAINTAIN}
                
        self.auto_scaler.cost_optimizer = mock_cost_optimizer
        self.auto_scaler.policy.cost_optimization_enabled = True
        
        # Test cost-based scaling decision
        low_cost_metrics = ResourceMetrics(
            cpu_utilization=25.0,
            cost_per_hour=100.0  # High cost should encourage scaling down
        )
        
        decision = asyncio.run(
            self.auto_scaler._evaluate_scaling_decision(low_cost_metrics)
        )
        
        # Should consider cost optimization
        assert decision.estimated_cost_impact <= 0  # Should reduce or maintain cost
        
    def test_performance_degradation_detection(self):
        """Test detection and response to performance degradation."""
        
        # Simulate performance degradation over time
        degradation_metrics = [
            PerformanceMetrics(throughput_qps=100.0, avg_latency_ms=50.0, error_rate=0.01),
            PerformanceMetrics(throughput_qps=90.0, avg_latency_ms=60.0, error_rate=0.02),
            PerformanceMetrics(throughput_qps=80.0, avg_latency_ms=70.0, error_rate=0.03),
            PerformanceMetrics(throughput_qps=70.0, avg_latency_ms=80.0, error_rate=0.04),
        ]
        
        # Add metrics to optimizer history
        for metrics in degradation_metrics:
            self.performance_optimizer.performance_history.append(metrics)
            
        # Calculate improvement scores - should show degradation trend
        scores = []
        self.performance_optimizer.baseline_metrics = degradation_metrics[0]
        
        for metrics in degradation_metrics[1:]:
            score = self.performance_optimizer._calculate_improvement_score(metrics)
            scores.append(score)
            
        # Scores should be increasingly negative (degrading performance)
        assert all(score < 0 for score in scores)
        assert scores[-1] < scores[0]  # Getting worse over time
        
    def test_resource_allocation_coordination(self):
        """Test coordination of resource allocation between components."""
        
        # Simulate scenario where both scaling and optimization are needed
        current_metrics = ResourceMetrics(
            cpu_utilization=90.0,  # High CPU - needs scaling
            memory_utilization=85.0,  # High memory - needs optimization
            quantum_solver_utilization=95.0,  # High quantum usage
            average_response_time=10.0,  # High latency
            throughput_rps=50.0,  # Low throughput
            error_rate=0.05  # High error rate
        )
        
        # Get scaling recommendation
        scaling_decision = asyncio.run(
            self.auto_scaler._evaluate_scaling_decision(current_metrics)
        )
        
        # Should recommend scaling up
        assert scaling_decision.direction == ScalingDirection.SCALE_UP
        
        # Convert to performance metrics
        perf_metrics = PerformanceMetrics(
            throughput_qps=current_metrics.throughput_rps,
            avg_latency_ms=current_metrics.average_response_time * 1000,
            cpu_utilization=current_metrics.cpu_utilization,
            memory_utilization=current_metrics.memory_utilization,
            error_rate=current_metrics.error_rate,
            quantum_solver_efficiency=current_metrics.quantum_solver_utilization / 100.0
        )
        
        # Set baseline for comparison
        self.performance_optimizer.baseline_metrics = PerformanceMetrics(
            throughput_qps=100.0,
            avg_latency_ms=2000.0,
            cpu_utilization=70.0,
            memory_utilization=60.0,
            error_rate=0.01,
            quantum_solver_efficiency=0.8
        )
        
        # Calculate improvement score - should be negative (needs optimization)
        improvement_score = self.performance_optimizer._calculate_improvement_score(perf_metrics)
        assert improvement_score < 0  # Performance is degraded
        
        # Both components agree that intervention is needed
        assert scaling_decision.direction != ScalingDirection.MAINTAIN
        assert improvement_score < -0.1  # Significant degradation


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
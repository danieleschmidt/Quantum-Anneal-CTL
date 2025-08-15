"""
Tests for the self-healing pipeline guard functionality.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch

from quantum_ctl.pipeline_guard.guard import PipelineGuard, PipelineStatus
from quantum_ctl.pipeline_guard.health_monitor import HealthMonitor
from quantum_ctl.pipeline_guard.recovery_manager import RecoveryManager
from quantum_ctl.pipeline_guard.circuit_breaker import CircuitBreaker, CircuitState
from quantum_ctl.pipeline_guard.metrics_collector import MetricsCollector
from quantum_ctl.pipeline_guard.security_monitor import SecurityMonitor
from quantum_ctl.pipeline_guard.performance_optimizer import PerformanceOptimizer


class TestPipelineGuard:
    """Test suite for PipelineGuard class."""
    
    @pytest.fixture
    def pipeline_guard(self):
        """Create a pipeline guard instance for testing."""
        return PipelineGuard(check_interval=0.1)  # Fast interval for testing
        
    @pytest.fixture
    def mock_health_check(self):
        """Mock health check function."""
        return Mock(return_value=True)
        
    @pytest.fixture
    def mock_recovery_action(self):
        """Mock recovery action function."""
        return AsyncMock(return_value=True)
        
    def test_pipeline_guard_initialization(self, pipeline_guard):
        """Test pipeline guard initialization."""
        assert pipeline_guard.status == PipelineStatus.HEALTHY
        assert pipeline_guard.check_interval == 0.1
        assert len(pipeline_guard.components) == 0
        
    def test_register_component(self, pipeline_guard, mock_health_check, mock_recovery_action):
        """Test component registration."""
        pipeline_guard.register_component(
            name="test_component",
            health_check=mock_health_check,
            recovery_action=mock_recovery_action,
            critical=True
        )
        
        assert "test_component" in pipeline_guard.components
        component = pipeline_guard.components["test_component"]
        assert component.name == "test_component"
        assert component.critical is True
        assert component.health_check == mock_health_check
        assert component.recovery_action == mock_recovery_action
        
    @pytest.mark.asyncio
    async def test_pipeline_monitoring(self, pipeline_guard, mock_health_check):
        """Test pipeline monitoring functionality."""
        pipeline_guard.register_component(
            name="test_component",
            health_check=mock_health_check
        )
        
        # Start monitoring
        pipeline_guard.start()
        
        # Wait for a few monitoring cycles
        await asyncio.sleep(0.3)
        
        # Check status
        status = pipeline_guard.get_status()
        assert status["pipeline_status"] == "healthy"
        assert status["total_components"] == 1
        assert status["healthy_components"] == 1
        
        # Stop monitoring
        await pipeline_guard.stop()
        
    @pytest.mark.asyncio
    async def test_component_failure_detection(self, pipeline_guard):
        """Test detection of component failures."""
        failing_health_check = Mock(return_value=False)
        
        pipeline_guard.register_component(
            name="failing_component",
            health_check=failing_health_check,
            critical=True
        )
        
        pipeline_guard.start()
        await asyncio.sleep(0.3)
        
        status = pipeline_guard.get_status()
        assert status["pipeline_status"] in ["critical", "degraded"]
        
        await pipeline_guard.stop()
        
    @pytest.mark.asyncio
    async def test_component_recovery(self, pipeline_guard):
        """Test component recovery functionality."""
        health_status = [False, False, True]  # Fail twice, then recover
        health_call_count = 0
        
        def dynamic_health_check():
            nonlocal health_call_count
            result = health_status[min(health_call_count, len(health_status) - 1)]
            health_call_count += 1
            return result
            
        recovery_action = AsyncMock(return_value=True)
        
        pipeline_guard.register_component(
            name="recovering_component",
            health_check=dynamic_health_check,
            recovery_action=recovery_action
        )
        
        pipeline_guard.start()
        await asyncio.sleep(0.5)  # Allow time for recovery attempts
        
        # Recovery action should have been called
        assert recovery_action.called
        
        await pipeline_guard.stop()


class TestHealthMonitor:
    """Test suite for HealthMonitor class."""
    
    @pytest.fixture
    def health_monitor(self):
        """Create a health monitor instance for testing."""
        return HealthMonitor()
        
    def test_health_monitor_initialization(self, health_monitor):
        """Test health monitor initialization."""
        assert len(health_monitor.components) == 0
        assert len(health_monitor.health_checks) == 0
        
    def test_register_component(self, health_monitor):
        """Test component registration in health monitor."""
        health_check = Mock(return_value=True)
        
        health_monitor.register_component("test_component", health_check)
        
        assert "test_component" in health_monitor.components
        assert "test_component" in health_monitor.health_checks
        assert health_monitor.health_checks["test_component"] == health_check
        
    @pytest.mark.asyncio
    async def test_check_component_health(self, health_monitor):
        """Test individual component health checking."""
        health_check = Mock(return_value=True)
        health_monitor.register_component("test_component", health_check)
        
        metric = await health_monitor.check_component_health("test_component")
        
        assert metric.component == "test_component"
        assert metric.healthy is True
        assert metric.response_time > 0
        assert health_check.called
        
    @pytest.mark.asyncio
    async def test_check_all_components(self, health_monitor):
        """Test checking all components concurrently."""
        health_check_1 = Mock(return_value=True)
        health_check_2 = Mock(return_value=False)
        
        health_monitor.register_component("component_1", health_check_1)
        health_monitor.register_component("component_2", health_check_2)
        
        results = await health_monitor.check_all_components()
        
        assert len(results) == 2
        assert "component_1" in results
        assert "component_2" in results
        assert results["component_1"].healthy is True
        assert results["component_2"].healthy is False
        
    def test_get_component_status(self, health_monitor):
        """Test getting component status information."""
        health_check = Mock(return_value=True)
        health_monitor.register_component("test_component", health_check)
        
        # Simulate some history
        component = health_monitor.components["test_component"]
        component.success_count = 8
        component.failure_count = 2
        
        status = health_monitor.get_component_status("test_component")
        
        assert status["name"] == "test_component"
        assert status["uptime_percentage"] == 80.0  # 8/10 * 100


class TestRecoveryManager:
    """Test suite for RecoveryManager class."""
    
    @pytest.fixture
    def recovery_manager(self):
        """Create a recovery manager instance for testing."""
        return RecoveryManager(max_retries=2, retry_delay=0.1)
        
    def test_recovery_manager_initialization(self, recovery_manager):
        """Test recovery manager initialization."""
        assert recovery_manager.max_retries == 2
        assert recovery_manager.retry_delay == 0.1
        assert len(recovery_manager.recovery_actions) == 0
        
    def test_register_recovery(self, recovery_manager):
        """Test recovery action registration."""
        recovery_action = Mock(return_value=True)
        
        recovery_manager.register_recovery("test_component", recovery_action)
        
        assert "test_component" in recovery_manager.recovery_actions
        assert recovery_manager.recovery_actions["test_component"] == recovery_action
        
    @pytest.mark.asyncio
    async def test_successful_recovery(self, recovery_manager):
        """Test successful component recovery."""
        recovery_action = AsyncMock(return_value=True)
        recovery_manager.register_recovery("test_component", recovery_action)
        
        success = await recovery_manager.recover_component("test_component")
        
        assert success is True
        assert recovery_action.called
        
    @pytest.mark.asyncio
    async def test_failed_recovery_with_retries(self, recovery_manager):
        """Test failed recovery with retry mechanism."""
        recovery_action = AsyncMock(return_value=False)
        recovery_manager.register_recovery("test_component", recovery_action)
        
        success = await recovery_manager.recover_component("test_component")
        
        assert success is False
        assert recovery_action.call_count == 2  # max_retries
        
    @pytest.mark.asyncio
    async def test_recovery_with_eventual_success(self, recovery_manager):
        """Test recovery that succeeds after initial failures."""
        call_count = 0
        
        async def flaky_recovery():
            nonlocal call_count
            call_count += 1
            return call_count >= 2  # Succeed on second attempt
            
        recovery_manager.register_recovery("test_component", flaky_recovery)
        
        success = await recovery_manager.recover_component("test_component")
        
        assert success is True
        assert call_count == 2
        
    def test_get_recovery_statistics(self, recovery_manager):
        """Test recovery statistics collection."""
        # Simulate some recovery history
        recovery_manager.recovery_history = [
            Mock(status="success"),
            Mock(status="failed"),
            Mock(status="success")
        ]
        
        stats = recovery_manager.get_recovery_statistics()
        
        assert stats["total_attempts"] == 3
        assert stats["successful_attempts"] == 2
        assert stats["success_rate"] == 200/3  # 2/3 * 100


class TestCircuitBreaker:
    """Test suite for CircuitBreaker class."""
    
    @pytest.fixture
    def circuit_breaker(self):
        """Create a circuit breaker instance for testing."""
        return CircuitBreaker(failure_threshold=3, recovery_timeout=0.1)
        
    def test_circuit_breaker_initialization(self, circuit_breaker):
        """Test circuit breaker initialization."""
        assert circuit_breaker.state == CircuitState.CLOSED
        assert circuit_breaker.failure_threshold == 3
        assert circuit_breaker.failure_count == 0
        
    def test_successful_call(self, circuit_breaker):
        """Test successful function call through circuit breaker."""
        mock_function = Mock(return_value="success")
        
        result = circuit_breaker.call(mock_function)
        
        assert result == "success"
        assert circuit_breaker.state == CircuitState.CLOSED
        assert circuit_breaker.failure_count == 0
        
    def test_failed_call_counting(self, circuit_breaker):
        """Test failure counting in circuit breaker."""
        mock_function = Mock(side_effect=Exception("test error"))
        
        # Make failing calls
        for i in range(2):
            with pytest.raises(Exception):
                circuit_breaker.call(mock_function)
                
        assert circuit_breaker.failure_count == 2
        assert circuit_breaker.state == CircuitState.CLOSED
        
    def test_circuit_opening(self, circuit_breaker):
        """Test circuit breaker opening after threshold failures."""
        mock_function = Mock(side_effect=Exception("test error"))
        
        # Make enough failing calls to open circuit
        for i in range(3):
            with pytest.raises(Exception):
                circuit_breaker.call(mock_function)
                
        assert circuit_breaker.state == CircuitState.OPEN
        
        # Further calls should raise CircuitBreakerException
        from quantum_ctl.pipeline_guard.circuit_breaker import CircuitBreakerException
        with pytest.raises(CircuitBreakerException):
            circuit_breaker.call(mock_function)
            
    def test_circuit_reset(self, circuit_breaker):
        """Test manual circuit breaker reset."""
        # Force circuit open
        circuit_breaker.force_open()
        assert circuit_breaker.state == CircuitState.OPEN
        
        # Reset circuit
        circuit_breaker.reset()
        assert circuit_breaker.state == CircuitState.CLOSED
        assert circuit_breaker.failure_count == 0


class TestMetricsCollector:
    """Test suite for MetricsCollector class."""
    
    @pytest.fixture
    def metrics_collector(self):
        """Create a metrics collector instance for testing."""
        return MetricsCollector(retention_hours=1)
        
    def test_metrics_collector_initialization(self, metrics_collector):
        """Test metrics collector initialization."""
        assert metrics_collector.retention_hours == 1
        assert len(metrics_collector.metrics) == 0
        
    def test_record_metric(self, metrics_collector):
        """Test recording metrics."""
        metrics_collector.record_metric(
            name="test_metric",
            value=42.5,
            labels={"component": "test"},
            unit="seconds"
        )
        
        assert "test_metric" in metrics_collector.metrics
        metrics = list(metrics_collector.metrics["test_metric"])
        assert len(metrics) == 1
        assert metrics[0].name == "test_metric"
        assert metrics[0].value == 42.5
        assert metrics[0].labels == {"component": "test"}
        
    def test_increment_counter(self, metrics_collector):
        """Test counter incrementation."""
        metrics_collector.increment_counter("test_counter", value=5)
        metrics_collector.increment_counter("test_counter", value=3)
        
        assert metrics_collector.counters["test_counter"] == 8
        
    def test_get_metric_summary(self, metrics_collector):
        """Test metric summary calculation."""
        # Record multiple values
        values = [10, 20, 30, 40, 50]
        for value in values:
            metrics_collector.record_metric("test_metric", value, "test_component")
            
        summary = metrics_collector.get_metric_summary("test_metric", hours=1)
        
        assert summary["count"] == 5
        assert summary["min"] == 10
        assert summary["max"] == 50
        assert summary["avg"] == 30


class TestSecurityMonitor:
    """Test suite for SecurityMonitor class."""
    
    @pytest.fixture
    def security_monitor(self):
        """Create a security monitor instance for testing."""
        return SecurityMonitor()
        
    def test_security_monitor_initialization(self, security_monitor):
        """Test security monitor initialization."""
        assert len(security_monitor.security_events) == 0
        assert len(security_monitor.allowed_ips) == 0
        assert len(security_monitor.blocked_ips) == 0
        
    def test_configure_allowed_ips(self, security_monitor):
        """Test IP allowlist configuration."""
        security_monitor.configure_allowed_ips(["192.168.1.0/24", "10.0.0.1"])
        
        assert len(security_monitor.allowed_ips) > 0
        assert "10.0.0.1" in security_monitor.allowed_ips
        
    def test_validate_access_allowed_ip(self, security_monitor):
        """Test access validation for allowed IP."""
        security_monitor.configure_allowed_ips(["192.168.1.100"])
        
        allowed = security_monitor.validate_access(
            "192.168.1.100", "test_user", "read_metrics"
        )
        
        assert allowed is True
        
    def test_validate_access_blocked_ip(self, security_monitor):
        """Test access validation for blocked IP."""
        security_monitor.configure_allowed_ips(["192.168.1.100"])
        
        allowed = security_monitor.validate_access(
            "172.16.0.1", "test_user", "read_metrics"
        )
        
        assert allowed is False
        
    def test_authentication_failure_tracking(self, security_monitor):
        """Test authentication failure tracking."""
        ip = "192.168.1.100"
        
        # Record multiple failures
        for i in range(3):
            security_monitor.record_auth_failure(ip, "test_user", "invalid_password")
            
        assert ip in security_monitor.failed_auth_attempts
        assert len(security_monitor.failed_auth_attempts[ip]) == 3
        
    def test_ip_blocking_after_failures(self, security_monitor):
        """Test automatic IP blocking after too many failures."""
        ip = "192.168.1.100"
        
        # Record enough failures to trigger blocking
        for i in range(6):  # More than max_auth_failures (5)
            security_monitor.record_auth_failure(ip, "test_user", "invalid_password")
            
        assert ip in security_monitor.blocked_ips


class TestPerformanceOptimizer:
    """Test suite for PerformanceOptimizer class."""
    
    @pytest.fixture
    def performance_optimizer(self):
        """Create a performance optimizer instance for testing."""
        return PerformanceOptimizer()
        
    def test_performance_optimizer_initialization(self, performance_optimizer):
        """Test performance optimizer initialization."""
        assert len(performance_optimizer.metrics_history) == 0
        assert len(performance_optimizer.optimization_rules) == 0
        
    def test_record_performance_metric(self, performance_optimizer):
        """Test recording performance metrics."""
        performance_optimizer.record_performance_metric(
            name="cpu_usage",
            value=75.5,
            component="system"
        )
        
        assert "cpu_usage" in performance_optimizer.metrics_history
        metrics = list(performance_optimizer.metrics_history["cpu_usage"])
        assert len(metrics) == 1
        assert metrics[0].value == 75.5
        
    def test_register_scaling_policy(self, performance_optimizer):
        """Test scaling policy registration."""
        performance_optimizer.register_scaling_policy(
            component="test_component",
            scale_up_threshold=80.0,
            scale_down_threshold=30.0,
            max_instances=5
        )
        
        assert "test_component" in performance_optimizer.scaling_policies
        policy = performance_optimizer.scaling_policies["test_component"]
        assert policy["scale_up_threshold"] == 80.0
        assert policy["max_instances"] == 5
        
    def test_add_optimization_rule(self, performance_optimizer):
        """Test optimization rule addition."""
        condition = lambda metrics: metrics.get("cpu_usage", 0) > 80
        action = lambda: "optimized"
        
        performance_optimizer.add_optimization_rule(
            name="high_cpu_rule",
            condition=condition,
            action=action,
            priority=1
        )
        
        assert len(performance_optimizer.optimization_rules) == 1
        rule = performance_optimizer.optimization_rules[0]
        assert rule["name"] == "high_cpu_rule"
        assert rule["priority"] == 1


@pytest.mark.asyncio
async def test_integration_pipeline_guard_with_components():
    """Integration test for pipeline guard with multiple components."""
    
    # Create components
    guard = PipelineGuard(check_interval=0.1)
    
    # Mock components with different behaviors
    healthy_component = Mock(return_value=True)
    flaky_component_calls = [False, False, True, True]  # Fail, then recover
    flaky_call_count = 0
    
    def flaky_component():
        nonlocal flaky_call_count
        result = flaky_component_calls[min(flaky_call_count, len(flaky_component_calls) - 1)]
        flaky_call_count += 1
        return result
        
    recovery_action = AsyncMock(return_value=True)
    
    # Register components
    guard.register_component(
        name="healthy_component",
        health_check=healthy_component,
        critical=False
    )
    
    guard.register_component(
        name="flaky_component", 
        health_check=flaky_component,
        recovery_action=recovery_action,
        critical=True
    )
    
    # Start monitoring
    guard.start()
    
    # Let it run for a bit
    await asyncio.sleep(0.5)
    
    # Check that recovery was attempted
    assert recovery_action.called
    
    # Get final status
    status = guard.get_status()
    
    # Should eventually be healthy after recovery
    await asyncio.sleep(0.3)
    final_status = guard.get_status()
    
    await guard.stop()
    
    # Verify results
    assert status["total_components"] == 2
    assert final_status["healthy_components"] >= 1  # At least healthy component should be working


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
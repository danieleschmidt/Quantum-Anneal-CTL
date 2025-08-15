#!/usr/bin/env python3
"""
Basic functionality test for the self-healing pipeline guard.
Tests core components without requiring pytest.
"""

import sys
import asyncio
import time
import traceback
from unittest.mock import Mock, AsyncMock

# Add current directory to path
sys.path.insert(0, '.')

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from quantum_ctl.pipeline_guard.guard import PipelineGuard, PipelineStatus
        print("✓ PipelineGuard imported successfully")
        
        from quantum_ctl.pipeline_guard.health_monitor import HealthMonitor
        print("✓ HealthMonitor imported successfully")
        
        from quantum_ctl.pipeline_guard.recovery_manager import RecoveryManager
        print("✓ RecoveryManager imported successfully")
        
        from quantum_ctl.pipeline_guard.circuit_breaker import CircuitBreaker, CircuitState
        print("✓ CircuitBreaker imported successfully")
        
        from quantum_ctl.pipeline_guard.metrics_collector import MetricsCollector
        print("✓ MetricsCollector imported successfully")
        
        from quantum_ctl.pipeline_guard.security_monitor import SecurityMonitor
        print("✓ SecurityMonitor imported successfully")
        
        from quantum_ctl.pipeline_guard.performance_optimizer import PerformanceOptimizer
        print("✓ PerformanceOptimizer imported successfully")
        
        print("All imports successful!")
        return True
        
    except Exception as e:
        print(f"✗ Import error: {e}")
        traceback.print_exc()
        return False


def test_basic_pipeline_guard():
    """Test basic pipeline guard functionality."""
    print("\nTesting basic pipeline guard...")
    
    try:
        from quantum_ctl.pipeline_guard.guard import PipelineGuard, PipelineStatus
        
        # Create pipeline guard
        guard = PipelineGuard(check_interval=0.1)
        
        # Test initialization
        assert guard.status == PipelineStatus.HEALTHY
        assert guard.check_interval == 0.1
        assert len(guard.components) == 0
        print("✓ Pipeline guard initialization")
        
        # Test component registration
        mock_health_check = Mock(return_value=True)
        mock_recovery = AsyncMock(return_value=True)
        
        guard.register_component(
            name="test_component",
            health_check=mock_health_check,
            recovery_action=mock_recovery,
            critical=True
        )
        
        assert "test_component" in guard.components
        assert guard.components["test_component"].critical is True
        print("✓ Component registration")
        
        # Test status retrieval
        status = guard.get_status()
        assert status["total_components"] == 1
        assert "components" in status
        print("✓ Status retrieval")
        
        print("Basic pipeline guard tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Pipeline guard test error: {e}")
        traceback.print_exc()
        return False


def test_health_monitor():
    """Test health monitor functionality."""
    print("\nTesting health monitor...")
    
    try:
        from quantum_ctl.pipeline_guard.health_monitor import HealthMonitor
        
        # Create health monitor
        monitor = HealthMonitor()
        
        # Test registration
        health_check = Mock(return_value=True)
        monitor.register_component("test_component", health_check)
        
        assert "test_component" in monitor.components
        assert "test_component" in monitor.health_checks
        print("✓ Component registration in health monitor")
        
        print("Health monitor tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Health monitor test error: {e}")
        traceback.print_exc()
        return False


def test_circuit_breaker():
    """Test circuit breaker functionality."""
    print("\nTesting circuit breaker...")
    
    try:
        from quantum_ctl.pipeline_guard.circuit_breaker import CircuitBreaker, CircuitState
        
        # Create circuit breaker
        breaker = CircuitBreaker(failure_threshold=3)
        
        # Test initial state
        assert breaker.state == CircuitState.CLOSED
        assert breaker.failure_count == 0
        print("✓ Circuit breaker initialization")
        
        # Test successful call
        mock_function = Mock(return_value="success")
        result = breaker.call(mock_function)
        assert result == "success"
        assert breaker.state == CircuitState.CLOSED
        print("✓ Successful function call through circuit breaker")
        
        # Test failure handling
        mock_function.side_effect = Exception("test error")
        
        # Make enough calls to open circuit
        for i in range(3):
            try:
                breaker.call(mock_function)
            except Exception:
                pass
                
        assert breaker.state == CircuitState.OPEN
        print("✓ Circuit breaker opens after failures")
        
        # Test that further calls raise CircuitBreakerException
        from quantum_ctl.pipeline_guard.circuit_breaker import CircuitBreakerException
        try:
            breaker.call(mock_function)
            assert False, "Should have raised CircuitBreakerException"
        except CircuitBreakerException:
            print("✓ Circuit breaker blocks calls when open")
            
        print("Circuit breaker tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Circuit breaker test error: {e}")
        traceback.print_exc()
        return False


def test_metrics_collector():
    """Test metrics collector functionality."""
    print("\nTesting metrics collector...")
    
    try:
        from quantum_ctl.pipeline_guard.metrics_collector import MetricsCollector
        
        # Create metrics collector
        collector = MetricsCollector(retention_hours=1)
        
        # Test metric recording
        collector.record_metric(
            name="test_metric",
            value=42.5,
            component="test_component",
            metadata={"unit": "seconds"}
        )
        
        assert "test_metric" in collector.metrics
        metrics = list(collector.metrics["test_metric"])
        assert len(metrics) == 1
        assert metrics[0].value == 42.5
        print("✓ Metric recording")
        
        # Test counter increment
        collector.increment_counter("test_counter", value=5)
        collector.increment_counter("test_counter", value=3)
        assert collector.counters["test_counter"] == 8
        print("✓ Counter increment")
        
        print("Metrics collector tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Metrics collector test error: {e}")
        traceback.print_exc()
        return False


def test_security_monitor():
    """Test security monitor functionality."""
    print("\nTesting security monitor...")
    
    try:
        from quantum_ctl.pipeline_guard.security_monitor import SecurityMonitor
        
        # Create security monitor
        monitor = SecurityMonitor()
        
        # Test IP configuration
        monitor.configure_allowed_ips(["192.168.1.0/24", "10.0.0.1"])
        assert len(monitor.allowed_ips) > 0
        print("✓ IP allowlist configuration")
        
        # Test access validation
        allowed = monitor.validate_access("10.0.0.1", "test_user", "read_metrics")
        assert allowed is True
        print("✓ Access validation for allowed IP")
        
        # Test blocking
        blocked = monitor.validate_access("172.16.0.1", "test_user", "read_metrics")
        assert blocked is False
        print("✓ Access blocking for disallowed IP")
        
        print("Security monitor tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Security monitor test error: {e}")
        traceback.print_exc()
        return False


async def test_async_functionality():
    """Test async functionality."""
    print("\nTesting async functionality...")
    
    try:
        from quantum_ctl.pipeline_guard.recovery_manager import RecoveryManager
        
        # Create recovery manager
        manager = RecoveryManager(max_retries=2, retry_delay=0.01)
        
        # Test successful recovery
        async def mock_recovery():
            return True
            
        manager.register_recovery("test_component", mock_recovery)
        success = await manager.recover_component("test_component")
        
        assert success is True
        print("✓ Async recovery functionality")
        
        print("Async functionality tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Async functionality test error: {e}")
        traceback.print_exc()
        return False


async def run_all_tests():
    """Run all tests."""
    print("Self-Healing Pipeline Guard - Basic Functionality Tests")
    print("=" * 60)
    
    test_results = []
    
    # Run synchronous tests
    test_results.append(test_imports())
    test_results.append(test_basic_pipeline_guard())
    test_results.append(test_health_monitor())
    test_results.append(test_circuit_breaker())
    test_results.append(test_metrics_collector())
    test_results.append(test_security_monitor())
    
    # Run asynchronous tests
    test_results.append(await test_async_functionality())
    
    # Summary
    print("\n" + "=" * 60)
    passed = sum(test_results)
    total = len(test_results)
    
    if passed == total:
        print(f"✓ ALL TESTS PASSED ({passed}/{total})")
        print("\nSelf-healing pipeline guard is ready for deployment!")
    else:
        print(f"✗ SOME TESTS FAILED ({passed}/{total} passed)")
        print("\nPlease review failed tests before deployment.")
        
    return passed == total


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
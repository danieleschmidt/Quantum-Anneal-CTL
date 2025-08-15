#!/usr/bin/env python3
"""
Direct test of pipeline guard components without full package imports.
"""

import sys
import asyncio
import time
import traceback
from unittest.mock import Mock, AsyncMock

# Add current directory to path
sys.path.insert(0, '.')

def test_direct_imports():
    """Test direct imports of pipeline guard modules."""
    print("Testing direct imports...")
    
    try:
        # Import individual modules directly
        sys.path.insert(0, './quantum_ctl/pipeline_guard')
        
        import guard
        print("✓ guard module imported")
        
        import health_monitor
        print("✓ health_monitor module imported")
        
        import recovery_manager
        print("✓ recovery_manager module imported")
        
        import circuit_breaker
        print("✓ circuit_breaker module imported")
        
        import metrics_collector
        print("✓ metrics_collector module imported")
        
        import security_monitor
        print("✓ security_monitor module imported")
        
        import performance_optimizer
        print("✓ performance_optimizer module imported")
        
        return True
        
    except Exception as e:
        print(f"✗ Import error: {e}")
        traceback.print_exc()
        return False


def test_pipeline_guard_functionality():
    """Test pipeline guard core functionality."""
    print("\nTesting pipeline guard functionality...")
    
    try:
        sys.path.insert(0, './quantum_ctl/pipeline_guard')
        from guard import PipelineGuard, PipelineStatus
        
        # Create pipeline guard
        guard = PipelineGuard(check_interval=0.1)
        
        # Test initialization
        assert guard.status == PipelineStatus.HEALTHY
        print("✓ PipelineGuard initialization")
        
        # Test component registration
        mock_health_check = Mock(return_value=True)
        guard.register_component(
            name="test_component",
            health_check=mock_health_check,
            critical=True
        )
        
        assert "test_component" in guard.components
        print("✓ Component registration")
        
        # Test status
        status = guard.get_status()
        assert status["total_components"] == 1
        print("✓ Status retrieval")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        traceback.print_exc()
        return False


def test_circuit_breaker_functionality():
    """Test circuit breaker functionality."""
    print("\nTesting circuit breaker...")
    
    try:
        sys.path.insert(0, './quantum_ctl/pipeline_guard')
        from circuit_breaker import CircuitBreaker, CircuitState, CircuitBreakerException
        
        # Create circuit breaker
        breaker = CircuitBreaker(failure_threshold=3)
        
        # Test successful call
        mock_function = Mock(return_value="success")
        result = breaker.call(mock_function)
        assert result == "success"
        print("✓ Successful function call")
        
        # Test failure handling
        mock_function.side_effect = Exception("test error")
        
        # Trigger failures
        for i in range(3):
            try:
                breaker.call(mock_function)
            except Exception:
                pass
                
        assert breaker.state == CircuitState.OPEN
        print("✓ Circuit opens after failures")
        
        # Test blocking
        try:
            breaker.call(mock_function)
            assert False, "Should have raised CircuitBreakerException"
        except CircuitBreakerException:
            print("✓ Circuit blocks when open")
            
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        traceback.print_exc()
        return False


def test_metrics_collector_functionality():
    """Test metrics collector functionality."""
    print("\nTesting metrics collector...")
    
    try:
        sys.path.insert(0, './quantum_ctl/pipeline_guard')
        from metrics_collector import MetricsCollector
        
        # Create metrics collector
        collector = MetricsCollector()
        
        # Test metric recording
        collector.record_metric("test_metric", 42.5, "test_component")
        assert "test_metric" in collector.metrics
        print("✓ Metric recording")
        
        # Test counter
        collector.increment_counter("test_counter", 5)
        assert collector.counters["test_counter"] == 5
        print("✓ Counter increment")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        traceback.print_exc()
        return False


def test_security_monitor_functionality():
    """Test security monitor functionality."""
    print("\nTesting security monitor...")
    
    try:
        sys.path.insert(0, './quantum_ctl/pipeline_guard')
        from security_monitor import SecurityMonitor
        
        # Create security monitor
        monitor = SecurityMonitor()
        
        # Test IP configuration
        monitor.configure_allowed_ips(["10.0.0.1"])
        assert "10.0.0.1" in monitor.allowed_ips
        print("✓ IP allowlist configuration")
        
        # Test access validation
        allowed = monitor.validate_access("10.0.0.1", "test_user", "read")
        assert allowed is True
        print("✓ Access validation")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        traceback.print_exc()
        return False


async def test_async_functionality():
    """Test async functionality."""
    print("\nTesting async functionality...")
    
    try:
        sys.path.insert(0, './quantum_ctl/pipeline_guard')
        from recovery_manager import RecoveryManager
        
        # Create recovery manager
        manager = RecoveryManager(max_retries=1, retry_delay=0.01)
        
        # Test successful recovery
        async def mock_recovery():
            return True
            
        manager.register_recovery("test_component", mock_recovery)
        success = await manager.recover_component("test_component")
        
        assert success is True
        print("✓ Async recovery")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        traceback.print_exc()
        return False


async def run_direct_tests():
    """Run direct tests."""
    print("Self-Healing Pipeline Guard - Direct Module Tests")
    print("=" * 55)
    
    test_results = []
    
    # Run tests
    test_results.append(test_direct_imports())
    test_results.append(test_pipeline_guard_functionality())
    test_results.append(test_circuit_breaker_functionality())
    test_results.append(test_metrics_collector_functionality())
    test_results.append(test_security_monitor_functionality())
    test_results.append(await test_async_functionality())
    
    # Summary
    print("\n" + "=" * 55)
    passed = sum(test_results)
    total = len(test_results)
    
    if passed == total:
        print(f"✓ ALL TESTS PASSED ({passed}/{total})")
        print("\nPipeline guard modules are working correctly!")
    else:
        print(f"✗ SOME TESTS FAILED ({passed}/{total} passed)")
        
    return passed == total


if __name__ == "__main__":
    success = asyncio.run(run_direct_tests())
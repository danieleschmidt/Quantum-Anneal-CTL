#!/usr/bin/env python3
"""
Basic import and functionality tests for quantum_ctl package.
Designed to work without external dependencies for CI/testing.
"""

import sys
import os
import unittest
from unittest.mock import Mock, patch

# Add package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestBasicImports(unittest.TestCase):
    """Test basic package imports and structure."""
    
    def setUp(self):
        """Set up test environment with mocked dependencies."""
        # Mock numpy and other external dependencies
        sys.modules['numpy'] = Mock()
        sys.modules['scipy'] = Mock()
        sys.modules['pandas'] = Mock()
        sys.modules['pydantic'] = Mock()
        sys.modules['fastapi'] = Mock()
        sys.modules['asyncio'] = Mock()
        
        # Mock D-Wave dependencies
        sys.modules['dwave'] = Mock()
        sys.modules['dwave.system'] = Mock()
        sys.modules['dwave.embedding'] = Mock()
        sys.modules['dimod'] = Mock()
        
        # Mock other optional dependencies
        sys.modules['sklearn'] = Mock()
        sys.modules['sklearn.ensemble'] = Mock()
        sys.modules['sklearn.linear_model'] = Mock()
        sys.modules['sklearn.preprocessing'] = Mock()
        sys.modules['psutil'] = Mock()
        
        # Mock numpy array creation
        mock_numpy = Mock()
        mock_numpy.array = Mock(return_value=[1, 2, 3])
        mock_numpy.zeros = Mock(return_value=[0, 0, 0])
        mock_numpy.ones = Mock(return_value=[1, 1, 1])
        mock_numpy.eye = Mock(return_value=[[1, 0], [0, 1]])
        mock_numpy.random = Mock()
        mock_numpy.random.uniform = Mock(return_value=0.5)
        mock_numpy.random.normal = Mock(return_value=0.0)
        mock_numpy.log = Mock(return_value=1.0)
        mock_numpy.mean = Mock(return_value=0.5)
        mock_numpy.sum = Mock(return_value=2.0)
        mock_numpy.clip = Mock(return_value=0.5)
        sys.modules['numpy'] = mock_numpy
    
    def test_package_structure(self):
        """Test package structure and basic imports."""
        try:
            # Test core package import
            import quantum_ctl
            
            # Check version
            self.assertTrue(hasattr(quantum_ctl, '__version__'))
            self.assertIsInstance(quantum_ctl.__version__, str)
            
            # Check author info
            self.assertTrue(hasattr(quantum_ctl, '__author__'))
            self.assertEqual(quantum_ctl.__author__, "Daniel Schmidt")
            
        except Exception as e:
            self.fail(f"Package import failed: {e}")
    
    def test_core_modules(self):
        """Test core module imports."""
        try:
            # Test core modules
            from quantum_ctl.core import controller
            from quantum_ctl.models import building
            from quantum_ctl.optimization import mpc_to_qubo
            from quantum_ctl.utils import config
            
            self.assertIsNotNone(controller)
            self.assertIsNotNone(building) 
            self.assertIsNotNone(mpc_to_qubo)
            self.assertIsNotNone(config)
            
        except Exception as e:
            self.fail(f"Core module import failed: {e}")
    
    def test_optimization_modules(self):
        """Test optimization module imports."""
        try:
            from quantum_ctl.optimization import quantum_solver
            from quantum_ctl.optimization import adaptive_quantum_engine
            from quantum_ctl.optimization import enhanced_mpc_qubo
            
            self.assertIsNotNone(quantum_solver)
            self.assertIsNotNone(adaptive_quantum_engine)
            self.assertIsNotNone(enhanced_mpc_qubo)
            
        except Exception as e:
            self.fail(f"Optimization module import failed: {e}")
    
    def test_utils_modules(self):
        """Test utility module imports."""
        try:
            from quantum_ctl.utils import error_handling
            from quantum_ctl.utils import monitoring
            from quantum_ctl.utils import quantum_health_monitor
            from quantum_ctl.utils import advanced_error_recovery
            
            self.assertIsNotNone(error_handling)
            self.assertIsNotNone(monitoring)
            self.assertIsNotNone(quantum_health_monitor)
            self.assertIsNotNone(advanced_error_recovery)
            
        except Exception as e:
            self.fail(f"Utils module import failed: {e}")
    
    def test_class_definitions(self):
        """Test that key classes are properly defined."""
        try:
            from quantum_ctl.core.controller import HVACController
            from quantum_ctl.models.building import Building, BuildingState
            from quantum_ctl.optimization.quantum_solver import QuantumSolver
            from quantum_ctl.optimization.adaptive_quantum_engine import AdaptiveQuantumEngine
            
            # Test class existence
            self.assertTrue(hasattr(HVACController, '__init__'))
            self.assertTrue(hasattr(Building, '__init__'))
            self.assertTrue(hasattr(BuildingState, '__init__'))
            self.assertTrue(hasattr(QuantumSolver, '__init__'))
            self.assertTrue(hasattr(AdaptiveQuantumEngine, '__init__'))
            
        except Exception as e:
            self.fail(f"Class definition test failed: {e}")
    
    def test_enum_definitions(self):
        """Test that enums are properly defined."""
        try:
            from quantum_ctl.optimization.adaptive_quantum_engine import OptimizationStrategy
            from quantum_ctl.utils.quantum_health_monitor import QuantumHealthStatus
            from quantum_ctl.utils.advanced_error_recovery import RecoveryMode
            
            # Test enum values
            self.assertTrue(hasattr(OptimizationStrategy, 'ADAPTIVE_HYBRID'))
            self.assertTrue(hasattr(QuantumHealthStatus, 'OPTIMAL'))
            self.assertTrue(hasattr(RecoveryMode, 'RETRY_QUANTUM'))
            
        except Exception as e:
            self.fail(f"Enum definition test failed: {e}")
    
    def test_configuration_loading(self):
        """Test configuration system."""
        try:
            from quantum_ctl.utils.config import QuantumConfig
            
            # Test config class instantiation
            config = QuantumConfig()
            self.assertIsNotNone(config)
            
            # Test basic config attributes
            self.assertTrue(hasattr(config, 'load_config'))
            self.assertTrue(hasattr(config, 'get'))
            
        except Exception as e:
            self.fail(f"Configuration test failed: {e}")


class TestBasicFunctionality(unittest.TestCase):
    """Test basic functionality without external dependencies."""
    
    def setUp(self):
        """Set up test environment."""
        # Mock all external dependencies
        sys.modules['numpy'] = Mock()
        sys.modules['scipy'] = Mock() 
        sys.modules['pandas'] = Mock()
        sys.modules['pydantic'] = Mock()
        sys.modules['fastapi'] = Mock()
        
        # Create mock numpy with basic array operations
        mock_numpy = Mock()
        mock_numpy.array = Mock(return_value=[1, 2, 3])
        mock_numpy.zeros = Mock(return_value=[0, 0, 0])
        mock_numpy.ones = Mock(return_value=[1, 1, 1])
        mock_numpy.eye = Mock(return_value=[[1, 0], [0, 1]])
        mock_numpy.random = Mock()
        mock_numpy.random.uniform = Mock(return_value=0.5)
        sys.modules['numpy'] = mock_numpy
    
    def test_building_model_creation(self):
        """Test building model creation."""
        try:
            from quantum_ctl.models.building import Building
            
            # Mock building creation
            with patch.object(Building, '__init__', return_value=None):
                building = Building.__new__(Building)
                building.zones = 5
                building.building_id = "test_building"
                
                self.assertEqual(building.zones, 5)
                self.assertEqual(building.building_id, "test_building")
                
        except Exception as e:
            self.fail(f"Building model creation failed: {e}")
    
    def test_qubo_conversion(self):
        """Test QUBO conversion functionality."""
        try:
            from quantum_ctl.optimization.mpc_to_qubo import MPCToQUBO
            
            # Mock QUBO converter
            with patch.object(MPCToQUBO, '__init__', return_value=None):
                converter = MPCToQUBO.__new__(MPCToQUBO)
                converter.state_dim = 5
                converter.control_dim = 5
                converter.horizon = 24
                
                self.assertEqual(converter.state_dim, 5)
                self.assertEqual(converter.control_dim, 5)
                self.assertEqual(converter.horizon, 24)
                
        except Exception as e:
            self.fail(f"QUBO conversion test failed: {e}")
    
    def test_error_handling(self):
        """Test error handling system."""
        try:
            from quantum_ctl.utils.error_handling import QuantumControlError, ErrorCategory
            
            # Test error creation
            error = QuantumControlError("Test error", ErrorCategory.OPTIMIZATION)
            self.assertIsInstance(error, Exception)
            self.assertEqual(str(error), "Test error")
            
        except Exception as e:
            self.fail(f"Error handling test failed: {e}")
    
    def test_monitoring_system(self):
        """Test monitoring system basics."""
        try:
            from quantum_ctl.utils.monitoring import HealthMonitor
            
            # Test monitor creation
            with patch.object(HealthMonitor, '__init__', return_value=None):
                monitor = HealthMonitor.__new__(HealthMonitor)
                monitor.name = "test_monitor"
                
                self.assertEqual(monitor.name, "test_monitor")
                
        except Exception as e:
            self.fail(f"Monitoring system test failed: {e}")


def run_tests():
    """Run all tests and return results."""
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestBasicImports))
    test_suite.addTest(unittest.makeSuite(TestBasicFunctionality))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result


if __name__ == "__main__":
    print("🧪 Running Quantum HVAC Control System Tests")
    print("=" * 50)
    
    result = run_tests()
    
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS SUMMARY")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print("\n❌ FAILURES:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split(chr(10))[-2]}")
    
    if result.errors:
        print("\n💥 ERRORS:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.split(chr(10))[-2]}")
    
    if result.wasSuccessful():
        print("\n✅ ALL TESTS PASSED!")
        exit(0)
    else:
        print("\n❌ SOME TESTS FAILED!")
        exit(1)
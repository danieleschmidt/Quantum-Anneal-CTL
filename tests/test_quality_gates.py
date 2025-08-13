"""
Quality gate tests for quantum HVAC system.
"""

import pytest
import asyncio
import time
import numpy as np
from unittest.mock import Mock, patch

from quantum_ctl import HVACController, Building
from quantum_ctl.core.controller import OptimizationConfig, ControlObjectives
from quantum_ctl.models.building import BuildingState, ZoneConfig
from quantum_ctl.utils.config_validator import SystemValidator, validate_system
from quantum_ctl.utils.health_dashboard import get_health_dashboard
from quantum_ctl.optimization.intelligent_caching import get_intelligent_cache
from quantum_ctl.optimization.load_balancer import get_load_balancer, WorkloadMetrics


class TestSecurityValidation:
    """Security validation tests."""
    
    def test_input_sanitization(self):
        """Test input sanitization and validation."""
        validator = SystemValidator()
        
        # Test malicious zone configuration
        zone = ZoneConfig(
            zone_id="'; DROP TABLE zones; --",  # SQL injection attempt
            area=-100.0,  # Invalid negative area
            volume=0.0,   # Invalid zero volume
            thermal_mass=-50.0,  # Invalid negative mass
            max_heating_power=1000000.0,  # Unreasonably high power
            max_cooling_power=1000000.0,
            comfort_temp_min=100.0,  # Invalid temperature range
            comfort_temp_max=50.0
        )
        
        building = Building(
            building_id="test_building",
            zones=[zone],
            occupancy_schedule="office_standard"
        )
        
        result = validator.validate_building_config(building)
        
        # Should catch all validation errors
        assert not result.is_valid
        assert len(result.errors) > 0
        
        # Check specific security/validation issues
        error_messages = ' '.join(result.errors)
        assert "invalid area" in error_messages.lower()
        assert "invalid volume" in error_messages.lower()
    
    def test_state_bounds_validation(self):
        """Test building state bounds validation."""
        validator = SystemValidator()
        
        zone = ZoneConfig(
            zone_id="test_zone",
            area=100.0,
            volume=300.0,
            thermal_mass=200.0,
            max_heating_power=10.0,
            max_cooling_power=8.0
        )
        
        building = Building(
            building_id="test_building",
            zones=[zone],
            occupancy_schedule="office_standard"
        )
        
        # Test malicious/invalid state
        malicious_state = BuildingState(
            timestamp=0.0,
            zone_temperatures=np.array([1000.0]),  # Extreme temperature
            outside_temperature=-273.0,  # Below absolute zero
            humidity=150.0,  # Invalid humidity
            occupancy=np.array([2.0]),  # Invalid occupancy > 1
            hvac_power=np.array([1000.0]),  # Extreme power
            control_setpoints=np.array([5.0])  # Invalid control > 1
        )
        
        result = validator.validate_state_data(malicious_state, building)
        
        # Should catch validation errors
        assert not result.is_valid or len(result.warnings) > 0
    
    def test_configuration_limits(self):
        """Test configuration parameter limits."""
        validator = SystemValidator()
        
        # Test extreme configuration
        extreme_config = OptimizationConfig(
            prediction_horizon=8760,  # 1 year - excessive
            control_interval=1,       # 1 minute - too frequent
            solver="malicious_solver",
            num_reads=1000000        # Extremely high
        )
        
        result = validator.validate_optimization_config(extreme_config)
        
        # Should generate warnings about extreme values
        assert len(result.warnings) > 0 or len(result.errors) > 0


class TestPerformanceValidation:
    """Performance validation tests."""
    
    @pytest.mark.asyncio
    async def test_optimization_performance(self):
        """Test optimization performance requirements."""
        # Create simple building
        zone = ZoneConfig(
            zone_id="perf_test_zone",
            area=100.0,
            volume=300.0,
            thermal_mass=200.0,
            max_heating_power=10.0,
            max_cooling_power=8.0
        )
        
        building = Building(
            building_id="perf_test_building",
            zones=[zone],
            occupancy_schedule="office_standard"
        )
        
        config = OptimizationConfig(
            prediction_horizon=2,
            control_interval=30,
            solver="classical_fallback",
            num_reads=10
        )
        
        objectives = ControlObjectives()
        controller = HVACController(building, config, objectives)
        
        # Create test state
        state = BuildingState(
            timestamp=0.0,
            zone_temperatures=np.array([22.0]),
            outside_temperature=15.0,
            humidity=50.0,
            occupancy=np.array([0.5]),
            hvac_power=np.array([5.0]),
            control_setpoints=np.array([0.5])
        )
        
        # Simple forecast
        n_steps = 4
        weather_forecast = np.array([[15.0, 200.0, 50.0]] * n_steps)
        energy_prices = np.array([0.12] * n_steps)
        
        # Performance test
        start_time = time.time()
        
        control_schedule = await controller.optimize(
            current_state=state,
            weather_forecast=weather_forecast,
            energy_prices=energy_prices
        )
        
        execution_time = time.time() - start_time
        
        # Performance requirements
        assert execution_time < 5.0, f"Optimization took {execution_time:.2f}s, should be < 5s"
        assert len(control_schedule) > 0, "Should return valid control schedule"
    
    def test_cache_performance(self):
        """Test intelligent cache performance."""
        cache = get_intelligent_cache()
        
        # Test cache operations
        problem_data = {
            'building_id': 'test_building',
            'weather_forecast': np.array([[20.0, 300.0, 60.0]] * 4),
            'energy_prices': np.array([0.12] * 4),
            'current_state': np.array([22.0]),
            'prediction_horizon': 2
        }
        
        solution = np.array([0.5, 0.6, 0.7, 0.8])
        
        # Store solution
        cache.store_solution(problem_data, solution, 10.0, 1.5)
        
        # Retrieve solution
        start_time = time.time()
        cached_result = cache.get_solution(problem_data)
        cache_time = time.time() - start_time
        
        # Cache should be fast
        assert cache_time < 0.1, f"Cache lookup took {cache_time:.3f}s, should be < 0.1s"
        assert cached_result is not None, "Should find cached solution"
        
        cached_solution, cached_energy, cached_comp_time = cached_result
        np.testing.assert_array_equal(cached_solution, solution)
        assert cached_energy == 10.0
        assert cached_comp_time == 1.5
    
    def test_load_balancer_performance(self):
        """Test load balancer performance."""
        load_balancer = get_load_balancer()
        
        # Test node selection performance
        workload = WorkloadMetrics(
            problem_size=100,
            priority=5,
            deadline=time.time() + 30.0,
            estimated_runtime=1.0,
            max_cost=1.0
        )
        
        start_time = time.time()
        selected_node = load_balancer.select_optimal_node(workload)
        selection_time = time.time() - start_time
        
        # Load balancer should be fast
        assert selection_time < 0.01, f"Node selection took {selection_time:.3f}s, should be < 0.01s"
        assert selected_node is not None, "Should select a node"


class TestReliabilityValidation:
    """Reliability and error handling tests."""
    
    @pytest.mark.asyncio
    async def test_graceful_degradation(self):
        """Test graceful degradation under failures."""
        # Test with broken quantum solver
        zone = ZoneConfig(
            zone_id="reliability_zone",
            area=100.0,
            volume=300.0,
            thermal_mass=200.0,
            max_heating_power=10.0,
            max_cooling_power=8.0
        )
        
        building = Building(
            building_id="reliability_building",
            zones=[zone],
            occupancy_schedule="office_standard"
        )
        
        config = OptimizationConfig(
            prediction_horizon=1,
            control_interval=30,
            solver="non_existent_solver",  # Should fall back
            num_reads=10
        )
        
        objectives = ControlObjectives()
        controller = HVACController(building, config, objectives)
        
        state = BuildingState(
            timestamp=0.0,
            zone_temperatures=np.array([22.0]),
            outside_temperature=15.0,
            humidity=50.0,
            occupancy=np.array([0.5]),
            hvac_power=np.array([5.0]),
            control_setpoints=np.array([0.5])
        )
        
        # Should not raise exception, should fall back gracefully
        n_steps = 2
        weather_forecast = np.array([[15.0, 200.0, 50.0]] * n_steps)
        energy_prices = np.array([0.12] * n_steps)
        
        control_schedule = await controller.optimize(
            current_state=state,
            weather_forecast=weather_forecast,
            energy_prices=energy_prices
        )
        
        # Should return some control schedule even with fallback
        assert len(control_schedule) > 0
    
    def test_error_recovery(self):
        """Test error recovery mechanisms."""
        dashboard = get_health_dashboard()
        
        # Simulate errors
        dashboard.record_error("Test error 1")
        dashboard.record_error("Test error 2")
        dashboard.record_optimization(False, 1.0, "Test optimization error")
        
        # Check health status reflects errors
        metrics = dashboard.get_current_metrics()
        assert metrics.error_count_last_hour >= 2
        assert metrics.last_error is not None
    
    def test_resource_limits(self):
        """Test resource limit enforcement."""
        cache = get_intelligent_cache()
        
        # Fill cache beyond capacity (simulate memory pressure)
        for i in range(cache.max_size + 100):
            problem_data = {
                'building_id': f'test_building_{i}',
                'scenario': i,
                'weather_forecast': np.random.rand(4, 3),
                'energy_prices': np.random.rand(4)
            }
            
            solution = np.random.rand(4)
            cache.store_solution(problem_data, solution, 0.0, 1.0)
        
        # Cache should enforce size limits
        stats = cache.get_statistics()
        assert stats['cache_size'] <= cache.max_size, "Cache should enforce size limits"
        assert stats['evictions'] > 0, "Should have evicted entries"


class TestSystemIntegration:
    """System integration tests."""
    
    def test_system_validation_integration(self):
        """Test complete system validation."""
        is_valid, results = validate_system()
        
        # Basic system should be valid (or have only warnings)
        assert 'environment' in results
        assert 'overall_valid' in results
        
        # Should not have critical errors that prevent operation
        env_result = results['environment']
        critical_errors = [
            error for error in env_result.errors 
            if 'python' in error.lower() or 'critical' in error.lower()
        ]
        assert len(critical_errors) == 0, f"Critical system errors: {critical_errors}"
    
    @pytest.mark.asyncio
    async def test_end_to_end_optimization(self):
        """Test complete end-to-end optimization flow."""
        # Full integration test
        dashboard = get_health_dashboard()
        cache = get_intelligent_cache()
        load_balancer = get_load_balancer()
        
        # Create test building
        zones = [
            ZoneConfig(
                zone_id=f"integration_zone_{i}",
                area=100.0,
                volume=300.0,
                thermal_mass=200.0,
                max_heating_power=10.0,
                max_cooling_power=8.0
            ) for i in range(2)
        ]
        
        building = Building(
            building_id="integration_test_building",
            zones=zones,
            occupancy_schedule="office_standard"
        )
        
        config = OptimizationConfig(
            prediction_horizon=1,
            control_interval=30,
            solver="classical_fallback",
            num_reads=10
        )
        
        objectives = ControlObjectives()
        controller = HVACController(building, config, objectives)
        
        state = BuildingState(
            timestamp=0.0,
            zone_temperatures=np.array([22.0, 21.5]),
            outside_temperature=15.0,
            humidity=50.0,
            occupancy=np.array([0.5, 0.7]),
            hvac_power=np.array([5.0, 6.0]),
            control_setpoints=np.array([0.5, 0.6])
        )
        
        n_steps = 2
        weather_forecast = np.array([[15.0, 200.0, 50.0]] * n_steps)
        energy_prices = np.array([0.12] * n_steps)
        
        # Run optimization with full system integration
        start_time = time.time()
        
        control_schedule = await controller.optimize(
            current_state=state,
            weather_forecast=weather_forecast,
            energy_prices=energy_prices
        )
        
        execution_time = time.time() - start_time
        
        # Verify results
        assert len(control_schedule) > 0, "Should produce control schedule"
        assert execution_time < 10.0, "End-to-end test should complete quickly"
        
        # Check that monitoring components recorded the operation
        health_report = dashboard.get_health_report()
        assert health_report['total_optimizations'] > 0, "Dashboard should record optimizations"
        
        # Apply controls
        controller.apply_schedule(control_schedule)
        
        # Verify controller status
        status = controller.get_status()
        assert status['building_id'] == building.building_id
        assert 'quantum_solver_status' in status


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
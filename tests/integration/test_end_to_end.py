"""
End-to-end integration tests for quantum HVAC control.
"""

import pytest
import numpy as np
import asyncio
from pathlib import Path

from quantum_ctl import HVACController, Building, MPCToQUBO
from quantum_ctl.core.controller import OptimizationConfig, ControlObjectives
from quantum_ctl.models.building import BuildingState, ZoneConfig
from quantum_ctl.optimization.quantum_solver import QuantumSolution


@pytest.mark.asyncio
class TestEndToEndOptimization:
    """Test complete optimization workflow."""
    
    async def test_simple_optimization_workflow(
        self,
        sample_building,
        sample_building_state,
        sample_weather_forecast,
        sample_energy_prices
    ):
        """Test complete optimization from building state to control schedule."""
        # Create controller with short horizon for testing
        config = OptimizationConfig(
            prediction_horizon=2,  # 2 hours
            control_interval=30,   # 30 minutes
            solver="classical_fallback",
            num_reads=50
        )
        
        objectives = ControlObjectives(
            energy_cost=0.5,
            comfort=0.4,
            carbon=0.1
        )
        
        controller = HVACController(sample_building, config, objectives)
        
        # Run optimization
        control_schedule = await controller.optimize(
            sample_building_state,
            sample_weather_forecast[:4],  # Match horizon
            sample_energy_prices[:4]
        )
        
        # Validate results
        assert isinstance(control_schedule, np.ndarray)
        assert len(control_schedule) > 0
        
        expected_length = (config.prediction_horizon * 60 // config.control_interval) * sample_building.get_control_dimension()
        assert len(control_schedule) == expected_length
        
        # Control values should be in valid range
        assert np.all(control_schedule >= 0.0)
        assert np.all(control_schedule <= 1.0)
        assert not np.any(np.isnan(control_schedule))
    
    async def test_multi_zone_optimization(self):
        """Test optimization with multiple zones."""
        # Create building with 5 zones
        zones = [
            ZoneConfig(
                zone_id=f"zone_{i}",
                area=120.0,
                volume=360.0,
                thermal_mass=300.0,
                max_heating_power=12.0,
                max_cooling_power=10.0,
                comfort_temp_min=19.0 + i * 0.5,  # Varying comfort ranges
                comfort_temp_max=23.0 + i * 0.5
            ) for i in range(5)
        ]
        
        building = Building(
            building_id="multi_zone_test",
            zones=zones,
            occupancy_schedule="office_standard"
        )
        
        # Create state with different temperatures per zone
        state = BuildingState(
            timestamp=0.0,
            zone_temperatures=np.array([20.0, 22.0, 24.0, 21.0, 23.0]),
            outside_temperature=10.0,
            humidity=45.0,
            occupancy=np.array([0.8, 0.6, 0.9, 0.4, 0.7]),
            hvac_power=np.array([6.0, 4.0, 8.0, 3.0, 5.0]),
            control_setpoints=np.array([0.6, 0.4, 0.8, 0.3, 0.5])
        )
        
        config = OptimizationConfig(
            prediction_horizon=3,
            control_interval=15,
            solver="classical_fallback"
        )
        
        controller = HVACController(building, config)
        
        # Weather and price data
        weather = np.array([[10.0, 400.0, 45.0]] * 12)  # 3 hours * 4 quarters
        prices = np.array([0.10] * 12)
        
        # Run optimization
        control_schedule = await controller.optimize(state, weather, prices)
        
        # Validate multi-zone results
        n_controls = building.get_control_dimension()
        n_steps = 3 * 60 // 15  # 3 hours, 15-min intervals
        
        assert len(control_schedule) == n_steps * n_controls
        
        # Reshape to analyze per-zone behavior
        schedule_2d = control_schedule.reshape((n_steps, n_controls))
        
        # Each zone should have reasonable control values
        for zone_i in range(n_controls):
            zone_controls = schedule_2d[:, zone_i]
            assert np.all(zone_controls >= 0.0)
            assert np.all(zone_controls <= 1.0)
            assert np.std(zone_controls) < 0.5  # Shouldn't vary too wildly
    
    async def test_optimization_with_extreme_weather(self):
        """Test optimization under extreme weather conditions."""
        building = Building(
            building_id="weather_test", 
            zones=3,
            thermal_mass=800.0,
            occupancy_schedule="office_standard"
        )
        
        # Extreme cold weather
        state = BuildingState(
            timestamp=0.0,
            zone_temperatures=np.array([18.0, 17.5, 19.0]),  # Below comfort
            outside_temperature=-15.0,  # Very cold
            humidity=70.0,
            occupancy=np.array([0.9, 0.8, 0.9]),  # High occupancy
            hvac_power=np.array([10.0, 9.0, 10.0]),  # High heating
            control_setpoints=np.array([0.9, 0.8, 0.9])
        )
        
        # Extreme weather forecast
        weather = np.array([
            [-15.0, 100.0, 70.0],  # Cold, low solar, high humidity
            [-12.0, 150.0, 65.0],
            [-10.0, 200.0, 60.0],
            [-8.0, 250.0, 55.0]
        ])
        
        # High energy prices during peak demand
        prices = np.array([0.25, 0.30, 0.35, 0.32])
        
        config = OptimizationConfig(
            prediction_horizon=1,  # 1 hour
            control_interval=15,
            solver="classical_fallback"
        )
        
        # Prioritize comfort over energy cost in extreme conditions
        objectives = ControlObjectives(
            energy_cost=0.3,
            comfort=0.6,
            carbon=0.1
        )
        
        controller = HVACController(building, config, objectives)
        
        control_schedule = await controller.optimize(state, weather, prices)
        
        # In extreme cold, system should prioritize heating
        assert len(control_schedule) == 12  # 4 steps * 3 zones
        schedule_2d = control_schedule.reshape((4, 3))
        
        # Most control values should be high (heating mode)
        assert np.mean(control_schedule) > 0.6
        
        # System should respond to extreme conditions
        assert np.max(control_schedule) > 0.8
    
    async def test_optimization_performance(self, large_building):
        """Test optimization performance with larger building."""
        # Create state for large building
        n_zones = len(large_building.zones)
        
        state = BuildingState(
            timestamp=0.0,
            zone_temperatures=np.random.uniform(20.0, 24.0, n_zones),
            outside_temperature=15.0,
            humidity=50.0,
            occupancy=np.random.uniform(0.3, 0.9, n_zones),
            hvac_power=np.random.uniform(2.0, 8.0, n_zones),
            control_setpoints=np.random.uniform(0.2, 0.8, n_zones)
        )
        
        # Large forecast data
        horizon_steps = 6 * 4  # 6 hours, 15-min intervals
        weather = np.random.normal(15.0, 5.0, (horizon_steps, 3))
        prices = np.random.uniform(0.08, 0.20, horizon_steps)
        
        config = OptimizationConfig(
            prediction_horizon=6,
            control_interval=15,
            solver="classical_fallback",
            num_reads=100  # Reduced for performance
        )
        
        controller = HVACController(large_building, config)
        
        # Time the optimization
        import time
        start_time = time.time()
        
        control_schedule = await controller.optimize(state, weather, prices)
        
        solve_time = time.time() - start_time
        
        # Should complete within reasonable time (even for fallback solver)
        assert solve_time < 30.0  # 30 seconds max
        
        # Validate large problem results
        expected_length = horizon_steps * n_zones
        assert len(control_schedule) == expected_length
        
        # Results should be reasonable
        assert np.all(control_schedule >= 0.0)
        assert np.all(control_schedule <= 1.0)
        assert not np.any(np.isnan(control_schedule))
        
        print(f"Large building optimization ({n_zones} zones) completed in {solve_time:.2f}s")
    
    async def test_optimization_error_handling(self, sample_building):
        """Test optimization with invalid inputs."""
        config = OptimizationConfig(
            prediction_horizon=1,
            solver="classical_fallback"
        )
        controller = HVACController(sample_building, config)
        
        # Test with mismatched zone count
        invalid_state = BuildingState(
            timestamp=0.0,
            zone_temperatures=np.array([22.0]),  # Wrong size
            outside_temperature=15.0,
            humidity=50.0,
            occupancy=np.array([0.5]),
            hvac_power=np.array([5.0]),
            control_setpoints=np.array([0.5])
        )
        
        weather = np.array([[15.0, 500.0, 50.0]] * 4)
        prices = np.array([0.12] * 4)
        
        # Should handle error gracefully (fallback to safe control)
        control_schedule = await controller.optimize(invalid_state, weather, prices)
        
        # Should return some control schedule (fallback)
        assert isinstance(control_schedule, np.ndarray)
        assert len(control_schedule) > 0
    
    async def test_controller_status_tracking(self, hvac_controller):
        """Test controller status and metrics tracking."""
        # Get initial status
        initial_status = hvac_controller.get_status()
        
        assert 'building_id' in initial_status
        assert 'last_optimization' in initial_status
        assert 'objectives' in initial_status
        assert 'config' in initial_status
        assert 'quantum_solver_status' in initial_status
        
        # Initially, no optimization should have run
        assert initial_status['last_optimization'] is None
        assert initial_status['history_length'] == 0


@pytest.mark.asyncio
class TestQuantumIntegration:
    """Test quantum solver integration."""
    
    async def test_quantum_solver_fallback(self, sample_qubo):
        """Test quantum solver with fallback to classical."""
        from quantum_ctl.optimization.quantum_solver import QuantumSolver
        
        # Create solver that will use classical fallback
        solver = QuantumSolver(solver_type="classical_fallback")
        
        # Test connection
        connection_result = await solver.test_connection()
        # For classical fallback, connection test might return error if D-Wave SDK not available
        # but the solver should still work
        assert connection_result['status'] in ['success', 'error']
        
        # Solve QUBO
        solution = await solver.solve(sample_qubo)
        
        assert isinstance(solution, QuantumSolution)
        assert solution.energy is not None
        assert len(solution.sample) > 0
        assert solution.chain_break_fraction == 0.0  # Classical solver
        
        # Validate solution
        for var, val in solution.sample.items():
            assert val in [0, 1]  # Binary variables
    
    async def test_mpc_to_qubo_integration(self):
        """Test MPC to QUBO conversion integration."""
        # Create simple MPC problem
        building = Building(building_id="qubo_test", zones=2)
        converter = MPCToQUBO(
            state_dim=building.get_state_dimension(),
            control_dim=building.get_control_dimension(),
            horizon=3,
            precision_bits=3
        )
        
        mcp_problem = {
            'state_dynamics': {
                'A': np.eye(building.get_state_dimension()),
                'B': np.random.randn(building.get_state_dimension(), building.get_control_dimension())
            },
            'initial_state': np.random.randn(building.get_state_dimension()),
            'objectives': {
                'weights': {'energy': 0.7, 'comfort': 0.3, 'carbon': 0.0}
            },
            'constraints': {
                'control_limits': [{'min': 0.0, 'max': 1.0}] * building.get_control_dimension(),
                'comfort_bounds': [{'temp_min': 20.0, 'temp_max': 24.0}] * building.get_control_dimension(),
                'power_limits': [{'heating_max': 10.0, 'cooling_max': -8.0}] * building.get_control_dimension()
            }
        }
        
        # Convert to QUBO
        Q = converter.to_qubo(mcp_problem)
        
        assert isinstance(Q, dict)
        assert len(Q) > 0
        
        # Solve with quantum solver
        solver = QuantumSolver(solver_type="classical_fallback")
        solution = await solver.solve(Q)
        
        # Decode back to control schedule
        control_schedule = converter.decode_solution(solution.sample)
        
        expected_length = converter.horizon * converter.control_dim
        assert len(control_schedule) == expected_length
        assert np.all(control_schedule >= 0.0)
        assert np.all(control_schedule <= 1.0)
    
    def test_solver_properties(self):
        """Test quantum solver properties and info."""
        from quantum_ctl.optimization.quantum_solver import QuantumSolver
        
        solver = QuantumSolver(solver_type="classical_fallback")
        
        # Get status
        status = solver.get_status()
        assert 'solver_type' in status
        assert 'is_available' in status
        assert 'dwave_sdk_available' in status
        
        # Get properties
        properties = solver.get_solver_properties()
        assert isinstance(properties, dict)


@pytest.mark.asyncio 
class TestRealTimeSimulation:
    """Test real-time control simulation."""
    
    async def test_control_application(self, sample_building):
        """Test applying control schedule to building."""
        # Create controller
        config = OptimizationConfig(prediction_horizon=1, solver="classical_fallback")
        controller = HVACController(sample_building, config)
        
        # Generate control schedule
        n_controls = sample_building.get_control_dimension()
        n_steps = 4  # 1 hour, 15-min intervals
        control_schedule = np.random.uniform(0.3, 0.7, n_controls * n_steps)
        
        # Apply first step
        controller.apply_schedule(control_schedule)
        
        # Check that building state was updated
        current_state = sample_building.get_state()
        expected_control = control_schedule[:n_controls]
        
        np.testing.assert_array_equal(
            current_state.control_setpoints,
            expected_control
        )
    
    async def test_building_state_evolution(self, sample_building):
        """Test building state evolution over time."""
        initial_state = sample_building.get_state()
        
        # Apply control and simulate step
        control = np.array([0.6, 0.5, 0.7])
        disturbances = np.zeros(sample_building.get_state_dimension())
        disturbances[sample_building.n_zones] = 20.0  # Outside temperature
        
        next_state = sample_building.simulate_step(control, disturbances)
        
        # State should have evolved
        assert next_state.timestamp > initial_state.timestamp
        
        # Temperatures should be influenced by control and disturbances
        temp_change = next_state.zone_temperatures - initial_state.zone_temperatures
        assert not np.allclose(temp_change, 0.0)  # Should have changed
    
    def test_constraint_validation(self, sample_building):
        """Test constraint validation for generated schedules."""
        from quantum_ctl.utils.validation import validate_control_schedule
        
        # Valid schedule
        valid_schedule = np.random.uniform(0.1, 0.9, 12)  # 3 zones * 4 time steps
        violations = validate_control_schedule(valid_schedule, sample_building, 4)
        
        assert violations['status'] != 'error'
        assert len(violations['control_bounds']) == 0
        
        # Invalid schedule (values outside bounds)
        invalid_schedule = np.array([1.5, -0.2, 0.5, 2.0, 0.0, 0.8, 0.4, 0.6, 0.3, 0.7, 0.1, 0.9])
        violations = validate_control_schedule(invalid_schedule, sample_building, 4)
        
        assert len(violations['control_bounds']) > 0
    
    def test_performance_metrics(self, hvac_controller, sample_building_state, sample_weather_forecast, sample_energy_prices):
        """Test performance metric calculation."""
        # This would test energy consumption, comfort violations, etc.
        # For now, basic validation that metrics can be computed
        
        status = hvac_controller.get_status()
        
        # Should have quantum solver status
        quantum_status = status['quantum_solver_status']
        assert 'solver_type' in quantum_status
        assert 'is_available' in quantum_status
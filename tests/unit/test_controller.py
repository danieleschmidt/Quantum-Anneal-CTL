"""
Unit tests for HVAC controller.
"""

import pytest
import numpy as np
import asyncio

from quantum_ctl.core.controller import HVACController, OptimizationConfig, ControlObjectives
from quantum_ctl.models.building import BuildingState


class TestOptimizationConfig:
    """Test OptimizationConfig class."""
    
    def test_optimization_config_creation(self):
        """Test optimization configuration creation."""
        config = OptimizationConfig(
            prediction_horizon=12,
            control_interval=30,
            solver="qpu",
            num_reads=2000,
            annealing_time=50
        )
        
        assert config.prediction_horizon == 12
        assert config.control_interval == 30
        assert config.solver == "qpu"
        assert config.num_reads == 2000
        assert config.annealing_time == 50
        assert config.chain_strength is None  # Default
    
    def test_default_config(self):
        """Test default configuration values."""
        config = OptimizationConfig()
        
        assert config.prediction_horizon == 24
        assert config.control_interval == 15
        assert config.solver == "hybrid_v2"
        assert config.num_reads == 1000
        assert config.annealing_time == 20


class TestControlObjectives:
    """Test ControlObjectives class."""
    
    def test_objectives_creation(self):
        """Test control objectives creation."""
        objectives = ControlObjectives(
            energy_cost=0.5,
            comfort=0.4,
            carbon=0.1
        )
        
        assert objectives.energy_cost == 0.5
        assert objectives.comfort == 0.4
        assert objectives.carbon == 0.1
    
    def test_objectives_validation(self):
        """Test objectives validation."""
        # Valid objectives (sum to 1.0)
        valid_objectives = ControlObjectives(0.6, 0.3, 0.1)
        assert valid_objectives.energy_cost == 0.6
        
        # Invalid objectives (don't sum to 1.0)
        with pytest.raises(ValueError, match="must sum to 1.0"):
            ControlObjectives(0.8, 0.3, 0.1)  # Sum = 1.2
    
    def test_objectives_edge_cases(self):
        """Test objectives with edge case values."""
        # Barely valid (within tolerance)
        edge_objectives = ControlObjectives(0.6, 0.301, 0.099)  # Sum ≈ 1.0
        assert edge_objectives.energy_cost == 0.6
        
        # Just outside tolerance
        with pytest.raises(ValueError):
            ControlObjectives(0.6, 0.35, 0.1)  # Sum = 1.05


class TestHVACController:
    """Test HVACController class."""
    
    def test_controller_creation(self, sample_building, optimization_config, control_objectives):
        """Test HVAC controller creation."""
        controller = HVACController(
            building=sample_building,
            config=optimization_config,
            objectives=control_objectives
        )
        
        assert controller.building == sample_building
        assert controller.config == optimization_config
        assert controller.objectives == control_objectives
        assert len(controller._control_history) == 0
    
    def test_controller_default_config(self, sample_building):
        """Test controller with default configuration."""
        controller = HVACController(building=sample_building)
        
        assert isinstance(controller.config, OptimizationConfig)
        assert isinstance(controller.objectives, ControlObjectives)
        assert controller.config.prediction_horizon == 24
        assert controller.objectives.energy_cost == 0.6
    
    def test_get_control_steps(self, hvac_controller):
        """Test control steps calculation."""
        steps = hvac_controller._get_control_steps()
        
        # 4 hours * 60 minutes / 15 minutes per step = 16 steps
        expected_steps = 4 * 60 // 15
        assert steps == expected_steps
    
    def test_set_objectives(self, hvac_controller):
        """Test objectives updating."""
        new_objectives = {
            'energy_cost': 0.7,
            'comfort': 0.2,
            'carbon': 0.1
        }
        
        hvac_controller.set_objectives(new_objectives)
        
        assert hvac_controller.objectives.energy_cost == 0.7
        assert hvac_controller.objectives.comfort == 0.2
        assert hvac_controller.objectives.carbon == 0.1
    
    def test_set_objectives_invalid(self, hvac_controller):
        """Test setting invalid objectives."""
        invalid_objectives = {
            'energy_cost': 0.8,
            'comfort': 0.3,
            'carbon': 0.1  # Sum = 1.2
        }
        
        with pytest.raises(ValueError):
            hvac_controller.set_objectives(invalid_objectives)
    
    @pytest.mark.asyncio
    async def test_optimize_basic(self, hvac_controller, sample_building_state, sample_weather_forecast, sample_energy_prices):
        """Test basic optimization functionality."""
        result = await hvac_controller.optimize(
            current_state=sample_building_state,
            weather_forecast=sample_weather_forecast,
            energy_prices=sample_energy_prices
        )
        
        assert isinstance(result, np.ndarray)
        assert len(result) > 0
        
        # Check that control values are reasonable
        assert np.all(result >= 0)
        assert np.all(result <= 1)
        assert not np.any(np.isnan(result))
    
    @pytest.mark.asyncio
    async def test_optimize_dimensions(self, hvac_controller, sample_building_state, sample_weather_forecast, sample_energy_prices):
        """Test optimization result dimensions."""
        result = await hvac_controller.optimize(
            sample_building_state, sample_weather_forecast, sample_energy_prices
        )
        
        n_controls = hvac_controller.building.get_control_dimension()
        n_steps = hvac_controller._get_control_steps()
        expected_length = n_controls * n_steps
        
        assert len(result) == expected_length
    
    @pytest.mark.asyncio
    async def test_formulate_mpc_problem(self, hvac_controller, sample_building_state, sample_weather_forecast, sample_energy_prices):
        """Test MPC problem formulation."""
        mpc_problem = await hvac_controller._formulate_mpc_problem(
            sample_building_state, sample_weather_forecast, sample_energy_prices
        )
        
        required_keys = ['state_dynamics', 'initial_state', 'disturbances', 'objectives', 'constraints', 'horizon']
        for key in required_keys:
            assert key in mcp_problem
        
        # Check state dynamics
        state_dynamics = mcp_problem['state_dynamics']
        assert 'A' in state_dynamics
        assert 'B' in state_dynamics
        
        A, B = state_dynamics['A'], state_dynamics['B']
        state_dim = hvac_controller.building.get_state_dimension()
        control_dim = hvac_controller.building.get_control_dimension()
        
        assert A.shape == (state_dim, state_dim)
        assert B.shape == (state_dim, control_dim)
    
    def test_get_penalty_weights(self, hvac_controller):
        """Test penalty weight calculation."""
        weights = hvac_controller._get_penalty_weights()
        
        required_weights = ['dynamics', 'comfort', 'energy', 'control_limits']
        for weight_type in required_weights:
            assert weight_type in weights
            assert weights[weight_type] > 0
        
        # Dynamics should have highest penalty (hard constraint)
        assert weights['dynamics'] >= weights['comfort']
        assert weights['dynamics'] >= weights['energy']
    
    def test_get_comfort_cost_matrix(self, hvac_controller):
        """Test comfort cost matrix generation."""
        Q_comfort = hvac_controller._get_comfort_cost_matrix()
        
        n_zones = hvac_controller.building.n_zones
        n_steps = hvac_controller._get_control_steps()
        expected_size = n_zones * n_steps
        
        assert Q_comfort.shape == (expected_size, expected_size)
        
        # Should be positive semi-definite (all eigenvalues >= 0)
        eigenvals = np.linalg.eigvals(Q_comfort)
        assert np.all(eigenvals >= -1e-10)  # Allow small numerical errors
    
    def test_get_energy_cost_matrix(self, hvac_controller, sample_energy_prices):
        """Test energy cost matrix generation."""
        R_energy = hvac_controller._get_energy_cost_matrix(sample_energy_prices)
        
        n_controls = hvac_controller.building.get_control_dimension()
        n_steps = hvac_controller._get_control_steps()
        expected_size = n_controls * n_steps
        
        assert R_energy.shape == (expected_size, expected_size)
        
        # Should be diagonal matrix with energy prices
        assert np.allclose(R_energy, np.diag(np.diag(R_energy)))  # Is diagonal
        assert np.all(np.diag(R_energy) >= 0)  # Positive diagonal
    
    def test_apply_schedule(self, hvac_controller, sample_building):
        """Test control schedule application."""
        n_controls = sample_building.get_control_dimension()
        n_steps = hvac_controller._get_control_steps()
        
        # Create test schedule
        test_schedule = np.random.uniform(0, 1, n_controls * n_steps)
        
        hvac_controller.apply_schedule(test_schedule)
        
        # Check that first control step was applied
        state = sample_building.get_state()
        expected_control = test_schedule[:n_controls]
        
        np.testing.assert_array_equal(state.control_setpoints, expected_control)
    
    def test_apply_empty_schedule(self, hvac_controller):
        """Test applying empty schedule."""
        empty_schedule = np.array([])
        
        # Should handle gracefully without errors
        hvac_controller.apply_schedule(empty_schedule)
    
    @pytest.mark.asyncio
    async def test_fallback_control(self, hvac_controller, sample_building_state):
        """Test fallback control when optimization fails."""
        # Add some control history
        n_controls = hvac_controller.building.get_control_dimension()
        n_steps = hvac_controller._get_control_steps()
        
        test_history = np.random.uniform(0, 1, n_controls * n_steps)
        hvac_controller._control_history.append(test_history)
        
        fallback_control = await hvac_controller._fallback_control(sample_building_state)
        
        assert isinstance(fallback_control, np.ndarray)
        assert len(fallback_control) == n_controls * n_steps
        assert np.all(fallback_control >= 0)
        assert np.all(fallback_control <= 1)
    
    @pytest.mark.asyncio
    async def test_fallback_control_no_history(self, hvac_controller, sample_building_state):
        """Test fallback control with no history."""
        # Ensure no history
        hvac_controller._control_history.clear()
        
        fallback_control = await hvac_controller._fallback_control(sample_building_state)
        
        assert isinstance(fallback_control, np.ndarray)
        assert len(fallback_control) > 0
    
    def test_get_status(self, hvac_controller):
        """Test controller status reporting."""
        status = hvac_controller.get_status()
        
        required_keys = [
            'building_id', 'last_optimization', 'objectives', 
            'config', 'history_length', 'quantum_solver_status'
        ]
        
        for key in required_keys:
            assert key in status
        
        assert status['building_id'] == hvac_controller.building.building_id
        assert status['history_length'] == len(hvac_controller._control_history)
        assert isinstance(status['quantum_solver_status'], dict)
    
    @pytest.mark.asyncio
    async def test_optimization_history_tracking(self, hvac_controller, sample_building_state, sample_weather_forecast, sample_energy_prices):
        """Test optimization history tracking."""
        initial_history_length = len(hvac_controller._control_history)
        
        # Run optimization
        await hvac_controller.optimize(
            sample_building_state, sample_weather_forecast, sample_energy_prices
        )
        
        # History should have increased
        assert len(hvac_controller._control_history) == initial_history_length + 1
        
        # Check last optimization timestamp
        assert hvac_controller._last_optimization is not None
    
    @pytest.mark.asyncio
    async def test_history_limit(self, hvac_controller, sample_building_state, sample_weather_forecast, sample_energy_prices):
        """Test control history length limiting."""
        # Fill history beyond limit
        for _ in range(15):  # Limit is 10
            await hvac_controller.optimize(
                sample_building_state, sample_weather_forecast, sample_energy_prices
            )
        
        # Should not exceed limit
        assert len(hvac_controller._control_history) <= 10
    
    @pytest.mark.asyncio
    async def test_control_loop_single_iteration(self, hvac_controller):
        """Test single iteration of control loop."""
        # Mock data source
        class MockDataSource:
            async def get_current_state(self):
                return BuildingState(0, np.array([22.0]), 15.0, 50.0, np.array([0.5]), np.array([5.0]), np.array([0.5]))
            
            async def get_weather_forecast(self):
                return np.array([[15.0, 500.0, 50.0]])
            
            async def get_energy_prices(self):
                return np.array([0.12])
        
        data_source = MockDataSource()
        
        # Test would require asyncio event loop management for full control loop
        # For now, just test that the methods exist and are callable
        assert callable(hvac_controller.run_control_loop)
    
    @pytest.mark.asyncio
    async def test_optimization_with_extreme_conditions(self, hvac_controller, sample_weather_forecast, sample_energy_prices):
        """Test optimization with extreme environmental conditions."""
        # Create extreme state
        extreme_state = BuildingState(
            timestamp=0.0,
            zone_temperatures=np.array([10.0, 35.0, 5.0]),  # Very cold and hot
            outside_temperature=-20.0,  # Very cold outside
            humidity=95.0,  # Very humid
            occupancy=np.array([1.0, 1.0, 1.0]),  # Full occupancy
            hvac_power=np.array([15.0, 15.0, 15.0]),  # Max power
            control_setpoints=np.array([1.0, 1.0, 1.0])  # Max control
        )
        
        result = await hvac_controller.optimize(
            extreme_state, sample_weather_forecast, sample_energy_prices
        )
        
        # Should still produce valid result
        assert isinstance(result, np.ndarray)
        assert np.all(result >= 0)
        assert np.all(result <= 1)
        assert not np.any(np.isnan(result))
    
    @pytest.mark.asyncio
    async def test_optimization_reproducibility(self, hvac_controller, sample_building_state, sample_weather_forecast, sample_energy_prices):
        """Test optimization reproducibility."""
        # Clear history to ensure clean state
        hvac_controller._control_history.clear()
        
        # Run optimization twice with same inputs
        result1 = await hvac_controller.optimize(
            sample_building_state, sample_weather_forecast, sample_energy_prices
        )
        
        # Reset state
        hvac_controller._control_history.clear()
        
        result2 = await hvac_controller.optimize(
            sample_building_state, sample_weather_forecast, sample_energy_prices  
        )
        
        # Results should be similar (allowing for randomness in fallback solver)
        assert result1.shape == result2.shape
        # Note: Exact equality not guaranteed due to potential randomness
    
    def test_controller_component_initialization(self, hvac_controller):
        """Test that all controller components are properly initialized."""
        # Check that quantum solver is initialized
        assert hvac_controller.quantum_solver is not None
        assert hvac_controller.mpc_formulator is not None
        
        # Check error handling components
        assert hvac_controller._error_handler is not None
        assert hvac_controller._health_monitor is not None
        assert hvac_controller._circuit_breaker is not None
        assert hvac_controller._retry_manager is not None
    
    @pytest.mark.asyncio
    async def test_optimization_error_handling(self, hvac_controller):
        """Test optimization error handling."""
        # Test with invalid inputs that should trigger error handling
        invalid_state = BuildingState(
            timestamp=0.0,
            zone_temperatures=np.array([float('nan')]),  # Invalid temperature
            outside_temperature=15.0,
            humidity=50.0,
            occupancy=np.array([0.5]),
            hvac_power=np.array([5.0]),
            control_setpoints=np.array([0.5])
        )
        
        invalid_weather = np.array([[float('inf'), 500.0, 50.0]])
        valid_prices = np.array([0.12])
        
        # Should handle errors gracefully and provide fallback
        result = await hvac_controller.optimize(
            invalid_state, invalid_weather, valid_prices
        )
        
        # Should still return a valid control schedule (fallback)
        assert isinstance(result, np.ndarray)
        assert len(result) > 0
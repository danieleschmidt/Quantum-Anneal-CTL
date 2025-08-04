"""
Pytest configuration and fixtures for quantum HVAC control tests.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import json
from typing import Generator

from quantum_ctl.models.building import Building, BuildingState, ZoneConfig
from quantum_ctl.core.controller import HVACController, OptimizationConfig, ControlObjectives
from quantum_ctl.optimization.quantum_solver import QuantumSolver


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Temporary directory fixture."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def sample_building() -> Building:
    """Create a sample building for testing."""
    zones = [
        ZoneConfig(
            zone_id=f"zone_{i}",
            area=100.0,
            volume=300.0,
            thermal_mass=200.0,
            max_heating_power=10.0,
            max_cooling_power=8.0
        ) for i in range(3)
    ]
    
    return Building(
        building_id="test_building",
        zones=zones,
        occupancy_schedule="office_standard"
    )


@pytest.fixture
def sample_building_state(sample_building: Building) -> BuildingState:
    """Create a sample building state."""
    n_zones = len(sample_building.zones)
    
    return BuildingState(
        timestamp=0.0,
        zone_temperatures=np.array([22.0, 21.5, 23.0]),
        outside_temperature=15.0,
        humidity=50.0,
        occupancy=np.array([0.7, 0.5, 0.8]),
        hvac_power=np.array([5.0, 3.0, 7.0]),
        control_setpoints=np.array([0.5, 0.3, 0.7])
    )


@pytest.fixture
def optimization_config() -> OptimizationConfig:
    """Create optimization configuration for testing."""
    return OptimizationConfig(
        prediction_horizon=4,  # Short horizon for testing
        control_interval=15,
        solver="classical_fallback",  # Use fallback for testing
        num_reads=100
    )


@pytest.fixture
def control_objectives() -> ControlObjectives:
    """Create control objectives for testing."""
    return ControlObjectives(
        energy_cost=0.6,
        comfort=0.3,
        carbon=0.1
    )


@pytest.fixture
def hvac_controller(
    sample_building: Building,
    optimization_config: OptimizationConfig,
    control_objectives: ControlObjectives
) -> HVACController:
    """Create HVAC controller for testing."""
    return HVACController(
        building=sample_building,
        config=optimization_config,
        objectives=control_objectives
    )


@pytest.fixture
def quantum_solver() -> QuantumSolver:
    """Create quantum solver for testing (with classical fallback)."""
    return QuantumSolver(
        solver_type="classical_fallback",
        num_reads=100,
        annealing_time=1
    )


@pytest.fixture
def sample_weather_forecast() -> np.ndarray:
    """Create sample weather forecast data."""
    # 4 time steps, 3 weather variables (temp, solar, humidity)
    return np.array([
        [15.0, 500.0, 50.0],
        [16.0, 600.0, 45.0],
        [17.0, 700.0, 40.0],
        [18.0, 650.0, 42.0]
    ])


@pytest.fixture
def sample_energy_prices() -> np.ndarray:
    """Create sample energy price data."""
    return np.array([0.12, 0.15, 0.18, 0.14])  # $/kWh


@pytest.fixture
def sample_qubo() -> dict:
    """Create a small sample QUBO problem."""
    return {
        (0, 0): 1.0,
        (1, 1): 1.0,
        (2, 2): 1.0,
        (0, 1): -2.0,
        (1, 2): -1.5,
        (0, 2): -1.0
    }


@pytest.fixture
def building_config_file(temp_dir: Path, sample_building: Building) -> Path:
    """Create a building configuration file."""
    config_file = temp_dir / "test_building.json"
    sample_building.save_config(config_file)
    return config_file


@pytest.fixture(scope="session")
def mock_dwave_available():
    """Mock D-Wave availability for testing."""
    # This would mock the D-Wave SDK availability
    return False


# Test data generators
@pytest.fixture
def random_building_state(sample_building: Building) -> BuildingState:
    """Generate random building state for testing."""
    n_zones = len(sample_building.zones)
    
    return BuildingState(
        timestamp=np.random.uniform(0, 86400),  # Random time in day
        zone_temperatures=np.random.uniform(18, 26, n_zones),
        outside_temperature=np.random.uniform(-10, 35),
        humidity=np.random.uniform(30, 80),
        occupancy=np.random.uniform(0, 1, n_zones),
        hvac_power=np.random.uniform(0, 10, n_zones),
        control_setpoints=np.random.uniform(0, 1, n_zones)
    )


@pytest.fixture
def large_building() -> Building:
    """Create a larger building for performance testing."""
    zones = [
        ZoneConfig(
            zone_id=f"zone_{i}",
            area=np.random.uniform(50, 200),
            volume=np.random.uniform(150, 600),
            thermal_mass=np.random.uniform(100, 500),
            max_heating_power=np.random.uniform(5, 15),
            max_cooling_power=np.random.uniform(4, 12)
        ) for i in range(20)  # 20 zones
    ]
    
    return Building(
        building_id="large_test_building",
        zones=zones,
        occupancy_schedule="office_standard"
    )


# Test utilities
def assert_control_schedule_valid(
    schedule: np.ndarray,
    building: Building,
    horizon: int
) -> None:
    """Assert that a control schedule is valid."""
    n_controls = building.get_control_dimension()
    expected_length = horizon * n_controls
    
    assert len(schedule) == expected_length, f"Schedule length {len(schedule)} != {expected_length}"
    assert np.all(schedule >= 0), "Control values must be non-negative"
    assert np.all(schedule <= 1), "Control values must be <= 1"
    assert not np.any(np.isnan(schedule)), "Control schedule contains NaN values"


def assert_building_state_valid(state: BuildingState, building: Building) -> None:
    """Assert that a building state is valid."""
    n_zones = len(building.zones)
    
    assert len(state.zone_temperatures) == n_zones
    assert len(state.occupancy) == n_zones
    assert len(state.hvac_power) == n_zones
    assert 0 <= state.humidity <= 100
    assert np.all(0 <= state.occupancy) and np.all(state.occupancy <= 1)
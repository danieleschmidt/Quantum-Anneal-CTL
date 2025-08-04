"""
Unit tests for building thermal model.
"""

import pytest
import numpy as np
from pathlib import Path
import json

from quantum_ctl.models.building import Building, BuildingState, ZoneConfig, OccupancyType


class TestZoneConfig:
    """Test ZoneConfig class."""
    
    def test_zone_config_creation(self):
        """Test zone configuration creation."""
        zone = ZoneConfig(
            zone_id="test_zone",
            area=100.0,
            volume=300.0,
            thermal_mass=200.0,
            max_heating_power=10.0,
            max_cooling_power=8.0
        )
        
        assert zone.zone_id == "test_zone"
        assert zone.area == 100.0
        assert zone.volume == 300.0
        assert zone.thermal_mass == 200.0
        assert zone.max_heating_power == 10.0
        assert zone.max_cooling_power == 8.0
        assert zone.comfort_temp_min == 20.0  # Default
        assert zone.comfort_temp_max == 24.0  # Default


class TestBuildingState:
    """Test BuildingState class."""
    
    def test_building_state_creation(self):
        """Test building state creation."""
        state = BuildingState(
            timestamp=1000.0,
            zone_temperatures=np.array([22.0, 21.0, 23.0]),
            outside_temperature=15.0,
            humidity=50.0,
            occupancy=np.array([0.7, 0.5, 0.8]),
            hvac_power=np.array([5.0, 3.0, 7.0]),
            control_setpoints=np.array([0.5, 0.3, 0.7])
        )
        
        assert state.timestamp == 1000.0
        assert len(state.zone_temperatures) == 3
        assert state.outside_temperature == 15.0
        assert state.humidity == 50.0
        np.testing.assert_array_equal(state.occupancy, [0.7, 0.5, 0.8])
    
    def test_state_to_vector(self):
        """Test state vector conversion."""
        state = BuildingState(
            timestamp=0.0,
            zone_temperatures=np.array([22.0, 21.0]),
            outside_temperature=15.0,
            humidity=50.0,
            occupancy=np.array([0.7, 0.5]),
            hvac_power=np.array([5.0, 3.0]),
            control_setpoints=np.array([0.5, 0.3])
        )
        
        vector = state.to_vector()
        expected = np.array([22.0, 21.0, 15.0, 50.0, 0.7, 0.5, 5.0, 3.0])
        np.testing.assert_array_equal(vector, expected)
    
    def test_state_from_vector(self):
        """Test state reconstruction from vector."""
        vector = np.array([22.0, 21.0, 15.0, 50.0, 0.7, 0.5, 5.0, 3.0, 0.5, 0.3])
        state = BuildingState.from_vector(vector, n_zones=2, timestamp=1000.0)
        
        assert state.timestamp == 1000.0
        np.testing.assert_array_equal(state.zone_temperatures, [22.0, 21.0])
        assert state.outside_temperature == 15.0
        assert state.humidity == 50.0
        np.testing.assert_array_equal(state.occupancy, [0.7, 0.5])
        np.testing.assert_array_equal(state.hvac_power, [5.0, 3.0])
        np.testing.assert_array_equal(state.control_setpoints, [0.5, 0.3])


class TestBuilding:
    """Test Building class."""
    
    def test_building_creation_with_zones(self):
        """Test building creation with zone configurations."""
        zones = [
            ZoneConfig(f"zone_{i}", 100.0, 300.0, 200.0, 10.0, 8.0)
            for i in range(3)
        ]
        
        building = Building("test_building", zones=zones)
        
        assert building.building_id == "test_building"
        assert building.n_zones == 3
        assert len(building.zones) == 3
        assert building.zones[0].zone_id == "zone_0"
    
    def test_building_creation_legacy_format(self):
        """Test building creation with legacy constructor."""
        building = Building(
            building_id="legacy_building",
            zones=5,  # Old format
            thermal_mass=1000.0,
            occupancy_schedule="office_standard"
        )
        
        assert building.building_id == "legacy_building"
        assert building.n_zones == 5
        assert len(building.zones) == 5
        assert all(zone.thermal_mass == 200.0 for zone in building.zones)  # 1000/5
    
    def test_get_dimensions(self, sample_building):
        """Test dimension calculations."""
        state_dim = sample_building.get_state_dimension()
        control_dim = sample_building.get_control_dimension()
        
        # 3 zones * 3 (temps + occupancy + hvac) + 2 (outside temp + humidity)
        assert state_dim == 11
        assert control_dim == 3  # One control per zone
    
    def test_get_dynamics_matrices(self, sample_building):
        """Test dynamics matrix generation."""
        A, B = sample_building.get_dynamics_matrices()
        
        state_dim = sample_building.get_state_dimension()
        control_dim = sample_building.get_control_dimension()
        
        assert A.shape == (state_dim, state_dim)
        assert B.shape == (state_dim, control_dim)
        
        # Check diagonal elements (should be positive for stability)
        assert np.all(np.diag(A) > 0)
    
    def test_get_constraints(self, sample_building):
        """Test constraint generation."""
        constraints = sample_building.get_constraints()
        
        assert 'comfort_bounds' in constraints
        assert 'power_limits' in constraints
        assert 'control_limits' in constraints
        
        assert len(constraints['comfort_bounds']) == sample_building.n_zones
        assert len(constraints['power_limits']) == sample_building.n_zones
        assert len(constraints['control_limits']) == sample_building.n_zones
        
        # Check constraint structure
        comfort_bound = constraints['comfort_bounds'][0]
        assert 'zone' in comfort_bound
        assert 'temp_min' in comfort_bound
        assert 'temp_max' in comfort_bound
        assert comfort_bound['temp_min'] < comfort_bound['temp_max']
    
    def test_predict_disturbances(self, sample_building, sample_weather_forecast):
        """Test disturbance prediction."""
        horizon = 4
        disturbances = sample_building.predict_disturbances(
            sample_weather_forecast, horizon
        )
        
        assert disturbances.shape == (horizon, sample_building.n_zones + 3)
        
        # Check weather data propagation
        np.testing.assert_array_equal(
            disturbances[:, -3], sample_weather_forecast[:, 0]  # Temperature
        )
    
    def test_apply_control(self, sample_building):
        """Test control application."""
        control = np.array([0.5, 0.7, 0.3])
        sample_building.apply_control(control)
        
        state = sample_building.get_state()
        np.testing.assert_array_equal(state.control_setpoints, control)
    
    def test_apply_control_wrong_size(self, sample_building):
        """Test control application with wrong size."""
        control = np.array([0.5, 0.7])  # Wrong size
        
        with pytest.raises(ValueError, match="Control vector length"):
            sample_building.apply_control(control)
    
    def test_simulate_step(self, sample_building):
        """Test single simulation step."""
        control = np.array([0.5, 0.6, 0.4])
        disturbances = np.zeros(sample_building.get_state_dimension())
        
        initial_state = sample_building.get_state()
        next_state = sample_building.simulate_step(control, disturbances)
        
        assert next_state.timestamp > initial_state.timestamp
        assert len(next_state.zone_temperatures) == sample_building.n_zones
    
    def test_occupancy_prediction(self, sample_building):
        """Test occupancy prediction patterns."""
        # Test office hours
        office_occupancy = sample_building._predict_occupancy(36)  # 9 AM (36 * 15min)
        residential_occupancy = sample_building._predict_occupancy(4)  # 1 AM
        
        # Office hours should have higher occupancy
        assert np.mean(office_occupancy) > np.mean(residential_occupancy)
        assert np.all(office_occupancy >= 0) and np.all(office_occupancy <= 1)
    
    def test_building_serialization(self, sample_building, temp_dir):
        """Test building configuration save/load."""
        config_file = temp_dir / "building_config.json"
        
        # Save configuration
        sample_building.save_config(config_file)
        assert config_file.exists()
        
        # Load configuration
        loaded_building = Building.load_config(config_file)
        
        assert loaded_building.building_id == sample_building.building_id
        assert loaded_building.n_zones == sample_building.n_zones
        assert len(loaded_building.zones) == len(sample_building.zones)
        
        # Check zone details
        for orig_zone, loaded_zone in zip(sample_building.zones, loaded_building.zones):
            assert orig_zone.zone_id == loaded_zone.zone_id
            assert orig_zone.area == loaded_zone.area
            assert orig_zone.thermal_mass == loaded_zone.thermal_mass
    
    def test_to_dict_from_dict(self, sample_building):
        """Test dictionary serialization."""
        building_dict = sample_building.to_dict()
        
        assert 'building_id' in building_dict
        assert 'zones' in building_dict
        assert 'thermal_coupling' in building_dict
        assert isinstance(building_dict['zones'], list)
        
        # Reconstruct from dictionary
        reconstructed = Building.from_dict(building_dict)
        
        assert reconstructed.building_id == sample_building.building_id
        assert reconstructed.n_zones == sample_building.n_zones
        np.testing.assert_array_equal(
            reconstructed.thermal_coupling,
            sample_building.thermal_coupling
        )
    
    def test_location_property(self, sample_building):
        """Test location property."""
        lat, lon = sample_building.location
        
        assert isinstance(lat, float)
        assert isinstance(lon, float)
        assert -90 <= lat <= 90
        assert -180 <= lon <= 180
    
    def test_default_coupling_matrix(self):
        """Test default thermal coupling matrix generation."""
        zones = [ZoneConfig(f"zone_{i}", 100, 300, 200, 10, 8) for i in range(4)]
        building = Building("test", zones=zones)
        
        coupling = building.thermal_coupling
        
        assert coupling.shape == (4, 4)
        # Diagonal should be negative (self-coupling)
        assert np.all(np.diag(coupling) < 0)
        # Adjacent zones should have positive coupling
        assert coupling[0, 1] > 0
        assert coupling[1, 0] > 0
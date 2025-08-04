"""
Building thermal model and state representation.
"""

from typing import Dict, Any, Optional, Tuple, List
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path


class OccupancyType(Enum):
    """Standard occupancy schedule types."""
    OFFICE_STANDARD = "office_standard"
    RESIDENTIAL = "residential"
    RETAIL = "retail"
    HOSPITAL = "hospital"
    SCHOOL = "school"
    CUSTOM = "custom"


@dataclass
class ZoneConfig:
    """Configuration for a single thermal zone."""
    zone_id: str
    area: float  # m²
    volume: float  # m³
    thermal_mass: float  # kJ/K
    max_heating_power: float  # kW
    max_cooling_power: float  # kW
    comfort_temp_min: float = 20.0  # °C
    comfort_temp_max: float = 24.0  # °C
    occupancy_schedule: Optional[np.ndarray] = None


@dataclass
class BuildingState:
    """Current state of the building system."""
    timestamp: float
    zone_temperatures: np.ndarray  # °C
    outside_temperature: float  # °C
    humidity: float  # %
    occupancy: np.ndarray  # People per zone
    hvac_power: np.ndarray  # kW per zone
    control_setpoints: np.ndarray  # Control settings
    
    def to_vector(self) -> np.ndarray:
        """Convert state to flat vector for optimization."""
        return np.concatenate([
            self.zone_temperatures,
            [self.outside_temperature, self.humidity],
            self.occupancy,
            self.hvac_power
        ])
    
    @classmethod
    def from_vector(cls, vector: np.ndarray, n_zones: int, timestamp: float):
        """Reconstruct state from flat vector."""
        idx = 0
        zone_temps = vector[idx:idx+n_zones]
        idx += n_zones
        
        outside_temp = vector[idx]
        humidity = vector[idx+1]
        idx += 2
        
        occupancy = vector[idx:idx+n_zones]
        idx += n_zones
        
        hvac_power = vector[idx:idx+n_zones]
        idx += n_zones
        
        control_setpoints = vector[idx:] if idx < len(vector) else np.zeros(n_zones)
        
        return cls(
            timestamp=timestamp,
            zone_temperatures=zone_temps,
            outside_temperature=outside_temp,
            humidity=humidity,
            occupancy=occupancy,
            hvac_power=hvac_power,
            control_setpoints=control_setpoints
        )


class Building:
    """
    Building thermal model for HVAC optimization.
    
    Represents a multi-zone building with thermal dynamics suitable for
    quantum optimization via Model Predictive Control.
    """
    
    def __init__(
        self,
        building_id: str,
        zones: List[ZoneConfig] = None,
        thermal_coupling_matrix: Optional[np.ndarray] = None,
        **kwargs
    ):
        self.building_id = building_id
        
        # Handle legacy constructor format
        if zones is None and 'zones' in kwargs:
            n_zones = kwargs['zones']
            zones = [ZoneConfig(
                zone_id=f"zone_{i}",
                area=100.0,  # m²
                volume=300.0,  # m³
                thermal_mass=kwargs.get('thermal_mass', 1000.0) // n_zones,
                max_heating_power=10.0,  # kW
                max_cooling_power=8.0   # kW
            ) for i in range(n_zones)]
        elif isinstance(zones, int):
            # Handle case where zones is passed as integer directly
            n_zones = zones
            zones = [ZoneConfig(
                zone_id=f"zone_{i}",
                area=100.0,
                volume=300.0,
                thermal_mass=kwargs.get('thermal_mass', 1000.0) // n_zones,
                max_heating_power=10.0,
                max_cooling_power=8.0
            ) for i in range(n_zones)]
        
        self.zones = zones or []
        self.n_zones = len(self.zones)
        
        # Thermal coupling between zones
        if thermal_coupling_matrix is not None:
            self.thermal_coupling = thermal_coupling_matrix
        elif 'heat_transfer_matrix' in kwargs:
            self.thermal_coupling = kwargs['heat_transfer_matrix']
        else:
            self.thermal_coupling = self._default_coupling_matrix()
        
        # Occupancy schedule
        self.occupancy_type = kwargs.get('occupancy_schedule', OccupancyType.OFFICE_STANDARD)
        
        # Building envelope parameters
        self.envelope_ua = kwargs.get('envelope_ua', 500.0)  # kW/K
        self.window_area = kwargs.get('window_area', 200.0)  # m²
        self.solar_heat_gain_coeff = kwargs.get('shgc', 0.6)
        
        # Geographic location for weather
        self.latitude = kwargs.get('latitude', 40.7128)  # NYC default
        self.longitude = kwargs.get('longitude', -74.0060)
        
        self._current_state: Optional[BuildingState] = None
    
    def _default_coupling_matrix(self) -> np.ndarray:
        """Generate default thermal coupling matrix between zones."""
        coupling = np.eye(self.n_zones) * -0.5  # Self-coupling
        
        # Adjacent zone coupling
        for i in range(self.n_zones - 1):
            coupling[i, i+1] = 0.1
            coupling[i+1, i] = 0.1
        
        return coupling
    
    def get_state_dimension(self) -> int:
        """Get dimension of state vector."""
        return self.n_zones * 3 + 2  # temps + occupancy + hvac + outside_temp + humidity
    
    def get_control_dimension(self) -> int:
        """Get dimension of control vector."""
        return self.n_zones  # One control per zone
    
    def get_dynamics_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get discrete-time state-space matrices A, B for thermal dynamics.
        
        State: [zone_temps, outside_temp, humidity, occupancy, hvac_power]
        Control: [hvac_setpoints]
        """
        dt = 900.0  # 15 minutes in seconds
        
        # State matrix A (thermal dynamics)
        n_states = self.get_state_dimension()
        A = np.eye(n_states)
        
        # Zone temperature dynamics
        for i in range(self.n_zones):
            zone = self.zones[i]
            thermal_time_constant = zone.thermal_mass / (self.envelope_ua / self.n_zones)
            
            # Temperature decay to outside
            A[i, i] = np.exp(-dt / thermal_time_constant)
            
            # Coupling with outside temperature
            A[i, self.n_zones] = 1 - A[i, i]
            
            # Inter-zone coupling
            for j in range(self.n_zones):
                if i != j:
                    coupling_strength = self.thermal_coupling[i, j]
                    A[i, j] = coupling_strength * dt / zone.thermal_mass
        
        # Outside temperature and humidity are external (identity)
        # Occupancy evolves slowly (high persistence)
        for i in range(self.n_zones):
            occ_idx = self.n_zones + 2 + i
            A[occ_idx, occ_idx] = 0.95  # Occupancy persistence
        
        # Control matrix B
        B = np.zeros((n_states, self.n_zones))
        
        for i in range(self.n_zones):
            zone = self.zones[i]
            # HVAC power affects zone temperature
            hvac_effectiveness = dt / zone.thermal_mass
            B[i, i] = hvac_effectiveness
        
        return A, B
    
    def get_constraints(self) -> Dict[str, Any]:
        """Get optimization constraints for MPC."""
        constraints = {
            'comfort_bounds': [],
            'power_limits': [],
            'control_limits': []
        }
        
        for i, zone in enumerate(self.zones):
            # Comfort temperature bounds
            constraints['comfort_bounds'].append({
                'zone': i,
                'temp_min': zone.comfort_temp_min,
                'temp_max': zone.comfort_temp_max
            })
            
            # Power limits
            constraints['power_limits'].append({
                'zone': i,
                'heating_max': zone.max_heating_power,
                'cooling_max': -zone.max_cooling_power
            })
            
            # Control signal limits (normalized 0-1)
            constraints['control_limits'].append({
                'zone': i,
                'min': 0.0,
                'max': 1.0
            })
        
        return constraints
    
    def predict_disturbances(
        self,
        weather_forecast: np.ndarray,
        horizon: int
    ) -> np.ndarray:
        """Predict external disturbances (weather, occupancy, solar gain)."""
        disturbances = np.zeros((horizon, self.n_zones + 3))  # zones + weather + solar
        
        for k in range(horizon):
            # Weather disturbances
            if k < len(weather_forecast):
                outside_temp = weather_forecast[k, 0] if weather_forecast.ndim > 1 else weather_forecast[k]
                solar_radiation = weather_forecast[k, 1] if weather_forecast.ndim > 1 and weather_forecast.shape[1] > 1 else 500.0
                humidity = weather_forecast[k, 2] if weather_forecast.ndim > 1 and weather_forecast.shape[1] > 2 else 50.0
            else:
                outside_temp = 20.0  # Default
                solar_radiation = 200.0
                humidity = 50.0
            
            disturbances[k, -3] = outside_temp
            disturbances[k, -2] = solar_radiation  
            disturbances[k, -1] = humidity
            
            # Occupancy predictions
            occupancy = self._predict_occupancy(k)
            disturbances[k, :self.n_zones] = occupancy
        
        return disturbances
    
    def _predict_occupancy(self, time_step: int) -> np.ndarray:
        """Predict occupancy for each zone at given time step."""
        # Simple occupancy model based on schedule type
        hour_of_day = (time_step * 0.25) % 24  # 15-min steps
        
        if self.occupancy_type == OccupancyType.OFFICE_STANDARD:
            if 8 <= hour_of_day <= 18:
                base_occupancy = np.random.normal(0.7, 0.1, self.n_zones)
            else:
                base_occupancy = np.random.normal(0.1, 0.05, self.n_zones)
        else:
            # Default residential pattern
            if 7 <= hour_of_day <= 9 or 17 <= hour_of_day <= 22:
                base_occupancy = np.random.normal(0.8, 0.1, self.n_zones)
            else:
                base_occupancy = np.random.normal(0.3, 0.1, self.n_zones)
        
        return np.clip(base_occupancy, 0.0, 1.0)
    
    def apply_control(self, control_vector: np.ndarray) -> None:
        """Apply control commands to the building."""
        if len(control_vector) != self.n_zones:
            raise ValueError(f"Control vector length {len(control_vector)} != {self.n_zones}")
        
        # In real implementation, this would send commands to BMS
        # For now, update internal state
        
        if self._current_state is not None:
            self._current_state.control_setpoints = control_vector.copy()
    
    def get_state(self) -> BuildingState:
        """Get current building state."""
        if self._current_state is None:
            # Initialize with default state
            self._current_state = BuildingState(
                timestamp=0.0,
                zone_temperatures=np.full(self.n_zones, 22.0),  # °C
                outside_temperature=15.0,
                humidity=50.0,
                occupancy=np.full(self.n_zones, 0.5),
                hvac_power=np.zeros(self.n_zones),
                control_setpoints=np.full(self.n_zones, 0.5)
            )
        
        return self._current_state
    
    def update_state(self, new_state: BuildingState) -> None:
        """Update building state."""
        self._current_state = new_state
    
    def simulate_step(self, control: np.ndarray, disturbances: np.ndarray) -> BuildingState:
        """Simulate one time step of building dynamics."""
        current = self.get_state()
        A, B = self.get_dynamics_matrices()
        
        # State vector
        x = current.to_vector()
        
        # Apply dynamics
        x_next = A @ x + B @ control
        
        # Add disturbances
        if len(disturbances) >= len(x):
            x_next += disturbances[:len(x)]
        
        # Create new state
        next_state = BuildingState.from_vector(
            x_next, 
            self.n_zones, 
            current.timestamp + 900  # 15 minutes
        )
        
        self.update_state(next_state)
        return next_state
    
    @property
    def location(self) -> Tuple[float, float]:
        """Get building location as (latitude, longitude)."""
        return (self.latitude, self.longitude)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize building configuration."""
        return {
            'building_id': self.building_id,
            'zones': [
                {
                    'zone_id': z.zone_id,
                    'area': z.area,
                    'volume': z.volume,  
                    'thermal_mass': z.thermal_mass,
                    'max_heating_power': z.max_heating_power,
                    'max_cooling_power': z.max_cooling_power,
                    'comfort_temp_min': z.comfort_temp_min,
                    'comfort_temp_max': z.comfort_temp_max
                } for z in self.zones
            ],
            'thermal_coupling': self.thermal_coupling.tolist(),
            'occupancy_schedule': self.occupancy_type.value,
            'envelope_ua': self.envelope_ua,
            'location': {'lat': self.latitude, 'lon': self.longitude}
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Building':
        """Deserialize building from dictionary."""
        zones = [ZoneConfig(**z) for z in data['zones']]
        
        return cls(
            building_id=data['building_id'],
            zones=zones,
            thermal_coupling_matrix=np.array(data['thermal_coupling']),
            occupancy_schedule=OccupancyType(data.get('occupancy_schedule', 'office_standard')),
            envelope_ua=data.get('envelope_ua', 500.0),
            latitude=data.get('location', {}).get('lat', 40.7128),
            longitude=data.get('location', {}).get('lon', -74.0060)
        )
    
    def save_config(self, filepath: Path) -> None:
        """Save building configuration to file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load_config(cls, filepath: Path) -> 'Building':
        """Load building configuration from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
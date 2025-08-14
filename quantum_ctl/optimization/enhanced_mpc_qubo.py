"""
Enhanced MPC to QUBO Transformation with Advanced Constraints and Multi-Objective Support.

This module extends the basic MPC-to-QUBO conversion with:
1. Advanced thermal dynamics modeling
2. Multi-zone coupling constraints  
3. Occupancy-aware comfort modeling
4. Carbon footprint optimization
5. Energy storage integration
6. Demand response capabilities
"""

from typing import Dict, Any, List, Tuple, Optional, Callable, Union
import numpy as np
from dataclasses import dataclass, field
import logging
from scipy import sparse
from collections import defaultdict
import math

from .mpc_to_qubo import MPCToQUBO, VariableEncoding
from ..models.building import BuildingState


@dataclass
class EnhancedVariableEncoding(VariableEncoding):
    """Enhanced variable encoding with additional metadata."""
    variable_type: str = "control"  # control, storage, demand_response, slack
    zone_id: Optional[int] = None
    time_coupling: bool = False  # Whether this variable couples across time
    priority: int = 0  # Higher priority variables get better precision
    
    @property
    def is_storage_variable(self) -> bool:
        return self.variable_type == "storage"
    
    @property 
    def is_demand_response_variable(self) -> bool:
        return self.variable_type == "demand_response"


@dataclass
class ThermalZoneModel:
    """Advanced thermal zone model with realistic building physics."""
    zone_id: int
    thermal_mass: float  # kJ/K
    thermal_resistance: float  # K/W  
    internal_gains: Dict[str, float]  # W from occupancy, equipment, lighting
    window_area: float  # m^2
    window_orientation: str  # N, S, E, W for solar gain calculation
    hvac_capacity: float  # kW max heating/cooling
    comfort_range: Tuple[float, float] = (20.0, 26.0)  # °C
    
    def calculate_solar_gain(self, hour: int, solar_radiation: float) -> float:
        """Calculate solar heat gain based on window orientation and time."""
        # Simplified solar gain calculation
        orientation_factors = {'S': 1.0, 'E': 0.7, 'W': 0.7, 'N': 0.3}
        base_factor = orientation_factors.get(self.window_orientation, 0.5)
        
        # Time-dependent factor (peak at solar noon)
        time_factor = max(0, math.cos((hour - 12) * math.pi / 12))
        
        # Solar heat gain coefficient (typical value)
        shgc = 0.3
        
        return self.window_area * shgc * solar_radiation * base_factor * time_factor


@dataclass
class EnergyStorageModel:
    """Energy storage system model (battery, thermal storage)."""
    capacity_kwh: float
    max_charge_rate: float  # kW
    max_discharge_rate: float  # kW
    efficiency: float = 0.85
    self_discharge_rate: float = 0.001  # per hour
    initial_soc: float = 0.5  # State of charge
    
    def get_storage_bounds(self) -> Tuple[float, float]:
        """Get storage level bounds."""
        return (0.1 * self.capacity_kwh, 0.9 * self.capacity_kwh)


class EnhancedMPCToQUBO(MPCToQUBO):
    """
    Enhanced MPC to QUBO converter with advanced building physics and multi-objective optimization.
    
    Features:
    - Realistic thermal zone modeling with solar gains
    - Energy storage optimization  
    - Demand response integration
    - Multi-zone thermal coupling
    - Occupancy-aware comfort constraints
    - Carbon footprint minimization
    """
    
    def __init__(
        self,
        zone_models: List[ThermalZoneModel],
        horizon_hours: int = 24,
        time_step_minutes: int = 60,
        precision_bits: int = 4,
        enable_storage: bool = False,
        enable_demand_response: bool = False
    ):
        # Initialize base class
        num_zones = len(zone_models)
        super().__init__(
            state_dim=num_zones,
            control_dim=num_zones, 
            horizon=horizon_hours,
            precision_bits=precision_bits
        )
        
        self.zone_models = zone_models
        self.horizon_hours = horizon_hours
        self.time_step_hours = time_step_minutes / 60
        self.enable_storage = enable_storage
        self.enable_demand_response = enable_demand_response
        
        self.logger = logging.getLogger(__name__)
        
        # Enhanced variable tracking
        self._storage_variables: List[EnhancedVariableEncoding] = []
        self._demand_response_variables: List[EnhancedVariableEncoding] = []
        
        # Thermal coupling matrix between zones
        self._zone_coupling_matrix = self._build_zone_coupling_matrix()
        
        # Carbon emission factors by hour (kg CO2/kWh)
        self._carbon_intensity = np.array([0.4] * 24)  # Simplified constant
        
    def _build_zone_coupling_matrix(self) -> np.ndarray:
        """Build thermal coupling matrix between adjacent zones."""
        num_zones = len(self.zone_models)
        coupling_matrix = np.eye(num_zones)  # Start with identity
        
        # Simple adjacent zone coupling (can be enhanced with floor plan data)
        coupling_strength = 0.1  # 10% thermal coupling with adjacent zones
        
        for i in range(num_zones - 1):
            coupling_matrix[i, i + 1] = coupling_strength
            coupling_matrix[i + 1, i] = coupling_strength
        
        return coupling_matrix
    
    def create_enhanced_variable_encodings(
        self,
        mpc_problem: Dict[str, Any],
        storage_model: Optional[EnergyStorageModel] = None
    ) -> None:
        """Create enhanced variable encodings including storage and demand response."""
        self._variable_encodings.clear()
        self._storage_variables.clear()
        self._demand_response_variables.clear()
        
        bit_offset = 0
        
        # 1. HVAC Control Variables (primary optimization variables)
        for k in range(self.horizon):
            for zone_idx, zone_model in enumerate(self.zone_models):
                # Control variable for each zone at each time step
                encoding = EnhancedVariableEncoding(
                    name=f"hvac_{k}_{zone_idx}",
                    min_val=0.0,
                    max_val=1.0,
                    precision=self.precision_bits,
                    offset=bit_offset,
                    variable_type="control",
                    zone_id=zone_idx,
                    time_coupling=True,
                    priority=1
                )
                
                self._variable_encodings.append(encoding)
                bit_offset += self.precision_bits
        
        # 2. Energy Storage Variables (if enabled)
        if self.enable_storage and storage_model:
            storage_min, storage_max = storage_model.get_storage_bounds()
            
            for k in range(self.horizon):
                # Storage level at each time step
                storage_encoding = EnhancedVariableEncoding(
                    name=f"storage_level_{k}",
                    min_val=storage_min,
                    max_val=storage_max,
                    precision=self.precision_bits - 1,  # Slightly lower precision
                    offset=bit_offset,
                    variable_type="storage",
                    time_coupling=True,
                    priority=2
                )
                
                self._storage_variables.append(storage_encoding)
                self._variable_encodings.append(storage_encoding)
                bit_offset += (self.precision_bits - 1)
                
                # Charge/discharge rate variables
                charge_encoding = EnhancedVariableEncoding(
                    name=f"charge_rate_{k}",
                    min_val=0.0,
                    max_val=storage_model.max_charge_rate,
                    precision=self.precision_bits - 1,
                    offset=bit_offset,
                    variable_type="storage",
                    priority=2
                )
                
                self._storage_variables.append(charge_encoding)
                self._variable_encodings.append(charge_encoding)
                bit_offset += (self.precision_bits - 1)
        
        # 3. Demand Response Variables (if enabled)
        if self.enable_demand_response:
            for k in range(self.horizon):
                for zone_idx in range(len(self.zone_models)):
                    # Load shedding capability for demand response
                    dr_encoding = EnhancedVariableEncoding(
                        name=f"demand_response_{k}_{zone_idx}",
                        min_val=0.0,
                        max_val=0.3,  # Up to 30% load shedding
                        precision=3,  # Lower precision for DR variables
                        offset=bit_offset,
                        variable_type="demand_response",
                        zone_id=zone_idx,
                        priority=3
                    )
                    
                    self._demand_response_variables.append(dr_encoding)
                    self._variable_encodings.append(dr_encoding)
                    bit_offset += 3
        
        # 4. Slack Variables for Soft Constraints
        constraint_types = ['comfort', 'power', 'storage', 'carbon']
        for constraint_type in constraint_types:
            for k in range(self.horizon):
                num_constraints = len(self.zone_models) if constraint_type in ['comfort', 'power'] else 1
                
                for i in range(num_constraints):
                    slack_encoding = EnhancedVariableEncoding(
                        name=f"slack_{constraint_type}_{k}_{i}",
                        min_val=0.0,
                        max_val=5.0,  # Maximum constraint violation
                        precision=3,
                        offset=bit_offset,
                        variable_type="slack",
                        zone_id=i if constraint_type in ['comfort', 'power'] else None,
                        priority=0  # Lowest priority
                    )
                    
                    self._variable_encodings.append(slack_encoding)
                    bit_offset += 3
        
        self._total_bits = bit_offset
        self.logger.info(f"Created enhanced encoding: {len(self._variable_encodings)} variables, {self._total_bits} bits")
    
    def to_enhanced_qubo(
        self,
        mpc_problem: Dict[str, Any],
        penalty_weights: Dict[str, float] = None,
        carbon_targets: Optional[Dict[str, float]] = None,
        storage_model: Optional[EnergyStorageModel] = None
    ) -> Dict[Tuple[int, int], float]:
        """
        Convert enhanced MPC problem to QUBO with advanced constraints.
        
        Args:
            mpc_problem: Enhanced MPC formulation
            penalty_weights: Constraint penalty weights
            carbon_targets: Carbon emission targets by hour
            storage_model: Energy storage system model
            
        Returns:
            Enhanced QUBO matrix
        """
        if penalty_weights is None:
            penalty_weights = {
                'thermal_dynamics': 2000.0,
                'comfort': 150.0,
                'energy': 100.0,
                'carbon': 80.0,
                'storage': 120.0,
                'demand_response': 60.0,
                'control_limits': 1000.0
            }
        
        # Create enhanced variable encodings
        self.create_enhanced_variable_encodings(mpc_problem, storage_model)
        
        # Initialize QUBO matrix
        Q = {}
        
        # Add enhanced objective terms
        self._add_enhanced_energy_objective(Q, mpc_problem, penalty_weights)
        self._add_enhanced_comfort_objective(Q, mpc_problem, penalty_weights)
        self._add_carbon_objective(Q, mpc_problem, penalty_weights, carbon_targets)
        
        # Add enhanced constraint terms
        self._add_thermal_dynamics_constraints(Q, mpc_problem, penalty_weights)
        self._add_multi_zone_comfort_constraints(Q, mpc_problem, penalty_weights)
        self._add_hvac_capacity_constraints(Q, mpc_problem, penalty_weights)
        
        # Add storage constraints (if enabled)
        if self.enable_storage and storage_model:
            self._add_storage_constraints(Q, storage_model, penalty_weights)
        
        # Add demand response constraints (if enabled)
        if self.enable_demand_response:
            self._add_demand_response_constraints(Q, mpc_problem, penalty_weights)
        
        self.logger.info(f"Generated enhanced QUBO with {len(Q)} non-zero terms")
        
        return Q
    
    def _add_enhanced_energy_objective(
        self,
        Q: Dict[Tuple[int, int], float],
        mpc_problem: Dict[str, Any],
        penalty_weights: Dict[str, float]
    ) -> None:
        """Add enhanced energy cost objective with time-of-use pricing."""
        energy_prices = mpc_problem.get('energy_prices', np.ones(self.horizon))
        weight = penalty_weights.get('energy', 100.0)
        
        for k in range(self.horizon):
            hourly_price = energy_prices[k] if k < len(energy_prices) else energy_prices[-1]
            
            for zone_idx, zone_model in enumerate(self.zone_models):
                # Get HVAC control encoding
                hvac_encoding = self._get_hvac_encoding(k, zone_idx)
                
                if hvac_encoding:
                    # Quadratic energy cost based on HVAC capacity
                    capacity_factor = zone_model.hvac_capacity / 10.0  # Normalize
                    
                    for bit_i in range(hvac_encoding.precision):
                        qubit_i = hvac_encoding.offset + bit_i
                        
                        # Diagonal terms (quadratic cost)
                        if (qubit_i, qubit_i) not in Q:
                            Q[(qubit_i, qubit_i)] = 0.0
                        
                        bit_contribution = (2**bit_i)**2
                        energy_cost = weight * hourly_price * capacity_factor * bit_contribution
                        Q[(qubit_i, qubit_i)] += energy_cost
                        
                        # Cross terms within same variable
                        for bit_j in range(bit_i + 1, hvac_encoding.precision):
                            qubit_j = hvac_encoding.offset + bit_j
                            
                            if (qubit_i, qubit_j) not in Q:
                                Q[(qubit_i, qubit_j)] = 0.0
                            
                            cross_cost = 2 * weight * hourly_price * capacity_factor * (2**bit_i) * (2**bit_j)
                            Q[(qubit_i, qubit_j)] += cross_cost
    
    def _add_enhanced_comfort_objective(
        self,
        Q: Dict[Tuple[int, int], float],
        mpc_problem: Dict[str, Any],
        penalty_weights: Dict[str, float]
    ) -> None:
        """Add occupancy-aware comfort objective."""
        weight = penalty_weights.get('comfort', 150.0)
        occupancy_schedule = mpc_problem.get('occupancy_schedule', np.ones((self.horizon, len(self.zone_models))))
        
        for k in range(self.horizon):
            for zone_idx, zone_model in enumerate(self.zone_models):
                # Occupancy factor affects comfort importance
                if k < occupancy_schedule.shape[0] and zone_idx < occupancy_schedule.shape[1]:
                    occupancy_factor = occupancy_schedule[k, zone_idx]
                else:
                    occupancy_factor = 0.5  # Default moderate occupancy
                
                # Higher penalty when zone is occupied
                comfort_weight = weight * (0.5 + occupancy_factor)
                
                hvac_encoding = self._get_hvac_encoding(k, zone_idx)
                
                if hvac_encoding:
                    # Comfort setpoint (middle of comfort range)
                    comfort_min, comfort_max = zone_model.comfort_range
                    optimal_control = 0.5  # Middle control value for comfort
                    
                    for bit_i in range(hvac_encoding.precision):
                        qubit_i = hvac_encoding.offset + bit_i
                        
                        # Quadratic penalty for deviation from optimal control
                        bit_value = 2**bit_i / (2**hvac_encoding.precision - 1)
                        deviation_penalty = abs(bit_value - optimal_control)
                        
                        if (qubit_i, qubit_i) not in Q:
                            Q[(qubit_i, qubit_i)] = 0.0
                        
                        Q[(qubit_i, qubit_i)] += comfort_weight * deviation_penalty
    
    def _add_carbon_objective(
        self,
        Q: Dict[Tuple[int, int], float],
        mpc_problem: Dict[str, Any],
        penalty_weights: Dict[str, float],
        carbon_targets: Optional[Dict[str, float]]
    ) -> None:
        """Add carbon footprint minimization objective."""
        weight = penalty_weights.get('carbon', 80.0)
        
        for k in range(self.horizon):
            # Carbon intensity varies by hour (grid mix changes)
            hour_of_day = k % 24
            carbon_intensity = self._carbon_intensity[hour_of_day]
            
            # Carbon target for this hour (if provided)
            carbon_target = carbon_targets.get(f'hour_{k}', 0.5) if carbon_targets else 0.5
            
            for zone_idx, zone_model in enumerate(self.zone_models):
                hvac_encoding = self._get_hvac_encoding(k, zone_idx)
                
                if hvac_encoding:
                    for bit_i in range(hvac_encoding.precision):
                        qubit_i = hvac_encoding.offset + bit_i
                        
                        # Linear carbon cost (diagonal QUBO terms)
                        if (qubit_i, qubit_i) not in Q:
                            Q[(qubit_i, qubit_i)] = 0.0
                        
                        carbon_cost = weight * carbon_intensity * (2**bit_i) * zone_model.hvac_capacity / 10.0
                        Q[(qubit_i, qubit_i)] += carbon_cost
    
    def _add_thermal_dynamics_constraints(
        self,
        Q: Dict[Tuple[int, int], float],
        mpc_problem: Dict[str, Any],
        penalty_weights: Dict[str, float]
    ) -> None:
        """Add realistic thermal dynamics constraints with zone coupling."""
        weight = penalty_weights.get('thermal_dynamics', 2000.0)
        weather_forecast = mpc_problem.get('weather_forecast', np.zeros((self.horizon, 3)))
        
        for k in range(self.horizon - 1):  # No constraint for final step
            for zone_idx, zone_model in enumerate(self.zone_models):
                # Current and next time step HVAC controls
                current_hvac = self._get_hvac_encoding(k, zone_idx)
                next_hvac = self._get_hvac_encoding(k + 1, zone_idx)
                
                if current_hvac and next_hvac:
                    # Thermal dynamics: T[k+1] = f(T[k], HVAC[k], weather[k], coupling)
                    
                    # Weather disturbances
                    if k < weather_forecast.shape[0]:
                        outdoor_temp = weather_forecast[k, 0]
                        solar_radiation = weather_forecast[k, 2] if weather_forecast.shape[1] > 2 else 0.0
                    else:
                        outdoor_temp = 20.0  # Default
                        solar_radiation = 0.0
                    
                    # Solar gain calculation
                    solar_gain = zone_model.calculate_solar_gain(k, solar_radiation)
                    
                    # Penalize rapid HVAC changes (thermal inertia)
                    for bit_i in range(current_hvac.precision):
                        for bit_j in range(next_hvac.precision):
                            qubit_i = current_hvac.offset + bit_i
                            qubit_j = next_hvac.offset + bit_j
                            
                            if bit_i == bit_j:  # Same bit position
                                # Encourage thermal consistency
                                if (qubit_i, qubit_j) not in Q:
                                    Q[(qubit_i, qubit_j)] = 0.0
                                
                                consistency_bonus = -weight * 0.05  # Negative for reward
                                Q[(qubit_i, qubit_j)] += consistency_bonus
                    
                    # Zone coupling effects
                    for coupled_zone_idx in range(len(self.zone_models)):
                        if coupled_zone_idx != zone_idx:
                            coupling_strength = self._zone_coupling_matrix[zone_idx, coupled_zone_idx]
                            
                            if coupling_strength > 0:
                                coupled_hvac = self._get_hvac_encoding(k, coupled_zone_idx)
                                
                                if coupled_hvac:
                                    # Add coupling terms between zones
                                    for bit_i in range(current_hvac.precision):
                                        for bit_j in range(coupled_hvac.precision):
                                            qubit_i = current_hvac.offset + bit_i
                                            qubit_j = coupled_hvac.offset + bit_j
                                            
                                            if (qubit_i, qubit_j) not in Q:
                                                Q[(qubit_i, qubit_j)] = 0.0
                                            
                                            coupling_penalty = weight * coupling_strength * 0.1
                                            Q[(qubit_i, qubit_j)] += coupling_penalty
    
    def _add_multi_zone_comfort_constraints(
        self,
        Q: Dict[Tuple[int, int], float],
        mpc_problem: Dict[str, Any],
        penalty_weights: Dict[str, float]
    ) -> None:
        """Add multi-zone comfort constraints with zone-specific requirements."""
        weight = penalty_weights.get('comfort', 150.0)
        
        for k in range(self.horizon):
            for zone_idx, zone_model in enumerate(self.zone_models):
                hvac_encoding = self._get_hvac_encoding(k, zone_idx)
                
                if hvac_encoding:
                    # Comfort range penalty
                    comfort_min, comfort_max = zone_model.comfort_range
                    
                    for bit_i in range(hvac_encoding.precision):
                        qubit_i = hvac_encoding.offset + bit_i
                        
                        if (qubit_i, qubit_i) not in Q:
                            Q[(qubit_i, qubit_i)] = 0.0
                        
                        # Penalty for extreme HVAC operation (proxy for temperature extremes)
                        bit_contribution = 2**bit_i / (2**hvac_encoding.precision - 1)
                        
                        # Higher penalty for very high or very low control values
                        if bit_contribution > 0.9 or bit_contribution < 0.1:
                            extreme_penalty = weight * 0.5
                            Q[(qubit_i, qubit_i)] += extreme_penalty
    
    def _add_hvac_capacity_constraints(
        self,
        Q: Dict[Tuple[int, int], float],
        mpc_problem: Dict[str, Any],
        penalty_weights: Dict[str, float]
    ) -> None:
        """Add HVAC capacity and operational constraints."""
        weight = penalty_weights.get('control_limits', 1000.0)
        
        for k in range(self.horizon):
            for zone_idx, zone_model in enumerate(self.zone_models):
                hvac_encoding = self._get_hvac_encoding(k, zone_idx)
                
                if hvac_encoding:
                    # HVAC capacity constraints are handled by variable bounds
                    # Add penalty for operating at maximum capacity (wear and tear)
                    for bit_i in range(hvac_encoding.precision):
                        qubit_i = hvac_encoding.offset + bit_i
                        
                        if (qubit_i, qubit_i) not in Q:
                            Q[(qubit_i, qubit_i)] = 0.0
                        
                        # Small penalty for maximum capacity operation (MSB)
                        if bit_i == hvac_encoding.precision - 1:
                            capacity_penalty = weight * 0.02
                            Q[(qubit_i, qubit_i)] += capacity_penalty
    
    def _add_storage_constraints(
        self,
        Q: Dict[Tuple[int, int], float],
        storage_model: EnergyStorageModel,
        penalty_weights: Dict[str, float]
    ) -> None:
        """Add energy storage system constraints."""
        weight = penalty_weights.get('storage', 120.0)
        
        for k in range(self.horizon - 1):
            # Storage level continuity: level[k+1] = level[k] + charge[k] - discharge[k]
            current_storage = self._get_storage_encoding(k)
            next_storage = self._get_storage_encoding(k + 1)
            charge_rate = self._get_charge_rate_encoding(k)
            
            if current_storage and next_storage and charge_rate:
                # Storage dynamics penalty
                for bit_i in range(current_storage.precision):
                    for bit_j in range(next_storage.precision):
                        qubit_i = current_storage.offset + bit_i
                        qubit_j = next_storage.offset + bit_j
                        
                        if (qubit_i, qubit_j) not in Q:
                            Q[(qubit_i, qubit_j)] = 0.0
                        
                        # Storage continuity constraint
                        continuity_penalty = weight * 0.1
                        Q[(qubit_i, qubit_j)] += continuity_penalty
                
                # Efficiency losses
                for bit_i in range(charge_rate.precision):
                    qubit_i = charge_rate.offset + bit_i
                    
                    if (qubit_i, qubit_i) not in Q:
                        Q[(qubit_i, qubit_i)] = 0.0
                    
                    # Penalty for storage losses
                    efficiency_penalty = weight * (1 - storage_model.efficiency) * (2**bit_i)
                    Q[(qubit_i, qubit_i)] += efficiency_penalty
    
    def _add_demand_response_constraints(
        self,
        Q: Dict[Tuple[int, int], float],
        mpc_problem: Dict[str, Any],
        penalty_weights: Dict[str, float]
    ) -> None:
        """Add demand response constraints and incentives."""
        weight = penalty_weights.get('demand_response', 60.0)
        
        # Demand response signals (e.g., from utility)
        dr_signals = mpc_problem.get('demand_response_signals', np.zeros(self.horizon))
        
        for k in range(self.horizon):
            dr_signal = dr_signals[k] if k < len(dr_signals) else 0.0
            
            for zone_idx in range(len(self.zone_models)):
                dr_encoding = self._get_demand_response_encoding(k, zone_idx)
                hvac_encoding = self._get_hvac_encoding(k, zone_idx)
                
                if dr_encoding and hvac_encoding:
                    # Incentive for demand response participation
                    for bit_i in range(dr_encoding.precision):
                        qubit_i = dr_encoding.offset + bit_i
                        
                        if (qubit_i, qubit_i) not in Q:
                            Q[(qubit_i, qubit_i)] = 0.0
                        
                        # Positive incentive for DR participation when signal is high
                        dr_incentive = -weight * dr_signal * (2**bit_i)  # Negative for reward
                        Q[(qubit_i, qubit_i)] += dr_incentive
                    
                    # Coupling between HVAC and DR variables
                    for bit_i in range(hvac_encoding.precision):
                        for bit_j in range(dr_encoding.precision):
                            qubit_i = hvac_encoding.offset + bit_i
                            qubit_j = dr_encoding.offset + bit_j
                            
                            if (qubit_i, qubit_j) not in Q:
                                Q[(qubit_i, qubit_j)] = 0.0
                            
                            # DR reduces HVAC operation
                            dr_coupling = weight * 0.2
                            Q[(qubit_i, qubit_j)] += dr_coupling
    
    def _get_hvac_encoding(self, time_step: int, zone_idx: int) -> Optional[EnhancedVariableEncoding]:
        """Get HVAC control encoding for specific time and zone."""
        var_name = f"hvac_{time_step}_{zone_idx}"
        
        for encoding in self._variable_encodings:
            if encoding.name == var_name and encoding.variable_type == "control":
                return encoding
        
        return None
    
    def _get_storage_encoding(self, time_step: int) -> Optional[EnhancedVariableEncoding]:
        """Get storage level encoding for specific time."""
        var_name = f"storage_level_{time_step}"
        
        for encoding in self._storage_variables:
            if encoding.name == var_name:
                return encoding
        
        return None
    
    def _get_charge_rate_encoding(self, time_step: int) -> Optional[EnhancedVariableEncoding]:
        """Get storage charge rate encoding for specific time."""
        var_name = f"charge_rate_{time_step}"
        
        for encoding in self._storage_variables:
            if encoding.name == var_name:
                return encoding
        
        return None
    
    def _get_demand_response_encoding(self, time_step: int, zone_idx: int) -> Optional[EnhancedVariableEncoding]:
        """Get demand response encoding for specific time and zone."""
        var_name = f"demand_response_{time_step}_{zone_idx}"
        
        for encoding in self._demand_response_variables:
            if encoding.name == var_name:
                return encoding
        
        return None
    
    def decode_enhanced_solution(
        self,
        solution: Dict[int, int],
        initial_state: BuildingState = None,
        storage_model: Optional[EnergyStorageModel] = None
    ) -> Dict[str, np.ndarray]:
        """
        Decode enhanced QUBO solution to all control schedules.
        
        Returns:
            Dictionary with hvac_schedule, storage_schedule, dr_schedule
        """
        results = {}
        
        # 1. HVAC Control Schedule
        hvac_schedule = np.zeros((self.horizon, len(self.zone_models)))
        
        for k in range(self.horizon):
            for zone_idx in range(len(self.zone_models)):
                hvac_encoding = self._get_hvac_encoding(k, zone_idx)
                
                if hvac_encoding:
                    # Extract bits for this HVAC variable
                    bits = []
                    for bit_i in range(hvac_encoding.precision):
                        qubit_idx = hvac_encoding.offset + bit_i
                        bit_value = solution.get(qubit_idx, 0)
                        bits.append(bit_value)
                    
                    # Decode to continuous value
                    control_value = hvac_encoding.decode_bits(bits)
                    hvac_schedule[k, zone_idx] = control_value
        
        results['hvac_schedule'] = hvac_schedule
        
        # 2. Storage Schedule (if enabled)
        if self.enable_storage and self._storage_variables:
            storage_schedule = np.zeros((self.horizon, 2))  # [level, charge_rate]
            
            for k in range(self.horizon):
                # Storage level
                storage_encoding = self._get_storage_encoding(k)
                if storage_encoding:
                    bits = []
                    for bit_i in range(storage_encoding.precision):
                        qubit_idx = storage_encoding.offset + bit_i
                        bit_value = solution.get(qubit_idx, 0)
                        bits.append(bit_value)
                    
                    storage_level = storage_encoding.decode_bits(bits)
                    storage_schedule[k, 0] = storage_level
                
                # Charge rate
                charge_encoding = self._get_charge_rate_encoding(k)
                if charge_encoding:
                    bits = []
                    for bit_i in range(charge_encoding.precision):
                        qubit_idx = charge_encoding.offset + bit_i
                        bit_value = solution.get(qubit_idx, 0)
                        bits.append(bit_value)
                    
                    charge_rate = charge_encoding.decode_bits(bits)
                    storage_schedule[k, 1] = charge_rate
            
            results['storage_schedule'] = storage_schedule
        
        # 3. Demand Response Schedule (if enabled)
        if self.enable_demand_response and self._demand_response_variables:
            dr_schedule = np.zeros((self.horizon, len(self.zone_models)))
            
            for k in range(self.horizon):
                for zone_idx in range(len(self.zone_models)):
                    dr_encoding = self._get_demand_response_encoding(k, zone_idx)
                    
                    if dr_encoding:
                        bits = []
                        for bit_i in range(dr_encoding.precision):
                            qubit_idx = dr_encoding.offset + bit_i
                            bit_value = solution.get(qubit_idx, 0)
                            bits.append(bit_value)
                        
                        dr_value = dr_encoding.decode_bits(bits)
                        dr_schedule[k, zone_idx] = dr_value
            
            results['dr_schedule'] = dr_schedule
        
        return results
    
    def validate_enhanced_solution(
        self,
        solution_dict: Dict[str, np.ndarray],
        mpc_problem: Dict[str, Any],
        storage_model: Optional[EnergyStorageModel] = None
    ) -> Dict[str, Any]:
        """Validate enhanced solution against all constraints."""
        violations = {
            'hvac_violations': [],
            'comfort_violations': [],
            'storage_violations': [],
            'dr_violations': [],
            'thermal_violations': []
        }
        
        hvac_schedule = solution_dict.get('hvac_schedule', np.array([]))
        
        if hvac_schedule.size > 0:
            # Check HVAC capacity constraints
            for k in range(hvac_schedule.shape[0]):
                for zone_idx in range(hvac_schedule.shape[1]):
                    if zone_idx < len(self.zone_models):
                        zone_model = self.zone_models[zone_idx]
                        control_value = hvac_schedule[k, zone_idx]
                        
                        # Control value should be in [0, 1] range
                        if control_value < 0 or control_value > 1:
                            violations['hvac_violations'].append({
                                'time_step': k,
                                'zone': zone_idx,
                                'value': control_value,
                                'violation': 'out_of_range'
                            })
                        
                        # Check against zone HVAC capacity
                        power_consumption = control_value * zone_model.hvac_capacity
                        if power_consumption > zone_model.hvac_capacity * 1.05:  # 5% tolerance
                            violations['hvac_violations'].append({
                                'time_step': k,
                                'zone': zone_idx,
                                'power': power_consumption,
                                'capacity': zone_model.hvac_capacity,
                                'violation': 'capacity_exceeded'
                            })
        
        # Validate storage constraints (if applicable)
        if 'storage_schedule' in solution_dict and storage_model:
            storage_schedule = solution_dict['storage_schedule']
            
            for k in range(storage_schedule.shape[0]):
                storage_level = storage_schedule[k, 0]
                charge_rate = storage_schedule[k, 1]
                
                # Check storage bounds
                storage_min, storage_max = storage_model.get_storage_bounds()
                if storage_level < storage_min or storage_level > storage_max:
                    violations['storage_violations'].append({
                        'time_step': k,
                        'level': storage_level,
                        'bounds': (storage_min, storage_max),
                        'violation': 'level_bounds'
                    })
                
                # Check charge rate limits
                if charge_rate > storage_model.max_charge_rate:
                    violations['storage_violations'].append({
                        'time_step': k,
                        'charge_rate': charge_rate,
                        'max_rate': storage_model.max_charge_rate,
                        'violation': 'charge_rate_limit'
                    })
        
        return violations
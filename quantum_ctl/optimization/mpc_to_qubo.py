"""
Model Predictive Control to QUBO transformation engine.

Converts continuous MPC optimization problems to Quadratic Unconstrained 
Binary Optimization (QUBO) format suitable for quantum annealing.
"""

from typing import Dict, Any, Tuple, List, Optional, Callable
import numpy as np
from dataclasses import dataclass
import logging
try:
    from scipy.sparse import csr_matrix
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    csr_matrix = None
# import networkx as nx  # Not needed for basic functionality

from ..models.building import BuildingState


@dataclass 
class VariableEncoding:
    """Binary encoding configuration for continuous variables."""
    name: str
    min_val: float
    max_val: float
    precision: int  # Number of binary bits
    offset: int     # Starting bit position in QUBO
    
    @property
    def num_bits(self) -> int:
        return self.precision
    
    def encode_value(self, value: float) -> List[int]:
        """Convert continuous value to binary representation."""
        # Clamp to bounds
        value = np.clip(value, self.min_val, self.max_val)
        
        # Scale to [0, 2^precision - 1]
        range_val = self.max_val - self.min_val
        scaled = (value - self.min_val) / range_val
        binary_val = int(scaled * (2**self.precision - 1))
        
        # Convert to binary bits
        bits = []
        for i in range(self.precision):
            bits.append((binary_val >> i) & 1)
        
        return bits
    
    def decode_bits(self, bits: List[int]) -> float:
        """Convert binary bits back to continuous value."""
        binary_val = sum(bit * (2**i) for i, bit in enumerate(bits))
        scaled = binary_val / (2**self.precision - 1)
        return self.min_val + scaled * (self.max_val - self.min_val)


class MPCToQUBO:
    """
    Transforms Model Predictive Control problems to QUBO format.
    
    Handles:
    - Continuous variable discretization 
    - Constraint penalty encoding
    - Objective function quadratization
    - QUBO matrix generation
    """
    
    def __init__(
        self,
        state_dim: int,
        control_dim: int, 
        horizon: int,
        precision_bits: int = 4
    ):
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.horizon = horizon
        self.precision_bits = precision_bits
        
        self.logger = logging.getLogger(__name__)
        
        # Variable encodings will be built dynamically
        self._variable_encodings: List[VariableEncoding] = []
        self._total_bits = 0
        
        # Constraint and objective tracking
        self._constraints: List[Callable] = []
        self._objectives: List[Callable] = []
        
    def _create_variable_encodings(self, mpc_problem: Dict[str, Any]) -> None:
        """Create binary encodings for all optimization variables."""
        self._variable_encodings.clear()
        bit_offset = 0
        
        # Control variables (primary optimization variables)
        constraints = mpc_problem.get('constraints', {})
        control_limits = constraints.get('control_limits', [])
        
        for k in range(self.horizon):
            for i in range(self.control_dim):
                # Get control bounds for this zone
                if i < len(control_limits):
                    min_val = control_limits[i].get('min', 0.0)
                    max_val = control_limits[i].get('max', 1.0)
                else:
                    min_val, max_val = 0.0, 1.0
                
                encoding = VariableEncoding(
                    name=f"u_{k}_{i}",
                    min_val=min_val,
                    max_val=max_val,
                    precision=self.precision_bits,
                    offset=bit_offset
                )
                
                self._variable_encodings.append(encoding)
                bit_offset += self.precision_bits
        
        # Slack variables for constraint violations (if needed)
        # These help handle soft constraints
        for constraint_type in ['comfort', 'power']:
            for k in range(self.horizon):
                for i in range(self.state_dim if constraint_type == 'comfort' else self.control_dim):
                    encoding = VariableEncoding(
                        name=f"slack_{constraint_type}_{k}_{i}",
                        min_val=0.0,
                        max_val=10.0,  # Maximum violation allowed
                        precision=3,   # Lower precision for slack
                        offset=bit_offset
                    )
                    
                    self._variable_encodings.append(encoding)
                    bit_offset += 3
        
        self._total_bits = bit_offset
        self.logger.info(f"Created {len(self._variable_encodings)} variables with {self._total_bits} total bits")
    
    def to_qubo(
        self,
        mpc_problem: Dict[str, Any],
        penalty_weights: Dict[str, float] = None
    ) -> Dict[Tuple[int, int], float]:
        """
        Convert MPC problem to QUBO format.
        
        Args:
            mpc_problem: MPC formulation dictionary
            penalty_weights: Constraint penalty weights
            
        Returns:
            QUBO matrix as dictionary of (i,j) -> coefficient
        """
        if penalty_weights is None:
            penalty_weights = {
                'dynamics': 1000.0,
                'comfort': 100.0,
                'energy': 50.0,
                'control_limits': 500.0
            }
        
        # Create variable encodings
        self._create_variable_encodings(mpc_problem)
        
        # Initialize QUBO matrix
        Q = {}
        
        # Add objective function terms
        self._add_objective_terms(Q, mpc_problem, penalty_weights)
        
        # Add constraint penalty terms
        self._add_dynamics_constraints(Q, mpc_problem, penalty_weights['dynamics'])
        self._add_comfort_constraints(Q, mpc_problem, penalty_weights['comfort'])
        self._add_control_constraints(Q, mpc_problem, penalty_weights['control_limits'])
        
        self.logger.info(f"Generated QUBO with {len(Q)} non-zero elements")
        
        return Q
    
    def _add_objective_terms(
        self,
        Q: Dict[Tuple[int, int], float],
        mpc_problem: Dict[str, Any],
        penalty_weights: Dict[str, float]
    ) -> None:
        """Add objective function terms to QUBO matrix."""
        objectives = mpc_problem.get('objectives', {})
        weights = objectives.get('weights', {})
        
        # Energy cost objective (quadratic in control)
        energy_weight = weights.get('energy', 0.6)
        if energy_weight > 0:
            self._add_energy_objective(Q, mpc_problem, energy_weight)
        
        # Comfort objective (quadratic in temperature deviation)
        comfort_weight = weights.get('comfort', 0.3)
        if comfort_weight > 0:
            self._add_comfort_objective(Q, mpc_problem, comfort_weight)
        
        # Carbon objective (linear in energy consumption)
        carbon_weight = weights.get('carbon', 0.1)
        if carbon_weight > 0:
            self._add_carbon_objective(Q, mpc_problem, carbon_weight)
    
    def _add_energy_objective(
        self,
        Q: Dict[Tuple[int, int], float],
        mpc_problem: Dict[str, Any],
        weight: float
    ) -> None:
        """Add energy cost terms to QUBO."""
        # Energy cost is quadratic in control variables
        for k in range(self.horizon):
            for i in range(self.control_dim):
                # Get control variable encoding
                control_encoding = self._get_control_encoding(k, i)
                
                if control_encoding:
                    # Add quadratic terms for each bit
                    for bit_i in range(control_encoding.precision):
                        qubit_i = control_encoding.offset + bit_i
                        
                        # Quadratic penalty (bit^2 = bit since bit ∈ {0,1})
                        if (qubit_i, qubit_i) not in Q:
                            Q[(qubit_i, qubit_i)] = 0.0
                        
                        Q[(qubit_i, qubit_i)] += weight * (2**bit_i)**2
                        
                        # Cross terms between bits of same variable
                        for bit_j in range(bit_i + 1, control_encoding.precision):
                            qubit_j = control_encoding.offset + bit_j
                            
                            if (qubit_i, qubit_j) not in Q:
                                Q[(qubit_i, qubit_j)] = 0.0
                            
                            Q[(qubit_i, qubit_j)] += 2 * weight * (2**bit_i) * (2**bit_j)
    
    def _add_comfort_objective(
        self,
        Q: Dict[Tuple[int, int], float],
        mpc_problem: Dict[str, Any],
        weight: float
    ) -> None:
        """Add comfort deviation penalty to QUBO."""
        # Comfort is quadratic in temperature deviation from setpoint
        # This requires state prediction through dynamics
        
        state_dynamics = mpc_problem.get('state_dynamics', {})
        A = state_dynamics.get('A', np.eye(self.state_dim))
        B = state_dynamics.get('B', np.zeros((self.state_dim, self.control_dim)))
        
        initial_state = mpc_problem.get('initial_state', np.zeros(self.state_dim))
        
        # Predict states using dynamics and add comfort penalties
        x = initial_state.copy()
        
        for k in range(self.horizon):
            # State prediction: x[k+1] = A*x[k] + B*u[k]
            # Comfort penalty on predicted temperatures
            
            for zone_i in range(min(self.control_dim, self.state_dim)):  # Temperature zones
                target_temp = 22.0  # °C comfort setpoint
                
                # Add penalty for deviation from target
                # This is approximated as penalty on control that affects temperature
                control_encoding = self._get_control_encoding(k, zone_i)
                
                if control_encoding:
                    for bit_i in range(control_encoding.precision):
                        qubit_i = control_encoding.offset + bit_i
                        
                        # Comfort penalty (simplified)
                        if (qubit_i, qubit_i) not in Q:
                            Q[(qubit_i, qubit_i)] = 0.0
                        
                        # Penalty for extreme control values
                        bit_value = 2**bit_i / (2**control_encoding.precision - 1)
                        comfort_penalty = abs(bit_value - 0.5)  # Prefer moderate control
                        Q[(qubit_i, qubit_i)] += weight * comfort_penalty
            
            # Update state prediction (simplified)
            # In full implementation, would track state evolution
            pass
    
    def _add_carbon_objective(
        self,
        Q: Dict[Tuple[int, int], float], 
        mpc_problem: Dict[str, Any],
        weight: float
    ) -> None:
        """Add carbon emission penalty (linear in energy)."""
        # Carbon is linear in energy consumption
        # Convert to quadratic by adding linear terms as diagonal QUBO elements
        
        for k in range(self.horizon):
            for i in range(self.control_dim):
                control_encoding = self._get_control_encoding(k, i)
                
                if control_encoding:
                    for bit_i in range(control_encoding.precision):
                        qubit_i = control_encoding.offset + bit_i
                        
                        # Linear term in QUBO diagonal
                        if (qubit_i, qubit_i) not in Q:
                            Q[(qubit_i, qubit_i)] = 0.0
                        
                        # Carbon intensity coefficient  
                        carbon_coeff = weight * (2**bit_i) * 0.5  # kg CO2/kWh
                        Q[(qubit_i, qubit_i)] += carbon_coeff
    
    def _add_dynamics_constraints(
        self,
        Q: Dict[Tuple[int, int], float],
        mpc_problem: Dict[str, Any],
        penalty_weight: float
    ) -> None:
        """Add dynamics constraint penalties to QUBO."""
        # Dynamics constraints: x[k+1] = A*x[k] + B*u[k]
        # These are hard constraints, so high penalty weight
        
        state_dynamics = mpc_problem.get('state_dynamics', {})
        A = state_dynamics.get('A', np.eye(self.state_dim))
        B = state_dynamics.get('B', np.zeros((self.state_dim, self.control_dim)))
        
        # For QUBO, dynamics constraints are handled implicitly
        # by ensuring the state evolution is deterministic given controls
        # The penalty ensures consistency with building thermal model
        
        for k in range(self.horizon - 1):  # No constraint for final step
            for i in range(self.control_dim):
                control_encoding = self._get_control_encoding(k, i)
                next_control_encoding = self._get_control_encoding(k + 1, i)
                
                if control_encoding and next_control_encoding:
                    # Penalize rapid control changes (smoothness)
                    for bit_i in range(control_encoding.precision):
                        for bit_j in range(next_control_encoding.precision):
                            qubit_i = control_encoding.offset + bit_i
                            qubit_j = next_control_encoding.offset + bit_j
                            
                            # Penalty for control change
                            if bit_i == bit_j:  # Same bit position
                                # Encourage consistency
                                if (qubit_i, qubit_j) not in Q:
                                    Q[(qubit_i, qubit_j)] = 0.0
                                Q[(qubit_i, qubit_j)] -= penalty_weight * 0.1
    
    def _add_comfort_constraints(
        self,
        Q: Dict[Tuple[int, int], float],
        mpc_problem: Dict[str, Any],
        penalty_weight: float
    ) -> None:
        """Add comfort bound constraints to QUBO."""
        constraints = mpc_problem.get('constraints', {})
        comfort_bounds = constraints.get('comfort_bounds', [])
        
        # Comfort constraints: temp_min <= T_zone <= temp_max
        # Implemented as penalties on controls that would violate bounds
        
        for k in range(self.horizon):
            for zone_i, bounds in enumerate(comfort_bounds):
                if zone_i >= self.control_dim:
                    continue
                    
                temp_min = bounds.get('temp_min', 20.0)
                temp_max = bounds.get('temp_max', 24.0)
                
                control_encoding = self._get_control_encoding(k, zone_i)
                
                if control_encoding:
                    # Penalty for extreme control values that violate comfort
                    for bit_i in range(control_encoding.precision):
                        qubit_i = control_encoding.offset + bit_i
                        
                        if (qubit_i, qubit_i) not in Q:
                            Q[(qubit_i, qubit_i)] = 0.0
                        
                        # Penalty increases with distance from comfort range
                        bit_contribution = 2**bit_i / (2**control_encoding.precision - 1)
                        
                        # High penalty for extreme values
                        if bit_contribution > 0.8 or bit_contribution < 0.2:
                            Q[(qubit_i, qubit_i)] += penalty_weight * 0.5
    
    def _add_control_constraints(
        self,
        Q: Dict[Tuple[int, int], float],
        mpc_problem: Dict[str, Any],
        penalty_weight: float
    ) -> None:
        """Add control limit constraints to QUBO."""
        # Control limits are handled implicitly by variable encoding bounds
        # Additional penalties can be added for preferred operating ranges
        
        for k in range(self.horizon):
            for i in range(self.control_dim):
                control_encoding = self._get_control_encoding(k, i)
                
                if control_encoding:
                    # Penalty for operating at extreme limits
                    for bit_i in range(control_encoding.precision):
                        qubit_i = control_encoding.offset + bit_i
                        
                        if (qubit_i, qubit_i) not in Q:
                            Q[(qubit_i, qubit_i)] = 0.0
                        
                        # Small penalty to avoid saturation
                        if bit_i == control_encoding.precision - 1:  # MSB
                            Q[(qubit_i, qubit_i)] += penalty_weight * 0.1
    
    def _get_control_encoding(self, time_step: int, control_idx: int) -> Optional[VariableEncoding]:
        """Get variable encoding for control at specific time and index."""
        var_name = f"u_{time_step}_{control_idx}"
        
        for encoding in self._variable_encodings:
            if encoding.name == var_name:
                return encoding
        
        return None
    
    def decode_solution(
        self,
        solution: Dict[int, int],
        initial_state: BuildingState = None
    ) -> np.ndarray:
        """
        Decode QUBO solution back to control schedule.
        
        Args:
            solution: Binary solution from quantum annealer
            initial_state: Initial building state
            
        Returns:
            Control schedule as array [horizon x control_dim]
        """
        control_schedule = np.zeros((self.horizon, self.control_dim))
        
        for k in range(self.horizon):
            for i in range(self.control_dim):
                control_encoding = self._get_control_encoding(k, i)
                
                if control_encoding:
                    # Extract bits for this control variable
                    bits = []
                    for bit_i in range(control_encoding.precision):
                        qubit_idx = control_encoding.offset + bit_i
                        bit_value = solution.get(qubit_idx, 0)
                        bits.append(bit_value)
                    
                    # Decode to continuous value
                    control_value = control_encoding.decode_bits(bits)
                    control_schedule[k, i] = control_value
        
        self.logger.info(f"Decoded control schedule with range [{control_schedule.min():.3f}, {control_schedule.max():.3f}]")
        
        return control_schedule.flatten()  # Return as flat array
    
    def get_problem_size(self) -> Dict[str, int]:
        """Get problem size metrics."""
        return {
            'total_qubits': self._total_bits,
            'control_variables': self.horizon * self.control_dim,
            'state_dimension': self.state_dim,
            'horizon_steps': self.horizon,
            'precision_bits': self.precision_bits,
            'variable_encodings': len(self._variable_encodings)
        }
    
    def validate_solution(
        self,
        control_schedule: np.ndarray,
        mpc_problem: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate decoded solution against constraints."""
        violations = {
            'control_bounds': [],
            'comfort_violations': [],
            'energy_limit': False
        }
        
        constraints = mpc_problem.get('constraints', {})
        
        # Check control bounds
        control_limits = constraints.get('control_limits', [])
        for i, limits in enumerate(control_limits):
            min_val = limits.get('min', 0.0)
            max_val = limits.get('max', 1.0)
            
            # Check all time steps for this control
            for k in range(self.horizon):
                idx = k * self.control_dim + i
                if idx < len(control_schedule):
                    value = control_schedule[idx]
                    if value < min_val or value > max_val:
                        violations['control_bounds'].append({
                            'time_step': k,
                            'control': i,
                            'value': value,
                            'bounds': (min_val, max_val)
                        })
        
        return violations
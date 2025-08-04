"""
Core HVAC controller using quantum annealing for Model Predictive Control.
"""

from typing import Dict, Any, Optional, List
import numpy as np
from dataclasses import dataclass
import asyncio
import logging
from datetime import datetime, timedelta

from ..models.building import Building, BuildingState
from ..optimization.mpc_to_qubo import MPCToQUBO
from ..optimization.quantum_solver import QuantumSolver
from ..utils.validation import validate_state, validate_forecast
from ..utils.error_handling import (
    ErrorHandler, async_error_handler, QuantumControlError, 
    OptimizationError, ErrorCategory, ErrorSeverity
)
from ..utils.monitoring import HealthMonitor, CircuitBreaker, RetryManager


@dataclass
class ControlObjectives:
    """Control optimization objectives with weights."""
    energy_cost: float = 0.6
    comfort: float = 0.3
    carbon: float = 0.1
    
    def __post_init__(self):
        total = self.energy_cost + self.comfort + self.carbon
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"Objective weights must sum to 1.0, got {total}")


@dataclass
class OptimizationConfig:
    """Configuration for quantum optimization."""
    prediction_horizon: int = 24  # hours
    control_interval: int = 15    # minutes
    solver: str = "hybrid_v2"
    num_reads: int = 1000
    annealing_time: int = 20      # microseconds
    chain_strength: Optional[float] = None


class HVACController:
    """
    Quantum-enhanced HVAC controller using D-Wave annealing.
    
    Converts Model Predictive Control problems to QUBO format and solves
    using quantum annealing for real-time building optimization.
    """
    
    def __init__(
        self,
        building: Building,
        config: OptimizationConfig = None,
        objectives: ControlObjectives = None,
    ):
        self.building = building
        self.config = config or OptimizationConfig()
        self.objectives = objectives or ControlObjectives()
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize quantum components
        self.mpc_formulator = MPCToQUBO(
            state_dim=building.get_state_dimension(),
            control_dim=building.get_control_dimension(),
            horizon=self._get_control_steps()
        )
        
        self.quantum_solver = QuantumSolver(
            solver_type=self.config.solver,
            num_reads=self.config.num_reads,
            annealing_time=self.config.annealing_time,
            chain_strength=self.config.chain_strength
        )
        
        # Control history for warm starts
        self._control_history: List[np.ndarray] = []
        self._last_optimization = None
        
        # Error handling and monitoring
        self._error_handler = ErrorHandler(f"{__name__}.{self.__class__.__name__}")
        self._health_monitor = HealthMonitor()
        self._circuit_breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=120)
        self._retry_manager = RetryManager(max_retries=2, base_delay=1.0)
        
        # Start health monitoring
        self._health_monitor.start_monitoring()
        
    def _get_control_steps(self) -> int:
        """Calculate number of control steps in prediction horizon."""
        minutes_per_step = self.config.control_interval
        total_minutes = self.config.prediction_horizon * 60
        return total_minutes // minutes_per_step
    
    def set_objectives(self, objectives: Dict[str, float]) -> None:
        """Update optimization objectives."""
        self.objectives = ControlObjectives(**objectives)
        self.logger.info(f"Updated objectives: {self.objectives}")
    
    @async_error_handler(category=ErrorCategory.OPTIMIZATION, severity=ErrorSeverity.HIGH)
    async def optimize(
        self,
        current_state: BuildingState,
        weather_forecast: np.ndarray,
        energy_prices: np.ndarray,
        **kwargs
    ) -> np.ndarray:
        """
        Optimize HVAC control using quantum annealing.
        
        Args:
            current_state: Current building state (temperatures, etc.)
            weather_forecast: Weather predictions for horizon
            energy_prices: Energy price forecast
            
        Returns:
            Optimal control schedule as numpy array
        """
        start_time = datetime.now()
        
        try:
            # Validate inputs
            validate_state(current_state, self.building)
            validate_forecast(weather_forecast, self._get_control_steps())
            
            # Formulate MPC problem
            self.logger.debug("Formulating MPC problem")
            mpc_problem = await self._formulate_mpc_problem(
                current_state, weather_forecast, energy_prices
            )
            
            # Convert to QUBO
            self.logger.debug("Converting MPC to QUBO")
            Q_matrix = self.mpc_formulator.to_qubo(
                mpc_problem,
                penalty_weights=self._get_penalty_weights()
            )
            
            # Solve with quantum annealing
            self.logger.debug("Solving with quantum annealing")
            solution = await self.quantum_solver.solve(Q_matrix)
            
            # Decode solution to control commands
            control_schedule = self.mpc_formulator.decode_solution(
                solution, current_state
            )
            
            # Store for warm starts
            self._control_history.append(control_schedule)
            if len(self._control_history) > 10:
                self._control_history.pop(0)
            
            self._last_optimization = datetime.now()
            
            optimization_time = (datetime.now() - start_time).total_seconds()
            self.logger.info(
                f"Optimization completed in {optimization_time:.2f}s, "
                f"energy: {solution.data_vectors['energy'][0]:.2f} kWh"
            )
            
            return control_schedule
            
        except Exception as e:
            self.logger.error(f"Optimization failed: {e}")
            # Fallback to last known good control or classical MPC
            return await self._fallback_control(current_state)
    
    async def _formulate_mpc_problem(
        self,
        current_state: BuildingState,
        weather_forecast: np.ndarray,
        energy_prices: np.ndarray
    ) -> Dict[str, Any]:
        """Formulate the MPC optimization problem."""
        
        # Building dynamics matrices
        A, B = self.building.get_dynamics_matrices()
        
        # Disturbance prediction (weather, occupancy)
        disturbances = self.building.predict_disturbances(
            weather_forecast, self._get_control_steps()
        )
        
        # Objective function components
        Q_comfort = self._get_comfort_cost_matrix()
        R_energy = self._get_energy_cost_matrix(energy_prices)
        
        return {
            'state_dynamics': {'A': A, 'B': B},
            'initial_state': current_state.to_vector(),
            'disturbances': disturbances,
            'objectives': {
                'comfort': Q_comfort,
                'energy': R_energy,
                'weights': {
                    'comfort': self.objectives.comfort,
                    'energy': self.objectives.energy_cost,
                    'carbon': self.objectives.carbon
                }
            },
            'constraints': self.building.get_constraints(),
            'horizon': self._get_control_steps()
        }
    
    def _get_penalty_weights(self) -> Dict[str, float]:
        """Get penalty weights for QUBO formulation."""
        return {
            'dynamics': 1000.0,  # Hard constraint
            'comfort': 100.0 * self.objectives.comfort,
            'energy': 50.0 * self.objectives.energy_cost,
            'control_limits': 500.0,  # Hard constraint
        }
    
    def _get_comfort_cost_matrix(self) -> np.ndarray:
        """Generate quadratic cost matrix for comfort deviations."""
        n_zones = self.building.zones
        n_steps = self._get_control_steps()
        
        # Penalize deviations from comfort setpoints
        Q = np.zeros((n_zones * n_steps, n_zones * n_steps))
        
        for k in range(n_steps):
            for i in range(n_zones):
                idx = k * n_zones + i
                Q[idx, idx] = 1.0  # Quadratic penalty on temperature deviation
        
        return Q
    
    def _get_energy_cost_matrix(self, energy_prices: np.ndarray) -> np.ndarray:
        """Generate cost matrix for energy consumption."""
        n_controls = self.building.get_control_dimension()
        n_steps = self._get_control_steps()
        
        R = np.zeros((n_controls * n_steps, n_controls * n_steps))
        
        for k in range(min(n_steps, len(energy_prices))):
            price = energy_prices[k] if k < len(energy_prices) else energy_prices[-1]
            for i in range(n_controls):
                idx = k * n_controls + i
                R[idx, idx] = price
        
        return R
    
    async def _perform_optimization_with_retry(
        self,
        current_state: BuildingState,
        weather_forecast: np.ndarray,
        energy_prices: np.ndarray,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform optimization with retry logic."""
        return await self._retry_manager.retry_async(
            self._core_optimization,
            current_state, weather_forecast, energy_prices, context
        )
    
    async def _core_optimization(
        self,
        current_state: BuildingState,
        weather_forecast: np.ndarray,
        energy_prices: np.ndarray,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Core optimization logic."""
        # Formulate MPC problem
        self.logger.debug("Formulating MPC problem")
        mpc_problem = await self._formulate_mpc_problem(
            current_state, weather_forecast, energy_prices
        )
        
        # Convert to QUBO
        self.logger.debug("Converting MPC to QUBO")
        Q_matrix = self.mpc_formulator.to_qubo(
            mpc_problem,
            penalty_weights=self._get_penalty_weights()
        )
        
        # Solve with quantum annealing
        self.logger.debug("Solving with quantum annealing")
        solution = await self.quantum_solver.solve(Q_matrix)
        
        # Decode solution to control commands
        control_schedule = self.mpc_formulator.decode_solution(
            solution, current_state
        )
        
        return {
            'control_schedule': control_schedule,
            'solution': solution,
            'mpc_problem': mpc_problem
        }

    async def _fallback_control(self, current_state: BuildingState) -> np.ndarray:
        """Fallback control when quantum optimization fails."""
        self.logger.warning("Using fallback control strategy")
        
        if self._control_history:
            # Use shifted version of last control
            last_control = self._control_history[-1]
            fallback = np.roll(last_control, -self.building.get_control_dimension())
            return fallback
        
        # Emergency: maintain current setpoints
        n_controls = self.building.get_control_dimension()
        n_steps = self._get_control_steps()
        return np.tile(current_state.control_setpoints, n_steps)
    
    def apply_schedule(self, control_schedule: np.ndarray) -> None:
        """Apply the first step of the control schedule to the building."""
        if len(control_schedule) == 0:
            self.logger.error("Empty control schedule provided")
            return
        
        # Extract first control step
        n_controls = self.building.get_control_dimension()
        current_control = control_schedule[:n_controls]
        
        # Apply to building (this would interface with actual BMS)
        self.building.apply_control(current_control)
        
        self.logger.info(f"Applied control: {current_control}")
    
    async def run_control_loop(
        self,
        data_source,
        update_interval: int = None
    ) -> None:
        """Run continuous control loop with quantum optimization."""
        interval = update_interval or self.config.control_interval * 60  # seconds
        
        self.logger.info(f"Starting control loop with {interval}s intervals")
        
        while True:
            try:
                # Get current data
                current_state = await data_source.get_current_state()
                weather_forecast = await data_source.get_weather_forecast()
                energy_prices = await data_source.get_energy_prices()
                
                # Optimize and apply
                control_schedule = await self.optimize(
                    current_state, weather_forecast, energy_prices
                )
                self.apply_schedule(control_schedule)
                
                await asyncio.sleep(interval)
                
            except Exception as e:
                self.logger.error(f"Control loop error: {e}")
                await asyncio.sleep(interval)
    
    def get_status(self) -> Dict[str, Any]:
        """Get controller status and metrics."""
        return {
            'building_id': self.building.building_id,
            'last_optimization': self._last_optimization,
            'objectives': self.objectives,
            'config': self.config,
            'history_length': len(self._control_history),
            'quantum_solver_status': self.quantum_solver.get_status()
        }
"""
Validation utilities for quantum HVAC control.
"""

import numpy as np
from typing import Any, Dict, List
import logging

from ..models.building import Building, BuildingState


logger = logging.getLogger(__name__)


def validate_state(state: BuildingState, building: Building) -> None:
    """
    Validate building state against building configuration.
    
    Args:
        state: Current building state
        building: Building configuration
        
    Raises:
        ValueError: If state is invalid
    """
    if len(state.zone_temperatures) != building.n_zones:
        raise ValueError(
            f"Zone temperature count {len(state.zone_temperatures)} != {building.n_zones}"
        )
    
    if len(state.occupancy) != building.n_zones:
        raise ValueError(
            f"Occupancy count {len(state.occupancy)} != {building.n_zones}"
        )
    
    if len(state.hvac_power) != building.n_zones:
        raise ValueError(
            f"HVAC power count {len(state.hvac_power)} != {building.n_zones}"
        )
    
    # Check temperature ranges
    for i, temp in enumerate(state.zone_temperatures):
        if not (-40 <= temp <= 60):  # Reasonable temperature range
            logger.warning(f"Zone {i} temperature {temp}°C outside expected range")
    
    # Check occupancy ranges
    for i, occ in enumerate(state.occupancy):
        if not (0 <= occ <= 1):
            raise ValueError(f"Zone {i} occupancy {occ} not in [0,1] range")
    
    # Check HVAC power ranges
    for i, power in enumerate(state.hvac_power):
        zone = building.zones[i] if i < len(building.zones) else None
        if zone:
            max_power = max(zone.max_heating_power, zone.max_cooling_power)
            if abs(power) > max_power * 1.1:  # 10% tolerance
                logger.warning(f"Zone {i} power {power} kW exceeds limits")


def validate_forecast(forecast: np.ndarray, expected_steps: int) -> None:
    """
    Validate weather forecast data.
    
    Args:
        forecast: Weather forecast array
        expected_steps: Expected number of time steps
        
    Raises:
        ValueError: If forecast is invalid
    """
    if len(forecast) < expected_steps:
        logger.warning(
            f"Forecast length {len(forecast)} < expected {expected_steps}, "
            "will extrapolate"
        )
    
    if forecast.ndim == 1:
        # Temperature only
        temp_data = forecast
    elif forecast.ndim == 2:
        # Multiple weather variables
        temp_data = forecast[:, 0]
    else:
        raise ValueError(f"Forecast must be 1D or 2D array, got {forecast.ndim}D")
    
    # Check temperature ranges
    if np.any(temp_data < -50) or np.any(temp_data > 60):
        logger.warning("Forecast temperatures outside expected range [-50, 60]°C")
    
    # Check for NaN values
    if np.any(np.isnan(forecast)):
        raise ValueError("Forecast contains NaN values")


def validate_control_schedule(
    schedule: np.ndarray,
    building: Building,
    horizon: int
) -> Dict[str, Any]:
    """
    Validate control schedule against building constraints.
    
    Args:
        schedule: Control schedule array
        building: Building configuration
        horizon: Time horizon steps
        
    Returns:
        Validation results with violations
    """
    violations = {
        'control_bounds': [],
        'power_limits': [],
        'rate_limits': []
    }
    
    n_controls = building.get_control_dimension()
    expected_length = horizon * n_controls
    
    if len(schedule) != expected_length:
        violations['control_bounds'].append({
            'error': f"Schedule length {len(schedule)} != expected {expected_length}"
        })
        return violations
    
    # Reshape to [horizon, n_controls]
    schedule_2d = schedule.reshape((horizon, n_controls))
    
    # Check control bounds
    for k in range(horizon):
        for i in range(n_controls):
            control_val = schedule_2d[k, i]
            
            # Basic bounds check [0, 1]
            if not (0 <= control_val <= 1):
                violations['control_bounds'].append({
                    'time_step': k,
                    'control': i,
                    'value': control_val,
                    'expected_range': [0, 1]
                })
            
            # Zone-specific power limits
            if i < len(building.zones):
                zone = building.zones[i]
                
                # Approximate power from control signal
                heating_power = control_val * zone.max_heating_power
                if heating_power > zone.max_heating_power * 1.05:  # 5% tolerance
                    violations['power_limits'].append({
                        'time_step': k,
                        'zone': i,
                        'power': heating_power,
                        'limit': zone.max_heating_power
                    })
    
    # Check rate of change limits
    for k in range(1, horizon):
        for i in range(n_controls):
            prev_val = schedule_2d[k-1, i]
            curr_val = schedule_2d[k, i]
            rate_change = abs(curr_val - prev_val)
            
            # Maximum 50% change per time step
            if rate_change > 0.5:
                violations['rate_limits'].append({
                    'time_step': k,
                    'control': i,
                    'rate_change': rate_change,
                    'limit': 0.5
                })
    
    return violations


def validate_qubo_matrix(Q: Dict[tuple, float]) -> Dict[str, Any]:
    """
    Validate QUBO matrix structure and properties.
    
    Args:
        Q: QUBO matrix as dictionary
        
    Returns:
        Validation results
    """
    if not Q:
        return {'status': 'error', 'message': 'Empty QUBO matrix'}
    
    # Get all variables
    variables = set()
    for (i, j) in Q.keys():
        variables.add(i)
        variables.add(j)
    
    variables = sorted(variables)
    n_vars = len(variables)
    
    # Check matrix properties
    n_linear = sum(1 for (i, j) in Q.keys() if i == j)
    n_quadratic = sum(1 for (i, j) in Q.keys() if i != j)
    
    # Check for symmetric quadratic terms
    symmetric_violations = []
    for (i, j) in Q.keys():
        if i != j:
            if (j, i) in Q:
                if abs(Q[(i, j)] - Q[(j, i)]) > 1e-10:
                    symmetric_violations.append(((i, j), Q[(i, j)], Q[(j, i)]))
    
    # Check coefficient magnitudes
    coeffs = list(Q.values())
    max_coeff = max(abs(c) for c in coeffs)
    min_coeff = min(abs(c) for c in coeffs if c != 0)
    
    return {
        'status': 'valid',
        'n_variables': n_vars,
        'n_linear_terms': n_linear,
        'n_quadratic_terms': n_quadratic,
        'max_coefficient': max_coeff,
        'min_coefficient': min_coeff,
        'coefficient_range': max_coeff / min_coeff if min_coeff > 0 else float('inf'),
        'symmetric_violations': len(symmetric_violations),
        'sparsity': len(Q) / (n_vars * (n_vars + 1) / 2) if n_vars > 0 else 0.0
    }


def validate_quantum_solution(
    solution: Dict[int, int],
    Q: Dict[tuple, float]
) -> Dict[str, Any]:
    """
    Validate quantum annealing solution.
    
    Args:
        solution: Binary variable assignments
        Q: Original QUBO matrix
        
    Returns:
        Validation results
    """
    # Check all variables are assigned
    qubo_vars = set()
    for (i, j) in Q.keys():
        qubo_vars.add(i)
        qubo_vars.add(j)
    
    missing_vars = qubo_vars - set(solution.keys())
    extra_vars = set(solution.keys()) - qubo_vars
    
    # Check binary values
    non_binary = [(var, val) for var, val in solution.items() 
                  if val not in {0, 1}]
    
    # Calculate energy
    energy = 0.0
    for (i, j), coeff in Q.items():
        if i == j:
            energy += coeff * solution.get(i, 0)
        else:
            energy += coeff * solution.get(i, 0) * solution.get(j, 0)
    
    return {
        'status': 'valid' if not missing_vars and not non_binary else 'invalid',
        'energy': energy,
        'missing_variables': list(missing_vars),
        'extra_variables': list(extra_vars),
        'non_binary_values': non_binary,
        'variable_count': len(solution)
    }
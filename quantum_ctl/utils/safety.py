"""
Safety mechanisms and emergency controls for HVAC systems.
"""

import numpy as np
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from ..models.building import Building, BuildingState


class SafetyLevel(Enum):
    """Safety alert levels."""
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class SafetyLimits:
    """Safety limits for building operation."""
    min_zone_temp: float = 15.0    # °C - Minimum safe temperature
    max_zone_temp: float = 35.0    # °C - Maximum safe temperature
    max_humidity: float = 95.0     # % - Maximum humidity
    min_humidity: float = 10.0     # % - Minimum humidity
    max_power_per_zone: float = 25.0  # kW - Maximum power per zone
    max_temp_change_rate: float = 5.0  # °C/hour - Maximum temperature change rate


class SafetyMonitor:
    """Safety monitoring and emergency control system."""
    
    def __init__(self, building: Building, limits: SafetyLimits = None):
        self.building = building
        self.limits = limits or SafetyLimits()
        self.logger = logging.getLogger(__name__)
        
        # Safety state tracking
        self.current_safety_level = SafetyLevel.NORMAL
        self.safety_violations: List[str] = []
        self.emergency_control_active = False
        
        # Historical data for rate calculations
        self._temp_history: List[Tuple[float, np.ndarray]] = []  # (timestamp, temperatures)
        
    def check_safety(self, state: BuildingState) -> SafetyLevel:
        """Check current safety status and return level."""
        violations = []
        
        # Temperature safety checks
        temp_violations = self._check_temperature_safety(state)
        violations.extend(temp_violations)
        
        # Humidity safety checks
        humidity_violations = self._check_humidity_safety(state)
        violations.extend(humidity_violations)
        
        # Power safety checks
        power_violations = self._check_power_safety(state)
        violations.extend(power_violations)
        
        # Rate of change checks
        rate_violations = self._check_rate_of_change(state)
        violations.extend(rate_violations)
        
        # Update safety state
        self.safety_violations = violations
        self.current_safety_level = self._determine_safety_level(violations)
        
        # Log safety issues
        if violations:
            self.logger.warning(f"Safety violations detected: {violations}")
        
        return self.current_safety_level
    
    def _check_temperature_safety(self, state: BuildingState) -> List[str]:
        """Check temperature safety limits."""
        violations = []
        
        for i, temp in enumerate(state.zone_temperatures):
            if temp < self.limits.min_zone_temp:
                violations.append(f"Zone {i} temperature too low: {temp:.1f}°C")
            elif temp > self.limits.max_zone_temp:
                violations.append(f"Zone {i} temperature too high: {temp:.1f}°C")
        
        return violations
    
    def _check_humidity_safety(self, state: BuildingState) -> List[str]:
        """Check humidity safety limits."""
        violations = []
        
        if state.humidity < self.limits.min_humidity:
            violations.append(f"Humidity too low: {state.humidity:.1f}%")
        elif state.humidity > self.limits.max_humidity:
            violations.append(f"Humidity too high: {state.humidity:.1f}%")
        
        return violations
    
    def _check_power_safety(self, state: BuildingState) -> List[str]:
        """Check power consumption safety limits."""
        violations = []
        
        for i, power in enumerate(state.hvac_power):
            if abs(power) > self.limits.max_power_per_zone:
                violations.append(f"Zone {i} power too high: {power:.1f} kW")
        
        return violations
    
    def _check_rate_of_change(self, state: BuildingState) -> List[str]:
        """Check temperature rate of change safety."""
        violations = []
        
        # Add current state to history
        self._temp_history.append((state.timestamp, state.zone_temperatures.copy()))
        
        # Keep only recent history (last hour)
        cutoff_time = state.timestamp - 3600  # 1 hour
        self._temp_history = [(t, temps) for t, temps in self._temp_history if t > cutoff_time]
        
        # Check rate of change if we have enough history
        if len(self._temp_history) >= 2:
            prev_time, prev_temps = self._temp_history[-2]
            curr_time, curr_temps = self._temp_history[-1]
            
            time_diff = (curr_time - prev_time) / 3600.0  # Convert to hours
            
            if time_diff > 0:
                temp_rates = (curr_temps - prev_temps) / time_diff
                
                for i, rate in enumerate(temp_rates):
                    if abs(rate) > self.limits.max_temp_change_rate:
                        violations.append(f"Zone {i} temperature changing too fast: {rate:.1f}°C/h")
        
        return violations
    
    def _determine_safety_level(self, violations: List[str]) -> SafetyLevel:
        """Determine overall safety level from violations."""
        if not violations:
            return SafetyLevel.NORMAL
        
        # Check for critical keywords
        critical_keywords = ['too high', 'too low', 'emergency']
        emergency_keywords = ['fire', 'flood', 'gas', 'electrical']
        
        has_critical = any(any(keyword in v.lower() for keyword in critical_keywords) for v in violations)
        has_emergency = any(any(keyword in v.lower() for keyword in emergency_keywords) for v in violations)
        
        if has_emergency:
            return SafetyLevel.EMERGENCY
        elif has_critical or len(violations) > 3:
            return SafetyLevel.CRITICAL
        else:
            return SafetyLevel.WARNING
    
    def get_emergency_control(self, state: BuildingState) -> np.ndarray:
        """Generate emergency control commands for safety."""
        self.logger.warning("Generating emergency control commands")
        self.emergency_control_active = True
        
        n_controls = self.building.get_control_dimension()
        emergency_control = np.zeros(n_controls)
        
        # Emergency control strategy based on violations
        for i, temp in enumerate(state.zone_temperatures):
            if i >= n_controls:
                break
                
            if temp < self.limits.min_zone_temp:
                # Emergency heating
                emergency_control[i] = 1.0
                self.logger.warning(f"Emergency heating for zone {i}")
            elif temp > self.limits.max_zone_temp:
                # Emergency cooling  
                emergency_control[i] = 0.0
                self.logger.warning(f"Emergency cooling for zone {i}")
            else:
                # Maintain current if safe
                if i < len(state.control_setpoints):
                    emergency_control[i] = np.clip(state.control_setpoints[i], 0.2, 0.8)
                else:
                    emergency_control[i] = 0.5
        
        return emergency_control
    
    def validate_control_schedule(self, control_schedule: np.ndarray, state: BuildingState) -> Tuple[np.ndarray, List[str]]:
        """Validate and potentially modify control schedule for safety."""
        violations = []
        safe_schedule = control_schedule.copy()
        
        # Predict potential issues
        predicted_violations = self._predict_safety_violations(control_schedule, state)
        
        if predicted_violations:
            violations.extend(predicted_violations)
            safe_schedule = self._make_schedule_safe(control_schedule, state)
            self.logger.warning(f"Modified control schedule for safety: {predicted_violations}")
        
        return safe_schedule, violations
    
    def _predict_safety_violations(self, control_schedule: np.ndarray, state: BuildingState) -> List[str]:
        """Predict potential safety violations from control schedule."""
        violations = []
        
        # Simple thermal model prediction
        # This is a simplified check - real implementation would use building thermal model
        
        n_controls = self.building.get_control_dimension()
        if len(control_schedule) >= n_controls:
            first_controls = control_schedule[:n_controls]
            
            for i, control in enumerate(first_controls):
                if i < len(state.zone_temperatures):
                    current_temp = state.zone_temperatures[i]
                    
                    # Predict temperature change based on control
                    if control > 0.8 and current_temp > 30.0:
                        violations.append(f"High control {control:.2f} may overheat zone {i}")
                    elif control < 0.2 and current_temp < 18.0:
                        violations.append(f"Low control {control:.2f} may undercool zone {i}")
        
        return violations
    
    def _make_schedule_safe(self, control_schedule: np.ndarray, state: BuildingState) -> np.ndarray:
        """Modify control schedule to ensure safety."""
        safe_schedule = control_schedule.copy()
        n_controls = self.building.get_control_dimension()
        
        # Apply safety constraints to first control step
        if len(safe_schedule) >= n_controls:
            for i in range(n_controls):
                current_control = safe_schedule[i]
                
                if i < len(state.zone_temperatures):
                    current_temp = state.zone_temperatures[i]
                    
                    # Safety limits based on current temperature
                    if current_temp > 30.0:
                        # Limit heating when already hot
                        safe_schedule[i] = min(current_control, 0.6)
                    elif current_temp < 18.0:
                        # Limit cooling when already cold
                        safe_schedule[i] = max(current_control, 0.4)
                    
                    # General bounds
                    safe_schedule[i] = np.clip(safe_schedule[i], 0.1, 0.9)
        
        return safe_schedule
    
    def get_safety_status(self) -> Dict[str, Any]:
        """Get current safety status."""
        return {
            'safety_level': self.current_safety_level.value,
            'violations': self.safety_violations,
            'emergency_control_active': self.emergency_control_active,
            'limits': {
                'min_zone_temp': self.limits.min_zone_temp,
                'max_zone_temp': self.limits.max_zone_temp,
                'max_humidity': self.limits.max_humidity,
                'max_power_per_zone': self.limits.max_power_per_zone
            }
        }
    
    def reset_safety_state(self):
        """Reset safety monitoring state."""
        self.current_safety_level = SafetyLevel.NORMAL
        self.safety_violations = []
        self.emergency_control_active = False
        self._temp_history = []
        self.logger.info("Safety monitoring state reset")


class SafetyInterlock:
    """Hardware safety interlock simulation."""
    
    def __init__(self):
        self.interlocks_active = False
        self.interlock_reasons: List[str] = []
        self.logger = logging.getLogger(__name__)
    
    def check_interlocks(self, state: BuildingState) -> bool:
        """Check if safety interlocks should be activated."""
        reasons = []
        
        # Temperature interlocks
        if np.any(state.zone_temperatures > 40.0):
            reasons.append("Zone temperature >40°C")
        if np.any(state.zone_temperatures < 10.0):
            reasons.append("Zone temperature <10°C")
        
        # Power interlocks
        if np.any(np.abs(state.hvac_power) > 30.0):
            reasons.append("HVAC power >30kW")
        
        # Humidity interlocks
        if state.humidity > 98.0:
            reasons.append("Humidity >98%")
        
        self.interlock_reasons = reasons
        self.interlocks_active = len(reasons) > 0
        
        if self.interlocks_active:
            self.logger.critical(f"Safety interlocks activated: {reasons}")
        
        return self.interlocks_active
    
    def get_interlock_status(self) -> Dict[str, Any]:
        """Get interlock status."""
        return {
            'active': self.interlocks_active,
            'reasons': self.interlock_reasons
        }
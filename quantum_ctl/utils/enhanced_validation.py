"""
Enhanced validation system for quantum HVAC control operations.
Provides comprehensive input validation, constraint checking, and safety validation.
"""

import numpy as np
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationResult:
    """Result of a validation check."""
    is_valid: bool
    severity: ValidationSeverity
    message: str
    field: Optional[str] = None
    suggestion: Optional[str] = None


class HVACValidator:
    """Comprehensive validator for HVAC control parameters."""
    
    def __init__(self):
        self.temperature_limits = {"min": -50.0, "max": 60.0}  # Celsius
        self.power_limits = {"min": 0.0, "max": 10000.0}  # kW
        self.efficiency_limits = {"min": 0.0, "max": 1.0}  # 0-100%
        self.occupancy_limits = {"min": 0.0, "max": 1.0}  # 0-100%
        
    def validate_temperature(
        self, 
        temp: float, 
        context: str = "general",
        comfort_range: Optional[tuple] = None
    ) -> ValidationResult:
        """Validate temperature values with context-aware limits."""
        if not isinstance(temp, (int, float)):
            return ValidationResult(
                False, 
                ValidationSeverity.ERROR,
                f"Temperature must be numeric, got {type(temp)}",
                "temperature"
            )
        
        if not (self.temperature_limits["min"] <= temp <= self.temperature_limits["max"]):
            return ValidationResult(
                False,
                ValidationSeverity.CRITICAL,
                f"Temperature {temp}°C outside safe range "
                f"[{self.temperature_limits['min']}, {self.temperature_limits['max']}]",
                "temperature",
                f"Use temperature between {self.temperature_limits['min']} and {self.temperature_limits['max']}°C"
            )
        
        # Context-specific validation
        if comfort_range and not (comfort_range[0] <= temp <= comfort_range[1]):
            return ValidationResult(
                True,  # Valid but not optimal
                ValidationSeverity.WARNING,
                f"Temperature {temp}°C outside comfort range {comfort_range}",
                "temperature",
                f"Consider adjusting to {comfort_range[0]}-{comfort_range[1]}°C for comfort"
            )
        
        return ValidationResult(True, ValidationSeverity.INFO, "Temperature validation passed")
    
    def validate_control_input(self, control: Union[float, List[float], np.ndarray]) -> ValidationResult:
        """Validate control input parameters."""
        try:
            if isinstance(control, (list, np.ndarray)):
                control_array = np.array(control)
                if control_array.size == 0:
                    return ValidationResult(
                        False,
                        ValidationSeverity.ERROR,
                        "Control input cannot be empty",
                        "control"
                    )
                
                # Check for invalid values
                if np.any(np.isnan(control_array)) or np.any(np.isinf(control_array)):
                    return ValidationResult(
                        False,
                        ValidationSeverity.CRITICAL,
                        "Control input contains NaN or infinite values",
                        "control",
                        "Ensure all control values are finite numbers"
                    )
                
                # Validate power limits for each control signal
                power_violations = np.where(
                    (control_array < self.power_limits["min"]) | 
                    (control_array > self.power_limits["max"])
                )[0]
                
                if len(power_violations) > 0:
                    return ValidationResult(
                        False,
                        ValidationSeverity.ERROR,
                        f"Control signals at indices {power_violations} exceed power limits "
                        f"[{self.power_limits['min']}, {self.power_limits['max']}]",
                        "control"
                    )
                
            else:
                # Single value validation
                if not isinstance(control, (int, float)):
                    return ValidationResult(
                        False,
                        ValidationSeverity.ERROR,
                        f"Control input must be numeric, got {type(control)}",
                        "control"
                    )
                
                if np.isnan(control) or np.isinf(control):
                    return ValidationResult(
                        False,
                        ValidationSeverity.CRITICAL,
                        "Control input is NaN or infinite",
                        "control"
                    )
            
            return ValidationResult(True, ValidationSeverity.INFO, "Control input validation passed")
            
        except Exception as e:
            return ValidationResult(
                False,
                ValidationSeverity.ERROR,
                f"Control input validation error: {str(e)}",
                "control"
            )
    
    def validate_building_config(self, config: Dict[str, Any]) -> List[ValidationResult]:
        """Validate building configuration parameters."""
        results = []
        
        # Required fields check
        required_fields = ["zones", "thermal_mass", "max_power"]
        for field in required_fields:
            if field not in config:
                results.append(ValidationResult(
                    False,
                    ValidationSeverity.ERROR,
                    f"Missing required field: {field}",
                    field,
                    f"Add {field} to building configuration"
                ))
        
        # Zones validation
        if "zones" in config:
            zones = config["zones"]
            if not isinstance(zones, int) or zones <= 0:
                results.append(ValidationResult(
                    False,
                    ValidationSeverity.ERROR,
                    f"Zones must be positive integer, got {zones}",
                    "zones"
                ))
            elif zones > 1000:
                results.append(ValidationResult(
                    True,
                    ValidationSeverity.WARNING,
                    f"Large number of zones ({zones}) may impact performance",
                    "zones",
                    "Consider zone aggregation for performance"
                ))
        
        # Thermal mass validation
        if "thermal_mass" in config:
            thermal_mass = config["thermal_mass"]
            if not isinstance(thermal_mass, (int, float)) or thermal_mass <= 0:
                results.append(ValidationResult(
                    False,
                    ValidationSeverity.ERROR,
                    f"Thermal mass must be positive number, got {thermal_mass}",
                    "thermal_mass"
                ))
        
        # Power limit validation
        if "max_power" in config:
            max_power = config["max_power"]
            if not isinstance(max_power, (int, float)) or max_power <= 0:
                results.append(ValidationResult(
                    False,
                    ValidationSeverity.ERROR,
                    f"Max power must be positive number, got {max_power}",
                    "max_power"
                ))
        
        return results if results else [ValidationResult(
            True, ValidationSeverity.INFO, "Building configuration validation passed"
        )]
    
    def validate_optimization_params(self, params: Dict[str, Any]) -> List[ValidationResult]:
        """Validate optimization parameters."""
        results = []
        
        # Horizon validation
        if "horizon" in params:
            horizon = params["horizon"]
            if not isinstance(horizon, int) or horizon <= 0:
                results.append(ValidationResult(
                    False,
                    ValidationSeverity.ERROR,
                    f"Optimization horizon must be positive integer, got {horizon}",
                    "horizon"
                ))
            elif horizon > 168:  # 1 week
                results.append(ValidationResult(
                    True,
                    ValidationSeverity.WARNING,
                    f"Long optimization horizon ({horizon}h) may be computationally expensive",
                    "horizon",
                    "Consider reducing horizon or using decomposition"
                ))
        
        # Solver validation
        if "solver" in params:
            solver = params["solver"]
            valid_solvers = ["classical", "quantum", "hybrid", "classical_fallback"]
            if solver not in valid_solvers:
                results.append(ValidationResult(
                    False,
                    ValidationSeverity.ERROR,
                    f"Invalid solver '{solver}', must be one of {valid_solvers}",
                    "solver"
                ))
        
        # Objectives validation
        if "objectives" in params:
            objectives = params["objectives"]
            if isinstance(objectives, dict):
                total_weight = sum(objectives.values())
                if abs(total_weight - 1.0) > 1e-6:
                    results.append(ValidationResult(
                        False,
                        ValidationSeverity.WARNING,
                        f"Objective weights sum to {total_weight}, should sum to 1.0",
                        "objectives",
                        "Normalize objective weights to sum to 1.0"
                    ))
                
                for obj_name, weight in objectives.items():
                    if not isinstance(weight, (int, float)) or weight < 0:
                        results.append(ValidationResult(
                            False,
                            ValidationSeverity.ERROR,
                            f"Objective weight for '{obj_name}' must be non-negative, got {weight}",
                            "objectives"
                        ))
        
        return results if results else [ValidationResult(
            True, ValidationSeverity.INFO, "Optimization parameters validation passed"
        )]


class SafetyValidator:
    """Safety-critical validation for HVAC operations."""
    
    def __init__(self):
        self.safety_limits = {
            "min_temp": 10.0,   # Minimum safe temperature
            "max_temp": 35.0,   # Maximum safe temperature
            "max_power_rate": 0.1,  # Maximum power change rate (10% per minute)
            "min_air_flow": 0.1     # Minimum air flow rate
        }
    
    def validate_safety_critical(
        self, 
        current_state: Dict[str, Any],
        proposed_control: Dict[str, Any]
    ) -> List[ValidationResult]:
        """Validate safety-critical parameters."""
        results = []
        
        # Temperature safety
        if "temperature" in current_state:
            temp = current_state["temperature"]
            if isinstance(temp, (list, np.ndarray)):
                temp_array = np.array(temp)
                unsafe_zones = np.where(
                    (temp_array < self.safety_limits["min_temp"]) |
                    (temp_array > self.safety_limits["max_temp"])
                )[0]
                
                if len(unsafe_zones) > 0:
                    results.append(ValidationResult(
                        False,
                        ValidationSeverity.CRITICAL,
                        f"Unsafe temperatures in zones {unsafe_zones}",
                        "temperature",
                        "Implement emergency temperature control"
                    ))
            else:
                if not (self.safety_limits["min_temp"] <= temp <= self.safety_limits["max_temp"]):
                    results.append(ValidationResult(
                        False,
                        ValidationSeverity.CRITICAL,
                        f"Unsafe temperature {temp}°C",
                        "temperature"
                    ))
        
        # Power change rate safety
        if "power" in current_state and "power" in proposed_control:
            current_power = np.array(current_state["power"])
            proposed_power = np.array(proposed_control["power"])
            
            if current_power.shape == proposed_power.shape:
                power_change_rate = np.abs(proposed_power - current_power) / np.maximum(current_power, 1.0)
                unsafe_changes = np.where(power_change_rate > self.safety_limits["max_power_rate"])[0]
                
                if len(unsafe_changes) > 0:
                    results.append(ValidationResult(
                        False,
                        ValidationSeverity.WARNING,
                        f"Rapid power changes in zones {unsafe_changes}",
                        "power",
                        "Implement gradual power ramping"
                    ))
        
        return results


class ValidationEngine:
    """Main validation engine that orchestrates all validation checks."""
    
    def __init__(self):
        self.hvac_validator = HVACValidator()
        self.safety_validator = SafetyValidator()
        self.custom_validators: List[Callable] = []
    
    def add_custom_validator(self, validator: Callable):
        """Add custom validation function."""
        self.custom_validators.append(validator)
    
    def validate_all(
        self,
        data: Dict[str, Any],
        validation_level: str = "standard"
    ) -> List[ValidationResult]:
        """Run comprehensive validation on data."""
        results = []
        
        try:
            # Basic HVAC validation
            if "temperature" in data:
                results.append(self.hvac_validator.validate_temperature(
                    data["temperature"],
                    data.get("context", "general"),
                    data.get("comfort_range")
                ))
            
            if "control" in data:
                results.append(self.hvac_validator.validate_control_input(data["control"]))
            
            if "building_config" in data:
                results.extend(self.hvac_validator.validate_building_config(data["building_config"]))
            
            if "optimization_params" in data:
                results.extend(self.hvac_validator.validate_optimization_params(data["optimization_params"]))
            
            # Safety validation (critical level)
            if validation_level in ["critical", "safety"]:
                current_state = data.get("current_state", {})
                proposed_control = data.get("proposed_control", {})
                if current_state and proposed_control:
                    results.extend(self.safety_validator.validate_safety_critical(
                        current_state, proposed_control
                    ))
            
            # Custom validations
            for custom_validator in self.custom_validators:
                try:
                    custom_results = custom_validator(data)
                    if isinstance(custom_results, list):
                        results.extend(custom_results)
                    elif isinstance(custom_results, ValidationResult):
                        results.append(custom_results)
                except Exception as e:
                    results.append(ValidationResult(
                        False,
                        ValidationSeverity.ERROR,
                        f"Custom validator error: {str(e)}",
                        "custom_validation"
                    ))
        
        except Exception as e:
            logger.error(f"Validation engine error: {str(e)}")
            results.append(ValidationResult(
                False,
                ValidationSeverity.ERROR,
                f"Validation engine error: {str(e)}",
                "validation_engine"
            ))
        
        return results
    
    def is_valid(self, results: List[ValidationResult]) -> bool:
        """Check if validation results indicate overall validity."""
        return all(
            result.is_valid for result in results 
            if result.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]
        )
    
    def get_errors(self, results: List[ValidationResult]) -> List[ValidationResult]:
        """Get only error-level validation results."""
        return [
            result for result in results
            if result.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]
        ]
    
    def get_warnings(self, results: List[ValidationResult]) -> List[ValidationResult]:
        """Get only warning-level validation results."""
        return [
            result for result in results
            if result.severity == ValidationSeverity.WARNING
        ]
"""
Comprehensive validation system for quantum HVAC control.
Includes input validation, constraint checking, and safety validation.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Validation strictness levels."""
    BASIC = "basic"
    STANDARD = "standard"
    STRICT = "strict"


@dataclass
class ValidationResult:
    """Result of validation check."""
    valid: bool
    message: str
    severity: str = "info"  # info, warning, error, critical
    suggestions: List[str] = None
    
    def __post_init__(self):
        if self.suggestions is None:
            self.suggestions = []


class ComprehensiveValidator:
    """Comprehensive validation for HVAC quantum control systems."""
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STANDARD):
        self.validation_level = validation_level
        self.logger = logger
    
    def validate_building_state(self, state: Dict[str, Any]) -> ValidationResult:
        """
        Validate building state input.
        
        Args:
            state: Building state dictionary
            
        Returns:
            ValidationResult indicating validation status
        """
        try:
            # Required fields
            required_fields = ['zone_temperatures', 'outside_temperature', 'occupancy']
            missing_fields = [f for f in required_fields if f not in state]
            
            if missing_fields:
                return ValidationResult(
                    valid=False,
                    message=f"Missing required fields: {missing_fields}",
                    severity="error",
                    suggestions=[f"Add {field} to state" for field in missing_fields]
                )
            
            # Temperature validation
            temps = np.array(state['zone_temperatures'])
            if not (-50 <= temps.min() <= temps.max() <= 60):
                return ValidationResult(
                    valid=False,
                    message=f"Zone temperatures out of range: {temps.min()}-{temps.max()}°C",
                    severity="error",
                    suggestions=["Check temperature sensors", "Verify units (Celsius expected)"]
                )
            
            outside_temp = state['outside_temperature']
            if not (-50 <= outside_temp <= 60):
                return ValidationResult(
                    valid=False,
                    message=f"Outside temperature out of range: {outside_temp}°C",
                    severity="error",
                    suggestions=["Check weather data source"]
                )
            
            # Occupancy validation
            occupancy = np.array(state['occupancy'])
            if not (0 <= occupancy.min() <= occupancy.max() <= 1):
                return ValidationResult(
                    valid=False,
                    message=f"Occupancy values out of range [0,1]: {occupancy.min()}-{occupancy.max()}",
                    severity="error",
                    suggestions=["Normalize occupancy to 0-1 range"]
                )
            
            # Warnings for unusual conditions
            warnings = []
            if abs(temps.mean() - outside_temp) < 2:
                warnings.append("Zone and outside temperatures very similar - check HVAC system")
            
            if occupancy.mean() > 0.9:
                warnings.append("Very high occupancy detected - verify sensors")
            
            return ValidationResult(
                valid=True,
                message="Building state validation passed",
                severity="warning" if warnings else "info",
                suggestions=warnings
            )
            
        except Exception as e:
            return ValidationResult(
                valid=False,
                message=f"Validation error: {str(e)}",
                severity="critical",
                suggestions=["Check input data format", "Verify numpy array compatibility"]
            )
    
    def validate_control_constraints(self, controls: np.ndarray, 
                                   building_config: Dict[str, Any]) -> ValidationResult:
        """
        Validate control outputs against system constraints.
        
        Args:
            controls: Control array
            building_config: Building configuration with limits
            
        Returns:
            ValidationResult for control constraints
        """
        try:
            # Power limits
            max_power = building_config.get('max_power_kw', 100)
            total_power = np.sum(np.abs(controls))
            
            if total_power > max_power:
                return ValidationResult(
                    valid=False,
                    message=f"Total power {total_power:.1f} kW exceeds limit {max_power} kW",
                    severity="error",
                    suggestions=[
                        "Reduce control magnitudes",
                        "Check power limits configuration",
                        "Consider load shedding"
                    ]
                )
            
            # Control rate limits
            max_rate = building_config.get('max_control_rate', 10)  # kW/min
            if len(controls) > 1:
                control_rates = np.abs(np.diff(controls))
                if control_rates.max() > max_rate:
                    return ValidationResult(
                        valid=False,
                        message=f"Control rate {control_rates.max():.1f} exceeds limit {max_rate}",
                        severity="warning",
                        suggestions=["Smooth control transitions", "Adjust rate limits"]
                    )
            
            return ValidationResult(
                valid=True,
                message="Control constraints satisfied",
                severity="info"
            )
            
        except Exception as e:
            return ValidationResult(
                valid=False,
                message=f"Control validation error: {str(e)}",
                severity="critical",
                suggestions=["Check control array format", "Verify building configuration"]
            )
    
    def validate_optimization_config(self, config: Dict[str, Any]) -> ValidationResult:
        """
        Validate optimization configuration.
        
        Args:
            config: Optimization configuration
            
        Returns:
            ValidationResult for configuration
        """
        try:
            # Horizon validation
            horizon = config.get('prediction_horizon', 24)
            if not (1 <= horizon <= 168):  # 1 hour to 1 week
                return ValidationResult(
                    valid=False,
                    message=f"Prediction horizon {horizon} hours outside reasonable range [1, 168]",
                    severity="warning",
                    suggestions=["Use 24-48 hours for typical applications"]
                )
            
            # Interval validation  
            interval = config.get('control_interval', 15)
            if not (5 <= interval <= 60):  # 5 minutes to 1 hour
                return ValidationResult(
                    valid=False,
                    message=f"Control interval {interval} minutes outside range [5, 60]",
                    severity="warning",
                    suggestions=["Use 15-30 minutes for most applications"]
                )
            
            # Solver validation
            solver = config.get('solver', 'classical_fallback')
            valid_solvers = ['classical_fallback', 'hybrid_v2', 'advantage_system4.1', 'quantum']
            if solver not in valid_solvers:
                return ValidationResult(
                    valid=False,
                    message=f"Unknown solver: {solver}",
                    severity="error",
                    suggestions=[f"Use one of: {', '.join(valid_solvers)}"]
                )
            
            return ValidationResult(
                valid=True,
                message="Optimization configuration valid",
                severity="info"
            )
            
        except Exception as e:
            return ValidationResult(
                valid=False,
                message=f"Configuration validation error: {str(e)}",
                severity="critical"
            )
    
    def validate_safety_limits(self, state: Dict[str, Any], 
                             controls: np.ndarray) -> ValidationResult:
        """
        Validate safety limits and emergency conditions.
        
        Args:
            state: Current building state
            controls: Proposed control actions
            
        Returns:
            ValidationResult for safety validation
        """
        try:
            safety_violations = []
            
            # Temperature safety limits
            temps = np.array(state.get('zone_temperatures', []))
            if len(temps) > 0:
                if temps.min() < 10:  # Too cold
                    safety_violations.append(f"Dangerously low temperature: {temps.min():.1f}°C")
                
                if temps.max() > 35:  # Too hot
                    safety_violations.append(f"Dangerously high temperature: {temps.max():.1f}°C")
            
            # Control magnitude safety
            if len(controls) > 0 and np.abs(controls).max() > 50:  # Very high control
                safety_violations.append(f"Extreme control magnitude: {np.abs(controls).max():.1f}")
            
            # Emergency conditions
            outside_temp = state.get('outside_temperature', 20)
            if outside_temp < -30 or outside_temp > 50:
                safety_violations.append(f"Extreme weather conditions: {outside_temp}°C")
            
            if safety_violations:
                return ValidationResult(
                    valid=False,
                    message=f"Safety violations detected: {'; '.join(safety_violations)}",
                    severity="critical",
                    suggestions=[
                        "Activate emergency protocols",
                        "Check sensor calibration", 
                        "Engage manual override if necessary"
                    ]
                )
            
            return ValidationResult(
                valid=True,
                message="All safety limits satisfied",
                severity="info"
            )
            
        except Exception as e:
            return ValidationResult(
                valid=False,
                message=f"Safety validation error: {str(e)}",
                severity="critical",
                suggestions=["Emergency stop recommended"]
            )
    
    def validate_complete_system(self, state: Dict[str, Any],
                               controls: np.ndarray,
                               config: Dict[str, Any],
                               building_config: Dict[str, Any]) -> Dict[str, ValidationResult]:
        """
        Run complete system validation.
        
        Args:
            state: Building state
            controls: Control outputs  
            config: Optimization config
            building_config: Building configuration
            
        Returns:
            Dictionary of validation results by category
        """
        results = {}
        
        # Run all validation checks
        results['state'] = self.validate_building_state(state)
        results['controls'] = self.validate_control_constraints(controls, building_config)
        results['config'] = self.validate_optimization_config(config)
        results['safety'] = self.validate_safety_limits(state, controls)
        
        return results
    
    def get_overall_status(self, validation_results: Dict[str, ValidationResult]) -> ValidationResult:
        """
        Get overall system validation status.
        
        Args:
            validation_results: Results from validate_complete_system
            
        Returns:
            Overall ValidationResult
        """
        # Check for any critical or error conditions
        critical_issues = [r for r in validation_results.values() 
                          if r.severity == "critical" or not r.valid]
        
        if critical_issues:
            return ValidationResult(
                valid=False,
                message=f"System validation failed: {len(critical_issues)} critical issues",
                severity="critical",
                suggestions=["Address all critical issues before proceeding"]
            )
        
        # Check for warnings
        warnings = [r for r in validation_results.values() if r.severity == "warning"]
        
        if warnings:
            return ValidationResult(
                valid=True,
                message=f"System validation passed with {len(warnings)} warnings",
                severity="warning",
                suggestions=["Review warnings for optimization opportunities"]
            )
        
        return ValidationResult(
            valid=True,
            message="Complete system validation passed",
            severity="info"
        )


# Convenience functions
def validate_system(state: Dict[str, Any],
                   controls: np.ndarray,
                   config: Dict[str, Any],
                   building_config: Dict[str, Any],
                   validation_level: ValidationLevel = ValidationLevel.STANDARD) -> Dict[str, ValidationResult]:
    """Convenience function for complete system validation."""
    validator = ComprehensiveValidator(validation_level)
    return validator.validate_complete_system(state, controls, config, building_config)


__all__ = [
    'ComprehensiveValidator', 
    'ValidationResult', 
    'ValidationLevel', 
    'validate_system'
]
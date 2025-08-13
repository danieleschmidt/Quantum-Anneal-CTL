"""
Configuration validation and system health checks.
"""

import os
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Result of a validation check."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    recommendations: List[str]

class SystemValidator:
    """Validates system configuration and environment."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def validate_environment(self) -> ValidationResult:
        """Validate the system environment."""
        errors = []
        warnings = []
        recommendations = []
        
        # Check Python version
        import sys
        if sys.version_info < (3, 9):
            errors.append(f"Python {sys.version_info.major}.{sys.version_info.minor} not supported. Requires Python 3.9+")
        
        # Check critical dependencies
        try:
            import numpy as np
            import scipy
            import dwave
        except ImportError as e:
            errors.append(f"Missing critical dependency: {e}")
        
        # Check D-Wave configuration
        dwave_token = os.getenv('DWAVE_API_TOKEN')
        if not dwave_token:
            warnings.append("D-Wave API token not configured. Quantum solving will use fallback.")
            recommendations.append("Set DWAVE_API_TOKEN environment variable for quantum access")
        
        # Check system resources
        try:
            import psutil
            memory = psutil.virtual_memory()
            if memory.available < 1024 * 1024 * 1024:  # 1GB
                warnings.append("Low system memory detected")
                recommendations.append("Ensure at least 2GB RAM for optimal performance")
        except ImportError:
            warnings.append("Cannot check system resources (psutil not available)")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            recommendations=recommendations
        )
    
    def validate_building_config(self, building) -> ValidationResult:
        """Validate building configuration."""
        errors = []
        warnings = []
        recommendations = []
        
        if not building.zones:
            errors.append("Building must have at least one zone")
        
        for i, zone in enumerate(building.zones):
            # Check zone configuration
            if zone.area <= 0:
                errors.append(f"Zone {i} has invalid area: {zone.area}")
            
            if zone.volume <= 0:
                errors.append(f"Zone {i} has invalid volume: {zone.volume}")
            
            if zone.max_heating_power <= 0:
                errors.append(f"Zone {i} has invalid heating power: {zone.max_heating_power}")
            
            if zone.max_cooling_power <= 0:
                errors.append(f"Zone {i} has invalid cooling power: {zone.max_cooling_power}")
            
            # Temperature range validation
            if zone.comfort_temp_min >= zone.comfort_temp_max:
                errors.append(f"Zone {i} has invalid comfort range: {zone.comfort_temp_min} >= {zone.comfort_temp_max}")
            
            # Reasonable ranges
            if zone.comfort_temp_min < 15 or zone.comfort_temp_max > 30:
                warnings.append(f"Zone {i} has unusual comfort range: {zone.comfort_temp_min}-{zone.comfort_temp_max}°C")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            recommendations=recommendations
        )
    
    def validate_optimization_config(self, config) -> ValidationResult:
        """Validate optimization configuration."""
        errors = []
        warnings = []
        recommendations = []
        
        if config.prediction_horizon <= 0:
            errors.append("Prediction horizon must be positive")
        elif config.prediction_horizon > 48:
            warnings.append("Very long prediction horizon may impact performance")
            recommendations.append("Consider reducing prediction horizon to 24 hours or less")
        
        if config.control_interval <= 0:
            errors.append("Control interval must be positive")
        elif config.control_interval < 5:
            warnings.append("Very short control interval may cause instability")
            recommendations.append("Consider using 15-minute or longer control intervals")
        
        if config.num_reads <= 0:
            errors.append("Number of reads must be positive")
        elif config.num_reads > 10000:
            warnings.append("Very high number of reads may be expensive")
            recommendations.append("Consider reducing num_reads for cost optimization")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            recommendations=recommendations
        )
    
    def validate_state_data(self, state, building) -> ValidationResult:
        """Validate building state data."""
        errors = []
        warnings = []
        recommendations = []
        
        expected_zones = len(building.zones)
        
        # Check array dimensions
        if len(state.zone_temperatures) != expected_zones:
            errors.append(f"Zone temperatures length {len(state.zone_temperatures)} != expected {expected_zones}")
        
        if len(state.occupancy) != expected_zones:
            errors.append(f"Occupancy length {len(state.occupancy)} != expected {expected_zones}")
        
        if len(state.hvac_power) != expected_zones:
            errors.append(f"HVAC power length {len(state.hvac_power)} != expected {expected_zones}")
        
        if len(state.control_setpoints) != expected_zones:
            errors.append(f"Control setpoints length {len(state.control_setpoints)} != expected {expected_zones}")
        
        # Check value ranges
        for i, temp in enumerate(state.zone_temperatures):
            if not 0 <= temp <= 50:  # Reasonable temperature range
                warnings.append(f"Zone {i} temperature {temp}°C is outside normal range")
        
        if not -50 <= state.outside_temperature <= 50:
            warnings.append(f"Outside temperature {state.outside_temperature}°C is extreme")
        
        if not 0 <= state.humidity <= 100:
            warnings.append(f"Humidity {state.humidity}% is outside valid range")
        
        for i, occ in enumerate(state.occupancy):
            if not 0 <= occ <= 1:
                errors.append(f"Zone {i} occupancy {occ} must be between 0 and 1")
        
        for i, power in enumerate(state.hvac_power):
            max_power = max(building.zones[i].max_heating_power, building.zones[i].max_cooling_power)
            if power > max_power * 1.1:  # Allow 10% overpower
                warnings.append(f"Zone {i} HVAC power {power}kW exceeds max {max_power}kW")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            recommendations=recommendations
        )

def validate_system() -> Tuple[bool, Dict[str, Any]]:
    """Complete system validation."""
    validator = SystemValidator()
    
    # Run all validations
    env_result = validator.validate_environment()
    
    results = {
        'environment': env_result,
        'overall_valid': env_result.is_valid,
        'total_errors': len(env_result.errors),
        'total_warnings': len(env_result.warnings)
    }
    
    # Log results
    if env_result.errors:
        for error in env_result.errors:
            logger.error(f"Validation error: {error}")
    
    if env_result.warnings:
        for warning in env_result.warnings:
            logger.warning(f"Validation warning: {warning}")
    
    if env_result.recommendations:
        for rec in env_result.recommendations:
            logger.info(f"Recommendation: {rec}")
    
    return results['overall_valid'], results
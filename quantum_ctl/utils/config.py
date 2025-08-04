"""
Configuration management utilities.
"""

import json
from pathlib import Path
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
from typing import Dict, Any, Union
import logging

logger = logging.getLogger(__name__)


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from JSON or YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config format is invalid
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            if config_path.suffix.lower() == '.json':
                config = json.load(f)
            elif config_path.suffix.lower() in ['.yaml', '.yml']:
                if not YAML_AVAILABLE:
                    raise ValueError("YAML support not available. Install PyYAML to use YAML configs.")
                config = yaml.safe_load(f)
            else:
                # Try JSON first, then YAML
                content = f.read()
                try:
                    config = json.loads(content)
                except json.JSONDecodeError:
                    if YAML_AVAILABLE:
                        config = yaml.safe_load(content)
                    else:
                        raise ValueError("Could not parse config as JSON and YAML not available")
        
        logger.info(f"Loaded configuration from {config_path}")
        return config
        
    except Exception as e:
        raise ValueError(f"Failed to parse configuration file {config_path}: {e}")


def save_config(config: Dict[str, Any], config_path: Union[str, Path]) -> None:
    """
    Save configuration to JSON or YAML file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(config_path, 'w') as f:
            if config_path.suffix.lower() == '.json':
                json.dump(config, f, indent=2)
            elif config_path.suffix.lower() in ['.yaml', '.yml']:
                if not YAML_AVAILABLE:
                    raise ValueError("YAML support not available. Install PyYAML to save YAML configs.")
                yaml.dump(config, f, default_flow_style=False, indent=2)
            else:
                # Default to JSON
                json.dump(config, f, indent=2)
        
        logger.info(f"Saved configuration to {config_path}")
        
    except Exception as e:
        raise ValueError(f"Failed to save configuration to {config_path}: {e}")


def get_default_config() -> Dict[str, Any]:
    """Get default system configuration."""
    return {
        'quantum': {
            'solver_type': 'hybrid_v2',
            'num_reads': 1000,
            'annealing_time': 20,
            'chain_strength': None,
            'auto_scale': True
        },
        'mpc': {
            'prediction_horizon': 24,  # hours
            'control_interval': 15,    # minutes
            'precision_bits': 4
        },
        'objectives': {
            'energy_cost': 0.6,
            'comfort': 0.3,
            'carbon': 0.1
        },
        'penalties': {
            'dynamics': 1000.0,
            'comfort': 100.0,
            'energy': 50.0,
            'control_limits': 500.0
        },
        'building': {
            'zones': 5,
            'thermal_mass': 1000.0,
            'envelope_ua': 500.0,
            'occupancy_schedule': 'office_standard'
        },
        'logging': {
            'level': 'INFO',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        },
        'api': {
            'host': '0.0.0.0',
            'port': 8000,
            'debug': False
        }
    }


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configuration dictionaries.
    
    Args:
        base_config: Base configuration
        override_config: Override configuration
        
    Returns:
        Merged configuration
    """
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged


def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate configuration dictionary.
    
    Args:
        config: Configuration to validate
        
    Returns:
        Validation results
    """
    issues = []
    warnings = []
    
    # Required sections
    required_sections = ['quantum', 'mpc', 'objectives']
    for section in required_sections:
        if section not in config:
            issues.append(f"Missing required section: {section}")
    
    # Quantum configuration
    if 'quantum' in config:
        quantum_config = config['quantum']
        
        if 'solver_type' not in quantum_config:
            issues.append("Missing quantum.solver_type")
        
        if 'num_reads' in quantum_config:
            if not isinstance(quantum_config['num_reads'], int) or quantum_config['num_reads'] <= 0:
                issues.append("quantum.num_reads must be positive integer")
    
    # MPC configuration
    if 'mpc' in config:
        mpc_config = config['mpc']
        
        if 'prediction_horizon' in mpc_config:
            if not isinstance(mpc_config['prediction_horizon'], (int, float)) or mpc_config['prediction_horizon'] <= 0:
                issues.append("mpc.prediction_horizon must be positive number")
        
        if 'control_interval' in mpc_config:
            if not isinstance(mpc_config['control_interval'], (int, float)) or mpc_config['control_interval'] <= 0:
                issues.append("mpc.control_interval must be positive number")
    
    # Objectives configuration
    if 'objectives' in config:
        objectives = config['objectives']
        
        required_objectives = ['energy_cost', 'comfort', 'carbon']
        for obj in required_objectives:
            if obj not in objectives:
                issues.append(f"Missing objective: {obj}")
            elif not isinstance(objectives[obj], (int, float)) or objectives[obj] < 0:
                issues.append(f"Objective {obj} must be non-negative number")
        
        # Check sum of objectives
        if all(obj in objectives for obj in required_objectives):
            total = sum(objectives[obj] for obj in required_objectives)
            if not (0.99 <= total <= 1.01):
                warnings.append(f"Objective weights sum to {total:.3f}, should be 1.0")
    
    return {
        'valid': len(issues) == 0,
        'issues': issues,
        'warnings': warnings
    }
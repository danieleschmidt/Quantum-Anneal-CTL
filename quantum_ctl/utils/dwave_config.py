"""
D-Wave quantum computer configuration and connection utilities.
"""

import os
import json
from typing import Dict, Any, Optional, List
import logging
from pathlib import Path

try:
    from dwave.cloud import Client
    from dwave.system import DWaveSampler
    DWAVE_AVAILABLE = True
except ImportError:
    DWAVE_AVAILABLE = False
    Client = None


logger = logging.getLogger(__name__)


class DWaveConfig:
    """D-Wave quantum computer configuration manager."""
    
    def __init__(self):
        self.config_path = Path.home() / ".dwave" / "dwave.conf"
        self.api_token = None
        self.endpoint = None
        self.solver = None
        
        # Load configuration
        self._load_config()
    
    def _load_config(self) -> None:
        """Load D-Wave configuration from environment and config files."""
        # Try environment variables first
        self.api_token = os.getenv("DWAVE_API_TOKEN")
        self.endpoint = os.getenv("DWAVE_API_ENDPOINT", "https://cloud.dwavesys.com/sapi")
        self.solver = os.getenv("DWAVE_SOLVER")
        
        # Try config file if environment variables not set
        if not self.api_token and self.config_path.exists():
            try:
                import configparser
                config = configparser.ConfigParser()
                config.read(self.config_path)
                
                if 'defaults' in config:
                    defaults = config['defaults']
                    self.api_token = self.api_token or defaults.get('token')
                    self.endpoint = self.endpoint or defaults.get('endpoint', self.endpoint)
                    self.solver = self.solver or defaults.get('solver')
                    
            except Exception as e:
                logger.warning(f"Could not read D-Wave config file: {e}")
    
    def is_configured(self) -> bool:
        """Check if D-Wave is properly configured."""
        return DWAVE_AVAILABLE and self.api_token is not None
    
    def get_config(self) -> Dict[str, Any]:
        """Get current D-Wave configuration."""
        return {
            'api_token': '***' if self.api_token else None,
            'endpoint': self.endpoint,
            'solver': self.solver,
            'configured': self.is_configured(),
            'sdk_available': DWAVE_AVAILABLE
        }
    
    def test_connection(self) -> Dict[str, Any]:
        """Test connection to D-Wave cloud service."""
        if not DWAVE_AVAILABLE:
            return {
                'status': 'error',
                'message': 'D-Wave Ocean SDK not installed'
            }
        
        if not self.api_token:
            return {
                'status': 'error', 
                'message': 'D-Wave API token not configured'
            }
        
        try:
            # Test connection
            client = Client.from_config(token=self.api_token, endpoint=self.endpoint)
            solvers = client.get_solvers()
            
            if not solvers:
                return {
                    'status': 'error',
                    'message': 'No solvers available'
                }
            
            # Categorize available solvers
            qpu_solvers = [s for s in solvers if s.solver_type == 'qpu']
            hybrid_solvers = [s for s in solvers if 'hybrid' in s.name.lower()]
            
            return {
                'status': 'success',
                'message': f'Connected successfully, {len(solvers)} solvers available',
                'solvers': {
                    'total': len(solvers),
                    'qpu': len(qpu_solvers),
                    'hybrid': len(hybrid_solvers),
                    'qpu_list': [
                        {
                            'name': s.name,
                            'qubits': s.properties.get('num_qubits', 0),
                            'topology': s.properties.get('topology', {}).get('type', 'unknown')
                        }
                        for s in qpu_solvers[:5]  # Limit to first 5
                    ],
                    'hybrid_list': [s.name for s in hybrid_solvers]
                }
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Connection failed: {str(e)}'
            }
    
    def get_recommended_solver(self, problem_size: int = 1000) -> Optional[str]:
        """Get recommended solver based on problem size."""
        if not self.is_configured():
            return None
        
        try:
            client = Client.from_config(token=self.api_token, endpoint=self.endpoint)
            solvers = client.get_solvers()
            
            # For small problems, prefer QPU
            if problem_size <= 1000:
                qpu_solvers = [s for s in solvers if s.solver_type == 'qpu']
                if qpu_solvers:
                    # Select largest available QPU
                    best_qpu = max(qpu_solvers, key=lambda s: s.properties.get('num_qubits', 0))
                    return best_qpu.name
            
            # For larger problems, use hybrid
            hybrid_solvers = [s for s in solvers if 'hybrid' in s.name.lower()]
            if hybrid_solvers:
                return hybrid_solvers[0].name
            
            # Fallback to any available solver
            if solvers:
                return solvers[0].name
                
        except Exception as e:
            logger.warning(f"Could not get solver recommendation: {e}")
        
        return None
    
    def create_config_file(self, api_token: str, endpoint: Optional[str] = None) -> bool:
        """Create D-Wave configuration file."""
        try:
            import configparser
            
            # Ensure config directory exists
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create configuration
            config = configparser.ConfigParser()
            config['defaults'] = {
                'token': api_token,
                'endpoint': endpoint or self.endpoint
            }
            
            # Write to file
            with open(self.config_path, 'w') as f:
                config.write(f)
            
            # Set restrictive permissions
            os.chmod(self.config_path, 0o600)
            
            # Reload configuration
            self._load_config()
            
            logger.info(f"D-Wave configuration saved to {self.config_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create D-Wave config: {e}")
            return False


def setup_dwave_config() -> DWaveConfig:
    """Setup and return D-Wave configuration."""
    config = DWaveConfig()
    
    if not config.is_configured():
        logger.warning("D-Wave not configured. Set DWAVE_API_TOKEN environment variable or use 'dwave config create'")
    
    return config


def get_solver_info(solver_name: str) -> Dict[str, Any]:
    """Get detailed information about a specific solver."""
    if not DWAVE_AVAILABLE:
        return {'error': 'D-Wave SDK not available'}
    
    try:
        client = Client.from_config()
        solver = client.get_solver(solver_name)
        
        info = {
            'name': solver.name,
            'solver_type': solver.solver_type,
            'status': solver.status,
            'properties': {}
        }
        
        # Add QPU-specific properties
        if solver.solver_type == 'qpu':
            props = solver.properties
            info['properties'] = {
                'num_qubits': props.get('num_qubits', 0),
                'chip_id': props.get('chip_id', 'unknown'),
                'topology': props.get('topology', {}),
                'programming_thermalization_range': props.get('programming_thermalization_range', []),
                'annealing_time_range': props.get('annealing_time_range', []),
                'num_reads_range': props.get('num_reads_range', [])
            }
        
        return info
        
    except Exception as e:
        return {'error': f'Could not get solver info: {e}'}


# Global configuration instance
dwave_config = setup_dwave_config()
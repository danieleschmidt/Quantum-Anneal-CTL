"""
Graceful fallback system for quantum computing dependencies.
Enables system to function without D-Wave or other quantum dependencies.
"""

import os
import logging
from typing import Any, Optional, Callable
from functools import wraps

logger = logging.getLogger(__name__)

# Global flags for available components
DWAVE_AVAILABLE = False
QUANTUM_AVAILABLE = False

def check_dwave_availability():
    """Check if D-Wave Ocean SDK is available and configured."""
    global DWAVE_AVAILABLE
    
    try:
        import dwave
        from dwave.system import DWaveSampler
        
        # Check configuration
        api_token = os.getenv('DWAVE_API_TOKEN')
        if not api_token:
            logger.warning("D-Wave not configured. Set DWAVE_API_TOKEN environment variable")
            DWAVE_AVAILABLE = False
        else:
            DWAVE_AVAILABLE = True
            logger.info("D-Wave Ocean SDK available and configured")
            
    except ImportError as e:
        logger.info("D-Wave Ocean SDK not available - using classical fallback")
        DWAVE_AVAILABLE = False
    
    return DWAVE_AVAILABLE


def quantum_fallback(classical_func: Callable):
    """
    Decorator to provide classical fallback for quantum operations.
    
    Args:
        classical_func: Classical implementation to use as fallback
    """
    def decorator(quantum_func: Callable):
        @wraps(quantum_func)
        def wrapper(*args, **kwargs):
            if DWAVE_AVAILABLE:
                try:
                    return quantum_func(*args, **kwargs)
                except Exception as e:
                    logger.warning(f"Quantum operation failed: {e}, using classical fallback")
                    return classical_func(*args, **kwargs)
            else:
                logger.info("Using classical fallback (quantum not available)")
                return classical_func(*args, **kwargs)
        return wrapper
    return decorator


def get_solver_type():
    """Return the best available solver type."""
    if DWAVE_AVAILABLE:
        return "quantum_hybrid"
    else:
        return "classical_fallback"


def mock_dwave_response(energy: float, num_reads: int = 100):
    """Create a mock D-Wave response for testing."""
    from collections import namedtuple
    
    MockSample = namedtuple('MockSample', ['sample', 'energy'])
    MockResponse = namedtuple('MockResponse', ['first', 'data_vectors'])
    
    import numpy as np
    sample = {i: np.random.choice([0, 1]) for i in range(10)}  # Mock binary variables
    
    return MockResponse(
        first=MockSample(sample=sample, energy=energy),
        data_vectors=['energy']
    )


# Initialize availability check
check_dwave_availability()
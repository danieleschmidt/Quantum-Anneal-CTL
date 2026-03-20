"""
Quantum Anneal CTL — Quantum annealing-based HVAC control for building microgrids.

This package implements a QUBO (Quadratic Unconstrained Binary Optimization)
formulation for HVAC control, solved via simulated annealing as a quantum-
annealing proxy.
"""

from quantum_ctl.hvac_model import HVACMicroGrid
from quantum_ctl.mpc_controller import MPCController
from quantum_ctl.optimizer import SimulatedAnnealingOptimizer

__all__ = ["HVACMicroGrid", "MPCController", "SimulatedAnnealingOptimizer"]
__version__ = "0.1.0"

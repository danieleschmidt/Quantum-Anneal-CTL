"""
Quantum-Anneal-CTL: Quantum annealing controller for HVAC micro-grids.

A framework for solving complex HVAC optimization problems using quantum annealing,
leveraging D-Wave quantum computers for real-time energy optimization in smart buildings.
"""

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"
__email__ = "daniel@terragonlabs.com"

from .core.controller import HVACController
from .core.microgrid import MicroGridController
from .models.building import Building
from .optimization.mpc_to_qubo import MPCToQUBO
from .integration.bms_connector import BMSConnector

__all__ = [
    "HVACController",
    "MicroGridController", 
    "Building",
    "MPCToQUBO",
    "BMSConnector",
]
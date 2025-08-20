"""
Research modules for novel quantum annealing algorithms and optimization techniques.

This module contains experimental and research-focused implementations for:
- Novel QUBO formulations
- Advanced embedding strategies  
- Hybrid quantum-classical algorithms
- Comparative performance studies
- Experimental benchmarking frameworks
"""

from .novel_qubo_formulations import NovelQUBOFormulator, AdaptiveConstraintWeighting
from .advanced_embedding_strategies import TopologyAwareEmbedder, DynamicEmbeddingOptimizer
from .experimental_benchmarks import ComparativeBenchmarkSuite, StatisticalValidator
from .adaptive_penalty_tuning import BayesianPenaltyOptimizer, MLPenaltyTuner

# New research enhancement modules
from .quantum_error_mitigation import QuantumErrorMitigationEngine, create_hvac_error_mitigator
from .multi_fidelity_optimization import MultiFidelityOptimizer, optimize_hvac_multifidelity
from .quantum_reinforcement_learning import QuantumQLearningAgent, create_quantum_hvac_agent
from .distributed_quantum_mesh import QuantumMeshCoordinator, create_quantum_mesh_network

__all__ = [
    "NovelQUBOFormulator",
    "AdaptiveConstraintWeighting", 
    "TopologyAwareEmbedder",
    "DynamicEmbeddingOptimizer",
    "ComparativeBenchmarkSuite",
    "StatisticalValidator",
    "BayesianPenaltyOptimizer",
    "MLPenaltyTuner",
    # New research enhancement modules
    "QuantumErrorMitigationEngine",
    "create_hvac_error_mitigator",
    "MultiFidelityOptimizer", 
    "optimize_hvac_multifidelity",
    "QuantumQLearningAgent",
    "create_quantum_hvac_agent",
    "QuantumMeshCoordinator",
    "create_quantum_mesh_network"
]
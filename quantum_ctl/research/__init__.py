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
from .hybrid_quantum_algorithms import VariationalQUBOSolver, QuantumApproximateOptimization
from .experimental_benchmarks import ComparativeBenchmarkSuite, StatisticalValidator
from .adaptive_penalty_tuning import BayesianPenaltyOptimizer, MLPenaltyTuner

__all__ = [
    "NovelQUBOFormulator",
    "AdaptiveConstraintWeighting", 
    "TopologyAwareEmbedder",
    "DynamicEmbeddingOptimizer",
    "VariationalQUBOSolver",
    "QuantumApproximateOptimization",
    "ComparativeBenchmarkSuite",
    "StatisticalValidator",
    "BayesianPenaltyOptimizer",
    "MLPenaltyTuner"
]
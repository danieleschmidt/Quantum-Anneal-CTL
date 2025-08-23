"""
Scaling and performance optimization modules for quantum HVAC control.

This module provides comprehensive scaling capabilities including:
- Auto-scaling quantum solver clusters
- Performance-aware load balancing
- Distributed computation coordination
- Resource optimization and caching
- Concurrent processing pipelines
"""

from .auto_scaler import QuantumAutoScaler, ScalingPolicy, ResourceMetrics
from .performance_optimizer import PerformanceOptimizer, OptimizationStrategy
from .distributed_quantum_coordinator import DistributedQuantumCoordinator, get_quantum_coordinator
from .global_quantum_orchestrator import GlobalQuantumOrchestrator

__all__ = [
    "QuantumAutoScaler",
    "ScalingPolicy", 
    "ResourceMetrics",
    "PerformanceOptimizer",
    "OptimizationStrategy",
    "DistributedQuantumCoordinator",
    "get_quantum_coordinator",
    "GlobalQuantumOrchestrator"
]
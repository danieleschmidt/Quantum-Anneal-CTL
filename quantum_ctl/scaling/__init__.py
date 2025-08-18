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
from .distributed_coordinator import DistributedCoordinator, CoordinationStrategy
from .concurrent_pipeline import ConcurrentPipeline, PipelineStage
from .resource_manager import ResourceManager, ResourceAllocation

__all__ = [
    "QuantumAutoScaler",
    "ScalingPolicy",
    "ResourceMetrics",
    "PerformanceOptimizer", 
    "OptimizationStrategy",
    "DistributedCoordinator",
    "CoordinationStrategy",
    "ConcurrentPipeline",
    "PipelineStage",
    "ResourceManager",
    "ResourceAllocation"
]
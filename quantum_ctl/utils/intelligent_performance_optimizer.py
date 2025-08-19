"""
Intelligent performance optimization system for quantum HVAC operations.
Provides adaptive performance tuning, resource optimization, and intelligent workload management.
"""

import asyncio
import time
import threading
import multiprocessing
import psutil
import logging
from typing import Any, Dict, List, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import statistics
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import queue

logger = logging.getLogger(__name__)


class OptimizationStrategy(Enum):
    """Performance optimization strategies."""
    AGGRESSIVE = "aggressive"
    BALANCED = "balanced"
    CONSERVATIVE = "conservative"
    ADAPTIVE = "adaptive"


class ResourceType(Enum):
    """Types of system resources."""
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"
    QUANTUM_QPU = "quantum_qpu"


@dataclass
class PerformanceMetrics:
    """Performance metrics for optimization decisions."""
    timestamp: float
    cpu_usage: float
    memory_usage: float
    disk_io: float
    network_io: float
    response_time: float
    throughput: float
    error_rate: float
    quantum_queue_time: Optional[float] = None
    active_connections: int = 0


@dataclass
class OptimizationAction:
    """Represents a performance optimization action."""
    action_type: str
    resource_type: ResourceType
    parameters: Dict[str, Any]
    expected_improvement: float
    confidence: float
    priority: int


class AdaptiveResourceManager:
    """Manages system resources with adaptive optimization."""
    
    def __init__(self, strategy: OptimizationStrategy = OptimizationStrategy.ADAPTIVE):
        self.strategy = strategy
        self.cpu_count = multiprocessing.cpu_count()
        self.memory_total = psutil.virtual_memory().total
        
        # Resource limits and thresholds
        self.resource_limits = {
            ResourceType.CPU: {"warning": 70, "critical": 85, "max": 95},
            ResourceType.MEMORY: {"warning": 70, "critical": 85, "max": 95},
            ResourceType.DISK: {"warning": 80, "critical": 90, "max": 95}
        }
        
        # Thread and process pools with adaptive sizing
        self.thread_pool = None
        self.process_pool = None
        self.pool_lock = threading.Lock()
        
        # Performance history for learning
        self.performance_history = deque(maxlen=1000)
        self.optimization_history = deque(maxlen=100)
        
        # Current resource allocation
        self.current_allocation = {
            "max_threads": min(32, self.cpu_count * 4),
            "max_processes": max(2, self.cpu_count // 2),
            "cache_size_mb": 256,
            "batch_size": 10,
            "timeout_seconds": 30
        }
        
        self._initialize_pools()
    
    def _initialize_pools(self):
        """Initialize thread and process pools with current allocation."""
        with self.pool_lock:
            # Clean up existing pools
            if self.thread_pool:
                self.thread_pool.shutdown(wait=False)
            if self.process_pool:
                self.process_pool.shutdown(wait=False)
            
            # Create new pools
            self.thread_pool = ThreadPoolExecutor(
                max_workers=self.current_allocation["max_threads"],
                thread_name_prefix="hvac_optimizer"
            )
            self.process_pool = ProcessPoolExecutor(
                max_workers=self.current_allocation["max_processes"]
            )
    
    async def optimize_resources(self, metrics: PerformanceMetrics) -> List[OptimizationAction]:
        """Analyze metrics and recommend optimization actions."""
        actions = []
        
        # Store metrics for learning
        self.performance_history.append(metrics)
        
        # CPU optimization
        cpu_actions = self._optimize_cpu_usage(metrics)
        actions.extend(cpu_actions)
        
        # Memory optimization
        memory_actions = self._optimize_memory_usage(metrics)
        actions.extend(memory_actions)
        
        # Concurrency optimization
        concurrency_actions = self._optimize_concurrency(metrics)
        actions.extend(concurrency_actions)
        
        # Quantum-specific optimizations
        quantum_actions = self._optimize_quantum_resources(metrics)
        actions.extend(quantum_actions)
        
        # Sort actions by priority and confidence
        actions.sort(key=lambda x: (x.priority, -x.confidence))
        
        return actions
    
    def _optimize_cpu_usage(self, metrics: PerformanceMetrics) -> List[OptimizationAction]:
        """Optimize CPU usage based on current metrics."""
        actions = []
        cpu_usage = metrics.cpu_usage
        
        if cpu_usage > self.resource_limits[ResourceType.CPU]["critical"]:
            # Critical CPU usage - aggressive optimization
            if self.current_allocation["max_threads"] > 8:
                actions.append(OptimizationAction(
                    action_type="reduce_thread_pool",
                    resource_type=ResourceType.CPU,
                    parameters={"new_size": max(8, self.current_allocation["max_threads"] // 2)},
                    expected_improvement=15.0,
                    confidence=0.8,
                    priority=1
                ))
            
            actions.append(OptimizationAction(
                action_type="enable_cpu_throttling",
                resource_type=ResourceType.CPU,
                parameters={"throttle_percent": 20},
                expected_improvement=20.0,
                confidence=0.7,
                priority=2
            ))
        
        elif cpu_usage < 30 and len(self.performance_history) > 10:
            # Low CPU usage - can increase parallelism
            avg_cpu = statistics.mean([m.cpu_usage for m in list(self.performance_history)[-10:]])
            if avg_cpu < 40:
                new_threads = min(64, int(self.current_allocation["max_threads"] * 1.5))
                actions.append(OptimizationAction(
                    action_type="increase_thread_pool",
                    resource_type=ResourceType.CPU,
                    parameters={"new_size": new_threads},
                    expected_improvement=10.0,
                    confidence=0.6,
                    priority=3
                ))
        
        return actions
    
    def _optimize_memory_usage(self, metrics: PerformanceMetrics) -> List[OptimizationAction]:
        """Optimize memory usage based on current metrics."""
        actions = []
        memory_usage = metrics.memory_usage
        
        if memory_usage > self.resource_limits[ResourceType.MEMORY]["critical"]:
            # Critical memory usage
            actions.append(OptimizationAction(
                action_type="reduce_cache_size",
                resource_type=ResourceType.MEMORY,
                parameters={"new_size_mb": self.current_allocation["cache_size_mb"] // 2},
                expected_improvement=25.0,
                confidence=0.9,
                priority=1
            ))
            
            actions.append(OptimizationAction(
                action_type="force_garbage_collection",
                resource_type=ResourceType.MEMORY,
                parameters={},
                expected_improvement=10.0,
                confidence=0.7,
                priority=2
            ))
        
        elif memory_usage < 50 and self.current_allocation["cache_size_mb"] < 1024:
            # Low memory usage - can increase cache
            actions.append(OptimizationAction(
                action_type="increase_cache_size",
                resource_type=ResourceType.MEMORY,
                parameters={"new_size_mb": min(1024, self.current_allocation["cache_size_mb"] * 2)},
                expected_improvement=15.0,
                confidence=0.6,
                priority=3
            ))
        
        return actions
    
    def _optimize_concurrency(self, metrics: PerformanceMetrics) -> List[OptimizationAction]:
        """Optimize concurrency based on response times and throughput."""
        actions = []
        
        # Analyze recent performance trends
        if len(self.performance_history) < 5:
            return actions
        
        recent_metrics = list(self.performance_history)[-5:]
        avg_response_time = statistics.mean([m.response_time for m in recent_metrics])
        avg_throughput = statistics.mean([m.throughput for m in recent_metrics])
        
        # High response time indicates overload
        if avg_response_time > 5.0:  # 5 seconds
            actions.append(OptimizationAction(
                action_type="reduce_batch_size",
                resource_type=ResourceType.CPU,
                parameters={"new_batch_size": max(1, self.current_allocation["batch_size"] // 2)},
                expected_improvement=20.0,
                confidence=0.8,
                priority=1
            ))
        
        # Low response time with low CPU - can increase concurrency
        elif avg_response_time < 1.0 and metrics.cpu_usage < 60:
            actions.append(OptimizationAction(
                action_type="increase_batch_size",
                resource_type=ResourceType.CPU,
                parameters={"new_batch_size": min(50, self.current_allocation["batch_size"] * 2)},
                expected_improvement=15.0,
                confidence=0.7,
                priority=3
            ))
        
        return actions
    
    def _optimize_quantum_resources(self, metrics: PerformanceMetrics) -> List[OptimizationAction]:
        """Optimize quantum-specific resources."""
        actions = []
        
        if metrics.quantum_queue_time and metrics.quantum_queue_time > 30:
            # Long quantum queue time - use hybrid approach
            actions.append(OptimizationAction(
                action_type="enable_hybrid_solver",
                resource_type=ResourceType.QUANTUM_QPU,
                parameters={"quantum_fraction": 0.3},
                expected_improvement=40.0,
                confidence=0.8,
                priority=1
            ))
            
            actions.append(OptimizationAction(
                action_type="increase_classical_fallback",
                resource_type=ResourceType.CPU,
                parameters={"fallback_threshold": 10},
                expected_improvement=30.0,
                confidence=0.7,
                priority=2
            ))
        
        return actions
    
    async def apply_optimization(self, action: OptimizationAction) -> bool:
        """Apply an optimization action."""
        try:
            success = False
            
            if action.action_type == "reduce_thread_pool":
                new_size = action.parameters["new_size"]
                self.current_allocation["max_threads"] = new_size
                self._initialize_pools()
                success = True
                logger.info(f"Reduced thread pool to {new_size}")
            
            elif action.action_type == "increase_thread_pool":
                new_size = action.parameters["new_size"]
                self.current_allocation["max_threads"] = new_size
                self._initialize_pools()
                success = True
                logger.info(f"Increased thread pool to {new_size}")
            
            elif action.action_type == "reduce_cache_size":
                new_size = action.parameters["new_size_mb"]
                self.current_allocation["cache_size_mb"] = new_size
                success = True
                logger.info(f"Reduced cache size to {new_size}MB")
            
            elif action.action_type == "increase_cache_size":
                new_size = action.parameters["new_size_mb"]
                self.current_allocation["cache_size_mb"] = new_size
                success = True
                logger.info(f"Increased cache size to {new_size}MB")
            
            elif action.action_type == "force_garbage_collection":
                import gc
                gc.collect()
                success = True
                logger.info("Forced garbage collection")
            
            elif action.action_type == "reduce_batch_size":
                new_size = action.parameters["new_batch_size"]
                self.current_allocation["batch_size"] = new_size
                success = True
                logger.info(f"Reduced batch size to {new_size}")
            
            elif action.action_type == "increase_batch_size":
                new_size = action.parameters["new_batch_size"]
                self.current_allocation["batch_size"] = new_size
                success = True
                logger.info(f"Increased batch size to {new_size}")
            
            # Record optimization action
            if success:
                self.optimization_history.append({
                    "timestamp": time.time(),
                    "action": action,
                    "success": True
                })
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to apply optimization {action.action_type}: {e}")
            self.optimization_history.append({
                "timestamp": time.time(),
                "action": action,
                "success": False,
                "error": str(e)
            })
            return False
    
    def get_current_allocation(self) -> Dict[str, Any]:
        """Get current resource allocation."""
        return self.current_allocation.copy()
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        if not self.optimization_history:
            return {"total_optimizations": 0}
        
        total_optimizations = len(self.optimization_history)
        successful_optimizations = sum(1 for o in self.optimization_history if o["success"])
        
        action_types = defaultdict(int)
        for opt in self.optimization_history:
            action_types[opt["action"].action_type] += 1
        
        return {
            "total_optimizations": total_optimizations,
            "success_rate": successful_optimizations / total_optimizations,
            "action_distribution": dict(action_types),
            "current_allocation": self.current_allocation
        }


class IntelligentWorkloadBalancer:
    """Intelligent load balancer for quantum HVAC workloads."""
    
    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or multiprocessing.cpu_count() * 2
        self.worker_queues = []
        self.worker_stats = []
        self.load_balancing_strategy = "least_loaded"
        
        # Initialize worker queues
        for i in range(self.max_workers):
            self.worker_queues.append(asyncio.Queue())
            self.worker_stats.append({
                "worker_id": i,
                "active_tasks": 0,
                "completed_tasks": 0,
                "total_time": 0.0,
                "avg_time": 0.0,
                "last_activity": time.time()
            })
    
    async def submit_task(self, task_func: Callable, *args, **kwargs) -> Any:
        """Submit a task to be executed by the load balancer."""
        
        # Select optimal worker
        worker_id = self._select_worker()
        
        # Create task wrapper
        task_wrapper = TaskWrapper(
            task_id=f"task_{int(time.time() * 1000000)}",
            func=task_func,
            args=args,
            kwargs=kwargs,
            worker_id=worker_id
        )
        
        # Submit to worker queue
        await self.worker_queues[worker_id].put(task_wrapper)
        self.worker_stats[worker_id]["active_tasks"] += 1
        
        # Wait for result
        return await task_wrapper.result_future
    
    def _select_worker(self) -> int:
        """Select optimal worker based on load balancing strategy."""
        
        if self.load_balancing_strategy == "round_robin":
            # Simple round robin
            return hash(time.time()) % len(self.worker_queues)
        
        elif self.load_balancing_strategy == "least_loaded":
            # Select worker with least active tasks
            min_tasks = min(stats["active_tasks"] for stats in self.worker_stats)
            for i, stats in enumerate(self.worker_stats):
                if stats["active_tasks"] == min_tasks:
                    return i
        
        elif self.load_balancing_strategy == "fastest_worker":
            # Select worker with best average performance
            best_worker = 0
            best_time = float('inf')
            
            for i, stats in enumerate(self.worker_stats):
                if stats["completed_tasks"] > 0:
                    avg_time = stats["avg_time"]
                    if avg_time < best_time:
                        best_time = avg_time
                        best_worker = i
            
            return best_worker
        
        return 0  # Default to first worker
    
    def get_load_stats(self) -> Dict[str, Any]:
        """Get current load balancing statistics."""
        total_active = sum(stats["active_tasks"] for stats in self.worker_stats)
        total_completed = sum(stats["completed_tasks"] for stats in self.worker_stats)
        
        return {
            "total_active_tasks": total_active,
            "total_completed_tasks": total_completed,
            "worker_count": len(self.worker_queues),
            "avg_tasks_per_worker": total_active / len(self.worker_queues),
            "worker_stats": self.worker_stats.copy(),
            "strategy": self.load_balancing_strategy
        }


@dataclass
class TaskWrapper:
    """Wrapper for tasks in the workload balancer."""
    task_id: str
    func: Callable
    args: tuple
    kwargs: dict
    worker_id: int
    result_future: asyncio.Future = field(default_factory=asyncio.Future)
    start_time: Optional[float] = None


class PerformanceOptimizationEngine:
    """Main performance optimization engine."""
    
    def __init__(self):
        self.resource_manager = AdaptiveResourceManager()
        self.workload_balancer = IntelligentWorkloadBalancer()
        self.running = False
        self.optimization_task = None
        self.metrics_collector = PerformanceMetricsCollector()
        
        # Optimization intervals
        self.optimization_interval = 30  # seconds
        self.metrics_interval = 5  # seconds
    
    async def start_optimization(self):
        """Start the performance optimization engine."""
        if self.running:
            logger.warning("Performance optimization already running")
            return
        
        self.running = True
        
        # Start metrics collection
        asyncio.create_task(self.metrics_collector.start_collection())
        
        # Start optimization loop
        self.optimization_task = asyncio.create_task(self._optimization_loop())
        
        logger.info("Performance optimization engine started")
    
    async def stop_optimization(self):
        """Stop the performance optimization engine."""
        self.running = False
        
        if self.optimization_task:
            self.optimization_task.cancel()
            try:
                await self.optimization_task
            except asyncio.CancelledError:
                pass
        
        await self.metrics_collector.stop_collection()
        logger.info("Performance optimization engine stopped")
    
    async def _optimization_loop(self):
        """Main optimization loop."""
        while self.running:
            try:
                # Get latest metrics
                metrics = await self.metrics_collector.get_latest_metrics()
                
                if metrics:
                    # Get optimization recommendations
                    actions = await self.resource_manager.optimize_resources(metrics)
                    
                    # Apply high-priority optimizations
                    for action in actions[:3]:  # Apply top 3 actions
                        if action.priority <= 2:  # High priority
                            success = await self.resource_manager.apply_optimization(action)
                            if success:
                                logger.info(f"Applied optimization: {action.action_type}")
                                # Give some time for the optimization to take effect
                                await asyncio.sleep(5)
                
                await asyncio.sleep(self.optimization_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in optimization loop: {e}")
                await asyncio.sleep(5)
    
    async def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        metrics = await self.metrics_collector.get_latest_metrics()
        resource_stats = self.resource_manager.get_optimization_stats()
        load_stats = self.workload_balancer.get_load_stats()
        
        return {
            "timestamp": time.time(),
            "current_metrics": metrics.__dict__ if metrics else None,
            "resource_optimization": resource_stats,
            "load_balancing": load_stats,
            "engine_status": {
                "running": self.running,
                "optimization_interval": self.optimization_interval,
                "metrics_interval": self.metrics_interval
            }
        }


class PerformanceMetricsCollector:
    """Collects system performance metrics for optimization."""
    
    def __init__(self):
        self.latest_metrics = None
        self.metrics_history = deque(maxlen=100)
        self.collection_task = None
        self.running = False
    
    async def start_collection(self):
        """Start metrics collection."""
        if self.running:
            return
        
        self.running = True
        self.collection_task = asyncio.create_task(self._collection_loop())
        logger.info("Performance metrics collection started")
    
    async def stop_collection(self):
        """Stop metrics collection."""
        self.running = False
        
        if self.collection_task:
            self.collection_task.cancel()
            try:
                await self.collection_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Performance metrics collection stopped")
    
    async def _collection_loop(self):
        """Main metrics collection loop."""
        while self.running:
            try:
                metrics = await self._collect_metrics()
                self.latest_metrics = metrics
                self.metrics_history.append(metrics)
                
                await asyncio.sleep(5)  # Collect every 5 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error collecting metrics: {e}")
                await asyncio.sleep(1)
    
    async def _collect_metrics(self) -> PerformanceMetrics:
        """Collect current system metrics."""
        
        # Basic system metrics
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        disk_io = psutil.disk_io_counters()
        network_io = psutil.net_io_counters()
        
        # Calculate rates (if we have previous data)
        disk_io_rate = 0.0
        network_io_rate = 0.0
        
        if hasattr(self, '_last_disk_io') and self._last_disk_io:
            disk_io_rate = ((disk_io.read_bytes + disk_io.write_bytes) - 
                           (self._last_disk_io.read_bytes + self._last_disk_io.write_bytes))
        
        if hasattr(self, '_last_network_io') and self._last_network_io:
            network_io_rate = ((network_io.bytes_sent + network_io.bytes_recv) - 
                              (self._last_network_io.bytes_sent + self._last_network_io.bytes_recv))
        
        self._last_disk_io = disk_io
        self._last_network_io = network_io
        
        # Mock response time and throughput (would come from application metrics)
        response_time = 1.0  # seconds
        throughput = 10.0    # requests per second
        error_rate = 0.05    # 5% error rate
        
        return PerformanceMetrics(
            timestamp=time.time(),
            cpu_usage=cpu_percent,
            memory_usage=memory.percent,
            disk_io=disk_io_rate,
            network_io=network_io_rate,
            response_time=response_time,
            throughput=throughput,
            error_rate=error_rate,
            active_connections=len(psutil.net_connections())
        )
    
    async def get_latest_metrics(self) -> Optional[PerformanceMetrics]:
        """Get the latest collected metrics."""
        return self.latest_metrics
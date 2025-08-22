"""
Distributed quantum coordination system for large-scale HVAC optimization.
Manages quantum resources across multiple solvers and coordinates problem distribution.
"""

import asyncio
import time
import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import deque, defaultdict
import hashlib

logger = logging.getLogger(__name__)


class ResourceType(Enum):
    """Quantum resource types."""
    QPU_ADVANTAGE = "qpu_advantage"
    QPU_ADVANTAGE2 = "qpu_advantage2"
    HYBRID_V1 = "hybrid_v1"
    HYBRID_V2 = "hybrid_v2"
    CLASSICAL = "classical"


class TaskPriority(Enum):
    """Task priority levels."""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    BACKGROUND = 5


@dataclass
class QuantumResource:
    """Quantum computing resource."""
    resource_id: str
    resource_type: ResourceType
    available: bool = True
    queue_length: int = 0
    avg_response_time: float = 1.0
    success_rate: float = 1.0
    max_problem_size: int = 5000
    cost_per_use: float = 0.0
    last_used: float = 0.0
    

@dataclass
class OptimizationTask:
    """Optimization task for distributed processing."""
    task_id: str
    problem_data: Dict[str, Any]
    priority: TaskPriority
    created_time: float
    deadline: Optional[float] = None
    estimated_runtime: float = 1.0
    required_resources: Set[ResourceType] = field(default_factory=set)
    callback: Optional[callable] = None
    

@dataclass
class TaskResult:
    """Result of distributed optimization task."""
    task_id: str
    success: bool
    result: Any
    execution_time: float
    resource_used: str
    error_message: Optional[str] = None
    

class DistributedQuantumCoordinator:
    """Coordinates distributed quantum optimization across multiple resources."""
    
    def __init__(self, max_concurrent_tasks: int = 10):
        self.max_concurrent_tasks = max_concurrent_tasks
        self.resources: Dict[str, QuantumResource] = {}
        self.task_queue = asyncio.Queue()
        self.active_tasks: Dict[str, OptimizationTask] = {}
        self.completed_tasks = deque(maxlen=1000)
        self.resource_performance = defaultdict(list)
        
        self.coordinator_active = False
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent_tasks)
        
        # Performance tracking
        self.throughput_history = deque(maxlen=100)
        self.load_balancer_weights = {}
        
        self._init_default_resources()
    
    def _init_default_resources(self):
        """Initialize default quantum resources."""
        # QPU Resources
        self.resources["advantage_system4.1"] = QuantumResource(
            resource_id="advantage_system4.1",
            resource_type=ResourceType.QPU_ADVANTAGE,
            max_problem_size=5000,
            cost_per_use=0.1,
            avg_response_time=5.0
        )
        
        self.resources["advantage2_prototype"] = QuantumResource(
            resource_id="advantage2_prototype", 
            resource_type=ResourceType.QPU_ADVANTAGE2,
            max_problem_size=7000,
            cost_per_use=0.15,
            avg_response_time=4.0
        )
        
        # Hybrid Resources
        self.resources["hybrid_v2"] = QuantumResource(
            resource_id="hybrid_v2",
            resource_type=ResourceType.HYBRID_V2,
            max_problem_size=100000,
            cost_per_use=0.05,
            avg_response_time=10.0
        )
        
        # Classical Fallback
        self.resources["classical_fallback"] = QuantumResource(
            resource_id="classical_fallback",
            resource_type=ResourceType.CLASSICAL,
            max_problem_size=1000000,
            cost_per_use=0.01,
            avg_response_time=2.0
        )
        
        # Initialize load balancer weights
        for resource_id in self.resources:
            self.load_balancer_weights[resource_id] = 1.0
    
    async def start_coordinator(self):
        """Start the distributed coordinator."""
        if self.coordinator_active:
            return
        
        self.coordinator_active = True
        
        # Start task processing loop
        self.coordination_task = asyncio.create_task(self._coordination_loop())
        
        # Start performance monitoring
        self.monitoring_task = asyncio.create_task(self._performance_monitoring_loop())
        
        logger.info("Distributed quantum coordinator started")
    
    async def stop_coordinator(self):
        """Stop the distributed coordinator."""
        self.coordinator_active = False
        
        if hasattr(self, 'coordination_task'):
            self.coordination_task.cancel()
            
        if hasattr(self, 'monitoring_task'):
            self.monitoring_task.cancel()
        
        self.executor.shutdown(wait=True)
        logger.info("Distributed quantum coordinator stopped")
    
    async def _coordination_loop(self):
        """Main coordination loop."""
        while self.coordinator_active:
            try:
                # Process pending tasks
                await self._process_task_queue()
                
                # Update resource availability
                self._update_resource_status()
                
                # Optimize load distribution
                self._optimize_load_balancing()
                
                await asyncio.sleep(1.0)
                
            except Exception as e:
                logger.error(f"Coordination loop error: {e}")
                await asyncio.sleep(5.0)
    
    async def _performance_monitoring_loop(self):
        """Performance monitoring loop."""
        while self.coordinator_active:
            try:
                # Calculate throughput
                current_throughput = len(self.completed_tasks) / max(time.time() - 
                    (self.completed_tasks[0].execution_time if self.completed_tasks else time.time()), 1.0)
                
                self.throughput_history.append({
                    'timestamp': time.time(),
                    'throughput': current_throughput,
                    'active_tasks': len(self.active_tasks),
                    'queue_size': self.task_queue.qsize()
                })
                
                # Update resource performance metrics
                self._update_resource_performance()
                
                await asyncio.sleep(30.0)  # Update every 30 seconds
                
            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")
                await asyncio.sleep(30.0)
    
    async def submit_task(self, problem_data: Dict[str, Any], 
                         priority: TaskPriority = TaskPriority.NORMAL,
                         deadline: Optional[float] = None,
                         callback: Optional[callable] = None) -> str:
        """
        Submit optimization task for distributed processing.
        
        Args:
            problem_data: Optimization problem data
            priority: Task priority level
            deadline: Optional deadline (timestamp)
            callback: Optional callback for completion
            
        Returns:
            Task ID for tracking
        """
        # Generate task ID
        task_id = hashlib.md5(f"{time.time()}_{id(problem_data)}".encode()).hexdigest()[:16]
        
        # Estimate required resources
        problem_size = self._estimate_problem_size(problem_data)
        required_resources = self._determine_required_resources(problem_size, priority)
        
        # Create task
        task = OptimizationTask(
            task_id=task_id,
            problem_data=problem_data.copy(),
            priority=priority,
            created_time=time.time(),
            deadline=deadline,
            estimated_runtime=self._estimate_runtime(problem_size, required_resources),
            required_resources=required_resources,
            callback=callback
        )
        
        # Add to queue
        await self.task_queue.put(task)
        
        logger.info(f"Task {task_id} submitted with priority {priority.name}")
        return task_id
    
    async def _process_task_queue(self):
        """Process tasks from the queue."""
        if self.task_queue.empty() or len(self.active_tasks) >= self.max_concurrent_tasks:
            return
        
        try:
            # Get next task (with timeout to avoid blocking)
            task = await asyncio.wait_for(self.task_queue.get(), timeout=0.1)
            
            # Find best resource for task
            best_resource = self._select_optimal_resource(task)
            
            if best_resource:
                # Mark resource as busy
                self.resources[best_resource].available = False
                self.resources[best_resource].queue_length += 1
                
                # Add to active tasks
                self.active_tasks[task.task_id] = task
                
                # Submit for execution
                future = self.executor.submit(self._execute_task, task, best_resource)
                
                # Handle completion asynchronously
                asyncio.create_task(self._handle_task_completion(future, task, best_resource))
                
                logger.debug(f"Task {task.task_id} assigned to {best_resource}")
            else:
                # No resources available, put back in queue
                await self.task_queue.put(task)
                
        except asyncio.TimeoutError:
            # No tasks in queue
            pass
        except Exception as e:
            logger.error(f"Task processing error: {e}")
    
    def _select_optimal_resource(self, task: OptimizationTask) -> Optional[str]:
        """Select optimal resource for task using intelligent scheduling."""
        available_resources = [
            (resource_id, resource) for resource_id, resource in self.resources.items()
            if resource.available and resource.resource_type in task.required_resources
        ]
        
        if not available_resources:
            return None
        
        # Multi-criteria optimization for resource selection
        best_resource = None
        best_score = float('-inf')
        
        for resource_id, resource in available_resources:
            # Calculate scoring factors
            
            # Performance factor (higher success rate and lower response time is better)
            performance_score = (resource.success_rate / max(resource.avg_response_time, 0.1)) * 100
            
            # Load factor (lower queue length is better)
            load_score = 1.0 / (resource.queue_length + 1)
            
            # Cost factor (lower cost is better for non-critical tasks)
            cost_factor = 1.0 if task.priority == TaskPriority.CRITICAL else (1.0 / (resource.cost_per_use + 0.01))
            
            # Deadline factor
            deadline_factor = 1.0
            if task.deadline:
                time_left = task.deadline - time.time()
                if time_left > 0:
                    deadline_factor = 1.0 / max(resource.avg_response_time / time_left, 0.1)
                else:
                    deadline_factor = 0.0  # Past deadline
            
            # Load balancer weight
            lb_weight = self.load_balancer_weights.get(resource_id, 1.0)
            
            # Combined score
            total_score = (performance_score * 0.4 + 
                          load_score * 0.3 + 
                          cost_factor * 0.1 + 
                          deadline_factor * 0.1 +
                          lb_weight * 0.1)
            
            if total_score > best_score:
                best_score = total_score
                best_resource = resource_id
        
        return best_resource
    
    def _execute_task(self, task: OptimizationTask, resource_id: str) -> TaskResult:
        """Execute optimization task on selected resource."""
        start_time = time.time()
        
        try:
            # Simulate quantum optimization execution
            # In real implementation, this would call the actual quantum solver
            
            resource = self.resources[resource_id]
            
            # Add realistic delay based on resource type
            execution_delay = max(resource.avg_response_time + np.random.normal(0, 0.5), 0.1)
            time.sleep(execution_delay)
            
            # Simulate success/failure based on resource success rate
            success = np.random.random() < resource.success_rate
            
            if success:
                # Generate mock result
                problem_size = self._estimate_problem_size(task.problem_data)
                result = {
                    'solution': np.random.rand(problem_size).tolist(),
                    'energy': np.random.random() * 100,
                    'num_reads': 1000,
                    'resource_used': resource_id
                }
                error_message = None
            else:
                result = None
                error_message = f"Optimization failed on {resource_id}"
            
            execution_time = time.time() - start_time
            
            return TaskResult(
                task_id=task.task_id,
                success=success,
                result=result,
                execution_time=execution_time,
                resource_used=resource_id,
                error_message=error_message
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return TaskResult(
                task_id=task.task_id,
                success=False,
                result=None,
                execution_time=execution_time,
                resource_used=resource_id,
                error_message=str(e)
            )
    
    async def _handle_task_completion(self, future, task: OptimizationTask, resource_id: str):
        """Handle task completion."""
        try:
            result = future.result()
            
            # Update resource status
            resource = self.resources[resource_id]
            resource.available = True
            resource.queue_length = max(resource.queue_length - 1, 0)
            resource.last_used = time.time()
            
            # Update resource performance
            if result.success:
                resource.success_rate = resource.success_rate * 0.95 + 0.05
            else:
                resource.success_rate = resource.success_rate * 0.95
            
            # Update response time
            resource.avg_response_time = resource.avg_response_time * 0.9 + result.execution_time * 0.1
            
            # Remove from active tasks
            if task.task_id in self.active_tasks:
                del self.active_tasks[task.task_id]
            
            # Add to completed tasks
            self.completed_tasks.append(result)
            
            # Call callback if provided
            if task.callback:
                try:
                    await task.callback(result)
                except Exception as e:
                    logger.error(f"Task callback error: {e}")
            
            # Record performance metrics
            self.resource_performance[resource_id].append({
                'timestamp': time.time(),
                'execution_time': result.execution_time,
                'success': result.success
            })
            
            logger.info(f"Task {task.task_id} completed on {resource_id} "
                       f"(success: {result.success}, time: {result.execution_time:.2f}s)")
            
        except Exception as e:
            logger.error(f"Task completion handling error: {e}")
            
            # Ensure resource is marked available
            if resource_id in self.resources:
                self.resources[resource_id].available = True
                self.resources[resource_id].queue_length = max(
                    self.resources[resource_id].queue_length - 1, 0)
    
    def _estimate_problem_size(self, problem_data: Dict[str, Any]) -> int:
        """Estimate problem size from problem data."""
        # Simple heuristic based on problem data
        if 'Q' in problem_data:
            return len(problem_data['Q'])
        elif 'horizon' in problem_data and 'zones' in problem_data:
            return problem_data['horizon'] * problem_data['zones'] * 2
        else:
            return 100  # Default size
    
    def _determine_required_resources(self, problem_size: int, 
                                    priority: TaskPriority) -> Set[ResourceType]:
        """Determine which resource types can handle the problem."""
        required = set()
        
        # Always include classical fallback
        required.add(ResourceType.CLASSICAL)
        
        # Add quantum resources based on problem size and priority
        if problem_size <= 5000:
            required.add(ResourceType.QPU_ADVANTAGE)
            required.add(ResourceType.HYBRID_V1)
            required.add(ResourceType.HYBRID_V2)
        
        if problem_size <= 7000 and priority in [TaskPriority.CRITICAL, TaskPriority.HIGH]:
            required.add(ResourceType.QPU_ADVANTAGE2)
        
        if problem_size <= 100000:
            required.add(ResourceType.HYBRID_V2)
        
        return required
    
    def _estimate_runtime(self, problem_size: int, required_resources: Set[ResourceType]) -> float:
        """Estimate task runtime."""
        # Base time estimation
        base_time = max(problem_size / 1000, 0.1)
        
        # Adjust based on available resource types
        if ResourceType.QPU_ADVANTAGE in required_resources:
            return base_time * 0.5  # Quantum speedup
        elif ResourceType.HYBRID_V2 in required_resources:
            return base_time * 0.7  # Hybrid efficiency
        else:
            return base_time  # Classical time
    
    def _update_resource_status(self):
        """Update resource availability status."""
        for resource_id, resource in self.resources.items():
            # Simulate resource availability updates
            # In real implementation, this would query actual D-Wave status
            
            # Occasionally mark resources as unavailable (maintenance, etc.)
            if np.random.random() < 0.01:  # 1% chance per update
                resource.available = False
                resource.queue_length = max(resource.queue_length + 5, 0)
            elif not resource.available and np.random.random() < 0.1:  # 10% chance to recover
                resource.available = True
                resource.queue_length = max(resource.queue_length - 5, 0)
    
    def _optimize_load_balancing(self):
        """Optimize load balancing weights based on performance."""
        for resource_id, performance_history in self.resource_performance.items():
            if len(performance_history) < 5:
                continue
            
            # Calculate recent performance metrics
            recent_performance = performance_history[-10:]
            avg_time = np.mean([p['execution_time'] for p in recent_performance])
            success_rate = np.mean([p['success'] for p in recent_performance])
            
            # Update load balancer weight
            # Higher weight for better performing resources
            performance_score = success_rate / max(avg_time, 0.1)
            self.load_balancer_weights[resource_id] = max(performance_score / 10, 0.1)
    
    def _update_resource_performance(self):
        """Update resource performance metrics."""
        for resource_id, resource in self.resources.items():
            # Clean old performance data
            cutoff_time = time.time() - 3600  # Keep 1 hour of data
            
            if resource_id in self.resource_performance:
                self.resource_performance[resource_id] = [
                    p for p in self.resource_performance[resource_id]
                    if p['timestamp'] > cutoff_time
                ]
    
    def get_coordinator_status(self) -> Dict[str, Any]:
        """Get coordinator status and metrics."""
        status = {
            'active': self.coordinator_active,
            'active_tasks': len(self.active_tasks),
            'queue_size': self.task_queue.qsize(),
            'completed_tasks': len(self.completed_tasks),
            'resources': {},
            'throughput': {},
            'load_balancing': self.load_balancer_weights.copy()
        }
        
        # Resource status
        for resource_id, resource in self.resources.items():
            status['resources'][resource_id] = {
                'type': resource.resource_type.value,
                'available': resource.available,
                'queue_length': resource.queue_length,
                'avg_response_time': resource.avg_response_time,
                'success_rate': resource.success_rate,
                'last_used': resource.last_used
            }
        
        # Throughput metrics
        if self.throughput_history:
            recent_throughput = self.throughput_history[-1]
            status['throughput'] = {
                'current': recent_throughput['throughput'],
                'active_tasks': recent_throughput['active_tasks'],
                'queue_size': recent_throughput['queue_size']
            }
        
        return status
    
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of specific task."""
        # Check active tasks
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            return {
                'status': 'running',
                'priority': task.priority.name,
                'created_time': task.created_time,
                'estimated_runtime': task.estimated_runtime
            }
        
        # Check completed tasks
        for result in self.completed_tasks:
            if result.task_id == task_id:
                return {
                    'status': 'completed',
                    'success': result.success,
                    'execution_time': result.execution_time,
                    'resource_used': result.resource_used
                }
        
        return None  # Task not found


# Global coordinator instance
_global_coordinator = None

async def get_quantum_coordinator() -> DistributedQuantumCoordinator:
    """Get global quantum coordinator instance."""
    global _global_coordinator
    if _global_coordinator is None:
        _global_coordinator = DistributedQuantumCoordinator()
        await _global_coordinator.start_coordinator()
    return _global_coordinator


__all__ = [
    'DistributedQuantumCoordinator',
    'ResourceType',
    'TaskPriority', 
    'QuantumResource',
    'OptimizationTask',
    'TaskResult',
    'get_quantum_coordinator'
]
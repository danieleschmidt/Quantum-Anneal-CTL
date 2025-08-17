"""
Distributed Quantum Cluster Management System.

High-performance distributed system for scaling quantum HVAC optimization
across multiple quantum processors, geographical regions, and building clusters.

Features:
1. Multi-QPU cluster orchestration with load balancing
2. Geographical distribution with latency optimization
3. Fault-tolerant quantum task distribution
4. Dynamic resource allocation and auto-scaling
5. Global optimization coordination across clusters
"""

from typing import Dict, Any, List, Optional, Tuple, Set, Callable, Union
import asyncio
import time
import logging
import json
import hashlib
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import numpy as np
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
import aiohttp
import ssl

try:
    import kubernetes
    from kubernetes import client, config
    KUBERNETES_AVAILABLE = True
except ImportError:
    KUBERNETES_AVAILABLE = False


class ClusterRole(Enum):
    """Roles in the distributed quantum cluster."""
    COORDINATOR = "coordinator"
    WORKER = "worker"
    GATEWAY = "gateway"
    MONITOR = "monitor"


class QuantumResourceType(Enum):
    """Types of quantum computing resources."""
    QPU_ADVANTAGE = "qpu_advantage"
    QPU_2000Q = "qpu_2000q"
    HYBRID_SOLVER = "hybrid_solver"
    CLASSICAL_SIMULATOR = "classical_simulator"
    GPU_SIMULATOR = "gpu_simulator"


class TaskPriority(Enum):
    """Task priority levels for distributed scheduling."""
    EMERGENCY = 5
    HIGH = 4
    NORMAL = 3
    LOW = 2
    BACKGROUND = 1


@dataclass
class ClusterNode:
    """Information about a cluster node."""
    node_id: str
    role: ClusterRole
    location: str  # Geographic location or data center
    endpoint: str
    quantum_resources: List[QuantumResourceType]
    capacity: int  # Max concurrent tasks
    current_load: int = 0
    
    # Performance metrics
    avg_response_time: float = 0.0
    success_rate: float = 1.0
    uptime: float = 1.0
    
    # Network metrics
    latency_to_coordinator: float = 0.0
    bandwidth_mbps: float = 1000.0
    
    last_heartbeat: datetime = field(default_factory=datetime.now)
    
    @property
    def load_factor(self) -> float:
        """Current load as fraction of capacity."""
        return self.current_load / max(self.capacity, 1)
    
    @property
    def efficiency_score(self) -> float:
        """Overall efficiency score for load balancing."""
        load_penalty = 1.0 - self.load_factor
        performance_bonus = self.success_rate * (1.0 / max(self.avg_response_time, 0.1))
        uptime_bonus = self.uptime
        
        return load_penalty * performance_bonus * uptime_bonus
    
    @property
    def is_healthy(self) -> bool:
        """Check if node is healthy."""
        age_minutes = (datetime.now() - self.last_heartbeat).total_seconds() / 60
        return age_minutes < 5.0 and self.uptime > 0.8


@dataclass
class DistributedQuantumTask:
    """Task for distributed quantum processing."""
    task_id: str
    building_cluster_id: str
    qubo_problem: Dict[Tuple[int, int], float]
    priority: TaskPriority
    deadline: datetime
    estimated_complexity: float  # 0-1 scale
    
    # Resource requirements
    required_qubits: int
    preferred_solver_type: QuantumResourceType
    allow_classical_fallback: bool = True
    
    # Distribution metadata
    can_be_decomposed: bool = True
    decomposition_strategy: Optional[str] = None
    geographic_constraints: List[str] = field(default_factory=list)
    
    # Execution tracking
    assigned_node: Optional[str] = None
    started_at: Optional[datetime] = None
    attempts: int = 0
    max_attempts: int = 3
    
    created_at: datetime = field(default_factory=datetime.now)
    
    @property
    def urgency_score(self) -> float:
        """Calculate urgency score for scheduling."""
        time_factor = max(0.1, (self.deadline - datetime.now()).total_seconds() / 3600)
        priority_weight = self.priority.value
        complexity_factor = 1.0 + self.estimated_complexity
        
        return (priority_weight * complexity_factor) / time_factor


@dataclass
class ClusterPerformanceMetrics:
    """Performance metrics for the entire cluster."""
    total_nodes: int
    active_nodes: int
    total_capacity: int
    current_utilization: float
    
    # Task metrics
    tasks_completed: int
    tasks_failed: int
    avg_task_time: float
    
    # Geographical distribution
    regions: Dict[str, int]  # region -> node count
    
    # Resource utilization
    qpu_utilization: float
    hybrid_utilization: float
    classical_utilization: float
    
    # Network metrics
    avg_inter_node_latency: float
    total_bandwidth_gbps: float


class QuantumTaskDecomposer:
    """Decomposes large quantum tasks for parallel execution."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    async def decompose_task(
        self, 
        task: DistributedQuantumTask,
        target_subtask_size: int = 1000
    ) -> List[DistributedQuantumTask]:
        """Decompose large task into smaller parallel subtasks."""
        if not task.can_be_decomposed or len(task.qubo_problem) <= target_subtask_size:
            return [task]
        
        try:
            if task.decomposition_strategy == "temporal":
                return await self._temporal_decomposition(task, target_subtask_size)
            elif task.decomposition_strategy == "spatial":
                return await self._spatial_decomposition(task, target_subtask_size)
            else:
                return await self._graph_decomposition(task, target_subtask_size)
        
        except Exception as e:
            self.logger.error(f"Task decomposition failed: {e}")
            return [task]  # Return original task if decomposition fails
    
    async def _temporal_decomposition(
        self, 
        task: DistributedQuantumTask, 
        target_size: int
    ) -> List[DistributedQuantumTask]:
        """Decompose by time horizon (for MPC problems)."""
        # Assume QUBO represents time-series optimization
        variables = set()
        for (i, j) in task.qubo_problem.keys():
            variables.add(i)
            variables.add(j)
        
        total_vars = len(variables)
        time_steps = int(np.sqrt(total_vars))  # Assume square time grid
        
        if time_steps <= 2:
            return [task]
        
        # Split into overlapping time windows
        subtasks = []
        window_size = max(2, time_steps // 3)
        overlap = window_size // 4
        
        for start_step in range(0, time_steps - window_size + 1, window_size - overlap):
            end_step = min(start_step + window_size, time_steps)
            
            # Extract QUBO subset for this time window
            subtask_qubo = {}
            var_mapping = {}
            new_var_idx = 0
            
            for (i, j), coeff in task.qubo_problem.items():
                i_step = i // (total_vars // time_steps)
                j_step = j // (total_vars // time_steps)
                
                if start_step <= i_step < end_step and start_step <= j_step < end_step:
                    if i not in var_mapping:
                        var_mapping[i] = new_var_idx
                        new_var_idx += 1
                    if j not in var_mapping:
                        var_mapping[j] = new_var_idx
                        new_var_idx += 1
                    
                    subtask_qubo[(var_mapping[i], var_mapping[j])] = coeff
            
            if subtask_qubo:
                subtask = DistributedQuantumTask(
                    task_id=f"{task.task_id}_t{start_step}",
                    building_cluster_id=task.building_cluster_id,
                    qubo_problem=subtask_qubo,
                    priority=task.priority,
                    deadline=task.deadline,
                    estimated_complexity=task.estimated_complexity * len(subtask_qubo) / len(task.qubo_problem),
                    required_qubits=len(var_mapping),
                    preferred_solver_type=task.preferred_solver_type,
                    can_be_decomposed=False
                )
                subtasks.append(subtask)
        
        return subtasks
    
    async def _spatial_decomposition(
        self, 
        task: DistributedQuantumTask, 
        target_size: int
    ) -> List[DistributedQuantumTask]:
        """Decompose by spatial zones (for building optimization)."""
        # Simple spatial decomposition based on variable grouping
        variables = list(set(var for (i, j) in task.qubo_problem.keys() for var in [i, j]))
        
        # Group variables into spatial clusters
        num_clusters = max(2, len(variables) // target_size)
        cluster_size = len(variables) // num_clusters
        
        subtasks = []
        for cluster_idx in range(num_clusters):
            start_var = cluster_idx * cluster_size
            end_var = min((cluster_idx + 1) * cluster_size, len(variables))
            cluster_vars = set(variables[start_var:end_var])
            
            # Add overlap with adjacent clusters
            if cluster_idx > 0:
                overlap_start = max(0, start_var - cluster_size // 4)
                cluster_vars.update(variables[overlap_start:start_var])
            
            if cluster_idx < num_clusters - 1:
                overlap_end = min(len(variables), end_var + cluster_size // 4)
                cluster_vars.update(variables[end_var:overlap_end])
            
            # Extract QUBO subset
            subtask_qubo = {}
            var_mapping = {var: idx for idx, var in enumerate(sorted(cluster_vars))}
            
            for (i, j), coeff in task.qubo_problem.items():
                if i in cluster_vars and j in cluster_vars:
                    subtask_qubo[(var_mapping[i], var_mapping[j])] = coeff
            
            if subtask_qubo:
                subtask = DistributedQuantumTask(
                    task_id=f"{task.task_id}_s{cluster_idx}",
                    building_cluster_id=task.building_cluster_id,
                    qubo_problem=subtask_qubo,
                    priority=task.priority,
                    deadline=task.deadline,
                    estimated_complexity=task.estimated_complexity * len(subtask_qubo) / len(task.qubo_problem),
                    required_qubits=len(var_mapping),
                    preferred_solver_type=task.preferred_solver_type,
                    can_be_decomposed=False
                )
                subtasks.append(subtask)
        
        return subtasks
    
    async def _graph_decomposition(
        self, 
        task: DistributedQuantumTask, 
        target_size: int
    ) -> List[DistributedQuantumTask]:
        """Decompose based on graph structure of QUBO."""
        # Use graph partitioning for general QUBO decomposition
        # Simplified implementation - in practice would use advanced graph algorithms
        
        variables = list(set(var for (i, j) in task.qubo_problem.keys() for var in [i, j]))
        
        if len(variables) <= target_size:
            return [task]
        
        # Simple random partitioning (would use better algorithms in practice)
        num_partitions = max(2, len(variables) // target_size)
        partition_size = len(variables) // num_partitions
        
        subtasks = []
        for p in range(num_partitions):
            start_idx = p * partition_size
            end_idx = min((p + 1) * partition_size, len(variables))
            partition_vars = set(variables[start_idx:end_idx])
            
            # Extract connected components
            subtask_qubo = {}
            var_mapping = {var: idx for idx, var in enumerate(sorted(partition_vars))}
            
            for (i, j), coeff in task.qubo_problem.items():
                if i in partition_vars and j in partition_vars:
                    subtask_qubo[(var_mapping[i], var_mapping[j])] = coeff
            
            if subtask_qubo:
                subtask = DistributedQuantumTask(
                    task_id=f"{task.task_id}_g{p}",
                    building_cluster_id=task.building_cluster_id,
                    qubo_problem=subtask_qubo,
                    priority=task.priority,
                    deadline=task.deadline,
                    estimated_complexity=task.estimated_complexity * len(subtask_qubo) / len(task.qubo_problem),
                    required_qubits=len(var_mapping),
                    preferred_solver_type=task.preferred_solver_type,
                    can_be_decomposed=False
                )
                subtasks.append(subtask)
        
        return subtasks


class DistributedLoadBalancer:
    """Intelligent load balancer for distributed quantum tasks."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.load_history: deque = deque(maxlen=1000)
        self.response_time_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
    
    async def select_optimal_node(
        self,
        task: DistributedQuantumTask,
        available_nodes: List[ClusterNode]
    ) -> Optional[ClusterNode]:
        """Select optimal node for task execution."""
        if not available_nodes:
            return None
        
        # Filter nodes by resource requirements
        suitable_nodes = []
        
        for node in available_nodes:
            if not node.is_healthy:
                continue
            
            # Check resource compatibility
            if task.preferred_solver_type not in node.quantum_resources:
                if not task.allow_classical_fallback:
                    continue
                if QuantumResourceType.CLASSICAL_SIMULATOR not in node.quantum_resources:
                    continue
            
            # Check capacity
            if node.current_load >= node.capacity:
                continue
            
            # Check geographic constraints
            if task.geographic_constraints and node.location not in task.geographic_constraints:
                continue
            
            suitable_nodes.append(node)
        
        if not suitable_nodes:
            return None
        
        # Score nodes based on multiple factors
        scored_nodes = []
        
        for node in suitable_nodes:
            score = await self._calculate_node_score(task, node)
            scored_nodes.append((score, node))
        
        # Select highest scoring node
        scored_nodes.sort(key=lambda x: x[0], reverse=True)
        return scored_nodes[0][1]
    
    async def _calculate_node_score(self, task: DistributedQuantumTask, node: ClusterNode) -> float:
        """Calculate comprehensive node suitability score."""
        # Base efficiency score
        efficiency_score = node.efficiency_score
        
        # Resource preference bonus
        resource_bonus = 1.0
        if task.preferred_solver_type in node.quantum_resources:
            resource_bonus = 1.2
        
        # Priority handling bonus
        priority_bonus = 1.0
        if task.priority in [TaskPriority.EMERGENCY, TaskPriority.HIGH]:
            # Prefer less loaded nodes for high priority tasks
            priority_bonus = 1.0 + (1.0 - node.load_factor) * 0.5
        
        # Latency penalty for time-sensitive tasks
        latency_penalty = 1.0
        if task.priority == TaskPriority.EMERGENCY:
            latency_penalty = 1.0 / (1.0 + node.latency_to_coordinator * 0.1)
        
        # Historical performance bonus
        history_bonus = 1.0
        if node.node_id in self.response_time_history:
            recent_times = list(self.response_time_history[node.node_id])
            if recent_times:
                avg_time = sum(recent_times) / len(recent_times)
                history_bonus = 1.0 / (1.0 + avg_time * 0.01)
        
        # Combine all factors
        total_score = (
            efficiency_score * resource_bonus * priority_bonus * 
            latency_penalty * history_bonus
        )
        
        return total_score
    
    def record_execution_result(
        self,
        node_id: str,
        task: DistributedQuantumTask,
        execution_time: float,
        success: bool
    ) -> None:
        """Record execution result for learning."""
        self.response_time_history[node_id].append(execution_time)
        
        self.load_history.append({
            'timestamp': datetime.now(),
            'node_id': node_id,
            'task_priority': task.priority.value,
            'execution_time': execution_time,
            'success': success
        })


class DistributedQuantumCluster:
    """
    Main distributed quantum cluster management system.
    
    Coordinates quantum task execution across multiple nodes,
    handles fault tolerance, and optimizes global performance.
    """
    
    def __init__(self, node_id: str, role: ClusterRole = ClusterRole.COORDINATOR):
        self.node_id = node_id
        self.role = role
        self.is_coordinator = (role == ClusterRole.COORDINATOR)
        
        self.cluster_nodes: Dict[str, ClusterNode] = {}
        self.task_queue: List[DistributedQuantumTask] = []
        self.active_tasks: Dict[str, DistributedQuantumTask] = {}
        self.completed_tasks: deque = deque(maxlen=1000)
        
        self.decomposer = QuantumTaskDecomposer()
        self.load_balancer = DistributedLoadBalancer()
        
        self.logger = logging.getLogger(f"{__name__}.{node_id}")
        self._shutdown_event = asyncio.Event()
        
        # Performance tracking
        self.cluster_metrics = ClusterPerformanceMetrics(
            total_nodes=0, active_nodes=0, total_capacity=0,
            current_utilization=0.0, tasks_completed=0, tasks_failed=0,
            avg_task_time=0.0, regions={}, qpu_utilization=0.0,
            hybrid_utilization=0.0, classical_utilization=0.0,
            avg_inter_node_latency=0.0, total_bandwidth_gbps=0.0
        )
        
        # Network communication
        self.http_session: Optional[aiohttp.ClientSession] = None
    
    async def start(self) -> None:
        """Start the distributed cluster node."""
        self.logger.info(f"Starting cluster node {self.node_id} as {self.role.value}")
        
        # Initialize HTTP session
        connector = aiohttp.TCPConnector(
            limit=100,
            ssl=ssl.create_default_context()
        )
        self.http_session = aiohttp.ClientSession(connector=connector)
        
        # Start role-specific services
        if self.role == ClusterRole.COORDINATOR:
            asyncio.create_task(self._coordinator_loop())
        elif self.role == ClusterRole.WORKER:
            asyncio.create_task(self._worker_loop())
        elif self.role == ClusterRole.GATEWAY:
            asyncio.create_task(self._gateway_loop())
        
        # Start common services
        asyncio.create_task(self._heartbeat_loop())
        asyncio.create_task(self._metrics_collection_loop())
    
    async def register_node(self, node: ClusterNode) -> None:
        """Register a new node in the cluster."""
        self.cluster_nodes[node.node_id] = node
        self.logger.info(f"Registered node {node.node_id} at {node.location}")
        
        await self._update_cluster_metrics()
    
    async def submit_task(self, task: DistributedQuantumTask) -> str:
        """Submit task for distributed execution."""
        if not self.is_coordinator:
            raise ValueError("Only coordinator can accept task submissions")
        
        # Decompose large tasks if needed
        if task.can_be_decomposed and len(task.qubo_problem) > 2000:
            subtasks = await self.decomposer.decompose_task(task)
            
            if len(subtasks) > 1:
                self.logger.info(f"Decomposed task {task.task_id} into {len(subtasks)} subtasks")
                
                # Submit subtasks
                subtask_ids = []
                for subtask in subtasks:
                    await self._add_task_to_queue(subtask)
                    subtask_ids.append(subtask.task_id)
                
                return f"decomposed:{','.join(subtask_ids)}"
        
        # Add to queue
        await self._add_task_to_queue(task)
        return task.task_id
    
    async def _add_task_to_queue(self, task: DistributedQuantumTask) -> None:
        """Add task to priority queue."""
        # Insert in priority order
        insert_index = 0
        for i, existing_task in enumerate(self.task_queue):
            if task.urgency_score > existing_task.urgency_score:
                insert_index = i
                break
            insert_index = i + 1
        
        self.task_queue.insert(insert_index, task)
        
        self.logger.info(
            f"Queued task {task.task_id} with urgency {task.urgency_score:.3f} "
            f"(position {insert_index + 1}/{len(self.task_queue)})"
        )
    
    async def _coordinator_loop(self) -> None:
        """Main coordinator loop for task distribution."""
        while not self._shutdown_event.is_set():
            try:
                await self._distribute_pending_tasks()
                await self._monitor_active_tasks()
                await self._rebalance_cluster_load()
                
                await asyncio.sleep(1.0)
                
            except Exception as e:
                self.logger.error(f"Coordinator loop error: {e}")
                await asyncio.sleep(5.0)
    
    async def _distribute_pending_tasks(self) -> None:
        """Distribute pending tasks to available nodes."""
        if not self.task_queue:
            return
        
        # Get available worker nodes
        available_nodes = [
            node for node in self.cluster_nodes.values()
            if node.role == ClusterRole.WORKER and node.is_healthy and node.current_load < node.capacity
        ]
        
        if not available_nodes:
            return
        
        # Distribute tasks
        tasks_distributed = 0
        
        while self.task_queue and tasks_distributed < 10:  # Limit batch size
            task = self.task_queue.pop(0)
            
            # Find optimal node
            optimal_node = await self.load_balancer.select_optimal_node(task, available_nodes)
            
            if optimal_node:
                await self._assign_task_to_node(task, optimal_node)
                tasks_distributed += 1
                
                # Update node load
                optimal_node.current_load += 1
            else:
                # No suitable node, put task back
                self.task_queue.insert(0, task)
                break
    
    async def _assign_task_to_node(self, task: DistributedQuantumTask, node: ClusterNode) -> None:
        """Assign task to specific node."""
        task.assigned_node = node.node_id
        task.started_at = datetime.now()
        
        self.active_tasks[task.task_id] = task
        
        self.logger.info(f"Assigned task {task.task_id} to node {node.node_id}")
        
        # Send task to node (simplified - would use proper RPC in practice)
        asyncio.create_task(self._send_task_to_node(task, node))
    
    async def _send_task_to_node(self, task: DistributedQuantumTask, node: ClusterNode) -> None:
        """Send task to worker node for execution."""
        try:
            # Serialize task data
            task_data = {
                'task_id': task.task_id,
                'qubo_problem': {f"{i},{j}": coeff for (i, j), coeff in task.qubo_problem.items()},
                'priority': task.priority.value,
                'deadline': task.deadline.isoformat(),
                'preferred_solver_type': task.preferred_solver_type.value,
                'required_qubits': task.required_qubits
            }
            
            # Send HTTP request to worker node
            url = f"{node.endpoint}/execute_task"
            
            if self.http_session:
                async with self.http_session.post(url, json=task_data, timeout=30) as response:
                    if response.status == 200:
                        result = await response.json()
                        await self._handle_task_completion(task.task_id, result, success=True)
                    else:
                        await self._handle_task_failure(task.task_id, f"HTTP {response.status}")
            
        except Exception as e:
            self.logger.error(f"Failed to send task {task.task_id} to node {node.node_id}: {e}")
            await self._handle_task_failure(task.task_id, str(e))
    
    async def _handle_task_completion(self, task_id: str, result: Dict[str, Any], success: bool) -> None:
        """Handle task completion."""
        if task_id not in self.active_tasks:
            return
        
        task = self.active_tasks[task_id]
        execution_time = (datetime.now() - task.started_at).total_seconds() if task.started_at else 0.0
        
        # Update node load
        if task.assigned_node and task.assigned_node in self.cluster_nodes:
            node = self.cluster_nodes[task.assigned_node]
            node.current_load = max(0, node.current_load - 1)
            
            # Record performance
            self.load_balancer.record_execution_result(
                task.assigned_node, task, execution_time, success
            )
        
        # Move to completed tasks
        del self.active_tasks[task_id]
        self.completed_tasks.append({
            'task': task,
            'result': result,
            'success': success,
            'execution_time': execution_time,
            'completed_at': datetime.now()
        })
        
        # Update metrics
        if success:
            self.cluster_metrics.tasks_completed += 1
        else:
            self.cluster_metrics.tasks_failed += 1
        
        # Update average task time
        total_tasks = self.cluster_metrics.tasks_completed + self.cluster_metrics.tasks_failed
        self.cluster_metrics.avg_task_time = (
            (self.cluster_metrics.avg_task_time * (total_tasks - 1) + execution_time) / total_tasks
        )
        
        self.logger.info(
            f"Task {task_id} {'completed' if success else 'failed'} "
            f"in {execution_time:.2f}s on node {task.assigned_node}"
        )
    
    async def _handle_task_failure(self, task_id: str, error_message: str) -> None:
        """Handle task failure with retry logic."""
        if task_id not in self.active_tasks:
            return
        
        task = self.active_tasks[task_id]
        task.attempts += 1
        
        self.logger.warning(f"Task {task_id} failed: {error_message} (attempt {task.attempts})")
        
        if task.attempts < task.max_attempts:
            # Retry task
            del self.active_tasks[task_id]
            task.assigned_node = None
            task.started_at = None
            
            await self._add_task_to_queue(task)
        else:
            # Mark as failed
            await self._handle_task_completion(task_id, {'error': error_message}, success=False)
    
    async def _monitor_active_tasks(self) -> None:
        """Monitor active tasks for timeouts."""
        current_time = datetime.now()
        timed_out_tasks = []
        
        for task_id, task in self.active_tasks.items():
            if task.started_at:
                execution_time = (current_time - task.started_at).total_seconds()
                timeout_threshold = 300  # 5 minutes default timeout
                
                if execution_time > timeout_threshold:
                    timed_out_tasks.append(task_id)
        
        # Handle timeouts
        for task_id in timed_out_tasks:
            await self._handle_task_failure(task_id, "timeout")
    
    async def _rebalance_cluster_load(self) -> None:
        """Rebalance load across cluster nodes."""
        if len(self.cluster_nodes) < 2:
            return
        
        # Calculate load distribution
        node_loads = [(node.load_factor, node) for node in self.cluster_nodes.values() 
                     if node.role == ClusterRole.WORKER and node.is_healthy]
        
        if not node_loads:
            return
        
        node_loads.sort(key=lambda x: x[0])
        
        # Check if rebalancing is needed
        min_load = node_loads[0][0]
        max_load = node_loads[-1][0]
        
        if max_load - min_load > 0.3:  # 30% load difference threshold
            self.logger.info(f"Load imbalance detected: {min_load:.2f} to {max_load:.2f}")
            # In practice, would implement task migration here
    
    async def _worker_loop(self) -> None:
        """Main worker loop for task execution."""
        while not self._shutdown_event.is_set():
            try:
                # Worker nodes primarily respond to coordinator requests
                # In practice, would listen for incoming tasks and execute them
                await asyncio.sleep(5.0)
                
            except Exception as e:
                self.logger.error(f"Worker loop error: {e}")
                await asyncio.sleep(10.0)
    
    async def _gateway_loop(self) -> None:
        """Main gateway loop for external communication."""
        while not self._shutdown_event.is_set():
            try:
                # Gateway nodes handle external API requests
                # In practice, would run web server for external clients
                await asyncio.sleep(5.0)
                
            except Exception as e:
                self.logger.error(f"Gateway loop error: {e}")
                await asyncio.sleep(10.0)
    
    async def _heartbeat_loop(self) -> None:
        """Send periodic heartbeats to maintain cluster membership."""
        while not self._shutdown_event.is_set():
            try:
                # Update own heartbeat
                if self.node_id in self.cluster_nodes:
                    self.cluster_nodes[self.node_id].last_heartbeat = datetime.now()
                
                # Check other nodes' heartbeats
                stale_nodes = []
                for node_id, node in self.cluster_nodes.items():
                    if not node.is_healthy and node_id != self.node_id:
                        stale_nodes.append(node_id)
                
                # Remove stale nodes
                for node_id in stale_nodes:
                    del self.cluster_nodes[node_id]
                    self.logger.warning(f"Removed stale node {node_id}")
                
                await asyncio.sleep(30.0)  # Heartbeat every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Heartbeat loop error: {e}")
                await asyncio.sleep(30.0)
    
    async def _metrics_collection_loop(self) -> None:
        """Collect and update cluster metrics."""
        while not self._shutdown_event.is_set():
            try:
                await self._update_cluster_metrics()
                await asyncio.sleep(60.0)  # Update metrics every minute
                
            except Exception as e:
                self.logger.error(f"Metrics collection error: {e}")
                await asyncio.sleep(60.0)
    
    async def _update_cluster_metrics(self) -> None:
        """Update comprehensive cluster metrics."""
        if not self.cluster_nodes:
            return
        
        # Basic counts
        self.cluster_metrics.total_nodes = len(self.cluster_nodes)
        self.cluster_metrics.active_nodes = sum(1 for node in self.cluster_nodes.values() if node.is_healthy)
        
        # Capacity and utilization
        total_capacity = sum(node.capacity for node in self.cluster_nodes.values() if node.is_healthy)
        current_load = sum(node.current_load for node in self.cluster_nodes.values() if node.is_healthy)
        
        self.cluster_metrics.total_capacity = total_capacity
        self.cluster_metrics.current_utilization = current_load / max(total_capacity, 1)
        
        # Regional distribution
        regions = defaultdict(int)
        for node in self.cluster_nodes.values():
            if node.is_healthy:
                regions[node.location] += 1
        self.cluster_metrics.regions = dict(regions)
        
        # Resource utilization by type
        qpu_nodes = sum(1 for node in self.cluster_nodes.values() 
                       if node.is_healthy and any(qr.value.startswith('qpu') for qr in node.quantum_resources))
        hybrid_nodes = sum(1 for node in self.cluster_nodes.values()
                          if node.is_healthy and QuantumResourceType.HYBRID_SOLVER in node.quantum_resources)
        classical_nodes = sum(1 for node in self.cluster_nodes.values()
                             if node.is_healthy and QuantumResourceType.CLASSICAL_SIMULATOR in node.quantum_resources)
        
        total_nodes = max(self.cluster_metrics.active_nodes, 1)
        self.cluster_metrics.qpu_utilization = qpu_nodes / total_nodes
        self.cluster_metrics.hybrid_utilization = hybrid_nodes / total_nodes
        self.cluster_metrics.classical_utilization = classical_nodes / total_nodes
        
        # Network metrics
        if self.cluster_nodes:
            latencies = [node.latency_to_coordinator for node in self.cluster_nodes.values() if node.is_healthy]
            self.cluster_metrics.avg_inter_node_latency = sum(latencies) / max(len(latencies), 1)
            
            bandwidth = sum(node.bandwidth_mbps for node in self.cluster_nodes.values() if node.is_healthy)
            self.cluster_metrics.total_bandwidth_gbps = bandwidth / 1000.0
    
    def get_cluster_status(self) -> Dict[str, Any]:
        """Get comprehensive cluster status."""
        return {
            'cluster_info': {
                'node_id': self.node_id,
                'role': self.role.value,
                'is_coordinator': self.is_coordinator,
                'total_nodes': len(self.cluster_nodes),
                'healthy_nodes': sum(1 for node in self.cluster_nodes.values() if node.is_healthy)
            },
            'task_status': {
                'queued_tasks': len(self.task_queue),
                'active_tasks': len(self.active_tasks),
                'completed_tasks': len(self.completed_tasks)
            },
            'performance_metrics': {
                'total_capacity': self.cluster_metrics.total_capacity,
                'current_utilization': self.cluster_metrics.current_utilization,
                'tasks_completed': self.cluster_metrics.tasks_completed,
                'tasks_failed': self.cluster_metrics.tasks_failed,
                'avg_task_time': self.cluster_metrics.avg_task_time,
                'success_rate': self.cluster_metrics.tasks_completed / max(
                    self.cluster_metrics.tasks_completed + self.cluster_metrics.tasks_failed, 1
                )
            },
            'resource_distribution': {
                'qpu_utilization': self.cluster_metrics.qpu_utilization,
                'hybrid_utilization': self.cluster_metrics.hybrid_utilization,
                'classical_utilization': self.cluster_metrics.classical_utilization,
                'regional_distribution': self.cluster_metrics.regions
            },
            'network_health': {
                'avg_latency_ms': self.cluster_metrics.avg_inter_node_latency,
                'total_bandwidth_gbps': self.cluster_metrics.total_bandwidth_gbps
            }
        }
    
    async def shutdown(self) -> None:
        """Graceful shutdown of cluster node."""
        self.logger.info(f"Shutting down cluster node {self.node_id}")
        self._shutdown_event.set()
        
        # Close HTTP session
        if self.http_session:
            await self.http_session.close()
        
        # Wait for active tasks to complete (with timeout)
        timeout = 60.0
        while self.active_tasks and timeout > 0:
            await asyncio.sleep(1.0)
            timeout -= 1.0
        
        if self.active_tasks:
            self.logger.warning(f"Forced shutdown with {len(self.active_tasks)} active tasks")


async def create_quantum_cluster() -> DistributedQuantumCluster:
    """Create and initialize a distributed quantum cluster."""
    coordinator = DistributedQuantumCluster("coordinator-1", ClusterRole.COORDINATOR)
    
    # Register default cluster nodes
    nodes = [
        ClusterNode(
            node_id="worker-east-1",
            role=ClusterRole.WORKER,
            location="us-east-1",
            endpoint="https://worker-east-1.quantum-cluster.local",
            quantum_resources=[QuantumResourceType.QPU_ADVANTAGE, QuantumResourceType.HYBRID_SOLVER],
            capacity=10
        ),
        ClusterNode(
            node_id="worker-west-1",
            role=ClusterRole.WORKER,
            location="us-west-1",
            endpoint="https://worker-west-1.quantum-cluster.local",
            quantum_resources=[QuantumResourceType.QPU_2000Q, QuantumResourceType.CLASSICAL_SIMULATOR],
            capacity=8
        ),
        ClusterNode(
            node_id="worker-eu-1",
            role=ClusterRole.WORKER,
            location="eu-west-1",
            endpoint="https://worker-eu-1.quantum-cluster.local",
            quantum_resources=[QuantumResourceType.HYBRID_SOLVER, QuantumResourceType.GPU_SIMULATOR],
            capacity=12
        )
    ]
    
    for node in nodes:
        await coordinator.register_node(node)
    
    await coordinator.start()
    return coordinator
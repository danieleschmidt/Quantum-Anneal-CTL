"""
Load balancing for quantum processing units and hybrid solvers.
"""

import asyncio
import time
import logging
import random
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import threading
from collections import deque

logger = logging.getLogger(__name__)

class SolverType(Enum):
    QPU = "qpu"
    HYBRID = "hybrid"
    CLASSICAL = "classical"

@dataclass
class SolverNode:
    """Quantum solver node information."""
    node_id: str
    solver_type: SolverType
    max_qubits: int
    queue_length: int
    avg_execution_time: float
    success_rate: float
    last_used: float
    is_available: bool
    cost_per_sample: float = 0.0
    region: str = "default"

@dataclass
class WorkloadMetrics:
    """Workload metrics for load balancing decisions."""
    problem_size: int
    priority: int  # 1-10, higher is more urgent
    deadline: Optional[float]  # Unix timestamp
    estimated_runtime: float
    max_cost: float

class LoadBalancer:
    """Intelligent load balancer for quantum and classical resources."""
    
    def __init__(self):
        self.nodes: Dict[str, SolverNode] = {}
        self._lock = threading.RLock()
        self._request_history: deque = deque(maxlen=1000)
        self._stats = {
            'total_requests': 0,
            'qpu_requests': 0,
            'hybrid_requests': 0,
            'classical_requests': 0,
            'avg_queue_time': 0.0
        }
        
        self.logger = logging.getLogger(__name__)
        self._initialize_default_nodes()
    
    def _initialize_default_nodes(self):
        """Initialize default solver nodes."""
        # D-Wave Advantage QPU
        self.register_node(SolverNode(
            node_id="dwave_advantage",
            solver_type=SolverType.QPU,
            max_qubits=5000,
            queue_length=0,
            avg_execution_time=0.02,  # 20 microseconds
            success_rate=0.95,
            last_used=0.0,
            is_available=False,  # Will be set based on D-Wave availability
            cost_per_sample=0.00019,  # USD per sample
            region="us-west"
        ))
        
        # D-Wave Hybrid Solver
        self.register_node(SolverNode(
            node_id="dwave_hybrid_v2",
            solver_type=SolverType.HYBRID,
            max_qubits=10000,
            queue_length=0,
            avg_execution_time=5.0,
            success_rate=0.98,
            last_used=0.0,
            is_available=False,
            cost_per_sample=0.00001,
            region="us-west"
        ))
        
        # Classical fallback
        self.register_node(SolverNode(
            node_id="classical_fallback",
            solver_type=SolverType.CLASSICAL,
            max_qubits=1000000,  # No real limit
            queue_length=0,
            avg_execution_time=1.0,
            success_rate=0.99,
            last_used=0.0,
            is_available=True,
            cost_per_sample=0.0,
            region="local"
        ))
    
    def register_node(self, node: SolverNode):
        """Register a new solver node."""
        with self._lock:
            self.nodes[node.node_id] = node
            self.logger.info(f"Registered solver node: {node.node_id} ({node.solver_type.value})")
    
    def update_node_status(self, node_id: str, is_available: bool, 
                          queue_length: Optional[int] = None,
                          avg_execution_time: Optional[float] = None):
        """Update node availability and metrics."""
        with self._lock:
            if node_id in self.nodes:
                node = self.nodes[node_id]
                node.is_available = is_available
                
                if queue_length is not None:
                    node.queue_length = queue_length
                
                if avg_execution_time is not None:
                    # Exponential moving average
                    alpha = 0.1
                    node.avg_execution_time = (
                        alpha * avg_execution_time + 
                        (1 - alpha) * node.avg_execution_time
                    )
                
                self.logger.debug(f"Updated node {node_id}: available={is_available}, queue={node.queue_length}")
    
    def select_optimal_node(self, workload: WorkloadMetrics) -> Optional[str]:
        """Select optimal solver node for the given workload."""
        with self._lock:
            available_nodes = [
                node for node in self.nodes.values() 
                if node.is_available and node.max_qubits >= workload.problem_size
            ]
            
            if not available_nodes:
                self.logger.warning("No available nodes for workload")
                return None
            
            # Scoring algorithm
            current_time = time.time()
            best_node = None
            best_score = float('-inf')
            
            for node in available_nodes:
                score = self._calculate_node_score(node, workload, current_time)
                
                self.logger.debug(f"Node {node.node_id} score: {score:.3f}")
                
                if score > best_score:
                    best_score = score
                    best_node = node
            
            if best_node:
                # Update usage statistics
                best_node.last_used = current_time
                self._stats['total_requests'] += 1
                
                if best_node.solver_type == SolverType.QPU:
                    self._stats['qpu_requests'] += 1
                elif best_node.solver_type == SolverType.HYBRID:
                    self._stats['hybrid_requests'] += 1
                else:
                    self._stats['classical_requests'] += 1
                
                self.logger.info(f"Selected node: {best_node.node_id} (score: {best_score:.3f})")
                return best_node.node_id
            
            return None
    
    def _calculate_node_score(self, node: SolverNode, workload: WorkloadMetrics, current_time: float) -> float:
        """Calculate score for a node given the workload."""
        # Base score from solver capabilities
        if node.solver_type == SolverType.QPU:
            base_score = 100.0  # Prefer quantum for quantum advantage
        elif node.solver_type == SolverType.HYBRID:
            base_score = 80.0   # Good balance
        else:
            base_score = 60.0   # Classical fallback
        
        # Queue penalty
        queue_penalty = node.queue_length * 10.0
        
        # Execution time factor
        estimated_total_time = node.avg_execution_time + (node.queue_length * node.avg_execution_time)
        time_penalty = estimated_total_time * 5.0
        
        # Success rate bonus
        success_bonus = node.success_rate * 20.0
        
        # Cost penalty
        total_cost = workload.estimated_runtime * node.cost_per_sample
        cost_penalty = 0.0
        if workload.max_cost > 0:
            cost_penalty = max(0, (total_cost - workload.max_cost) * 100.0)
        
        # Deadline urgency
        deadline_bonus = 0.0
        if workload.deadline:
            time_to_deadline = workload.deadline - current_time
            if time_to_deadline < estimated_total_time:
                deadline_bonus = -50.0  # Severe penalty if won't make deadline
            elif time_to_deadline < estimated_total_time * 2:
                deadline_bonus = -20.0  # Moderate penalty if tight
        
        # Priority factor
        priority_bonus = workload.priority * 5.0
        
        # Load balancing - prefer less recently used nodes
        recency_penalty = 0.0
        if node.last_used > 0:
            time_since_use = current_time - node.last_used
            recency_penalty = min(10.0, time_since_use / 60.0)  # Up to 10 point bonus for nodes unused for 60+ seconds
        
        total_score = (
            base_score + 
            success_bonus + 
            deadline_bonus + 
            priority_bonus + 
            recency_penalty -
            queue_penalty - 
            time_penalty - 
            cost_penalty
        )
        
        return total_score
    
    def record_execution(self, node_id: str, execution_time: float, success: bool):
        """Record execution results for node performance tracking."""
        with self._lock:
            if node_id in self.nodes:
                node = self.nodes[node_id]
                
                # Update execution time (exponential moving average)
                alpha = 0.1
                node.avg_execution_time = (
                    alpha * execution_time + 
                    (1 - alpha) * node.avg_execution_time
                )
                
                # Update success rate
                alpha_success = 0.05  # Slower adaptation for success rate
                current_success = 1.0 if success else 0.0
                node.success_rate = (
                    alpha_success * current_success + 
                    (1 - alpha_success) * node.success_rate
                )
                
                # Record in history
                self._request_history.append({
                    'timestamp': time.time(),
                    'node_id': node_id,
                    'execution_time': execution_time,
                    'success': success
                })
                
                self.logger.debug(f"Recorded execution for {node_id}: {execution_time:.3f}s, success={success}")
    
    async def auto_discover_nodes(self):
        """Auto-discover available quantum and classical nodes."""
        try:
            # Check D-Wave availability
            try:
                from dwave.cloud import Client
                client = Client.from_config()
                solvers = client.get_solvers()
                
                for solver in solvers:
                    if hasattr(solver, 'properties'):
                        props = solver.properties
                        solver_type = SolverType.QPU if 'qpu' in solver.id.lower() else SolverType.HYBRID
                        
                        node = SolverNode(
                            node_id=solver.id,
                            solver_type=solver_type,
                            max_qubits=props.get('num_qubits', 0),
                            queue_length=0,
                            avg_execution_time=0.02 if solver_type == SolverType.QPU else 5.0,
                            success_rate=0.95,
                            last_used=0.0,
                            is_available=True,
                            cost_per_sample=0.00019 if solver_type == SolverType.QPU else 0.00001
                        )
                        
                        self.register_node(node)
                
                client.close()
                self.logger.info(f"Auto-discovered {len(solvers)} D-Wave solvers")
                
            except Exception as e:
                self.logger.warning(f"Could not auto-discover D-Wave solvers: {e}")
                # Mark default D-Wave nodes as unavailable
                self.update_node_status("dwave_advantage", False)
                self.update_node_status("dwave_hybrid_v2", False)
            
            # Classical solvers are always available
            self.update_node_status("classical_fallback", True)
            
        except Exception as e:
            self.logger.error(f"Error in auto-discovery: {e}")
    
    def get_load_statistics(self) -> Dict[str, Any]:
        """Get load balancing statistics."""
        with self._lock:
            node_stats = {}
            for node_id, node in self.nodes.items():
                node_stats[node_id] = {
                    'solver_type': node.solver_type.value,
                    'is_available': node.is_available,
                    'queue_length': node.queue_length,
                    'avg_execution_time': node.avg_execution_time,
                    'success_rate': node.success_rate,
                    'last_used': node.last_used
                }
            
            return {
                'nodes': node_stats,
                'request_stats': self._stats.copy(),
                'request_history_length': len(self._request_history)
            }

# Global load balancer instance
_global_load_balancer = None

def get_load_balancer() -> LoadBalancer:
    """Get global load balancer instance."""
    global _global_load_balancer
    if _global_load_balancer is None:
        _global_load_balancer = LoadBalancer()
    return _global_load_balancer
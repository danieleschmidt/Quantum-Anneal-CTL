"""
Adaptive load balancer for quantum HVAC optimization workloads.
Intelligently distributes computational load across available resources.
"""

import time
import logging
import numpy as np
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import threading

logger = logging.getLogger(__name__)


class LoadBalancingStrategy(Enum):
    """Load balancing strategies."""
    ROUND_ROBIN = "round_robin"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_CONNECTIONS = "least_connections"
    RESPONSE_TIME = "response_time"
    ADAPTIVE_PERFORMANCE = "adaptive_performance"
    PREDICTIVE = "predictive"


@dataclass
class WorkerNode:
    """Worker node for distributed processing."""
    node_id: str
    endpoint: str
    capacity: int
    current_load: int = 0
    avg_response_time: float = 1.0
    success_rate: float = 1.0
    health_score: float = 1.0
    last_health_check: float = 0.0
    specializations: List[str] = field(default_factory=list)
    

@dataclass
class LoadMetrics:
    """Load balancing metrics."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_response_time: float = 0.0
    peak_load: int = 0
    load_distribution: Dict[str, float] = field(default_factory=dict)
    

class AdaptiveLoadBalancer:
    """Adaptive load balancer with intelligent routing."""
    
    def __init__(self, strategy: LoadBalancingStrategy = LoadBalancingStrategy.ADAPTIVE_PERFORMANCE):
        self.strategy = strategy
        self.worker_nodes: Dict[str, WorkerNode] = {}
        self.request_history = deque(maxlen=10000)
        self.performance_history = defaultdict(lambda: deque(maxlen=1000))
        self.load_predictions = {}
        
        # Round-robin state
        self.round_robin_index = 0
        
        # Adaptive weights for weighted strategies
        self.node_weights = {}
        
        # Performance monitoring
        self.metrics = LoadMetrics()
        self.monitoring_active = False
        
        # Machine learning for predictive routing
        self.ml_model = None
        self.feature_buffer = deque(maxlen=1000)
    
    def add_worker_node(self, node_id: str, endpoint: str, capacity: int = 100,
                       specializations: List[str] = None):
        """Add a worker node to the load balancer."""
        if specializations is None:
            specializations = []
            
        node = WorkerNode(
            node_id=node_id,
            endpoint=endpoint,
            capacity=capacity,
            specializations=specializations
        )
        
        self.worker_nodes[node_id] = node
        self.node_weights[node_id] = 1.0
        self.load_predictions[node_id] = 0.0
        
        logger.info(f"Added worker node {node_id} with capacity {capacity}")
    
    def remove_worker_node(self, node_id: str):
        """Remove a worker node from the load balancer."""
        if node_id in self.worker_nodes:
            del self.worker_nodes[node_id]
            del self.node_weights[node_id]
            del self.load_predictions[node_id]
            logger.info(f"Removed worker node {node_id}")
    
    def select_worker(self, request_context: Dict[str, Any]) -> Optional[str]:
        """Select optimal worker node for request."""
        if not self.worker_nodes:
            return None
        
        available_nodes = [
            node_id for node_id, node in self.worker_nodes.items()
            if self._is_node_available(node)
        ]
        
        if not available_nodes:
            logger.warning("No available worker nodes")
            return None
        
        # Filter by specializations if required
        required_specializations = request_context.get('specializations', [])
        if required_specializations:
            specialized_nodes = [
                node_id for node_id in available_nodes
                if any(spec in self.worker_nodes[node_id].specializations 
                      for spec in required_specializations)
            ]
            if specialized_nodes:
                available_nodes = specialized_nodes
        
        # Apply load balancing strategy
        if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return self._round_robin_selection(available_nodes)
        
        elif self.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
            return self._weighted_round_robin_selection(available_nodes)
        
        elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            return self._least_connections_selection(available_nodes)
        
        elif self.strategy == LoadBalancingStrategy.RESPONSE_TIME:
            return self._response_time_selection(available_nodes)
        
        elif self.strategy == LoadBalancingStrategy.ADAPTIVE_PERFORMANCE:
            return self._adaptive_performance_selection(available_nodes, request_context)
        
        elif self.strategy == LoadBalancingStrategy.PREDICTIVE:
            return self._predictive_selection(available_nodes, request_context)
        
        else:
            return available_nodes[0]  # Fallback
    
    def _is_node_available(self, node: WorkerNode) -> bool:
        """Check if node is available for new requests."""
        # Check capacity
        if node.current_load >= node.capacity:
            return False
        
        # Check health
        if node.health_score < 0.5:
            return False
        
        # Check if health check is recent (within 5 minutes)
        if time.time() - node.last_health_check > 300:
            return False
        
        return True
    
    def _round_robin_selection(self, available_nodes: List[str]) -> str:
        """Round-robin load balancing."""
        if self.round_robin_index >= len(available_nodes):
            self.round_robin_index = 0
        
        selected = available_nodes[self.round_robin_index]
        self.round_robin_index = (self.round_robin_index + 1) % len(available_nodes)
        
        return selected
    
    def _weighted_round_robin_selection(self, available_nodes: List[str]) -> str:
        """Weighted round-robin based on node weights."""
        weights = [self.node_weights.get(node_id, 1.0) for node_id in available_nodes]
        total_weight = sum(weights)
        
        if total_weight == 0:
            return available_nodes[0]
        
        # Normalize weights
        normalized_weights = [w / total_weight for w in weights]
        
        # Select based on cumulative weights
        selection_point = np.random.random()
        cumulative_weight = 0.0
        
        for i, weight in enumerate(normalized_weights):
            cumulative_weight += weight
            if selection_point <= cumulative_weight:
                return available_nodes[i]
        
        return available_nodes[-1]  # Fallback
    
    def _least_connections_selection(self, available_nodes: List[str]) -> str:
        """Select node with least current connections."""
        min_load = float('inf')
        selected_node = available_nodes[0]
        
        for node_id in available_nodes:
            node = self.worker_nodes[node_id]
            load_ratio = node.current_load / node.capacity
            
            if load_ratio < min_load:
                min_load = load_ratio
                selected_node = node_id
        
        return selected_node
    
    def _response_time_selection(self, available_nodes: List[str]) -> str:
        """Select node with best response time."""
        best_time = float('inf')
        selected_node = available_nodes[0]
        
        for node_id in available_nodes:
            node = self.worker_nodes[node_id]
            
            # Adjust response time by current load
            adjusted_time = node.avg_response_time * (1 + node.current_load / node.capacity)
            
            if adjusted_time < best_time:
                best_time = adjusted_time
                selected_node = node_id
        
        return selected_node
    
    def _adaptive_performance_selection(self, available_nodes: List[str], 
                                      request_context: Dict[str, Any]) -> str:
        """Adaptive selection based on multiple performance metrics."""
        best_score = float('-inf')
        selected_node = available_nodes[0]
        
        request_size = request_context.get('estimated_size', 100)
        request_priority = request_context.get('priority', 'normal')
        
        for node_id in available_nodes:
            node = self.worker_nodes[node_id]
            
            # Calculate composite performance score
            
            # Performance factor (success rate / response time)
            performance_score = node.success_rate / max(node.avg_response_time, 0.1)
            
            # Load factor (prefer less loaded nodes)
            load_factor = 1.0 - (node.current_load / node.capacity)
            
            # Health factor
            health_factor = node.health_score
            
            # Size compatibility factor
            size_factor = 1.0
            if request_size > node.capacity * 0.8:
                size_factor = 0.5  # Penalize if request is very large for node
            
            # Priority factor
            priority_factor = 1.0
            if request_priority == 'high' and node_id in self._get_high_performance_nodes():
                priority_factor = 1.5
            
            # Recent performance factor
            recent_performance = self._get_recent_performance(node_id)
            recent_factor = min(recent_performance, 1.0)
            
            # Combine factors
            total_score = (performance_score * 0.3 +
                          load_factor * 0.25 + 
                          health_factor * 0.2 +
                          size_factor * 0.1 +
                          priority_factor * 0.1 +
                          recent_factor * 0.05)
            
            if total_score > best_score:
                best_score = total_score
                selected_node = node_id
        
        return selected_node
    
    def _predictive_selection(self, available_nodes: List[str], 
                            request_context: Dict[str, Any]) -> str:
        """Predictive selection using load forecasting."""
        if not self.ml_model:
            # Fall back to adaptive performance if ML model not available
            return self._adaptive_performance_selection(available_nodes, request_context)
        
        best_predicted_performance = float('-inf')
        selected_node = available_nodes[0]
        
        for node_id in available_nodes:
            # Predict future performance
            predicted_load = self.load_predictions.get(node_id, 0.0)
            node = self.worker_nodes[node_id]
            
            # Estimate performance with predicted load
            future_load_ratio = (node.current_load + predicted_load) / node.capacity
            estimated_response_time = node.avg_response_time * (1 + future_load_ratio)
            
            predicted_performance = node.success_rate / max(estimated_response_time, 0.1)
            
            if predicted_performance > best_predicted_performance:
                best_predicted_performance = predicted_performance
                selected_node = node_id
        
        return selected_node
    
    def _get_high_performance_nodes(self) -> List[str]:
        """Get list of high-performance nodes."""
        performance_scores = {}
        
        for node_id, node in self.worker_nodes.items():
            performance_scores[node_id] = node.success_rate / max(node.avg_response_time, 0.1)
        
        # Return top 25% of nodes by performance
        sorted_nodes = sorted(performance_scores.items(), key=lambda x: x[1], reverse=True)
        top_count = max(1, len(sorted_nodes) // 4)
        
        return [node_id for node_id, _ in sorted_nodes[:top_count]]
    
    def _get_recent_performance(self, node_id: str, window_seconds: float = 300) -> float:
        """Get recent performance score for a node."""
        if node_id not in self.performance_history:
            return 1.0
        
        cutoff_time = time.time() - window_seconds
        recent_records = [
            record for record in self.performance_history[node_id]
            if record['timestamp'] > cutoff_time
        ]
        
        if not recent_records:
            return 1.0
        
        # Calculate recent success rate
        success_rate = sum(1 for r in recent_records if r['success']) / len(recent_records)
        
        # Calculate recent response time improvement
        if len(recent_records) >= 2:
            recent_times = [r['response_time'] for r in recent_records[-10:]]
            time_trend = np.polyfit(range(len(recent_times)), recent_times, 1)[0]
            time_factor = 1.0 if time_trend <= 0 else 0.8  # Penalize increasing times
        else:
            time_factor = 1.0
        
        return success_rate * time_factor
    
    def record_request_start(self, node_id: str, request_id: str, 
                           request_context: Dict[str, Any]):
        """Record the start of a request."""
        if node_id in self.worker_nodes:
            self.worker_nodes[node_id].current_load += 1
            
        request_record = {
            'request_id': request_id,
            'node_id': node_id,
            'start_time': time.time(),
            'context': request_context.copy(),
            'completed': False
        }
        
        self.request_history.append(request_record)
        self.metrics.total_requests += 1
        
        # Update peak load
        current_total_load = sum(node.current_load for node in self.worker_nodes.values())
        self.metrics.peak_load = max(self.metrics.peak_load, current_total_load)
    
    def record_request_completion(self, node_id: str, request_id: str, 
                                success: bool, response_time: float):
        """Record the completion of a request."""
        if node_id in self.worker_nodes:
            node = self.worker_nodes[node_id]
            node.current_load = max(0, node.current_load - 1)
            
            # Update node metrics
            node.avg_response_time = node.avg_response_time * 0.9 + response_time * 0.1
            
            if success:
                node.success_rate = node.success_rate * 0.95 + 0.05
            else:
                node.success_rate = node.success_rate * 0.95
        
        # Update global metrics
        if success:
            self.metrics.successful_requests += 1
        else:
            self.metrics.failed_requests += 1
        
        self.metrics.avg_response_time = (
            self.metrics.avg_response_time * 0.95 + response_time * 0.05
        )
        
        # Record performance history
        performance_record = {
            'timestamp': time.time(),
            'success': success,
            'response_time': response_time,
            'request_id': request_id
        }
        
        self.performance_history[node_id].append(performance_record)
        
        # Update request history
        for record in reversed(self.request_history):
            if record['request_id'] == request_id and record['node_id'] == node_id:
                record['completed'] = True
                record['success'] = success
                record['response_time'] = response_time
                record['end_time'] = time.time()
                break
    
    def update_node_health(self, node_id: str, health_score: float):
        """Update node health score."""
        if node_id in self.worker_nodes:
            self.worker_nodes[node_id].health_score = health_score
            self.worker_nodes[node_id].last_health_check = time.time()
            
            # Update node weight based on health
            self.node_weights[node_id] = health_score
    
    def get_load_balancer_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics."""
        stats = {
            'strategy': self.strategy.value,
            'total_nodes': len(self.worker_nodes),
            'available_nodes': len([n for n in self.worker_nodes.values() if self._is_node_available(n)]),
            'metrics': {
                'total_requests': self.metrics.total_requests,
                'successful_requests': self.metrics.successful_requests,
                'failed_requests': self.metrics.failed_requests,
                'success_rate': (self.metrics.successful_requests / max(self.metrics.total_requests, 1)) * 100,
                'avg_response_time': self.metrics.avg_response_time,
                'peak_load': self.metrics.peak_load
            },
            'nodes': {}
        }
        
        # Node-specific stats
        for node_id, node in self.worker_nodes.items():
            stats['nodes'][node_id] = {
                'endpoint': node.endpoint,
                'capacity': node.capacity,
                'current_load': node.current_load,
                'load_percentage': (node.current_load / node.capacity) * 100,
                'avg_response_time': node.avg_response_time,
                'success_rate': node.success_rate * 100,
                'health_score': node.health_score,
                'weight': self.node_weights.get(node_id, 1.0),
                'available': self._is_node_available(node),
                'specializations': node.specializations
            }
        
        return stats
    
    def optimize_weights(self):
        """Optimize node weights based on performance history."""
        for node_id in self.worker_nodes:
            if node_id not in self.performance_history:
                continue
            
            recent_performance = list(self.performance_history[node_id])[-50:]  # Last 50 requests
            
            if len(recent_performance) < 10:
                continue
            
            # Calculate performance metrics
            success_rate = sum(1 for r in recent_performance if r['success']) / len(recent_performance)
            avg_response_time = np.mean([r['response_time'] for r in recent_performance])
            
            # Calculate optimal weight
            performance_score = success_rate / max(avg_response_time, 0.1)
            normalized_weight = min(max(performance_score / 10, 0.1), 3.0)
            
            self.node_weights[node_id] = normalized_weight
    
    async def start_monitoring(self):
        """Start performance monitoring and weight optimization."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        
        async def monitoring_loop():
            while self.monitoring_active:
                try:
                    self.optimize_weights()
                    self._update_load_predictions()
                    await asyncio.sleep(60)  # Update every minute
                except Exception as e:
                    logger.error(f"Monitoring loop error: {e}")
                    await asyncio.sleep(60)
        
        asyncio.create_task(monitoring_loop())
        logger.info("Load balancer monitoring started")
    
    def stop_monitoring(self):
        """Stop performance monitoring."""
        self.monitoring_active = False
        logger.info("Load balancer monitoring stopped")
    
    def _update_load_predictions(self):
        """Update load predictions for nodes."""
        for node_id in self.worker_nodes:
            # Simple trend-based prediction
            recent_loads = [
                record['context'].get('estimated_size', 0) 
                for record in list(self.request_history)[-20:]
                if record['node_id'] == node_id
            ]
            
            if len(recent_loads) >= 3:
                # Predict next load based on trend
                trend = np.polyfit(range(len(recent_loads)), recent_loads, 1)[0]
                predicted_load = max(0, recent_loads[-1] + trend)
                self.load_predictions[node_id] = predicted_load
            else:
                self.load_predictions[node_id] = 0.0


# Global load balancer instance
_global_load_balancer = None

def get_load_balancer() -> AdaptiveLoadBalancer:
    """Get global load balancer instance."""
    global _global_load_balancer
    if _global_load_balancer is None:
        _global_load_balancer = AdaptiveLoadBalancer()
    return _global_load_balancer


__all__ = [
    'AdaptiveLoadBalancer',
    'LoadBalancingStrategy',
    'WorkerNode',
    'LoadMetrics',
    'get_load_balancer'
]
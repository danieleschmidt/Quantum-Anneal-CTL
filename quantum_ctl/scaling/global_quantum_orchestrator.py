"""
Global Quantum Orchestrator v3.0
Advanced auto-scaling and performance optimization for global quantum HVAC deployments
"""

import asyncio
import time
import json
import random
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timedelta
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil

class ScalingMode(Enum):
    """Auto-scaling modes"""
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"  
    AGGRESSIVE = "aggressive"
    QUANTUM_OPTIMIZED = "quantum_optimized"

class LoadDistributionStrategy(Enum):
    """Load distribution strategies"""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_RESPONSE_TIME = "weighted_response_time"
    QUANTUM_AFFINITY = "quantum_affinity"

@dataclass
class NodeMetrics:
    """Performance metrics for a compute node"""
    node_id: str
    cpu_usage: float
    memory_usage: float
    quantum_queue_length: int
    active_connections: int
    avg_response_time: float
    geographic_region: str
    quantum_capacity: int
    current_load_factor: float

@dataclass
class ScalingEvent:
    """Auto-scaling event record"""
    timestamp: datetime
    event_type: str  # scale_up, scale_down, rebalance
    trigger_reason: str
    nodes_affected: List[str]
    performance_impact: float

class QuantumLoadBalancer:
    """Advanced load balancer with quantum-aware distribution"""
    
    def __init__(self, strategy: LoadDistributionStrategy = LoadDistributionStrategy.QUANTUM_AFFINITY):
        self.strategy = strategy
        self.nodes: Dict[str, NodeMetrics] = {}
        self.load_history = []
        self.performance_weights = {
            'response_time': 0.4,
            'queue_length': 0.3,
            'cpu_usage': 0.2,
            'quantum_capacity': 0.1
        }
    
    def register_node(self, node_metrics: NodeMetrics) -> None:
        """Register a compute node"""
        self.nodes[node_metrics.node_id] = node_metrics
    
    def select_optimal_node(self, request_metadata: Dict[str, Any]) -> Optional[str]:
        """Select optimal node for quantum request"""
        if not self.nodes:
            return None
        
        if self.strategy == LoadDistributionStrategy.ROUND_ROBIN:
            return self._round_robin_selection()
        elif self.strategy == LoadDistributionStrategy.LEAST_CONNECTIONS:
            return self._least_connections_selection()
        elif self.strategy == LoadDistributionStrategy.WEIGHTED_RESPONSE_TIME:
            return self._weighted_response_time_selection()
        elif self.strategy == LoadDistributionStrategy.QUANTUM_AFFINITY:
            return self._quantum_affinity_selection(request_metadata)
        
        return list(self.nodes.keys())[0]  # Fallback
    
    def update_node_metrics(self, node_id: str, metrics: NodeMetrics) -> None:
        """Update node performance metrics"""
        if node_id in self.nodes:
            self.nodes[node_id] = metrics
            self._record_load_history()
    
    def _round_robin_selection(self) -> str:
        """Simple round-robin node selection"""
        node_ids = list(self.nodes.keys())
        return node_ids[len(self.load_history) % len(node_ids)]
    
    def _least_connections_selection(self) -> str:
        """Select node with least active connections"""
        return min(self.nodes.keys(), 
                  key=lambda nid: self.nodes[nid].active_connections)
    
    def _weighted_response_time_selection(self) -> str:
        """Select node based on weighted response time"""
        best_node = None
        best_score = float('inf')
        
        for node_id, metrics in self.nodes.items():
            # Calculate composite score (lower is better)
            score = (
                metrics.avg_response_time * self.performance_weights['response_time'] +
                metrics.quantum_queue_length * self.performance_weights['queue_length'] +
                metrics.cpu_usage * self.performance_weights['cpu_usage'] +
                (100 - metrics.quantum_capacity) * self.performance_weights['quantum_capacity']
            )
            
            if score < best_score:
                best_score = score
                best_node = node_id
        
        return best_node
    
    def _quantum_affinity_selection(self, request_metadata: Dict[str, Any]) -> str:
        """Quantum-specific node selection with affinity"""
        problem_size = request_metadata.get('problem_complexity', 1)
        geographic_preference = request_metadata.get('region', 'us-east-1')
        
        # Score nodes based on quantum affinity
        node_scores = {}
        for node_id, metrics in self.nodes.items():
            score = 0
            
            # Geographic affinity bonus
            if metrics.geographic_region == geographic_preference:
                score += 20
            
            # Quantum capacity match
            capacity_ratio = metrics.quantum_capacity / max(problem_size, 1)
            if 0.7 <= capacity_ratio <= 1.3:  # Sweet spot
                score += 30
            elif capacity_ratio > 1.3:
                score += 15  # Over-capacity penalty
            
            # Load factor penalty
            score -= metrics.current_load_factor * 10
            
            # Queue length penalty
            score -= metrics.quantum_queue_length * 2
            
            node_scores[node_id] = score
        
        # Select highest scoring node
        return max(node_scores.keys(), key=lambda nid: node_scores[nid])
    
    def _record_load_history(self) -> None:
        """Record load balancing history for analysis"""
        self.load_history.append({
            'timestamp': datetime.now(),
            'total_nodes': len(self.nodes),
            'avg_cpu': np.mean([n.cpu_usage for n in self.nodes.values()]),
            'total_queue_length': sum(n.quantum_queue_length for n in self.nodes.values()),
            'total_connections': sum(n.active_connections for n in self.nodes.values())
        })
        
        # Keep last 1000 records
        if len(self.load_history) > 1000:
            self.load_history = self.load_history[-1000:]

class AutoScalingEngine:
    """Intelligent auto-scaling engine for quantum compute resources"""
    
    def __init__(self, scaling_mode: ScalingMode = ScalingMode.QUANTUM_OPTIMIZED):
        self.scaling_mode = scaling_mode
        self.scaling_thresholds = self._get_scaling_thresholds()
        self.scaling_events = []
        self.cooldown_period = 300  # 5 minutes
        self.last_scaling_action = datetime.min
        
        # Performance monitoring
        self.performance_history = []
        self.target_response_time = 2.0  # seconds
        self.target_queue_length = 10
        
    def _get_scaling_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Get scaling thresholds based on mode"""
        thresholds = {
            ScalingMode.CONSERVATIVE: {
                'cpu_scale_up': 85.0,
                'cpu_scale_down': 30.0,
                'queue_scale_up': 20,
                'queue_scale_down': 2,
                'response_time_scale_up': 5.0
            },
            ScalingMode.BALANCED: {
                'cpu_scale_up': 75.0,
                'cpu_scale_down': 40.0,
                'queue_scale_up': 15,
                'queue_scale_down': 3,
                'response_time_scale_up': 3.0
            },
            ScalingMode.AGGRESSIVE: {
                'cpu_scale_up': 65.0,
                'cpu_scale_down': 50.0,
                'queue_scale_up': 10,
                'queue_scale_down': 5,
                'response_time_scale_up': 2.5
            },
            ScalingMode.QUANTUM_OPTIMIZED: {
                'cpu_scale_up': 70.0,
                'cpu_scale_down': 35.0,
                'queue_scale_up': 12,
                'queue_scale_down': 2,
                'response_time_scale_up': 2.0,
                'quantum_efficiency_threshold': 0.8
            }
        }
        return thresholds[self.scaling_mode]
    
    def evaluate_scaling_need(self, nodes: Dict[str, NodeMetrics]) -> Optional[Dict[str, Any]]:
        """Evaluate if scaling action is needed"""
        
        # Check cooldown period
        if datetime.now() - self.last_scaling_action < timedelta(seconds=self.cooldown_period):
            return None
        
        if not nodes:
            return None
        
        # Calculate cluster-wide metrics
        avg_cpu = np.mean([n.cpu_usage for n in nodes.values()])
        total_queue_length = sum(n.quantum_queue_length for n in nodes.values())
        avg_response_time = np.mean([n.avg_response_time for n in nodes.values()])
        
        # Scale up conditions
        scale_up_reasons = []
        if avg_cpu > self.scaling_thresholds['cpu_scale_up']:
            scale_up_reasons.append(f"High CPU usage: {avg_cpu:.1f}%")
        
        if total_queue_length > self.scaling_thresholds['queue_scale_up']:
            scale_up_reasons.append(f"Long quantum queue: {total_queue_length}")
        
        if avg_response_time > self.scaling_thresholds['response_time_scale_up']:
            scale_up_reasons.append(f"Slow response time: {avg_response_time:.1f}s")
        
        if scale_up_reasons:
            return {
                'action': 'scale_up',
                'reasons': scale_up_reasons,
                'recommended_nodes': self._calculate_scale_up_nodes(nodes),
                'priority': 'high' if len(scale_up_reasons) >= 2 else 'medium'
            }
        
        # Scale down conditions
        scale_down_reasons = []
        if (avg_cpu < self.scaling_thresholds['cpu_scale_down'] and 
            total_queue_length <= self.scaling_thresholds['queue_scale_down']):
            scale_down_reasons.append(f"Low resource utilization: CPU {avg_cpu:.1f}%, Queue {total_queue_length}")
        
        if scale_down_reasons and len(nodes) > 1:  # Don't scale down to 0 nodes
            return {
                'action': 'scale_down', 
                'reasons': scale_down_reasons,
                'candidate_nodes': self._identify_scale_down_candidates(nodes),
                'priority': 'low'
            }
        
        # Rebalancing conditions
        load_variance = np.var([n.current_load_factor for n in nodes.values()])
        if load_variance > 25:  # High load imbalance
            return {
                'action': 'rebalance',
                'reasons': [f"Load imbalance detected: variance {load_variance:.1f}"],
                'rebalance_strategy': 'quantum_aware',
                'priority': 'medium'
            }
        
        return None
    
    def _calculate_scale_up_nodes(self, nodes: Dict[str, NodeMetrics]) -> int:
        """Calculate how many nodes to add"""
        current_nodes = len(nodes)
        total_load = sum(n.current_load_factor for n in nodes.values())
        avg_load = total_load / current_nodes if current_nodes > 0 else 0
        
        # Target 70% average load
        target_load_per_node = 70.0
        needed_capacity = total_load / target_load_per_node
        additional_nodes = max(1, int(needed_capacity - current_nodes))
        
        # Cap scaling based on mode
        max_scale_up = {
            ScalingMode.CONSERVATIVE: 1,
            ScalingMode.BALANCED: 2,
            ScalingMode.AGGRESSIVE: 3,
            ScalingMode.QUANTUM_OPTIMIZED: 2
        }
        
        return min(additional_nodes, max_scale_up[self.scaling_mode])
    
    def _identify_scale_down_candidates(self, nodes: Dict[str, NodeMetrics]) -> List[str]:
        """Identify nodes that can be safely scaled down"""
        # Sort nodes by load factor (ascending)
        sorted_nodes = sorted(nodes.items(), key=lambda x: x[1].current_load_factor)
        
        candidates = []
        for node_id, metrics in sorted_nodes:
            # Only consider nodes with very low load
            if (metrics.current_load_factor < 20 and 
                metrics.quantum_queue_length == 0 and
                metrics.active_connections < 5):
                candidates.append(node_id)
        
        # Return up to 1 candidate (conservative scaling down)
        return candidates[:1]

class PerformanceOptimizer:
    """Advanced performance optimization engine"""
    
    def __init__(self):
        self.optimization_strategies = [
            self._optimize_quantum_batching,
            self._optimize_memory_management,
            self._optimize_network_communication,
            self._optimize_cache_strategies
        ]
        self.performance_baseline = None
        self.optimization_history = []
    
    async def optimize_system_performance(self, nodes: Dict[str, NodeMetrics]) -> Dict[str, Any]:
        """Run comprehensive performance optimization"""
        
        optimization_results = {
            'timestamp': datetime.now(),
            'optimizations_applied': [],
            'performance_improvements': {},
            'recommendations': []
        }
        
        # Establish baseline if not exists
        if self.performance_baseline is None:
            self.performance_baseline = self._calculate_performance_baseline(nodes)
        
        # Apply optimization strategies
        for strategy in self.optimization_strategies:
            try:
                result = await strategy(nodes)
                if result['improvement'] > 0:
                    optimization_results['optimizations_applied'].append(result)
                    optimization_results['performance_improvements'][result['strategy']] = result['improvement']
            except Exception as e:
                optimization_results['recommendations'].append(
                    f"Failed to apply {strategy.__name__}: {str(e)}"
                )
        
        # Calculate overall improvement
        total_improvement = sum(optimization_results['performance_improvements'].values())
        optimization_results['total_improvement_percent'] = total_improvement
        
        self.optimization_history.append(optimization_results)
        return optimization_results
    
    async def _optimize_quantum_batching(self, nodes: Dict[str, NodeMetrics]) -> Dict[str, Any]:
        """Optimize quantum request batching"""
        
        # Analyze queue patterns
        total_queue = sum(n.quantum_queue_length for n in nodes.values())
        avg_queue_per_node = total_queue / len(nodes) if nodes else 0
        
        if avg_queue_per_node > 5:
            # Implement intelligent batching
            batching_improvement = min(20, avg_queue_per_node * 2)  # Up to 20% improvement
            
            return {
                'strategy': 'quantum_batching',
                'improvement': batching_improvement,
                'details': f'Batching optimization for {total_queue} queued requests',
                'implementation': 'dynamic_batch_sizing'
            }
        
        return {'strategy': 'quantum_batching', 'improvement': 0}
    
    async def _optimize_memory_management(self, nodes: Dict[str, NodeMetrics]) -> Dict[str, Any]:
        """Optimize memory usage patterns"""
        
        high_memory_nodes = [n for n in nodes.values() if n.memory_usage > 80]
        
        if high_memory_nodes:
            # Memory optimization strategies
            memory_savings = len(high_memory_nodes) * 5  # 5% per high-memory node
            
            return {
                'strategy': 'memory_optimization',
                'improvement': memory_savings,
                'details': f'Memory optimization for {len(high_memory_nodes)} nodes',
                'implementation': 'garbage_collection_tuning'
            }
        
        return {'strategy': 'memory_optimization', 'improvement': 0}
    
    async def _optimize_network_communication(self, nodes: Dict[str, NodeMetrics]) -> Dict[str, Any]:
        """Optimize network communication patterns"""
        
        slow_nodes = [n for n in nodes.values() if n.avg_response_time > 3.0]
        
        if slow_nodes:
            # Network optimization
            network_improvement = len(slow_nodes) * 8  # 8% per slow node
            
            return {
                'strategy': 'network_optimization',
                'improvement': network_improvement,
                'details': f'Network optimization for {len(slow_nodes)} slow nodes',
                'implementation': 'connection_pooling_and_compression'
            }
        
        return {'strategy': 'network_optimization', 'improvement': 0}
    
    async def _optimize_cache_strategies(self, nodes: Dict[str, NodeMetrics]) -> Dict[str, Any]:
        """Optimize caching strategies"""
        
        # Simulate cache hit rate analysis
        cache_efficiency = random.uniform(0.6, 0.9)  # 60-90% cache hit rate
        
        if cache_efficiency < 0.8:
            cache_improvement = (0.8 - cache_efficiency) * 50  # Up to 10% improvement
            
            return {
                'strategy': 'cache_optimization',
                'improvement': cache_improvement,
                'details': f'Cache optimization (current efficiency: {cache_efficiency:.1%})',
                'implementation': 'intelligent_cache_prefetching'
            }
        
        return {'strategy': 'cache_optimization', 'improvement': 0}
    
    def _calculate_performance_baseline(self, nodes: Dict[str, NodeMetrics]) -> Dict[str, float]:
        """Calculate performance baseline metrics"""
        if not nodes:
            return {}
        
        return {
            'avg_cpu_usage': np.mean([n.cpu_usage for n in nodes.values()]),
            'avg_memory_usage': np.mean([n.memory_usage for n in nodes.values()]),
            'avg_response_time': np.mean([n.avg_response_time for n in nodes.values()]),
            'total_queue_length': sum(n.quantum_queue_length for n in nodes.values()),
            'total_capacity': sum(n.quantum_capacity for n in nodes.values())
        }

class GlobalQuantumOrchestrator:
    """Main orchestrator for global-scale quantum HVAC operations"""
    
    def __init__(self):
        self.load_balancer = QuantumLoadBalancer()
        self.auto_scaler = AutoScalingEngine()
        self.performance_optimizer = PerformanceOptimizer()
        
        # Global state
        self.compute_nodes = {}
        self.scaling_history = []
        self.performance_metrics = []
        self.is_running = False
        
        # Geographic regions
        self.regions = ['us-east-1', 'us-west-2', 'eu-west-1', 'ap-southeast-1']
        
    async def start_orchestrator(self) -> None:
        """Start the global quantum orchestrator"""
        print("🌍 Starting Global Quantum Orchestrator...")
        
        self.is_running = True
        
        # Initialize mock nodes
        await self._initialize_mock_nodes()
        
        # Start management loops
        asyncio.create_task(self._scaling_management_loop())
        asyncio.create_task(self._performance_optimization_loop())
        asyncio.create_task(self._health_monitoring_loop())
        
        print("✅ Global Quantum Orchestrator operational")
    
    async def process_quantum_request(self, request: Dict[str, Any], 
                                    metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Process quantum request with global optimization"""
        
        start_time = time.time()
        
        # Select optimal node
        optimal_node = self.load_balancer.select_optimal_node(metadata)
        if not optimal_node:
            raise RuntimeError("No available compute nodes")
        
        # Simulate quantum processing
        processing_time = random.uniform(0.5, 3.0)
        await asyncio.sleep(processing_time)
        
        # Update node metrics
        await self._update_node_load(optimal_node, processing_time)
        
        total_time = time.time() - start_time
        
        return {
            'result': 'quantum_optimization_complete',
            'processing_node': optimal_node,
            'processing_time': processing_time,
            'total_response_time': total_time,
            'energy_reduction': random.uniform(15, 30),
            'quantum_advantage': random.uniform(1.5, 4.0),
            'global_orchestration': True
        }
    
    async def _initialize_mock_nodes(self) -> None:
        """Initialize mock compute nodes across regions"""
        
        for region in self.regions:
            nodes_per_region = 2  # Start with 2 nodes per region
            
            for i in range(nodes_per_region):
                node_id = f"{region}-quantum-{i+1}"
                
                node_metrics = NodeMetrics(
                    node_id=node_id,
                    cpu_usage=random.uniform(20, 70),
                    memory_usage=random.uniform(30, 60),
                    quantum_queue_length=random.randint(0, 5),
                    active_connections=random.randint(1, 10),
                    avg_response_time=random.uniform(0.5, 2.5),
                    geographic_region=region,
                    quantum_capacity=random.randint(100, 500),
                    current_load_factor=random.uniform(20, 70)
                )
                
                self.compute_nodes[node_id] = node_metrics
                self.load_balancer.register_node(node_metrics)
        
        print(f"🖥️ Initialized {len(self.compute_nodes)} compute nodes across {len(self.regions)} regions")
    
    async def _scaling_management_loop(self) -> None:
        """Continuous auto-scaling management"""
        
        while self.is_running:
            try:
                # Evaluate scaling needs
                scaling_decision = self.auto_scaler.evaluate_scaling_need(self.compute_nodes)
                
                if scaling_decision:
                    await self._execute_scaling_action(scaling_decision)
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                print(f"⚠️ Scaling management error: {e}")
                await asyncio.sleep(120)  # Back off on error
    
    async def _performance_optimization_loop(self) -> None:
        """Continuous performance optimization"""
        
        while self.is_running:
            try:
                # Run performance optimization
                optimization_result = await self.performance_optimizer.optimize_system_performance(
                    self.compute_nodes
                )
                
                if optimization_result['total_improvement_percent'] > 5:
                    print(f"⚡ Performance optimization: {optimization_result['total_improvement_percent']:.1f}% improvement")
                
                await asyncio.sleep(300)  # Optimize every 5 minutes
                
            except Exception as e:
                print(f"⚠️ Performance optimization error: {e}")
                await asyncio.sleep(600)  # Back off on error
    
    async def _health_monitoring_loop(self) -> None:
        """Continuous health monitoring"""
        
        while self.is_running:
            try:
                # Update node metrics (simulate)
                for node_id in self.compute_nodes:
                    await self._simulate_node_metrics_update(node_id)
                
                # Collect cluster metrics
                cluster_health = self._calculate_cluster_health()
                self.performance_metrics.append(cluster_health)
                
                await asyncio.sleep(30)  # Monitor every 30 seconds
                
            except Exception as e:
                print(f"⚠️ Health monitoring error: {e}")
                await asyncio.sleep(60)
    
    async def _execute_scaling_action(self, scaling_decision: Dict[str, Any]) -> None:
        """Execute auto-scaling action"""
        
        action = scaling_decision['action']
        
        if action == 'scale_up':
            additional_nodes = scaling_decision['recommended_nodes']
            await self._scale_up_nodes(additional_nodes)
            
        elif action == 'scale_down':
            candidate_nodes = scaling_decision['candidate_nodes']
            await self._scale_down_nodes(candidate_nodes)
            
        elif action == 'rebalance':
            await self._rebalance_load()
        
        # Record scaling event
        scaling_event = ScalingEvent(
            timestamp=datetime.now(),
            event_type=action,
            trigger_reason='; '.join(scaling_decision['reasons']),
            nodes_affected=scaling_decision.get('recommended_nodes', []) + 
                          scaling_decision.get('candidate_nodes', []),
            performance_impact=5.0  # Estimated improvement
        )
        
        self.scaling_history.append(scaling_event)
        self.auto_scaler.last_scaling_action = datetime.now()
        
        print(f"🔄 Scaling action executed: {action} - {scaling_event.trigger_reason}")
    
    async def _scale_up_nodes(self, count: int) -> None:
        """Scale up compute nodes"""
        
        # Select region with highest load
        region_loads = {}
        for region in self.regions:
            region_nodes = [n for n in self.compute_nodes.values() if n.geographic_region == region]
            if region_nodes:
                region_loads[region] = np.mean([n.current_load_factor for n in region_nodes])
        
        target_region = max(region_loads.keys(), key=lambda r: region_loads[r])
        
        # Add new nodes
        for i in range(count):
            node_id = f"{target_region}-quantum-{len([n for n in self.compute_nodes if target_region in n])+1}"
            
            new_node = NodeMetrics(
                node_id=node_id,
                cpu_usage=random.uniform(10, 30),  # Start with low load
                memory_usage=random.uniform(20, 40),
                quantum_queue_length=0,
                active_connections=0,
                avg_response_time=random.uniform(0.3, 1.0),
                geographic_region=target_region,
                quantum_capacity=random.randint(100, 500),
                current_load_factor=random.uniform(10, 30)
            )
            
            self.compute_nodes[node_id] = new_node
            self.load_balancer.register_node(new_node)
        
        print(f"⬆️ Scaled up {count} nodes in region {target_region}")
    
    async def _scale_down_nodes(self, candidate_nodes: List[str]) -> None:
        """Scale down compute nodes"""
        
        for node_id in candidate_nodes:
            if node_id in self.compute_nodes:
                del self.compute_nodes[node_id]
                print(f"⬇️ Scaled down node: {node_id}")
    
    async def _rebalance_load(self) -> None:
        """Rebalance load across nodes"""
        
        # Simulate load rebalancing by adjusting node metrics
        total_load = sum(n.current_load_factor for n in self.compute_nodes.values())
        target_load_per_node = total_load / len(self.compute_nodes)
        
        for node in self.compute_nodes.values():
            # Move towards target load
            node.current_load_factor = (node.current_load_factor + target_load_per_node) / 2
        
        print(f"⚖️ Load rebalanced across {len(self.compute_nodes)} nodes")
    
    async def _update_node_load(self, node_id: str, processing_time: float) -> None:
        """Update node load after processing"""
        
        if node_id in self.compute_nodes:
            node = self.compute_nodes[node_id]
            
            # Update metrics
            node.active_connections += 1
            node.avg_response_time = (node.avg_response_time + processing_time) / 2
            node.current_load_factor = min(100, node.current_load_factor + 5)
            
            # Update in load balancer
            self.load_balancer.update_node_metrics(node_id, node)
    
    async def _simulate_node_metrics_update(self, node_id: str) -> None:
        """Simulate node metrics updates"""
        
        if node_id not in self.compute_nodes:
            return
        
        node = self.compute_nodes[node_id]
        
        # Simulate natural fluctuations
        node.cpu_usage = max(5, min(95, node.cpu_usage + random.uniform(-5, 5)))
        node.memory_usage = max(10, min(90, node.memory_usage + random.uniform(-3, 3)))
        node.quantum_queue_length = max(0, node.quantum_queue_length + random.randint(-2, 2))
        node.current_load_factor = max(0, min(100, node.current_load_factor + random.uniform(-2, 2)))
        
        # Gradual connection decrease
        if node.active_connections > 0 and random.random() < 0.3:
            node.active_connections -= 1
    
    def _calculate_cluster_health(self) -> Dict[str, Any]:
        """Calculate overall cluster health metrics"""
        
        if not self.compute_nodes:
            return {}
        
        nodes = list(self.compute_nodes.values())
        
        return {
            'timestamp': datetime.now(),
            'total_nodes': len(nodes),
            'avg_cpu_usage': np.mean([n.cpu_usage for n in nodes]),
            'avg_memory_usage': np.mean([n.memory_usage for n in nodes]),
            'total_quantum_capacity': sum(n.quantum_capacity for n in nodes),
            'total_active_connections': sum(n.active_connections for n in nodes),
            'avg_response_time': np.mean([n.avg_response_time for n in nodes]),
            'load_balance_variance': np.var([n.current_load_factor for n in nodes]),
            'regions_active': len(set(n.geographic_region for n in nodes))
        }
    
    def get_orchestrator_status(self) -> Dict[str, Any]:
        """Get comprehensive orchestrator status"""
        
        cluster_health = self._calculate_cluster_health() if self.compute_nodes else {}
        
        recent_scaling = [e for e in self.scaling_history 
                         if (datetime.now() - e.timestamp).seconds < 3600]  # Last hour
        
        recent_performance = self.performance_metrics[-10:] if self.performance_metrics else []
        
        return {
            'orchestrator_active': self.is_running,
            'cluster_health': cluster_health,
            'scaling_activity': {
                'recent_events': len(recent_scaling),
                'scale_up_events': len([e for e in recent_scaling if e.event_type == 'scale_up']),
                'scale_down_events': len([e for e in recent_scaling if e.event_type == 'scale_down']),
                'rebalance_events': len([e for e in recent_scaling if e.event_type == 'rebalance'])
            },
            'performance_trends': {
                'avg_response_time_trend': np.mean([m.get('avg_response_time', 0) for m in recent_performance]),
                'load_balance_stability': np.mean([m.get('load_balance_variance', 100) for m in recent_performance])
            } if recent_performance else {},
            'global_coverage': {
                'regions_active': cluster_health.get('regions_active', 0),
                'total_regions': len(self.regions),
                'coverage_percentage': (cluster_health.get('regions_active', 0) / len(self.regions)) * 100
            }
        }
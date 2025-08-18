"""
Advanced embedding strategies for quantum annealing hardware.

This module implements novel embedding techniques that go beyond
standard chain embedding, optimizing for quantum hardware topology
and problem structure.
"""

import numpy as np
from typing import Dict, Any, Tuple, List, Optional, Set
from dataclasses import dataclass
import logging
import networkx as nx
from abc import ABC, abstractmethod
import asyncio
from concurrent.futures import ThreadPoolExecutor

try:
    from dwave.system import DWaveSampler
    from dwave.embedding import embed_bqm, unembed_sampleset
    from dwave.embedding.utilities import edgelist_to_adjacency
    from minorminer import find_embedding
    import dwave_networkx as dnx
    DWAVE_AVAILABLE = True
except ImportError:
    DWAVE_AVAILABLE = False


@dataclass
class EmbeddingQualityMetrics:
    """Metrics for evaluating embedding quality."""
    max_chain_length: int
    avg_chain_length: float
    total_qubits_used: int
    topology_efficiency: float
    chain_break_risk: float
    embedding_overhead: float


@dataclass
class TopologyParameters:
    """Parameters for topology-aware embedding."""
    topology_type: str = "pegasus"
    prefer_short_chains: bool = True
    locality_weight: float = 0.7
    connectivity_weight: float = 0.3
    max_retries: int = 10


class TopologyAwareEmbedder:
    """
    Topology-aware embedding that optimizes for specific quantum hardware.
    
    Uses knowledge of quantum hardware topology (Chimera, Pegasus, Zephyr)
    to create embeddings that minimize chain breaks and improve solution quality.
    """
    
    def __init__(self, topology_params: TopologyParameters = None):
        self.topology_params = topology_params or TopologyParameters()
        self.logger = logging.getLogger(__name__)
        
        # Hardware topology graphs
        self._topology_graphs = {}
        self._embedding_cache = {}
        
        # Performance tracking
        self._embedding_stats = []
        
    async def find_optimal_embedding(
        self,
        problem_graph: Dict[Tuple[int, int], float],
        hardware_graph: nx.Graph = None,
        optimization_strategy: str = "topology_aware"
    ) -> Tuple[Dict[int, List[int]], EmbeddingQualityMetrics]:
        """
        Find optimal embedding using topology-aware strategies.
        
        Args:
            problem_graph: QUBO problem as graph
            hardware_graph: Target hardware topology
            optimization_strategy: Embedding strategy to use
            
        Returns:
            Tuple of (embedding, quality_metrics)
        """
        
        if not DWAVE_AVAILABLE:
            self.logger.warning("D-Wave not available, using mock embedding")
            return self._mock_embedding(problem_graph)
            
        # Get hardware graph if not provided
        if hardware_graph is None:
            hardware_graph = self._get_hardware_topology()
            
        # Convert QUBO to problem graph
        problem_nx = self._qubo_to_networkx(problem_graph)
        
        if optimization_strategy == "topology_aware":
            embedding = await self._topology_aware_embedding(problem_nx, hardware_graph)
        elif optimization_strategy == "hierarchical":
            embedding = await self._hierarchical_embedding(problem_nx, hardware_graph)
        elif optimization_strategy == "locality_optimized":
            embedding = await self._locality_optimized_embedding(problem_nx, hardware_graph)
        else:
            # Fallback to standard embedding
            embedding = await self._standard_embedding(problem_nx, hardware_graph)
            
        # Analyze embedding quality
        quality_metrics = self._analyze_embedding_quality(embedding, problem_nx, hardware_graph)
        
        # Cache successful embedding
        self._cache_embedding(problem_graph, embedding, quality_metrics)
        
        return embedding, quality_metrics
        
    async def _topology_aware_embedding(
        self,
        problem_graph: nx.Graph,
        hardware_graph: nx.Graph
    ) -> Dict[int, List[int]]:
        """Topology-aware embedding optimization."""
        
        # Analyze problem structure
        problem_analysis = self._analyze_problem_structure(problem_graph)
        hardware_analysis = self._analyze_hardware_structure(hardware_graph)
        
        # Use problem structure to guide embedding
        if problem_analysis['has_grid_structure']:
            return await self._grid_aware_embedding(problem_graph, hardware_graph)
        elif problem_analysis['has_hierarchical_structure']:
            return await self._hierarchical_embedding(problem_graph, hardware_graph)
        else:
            return await self._general_topology_embedding(problem_graph, hardware_graph)
            
    async def _grid_aware_embedding(
        self,
        problem_graph: nx.Graph,
        hardware_graph: nx.Graph
    ) -> Dict[int, List[int]]:
        """Embedding optimized for grid-like problem structures."""
        
        # Detect grid dimensions
        grid_dims = self._detect_grid_dimensions(problem_graph)
        
        # Map to hardware grid structure
        if self.topology_params.topology_type == "pegasus":
            hardware_dims = self._get_pegasus_grid_mapping(hardware_graph)
        elif self.topology_params.topology_type == "chimera":
            hardware_dims = self._get_chimera_grid_mapping(hardware_graph)
        else:
            hardware_dims = self._get_generic_grid_mapping(hardware_graph)
            
        # Create grid-aware embedding
        embedding = {}
        
        for logical_var in problem_graph.nodes():
            # Map logical variable to grid position
            grid_pos = self._logical_to_grid_position(logical_var, grid_dims)
            
            # Find best hardware location for this grid position
            hardware_qubits = self._find_hardware_grid_location(grid_pos, hardware_dims, hardware_graph)
            
            embedding[logical_var] = hardware_qubits
            
        # Optimize chain lengths and connectivity
        embedding = self._optimize_embedding_chains(embedding, problem_graph, hardware_graph)
        
        return embedding
        
    async def _hierarchical_embedding(
        self,
        problem_graph: nx.Graph,
        hardware_graph: nx.Graph
    ) -> Dict[int, List[int]]:
        """Hierarchical embedding for problems with natural hierarchy."""
        
        # Decompose problem into hierarchical levels
        hierarchy_levels = self._decompose_into_hierarchy(problem_graph)
        
        embedding = {}
        used_qubits = set()
        
        # Embed each hierarchy level
        for level, variables in hierarchy_levels.items():
            level_subgraph = problem_graph.subgraph(variables)
            
            # Find available hardware region for this level
            available_hardware = self._get_available_hardware_region(
                hardware_graph, used_qubits, len(variables)
            )
            
            # Embed this level
            level_embedding = await self._embed_subgraph(
                level_subgraph, available_hardware
            )
            
            # Merge into main embedding
            embedding.update(level_embedding)
            
            # Mark qubits as used
            for qubit_list in level_embedding.values():
                used_qubits.update(qubit_list)
                
        return embedding
        
    async def _locality_optimized_embedding(
        self,
        problem_graph: nx.Graph,
        hardware_graph: nx.Graph
    ) -> Dict[int, List[int]]:
        """Embedding optimized for locality preservation."""
        
        # Use community detection to find tightly connected variable groups
        communities = self._detect_variable_communities(problem_graph)
        
        embedding = {}
        used_qubits = set()
        
        for community in communities:
            community_subgraph = problem_graph.subgraph(community)
            
            # Find compact hardware region for this community
            compact_region = self._find_compact_hardware_region(
                hardware_graph, used_qubits, len(community)
            )
            
            # Embed community with minimal chain lengths
            community_embedding = await self._embed_compact_subgraph(
                community_subgraph, compact_region
            )
            
            embedding.update(community_embedding)
            used_qubits.update([q for qubit_list in community_embedding.values() for q in qubit_list])
            
        return embedding
        
    def _analyze_problem_structure(self, problem_graph: nx.Graph) -> Dict[str, Any]:
        """Analyze the structure of the problem graph."""
        
        analysis = {
            'node_count': len(problem_graph.nodes()),
            'edge_count': len(problem_graph.edges()),
            'avg_degree': np.mean([d for n, d in problem_graph.degree()]),
            'clustering_coefficient': nx.average_clustering(problem_graph),
            'has_grid_structure': False,
            'has_hierarchical_structure': False
        }
        
        # Detect grid structure
        if self._is_grid_like(problem_graph):
            analysis['has_grid_structure'] = True
            analysis['grid_dimensions'] = self._detect_grid_dimensions(problem_graph)
            
        # Detect hierarchical structure
        if self._is_hierarchical(problem_graph):
            analysis['has_hierarchical_structure'] = True
            analysis['hierarchy_depth'] = self._calculate_hierarchy_depth(problem_graph)
            
        return analysis
        
    def _is_grid_like(self, graph: nx.Graph) -> bool:
        """Check if graph has grid-like structure."""
        degrees = [d for n, d in graph.degree()]
        
        # Grid graphs have mostly degree 2-4 nodes
        low_degree_fraction = sum(1 for d in degrees if d <= 4) / len(degrees)
        
        # Check for regular structure
        degree_variance = np.var(degrees)
        
        return low_degree_fraction > 0.8 and degree_variance < 2.0
        
    def _is_hierarchical(self, graph: nx.Graph) -> bool:
        """Check if graph has hierarchical structure."""
        # Use clustering coefficient variation as indicator
        clustering_coeffs = nx.clustering(graph)
        clustering_variance = np.var(list(clustering_coeffs.values()))
        
        # Hierarchical graphs have high clustering coefficient variation
        return clustering_variance > 0.1
        
    def _detect_grid_dimensions(self, graph: nx.Graph) -> Tuple[int, int]:
        """Detect dimensions of grid-like graph."""
        n_nodes = len(graph.nodes())
        
        # Try common grid dimensions
        for width in range(2, int(np.sqrt(n_nodes)) + 2):
            if n_nodes % width == 0:
                height = n_nodes // width
                if self._validate_grid_dimensions(graph, width, height):
                    return (width, height)
                    
        # Fallback to square-ish grid
        side = int(np.sqrt(n_nodes))
        return (side, side)
        
    def _validate_grid_dimensions(self, graph: nx.Graph, width: int, height: int) -> bool:
        """Validate if graph matches given grid dimensions."""
        if width * height != len(graph.nodes()):
            return False
            
        # Check if connectivity matches grid pattern
        expected_edges = (width - 1) * height + width * (height - 1)
        actual_edges = len(graph.edges())
        
        # Allow some tolerance for missing edges
        return abs(expected_edges - actual_edges) <= 0.1 * expected_edges
        
    def _decompose_into_hierarchy(self, graph: nx.Graph) -> Dict[int, List[int]]:
        """Decompose graph into hierarchical levels."""
        # Use betweenness centrality to identify hierarchy levels
        centrality = nx.betweenness_centrality(graph)
        
        # Sort nodes by centrality
        sorted_nodes = sorted(graph.nodes(), key=lambda n: centrality[n], reverse=True)
        
        # Create hierarchy levels
        levels = {}
        nodes_per_level = max(1, len(sorted_nodes) // 3)
        
        for i, level in enumerate(range(3)):
            start_idx = level * nodes_per_level
            end_idx = min((level + 1) * nodes_per_level, len(sorted_nodes))
            levels[level] = sorted_nodes[start_idx:end_idx]
            
        return levels
        
    def _detect_variable_communities(self, graph: nx.Graph) -> List[List[int]]:
        """Detect communities of tightly connected variables."""
        try:
            import community as community_louvain
            partition = community_louvain.best_partition(graph)
            
            # Convert partition to community lists
            communities = {}
            for node, community_id in partition.items():
                if community_id not in communities:
                    communities[community_id] = []
                communities[community_id].append(node)
                
            return list(communities.values())
            
        except ImportError:
            # Fallback: use simple clustering based on connectivity
            return self._simple_community_detection(graph)
            
    def _simple_community_detection(self, graph: nx.Graph) -> List[List[int]]:
        """Simple community detection fallback."""
        communities = []
        unvisited = set(graph.nodes())
        
        while unvisited:
            # Start new community with highest degree unvisited node
            start_node = max(unvisited, key=lambda n: graph.degree(n))
            community = [start_node]
            unvisited.remove(start_node)
            
            # Add neighbors with high connectivity
            for neighbor in graph.neighbors(start_node):
                if neighbor in unvisited:
                    # Add neighbor if it has strong connection to community
                    community_connections = sum(1 for c in community if graph.has_edge(neighbor, c))
                    if community_connections >= len(community) * 0.5:
                        community.append(neighbor)
                        unvisited.remove(neighbor)
                        
            communities.append(community)
            
        return communities
        
    async def _embed_subgraph(
        self,
        subgraph: nx.Graph,
        hardware_region: nx.Graph
    ) -> Dict[int, List[int]]:
        """Embed subgraph into hardware region."""
        
        if not DWAVE_AVAILABLE:
            return {node: [node] for node in subgraph.nodes()}
            
        # Use minorminer to find embedding
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    find_embedding,
                    subgraph.edges(),
                    hardware_region.edges(),
                    tries=self.topology_params.max_retries
                )
                embedding = await asyncio.wrap_future(future)
                
            return embedding if embedding else {}
            
        except Exception as e:
            self.logger.warning(f"Subgraph embedding failed: {e}")
            return {}
            
    def _get_hardware_topology(self) -> nx.Graph:
        """Get hardware topology graph."""
        
        if not DWAVE_AVAILABLE:
            # Mock hardware graph for testing
            return nx.grid_2d_graph(4, 4)
            
        try:
            sampler = DWaveSampler()
            return sampler.to_networkx_graph()
            
        except Exception as e:
            self.logger.warning(f"Could not get hardware topology: {e}")
            # Return mock topology
            if self.topology_params.topology_type == "pegasus":
                return dnx.pegasus_graph(4)
            elif self.topology_params.topology_type == "chimera":
                return dnx.chimera_graph(4)
            else:
                return nx.grid_2d_graph(8, 8)
                
    def _qubo_to_networkx(self, qubo: Dict[Tuple[int, int], float]) -> nx.Graph:
        """Convert QUBO to NetworkX graph."""
        graph = nx.Graph()
        
        # Add all variables as nodes
        variables = set()
        for (i, j) in qubo.keys():
            variables.add(i)
            variables.add(j)
            
        graph.add_nodes_from(variables)
        
        # Add edges for quadratic terms
        for (i, j), coeff in qubo.items():
            if i != j and coeff != 0:
                graph.add_edge(i, j, weight=abs(coeff))
                
        return graph
        
    def _analyze_embedding_quality(
        self,
        embedding: Dict[int, List[int]],
        problem_graph: nx.Graph,
        hardware_graph: nx.Graph
    ) -> EmbeddingQualityMetrics:
        """Analyze the quality of an embedding."""
        
        if not embedding:
            return EmbeddingQualityMetrics(0, 0.0, 0, 0.0, 1.0, 1.0)
            
        chain_lengths = [len(chain) for chain in embedding.values()]
        total_qubits = sum(chain_lengths)
        
        max_chain_length = max(chain_lengths) if chain_lengths else 0
        avg_chain_length = np.mean(chain_lengths) if chain_lengths else 0.0
        
        # Topology efficiency: ratio of logical to physical qubits
        topology_efficiency = len(embedding) / total_qubits if total_qubits > 0 else 0.0
        
        # Chain break risk: longer chains have higher risk
        chain_break_risk = max_chain_length / 10.0  # Normalized risk estimate
        
        # Embedding overhead
        embedding_overhead = (total_qubits - len(embedding)) / len(embedding) if embedding else 1.0
        
        return EmbeddingQualityMetrics(
            max_chain_length=max_chain_length,
            avg_chain_length=avg_chain_length,
            total_qubits_used=total_qubits,
            topology_efficiency=topology_efficiency,
            chain_break_risk=min(1.0, chain_break_risk),
            embedding_overhead=embedding_overhead
        )
        
    def _mock_embedding(self, problem_graph: Dict[Tuple[int, int], float]) -> Tuple[Dict[int, List[int]], EmbeddingQualityMetrics]:
        """Mock embedding for testing without D-Wave access."""
        variables = set()
        for (i, j) in problem_graph.keys():
            variables.add(i)
            variables.add(j)
            
        # Simple 1:1 mapping
        embedding = {var: [var] for var in variables}
        
        metrics = EmbeddingQualityMetrics(
            max_chain_length=1,
            avg_chain_length=1.0,
            total_qubits_used=len(variables),
            topology_efficiency=1.0,
            chain_break_risk=0.0,
            embedding_overhead=0.0
        )
        
        return embedding, metrics


class DynamicEmbeddingOptimizer:
    """
    Dynamic embedding optimization that adapts based on solution quality feedback.
    
    Learns from previous embeddings to improve future embedding decisions,
    using reinforcement learning principles.
    """
    
    def __init__(self, learning_rate: float = 0.1):
        self.learning_rate = learning_rate
        self.embedding_performance_history = []
        self.strategy_scores = {
            'topology_aware': 0.0,
            'hierarchical': 0.0,
            'locality_optimized': 0.0,
            'standard': 0.0
        }
        
        self.logger = logging.getLogger(__name__)
        
    async def optimize_embedding_dynamically(
        self,
        problem_graph: Dict[Tuple[int, int], float],
        solution_quality_feedback: Optional[float] = None
    ) -> Tuple[Dict[int, List[int]], str]:
        """
        Optimize embedding using dynamic strategy selection.
        
        Args:
            problem_graph: QUBO problem graph
            solution_quality_feedback: Quality of previous solution (0-1)
            
        Returns:
            Tuple of (embedding, strategy_used)
        """
        
        # Update strategy scores based on feedback
        if solution_quality_feedback is not None and self.embedding_performance_history:
            last_strategy = self.embedding_performance_history[-1]['strategy']
            self._update_strategy_score(last_strategy, solution_quality_feedback)
            
        # Select best strategy based on current scores
        selected_strategy = self._select_optimal_strategy(problem_graph)
        
        # Create embedder and find embedding
        embedder = TopologyAwareEmbedder()
        embedding, quality_metrics = await embedder.find_optimal_embedding(
            problem_graph, optimization_strategy=selected_strategy
        )
        
        # Record this embedding attempt
        self.embedding_performance_history.append({
            'strategy': selected_strategy,
            'quality_metrics': quality_metrics,
            'problem_size': len(set(i for (i, j) in problem_graph.keys()) | set(j for (i, j) in problem_graph.keys()))
        })
        
        return embedding, selected_strategy
        
    def _update_strategy_score(self, strategy: str, quality_feedback: float) -> None:
        """Update strategy score based on solution quality feedback."""
        
        if strategy in self.strategy_scores:
            # Exponential moving average update
            current_score = self.strategy_scores[strategy]
            self.strategy_scores[strategy] = (
                (1 - self.learning_rate) * current_score + 
                self.learning_rate * quality_feedback
            )
            
            self.logger.debug(f"Updated {strategy} score: {current_score:.3f} -> {self.strategy_scores[strategy]:.3f}")
            
    def _select_optimal_strategy(self, problem_graph: Dict[Tuple[int, int], float]) -> str:
        """Select optimal embedding strategy based on problem characteristics and scores."""
        
        # Analyze problem characteristics
        variables = set()
        for (i, j) in problem_graph.keys():
            variables.add(i)
            variables.add(j)
            
        problem_size = len(variables)
        edge_count = sum(1 for (i, j) in problem_graph.keys() if i != j)
        density = edge_count / (problem_size * (problem_size - 1) / 2) if problem_size > 1 else 0
        
        # Weight strategies based on problem characteristics
        strategy_weights = {}
        
        # Topology-aware is good for structured problems
        if density < 0.3:  # Sparse problems
            strategy_weights['topology_aware'] = self.strategy_scores['topology_aware'] * 1.2
        else:
            strategy_weights['topology_aware'] = self.strategy_scores['topology_aware']
            
        # Hierarchical is good for large problems
        if problem_size > 100:
            strategy_weights['hierarchical'] = self.strategy_scores['hierarchical'] * 1.3
        else:
            strategy_weights['hierarchical'] = self.strategy_scores['hierarchical']
            
        # Locality-optimized is good for dense problems
        if density > 0.5:
            strategy_weights['locality_optimized'] = self.strategy_scores['locality_optimized'] * 1.1
        else:
            strategy_weights['locality_optimized'] = self.strategy_scores['locality_optimized']
            
        # Standard is the baseline
        strategy_weights['standard'] = self.strategy_scores['standard']
        
        # Select strategy with highest weighted score
        best_strategy = max(strategy_weights.keys(), key=lambda k: strategy_weights[k])
        
        self.logger.info(f"Selected embedding strategy: {best_strategy} (score: {strategy_weights[best_strategy]:.3f})")
        return best_strategy
        
    def get_strategy_performance_summary(self) -> Dict[str, Any]:
        """Get summary of strategy performance."""
        
        summary = {
            'strategy_scores': self.strategy_scores.copy(),
            'total_embeddings': len(self.embedding_performance_history),
            'strategy_usage': {}
        }
        
        # Count strategy usage
        for record in self.embedding_performance_history:
            strategy = record['strategy']
            summary['strategy_usage'][strategy] = summary['strategy_usage'].get(strategy, 0) + 1
            
        # Calculate average quality metrics by strategy
        strategy_metrics = {}
        for strategy in self.strategy_scores.keys():
            strategy_records = [r for r in self.embedding_performance_history if r['strategy'] == strategy]
            
            if strategy_records:
                avg_max_chain = np.mean([r['quality_metrics'].max_chain_length for r in strategy_records])
                avg_efficiency = np.mean([r['quality_metrics'].topology_efficiency for r in strategy_records])
                
                strategy_metrics[strategy] = {
                    'avg_max_chain_length': avg_max_chain,
                    'avg_topology_efficiency': avg_efficiency,
                    'usage_count': len(strategy_records)
                }
                
        summary['strategy_performance'] = strategy_metrics
        
        return summary
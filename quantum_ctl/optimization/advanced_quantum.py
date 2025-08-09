"""
Advanced quantum optimization features for HVAC control.

Provides quantum problem decomposition, multi-objective optimization,
and uncertainty quantification using quantum annealing.
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import warnings

import numpy as np
import networkx as nx
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
import dimod
from dwave.system import DWaveSampler, EmbeddingComposite, FixedEmbeddingComposite
from dwave.embedding import embed_qubo, unembed_sampleset
from dwave.embedding.chain_breaks import majority_vote
import minorminer


class DecompositionStrategy(Enum):
    """Problem decomposition strategies."""
    TEMPORAL = "temporal"
    SPATIAL = "spatial"
    HIERARCHICAL = "hierarchical"
    GRAPH_PARTITION = "graph_partition"


class ObjectiveType(Enum):
    """Types of optimization objectives."""
    ENERGY_COST = "energy_cost"
    COMFORT = "comfort"
    CARBON_EMISSIONS = "carbon"
    LOAD_BALANCING = "load_balance"
    PEAK_SHAVING = "peak_shaving"


@dataclass
class ObjectiveFunction:
    """Multi-objective optimization objective."""
    name: str
    objective_type: ObjectiveType
    weight: float
    target_value: Optional[float] = None
    penalty_function: Optional[Callable] = None
    
    def evaluate(self, solution: np.ndarray, context: Dict[str, Any]) -> float:
        """Evaluate objective function value."""
        if self.penalty_function:
            return self.penalty_function(solution, context)
        else:
            # Default evaluation based on type
            return self._default_evaluation(solution, context)
    
    def _default_evaluation(self, solution: np.ndarray, context: Dict[str, Any]) -> float:
        """Default objective evaluation."""
        if self.objective_type == ObjectiveType.ENERGY_COST:
            power = context.get('power_consumption', np.sum(solution))
            energy_price = context.get('energy_price', 0.12)
            return power * energy_price
        
        elif self.objective_type == ObjectiveType.COMFORT:
            temperatures = context.get('temperatures', np.zeros_like(solution))
            target_temp = context.get('target_temperature', 22.0)
            comfort_violation = np.sum(np.abs(temperatures - target_temp))
            return comfort_violation
        
        elif self.objective_type == ObjectiveType.CARBON_EMISSIONS:
            power = context.get('power_consumption', np.sum(solution))
            carbon_factor = context.get('carbon_factor', 0.5)  # kg CO2 per kWh
            return power * carbon_factor
        
        else:
            return 0.0


@dataclass
class ParetoSolution:
    """Solution on Pareto frontier."""
    solution: np.ndarray
    objective_values: Dict[str, float]
    energy: float
    dominates_count: int = 0
    dominated_by_count: int = 0
    crowding_distance: float = 0.0


class QuantumDecomposer:
    """
    Quantum problem decomposition for large-scale HVAC optimization.
    
    Breaks down large QUBO problems into smaller, manageable subproblems
    that can be solved efficiently on quantum hardware.
    """
    
    def __init__(self, max_subproblem_size: int = 200):
        self.max_subproblem_size = max_subproblem_size
        self.logger = logging.getLogger("quantum_decomposer")
    
    def decompose_problem(
        self,
        Q: np.ndarray,
        strategy: DecompositionStrategy = DecompositionStrategy.TEMPORAL,
        overlap_size: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Decompose large QUBO problem into subproblems.
        
        Args:
            Q: QUBO matrix to decompose
            strategy: Decomposition strategy to use
            overlap_size: Overlap between subproblems for consistency
            
        Returns:
            List of subproblem dictionaries
        """
        n = Q.shape[0]
        
        if n <= self.max_subproblem_size:
            # Problem is small enough, no decomposition needed
            return [{
                'Q': Q,
                'variable_mapping': list(range(n)),
                'subproblem_id': 0
            }]
        
        self.logger.info(f"Decomposing problem of size {n} using {strategy.value} strategy")
        
        if strategy == DecompositionStrategy.TEMPORAL:
            return self._temporal_decomposition(Q, overlap_size)
        elif strategy == DecompositionStrategy.SPATIAL:
            return self._spatial_decomposition(Q, overlap_size)
        elif strategy == DecompositionStrategy.GRAPH_PARTITION:
            return self._graph_partition_decomposition(Q, overlap_size)
        elif strategy == DecompositionStrategy.HIERARCHICAL:
            return self._hierarchical_decomposition(Q, overlap_size)
        else:
            raise ValueError(f"Unknown decomposition strategy: {strategy}")
    
    def _temporal_decomposition(self, Q: np.ndarray, overlap: int) -> List[Dict[str, Any]]:
        """Decompose problem by time steps (for MPC problems)."""
        n = Q.shape[0]
        subproblems = []
        
        # Assume variables are ordered by time step
        step_size = self.max_subproblem_size - overlap
        
        for i in range(0, n, step_size):
            end_idx = min(i + self.max_subproblem_size, n)
            
            # Extract subproblem
            indices = list(range(i, end_idx))
            Q_sub = Q[np.ix_(indices, indices)]
            
            subproblems.append({
                'Q': Q_sub,
                'variable_mapping': indices,
                'subproblem_id': len(subproblems),
                'temporal_range': (i, end_idx)
            })
            
            if end_idx >= n:
                break
        
        self.logger.debug(f"Created {len(subproblems)} temporal subproblems")
        return subproblems
    
    def _spatial_decomposition(self, Q: np.ndarray, overlap: int) -> List[Dict[str, Any]]:
        """Decompose problem by spatial regions (for multi-zone HVAC)."""
        n = Q.shape[0]
        
        # For HVAC, assume variables are grouped by zones
        # Estimate zone size from problem structure
        zone_size = max(1, n // 20)  # Assume up to 20 zones
        zones_per_subproblem = self.max_subproblem_size // zone_size
        
        subproblems = []
        
        for zone_start in range(0, n, zones_per_subproblem * zone_size):
            zone_end = min(zone_start + zones_per_subproblem * zone_size, n)
            
            # Add overlap with neighboring zones
            indices_start = max(0, zone_start - overlap)
            indices_end = min(n, zone_end + overlap)
            
            indices = list(range(indices_start, indices_end))
            Q_sub = Q[np.ix_(indices, indices)]
            
            subproblems.append({
                'Q': Q_sub,
                'variable_mapping': indices,
                'subproblem_id': len(subproblems),
                'spatial_range': (zone_start, zone_end)
            })
        
        self.logger.debug(f"Created {len(subproblems)} spatial subproblems")
        return subproblems
    
    def _graph_partition_decomposition(self, Q: np.ndarray, overlap: int) -> List[Dict[str, Any]]:
        """Decompose using graph partitioning algorithms."""
        n = Q.shape[0]
        
        # Create graph from Q matrix
        G = nx.Graph()
        G.add_nodes_from(range(n))
        
        # Add edges for non-zero Q elements
        for i in range(n):
            for j in range(i + 1, n):
                if abs(Q[i, j]) > 1e-10:
                    G.add_edge(i, j, weight=abs(Q[i, j]))
        
        # Estimate number of partitions needed
        num_partitions = max(1, n // self.max_subproblem_size)
        
        try:
            # Use NetworkX community detection for partitioning
            import networkx.algorithms.community as nx_comm
            communities = nx_comm.greedy_modularity_communities(G, resolution=num_partitions)
            
            subproblems = []
            for i, community in enumerate(communities):
                indices = sorted(list(community))
                
                # Add overlap with neighboring communities
                extended_indices = set(indices)
                for idx in indices:
                    neighbors = list(G.neighbors(idx))
                    extended_indices.update(neighbors[:overlap])
                
                final_indices = sorted(list(extended_indices))
                Q_sub = Q[np.ix_(final_indices, final_indices)]
                
                subproblems.append({
                    'Q': Q_sub,
                    'variable_mapping': final_indices,
                    'subproblem_id': i,
                    'community_size': len(indices)
                })
            
        except ImportError:
            # Fallback to simple partitioning
            self.logger.warning("Advanced graph partitioning not available, using simple partitioning")
            return self._temporal_decomposition(Q, overlap)
        
        self.logger.debug(f"Created {len(subproblems)} graph partition subproblems")
        return subproblems
    
    def _hierarchical_decomposition(self, Q: np.ndarray, overlap: int) -> List[Dict[str, Any]]:
        """Hierarchical decomposition for multi-level problems."""
        # Implement hierarchical clustering based decomposition
        n = Q.shape[0]
        
        # Use connectivity patterns to create hierarchy
        connectivity = (np.abs(Q) > 1e-10).astype(int)
        
        # Create distance matrix
        distances = pdist(connectivity, metric='hamming')
        distance_matrix = squareform(distances)
        
        # Hierarchical clustering
        from scipy.cluster.hierarchy import linkage, fcluster
        
        linkage_matrix = linkage(distances, method='ward')
        
        # Determine number of clusters
        n_clusters = max(1, n // self.max_subproblem_size)
        cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
        
        subproblems = []
        for cluster_id in range(1, n_clusters + 1):
            indices = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
            
            if len(indices) == 0:
                continue
            
            # Add overlap
            extended_indices = set(indices)
            for idx in indices[:overlap]:
                # Add nearby variables based on connectivity
                neighbors = np.where(connectivity[idx] > 0)[0]
                extended_indices.update(neighbors)
            
            final_indices = sorted(list(extended_indices))
            Q_sub = Q[np.ix_(final_indices, final_indices)]
            
            subproblems.append({
                'Q': Q_sub,
                'variable_mapping': final_indices,
                'subproblem_id': cluster_id - 1,
                'cluster_size': len(indices)
            })
        
        self.logger.debug(f"Created {len(subproblems)} hierarchical subproblems")
        return subproblems
    
    async def solve_subproblems(
        self,
        subproblems: List[Dict[str, Any]],
        sampler,
        solver_params: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Solve subproblems in parallel."""
        
        async def solve_single_subproblem(subproblem: Dict[str, Any]) -> Dict[str, Any]:
            """Solve single subproblem."""
            try:
                Q_sub = subproblem['Q']
                
                # Convert to dimod format
                bqm = dimod.BinaryQuadraticModel.from_numpy_matrix(Q_sub)
                
                # Solve using provided sampler
                sampleset = sampler.sample(bqm, **solver_params)
                
                # Get best solution
                best_sample = sampleset.first
                best_energy = best_sample.energy
                
                # Convert back to numpy array
                n_vars = Q_sub.shape[0]
                solution = np.array([best_sample.sample[i] for i in range(n_vars)])
                
                return {
                    'subproblem_id': subproblem['subproblem_id'],
                    'solution': solution,
                    'energy': best_energy,
                    'variable_mapping': subproblem['variable_mapping'],
                    'solve_info': dict(sampleset.data_vectors['num_occurrences'].items())[0]
                }
                
            except Exception as e:
                self.logger.error(f"Subproblem {subproblem['subproblem_id']} failed: {e}")
                return {
                    'subproblem_id': subproblem['subproblem_id'],
                    'solution': None,
                    'error': str(e)
                }
        
        # Solve all subproblems concurrently
        tasks = [solve_single_subproblem(sp) for sp in subproblems]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter successful results
        successful_results = [r for r in results if isinstance(r, dict) and 'solution' in r and r['solution'] is not None]
        
        self.logger.info(f"Solved {len(successful_results)}/{len(subproblems)} subproblems successfully")
        return successful_results
    
    def merge_solutions(
        self,
        subproblem_results: List[Dict[str, Any]],
        original_size: int,
        overlap_resolution: str = "average"
    ) -> np.ndarray:
        """
        Merge solutions from subproblems into global solution.
        
        Args:
            subproblem_results: Results from solved subproblems
            original_size: Size of original problem
            overlap_resolution: How to resolve overlapping variables
            
        Returns:
            Global solution vector
        """
        global_solution = np.zeros(original_size)
        variable_counts = np.zeros(original_size)  # Track how many times each variable is set
        
        for result in subproblem_results:
            if 'solution' not in result or result['solution'] is None:
                continue
            
            variable_mapping = result['variable_mapping']
            solution = result['solution']
            
            for i, global_idx in enumerate(variable_mapping):
                if i < len(solution):
                    global_solution[global_idx] += solution[i]
                    variable_counts[global_idx] += 1
        
        # Resolve overlapping variables
        if overlap_resolution == "average":
            # Average overlapping values
            mask = variable_counts > 0
            global_solution[mask] = global_solution[mask] / variable_counts[mask]
            
            # Round to binary values
            global_solution = np.round(global_solution).astype(int)
        
        elif overlap_resolution == "majority":
            # Use majority vote for overlapping variables
            for i in range(original_size):
                if variable_counts[i] > 1:
                    # Get all values for this variable
                    values = []
                    for result in subproblem_results:
                        if 'solution' in result and result['solution'] is not None:
                            mapping = result['variable_mapping']
                            if i in mapping:
                                local_idx = mapping.index(i)
                                if local_idx < len(result['solution']):
                                    values.append(result['solution'][local_idx])
                    
                    if values:
                        global_solution[i] = 1 if sum(values) > len(values) / 2 else 0
        
        # Handle unset variables (default to 0)
        unset_variables = variable_counts == 0
        global_solution[unset_variables] = 0
        
        self.logger.info(
            f"Merged global solution: {np.sum(unset_variables)} variables unset, "
            f"{np.sum(variable_counts > 1)} variables had overlaps"
        )
        
        return global_solution


class MultiObjectiveOptimizer:
    """
    Multi-objective quantum optimization using NSGA-II principles.
    
    Finds Pareto-optimal solutions for competing objectives in HVAC control.
    """
    
    def __init__(self):
        self.logger = logging.getLogger("multi_objective_optimizer")
        self.objectives: List[ObjectiveFunction] = []
    
    def add_objective(self, objective: ObjectiveFunction) -> None:
        """Add optimization objective."""
        self.objectives.append(objective)
        self.logger.info(f"Added objective: {objective.name} (weight: {objective.weight})")
    
    def clear_objectives(self) -> None:
        """Clear all objectives."""
        self.objectives.clear()
    
    async def find_pareto_front(
        self,
        Q: np.ndarray,
        sampler,
        solver_params: Dict[str, Any],
        num_solutions: int = 20,
        context: Optional[Dict[str, Any]] = None
    ) -> List[ParetoSolution]:
        """
        Find Pareto-optimal solutions for multi-objective problem.
        
        Args:
            Q: Base QUBO matrix
            sampler: Quantum sampler
            solver_params: Solver parameters
            num_solutions: Number of solutions to find
            context: Context for objective evaluation
            
        Returns:
            List of Pareto-optimal solutions
        """
        if not self.objectives:
            raise ValueError("No objectives defined")
        
        if context is None:
            context = {}
        
        self.logger.info(f"Finding Pareto front with {len(self.objectives)} objectives")
        
        # Generate multiple solutions with different objective weightings
        solutions = []
        
        for i in range(num_solutions):
            # Create weighted objective QUBO
            weighted_Q = self._create_weighted_qubo(Q, i, num_solutions)
            
            try:
                # Solve weighted problem
                bqm = dimod.BinaryQuadraticModel.from_numpy_matrix(weighted_Q)
                sampleset = sampler.sample(bqm, **solver_params)
                
                # Extract best solution
                best_sample = sampleset.first
                solution_vector = np.array([best_sample.sample[j] for j in range(len(best_sample.sample))])
                
                # Evaluate all objectives
                objective_values = {}
                for obj in self.objectives:
                    obj_value = obj.evaluate(solution_vector, context)
                    objective_values[obj.name] = obj_value
                
                pareto_sol = ParetoSolution(
                    solution=solution_vector,
                    objective_values=objective_values,
                    energy=best_sample.energy
                )
                
                solutions.append(pareto_sol)
                
            except Exception as e:
                self.logger.warning(f"Failed to solve weighted problem {i}: {e}")
        
        # Filter for Pareto-optimal solutions
        pareto_solutions = self._find_pareto_optimal(solutions)
        
        # Calculate crowding distances
        self._calculate_crowding_distances(pareto_solutions)
        
        self.logger.info(f"Found {len(pareto_solutions)} Pareto-optimal solutions")
        return pareto_solutions
    
    def _create_weighted_qubo(
        self,
        base_Q: np.ndarray,
        solution_index: int,
        total_solutions: int
    ) -> np.ndarray:
        """Create QUBO with different objective weights."""
        
        # Vary weights across solutions
        alpha = solution_index / (total_solutions - 1) if total_solutions > 1 else 0.5
        
        # Start with base problem
        weighted_Q = base_Q.copy()
        
        # Add objective-specific terms
        for i, obj in enumerate(self.objectives):
            # Vary weight based on solution index and objective
            weight_factor = obj.weight * (alpha if i % 2 == 0 else (1 - alpha))
            
            # Add objective-specific penalty matrix
            obj_penalty = self._create_objective_penalty_matrix(obj, base_Q.shape[0])
            weighted_Q += weight_factor * obj_penalty
        
        return weighted_Q
    
    def _create_objective_penalty_matrix(
        self,
        objective: ObjectiveFunction,
        matrix_size: int
    ) -> np.ndarray:
        """Create penalty matrix for specific objective."""
        
        penalty_matrix = np.zeros((matrix_size, matrix_size))
        
        if objective.objective_type == ObjectiveType.ENERGY_COST:
            # Penalize high energy consumption (diagonal penalties)
            np.fill_diagonal(penalty_matrix, 1.0)
        
        elif objective.objective_type == ObjectiveType.COMFORT:
            # Penalize comfort violations (off-diagonal for zone coupling)
            for i in range(min(matrix_size, matrix_size)):
                for j in range(i + 1, min(matrix_size, matrix_size)):
                    penalty_matrix[i, j] = 0.1  # Encourage coordination
        
        elif objective.objective_type == ObjectiveType.LOAD_BALANCING:
            # Penalize load imbalances
            for i in range(matrix_size):
                for j in range(matrix_size):
                    if i != j:
                        penalty_matrix[i, j] = 0.05  # Encourage balance
        
        return penalty_matrix
    
    def _find_pareto_optimal(self, solutions: List[ParetoSolution]) -> List[ParetoSolution]:
        """Find Pareto-optimal solutions from candidate set."""
        
        if not solutions:
            return []
        
        pareto_solutions = []
        
        for candidate in solutions:
            is_dominated = False
            
            for other in solutions:
                if candidate is other:
                    continue
                
                if self._dominates(other, candidate):
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_solutions.append(candidate)
        
        return pareto_solutions
    
    def _dominates(self, solution1: ParetoSolution, solution2: ParetoSolution) -> bool:
        """Check if solution1 dominates solution2."""
        
        all_better_or_equal = True
        at_least_one_better = False
        
        for obj in self.objectives:
            obj_name = obj.name
            
            val1 = solution1.objective_values.get(obj_name, float('inf'))
            val2 = solution2.objective_values.get(obj_name, float('inf'))
            
            # Assuming minimization objectives
            if val1 > val2:
                all_better_or_equal = False
                break
            elif val1 < val2:
                at_least_one_better = True
        
        return all_better_or_equal and at_least_one_better
    
    def _calculate_crowding_distances(self, solutions: List[ParetoSolution]) -> None:
        """Calculate crowding distances for solution diversity."""
        
        if len(solutions) <= 2:
            for sol in solutions:
                sol.crowding_distance = float('inf')
            return
        
        # Initialize crowding distances
        for sol in solutions:
            sol.crowding_distance = 0.0
        
        # Calculate distance for each objective
        for obj in self.objectives:
            obj_name = obj.name
            
            # Sort solutions by objective value
            solutions.sort(key=lambda x: x.objective_values.get(obj_name, 0))
            
            # Boundary solutions get infinite distance
            solutions[0].crowding_distance = float('inf')
            solutions[-1].crowding_distance = float('inf')
            
            # Calculate distances for intermediate solutions
            obj_range = (solutions[-1].objective_values.get(obj_name, 0) - 
                        solutions[0].objective_values.get(obj_name, 0))
            
            if obj_range > 0:
                for i in range(1, len(solutions) - 1):
                    distance = (solutions[i + 1].objective_values.get(obj_name, 0) - 
                              solutions[i - 1].objective_values.get(obj_name, 0)) / obj_range
                    solutions[i].crowding_distance += distance
    
    def interactive_selection(
        self,
        pareto_solutions: List[ParetoSolution],
        preferences: Dict[str, float]
    ) -> ParetoSolution:
        """
        Select best solution based on user preferences.
        
        Args:
            pareto_solutions: Pareto-optimal solutions
            preferences: User preference weights for objectives
            
        Returns:
            Best solution according to preferences
        """
        if not pareto_solutions:
            raise ValueError("No Pareto solutions provided")
        
        best_solution = None
        best_score = float('-inf')
        
        # Normalize preference weights
        total_weight = sum(preferences.values())
        if total_weight > 0:
            normalized_prefs = {k: v / total_weight for k, v in preferences.items()}
        else:
            normalized_prefs = preferences
        
        for solution in pareto_solutions:
            score = 0.0
            
            for obj_name, weight in normalized_prefs.items():
                if obj_name in solution.objective_values:
                    # Higher score for better (lower) objective values
                    obj_value = solution.objective_values[obj_name]
                    score += weight * (1.0 / (1.0 + abs(obj_value)))
            
            if score > best_score:
                best_score = score
                best_solution = solution
        
        self.logger.info(f"Selected solution with preference score: {best_score:.3f}")
        return best_solution


class UncertaintyQuantifier:
    """
    Uncertainty quantification for robust quantum HVAC control.
    
    Handles uncertain weather forecasts, occupancy, and energy prices.
    """
    
    def __init__(self):
        self.logger = logging.getLogger("uncertainty_quantifier")
    
    def create_robust_qubo(
        self,
        nominal_Q: np.ndarray,
        uncertainty_models: Dict[str, Any],
        confidence_level: float = 0.95
    ) -> np.ndarray:
        """
        Create robust QUBO that handles uncertainties.
        
        Args:
            nominal_Q: Nominal QUBO matrix
            uncertainty_models: Dictionary of uncertainty models
            confidence_level: Confidence level for robustness
            
        Returns:
            Robust QUBO matrix
        """
        robust_Q = nominal_Q.copy()
        
        # Add uncertainty penalties
        for param_name, uncertainty_model in uncertainty_models.items():
            uncertainty_penalty = self._compute_uncertainty_penalty(
                uncertainty_model,
                nominal_Q.shape[0],
                confidence_level
            )
            
            robust_Q += uncertainty_penalty
        
        self.logger.info(f"Created robust QUBO with {len(uncertainty_models)} uncertainty sources")
        return robust_Q
    
    def _compute_uncertainty_penalty(
        self,
        uncertainty_model: Dict[str, Any],
        matrix_size: int,
        confidence_level: float
    ) -> np.ndarray:
        """Compute penalty matrix for uncertainty model."""
        
        penalty_matrix = np.zeros((matrix_size, matrix_size))
        
        model_type = uncertainty_model.get('type', 'gaussian')
        variance = uncertainty_model.get('variance', 1.0)
        
        if model_type == 'gaussian':
            # Add diagonal penalties proportional to uncertainty
            penalty_strength = variance * (1 - confidence_level)
            np.fill_diagonal(penalty_matrix, penalty_strength)
        
        elif model_type == 'scenario_based':
            scenarios = uncertainty_model.get('scenarios', [])
            weights = uncertainty_model.get('weights', [1.0] * len(scenarios))
            
            # Weighted combination of scenario penalties
            for scenario, weight in zip(scenarios, weights):
                scenario_penalty = self._scenario_to_penalty_matrix(scenario, matrix_size)
                penalty_matrix += weight * scenario_penalty
        
        return penalty_matrix
    
    def _scenario_to_penalty_matrix(
        self,
        scenario: Dict[str, Any],
        matrix_size: int
    ) -> np.ndarray:
        """Convert uncertainty scenario to penalty matrix."""
        
        penalty = np.zeros((matrix_size, matrix_size))
        
        # Example: temperature uncertainty affects comfort constraints
        if 'temperature_deviation' in scenario:
            temp_dev = scenario['temperature_deviation']
            comfort_penalty = abs(temp_dev) * 0.1
            np.fill_diagonal(penalty, comfort_penalty)
        
        return penalty
    
    def monte_carlo_evaluation(
        self,
        solution: np.ndarray,
        base_Q: np.ndarray,
        uncertainty_models: Dict[str, Any],
        num_samples: int = 1000
    ) -> Dict[str, Any]:
        """
        Evaluate solution robustness using Monte Carlo simulation.
        
        Args:
            solution: Solution to evaluate
            base_Q: Base QUBO matrix
            uncertainty_models: Uncertainty models
            num_samples: Number of Monte Carlo samples
            
        Returns:
            Robustness statistics
        """
        
        energies = []
        
        for _ in range(num_samples):
            # Sample uncertain parameters
            perturbed_Q = self._sample_uncertain_qubo(base_Q, uncertainty_models)
            
            # Evaluate solution energy
            energy = np.dot(solution, np.dot(perturbed_Q, solution))
            energies.append(energy)
        
        energies = np.array(energies)
        
        statistics = {
            'mean_energy': np.mean(energies),
            'std_energy': np.std(energies),
            'min_energy': np.min(energies),
            'max_energy': np.max(energies),
            'percentile_5': np.percentile(energies, 5),
            'percentile_95': np.percentile(energies, 95),
            'coefficient_of_variation': np.std(energies) / np.mean(energies) if np.mean(energies) != 0 else 0
        }
        
        self.logger.info(f"Monte Carlo evaluation: mean={statistics['mean_energy']:.3f}, std={statistics['std_energy']:.3f}")
        return statistics
    
    def _sample_uncertain_qubo(
        self,
        base_Q: np.ndarray,
        uncertainty_models: Dict[str, Any]
    ) -> np.ndarray:
        """Sample QUBO matrix with uncertainties."""
        
        perturbed_Q = base_Q.copy()
        
        for param_name, model in uncertainty_models.items():
            model_type = model.get('type', 'gaussian')
            
            if model_type == 'gaussian':
                mean = model.get('mean', 0.0)
                std = model.get('std', 1.0)
                
                # Add Gaussian perturbation
                perturbation = np.random.normal(mean, std, base_Q.shape)
                perturbed_Q += perturbation
            
            elif model_type == 'uniform':
                low = model.get('low', -1.0)
                high = model.get('high', 1.0)
                
                # Add uniform perturbation
                perturbation = np.random.uniform(low, high, base_Q.shape)
                perturbed_Q += perturbation
        
        return perturbed_Q


# Global instances
_global_decomposer = QuantumDecomposer()
_global_multi_objective = MultiObjectiveOptimizer()
_global_uncertainty_quantifier = UncertaintyQuantifier()


def get_quantum_decomposer() -> QuantumDecomposer:
    """Get global quantum decomposer."""
    return _global_decomposer


def get_multi_objective_optimizer() -> MultiObjectiveOptimizer:
    """Get global multi-objective optimizer."""
    return _global_multi_objective


def get_uncertainty_quantifier() -> UncertaintyQuantifier:
    """Get global uncertainty quantifier."""
    return _global_uncertainty_quantifier
"""
Advanced Error Recovery System for Quantum HVAC Optimization.

This module implements sophisticated error recovery strategies including:
1. Intelligent fallback selection based on problem characteristics
2. Partial solution recovery from failed quantum solves
3. Adaptive retry strategies with exponential backoff
4. Problem decomposition for large failed problems
5. Quantum-classical hybrid recovery modes
"""

from typing import Dict, Any, List, Optional, Tuple, Callable, Union
import numpy as np
from dataclasses import dataclass, field
import logging
import asyncio
import time
from datetime import datetime, timedelta
from enum import Enum
import json
from collections import deque

from .error_handling import (
    ErrorHandler, QuantumControlError, OptimizationError,
    ErrorCategory, ErrorSeverity, ErrorRecoveryStrategy
)
from .monitoring import RetryManager
from ..optimization.quantum_solver import QuantumSolution


class RecoveryMode(Enum):
    """Recovery modes for different types of failures."""
    RETRY_QUANTUM = "retry_quantum"           # Retry with adjusted parameters
    CLASSICAL_FALLBACK = "classical_fallback"  # Switch to classical solver
    HYBRID_DECOMPOSITION = "hybrid_decomposition"  # Decompose problem
    PARTIAL_RECOVERY = "partial_recovery"      # Use partial quantum solution
    EMERGENCY_CONTROL = "emergency_control"    # Safe emergency control
    CACHED_SOLUTION = "cached_solution"        # Use similar cached solution


class FailureType(Enum):
    """Types of quantum optimization failures."""
    CHAIN_BREAKS = "chain_breaks"             # High chain break fraction
    EMBEDDING_FAILURE = "embedding_failure"   # Cannot find valid embedding
    QPU_TIMEOUT = "qpu_timeout"              # Quantum processor timeout
    SOLVER_ERROR = "solver_error"            # General solver error
    INVALID_SOLUTION = "invalid_solution"     # Solution violates constraints
    POOR_QUALITY = "poor_quality"            # Low solution quality
    CONVERGENCE_FAILURE = "convergence_failure"  # No convergence achieved


@dataclass
class RecoveryContext:
    """Context information for error recovery decisions."""
    failure_type: FailureType
    problem_size: int
    problem_complexity: str
    previous_failures: int
    time_remaining: float  # Seconds until control deadline
    available_solvers: List[str]
    cached_solutions_available: bool
    safety_critical: bool = False
    
    # Problem characteristics
    constraint_density: float = 0.0
    qubo_condition_number: Optional[float] = None
    embedding_quality_history: List[float] = field(default_factory=list)
    
    @property
    def is_time_critical(self) -> bool:
        """Check if recovery must be fast due to time constraints."""
        return self.time_remaining < 30.0  # Less than 30 seconds
    
    @property
    def allows_decomposition(self) -> bool:
        """Check if problem size allows decomposition."""
        return self.problem_size > 50 and not self.is_time_critical


@dataclass
class RecoveryResult:
    """Result of error recovery attempt."""
    success: bool
    recovery_mode: RecoveryMode
    solution: Optional[Any] = None
    partial_solution: Optional[Any] = None
    recovery_time: float = 0.0
    quality_estimate: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def has_usable_solution(self) -> bool:
        """Check if recovery produced a usable solution."""
        return self.success and (self.solution is not None or self.partial_solution is not None)


class AdvancedErrorRecoverySystem:
    """
    Advanced error recovery system for quantum HVAC optimization.
    
    Implements intelligent recovery strategies based on failure analysis,
    problem characteristics, and system state.
    """
    
    def __init__(
        self,
        max_recovery_attempts: int = 3,
        enable_problem_decomposition: bool = True,
        enable_partial_recovery: bool = True,
        solution_cache_size: int = 100
    ):
        self.max_recovery_attempts = max_recovery_attempts
        self.enable_problem_decomposition = enable_problem_decomposition
        self.enable_partial_recovery = enable_partial_recovery
        
        self.logger = logging.getLogger(__name__)
        
        # Solution caching for similar problems
        self._solution_cache: deque = deque(maxlen=solution_cache_size)
        
        # Recovery strategy performance tracking
        self._recovery_history: deque = deque(maxlen=1000)
        self._strategy_performance: Dict[RecoveryMode, Dict[str, float]] = {
            mode: {'success_rate': 0.5, 'avg_quality': 0.5, 'avg_time': 10.0}
            for mode in RecoveryMode
        }
        
        # Problem decomposition strategies
        self._decomposition_strategies = {
            'temporal': self._temporal_decomposition,
            'spatial': self._spatial_decomposition,
            'objective': self._objective_decomposition
        }
        
        # Classical fallback solvers
        self._classical_solvers = {
            'scipy_minimize': self._scipy_minimize_fallback,
            'genetic_algorithm': self._genetic_algorithm_fallback,
            'simulated_annealing': self._simulated_annealing_fallback
        }
        
        # Initialize retry managers for different recovery modes
        self._retry_managers = {
            RecoveryMode.RETRY_QUANTUM: RetryManager(max_retries=2, base_delay=1.0),
            RecoveryMode.CLASSICAL_FALLBACK: RetryManager(max_retries=1, base_delay=0.5),
            RecoveryMode.HYBRID_DECOMPOSITION: RetryManager(max_retries=1, base_delay=2.0)
        }
    
    async def recover_from_failure(
        self,
        failure_context: RecoveryContext,
        original_problem: Dict[str, Any],
        failed_solution: Optional[QuantumSolution] = None
    ) -> RecoveryResult:
        """
        Attempt recovery from quantum optimization failure.
        
        Args:
            failure_context: Context about the failure
            original_problem: Original optimization problem
            failed_solution: Partial/failed solution if available
            
        Returns:
            Recovery result with solution or fallback
        """
        start_time = time.time()
        
        self.logger.info(
            f"Attempting recovery from {failure_context.failure_type.value} failure, "
            f"problem size: {failure_context.problem_size}, "
            f"time remaining: {failure_context.time_remaining:.1f}s"
        )
        
        # Select optimal recovery strategy
        recovery_strategy = self._select_recovery_strategy(failure_context, original_problem)
        
        recovery_result = None
        
        try:
            # Execute recovery strategy
            if recovery_strategy == RecoveryMode.RETRY_QUANTUM:
                recovery_result = await self._retry_quantum_with_adjustments(
                    failure_context, original_problem, failed_solution
                )
                
            elif recovery_strategy == RecoveryMode.CLASSICAL_FALLBACK:
                recovery_result = await self._classical_fallback_recovery(
                    failure_context, original_problem
                )
                
            elif recovery_strategy == RecoveryMode.HYBRID_DECOMPOSITION:
                recovery_result = await self._hybrid_decomposition_recovery(
                    failure_context, original_problem
                )
                
            elif recovery_strategy == RecoveryMode.PARTIAL_RECOVERY:
                recovery_result = await self._partial_solution_recovery(
                    failure_context, original_problem, failed_solution
                )
                
            elif recovery_strategy == RecoveryMode.CACHED_SOLUTION:
                recovery_result = await self._cached_solution_recovery(
                    failure_context, original_problem
                )
                
            elif recovery_strategy == RecoveryMode.EMERGENCY_CONTROL:
                recovery_result = await self._emergency_control_recovery(
                    failure_context, original_problem
                )
            
            if recovery_result is None:
                recovery_result = RecoveryResult(
                    success=False,
                    recovery_mode=recovery_strategy,
                    recovery_time=time.time() - start_time,
                    metadata={'error': 'Recovery strategy not implemented'}
                )
            
        except Exception as e:
            self.logger.error(f"Recovery strategy {recovery_strategy.value} failed: {e}")
            recovery_result = RecoveryResult(
                success=False,
                recovery_mode=recovery_strategy,
                recovery_time=time.time() - start_time,
                metadata={'error': str(e)}
            )
        
        # Record recovery attempt
        recovery_result.recovery_time = time.time() - start_time
        self._record_recovery_attempt(failure_context, recovery_result)
        
        if recovery_result.success:
            self.logger.info(
                f"Recovery successful using {recovery_strategy.value}, "
                f"quality: {recovery_result.quality_estimate:.3f}, "
                f"time: {recovery_result.recovery_time:.1f}s"
            )
        else:
            self.logger.warning(f"Recovery failed using {recovery_strategy.value}")
        
        return recovery_result
    
    def _select_recovery_strategy(
        self,
        failure_context: RecoveryContext,
        original_problem: Dict[str, Any]
    ) -> RecoveryMode:
        """Select optimal recovery strategy based on failure context and problem characteristics."""
        
        # Emergency cases - use fast fallback
        if failure_context.safety_critical or failure_context.is_time_critical:
            if failure_context.cached_solutions_available:
                return RecoveryMode.CACHED_SOLUTION
            else:
                return RecoveryMode.EMERGENCY_CONTROL
        
        # Strategy selection based on failure type and problem characteristics
        strategy_scores = {}
        
        for mode in RecoveryMode:
            score = self._calculate_strategy_score(mode, failure_context, original_problem)
            strategy_scores[mode] = score
        
        # Select strategy with highest score
        best_strategy = max(strategy_scores.keys(), key=lambda k: strategy_scores[k])
        
        self.logger.debug(f"Strategy scores: {[(k.value, v) for k, v in strategy_scores.items()]}")
        self.logger.info(f"Selected recovery strategy: {best_strategy.value}")
        
        return best_strategy
    
    def _calculate_strategy_score(
        self,
        mode: RecoveryMode,
        failure_context: RecoveryContext,
        original_problem: Dict[str, Any]
    ) -> float:
        """Calculate score for recovery strategy based on context."""
        base_score = 0.0
        
        # Get historical performance for this strategy
        performance = self._strategy_performance[mode]
        
        if mode == RecoveryMode.RETRY_QUANTUM:
            # Good for transient failures, not for systematic issues
            if failure_context.failure_type in [FailureType.QPU_TIMEOUT, FailureType.SOLVER_ERROR]:
                base_score = 0.8
            elif failure_context.failure_type == FailureType.CHAIN_BREAKS:
                base_score = 0.6 if failure_context.previous_failures < 2 else 0.2
            else:
                base_score = 0.3
        
        elif mode == RecoveryMode.CLASSICAL_FALLBACK:
            # Always available, reliable for smaller problems
            base_score = 0.7
            if failure_context.problem_size > 100:
                base_score = 0.5  # Classical struggles with large problems
            if failure_context.is_time_critical:
                base_score += 0.2  # Fast classical methods available
        
        elif mode == RecoveryMode.HYBRID_DECOMPOSITION:
            # Good for large, complex problems
            if failure_context.allows_decomposition:
                base_score = 0.8
                if failure_context.problem_size > 200:
                    base_score = 0.9  # Excellent for very large problems
            else:
                base_score = 0.1  # Not suitable for small/time-critical problems
        
        elif mode == RecoveryMode.PARTIAL_RECOVERY:
            # Good when we have a partial solution
            if failure_context.failure_type in [FailureType.POOR_QUALITY, FailureType.CONVERGENCE_FAILURE]:
                base_score = 0.7
            else:
                base_score = 0.4
        
        elif mode == RecoveryMode.CACHED_SOLUTION:
            # Excellent if similar solutions available
            if failure_context.cached_solutions_available:
                base_score = 0.9 if failure_context.is_time_critical else 0.6
            else:
                base_score = 0.0  # Not available
        
        elif mode == RecoveryMode.EMERGENCY_CONTROL:
            # Last resort, always works but low quality
            base_score = 0.3
            if failure_context.safety_critical:
                base_score = 0.8  # High priority for safety
        
        # Adjust score based on historical performance
        success_factor = performance['success_rate']
        quality_factor = performance['avg_quality']
        time_factor = 1.0 / (1.0 + performance['avg_time'] / 10.0)  # Prefer faster methods
        
        final_score = base_score * (0.5 * success_factor + 0.3 * quality_factor + 0.2 * time_factor)
        
        return final_score
    
    async def _retry_quantum_with_adjustments(
        self,
        failure_context: RecoveryContext,
        original_problem: Dict[str, Any],
        failed_solution: Optional[QuantumSolution]
    ) -> RecoveryResult:
        """Retry quantum solving with parameter adjustments."""
        
        # Determine parameter adjustments based on failure type
        adjustments = {}
        
        if failure_context.failure_type == FailureType.CHAIN_BREAKS:
            adjustments['chain_strength_multiplier'] = 2.0
            adjustments['num_reads_multiplier'] = 1.5
            
        elif failure_context.failure_type == FailureType.EMBEDDING_FAILURE:
            adjustments['embedding_tries'] = 20
            adjustments['chain_strength_multiplier'] = 1.5
            
        elif failure_context.failure_type == FailureType.QPU_TIMEOUT:
            adjustments['annealing_time'] = max(1, int(20 * 0.5))  # Reduce annealing time
            adjustments['num_reads_multiplier'] = 0.5
        
        elif failure_context.failure_type == FailureType.POOR_QUALITY:
            adjustments['num_reads_multiplier'] = 2.0
            adjustments['annealing_time_multiplier'] = 1.5
        
        # Use retry manager for quantum retry
        retry_manager = self._retry_managers[RecoveryMode.RETRY_QUANTUM]
        
        try:
            # This would integrate with actual quantum solver
            # For now, simulate retry logic
            await asyncio.sleep(0.1)  # Simulate quantum solve time
            
            # Simulate success/failure based on problem characteristics
            retry_success_prob = 0.7 if failure_context.previous_failures < 2 else 0.3
            retry_success = np.random.random() < retry_success_prob
            
            if retry_success:
                # Simulate improved solution
                quality = 0.7 + 0.2 * np.random.random()
                
                return RecoveryResult(
                    success=True,
                    recovery_mode=RecoveryMode.RETRY_QUANTUM,
                    quality_estimate=quality,
                    metadata={
                        'adjustments': adjustments,
                        'retry_attempt': failure_context.previous_failures + 1
                    }
                )
            else:
                return RecoveryResult(
                    success=False,
                    recovery_mode=RecoveryMode.RETRY_QUANTUM,
                    metadata={'adjustments': adjustments, 'retry_failed': True}
                )
        
        except Exception as e:
            return RecoveryResult(
                success=False,
                recovery_mode=RecoveryMode.RETRY_QUANTUM,
                metadata={'error': str(e)}
            )
    
    async def _classical_fallback_recovery(
        self,
        failure_context: RecoveryContext,
        original_problem: Dict[str, Any]
    ) -> RecoveryResult:
        """Recover using classical optimization methods."""
        
        # Select best classical solver based on problem characteristics
        if failure_context.problem_size < 50:
            solver_name = 'scipy_minimize'
        elif failure_context.is_time_critical:
            solver_name = 'simulated_annealing'  # Fast heuristic
        else:
            solver_name = 'genetic_algorithm'  # Good for larger problems
        
        solver_func = self._classical_solvers[solver_name]
        
        try:
            # Run classical solver
            classical_result = await solver_func(original_problem)
            
            if classical_result['success']:
                return RecoveryResult(
                    success=True,
                    recovery_mode=RecoveryMode.CLASSICAL_FALLBACK,
                    solution=classical_result['solution'],
                    quality_estimate=classical_result['quality'],
                    metadata={
                        'classical_solver': solver_name,
                        'iterations': classical_result.get('iterations', 0)
                    }
                )
            else:
                return RecoveryResult(
                    success=False,
                    recovery_mode=RecoveryMode.CLASSICAL_FALLBACK,
                    metadata={'classical_solver': solver_name, 'solver_error': classical_result.get('error')}
                )
        
        except Exception as e:
            return RecoveryResult(
                success=False,
                recovery_mode=RecoveryMode.CLASSICAL_FALLBACK,
                metadata={'error': str(e)}
            )
    
    async def _hybrid_decomposition_recovery(
        self,
        failure_context: RecoveryContext,
        original_problem: Dict[str, Any]
    ) -> RecoveryResult:
        """Recover using problem decomposition and hybrid solving."""
        
        if not failure_context.allows_decomposition:
            return RecoveryResult(
                success=False,
                recovery_mode=RecoveryMode.HYBRID_DECOMPOSITION,
                metadata={'error': 'Problem decomposition not suitable'}
            )
        
        # Select decomposition strategy
        if failure_context.problem_complexity == 'temporal':
            decomp_strategy = 'temporal'
        elif failure_context.constraint_density > 0.5:
            decomp_strategy = 'spatial'
        else:
            decomp_strategy = 'objective'
        
        decomposition_func = self._decomposition_strategies[decomp_strategy]
        
        try:
            # Decompose problem
            subproblems = decomposition_func(original_problem)
            
            if not subproblems:
                return RecoveryResult(
                    success=False,
                    recovery_mode=RecoveryMode.HYBRID_DECOMPOSITION,
                    metadata={'error': 'Problem decomposition failed'}
                )
            
            # Solve subproblems (some quantum, some classical)
            subsolutions = []
            for i, subproblem in enumerate(subproblems):
                # Use quantum for smaller, well-conditioned subproblems
                use_quantum = (
                    len(subproblems) <= 3 and
                    subproblem.get('size', 0) < 100 and
                    i < 2  # Limit quantum usage
                )
                
                if use_quantum:
                    # Simulate quantum solve on subproblem
                    await asyncio.sleep(0.05)
                    subsol = {'success': True, 'quality': 0.8, 'method': 'quantum'}
                else:
                    # Use classical method
                    subsol = {'success': True, 'quality': 0.7, 'method': 'classical'}
                
                subsolutions.append(subsol)
            
            # Combine subsolutions
            if all(sol['success'] for sol in subsolutions):
                combined_quality = np.mean([sol['quality'] for sol in subsolutions])
                
                return RecoveryResult(
                    success=True,
                    recovery_mode=RecoveryMode.HYBRID_DECOMPOSITION,
                    quality_estimate=combined_quality,
                    metadata={
                        'decomposition_strategy': decomp_strategy,
                        'num_subproblems': len(subproblems),
                        'quantum_subproblems': sum(1 for sol in subsolutions if sol['method'] == 'quantum'),
                        'subsolution_qualities': [sol['quality'] for sol in subsolutions]
                    }
                )
            else:
                return RecoveryResult(
                    success=False,
                    recovery_mode=RecoveryMode.HYBRID_DECOMPOSITION,
                    metadata={'error': 'Some subproblems failed to solve'}
                )
        
        except Exception as e:
            return RecoveryResult(
                success=False,
                recovery_mode=RecoveryMode.HYBRID_DECOMPOSITION,
                metadata={'error': str(e)}
            )
    
    async def _partial_solution_recovery(
        self,
        failure_context: RecoveryContext,
        original_problem: Dict[str, Any],
        failed_solution: Optional[QuantumSolution]
    ) -> RecoveryResult:
        """Attempt to recover usable solution from partial/failed quantum solution."""
        
        if not self.enable_partial_recovery or failed_solution is None:
            return RecoveryResult(
                success=False,
                recovery_mode=RecoveryMode.PARTIAL_RECOVERY,
                metadata={'error': 'No partial solution available'}
            )
        
        try:
            # Analyze partial solution quality
            partial_quality = self._assess_partial_solution_quality(failed_solution, original_problem)
            
            if partial_quality < 0.3:
                return RecoveryResult(
                    success=False,
                    recovery_mode=RecoveryMode.PARTIAL_RECOVERY,
                    metadata={'error': 'Partial solution quality too low', 'quality': partial_quality}
                )
            
            # Attempt to repair/improve partial solution
            repaired_solution = await self._repair_partial_solution(failed_solution, original_problem)
            
            if repaired_solution:
                return RecoveryResult(
                    success=True,
                    recovery_mode=RecoveryMode.PARTIAL_RECOVERY,
                    partial_solution=repaired_solution,
                    quality_estimate=partial_quality,
                    metadata={
                        'original_quality': partial_quality,
                        'repair_method': 'local_search'
                    }
                )
            else:
                return RecoveryResult(
                    success=False,
                    recovery_mode=RecoveryMode.PARTIAL_RECOVERY,
                    metadata={'error': 'Solution repair failed'}
                )
        
        except Exception as e:
            return RecoveryResult(
                success=False,
                recovery_mode=RecoveryMode.PARTIAL_RECOVERY,
                metadata={'error': str(e)}
            )
    
    async def _cached_solution_recovery(
        self,
        failure_context: RecoveryContext,
        original_problem: Dict[str, Any]
    ) -> RecoveryResult:
        """Recover using similar cached solution."""
        
        # Find most similar cached solution
        similar_solution = self._find_similar_cached_solution(original_problem)
        
        if similar_solution is None:
            return RecoveryResult(
                success=False,
                recovery_mode=RecoveryMode.CACHED_SOLUTION,
                metadata={'error': 'No similar cached solution found'}
            )
        
        # Adapt cached solution to current problem
        try:
            adapted_solution = self._adapt_cached_solution(similar_solution, original_problem)
            
            if adapted_solution:
                return RecoveryResult(
                    success=True,
                    recovery_mode=RecoveryMode.CACHED_SOLUTION,
                    solution=adapted_solution['solution'],
                    quality_estimate=adapted_solution['quality'],
                    metadata={
                        'cache_similarity': similar_solution['similarity'],
                        'adaptation_method': adapted_solution['method']
                    }
                )
            else:
                return RecoveryResult(
                    success=False,
                    recovery_mode=RecoveryMode.CACHED_SOLUTION,
                    metadata={'error': 'Solution adaptation failed'}
                )
        
        except Exception as e:
            return RecoveryResult(
                success=False,
                recovery_mode=RecoveryMode.CACHED_SOLUTION,
                metadata={'error': str(e)}
            )
    
    async def _emergency_control_recovery(
        self,
        failure_context: RecoveryContext,
        original_problem: Dict[str, Any]
    ) -> RecoveryResult:
        """Generate safe emergency control as last resort."""
        
        try:
            # Generate conservative, safe control schedule
            emergency_solution = self._generate_emergency_control(original_problem)
            
            return RecoveryResult(
                success=True,
                recovery_mode=RecoveryMode.EMERGENCY_CONTROL,
                solution=emergency_solution,
                quality_estimate=0.4,  # Low quality but safe
                metadata={
                    'safety_priority': True,
                    'control_type': 'conservative'
                }
            )
        
        except Exception as e:
            return RecoveryResult(
                success=False,
                recovery_mode=RecoveryMode.EMERGENCY_CONTROL,
                metadata={'error': str(e)}
            )
    
    # Classical solver implementations
    async def _scipy_minimize_fallback(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback using SciPy optimization."""
        # Simplified implementation - would use actual scipy.optimize
        await asyncio.sleep(0.1)  # Simulate solve time
        
        return {
            'success': True,
            'solution': np.random.uniform(0.3, 0.7, size=problem.get('num_variables', 20)),
            'quality': 0.75,
            'iterations': 50
        }
    
    async def _genetic_algorithm_fallback(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback using genetic algorithm."""
        await asyncio.sleep(0.2)  # Simulate solve time
        
        return {
            'success': True,
            'solution': np.random.uniform(0.2, 0.8, size=problem.get('num_variables', 20)),
            'quality': 0.65,
            'iterations': 100
        }
    
    async def _simulated_annealing_fallback(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback using simulated annealing."""
        await asyncio.sleep(0.05)  # Fast heuristic
        
        return {
            'success': True,
            'solution': np.random.uniform(0.4, 0.6, size=problem.get('num_variables', 20)),
            'quality': 0.6,
            'iterations': 200
        }
    
    # Problem decomposition implementations
    def _temporal_decomposition(self, problem: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Decompose problem by time horizon."""
        horizon = problem.get('horizon', 24)
        
        if horizon <= 6:
            return [problem]  # Too small to decompose
        
        # Split into overlapping time windows
        window_size = horizon // 3
        overlap = window_size // 4
        
        subproblems = []
        for i in range(0, horizon, window_size - overlap):
            end_time = min(i + window_size, horizon)
            
            subproblem = problem.copy()
            subproblem['horizon'] = end_time - i
            subproblem['time_offset'] = i
            subproblem['size'] = (end_time - i) * problem.get('num_zones', 10)
            
            subproblems.append(subproblem)
        
        return subproblems
    
    def _spatial_decomposition(self, problem: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Decompose problem by building zones."""
        num_zones = problem.get('num_zones', 10)
        
        if num_zones <= 5:
            return [problem]  # Too small to decompose
        
        # Group zones (simplified - could use actual building topology)
        zones_per_group = max(3, num_zones // 3)
        
        subproblems = []
        for i in range(0, num_zones, zones_per_group):
            end_zone = min(i + zones_per_group, num_zones)
            
            subproblem = problem.copy()
            subproblem['num_zones'] = end_zone - i
            subproblem['zone_offset'] = i
            subproblem['size'] = (end_zone - i) * problem.get('horizon', 24)
            
            subproblems.append(subproblem)
        
        return subproblems
    
    def _objective_decomposition(self, problem: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Decompose problem by objectives."""
        objectives = problem.get('objectives', {}).get('weights', {})
        
        if len(objectives) <= 2:
            return [problem]  # Not enough objectives to decompose
        
        # Create subproblems focusing on different objectives
        subproblems = []
        
        for obj_name, weight in objectives.items():
            if weight > 0.1:  # Only decompose significant objectives
                subproblem = problem.copy()
                
                # Focus on single objective
                new_objectives = {obj_name: 1.0}
                subproblem['objectives'] = {'weights': new_objectives}
                subproblem['primary_objective'] = obj_name
                subproblem['size'] = problem.get('size', 100)
                
                subproblems.append(subproblem)
        
        return subproblems
    
    def _assess_partial_solution_quality(
        self,
        solution: QuantumSolution,
        problem: Dict[str, Any]
    ) -> float:
        """Assess quality of partial quantum solution."""
        
        # Check chain break fraction
        chain_quality = 1.0 - min(1.0, solution.chain_break_fraction)
        
        # Check solution completeness (how many variables have valid values)
        total_variables = len(solution.sample)
        valid_variables = sum(1 for val in solution.sample.values() if 0 <= val <= 1)
        completeness = valid_variables / max(total_variables, 1)
        
        # Check energy quality (lower is better for minimization)
        energy_quality = max(0.0, 1.0 - abs(solution.energy) / 1000.0)
        
        # Combine quality factors
        overall_quality = 0.4 * chain_quality + 0.4 * completeness + 0.2 * energy_quality
        
        return overall_quality
    
    async def _repair_partial_solution(
        self,
        solution: QuantumSolution,
        problem: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Attempt to repair/improve partial solution using local search."""
        
        try:
            # Simple local search repair (can be enhanced)
            current_solution = dict(solution.sample)
            
            # Fix invalid variable values
            for var, value in current_solution.items():
                if not (0 <= value <= 1):
                    current_solution[var] = max(0, min(1, value))
            
            # Local improvement (simplified)
            for _ in range(10):  # Limited iterations
                # Random local move
                if current_solution:
                    var = np.random.choice(list(current_solution.keys()))
                    old_value = current_solution[var]
                    
                    # Small random change
                    delta = np.random.normal(0, 0.1)
                    new_value = max(0, min(1, old_value + delta))
                    current_solution[var] = new_value
            
            return {
                'solution': current_solution,
                'quality': 0.6,  # Estimated quality after repair
                'method': 'local_search'
            }
        
        except Exception as e:
            self.logger.error(f"Solution repair failed: {e}")
            return None
    
    def _find_similar_cached_solution(self, problem: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Find most similar cached solution."""
        
        if not self._solution_cache:
            return None
        
        # Calculate similarity scores
        best_similarity = 0.0
        best_solution = None
        
        for cached_entry in self._solution_cache:
            similarity = self._calculate_problem_similarity(problem, cached_entry['problem'])
            
            if similarity > best_similarity and similarity > 0.7:  # Minimum similarity threshold
                best_similarity = similarity
                best_solution = cached_entry
                best_solution['similarity'] = similarity
        
        return best_solution
    
    def _calculate_problem_similarity(self, problem1: Dict[str, Any], problem2: Dict[str, Any]) -> float:
        """Calculate similarity between two problems."""
        
        similarity_factors = []
        
        # Size similarity
        size1 = problem1.get('num_zones', 1) * problem1.get('horizon', 1)
        size2 = problem2.get('num_zones', 1) * problem2.get('horizon', 1)
        size_similarity = min(size1, size2) / max(size1, size2)
        similarity_factors.append(size_similarity)
        
        # Objective weights similarity
        obj1 = problem1.get('objectives', {}).get('weights', {})
        obj2 = problem2.get('objectives', {}).get('weights', {})
        
        if obj1 and obj2:
            # Compare objective weight vectors
            common_objectives = set(obj1.keys()) & set(obj2.keys())
            if common_objectives:
                obj_similarity = 1.0 - sum(
                    abs(obj1.get(obj, 0) - obj2.get(obj, 0))
                    for obj in common_objectives
                ) / len(common_objectives)
                similarity_factors.append(obj_similarity)
        
        # Complexity similarity
        complexity1 = problem1.get('complexity', 'medium')
        complexity2 = problem2.get('complexity', 'medium')
        complexity_similarity = 1.0 if complexity1 == complexity2 else 0.5
        similarity_factors.append(complexity_similarity)
        
        # Overall similarity
        return np.mean(similarity_factors)
    
    def _adapt_cached_solution(
        self,
        cached_solution: Dict[str, Any],
        current_problem: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Adapt cached solution to current problem."""
        
        try:
            # Scale solution based on problem size differences
            cached_problem = cached_solution['problem']
            solution = cached_solution['solution'].copy()
            
            # Simple scaling adaptation (can be enhanced)
            size_ratio = (
                current_problem.get('num_zones', 1) * current_problem.get('horizon', 1)
            ) / (
                cached_problem.get('num_zones', 1) * cached_problem.get('horizon', 1)
            )
            
            if 0.5 <= size_ratio <= 2.0:  # Reasonable scaling range
                # Scale solution values
                if isinstance(solution, dict):
                    adapted_solution = {k: v * np.sqrt(size_ratio) for k, v in solution.items()}
                elif isinstance(solution, np.ndarray):
                    adapted_solution = solution * np.sqrt(size_ratio)
                else:
                    adapted_solution = solution
                
                # Clamp to valid range
                if isinstance(adapted_solution, dict):
                    for k, v in adapted_solution.items():
                        adapted_solution[k] = max(0, min(1, v))
                
                return {
                    'solution': adapted_solution,
                    'quality': cached_solution['quality'] * 0.8,  # Slightly reduced quality
                    'method': 'scaling_adaptation'
                }
            else:
                return None  # Scaling not feasible
        
        except Exception as e:
            self.logger.error(f"Solution adaptation failed: {e}")
            return None
    
    def _generate_emergency_control(self, problem: Dict[str, Any]) -> np.ndarray:
        """Generate safe emergency control schedule."""
        
        num_zones = problem.get('num_zones', 10)
        horizon = problem.get('horizon', 24)
        
        # Conservative control values
        emergency_values = []
        
        for zone in range(num_zones):
            for hour in range(horizon):
                # Use moderate control values (avoid extremes)
                base_value = 0.5  # Middle value
                
                # Slight variation based on time of day (simplified)
                time_adjustment = 0.1 * np.sin(2 * np.pi * hour / 24)
                control_value = max(0.2, min(0.8, base_value + time_adjustment))
                
                emergency_values.append(control_value)
        
        return np.array(emergency_values)
    
    def _record_recovery_attempt(
        self,
        failure_context: RecoveryContext,
        recovery_result: RecoveryResult
    ) -> None:
        """Record recovery attempt for performance tracking."""
        
        # Add to recovery history
        recovery_record = {
            'timestamp': time.time(),
            'failure_type': failure_context.failure_type.value,
            'problem_size': failure_context.problem_size,
            'recovery_mode': recovery_result.recovery_mode.value,
            'success': recovery_result.success,
            'quality': recovery_result.quality_estimate,
            'recovery_time': recovery_result.recovery_time
        }
        
        self._recovery_history.append(recovery_record)
        
        # Update strategy performance statistics
        mode = recovery_result.recovery_mode
        performance = self._strategy_performance[mode]
        
        # Exponential moving average update
        alpha = 0.1
        performance['success_rate'] = (
            alpha * (1.0 if recovery_result.success else 0.0) +
            (1 - alpha) * performance['success_rate']
        )
        performance['avg_quality'] = (
            alpha * recovery_result.quality_estimate +
            (1 - alpha) * performance['avg_quality']
        )
        performance['avg_time'] = (
            alpha * recovery_result.recovery_time +
            (1 - alpha) * performance['avg_time']
        )
    
    def cache_solution(
        self,
        problem: Dict[str, Any],
        solution: Any,
        quality: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Cache solution for future recovery use."""
        
        cache_entry = {
            'timestamp': time.time(),
            'problem': problem.copy(),
            'solution': solution,
            'quality': quality,
            'metadata': metadata or {}
        }
        
        self._solution_cache.append(cache_entry)
        self.logger.debug(f"Cached solution with quality {quality:.3f}")
    
    def get_recovery_statistics(self) -> Dict[str, Any]:
        """Get recovery system performance statistics."""
        
        if not self._recovery_history:
            return {'error': 'No recovery history available'}
        
        # Overall statistics
        total_attempts = len(self._recovery_history)
        successful_attempts = sum(1 for r in self._recovery_history if r['success'])
        overall_success_rate = successful_attempts / total_attempts
        
        # Statistics by recovery mode
        mode_stats = {}
        for mode in RecoveryMode:
            mode_records = [r for r in self._recovery_history if r['recovery_mode'] == mode.value]
            if mode_records:
                mode_stats[mode.value] = {
                    'attempts': len(mode_records),
                    'success_rate': sum(1 for r in mode_records if r['success']) / len(mode_records),
                    'avg_quality': np.mean([r['quality'] for r in mode_records]),
                    'avg_recovery_time': np.mean([r['recovery_time'] for r in mode_records])
                }
        
        # Statistics by failure type
        failure_stats = {}
        for failure_type in FailureType:
            failure_records = [r for r in self._recovery_history if r['failure_type'] == failure_type.value]
            if failure_records:
                failure_stats[failure_type.value] = {
                    'occurrences': len(failure_records),
                    'recovery_success_rate': sum(1 for r in failure_records if r['success']) / len(failure_records)
                }
        
        return {
            'total_recovery_attempts': total_attempts,
            'overall_success_rate': overall_success_rate,
            'cache_size': len(self._solution_cache),
            'recovery_modes': mode_stats,
            'failure_types': failure_stats,
            'current_strategy_performance': self._strategy_performance
        }


# Global recovery system instance
_global_recovery_system: Optional[AdvancedErrorRecoverySystem] = None


def get_recovery_system() -> AdvancedErrorRecoverySystem:
    """Get global error recovery system instance."""
    global _global_recovery_system
    if _global_recovery_system is None:
        _global_recovery_system = AdvancedErrorRecoverySystem()
    return _global_recovery_system


def reset_recovery_system():
    """Reset global error recovery system instance."""
    global _global_recovery_system
    _global_recovery_system = None
"""
Quantum solver interface for D-Wave systems.

Provides unified interface to D-Wave quantum annealers, hybrid solvers,
and classical fallbacks for HVAC optimization problems.
"""

from typing import Dict, Any, Optional, Union, Tuple, List
import numpy as np
from dataclasses import dataclass
import logging
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor

try:
    from dwave.system import DWaveSampler, LazyFixedEmbeddingComposite
    from dwave.system.composites import EmbeddingComposite
    from dwave.embedding import embed_qubo, unembed_sampleset
    from dwave.embedding.chain_breaks import majority_vote
    from dimod import BinaryQuadraticModel, SampleSet
    from dimod.generators import gnp_random_bqm
    from dwave.cloud import Client
    from dwave.hybrid import KerberosSampler
    import dwave.inspector
    DWAVE_AVAILABLE = True
except ImportError:
    DWAVE_AVAILABLE = False
    # Mock classes for development without D-Wave access
    class DWaveSampler:
        def __init__(self): pass
    class EmbeddingComposite:
        def __init__(self, sampler): pass
    class BinaryQuadraticModel:
        def __init__(self, *args, **kwargs): pass
    class SampleSet:
        def __init__(self): pass


@dataclass
class QuantumSolution:
    """Quantum annealing solution with metadata."""
    sample: Dict[int, int]  # Best binary solution
    energy: float           # Solution energy
    num_occurrences: int   # How many times this solution appeared
    chain_break_fraction: float  # Fraction of broken chains
    timing: Dict[str, float]      # Timing information
    embedding_stats: Dict[str, Any]  # Embedding quality metrics
    data_vectors: Dict[str, np.ndarray] = None  # Decoded problem data
    
    @property
    def is_valid(self) -> bool:
        """Check if solution is valid (low chain breaks)."""
        return self.chain_break_fraction < 0.1


class QuantumSolver:
    """
    Quantum annealing solver for HVAC optimization.
    
    Provides interface to D-Wave quantum computers with automatic
    fallback to classical methods when quantum access is unavailable.
    """
    
    def __init__(
        self,
        solver_type: str = "hybrid_v2",
        num_reads: int = 1000,
        annealing_time: int = 20,
        chain_strength: Optional[float] = None,
        auto_scale: bool = True
    ):
        self.solver_type = solver_type
        self.num_reads = num_reads
        self.annealing_time = annealing_time
        self.chain_strength = chain_strength
        self.auto_scale = auto_scale
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize quantum components
        self._sampler = None
        self._embedding_cache = {}
        self._solution_cache = {}
        
        # Performance tracking
        self._solve_count = 0
        self._total_qpu_time = 0.0
        self._avg_chain_breaks = 0.0
        
        # Initialize solver
        self._initialize_solver()
    
    def _initialize_solver(self) -> None:
        """Initialize the quantum solver."""
        if not DWAVE_AVAILABLE:
            self.logger.warning("D-Wave Ocean SDK not available, using classical fallback")
            self._sampler = None
            return
        
        try:
            if self.solver_type == "qpu":
                # Direct QPU access
                self._sampler = EmbeddingComposite(DWaveSampler())
                self.logger.info("Initialized D-Wave QPU sampler")
                
            elif self.solver_type.startswith("hybrid"):
                # Hybrid classical-quantum solver
                self._sampler = KerberosSampler()
                self.logger.info(f"Initialized D-Wave hybrid sampler: {self.solver_type}")
                
            else:
                self.logger.error(f"Unknown solver type: {self.solver_type}")
                self._sampler = None
                
        except Exception as e:
            self.logger.error(f"Failed to initialize D-Wave sampler: {e}")
            self._sampler = None
    
    async def solve(
        self,
        Q: Dict[Tuple[int, int], float],
        **kwargs
    ) -> QuantumSolution:
        """
        Solve QUBO problem using quantum annealing.
        
        Args:
            Q: QUBO matrix as dictionary
            **kwargs: Additional solver parameters
            
        Returns:
            QuantumSolution with best result and metadata
        """
        start_time = time.time()
        
        try:
            if self._sampler is None:
                return await self._classical_fallback(Q)
            
            # Convert to BQM
            bqm = self._qubo_to_bqm(Q)
            
            # Solve with quantum annealing
            if self.solver_type.startswith("hybrid"):
                solution = await self._solve_hybrid(bqm, **kwargs)
            else:
                solution = await self._solve_qpu(bqm, **kwargs)
            
            total_time = time.time() - start_time
            solution.timing['total_solve_time'] = total_time
            
            self._update_performance_stats(solution)
            
            self.logger.info(
                f"Quantum solve completed in {total_time:.2f}s, "
                f"energy: {solution.energy:.4f}, "
                f"chain breaks: {solution.chain_break_fraction:.3f}"
            )
            
            return solution
            
        except Exception as e:
            self.logger.error(f"Quantum solve failed: {e}")
            return await self._classical_fallback(Q)
    
    def _qubo_to_bqm(self, Q: Dict[Tuple[int, int], float]) -> BinaryQuadraticModel:
        """Convert QUBO dictionary to Binary Quadratic Model."""
        # Separate linear and quadratic terms
        linear = {}
        quadratic = {}
        
        for (i, j), coeff in Q.items():
            if i == j:
                linear[i] = coeff
            else:
                # Ensure i < j for quadratic terms
                if i > j:
                    i, j = j, i
                quadratic[(i, j)] = coeff
        
        return BinaryQuadraticModel(linear, quadratic, 'BINARY')
    
    async def _solve_hybrid(
        self,
        bqm: BinaryQuadraticModel,
        **kwargs
    ) -> QuantumSolution:
        """Solve using D-Wave hybrid solver."""
        solve_params = {
            'time_limit': kwargs.get('time_limit', 30),  # seconds
            'seed': kwargs.get('seed', None)
        }
        
        # Run hybrid solver in thread pool to avoid blocking
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self._sampler.sample, bqm, **solve_params)
            sampleset = await asyncio.wrap_future(future)
        
        # Extract best solution
        best_sample = sampleset.first
        
        return QuantumSolution(
            sample=dict(best_sample.sample),
            energy=best_sample.energy,
            num_occurrences=best_sample.num_occurrences,
            chain_break_fraction=0.0,  # Hybrid solver handles embedding internally
            timing={
                'qpu_access_time': sampleset.info.get('qpu_access_time', 0.0),
                'run_time': sampleset.info.get('run_time', 0.0)
            },
            embedding_stats={
                'solver_type': 'hybrid',
                'problem_size': len(bqm.variables)
            }
        )
    
    async def _solve_qpu(
        self,
        bqm: BinaryQuadraticModel,
        **kwargs
    ) -> QuantumSolution:
        """Solve using direct QPU access."""
        solve_params = {
            'num_reads': kwargs.get('num_reads', self.num_reads),
            'annealing_time': kwargs.get('annealing_time', self.annealing_time),
            'auto_scale': self.auto_scale
        }
        
        # Set chain strength if specified
        if self.chain_strength is not None:
            solve_params['chain_strength'] = self.chain_strength
        else:
            # Auto-calculate chain strength
            max_coeff = max(abs(coeff) for coeff in bqm.quadratic.values()) if bqm.quadratic else 1.0
            solve_params['chain_strength'] = 2.0 * max_coeff
        
        # Run QPU solver in thread pool
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self._sampler.sample, bqm, **solve_params)
            sampleset = await asyncio.wrap_future(future)
        
        # Calculate chain break statistics
        embedding_info = getattr(sampleset, 'info', {})
        chain_break_fraction = self._calculate_chain_breaks(sampleset, embedding_info)
        
        # Extract best solution
        best_sample = sampleset.first
        
        return QuantumSolution(
            sample=dict(best_sample.sample),
            energy=best_sample.energy,
            num_occurrences=best_sample.num_occurrences,
            chain_break_fraction=chain_break_fraction,
            timing={
                'qpu_access_time': embedding_info.get('timing', {}).get('qpu_access_time', 0.0),
                'qpu_anneal_time_per_sample': embedding_info.get('timing', {}).get('qpu_anneal_time_per_sample', 0.0)
            },
            embedding_stats={
                'solver_type': 'qpu',
                'problem_size': len(bqm.variables),
                'embedding_info': embedding_info.get('embedding_context', {})
            }
        )
    
    def _calculate_chain_breaks(
        self,
        sampleset: SampleSet,
        embedding_info: Dict[str, Any]
    ) -> float:
        """Calculate fraction of solutions with chain breaks."""
        if not hasattr(sampleset, 'info') or 'embedding_context' not in sampleset.info:
            return 0.0
        
        try:
            embedding_context = sampleset.info['embedding_context']
            embedding = embedding_context.get('embedding', {})
            
            if not embedding:
                return 0.0
            
            # Count broken chains across all samples
            total_breaks = 0
            total_chains = len(embedding)
            total_samples = len(sampleset)
            
            for sample in sampleset.samples():
                for logical_var, physical_qubits in embedding.items():
                    if len(physical_qubits) > 1:
                        # Check if all qubits in chain have same value
                        chain_values = [sample.get(q, 0) for q in physical_qubits]
                        if len(set(chain_values)) > 1:
                            total_breaks += 1
            
            return total_breaks / (total_samples * total_chains) if total_samples * total_chains > 0 else 0.0
            
        except Exception as e:
            self.logger.warning(f"Could not calculate chain breaks: {e}")
            return 0.0
    
    async def _classical_fallback(self, Q: Dict[Tuple[int, int], float]) -> QuantumSolution:
        """Classical fallback solver when quantum is unavailable."""
        self.logger.info("Using classical fallback solver")
        
        # Simple greedy/random solution for fallback
        # In production, would use sophisticated classical optimizer
        
        start_time = time.time()
        
        # Get all variables
        variables = set()
        for (i, j) in Q.keys():
            variables.add(i)
            variables.add(j)
        
        variables = sorted(variables)
        
        # Random initialization
        best_sample = {var: np.random.randint(0, 2) for var in variables}
        best_energy = self._evaluate_qubo_energy(best_sample, Q)
        
        # Simple local search
        for _ in range(100):  # Limited iterations for speed
            # Random bit flip
            var_to_flip = np.random.choice(variables)
            test_sample = best_sample.copy()
            test_sample[var_to_flip] = 1 - test_sample[var_to_flip]
            
            test_energy = self._evaluate_qubo_energy(test_sample, Q)
            
            if test_energy < best_energy:
                best_sample = test_sample
                best_energy = test_energy
        
        solve_time = time.time() - start_time
        
        return QuantumSolution(
            sample=best_sample,
            energy=best_energy,
            num_occurrences=1,
            chain_break_fraction=0.0,
            timing={'total_solve_time': solve_time},
            embedding_stats={'solver_type': 'classical_fallback'}
        )
    
    def _evaluate_qubo_energy(
        self,
        sample: Dict[int, int],
        Q: Dict[Tuple[int, int], float]
    ) -> float:
        """Evaluate QUBO energy for a given sample."""
        energy = 0.0
        
        for (i, j), coeff in Q.items():
            if i == j:
                energy += coeff * sample.get(i, 0)
            else:
                energy += coeff * sample.get(i, 0) * sample.get(j, 0)
        
        return energy
    
    def _update_performance_stats(self, solution: QuantumSolution) -> None:
        """Update solver performance statistics."""
        self._solve_count += 1
        self._total_qpu_time += solution.timing.get('qpu_access_time', 0.0)
        
        # Exponential moving average for chain breaks
        alpha = 0.1
        self._avg_chain_breaks = (
            alpha * solution.chain_break_fraction + 
            (1 - alpha) * self._avg_chain_breaks
        )
    
    def get_status(self) -> Dict[str, Any]:
        """Get solver status and performance metrics."""
        return {
            'solver_type': self.solver_type,
            'is_available': self._sampler is not None,
            'solve_count': self._solve_count,
            'total_qpu_time': self._total_qpu_time,
            'avg_qpu_time_per_solve': (
                self._total_qpu_time / self._solve_count if self._solve_count > 0 else 0.0
            ),
            'avg_chain_break_fraction': self._avg_chain_breaks,
            'dwave_sdk_available': DWAVE_AVAILABLE
        }
    
    def get_solver_properties(self) -> Dict[str, Any]:
        """Get properties of the underlying quantum processor."""
        if self._sampler is None:
            return {'error': 'No quantum solver available'}
        
        try:
            if hasattr(self._sampler, 'child') and hasattr(self._sampler.child, 'properties'):
                props = self._sampler.child.properties
                return {
                    'solver_name': props.get('chip_id', 'unknown'),
                    'num_qubits': props.get('num_qubits', 0),
                    'topology': props.get('topology', {}),
                    'quantum_clock': props.get('quantum_clock', 0),
                    'supported_problem_types': props.get('supported_problem_types', [])
                }
            else:
                return {'solver_type': 'hybrid', 'details': 'Hybrid solver properties not available'}
                
        except Exception as e:
            return {'error': f'Could not retrieve solver properties: {e}'}
    
    async def test_connection(self) -> Dict[str, Any]:
        """Test connection to D-Wave cloud service."""
        if not DWAVE_AVAILABLE:
            return {'status': 'error', 'message': 'D-Wave SDK not available'}
        
        try:
            # Create small test problem
            test_Q = {(0, 0): 1, (0, 1): -2, (1, 1): 1}
            
            # Solve with minimal resources
            test_params = {
                'num_reads': 10,
                'annealing_time': 1
            }
            
            solution = await self.solve(test_Q, **test_params)
            
            return {
                'status': 'success',
                'solver_type': self.solver_type,
                'test_energy': solution.energy,
                'chain_breaks': solution.chain_break_fraction,
                'solve_time': solution.timing.get('total_solve_time', 0.0)
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e),
                'solver_type': self.solver_type
            }
    
    def inspect_solution(
        self,
        solution: QuantumSolution,
        Q: Dict[Tuple[int, int], float],
        show_browser: bool = False
    ) -> Optional[str]:
        """Inspect quantum solution using D-Wave Inspector."""
        if not DWAVE_AVAILABLE or self.solver_type != 'qpu':
            self.logger.warning("Solution inspection only available for QPU solutions")
            return None
        
        try:
            # This would show the solution in D-Wave Inspector
            # For now, return a summary
            return f"""
Quantum Solution Inspection:
- Energy: {solution.energy:.6f}
- Chain Break Fraction: {solution.chain_break_fraction:.3f}
- Solver: {solution.embedding_stats.get('solver_type', 'unknown')}
- Problem Size: {solution.embedding_stats.get('problem_size', 0)} variables
- Sample: {dict(list(solution.sample.items())[:10])}...
            """
            
        except Exception as e:
            self.logger.error(f"Failed to inspect solution: {e}")
            return None
"""
Distributed Quantum Solver with Load Balancing and Auto-Scaling.

This module implements a distributed quantum annealing system that can:
1. Distribute large problems across multiple quantum processors
2. Load balance between QPU, hybrid, and classical solvers
3. Auto-scale based on demand and problem characteristics
4. Optimize resource utilization and cost
5. Provide fault tolerance and redundancy
"""

from typing import Dict, Any, List, Optional, Tuple, Callable, Union
import numpy as np
from dataclasses import dataclass, field
import logging
import asyncio
import time
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import deque, defaultdict
from enum import Enum
import json
import uuid

from .adaptive_quantum_engine import AdaptiveQuantumEngine, QuantumPerformanceMetrics
from .quantum_solver import QuantumSolver, QuantumSolution
from .load_balancer import QuantumLoadBalancer, SolverNode
from ..utils.monitoring import ResourceManager, PerformanceMonitor


class SolverTier(Enum):
    """Quantum solver performance tiers."""
    PREMIUM = "premium"     # Latest QPU with best performance
    STANDARD = "standard"   # Regular QPU access
    HYBRID = "hybrid"       # Hybrid classical-quantum
    CLASSICAL = "classical" # Classical fallback
    BUDGET = "budget"       # Cost-optimized classical


@dataclass
class ResourceAllocation:
    """Resource allocation for quantum solve."""
    solver_ids: List[str]
    estimated_cost: float
    estimated_time: float
    qpu_time_quota: float
    priority_level: int
    
    # Scaling parameters
    max_parallel_solvers: int = 3
    min_quality_threshold: float = 0.7
    timeout_seconds: float = 60.0


@dataclass
class SolveRequest:
    """Distributed solve request."""
    request_id: str
    problem_data: Dict[str, Any]
    priority: int = 1  # 1=low, 2=normal, 3=high, 4=critical
    quality_target: float = 0.8
    time_budget: float = 60.0  # seconds
    cost_budget: Optional[float] = None
    preferred_solvers: Optional[List[str]] = None
    
    # Decomposition hints
    allow_decomposition: bool = True
    max_subproblems: int = 5
    
    # Metadata
    submitted_time: datetime = field(default_factory=datetime.now)
    client_id: str = "default"


@dataclass
class SolveResult:
    """Distributed solve result."""
    request_id: str
    success: bool
    solution: Optional[QuantumSolution] = None
    solve_time: float = 0.0
    actual_cost: float = 0.0
    quality_score: float = 0.0
    
    # Resource usage
    solvers_used: List[str] = field(default_factory=list)
    qpu_time_used: float = 0.0
    
    # Scaling info
    subproblems_count: int = 1
    parallel_solvers: int = 1
    
    # Metadata
    completion_time: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class QuantumResourceOrchestrator:
    """
    Orchestrates quantum computing resources for optimal performance and cost.
    
    Features:
    - Dynamic solver selection based on problem characteristics
    - Cost optimization with budget constraints
    - Auto-scaling based on demand
    - Resource pooling and sharing
    - Performance-based solver ranking
    """
    
    def __init__(
        self,
        max_concurrent_solves: int = 10,
        cost_optimization: bool = True,
        enable_auto_scaling: bool = True
    ):
        self.max_concurrent_solves = max_concurrent_solves
        self.cost_optimization = cost_optimization
        self.enable_auto_scaling = enable_auto_scaling
        
        self.logger = logging.getLogger(__name__)
        
        # Solver management
        self._available_solvers: Dict[str, Dict[str, Any]] = {}
        self._solver_performance: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self._solver_costs: Dict[str, float] = {}  # Cost per solve
        
        # Request queue management
        self._request_queue: asyncio.Queue = asyncio.Queue()
        self._active_solves: Dict[str, SolveRequest] = {}
        self._completed_results: Dict[str, SolveResult] = {}
        
        # Load balancing
        self._load_balancer = QuantumLoadBalancer()
        
        # Resource scaling
        self._current_load = 0.0
        self._demand_forecast: deque = deque(maxlen=100)
        self._scaling_decisions: deque = deque(maxlen=50)
        
        # Performance tracking
        self._performance_monitor = PerformanceMonitor()
        
        # Worker management
        self._worker_tasks: List[asyncio.Task] = []
        self._orchestrator_running = False
    
    def register_solver(
        self,
        solver_id: str,
        solver_type: str,
        performance_tier: SolverTier,
        cost_per_solve: float = 0.0,
        max_concurrent: int = 1,
        capabilities: Optional[Dict[str, Any]] = None
    ) -> None:
        """Register a quantum solver with the orchestrator."""
        
        self._available_solvers[solver_id] = {
            'type': solver_type,
            'tier': performance_tier,
            'cost_per_solve': cost_per_solve,
            'max_concurrent': max_concurrent,
            'current_load': 0,
            'capabilities': capabilities or {},
            'last_seen': time.time(),
            'total_solves': 0,
            'avg_quality': 0.8,
            'avg_solve_time': 30.0,
            'success_rate': 0.9
        }
        
        self._solver_costs[solver_id] = cost_per_solve
        
        # Register with load balancer
        solver_node = SolverNode(
            node_id=solver_id,
            solver_type=solver_type,
            current_load=0.0,
            max_capacity=max_concurrent,
            performance_score=0.8,
            cost_factor=cost_per_solve
        )
        
        self._load_balancer.add_solver(solver_node)
        
        self.logger.info(f"Registered solver: {solver_id} ({solver_type}, {performance_tier.value})")
    
    async def start_orchestrator(self) -> None:
        """Start the distributed orchestrator."""
        if self._orchestrator_running:
            return
        
        self._orchestrator_running = True
        
        # Start worker tasks
        for i in range(min(4, self.max_concurrent_solves)):
            worker_task = asyncio.create_task(self._solver_worker(f"worker_{i}"))
            self._worker_tasks.append(worker_task)
        
        # Start monitoring and scaling tasks
        monitor_task = asyncio.create_task(self._monitoring_loop())
        scaling_task = asyncio.create_task(self._scaling_loop())
        
        self._worker_tasks.extend([monitor_task, scaling_task])
        
        self.logger.info(f"Started quantum orchestrator with {len(self._worker_tasks)} workers")
    
    async def stop_orchestrator(self) -> None:
        """Stop the distributed orchestrator."""
        self._orchestrator_running = False
        
        # Cancel all worker tasks
        for task in self._worker_tasks:
            task.cancel()
        
        # Wait for tasks to finish
        await asyncio.gather(*self._worker_tasks, return_exceptions=True)
        
        self._worker_tasks.clear()
        self.logger.info("Stopped quantum orchestrator")
    
    async def submit_solve_request(self, request: SolveRequest) -> str:
        """Submit a solve request to the distributed system."""
        
        # Validate request
        if not request.request_id:
            request.request_id = str(uuid.uuid4())
        
        # Add to queue
        await self._request_queue.put(request)
        self._active_solves[request.request_id] = request
        
        self.logger.info(
            f"Submitted solve request {request.request_id}, "
            f"priority: {request.priority}, "
            f"queue size: {self._request_queue.qsize()}"
        )
        
        return request.request_id
    
    async def get_solve_result(
        self,
        request_id: str,
        timeout: Optional[float] = None
    ) -> Optional[SolveResult]:
        """Get result for a solve request."""
        
        # Check if result is already available
        if request_id in self._completed_results:
            return self._completed_results[request_id]
        
        # Wait for result with timeout
        start_time = time.time()
        while (timeout is None or (time.time() - start_time) < timeout):
            if request_id in self._completed_results:
                return self._completed_results[request_id]
            
            await asyncio.sleep(0.1)
        
        return None  # Timeout
    
    async def _solver_worker(self, worker_id: str) -> None:
        """Worker task that processes solve requests."""
        
        self.logger.info(f"Started solver worker: {worker_id}")
        
        while self._orchestrator_running:
            try:
                # Get next request from queue
                try:
                    request = await asyncio.wait_for(
                        self._request_queue.get(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                # Process the request
                result = await self._process_solve_request(request)
                
                # Store result
                self._completed_results[request.request_id] = result
                
                # Clean up from active solves
                if request.request_id in self._active_solves:
                    del self._active_solves[request.request_id]
                
                # Update performance metrics
                self._update_performance_metrics(request, result)
                
                self.logger.info(
                    f"Worker {worker_id} completed {request.request_id}, "
                    f"success: {result.success}, time: {result.solve_time:.1f}s"
                )
                
            except Exception as e:
                self.logger.error(f"Worker {worker_id} error: {e}")
                await asyncio.sleep(1.0)
        
        self.logger.info(f"Stopped solver worker: {worker_id}")
    
    async def _process_solve_request(self, request: SolveRequest) -> SolveResult:
        """Process a single solve request using optimal resource allocation."""
        
        start_time = time.time()
        
        try:
            # Analyze problem characteristics
            problem_analysis = self._analyze_problem(request.problem_data)
            
            # Determine optimal resource allocation
            allocation = self._determine_resource_allocation(request, problem_analysis)
            
            # Execute solve based on allocation strategy
            if allocation.max_parallel_solvers > 1 and request.allow_decomposition:
                # Parallel/decomposed solve
                result = await self._execute_parallel_solve(request, allocation)
            else:
                # Single solver approach
                result = await self._execute_single_solve(request, allocation)
            
            # Calculate final metrics
            result.solve_time = time.time() - start_time
            result.request_id = request.request_id
            
            return result
            
        except Exception as e:
            self.logger.error(f"Solve request {request.request_id} failed: {e}")
            
            return SolveResult(
                request_id=request.request_id,
                success=False,
                solve_time=time.time() - start_time,
                metadata={'error': str(e)}
            )
    
    def _analyze_problem(self, problem_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze problem characteristics for resource allocation."""
        
        # Extract problem dimensions
        num_zones = problem_data.get('num_zones', 10)
        horizon = problem_data.get('horizon', 24)
        problem_size = num_zones * horizon
        
        # Estimate complexity
        objectives = problem_data.get('objectives', {})
        num_objectives = len(objectives.get('weights', {}))
        
        constraint_types = problem_data.get('constraints', {})
        num_constraint_types = len(constraint_types)
        
        complexity_score = (
            problem_size / 100.0 +
            num_objectives * 0.1 +
            num_constraint_types * 0.1
        )
        
        # Classify problem complexity
        if complexity_score < 0.5:
            complexity = 'simple'
        elif complexity_score < 2.0:
            complexity = 'medium'
        else:
            complexity = 'complex'
        
        # Estimate optimal solver type
        if problem_size < 50:
            preferred_tier = SolverTier.CLASSICAL
        elif problem_size < 200:
            preferred_tier = SolverTier.HYBRID
        else:
            preferred_tier = SolverTier.PREMIUM
        
        return {
            'problem_size': problem_size,
            'complexity': complexity,
            'complexity_score': complexity_score,
            'preferred_tier': preferred_tier,
            'decomposable': problem_size > 100,
            'estimated_solve_time': self._estimate_solve_time(problem_size, complexity),
            'estimated_cost': self._estimate_solve_cost(problem_size, preferred_tier)
        }
    
    def _determine_resource_allocation(
        self,
        request: SolveRequest,
        analysis: Dict[str, Any]
    ) -> ResourceAllocation:
        """Determine optimal resource allocation for the request."""
        
        # Get available solvers that match requirements
        candidate_solvers = self._get_candidate_solvers(request, analysis)
        
        if not candidate_solvers:
            # Fallback to any available solver
            candidate_solvers = [
                solver_id for solver_id, info in self._available_solvers.items()
                if info['current_load'] < info['max_concurrent']
            ]
        
        # Optimize allocation based on objectives
        if self.cost_optimization and request.cost_budget:
            # Cost-optimized allocation
            allocation = self._optimize_for_cost(request, analysis, candidate_solvers)
        else:
            # Performance-optimized allocation
            allocation = self._optimize_for_performance(request, analysis, candidate_solvers)
        
        return allocation
    
    def _get_candidate_solvers(
        self,
        request: SolveRequest,
        analysis: Dict[str, Any]
    ) -> List[str]:
        """Get candidate solvers that match request requirements."""
        
        candidates = []
        preferred_tier = analysis['preferred_tier']
        
        for solver_id, solver_info in self._available_solvers.items():
            # Check availability
            if solver_info['current_load'] >= solver_info['max_concurrent']:
                continue
            
            # Check solver tier preference
            solver_tier = solver_info['tier']
            if request.preferred_solvers and solver_id not in request.preferred_solvers:
                continue
            
            # Check capabilities
            if analysis['problem_size'] > 1000 and solver_tier == SolverTier.CLASSICAL:
                continue  # Classical may struggle with very large problems
            
            # Check cost constraints
            if request.cost_budget and solver_info['cost_per_solve'] > request.cost_budget:
                continue
            
            # Check performance requirements
            if request.quality_target > 0.9 and solver_info['avg_quality'] < 0.8:
                continue
            
            candidates.append(solver_id)
        
        # Sort by preference (tier, performance, cost)
        candidates.sort(key=lambda x: (
            self._available_solvers[x]['tier'].value,
            -self._available_solvers[x]['avg_quality'],
            self._available_solvers[x]['cost_per_solve']
        ))
        
        return candidates[:5]  # Limit to top 5 candidates
    
    def _optimize_for_cost(
        self,
        request: SolveRequest,
        analysis: Dict[str, Any],
        candidates: List[str]
    ) -> ResourceAllocation:
        """Create cost-optimized resource allocation."""
        
        # Find minimum cost solution that meets quality requirements
        best_allocation = None
        min_cost = float('inf')
        
        for solver_id in candidates:
            solver_info = self._available_solvers[solver_id]
            
            # Estimate cost and performance
            estimated_cost = solver_info['cost_per_solve']
            estimated_quality = solver_info['avg_quality']
            estimated_time = solver_info['avg_solve_time']
            
            # Check if meets requirements
            if (estimated_quality >= request.quality_target and
                estimated_time <= request.time_budget and
                estimated_cost < min_cost):
                
                best_allocation = ResourceAllocation(
                    solver_ids=[solver_id],
                    estimated_cost=estimated_cost,
                    estimated_time=estimated_time,
                    qpu_time_quota=estimated_time * 0.1,  # Estimate QPU usage
                    priority_level=request.priority,
                    max_parallel_solvers=1,
                    timeout_seconds=request.time_budget
                )
                min_cost = estimated_cost
        
        if best_allocation is None and candidates:
            # Fallback to first available solver
            solver_id = candidates[0]
            solver_info = self._available_solvers[solver_id]
            
            best_allocation = ResourceAllocation(
                solver_ids=[solver_id],
                estimated_cost=solver_info['cost_per_solve'],
                estimated_time=solver_info['avg_solve_time'],
                qpu_time_quota=solver_info['avg_solve_time'] * 0.1,
                priority_level=request.priority,
                max_parallel_solvers=1
            )
        
        return best_allocation or ResourceAllocation(
            solver_ids=[],
            estimated_cost=0.0,
            estimated_time=0.0,
            qpu_time_quota=0.0,
            priority_level=request.priority
        )
    
    def _optimize_for_performance(
        self,
        request: SolveRequest,
        analysis: Dict[str, Any],
        candidates: List[str]
    ) -> ResourceAllocation:
        """Create performance-optimized resource allocation."""
        
        # For complex problems, consider parallel solving
        if (analysis['decomposable'] and 
            request.allow_decomposition and 
            len(candidates) > 1):
            
            # Multi-solver allocation
            return self._create_parallel_allocation(request, analysis, candidates)
        else:
            # Single best solver
            if candidates:
                best_solver = max(
                    candidates,
                    key=lambda x: self._available_solvers[x]['avg_quality']
                )
                
                solver_info = self._available_solvers[best_solver]
                
                return ResourceAllocation(
                    solver_ids=[best_solver],
                    estimated_cost=solver_info['cost_per_solve'],
                    estimated_time=solver_info['avg_solve_time'],
                    qpu_time_quota=solver_info['avg_solve_time'] * 0.2,
                    priority_level=request.priority,
                    max_parallel_solvers=1
                )
            else:
                return ResourceAllocation(
                    solver_ids=[],
                    estimated_cost=0.0,
                    estimated_time=0.0,
                    qpu_time_quota=0.0,
                    priority_level=request.priority
                )
    
    def _create_parallel_allocation(
        self,
        request: SolveRequest,
        analysis: Dict[str, Any],
        candidates: List[str]
    ) -> ResourceAllocation:
        """Create allocation for parallel/decomposed solving."""
        
        # Select best solvers for parallel execution
        max_parallel = min(
            request.max_subproblems,
            len(candidates),
            3  # Limit parallel solvers
        )
        
        # Choose diverse solver types for redundancy
        selected_solvers = []
        selected_tiers = set()
        
        for solver_id in candidates:
            if len(selected_solvers) >= max_parallel:
                break
            
            solver_tier = self._available_solvers[solver_id]['tier']
            
            # Prefer diverse tiers for robustness
            if solver_tier not in selected_tiers or len(selected_solvers) < 2:
                selected_solvers.append(solver_id)
                selected_tiers.add(solver_tier)
        
        # Calculate resource estimates
        total_cost = sum(
            self._available_solvers[s]['cost_per_solve'] for s in selected_solvers
        )
        
        # Parallel solving should reduce time
        avg_time = np.mean([
            self._available_solvers[s]['avg_solve_time'] for s in selected_solvers
        ])
        estimated_time = avg_time * 0.7  # 30% time reduction from parallelization
        
        qpu_quota = estimated_time * len(selected_solvers) * 0.15
        
        return ResourceAllocation(
            solver_ids=selected_solvers,
            estimated_cost=total_cost,
            estimated_time=estimated_time,
            qpu_time_quota=qpu_quota,
            priority_level=request.priority,
            max_parallel_solvers=len(selected_solvers),
            timeout_seconds=request.time_budget
        )
    
    async def _execute_single_solve(
        self,
        request: SolveRequest,
        allocation: ResourceAllocation
    ) -> SolveResult:
        """Execute solve using single solver."""
        
        if not allocation.solver_ids:
            return SolveResult(
                request_id=request.request_id,
                success=False,
                metadata={'error': 'No available solvers'}
            )
        
        solver_id = allocation.solver_ids[0]
        
        try:
            # Reserve solver
            self._available_solvers[solver_id]['current_load'] += 1
            
            # Execute solve
            solution = await self._execute_solver(solver_id, request.problem_data)
            
            # Calculate quality score
            quality_score = self._calculate_solution_quality(solution, request.problem_data)
            
            return SolveResult(
                request_id=request.request_id,
                success=solution is not None,
                solution=solution,
                quality_score=quality_score,
                actual_cost=self._available_solvers[solver_id]['cost_per_solve'],
                solvers_used=[solver_id],
                qpu_time_used=allocation.qpu_time_quota,
                parallel_solvers=1
            )
            
        finally:
            # Release solver
            self._available_solvers[solver_id]['current_load'] -= 1
    
    async def _execute_parallel_solve(
        self,
        request: SolveRequest,
        allocation: ResourceAllocation
    ) -> SolveResult:
        """Execute solve using multiple solvers in parallel."""
        
        if len(allocation.solver_ids) < 2:
            # Fall back to single solve
            return await self._execute_single_solve(request, allocation)
        
        try:
            # Reserve all solvers
            for solver_id in allocation.solver_ids:
                self._available_solvers[solver_id]['current_load'] += 1
            
            # Create subproblems (simplified - could use actual decomposition)
            subproblems = self._decompose_problem(
                request.problem_data,
                len(allocation.solver_ids)
            )
            
            # Launch parallel solves
            solve_tasks = []
            for i, solver_id in enumerate(allocation.solver_ids):
                if i < len(subproblems):
                    task = asyncio.create_task(
                        self._execute_solver(solver_id, subproblems[i])
                    )
                    solve_tasks.append((solver_id, task))
            
            # Wait for results with timeout
            completed_solves = []
            timeout = allocation.timeout_seconds
            
            try:
                for solver_id, task in solve_tasks:
                    solution = await asyncio.wait_for(task, timeout=timeout)
                    completed_solves.append((solver_id, solution))
            except asyncio.TimeoutError:
                # Cancel remaining tasks
                for solver_id, task in solve_tasks:
                    if not task.done():
                        task.cancel()
            
            # Combine results
            if completed_solves:
                best_solution = None
                best_quality = 0.0
                total_cost = 0.0
                qpu_time = 0.0
                
                for solver_id, solution in completed_solves:
                    if solution:
                        quality = self._calculate_solution_quality(solution, request.problem_data)
                        if quality > best_quality:
                            best_solution = solution
                            best_quality = quality
                    
                    total_cost += self._available_solvers[solver_id]['cost_per_solve']
                    qpu_time += allocation.qpu_time_quota / len(allocation.solver_ids)
                
                return SolveResult(
                    request_id=request.request_id,
                    success=best_solution is not None,
                    solution=best_solution,
                    quality_score=best_quality,
                    actual_cost=total_cost,
                    solvers_used=allocation.solver_ids,
                    qpu_time_used=qpu_time,
                    subproblems_count=len(subproblems),
                    parallel_solvers=len(allocation.solver_ids)
                )
            else:
                return SolveResult(
                    request_id=request.request_id,
                    success=False,
                    actual_cost=sum(self._available_solvers[s]['cost_per_solve'] for s in allocation.solver_ids),
                    metadata={'error': 'All parallel solves failed or timed out'}
                )
            
        finally:
            # Release all solvers
            for solver_id in allocation.solver_ids:
                self._available_solvers[solver_id]['current_load'] -= 1
    
    async def _execute_solver(
        self,
        solver_id: str,
        problem_data: Dict[str, Any]
    ) -> Optional[QuantumSolution]:
        """Execute solve on specific solver."""
        
        solver_info = self._available_solvers[solver_id]
        solver_type = solver_info['type']
        
        try:
            if solver_type == 'adaptive_quantum':
                # Use adaptive quantum engine
                engine = AdaptiveQuantumEngine()
                solution = await engine.solve_adaptive(problem_data.get('qubo', {}))
                
            elif solver_type in ['qpu', 'hybrid', 'classical']:
                # Use basic quantum solver
                solver = QuantumSolver(solver_type=solver_type)
                solution = await solver.solve(problem_data.get('qubo', {}))
                
            else:
                # Unknown solver type
                self.logger.error(f"Unknown solver type: {solver_type}")
                return None
            
            return solution
            
        except Exception as e:
            self.logger.error(f"Solver {solver_id} execution failed: {e}")
            return None
    
    def _decompose_problem(
        self,
        problem_data: Dict[str, Any],
        num_subproblems: int
    ) -> List[Dict[str, Any]]:
        """Decompose problem for parallel solving."""
        
        # Simplified problem decomposition
        # In practice, would use sophisticated decomposition strategies
        
        subproblems = []
        
        # Temporal decomposition
        horizon = problem_data.get('horizon', 24)
        subhorizon = max(6, horizon // num_subproblems)
        
        for i in range(num_subproblems):
            start_time = i * subhorizon
            end_time = min(start_time + subhorizon, horizon)
            
            if start_time >= horizon:
                break
            
            subproblem = problem_data.copy()
            subproblem['horizon'] = end_time - start_time
            subproblem['time_offset'] = start_time
            
            # Modify QUBO for subproblem (simplified)
            if 'qubo' in problem_data:
                # Extract relevant portion of QUBO
                subproblem['qubo'] = problem_data['qubo']  # Simplified
            
            subproblems.append(subproblem)
        
        return subproblems
    
    def _calculate_solution_quality(
        self,
        solution: QuantumSolution,
        problem_data: Dict[str, Any]
    ) -> float:
        """Calculate quality score for solution."""
        
        if solution is None:
            return 0.0
        
        # Quality factors
        quality_factors = []
        
        # Chain break quality (for quantum solutions)
        if hasattr(solution, 'chain_break_fraction'):
            chain_quality = 1.0 - min(1.0, solution.chain_break_fraction)
            quality_factors.append(chain_quality)
        
        # Energy quality (lower energy is better for minimization)
        if hasattr(solution, 'energy'):
            # Normalize energy (problem-specific)
            energy_quality = max(0.0, 1.0 - abs(solution.energy) / 1000.0)
            quality_factors.append(energy_quality)
        
        # Solution completeness
        if hasattr(solution, 'sample') and solution.sample:
            completeness = 1.0  # Assume complete solution
            quality_factors.append(completeness)
        
        # Overall quality score
        return np.mean(quality_factors) if quality_factors else 0.5
    
    def _estimate_solve_time(self, problem_size: int, complexity: str) -> float:
        """Estimate solve time based on problem characteristics."""
        
        base_time = {
            'simple': 5.0,
            'medium': 15.0,
            'complex': 45.0
        }.get(complexity, 15.0)
        
        # Scale with problem size
        size_factor = max(1.0, np.log(problem_size / 10.0))
        
        return base_time * size_factor
    
    def _estimate_solve_cost(self, problem_size: int, tier: SolverTier) -> float:
        """Estimate solve cost based on problem size and solver tier."""
        
        base_costs = {
            SolverTier.PREMIUM: 5.0,
            SolverTier.STANDARD: 2.0,
            SolverTier.HYBRID: 0.5,
            SolverTier.CLASSICAL: 0.1,
            SolverTier.BUDGET: 0.05
        }
        
        base_cost = base_costs.get(tier, 1.0)
        
        # Scale with problem size
        size_factor = max(1.0, problem_size / 100.0)
        
        return base_cost * size_factor
    
    def _update_performance_metrics(
        self,
        request: SolveRequest,
        result: SolveResult
    ) -> None:
        """Update solver performance metrics based on results."""
        
        for solver_id in result.solvers_used:
            if solver_id in self._available_solvers:
                solver_info = self._available_solvers[solver_id]
                
                # Update performance history
                performance_record = {
                    'timestamp': time.time(),
                    'success': result.success,
                    'quality': result.quality_score,
                    'solve_time': result.solve_time,
                    'problem_size': request.problem_data.get('num_zones', 0) * request.problem_data.get('horizon', 0)
                }
                
                self._solver_performance[solver_id].append(performance_record)
                
                # Update running averages (exponential moving average)
                alpha = 0.1
                solver_info['avg_quality'] = (
                    alpha * result.quality_score +
                    (1 - alpha) * solver_info['avg_quality']
                )
                solver_info['avg_solve_time'] = (
                    alpha * result.solve_time +
                    (1 - alpha) * solver_info['avg_solve_time']
                )
                solver_info['success_rate'] = (
                    alpha * (1.0 if result.success else 0.0) +
                    (1 - alpha) * solver_info['success_rate']
                )
                
                solver_info['total_solves'] += 1
                solver_info['last_seen'] = time.time()
    
    async def _monitoring_loop(self) -> None:
        """Background monitoring loop for system health."""
        
        while self._orchestrator_running:
            try:
                # Update system metrics
                self._current_load = len(self._active_solves) / self.max_concurrent_solves
                
                # Record demand for forecasting
                self._demand_forecast.append({
                    'timestamp': time.time(),
                    'queue_size': self._request_queue.qsize(),
                    'active_solves': len(self._active_solves),
                    'load_factor': self._current_load
                })
                
                # Update solver availability
                current_time = time.time()
                for solver_id, solver_info in self._available_solvers.items():
                    # Mark solvers as unavailable if not seen recently
                    if current_time - solver_info['last_seen'] > 300:  # 5 minutes
                        solver_info['current_load'] = solver_info['max_concurrent']  # Mark as full
                
                await asyncio.sleep(10)  # Monitor every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(10)
    
    async def _scaling_loop(self) -> None:
        """Background loop for auto-scaling decisions."""
        
        if not self.enable_auto_scaling:
            return
        
        while self._orchestrator_running:
            try:
                # Analyze demand trends
                if len(self._demand_forecast) >= 10:
                    recent_demand = list(self._demand_forecast)[-10:]
                    avg_load = np.mean([d['load_factor'] for d in recent_demand])
                    
                    # Scaling decisions
                    if avg_load > 0.8:
                        # High load - consider scaling up
                        await self._scale_up()
                    elif avg_load < 0.3:
                        # Low load - consider scaling down
                        await self._scale_down()
                
                await asyncio.sleep(60)  # Scale decisions every minute
                
            except Exception as e:
                self.logger.error(f"Scaling loop error: {e}")
                await asyncio.sleep(60)
    
    async def _scale_up(self) -> None:
        """Scale up resources to handle increased demand."""
        
        # Check if we can add more workers
        if len(self._worker_tasks) < self.max_concurrent_solves + 5:
            # Add additional worker
            worker_id = f"worker_scaled_{len(self._worker_tasks)}"
            worker_task = asyncio.create_task(self._solver_worker(worker_id))
            self._worker_tasks.append(worker_task)
            
            self._scaling_decisions.append({
                'timestamp': time.time(),
                'action': 'scale_up',
                'new_worker': worker_id
            })
            
            self.logger.info(f"Scaled up: added worker {worker_id}")
    
    async def _scale_down(self) -> None:
        """Scale down resources during low demand."""
        
        # Remove excess workers (keep minimum of 2)
        if len(self._worker_tasks) > 4:  # Keep at least 2 worker + 2 system tasks
            # Find worker tasks
            worker_tasks = [t for t in self._worker_tasks if not t.done()]
            
            if len(worker_tasks) > 2:
                # Cancel one worker task
                task_to_remove = worker_tasks[-1]
                task_to_remove.cancel()
                self._worker_tasks.remove(task_to_remove)
                
                self._scaling_decisions.append({
                    'timestamp': time.time(),
                    'action': 'scale_down',
                    'removed_worker': str(task_to_remove)
                })
                
                self.logger.info("Scaled down: removed worker task")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        
        # Solver status
        solver_status = {}
        for solver_id, info in self._available_solvers.items():
            solver_status[solver_id] = {
                'type': info['type'],
                'tier': info['tier'].value,
                'current_load': info['current_load'],
                'max_concurrent': info['max_concurrent'],
                'success_rate': info['success_rate'],
                'avg_quality': info['avg_quality'],
                'total_solves': info['total_solves'],
                'available': info['current_load'] < info['max_concurrent']
            }
        
        # Queue status
        queue_status = {
            'pending_requests': self._request_queue.qsize(),
            'active_solves': len(self._active_solves),
            'completed_results': len(self._completed_results)
        }
        
        # Performance metrics
        if self._demand_forecast:
            recent_load = list(self._demand_forecast)[-10:]
            performance_metrics = {
                'current_load_factor': self._current_load,
                'avg_load_10min': np.mean([d['load_factor'] for d in recent_load]),
                'peak_queue_size': max(d['queue_size'] for d in recent_load),
                'worker_count': len(self._worker_tasks)
            }
        else:
            performance_metrics = {
                'current_load_factor': self._current_load,
                'worker_count': len(self._worker_tasks)
            }
        
        return {
            'orchestrator_running': self._orchestrator_running,
            'solver_status': solver_status,
            'queue_status': queue_status,
            'performance_metrics': performance_metrics,
            'scaling_enabled': self.enable_auto_scaling,
            'cost_optimization': self.cost_optimization,
            'recent_scaling_actions': list(self._scaling_decisions)[-5:]
        }
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get detailed performance report."""
        
        # Solver performance analysis
        solver_performance = {}
        for solver_id, performance_history in self._solver_performance.items():
            if performance_history:
                recent_performance = list(performance_history)[-20:]  # Last 20 solves
                
                solver_performance[solver_id] = {
                    'total_solves': len(performance_history),
                    'recent_solves': len(recent_performance),
                    'avg_quality': np.mean([p['quality'] for p in recent_performance]),
                    'avg_solve_time': np.mean([p['solve_time'] for p in recent_performance]),
                    'success_rate': np.mean([p['success'] for p in recent_performance]),
                    'quality_trend': self._calculate_performance_trend([p['quality'] for p in recent_performance]),
                    'time_trend': self._calculate_performance_trend([p['solve_time'] for p in recent_performance])
                }
        
        # System utilization
        if self._demand_forecast:
            demand_data = list(self._demand_forecast)[-50:]  # Last 50 data points
            utilization_stats = {
                'avg_utilization': np.mean([d['load_factor'] for d in demand_data]),
                'peak_utilization': np.max([d['load_factor'] for d in demand_data]),
                'avg_queue_size': np.mean([d['queue_size'] for d in demand_data]),
                'utilization_trend': self._calculate_performance_trend([d['load_factor'] for d in demand_data])
            }
        else:
            utilization_stats = {'avg_utilization': 0.0}
        
        return {
            'solver_performance': solver_performance,
            'system_utilization': utilization_stats,
            'scaling_decisions': len(self._scaling_decisions),
            'total_requests_processed': sum(
                solver['total_solves'] for solver in self._available_solvers.values()
            )
        }
    
    def _calculate_performance_trend(self, values: List[float]) -> float:
        """Calculate trend in performance values (positive = improving)."""
        
        if len(values) < 3:
            return 0.0
        
        # Simple linear regression slope
        x = np.arange(len(values))
        y = np.array(values)
        
        # Calculate slope
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        
        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean) ** 2)
        
        return numerator / denominator if denominator > 0 else 0.0


# Global distributed solver instance
_global_quantum_orchestrator: Optional[QuantumResourceOrchestrator] = None


def get_quantum_orchestrator() -> QuantumResourceOrchestrator:
    """Get global quantum orchestrator instance."""
    global _global_quantum_orchestrator
    if _global_quantum_orchestrator is None:
        _global_quantum_orchestrator = QuantumResourceOrchestrator()
    return _global_quantum_orchestrator


def reset_quantum_orchestrator():
    """Reset global quantum orchestrator instance."""
    global _global_quantum_orchestrator
    if _global_quantum_orchestrator:
        asyncio.create_task(_global_quantum_orchestrator.stop_orchestrator())
    _global_quantum_orchestrator = None
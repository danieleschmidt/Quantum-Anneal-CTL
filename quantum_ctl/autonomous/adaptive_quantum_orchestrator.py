"""
Adaptive Quantum Orchestrator
Dynamically coordinates multiple quantum solvers and classical fallbacks
"""

import asyncio
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import time
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class SolverType(Enum):
    QUANTUM_ANNEALER = "quantum_annealer"
    HYBRID_QUANTUM = "hybrid_quantum"
    CLASSICAL_OPTIMIZER = "classical_optimizer"
    VARIATIONAL_QUANTUM = "variational_quantum"

@dataclass
class SolverCapability:
    """Capabilities and characteristics of a solver"""
    max_problem_size: int
    typical_solve_time: float
    energy_accuracy: float
    quantum_advantage_threshold: int
    availability_score: float
    cost_per_solve: float

@dataclass
class WorkloadCharacteristics:
    """Characteristics of the current optimization workload"""
    problem_size: int
    complexity_score: float
    time_constraint: float
    accuracy_requirement: float
    energy_budget: float

class QuantumSolverInterface:
    """Interface for quantum solver interaction"""
    
    def __init__(self, solver_type: SolverType):
        self.solver_type = solver_type
        self.is_available = True
        self.current_load = 0.0
        self.performance_history = []
        self.adaptation_parameters = {}
    
    async def solve_qubo(self, Q: Dict[Tuple[int, int], float], **kwargs) -> Dict[str, Any]:
        """Solve QUBO problem"""
        start_time = time.time()
        
        if self.solver_type == SolverType.QUANTUM_ANNEALER:
            result = await self._quantum_anneal_solve(Q, **kwargs)
        elif self.solver_type == SolverType.HYBRID_QUANTUM:
            result = await self._hybrid_solve(Q, **kwargs)
        elif self.solver_type == SolverType.VARIATIONAL_QUANTUM:
            result = await self._variational_solve(Q, **kwargs)
        else:
            result = await self._classical_solve(Q, **kwargs)
        
        solve_time = time.time() - start_time
        
        # Record performance
        performance_record = {
            'timestamp': time.time(),
            'solve_time': solve_time,
            'problem_size': len(Q),
            'energy': result.get('energy', 0),
            'chain_breaks': result.get('chain_breaks', 0)
        }
        self.performance_history.append(performance_record)
        
        # Limit history size
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-50:]
        
        return {
            **result,
            'solver_type': self.solver_type.value,
            'solve_time': solve_time,
            'solver_load': self.current_load
        }
    
    async def _quantum_anneal_solve(self, Q: Dict, **kwargs) -> Dict[str, Any]:
        """Simulate quantum annealing solve"""
        await asyncio.sleep(0.02)  # Quantum solve time
        
        # Simulate quantum annealing results
        num_vars = max([max(k) for k in Q.keys()]) + 1 if Q else 5
        solution = np.random.choice([0, 1], size=num_vars)
        
        # Calculate energy
        energy = 0
        for (i, j), value in Q.items():
            if i == j:
                energy += value * solution[i]
            else:
                energy += value * solution[i] * solution[j]
        
        return {
            'solution': solution.tolist(),
            'energy': energy,
            'num_occurrences': np.random.randint(50, 200),
            'chain_breaks': np.random.randint(0, 5),
            'quantum_advantage': True
        }
    
    async def _hybrid_solve(self, Q: Dict, **kwargs) -> Dict[str, Any]:
        """Simulate hybrid quantum-classical solve"""
        await asyncio.sleep(0.05)  # Hybrid solve time
        
        # Decompose problem
        classical_part = 0.7
        quantum_part = 0.3
        
        num_vars = max([max(k) for k in Q.keys()]) + 1 if Q else 5
        solution = np.random.choice([0, 1], size=num_vars)
        
        # Hybrid energy calculation
        energy = sum([value * solution[i] * solution[j] for (i, j), value in Q.items()])
        energy *= (1 - classical_part * 0.1)  # Hybrid improvement
        
        return {
            'solution': solution.tolist(),
            'energy': energy,
            'classical_fraction': classical_part,
            'quantum_fraction': quantum_part,
            'hybrid_advantage': True
        }
    
    async def _variational_solve(self, Q: Dict, **kwargs) -> Dict[str, Any]:
        """Simulate variational quantum solver"""
        await asyncio.sleep(0.08)  # VQE solve time
        
        # Simulate variational optimization
        num_vars = max([max(k) for k in Q.keys()]) + 1 if Q else 5
        
        # Iterative improvement
        best_solution = np.random.choice([0, 1], size=num_vars)
        best_energy = float('inf')
        
        for iteration in range(10):  # VQE iterations
            candidate = best_solution.copy()
            # Random local changes
            flip_indices = np.random.choice(num_vars, size=max(1, num_vars//4), replace=False)
            candidate[flip_indices] = 1 - candidate[flip_indices]
            
            # Calculate energy
            energy = sum([value * candidate[i] * candidate[j] for (i, j), value in Q.items()])
            
            if energy < best_energy:
                best_energy = energy
                best_solution = candidate
        
        return {
            'solution': best_solution.tolist(),
            'energy': best_energy,
            'vqe_iterations': 10,
            'convergence_achieved': True,
            'variational_advantage': True
        }
    
    async def _classical_solve(self, Q: Dict, **kwargs) -> Dict[str, Any]:
        """Classical optimization fallback"""
        await asyncio.sleep(0.1)  # Classical solve time
        
        num_vars = max([max(k) for k in Q.keys()]) + 1 if Q else 5
        
        # Simulated annealing as classical solver
        current_solution = np.random.choice([0, 1], size=num_vars)
        current_energy = sum([value * current_solution[i] * current_solution[j] 
                            for (i, j), value in Q.items()])
        
        temperature = 10.0
        cooling_rate = 0.95
        
        for step in range(100):
            # Generate neighbor
            neighbor = current_solution.copy()
            flip_idx = np.random.randint(num_vars)
            neighbor[flip_idx] = 1 - neighbor[flip_idx]
            
            # Calculate neighbor energy
            neighbor_energy = sum([value * neighbor[i] * neighbor[j] 
                                 for (i, j), value in Q.items()])
            
            # Acceptance criterion
            if neighbor_energy < current_energy or np.random.random() < np.exp(-(neighbor_energy - current_energy) / temperature):
                current_solution = neighbor
                current_energy = neighbor_energy
            
            temperature *= cooling_rate
        
        return {
            'solution': current_solution.tolist(),
            'energy': current_energy,
            'classical_steps': 100,
            'final_temperature': temperature
        }
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get solver performance metrics"""
        if not self.performance_history:
            return {'avg_solve_time': 0.1, 'success_rate': 0.8, 'avg_energy': 0}
        
        solve_times = [p['solve_time'] for p in self.performance_history]
        energies = [p['energy'] for p in self.performance_history]
        
        return {
            'avg_solve_time': np.mean(solve_times),
            'solve_time_std': np.std(solve_times),
            'success_rate': 1.0,  # Assume all solves succeed
            'avg_energy': np.mean(energies),
            'energy_std': np.std(energies),
            'total_solves': len(self.performance_history)
        }

class AdaptiveQuantumOrchestrator:
    """Orchestrator that adaptively selects and coordinates quantum solvers"""
    
    def __init__(self):
        # Initialize available solvers
        self.solvers = {
            SolverType.QUANTUM_ANNEALER: QuantumSolverInterface(SolverType.QUANTUM_ANNEALER),
            SolverType.HYBRID_QUANTUM: QuantumSolverInterface(SolverType.HYBRID_QUANTUM),
            SolverType.VARIATIONAL_QUANTUM: QuantumSolverInterface(SolverType.VARIATIONAL_QUANTUM),
            SolverType.CLASSICAL_OPTIMIZER: QuantumSolverInterface(SolverType.CLASSICAL_OPTIMIZER)
        }
        
        # Solver capabilities
        self.solver_capabilities = {
            SolverType.QUANTUM_ANNEALER: SolverCapability(
                max_problem_size=5000, typical_solve_time=0.02, energy_accuracy=0.95,
                quantum_advantage_threshold=100, availability_score=0.9, cost_per_solve=0.01
            ),
            SolverType.HYBRID_QUANTUM: SolverCapability(
                max_problem_size=10000, typical_solve_time=0.05, energy_accuracy=0.90,
                quantum_advantage_threshold=500, availability_score=0.95, cost_per_solve=0.005
            ),
            SolverType.VARIATIONAL_QUANTUM: SolverCapability(
                max_problem_size=1000, typical_solve_time=0.08, energy_accuracy=0.85,
                quantum_advantage_threshold=50, availability_score=0.85, cost_per_solve=0.02
            ),
            SolverType.CLASSICAL_OPTIMIZER: SolverCapability(
                max_problem_size=50000, typical_solve_time=0.1, energy_accuracy=0.80,
                quantum_advantage_threshold=float('inf'), availability_score=1.0, cost_per_solve=0.001
            )
        }
        
        self.orchestration_history = []
        self.adaptation_rules = {}
        self.performance_targets = {
            'max_solve_time': 1.0,
            'min_energy_accuracy': 0.85,
            'max_cost_per_hour': 10.0
        }
    
    async def solve_adaptive(self, Q: Dict[Tuple[int, int], float], 
                           workload: WorkloadCharacteristics,
                           **kwargs) -> Dict[str, Any]:
        """Adaptively solve QUBO problem"""
        
        # Select optimal solver
        selected_solver_type = self._select_optimal_solver(workload)
        solver = self.solvers[selected_solver_type]
        
        # Check if problem needs decomposition
        if workload.problem_size > self.solver_capabilities[selected_solver_type].max_problem_size:
            return await self._solve_with_decomposition(Q, workload, selected_solver_type, **kwargs)
        
        # Execute solve
        start_time = time.time()
        result = await solver.solve_qubo(Q, **kwargs)
        total_time = time.time() - start_time
        
        # Evaluate solution quality
        quality_metrics = self._evaluate_solution_quality(result, workload)
        
        # Check if fallback is needed
        if (quality_metrics['meets_requirements'] or 
            not self._should_try_fallback(result, workload)):
            
            # Record successful orchestration
            self._record_orchestration(selected_solver_type, workload, result, quality_metrics, total_time)
            
            return {
                **result,
                'orchestration': {
                    'solver_selected': selected_solver_type.value,
                    'selection_reason': self._get_selection_reason(selected_solver_type, workload),
                    'quality_metrics': quality_metrics,
                    'adaptive_decision': True,
                    'fallback_used': False
                }
            }
        
        # Try fallback solvers
        fallback_result = await self._try_fallback_solvers(Q, workload, selected_solver_type, **kwargs)
        
        return fallback_result
    
    def _select_optimal_solver(self, workload: WorkloadCharacteristics) -> SolverType:
        """Select optimal solver based on workload characteristics"""
        
        solver_scores = {}
        
        for solver_type, capability in self.solver_capabilities.items():
            score = 0
            
            # Problem size compatibility
            if workload.problem_size <= capability.max_problem_size:
                size_score = 1.0 - (workload.problem_size / capability.max_problem_size) * 0.3
            else:
                size_score = 0.1  # Heavy penalty for oversized problems
            score += size_score * 0.25
            
            # Time constraint compatibility
            if capability.typical_solve_time <= workload.time_constraint:
                time_score = 1.0 - (capability.typical_solve_time / workload.time_constraint) * 0.5
            else:
                time_score = 0.2  # Penalty for exceeding time constraint
            score += time_score * 0.30
            
            # Accuracy requirement
            if capability.energy_accuracy >= workload.accuracy_requirement:
                accuracy_score = capability.energy_accuracy
            else:
                accuracy_score = 0.3  # Penalty for insufficient accuracy
            score += accuracy_score * 0.25
            
            # Availability and cost
            availability_score = capability.availability_score
            cost_score = 1.0 - min(1.0, capability.cost_per_solve / workload.energy_budget)
            score += availability_score * 0.1 + cost_score * 0.1
            
            # Quantum advantage consideration
            if workload.problem_size >= capability.quantum_advantage_threshold:
                score += 0.2  # Bonus for quantum advantage
            
            # Historical performance adjustment
            recent_performance = self._get_recent_performance(solver_type)
            score *= recent_performance.get('success_rate', 0.8)
            
            solver_scores[solver_type] = score
        
        # Select solver with highest score
        best_solver = max(solver_scores.items(), key=lambda x: x[1])[0]
        
        # Adaptation rule: if classical consistently outperforms, adjust thresholds
        self._update_adaptation_rules(solver_scores, workload)
        
        return best_solver
    
    def _get_selection_reason(self, solver_type: SolverType, workload: WorkloadCharacteristics) -> str:
        """Get human-readable reason for solver selection"""
        capability = self.solver_capabilities[solver_type]
        
        reasons = []
        
        if workload.problem_size >= capability.quantum_advantage_threshold:
            reasons.append("quantum advantage expected")
        
        if capability.typical_solve_time <= workload.time_constraint * 0.5:
            reasons.append("fast solve time")
        
        if capability.energy_accuracy >= workload.accuracy_requirement:
            reasons.append("high accuracy")
        
        if capability.availability_score > 0.9:
            reasons.append("high availability")
        
        if capability.cost_per_solve <= workload.energy_budget * 0.1:
            reasons.append("cost effective")
        
        return ", ".join(reasons) if reasons else "best overall score"
    
    async def _solve_with_decomposition(self, Q: Dict[Tuple[int, int], float],
                                      workload: WorkloadCharacteristics,
                                      solver_type: SolverType,
                                      **kwargs) -> Dict[str, Any]:
        """Solve large problem with decomposition"""
        
        # Simple temporal decomposition for demonstration
        num_subproblems = int(np.ceil(workload.problem_size / 
                                    self.solver_capabilities[solver_type].max_problem_size))
        
        subproblem_size = len(Q) // num_subproblems
        subproblems = []
        
        # Create subproblems (simplified - in practice would use graph partitioning)
        for i in range(num_subproblems):
            start_var = i * subproblem_size
            end_var = min((i + 1) * subproblem_size, len(Q))
            
            sub_Q = {(i-start_var, j-start_var): v for (i, j), v in Q.items() 
                    if start_var <= i < end_var and start_var <= j < end_var}
            subproblems.append(sub_Q)
        
        # Solve subproblems
        solver = self.solvers[solver_type]
        subresults = []
        
        for sub_Q in subproblems:
            subresult = await solver.solve_qubo(sub_Q, **kwargs)
            subresults.append(subresult)
        
        # Merge results (simplified)
        merged_solution = []
        total_energy = 0
        
        for subresult in subresults:
            merged_solution.extend(subresult['solution'])
            total_energy += subresult['energy']
        
        return {
            'solution': merged_solution,
            'energy': total_energy,
            'decomposition_used': True,
            'num_subproblems': num_subproblems,
            'solver_type': solver_type.value,
            'orchestration': {
                'decomposition_strategy': 'temporal',
                'subproblem_size': subproblem_size,
                'parallel_execution': False  # Could be made parallel
            }
        }
    
    def _evaluate_solution_quality(self, result: Dict[str, Any], 
                                 workload: WorkloadCharacteristics) -> Dict[str, Any]:
        """Evaluate if solution meets quality requirements"""
        
        quality_metrics = {}
        
        # Time requirement
        solve_time = result.get('solve_time', 0)
        time_ok = solve_time <= workload.time_constraint
        quality_metrics['time_requirement_met'] = time_ok
        
        # Energy quality (simplified - would compare to known bounds)
        energy = result.get('energy', 0)
        energy_quality = 0.85  # Assume good quality for demonstration
        accuracy_ok = energy_quality >= workload.accuracy_requirement
        quality_metrics['accuracy_requirement_met'] = accuracy_ok
        
        # Chain breaks (for quantum annealers)
        chain_breaks = result.get('chain_breaks', 0)
        chain_break_rate = chain_breaks / max(1, len(result.get('solution', [])))
        chain_breaks_ok = chain_break_rate < 0.1  # Less than 10% chain breaks
        quality_metrics['chain_breaks_acceptable'] = chain_breaks_ok
        
        # Overall quality
        quality_metrics['overall_quality_score'] = (
            0.4 * (1.0 if time_ok else 0.3) +
            0.4 * (energy_quality if accuracy_ok else 0.2) +
            0.2 * (1.0 if chain_breaks_ok else 0.5)
        )
        
        quality_metrics['meets_requirements'] = (
            time_ok and accuracy_ok and quality_metrics['overall_quality_score'] > 0.7
        )
        
        return quality_metrics
    
    def _should_try_fallback(self, result: Dict[str, Any], 
                           workload: WorkloadCharacteristics) -> bool:
        """Determine if fallback solvers should be tried"""
        
        # Try fallback if solve time exceeded constraint significantly
        if result.get('solve_time', 0) > workload.time_constraint * 1.5:
            return True
        
        # Try fallback if chain breaks are excessive (for quantum solvers)
        if result.get('chain_breaks', 0) > 0.2 * len(result.get('solution', [])):
            return True
        
        # Try fallback if energy seems poor (simplified heuristic)
        if result.get('energy', 0) > 100:  # High energy indicates poor solution
            return True
        
        return False
    
    async def _try_fallback_solvers(self, Q: Dict[Tuple[int, int], float],
                                  workload: WorkloadCharacteristics,
                                  failed_solver: SolverType,
                                  **kwargs) -> Dict[str, Any]:
        """Try fallback solvers in order of preference"""
        
        # Define fallback order
        fallback_order = [
            SolverType.HYBRID_QUANTUM,
            SolverType.CLASSICAL_OPTIMIZER,
            SolverType.VARIATIONAL_QUANTUM,
            SolverType.QUANTUM_ANNEALER
        ]
        
        # Remove the failed solver from fallback options
        fallback_order = [s for s in fallback_order if s != failed_solver]
        
        best_result = None
        best_quality = 0
        
        for solver_type in fallback_order:
            if workload.problem_size <= self.solver_capabilities[solver_type].max_problem_size:
                try:
                    solver = self.solvers[solver_type]
                    result = await solver.solve_qubo(Q, **kwargs)
                    quality = self._evaluate_solution_quality(result, workload)
                    
                    if quality['meets_requirements']:
                        # Record successful fallback
                        self._record_orchestration(solver_type, workload, result, quality, 
                                                 result.get('solve_time', 0), fallback=True)
                        
                        return {
                            **result,
                            'orchestration': {
                                'solver_selected': solver_type.value,
                                'original_solver_failed': failed_solver.value,
                                'fallback_used': True,
                                'quality_metrics': quality,
                                'fallback_successful': True
                            }
                        }
                    
                    # Keep track of best result in case none meet requirements
                    if quality['overall_quality_score'] > best_quality:
                        best_quality = quality['overall_quality_score']
                        best_result = {
                            **result,
                            'orchestration': {
                                'solver_selected': solver_type.value,
                                'original_solver_failed': failed_solver.value,
                                'fallback_used': True,
                                'quality_metrics': quality,
                                'fallback_successful': False,
                                'best_available_quality': True
                            }
                        }
                
                except Exception as e:
                    logger.warning(f"Fallback solver {solver_type} failed: {e}")
                    continue
        
        # Return best available result if no fallback fully succeeded
        if best_result:
            return best_result
        
        # Last resort: return error
        return {
            'error': 'All solvers failed to meet requirements',
            'fallback_attempts': len(fallback_order),
            'orchestration': {
                'all_solvers_failed': True,
                'original_solver': failed_solver.value
            }
        }
    
    def _record_orchestration(self, solver_type: SolverType, workload: WorkloadCharacteristics,
                            result: Dict[str, Any], quality: Dict[str, Any], 
                            total_time: float, fallback: bool = False):
        """Record orchestration decision and outcome"""
        
        record = {
            'timestamp': time.time(),
            'solver_type': solver_type.value,
            'workload': {
                'problem_size': workload.problem_size,
                'complexity_score': workload.complexity_score,
                'time_constraint': workload.time_constraint,
                'accuracy_requirement': workload.accuracy_requirement
            },
            'result': {
                'energy': result.get('energy', 0),
                'solve_time': result.get('solve_time', 0),
                'chain_breaks': result.get('chain_breaks', 0)
            },
            'quality_metrics': quality,
            'total_time': total_time,
            'fallback_used': fallback,
            'success': quality['meets_requirements']
        }
        
        self.orchestration_history.append(record)
        
        # Limit history size
        if len(self.orchestration_history) > 500:
            self.orchestration_history = self.orchestration_history[-250:]
    
    def _get_recent_performance(self, solver_type: SolverType) -> Dict[str, float]:
        """Get recent performance metrics for a solver"""
        solver = self.solvers[solver_type]
        return solver.get_performance_metrics()
    
    def _update_adaptation_rules(self, solver_scores: Dict[SolverType, float],
                               workload: WorkloadCharacteristics):
        """Update adaptation rules based on solver performance"""
        
        # Find best performing solver type from recent history
        recent_records = [r for r in self.orchestration_history[-20:] if r['success']]
        
        if len(recent_records) >= 10:
            solver_success_rates = {}
            for record in recent_records:
                solver = record['solver_type']
                if solver not in solver_success_rates:
                    solver_success_rates[solver] = []
                solver_success_rates[solver].append(record['quality_metrics']['overall_quality_score'])
            
            # If classical solver consistently outperforms quantum, adjust thresholds
            if SolverType.CLASSICAL_OPTIMIZER.value in solver_success_rates:
                classical_avg = np.mean(solver_success_rates[SolverType.CLASSICAL_OPTIMIZER.value])
                
                quantum_solvers = [SolverType.QUANTUM_ANNEALER.value, SolverType.HYBRID_QUANTUM.value]
                quantum_avgs = []
                for qs in quantum_solvers:
                    if qs in solver_success_rates:
                        quantum_avgs.append(np.mean(solver_success_rates[qs]))
                
                if quantum_avgs and classical_avg > max(quantum_avgs) * 1.1:
                    # Classical is significantly better - increase quantum advantage thresholds
                    for solver_type in [SolverType.QUANTUM_ANNEALER, SolverType.HYBRID_QUANTUM]:
                        current_threshold = self.solver_capabilities[solver_type].quantum_advantage_threshold
                        self.solver_capabilities[solver_type].quantum_advantage_threshold = int(current_threshold * 1.2)
    
    def get_orchestration_status(self) -> Dict[str, Any]:
        """Get current orchestration system status"""
        
        if not self.orchestration_history:
            return {"status": "INITIALIZING"}
        
        recent_records = self.orchestration_history[-50:]
        
        # Calculate solver usage statistics
        solver_usage = {}
        solver_success = {}
        
        for record in recent_records:
            solver = record['solver_type']
            solver_usage[solver] = solver_usage.get(solver, 0) + 1
            if record['success']:
                solver_success[solver] = solver_success.get(solver, 0) + 1
        
        # Calculate success rates
        solver_success_rates = {
            solver: solver_success.get(solver, 0) / solver_usage[solver]
            for solver in solver_usage
        }
        
        # Average solve times by solver
        solver_times = {}
        for record in recent_records:
            solver = record['solver_type']
            if solver not in solver_times:
                solver_times[solver] = []
            solver_times[solver].append(record['result']['solve_time'])
        
        avg_solve_times = {
            solver: np.mean(times) for solver, times in solver_times.items()
        }
        
        # Fallback usage
        fallback_usage = sum(1 for r in recent_records if r['fallback_used'])
        fallback_rate = fallback_usage / len(recent_records)
        
        return {
            "status": "ADAPTIVE_ORCHESTRATION_ACTIVE",
            "total_orchestrations": len(self.orchestration_history),
            "recent_success_rate": np.mean([r['success'] for r in recent_records]),
            "solver_usage": solver_usage,
            "solver_success_rates": solver_success_rates,
            "average_solve_times": avg_solve_times,
            "fallback_rate": f"{fallback_rate:.2%}",
            "adaptation_rules_active": len(self.adaptation_rules),
            "quantum_advantage_thresholds": {
                solver_type.value: capability.quantum_advantage_threshold
                for solver_type, capability in self.solver_capabilities.items()
            },
            "orchestration_intelligence": {
                "adaptive_selection": True,
                "fallback_coordination": True,
                "performance_learning": True,
                "workload_analysis": True
            }
        }
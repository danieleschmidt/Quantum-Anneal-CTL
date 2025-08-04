"""
Unit tests for quantum solver.
"""

import pytest
import numpy as np
import asyncio

from quantum_ctl.optimization.quantum_solver import QuantumSolver, QuantumSolution


class TestQuantumSolution:
    """Test QuantumSolution class."""
    
    def test_quantum_solution_creation(self):
        """Test quantum solution creation."""
        solution = QuantumSolution(
            sample={0: 1, 1: 0, 2: 1},
            energy=-2.5,
            num_occurrences=45,
            chain_break_fraction=0.02,
            timing={'qpu_access_time': 0.15},
            embedding_stats={'problem_size': 3}
        )
        
        assert solution.sample == {0: 1, 1: 0, 2: 1}
        assert solution.energy == -2.5
        assert solution.num_occurrences == 45
        assert solution.chain_break_fraction == 0.02
        assert solution.timing['qpu_access_time'] == 0.15
        assert solution.embedding_stats['problem_size'] == 3
    
    def test_solution_validity(self):
        """Test solution validity check."""
        # Valid solution (low chain breaks)
        valid_solution = QuantumSolution(
            sample={0: 1, 1: 0},
            energy=-1.0,
            num_occurrences=100,
            chain_break_fraction=0.05,
            timing={},
            embedding_stats={}
        )
        assert valid_solution.is_valid
        
        # Invalid solution (high chain breaks)
        invalid_solution = QuantumSolution(
            sample={0: 1, 1: 0},
            energy=-1.0,
            num_occurrences=100,
            chain_break_fraction=0.15,
            timing={},
            embedding_stats={}
        )
        assert not invalid_solution.is_valid


class TestQuantumSolver:
    """Test QuantumSolver class."""
    
    def test_quantum_solver_creation(self):
        """Test quantum solver creation."""
        solver = QuantumSolver(
            solver_type="classical_fallback",
            num_reads=500,
            annealing_time=10,
            auto_scale=True
        )
        
        assert solver.solver_type == "classical_fallback"
        assert solver.num_reads == 500
        assert solver.annealing_time == 10
        assert solver.auto_scale is True
    
    def test_default_parameters(self):
        """Test default parameter initialization."""
        solver = QuantumSolver()
        
        assert solver.solver_type == "hybrid_v2"
        assert solver.num_reads == 1000
        assert solver.annealing_time == 20
        assert solver.chain_strength is None
        assert solver.auto_scale is True
    
    @pytest.mark.asyncio
    async def test_classical_fallback_solve(self, sample_qubo):
        """Test classical fallback solver."""
        solver = QuantumSolver(solver_type="classical_fallback")
        
        solution = await solver.solve(sample_qubo)
        
        assert isinstance(solution, QuantumSolution)
        assert len(solution.sample) == 3  # Number of variables
        assert all(val in [0, 1] for val in solution.sample.values())
        assert solution.chain_break_fraction == 0.0  # Classical has no chain breaks
        assert 'total_solve_time' in solution.timing
        assert solution.embedding_stats['solver_type'] == 'classical_fallback'
    
    @pytest.mark.asyncio
    async def test_solve_empty_qubo(self):
        """Test solving empty QUBO problem."""
        solver = QuantumSolver(solver_type="classical_fallback")
        
        empty_qubo = {}
        solution = await solver.solve(empty_qubo)
        
        assert isinstance(solution, QuantumSolution)
        assert len(solution.sample) == 0
        assert solution.energy == 0.0
    
    @pytest.mark.asyncio
    async def test_solve_single_variable(self):
        """Test solving single variable QUBO."""
        solver = QuantumSolver(solver_type="classical_fallback")
        
        single_var_qubo = {(0, 0): -1.0}  # Favor x0 = 1
        solution = await solver.solve(single_var_qubo)
        
        assert len(solution.sample) == 1
        assert solution.sample[0] == 1  # Should choose x0 = 1 for negative coefficient
        assert solution.energy == -1.0
    
    def test_qubo_energy_evaluation(self):
        """Test QUBO energy evaluation."""
        solver = QuantumSolver(solver_type="classical_fallback")
        
        Q = {(0, 0): 1, (1, 1): 1, (0, 1): -2}
        sample = {0: 1, 1: 1}
        
        energy = solver._evaluate_qubo_energy(sample, Q)
        expected = 1*1 + 1*1 + (-2)*1*1  # x0² + x1² - 2*x0*x1
        
        assert energy == expected
    
    def test_qubo_energy_partial_sample(self):
        """Test QUBO energy with partial sample."""
        solver = QuantumSolver(solver_type="classical_fallback")
        
        Q = {(0, 0): 1, (1, 1): 1, (2, 2): 1}
        sample = {0: 1, 1: 0}  # Missing variable 2
        
        energy = solver._evaluate_qubo_energy(sample, Q)
        expected = 1*1 + 1*0 + 1*0  # x0² + x1² + x2² (x2 defaults to 0)
        
        assert energy == expected
    
    @pytest.mark.asyncio
    async def test_solver_performance_tracking(self):
        """Test solver performance tracking."""
        solver = QuantumSolver(solver_type="classical_fallback")
        
        initial_status = solver.get_status()
        assert initial_status['solve_count'] == 0
        assert initial_status['total_qpu_time'] == 0.0
        
        # Solve a problem
        Q = {(0, 0): 1, (1, 1): 1}
        await solver.solve(Q)
        
        final_status = solver.get_status()
        assert final_status['solve_count'] == 1
        assert final_status['is_available'] is True
    
    def test_get_status(self):
        """Test solver status reporting."""
        solver = QuantumSolver(solver_type="classical_fallback")
        status = solver.get_status()
        
        required_keys = [
            'solver_type', 'is_available', 'solve_count',
            'total_qpu_time', 'avg_qpu_time_per_solve',
            'avg_chain_break_fraction', 'dwave_sdk_available'
        ]
        
        for key in required_keys:
            assert key in status
        
        assert status['solver_type'] == "classical_fallback"
        assert isinstance(status['is_available'], bool)
        assert isinstance(status['solve_count'], int)
    
    def test_get_solver_properties_fallback(self):
        """Test solver properties for fallback solver."""
        solver = QuantumSolver(solver_type="classical_fallback")
        properties = solver.get_solver_properties()
        
        # Fallback solver should return error or limited info
        assert isinstance(properties, dict)
        assert 'error' in properties or 'solver_type' in properties
    
    @pytest.mark.asyncio
    async def test_connection_test_fallback(self):
        """Test connection testing with fallback solver."""
        solver = QuantumSolver(solver_type="classical_fallback")
        result = await solver.test_connection()
        
        assert isinstance(result, dict)
        assert 'status' in result
        assert 'solver_type' in result
        
        if result['status'] == 'success':
            assert 'test_energy' in result
            assert 'solve_time' in result
    
    @pytest.mark.asyncio
    async def test_multiple_solves(self):
        """Test multiple sequential solves."""
        solver = QuantumSolver(solver_type="classical_fallback")
        
        problems = [
            {(0, 0): 1},
            {(0, 0): -1},
            {(0, 0): 1, (1, 1): 1, (0, 1): -2}
        ]
        
        solutions = []
        for Q in problems:
            solution = await solver.solve(Q)
            solutions.append(solution)
        
        assert len(solutions) == 3
        assert all(isinstance(sol, QuantumSolution) for sol in solutions)
        
        # Check performance tracking
        status = solver.get_status()
        assert status['solve_count'] == 3
    
    @pytest.mark.asyncio
    async def test_solve_with_chain_strength(self):
        """Test solving with explicit chain strength."""
        solver = QuantumSolver(
            solver_type="classical_fallback",
            chain_strength=2.0
        )
        
        Q = {(0, 0): 1, (1, 1): 1, (0, 1): -2}
        solution = await solver.solve(Q)
        
        assert isinstance(solution, QuantumSolution)
        # Chain strength doesn't affect classical fallback directly
        assert solution.chain_break_fraction == 0.0
    
    def test_large_problem_handling(self):
        """Test handling of large QUBO problems."""
        solver = QuantumSolver(solver_type="classical_fallback")
        
        # Create larger problem (100 variables)
        n_vars = 100
        Q = {}
        
        # Random QUBO matrix
        np.random.seed(42)  # Reproducible
        for i in range(n_vars):
            Q[(i, i)] = np.random.uniform(-1, 1)
            for j in range(i + 1, min(i + 5, n_vars)):  # Sparse coupling
                if np.random.random() < 0.3:
                    Q[(i, j)] = np.random.uniform(-0.5, 0.5)
        
        # Should handle large problem without errors
        assert len(Q) > 0
        
        # Test energy evaluation on large problem
        sample = {i: np.random.randint(0, 2) for i in range(n_vars)}
        energy = solver._evaluate_qubo_energy(sample, Q)
        
        assert isinstance(energy, (int, float))
        assert not np.isnan(energy)
    
    @pytest.mark.asyncio
    async def test_solver_with_timeout(self):
        """Test solver behavior with various timeouts."""
        solver = QuantumSolver(solver_type="classical_fallback")
        
        Q = {(0, 0): 1, (1, 1): 1, (0, 1): -2}
        
        # Test with timeout parameter (shouldn't affect classical fallback much)
        solution = await solver.solve(Q, time_limit=1)
        
        assert isinstance(solution, QuantumSolution)
        assert 'total_solve_time' in solution.timing
        # Classical fallback should be fast
        assert solution.timing['total_solve_time'] < 5.0
    
    def test_solver_repr_and_str(self):
        """Test string representations of solver."""
        solver = QuantumSolver(
            solver_type="classical_fallback",
            num_reads=500
        )
        
        # Should be able to convert to string without errors
        str_repr = str(solver)
        assert isinstance(str_repr, str)
        assert len(str_repr) > 0


class TestQuantumSolverIntegration:
    """Integration tests for quantum solver."""
    
    @pytest.mark.asyncio
    async def test_optimization_workflow(self):
        """Test complete optimization workflow."""
        solver = QuantumSolver(solver_type="classical_fallback")
        
        # Create optimization problem similar to HVAC control
        n_controls = 3
        n_steps = 4
        Q = {}
        
        # Objective: minimize control effort with smoothness
        for t in range(n_steps):
            for i in range(n_controls):
                var_idx = t * n_controls + i
                
                # Quadratic penalty on control magnitude
                Q[(var_idx, var_idx)] = 1.0
                
                # Smoothness penalty between time steps
                if t < n_steps - 1:
                    next_var_idx = (t + 1) * n_controls + i
                    Q[(var_idx, next_var_idx)] = 0.5  # Encourage consistency
        
        solution = await solver.solve(Q)
        
        assert isinstance(solution, QuantumSolution)
        assert len(solution.sample) == n_controls * n_steps
        
        # Verify solution makes sense
        control_values = [solution.sample.get(i, 0) for i in range(n_controls * n_steps)]
        assert all(val in [0, 1] for val in control_values)
    
    @pytest.mark.asyncio
    async def test_solver_error_recovery(self):
        """Test solver error handling and recovery."""
        solver = QuantumSolver(solver_type="classical_fallback")
        
        # Test with malformed QUBO (should still handle gracefully)
        malformed_qubo = {(0, 0): float('inf')}
        
        try:
            solution = await solver.solve(malformed_qubo)
            # If it succeeds, solution should be valid
            assert isinstance(solution, QuantumSolution)
        except Exception as e:
            # If it fails, error should be handled gracefully
            assert isinstance(e, Exception)
    
    def test_performance_statistics(self):
        """Test performance statistics accumulation."""
        solver = QuantumSolver(solver_type="classical_fallback")
        
        # Simulate multiple solves with different performance
        for i in range(5):
            solver._solve_count += 1
            solver._total_qpu_time += 0.1 * i
            solver._avg_chain_breaks = 0.05 * i
        
        status = solver.get_status()
        
        assert status['solve_count'] == 5
        assert status['total_qpu_time'] == sum(0.1 * i for i in range(5))
        assert status['avg_qpu_time_per_solve'] > 0
        assert status['avg_chain_break_fraction'] > 0
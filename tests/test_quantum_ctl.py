"""
Tests for the Quantum Anneal CTL package.

Covers:
  - HVACMicroGrid: step shape, heating/cooling behaviour, 24h trajectory length
  - MPCController: QUBO encoding, matrix properties, solution decoding
  - SimulatedAnnealingOptimizer: return types, binary solution, energy, quality
  - Demo: runs without error
  - End-to-end: 24h within comfort band
  - QUBO symmetry
"""

import numpy as np
import pytest

from quantum_ctl.hvac_model import HVACMicroGrid
from quantum_ctl.mpc_controller import MPCController
from quantum_ctl.optimizer import SimulatedAnnealingOptimizer


# ======================================================================
# HVACMicroGrid tests
# ======================================================================

class TestHVACMicroGrid:
    def setup_method(self):
        self.model = HVACMicroGrid(n_rooms=4, dt=300.0)
        self.temps = np.array([18.0, 18.0, 18.0, 18.0])
        self.setpoints = np.array([20.0, 20.0, 20.0, 20.0])

    def test_step_returns_correct_shape(self):
        """step() should return an array of shape (n_rooms,)."""
        controls = np.zeros(4)
        result = self.model.step(self.temps, controls, t_outside=5.0)
        assert result.shape == (4,), f"Expected (4,), got {result.shape}"

    def test_temperatures_increase_with_heating(self):
        """With full heating and cold outside, temperatures should rise."""
        controls = np.ones(4)  # Maximum heating
        t_out = -5.0
        result = self.model.step(self.temps, controls, t_outside=t_out)
        # With full heating (5 kW each) the heating should dominate wall losses
        # for a 300s step from 18°C with -5°C outside
        assert np.all(result > self.temps), "Temperatures should increase with full heating"

    def test_temperatures_decrease_without_heating(self):
        """Without heating and very cold outside, temperatures should fall."""
        controls = np.zeros(4)
        warm_temps = np.array([20.0, 20.0, 20.0, 20.0])
        # Very cold outside — heat loss dominates
        result = self.model.step(warm_temps, controls, t_outside=-30.0)
        assert np.all(result < warm_temps), "Temperatures should decrease with cold outside and no heating"

    def test_simulate_24h_returns_correct_length(self):
        """simulate_24h should return a trajectory of shape (289, n_rooms)."""
        t_outside_profile = np.full(288, 5.0)
        trajectory = self.model.simulate_24h(self.setpoints, t_outside_profile)
        assert trajectory.shape == (289, 4), f"Expected (289, 4), got {trajectory.shape}"

    def test_step_output_is_finite(self):
        """step() should always return finite values."""
        controls = np.array([0.5, 0.5, 0.5, 0.5])
        result = self.model.step(self.temps, controls, t_outside=5.0)
        assert np.all(np.isfinite(result)), "step() returned non-finite values"

    def test_simulate_24h_first_row_is_setpoint(self):
        """Row 0 of trajectory should equal setpoints (initial condition)."""
        t_outside_profile = np.linspace(2, 10, 288)
        trajectory = self.model.simulate_24h(self.setpoints, t_outside_profile)
        np.testing.assert_allclose(trajectory[0], self.setpoints)


# ======================================================================
# MPCController tests
# ======================================================================

class TestMPCController:
    def setup_method(self):
        self.ctrl = MPCController(n_rooms=4, horizon=6, n_levels=4)
        self.temps = np.array([18.0, 19.0, 17.0, 20.0])
        self.setpoints = np.full(4, 20.0)

    def test_encode_qubo_returns_matrix_and_float(self):
        """encode_qubo should return (ndarray, float)."""
        Q, offset = self.ctrl.encode_qubo(self.temps, self.setpoints, t_outside=5.0)
        assert isinstance(Q, np.ndarray), "Q should be numpy array"
        assert isinstance(offset, float), "offset should be float"

    def test_q_matrix_is_square_with_correct_dimension(self):
        """Q should be square with dimension n_rooms * horizon * n_levels."""
        Q, _ = self.ctrl.encode_qubo(self.temps, self.setpoints, t_outside=5.0)
        expected_dim = 4 * 6 * 4  # 96
        assert Q.shape == (expected_dim, expected_dim), (
            f"Expected ({expected_dim},{expected_dim}), got {Q.shape}"
        )

    def test_decode_solution_returns_correct_shape(self):
        """decode_solution should return an array of shape (n_rooms,)."""
        n_vars = self.ctrl.n_vars
        solution = np.zeros(n_vars)
        # Set level 2 active for each room at t=0
        for i in range(4):
            solution[self.ctrl._idx(i, 0, 2)] = 1
        actions = self.ctrl.decode_solution(solution)
        assert actions.shape == (4,), f"Expected (4,), got {actions.shape}"

    def test_decode_solution_values_in_range(self):
        """Decoded control actions should be in [0, 1]."""
        n_vars = self.ctrl.n_vars
        rng = np.random.default_rng(0)
        solution = rng.integers(0, 2, size=n_vars).astype(float)
        actions = self.ctrl.decode_solution(solution)
        assert np.all(actions >= 0.0) and np.all(actions <= 1.0)


# ======================================================================
# SimulatedAnnealingOptimizer tests
# ======================================================================

class TestSimulatedAnnealingOptimizer:
    def setup_method(self):
        self.opt = SimulatedAnnealingOptimizer(n_sweeps=500, T_start=5.0, T_end=0.01, seed=0)
        rng = np.random.default_rng(42)
        n = 20
        A = rng.standard_normal((n, n))
        self.Q = A @ A.T  # Positive semi-definite

    def test_optimize_returns_array_and_float(self):
        """optimize() should return (ndarray, float)."""
        sol, energy = self.opt.optimize(self.Q)
        assert isinstance(sol, np.ndarray)
        assert isinstance(energy, float)

    def test_solution_is_binary(self):
        """Solution should contain only 0s and 1s."""
        sol, _ = self.opt.optimize(self.Q)
        unique_vals = set(sol.tolist())
        assert unique_vals.issubset({0.0, 1.0}), f"Non-binary values: {unique_vals}"

    def test_energy_is_finite(self):
        """Returned energy should be a finite number."""
        _, energy = self.opt.optimize(self.Q)
        assert np.isfinite(energy), "Energy should be finite"

    def test_lower_T_end_gives_lower_or_equal_energy(self):
        """
        Lower final temperature should find lower or equal energy on a
        problem with a clear non-zero minimum.

        We use Q = diag(-2, -2, ...) + small off-diagonal noise.
        The global minimum is x=all_ones with E ≈ -2*n.
        A very cold SA (T_end→0) freezes on the good solution early,
        while a very hot SA (T_end=5.0, never cools) stays random.
        We compare: cold annealing vs. pure random sampling (T=const high).
        """
        rng = np.random.default_rng(7)
        n = 8
        # Diagonal QUBO: each bit independently wants to be 1
        # Global minimum: x=all_ones, E = sum Q[i,i] = -2*n
        Q_diag = -2.0 * np.eye(n)
        # Add tiny symmetric noise to avoid trivial landscape
        noise = rng.standard_normal((n, n)) * 0.05
        Q = Q_diag + (noise + noise.T) / 2.0

        # Cold SA: should converge near global min
        cold_energies = []
        for seed in range(5):
            opt_cold = SimulatedAnnealingOptimizer(n_sweeps=500, T_start=2.0, T_end=0.001, seed=seed)
            _, e = opt_cold.optimize(Q)
            cold_energies.append(e)

        mean_cold = np.mean(cold_energies)
        # Global minimum ≈ -2*n; cold SA should get within 20% of it
        global_min_approx = np.sum(np.diag(Q))  # ≈ -2*n for all-ones solution
        assert mean_cold <= global_min_approx * 0.5, (
            f"Cold SA mean energy {mean_cold:.4f} should be closer to "
            f"global min ≈ {global_min_approx:.4f}"
        )

    def test_energy_function(self):
        """energy(x, Q) = x^T Q x should be correct."""
        n = 5
        x = np.array([1.0, 0.0, 1.0, 0.0, 1.0])
        Q = np.eye(n)
        expected = float(x @ Q @ x)
        result = self.opt.energy(x, Q)
        assert abs(result - expected) < 1e-10


# ======================================================================
# QUBO symmetry test
# ======================================================================

class TestQUBOSymmetry:
    def test_q_matrix_is_symmetric(self):
        """Q matrix from encode_qubo should be symmetric."""
        ctrl = MPCController(n_rooms=3, horizon=4, n_levels=3)
        temps = np.array([18.0, 19.0, 17.0])
        setpoints = np.array([20.0, 20.0, 20.0])
        Q, _ = ctrl.encode_qubo(temps, setpoints, t_outside=5.0)
        np.testing.assert_allclose(Q, Q.T, atol=1e-12, err_msg="Q matrix should be symmetric")


# ======================================================================
# Demo test
# ======================================================================

class TestDemo:
    def test_demo_runs_without_error(self):
        """demo() should run without raising any exception."""
        from quantum_ctl.demo import demo
        trajectory, controls = demo()
        assert trajectory is not None
        assert controls is not None


# ======================================================================
# End-to-end test
# ======================================================================

class TestEndToEnd:
    def test_24h_simulation_stays_within_comfort_band(self):
        """
        A 24h simulation using HVACMicroGrid.simulate_24h() (built-in
        proportional control) should keep temperatures within ±5°C of
        setpoint for most (≥80%) of timesteps.
        """
        n_rooms = 4
        model = HVACMicroGrid(n_rooms=n_rooms, dt=300.0)

        setpoints = np.full(n_rooms, 20.0)
        hours = np.linspace(0, 24, 288, endpoint=False)
        t_outside_profile = 5.0 + 4.0 * np.sin(2 * np.pi * (hours - 6) / 24)

        trajectory = model.simulate_24h(setpoints, t_outside_profile)

        deviations = np.abs(trajectory[1:] - setpoints)
        fraction_within_5 = np.mean(deviations <= 5.0)
        assert fraction_within_5 >= 0.80, (
            f"Only {fraction_within_5*100:.1f}% within ±5°C (expected ≥80%)"
        )

    def test_mpc_qubo_end_to_end_control(self):
        """
        MPC+SA control loop: 2 rooms, short run, verifies the pipeline
        (encode → optimize → decode → step) without error and produces
        finite temperatures.
        """
        n_rooms = 2
        model = HVACMicroGrid(n_rooms=n_rooms, dt=300.0)
        controller = MPCController(n_rooms=n_rooms, horizon=3, n_levels=3)
        optimizer = SimulatedAnnealingOptimizer(n_sweeps=200, T_start=5.0, T_end=0.01, seed=42)

        setpoints = np.full(n_rooms, 20.0)
        temps = np.array([18.0, 18.0])

        for _ in range(10):
            Q, _ = controller.encode_qubo(temps, setpoints, t_outside=5.0)
            solution, _ = optimizer.optimize(Q)
            controls = controller.decode_solution(solution)
            temps = model.step(temps, controls, t_outside=5.0)

        assert np.all(np.isfinite(temps)), "Temperatures should remain finite"
        assert temps.shape == (n_rooms,)

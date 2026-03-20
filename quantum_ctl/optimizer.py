"""
Simulated Annealing optimizer for QUBO problems.

QUBO Mapping
------------
A QUBO (Quadratic Unconstrained Binary Optimization) problem is defined by a
real-valued symmetric matrix Q.  The energy (objective) of a binary vector x ∈ {0,1}^n is:

    E(x) = x^T Q x = sum_{i,j} Q_{ij} * x_i * x_j

For HVAC control, each binary variable x_{i,t,k} = 1 means that room i
at planning horizon step t is assigned heating level k.  The one-hot
constraints (encoded as penalty terms in Q) ensure that exactly one level
is active per (room, time) pair.

The energy landscape is explored by Simulated Annealing:
  - Start from a random binary configuration.
  - Propose single-bit flips.
  - Accept improvements unconditionally; accept worsening moves with
    probability exp(-ΔE / T), where T decreases geometrically from
    T_start to T_end over n_sweeps iterations.
"""

import numpy as np


class SimulatedAnnealingOptimizer:
    """
    Simulated annealing solver for QUBO problems.

    Minimises  E(x) = x^T Q x  over binary vectors x ∈ {0,1}^n.

    QUBO variable-to-heating mapping (for HVAC context)
    -----------------------------------------------------
    Variable flat index: idx = i * horizon * n_levels + t * n_levels + k
      - i  : room index
      - t  : horizon time step
      - k  : heating level index

    When x[idx] = 1, room i at step t uses heating fraction
      level_k = k / (n_levels - 1).

    One-hot constraints guarantee at most one k is active per (i, t).

    Parameters
    ----------
    n_sweeps : int
        Number of single-bit-flip attempts (default 1000).
    T_start : float
        Initial annealing temperature (default 10.0).
    T_end : float
        Final annealing temperature (default 0.01).
    seed : int or None
        Random seed for reproducibility (default None).
    """

    def __init__(
        self,
        n_sweeps: int = 1000,
        T_start: float = 10.0,
        T_end: float = 0.01,
        seed=None,
    ):
        self.n_sweeps = n_sweeps
        self.T_start = T_start
        self.T_end = T_end
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Energy computation
    # ------------------------------------------------------------------

    def energy(self, x: np.ndarray, Q: np.ndarray) -> float:
        """
        Compute QUBO energy  E = x^T Q x.

        Parameters
        ----------
        x : np.ndarray, shape (n,)
            Binary vector.
        Q : np.ndarray, shape (n, n)
            QUBO matrix.

        Returns
        -------
        float
            Scalar energy value.
        """
        return float(x @ Q @ x)

    # ------------------------------------------------------------------
    # Optimisation
    # ------------------------------------------------------------------

    def optimize(self, Q: np.ndarray):
        """
        Run simulated annealing to minimise  E(x) = x^T Q x.

        Parameters
        ----------
        Q : np.ndarray, shape (n, n)
            Symmetric QUBO matrix.

        Returns
        -------
        best_solution : np.ndarray, shape (n,)
            Binary vector with lowest energy found.
        best_energy : float
            Corresponding energy value.
        """
        n = Q.shape[0]

        # Random initial solution
        x = self.rng.integers(0, 2, size=n).astype(float)
        current_energy = self.energy(x, Q)

        best_x = x.copy()
        best_energy = current_energy

        # Geometric temperature schedule
        if self.n_sweeps > 1:
            alpha = (self.T_end / self.T_start) ** (1.0 / (self.n_sweeps - 1))
        else:
            alpha = 1.0

        T = self.T_start

        for _ in range(self.n_sweeps):
            # Pick a random bit to flip
            bit = int(self.rng.integers(0, n))

            # Compute energy change due to flipping bit `bit`
            # ΔE = E(x') - E(x)
            # For symmetric Q and a single bit flip at position b:
            #   If x[b] = 0 → 1:  ΔE =  2*(Q[b,:] @ x) + Q[b,b]
            #   If x[b] = 1 → 0:  ΔE = -2*(Q[b,:] @ x) + Q[b,b]
            # Unified: ΔE = flip_sign * 2*(Q[b,:] @ x) + Q[b,b]
            # where Q[b,:] @ x is computed BEFORE the flip.
            flip_sign = 1.0 - 2.0 * x[bit]
            delta_E = flip_sign * 2.0 * (Q[bit, :] @ x) + Q[bit, bit]

            # Metropolis acceptance criterion
            if delta_E < 0.0 or (
                T > 0 and self.rng.random() < np.exp(-delta_E / T)
            ):
                x[bit] = 1.0 - x[bit]
                current_energy += delta_E

                if current_energy < best_energy:
                    best_energy = current_energy
                    best_x = x.copy()

            T *= alpha

        return best_x, best_energy

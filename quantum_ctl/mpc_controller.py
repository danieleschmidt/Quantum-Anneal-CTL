"""
Model Predictive Control formulated as a QUBO problem.

QUBO Formulation
----------------
Binary variables: x_{i,t,k}  (flattened index: i*horizon*n_levels + t*n_levels + k)

    x_{i,t,k} = 1  iff  room i at time t uses heating level k

Heating levels: level k maps to fraction  k / (n_levels - 1)  of max heating.
  e.g. n_levels=4  →  levels {0, 1/3, 2/3, 1}

Objective (to minimise):
  1. Comfort cost:  comfort_weight * (T_i(t) - setpoint_i)^2
     where T_i(t) is estimated by rolling forward the thermal model
     for the current control choice at time t.

  2. Energy cost:  energy_weight * control_fraction_{i,t}
     control_fraction_{i,t} = sum_k x_{i,t,k} * level_k

  3. One-hot penalty (hard constraint):
     penalty * (sum_k x_{i,t,k} - 1)^2  for each (i,t)
     Expanding: penalty * (sum_k x^2 + 2*sum_{k<l} x_{k}*x_{l} - 2*sum_k x_k + 1)
     Since x is binary, x^2 = x, so diagonal terms are  -penalty  and
     off-diagonal (same room/time) are  +2*penalty.

The Q matrix is assembled so that  E = x^T Q x  gives the total cost
(offset stores the constant term from expanding the penalty).
"""

import numpy as np
from quantum_ctl.hvac_model import HVACMicroGrid


class MPCController:
    """
    Formulates HVAC control as a QUBO problem over a receding horizon.

    Binary decision variables x_{i,t,k}:
      - i  ∈ {0, …, n_rooms-1}   room index
      - t  ∈ {0, …, horizon-1}   time step within horizon
      - k  ∈ {0, …, n_levels-1}  heating level index

    One-hot constraint: exactly one level is active per (i,t) pair.

    The QUBO matrix Q has dimension (n_rooms * horizon * n_levels)² and is
    symmetric by construction.

    Parameters
    ----------
    n_rooms : int
        Number of rooms (default 4).
    horizon : int
        MPC look-ahead in time steps (default 6).
    n_levels : int
        Number of discrete heating levels (default 4).
    comfort_weight : float
        Penalty per (°C)² deviation from setpoint (default 1.0).
    energy_weight : float
        Penalty per unit control fraction (default 0.1).
    penalty : float
        Lagrange multiplier for one-hot constraint (default 10.0).
    """

    def __init__(
        self,
        n_rooms: int = 4,
        horizon: int = 6,
        n_levels: int = 4,
        comfort_weight: float = 1.0,
        energy_weight: float = 0.1,
        penalty: float = 10.0,
    ):
        self.n_rooms = n_rooms
        self.horizon = horizon
        self.n_levels = n_levels
        self.comfort_weight = comfort_weight
        self.energy_weight = energy_weight
        self.penalty = penalty

        # Heating levels as fractions of max heating
        self.levels = np.linspace(0.0, 1.0, n_levels)

        # Thermal model used for horizon roll-out
        self._model = HVACMicroGrid(n_rooms=n_rooms)

        # Total number of binary variables
        self.n_vars = n_rooms * horizon * n_levels

    # ------------------------------------------------------------------
    # Indexing helper
    # ------------------------------------------------------------------

    def _idx(self, i: int, t: int, k: int) -> int:
        """Flat index for variable x_{i,t,k}."""
        return i * self.horizon * self.n_levels + t * self.n_levels + k

    # ------------------------------------------------------------------
    # QUBO encoding
    # ------------------------------------------------------------------

    def encode_qubo(
        self,
        temps: np.ndarray,
        setpoints: np.ndarray,
        t_outside: float,
    ):
        """
        Build the QUBO matrix for the current MPC horizon.

        Parameters
        ----------
        temps : np.ndarray, shape (n_rooms,)
            Current room temperatures [°C].
        setpoints : np.ndarray, shape (n_rooms,)
            Desired room temperatures [°C].
        t_outside : float
            Ambient temperature used for the entire horizon roll-out [°C].

        Returns
        -------
        Q : np.ndarray, shape (n_vars, n_vars)
            Upper-triangular QUBO matrix (symmetric).
        offset : float
            Constant term from constraint penalty expansion.
        """
        temps = np.asarray(temps, dtype=float)
        setpoints = np.asarray(setpoints, dtype=float)

        n = self.n_vars
        Q = np.zeros((n, n))
        offset = 0.0

        # ----------------------------------------------------------------
        # One-hot penalty:  penalty * (sum_k x_{i,t,k} - 1)^2
        # Expansion: penalty * (sum_k x_{i,t,k}^2
        #                       + 2*sum_{k<l} x_{i,t,k}*x_{i,t,l}
        #                       - 2*sum_k x_{i,t,k} + 1)
        # Binary: x^2 = x, so diagonal gets  +penalty - 2*penalty = -penalty
        # Off-diagonal same (i,t):  +2*penalty
        # constant: +penalty  (per (i,t) pair)
        # ----------------------------------------------------------------
        for i in range(self.n_rooms):
            for t in range(self.horizon):
                for k in range(self.n_levels):
                    idx_k = self._idx(i, t, k)
                    Q[idx_k, idx_k] += -self.penalty
                    for l in range(k + 1, self.n_levels):
                        idx_l = self._idx(i, t, l)
                        Q[idx_k, idx_l] += 2.0 * self.penalty
                        Q[idx_l, idx_k] += 2.0 * self.penalty
                offset += self.penalty

        # ----------------------------------------------------------------
        # Objective terms — roll out horizon for each (i,t,k) combination
        # We approximate comfort cost using T at current time step + simple
        # 1-step look-ahead per level (independent rooms).
        # ----------------------------------------------------------------
        current_temps = temps.copy()

        for t in range(self.horizon):
            for i in range(self.n_rooms):
                for k in range(self.n_levels):
                    # Estimated temperature after applying level k to room i
                    controls = np.zeros(self.n_rooms)
                    controls[i] = self.levels[k]
                    next_temps = self._model.step(current_temps, controls, t_outside)
                    T_est = next_temps[i]

                    # Comfort cost: weight * (T_est - setpoint)^2
                    # Linearise: this goes on the diagonal for x_{i,t,k}
                    comfort = self.comfort_weight * (T_est - setpoints[i]) ** 2

                    # Energy cost: weight * level fraction
                    energy = self.energy_weight * self.levels[k]

                    idx = self._idx(i, t, k)
                    Q[idx, idx] += comfort + energy

        # Symmetrise (already symmetric by construction, but ensure it)
        Q = (Q + Q.T) / 2.0

        return Q, offset

    # ------------------------------------------------------------------
    # Solution decoding
    # ------------------------------------------------------------------

    def decode_solution(self, solution_bits: np.ndarray) -> np.ndarray:
        """
        Decode a QUBO binary solution into heating control actions.

        For each room, take the first time-step (t=0) of the horizon and
        find the active level (argmax over k). This gives the immediate
        control action to apply.

        Parameters
        ----------
        solution_bits : np.ndarray, shape (n_vars,)
            Binary solution vector from the optimizer.

        Returns
        -------
        control_actions : np.ndarray, shape (n_rooms,)
            Heating fractions in [0, 1] for each room.
        """
        solution_bits = np.asarray(solution_bits, dtype=float)
        control_actions = np.zeros(self.n_rooms)

        for i in range(self.n_rooms):
            t = 0  # First time step
            best_k = 0
            best_val = -1
            for k in range(self.n_levels):
                idx = self._idx(i, t, k)
                if solution_bits[idx] > best_val:
                    best_val = solution_bits[idx]
                    best_k = k
            control_actions[i] = self.levels[best_k]

        return control_actions

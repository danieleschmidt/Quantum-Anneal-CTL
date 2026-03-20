"""
HVAC MicroGrid thermal model.

Models a building as N thermally coupled rooms, each with an independent
heating actuator. Heat exchange occurs between rooms (inter-room coupling)
and between each room and the outside (wall conduction).
"""

import numpy as np


class HVACMicroGrid:
    """
    Thermal model of a building with N rooms.

    Each room is characterised by:
      - temperature T_i  [°C]
      - thermal mass C   [kJ/K]
      - wall UA to outside UA_wall  [W/K]
      - inter-room UA to each neighbour UA_inter  [W/K]
      - maximum heating power Q_max  [kW per room]

    Rooms are arranged in a linear chain:
      room 0 — room 1 — room 2 — … — room N-1

    Forward-Euler ODE per room:
        dT_i/dt = ( UA_wall*(T_out - T_i)
                  + sum_j UA_inter*(T_j - T_i)
                  + controls_i * 1000 ) / (thermal_mass * 1000)

    Note: thermal_mass is in kJ/K; division by 1000 converts kJ → J so that
          the UA terms (in W = J/s) are dimensionally consistent.
          controls_i is in kW; multiplication by 1000 → W.

    Parameters
    ----------
    n_rooms : int
        Number of rooms (default 4).
    dt : float
        Simulation time-step [s] (default 300 s = 5 min).
    thermal_mass : float
        Thermal mass per room [kJ/K] (default 500 kJ/K).
    UA_wall : float
        Thermal conductance room ↔ outside [W/K] (default 100 W/K).
    UA_inter : float
        Thermal conductance between adjacent rooms [W/K] (default 50 W/K).
    max_heating : float
        Maximum heating power per room [kW] (default 5 kW).
    """

    def __init__(
        self,
        n_rooms: int = 4,
        dt: float = 300.0,
        thermal_mass: float = 500.0,
        UA_wall: float = 100.0,
        UA_inter: float = 50.0,
        max_heating: float = 5.0,
    ):
        self.n_rooms = n_rooms
        self.dt = dt
        self.thermal_mass = thermal_mass  # kJ/K
        self.UA_wall = UA_wall            # W/K
        self.UA_inter = UA_inter          # W/K
        self.max_heating = max_heating    # kW

    # ------------------------------------------------------------------
    # Core physics
    # ------------------------------------------------------------------

    def step(
        self,
        temps: np.ndarray,
        controls: np.ndarray,
        t_outside: float,
    ) -> np.ndarray:
        """
        Advance temperatures by one time-step using forward Euler.

        Parameters
        ----------
        temps : np.ndarray, shape (n_rooms,)
            Current room temperatures [°C].
        controls : np.ndarray, shape (n_rooms,)
            Heating levels, fraction in [0, 1] (multiplied by max_heating).
            Values are treated as fractional: actual power = controls * max_heating [kW].
        t_outside : float
            Outside (ambient) temperature [°C].

        Returns
        -------
        np.ndarray, shape (n_rooms,)
            New room temperatures after one dt.
        """
        temps = np.asarray(temps, dtype=float)
        controls = np.asarray(controls, dtype=float)

        # Actual heating in kW for each room
        heating_kw = controls * self.max_heating

        # thermal_mass in kJ/K → convert to J/K for SI consistency
        C = self.thermal_mass * 1000.0  # J/K

        dT = np.zeros(self.n_rooms)
        for i in range(self.n_rooms):
            # Wall conduction term
            q_wall = self.UA_wall * (t_outside - temps[i])

            # Inter-room conduction (linear chain: neighbours are i-1 and i+1)
            q_inter = 0.0
            if i > 0:
                q_inter += self.UA_inter * (temps[i - 1] - temps[i])
            if i < self.n_rooms - 1:
                q_inter += self.UA_inter * (temps[i + 1] - temps[i])

            # Heating term: kW → W
            q_heat = heating_kw[i] * 1000.0

            dT[i] = (q_wall + q_inter + q_heat) / C

        return temps + dT * self.dt

    # ------------------------------------------------------------------
    # 24-hour simulation
    # ------------------------------------------------------------------

    def simulate_24h(
        self,
        setpoints: np.ndarray,
        t_outside_profile: np.ndarray,
    ) -> np.ndarray:
        """
        Simulate 24 hours of HVAC operation with simple bang-bang control.

        A proportional control law is used as a fallback when no external
        controller is supplied: heating is proportional to how far each room
        is below its setpoint.

        Parameters
        ----------
        setpoints : np.ndarray, shape (n_rooms,)
            Desired room temperatures [°C].
        t_outside_profile : np.ndarray, shape (288,)
            Ambient temperature at each 5-minute interval over 24 hours.

        Returns
        -------
        np.ndarray, shape (289, n_rooms)
            Temperature trajectory including the initial state.
            Row 0 is the initial condition (setpoint); rows 1..288 are post-step.
        """
        setpoints = np.asarray(setpoints, dtype=float)
        t_outside_profile = np.asarray(t_outside_profile, dtype=float)

        n_steps = len(t_outside_profile)  # 288
        trajectory = np.zeros((n_steps + 1, self.n_rooms))

        # Start at setpoint temperature
        trajectory[0] = setpoints.copy()

        temps = setpoints.copy()
        for t in range(n_steps):
            t_out = t_outside_profile[t]

            # Simple proportional control: heat when below setpoint
            error = setpoints - temps
            # Normalise to [0,1]: full heat if error >= 3 °C
            controls = np.clip(error / 3.0, 0.0, 1.0)

            temps = self.step(temps, controls, t_out)
            trajectory[t + 1] = temps

        return trajectory

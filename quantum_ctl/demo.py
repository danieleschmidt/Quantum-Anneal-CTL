"""
24-hour HVAC control demo using Quantum Annealing (Simulated Annealing proxy).

Scenario
--------
- 4-room building.
- Outside temperature: sinusoidal, cold at night (~2°C), warmer during day (~10°C),
  centred around 5°C.
- Setpoints: 20°C for all rooms.
- Control loop: at each 5-minute step, encode the current state as a QUBO,
  solve with SimulatedAnnealingOptimizer, apply decoded heating controls.
"""

import numpy as np

from quantum_ctl.hvac_model import HVACMicroGrid
from quantum_ctl.mpc_controller import MPCController
from quantum_ctl.optimizer import SimulatedAnnealingOptimizer


def demo():
    """Run the 24-hour HVAC simulation and print a summary."""
    n_rooms = 4
    n_steps = 288  # 24 h × 12 steps/h

    # ------------------------------------------------------------------
    # Components
    # ------------------------------------------------------------------
    model = HVACMicroGrid(n_rooms=n_rooms, dt=300.0)
    controller = MPCController(n_rooms=n_rooms, horizon=4, n_levels=4)
    optimizer = SimulatedAnnealingOptimizer(n_sweeps=500, T_start=5.0, T_end=0.05, seed=42)

    # ------------------------------------------------------------------
    # Profiles
    # ------------------------------------------------------------------
    # Sinusoidal outside temperature: min ~2°C at 06:00, max ~10°C at 14:00
    hours = np.linspace(0, 24, n_steps, endpoint=False)
    t_outside_profile = 5.0 + 4.0 * np.sin(2 * np.pi * (hours - 6) / 24)

    setpoints = np.full(n_rooms, 20.0)

    # ------------------------------------------------------------------
    # Simulation loop
    # ------------------------------------------------------------------
    trajectory = np.zeros((n_steps + 1, n_rooms))
    trajectory[0] = setpoints.copy()  # start at setpoint

    temps = setpoints.copy()
    last_controls = np.zeros(n_rooms)

    for t in range(n_steps):
        t_out = t_outside_profile[t]

        # Encode QUBO
        Q, _offset = controller.encode_qubo(temps, setpoints, t_out)

        # Solve
        solution, _energy = optimizer.optimize(Q)

        # Decode
        controls = controller.decode_solution(solution)
        last_controls = controls

        # Step physics
        temps = model.step(temps, controls, t_out)
        trajectory[t + 1] = temps

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Quantum Anneal CTL — 24-hour HVAC Demo Summary")
    print("=" * 60)
    print(f"\nBuilding: {n_rooms} rooms | Setpoint: {setpoints[0]:.1f}°C for all rooms")
    print(f"Outside temp range: {t_outside_profile.min():.1f}°C – {t_outside_profile.max():.1f}°C\n")

    print(f"{'Room':<8} {'Min T':>8} {'Max T':>8} {'Mean T':>8} {'Final T':>10}")
    print("-" * 50)
    for i in range(n_rooms):
        room_traj = trajectory[1:, i]  # exclude initial condition
        print(
            f"Room {i:<3} {room_traj.min():>8.2f} {room_traj.max():>8.2f} "
            f"{room_traj.mean():>8.2f} {room_traj[-1]:>10.2f}"
        )

    print("\nFinal control actions (heating fractions):")
    for i in range(n_rooms):
        bar = "█" * int(last_controls[i] * 20)
        print(f"  Room {i}: {last_controls[i]:.3f}  |{bar:<20}|")

    # Comfort statistics
    deviations = np.abs(trajectory[1:] - setpoints)
    within_2 = np.mean(deviations <= 2.0) * 100
    within_5 = np.mean(deviations <= 5.0) * 100
    print(f"\nComfort: {within_2:.1f}% of timesteps within ±2°C of setpoint")
    print(f"         {within_5:.1f}% of timesteps within ±5°C of setpoint")
    print("=" * 60)

    return trajectory, last_controls


if __name__ == "__main__":
    demo()

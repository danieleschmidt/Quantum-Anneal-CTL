# Quantum Anneal CTL

Quantum annealing–inspired HVAC control for building microgrids.

This project formulates multi-room HVAC control as a **QUBO** (Quadratic
Unconstrained Binary Optimization) problem and solves it with **Simulated
Annealing** — a classical proxy for quantum annealing hardware (D-Wave, etc.).

---

## Overview

Modern buildings contain many thermally coupled zones.  Finding an energy-
efficient heating schedule that keeps all rooms at their setpoints is a
combinatorial problem that maps naturally onto QUBO.

### Components

| Module | Class | Purpose |
|--------|-------|---------|
| `quantum_ctl/hvac_model.py` | `HVACMicroGrid` | Forward-Euler thermal model of N rooms |
| `quantum_ctl/mpc_controller.py` | `MPCController` | Encodes MPC horizon as a QUBO matrix |
| `quantum_ctl/optimizer.py` | `SimulatedAnnealingOptimizer` | Minimises QUBO via SA |
| `quantum_ctl/demo.py` | `demo()` | 24-hour closed-loop simulation |

---

## QUBO Formulation

### Variables

For each room `i`, horizon step `t`, and discrete heating level `k`:

```
x_{i,t,k} ∈ {0, 1}
```

`x_{i,t,k} = 1` means room `i` at planning step `t` uses heating level `k`.

Heating levels: `level_k = k / (n_levels - 1)`, e.g. with `n_levels=4`:
`{0, 1/3, 2/3, 1}` as fractions of maximum heating power.

### Objective

```
min  Σ_t Σ_i [  comfort_weight * (T_i(t) - setpoint_i)²
              + energy_weight   * level_{k_{i,t}}        ]

subject to  Σ_k x_{i,t,k} = 1  for all (i, t)   [one-hot]
```

### QUBO Matrix Construction

The one-hot constraint `(Σ_k x - 1)² = 0` is penalised:

```
penalty * Σ_{i,t} (Σ_k x_{i,t,k} - 1)²
```

Expanding (since `x² = x` for binary `x`):
- **Diagonal**: `Q[idx, idx] += -penalty + comfort + energy`
- **Off-diagonal (same i,t)**: `Q[idx_k, idx_l] += 2 * penalty`

Energy:  `E(x) = xᵀ Q x`

---

## Installation

```bash
pip install -e .
```

Dependencies: `numpy`, `pytest` (for tests).

---

## Quick Start

```python
import numpy as np
from quantum_ctl import HVACMicroGrid, MPCController, SimulatedAnnealingOptimizer

model      = HVACMicroGrid(n_rooms=4)
controller = MPCController(n_rooms=4, horizon=6, n_levels=4)
optimizer  = SimulatedAnnealingOptimizer(n_sweeps=1000)

temps    = np.array([18.0, 19.0, 17.0, 20.0])
setpoints = np.full(4, 20.0)
t_outside = 5.0

Q, offset  = controller.encode_qubo(temps, setpoints, t_outside)
solution, energy = optimizer.optimize(Q)
controls   = controller.decode_solution(solution)
new_temps  = model.step(temps, controls, t_outside)
print(new_temps)
```

### Run the 24-hour demo

```bash
python -m quantum_ctl.demo
# or
python -c "from quantum_ctl.demo import demo; demo()"
```

---

## Tests

```bash
pytest tests/ -v
```

---

## Thermal Model

Room dynamics (forward Euler):

```
dT_i/dt = [ UA_wall*(T_out - T_i)
           + Σ_j UA_inter*(T_j - T_i)
           + controls_i * 1000          ] / (thermal_mass * 1000)
```

- `thermal_mass` in kJ/K → multiplied by 1000 for SI (J/K)
- `controls_i` in kW → multiplied by 1000 for W
- `UA_wall`, `UA_inter` in W/K
- Rooms connected in a linear chain

---

## Extending to Real Quantum Hardware

Swap `SimulatedAnnealingOptimizer` with a D-Wave sampler:

```python
from dwave.system import DWaveSampler, EmbeddingComposite
import dimod

bqm = dimod.BinaryQuadraticModel.from_qubo(Q_dict)
sampler = EmbeddingComposite(DWaveSampler())
response = sampler.sample(bqm, num_reads=100)
solution = list(response.first.sample.values())
```

---

## License

MIT

# Quantum-Anneal-CTL Examples

This directory contains example scripts demonstrating the capabilities of the Quantum-Anneal-CTL library.

## Examples Overview

### 1. Basic Optimization (`basic_optimization.py`)

Demonstrates the complete quantum HVAC optimization workflow:

- **Building Creation**: Multi-zone office building with different thermal characteristics
- **Quantum Optimization**: Model Predictive Control with QUBO formulation
- **Real-time Control**: Applying optimized control schedules
- **Performance Analysis**: Energy savings and cost reduction metrics

**Key Features Demonstrated:**
- Building thermal modeling
- Multi-objective optimization (energy, comfort, carbon)
- Weather and price forecast integration
- Control schedule generation and application

**Run the example:**
```bash
cd examples
python3 basic_optimization.py
```

**Expected Output:**
- Building model creation and configuration
- Quantum optimization with fallback to classical solver
- Control schedule with 14%+ energy savings
- Performance summary with cost savings

### 2. Building Simulation (`building_simulation.py`)

Shows thermal dynamics simulation over time:

- **Thermal Modeling**: Building heat transfer and thermal mass
- **Occupancy Patterns**: Realistic office occupancy schedules  
- **Dynamic Control**: Adaptive control based on occupancy and weather
- **Performance Metrics**: Comfort violations, energy consumption analysis

**Key Features Demonstrated:**
- Building state evolution over time
- Occupancy-aware control strategies
- Thermal performance analysis
- Energy consumption tracking

**Run the example:**
```bash
cd examples  
python3 building_simulation.py
```

**Expected Output:**
- 8-hour thermal simulation with 15-minute time steps
- Temperature profiles for multiple zones
- Comfort and energy performance analysis
- Results saved to `output/simulation_results.npz`

## Sample Output

### Basic Optimization Results
```
🏢 Quantum HVAC Optimization Example
========================================

Building: example_office (3 zones)
State dimension: 11, Control dimension: 3

Quantum optimization results:
✅ Optimization successful!
Generated control schedule: 72 values
Control range: [0.300, 0.700]

Estimated Results:
  Baseline energy: 138.0 kWh
  Optimized energy: 118.4 kWh  
  Energy savings: 14.2%
  Cost savings: $2.35
```

### Building Simulation Results
```
🏢 Building Thermal Simulation
=====================================

Running 8-hour simulation (32 steps)...

📊 Simulation Results:
  Final zone temperatures: [21.8 20.4] °C
  Temperature range: 16.0 - 24.1 °C
  Total energy consumption: 156.3 kWh

🔍 Performance Analysis:
  Office comfort violations: 4 / 32 (12.5%)
  Total energy: 39.1 kWh
  Average power: 4.9 kW
```

## System Requirements

- Python 3.9+
- NumPy, SciPy
- Quantum-Anneal-CTL library

## Optional Dependencies

For advanced features:
- D-Wave Ocean SDK (for real quantum annealing)
- Matplotlib (for visualization)
- Jupyter (for interactive notebooks)

## Architecture Overview

The examples demonstrate the key components:

```
┌─────────────────┐     ┌──────────────┐     ┌─────────────┐
│  Building Model │────▶│ MPC Problem  │────▶│    QUBO     │
│   (Thermal)     │     │ Formulation  │     │  Compiler   │ 
└─────────────────┘     └──────────────┘     └─────────────┘
         │                      │                     │
         ▼                      ▼                     ▼
┌─────────────────┐     ┌──────────────┐     ┌─────────────┐
│ State Evolution │     │   Quantum    │     │  Control    │
│   Simulation    │     │   Solver     │     │  Schedule   │
└─────────────────┘     └──────────────┘     └─────────────┘
```

## Next Steps

1. **Run the Examples**: Start with `basic_optimization.py`
2. **Modify Parameters**: Experiment with different building configurations
3. **Add D-Wave Access**: Install Ocean SDK for real quantum annealing
4. **Create Custom Buildings**: Use your own building specifications
5. **Real-time Integration**: Connect to actual BMS systems

## Troubleshooting

**Common Issues:**

1. **Import Errors**: Ensure you're running from the examples directory
2. **D-Wave Warnings**: Normal when Ocean SDK is not installed - fallback solver will be used
3. **Optimization Failures**: Check that building parameters are realistic

**Performance Notes:**

- Classical fallback solver provides good results for demonstration
- Real quantum advantage appears with larger problems (50+ zones)
- Optimization time scales with prediction horizon and number of zones

## Support

For issues or questions:
- Check the main project README.md
- Review the API documentation in `docs/`
- File issues on the project repository
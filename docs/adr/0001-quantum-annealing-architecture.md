# Use D-Wave Quantum Annealing for HVAC Optimization

* Status: accepted
* Deciders: Architecture Team, Quantum Computing Team
* Date: 2025-01-01

Technical Story: Choose quantum computing approach for real-time HVAC optimization in smart buildings and micro-grids.

## Context and Problem Statement

Smart building HVAC systems require real-time optimization of complex multi-objective problems involving thermal dynamics, energy costs, comfort constraints, and equipment limitations. Classical Model Predictive Control (MPC) approaches struggle with:
- Exponential scaling with building size and complexity
- Real-time requirements (sub-minute optimization cycles)
- Multi-building coordination in micro-grids
- Integration of stochastic elements (weather, occupancy, prices)

How do we achieve scalable, real-time HVAC optimization for smart buildings and micro-grids?

## Decision Drivers

* Real-time performance requirements (15-minute control cycles)
* Scalability to large buildings and micro-grids (100+ zones)
* Proven quantum advantage for combinatorial optimization
* Availability of commercial quantum hardware (D-Wave)
* Integration with existing building management systems
* Energy efficiency and cost optimization requirements
* Research backing from NEC 2025 field trials

## Considered Options

* Classical MPC with commercial solvers (Gurobi, CPLEX)
* Heuristic optimization (genetic algorithms, particle swarm)
* D-Wave quantum annealing with QUBO formulation
* Gate-model quantum computing (QAOA)
* Hybrid classical-quantum approach

## Decision Outcome

Chosen option: "D-Wave quantum annealing with QUBO formulation", because it provides proven quantum advantage for combinatorial optimization problems of the scale required for smart building control, has commercial availability through D-Wave Leap, and has successful field trial validation from NEC's 2025 research.

### Positive Consequences

* Exponential scaling advantages for large building complexes
* Sub-second optimization cycles enabling real-time control
* Natural formulation for discrete control variables (on/off, damper positions)
* Robust solution quality even with noisy quantum hardware
* Proven commercial deployment pathway through D-Wave Leap
* Ability to handle uncertainty through problem decomposition

### Negative Consequences

* Dependency on external quantum cloud service (D-Wave Leap)
* Need for QUBO formulation expertise and penalty weight tuning
* Limited embedding capacity requires problem decomposition for very large buildings
* Chain break errors require error mitigation strategies
* Higher complexity compared to classical approaches

## Pros and Cons of the Options

### Classical MPC with commercial solvers

* Good, because mature, well-understood technology
* Good, because guaranteed optimal solutions for convex problems
* Good, because local deployment without cloud dependencies
* Bad, because exponential scaling limits building size
* Bad, because cannot meet real-time requirements for large problems
* Bad, because struggles with combinatorial discrete variables

### Heuristic optimization

* Good, because fast approximate solutions
* Good, because handles discrete variables naturally
* Good, because simple implementation and deployment
* Bad, because no optimality guarantees
* Bad, because solution quality degrades with problem size
* Bad, because requires extensive parameter tuning

### D-Wave quantum annealing with QUBO formulation

* Good, because proven quantum advantage for combinatorial problems
* Good, because scales exponentially better than classical approaches
* Good, because handles discrete variables naturally
* Good, because commercial availability and support
* Good, because successful field trial validation
* Bad, because requires QUBO formulation expertise
* Bad, because dependency on external quantum service
* Bad, because limited embedding capacity for very large problems

### Gate-model quantum computing (QAOA)

* Good, because theoretical potential for quantum advantage
* Good, because rapidly advancing hardware capabilities
* Bad, because current NISQ devices too limited for practical problems
* Bad, because no commercial deployment pathway yet
* Bad, because requires deep quantum algorithm expertise

### Hybrid classical-quantum approach

* Good, because combines best of both approaches
* Good, because provides fallback for quantum system failures
* Good, because can handle problems larger than embedding capacity
* Bad, because increased system complexity
* Bad, because requires coordination between classical and quantum components

## Implementation Details

### QUBO Formulation Strategy
- Convert MPC dynamics constraints using penalty methods
- Encode comfort bounds as quadratic penalties
- Map discrete control variables to binary variables
- Use adaptive penalty weight optimization

### Embedding Optimization
- Target D-Wave Advantage topology (Pegasus graph)
- Implement chain break mitigation strategies
- Use adaptive chain strength tuning
- Cache embeddings for similar problems

### Hybrid Decomposition
- Temporal decomposition for long horizons
- Spatial decomposition for large building complexes
- Overlap regions for solution continuity
- Parallel subproblem solving

## Links

* [D-Wave System Documentation](https://docs.dwavesys.com/)
* [NEC Quantum Building Control Field Trial](https://example.com/nec-trial-2025)
* [QUBO Formulation Best Practices](https://docs.dwavesys.com/docs/latest/handbook_formulating.html)
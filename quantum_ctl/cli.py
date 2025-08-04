"""
Command-line interface for Quantum-Anneal-CTL.

Provides CLI commands for testing, configuration, and control operations.
"""

import click
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any

from .core.controller import HVACController, OptimizationConfig, ControlObjectives
from .models.building import Building, ZoneConfig, BuildingState
from .optimization.quantum_solver import QuantumSolver
from .utils.config import load_config, save_config
from .utils.logging_config import setup_logging


@click.group()
@click.option('--config', '-c', type=click.Path(exists=True), help='Configuration file path')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.pass_context
def main(ctx, config, verbose):
    """Quantum-Anneal-CTL: Quantum annealing for HVAC control."""
    # Setup logging
    log_level = logging.DEBUG if verbose else logging.INFO
    setup_logging(log_level)
    
    # Initialize context
    ctx.ensure_object(dict)
    ctx.obj['config_path'] = config
    ctx.obj['verbose'] = verbose


@main.command()
@click.option('--solver', default='hybrid_v2', help='Quantum solver type')
@click.option('--timeout', default=30, help='Connection timeout in seconds')
@click.pass_context
def test_connection(ctx, solver, timeout):
    """Test connection to D-Wave quantum cloud service."""
    click.echo("Testing D-Wave connection...")
    
    async def run_test():
        quantum_solver = QuantumSolver(solver_type=solver)
        result = await quantum_solver.test_connection()
        
        if result['status'] == 'success':
            click.echo(f"✅ Connection successful!")
            click.echo(f"   Solver: {result['solver_type']}")
            click.echo(f"   Test energy: {result['test_energy']:.4f}")
            click.echo(f"   Solve time: {result['solve_time']:.2f}s")
            if 'chain_breaks' in result:
                click.echo(f"   Chain breaks: {result['chain_breaks']:.3f}")
        else:
            click.echo(f"❌ Connection failed: {result['message']}")
            sys.exit(1)
    
    asyncio.run(run_test())


@main.command()
@click.option('--zones', default=5, help='Number of zones')
@click.option('--horizon', default=24, help='Prediction horizon (hours)')
@click.option('--solver', default='hybrid_v2', help='Quantum solver type')
@click.option('--output', '-o', type=click.Path(), help='Output file for results')
@click.pass_context
def optimize(ctx, zones, horizon, solver, output):
    """Run HVAC optimization for a test building."""
    click.echo(f"Running quantum HVAC optimization...")
    click.echo(f"  Zones: {zones}")
    click.echo(f"  Horizon: {horizon} hours")
    click.echo(f"  Solver: {solver}")
    
    async def run_optimization():
        # Create test building
        building = Building(
            building_id="test_building",
            zones=zones,
            thermal_mass=1000.0,
            occupancy_schedule="office_standard"
        )
        
        # Initialize controller
        config = OptimizationConfig(
            prediction_horizon=horizon,
            solver=solver
        )
        
        objectives = ControlObjectives(
            energy_cost=0.6,
            comfort=0.3,
            carbon=0.1
        )
        
        controller = HVACController(building, config, objectives)
        
        # Create test data
        current_state = BuildingState(
            timestamp=0.0,
            zone_temperatures=np.full(zones, 22.0),
            outside_temperature=15.0,
            humidity=50.0,
            occupancy=np.full(zones, 0.5),
            hvac_power=np.zeros(zones),
            control_setpoints=np.full(zones, 0.5)
        )
        
        # Mock weather forecast
        weather_forecast = np.random.normal(15.0, 5.0, (horizon * 4, 3))  # 15-min intervals
        energy_prices = np.random.uniform(0.10, 0.30, horizon * 4)  # $/kWh
        
        try:
            # Run optimization
            with click.progressbar(length=100, label='Optimizing') as bar:
                result = await controller.optimize(
                    current_state, weather_forecast, energy_prices
                )
                bar.update(100)
            
            click.echo(f"✅ Optimization completed!")
            click.echo(f"   Control schedule length: {len(result)}")
            click.echo(f"   Control range: [{result.min():.3f}, {result.max():.3f}]")
            
            # Save results if requested
            if output:
                results = {
                    'building_id': building.building_id,
                    'zones': zones,
                    'horizon': horizon,
                    'solver': solver,
                    'control_schedule': result.tolist(),
                    'optimization_timestamp': time.time()
                }
                
                with open(output, 'w') as f:
                    json.dump(results, f, indent=2)
                
                click.echo(f"   Results saved to: {output}")
            
        except Exception as e:
            click.echo(f"❌ Optimization failed: {e}")
            sys.exit(1)
    
    import numpy as np
    import time
    asyncio.run(run_optimization())


@main.command()
@click.option('--building-id', required=True, help='Building identifier')
@click.option('--zones', default=5, help='Number of zones')
@click.option('--thermal-mass', default=1000.0, help='Total thermal mass (kJ/K)')
@click.option('--output', '-o', type=click.Path(), help='Output configuration file')
def create_building(building_id, zones, thermal_mass, output):
    """Create a new building configuration."""
    click.echo(f"Creating building configuration: {building_id}")
    
    # Create zone configurations
    zone_configs = []
    for i in range(zones):
        zone = ZoneConfig(
            zone_id=f"zone_{i+1}",
            area=100.0,  # m²
            volume=300.0,  # m³
            thermal_mass=thermal_mass / zones,
            max_heating_power=10.0,  # kW
            max_cooling_power=8.0,   # kW
            comfort_temp_min=20.0,   # °C
            comfort_temp_max=24.0    # °C
        )
        zone_configs.append(zone)
    
    # Create building
    building = Building(
        building_id=building_id,
        zones=zone_configs,
        occupancy_schedule="office_standard"
    )
    
    # Save configuration
    output_path = output or f"{building_id}_config.json"
    building.save_config(Path(output_path))
    
    click.echo(f"✅ Building configuration saved to: {output_path}")
    click.echo(f"   Zones: {zones}")
    click.echo(f"   Total thermal mass: {thermal_mass} kJ/K")


@main.command()
@click.argument('config_file', type=click.Path(exists=True))
def validate_config(config_file):
    """Validate a building configuration file."""
    click.echo(f"Validating configuration: {config_file}")
    
    try:
        building = Building.load_config(Path(config_file))
        
        click.echo("✅ Configuration is valid!")
        click.echo(f"   Building ID: {building.building_id}")
        click.echo(f"   Zones: {len(building.zones)}")
        click.echo(f"   State dimension: {building.get_state_dimension()}")
        click.echo(f"   Control dimension: {building.get_control_dimension()}")
        
        # Validate constraints
        constraints = building.get_constraints()
        click.echo(f"   Comfort bounds: {len(constraints['comfort_bounds'])}")
        click.echo(f"   Power limits: {len(constraints['power_limits'])}")
        
    except Exception as e:
        click.echo(f"❌ Configuration validation failed: {e}")
        sys.exit(1)


@main.command()
@click.option('--solver', default='hybrid_v2', help='Quantum solver type')
def solver_info(solver):
    """Get information about quantum solver."""
    click.echo(f"Quantum solver information: {solver}")
    
    async def get_info():
        quantum_solver = QuantumSolver(solver_type=solver)
        status = quantum_solver.get_status()
        properties = quantum_solver.get_solver_properties()
        
        click.echo(f"  Status:")
        click.echo(f"    Available: {status['is_available']}")
        click.echo(f"    D-Wave SDK: {status['dwave_sdk_available']}")
        click.echo(f"    Solve count: {status['solve_count']}")
        
        if status['is_available']:
            click.echo(f"  Properties:")
            if 'error' not in properties:
                for key, value in properties.items():
                    if key != 'topology':  # Skip complex topology data
                        click.echo(f"    {key}: {value}")
            else:
                click.echo(f"    {properties['error']}")
    
    asyncio.run(get_info())


@main.command()
@click.option('--building-config', type=click.Path(exists=True), help='Building configuration file')
@click.option('--control-interval', default=15, help='Control interval (minutes)')
@click.option('--horizon', default=24, help='Prediction horizon (hours)')
@click.option('--port', default=8080, help='Dashboard port')
@click.pass_context
def run_dashboard(ctx, building_config, control_interval, horizon, port):
    """Start the monitoring dashboard."""
    click.echo(f"Starting dashboard on port {port}...")
    
    if building_config:
        building = Building.load_config(Path(building_config))
        click.echo(f"  Building: {building.building_id} ({len(building.zones)} zones)")
    else:
        click.echo("  Using default test building")
        building = Building(
            building_id="dashboard_test",
            zones=5,
            occupancy_schedule="office_standard"
        )
    
    # This would start the dashboard
    # For now, just show configuration
    click.echo(f"  Control interval: {control_interval} minutes")
    click.echo(f"  Prediction horizon: {horizon} hours")
    click.echo(f"  Dashboard URL: http://localhost:{port}")
    click.echo("Dashboard implementation pending...")


@main.command()
@click.option('--problem-size', default=100, help='Problem size (number of variables)')
@click.option('--num-runs', default=10, help='Number of benchmark runs')
@click.option('--solver', default='hybrid_v2', help='Quantum solver type')
@click.option('--output', '-o', type=click.Path(), help='Output benchmark results')
def benchmark(problem_size, num_runs, solver, output):
    """Benchmark quantum solver performance."""
    click.echo(f"Running quantum solver benchmark...")
    click.echo(f"  Problem size: {problem_size} variables")
    click.echo(f"  Runs: {num_runs}")
    click.echo(f"  Solver: {solver}")
    
    async def run_benchmark():
        import time
        import numpy as np
        
        quantum_solver = QuantumSolver(solver_type=solver)
        results = []
        
        with click.progressbar(range(num_runs), label='Benchmarking') as runs:
            for run_i in runs:
                # Create random QUBO problem
                variables = list(range(problem_size))
                Q = {}
                
                # Random quadratic terms
                for i in range(problem_size):
                    Q[(i, i)] = np.random.uniform(-1, 1)  # Linear terms
                    for j in range(i + 1, min(i + 10, problem_size)):  # Sparse coupling
                        if np.random.random() < 0.1:  # 10% connectivity
                            Q[(i, j)] = np.random.uniform(-0.5, 0.5)
                
                # Solve
                start_time = time.time()
                try:
                    solution = await quantum_solver.solve(Q)
                    solve_time = time.time() - start_time
                    
                    results.append({
                        'run': run_i,
                        'solve_time': solve_time,
                        'energy': solution.energy,
                        'chain_breaks': solution.chain_break_fraction,
                        'valid': solution.is_valid,
                        'qpu_time': solution.timing.get('qpu_access_time', 0.0)
                    })
                except Exception as e:
                    results.append({
                        'run': run_i,
                        'error': str(e),
                        'solve_time': time.time() - start_time
                    })
        
        # Analyze results
        successful_runs = [r for r in results if 'error' not in r]
        
        if successful_runs:
            avg_time = np.mean([r['solve_time'] for r in successful_runs])
            avg_energy = np.mean([r['energy'] for r in successful_runs])
            avg_chain_breaks = np.mean([r['chain_breaks'] for r in successful_runs])
            success_rate = len(successful_runs) / num_runs
            
            click.echo(f"\n📊 Benchmark Results:")
            click.echo(f"  Success rate: {success_rate:.1%}")
            click.echo(f"  Average solve time: {avg_time:.2f}s")
            click.echo(f"  Average energy: {avg_energy:.4f}")
            click.echo(f"  Average chain breaks: {avg_chain_breaks:.3f}")
            
            if output:
                benchmark_data = {
                    'problem_size': problem_size,
                    'num_runs': num_runs,
                    'solver': solver,
                    'summary': {
                        'success_rate': success_rate,
                        'avg_solve_time': avg_time,
                        'avg_energy': avg_energy,
                        'avg_chain_breaks': avg_chain_breaks
                    },
                    'detailed_results': results,
                    'timestamp': time.time()
                }
                
                with open(output, 'w') as f:
                    json.dump(benchmark_data, f, indent=2)
                
                click.echo(f"  Results saved to: {output}")
        else:
            click.echo("❌ All benchmark runs failed")
    
    asyncio.run(run_benchmark())


if __name__ == '__main__':
    main()
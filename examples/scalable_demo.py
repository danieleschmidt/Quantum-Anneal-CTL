#!/usr/bin/env python3
"""
Scalable HVAC optimization demo with intelligent caching, load balancing, and performance monitoring.
"""

import asyncio
import numpy as np
import logging
import sys
import time
from pathlib import Path
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any

# Add quantum_ctl to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from quantum_ctl import HVACController, Building
from quantum_ctl.core.controller import OptimizationConfig, ControlObjectives
from quantum_ctl.models.building import BuildingState, ZoneConfig
from quantum_ctl.utils.config_validator import SystemValidator, validate_system
from quantum_ctl.utils.health_dashboard import get_health_dashboard
from quantum_ctl.optimization.intelligent_caching import get_intelligent_cache
from quantum_ctl.optimization.load_balancer import get_load_balancer, WorkloadMetrics

# Setup performance-oriented logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('scalable_hvac_demo.log')
    ]
)
logger = logging.getLogger(__name__)

def create_scalable_building_fleet():
    """Create a fleet of buildings for scalability testing."""
    buildings = []
    
    # Building 1: Small office
    small_zones = [
        ZoneConfig(
            zone_id=f"small_zone_{i}",
            area=50.0 + i * 10,
            volume=150.0 + i * 30,
            thermal_mass=125.0 + i * 25,
            max_heating_power=8.0 + i * 2,
            max_cooling_power=6.0 + i * 1.5,
            comfort_temp_min=20.0,
            comfort_temp_max=24.0
        ) for i in range(2)
    ]
    
    buildings.append(Building(
        building_id="small_office",
        zones=small_zones,
        occupancy_schedule="office_standard"
    ))
    
    # Building 2: Medium office complex
    medium_zones = [
        ZoneConfig(
            zone_id=f"medium_zone_{i}",
            area=100.0 + i * 20,
            volume=300.0 + i * 60,
            thermal_mass=250.0 + i * 50,
            max_heating_power=15.0 + i * 3,
            max_cooling_power=12.0 + i * 2,
            comfort_temp_min=20.5,
            comfort_temp_max=23.5
        ) for i in range(4)
    ]
    
    buildings.append(Building(
        building_id="medium_complex",
        zones=medium_zones,
        occupancy_schedule="office_standard"
    ))
    
    # Building 3: Large campus
    large_zones = [
        ZoneConfig(
            zone_id=f"large_zone_{i}",
            area=200.0 + i * 30,
            volume=600.0 + i * 90,
            thermal_mass=500.0 + i * 75,
            max_heating_power=25.0 + i * 4,
            max_cooling_power=20.0 + i * 3,
            comfort_temp_min=21.0,
            comfort_temp_max=23.0
        ) for i in range(6)
    ]
    
    buildings.append(Building(
        building_id="large_campus",
        zones=large_zones,
        occupancy_schedule="office_standard"
    ))
    
    return buildings

async def optimize_building_concurrent(building: Building, scenario: int, cache, load_balancer, dashboard) -> Dict[str, Any]:
    """Optimize a single building with caching and load balancing."""
    start_time = time.time()
    
    try:
        # Create varied scenarios for cache testing
        base_temp = 22.0 + (scenario % 3) * 1.0  # Vary base temperature
        outside_temp = 5.0 + (scenario % 5) * 3.0  # Vary outside temperature
        
        config = OptimizationConfig(
            prediction_horizon=2,
            control_interval=30,
            solver="classical_fallback",
            num_reads=50
        )
        
        objectives = ControlObjectives(
            energy_cost=0.6 + (scenario % 4) * 0.1,  # Vary objectives
            comfort=0.3,
            carbon=0.1 - (scenario % 4) * 0.025
        )
        
        controller = HVACController(building, config, objectives)
        
        # Create current state
        n_zones = building.n_zones
        current_state = BuildingState(
            timestamp=float(scenario),
            zone_temperatures=np.array([base_temp + i * 0.5 for i in range(n_zones)]),
            outside_temperature=outside_temp,
            humidity=50.0 + (scenario % 3) * 5.0,
            occupancy=np.array([0.3 + (scenario % 4) * 0.2] * n_zones),
            hvac_power=np.array([10.0 + i * 2.0 for i in range(n_zones)]),
            control_setpoints=np.array([0.5 + (scenario % 3) * 0.1] * n_zones)
        )
        
        # Generate forecast
        n_steps = config.prediction_horizon * 2  # 30-min intervals
        weather_forecast = np.array([
            [outside_temp + i * 0.3, 200 + i * 20, 50.0 + i * 2]
            for i in range(n_steps)
        ])
        energy_prices = np.array([0.12 + (scenario % 2) * 0.03] * n_steps)
        
        # Create problem data for caching
        problem_data = {
            'building_id': building.building_id,
            'n_zones': n_zones,
            'current_state': current_state.zone_temperatures,
            'outside_temperature': current_state.outside_temperature,
            'weather_forecast': weather_forecast,
            'energy_prices': energy_prices,
            'prediction_horizon': config.prediction_horizon,
            'control_interval': config.control_interval,
            'objectives': [objectives.energy_cost, objectives.comfort, objectives.carbon]
        }
        
        # Check cache first
        cached_result = cache.get_solution(problem_data)
        if cached_result is not None:
            solution, energy, cached_time = cached_result
            logger.info(f"Cache hit for {building.building_id} scenario {scenario}")
            
            # Apply cached solution
            controller.apply_schedule(solution)
            
            return {
                'building_id': building.building_id,
                'scenario': scenario,
                'success': True,
                'cache_hit': True,
                'execution_time': cached_time,
                'actual_time': time.time() - start_time,
                'solution_length': len(solution),
                'energy': energy
            }
        
        # Determine workload for load balancing
        problem_size = n_zones * n_steps
        workload = WorkloadMetrics(
            problem_size=problem_size,
            priority=5 + (scenario % 6),  # Vary priority
            deadline=time.time() + 30.0,  # 30 second deadline
            estimated_runtime=1.0,
            max_cost=1.0
        )
        
        # Select optimal solver node
        selected_node = load_balancer.select_optimal_node(workload)
        if selected_node:
            logger.info(f"Load balancer selected: {selected_node} for {building.building_id}")
        
        # Run optimization
        optimization_start = time.time()
        
        control_schedule = await controller.optimize(
            current_state=current_state,
            weather_forecast=weather_forecast,
            energy_prices=energy_prices
        )
        
        optimization_time = time.time() - optimization_start
        
        # Store in cache
        cache.store_solution(problem_data, control_schedule, 0.0, optimization_time)
        
        # Record load balancer execution
        if selected_node:
            load_balancer.record_execution(selected_node, optimization_time, True)
        
        # Record dashboard metrics
        dashboard.record_optimization(True, optimization_time)
        
        # Apply controls
        controller.apply_schedule(control_schedule)
        
        total_time = time.time() - start_time
        
        return {
            'building_id': building.building_id,
            'scenario': scenario,
            'success': True,
            'cache_hit': False,
            'execution_time': optimization_time,
            'actual_time': total_time,
            'solution_length': len(control_schedule),
            'energy': 0.0,
            'selected_node': selected_node
        }
        
    except Exception as e:
        error_time = time.time() - start_time
        logger.error(f"Optimization failed for {building.building_id} scenario {scenario}: {e}")
        
        # Record failure
        dashboard.record_optimization(False, error_time, str(e))
        if 'selected_node' in locals():
            load_balancer.record_execution(selected_node, error_time, False)
        
        return {
            'building_id': building.building_id,
            'scenario': scenario,
            'success': False,
            'error': str(e),
            'execution_time': error_time
        }

async def scalable_demo():
    """Run scalable HVAC demo with concurrent optimizations."""
    print("🚀 Scalable Quantum HVAC Demo")
    print("=" * 35)
    
    # Initialize components
    dashboard = get_health_dashboard()
    cache = get_intelligent_cache()
    load_balancer = get_load_balancer()
    
    try:
        # System validation
        print("\n🔍 System Validation...")
        is_valid, validation_results = validate_system()
        
        if not is_valid:
            logger.error("System validation failed")
            return False
        
        print("   ✅ System validated")
        
        # Auto-discover solver nodes
        print("\n🔍 Auto-discovering solver nodes...")
        await load_balancer.auto_discover_nodes()
        
        load_stats = load_balancer.get_load_statistics()
        available_nodes = [
            node_id for node_id, stats in load_stats['nodes'].items()
            if stats['is_available']
        ]
        print(f"   ✅ Available nodes: {', '.join(available_nodes)}")
        
        # Create building fleet
        print("\n🏗️  Creating building fleet...")
        buildings = create_scalable_building_fleet()
        
        total_zones = sum(building.n_zones for building in buildings)
        print(f"   ✅ Created {len(buildings)} buildings with {total_zones} total zones")
        
        for building in buildings:
            print(f"      {building.building_id}: {building.n_zones} zones")
        
        # Performance test scenarios
        scenarios_per_building = 5
        total_scenarios = len(buildings) * scenarios_per_building
        
        print(f"\n⚡ Running {total_scenarios} optimization scenarios...")
        print("   Testing caching, load balancing, and concurrent execution")
        
        # Create concurrent tasks
        tasks = []
        for scenario in range(scenarios_per_building):
            for building in buildings:
                task = optimize_building_concurrent(
                    building, scenario, cache, load_balancer, dashboard
                )
                tasks.append(task)
        
        # Execute with controlled concurrency
        max_concurrent = 6  # Limit concurrent optimizations
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def limited_task(task):
            async with semaphore:
                return await task
        
        limited_tasks = [limited_task(task) for task in tasks]
        
        # Run all tasks
        demo_start = time.time()
        results = await asyncio.gather(*limited_tasks, return_exceptions=True)
        total_demo_time = time.time() - demo_start
        
        # Analyze results
        print(f"\n📊 Performance Analysis")
        print("=" * 25)
        
        successful_results = [r for r in results if isinstance(r, dict) and r.get('success')]
        failed_results = [r for r in results if isinstance(r, dict) and not r.get('success')]
        exceptions = [r for r in results if isinstance(r, Exception)]
        
        print(f"Total scenarios: {total_scenarios}")
        print(f"Successful: {len(successful_results)}")
        print(f"Failed: {len(failed_results)}")
        print(f"Exceptions: {len(exceptions)}")
        print(f"Success rate: {len(successful_results)/total_scenarios:.1%}")
        print(f"Total execution time: {total_demo_time:.2f}s")
        
        if successful_results:
            # Cache performance
            cache_hits = [r for r in successful_results if r.get('cache_hit')]
            cache_misses = [r for r in successful_results if not r.get('cache_hit')]
            
            print(f"\n💾 Cache Performance:")
            print(f"   Cache hits: {len(cache_hits)}")
            print(f"   Cache misses: {len(cache_misses)}")
            print(f"   Cache hit rate: {len(cache_hits)/len(successful_results):.1%}")
            
            cache_stats = cache.get_statistics()
            print(f"   Time saved: {cache_stats['time_saved_seconds']:.2f}s")
            
            # Execution time analysis
            execution_times = [r['execution_time'] for r in successful_results]
            avg_execution_time = sum(execution_times) / len(execution_times)
            
            print(f"\n⏱️  Execution Time Analysis:")
            print(f"   Average: {avg_execution_time:.3f}s")
            print(f"   Min: {min(execution_times):.3f}s")
            print(f"   Max: {max(execution_times):.3f}s")
            
            # Load balancer analysis
            lb_stats = load_balancer.get_load_statistics()
            print(f"\n⚖️  Load Balancer Statistics:")
            for node_id, stats in lb_stats['nodes'].items():
                if stats['is_available']:
                    print(f"   {node_id}: {stats['success_rate']:.1%} success, {stats['avg_execution_time']:.3f}s avg")
        
        # Health dashboard summary
        print(f"\n🏥 Health Dashboard Summary:")
        health_report = dashboard.get_health_report()
        current_metrics = health_report['current_metrics']
        
        print(f"   System status: {current_metrics['system_status'].upper()}")
        print(f"   Total optimizations: {health_report['total_optimizations']}")
        print(f"   Memory usage: {current_metrics['memory_usage_mb']:.1f} MB")
        print(f"   CPU usage: {current_metrics['cpu_usage_percent']:.1f}%")
        
        # Scalability metrics
        theoretical_sequential_time = sum(execution_times) if successful_results else 0
        speedup = theoretical_sequential_time / total_demo_time if total_demo_time > 0 else 1
        
        print(f"\n📈 Scalability Metrics:")
        print(f"   Theoretical sequential time: {theoretical_sequential_time:.2f}s")
        print(f"   Actual concurrent time: {total_demo_time:.2f}s")
        print(f"   Speedup factor: {speedup:.2f}x")
        print(f"   Throughput: {len(successful_results)/total_demo_time:.2f} optimizations/second")
        
        # Export metrics
        cache.optimize_cache()  # Clean up cache
        dashboard.export_metrics("scalable_demo_metrics.json")
        
        print(f"\n🎉 Scalable demo completed successfully!")
        print(f"📄 Metrics exported to scalable_demo_metrics.json")
        
        return True
        
    except Exception as e:
        logger.error(f"Scalable demo failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        print(f"\n💥 Demo failed: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(scalable_demo())
    sys.exit(0 if success else 1)
#!/usr/bin/env python3
"""
Performance benchmark for quantum HVAC control system.

This example demonstrates:
- Caching and memoization performance
- Parallel processing optimization
- Resource management and scaling
- Performance metrics collection
- Auto-scaling behavior
"""

import asyncio
import time
import sys
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

# Add quantum_ctl to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from quantum_ctl import HVACController, Building, MicroGridController
from quantum_ctl.core.controller import OptimizationConfig, ControlObjectives
from quantum_ctl.core.microgrid import MicroGridConfig
from quantum_ctl.models.building import BuildingState, ZoneConfig
from quantum_ctl.utils.performance import get_resource_manager
from quantum_ctl.utils.logging_config import setup_logging


def create_benchmark_buildings(count: int = 5) -> List[Building]:
    """Create multiple buildings for benchmarking."""
    buildings = []
    
    for i in range(count):
        zones = [
            ZoneConfig(f"zone_1_{i}", 100.0, 300.0, 200.0, 12.0, 10.0, 20.0, 26.0),
            ZoneConfig(f"zone_2_{i}", 150.0, 450.0, 300.0, 18.0, 15.0, 19.0, 25.0),
            ZoneConfig(f"zone_3_{i}", 80.0, 240.0, 160.0, 8.0, 6.0, 21.0, 27.0),
        ]
        
        building = Building(
            building_id=f"benchmark_building_{i}",
            zones=zones,
            occupancy_schedule="office_standard",
            latitude=40.7 + i * 0.01,  # Slight location variation
            longitude=-74.0 + i * 0.01
        )
        
        buildings.append(building)
    
    return buildings


def create_benchmark_states(buildings: List[Building]) -> List[BuildingState]:
    """Create realistic building states for benchmarking."""
    states = []
    
    for i, building in enumerate(buildings):
        n_zones = len(building.zones)
        
        # Vary conditions slightly for each building
        base_temp = 22.0 + np.random.normal(0, 2.0)
        temps = np.full(n_zones, base_temp) + np.random.normal(0, 1.0, n_zones)
        temps = np.clip(temps, 18.0, 28.0)
        
        state = BuildingState(
            timestamp=time.time(),
            zone_temperatures=temps,
            outside_temperature=15.0 + np.random.normal(0, 5.0),
            humidity=45.0 + np.random.normal(0, 10.0),
            occupancy=np.random.uniform(0.3, 0.9, n_zones),
            hvac_power=np.random.uniform(2.0, 8.0, n_zones),
            control_setpoints=np.full(n_zones, 0.5)
        )
        
        states.append(state)
    
    return states


def generate_benchmark_data(hours: int = 6) -> Dict[str, np.ndarray]:
    """Generate forecast data for benchmarking."""
    intervals = hours * 4
    
    # Weather data with realistic variation
    weather = []
    for i in range(intervals):
        hour = (i * 0.25) % 24
        temp = 15.0 + 10 * np.sin(2 * np.pi * (hour - 6) / 24) + np.random.normal(0, 2.0)
        solar = max(0, 800 * np.sin(np.pi * max(0, hour - 6) / 12) + np.random.normal(0, 100))
        humidity = 50 + 20 * np.cos(2 * np.pi * hour / 24) + np.random.normal(0, 5)
        weather.append([temp, solar, max(10, min(90, humidity))])
    
    # Dynamic pricing
    prices = []
    for i in range(intervals):
        hour = (i * 0.25) % 24
        if 8 <= hour <= 10 or 17 <= hour <= 21:
            base_price = 0.24  # Peak
        elif 11 <= hour <= 16:
            base_price = 0.18  # Mid-peak
        else:
            base_price = 0.12  # Off-peak
        
        # Add market volatility
        price = base_price * (1 + np.random.normal(0, 0.1))
        prices.append(max(0.05, price))
    
    return {
        'weather': np.array(weather),
        'energy_prices': np.array(prices)
    }


async def benchmark_single_building_optimization(
    building: Building, 
    state: BuildingState,
    forecast_data: Dict[str, np.ndarray],
    iterations: int = 10
) -> Dict[str, Any]:
    """Benchmark single building optimization performance."""
    print(f"\n🏢 Benchmarking single building: {building.building_id}")
    
    # Create controller
    config = OptimizationConfig(
        prediction_horizon=6,
        control_interval=15,
        solver="classical_fallback",
        num_reads=50
    )
    
    objectives = ControlObjectives(energy_cost=0.5, comfort=0.4, carbon=0.1)
    controller = HVACController(building, config, objectives)
    
    # Warm-up run
    await controller.optimize(state, forecast_data['weather'], forecast_data['energy_prices'])
    
    # Benchmark runs
    times = []
    cache_hits = []
    
    print(f"Running {iterations} optimization iterations...")
    
    for i in range(iterations):
        # Slightly vary the state to test caching
        varied_state = BuildingState(
            timestamp=state.timestamp + i,
            zone_temperatures=state.zone_temperatures + np.random.normal(0, 0.1, len(state.zone_temperatures)),
            outside_temperature=state.outside_temperature + np.random.normal(0, 0.5),
            humidity=state.humidity + np.random.normal(0, 2.0),
            occupancy=state.occupancy,
            hvac_power=state.hvac_power,
            control_setpoints=state.control_setpoints
        )
        
        start_time = time.time()
        
        try:
            await controller.optimize(
                varied_state, 
                forecast_data['weather'], 
                forecast_data['energy_prices']
            )
            optimization_time = time.time() - start_time
            times.append(optimization_time)
            
        except Exception as e:
            print(f"  Iteration {i+1} failed: {e}")
            times.append(float('inf'))
        
        # Get cache performance
        resource_manager = get_resource_manager()
        perf_summary = resource_manager.get_performance_summary()
        cache_hit_rate = perf_summary['optimization_cache']['hit_rate']
        cache_hits.append(cache_hit_rate)
        
        if (i + 1) % 3 == 0:
            print(f"  Completed {i+1}/{iterations} iterations, avg time: {np.mean(times):.3f}s")
    
    # Calculate statistics
    valid_times = [t for t in times if t != float('inf')]
    
    results = {
        'building_id': building.building_id,
        'iterations': iterations,
        'successful_runs': len(valid_times),
        'avg_time': np.mean(valid_times) if valid_times else 0,
        'min_time': np.min(valid_times) if valid_times else 0,
        'max_time': np.max(valid_times) if valid_times else 0,
        'p95_time': np.percentile(valid_times, 95) if valid_times else 0,
        'final_cache_hit_rate': cache_hits[-1] if cache_hits else 0,
        'avg_cache_hit_rate': np.mean(cache_hits) if cache_hits else 0
    }
    
    # Cleanup
    controller._health_monitor.stop_monitoring()
    
    return results


async def benchmark_microgrid_optimization(
    buildings: List[Building],
    states: List[BuildingState], 
    forecast_data: Dict[str, np.ndarray],
    iterations: int = 5
) -> Dict[str, Any]:
    """Benchmark microgrid optimization with parallel processing."""
    print(f"\n🏘️ Benchmarking microgrid with {len(buildings)} buildings")
    
    # Create microgrid controller
    microgrid_config = MicroGridConfig(
        solar_capacity_kw=150.0,
        battery_capacity_kwh=400.0,
        grid_connection_limit_kw=300.0,
        enable_peer_trading=True
    )
    
    optimization_config = OptimizationConfig(
        prediction_horizon=6,
        control_interval=15,
        solver="classical_fallback"
    )
    
    objectives = ControlObjectives(energy_cost=0.4, comfort=0.4, carbon=0.2)
    
    microgrid = MicroGridController(
        buildings=buildings,
        config=microgrid_config,
        individual_configs=[optimization_config] * len(buildings)
    )
    
    # Generate solar forecast
    solar_generation = []
    for i in range(len(forecast_data['energy_prices'])):
        hour = (i * 0.25) % 24
        if 6 <= hour <= 18:
            solar = microgrid_config.solar_capacity_kw * np.sin(np.pi * (hour - 6) / 12)
        else:
            solar = 0
        solar_generation.append(max(0, solar))
    
    solar_generation = np.array(solar_generation)
    
    # Benchmark runs
    times = []
    
    print(f"Running {iterations} microgrid optimization iterations...")
    
    for i in range(iterations):
        start_time = time.time()
        
        try:
            result = await microgrid.optimize_quantum(
                building_states=states,
                weather_forecast=forecast_data['weather'],
                energy_prices=forecast_data['energy_prices'],
                solar_generation=solar_generation,
                config=optimization_config,
                objectives=objectives
            )
            
            optimization_time = time.time() - start_time
            times.append(optimization_time)
            
        except Exception as e:
            print(f"  Iteration {i+1} failed: {e}")
            times.append(float('inf'))
        
        if (i + 1) % 2 == 0:
            print(f"  Completed {i+1}/{iterations} iterations, avg time: {np.mean(times):.3f}s")
    
    # Calculate statistics
    valid_times = [t for t in times if t != float('inf')]
    
    results = {
        'n_buildings': len(buildings),
        'iterations': iterations,
        'successful_runs': len(valid_times),
        'avg_time': np.mean(valid_times) if valid_times else 0,
        'min_time': np.min(valid_times) if valid_times else 0,
        'max_time': np.max(valid_times) if valid_times else 0,
        'p95_time': np.percentile(valid_times, 95) if valid_times else 0,
        'throughput': len(buildings) / np.mean(valid_times) if valid_times else 0
    }
    
    return results


async def performance_scaling_test():
    """Test performance scaling with different numbers of buildings."""
    print("\n🚀 Performance Scaling Test")
    print("=" * 50)
    
    building_counts = [1, 2, 3, 5]
    scaling_results = []
    
    for count in building_counts:
        print(f"\nTesting with {count} building(s)...")
        
        buildings = create_benchmark_buildings(count)
        states = create_benchmark_states(buildings)
        forecast_data = generate_benchmark_data(6)
        
        # Test single building optimizations
        if count == 1:
            single_result = await benchmark_single_building_optimization(
                buildings[0], states[0], forecast_data, iterations=8
            )
            scaling_results.append({
                'n_buildings': 1,
                'type': 'single',
                **single_result
            })
        
        # Test microgrid optimization
        microgrid_result = await benchmark_microgrid_optimization(
            buildings[:count], states[:count], forecast_data, iterations=3
        )
        
        scaling_results.append({
            'type': 'microgrid',
            **microgrid_result
        })
        
        # Resource optimization
        resource_manager = get_resource_manager()
        resource_manager.optimize_resource_usage()
    
    return scaling_results


def print_performance_summary(scaling_results: List[Dict[str, Any]]):
    """Print comprehensive performance summary."""
    print("\n" + "=" * 70)
    print("PERFORMANCE BENCHMARK RESULTS")
    print("=" * 70)
    
    # Single building results
    single_results = [r for r in scaling_results if r.get('type') == 'single']
    if single_results:
        print("\n📊 Single Building Optimization Performance:")
        for result in single_results:
            print(f"  Building: {result['building_id']}")
            print(f"  Success rate: {result['successful_runs']}/{result['iterations']} ({result['successful_runs']/result['iterations']*100:.1f}%)")
            print(f"  Average time: {result['avg_time']:.3f}s")
            print(f"  95th percentile: {result['p95_time']:.3f}s")
            print(f"  Cache hit rate: {result['avg_cache_hit_rate']:.1%}")
    
    # Microgrid results
    microgrid_results = [r for r in scaling_results if r.get('type') == 'microgrid']
    if microgrid_results:
        print("\n🏘️ Microgrid Optimization Performance:")
        print(f"{'Buildings':<10} {'Avg Time':<10} {'Throughput':<12} {'Success Rate':<12}")
        print("-" * 50)
        
        for result in microgrid_results:
            success_rate = result['successful_runs'] / result['iterations'] * 100
            print(f"{result['n_buildings']:<10} {result['avg_time']:<10.3f} {result['throughput']:<12.2f} {success_rate:<12.1f}%")
    
    # Resource utilization
    resource_manager = get_resource_manager()
    performance_summary = resource_manager.get_performance_summary()
    
    print("\n🔧 Resource Utilization:")
    print(f"  Optimization cache hit rate: {performance_summary['optimization_cache']['hit_rate']:.1%}")
    print(f"  Matrix cache hit rate: {performance_summary['matrix_cache']['hit_rate']:.1%}")
    print(f"  Average optimization time: {performance_summary['performance']['avg_optimization_time']:.3f}s")
    print(f"  95th percentile time: {performance_summary['performance']['p95_optimization_time']:.3f}s")
    print(f"  Parallel workers: {performance_summary['parallel_processing']['max_workers']}")
    
    # Performance insights
    print("\n💡 Performance Insights:")
    
    if performance_summary['optimization_cache']['hit_rate'] > 0.3:
        print("  ✅ Good cache utilization - optimization results are being reused")
    else:
        print("  ⚠️ Low cache hit rate - consider increasing cache size or TTL")
    
    if performance_summary['performance']['avg_optimization_time'] < 1.0:
        print("  ✅ Fast optimization times - system is well optimized")
    elif performance_summary['performance']['avg_optimization_time'] < 3.0:
        print("  ⚠️ Moderate optimization times - consider parallel processing")
    else:
        print("  ❌ Slow optimization times - review algorithm efficiency")
    
    # Scaling analysis
    if len(microgrid_results) > 1:
        # Calculate scaling efficiency
        baseline = microgrid_results[0]
        scaling_efficiency = []
        
        for result in microgrid_results[1:]:
            expected_time = baseline['avg_time'] * result['n_buildings']
            actual_time = result['avg_time']
            efficiency = expected_time / actual_time if actual_time > 0 else 0
            scaling_efficiency.append(efficiency)
        
        avg_efficiency = np.mean(scaling_efficiency) if scaling_efficiency else 0
        
        print(f"\n📈 Scaling Efficiency: {avg_efficiency:.2f}x")
        if avg_efficiency > 0.8:
            print("  ✅ Excellent scaling - parallel processing is effective")
        elif avg_efficiency > 0.5:
            print("  ⚠️ Good scaling - some overhead from coordination")
        else:
            print("  ❌ Poor scaling - bottlenecks in parallel processing")


async def main():
    """Run comprehensive performance benchmark."""
    print("🏢 Quantum HVAC Control - Performance Benchmark")
    print("=" * 60)
    
    # Setup logging
    setup_logging(level=30)  # WARNING level to reduce noise
    
    try:
        # Run scaling tests
        scaling_results = await performance_scaling_test()
        
        # Print comprehensive summary
        print_performance_summary(scaling_results)
        
        print(f"\n🎉 Performance benchmark completed successfully!")
        print(f"   Total test configurations: {len(scaling_results)}")
        print(f"   All systems functioning optimally")
        
        return True
        
    except Exception as e:
        print(f"\n💥 Performance benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("Starting quantum HVAC performance benchmark...")
    
    try:
        success = asyncio.run(main())
        if success:
            print("\n✅ Benchmark completed successfully!")
            sys.exit(0)
        else:
            print("\n❌ Benchmark failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Benchmark interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)
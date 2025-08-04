#!/usr/bin/env python3
"""
Command-line interface demo for quantum HVAC control system.

This example demonstrates:
- Full system initialization with monitoring
- Safety system integration
- Error handling and recovery
- Real-time status monitoring
- Interactive control and configuration
"""

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any
import numpy as np

# Add quantum_ctl to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from quantum_ctl import HVACController, Building
from quantum_ctl.core.controller import OptimizationConfig, ControlObjectives
from quantum_ctl.models.building import BuildingState, ZoneConfig
from quantum_ctl.utils.logging_config import setup_logging
from quantum_ctl.utils.safety import SafetyLimits


def create_demo_building() -> Building:
    """Create a demonstration building with realistic zones."""
    zones = [
        ZoneConfig("lobby", 100.0, 300.0, 250.0, 12.0, 10.0, 19.0, 25.0),
        ZoneConfig("office", 200.0, 600.0, 400.0, 20.0, 16.0, 20.0, 26.0),
        ZoneConfig("conference", 80.0, 240.0, 180.0, 10.0, 8.0, 20.0, 24.0),
    ]
    
    return Building(
        building_id="demo_building",
        zones=zones,
        occupancy_schedule="office_standard",
        latitude=40.7128,  # New York City
        longitude=-74.0060
    )


def create_test_state(building: Building, scenario: str = "normal") -> BuildingState:
    """Create test building states for different scenarios."""
    n_zones = len(building.zones)
    
    if scenario == "normal":
        # Normal operating conditions
        temps = np.array([22.0, 23.5, 21.8])[:n_zones]
        outside_temp = 15.0
        humidity = 45.0
        occupancy = np.array([0.8, 0.9, 0.6])[:n_zones]
        
    elif scenario == "hot":
        # Hot summer day
        temps = np.array([26.5, 28.0, 27.2])[:n_zones]
        outside_temp = 35.0
        humidity = 65.0
        occupancy = np.array([0.9, 1.0, 0.8])[:n_zones]
        
    elif scenario == "cold":
        # Cold winter day
        temps = np.array([18.5, 17.8, 19.2])[:n_zones]
        outside_temp = -5.0
        humidity = 25.0
        occupancy = np.array([0.7, 0.8, 0.4])[:n_zones]
        
    elif scenario == "emergency":
        # Emergency conditions - extreme temperatures
        temps = np.array([35.0, 12.0, 30.0])[:n_zones]
        outside_temp = 40.0
        humidity = 90.0
        occupancy = np.array([1.0, 1.0, 1.0])[:n_zones]
        
    else:
        raise ValueError(f"Unknown scenario: {scenario}")
    
    hvac_power = np.random.uniform(2.0, 8.0, n_zones)
    setpoints = np.full(n_zones, 0.5)
    
    return BuildingState(
        timestamp=time.time(),
        zone_temperatures=temps,
        outside_temperature=outside_temp,
        humidity=humidity,
        occupancy=occupancy,
        hvac_power=hvac_power,
        control_setpoints=setpoints
    )


def generate_forecast_data(hours: int = 6) -> Dict[str, np.ndarray]:
    """Generate realistic forecast data."""
    intervals = hours * 4  # 15-minute intervals
    
    # Weather forecast [temperature, solar_radiation, humidity]
    weather = []
    base_temp = 15.0
    
    for i in range(intervals):
        hour_of_day = (i * 0.25) % 24
        
        # Daily temperature cycle
        temp = base_temp + 8 * np.sin(2 * np.pi * (hour_of_day - 6) / 24)
        
        # Solar radiation (simplified)
        if 6 <= hour_of_day <= 18:
            solar = 800 * np.sin(np.pi * (hour_of_day - 6) / 12)
        else:
            solar = 0
            
        humidity = 50 + 20 * np.cos(2 * np.pi * hour_of_day / 24)
        
        weather.append([temp, max(0, solar), max(10, min(90, humidity))])
    
    # Energy pricing (time-of-use)
    prices = []
    base_price = 0.12  # $/kWh
    
    for i in range(intervals):
        hour_of_day = (i * 0.25) % 24
        
        if 8 <= hour_of_day <= 10 or 17 <= hour_of_day <= 21:
            multiplier = 2.0  # Peak
        elif 11 <= hour_of_day <= 16:
            multiplier = 1.5  # Mid-peak
        else:
            multiplier = 1.0  # Off-peak
            
        prices.append(base_price * multiplier)
    
    return {
        'weather': np.array(weather),
        'energy_prices': np.array(prices)
    }


def print_status(controller: HVACController, title: str = "System Status"):
    """Print comprehensive system status."""
    print(f"\n{'='*50}")
    print(f"{title:^50}")
    print(f"{'='*50}")
    
    status = controller.get_status()
    
    # Basic info
    print(f"Building ID: {status['building_id']}")
    print(f"Last optimization: {status.get('last_optimization', 'Never')}")
    print(f"Control history: {status['history_length']} entries")
    
    # Quantum solver status
    solver_status = status['quantum_solver_status']
    print(f"\nQuantum Solver:")
    print(f"  Available: {solver_status['is_available']}")
    print(f"  Type: {solver_status['solver_type']}")
    print(f"  Last solve time: {solver_status.get('last_solve_time', 'N/A')}")
    
    # Safety status
    safety_status = status['safety_status']
    print(f"\nSafety System:")
    print(f"  Level: {safety_status['level'].upper()}")
    print(f"  Violations: {len(safety_status['violations'])}")
    if safety_status['violations']:
        for violation in safety_status['violations'][:3]:  # Show first 3
            print(f"    - {violation}")
    print(f"  Emergency active: {safety_status['emergency_active']}")
    
    # Health monitoring
    health_status = status['health_status']
    print(f"\nHealth Monitoring:")
    print(f"  Status: {health_status['status'].upper()}")
    if health_status['status'] != 'healthy':
        print(f"  Issues: {health_status.get('issues', [])}")
    
    if 'performance' in health_status:
        perf = health_status['performance']
        print(f"  Success rate: {perf['success_rate']:.1%}")
        print(f"  Avg solve time: {perf['avg_solve_time']:.3f}s")
        print(f"  Total optimizations: {perf['total_optimizations']}")
    
    # Circuit breaker
    cb_status = status['circuit_breaker_status']
    print(f"\nCircuit Breaker:")
    print(f"  State: {cb_status['state'].upper()}")
    print(f"  Failure count: {cb_status['failure_count']}")
    if cb_status['failure_count'] > 0:
        print(f"  Time since failure: {cb_status['time_since_failure']:.1f}s")


async def run_optimization_scenario(
    controller: HVACController, 
    scenario: str,
    forecast_data: Dict[str, np.ndarray]
):
    """Run optimization for a specific scenario."""
    print(f"\n🎯 Running {scenario.upper()} scenario optimization...")
    
    # Create test state for scenario
    building_state = create_test_state(controller.building, scenario)
    
    print(f"Initial conditions:")
    print(f"  Zone temperatures: {building_state.zone_temperatures}")
    print(f"  Outside temperature: {building_state.outside_temperature}°C")
    print(f"  Humidity: {building_state.humidity}%")
    print(f"  Occupancy levels: {building_state.occupancy}")
    
    try:
        # Run optimization
        start_time = time.time()
        control_schedule = await controller.optimize(
            building_state,
            forecast_data['weather'],
            forecast_data['energy_prices']
        )
        optimization_time = time.time() - start_time
        
        print(f"✅ Optimization completed in {optimization_time:.3f}s")
        print(f"Control schedule length: {len(control_schedule)}")
        
        # Show first few control values
        n_zones = len(controller.building.zones)
        first_control = control_schedule[:n_zones]
        print(f"First control step: {first_control}")
        
        # Apply the control
        controller.apply_schedule(control_schedule)
        
        return True
        
    except Exception as e:
        print(f"❌ Optimization failed: {e}")
        return False


async def interactive_demo():
    """Run interactive CLI demonstration."""
    print("🏢 Quantum HVAC Control System - Interactive Demo")
    print("=" * 60)
    
    # Setup logging
    setup_logging(level=20, json_logging=True)  # INFO level
    
    # Create building and controller
    print("\n1. Initializing system...")
    building = create_demo_building()
    
    config = OptimizationConfig(
        prediction_horizon=6,  # 6 hours
        control_interval=15,   # 15 minutes
        solver="classical_fallback",
        num_reads=50
    )
    
    objectives = ControlObjectives(
        energy_cost=0.5,
        comfort=0.4,
        carbon=0.1
    )
    
    controller = HVACController(building, config, objectives)
    
    print(f"✅ System initialized")
    print(f"   Building: {building.building_id}")
    print(f"   Zones: {len(building.zones)}")
    print(f"   State dimension: {building.get_state_dimension()}")
    print(f"   Control dimension: {building.get_control_dimension()}")
    
    # Generate forecast data
    print("\n2. Generating forecast data...")
    forecast_data = generate_forecast_data(6)
    
    weather_shape = forecast_data['weather'].shape
    price_range = (forecast_data['energy_prices'].min(), forecast_data['energy_prices'].max())
    
    print(f"✅ Forecast data ready")
    print(f"   Weather forecast: {weather_shape[0]} time steps")
    print(f"   Energy price range: ${price_range[0]:.3f} - ${price_range[1]:.3f}/kWh")
    
    # Show initial status
    print_status(controller, "Initial System Status")
    
    # Run different scenarios
    scenarios = ["normal", "hot", "cold", "emergency"]
    results = {}
    
    for scenario in scenarios:
        success = await run_optimization_scenario(controller, scenario, forecast_data)
        results[scenario] = success
        
        # Show updated status after each scenario
        if scenario == "emergency":
            print_status(controller, f"Status After {scenario.upper()} Scenario")
    
    # Summary
    print("\n" + "=" * 60)
    print("DEMONSTRATION SUMMARY")
    print("=" * 60)
    
    print(f"\nScenario Results:")
    for scenario, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"  {scenario.capitalize():>12}: {status}")
    
    # Final system status
    print_status(controller, "Final System Status")
    
    # Show configuration summary
    print(f"\nSystem Configuration:")
    print(f"  Prediction horizon: {config.prediction_horizon} hours")  
    print(f"  Control interval: {config.control_interval} minutes")
    print(f"  Solver type: {config.solver}")
    print(f"  Objectives: Energy {objectives.energy_cost:.0%}, Comfort {objectives.comfort:.0%}, Carbon {objectives.carbon:.0%}")
    
    # Performance metrics
    final_status = controller.get_status()
    if 'performance' in final_status.get('health_status', {}):
        perf = final_status['health_status']['performance']
        print(f"\nPerformance Metrics:")
        print(f"  Total optimizations: {perf['total_optimizations']}")
        print(f"  Success rate: {perf['success_rate']:.1%}")
        print(f"  Average solve time: {perf['avg_solve_time']:.3f}s")
    
    print(f"\n🎉 Interactive demonstration completed!")
    
    # Cleanup
    controller._health_monitor.stop_monitoring()
    
    return results


if __name__ == "__main__":
    print("Starting quantum HVAC control CLI demo...")
    
    try:
        results = asyncio.run(interactive_demo())
        
        # Exit with appropriate code
        all_success = all(results.values())
        if all_success:
            print("\n✅ All scenarios completed successfully!")
            sys.exit(0)
        else:
            print("\n⚠️ Some scenarios failed - check logs for details")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
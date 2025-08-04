#!/usr/bin/env python3
"""
Basic HVAC optimization example using quantum annealing.

This example demonstrates:
- Creating a building model
- Setting up quantum optimization
- Running optimization with mock data
- Applying control schedule
"""

import asyncio
import numpy as np
from pathlib import Path
import sys

# Add quantum_ctl to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from quantum_ctl import HVACController, Building
from quantum_ctl.core.controller import OptimizationConfig, ControlObjectives
from quantum_ctl.models.building import BuildingState, ZoneConfig


async def basic_optimization_example():
    """Run basic HVAC optimization example."""
    print("🏢 Quantum HVAC Optimization Example")
    print("=" * 40)
    
    # 1. Create Building Model
    print("\n1. Creating building model...")
    
    # Define zones with specific characteristics
    zones = [
        ZoneConfig(
            zone_id="conference_room",
            area=80.0,    # m²
            volume=240.0, # m³ 
            thermal_mass=200.0,  # kJ/K
            max_heating_power=12.0,  # kW
            max_cooling_power=10.0,  # kW
            comfort_temp_min=21.0,   # °C
            comfort_temp_max=25.0    # °C
        ),
        ZoneConfig(
            zone_id="open_office",
            area=200.0,
            volume=600.0,
            thermal_mass=500.0,
            max_heating_power=20.0,
            max_cooling_power=16.0,
            comfort_temp_min=20.0,
            comfort_temp_max=24.0
        ),
        ZoneConfig(
            zone_id="server_room",
            area=40.0,
            volume=120.0,
            thermal_mass=100.0,
            max_heating_power=5.0,
            max_cooling_power=15.0,  # Needs more cooling
            comfort_temp_min=18.0,   # Cooler for servers
            comfort_temp_max=22.0
        )
    ]
    
    building = Building(
        building_id="example_office",
        zones=zones,
        occupancy_schedule="office_standard"
    )
    
    print(f"   Building: {building.building_id}")
    print(f"   Zones: {building.n_zones}")
    print(f"   State dimension: {building.get_state_dimension()}")
    print(f"   Control dimension: {building.get_control_dimension()}")
    
    # 2. Configure Optimization
    print("\n2. Configuring quantum optimization...")
    
    config = OptimizationConfig(
        prediction_horizon=6,     # 6 hours ahead
        control_interval=15,      # 15-minute control updates
        solver="classical_fallback",  # Use fallback for demo
        num_reads=100
    )
    
    objectives = ControlObjectives(
        energy_cost=0.5,  # 50% energy cost minimization
        comfort=0.4,      # 40% comfort optimization  
        carbon=0.1        # 10% carbon footprint reduction
    )
    
    controller = HVACController(building, config, objectives)
    
    print(f"   Prediction horizon: {config.prediction_horizon} hours")
    print(f"   Control interval: {config.control_interval} minutes")
    print(f"   Solver: {config.solver}")
    print(f"   Objectives: Energy {objectives.energy_cost:.1%}, "
          f"Comfort {objectives.comfort:.1%}, Carbon {objectives.carbon:.1%}")
    
    # 3. Create Current Building State
    print("\n3. Setting current building state...")
    
    current_state = BuildingState(
        timestamp=0.0,
        zone_temperatures=np.array([23.5, 22.0, 20.5]),  # Current temps
        outside_temperature=10.0,  # Cold outside
        humidity=45.0,
        occupancy=np.array([0.8, 0.9, 0.1]),  # High occupancy in office areas
        hvac_power=np.array([8.0, 12.0, 3.0]),  # Current HVAC power
        control_setpoints=np.array([0.6, 0.7, 0.3])  # Current control settings
    )
    
    print(f"   Zone temperatures: {current_state.zone_temperatures} °C")
    print(f"   Outside temperature: {current_state.outside_temperature} °C")
    print(f"   Occupancy levels: {current_state.occupancy}")
    
    # 4. Create Forecast Data
    print("\n4. Generating forecast data...")
    
    # 6 hours * 4 (15-min intervals per hour) = 24 time steps
    n_steps = config.prediction_horizon * 4
    
    # Weather forecast: [temperature, solar_radiation, humidity]
    weather_forecast = np.array([
        [10.0 + 2*np.sin(i/4), 200 + 300*np.sin(i/6), 45 + 10*np.cos(i/8)]
        for i in range(n_steps)
    ])
    
    # Energy prices ($/kWh) - higher during peak hours
    base_price = 0.12
    peak_hours = [8, 9, 10, 17, 18, 19, 20]  # Peak hours
    energy_prices = np.array([
        base_price * (1.5 if (i//4) % 24 in peak_hours else 1.0)
        for i in range(n_steps)
    ])
    
    print(f"   Weather forecast: {n_steps} time steps")
    print(f"   Temperature range: {weather_forecast[:, 0].min():.1f} - {weather_forecast[:, 0].max():.1f} °C")
    print(f"   Energy price range: ${energy_prices.min():.3f} - ${energy_prices.max():.3f}/kWh")
    
    # 5. Run Quantum Optimization
    print("\n5. Running quantum optimization...")
    print("   This may take a few moments...")
    
    try:
        control_schedule = await controller.optimize(
            current_state=current_state,
            weather_forecast=weather_forecast,
            energy_prices=energy_prices
        )
        
        print(f"   ✅ Optimization successful!")
        print(f"   Generated control schedule: {len(control_schedule)} values")
        print(f"   Control range: [{control_schedule.min():.3f}, {control_schedule.max():.3f}]")
        
        # Reshape for analysis
        n_controls = building.get_control_dimension()
        schedule_2d = control_schedule.reshape((-1, n_controls))
        
        print(f"\n   Control Schedule Preview (first 4 time steps):")
        print(f"   Time Step | Conference | Office | Server")
        print(f"   --------- | ---------- | ------ | ------")
        for i in range(min(4, len(schedule_2d))):
            print(f"   {i+1:8d} | {schedule_2d[i,0]:9.3f} | {schedule_2d[i,1]:6.3f} | {schedule_2d[i,2]:6.3f}")
        
    except Exception as e:
        print(f"   ❌ Optimization failed: {e}")
        return False
    
    # 6. Apply Control Schedule
    print("\n6. Applying control schedule...")
    
    # Apply the first control step
    controller.apply_schedule(control_schedule)
    
    # Get updated building state
    updated_state = building.get_state()
    
    print(f"   Control commands applied:")
    for i, zone in enumerate(building.zones):
        old_setpoint = current_state.control_setpoints[i]
        new_setpoint = updated_state.control_setpoints[i]
        print(f"   {zone.zone_id:15s}: {old_setpoint:.3f} → {new_setpoint:.3f}")
    
    # 7. Performance Summary
    print("\n7. Performance Summary")
    print("=" * 40)
    
    status = controller.get_status()
    
    print(f"Building ID: {status['building_id']}")
    print(f"Optimization count: {status['history_length']}")
    print(f"Last optimization: {status['last_optimization']}")
    
    quantum_status = status['quantum_solver_status']
    print(f"Quantum solver available: {quantum_status['is_available']}")
    print(f"Solver type: {quantum_status['solver_type']}")
    
    # Estimate energy savings (simplified calculation)
    baseline_energy = np.sum(current_state.hvac_power) * 6  # 6 hours
    optimized_energy = np.mean(control_schedule) * np.sum([z.max_heating_power for z in zones]) * 6
    energy_savings = max(0, (baseline_energy - optimized_energy) / baseline_energy * 100)
    
    print(f"\nEstimated Results:")
    print(f"  Baseline energy: {baseline_energy:.1f} kWh")
    print(f"  Optimized energy: {optimized_energy:.1f} kWh") 
    print(f"  Energy savings: {energy_savings:.1f}%")
    
    cost_savings = energy_savings / 100 * np.mean(energy_prices) * baseline_energy
    print(f"  Cost savings: ${cost_savings:.2f}")
    
    print(f"\n🎉 Quantum HVAC optimization complete!")
    return True


if __name__ == "__main__":
    print("Starting quantum HVAC optimization example...")
    
    try:
        success = asyncio.run(basic_optimization_example())
        if success:
            print("\n✅ Example completed successfully!")
        else:
            print("\n❌ Example failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Example interrupted by user")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)
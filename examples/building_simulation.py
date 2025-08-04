#!/usr/bin/env python3
"""
Building thermal simulation example.

Demonstrates building thermal dynamics and state evolution over time.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add quantum_ctl to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from quantum_ctl.models.building import Building, BuildingState, ZoneConfig


def simulate_building_dynamics():
    """Simulate building thermal dynamics over time."""
    print("🏢 Building Thermal Simulation")
    print("=" * 35)
    
    # Create a test building
    zones = [
        ZoneConfig("office", 150.0, 450.0, 400.0, 15.0, 12.0, 20.0, 24.0),
        ZoneConfig("lobby", 100.0, 300.0, 250.0, 10.0, 8.0, 18.0, 26.0)
    ]
    
    building = Building("simulation_building", zones=zones)
    
    print(f"Building: {building.building_id}")
    print(f"Zones: {[z.zone_id for z in building.zones]}")
    print(f"State dimension: {building.get_state_dimension()}")
    
    # Initial conditions
    initial_state = BuildingState(
        timestamp=0.0,
        zone_temperatures=np.array([18.0, 16.0]),  # Cold start
        outside_temperature=5.0,  # Cold outside
        humidity=60.0,
        occupancy=np.array([0.0, 0.0]),  # Empty building
        hvac_power=np.array([0.0, 0.0]),  # HVAC off
        control_setpoints=np.array([0.8, 0.6])  # High heating demand
    )
    
    building.update_state(initial_state)
    
    # Simulation parameters
    simulation_hours = 8
    time_step_minutes = 15
    n_steps = simulation_hours * 60 // time_step_minutes
    
    print(f"\nSimulation parameters:")
    print(f"  Duration: {simulation_hours} hours")
    print(f"  Time step: {time_step_minutes} minutes")
    print(f"  Total steps: {n_steps}")
    
    # Storage for results
    results = {
        'time': [],
        'zone_temps': [],
        'outside_temp': [],
        'occupancy': [],
        'control_signals': [],
        'hvac_power': []
    }
    
    print(f"\nRunning simulation...")
    
    for step in range(n_steps):
        current_time = step * time_step_minutes / 60.0  # Hours
        
        # Get current state
        state = building.get_state()
        
        # Simulate occupancy schedule (office hours)
        if 2 <= current_time <= 6:  # 8 AM to 12 PM
            occupancy = np.array([0.8, 0.3])  # High office, low lobby
        elif 6 <= current_time <= 8:  # 12 PM to 2 PM  
            occupancy = np.array([0.6, 0.5])  # Lunch time
        else:
            occupancy = np.array([0.1, 0.1])  # After hours
        
        # Outside temperature variation (daily cycle)
        outside_temp = 5.0 + 8.0 * np.sin(2 * np.pi * current_time / 24)
        
        # Control strategy: maintain comfort when occupied
        if np.max(occupancy) > 0.5:
            # Occupied - maintain comfort
            temp_error = 22.0 - state.zone_temperatures  # Target 22°C
            control_signals = np.clip(0.5 + 0.1 * temp_error, 0.0, 1.0)
        else:
            # Unoccupied - energy saving mode
            temp_error = 18.0 - state.zone_temperatures  # Lower target
            control_signals = np.clip(0.3 + 0.05 * temp_error, 0.0, 0.7)
        
        # Create disturbances (weather, occupancy, solar gain)
        disturbances = np.zeros(building.get_state_dimension())
        disturbances[building.n_zones] = outside_temp  # Outside temperature
        disturbances[:building.n_zones] = occupancy * 2.0  # Occupant heat gain
        
        # Simulate one time step
        next_state = building.simulate_step(control_signals, disturbances)
        
        # Update occupancy in state
        next_state.occupancy = occupancy
        next_state.outside_temperature = outside_temp
        
        # Store results
        results['time'].append(current_time)
        results['zone_temps'].append(next_state.zone_temperatures.copy())
        results['outside_temp'].append(outside_temp)
        results['occupancy'].append(occupancy.copy())
        results['control_signals'].append(control_signals.copy())
        results['hvac_power'].append(next_state.hvac_power.copy())
        
        if step % 8 == 0:  # Print every 2 hours
            print(f"  t={current_time:4.1f}h: T={next_state.zone_temperatures}, "
                  f"Occ={occupancy}, Ctrl={control_signals}")
    
    # Convert to numpy arrays
    results['time'] = np.array(results['time'])
    results['zone_temps'] = np.array(results['zone_temps'])
    results['outside_temp'] = np.array(results['outside_temp'])
    results['occupancy'] = np.array(results['occupancy'])
    results['control_signals'] = np.array(results['control_signals'])
    results['hvac_power'] = np.array(results['hvac_power'])
    
    print(f"\n📊 Simulation Results:")
    print(f"  Final zone temperatures: {results['zone_temps'][-1]} °C")
    print(f"  Temperature range: {results['zone_temps'].min():.1f} - {results['zone_temps'].max():.1f} °C")
    print(f"  Average control signals: {np.mean(results['control_signals'], axis=0)}")
    print(f"  Total energy consumption: {np.sum(results['hvac_power']):.1f} kWh")
    
    # Save results to file
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / "simulation_results.npz"
    np.savez(output_file, **results)
    print(f"  Results saved to: {output_file}")
    
    # Create simple text plot
    print(f"\n📈 Temperature Profile:")
    print(f"Time(h) | Office | Lobby | Outside")
    print(f"------- | ------ | ----- | -------")
    
    for i in range(0, len(results['time']), 8):  # Every 2 hours
        t = results['time'][i]
        office_temp = results['zone_temps'][i, 0]
        lobby_temp = results['zone_temps'][i, 1] 
        outside_temp = results['outside_temp'][i]
        
        print(f"{t:6.1f}  | {office_temp:5.1f}  | {lobby_temp:4.1f}  | {outside_temp:6.1f}")
    
    return results


def analyze_thermal_performance(results):
    """Analyze thermal performance metrics."""
    print(f"\n🔍 Performance Analysis:")
    
    # Comfort analysis
    office_temps = results['zone_temps'][:, 0]
    lobby_temps = results['zone_temps'][:, 1]
    
    office_comfort_violations = np.sum((office_temps < 20.0) | (office_temps > 24.0))
    lobby_comfort_violations = np.sum((lobby_temps < 18.0) | (lobby_temps > 26.0))
    
    total_hours = len(results['time']) * 0.25  # 15-min steps
    
    print(f"Comfort Performance:")
    print(f"  Office comfort violations: {office_comfort_violations} / {len(office_temps)} ({office_comfort_violations/len(office_temps)*100:.1f}%)")
    print(f"  Lobby comfort violations: {lobby_comfort_violations} / {len(lobby_temps)} ({lobby_comfort_violations/len(lobby_temps)*100:.1f}%)")
    
    # Energy analysis
    total_energy = np.sum(results['hvac_power']) * 0.25  # kWh (15-min intervals)
    avg_power = np.mean(results['hvac_power'])
    peak_power = np.max(results['hvac_power'])
    
    print(f"Energy Performance:")
    print(f"  Total energy: {total_energy:.1f} kWh")
    print(f"  Average power: {avg_power:.1f} kW")
    print(f"  Peak power: {peak_power:.1f} kW")
    print(f"  Energy intensity: {total_energy/total_hours:.2f} kWh/h")
    
    # Control analysis
    avg_control = np.mean(results['control_signals'], axis=0)
    control_variation = np.std(results['control_signals'], axis=0)
    
    print(f"Control Performance:")
    print(f"  Average control signals: {avg_control}")
    print(f"  Control variation (std): {control_variation}")
    
    # Responsiveness analysis
    temp_changes = np.abs(np.diff(results['zone_temps'], axis=0))
    avg_temp_change = np.mean(temp_changes, axis=0)
    
    print(f"System Responsiveness:")
    print(f"  Average temperature change per step: {avg_temp_change} °C")


if __name__ == "__main__":
    print("Starting building thermal simulation...")
    
    try:
        results = simulate_building_dynamics()
        analyze_thermal_performance(results)
        
        print(f"\n✅ Simulation completed successfully!")
        print(f"   Check the output/ directory for detailed results")
        
    except Exception as e:
        print(f"\n💥 Simulation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
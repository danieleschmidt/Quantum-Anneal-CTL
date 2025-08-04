#!/usr/bin/env python3
"""
Microgrid quantum optimization example.

Demonstrates:
- Multi-building coordination
- Energy storage optimization  
- Peer-to-peer energy trading
- Renewable energy integration
"""

import asyncio
import numpy as np
from pathlib import Path
import sys
import json
from datetime import datetime, timedelta

# Add quantum_ctl to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from quantum_ctl import Building, HVACController
from quantum_ctl.core.controller import OptimizationConfig, ControlObjectives
from quantum_ctl.core.microgrid import MicroGridController
from quantum_ctl.models.building import BuildingState, ZoneConfig


async def microgrid_optimization_example():
    """Run comprehensive microgrid optimization example."""
    print("🏘️  Quantum Microgrid Optimization Example")
    print("=" * 45)
    
    # 1. Create Multiple Buildings
    print("\n1. Creating microgrid buildings...")
    
    buildings = []
    
    # Office Building
    office_zones = [
        ZoneConfig("office_1", 120.0, 360.0, 300.0, 18.0, 15.0, 20.0, 24.0),
        ZoneConfig("office_2", 120.0, 360.0, 300.0, 18.0, 15.0, 20.0, 24.0),
        ZoneConfig("conference", 80.0, 240.0, 200.0, 12.0, 10.0, 21.0, 25.0),
    ]
    
    office_building = Building(
        building_id="office_building",
        zones=office_zones,
        occupancy_schedule="office_standard",
        latitude=40.7128,
        longitude=-74.0060
    )
    buildings.append(office_building)
    
    # Retail Store
    retail_zones = [
        ZoneConfig("sales_floor", 200.0, 600.0, 500.0, 25.0, 20.0, 19.0, 26.0),
        ZoneConfig("storage", 60.0, 180.0, 150.0, 8.0, 6.0, 15.0, 28.0),
    ]
    
    retail_building = Building(
        building_id="retail_store", 
        zones=retail_zones,
        occupancy_schedule="retail",
        latitude=40.7128,
        longitude=-74.0060
    )
    buildings.append(retail_building)
    
    # Residential Complex
    residential_zones = [
        ZoneConfig("apt_1", 80.0, 240.0, 200.0, 12.0, 10.0, 20.0, 25.0),
        ZoneConfig("apt_2", 80.0, 240.0, 200.0, 12.0, 10.0, 20.0, 25.0),
        ZoneConfig("common_area", 100.0, 300.0, 250.0, 15.0, 12.0, 18.0, 26.0),
    ]
    
    residential_building = Building(
        building_id="residential_complex",
        zones=residential_zones, 
        occupancy_schedule="residential",
        latitude=40.7128,
        longitude=-74.0060
    )
    buildings.append(residential_building)
    
    print(f"   Created {len(buildings)} buildings:")
    for building in buildings:
        print(f"   - {building.building_id}: {len(building.zones)} zones")
    
    # 2. Create Microgrid Controller
    print("\n2. Initializing microgrid controller...")
    
    microgrid = MicroGridController(
        buildings=buildings,
        solar_capacity_kw=500.0,     # 500 kW solar array
        battery_capacity_kwh=1000.0, # 1 MWh battery storage
        grid_connection_limit_kw=300.0,
        enable_peer_trading=True
    )
    
    print(f"   Solar capacity: {microgrid.solar_capacity_kw} kW")
    print(f"   Battery capacity: {microgrid.battery_capacity_kwh} kWh")
    print(f"   Grid limit: {microgrid.grid_connection_limit_kw} kW")
    print(f"   Buildings connected: {len(microgrid.buildings)}")
    
    # 3. Generate Microgrid Data
    print("\n3. Generating microgrid forecast data...")
    
    # 12-hour optimization horizon
    horizon_hours = 12
    n_steps = horizon_hours * 4  # 15-minute intervals
    
    # Solar generation forecast (kW)
    current_hour = 8  # Start at 8 AM
    solar_generation = []
    for i in range(n_steps):
        hour = (current_hour + i/4) % 24
        if 6 <= hour <= 18:  # Daylight hours
            # Peak solar at noon, realistic curve
            solar_fraction = np.sin(np.pi * (hour - 6) / 12) ** 2
            cloud_factor = 0.8 + 0.2 * np.sin(i/3)  # Cloud variation
            generation = microgrid.solar_capacity_kw * solar_fraction * cloud_factor
        else:
            generation = 0.0
        solar_generation.append(generation)
    
    solar_generation = np.array(solar_generation)
    
    # Grid energy prices ($/kWh) - time-of-use pricing
    base_price = 0.12
    energy_prices = []
    for i in range(n_steps):
        hour = (current_hour + i/4) % 24
        if 7 <= hour <= 10 or 17 <= hour <= 21:  # Peak hours
            price = base_price * 1.8  # Peak pricing
        elif 11 <= hour <= 16:  # Mid-peak
            price = base_price * 1.3
        else:  # Off-peak
            price = base_price * 0.8
        
        # Add some market volatility
        price *= (1.0 + 0.1 * np.sin(i/8))
        energy_prices.append(price)
    
    energy_prices = np.array(energy_prices)
    
    # Weather forecast for all buildings
    weather_base_temp = 15.0  # Base temperature
    weather_forecast = np.array([
        [
            weather_base_temp + 5 * np.sin(2*np.pi*i/96),  # Daily temperature cycle
            300 + 400 * np.maximum(0, np.sin(np.pi*i/48)),  # Solar radiation
            50 + 15 * np.cos(i/12)  # Humidity
        ]
        for i in range(n_steps)
    ])
    
    print(f"   Forecast horizon: {horizon_hours} hours ({n_steps} steps)")
    print(f"   Solar generation range: {solar_generation.min():.0f} - {solar_generation.max():.0f} kW")
    print(f"   Energy price range: ${energy_prices.min():.3f} - ${energy_prices.max():.3f}/kWh")
    print(f"   Weather temperature range: {weather_forecast[:,0].min():.1f} - {weather_forecast[:,0].max():.1f} °C")
    
    # 4. Set Building States
    print("\n4. Setting initial building states...")
    
    building_states = []
    
    # Office building state (morning startup)
    office_state = BuildingState(
        timestamp=current_hour * 3600,
        zone_temperatures=np.array([18.0, 17.5, 19.0]),  # Cold overnight
        outside_temperature=weather_forecast[0, 0],
        humidity=weather_forecast[0, 2],
        occupancy=np.array([0.1, 0.1, 0.05]),  # Early morning, few people
        hvac_power=np.array([5.0, 5.0, 2.0]),
        control_setpoints=np.array([0.7, 0.7, 0.5])  # Pre-heating
    )
    building_states.append(office_state)
    
    # Retail store state
    retail_state = BuildingState(
        timestamp=current_hour * 3600,
        zone_temperatures=np.array([16.0, 15.0]),  # Cold overnight
        outside_temperature=weather_forecast[0, 0],
        humidity=weather_forecast[0, 2],
        occupancy=np.array([0.2, 0.0]),  # Opening preparations
        hvac_power=np.array([8.0, 2.0]),
        control_setpoints=np.array([0.8, 0.3])
    )
    building_states.append(retail_state)
    
    # Residential state
    residential_state = BuildingState(
        timestamp=current_hour * 3600,
        zone_temperatures=np.array([20.0, 19.5, 17.0]),  # Maintained overnight
        outside_temperature=weather_forecast[0, 0],
        humidity=weather_forecast[0, 2],
        occupancy=np.array([0.8, 0.9, 0.1]),  # People at home in morning
        hvac_power=np.array([3.0, 3.5, 1.0]),
        control_setpoints=np.array([0.4, 0.45, 0.2])
    )
    building_states.append(residential_state)
    
    print(f"   Initial building temperatures:")
    for i, (building, state) in enumerate(zip(buildings, building_states)):
        print(f"   - {building.building_id}: {state.zone_temperatures.mean():.1f}°C avg")
    
    # 5. Run Microgrid Optimization
    print("\n5. Running quantum microgrid optimization...")
    print("   This optimizes HVAC, storage, and trading simultaneously...")
    
    try:
        # Microgrid optimization config
        config = OptimizationConfig(
            prediction_horizon=horizon_hours,
            control_interval=15,
            solver="hybrid_v2",  # Use hybrid for larger problems
            num_reads=500
        )
        
        # Multi-objective optimization for microgrid
        objectives = ControlObjectives(
            energy_cost=0.4,  # 40% cost minimization
            comfort=0.4,      # 40% comfort across all buildings
            carbon=0.2        # 20% carbon footprint reduction
        )
        
        # Run comprehensive optimization
        optimization_result = await microgrid.optimize_quantum(
            building_states=building_states,
            weather_forecast=weather_forecast,
            energy_prices=energy_prices,
            solar_generation=solar_generation,
            config=config,
            objectives=objectives
        )
        
        print(f"   ✅ Microgrid optimization successful!")
        
        # Extract results
        hvac_schedules = optimization_result['hvac_schedules']
        battery_schedule = optimization_result['battery_schedule']
        trading_schedule = optimization_result.get('peer_trading', {})
        grid_schedule = optimization_result['grid_interaction']
        
        print(f"   Generated schedules:")
        print(f"   - HVAC controls: {len(hvac_schedules)} buildings")
        print(f"   - Battery operations: {len(battery_schedule)} time steps")
        print(f"   - Grid interactions: {len(grid_schedule)} time steps")
        
    except Exception as e:
        print(f"   ❌ Microgrid optimization failed: {e}")
        print(f"   Using fallback optimization...")
        
        # Fallback: optimize each building individually
        hvac_schedules = []
        for i, (building, state) in enumerate(zip(buildings, building_states)):
            controller = HVACController(building, config, objectives)
            schedule = await controller.optimize(state, weather_forecast, energy_prices)
            hvac_schedules.append(schedule)
        
        # Simple battery and grid schedules
        battery_schedule = np.zeros(n_steps)  # No battery operation
        grid_schedule = np.ones(n_steps) * 100  # Constant grid draw
        trading_schedule = {}
    
    # 6. Analyze Microgrid Performance
    print("\n6. Microgrid Performance Analysis")
    print("=" * 45)
    
    # Energy balance analysis
    total_building_demand = 0
    for i, (building, schedule) in enumerate(zip(buildings, hvac_schedules)):
        # Estimate building power demand
        n_zones = len(building.zones)
        schedule_2d = schedule.reshape((-1, n_zones))
        avg_control = np.mean(schedule_2d, axis=0)
        max_powers = [zone.max_heating_power + zone.max_cooling_power for zone in building.zones]
        building_demand = np.sum(avg_control * max_powers) * horizon_hours
        total_building_demand += building_demand
        
        print(f"Building {building.building_id}:")
        print(f"  Average control: {avg_control}")
        print(f"  Estimated energy: {building_demand:.1f} kWh")
    
    # Solar and grid analysis
    total_solar = np.sum(solar_generation) * 0.25  # kWh (15-min intervals)
    avg_grid_price = np.mean(energy_prices)
    
    print(f"\nMicrogrid Energy Balance:")
    print(f"  Total building demand: {total_building_demand:.1f} kWh")
    print(f"  Solar generation: {total_solar:.1f} kWh")
    print(f"  Solar coverage: {min(100, total_solar/total_building_demand*100):.1f}%")
    print(f"  Average grid price: ${avg_grid_price:.3f}/kWh")
    
    # Cost analysis
    if total_solar >= total_building_demand:
        net_grid_usage = 0
        excess_solar = total_solar - total_building_demand
        revenue = excess_solar * avg_grid_price * 0.8  # Sell-back rate
        cost = 0
    else:
        net_grid_usage = total_building_demand - total_solar
        cost = net_grid_usage * avg_grid_price
        revenue = 0
        excess_solar = 0
    
    print(f"\nCost Analysis:")
    print(f"  Net grid usage: {net_grid_usage:.1f} kWh")
    print(f"  Grid cost: ${cost:.2f}")
    print(f"  Excess solar: {excess_solar:.1f} kWh")
    print(f"  Solar revenue: ${revenue:.2f}")
    print(f"  Net cost: ${cost - revenue:.2f}")
    
    # Peak demand analysis
    peak_building_demand = 0
    for i, (building, schedule) in enumerate(zip(buildings, hvac_schedules)):
        n_zones = len(building.zones)
        schedule_2d = schedule.reshape((-1, n_zones))
        max_powers = [zone.max_heating_power + zone.max_cooling_power for zone in building.zones]
        peak_power = np.max([np.sum(step * max_powers) for step in schedule_2d])
        peak_building_demand += peak_power
    
    peak_solar = np.max(solar_generation)
    peak_net_demand = max(0, peak_building_demand - peak_solar)
    
    print(f"\nPeak Demand Analysis:")
    print(f"  Peak building demand: {peak_building_demand:.1f} kW")
    print(f"  Peak solar generation: {peak_solar:.1f} kW")
    print(f"  Peak net demand: {peak_net_demand:.1f} kW")
    print(f"  Grid limit utilization: {peak_net_demand/microgrid.grid_connection_limit_kw*100:.1f}%")
    
    # Environmental impact
    grid_carbon_intensity = 0.4  # kg CO2/kWh (US average)
    solar_carbon_intensity = 0.05  # kg CO2/kWh (lifecycle)
    
    grid_carbon = net_grid_usage * grid_carbon_intensity
    solar_carbon = total_solar * solar_carbon_intensity
    total_carbon = grid_carbon + solar_carbon
    
    print(f"\nEnvironmental Impact:")
    print(f"  Grid carbon: {grid_carbon:.1f} kg CO2")
    print(f"  Solar carbon: {solar_carbon:.1f} kg CO2") 
    print(f"  Total carbon: {total_carbon:.1f} kg CO2")
    print(f"  Carbon intensity: {total_carbon/total_building_demand:.3f} kg CO2/kWh")
    
    # 7. Save Results
    print("\n7. Saving optimization results...")
    
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    
    results = {
        'microgrid_config': {
            'buildings': [b.building_id for b in buildings],
            'solar_capacity_kw': microgrid.solar_capacity_kw,
            'battery_capacity_kwh': microgrid.battery_capacity_kwh,
            'grid_limit_kw': microgrid.grid_connection_limit_kw
        },
        'optimization_params': {
            'horizon_hours': horizon_hours,
            'solver': config.solver,
            'objectives': {
                'energy_cost': objectives.energy_cost,
                'comfort': objectives.comfort,
                'carbon': objectives.carbon
            }
        },
        'forecasts': {
            'solar_generation': solar_generation.tolist(),
            'energy_prices': energy_prices.tolist(),
            'weather': weather_forecast.tolist()
        },
        'results': {
            'total_building_demand_kwh': total_building_demand,
            'total_solar_kwh': total_solar,
            'net_cost_usd': cost - revenue,
            'total_carbon_kg': total_carbon,
            'peak_demand_kw': peak_building_demand,
            'solar_coverage_percent': min(100, total_solar/total_building_demand*100)
        },
        'hvac_schedules': {
            buildings[i].building_id: schedule.tolist() 
            for i, schedule in enumerate(hvac_schedules)
        },
        'timestamp': datetime.now().isoformat()
    }
    
    output_file = output_dir / "microgrid_optimization.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"   Results saved to: {output_file}")
    
    print(f"\n🎉 Quantum microgrid optimization complete!")
    print(f"   Optimized {len(buildings)} buildings with renewable integration")
    print(f"   Achieved {min(100, total_solar/total_building_demand*100):.1f}% solar coverage")
    print(f"   Net operating cost: ${cost - revenue:.2f}")
    
    return results


if __name__ == "__main__":
    print("Starting quantum microgrid optimization example...")
    
    try:
        results = asyncio.run(microgrid_optimization_example())
        print("\n✅ Microgrid example completed successfully!")
        print("   Check the output/ directory for detailed results")
        
    except KeyboardInterrupt:
        print("\n🛑 Example interrupted by user")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
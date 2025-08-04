#!/usr/bin/env python3
"""
Complete microgrid optimization example using quantum annealing.

This example demonstrates:
- Creating multiple buildings in a microgrid
- Solar generation and battery storage
- Coordinated quantum optimization
- Energy trading between buildings
- Real-time monitoring and control
"""

import asyncio
import numpy as np
from pathlib import Path
import sys
import time

# Add quantum_ctl to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from quantum_ctl import MicroGridController, Building
from quantum_ctl.core.controller import OptimizationConfig, ControlObjectives
from quantum_ctl.core.microgrid import MicroGridConfig
from quantum_ctl.models.building import BuildingState, ZoneConfig


async def create_sample_buildings():
    """Create sample buildings for microgrid demonstration."""
    
    # Building 1: Office Complex
    office_zones = [
        ZoneConfig("lobby", 150.0, 450.0, 300.0, 15.0, 12.0, 20.0, 24.0),
        ZoneConfig("open_office", 400.0, 1200.0, 800.0, 25.0, 20.0, 21.0, 25.0),
        ZoneConfig("conference", 100.0, 300.0, 200.0, 12.0, 10.0, 20.0, 24.0),
        ZoneConfig("server_room", 50.0, 150.0, 100.0, 8.0, 18.0, 18.0, 22.0)
    ]
    
    office_building = Building(
        building_id="office_complex",
        zones=office_zones,
        occupancy_schedule="office_standard",
        latitude=40.7128,
        longitude=-74.0060
    )
    
    # Building 2: Retail Store
    retail_zones = [
        ZoneConfig("sales_floor", 300.0, 900.0, 600.0, 20.0, 16.0, 20.0, 26.0),
        ZoneConfig("storage", 100.0, 300.0, 200.0, 8.0, 6.0, 18.0, 28.0),
        ZoneConfig("office", 50.0, 150.0, 100.0, 6.0, 5.0, 21.0, 25.0)
    ]
    
    retail_building = Building(
        building_id="retail_store",
        zones=retail_zones,
        occupancy_schedule="retail",
        latitude=40.7130,
        longitude=-74.0058
    )
    
    # Building 3: Residential Complex
    residential_zones = [
        ZoneConfig("apartment_1", 80.0, 240.0, 160.0, 8.0, 6.0, 20.0, 26.0),
        ZoneConfig("apartment_2", 80.0, 240.0, 160.0, 8.0, 6.0, 20.0, 26.0),
        ZoneConfig("common_area", 120.0, 360.0, 240.0, 12.0, 10.0, 21.0, 25.0)
    ]
    
    residential_building = Building(
        building_id="residential_complex",
        zones=residential_zones,
        occupancy_schedule="residential",
        latitude=40.7132,
        longitude=-74.0056
    )
    
    return [office_building, retail_building, residential_building]


def generate_solar_forecast(hours: int = 24) -> np.ndarray:
    """Generate realistic solar generation forecast."""
    # 15-minute intervals
    intervals = hours * 4
    solar_generation = []
    
    for i in range(intervals):
        hour_of_day = (i * 0.25) % 24
        
        # Solar generation profile (simplified)
        if 6 <= hour_of_day <= 18:
            # Peak generation around noon
            solar_factor = np.sin(np.pi * (hour_of_day - 6) / 12)
            # Add some variability for clouds
            cloud_factor = 0.8 + 0.4 * np.sin(i * 0.1) * np.cos(i * 0.05)
            generation = 100 * solar_factor * cloud_factor  # kW
        else:
            generation = 0.0
        
        solar_generation.append(max(0, generation))
    
    return np.array(solar_generation)


def generate_dynamic_pricing(hours: int = 24) -> np.ndarray:
    """Generate time-of-use electricity pricing."""
    intervals = hours * 4
    prices = []
    
    base_price = 0.12  # $/kWh
    
    for i in range(intervals):
        hour_of_day = (i * 0.25) % 24
        
        # Peak pricing during high demand hours
        if 8 <= hour_of_day <= 10 or 17 <= hour_of_day <= 21:
            price_multiplier = 2.0  # Peak hours
        elif 11 <= hour_of_day <= 16:
            price_multiplier = 1.5  # Mid-peak
        else:
            price_multiplier = 1.0  # Off-peak
        
        # Add some market volatility
        volatility = 1.0 + 0.1 * np.sin(i * 0.2)
        final_price = base_price * price_multiplier * volatility
        prices.append(final_price)
    
    return np.array(prices)


async def create_building_states(buildings):
    """Create realistic current states for all buildings."""
    states = []
    
    for i, building in enumerate(buildings):
        n_zones = len(building.zones)
        
        # Different temperature profiles for different building types
        if "office" in building.building_id:
            base_temp = 22.0
            temp_variation = np.random.normal(0, 1.0, n_zones)
        elif "retail" in building.building_id:
            base_temp = 21.5
            temp_variation = np.random.normal(0, 0.8, n_zones)
        else:  # residential
            base_temp = 23.0
            temp_variation = np.random.normal(0, 1.2, n_zones)
        
        zone_temps = np.clip(base_temp + temp_variation, 18.0, 28.0)
        
        # Different occupancy patterns
        if "office" in building.building_id:
            occupancy = np.array([0.8, 0.9, 0.6, 0.1])[:n_zones]
        elif "retail" in building.building_id:
            occupancy = np.array([0.7, 0.2, 0.3])[:n_zones]
        else:  # residential
            occupancy = np.array([0.5, 0.6, 0.8])[:n_zones]
        
        # Current HVAC power usage
        hvac_power = np.random.uniform(2.0, 8.0, n_zones)
        
        state = BuildingState(
            timestamp=time.time(),
            zone_temperatures=zone_temps,
            outside_temperature=8.0,  # Cold winter day
            humidity=45.0,
            occupancy=occupancy,
            hvac_power=hvac_power,
            control_setpoints=np.full(n_zones, 0.6)
        )
        
        states.append(state)
    
    return states


async def microgrid_optimization_demo():
    """Complete microgrid optimization demonstration."""
    print("🏢🏪🏠 Quantum Microgrid Optimization Demo")
    print("=" * 50)
    
    # 1. Create Buildings
    print("\n1. Creating microgrid buildings...")
    buildings = await create_sample_buildings()
    
    for building in buildings:
        print(f"   {building.building_id}: {len(building.zones)} zones")
    
    # 2. Configure Microgrid
    print("\n2. Configuring microgrid controller...")
    
    microgrid_config = MicroGridConfig(
        solar_capacity_kw=200.0,        # 200 kW solar array
        battery_capacity_kwh=500.0,     # 500 kWh battery storage
        grid_connection_limit_kw=400.0, # 400 kW grid connection
        enable_peer_trading=True,       # Enable P2P energy trading
        coordination_interval=300       # 5-minute coordination
    )
    
    optimization_config = OptimizationConfig(
        prediction_horizon=12,          # 12 hours ahead
        control_interval=15,            # 15-minute intervals
        solver="classical_fallback",    # Use fallback for demo
        num_reads=100
    )
    
    objectives = ControlObjectives(
        energy_cost=0.4,  # 40% cost optimization
        comfort=0.4,      # 40% comfort optimization
        carbon=0.2        # 20% carbon reduction
    )
    
    microgrid = MicroGridController(
        buildings=buildings,
        config=microgrid_config,
        individual_configs=[optimization_config] * len(buildings)
    )
    
    print(f"   Solar capacity: {microgrid_config.solar_capacity_kw} kW")
    print(f"   Battery capacity: {microgrid_config.battery_capacity_kwh} kWh")
    print(f"   Grid connection: {microgrid_config.grid_connection_limit_kw} kW")
    print(f"   Peer trading: {'Enabled' if microgrid_config.enable_peer_trading else 'Disabled'}")
    
    # 3. Generate Forecasts
    print("\n3. Generating forecast data...")
    
    # Weather forecast (temperature, solar radiation, humidity)
    weather_forecast = np.array([
        [8.0 + 3*np.sin(i/6), 300 + 400*np.sin(i/8), 50 + 15*np.cos(i/10)]
        for i in range(48)  # 12 hours * 4 intervals
    ])
    
    energy_prices = generate_dynamic_pricing(12)
    solar_generation = generate_solar_forecast(12)
    
    print(f"   Weather forecast: 12 hours")
    print(f"   Temperature range: {weather_forecast[:, 0].min():.1f} - {weather_forecast[:, 0].max():.1f} °C")
    print(f"   Energy price range: ${energy_prices.min():.3f} - ${energy_prices.max():.3f}/kWh")
    print(f"   Solar generation: {solar_generation.max():.1f} kW peak")
    
    # 4. Get Current Building States
    print("\n4. Reading current building states...")
    
    building_states = await create_building_states(buildings)
    
    for i, (building, state) in enumerate(zip(buildings, building_states)):
        avg_temp = np.mean(state.zone_temperatures)
        total_power = np.sum(state.hvac_power)
        avg_occupancy = np.mean(state.occupancy)
        
        print(f"   {building.building_id}:")
        print(f"     Avg temperature: {avg_temp:.1f}°C")
        print(f"     Total HVAC power: {total_power:.1f} kW")
        print(f"     Avg occupancy: {avg_occupancy:.1%}")
    
    # 5. Run Quantum Microgrid Optimization
    print("\n5. Running quantum microgrid optimization...")
    print("   This may take a few moments...")
    
    start_time = time.time()
    
    try:
        optimization_result = await microgrid.optimize_quantum(
            building_states=building_states,
            weather_forecast=weather_forecast,
            energy_prices=energy_prices,
            solar_generation=solar_generation,
            config=optimization_config,
            objectives=objectives
        )
        
        optimization_time = time.time() - start_time
        print(f"   ✅ Optimization completed in {optimization_time:.2f}s")
        
        # 6. Analyze Results
        print("\n6. Optimization Results")
        print("=" * 30)
        
        energy_balance = optimization_result['energy_balance']
        print(f"Energy Balance (12-hour period):")
        print(f"  Total demand: {energy_balance['total_demand_kwh']:.1f} kWh")
        print(f"  Solar generation: {energy_balance['solar_generation_kwh']:.1f} kWh")
        print(f"  Grid import: {energy_balance['grid_import_kwh']:.1f} kWh")
        print(f"  Battery throughput: {energy_balance['battery_throughput_kwh']:.1f} kWh")
        
        # Solar utilization
        solar_utilization = optimization_result['solar_utilization']
        solar_efficiency = np.sum(solar_utilization) / np.sum(solar_generation) * 100
        print(f"  Solar utilization: {solar_efficiency:.1f}%")
        
        # Building demand breakdown
        building_demands = optimization_result['building_demands']
        print(f"\nBuilding Energy Demand:")
        for i, (building, demand) in enumerate(zip(buildings, building_demands)):
            total_demand = np.sum(demand) * 0.25  # Convert to kWh
            peak_demand = np.max(demand)
            print(f"  {building.building_id}:")
            print(f"    Total: {total_demand:.1f} kWh")
            print(f"    Peak: {peak_demand:.1f} kW")
        
        # Battery schedule analysis
        battery_schedule = optimization_result['battery_schedule']
        charging_periods = len([x for x in battery_schedule if x > 0])
        discharging_periods = len([x for x in battery_schedule if x < 0])
        
        print(f"\nBattery Operation:")
        print(f"  Charging periods: {charging_periods}")
        print(f"  Discharging periods: {discharging_periods}")
        print(f"  Peak charging: {max(battery_schedule):.1f} kW")
        print(f"  Peak discharging: {abs(min(battery_schedule)):.1f} kW")
        
        # Cost Analysis
        total_grid_cost = np.sum(optimization_result['grid_interaction'] * energy_prices * 0.25)
        baseline_cost = np.sum(optimization_result['total_demand'] * energy_prices * 0.25)
        cost_savings = baseline_cost - total_grid_cost
        savings_percentage = cost_savings / baseline_cost * 100 if baseline_cost > 0 else 0
        
        print(f"\nCost Analysis:")
        print(f"  Baseline cost (no optimization): ${baseline_cost:.2f}")
        print(f"  Optimized cost: ${total_grid_cost:.2f}")
        print(f"  Cost savings: ${cost_savings:.2f} ({savings_percentage:.1f}%)")
        
        # Peer-to-peer trading
        if microgrid_config.enable_peer_trading:
            peer_trading = optimization_result['peer_trading']
            total_trading_volume = sum(
                step_data['total_trading_volume'] 
                for step_data in peer_trading.values()
            ) * 0.25  # Convert to kWh
            
            print(f"\nPeer-to-Peer Trading:")
            print(f"  Total trading volume: {total_trading_volume:.1f} kWh")
            print(f"  Trading efficiency: {total_trading_volume/energy_balance['total_demand_kwh']*100:.1f}%")
        
        # 7. Control Schedule Preview
        print("\n7. Control Schedule Preview (First 4 intervals)")
        print("=" * 60)
        
        hvac_schedules = optimization_result['hvac_schedules']
        
        for building_idx, (building, schedule) in enumerate(zip(buildings, hvac_schedules)):
            n_zones = len(building.zones)
            n_steps = len(schedule) // n_zones
            schedule_2d = schedule.reshape((n_steps, n_zones))
            
            print(f"\n{building.building_id}:")
            zone_names = [zone.zone_id for zone in building.zones]
            header = "Time | " + " | ".join(f"{name:>8s}" for name in zone_names)
            print(header)
            print("-" * len(header))
            
            for step in range(min(4, n_steps)):
                values = " | ".join(f"{schedule_2d[step, i]:8.3f}" for i in range(n_zones))
                print(f"{step+1:4d} | {values}")
        
        # 8. System Status
        print("\n8. Microgrid System Status")
        print("=" * 30)
        
        status = microgrid.get_status()
        print(f"Buildings: {status['n_buildings']}")
        print(f"Battery SOC: {status['battery_soc']:.1%}")
        print(f"Solar generation: {status['solar_generation']:.1f} kW")
        print(f"Grid import: {status['grid_import']:.1f} kW")
        
        print(f"\nController Status:")
        for i, controller_status in enumerate(status['controller_status']):
            building_id = controller_status['building_id']
            solver_status = controller_status['quantum_solver_status']
            print(f"  {building_id}:")
            print(f"    Solver available: {solver_status['is_available']}")
            print(f"    Solver type: {solver_status['solver_type']}")
            print(f"    History length: {controller_status['history_length']}")
        
        print(f"\n🎉 Quantum microgrid optimization complete!")
        return True
        
    except Exception as e:
        print(f"   ❌ Optimization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("Starting quantum microgrid optimization demo...")
    
    try:
        success = asyncio.run(microgrid_optimization_demo())
        if success:
            print("\n✅ Demo completed successfully!")
            print("\nKey achievements:")
            print("  • Multi-building quantum optimization")
            print("  • Solar generation integration") 
            print("  • Battery storage optimization")
            print("  • Energy cost minimization")
            print("  • Peer-to-peer energy trading")
            print("  • Real-time control coordination")
        else:
            print("\n❌ Demo failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted by user")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)
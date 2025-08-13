#!/usr/bin/env python3
"""
Simple HVAC optimization demo with enhanced error handling and monitoring.
"""

import asyncio
import numpy as np
import logging
from pathlib import Path
import sys

# Add quantum_ctl to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from quantum_ctl import HVACController, Building
from quantum_ctl.core.controller import OptimizationConfig, ControlObjectives
from quantum_ctl.models.building import BuildingState, ZoneConfig

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_simple_building():
    """Create a simple 1-zone building for demo."""
    zones = [
        ZoneConfig(
            zone_id="main_zone",
            area=100.0,
            volume=300.0,
            thermal_mass=250.0,
            max_heating_power=15.0,
            max_cooling_power=12.0,
            comfort_temp_min=20.0,
            comfort_temp_max=24.0
        )
    ]
    
    return Building(
        building_id="simple_demo",
        zones=zones,
        occupancy_schedule="office_standard"
    )

async def simple_demo():
    """Run simple HVAC demo."""
    print("🏢 Simple Quantum HVAC Demo")
    print("=" * 30)
    
    try:
        # Create building
        building = create_simple_building()
        logger.info(f"Created building: {building.building_id}")
        
        # Configure optimization
        config = OptimizationConfig(
            prediction_horizon=2,  # 2 hours for simplicity
            control_interval=30,   # 30-minute intervals
            solver="classical_fallback",
            num_reads=50
        )
        
        objectives = ControlObjectives(
            energy_cost=0.7,
            comfort=0.3,
            carbon=0.0
        )
        
        # Create controller
        controller = HVACController(building, config, objectives)
        logger.info("Controller initialized successfully")
        
        # Current state
        current_state = BuildingState(
            timestamp=0.0,
            zone_temperatures=np.array([22.0]),
            outside_temperature=15.0,
            humidity=50.0,
            occupancy=np.array([0.8]),
            hvac_power=np.array([10.0]),
            control_setpoints=np.array([0.5])
        )
        
        # Simple forecast
        n_steps = config.prediction_horizon * 2  # 30-min intervals
        weather_forecast = np.array([
            [15.0 + i * 0.5, 250.0, 50.0] for i in range(n_steps)
        ])
        energy_prices = np.array([0.12] * n_steps)
        
        print(f"\n📊 Current State:")
        print(f"   Temperature: {current_state.zone_temperatures[0]:.1f}°C")
        print(f"   Outside: {current_state.outside_temperature:.1f}°C")
        print(f"   Occupancy: {current_state.occupancy[0]:.1%}")
        
        # Run optimization
        print(f"\n⚡ Running optimization...")
        control_schedule = await controller.optimize(
            current_state=current_state,
            weather_forecast=weather_forecast,
            energy_prices=energy_prices
        )
        
        print(f"✅ Optimization complete!")
        print(f"   Schedule length: {len(control_schedule)}")
        
        # Apply controls
        controller.apply_schedule(control_schedule)
        
        # Show results
        status = controller.get_status()
        print(f"\n📈 Results:")
        print(f"   Building: {status['building_id']}")
        print(f"   Solver: {status['quantum_solver_status']['solver_type']}")
        
        return True
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(simple_demo())
    sys.exit(0 if success else 1)
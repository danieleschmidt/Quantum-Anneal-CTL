#!/usr/bin/env python3
"""
Complete quantum HVAC control system demonstration.

Shows end-to-end functionality with error handling and safety monitoring.
"""

import asyncio
import logging
import numpy as np
import sys
import time
from pathlib import Path

# Add quantum_ctl to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from quantum_ctl.core.controller import HVACController
from quantum_ctl.models.building import Building, ZoneConfig, BuildingState
from quantum_ctl.utils.safety import SafetyLimits
from quantum_ctl.utils.performance import get_resource_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


class MockDataSource:
    """Mock data source for demonstration."""
    
    def __init__(self):
        self.time_step = 0
    
    async def get_current_state(self) -> BuildingState:
        """Get current building state."""
        # Simulate realistic building state
        n_zones = 5
        base_temp = 22.0
        temp_variation = np.random.normal(0, 1.0, n_zones)
        
        return BuildingState(
            timestamp=time.time() + self.time_step * 900,  # 15-min intervals
            zone_temperatures=base_temp + temp_variation,
            outside_temperature=15.0 + 5 * np.sin(self.time_step * 0.1),  # Varying outdoor temp
            humidity=50.0 + 10 * np.random.random(),
            occupancy=np.random.uniform(0.2, 0.8, n_zones),
            hvac_power=np.random.uniform(0, 10, n_zones),
            control_setpoints=np.full(n_zones, 0.5)
        )
    
    async def get_weather_forecast(self) -> np.ndarray:
        """Get weather forecast."""
        horizon = 24  # 24 time steps (6 hours with 15-min intervals)
        
        # Generate realistic weather forecast
        forecast = []
        for i in range(horizon):
            temp = 15.0 + 8 * np.sin((self.time_step + i) * 0.05)  # Daily cycle
            solar = max(0, 800 * np.sin((self.time_step + i) * 0.02))  # Solar cycle
            humidity = 50 + 20 * np.random.random()
            
            forecast.append([temp, solar, humidity])
        
        return np.array(forecast)
    
    async def get_energy_prices(self) -> np.ndarray:
        """Get energy price forecast."""
        horizon = 24
        
        # Generate time-of-use pricing
        prices = []
        for i in range(horizon):
            hour = ((self.time_step + i) * 0.25) % 24  # Convert to hour of day
            
            if 9 <= hour <= 17:  # Peak hours
                price = 0.18 + 0.05 * np.random.random()
            elif 18 <= hour <= 21:  # Evening peak
                price = 0.22 + 0.05 * np.random.random()
            else:  # Off-peak
                price = 0.12 + 0.03 * np.random.random()
            
            prices.append(price)
        
        return np.array(prices)
    
    def advance_time(self):
        """Advance simulation time."""
        self.time_step += 1


async def demonstrate_quantum_hvac():
    """Demonstrate complete quantum HVAC control system."""
    
    logger.info("🏢 Initializing Quantum HVAC Control System")
    
    # Create building configuration
    zones = [
        ZoneConfig("office_north", area=150, volume=450, thermal_mass=2000, 
                  max_heating_power=12, max_cooling_power=10),
        ZoneConfig("office_south", area=150, volume=450, thermal_mass=2000,
                  max_heating_power=12, max_cooling_power=10),  
        ZoneConfig("conference", area=100, volume=300, thermal_mass=1500,
                  max_heating_power=8, max_cooling_power=6),
        ZoneConfig("lobby", area=200, volume=600, thermal_mass=2500,
                  max_heating_power=15, max_cooling_power=12),
        ZoneConfig("server_room", area=50, volume=150, thermal_mass=800,
                  max_heating_power=5, max_cooling_power=15)  # High cooling need
    ]
    
    building = Building(
        building_id="demo_building",
        zones=zones,
        envelope_ua=600.0,
        latitude=40.7128,  # NYC
        longitude=-74.0060
    )
    
    # Create controller with custom safety limits
    safety_limits = SafetyLimits(
        min_zone_temp=18.0,
        max_zone_temp=26.0,
        max_humidity=75.0,
        max_power_per_zone=20.0
    )
    
    controller = HVACController(building=building)
    logger.info(f"✅ Controller initialized for {building.n_zones} zones")
    
    # Create data source
    data_source = MockDataSource()
    
    # Performance monitoring
    resource_manager = get_resource_manager()
    
    # Run optimization cycles
    for cycle in range(10):
        logger.info(f"🔄 Optimization Cycle {cycle + 1}/10")
        
        try:
            # Get current data
            start_time = time.time()
            current_state = await data_source.get_current_state()
            weather_forecast = await data_source.get_weather_forecast()
            energy_prices = await data_source.get_energy_prices()
            
            data_time = time.time() - start_time
            logger.info(f"📊 Data collection: {data_time:.3f}s")
            
            # Log current conditions
            logger.info(f"🌡️  Zone temperatures: {current_state.zone_temperatures.round(1)}")
            logger.info(f"🌤️  Outside temperature: {current_state.outside_temperature:.1f}°C")
            logger.info(f"💧 Humidity: {current_state.humidity:.1f}%")
            logger.info(f"💰 Current energy price: ${energy_prices[0]:.3f}/kWh")
            
            # Run optimization
            opt_start = time.time()
            control_schedule = await controller.optimize(
                current_state, weather_forecast, energy_prices
            )
            opt_time = time.time() - opt_start
            
            logger.info(f"⚡ Optimization completed in {opt_time:.3f}s")
            
            # Apply control
            controller.apply_schedule(control_schedule)
            
            # Log results
            next_controls = control_schedule[:building.n_zones]
            logger.info(f"🎛️  Next control actions: {next_controls.round(3)}")
            
            # Safety and performance status
            status = controller.get_status()
            safety_level = status['safety_status']['level']
            performance = resource_manager.get_performance_summary()
            
            logger.info(f"🛡️  Safety level: {safety_level}")
            logger.info(f"📈 Avg optimization time: {performance['performance']['avg_optimization_time']:.3f}s")
            
            # Check for issues
            if safety_level != 'normal':
                logger.warning(f"⚠️ Safety issue detected: {status['safety_status']['violations']}")
            
            # Simulate time advancement
            data_source.advance_time()
            
            # Brief pause between cycles
            await asyncio.sleep(1)
            
        except Exception as e:
            logger.error(f"❌ Cycle {cycle + 1} failed: {e}")
            continue
    
    # Final performance report
    logger.info("📊 Final Performance Report")
    final_status = controller.get_status()
    performance_summary = resource_manager.get_performance_summary()
    
    logger.info(f"✅ Total optimizations: {performance_summary['performance'].get('avg_optimization_time', 0)}")
    logger.info(f"🏢 Building status: {final_status['safety_status']['level']}")
    logger.info(f"🧮 Quantum solver: {final_status['quantum_solver_status']['solver_type']}")
    logger.info(f"💾 Cache performance: {performance_summary.get('optimization_cache', {}).get('hit_rate', 0):.3f}")
    
    logger.info("🎉 Quantum HVAC Control System demonstration completed successfully!")


async def demonstrate_emergency_response():
    """Demonstrate emergency response capabilities."""
    
    logger.info("🚨 Testing Emergency Response System")
    
    # Create simple building for emergency test
    building = Building(building_id="emergency_test", zones=2)
    controller = HVACController(building=building)
    
    # Create emergency state (very high temperature)
    emergency_state = BuildingState(
        timestamp=time.time(),
        zone_temperatures=np.array([35.0, 40.0]),  # Dangerously high
        outside_temperature=45.0,  # Heat wave
        humidity=90.0,  # Very humid
        occupancy=np.array([0.8, 0.9]),  # High occupancy
        hvac_power=np.array([25.0, 30.0]),  # Over limit
        control_setpoints=np.array([0.9, 1.0])
    )
    
    # Test safety monitoring
    safety_level = controller._safety_monitor.check_safety(emergency_state)
    logger.info(f"🛡️  Safety level: {safety_level.value}")
    logger.info(f"⚠️  Violations: {controller._safety_monitor.safety_violations}")
    
    # Emergency control should be activated
    if safety_level.value in ['critical', 'emergency']:
        emergency_schedule = await controller._emergency_control(emergency_state)
        logger.info(f"🚨 Emergency control activated")
        logger.info(f"🎛️  Emergency controls: {emergency_schedule[:2].round(3)}")
    
    logger.info("✅ Emergency response test completed")


async def main():
    """Main demonstration function."""
    
    print("=" * 60)
    print("🌟 QUANTUM-INSPIRED HVAC CONTROL SYSTEM DEMO")
    print("=" * 60)
    
    try:
        # Main demonstration
        await demonstrate_quantum_hvac()
        
        print("\n" + "=" * 60)
        
        # Emergency response demo  
        await demonstrate_emergency_response()
        
        print("\n" + "=" * 60)
        print("✅ All demonstrations completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        logger.error(f"❌ Demonstration failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
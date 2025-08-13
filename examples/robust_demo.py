#!/usr/bin/env python3
"""
Robust HVAC optimization demo with comprehensive error handling and monitoring.
"""

import asyncio
import numpy as np
import logging
import sys
from pathlib import Path
import traceback

# Add quantum_ctl to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from quantum_ctl import HVACController, Building
from quantum_ctl.core.controller import OptimizationConfig, ControlObjectives
from quantum_ctl.models.building import BuildingState, ZoneConfig
from quantum_ctl.utils.config_validator import SystemValidator, validate_system
from quantum_ctl.utils.health_dashboard import get_health_dashboard

# Setup comprehensive logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('hvac_demo.log')
    ]
)
logger = logging.getLogger(__name__)

def create_robust_building():
    """Create a robust multi-zone building with validation."""
    zones = [
        ZoneConfig(
            zone_id="lobby",
            area=150.0,
            volume=450.0,
            thermal_mass=300.0,
            max_heating_power=20.0,
            max_cooling_power=18.0,
            comfort_temp_min=20.0,
            comfort_temp_max=25.0
        ),
        ZoneConfig(
            zone_id="office_east",
            area=120.0,
            volume=360.0,
            thermal_mass=240.0,
            max_heating_power=15.0,
            max_cooling_power=12.0,
            comfort_temp_min=21.0,
            comfort_temp_max=24.0
        ),
        ZoneConfig(
            zone_id="office_west",
            area=110.0,
            volume=330.0,
            thermal_mass=220.0,
            max_heating_power=14.0,
            max_cooling_power=11.0,
            comfort_temp_min=21.0,
            comfort_temp_max=24.0
        )
    ]
    
    return Building(
        building_id="robust_demo_building",
        zones=zones,
        occupancy_schedule="office_standard"
    )

async def robust_demo():
    """Run robust HVAC demo with comprehensive error handling."""
    print("🏢 Robust Quantum HVAC Demo")
    print("=" * 35)
    
    # Get health dashboard
    dashboard = get_health_dashboard()
    
    try:
        # 1. System Validation
        print("\n🔍 System Validation...")
        is_valid, validation_results = validate_system()
        
        if not is_valid:
            logger.error("System validation failed")
            for error in validation_results['environment'].errors:
                print(f"   ❌ {error}")
            return False
        
        for warning in validation_results['environment'].warnings:
            print(f"   ⚠️  {warning}")
        
        for rec in validation_results['environment'].recommendations:
            print(f"   💡 {rec}")
        
        print("   ✅ System validation passed")
        
        # 2. Building Configuration
        print("\n🏗️  Building Configuration...")
        building = create_robust_building()
        
        # Validate building
        validator = SystemValidator()
        building_result = validator.validate_building_config(building)
        
        if not building_result.is_valid:
            logger.error("Building validation failed")
            for error in building_result.errors:
                print(f"   ❌ {error}")
            return False
        
        print(f"   ✅ Building validated: {building.building_id}")
        print(f"   📊 Zones: {building.n_zones}")
        
        # 3. Optimization Configuration
        print("\n⚙️  Optimization Configuration...")
        config = OptimizationConfig(
            prediction_horizon=4,
            control_interval=15,
            solver="classical_fallback",
            num_reads=100
        )
        
        # Validate config
        config_result = validator.validate_optimization_config(config)
        if not config_result.is_valid:
            logger.error("Configuration validation failed")
            for error in config_result.errors:
                print(f"   ❌ {error}")
            return False
        
        for warning in config_result.warnings:
            print(f"   ⚠️  {warning}")
        
        objectives = ControlObjectives(
            energy_cost=0.6,
            comfort=0.35,
            carbon=0.05
        )
        
        print(f"   ✅ Configuration validated")
        print(f"   📈 Horizon: {config.prediction_horizon}h, Interval: {config.control_interval}min")
        
        # 4. Controller Initialization with Error Handling
        print("\n🎮 Controller Initialization...")
        
        try:
            controller = HVACController(building, config, objectives)
            logger.info("Controller initialized successfully")
            print("   ✅ Controller ready")
        except Exception as e:
            logger.error(f"Controller initialization failed: {e}")
            dashboard.record_error(f"Controller init failed: {str(e)}")
            print(f"   ❌ Controller failed: {e}")
            return False
        
        # 5. Building State with Validation
        print("\n📊 Building State Setup...")
        current_state = BuildingState(
            timestamp=0.0,
            zone_temperatures=np.array([23.0, 22.5, 22.0]),
            outside_temperature=8.0,
            humidity=55.0,
            occupancy=np.array([0.2, 0.8, 0.7]),  # Morning arrival
            hvac_power=np.array([12.0, 10.0, 9.0]),
            control_setpoints=np.array([0.6, 0.7, 0.65])
        )
        
        # Validate state
        state_result = validator.validate_state_data(current_state, building)
        if not state_result.is_valid:
            logger.error("State validation failed")
            for error in state_result.errors:
                print(f"   ❌ {error}")
            return False
        
        for warning in state_result.warnings:
            print(f"   ⚠️  {warning}")
        
        print("   ✅ State validated")
        print(f"   🌡️  Temperatures: {current_state.zone_temperatures}")
        print(f"   👥 Occupancy: {current_state.occupancy}")
        
        # 6. Forecast Generation
        print("\n🔮 Forecast Generation...")
        n_steps = config.prediction_horizon * 4  # 15-min intervals
        
        # Weather forecast with realistic variation
        weather_forecast = np.array([
            [
                8.0 + 3.0 * np.sin(i * np.pi / 12),  # Temperature cycle
                150 + 400 * max(0, np.sin(i * np.pi / 24)),  # Solar radiation
                55.0 + 10 * np.cos(i * np.pi / 8)  # Humidity variation
            ]
            for i in range(n_steps)
        ])
        
        # Energy prices with peak/off-peak structure
        base_price = 0.12
        energy_prices = np.array([
            base_price * (1.8 if 8 <= (i // 4) % 24 <= 20 else 0.8)
            for i in range(n_steps)
        ])
        
        print(f"   ✅ Forecast generated: {n_steps} steps")
        print(f"   🌡️  Temp range: {weather_forecast[:, 0].min():.1f}-{weather_forecast[:, 0].max():.1f}°C")
        print(f"   💰 Price range: ${energy_prices.min():.3f}-${energy_prices.max():.3f}/kWh")
        
        # 7. Optimization with Robust Error Handling
        print("\n⚡ Running Optimization...")
        optimization_start = asyncio.get_event_loop().time()
        
        try:
            control_schedule = await controller.optimize(
                current_state=current_state,
                weather_forecast=weather_forecast,
                energy_prices=energy_prices
            )
            
            optimization_time = asyncio.get_event_loop().time() - optimization_start
            dashboard.record_optimization(True, optimization_time)
            
            print(f"   ✅ Optimization successful in {optimization_time:.2f}s")
            print(f"   📋 Schedule: {len(control_schedule)} control values")
            
            # Validate control schedule
            if len(control_schedule) != n_steps * building.get_control_dimension():
                logger.warning(f"Unexpected schedule length: {len(control_schedule)}")
                dashboard.record_error("Schedule length mismatch")
            
            # Check control bounds
            if np.any(control_schedule < 0) or np.any(control_schedule > 1):
                logger.warning("Control values outside [0,1] bounds")
                dashboard.record_error("Control bounds violation")
            
        except Exception as e:
            optimization_time = asyncio.get_event_loop().time() - optimization_start
            dashboard.record_optimization(False, optimization_time, str(e))
            
            logger.error(f"Optimization failed: {e}")
            print(f"   ❌ Optimization failed: {e}")
            return False
        
        # 8. Apply Controls Safely
        print("\n🎛️  Applying Controls...")
        try:
            controller.apply_schedule(control_schedule)
            print("   ✅ Controls applied successfully")
        except Exception as e:
            logger.error(f"Control application failed: {e}")
            dashboard.record_error(f"Control application failed: {str(e)}")
            print(f"   ❌ Control application failed: {e}")
        
        # 9. Health Report
        print("\n📊 System Health Report")
        print("=" * 25)
        
        health_report = dashboard.get_health_report()
        current_metrics = health_report['current_metrics']
        
        print(f"Status: {current_metrics['system_status'].upper()}")
        print(f"Success Rate: {current_metrics['optimization_success_rate']:.1%}")
        print(f"Avg Optimization Time: {current_metrics['avg_optimization_time']:.2f}s")
        print(f"Memory Usage: {current_metrics['memory_usage_mb']:.1f} MB")
        print(f"CPU Usage: {current_metrics['cpu_usage_percent']:.1f}%")
        print(f"Quantum Status: {current_metrics['quantum_solver_status']}")
        print(f"Errors (1h): {current_metrics['error_count_last_hour']}")
        
        if current_metrics['last_error']:
            print(f"Last Error: {current_metrics['last_error']}")
        
        # 10. Performance Summary
        print("\n🎯 Performance Summary")
        print("=" * 22)
        
        controller_status = controller.get_status()
        print(f"Building: {controller_status['building_id']}")
        print(f"Solver: {controller_status['quantum_solver_status']['solver_type']}")
        print(f"Total Optimizations: {health_report['total_optimizations']}")
        
        # Simple energy estimation
        baseline_power = np.sum(current_state.hvac_power)
        avg_control = np.mean(control_schedule)
        max_power = sum(max(z.max_heating_power, z.max_cooling_power) for z in building.zones)
        estimated_power = avg_control * max_power
        
        savings_percent = max(0, (baseline_power - estimated_power) / baseline_power * 100)
        cost_savings = savings_percent / 100 * np.mean(energy_prices) * baseline_power * config.prediction_horizon
        
        print(f"Estimated Energy Savings: {savings_percent:.1f}%")
        print(f"Estimated Cost Savings: ${cost_savings:.2f}")
        
        print(f"\n🎉 Robust demo completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Demo failed with unexpected error: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        dashboard.record_error(f"Demo failure: {str(e)}")
        print(f"\n💥 Demo failed: {e}")
        return False
    
    finally:
        # Export health metrics
        try:
            dashboard.export_metrics("demo_health_metrics.json")
            print("📄 Health metrics exported to demo_health_metrics.json")
        except Exception as e:
            logger.warning(f"Failed to export metrics: {e}")

if __name__ == "__main__":
    success = asyncio.run(robust_demo())
    sys.exit(0 if success else 1)
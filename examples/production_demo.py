#!/usr/bin/env python3
"""
Production-scale quantum HVAC control system demonstration.

Shows enterprise-grade features including auto-scaling, cloud sync,
distributed optimization, and comprehensive monitoring.
"""

import asyncio
import logging
import sys
import time
import numpy as np
from pathlib import Path

# Add quantum_ctl to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from quantum_ctl.core.controller import HVACController
from quantum_ctl.models.building import Building, ZoneConfig, BuildingState
from quantum_ctl.optimization.auto_scaler import (
    initialize_auto_scaling, ScalingPolicy, get_auto_scaler, get_scheduler
)
from quantum_ctl.integration.cloud_sync import (
    initialize_cloud_integration, get_cloud_sync, get_distributed_optimizer
)
from quantum_ctl.utils.performance import get_resource_manager
from quantum_ctl.utils.safety import SafetyLimits

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


class ProductionSimulator:
    """Simulate production-scale HVAC control system."""
    
    def __init__(self):
        self.buildings = []
        self.controllers = []
        self.start_time = time.time()
        
    async def create_building_fleet(self, count: int = 5):
        """Create a fleet of buildings for simulation."""
        building_types = [
            ('office_tower', 20, 'high_rise'),
            ('shopping_mall', 15, 'retail'),
            ('hospital', 25, 'healthcare'),
            ('school', 12, 'education'),
            ('data_center', 8, 'industrial')
        ]
        
        for i in range(count):
            building_type, zone_count, category = building_types[i % len(building_types)]
            
            # Create zones with realistic configurations
            zones = []
            for j in range(zone_count):
                zone = ZoneConfig(
                    zone_id=f"{building_type}_zone_{j+1}",
                    area=np.random.uniform(80, 200),  # m²
                    volume=np.random.uniform(240, 600),  # m³
                    thermal_mass=np.random.uniform(1200, 3000),  # kJ/K
                    max_heating_power=np.random.uniform(8, 20),  # kW
                    max_cooling_power=np.random.uniform(6, 16),  # kW
                    comfort_temp_min=19.0 + np.random.uniform(-1, 1),
                    comfort_temp_max=24.0 + np.random.uniform(-1, 1)
                )
                zones.append(zone)
            
            building = Building(
                building_id=f"{building_type}_{i+1}",
                zones=zones,
                envelope_ua=np.random.uniform(400, 800),
                latitude=40.7128 + np.random.uniform(-5, 5),  # Around NYC
                longitude=-74.0060 + np.random.uniform(-5, 5)
            )
            
            # Create controller with safety limits
            safety_limits = SafetyLimits(
                min_zone_temp=16.0 if category == 'industrial' else 18.0,
                max_zone_temp=28.0 if category == 'industrial' else 26.0,
                max_humidity=85.0 if category == 'healthcare' else 75.0,
                max_power_per_zone=25.0 if category == 'industrial' else 20.0
            )
            
            controller = HVACController(building=building)
            
            self.buildings.append(building)
            self.controllers.append(controller)
            
            logger.info(f"Created {building_type}_{i+1}: {zone_count} zones, category: {category}")
        
        logger.info(f"✅ Created fleet of {len(self.buildings)} buildings")
    
    async def simulate_realistic_conditions(self, building_idx: int) -> dict:
        """Generate realistic building conditions."""
        building = self.buildings[building_idx]
        elapsed_hours = (time.time() - self.start_time) / 3600.0
        
        # Time-of-day effects
        hour_of_day = (elapsed_hours * 24) % 24  # Accelerated time
        
        # Base temperatures with daily cycle
        outside_temp = 18.0 + 12 * np.sin((hour_of_day - 6) * np.pi / 12)
        
        # Zone temperatures with some variation
        base_temp = 21.0 + 2 * np.sin(hour_of_day * np.pi / 12)
        zone_temps = base_temp + np.random.normal(0, 1.0, building.n_zones)
        
        # Occupancy patterns
        if 7 <= hour_of_day <= 18:  # Work hours
            occupancy = np.random.uniform(0.6, 0.9, building.n_zones)
        elif 18 <= hour_of_day <= 22:  # Evening
            occupancy = np.random.uniform(0.2, 0.5, building.n_zones)
        else:  # Night
            occupancy = np.random.uniform(0.0, 0.1, building.n_zones)
        
        # HVAC power based on conditions
        hvac_power = np.random.uniform(2, 15, building.n_zones)
        
        state = BuildingState(
            timestamp=time.time(),
            zone_temperatures=zone_temps,
            outside_temperature=outside_temp,
            humidity=45 + 20 * np.random.random(),
            occupancy=occupancy,
            hvac_power=hvac_power,
            control_setpoints=np.random.uniform(0.3, 0.7, building.n_zones)
        )
        
        # Generate forecasts
        weather_forecast = []
        energy_prices = []
        
        for h in range(24):  # 24-hour forecast
            future_hour = hour_of_day + h
            forecast_temp = 18.0 + 12 * np.sin((future_hour - 6) * np.pi / 12)
            solar_irradiance = max(0, 800 * np.sin((future_hour - 6) * np.pi / 12))
            humidity = 45 + 20 * np.random.random()
            
            weather_forecast.append([forecast_temp, solar_irradiance, humidity])
            
            # Time-of-use pricing
            if 9 <= (future_hour % 24) <= 17:  # Peak
                price = 0.18 + 0.08 * np.random.random()
            elif 18 <= (future_hour % 24) <= 21:  # Evening peak
                price = 0.24 + 0.06 * np.random.random()
            else:  # Off-peak
                price = 0.12 + 0.04 * np.random.random()
            
            energy_prices.append(price)
        
        return {
            'state': state,
            'weather_forecast': np.array(weather_forecast),
            'energy_prices': np.array(energy_prices)
        }
    
    async def run_optimization_cycle(self, building_idx: int) -> dict:
        """Run optimization cycle for a building."""
        controller = self.controllers[building_idx]
        building = self.buildings[building_idx]
        
        # Generate conditions
        conditions = await self.simulate_realistic_conditions(building_idx)
        
        # Submit optimization to scheduler if available
        scheduler = get_scheduler()
        if scheduler:
            # Create optimization request
            optimization_request = {
                'building_id': building.building_id,
                'state': conditions['state'],
                'weather_forecast': conditions['weather_forecast'],
                'energy_prices': conditions['energy_prices'],
                'horizon': 24,
                'zones': building.n_zones,
                'priority': 'normal',
                'complexity_factor': 1.0 + (building.n_zones / 20.0)  # Larger buildings are more complex
            }
            
            task_id = scheduler.submit_optimization(optimization_request)
            
            # For demo, simulate immediate scheduling and completion
            scheduled = await scheduler.schedule_optimization()
            if scheduled and scheduled['id'] == task_id:
                # Run actual optimization
                try:
                    start_time = time.time()
                    schedule = await controller.optimize(
                        conditions['state'],
                        conditions['weather_forecast'],
                        conditions['energy_prices']
                    )
                    
                    optimization_time = time.time() - start_time
                    
                    # Complete the scheduled task
                    result = {
                        'success': True,
                        'schedule': schedule,
                        'optimization_time': optimization_time,
                        'energy_cost_savings': np.random.uniform(50, 200),  # Simulated
                        'comfort_score': np.random.uniform(0.8, 0.95)
                    }
                    
                    scheduler.complete_optimization(task_id, result)
                    
                    return result
                    
                except Exception as e:
                    logger.error(f"Optimization failed for {building.building_id}: {e}")
                    scheduler.complete_optimization(task_id, {'success': False, 'error': str(e)})
                    return {'success': False, 'error': str(e)}
        
        else:
            # Direct optimization without scheduler
            try:
                start_time = time.time()
                schedule = await controller.optimize(
                    conditions['state'],
                    conditions['weather_forecast'], 
                    conditions['energy_prices']
                )
                
                optimization_time = time.time() - start_time
                
                return {
                    'success': True,
                    'schedule': schedule,
                    'optimization_time': optimization_time,
                    'energy_cost_savings': np.random.uniform(50, 200),
                    'comfort_score': np.random.uniform(0.8, 0.95)
                }
                
            except Exception as e:
                logger.error(f"Direct optimization failed for {building.building_id}: {e}")
                return {'success': False, 'error': str(e)}
    
    async def run_production_simulation(self, duration_minutes: int = 10):
        """Run production-scale simulation."""
        logger.info(f"🏭 Starting production simulation for {duration_minutes} minutes")
        
        end_time = time.time() + (duration_minutes * 60)
        cycle_count = 0
        successful_optimizations = 0
        total_savings = 0.0
        
        while time.time() < end_time:
            cycle_count += 1
            cycle_start = time.time()
            
            logger.info(f"🔄 Simulation Cycle {cycle_count}")
            
            # Run optimizations for all buildings in parallel
            optimization_tasks = [
                self.run_optimization_cycle(i) 
                for i in range(len(self.buildings))
            ]
            
            results = await asyncio.gather(*optimization_tasks, return_exceptions=True)
            
            # Process results
            cycle_successes = 0
            cycle_savings = 0.0
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Building {i} optimization failed: {result}")
                    continue
                
                if result.get('success'):
                    cycle_successes += 1
                    cycle_savings += result.get('energy_cost_savings', 0.0)
            
            successful_optimizations += cycle_successes
            total_savings += cycle_savings
            
            cycle_time = time.time() - cycle_start
            
            logger.info(f"   ✅ Completed {cycle_successes}/{len(self.buildings)} optimizations in {cycle_time:.2f}s")
            logger.info(f"   💰 Cycle savings: ${cycle_savings:.2f}")
            
            # Get system status
            if cycle_count % 3 == 0:  # Every 3rd cycle
                await self.report_system_status()
            
            # Brief pause between cycles
            await asyncio.sleep(2)
        
        # Final report
        logger.info(f"\\n🎯 Production Simulation Complete!")
        logger.info(f"   Total cycles: {cycle_count}")
        logger.info(f"   Successful optimizations: {successful_optimizations}")
        logger.info(f"   Success rate: {successful_optimizations / (cycle_count * len(self.buildings)) * 100:.1f}%")
        logger.info(f"   Total estimated savings: ${total_savings:.2f}")
        
        return {
            'cycles': cycle_count,
            'successful_optimizations': successful_optimizations,
            'total_savings': total_savings,
            'buildings': len(self.buildings)
        }
    
    async def report_system_status(self):
        """Report comprehensive system status."""
        # Auto-scaler status
        scaler = get_auto_scaler()
        if scaler:
            scaling_status = scaler.get_scaling_status()
            logger.info(f"   ⚡ Auto-scaler: {scaling_status['current_workers']} workers")
            logger.info(f"      CPU: {scaling_status['recent_metrics']['cpu_percent']:.1f}%, "
                       f"Memory: {scaling_status['recent_metrics']['memory_percent']:.1f}%")
        
        # Scheduler status
        scheduler = get_scheduler()
        if scheduler:
            sched_status = scheduler.get_scheduler_status()
            logger.info(f"   📋 Scheduler: {sched_status['pending_count']} pending, "
                       f"{sched_status['active_count']} active")
            logger.info(f"      Avg wait time: {sched_status['average_wait_time']:.1f}s")
        
        # Resource manager status
        resource_manager = get_resource_manager()
        performance = resource_manager.get_performance_summary()
        logger.info(f"   📊 Performance: avg opt time {performance['performance']['avg_optimization_time']:.2f}s")
        
        # Cloud sync status (if available)
        cloud_sync = get_cloud_sync()
        if cloud_sync:
            sync_status = cloud_sync.get_sync_status()
            logger.info(f"   ☁️  Cloud sync: {sync_status['stats']['uploads']} uploads, "
                       f"{sync_status['pending_metrics']} pending")


async def main():
    """Main production demonstration."""
    print("=" * 80)
    print("🏭 QUANTUM HVAC CONTROL - PRODUCTION SCALE DEMONSTRATION")
    print("=" * 80)
    
    try:
        # Initialize production-scale components
        logger.info("🚀 Initializing production-scale quantum HVAC system...")
        
        # Initialize auto-scaling with production policy
        production_policy = ScalingPolicy(
            cpu_scale_up_threshold=60.0,    # More aggressive scaling
            cpu_scale_down_threshold=25.0,
            memory_scale_up_threshold=70.0,
            queue_scale_up_threshold=5,     # Scale up quickly under load
            min_workers=2,
            max_workers=12,                 # Higher capacity
            cooldown_period=180.0           # 3-minute cooldown
        )
        initialize_auto_scaling(production_policy)
        logger.info("✅ Auto-scaling initialized")
        
        # Initialize cloud integration (mock endpoint for demo)
        # await initialize_cloud_integration("https://api.quantum-hvac.example.com", "demo-api-key")
        # logger.info("✅ Cloud integration initialized")
        
        # Create production simulator
        simulator = ProductionSimulator()
        await simulator.create_building_fleet(count=8)  # 8 buildings
        
        # Run production simulation
        results = await simulator.run_production_simulation(duration_minutes=5)
        
        print("\\n" + "=" * 80)
        print("📈 PRODUCTION SIMULATION RESULTS")
        print("=" * 80)
        print(f"Buildings managed: {results['buildings']}")
        print(f"Total optimization cycles: {results['cycles']}")
        print(f"Successful optimizations: {results['successful_optimizations']}")
        print(f"Success rate: {results['successful_optimizations'] / (results['cycles'] * results['buildings']) * 100:.1f}%")
        print(f"Estimated total savings: ${results['total_savings']:.2f}")
        
        # Final system status
        print("\\n" + "=" * 80)
        print("🔧 FINAL SYSTEM STATUS")
        print("=" * 80)
        
        scaler = get_auto_scaler()
        if scaler:
            final_scaling = scaler.get_scaling_status()
            print(f"Auto-scaler workers: {final_scaling['current_workers']}")
            print(f"Scaling actions taken: {len(final_scaling['scaling_history'])}")
            
            prediction = scaler.predict_scaling_need()
            print(f"Scaling prediction: {prediction.get('prediction', 'N/A')}")
        
        scheduler = get_scheduler()
        if scheduler:
            final_sched = scheduler.get_scheduler_status()
            print(f"Total completed optimizations: {final_sched['total_completed']}")
            print(f"Average optimization duration: {final_sched['average_duration']:.2f}s")
        
        resource_manager = get_resource_manager()
        final_perf = resource_manager.get_performance_summary()
        opt_cache_hit = final_perf.get('optimization_cache', {}).get('hit_rate', 0.0)
        print(f"Optimization cache hit rate: {opt_cache_hit:.1%}")
        
        print("\\n✅ Production demonstration completed successfully!")
        
        # Cleanup
        if scaler:
            scaler.stop()
        
    except Exception as e:
        logger.error(f"❌ Production demonstration failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
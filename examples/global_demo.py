#!/usr/bin/env python3
"""
Global-ready HVAC optimization demo with internationalization and compliance.
"""

import asyncio
import numpy as np
import logging
import sys
import time
from pathlib import Path
import traceback
from typing import Dict, Any

# Add quantum_ctl to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from quantum_ctl import HVACController, Building
from quantum_ctl.core.controller import OptimizationConfig, ControlObjectives
from quantum_ctl.models.building import BuildingState, ZoneConfig
from quantum_ctl.utils.config_validator import SystemValidator, validate_system
from quantum_ctl.utils.health_dashboard import get_health_dashboard
from quantum_ctl.utils.i18n import get_i18n_manager, _
from quantum_ctl.utils.compliance import get_compliance_manager, ComplianceRegulation

# Setup multilingual logging
def setup_multilingual_logging(locale: str = "en"):
    """Setup logging with locale-specific formatting."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f'global_hvac_demo_{locale}.log')
        ]
    )
    return logging.getLogger(__name__)

def create_global_buildings() -> Dict[str, Building]:
    """Create buildings for different regions with localized configurations."""
    buildings = {}
    
    # European Office (GDPR compliance, Celsius, kW)
    eu_zones = [
        ZoneConfig(
            zone_id="eu_office_main",
            area=120.0,
            volume=360.0,
            thermal_mass=280.0,
            max_heating_power=18.0,
            max_cooling_power=14.0,
            comfort_temp_min=20.0,  # EU standard
            comfort_temp_max=24.0
        ),
        ZoneConfig(
            zone_id="eu_meeting_room",
            area=60.0,
            volume=180.0,
            thermal_mass=140.0,
            max_heating_power=9.0,
            max_cooling_power=7.0,
            comfort_temp_min=21.0,
            comfort_temp_max=23.0
        )
    ]
    
    buildings["eu_office"] = Building(
        building_id="european_headquarters",
        zones=eu_zones,
        occupancy_schedule="office_standard"
    )
    
    # US Office (CCPA compliance, mixed units)
    us_zones = [
        ZoneConfig(
            zone_id="us_open_plan",
            area=200.0,
            volume=600.0,
            thermal_mass=400.0,
            max_heating_power=25.0,
            max_cooling_power=20.0,
            comfort_temp_min=68.0,  # Fahrenheit equivalent
            comfort_temp_max=75.0
        ),
        ZoneConfig(
            zone_id="us_executive",
            area=80.0,
            volume=240.0,
            thermal_mass=160.0,
            max_heating_power=12.0,
            max_cooling_power=10.0,
            comfort_temp_min=70.0,
            comfort_temp_max=74.0
        )
    ]
    
    buildings["us_office"] = Building(
        building_id="american_branch",
        zones=us_zones,
        occupancy_schedule="office_standard"
    )
    
    # Asian Office (PDPA compliance, high occupancy)
    asia_zones = [
        ZoneConfig(
            zone_id="asia_workspace",
            area=150.0,
            volume=450.0,
            thermal_mass=320.0,
            max_heating_power=20.0,
            max_cooling_power=25.0,  # More cooling for hot climate
            comfort_temp_min=22.0,
            comfort_temp_max=26.0   # Higher tolerance for humid climate
        )
    ]
    
    buildings["asia_office"] = Building(
        building_id="asian_operations",
        zones=asia_zones,
        occupancy_schedule="office_standard"
    )
    
    return buildings

async def run_localized_optimization(building: Building, locale: str, 
                                   compliance_manager, logger) -> Dict[str, Any]:
    """Run optimization with locale-specific settings and compliance."""
    i18n = get_i18n_manager()
    i18n.set_locale(locale)
    
    logger.info(_(
        "optimization.started", 
        building=building.building_id, 
        locale=locale
    ))
    
    # Configure optimization based on locale
    if locale == "eu" or locale == "de":
        # European standards: strict energy efficiency
        config = OptimizationConfig(
            prediction_horizon=4,
            control_interval=15,
            solver="classical_fallback",
            num_reads=100
        )
        objectives = ControlObjectives(
            energy_cost=0.7,  # High energy cost focus
            comfort=0.25,
            carbon=0.05       # Carbon footprint consideration
        )
        base_temp = 22.0      # Celsius
        outside_temp = 8.0    # Cold European winter
        currency = "EUR"
        
    elif locale == "en":
        # US standards: balanced approach
        config = OptimizationConfig(
            prediction_horizon=6,
            control_interval=30,
            solver="classical_fallback",
            num_reads=75
        )
        objectives = ControlObjectives(
            energy_cost=0.6,
            comfort=0.35,
            carbon=0.05
        )
        base_temp = 72.0      # Fahrenheit equivalent
        outside_temp = 46.0   # Fahrenheit
        currency = "USD"
        
    else:  # Asian markets
        # Asian standards: comfort in hot climates
        config = OptimizationConfig(
            prediction_horizon=3,
            control_interval=20,
            solver="classical_fallback",
            num_reads=50
        )
        objectives = ControlObjectives(
            energy_cost=0.5,
            comfort=0.45,     # Higher comfort priority
            carbon=0.05
        )
        base_temp = 25.0      # Higher for hot climate
        outside_temp = 32.0   # Hot outside
        currency = "SGD"
    
    try:
        # Record data processing for compliance
        compliance_manager.record_data_processing(
            data_type="temperature_data",
            purpose="hvac_optimization",
            legal_basis="legitimate_interest",
            building_id=building.building_id
        )
        
        compliance_manager.record_data_processing(
            data_type="energy_consumption",
            purpose="efficiency_monitoring",
            legal_basis="contractual",
            building_id=building.building_id
        )
        
        # Create controller
        controller = HVACController(building, config, objectives)
        
        # Create current state
        n_zones = building.n_zones
        current_state = BuildingState(
            timestamp=time.time(),
            zone_temperatures=np.array([base_temp + i * 0.5 for i in range(n_zones)]),
            outside_temperature=outside_temp,
            humidity=55.0,
            occupancy=np.array([0.7 + i * 0.1 for i in range(n_zones)]),
            hvac_power=np.array([10.0 + i * 2.0 for i in range(n_zones)]),
            control_setpoints=np.array([0.6 + i * 0.05 for i in range(n_zones)])
        )
        
        # Generate localized forecast
        n_steps = config.prediction_horizon * (60 // config.control_interval)
        weather_forecast = np.array([
            [outside_temp + i * 0.5, 200 + i * 15, 55.0 + i * 2]
            for i in range(n_steps)
        ])
        
        # Localized energy pricing
        if currency == "EUR":
            base_price = 0.25  # EUR per kWh (higher EU prices)
        elif currency == "USD":
            base_price = 0.12  # USD per kWh
        else:
            base_price = 0.18  # SGD per kWh
        
        energy_prices = np.array([base_price] * n_steps)
        
        # Run optimization
        start_time = time.time()
        
        control_schedule = await controller.optimize(
            current_state=current_state,
            weather_forecast=weather_forecast,
            energy_prices=energy_prices
        )
        
        optimization_time = time.time() - start_time
        
        # Apply controls
        controller.apply_schedule(control_schedule)
        
        # Calculate localized results
        baseline_energy = np.sum(current_state.hvac_power) * config.prediction_horizon
        estimated_optimized = np.mean(control_schedule) * baseline_energy * 0.8
        energy_savings = max(0, baseline_energy - estimated_optimized)
        
        cost_savings = energy_savings * np.mean(energy_prices)
        formatted_savings = i18n.format_currency(cost_savings, currency.replace("SGD", "USD"), locale)
        
        logger.info(_(
            "optimization.completed",
            building=building.building_id,
            time=f"{optimization_time:.2f}s"
        ))
        
        return {
            "building_id": building.building_id,
            "locale": locale,
            "success": True,
            "optimization_time": optimization_time,
            "energy_savings_kwh": energy_savings,
            "cost_savings": formatted_savings,
            "currency": currency,
            "localized_results": {
                "base_temperature": f"{base_temp} {i18n.get_text('units.celsius' if locale != 'en' else 'units.fahrenheit')}",
                "outside_temperature": f"{outside_temp} {i18n.get_text('units.celsius' if locale != 'en' else 'units.fahrenheit')}",
                "energy_saved": f"{i18n.format_number(energy_savings, locale)} {i18n.get_text('units.kilowatthour')}",
                "money_saved": formatted_savings
            }
        }
        
    except Exception as e:
        logger.error(f"Optimization failed for {building.building_id} ({locale}): {e}")
        
        compliance_manager._log_audit_event(
            event_type="system_error",
            building_id=building.building_id,
            action="optimization_failure",
            details={"error": str(e), "locale": locale}
        )
        
        return {
            "building_id": building.building_id,
            "locale": locale,
            "success": False,
            "error": str(e)
        }

async def global_demo():
    """Run global-ready HVAC demo with multiple locales and compliance."""
    print("🌍 Global Quantum HVAC Demo")
    print("=" * 30)
    
    # Initialize global components
    dashboard = get_health_dashboard()
    compliance_manager = get_compliance_manager()
    
    # Enable multiple compliance frameworks
    compliance_manager.enable_regulation(ComplianceRegulation.GDPR)
    compliance_manager.enable_regulation(ComplianceRegulation.CCPA)
    compliance_manager.enable_regulation(ComplianceRegulation.PDPA)
    compliance_manager.enable_regulation(ComplianceRegulation.ENERGY_STAR)
    compliance_manager.enable_regulation(ComplianceRegulation.ISO27001)
    
    try:
        # System validation
        print("\n🔍 Global System Validation...")
        is_valid, validation_results = validate_system()
        
        if not is_valid:
            print("   ❌ System validation failed")
            return False
        
        print("   ✅ System validated globally")
        
        # Create global building fleet
        print("\n🏗️  Creating Global Building Fleet...")
        buildings = create_global_buildings()
        
        total_zones = sum(building.n_zones for building in buildings.values())
        print(f"   ✅ Created {len(buildings)} buildings across regions with {total_zones} total zones")
        
        # Define regional configurations
        regions = [
            {"locale": "de", "building": "eu_office", "name": "European Headquarters (GDPR)"},
            {"locale": "en", "building": "us_office", "name": "American Branch (CCPA)"},
            {"locale": "en", "building": "asia_office", "name": "Asian Operations (PDPA)"}  # Using 'en' for Asian demo
        ]
        
        # Run optimizations for each region
        print(f"\n⚡ Running Multi-Regional Optimizations...")
        results = []
        
        for region in regions:
            locale = region["locale"]
            building = buildings[region["building"]]
            logger = setup_multilingual_logging(locale)
            
            print(f"\n   🌐 {region['name']} ({locale.upper()})")
            
            # Record user consent (simulated)
            compliance_manager.record_user_consent(
                user_id=f"facility_manager_{region['building']}",
                purpose="hvac_optimization",
                granted=True,
                consent_text=f"I consent to HVAC optimization data processing for {building.building_id}",
                building_id=building.building_id
            )
            
            result = await run_localized_optimization(
                building, locale, compliance_manager, logger
            )
            results.append(result)
            
            if result["success"]:
                localized = result["localized_results"]
                print(f"      ✅ Optimization: {result['optimization_time']:.2f}s")
                print(f"      💡 Energy Saved: {localized['energy_saved']}")
                print(f"      💰 Cost Saved: {localized['money_saved']}")
                print(f"      🌡️  Base Temp: {localized['base_temperature']}")
            else:
                print(f"      ❌ Failed: {result['error']}")
        
        # Global Performance Summary
        print(f"\n📊 Global Performance Summary")
        print("=" * 30)
        
        successful_results = [r for r in results if r["success"]]
        failed_results = [r for r in results if not r["success"]]
        
        print(f"Regions optimized: {len(successful_results)}/{len(results)}")
        print(f"Global success rate: {len(successful_results)/len(results):.1%}")
        
        if successful_results:
            total_energy_saved = sum(r["energy_savings_kwh"] for r in successful_results)
            avg_optimization_time = sum(r["optimization_time"] for r in successful_results) / len(successful_results)
            
            print(f"Total energy saved: {total_energy_saved:.1f} kWh")
            print(f"Average optimization time: {avg_optimization_time:.2f}s")
            
            # Show localized cost savings
            print(f"\nRegional Cost Savings:")
            for result in successful_results:
                if result["success"]:
                    print(f"  {result['building_id']}: {result['cost_savings']}")
        
        # Compliance Summary
        print(f"\n🛡️  Compliance Summary")
        print("=" * 20)
        
        compliance_status = compliance_manager.get_compliance_status()
        enabled_regs = compliance_status["enabled_regulations"]
        
        print(f"Active Regulations: {', '.join(enabled_regs)}")
        print(f"Data Processing Records: {compliance_status['data_processing_records']}")
        print(f"Audit Log Entries: {compliance_status['audit_log_entries']}")
        print(f"Consent Records: {compliance_status['consent_records']}")
        
        # Multi-language Status Report
        print(f"\n🌐 Multi-Language Status")
        print("=" * 25)
        
        i18n = get_i18n_manager()
        available_locales = i18n.get_available_locales()
        print(f"Supported Languages: {', '.join(available_locales)}")
        
        # Show status in different languages
        for locale in ["en", "es", "de", "fr"]:
            i18n.set_locale(locale)
            status_msg = i18n.get_text("status.healthy")
            energy_msg = i18n.get_text("energy.efficiency")
            print(f"  {locale.upper()}: {status_msg} - {energy_msg}")
        
        # Export compliance data
        print(f"\n📄 Exporting Compliance Data...")
        compliance_manager.cleanup_expired_data()
        compliance_manager.export_compliance_data("global_compliance_report.json")
        
        # Export health metrics
        dashboard.export_metrics("global_health_metrics.json")
        
        print(f"\n🎉 Global demo completed successfully!")
        print(f"📄 Reports exported:")
        print(f"   - global_compliance_report.json")
        print(f"   - global_health_metrics.json")
        
        return True
        
    except Exception as e:
        print(f"\n💥 Global demo failed: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = asyncio.run(global_demo())
    sys.exit(0 if success else 1)
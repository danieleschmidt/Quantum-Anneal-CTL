"""
Micro-grid controller for coordinating multiple buildings.
"""

from typing import List, Dict, Any, Optional
import numpy as np
import asyncio
import logging
from dataclasses import dataclass

from .controller import HVACController, OptimizationConfig, ControlObjectives
from ..models.building import Building, BuildingState


@dataclass
class MicroGridConfig:
    """Configuration for micro-grid coordination."""
    solar_capacity_kw: float = 0.0
    battery_capacity_kwh: float = 0.0
    grid_connection_limit_kw: float = 1000.0
    enable_peer_trading: bool = False
    coordination_interval: int = 300  # seconds


class MicroGridController:
    """
    Micro-grid controller for coordinating multiple buildings.
    
    Manages energy flows between buildings, storage, and grid connection
    using quantum optimization for global optimality.
    """
    
    def __init__(
        self,
        buildings: List[Building],
        solar_capacity_kw: float = 0.0,
        battery_capacity_kwh: float = 0.0,
        grid_connection_limit_kw: float = 1000.0,
        enable_peer_trading: bool = False,
        config: MicroGridConfig = None,
        individual_configs: Optional[List[OptimizationConfig]] = None
    ):
        self.buildings = buildings
        self.solar_capacity_kw = solar_capacity_kw
        self.battery_capacity_kwh = battery_capacity_kwh  
        self.grid_connection_limit_kw = grid_connection_limit_kw
        self.enable_peer_trading = enable_peer_trading
        
        self.config = config or MicroGridConfig(
            solar_capacity_kw=solar_capacity_kw,
            battery_capacity_kwh=battery_capacity_kwh,
            grid_connection_limit_kw=grid_connection_limit_kw,
            enable_peer_trading=enable_peer_trading
        )
        
        self.logger = logging.getLogger(__name__)
        
        # Create individual building controllers
        self.controllers = []
        for i, building in enumerate(buildings):
            building_config = (
                individual_configs[i] if individual_configs and i < len(individual_configs)
                else OptimizationConfig()
            )
            controller = HVACController(building, building_config)
            self.controllers.append(controller)
        
        # Micro-grid state
        self.battery_soc = 0.5  # State of charge (0-1)
        self.solar_generation = 0.0  # Current solar output (kW)
        self.grid_import = 0.0  # Current grid import (kW)
        
    async def optimize_microgrid(
        self,
        building_states: List[BuildingState],
        weather_forecast: np.ndarray,
        energy_prices: np.ndarray,
        solar_forecast: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Optimize entire micro-grid including all buildings and energy flows.
        
        Args:
            building_states: Current states of all buildings
            weather_forecast: Weather predictions
            energy_prices: Energy price forecast
            solar_forecast: Solar generation forecast
            
        Returns:
            Coordinated optimization results
        """
        if len(building_states) != len(self.buildings):
            raise ValueError("Number of states must match number of buildings")
        
        self.logger.info(f"Optimizing micro-grid with {len(self.buildings)} buildings")
        
        # Individual building optimizations
        building_schedules = []
        total_demand = np.zeros(len(energy_prices))
        
        for i, (controller, state) in enumerate(zip(self.controllers, building_states)):
            try:
                schedule = await controller.optimize(
                    state, weather_forecast, energy_prices
                )
                building_schedules.append(schedule)
                
                # Estimate energy demand from control schedule
                # This is simplified - would need building power model
                estimated_demand = np.mean(schedule) * self.buildings[i].zones[0].max_heating_power
                total_demand += estimated_demand
                
            except Exception as e:
                self.logger.error(f"Building {i} optimization failed: {e}")
                # Use fallback schedule
                fallback_schedule = np.full(
                    controller._get_control_steps() * self.buildings[i].get_control_dimension(),
                    0.5
                )
                building_schedules.append(fallback_schedule)
        
        # Optimize energy flows
        energy_plan = self._optimize_energy_flows(
            total_demand, energy_prices, solar_forecast
        )
        
        return {
            'building_schedules': building_schedules,
            'energy_plan': energy_plan,
            'total_demand': total_demand,
            'optimization_timestamp': asyncio.get_event_loop().time()
        }
    
    async def optimize_quantum(
        self,
        building_states: List[BuildingState],
        weather_forecast: np.ndarray,
        energy_prices: np.ndarray,
        solar_generation: np.ndarray,
        config: OptimizationConfig,
        objectives: ControlObjectives
    ) -> Dict[str, Any]:
        """
        Quantum optimization of entire microgrid.
        
        Args:
            building_states: Current states of all buildings
            weather_forecast: Weather predictions
            energy_prices: Energy price forecast
            solar_generation: Solar generation forecast
            config: Optimization configuration
            objectives: Multi-objective weights
            
        Returns:
            Comprehensive optimization results
        """
        self.logger.info(f"Starting quantum microgrid optimization")
        
        # Update individual controller configurations
        for controller in self.controllers:
            controller.config = config
            controller.objectives = objectives
        
        # Optimize each building with microgrid awareness
        hvac_schedules = []
        building_demands = []
        
        for i, (controller, state) in enumerate(zip(self.controllers, building_states)):
            try:
                # Individual building optimization
                schedule = await controller.optimize(
                    state, weather_forecast, energy_prices
                )
                hvac_schedules.append(schedule)
                
                # Estimate building energy demand
                n_zones = len(self.buildings[i].zones)
                n_steps = len(schedule) // n_zones
                schedule_2d = schedule.reshape((n_steps, n_zones))
                
                # Calculate power demand for each time step
                zone_powers = [zone.max_heating_power + zone.max_cooling_power 
                              for zone in self.buildings[i].zones]
                demand_profile = np.array([
                    np.sum(step * zone_powers) for step in schedule_2d
                ])
                building_demands.append(demand_profile)
                
            except Exception as e:
                self.logger.error(f"Building {i} quantum optimization failed: {e}")
                # Fallback schedule
                n_controls = self.buildings[i].get_control_dimension()
                n_steps = config.prediction_horizon * 4  # 15-min intervals
                fallback = np.full(n_steps * n_controls, 0.5)
                hvac_schedules.append(fallback)
                
                # Default demand profile
                avg_power = np.mean([zone.max_heating_power for zone in self.buildings[i].zones])
                building_demands.append(np.full(n_steps, avg_power))
        
        # Aggregate building demand
        total_demand = np.sum(building_demands, axis=0)
        
        # Optimize energy storage and grid interaction
        battery_schedule = self._optimize_battery_schedule(
            total_demand, energy_prices, solar_generation, objectives
        )
        
        # Calculate grid interaction
        net_demand = total_demand - solar_generation
        battery_contribution = np.array(battery_schedule)
        grid_interaction = np.maximum(0, net_demand + battery_contribution)
        
        # Peer-to-peer trading (simplified)
        peer_trading = {}
        if self.enable_peer_trading:
            peer_trading = self._optimize_peer_trading(
                building_demands, solar_generation, energy_prices
            )
        
        return {
            'hvac_schedules': hvac_schedules,
            'battery_schedule': battery_schedule,
            'grid_interaction': grid_interaction,
            'peer_trading': peer_trading,
            'building_demands': building_demands,
            'total_demand': total_demand,
            'solar_utilization': np.minimum(solar_generation, total_demand),
            'energy_balance': {
                'total_demand_kwh': np.sum(total_demand) * 0.25,  # 15-min intervals
                'solar_generation_kwh': np.sum(solar_generation) * 0.25,
                'grid_import_kwh': np.sum(grid_interaction) * 0.25,
                'battery_throughput_kwh': np.sum(np.abs(battery_schedule)) * 0.25
            }
        }
    
    def _optimize_energy_flows(
        self,
        demand_forecast: np.ndarray,
        energy_prices: np.ndarray,
        solar_forecast: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Optimize energy flows between sources."""
        
        if solar_forecast is None:
            solar_forecast = np.zeros_like(demand_forecast)
        
        # Simple energy flow optimization
        # In full implementation, would use quantum optimization
        
        battery_schedule = []
        grid_schedule = []
        solar_schedule = []
        
        current_soc = self.battery_soc
        
        for t, (demand, price, solar) in enumerate(zip(demand_forecast, energy_prices, solar_forecast)):
            # Limit solar to capacity
            available_solar = min(solar, self.config.solar_capacity_kw)
            
            # Net demand after solar
            net_demand = max(0, demand - available_solar)
            
            # Battery decision (simplified)
            if price < np.mean(energy_prices) and current_soc < 0.9:
                # Charge battery during low prices
                battery_power = min(
                    self.config.battery_capacity_kwh * 0.2,  # 20% per hour max charge rate
                    self.config.grid_connection_limit_kw - net_demand
                )
                current_soc = min(1.0, current_soc + battery_power / self.config.battery_capacity_kwh)
            elif price > np.mean(energy_prices) and current_soc > 0.1:
                # Discharge battery during high prices
                battery_power = -min(
                    current_soc * self.config.battery_capacity_kwh,
                    net_demand,
                    self.config.battery_capacity_kwh * 0.2
                )
                current_soc = max(0.0, current_soc + battery_power / self.config.battery_capacity_kwh)
            else:
                battery_power = 0.0
            
            # Grid import to meet remaining demand
            grid_import = max(0, net_demand + battery_power)
            grid_import = min(grid_import, self.config.grid_connection_limit_kw)
            
            battery_schedule.append(battery_power)
            grid_schedule.append(grid_import)
            solar_schedule.append(available_solar)
        
        return {
            'battery_schedule': np.array(battery_schedule),
            'grid_schedule': np.array(grid_schedule),
            'solar_schedule': np.array(solar_schedule),
            'final_battery_soc': current_soc
        }
    
    def _optimize_battery_schedule(
        self,
        demand_profile: np.ndarray,
        energy_prices: np.ndarray,
        solar_generation: np.ndarray,
        objectives: ControlObjectives
    ) -> List[float]:
        """Optimize battery charging/discharging schedule."""
        
        battery_schedule = []
        current_soc = self.battery_soc
        
        # Price thresholds for battery operation
        price_mean = np.mean(energy_prices)
        price_std = np.std(energy_prices)
        charge_threshold = price_mean - 0.5 * price_std
        discharge_threshold = price_mean + 0.5 * price_std
        
        for t, (demand, price, solar) in enumerate(zip(demand_profile, energy_prices, solar_generation)):
            net_demand = demand - solar
            
            # Battery operation decision
            if price < charge_threshold and current_soc < 0.9 and net_demand < 0:
                # Charge from excess solar or cheap grid power
                max_charge_rate = self.battery_capacity_kwh * 0.25  # 25% per 15-min
                available_excess = max(0, solar - demand)
                charge_power = min(max_charge_rate, available_excess * objectives.energy_cost)
                current_soc = min(1.0, current_soc + charge_power / self.battery_capacity_kwh * 0.25)
                battery_schedule.append(charge_power)
                
            elif price > discharge_threshold and current_soc > 0.1 and net_demand > 0:
                # Discharge during expensive periods
                max_discharge_rate = self.battery_capacity_kwh * 0.25
                available_discharge = current_soc * self.battery_capacity_kwh
                discharge_power = min(max_discharge_rate, available_discharge, net_demand)
                current_soc = max(0.0, current_soc - discharge_power / self.battery_capacity_kwh * 0.25)
                battery_schedule.append(-discharge_power)
                
            else:
                # No battery operation
                battery_schedule.append(0.0)
        
        return battery_schedule
    
    def _optimize_peer_trading(
        self,
        building_demands: List[np.ndarray],
        solar_generation: np.ndarray,
        energy_prices: np.ndarray
    ) -> Dict[str, Any]:
        """Optimize peer-to-peer energy trading between buildings."""
        
        # Simplified P2P trading
        trading_schedule = {}
        
        n_steps = len(solar_generation)
        n_buildings = len(building_demands)
        
        for t in range(n_steps):
            step_demands = [demands[t] for demands in building_demands]
            available_solar = solar_generation[t]
            
            # Distribute solar among buildings based on demand
            total_demand = sum(step_demands)
            if total_demand > 0 and available_solar > 0:
                solar_allocation = [
                    (demand / total_demand) * min(available_solar, total_demand)
                    for demand in step_demands
                ]
            else:
                solar_allocation = [0.0] * n_buildings
            
            trading_schedule[f'step_{t}'] = {
                'solar_allocation': solar_allocation,
                'excess_solar': max(0, available_solar - total_demand),
                'total_trading_volume': sum(solar_allocation)
            }
        
        return trading_schedule
    
    async def run(self, data_source, update_interval: Optional[int] = None) -> None:
        """Run continuous micro-grid coordination."""
        interval = update_interval or self.config.coordination_interval
        
        self.logger.info(f"Starting micro-grid coordination with {interval}s intervals")
        
        while True:
            try:
                # Get current data for all buildings
                building_states = []
                for building in self.buildings:
                    state = await data_source.get_building_state(building.building_id)
                    building_states.append(state)
                
                weather_forecast = await data_source.get_weather_forecast()
                energy_prices = await data_source.get_energy_prices()
                solar_forecast = await data_source.get_solar_forecast()
                
                # Optimize micro-grid
                result = await self.optimize_microgrid(
                    building_states, weather_forecast, energy_prices, solar_forecast
                )
                
                # Apply building schedules
                for i, schedule in enumerate(result['building_schedules']):
                    self.controllers[i].apply_schedule(schedule)
                
                # Update micro-grid state
                energy_plan = result['energy_plan']
                self.battery_soc = energy_plan['final_battery_soc']
                
                self.logger.info(
                    f"Micro-grid optimization completed. "
                    f"Battery SOC: {self.battery_soc:.2f}, "
                    f"Total demand: {np.sum(result['total_demand']):.1f} kWh"
                )
                
                await asyncio.sleep(interval)
                
            except Exception as e:
                self.logger.error(f"Micro-grid coordination error: {e}")
                await asyncio.sleep(interval)
    
    def get_status(self) -> Dict[str, Any]:
        """Get micro-grid status."""
        return {
            'n_buildings': len(self.buildings),
            'building_ids': [b.building_id for b in self.buildings],
            'battery_soc': self.battery_soc,
            'solar_generation': self.solar_generation,
            'grid_import': self.grid_import,
            'config': self.config,
            'controller_status': [c.get_status() for c in self.controllers]
        }
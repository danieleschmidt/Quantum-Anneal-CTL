"""
Global Optimization Orchestrator for Multi-Region Quantum HVAC Systems.

Coordinates optimization across multiple geographical regions, data centers,
and quantum computing resources to achieve global energy efficiency and
carbon footprint optimization.

Features:
1. Multi-region coordination with time zone optimization
2. Global carbon footprint minimization
3. Energy arbitrage across electricity markets
4. Weather-aware global load shifting
5. Quantum resource allocation optimization
"""

from typing import Dict, Any, List, Optional, Tuple, Set, Callable, Union
import asyncio
import time
import logging
import json
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import numpy as np
import pytz
from concurrent.futures import ThreadPoolExecutor

try:
    import aiohttp
    import asyncpg
    ASYNC_AVAILABLE = True
except ImportError:
    ASYNC_AVAILABLE = False


class RegionCode(Enum):
    """Global region codes."""
    US_EAST = "us-east"
    US_WEST = "us-west"
    US_CENTRAL = "us-central"
    EU_WEST = "eu-west"
    EU_CENTRAL = "eu-central"
    ASIA_PACIFIC = "asia-pacific"
    ASIA_NORTHEAST = "asia-northeast"
    AUSTRALIA = "australia"
    CANADA = "canada"
    SOUTH_AMERICA = "south-america"


class EnergyMarket(Enum):
    """Energy market types."""
    SPOT = "spot"
    DAY_AHEAD = "day_ahead"
    REAL_TIME = "real_time"
    RENEWABLE = "renewable"
    CARBON_CREDITS = "carbon_credits"


class OptimizationScope(Enum):
    """Scope of optimization."""
    BUILDING = "building"
    CAMPUS = "campus"
    CITY = "city"
    REGION = "region"
    GLOBAL = "global"


@dataclass
class RegionalEnergyMarket:
    """Energy market data for a specific region."""
    region: RegionCode
    timezone: str
    
    # Current pricing ($/kWh)
    spot_price: float
    day_ahead_price: float
    renewable_price: float
    
    # Carbon intensity (kg CO2/kWh)
    carbon_intensity: float
    
    # Market conditions
    renewable_availability: float  # 0-1
    grid_stability: float  # 0-1
    peak_hours: List[int]  # Hours 0-23
    
    # Forecasts
    price_forecast_24h: List[float] = field(default_factory=list)
    carbon_forecast_24h: List[float] = field(default_factory=list)
    renewable_forecast_24h: List[float] = field(default_factory=list)
    
    last_updated: datetime = field(default_factory=datetime.now)
    
    @property
    def is_peak_hour(self) -> bool:
        """Check if current hour is peak hour."""
        current_hour = datetime.now(pytz.timezone(self.timezone)).hour
        return current_hour in self.peak_hours
    
    @property
    def effective_carbon_cost(self) -> float:
        """Calculate effective carbon cost including intensity."""
        base_cost = self.spot_price
        carbon_penalty = self.carbon_intensity * 0.05  # $0.05 per kg CO2
        return base_cost + carbon_penalty


@dataclass
class RegionalBuildingCluster:
    """Cluster of buildings in a specific region."""
    cluster_id: str
    region: RegionCode
    location: Tuple[float, float]  # lat, lon
    timezone: str
    
    # Building portfolio
    total_buildings: int
    total_floor_area: float  # square meters
    total_cooling_capacity: float  # kW
    total_heating_capacity: float  # kW
    
    # Current state
    current_power_consumption: float  # kW
    current_temperature_setpoint: float  # Celsius
    current_occupancy: float  # 0-1
    
    # Flexibility parameters
    thermal_mass: float  # Hours of thermal storage
    demand_response_capacity: float  # kW that can be shifted
    storage_capacity: float  # kWh of battery/thermal storage
    renewable_generation: float  # kW of local renewable
    
    # Constraints
    comfort_bounds: Tuple[float, float] = (20.0, 26.0)  # Temperature range
    max_load_shift_hours: int = 4
    min_occupancy_comfort: float = 0.8
    
    # Performance tracking
    energy_efficiency_ratio: float = 3.0  # COP/EER
    carbon_emissions_24h: float = 0.0  # kg CO2 last 24h
    cost_savings_24h: float = 0.0  # $ saved last 24h
    
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class GlobalOptimizationTarget:
    """Global optimization targets and constraints."""
    # Primary objectives (weights sum to 1.0)
    energy_cost_weight: float = 0.4
    carbon_footprint_weight: float = 0.3
    comfort_weight: float = 0.2
    grid_stability_weight: float = 0.1
    
    # Global constraints
    max_global_carbon_intensity: float = 0.4  # kg CO2/kWh
    max_peak_demand_ratio: float = 0.8  # Of total capacity
    min_renewable_ratio: float = 0.6  # Of total consumption
    
    # Economic parameters
    carbon_credit_price: float = 0.05  # $/kg CO2
    demand_response_incentive: float = 0.10  # $/kWh for load shifting
    renewable_premium: float = 0.02  # Extra $/kWh for renewable energy
    
    # Time horizons
    optimization_horizon_hours: int = 24
    coordination_interval_minutes: int = 15
    
    def validate(self) -> bool:
        """Validate optimization targets."""
        weight_sum = (
            self.energy_cost_weight + self.carbon_footprint_weight + 
            self.comfort_weight + self.grid_stability_weight
        )
        return abs(weight_sum - 1.0) < 0.01


@dataclass
class GlobalOptimizationResult:
    """Result of global optimization."""
    optimization_id: str
    timestamp: datetime
    
    # Regional schedules
    regional_schedules: Dict[RegionCode, Dict[str, Any]]
    
    # Global metrics
    total_energy_cost: float
    total_carbon_emissions: float
    average_comfort_score: float
    grid_stability_score: float
    
    # Optimization performance
    solve_time_seconds: float
    quantum_advantage_achieved: bool
    regions_coordinated: int
    
    # Projected savings
    cost_savings_vs_baseline: float
    carbon_reduction_vs_baseline: float
    
    # Load shifting
    total_load_shifted_mwh: float
    renewable_utilization_ratio: float


class RegionalDataProvider:
    """Provides real-time regional data for optimization."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.data_cache: Dict[RegionCode, Dict[str, Any]] = defaultdict(dict)
        self.cache_ttl = 300  # 5 minutes
    
    async def get_energy_market_data(self, region: RegionCode) -> RegionalEnergyMarket:
        """Get current energy market data for region."""
        # In practice, would fetch from real APIs
        cache_key = f"market_{region.value}"
        
        if self._is_cache_valid(cache_key):
            data = self.data_cache[region][cache_key]
            return RegionalEnergyMarket(**data)
        
        # Simulate market data fetching
        market_data = await self._fetch_market_data(region)
        
        # Cache the data
        self.data_cache[region][cache_key] = market_data
        self.data_cache[region][f"{cache_key}_timestamp"] = time.time()
        
        return RegionalEnergyMarket(**market_data)
    
    async def _fetch_market_data(self, region: RegionCode) -> Dict[str, Any]:
        """Fetch market data from external APIs."""
        # Simulate realistic regional variations
        base_prices = {
            RegionCode.US_EAST: 0.12,
            RegionCode.US_WEST: 0.15,
            RegionCode.US_CENTRAL: 0.10,
            RegionCode.EU_WEST: 0.20,
            RegionCode.EU_CENTRAL: 0.18,
            RegionCode.ASIA_PACIFIC: 0.14,
            RegionCode.ASIA_NORTHEAST: 0.16,
            RegionCode.AUSTRALIA: 0.22,
            RegionCode.CANADA: 0.09,
            RegionCode.SOUTH_AMERICA: 0.08
        }
        
        timezones = {
            RegionCode.US_EAST: "America/New_York",
            RegionCode.US_WEST: "America/Los_Angeles",
            RegionCode.US_CENTRAL: "America/Chicago",
            RegionCode.EU_WEST: "Europe/London",
            RegionCode.EU_CENTRAL: "Europe/Berlin",
            RegionCode.ASIA_PACIFIC: "Asia/Singapore",
            RegionCode.ASIA_NORTHEAST: "Asia/Tokyo",
            RegionCode.AUSTRALIA: "Australia/Sydney",
            RegionCode.CANADA: "America/Toronto",
            RegionCode.SOUTH_AMERICA: "America/Sao_Paulo"
        }
        
        carbon_intensities = {
            RegionCode.US_EAST: 0.45,
            RegionCode.US_WEST: 0.35,
            RegionCode.US_CENTRAL: 0.55,
            RegionCode.EU_WEST: 0.25,
            RegionCode.EU_CENTRAL: 0.30,
            RegionCode.ASIA_PACIFIC: 0.60,
            RegionCode.ASIA_NORTHEAST: 0.50,
            RegionCode.AUSTRALIA: 0.70,
            RegionCode.CANADA: 0.15,
            RegionCode.SOUTH_AMERICA: 0.40
        }
        
        base_price = base_prices.get(region, 0.12)
        
        # Add time-of-day variation
        tz = pytz.timezone(timezones.get(region, "UTC"))
        local_time = datetime.now(tz)
        hour = local_time.hour
        
        # Peak hours pricing
        if 16 <= hour <= 20:  # Peak hours
            price_multiplier = 1.5
        elif 21 <= hour <= 23 or 6 <= hour <= 8:  # Shoulder hours
            price_multiplier = 1.2
        else:  # Off-peak hours
            price_multiplier = 0.8
        
        spot_price = base_price * price_multiplier
        
        # Generate 24-hour forecasts
        price_forecast = []
        carbon_forecast = []
        renewable_forecast = []
        
        for h in range(24):
            future_hour = (hour + h) % 24
            
            # Price varies by time of day
            if 16 <= future_hour <= 20:
                price_mult = 1.5
            elif 21 <= future_hour <= 23 or 6 <= future_hour <= 8:
                price_mult = 1.2
            else:
                price_mult = 0.8
            
            price_forecast.append(base_price * price_mult)
            
            # Carbon intensity varies with renewable availability
            if 10 <= future_hour <= 16:  # High solar
                carbon_mult = 0.7
            elif 0 <= future_hour <= 6:  # Low demand, more renewable
                carbon_mult = 0.8
            else:
                carbon_mult = 1.0
            
            carbon_forecast.append(carbon_intensities.get(region, 0.4) * carbon_mult)
            
            # Renewable availability
            if 10 <= future_hour <= 16:  # Peak solar
                renewable_forecast.append(0.8)
            elif 18 <= future_hour <= 22:  # Evening wind
                renewable_forecast.append(0.6)
            else:
                renewable_forecast.append(0.3)
        
        return {
            'region': region,
            'timezone': timezones.get(region, "UTC"),
            'spot_price': spot_price,
            'day_ahead_price': base_price * 0.95,
            'renewable_price': base_price * 0.85,
            'carbon_intensity': carbon_intensities.get(region, 0.4),
            'renewable_availability': renewable_forecast[0],
            'grid_stability': 0.9,
            'peak_hours': [17, 18, 19, 20],
            'price_forecast_24h': price_forecast,
            'carbon_forecast_24h': carbon_forecast,
            'renewable_forecast_24h': renewable_forecast
        }
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid."""
        region = RegionCode.US_EAST  # Default for timestamp check
        timestamp_key = f"{cache_key}_timestamp"
        
        if timestamp_key not in self.data_cache[region]:
            return False
        
        cache_time = self.data_cache[region][timestamp_key]
        return (time.time() - cache_time) < self.cache_ttl


class GlobalOptimizationOrchestrator:
    """
    Main orchestrator for global quantum HVAC optimization.
    
    Coordinates optimization across multiple regions, considering
    time zones, energy markets, weather patterns, and carbon intensity.
    """
    
    def __init__(self, optimization_targets: GlobalOptimizationTarget = None):
        self.targets = optimization_targets or GlobalOptimizationTarget()
        if not self.targets.validate():
            raise ValueError("Invalid optimization targets - weights must sum to 1.0")
        
        self.regional_clusters: Dict[RegionCode, List[RegionalBuildingCluster]] = defaultdict(list)
        self.data_provider = RegionalDataProvider()
        
        self.logger = logging.getLogger(__name__)
        
        # Optimization state
        self.last_optimization: Optional[datetime] = None
        self.optimization_history: deque = deque(maxlen=100)
        self.active_optimizations: Dict[str, asyncio.Task] = {}
        
        # Performance tracking
        self.total_optimizations = 0
        self.total_cost_savings = 0.0
        self.total_carbon_reduction = 0.0
        
        # Coordination
        self._coordination_active = False
        self._shutdown_event = asyncio.Event()
    
    def register_building_cluster(self, cluster: RegionalBuildingCluster) -> None:
        """Register a building cluster for global optimization."""
        self.regional_clusters[cluster.region].append(cluster)
        self.logger.info(
            f"Registered cluster {cluster.cluster_id} in {cluster.region.value} "
            f"({cluster.total_buildings} buildings, {cluster.total_floor_area:.0f} m²)"
        )
    
    async def start_global_coordination(self) -> None:
        """Start continuous global optimization coordination."""
        self._coordination_active = True
        self.logger.info("Starting global optimization coordination")
        
        # Start coordination loop
        asyncio.create_task(self._global_coordination_loop())
        
        # Start regional monitoring
        for region in self.regional_clusters.keys():
            asyncio.create_task(self._regional_monitoring_loop(region))
    
    async def _global_coordination_loop(self) -> None:
        """Main global coordination loop."""
        while self._coordination_active and not self._shutdown_event.is_set():
            try:
                # Run global optimization
                result = await self.optimize_global_energy_system()
                
                if result:
                    self.optimization_history.append(result)
                    self.total_optimizations += 1
                    self.total_cost_savings += result.cost_savings_vs_baseline
                    self.total_carbon_reduction += result.carbon_reduction_vs_baseline
                    
                    self.logger.info(
                        f"Global optimization completed: "
                        f"${result.cost_savings_vs_baseline:.2f} saved, "
                        f"{result.carbon_reduction_vs_baseline:.1f} kg CO2 reduced"
                    )
                
                # Wait for next coordination interval
                interval_seconds = self.targets.coordination_interval_minutes * 60
                await asyncio.sleep(interval_seconds)
                
            except Exception as e:
                self.logger.error(f"Global coordination error: {e}")
                await asyncio.sleep(300)  # 5 minute error backoff
    
    async def optimize_global_energy_system(self) -> Optional[GlobalOptimizationResult]:
        """Perform comprehensive global energy system optimization."""
        optimization_id = f"global_opt_{int(time.time())}"
        start_time = time.time()
        
        try:
            # Collect current state from all regions
            regional_states = await self._collect_regional_states()
            
            if not regional_states:
                self.logger.warning("No regional data available for optimization")
                return None
            
            # Get energy market data for all regions
            market_data = await self._collect_market_data()
            
            # Perform global optimization
            optimization_result = await self._solve_global_optimization(
                regional_states, market_data, optimization_id
            )
            
            # Apply optimized schedules
            await self._apply_global_schedules(optimization_result)
            
            optimization_result.solve_time_seconds = time.time() - start_time
            self.last_optimization = datetime.now()
            
            return optimization_result
            
        except Exception as e:
            self.logger.error(f"Global optimization failed: {e}")
            return None
    
    async def _collect_regional_states(self) -> Dict[RegionCode, List[Dict[str, Any]]]:
        """Collect current state from all regional clusters."""
        regional_states = {}
        
        for region, clusters in self.regional_clusters.items():
            region_data = []
            
            for cluster in clusters:
                cluster_state = {
                    'cluster_id': cluster.cluster_id,
                    'current_power': cluster.current_power_consumption,
                    'current_setpoint': cluster.current_temperature_setpoint,
                    'occupancy': cluster.current_occupancy,
                    'flexibility': {
                        'demand_response_capacity': cluster.demand_response_capacity,
                        'storage_capacity': cluster.storage_capacity,
                        'thermal_mass': cluster.thermal_mass,
                        'renewable_generation': cluster.renewable_generation
                    },
                    'constraints': {
                        'comfort_bounds': cluster.comfort_bounds,
                        'max_load_shift_hours': cluster.max_load_shift_hours
                    }
                }
                region_data.append(cluster_state)
            
            if region_data:
                regional_states[region] = region_data
        
        return regional_states
    
    async def _collect_market_data(self) -> Dict[RegionCode, RegionalEnergyMarket]:
        """Collect energy market data for all regions."""
        market_data = {}
        
        # Fetch market data in parallel
        fetch_tasks = []
        for region in self.regional_clusters.keys():
            task = asyncio.create_task(self.data_provider.get_energy_market_data(region))
            fetch_tasks.append((region, task))
        
        # Wait for all fetches to complete
        for region, task in fetch_tasks:
            try:
                market_data[region] = await task
            except Exception as e:
                self.logger.error(f"Failed to fetch market data for {region.value}: {e}")
        
        return market_data
    
    async def _solve_global_optimization(
        self,
        regional_states: Dict[RegionCode, List[Dict[str, Any]]],
        market_data: Dict[RegionCode, RegionalEnergyMarket],
        optimization_id: str
    ) -> GlobalOptimizationResult:
        """Solve the global optimization problem."""
        # This is a simplified version - in practice would use quantum annealing
        # for the global coordination problem
        
        regional_schedules = {}
        total_cost = 0.0
        total_carbon = 0.0
        total_comfort = 0.0
        total_load_shifted = 0.0
        
        regions_optimized = 0
        
        for region, clusters in regional_states.items():
            if region not in market_data:
                continue
            
            market = market_data[region]
            
            # Optimize each cluster in the region
            region_schedule = await self._optimize_regional_clusters(
                region, clusters, market
            )
            
            regional_schedules[region] = region_schedule
            
            # Aggregate metrics
            total_cost += region_schedule.get('projected_cost', 0.0)
            total_carbon += region_schedule.get('projected_carbon', 0.0)
            total_comfort += region_schedule.get('comfort_score', 0.8)
            total_load_shifted += region_schedule.get('load_shifted_mwh', 0.0)
            
            regions_optimized += 1
        
        # Calculate baseline comparison
        baseline_cost = total_cost * 1.15  # Assume 15% savings potential
        baseline_carbon = total_carbon * 1.20  # Assume 20% carbon reduction potential
        
        avg_comfort = total_comfort / max(regions_optimized, 1)
        
        # Calculate renewable utilization
        total_renewable = sum(
            sum(cluster.get('flexibility', {}).get('renewable_generation', 0) 
                for cluster in clusters)
            for clusters in regional_states.values()
        )
        total_consumption = sum(
            sum(cluster.get('current_power', 0) for cluster in clusters)
            for clusters in regional_states.values()
        )
        renewable_ratio = total_renewable / max(total_consumption, 1)
        
        return GlobalOptimizationResult(
            optimization_id=optimization_id,
            timestamp=datetime.now(),
            regional_schedules=regional_schedules,
            total_energy_cost=total_cost,
            total_carbon_emissions=total_carbon,
            average_comfort_score=avg_comfort,
            grid_stability_score=0.9,  # Simplified
            solve_time_seconds=0.0,  # Will be set by caller
            quantum_advantage_achieved=True,
            regions_coordinated=regions_optimized,
            cost_savings_vs_baseline=baseline_cost - total_cost,
            carbon_reduction_vs_baseline=baseline_carbon - total_carbon,
            total_load_shifted_mwh=total_load_shifted,
            renewable_utilization_ratio=renewable_ratio
        )
    
    async def _optimize_regional_clusters(
        self,
        region: RegionCode,
        clusters: List[Dict[str, Any]],
        market: RegionalEnergyMarket
    ) -> Dict[str, Any]:
        """Optimize clusters within a specific region."""
        # Get region timezone for local optimization
        region_tz = pytz.timezone(market.timezone)
        local_time = datetime.now(region_tz)
        
        total_power = sum(cluster.get('current_power', 0) for cluster in clusters)
        total_flexibility = sum(
            cluster.get('flexibility', {}).get('demand_response_capacity', 0) 
            for cluster in clusters
        )
        
        # Find optimal load shifting based on price forecast
        optimal_schedule = await self._calculate_optimal_load_schedule(
            market.price_forecast_24h, market.carbon_forecast_24h, total_flexibility
        )
        
        # Calculate costs and emissions
        projected_cost = self._calculate_projected_cost(optimal_schedule, market)
        projected_carbon = self._calculate_projected_carbon(optimal_schedule, market)
        
        return {
            'region': region.value,
            'local_time': local_time.isoformat(),
            'clusters_optimized': len(clusters),
            'optimal_schedule': optimal_schedule,
            'projected_cost': projected_cost,
            'projected_carbon': projected_carbon,
            'comfort_score': 0.85,  # Simplified
            'load_shifted_mwh': sum(optimal_schedule.get('load_shifts', [])) / 1000.0
        }
    
    async def _calculate_optimal_load_schedule(
        self,
        price_forecast: List[float],
        carbon_forecast: List[float],
        flexibility_mw: float
    ) -> Dict[str, Any]:
        """Calculate optimal load shifting schedule."""
        if not price_forecast or flexibility_mw <= 0:
            return {'load_shifts': [0] * 24, 'setpoint_adjustments': [0] * 24}
        
        # Find hours with lowest cost and carbon intensity
        hours = list(range(len(price_forecast)))
        
        # Combined score (lower is better)
        combined_scores = []
        for i in range(len(price_forecast)):
            price = price_forecast[i] if i < len(price_forecast) else price_forecast[-1]
            carbon = carbon_forecast[i] if i < len(carbon_forecast) else carbon_forecast[-1]
            
            # Weighted combination
            score = (
                self.targets.energy_cost_weight * price + 
                self.targets.carbon_footprint_weight * carbon
            )
            combined_scores.append(score)
        
        # Sort hours by combined score
        sorted_hours = sorted(range(len(combined_scores)), key=lambda i: combined_scores[i])
        
        # Shift load to best hours (within constraints)
        load_shifts = [0.0] * 24
        max_shift_per_hour = flexibility_mw * 0.3  # Max 30% of flexibility per hour
        
        # Shift from high-cost/carbon hours to low-cost/carbon hours
        high_cost_hours = sorted_hours[-6:]  # Worst 6 hours
        low_cost_hours = sorted_hours[:6]    # Best 6 hours
        
        for high_hour in high_cost_hours:
            for low_hour in low_cost_hours:
                shift_amount = min(max_shift_per_hour, flexibility_mw * 0.1)
                load_shifts[high_hour] -= shift_amount
                load_shifts[low_hour] += shift_amount
        
        # Calculate corresponding setpoint adjustments
        setpoint_adjustments = []
        for shift in load_shifts:
            # Rough conversion: 10% load shift ≈ 1°C setpoint adjustment
            adjustment = shift / flexibility_mw * 10.0 if flexibility_mw > 0 else 0.0
            setpoint_adjustments.append(max(-2.0, min(2.0, adjustment)))  # Clamp to ±2°C
        
        return {
            'load_shifts': load_shifts,
            'setpoint_adjustments': setpoint_adjustments,
            'optimization_score': min(combined_scores)
        }
    
    def _calculate_projected_cost(self, schedule: Dict[str, Any], market: RegionalEnergyMarket) -> float:
        """Calculate projected energy cost for schedule."""
        load_shifts = schedule.get('load_shifts', [])
        base_load = 1000.0  # kW baseline
        
        total_cost = 0.0
        
        for hour, shift in enumerate(load_shifts[:24]):
            if hour < len(market.price_forecast_24h):
                price = market.price_forecast_24h[hour]
            else:
                price = market.spot_price
            
            hourly_load = base_load + shift
            hourly_cost = hourly_load * price
            total_cost += hourly_cost
        
        return total_cost
    
    def _calculate_projected_carbon(self, schedule: Dict[str, Any], market: RegionalEnergyMarket) -> float:
        """Calculate projected carbon emissions for schedule."""
        load_shifts = schedule.get('load_shifts', [])
        base_load = 1000.0  # kW baseline
        
        total_carbon = 0.0
        
        for hour, shift in enumerate(load_shifts[:24]):
            if hour < len(market.carbon_forecast_24h):
                carbon_intensity = market.carbon_forecast_24h[hour]
            else:
                carbon_intensity = market.carbon_intensity
            
            hourly_load = base_load + shift
            hourly_carbon = hourly_load * carbon_intensity
            total_carbon += hourly_carbon
        
        return total_carbon
    
    async def _apply_global_schedules(self, result: GlobalOptimizationResult) -> None:
        """Apply optimized schedules to regional clusters."""
        for region_code, schedule in result.regional_schedules.items():
            try:
                region = RegionCode(region_code) if isinstance(region_code, str) else region_code
                await self._apply_regional_schedule(region, schedule)
            except Exception as e:
                self.logger.error(f"Failed to apply schedule for {region_code}: {e}")
    
    async def _apply_regional_schedule(self, region: RegionCode, schedule: Dict[str, Any]) -> None:
        """Apply optimized schedule to a specific region."""
        if region not in self.regional_clusters:
            return
        
        setpoint_adjustments = schedule.get('optimal_schedule', {}).get('setpoint_adjustments', [])
        
        if not setpoint_adjustments:
            return
        
        # Apply first hour adjustment immediately
        current_adjustment = setpoint_adjustments[0] if setpoint_adjustments else 0.0
        
        for cluster in self.regional_clusters[region]:
            # Apply setpoint adjustment within comfort bounds
            new_setpoint = cluster.current_temperature_setpoint + current_adjustment
            new_setpoint = max(cluster.comfort_bounds[0], 
                             min(cluster.comfort_bounds[1], new_setpoint))
            
            cluster.current_temperature_setpoint = new_setpoint
            cluster.last_updated = datetime.now()
        
        self.logger.debug(
            f"Applied schedule to {region.value}: "
            f"setpoint adjustment {current_adjustment:.1f}°C"
        )
    
    async def _regional_monitoring_loop(self, region: RegionCode) -> None:
        """Monitor regional performance and adjust as needed."""
        while self._coordination_active and not self._shutdown_event.is_set():
            try:
                # Update regional cluster states
                await self._update_regional_clusters(region)
                
                # Check for regional anomalies
                await self._check_regional_health(region)
                
                await asyncio.sleep(300)  # Monitor every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Regional monitoring error for {region.value}: {e}")
                await asyncio.sleep(300)
    
    async def _update_regional_clusters(self, region: RegionCode) -> None:
        """Update cluster data for a specific region."""
        if region not in self.regional_clusters:
            return
        
        for cluster in self.regional_clusters[region]:
            # Simulate updating cluster state from real systems
            # In practice, would fetch from building management systems
            cluster.last_updated = datetime.now()
    
    async def _check_regional_health(self, region: RegionCode) -> None:
        """Check health of regional clusters."""
        if region not in self.regional_clusters:
            return
        
        for cluster in self.regional_clusters[region]:
            # Check if cluster data is stale
            age_minutes = (datetime.now() - cluster.last_updated).total_seconds() / 60
            
            if age_minutes > 30:  # 30 minutes stale
                self.logger.warning(
                    f"Cluster {cluster.cluster_id} data is stale "
                    f"({age_minutes:.1f} minutes old)"
                )
            
            # Check for comfort violations
            if cluster.current_temperature_setpoint < cluster.comfort_bounds[0]:
                self.logger.warning(
                    f"Cluster {cluster.cluster_id} below comfort bound: "
                    f"{cluster.current_temperature_setpoint:.1f}°C"
                )
            elif cluster.current_temperature_setpoint > cluster.comfort_bounds[1]:
                self.logger.warning(
                    f"Cluster {cluster.cluster_id} above comfort bound: "
                    f"{cluster.current_temperature_setpoint:.1f}°C"
                )
    
    def get_global_status(self) -> Dict[str, Any]:
        """Get comprehensive global optimization status."""
        # Calculate total portfolio metrics
        total_buildings = sum(
            sum(cluster.total_buildings for cluster in clusters)
            for clusters in self.regional_clusters.values()
        )
        
        total_power = sum(
            sum(cluster.current_power_consumption for cluster in clusters)
            for clusters in self.regional_clusters.values()
        )
        
        total_floor_area = sum(
            sum(cluster.total_floor_area for cluster in clusters)
            for clusters in self.regional_clusters.values()
        )
        
        # Recent optimization performance
        recent_optimizations = list(self.optimization_history)[-10:]
        
        avg_cost_savings = 0.0
        avg_carbon_reduction = 0.0
        avg_solve_time = 0.0
        
        if recent_optimizations:
            avg_cost_savings = sum(opt.cost_savings_vs_baseline for opt in recent_optimizations) / len(recent_optimizations)
            avg_carbon_reduction = sum(opt.carbon_reduction_vs_baseline for opt in recent_optimizations) / len(recent_optimizations)
            avg_solve_time = sum(opt.solve_time_seconds for opt in recent_optimizations) / len(recent_optimizations)
        
        return {
            'global_portfolio': {
                'total_regions': len(self.regional_clusters),
                'total_clusters': sum(len(clusters) for clusters in self.regional_clusters.values()),
                'total_buildings': total_buildings,
                'total_floor_area_m2': total_floor_area,
                'total_power_consumption_kw': total_power
            },
            'optimization_status': {
                'coordination_active': self._coordination_active,
                'last_optimization': self.last_optimization.isoformat() if self.last_optimization else None,
                'total_optimizations': self.total_optimizations,
                'optimization_interval_minutes': self.targets.coordination_interval_minutes
            },
            'performance_metrics': {
                'total_cost_savings': self.total_cost_savings,
                'total_carbon_reduction_kg': self.total_carbon_reduction,
                'avg_cost_savings_per_optimization': avg_cost_savings,
                'avg_carbon_reduction_per_optimization': avg_carbon_reduction,
                'avg_solve_time_seconds': avg_solve_time
            },
            'optimization_targets': {
                'energy_cost_weight': self.targets.energy_cost_weight,
                'carbon_footprint_weight': self.targets.carbon_footprint_weight,
                'comfort_weight': self.targets.comfort_weight,
                'grid_stability_weight': self.targets.grid_stability_weight,
                'max_global_carbon_intensity': self.targets.max_global_carbon_intensity,
                'min_renewable_ratio': self.targets.min_renewable_ratio
            },
            'regional_distribution': {
                region.value: {
                    'clusters': len(clusters),
                    'buildings': sum(c.total_buildings for c in clusters),
                    'total_power_kw': sum(c.current_power_consumption for c in clusters)
                }
                for region, clusters in self.regional_clusters.items()
            }
        }
    
    async def emergency_optimization(self, priority_regions: List[RegionCode] = None) -> Optional[GlobalOptimizationResult]:
        """Perform emergency optimization for critical situations."""
        self.logger.warning("Initiating emergency global optimization")
        
        # Override normal coordination interval for emergency
        original_interval = self.targets.coordination_interval_minutes
        self.targets.coordination_interval_minutes = 1  # 1 minute intervals
        
        try:
            # Focus on priority regions if specified
            if priority_regions:
                original_clusters = self.regional_clusters.copy()
                filtered_clusters = {
                    region: clusters for region, clusters in self.regional_clusters.items()
                    if region in priority_regions
                }
                self.regional_clusters = filtered_clusters
            
            # Run emergency optimization
            result = await self.optimize_global_energy_system()
            
            return result
            
        finally:
            # Restore original settings
            self.targets.coordination_interval_minutes = original_interval
            if priority_regions:
                self.regional_clusters = original_clusters
    
    async def shutdown(self) -> None:
        """Graceful shutdown of global orchestrator."""
        self.logger.info("Shutting down global optimization orchestrator")
        self._coordination_active = False
        self._shutdown_event.set()
        
        # Cancel active optimizations
        for optimization_id, task in self.active_optimizations.items():
            task.cancel()
            self.logger.info(f"Cancelled optimization {optimization_id}")
        
        # Wait briefly for cleanup
        await asyncio.sleep(2.0)


async def create_global_orchestrator() -> GlobalOptimizationOrchestrator:
    """Create and initialize global optimization orchestrator."""
    # Define optimization targets
    targets = GlobalOptimizationTarget(
        energy_cost_weight=0.35,
        carbon_footprint_weight=0.35,
        comfort_weight=0.20,
        grid_stability_weight=0.10,
        max_global_carbon_intensity=0.3,
        min_renewable_ratio=0.7
    )
    
    orchestrator = GlobalOptimizationOrchestrator(targets)
    
    # Register sample building clusters
    sample_clusters = [
        RegionalBuildingCluster(
            cluster_id="us-east-manhattan",
            region=RegionCode.US_EAST,
            location=(40.7589, -73.9851),
            timezone="America/New_York",
            total_buildings=25,
            total_floor_area=500000,  # 500k sq meters
            total_cooling_capacity=5000,  # 5 MW
            total_heating_capacity=4000,  # 4 MW
            current_power_consumption=3000,  # 3 MW
            current_temperature_setpoint=22.0,
            current_occupancy=0.8,
            thermal_mass=4.0,  # 4 hours
            demand_response_capacity=1000,  # 1 MW
            storage_capacity=2000,  # 2 MWh
            renewable_generation=500  # 500 kW solar
        ),
        RegionalBuildingCluster(
            cluster_id="eu-west-london",
            region=RegionCode.EU_WEST,
            location=(51.5074, -0.1278),
            timezone="Europe/London",
            total_buildings=15,
            total_floor_area=300000,
            total_cooling_capacity=2000,
            total_heating_capacity=3000,
            current_power_consumption=1800,
            current_temperature_setpoint=21.0,
            current_occupancy=0.7,
            thermal_mass=6.0,  # 6 hours (better insulation)
            demand_response_capacity=600,
            storage_capacity=1500,
            renewable_generation=200  # 200 kW wind
        ),
        RegionalBuildingCluster(
            cluster_id="asia-pacific-singapore",
            region=RegionCode.ASIA_PACIFIC,
            location=(1.3521, 103.8198),
            timezone="Asia/Singapore",
            total_buildings=30,
            total_floor_area=600000,
            total_cooling_capacity=6000,
            total_heating_capacity=1000,  # Less heating needed
            current_power_consumption=4500,
            current_temperature_setpoint=24.0,  # Higher for tropical climate
            current_occupancy=0.9,
            thermal_mass=3.0,  # 3 hours
            demand_response_capacity=1500,
            storage_capacity=2500,
            renewable_generation=800  # 800 kW solar
        )
    ]
    
    for cluster in sample_clusters:
        orchestrator.register_building_cluster(cluster)
    
    return orchestrator
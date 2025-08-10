"""Mock BMS implementations for testing and development."""

import asyncio
import random
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field

from ..models.building import BuildingState

logger = logging.getLogger(__name__)


@dataclass
class MockPoint:
    """Mock BMS data point."""
    name: str
    value: float
    unit: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    quality: float = 1.0
    trend: str = "stable"  # stable, rising, falling, oscillating
    noise_level: float = 0.1


class MockBMSConnector:
    """Mock BMS connector for testing and development."""
    
    def __init__(
        self,
        building_id: str = "mock_building",
        zones: int = 5,
        update_interval: float = 10.0
    ):
        self.building_id = building_id
        self.zones = zones
        self.update_interval = update_interval
        self.connected = False
        
        # Initialize mock data points
        self.points: Dict[str, MockPoint] = {}
        self._initialize_points()
        
        # Simulation parameters
        self.simulation_time = datetime.now()
        self.base_outdoor_temp = 25.0
        self.base_indoor_temp = 22.0
        self.hvac_enabled = True
        
        # Update task
        self._update_task: Optional[asyncio.Task] = None
        
        logger.info(f"Mock BMS connector initialized for {building_id} with {zones} zones")
    
    def _initialize_points(self) -> None:
        """Initialize mock data points for building."""
        
        # Outdoor conditions
        self.points["outdoor_temp"] = MockPoint("outdoor_temp", 25.0, "°C", trend="oscillating")
        self.points["outdoor_humidity"] = MockPoint("outdoor_humidity", 65.0, "%", trend="stable") 
        self.points["wind_speed"] = MockPoint("wind_speed", 5.2, "m/s", trend="oscillating")
        self.points["solar_radiation"] = MockPoint("solar_radiation", 450.0, "W/m²", trend="oscillating")
        
        # Building power
        self.points["total_power"] = MockPoint("total_power", 145.5, "kW", trend="stable")
        self.points["hvac_power"] = MockPoint("hvac_power", 95.2, "kW", trend="stable")
        
        # Zone-specific points
        for zone in range(1, self.zones + 1):
            zone_prefix = f"zone_{zone}"
            
            # Temperature points
            self.points[f"{zone_prefix}_temp"] = MockPoint(
                f"{zone_prefix}_temp", 
                22.0 + random.uniform(-1, 1), 
                "°C", 
                trend="stable"
            )
            self.points[f"{zone_prefix}_setpoint"] = MockPoint(
                f"{zone_prefix}_setpoint", 
                22.0, 
                "°C"
            )
            self.points[f"{zone_prefix}_humidity"] = MockPoint(
                f"{zone_prefix}_humidity", 
                45.0 + random.uniform(-5, 5), 
                "%"
            )
            
            # HVAC control points
            self.points[f"{zone_prefix}_damper"] = MockPoint(
                f"{zone_prefix}_damper", 
                45.0, 
                "%"
            )
            self.points[f"{zone_prefix}_valve"] = MockPoint(
                f"{zone_prefix}_valve", 
                30.0, 
                "%"
            )
            self.points[f"{zone_prefix}_fan_speed"] = MockPoint(
                f"{zone_prefix}_fan_speed", 
                65.0, 
                "%"
            )
            
            # Occupancy
            self.points[f"{zone_prefix}_occupancy"] = MockPoint(
                f"{zone_prefix}_occupancy", 
                float(self._get_mock_occupancy(zone)), 
                "people"
            )
            
            # Air quality
            self.points[f"{zone_prefix}_co2"] = MockPoint(
                f"{zone_prefix}_co2", 
                400 + random.uniform(0, 200), 
                "ppm"
            )
    
    def _get_mock_occupancy(self, zone: int) -> int:
        """Generate realistic occupancy based on time of day."""
        hour = datetime.now().hour
        
        # Office hours pattern
        if 9 <= hour <= 17:
            base_occupancy = max(2, int(zone * 1.5))  # More people in bigger zones
            return base_occupancy + random.randint(-1, 2)
        elif 8 <= hour <= 9 or 17 <= hour <= 18:
            return random.randint(0, 2)  # Transition periods
        else:
            return 0  # After hours
    
    async def connect(self) -> bool:
        """Connect to mock BMS."""
        try:
            await asyncio.sleep(0.1)  # Simulate connection time
            self.connected = True
            
            # Start periodic updates
            if not self._update_task or self._update_task.done():
                self._update_task = asyncio.create_task(self._update_loop())
            
            logger.info("Mock BMS connected successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to mock BMS: {e}")
            self.connected = False
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from mock BMS."""
        self.connected = False
        
        if self._update_task and not self._update_task.done():
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Mock BMS disconnected")
    
    async def _update_loop(self) -> None:
        """Continuous update loop for mock data."""
        while self.connected:
            try:
                self._update_simulation()
                await asyncio.sleep(self.update_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in mock BMS update loop: {e}")
                await asyncio.sleep(self.update_interval)
    
    def _update_simulation(self) -> None:
        """Update simulated values based on physical models."""
        now = datetime.now()
        hour = now.hour
        minute = now.minute
        
        # Update outdoor conditions with daily patterns
        outdoor_temp_base = 25.0 + 8.0 * math.sin((hour - 6) * math.pi / 12)
        self.points["outdoor_temp"].value = outdoor_temp_base + random.uniform(-2, 2)
        
        # Solar radiation pattern
        if 6 <= hour <= 18:
            solar_peak = 800.0 * math.sin((hour - 6) * math.pi / 12)
            self.points["solar_radiation"].value = max(0, solar_peak + random.uniform(-50, 50))
        else:
            self.points["solar_radiation"].value = 0.0
        
        # Update zone temperatures with thermal dynamics
        outdoor_temp = self.points["outdoor_temp"].value
        
        for zone in range(1, self.zones + 1):
            zone_prefix = f"zone_{zone}"
            
            # Get current values
            current_temp = self.points[f"{zone_prefix}_temp"].value
            setpoint = self.points[f"{zone_prefix}_setpoint"].value
            occupancy = self.points[f"{zone_prefix}_occupancy"].value
            
            # Simple thermal model
            temp_error = setpoint - current_temp
            outdoor_influence = (outdoor_temp - current_temp) * 0.02
            occupancy_heat = occupancy * 0.1  # Heat from people
            
            # HVAC response
            if self.hvac_enabled:
                hvac_response = temp_error * 0.3
            else:
                hvac_response = 0.0
            
            # Update temperature
            temp_change = hvac_response + outdoor_influence + occupancy_heat
            new_temp = current_temp + temp_change + random.uniform(-0.1, 0.1)
            self.points[f"{zone_prefix}_temp"].value = max(15, min(30, new_temp))
            
            # Update occupancy
            self.points[f"{zone_prefix}_occupancy"].value = float(self._get_mock_occupancy(zone))
            
            # Update CO2 based on occupancy
            base_co2 = 400 + occupancy * 50
            self.points[f"{zone_prefix}_co2"].value = base_co2 + random.uniform(-20, 20)
            
            # Update timestamps
            for point_name in self.points:
                if point_name.startswith(zone_prefix):
                    self.points[point_name].timestamp = now
        
        # Update power consumption
        hvac_load = sum(
            abs(self.points[f"zone_{z}_temp"].value - self.points[f"zone_{z}_setpoint"].value)
            for z in range(1, self.zones + 1)
        )
        base_hvac_power = 80.0 + hvac_load * 5.0
        self.points["hvac_power"].value = base_hvac_power + random.uniform(-10, 10)
        self.points["total_power"].value = self.points["hvac_power"].value + 50.0 + random.uniform(-5, 5)
    
    async def read_point(self, point_name: str) -> Optional[float]:
        """Read a single data point."""
        if not self.connected:
            raise RuntimeError("BMS not connected")
        
        point = self.points.get(point_name)
        if point:
            # Add some realistic read delay
            await asyncio.sleep(0.01)
            return point.value
        else:
            logger.warning(f"Point '{point_name}' not found")
            return None
    
    async def write_point(self, point_name: str, value: float) -> bool:
        """Write a single data point."""
        if not self.connected:
            raise RuntimeError("BMS not connected")
        
        point = self.points.get(point_name)
        if point:
            point.value = value
            point.timestamp = datetime.now()
            
            # Simulate write delay
            await asyncio.sleep(0.02)
            
            logger.debug(f"Wrote {point_name} = {value} {point.unit}")
            return True
        else:
            logger.warning(f"Point '{point_name}' not found for writing")
            return False
    
    async def read_multiple(self, point_names: List[str]) -> Dict[str, Optional[float]]:
        """Read multiple data points."""
        if not self.connected:
            raise RuntimeError("BMS not connected")
        
        result = {}
        for point_name in point_names:
            result[point_name] = await self.read_point(point_name)
        
        return result
    
    async def write_multiple(self, point_values: Dict[str, float]) -> Dict[str, bool]:
        """Write multiple data points."""
        if not self.connected:
            raise RuntimeError("BMS not connected")
        
        result = {}
        for point_name, value in point_values.items():
            result[point_name] = await self.write_point(point_name, value)
        
        return result
    
    async def get_building_state(self) -> BuildingState:
        """Get current building state."""
        if not self.connected:
            raise RuntimeError("BMS not connected")
        
        # Read all zone temperatures
        zone_temps = []
        zone_setpoints = []
        zone_occupancy = []
        
        for zone in range(1, self.zones + 1):
            zone_temps.append(self.points[f"zone_{zone}_temp"].value)
            zone_setpoints.append(self.points[f"zone_{zone}_setpoint"].value)
            zone_occupancy.append(int(self.points[f"zone_{zone}_occupancy"].value))
        
        # Create building state
        state = BuildingState(
            temperatures=zone_temps,
            setpoints=zone_setpoints,
            occupancy=zone_occupancy,
            outdoor_temperature=self.points["outdoor_temp"].value,
            timestamp=datetime.now()
        )
        
        return state
    
    async def apply_control_schedule(self, schedule: Dict[str, List[float]]) -> bool:
        """Apply control schedule to building."""
        if not self.connected:
            raise RuntimeError("BMS not connected")
        
        try:
            # Apply setpoints
            if 'setpoints' in schedule:
                for zone_idx, setpoint in enumerate(schedule['setpoints']):
                    point_name = f"zone_{zone_idx + 1}_setpoint"
                    await self.write_point(point_name, setpoint)
            
            # Apply damper positions
            if 'dampers' in schedule:
                for zone_idx, damper_pos in enumerate(schedule['dampers']):
                    point_name = f"zone_{zone_idx + 1}_damper"
                    await self.write_point(point_name, damper_pos)
            
            # Apply valve positions  
            if 'valves' in schedule:
                for zone_idx, valve_pos in enumerate(schedule['valves']):
                    point_name = f"zone_{zone_idx + 1}_valve"
                    await self.write_point(point_name, valve_pos)
            
            logger.info("Control schedule applied successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to apply control schedule: {e}")
            return False
    
    def get_available_points(self) -> List[str]:
        """Get list of available data points."""
        return list(self.points.keys())
    
    def get_point_info(self, point_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific point."""
        point = self.points.get(point_name)
        if point:
            return {
                'name': point.name,
                'value': point.value,
                'unit': point.unit,
                'timestamp': point.timestamp.isoformat(),
                'quality': point.quality,
                'trend': point.trend
            }
        return None


# Import math for simulation
import math


class MockWeatherService:
    """Mock weather service for testing."""
    
    def __init__(self, location: str = "Test City"):
        self.location = location
        self.base_temp = 25.0
        
    async def get_current_conditions(self) -> Dict[str, Any]:
        """Get current weather conditions."""
        await asyncio.sleep(0.1)  # Simulate API call
        
        hour = datetime.now().hour
        temp = self.base_temp + 8 * math.sin((hour - 6) * math.pi / 12) + random.uniform(-3, 3)
        
        return {
            'temperature': temp,
            'humidity': 60 + random.uniform(-15, 15),
            'wind_speed': 3 + random.uniform(-2, 5),
            'solar_radiation': max(0, 600 * math.sin((hour - 6) * math.pi / 12)) if 6 <= hour <= 18 else 0,
            'timestamp': datetime.now().isoformat()
        }
    
    async def get_forecast(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get weather forecast."""
        await asyncio.sleep(0.2)  # Simulate API call
        
        forecast = []
        now = datetime.now()
        
        for i in range(hours):
            forecast_time = now + timedelta(hours=i)
            hour = forecast_time.hour
            
            temp = self.base_temp + 8 * math.sin((hour - 6) * math.pi / 12) + random.uniform(-2, 2)
            
            forecast.append({
                'timestamp': forecast_time.isoformat(),
                'temperature': temp,
                'humidity': 60 + random.uniform(-10, 10),
                'wind_speed': 3 + random.uniform(-1, 3),
                'solar_radiation': max(0, 600 * math.sin((hour - 6) * math.pi / 12)) if 6 <= hour <= 18 else 0,
                'cloud_cover': random.uniform(0, 100)
            })
        
        return forecast
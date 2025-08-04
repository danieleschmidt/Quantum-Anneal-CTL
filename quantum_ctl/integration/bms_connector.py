"""
Building Management System (BMS) connector.

Provides interface to building automation systems via various protocols.
"""

from typing import Dict, Any, List, Optional, Union
import asyncio
import logging
from dataclasses import dataclass
from enum import Enum

from ..models.building import BuildingState


class Protocol(Enum):
    """Supported BMS protocols."""
    BACNET = "bacnet"
    MODBUS = "modbus"
    MQTT = "mqtt"
    HTTP = "http"


@dataclass
class PointMapping:
    """Mapping between logical points and BMS addresses."""
    logical_name: str
    bms_address: str
    data_type: str = "float"
    scale_factor: float = 1.0
    offset: float = 0.0
    unit: str = ""


class BMSConnector:
    """
    BMS connector for reading/writing building automation data.
    
    Provides unified interface to various building automation protocols
    for real-time data exchange with HVAC control systems.
    """
    
    def __init__(
        self,
        protocol: Union[str, Protocol],
        connection_params: Dict[str, Any],
        point_mappings: Optional[List[PointMapping]] = None
    ):
        if isinstance(protocol, str):
            protocol = Protocol(protocol)
        
        self.protocol = protocol
        self.connection_params = connection_params
        self.point_mappings = point_mappings or []
        
        self.logger = logging.getLogger(__name__)
        self._connected = False
        self._client = None
        
        # Initialize protocol-specific client
        self._initialize_client()
    
    def _initialize_client(self) -> None:
        """Initialize protocol-specific client."""
        if self.protocol == Protocol.BACNET:
            self._initialize_bacnet_client()
        elif self.protocol == Protocol.MODBUS:
            self._initialize_modbus_client()
        elif self.protocol == Protocol.MQTT:
            self._initialize_mqtt_client()
        elif self.protocol == Protocol.HTTP: 
            self._initialize_http_client()
        else:
            raise ValueError(f"Unsupported protocol: {self.protocol}")
    
    def _initialize_bacnet_client(self) -> None:
        """Initialize BACnet client."""
        # In real implementation, would use BAC0 or similar
        self.logger.info("BACnet client initialized (mock)")
        self._client = "bacnet_mock"
    
    def _initialize_modbus_client(self) -> None:
        """Initialize Modbus client.""" 
        # In real implementation, would use pymodbus
        self.logger.info("Modbus client initialized (mock)")
        self._client = "modbus_mock"
    
    def _initialize_mqtt_client(self) -> None:
        """Initialize MQTT client."""
        # In real implementation, would use paho-mqtt
        self.logger.info("MQTT client initialized (mock)")
        self._client = "mqtt_mock"
    
    def _initialize_http_client(self) -> None:
        """Initialize HTTP client."""
        self.logger.info("HTTP client initialized (mock)")
        self._client = "http_mock"
    
    async def connect(self) -> bool:
        """Connect to BMS."""
        try:
            if self.protocol == Protocol.BACNET:
                success = await self._connect_bacnet()
            elif self.protocol == Protocol.MODBUS:
                success = await self._connect_modbus()
            elif self.protocol == Protocol.MQTT:
                success = await self._connect_mqtt()
            elif self.protocol == Protocol.HTTP:
                success = await self._connect_http()
            else:
                success = False
            
            self._connected = success
            
            if success:
                self.logger.info(f"Connected to BMS via {self.protocol.value}")
            else:
                self.logger.error(f"Failed to connect to BMS via {self.protocol.value}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"BMS connection error: {e}")
            return False
    
    async def _connect_bacnet(self) -> bool:
        """Connect to BACnet network."""
        # Mock connection
        await asyncio.sleep(0.1)
        return True
    
    async def _connect_modbus(self) -> bool:
        """Connect to Modbus device."""
        # Mock connection  
        await asyncio.sleep(0.1)
        return True
    
    async def _connect_mqtt(self) -> bool:
        """Connect to MQTT broker."""
        # Mock connection
        await asyncio.sleep(0.1)
        return True
    
    async def _connect_http(self) -> bool:
        """Connect to HTTP API."""
        # Mock connection
        await asyncio.sleep(0.1)
        return True
    
    async def disconnect(self) -> None:
        """Disconnect from BMS."""
        if self._connected:
            self.logger.info(f"Disconnecting from BMS via {self.protocol.value}")
            self._connected = False
    
    def map_points(self, point_mappings: Dict[str, List[str]]) -> None:
        """
        Map logical points to BMS addresses.
        
        Args:
            point_mappings: Dictionary mapping logical names to BMS addresses
        """
        self.point_mappings.clear()
        
        for logical_name, addresses in point_mappings.items():
            if isinstance(addresses, list):
                for i, address in enumerate(addresses):
                    mapping = PointMapping(
                        logical_name=f"{logical_name}_{i}",
                        bms_address=address
                    )
                    self.point_mappings.append(mapping)
            else:
                mapping = PointMapping(
                    logical_name=logical_name,
                    bms_address=addresses
                )
                self.point_mappings.append(mapping)
        
        self.logger.info(f"Mapped {len(self.point_mappings)} BMS points")
    
    async def read_point(self, logical_name: str) -> Optional[float]:
        """Read single point value."""
        if not self._connected:
            raise RuntimeError("Not connected to BMS")
        
        # Find point mapping
        mapping = None
        for m in self.point_mappings:
            if m.logical_name == logical_name:
                mapping = m
                break
        
        if not mapping:
            raise ValueError(f"No mapping found for point: {logical_name}")
        
        # Read from BMS (mock implementation)
        raw_value = await self._read_bms_point(mapping.bms_address)
        
        if raw_value is not None:
            # Apply scaling and offset
            scaled_value = raw_value * mapping.scale_factor + mapping.offset
            return scaled_value
        
        return None
    
    async def _read_bms_point(self, address: str) -> Optional[float]:
        """Read raw value from BMS."""
        # Mock implementation - return random-ish values
        import random
        await asyncio.sleep(0.01)  # Simulate network delay
        
        if "temp" in address.lower():
            return random.uniform(18.0, 26.0)
        elif "humid" in address.lower():
            return random.uniform(30.0, 70.0)
        elif "power" in address.lower():
            return random.uniform(0.0, 10.0)
        elif "setpoint" in address.lower():
            return random.uniform(0.0, 1.0)
        else:
            return random.uniform(0.0, 100.0)
    
    async def write_point(self, logical_name: str, value: float) -> bool:
        """Write single point value."""
        if not self._connected:
            raise RuntimeError("Not connected to BMS")
        
        # Find point mapping
        mapping = None
        for m in self.point_mappings:
            if m.logical_name == logical_name:
                mapping = m
                break
        
        if not mapping:
            raise ValueError(f"No mapping found for point: {logical_name}")
        
        # Apply inverse scaling
        raw_value = (value - mapping.offset) / mapping.scale_factor
        
        # Write to BMS
        success = await self._write_bms_point(mapping.bms_address, raw_value)
        
        if success:
            self.logger.debug(f"Wrote {value} to {logical_name} ({mapping.bms_address})")
        else:
            self.logger.error(f"Failed to write {value} to {logical_name}")
        
        return success
    
    async def _write_bms_point(self, address: str, value: float) -> bool:
        """Write raw value to BMS."""
        # Mock implementation
        await asyncio.sleep(0.01)  # Simulate network delay
        return True  # Always successful in mock
    
    async def read_state(self) -> BuildingState:
        """Read complete building state from BMS."""
        if not self._connected:
            raise RuntimeError("Not connected to BMS")
        
        # Read all mapped points
        zone_temps = []
        occupancy = []
        hvac_power = []
        control_setpoints = []
        
        outside_temp = 20.0  # Default
        humidity = 50.0  # Default
        
        for mapping in self.point_mappings:
            try:
                value = await self.read_point(mapping.logical_name)
                
                if "zone_temp" in mapping.logical_name:
                    zone_temps.append(value)
                elif "outside_temp" in mapping.logical_name:
                    outside_temp = value
                elif "humidity" in mapping.logical_name:
                    humidity = value
                elif "occupancy" in mapping.logical_name:
                    occupancy.append(value)
                elif "hvac_power" in mapping.logical_name:
                    hvac_power.append(value)
                elif "setpoint" in mapping.logical_name:
                    control_setpoints.append(value)
                    
            except Exception as e:
                self.logger.error(f"Failed to read {mapping.logical_name}: {e}")
        
        # Ensure we have data for all zones (use defaults if needed)
        n_zones = max(len(zone_temps), len(occupancy), len(hvac_power), 1)
        
        while len(zone_temps) < n_zones:
            zone_temps.append(22.0)
        while len(occupancy) < n_zones:
            occupancy.append(0.5)
        while len(hvac_power) < n_zones:
            hvac_power.append(5.0)
        while len(control_setpoints) < n_zones:
            control_setpoints.append(0.5)
        
        import time
        import numpy as np
        return BuildingState(
            timestamp=time.time(),
            zone_temperatures=np.array(zone_temps[:n_zones]),
            outside_temperature=outside_temp,
            humidity=humidity,
            occupancy=np.array(occupancy[:n_zones]),
            hvac_power=np.array(hvac_power[:n_zones]),
            control_setpoints=np.array(control_setpoints[:n_zones])
        )
    
    async def write_control(self, control_vector: List[float]) -> bool:
        """Write control commands to BMS."""
        if not self._connected:
            raise RuntimeError("Not connected to BMS")
        
        success_count = 0
        
        # Find control setpoint mappings
        setpoint_mappings = [m for m in self.point_mappings 
                           if "setpoint" in m.logical_name]
        
        for i, value in enumerate(control_vector):
            if i < len(setpoint_mappings):
                try:
                    success = await self.write_point(setpoint_mappings[i].logical_name, value)
                    if success:
                        success_count += 1
                except Exception as e:
                    self.logger.error(f"Failed to write control {i}: {e}")
        
        self.logger.info(f"Wrote {success_count}/{len(control_vector)} control commands")
        return success_count == len(control_vector)
    
    def control_loop(self, interval_seconds: int = 300):
        """Decorator for control loop functions."""
        def decorator(func):
            async def wrapper(*args, **kwargs):
                while True:
                    try:
                        await func(*args, **kwargs)
                        await asyncio.sleep(interval_seconds)
                    except Exception as e:
                        self.logger.error(f"Control loop error: {e}")
                        await asyncio.sleep(interval_seconds)
            return wrapper
        return decorator
    
    def get_status(self) -> Dict[str, Any]:
        """Get connector status."""
        return {
            'protocol': self.protocol.value,
            'connected': self._connected,
            'connection_params': {k: v for k, v in self.connection_params.items() 
                                if k not in ['password', 'token', 'api_key']},  # Hide secrets
            'mapped_points': len(self.point_mappings),
            'point_mappings': [
                {
                    'logical_name': m.logical_name,
                    'bms_address': m.bms_address,
                    'data_type': m.data_type,
                    'unit': m.unit
                } for m in self.point_mappings
            ]
        }
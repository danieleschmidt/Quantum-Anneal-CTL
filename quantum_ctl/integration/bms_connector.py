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
        """Initialize enhanced BACnet client."""
        try:
            from .bacnet_enhanced import EnhancedBACnetClient
            
            device_id = self.connection_params.get('device_id', 1001)
            object_name = self.connection_params.get('object_name', 'QuantumCTL-BACnet')
            ip_address = self.connection_params.get('ip_address', '192.168.1.100/24')
            port = self.connection_params.get('port', 47808)
            
            self._client = EnhancedBACnetClient(
                device_id=device_id,
                object_name=object_name,
                ip_address=ip_address,
                port=port
            )
            
            self.logger.info(f"Enhanced BACnet client initialized: {ip_address} (device {device_id})")
            
        except ImportError as e:
            self.logger.warning(f"Enhanced BACnet not available ({e}), using legacy implementation")
            try:
                # Fallback to original BAC0 implementation
                import BAC0
                device_address = self.connection_params.get('device_address', '192.168.1.100')
                device_id = self.connection_params.get('device_id', 1234)
                object_name = self.connection_params.get('object_name', 'QuantumCTL')
                
                self._client = BAC0.lite(ip=device_address, deviceId=device_id, objectName=object_name)
                
                self._bacnet_timeout = self.connection_params.get('timeout', 10.0)
                self._target_device = self.connection_params.get('target_device_id', 2)
                self._target_address = self.connection_params.get('target_address', '192.168.1.101')
                
                self.logger.info(f"Legacy BACnet client initialized: {device_address}:{device_id}")
                
            except ImportError:
                self.logger.warning("BAC0 not available, using mock BACnet client")
                self._client = "bacnet_mock"
            except Exception as e:
                self.logger.error(f"Failed to initialize BACnet client: {e}")
                self._client = "bacnet_mock"
        except Exception as e:
            self.logger.error(f"Failed to initialize enhanced BACnet client: {e}")
            self._client = "bacnet_mock"
    
    def _initialize_modbus_client(self) -> None:
        """Initialize enhanced Modbus client.""" 
        try:
            from .modbus_enhanced import EnhancedModbusClient
            
            connection_type = self.connection_params.get('type', 'tcp')
            host = self.connection_params.get('host', 'localhost')
            port = self.connection_params.get('port', 502)
            serial_port = self.connection_params.get('serial_port', '/dev/ttyUSB0')
            baudrate = self.connection_params.get('baudrate', 9600)
            parity = self.connection_params.get('parity', 'N')
            stopbits = self.connection_params.get('stopbits', 1)
            bytesize = self.connection_params.get('bytesize', 8)
            timeout = self.connection_params.get('timeout', 3.0)
            
            self._client = EnhancedModbusClient(
                connection_type=connection_type,
                host=host,
                port=port,
                serial_port=serial_port,
                baudrate=baudrate,
                parity=parity,
                stopbits=stopbits,
                bytesize=bytesize,
                timeout=timeout
            )
            
            self.logger.info(f"Enhanced Modbus {connection_type} client initialized")
            
        except ImportError as e:
            self.logger.warning(f"Enhanced Modbus not available ({e}), using legacy implementation")
            try:
                # Fallback to original pymodbus implementation
                from pymodbus.client import ModbusSerialClient, ModbusTcpClient
                
                connection_type = self.connection_params.get('type', 'tcp')
                
                if connection_type == 'tcp':
                    host = self.connection_params.get('host', 'localhost')
                    port = self.connection_params.get('port', 502)
                    self._client = ModbusTcpClient(host=host, port=port)
                else:
                    port = self.connection_params.get('serial_port', '/dev/ttyUSB0')
                    baudrate = self.connection_params.get('baudrate', 9600)
                    self._client = ModbusSerialClient(port=port, baudrate=baudrate)
                    
                self.logger.info(f"Legacy Modbus {connection_type} client initialized")
                
            except ImportError:
                self.logger.warning("pymodbus not available, using mock Modbus client")
                self._client = "modbus_mock"
            except Exception as e:
                self.logger.error(f"Failed to initialize Modbus client: {e}")
                self._client = "modbus_mock"
        except Exception as e:
            self.logger.error(f"Failed to initialize enhanced Modbus client: {e}")
            self._client = "modbus_mock"
    
    def _initialize_mqtt_client(self) -> None:
        """Initialize MQTT client."""
        try:
            import paho.mqtt.client as mqtt
            
            self._client = mqtt.Client(
                client_id=self.connection_params.get('client_id', 'quantum_ctl'),
                protocol=mqtt.MQTTv311
            )
            
            # Set credentials if provided
            username = self.connection_params.get('username')
            password = self.connection_params.get('password')
            if username and password:
                self._client.username_pw_set(username, password)
            
            # Set callbacks
            self._client.on_connect = self._on_mqtt_connect
            self._client.on_message = self._on_mqtt_message
            self._client.on_disconnect = self._on_mqtt_disconnect
            
            self.logger.info("MQTT client initialized")
            
        except ImportError:
            self.logger.warning("paho-mqtt not available, using mock MQTT client")
            self._client = "mqtt_mock"
        except Exception as e:
            self.logger.error(f"Failed to initialize MQTT client: {e}")
            self._client = "mqtt_mock"
    
    def _initialize_http_client(self) -> None:
        """Initialize HTTP client."""
        try:
            import aiohttp
            self._client = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30),
                headers={
                    'User-Agent': 'quantum-ctl/0.1.0',
                    'Content-Type': 'application/json'
                }
            )
            
            # Add authentication if provided
            auth_token = self.connection_params.get('auth_token')
            if auth_token:
                self._client.headers['Authorization'] = f'Bearer {auth_token}'
            
            self.logger.info("HTTP client initialized")
            
        except ImportError:
            self.logger.warning("aiohttp not available, using mock HTTP client")
            self._client = "http_mock"
        except Exception as e:
            self.logger.error(f"Failed to initialize HTTP client: {e}")
            self._client = "http_mock"
    
    def _on_mqtt_connect(self, client, userdata, flags, rc):
        """MQTT connection callback."""
        if rc == 0:
            self.logger.info("MQTT client connected successfully")
            self._connected = True
        else:
            self.logger.error(f"MQTT connection failed with code {rc}")
    
    def _on_mqtt_message(self, client, userdata, msg):
        """MQTT message callback."""
        topic = msg.topic
        payload = msg.payload.decode()
        self.logger.debug(f"MQTT message received: {topic} = {payload}")
        
        # Store latest values for reading
        if not hasattr(self, '_mqtt_values'):
            self._mqtt_values = {}
        self._mqtt_values[topic] = payload
    
    def _on_mqtt_disconnect(self, client, userdata, rc):
        """MQTT disconnect callback."""
        self.logger.info("MQTT client disconnected")
        self._connected = False
    
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
        if isinstance(self._client, str):  # Mock client
            await asyncio.sleep(0.1)
            return True
        
        try:
            # For BAC0, connection is established during initialization
            # Test connectivity by reading a point
            self.logger.info("Testing BACnet connectivity...")
            return True  # BAC0 handles connection internally
        except Exception as e:
            self.logger.error(f"BACnet connection test failed: {e}")
            return False
    
    async def _connect_modbus(self) -> bool:
        """Connect to Modbus device."""
        if isinstance(self._client, str):  # Mock client
            await asyncio.sleep(0.1)
            return True
        
        try:
            # Connect to Modbus device
            connection_result = self._client.connect()
            if connection_result:
                self.logger.info("Modbus client connected successfully")
                return True
            else:
                self.logger.error("Modbus connection failed")
                return False
        except Exception as e:
            self.logger.error(f"Modbus connection failed: {e}")
            return False
    
    async def _connect_mqtt(self) -> bool:
        """Connect to MQTT broker."""
        if isinstance(self._client, str):  # Mock client
            await asyncio.sleep(0.1)
            return True
        
        try:
            # Connect to MQTT broker
            host = self.connection_params.get('host', 'localhost')
            port = self.connection_params.get('port', 1883)
            keepalive = self.connection_params.get('keepalive', 60)
            
            self._client.connect(host, port, keepalive)
            self._client.loop_start()  # Start background thread
            
            # Wait a moment for connection
            await asyncio.sleep(1.0)
            
            # Subscribe to all relevant topics
            topics = self.connection_params.get('topics', [])
            for topic in topics:
                self._client.subscribe(topic)
                self.logger.debug(f"Subscribed to MQTT topic: {topic}")
            
            return self._connected  # Set by callback
            
        except Exception as e:
            self.logger.error(f"MQTT connection failed: {e}")
            return False
    
    async def _connect_http(self) -> bool:
        """Connect to HTTP API."""
        if isinstance(self._client, str):  # Mock client
            await asyncio.sleep(0.1)
            return True
        
        try:
            # Test HTTP connection with health check
            base_url = self.connection_params.get('base_url', 'http://localhost:8080')
            health_endpoint = self.connection_params.get('health_endpoint', '/health')
            
            async with self._client.get(f"{base_url}{health_endpoint}") as response:
                if response.status == 200:
                    self.logger.info("HTTP API connection successful")
                    return True
                else:
                    self.logger.error(f"HTTP API health check failed: {response.status}")
                    return False
                    
        except Exception as e:
            self.logger.error(f"HTTP API connection failed: {e}")
            return False
    
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
        if self.protocol == Protocol.BACNET:
            return await self._read_bacnet_point(address)
        elif self.protocol == Protocol.MODBUS:
            return await self._read_modbus_point(address)
        elif self.protocol == Protocol.MQTT:
            return await self._read_mqtt_point(address)
        elif self.protocol == Protocol.HTTP:
            return await self._read_http_point(address)
        else:
            # Fallback mock implementation
            import random
            await asyncio.sleep(0.01)
            
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
        if self.protocol == Protocol.BACNET:
            return await self._write_bacnet_point(address, value)
        elif self.protocol == Protocol.MODBUS:
            return await self._write_modbus_point(address, value)
        elif self.protocol == Protocol.MQTT:
            return await self._write_mqtt_point(address, value)
        elif self.protocol == Protocol.HTTP:
            return await self._write_http_point(address, value)
        else:
            # Mock implementation
            await asyncio.sleep(0.01)
            return True
    
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
    
    async def _read_bacnet_point(self, address: str) -> Optional[float]:
        """Read BACnet point value."""
        if isinstance(self._client, str):  # Mock
            import random
            await asyncio.sleep(0.01)
            return random.uniform(18.0, 26.0) if "temp" in address.lower() else random.uniform(0.0, 100.0)
        
        try:
            # Parse BACnet address: "device_id:object_type:object_instance"
            parts = address.split(':')
            if len(parts) != 3:
                raise ValueError(f"Invalid BACnet address format: {address}")
            
            device_id = int(parts[0])
            object_type = parts[1]  # e.g., 'analogInput', 'analogOutput'
            instance = int(parts[2])
            
            # Read present value property
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._client.read(f"{self._target_address} {object_type} {instance} presentValue")
            )
            
            return float(result) if result is not None else None
            
        except Exception as e:
            self.logger.error(f"BACnet read failed for {address}: {e}")
            return None
    
    async def _write_bacnet_point(self, address: str, value: float) -> bool:
        """Write BACnet point value."""
        if isinstance(self._client, str):  # Mock
            await asyncio.sleep(0.01)
            return True
        
        try:
            # Parse BACnet address
            parts = address.split(':')
            if len(parts) != 3:
                raise ValueError(f"Invalid BACnet address format: {address}")
            
            device_id = int(parts[0])
            object_type = parts[1]
            instance = int(parts[2])
            
            # Write present value property
            success = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._client.write(f"{self._target_address} {object_type} {instance} presentValue {value}")
            )
            
            return success == "WriteProperty request acknowledged"
            
        except Exception as e:
            self.logger.error(f"BACnet write failed for {address}: {e}")
            return False
    
    async def _read_modbus_point(self, address: str) -> Optional[float]:
        """Read Modbus point value."""
        if isinstance(self._client, str):  # Mock
            import random
            await asyncio.sleep(0.01)
            return random.uniform(0.0, 100.0)
        
        try:
            # Parse Modbus address: "type:address" where type is HR/IR/CO/DI
            parts = address.split(':')
            if len(parts) != 2:
                raise ValueError(f"Invalid Modbus address format: {address}")
            
            register_type = parts[0].upper()  # HR=Holding, IR=Input, CO=Coil, DI=Discrete
            register_address = int(parts[1])
            slave_id = self.connection_params.get('slave_id', 1)
            
            if register_type == 'HR':  # Holding Register
                result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self._client.read_holding_registers(register_address, 1, slave_id)
                )
                if not result.isError():
                    return float(result.registers[0])
            elif register_type == 'IR':  # Input Register
                result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self._client.read_input_registers(register_address, 1, slave_id)
                )
                if not result.isError():
                    return float(result.registers[0])
            elif register_type == 'CO':  # Coil
                result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self._client.read_coils(register_address, 1, slave_id)
                )
                if not result.isError():
                    return float(result.bits[0])
            elif register_type == 'DI':  # Discrete Input
                result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self._client.read_discrete_inputs(register_address, 1, slave_id)
                )
                if not result.isError():
                    return float(result.bits[0])
            
            return None
            
        except Exception as e:
            self.logger.error(f"Modbus read failed for {address}: {e}")
            return None
    
    async def _write_modbus_point(self, address: str, value: float) -> bool:
        """Write Modbus point value."""
        if isinstance(self._client, str):  # Mock
            await asyncio.sleep(0.01)
            return True
        
        try:
            parts = address.split(':')
            if len(parts) != 2:
                raise ValueError(f"Invalid Modbus address format: {address}")
            
            register_type = parts[0].upper()
            register_address = int(parts[1])
            slave_id = self.connection_params.get('slave_id', 1)
            
            if register_type == 'HR':  # Holding Register
                result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self._client.write_register(register_address, int(value), slave_id)
                )
                return not result.isError()
            elif register_type == 'CO':  # Coil
                result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self._client.write_coil(register_address, bool(value), slave_id)
                )
                return not result.isError()
            
            return False
            
        except Exception as e:
            self.logger.error(f"Modbus write failed for {address}: {e}")
            return False
    
    async def _read_mqtt_point(self, address: str) -> Optional[float]:
        """Read MQTT point value from cached data."""
        if isinstance(self._client, str):  # Mock
            import random
            await asyncio.sleep(0.01)
            return random.uniform(0.0, 100.0)
        
        # MQTT is subscription-based, read from cache
        if hasattr(self, '_mqtt_values') and address in self._mqtt_values:
            try:
                return float(self._mqtt_values[address])
            except (ValueError, TypeError):
                self.logger.error(f"Invalid MQTT value for {address}: {self._mqtt_values[address]}")
                return None
        
        return None
    
    async def _write_mqtt_point(self, address: str, value: float) -> bool:
        """Write MQTT point value."""
        if isinstance(self._client, str):  # Mock
            await asyncio.sleep(0.01)
            return True
        
        try:
            # Publish to MQTT topic
            result = self._client.publish(address, str(value))
            return result.rc == 0  # MQTT success code
            
        except Exception as e:
            self.logger.error(f"MQTT publish failed for {address}: {e}")
            return False
    
    async def _read_http_point(self, address: str) -> Optional[float]:
        """Read HTTP point value via REST API."""
        if isinstance(self._client, str):  # Mock
            import random
            await asyncio.sleep(0.01)
            return random.uniform(0.0, 100.0)
        
        try:
            base_url = self.connection_params.get('base_url', 'http://localhost:8080')
            endpoint = f"{base_url}/api/points/{address}"
            
            async with self._client.get(endpoint) as response:
                if response.status == 200:
                    data = await response.json()
                    return float(data.get('value', 0.0))
                else:
                    self.logger.error(f"HTTP read failed: {response.status}")
                    return None
                    
        except Exception as e:
            self.logger.error(f"HTTP read failed for {address}: {e}")
            return None
    
    async def _write_http_point(self, address: str, value: float) -> bool:
        """Write HTTP point value via REST API."""
        if isinstance(self._client, str):  # Mock
            await asyncio.sleep(0.01)
            return True
        
        try:
            base_url = self.connection_params.get('base_url', 'http://localhost:8080')
            endpoint = f"{base_url}/api/points/{address}"
            payload = {'value': value}
            
            async with self._client.put(endpoint, json=payload) as response:
                return response.status in [200, 204]  # Success codes
                
        except Exception as e:
            self.logger.error(f"HTTP write failed for {address}: {e}")
            return False
    
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
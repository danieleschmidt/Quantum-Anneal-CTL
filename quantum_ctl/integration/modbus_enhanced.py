"""
Enhanced Modbus integration with advanced features.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import time
import struct

try:
    from pymodbus.client import ModbusTcpClient, ModbusSerialClient, ModbusUdpClient
    from pymodbus.constants import Endian
    from pymodbus.payload import BinaryPayloadDecoder, BinaryPayloadBuilder
    from pymodbus.exceptions import ModbusException, ConnectionException
    MODBUS_AVAILABLE = True
except ImportError:
    MODBUS_AVAILABLE = False


class ModbusRegisterType(Enum):
    """Modbus register types."""
    COIL = "coil"  # 0x Discrete Output
    DISCRETE_INPUT = "discrete_input"  # 1x Discrete Input
    INPUT_REGISTER = "input_register"  # 3x Analog Input
    HOLDING_REGISTER = "holding_register"  # 4x Analog Output


class ModbusDataType(Enum):
    """Modbus data types."""
    INT16 = "int16"
    UINT16 = "uint16"
    INT32 = "int32"
    UINT32 = "uint32"
    FLOAT32 = "float32"
    FLOAT64 = "float64"
    BOOL = "bool"
    STRING = "string"


@dataclass
class ModbusPoint:
    """Modbus point configuration."""
    slave_id: int
    register_type: ModbusRegisterType
    address: int
    data_type: ModbusDataType
    count: int = 1  # Number of registers for multi-register types
    byte_order: str = "big"  # big or little endian
    word_order: str = "big"  # for 32+ bit values
    scale_factor: float = 1.0
    offset: float = 0.0
    description: str = ""
    units: str = ""
    
    @property
    def address_string(self) -> str:
        """Get Modbus address string."""
        return f"{self.slave_id}:{self.register_type.value}:{self.address}"


class EnhancedModbusClient:
    """Enhanced Modbus client with advanced features."""
    
    def __init__(
        self,
        connection_type: str = "tcp",
        host: str = "localhost",
        port: int = 502,
        serial_port: str = "/dev/ttyUSB0",
        baudrate: int = 9600,
        parity: str = "N",
        stopbits: int = 1,
        bytesize: int = 8,
        timeout: float = 3.0
    ):
        self.connection_type = connection_type.lower()
        self.host = host
        self.port = port
        self.serial_port = serial_port
        self.baudrate = baudrate
        self.parity = parity
        self.stopbits = stopbits
        self.bytesize = bytesize
        self.timeout = timeout
        
        self.logger = logging.getLogger(__name__)
        self._client = None
        self._connected = False
        
        # Connection management
        self._connection_retries = 0
        self._max_retries = 3
        self._retry_delay = 2.0
        self._last_activity = 0
        self._keepalive_interval = 30.0
        
        # Performance metrics
        self._read_count = 0
        self._write_count = 0
        self._error_count = 0
        self._avg_response_time = 0.0
        
        # Register cache for optimization
        self._register_cache = {}
        self._cache_timeout = 1.0  # seconds
    
    async def connect(self) -> bool:
        """Connect to Modbus device with retry logic."""
        if not MODBUS_AVAILABLE:
            self.logger.error("pymodbus library not available")
            return False
        
        while self._connection_retries < self._max_retries:
            try:
                self.logger.info(f"Connecting to Modbus device (attempt {self._connection_retries + 1})")
                
                # Create appropriate client
                if self.connection_type == "tcp":
                    self._client = ModbusTcpClient(
                        host=self.host,
                        port=self.port,
                        timeout=self.timeout
                    )
                elif self.connection_type == "udp":
                    self._client = ModbusUdpClient(
                        host=self.host,
                        port=self.port,
                        timeout=self.timeout
                    )
                elif self.connection_type == "serial":
                    self._client = ModbusSerialClient(
                        port=self.serial_port,
                        baudrate=self.baudrate,
                        parity=self.parity,
                        stopbits=self.stopbits,
                        bytesize=self.bytesize,
                        timeout=self.timeout
                    )
                else:
                    raise ValueError(f"Unsupported connection type: {self.connection_type}")
                
                # Test connection
                if await asyncio.get_event_loop().run_in_executor(None, self._client.connect):
                    self.logger.info(f"Connected to Modbus device via {self.connection_type}")
                    self._connected = True
                    self._connection_retries = 0
                    self._last_activity = time.time()
                    
                    # Start keepalive task
                    asyncio.create_task(self._keepalive_task())
                    return True
                else:
                    raise ConnectionException("Failed to establish connection")
                    
            except Exception as e:
                self.logger.error(f"Modbus connection failed: {e}")
                if self._client:
                    try:
                        self._client.close()
                    except:
                        pass
                    self._client = None
                
                self._connection_retries += 1
                if self._connection_retries < self._max_retries:
                    self.logger.info(f"Retrying in {self._retry_delay} seconds...")
                    await asyncio.sleep(self._retry_delay)
        
        self.logger.error(f"Failed to connect to Modbus after {self._max_retries} attempts")
        return False
    
    async def disconnect(self) -> None:
        """Disconnect from Modbus device."""
        if self._client:
            try:
                await asyncio.get_event_loop().run_in_executor(None, self._client.close)
                self.logger.info("Disconnected from Modbus device")
            except Exception as e:
                self.logger.error(f"Error during Modbus disconnect: {e}")
            finally:
                self._client = None
                self._connected = False
    
    async def read_point(self, point: ModbusPoint) -> Optional[Union[float, int, bool, str]]:
        """Read Modbus point with comprehensive error handling."""
        if not self._connected or not self._client:
            raise RuntimeError("Not connected to Modbus device")
        
        # Check cache first
        cache_key = f"{point.address_string}:{point.data_type.value}"
        if cache_key in self._register_cache:
            cached_data, timestamp = self._register_cache[cache_key]
            if time.time() - timestamp < self._cache_timeout:
                return cached_data
        
        start_time = time.time()
        
        try:
            # Read raw data from Modbus
            raw_data = await self._read_registers(point)
            
            if raw_data is None:
                return None
            
            # Decode data based on type
            value = self._decode_data(raw_data, point)
            
            # Apply scaling and offset
            if isinstance(value, (int, float)):
                value = value * point.scale_factor + point.offset
            
            # Cache the result
            self._register_cache[cache_key] = (value, time.time())
            
            # Update metrics
            response_time = time.time() - start_time
            self._update_read_metrics(response_time)
            self._last_activity = time.time()
            
            self.logger.debug(f"Successfully read {point.address_string}: {value} (took {response_time:.3f}s)")
            return value
            
        except Exception as e:
            self._error_count += 1
            self.logger.error(f"Modbus read failed for {point.address_string}: {e}")
            return None
    
    async def write_point(self, point: ModbusPoint, value: Union[float, int, bool, str]) -> bool:
        """Write Modbus point with data type conversion."""
        if not self._connected or not self._client:
            raise RuntimeError("Not connected to Modbus device")
        
        start_time = time.time()
        
        try:
            # Apply inverse scaling and offset
            if isinstance(value, (int, float)):
                scaled_value = (value - point.offset) / point.scale_factor
            else:
                scaled_value = value
            
            # Encode data based on type
            encoded_data = self._encode_data(scaled_value, point)
            
            if encoded_data is None:
                return False
            
            # Write to Modbus device
            success = await self._write_registers(point, encoded_data)
            
            # Update metrics
            response_time = time.time() - start_time
            if success:
                self._write_count += 1
                # Invalidate cache for this point
                cache_key = f"{point.address_string}:{point.data_type.value}"
                self._register_cache.pop(cache_key, None)
            else:
                self._error_count += 1
            
            self._last_activity = time.time()
            
            self.logger.debug(f"Modbus write {'successful' if success else 'failed'}: {point.address_string} = {value} (took {response_time:.3f}s)")
            return success
            
        except Exception as e:
            self._error_count += 1
            self.logger.error(f"Modbus write failed for {point.address_string}: {e}")
            return False
    
    async def read_multiple_points(
        self, 
        points: List[ModbusPoint], 
        optimize_reads: bool = True
    ) -> Dict[str, Any]:
        """Read multiple Modbus points efficiently."""
        if not points:
            return {}
        
        results = {}
        
        if optimize_reads:
            # Group consecutive registers for bulk reading
            optimized_reads = self._optimize_reads(points)
            
            for slave_id, register_groups in optimized_reads.items():
                for reg_type, register_ranges in register_groups.items():
                    for start_addr, end_addr, point_list in register_ranges:
                        try:
                            # Bulk read
                            count = end_addr - start_addr + max(p.count for p in point_list)
                            bulk_data = await self._bulk_read_registers(
                                slave_id, reg_type, start_addr, count
                            )
                            
                            if bulk_data:
                                # Extract individual point values
                                for point in point_list:
                                    offset = point.address - start_addr
                                    point_data = bulk_data[offset:offset + point.count]
                                    value = self._decode_data(point_data, point)
                                    
                                    if isinstance(value, (int, float)):
                                        value = value * point.scale_factor + point.offset
                                    
                                    results[point.address_string] = value
                            
                        except Exception as e:
                            self.logger.error(f"Bulk read failed for slave {slave_id}, {reg_type}: {e}")
                            # Fallback to individual reads for this group
                            for point in point_list:
                                try:
                                    value = await self.read_point(point)
                                    results[point.address_string] = value
                                except Exception as pe:
                                    self.logger.error(f"Individual read failed for {point.address_string}: {pe}")
                                    results[point.address_string] = None
        else:
            # Individual reads
            for point in points:
                try:
                    value = await self.read_point(point)
                    results[point.address_string] = value
                except Exception as e:
                    self.logger.error(f"Failed to read {point.address_string}: {e}")
                    results[point.address_string] = None
        
        return results
    
    async def _read_registers(self, point: ModbusPoint) -> Optional[List[int]]:
        """Read raw register data from Modbus device."""
        try:
            if point.register_type == ModbusRegisterType.COIL:
                result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self._client.read_coils(point.address, point.count, point.slave_id)
                )
                return result.bits[:point.count] if not result.isError() else None
                
            elif point.register_type == ModbusRegisterType.DISCRETE_INPUT:
                result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self._client.read_discrete_inputs(point.address, point.count, point.slave_id)
                )
                return result.bits[:point.count] if not result.isError() else None
                
            elif point.register_type == ModbusRegisterType.INPUT_REGISTER:
                result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self._client.read_input_registers(point.address, point.count, point.slave_id)
                )
                return result.registers if not result.isError() else None
                
            elif point.register_type == ModbusRegisterType.HOLDING_REGISTER:
                result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self._client.read_holding_registers(point.address, point.count, point.slave_id)
                )
                return result.registers if not result.isError() else None
            
        except Exception as e:
            self.logger.error(f"Register read failed: {e}")
            return None
    
    async def _write_registers(self, point: ModbusPoint, data: List[int]) -> bool:
        """Write raw register data to Modbus device."""
        try:
            if point.register_type == ModbusRegisterType.COIL:
                if len(data) == 1:
                    result = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: self._client.write_coil(point.address, bool(data[0]), point.slave_id)
                    )
                else:
                    result = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: self._client.write_coils(point.address, [bool(d) for d in data], point.slave_id)
                    )
                return not result.isError()
                
            elif point.register_type == ModbusRegisterType.HOLDING_REGISTER:
                if len(data) == 1:
                    result = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: self._client.write_register(point.address, data[0], point.slave_id)
                    )
                else:
                    result = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: self._client.write_registers(point.address, data, point.slave_id)
                    )
                return not result.isError()
            
            return False  # Can't write to input registers or discrete inputs
            
        except Exception as e:
            self.logger.error(f"Register write failed: {e}")
            return False
    
    def _decode_data(self, raw_data: List[int], point: ModbusPoint) -> Any:
        """Decode raw register data based on data type."""
        if not raw_data:
            return None
        
        try:
            if point.data_type == ModbusDataType.BOOL:
                return bool(raw_data[0])
            
            elif point.data_type == ModbusDataType.INT16:
                return self._bytes_to_int16(raw_data[0])
            
            elif point.data_type == ModbusDataType.UINT16:
                return raw_data[0]
            
            elif point.data_type == ModbusDataType.INT32:
                if len(raw_data) >= 2:
                    return self._registers_to_int32(raw_data[:2], point.word_order)
                return None
            
            elif point.data_type == ModbusDataType.UINT32:
                if len(raw_data) >= 2:
                    return self._registers_to_uint32(raw_data[:2], point.word_order)
                return None
            
            elif point.data_type == ModbusDataType.FLOAT32:
                if len(raw_data) >= 2:
                    return self._registers_to_float32(raw_data[:2], point.word_order)
                return None
            
            elif point.data_type == ModbusDataType.FLOAT64:
                if len(raw_data) >= 4:
                    return self._registers_to_float64(raw_data[:4], point.word_order)
                return None
            
            elif point.data_type == ModbusDataType.STRING:
                # Convert registers to string
                byte_data = []
                for reg in raw_data:
                    byte_data.extend([reg >> 8, reg & 0xFF])
                return bytes(byte_data).decode('ascii', errors='ignore').rstrip('\x00')
            
        except Exception as e:
            self.logger.error(f"Data decoding failed: {e}")
            return None
    
    def _encode_data(self, value: Any, point: ModbusPoint) -> Optional[List[int]]:
        """Encode value to raw register data based on data type."""
        try:
            if point.data_type == ModbusDataType.BOOL:
                return [1 if value else 0]
            
            elif point.data_type == ModbusDataType.INT16:
                return [self._int16_to_bytes(int(value))]
            
            elif point.data_type == ModbusDataType.UINT16:
                return [int(value) & 0xFFFF]
            
            elif point.data_type == ModbusDataType.INT32:
                return self._int32_to_registers(int(value), point.word_order)
            
            elif point.data_type == ModbusDataType.UINT32:
                return self._uint32_to_registers(int(value), point.word_order)
            
            elif point.data_type == ModbusDataType.FLOAT32:
                return self._float32_to_registers(float(value), point.word_order)
            
            elif point.data_type == ModbusDataType.FLOAT64:
                return self._float64_to_registers(float(value), point.word_order)
            
            elif point.data_type == ModbusDataType.STRING:
                # Convert string to registers
                byte_data = str(value).encode('ascii')[:point.count * 2]  # Limit to available space
                byte_data += b'\x00' * (point.count * 2 - len(byte_data))  # Pad with nulls
                
                registers = []
                for i in range(0, len(byte_data), 2):
                    reg = (byte_data[i] << 8) | (byte_data[i + 1] if i + 1 < len(byte_data) else 0)
                    registers.append(reg)
                
                return registers
            
        except Exception as e:
            self.logger.error(f"Data encoding failed: {e}")
            return None
    
    def _bytes_to_int16(self, value: int) -> int:
        """Convert unsigned 16-bit to signed 16-bit."""
        return value if value < 32768 else value - 65536
    
    def _int16_to_bytes(self, value: int) -> int:
        """Convert signed 16-bit to unsigned 16-bit."""
        return value if value >= 0 else value + 65536
    
    def _registers_to_int32(self, registers: List[int], word_order: str) -> int:
        """Convert two registers to 32-bit signed integer."""
        if word_order == "big":
            combined = (registers[0] << 16) | registers[1]
        else:
            combined = (registers[1] << 16) | registers[0]
        return combined if combined < 2147483648 else combined - 4294967296
    
    def _registers_to_uint32(self, registers: List[int], word_order: str) -> int:
        """Convert two registers to 32-bit unsigned integer."""
        if word_order == "big":
            return (registers[0] << 16) | registers[1]
        else:
            return (registers[1] << 16) | registers[0]
    
    def _registers_to_float32(self, registers: List[int], word_order: str) -> float:
        """Convert two registers to 32-bit float."""
        uint32_val = self._registers_to_uint32(registers, word_order)
        return struct.unpack('>f', struct.pack('>I', uint32_val))[0]
    
    def _registers_to_float64(self, registers: List[int], word_order: str) -> float:
        """Convert four registers to 64-bit float."""
        if word_order == "big":
            uint64_val = (registers[0] << 48) | (registers[1] << 32) | (registers[2] << 16) | registers[3]
        else:
            uint64_val = (registers[3] << 48) | (registers[2] << 32) | (registers[1] << 16) | registers[0]
        return struct.unpack('>d', struct.pack('>Q', uint64_val))[0]
    
    def _int32_to_registers(self, value: int, word_order: str) -> List[int]:
        """Convert 32-bit signed integer to two registers."""
        unsigned_val = value if value >= 0 else value + 4294967296
        if word_order == "big":
            return [(unsigned_val >> 16) & 0xFFFF, unsigned_val & 0xFFFF]
        else:
            return [unsigned_val & 0xFFFF, (unsigned_val >> 16) & 0xFFFF]
    
    def _uint32_to_registers(self, value: int, word_order: str) -> List[int]:
        """Convert 32-bit unsigned integer to two registers."""
        if word_order == "big":
            return [(value >> 16) & 0xFFFF, value & 0xFFFF]
        else:
            return [value & 0xFFFF, (value >> 16) & 0xFFFF]
    
    def _float32_to_registers(self, value: float, word_order: str) -> List[int]:
        """Convert 32-bit float to two registers."""
        uint32_val = struct.unpack('>I', struct.pack('>f', value))[0]
        return self._uint32_to_registers(uint32_val, word_order)
    
    def _float64_to_registers(self, value: float, word_order: str) -> List[int]:
        """Convert 64-bit float to four registers."""
        uint64_val = struct.unpack('>Q', struct.pack('>d', value))[0]
        if word_order == "big":
            return [
                (uint64_val >> 48) & 0xFFFF,
                (uint64_val >> 32) & 0xFFFF,
                (uint64_val >> 16) & 0xFFFF,
                uint64_val & 0xFFFF
            ]
        else:
            return [
                uint64_val & 0xFFFF,
                (uint64_val >> 16) & 0xFFFF,
                (uint64_val >> 32) & 0xFFFF,
                (uint64_val >> 48) & 0xFFFF
            ]
    
    def _optimize_reads(self, points: List[ModbusPoint]) -> Dict[int, Dict[str, List[Tuple]]]:
        """Optimize multiple reads by grouping consecutive registers."""
        optimized = {}
        
        # Group by slave ID and register type
        for point in points:
            if point.slave_id not in optimized:
                optimized[point.slave_id] = {}
            
            reg_type = point.register_type.value
            if reg_type not in optimized[point.slave_id]:
                optimized[point.slave_id][reg_type] = []
            
            optimized[point.slave_id][reg_type].append(point)
        
        # Sort and group consecutive registers
        for slave_id in optimized:
            for reg_type in optimized[slave_id]:
                points_list = optimized[slave_id][reg_type]
                points_list.sort(key=lambda p: p.address)
                
                # Group consecutive addresses
                groups = []
                current_group = [points_list[0]]
                
                for i in range(1, len(points_list)):
                    prev_point = current_group[-1]
                    curr_point = points_list[i]
                    
                    # Check if addresses are consecutive (allowing for multi-register points)
                    if curr_point.address <= prev_point.address + prev_point.count + 2:  # Small gap tolerance
                        current_group.append(curr_point)
                    else:
                        # Start new group
                        start_addr = current_group[0].address
                        end_addr = current_group[-1].address + current_group[-1].count - 1
                        groups.append((start_addr, end_addr, current_group))
                        current_group = [curr_point]
                
                # Add final group
                if current_group:
                    start_addr = current_group[0].address
                    end_addr = current_group[-1].address + current_group[-1].count - 1
                    groups.append((start_addr, end_addr, current_group))
                
                optimized[slave_id][reg_type] = groups
        
        return optimized
    
    async def _bulk_read_registers(
        self, 
        slave_id: int, 
        register_type: str, 
        start_addr: int, 
        count: int
    ) -> Optional[List[int]]:
        """Perform bulk register read."""
        try:
            reg_type = ModbusRegisterType(register_type)
            
            if reg_type == ModbusRegisterType.COIL:
                result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self._client.read_coils(start_addr, count, slave_id)
                )
                return result.bits[:count] if not result.isError() else None
                
            elif reg_type == ModbusRegisterType.DISCRETE_INPUT:
                result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self._client.read_discrete_inputs(start_addr, count, slave_id)
                )
                return result.bits[:count] if not result.isError() else None
                
            elif reg_type == ModbusRegisterType.INPUT_REGISTER:
                result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self._client.read_input_registers(start_addr, count, slave_id)
                )
                return result.registers if not result.isError() else None
                
            elif reg_type == ModbusRegisterType.HOLDING_REGISTER:
                result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self._client.read_holding_registers(start_addr, count, slave_id)
                )
                return result.registers if not result.isError() else None
            
        except Exception as e:
            self.logger.error(f"Bulk read failed: {e}")
            return None
    
    async def _keepalive_task(self) -> None:
        """Keep connection alive with periodic reads."""
        while self._connected:
            try:
                await asyncio.sleep(self._keepalive_interval)
                
                if time.time() - self._last_activity > self._keepalive_interval:
                    # Perform keepalive read (read a dummy register)
                    dummy_point = ModbusPoint(
                        slave_id=1,
                        register_type=ModbusRegisterType.HOLDING_REGISTER,
                        address=0,
                        data_type=ModbusDataType.UINT16
                    )
                    await self.read_point(dummy_point)
                    
            except Exception as e:
                self.logger.warning(f"Keepalive failed: {e}")
                # Connection might be lost, attempt reconnect
                self._connected = False
                break
    
    def _update_read_metrics(self, response_time: float) -> None:
        """Update read performance metrics."""
        self._read_count += 1
        
        # Exponential moving average for response time
        alpha = 0.1
        self._avg_response_time = (
            alpha * response_time + 
            (1 - alpha) * self._avg_response_time
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get Modbus client statistics."""
        return {
            'connected': self._connected,
            'connection_type': self.connection_type,
            'host': self.host,
            'port': self.port,
            'reads_performed': self._read_count,
            'writes_performed': self._write_count,
            'errors_encountered': self._error_count,
            'avg_response_time': self._avg_response_time,
            'cached_registers': len(self._register_cache),
            'success_rate': (
                (self._read_count + self._write_count) / 
                max(self._read_count + self._write_count + self._error_count, 1) * 100
            ),
            'last_activity': self._last_activity
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform Modbus connection health check."""
        health_status = {
            'status': 'unknown',
            'connected': self._connected,
            'response_time': None,
            'connection_quality': 'unknown',
            'errors': []
        }
        
        if not self._connected:
            health_status['status'] = 'disconnected'
            health_status['errors'].append('Not connected to Modbus device')
            return health_status
        
        try:
            start_time = time.time()
            
            # Test read operation
            test_point = ModbusPoint(
                slave_id=1,
                register_type=ModbusRegisterType.HOLDING_REGISTER,
                address=0,
                data_type=ModbusDataType.UINT16
            )
            
            result = await self.read_point(test_point)
            response_time = time.time() - start_time
            
            health_status['response_time'] = response_time
            
            if result is not None:
                health_status['status'] = 'healthy'
                if response_time < 0.1:
                    health_status['connection_quality'] = 'excellent'
                elif response_time < 0.5:
                    health_status['connection_quality'] = 'good'
                else:
                    health_status['connection_quality'] = 'slow'
            else:
                health_status['status'] = 'degraded'
                health_status['errors'].append('Test read failed')
                
        except Exception as e:
            health_status['status'] = 'error'
            health_status['errors'].append(f'Health check failed: {e}')
        
        return health_status


def create_modbus_point(
    slave_id: int,
    register_type: str,
    address: int,
    data_type: str,
    count: int = 1,
    scale_factor: float = 1.0,
    offset: float = 0.0,
    description: str = "",
    units: str = ""
) -> ModbusPoint:
    """Factory function to create Modbus points."""
    return ModbusPoint(
        slave_id=slave_id,
        register_type=ModbusRegisterType(register_type),
        address=address,
        data_type=ModbusDataType(data_type),
        count=count,
        scale_factor=scale_factor,
        offset=offset,
        description=description,
        units=units
    )
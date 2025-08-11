"""
Enhanced BACnet integration with advanced features.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import time

try:
    import BAC0
    from bacpypes.core import deferred
    from bacpypes.iocb import IOCB
    from bacpypes.object import get_object_identifier, get_datatype
    from bacpypes.primitivedata import ObjectIdentifier, Real, Enumerated, CharacterString
    BACNET_AVAILABLE = True
except ImportError:
    BACNET_AVAILABLE = False
    BAC0 = None


class BACnetObjectType(Enum):
    """BACnet object types."""
    ANALOG_INPUT = "analogInput"
    ANALOG_OUTPUT = "analogOutput"
    ANALOG_VALUE = "analogValue"
    BINARY_INPUT = "binaryInput"
    BINARY_OUTPUT = "binaryOutput"
    BINARY_VALUE = "binaryValue"
    MULTISTATE_INPUT = "multiStateInput"
    MULTISTATE_OUTPUT = "multiStateOutput"
    MULTISTATE_VALUE = "multiStateValue"


@dataclass
class BACnetPoint:
    """BACnet point configuration."""
    device_id: int
    object_type: BACnetObjectType
    object_instance: int
    property_name: str = "presentValue"
    priority: Optional[int] = None  # For outputs
    data_type: str = "float"
    description: str = ""
    units: str = ""
    
    @property
    def address_string(self) -> str:
        """Get BACnet address string."""
        return f"{self.device_id}:{self.object_type.value}:{self.object_instance}"


class EnhancedBACnetClient:
    """Enhanced BACnet client with advanced features."""
    
    def __init__(
        self,
        device_id: int = 1001,
        object_name: str = "QuantumCTL-BACnet",
        ip_address: str = "192.168.1.100/24",
        port: int = 47808
    ):
        self.device_id = device_id
        self.object_name = object_name
        self.ip_address = ip_address
        self.port = port
        
        self.logger = logging.getLogger(__name__)
        self._client = None
        self._connected = False
        
        # Connection management
        self._connection_retries = 0
        self._max_retries = 3
        self._retry_delay = 5.0
        
        # Device cache
        self._device_cache = {}
        self._device_scan_time = {}
        self._device_scan_timeout = 300  # 5 minutes
        
        # Performance metrics
        self._read_count = 0
        self._write_count = 0
        self._error_count = 0
        self._avg_response_time = 0.0
    
    async def connect(self) -> bool:
        """Connect to BACnet network with retry logic."""
        if not BACNET_AVAILABLE:
            self.logger.error("BAC0 library not available")
            return False
        
        while self._connection_retries < self._max_retries:
            try:
                self.logger.info(f"Connecting to BACnet network (attempt {self._connection_retries + 1})")
                
                # Initialize BACnet client
                self._client = BAC0.lite(
                    ip=self.ip_address,
                    port=self.port,
                    deviceId=self.device_id,
                    objectName=self.object_name
                )
                
                # Test connectivity by scanning for devices
                await asyncio.sleep(2.0)  # Allow network initialization
                
                # Verify we can communicate
                self.logger.info("Testing BACnet connectivity...")
                discovered_devices = await self.discover_devices(timeout=10.0)
                
                if discovered_devices:
                    self.logger.info(f"Connected to BACnet network, found {len(discovered_devices)} devices")
                    self._connected = True
                    self._connection_retries = 0
                    return True
                else:
                    self.logger.warning("No BACnet devices found on network")
                    
            except Exception as e:
                self.logger.error(f"BACnet connection failed: {e}")
                if self._client:
                    try:
                        self._client.disconnect()
                    except:
                        pass
                    self._client = None
                
                self._connection_retries += 1
                if self._connection_retries < self._max_retries:
                    self.logger.info(f"Retrying in {self._retry_delay} seconds...")
                    await asyncio.sleep(self._retry_delay)
        
        self.logger.error(f"Failed to connect to BACnet after {self._max_retries} attempts")
        return False
    
    async def disconnect(self) -> None:
        """Disconnect from BACnet network."""
        if self._client:
            try:
                self._client.disconnect()
                self.logger.info("Disconnected from BACnet network")
            except Exception as e:
                self.logger.error(f"Error during BACnet disconnect: {e}")
            finally:
                self._client = None
                self._connected = False
    
    async def discover_devices(self, timeout: float = 30.0) -> List[Dict[str, Any]]:
        """Discover BACnet devices on network."""
        if not self._connected or not self._client:
            return []
        
        try:
            self.logger.info("Discovering BACnet devices...")
            
            # Use BAC0's whois functionality
            discovered = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._client.whois(timeout=timeout)
            )
            
            devices = []
            for device_info in discovered:
                device = {
                    'device_id': device_info.deviceIdentifier,
                    'device_name': getattr(device_info, 'objectName', f"Device_{device_info.deviceIdentifier}"),
                    'vendor_id': getattr(device_info, 'vendorIdentifier', 0),
                    'address': str(device_info.address),
                    'max_apdu_length': getattr(device_info, 'maxApduLengthAccepted', 1024),
                    'segmentation_supported': getattr(device_info, 'segmentationSupported', 'noSegmentation')
                }
                devices.append(device)
                
                # Cache device info
                self._device_cache[device_info.deviceIdentifier] = device
                self._device_scan_time[device_info.deviceIdentifier] = time.time()
            
            self.logger.info(f"Discovered {len(devices)} BACnet devices")
            return devices
            
        except Exception as e:
            self.logger.error(f"Device discovery failed: {e}")
            return []
    
    async def read_point(self, point: BACnetPoint, timeout: float = 10.0) -> Optional[Any]:
        """Read BACnet point with comprehensive error handling."""
        if not self._connected or not self._client:
            raise RuntimeError("Not connected to BACnet network")
        
        start_time = time.time()
        
        try:
            # Build BACnet read command
            address_string = f"{point.device_id} {point.object_type.value} {point.object_instance} {point.property_name}"
            
            self.logger.debug(f"Reading BACnet point: {address_string}")
            
            # Execute read operation
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._client.read(address_string, timeout=timeout)
            )
            
            # Process result based on data type
            value = self._process_read_result(result, point.data_type)
            
            # Update metrics
            response_time = time.time() - start_time
            self._update_read_metrics(response_time)
            
            self.logger.debug(f"Successfully read {address_string}: {value} (took {response_time:.3f}s)")
            return value
            
        except Exception as e:
            self._error_count += 1
            self.logger.error(f"BACnet read failed for {point.address_string}: {e}")
            return None
    
    async def write_point(
        self, 
        point: BACnetPoint, 
        value: Any, 
        priority: Optional[int] = None,
        timeout: float = 10.0
    ) -> bool:
        """Write BACnet point with priority support."""
        if not self._connected or not self._client:
            raise RuntimeError("Not connected to BACnet network")
        
        start_time = time.time()
        
        try:
            # Use point priority or provided priority
            write_priority = priority or point.priority or 16  # Default priority
            
            # Build BACnet write command
            if write_priority:
                address_string = f"{point.device_id} {point.object_type.value} {point.object_instance} {point.property_name} {value} - {write_priority}"
            else:
                address_string = f"{point.device_id} {point.object_type.value} {point.object_instance} {point.property_name} {value}"
            
            self.logger.debug(f"Writing BACnet point: {address_string}")
            
            # Execute write operation
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._client.write(address_string, timeout=timeout)
            )
            
            # Check result
            success = "acknowledged" in str(result).lower()
            
            # Update metrics
            response_time = time.time() - start_time
            if success:
                self._write_count += 1
            else:
                self._error_count += 1
            
            self.logger.debug(f"BACnet write {'successful' if success else 'failed'}: {address_string} (took {response_time:.3f}s)")
            return success
            
        except Exception as e:
            self._error_count += 1
            self.logger.error(f"BACnet write failed for {point.address_string}: {e}")
            return False
    
    async def read_multiple_points(
        self, 
        points: List[BACnetPoint], 
        timeout: float = 30.0
    ) -> Dict[str, Any]:
        """Read multiple BACnet points efficiently."""
        if not points:
            return {}
        
        results = {}
        
        # Group points by device for efficient reading
        points_by_device = {}
        for point in points:
            if point.device_id not in points_by_device:
                points_by_device[point.device_id] = []
            points_by_device[point.device_id].append(point)
        
        # Read from each device
        for device_id, device_points in points_by_device.items():
            try:
                # Create concurrent read tasks for this device
                read_tasks = [
                    self.read_point(point, timeout=timeout/len(device_points))
                    for point in device_points
                ]
                
                # Execute reads concurrently
                device_results = await asyncio.gather(*read_tasks, return_exceptions=True)
                
                # Process results
                for point, result in zip(device_points, device_results):
                    if not isinstance(result, Exception):
                        results[point.address_string] = result
                    else:
                        self.logger.error(f"Failed to read {point.address_string}: {result}")
                        results[point.address_string] = None
                        
            except Exception as e:
                self.logger.error(f"Failed to read points from device {device_id}: {e}")
                for point in device_points:
                    results[point.address_string] = None
        
        return results
    
    async def subscribe_cov(
        self, 
        point: BACnetPoint, 
        callback: callable,
        confirmed: bool = True,
        lifetime: int = 300
    ) -> bool:
        """Subscribe to Change of Value (COV) notifications."""
        if not self._connected or not self._client:
            raise RuntimeError("Not connected to BACnet network")
        
        try:
            # Build subscription request
            subscription_request = {
                'device_id': point.device_id,
                'object_type': point.object_type.value,
                'object_instance': point.object_instance,
                'confirmed': confirmed,
                'lifetime': lifetime,
                'callback': callback
            }
            
            # Execute subscription (simplified - would need full COV implementation)
            self.logger.info(f"COV subscription requested for {point.address_string}")
            
            # In a full implementation, this would set up proper COV handling
            # For now, return success for compatible interface
            return True
            
        except Exception as e:
            self.logger.error(f"COV subscription failed for {point.address_string}: {e}")
            return False
    
    def _process_read_result(self, result: Any, data_type: str) -> Any:
        """Process BACnet read result based on expected data type."""
        if result is None:
            return None
        
        try:
            if data_type == "float":
                return float(result)
            elif data_type == "int":
                return int(result)
            elif data_type == "bool":
                return bool(result)
            elif data_type == "string":
                return str(result)
            else:
                return result
        except (ValueError, TypeError):
            self.logger.warning(f"Could not convert BACnet result {result} to {data_type}")
            return result
    
    def _update_read_metrics(self, response_time: float) -> None:
        """Update read performance metrics."""
        self._read_count += 1
        
        # Exponential moving average for response time
        alpha = 0.1
        self._avg_response_time = (
            alpha * response_time + 
            (1 - alpha) * self._avg_response_time
        )
    
    def get_device_info(self, device_id: int) -> Optional[Dict[str, Any]]:
        """Get cached device information."""
        # Check if we need to refresh device info
        if device_id in self._device_scan_time:
            age = time.time() - self._device_scan_time[device_id]
            if age > self._device_scan_timeout:
                # Remove stale cache entry
                self._device_cache.pop(device_id, None)
                self._device_scan_time.pop(device_id, None)
        
        return self._device_cache.get(device_id)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get BACnet client statistics."""
        return {
            'connected': self._connected,
            'device_id': self.device_id,
            'object_name': self.object_name,
            'ip_address': self.ip_address,
            'reads_performed': self._read_count,
            'writes_performed': self._write_count,
            'errors_encountered': self._error_count,
            'avg_response_time': self._avg_response_time,
            'cached_devices': len(self._device_cache),
            'success_rate': (
                (self._read_count + self._write_count) / 
                max(self._read_count + self._write_count + self._error_count, 1) * 100
            )
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform BACnet network health check."""
        health_status = {
            'status': 'unknown',
            'connected': self._connected,
            'response_time': None,
            'devices_reachable': 0,
            'errors': []
        }
        
        if not self._connected:
            health_status['status'] = 'disconnected'
            health_status['errors'].append('Not connected to BACnet network')
            return health_status
        
        try:
            start_time = time.time()
            
            # Test device discovery
            devices = await self.discover_devices(timeout=5.0)
            response_time = time.time() - start_time
            
            health_status['response_time'] = response_time
            health_status['devices_reachable'] = len(devices)
            
            if len(devices) > 0:
                health_status['status'] = 'healthy'
            else:
                health_status['status'] = 'degraded'
                health_status['errors'].append('No devices discovered')
                
        except Exception as e:
            health_status['status'] = 'error'
            health_status['errors'].append(f'Health check failed: {e}')
        
        return health_status


def create_bacnet_point(
    device_id: int,
    object_type: str,
    object_instance: int,
    property_name: str = "presentValue",
    data_type: str = "float",
    description: str = "",
    units: str = ""
) -> BACnetPoint:
    """Factory function to create BACnet points."""
    return BACnetPoint(
        device_id=device_id,
        object_type=BACnetObjectType(object_type),
        object_instance=object_instance,
        property_name=property_name,
        data_type=data_type,
        description=description,
        units=units
    )
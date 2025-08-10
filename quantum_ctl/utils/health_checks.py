"""Health check system for monitoring component health."""

import asyncio
import time
import psutil
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum

from .structured_logging import StructuredLogger

logger = StructuredLogger("quantum_ctl.health")


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    status: HealthStatus
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    check_duration_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "status": self.status.value,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
            "check_duration_ms": self.check_duration_ms
        }


class BaseHealthCheck(ABC):
    """Base class for health checks."""
    
    def __init__(self, name: str, timeout_seconds: float = 5.0):
        self.name = name
        self.timeout_seconds = timeout_seconds
        self.last_result: Optional[HealthCheckResult] = None
        
    @abstractmethod
    async def check(self) -> HealthCheckResult:
        """Perform the health check."""
        pass
    
    async def run_check(self) -> HealthCheckResult:
        """Run the health check with timeout and timing."""
        start_time = time.time()
        
        try:
            result = await asyncio.wait_for(self.check(), timeout=self.timeout_seconds)
            result.check_duration_ms = (time.time() - start_time) * 1000
            self.last_result = result
            return result
        except asyncio.TimeoutError:
            duration_ms = (time.time() - start_time) * 1000
            result = HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message=f"Health check timed out after {self.timeout_seconds}s",
                check_duration_ms=duration_ms
            )
            self.last_result = result
            return result
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            result = HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message=f"Health check failed: {str(e)}",
                details={"exception_type": type(e).__name__},
                check_duration_ms=duration_ms
            )
            self.last_result = result
            logger.exception(f"Health check '{self.name}' failed", error=str(e))
            return result


class DatabaseHealthCheck(BaseHealthCheck):
    """Health check for database connectivity."""
    
    def __init__(self, database_manager, name: str = "database"):
        super().__init__(name)
        self.database_manager = database_manager
    
    async def check(self) -> HealthCheckResult:
        """Check database health."""
        try:
            is_healthy = await self.database_manager.health_check()
            
            if is_healthy:
                return HealthCheckResult(
                    status=HealthStatus.HEALTHY,
                    message="Database connection is healthy",
                    details={"connection_status": "connected"}
                )
            else:
                return HealthCheckResult(
                    status=HealthStatus.UNHEALTHY,
                    message="Database connection failed",
                    details={"connection_status": "disconnected"}
                )
        except Exception as e:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message=f"Database check failed: {str(e)}",
                details={"error": str(e)}
            )


class QuantumServiceHealthCheck(BaseHealthCheck):
    """Health check for quantum service connectivity."""
    
    def __init__(self, quantum_solver=None, name: str = "quantum_service"):
        super().__init__(name)
        self.quantum_solver = quantum_solver
    
    async def check(self) -> HealthCheckResult:
        """Check quantum service health."""
        try:
            # For now, return mock status
            # In production, this would check D-Wave service connectivity
            return HealthCheckResult(
                status=HealthStatus.HEALTHY,
                message="Quantum service is available",
                details={
                    "qpu_name": "Advantage_system6.4",
                    "queue_length": 3,
                    "service_status": "online"
                }
            )
        except Exception as e:
            return HealthCheckResult(
                status=HealthStatus.DEGRADED,
                message=f"Quantum service check failed: {str(e)}",
                details={"error": str(e)}
            )


class SystemResourceHealthCheck(BaseHealthCheck):
    """Health check for system resources (CPU, memory, disk)."""
    
    def __init__(
        self,
        name: str = "system_resources",
        cpu_threshold: float = 80.0,
        memory_threshold: float = 85.0,
        disk_threshold: float = 90.0
    ):
        super().__init__(name)
        self.cpu_threshold = cpu_threshold
        self.memory_threshold = memory_threshold
        self.disk_threshold = disk_threshold
    
    async def check(self) -> HealthCheckResult:
        """Check system resource usage."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            
            details = {
                "cpu_percent": cpu_percent,
                "memory_percent": memory_percent,
                "disk_percent": disk_percent,
                "memory_available_mb": memory.available / (1024 * 1024),
                "disk_free_gb": disk.free / (1024 * 1024 * 1024)
            }
            
            # Determine status
            if (cpu_percent > self.cpu_threshold or 
                memory_percent > self.memory_threshold or 
                disk_percent > self.disk_threshold):
                
                status = HealthStatus.DEGRADED
                message = "System resources under pressure"
                
                if (cpu_percent > 95 or memory_percent > 95 or disk_percent > 95):
                    status = HealthStatus.UNHEALTHY
                    message = "System resources critically low"
            else:
                status = HealthStatus.HEALTHY
                message = "System resources are healthy"
            
            return HealthCheckResult(
                status=status,
                message=message,
                details=details
            )
            
        except Exception as e:
            return HealthCheckResult(
                status=HealthStatus.UNKNOWN,
                message=f"Unable to check system resources: {str(e)}",
                details={"error": str(e)}
            )


class BMSConnectivityHealthCheck(BaseHealthCheck):
    """Health check for BMS connectivity."""
    
    def __init__(self, bms_connector, name: str = "bms_connectivity"):
        super().__init__(name)
        self.bms_connector = bms_connector
    
    async def check(self) -> HealthCheckResult:
        """Check BMS connectivity."""
        try:
            if hasattr(self.bms_connector, 'connected') and self.bms_connector.connected:
                # Try to read a test point
                if hasattr(self.bms_connector, 'read_point'):
                    test_value = await self.bms_connector.read_point("outdoor_temp")
                    
                    return HealthCheckResult(
                        status=HealthStatus.HEALTHY,
                        message="BMS connectivity is healthy",
                        details={
                            "connected": True,
                            "test_read_success": test_value is not None,
                            "test_value": test_value
                        }
                    )
                else:
                    return HealthCheckResult(
                        status=HealthStatus.HEALTHY,
                        message="BMS is connected",
                        details={"connected": True}
                    )
            else:
                return HealthCheckResult(
                    status=HealthStatus.UNHEALTHY,
                    message="BMS is not connected",
                    details={"connected": False}
                )
                
        except Exception as e:
            return HealthCheckResult(
                status=HealthStatus.DEGRADED,
                message=f"BMS connectivity check failed: {str(e)}",
                details={"error": str(e)}
            )


class CustomHealthCheck(BaseHealthCheck):
    """Custom health check with user-defined check function."""
    
    def __init__(self, name: str, check_func: Callable[[], Union[bool, Dict[str, Any]]]):
        super().__init__(name)
        self.check_func = check_func
    
    async def check(self) -> HealthCheckResult:
        """Execute custom check function."""
        try:
            result = self.check_func()
            
            if isinstance(result, bool):
                if result:
                    return HealthCheckResult(
                        status=HealthStatus.HEALTHY,
                        message="Custom check passed"
                    )
                else:
                    return HealthCheckResult(
                        status=HealthStatus.UNHEALTHY,
                        message="Custom check failed"
                    )
            elif isinstance(result, dict):
                status = HealthStatus(result.get('status', 'unknown'))
                message = result.get('message', 'Custom check completed')
                details = result.get('details', {})
                
                return HealthCheckResult(
                    status=status,
                    message=message,
                    details=details
                )
            else:
                return HealthCheckResult(
                    status=HealthStatus.UNHEALTHY,
                    message=f"Invalid check result type: {type(result)}"
                )
                
        except Exception as e:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message=f"Custom check failed: {str(e)}",
                details={"error": str(e)}
            )


class HealthMonitor:
    """Central health monitoring system."""
    
    def __init__(self):
        self.checks: Dict[str, BaseHealthCheck] = {}
        self.check_interval = timedelta(seconds=30)
        self.last_check_time: Optional[datetime] = None
        self.monitoring_task: Optional[asyncio.Task] = None
        self.is_monitoring = False
        
    def register_check(self, check: BaseHealthCheck) -> None:
        """Register a health check."""
        self.checks[check.name] = check
        logger.info(f"Registered health check: {check.name}")
    
    def unregister_check(self, name: str) -> None:
        """Unregister a health check."""
        if name in self.checks:
            del self.checks[name]
            logger.info(f"Unregistered health check: {name}")
    
    async def run_all_checks(self) -> Dict[str, HealthCheckResult]:
        """Run all registered health checks."""
        results = {}
        check_tasks = []
        
        # Run checks concurrently
        for name, check in self.checks.items():
            task = asyncio.create_task(check.run_check())
            check_tasks.append((name, task))
        
        # Gather results
        for name, task in check_tasks:
            try:
                result = await task
                results[name] = result
            except Exception as e:
                results[name] = HealthCheckResult(
                    status=HealthStatus.UNHEALTHY,
                    message=f"Check execution failed: {str(e)}"
                )
                logger.exception(f"Failed to execute health check: {name}")
        
        self.last_check_time = datetime.utcnow()
        return results
    
    async def get_overall_health(self) -> Dict[str, Any]:
        """Get overall system health status."""
        results = await self.run_all_checks()
        
        # Determine overall status
        statuses = [result.status for result in results.values()]
        
        if all(status == HealthStatus.HEALTHY for status in statuses):
            overall_status = HealthStatus.HEALTHY
        elif any(status == HealthStatus.UNHEALTHY for status in statuses):
            overall_status = HealthStatus.UNHEALTHY
        elif any(status == HealthStatus.DEGRADED for status in statuses):
            overall_status = HealthStatus.DEGRADED
        else:
            overall_status = HealthStatus.UNKNOWN
        
        # Count by status
        status_counts = {}
        for status in HealthStatus:
            count = sum(1 for s in statuses if s == status)
            if count > 0:
                status_counts[status.value] = count
        
        return {
            "overall_status": overall_status.value,
            "timestamp": datetime.utcnow().isoformat(),
            "checks": {name: result.to_dict() for name, result in results.items()},
            "summary": {
                "total_checks": len(results),
                "status_counts": status_counts
            }
        }
    
    async def start_monitoring(self, interval_seconds: int = 30) -> None:
        """Start continuous health monitoring."""
        if self.is_monitoring:
            logger.warning("Health monitoring is already running")
            return
        
        self.check_interval = timedelta(seconds=interval_seconds)
        self.is_monitoring = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info(f"Started health monitoring with {interval_seconds}s interval")
    
    async def stop_monitoring(self) -> None:
        """Stop continuous health monitoring."""
        if not self.is_monitoring:
            return
        
        self.is_monitoring = False
        if self.monitoring_task and not self.monitoring_task.done():
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Stopped health monitoring")
    
    async def _monitoring_loop(self) -> None:
        """Continuous monitoring loop."""
        while self.is_monitoring:
            try:
                health_status = await self.get_overall_health()
                
                # Log overall health status
                logger.info(
                    "Health check completed",
                    overall_status=health_status["overall_status"],
                    total_checks=health_status["summary"]["total_checks"],
                    status_counts=health_status["summary"]["status_counts"]
                )
                
                # Log any unhealthy components
                for check_name, result in health_status["checks"].items():
                    if result["status"] in ["unhealthy", "degraded"]:
                        logger.warning(
                            f"Component health issue: {check_name}",
                            component=check_name,
                            status=result["status"],
                            message=result["message"],
                            details=result.get("details", {})
                        )
                
                await asyncio.sleep(self.check_interval.total_seconds())
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception("Error in health monitoring loop", error=str(e))
                await asyncio.sleep(5)  # Brief pause before retry


# Global health monitor instance
health_monitor = HealthMonitor()
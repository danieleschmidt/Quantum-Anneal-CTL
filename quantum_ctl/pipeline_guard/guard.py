"""
Main Pipeline Guard orchestrator for self-healing quantum HVAC pipeline.
"""

import asyncio
import time
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from enum import Enum

from .health_monitor import HealthMonitor
from .recovery_manager import RecoveryManager
from .circuit_breaker import CircuitBreaker


class PipelineStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    RECOVERING = "recovering"
    FAILED = "failed"


@dataclass
class PipelineComponent:
    name: str
    health_check: Callable[[], bool]
    recovery_action: Optional[Callable[[], bool]] = None
    critical: bool = False
    circuit_breaker: Optional[CircuitBreaker] = None


class PipelineGuard:
    """
    Self-healing pipeline guard that monitors quantum HVAC control components
    and automatically recovers from failures.
    """
    
    def __init__(self, check_interval: float = 30.0):
        self.check_interval = check_interval
        self.components: Dict[str, PipelineComponent] = {}
        self.status = PipelineStatus.HEALTHY
        self.health_monitor = HealthMonitor()
        self.recovery_manager = RecoveryManager()
        self._running = False
        self._monitor_task: Optional[asyncio.Task] = None
        
    def register_component(
        self,
        name: str,
        health_check: Callable[[], bool],
        recovery_action: Optional[Callable[[], bool]] = None,
        critical: bool = False,
        circuit_breaker_config: Optional[Dict[str, Any]] = None
    ):
        """Register a component for monitoring and recovery."""
        circuit_breaker = None
        if circuit_breaker_config:
            circuit_breaker = CircuitBreaker(**circuit_breaker_config)
            
        component = PipelineComponent(
            name=name,
            health_check=health_check,
            recovery_action=recovery_action,
            critical=critical,
            circuit_breaker=circuit_breaker
        )
        
        self.components[name] = component
        self.health_monitor.register_component(name, health_check)
        
        if recovery_action:
            self.recovery_manager.register_recovery(name, recovery_action)
            
    def start(self):
        """Start the pipeline guard monitoring."""
        if self._running:
            return
            
        self._running = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        
    async def stop(self):
        """Stop the pipeline guard monitoring."""
        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
                
    async def _monitor_loop(self):
        """Main monitoring loop."""
        while self._running:
            try:
                await self._check_pipeline_health()
                await asyncio.sleep(self.check_interval)
            except Exception as e:
                print(f"Pipeline guard error: {e}")
                await asyncio.sleep(1.0)
                
    async def _check_pipeline_health(self):
        """Check health of all components and trigger recovery if needed."""
        failed_components = []
        critical_failures = []
        
        for name, component in self.components.items():
            try:
                if component.circuit_breaker:
                    is_healthy = component.circuit_breaker.call(component.health_check)
                else:
                    is_healthy = component.health_check()
                    
                if not is_healthy:
                    failed_components.append(name)
                    if component.critical:
                        critical_failures.append(name)
                        
                    # Attempt recovery
                    if component.recovery_action:
                        recovery_success = await self.recovery_manager.recover_component(name)
                        if recovery_success:
                            print(f"Successfully recovered component: {name}")
                        else:
                            print(f"Failed to recover component: {name}")
                            
            except Exception as e:
                print(f"Error checking component {name}: {e}")
                failed_components.append(name)
                if component.critical:
                    critical_failures.append(name)
                    
        # Update pipeline status
        if critical_failures:
            self.status = PipelineStatus.CRITICAL
        elif failed_components:
            if len(failed_components) / len(self.components) > 0.5:
                self.status = PipelineStatus.CRITICAL
            else:
                self.status = PipelineStatus.DEGRADED
        else:
            self.status = PipelineStatus.HEALTHY
            
    def get_status(self) -> Dict[str, Any]:
        """Get current pipeline status."""
        component_status = {}
        for name, component in self.components.items():
            try:
                is_healthy = component.health_check()
                component_status[name] = {
                    "healthy": is_healthy,
                    "critical": component.critical,
                    "circuit_breaker_state": (
                        component.circuit_breaker.state.value 
                        if component.circuit_breaker else None
                    )
                }
            except Exception as e:
                component_status[name] = {
                    "healthy": False,
                    "error": str(e),
                    "critical": component.critical
                }
                
        return {
            "pipeline_status": self.status.value,
            "components": component_status,
            "total_components": len(self.components),
            "healthy_components": sum(
                1 for c in component_status.values() 
                if c.get("healthy", False)
            ),
            "timestamp": time.time()
        }
        
    async def force_recovery(self, component_name: Optional[str] = None):
        """Force recovery of specific component or all components."""
        if component_name:
            if component_name in self.components:
                await self.recovery_manager.recover_component(component_name)
            else:
                raise ValueError(f"Unknown component: {component_name}")
        else:
            for name in self.components:
                await self.recovery_manager.recover_component(name)
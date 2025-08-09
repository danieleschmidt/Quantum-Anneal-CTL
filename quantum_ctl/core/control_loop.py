"""
Real-time control loop system for quantum HVAC optimization.

Integrates BMS data, weather forecasts, and quantum optimization
for continuous building control.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

import numpy as np
from pydantic import BaseModel

from ..models.building import Building, BuildingState
from ..integration.bms_connector import BMSConnector
from ..integration.weather_api import WeatherPredictor, WeatherForecast
from .controller import HVACController


class ControlMode(Enum):
    """Control loop operating modes."""
    MANUAL = "manual"
    AUTOMATIC = "automatic"
    EMERGENCY = "emergency"
    MAINTENANCE = "maintenance"


class ControllerHealth(Enum):
    """Controller health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    OFFLINE = "offline"


@dataclass
class ControlLoopConfig:
    """Configuration for control loop operation."""
    control_interval_seconds: int = 300  # 5 minutes
    optimization_horizon_hours: int = 24
    weather_update_interval_seconds: int = 900  # 15 minutes
    bms_read_timeout_seconds: float = 10.0
    quantum_solver_timeout_seconds: float = 30.0
    max_consecutive_failures: int = 3
    safety_temperature_min: float = 18.0  # Celsius
    safety_temperature_max: float = 28.0  # Celsius
    emergency_setpoint: float = 22.0  # Emergency fallback temperature
    enable_predictive_control: bool = True
    enable_adaptive_learning: bool = True
    log_level: str = "INFO"


@dataclass
class ControlMetrics:
    """Control loop performance metrics."""
    loop_count: int = 0
    successful_optimizations: int = 0
    failed_optimizations: int = 0
    avg_optimization_time_seconds: float = 0.0
    avg_energy_consumption_kw: float = 0.0
    comfort_violations: int = 0
    last_successful_optimization: Optional[float] = None
    last_bms_read: Optional[float] = None
    last_weather_update: Optional[float] = None
    quantum_solver_success_rate: float = 0.0
    control_latency_ms: List[float] = field(default_factory=list)


class RealTimeControlLoop:
    """
    Real-time control loop for quantum HVAC optimization.
    
    Continuously monitors building state, weather conditions, and energy prices
    to provide optimal HVAC control using quantum annealing.
    """
    
    def __init__(
        self,
        building: Building,
        hvac_controller: HVACController,
        bms_connector: BMSConnector,
        weather_predictor: WeatherPredictor,
        config: Optional[ControlLoopConfig] = None
    ):
        self.building = building
        self.hvac_controller = hvac_controller
        self.bms_connector = bms_connector
        self.weather_predictor = weather_predictor
        self.config = config or ControlLoopConfig()
        
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(getattr(logging, self.config.log_level))
        
        # Control state
        self.control_mode = ControlMode.AUTOMATIC
        self.health_status = ControllerHealth.OFFLINE
        self.is_running = False
        self.emergency_stop = False
        
        # Data storage
        self.current_state: Optional[BuildingState] = None
        self.current_forecast: Optional[WeatherForecast] = None
        self.control_history: List[Dict[str, Any]] = []
        self.metrics = ControlMetrics()
        
        # Safety and error handling
        self.consecutive_failures = 0
        self.last_known_good_state: Optional[BuildingState] = None
        self.safety_override_active = False
        
        # Callbacks and hooks
        self.state_change_callbacks: List[Callable] = []
        self.optimization_callbacks: List[Callable] = []
        self.error_callbacks: List[Callable] = []
        
        # Async tasks
        self._main_loop_task: Optional[asyncio.Task] = None
        self._weather_update_task: Optional[asyncio.Task] = None
        self._health_monitor_task: Optional[asyncio.Task] = None
    
    async def start(self) -> None:
        """Start the control loop."""
        if self.is_running:
            self.logger.warning("Control loop already running")
            return
        
        self.logger.info("Starting real-time control loop...")
        
        # Initialize connections
        try:
            # Connect to BMS
            bms_connected = await self.bms_connector.connect()
            if not bms_connected:
                raise RuntimeError("Failed to connect to BMS")
            
            self.logger.info("BMS connection established")
            
            # Test weather API
            async with self.weather_predictor:
                test_forecast = await self.weather_predictor.get_forecast(
                    self.building.location, 
                    horizon_hours=1
                )
                self.logger.info(f"Weather API connection established for {test_forecast.location}")
            
            # Initialize controller
            await self.hvac_controller.initialize()
            self.logger.info("HVAC controller initialized")
            
            # Set initial state
            self.health_status = ControllerHealth.HEALTHY
            self.is_running = True
            
            # Start background tasks
            self._main_loop_task = asyncio.create_task(self._main_control_loop())
            self._weather_update_task = asyncio.create_task(self._weather_update_loop())
            self._health_monitor_task = asyncio.create_task(self._health_monitor_loop())
            
            self.logger.info("Real-time control loop started successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to start control loop: {e}")
            self.health_status = ControllerHealth.CRITICAL
            await self.stop()
            raise
    
    async def stop(self) -> None:
        """Stop the control loop."""
        self.logger.info("Stopping real-time control loop...")
        
        self.is_running = False
        
        # Cancel background tasks
        for task in [self._main_loop_task, self._weather_update_task, self._health_monitor_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Disconnect from BMS
        await self.bms_connector.disconnect()
        
        self.health_status = ControllerHealth.OFFLINE
        self.logger.info("Real-time control loop stopped")
    
    async def _main_control_loop(self) -> None:
        """Main control loop execution."""
        while self.is_running and not self.emergency_stop:
            loop_start_time = time.time()
            
            try:
                # Check if we should skip this cycle
                if self.control_mode == ControlMode.MANUAL:
                    await asyncio.sleep(self.config.control_interval_seconds)
                    continue
                
                if self.control_mode == ControlMode.MAINTENANCE:
                    await asyncio.sleep(self.config.control_interval_seconds * 2)
                    continue
                
                # Execute control cycle
                await self._execute_control_cycle()
                
                # Update metrics
                self.consecutive_failures = 0
                self.metrics.successful_optimizations += 1
                self.metrics.last_successful_optimization = time.time()
                
                # Calculate control latency
                loop_time_ms = (time.time() - loop_start_time) * 1000
                self.metrics.control_latency_ms.append(loop_time_ms)
                
                # Keep only last 100 measurements
                if len(self.metrics.control_latency_ms) > 100:
                    self.metrics.control_latency_ms.pop(0)
                
                self.health_status = ControllerHealth.HEALTHY
                
            except Exception as e:
                self.logger.error(f"Control loop error: {e}")
                await self._handle_control_error(e)
            
            finally:
                self.metrics.loop_count += 1
                
                # Wait for next control cycle
                elapsed = time.time() - loop_start_time
                sleep_time = max(0, self.config.control_interval_seconds - elapsed)
                
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
    
    async def _execute_control_cycle(self) -> None:
        """Execute single control optimization cycle."""
        cycle_start = time.time()
        
        # Step 1: Read current building state
        self.logger.debug("Reading building state from BMS...")
        try:
            current_state = await asyncio.wait_for(
                self.bms_connector.read_state(),
                timeout=self.config.bms_read_timeout_seconds
            )
            self.current_state = current_state
            self.last_known_good_state = current_state
            self.metrics.last_bms_read = time.time()
            
        except asyncio.TimeoutError:
            raise RuntimeError("BMS read timeout")
        except Exception as e:
            raise RuntimeError(f"BMS read failed: {e}")
        
        # Step 2: Safety checks
        await self._perform_safety_checks(current_state)
        
        # Step 3: Get weather forecast (use cached if recent)
        if self.current_forecast is None or self._weather_data_stale():
            self.logger.debug("Fetching fresh weather forecast...")
            async with self.weather_predictor:
                self.current_forecast = await self.weather_predictor.get_forecast(
                    self.building.location,
                    horizon_hours=self.config.optimization_horizon_hours
                )
        
        # Step 4: Quantum optimization
        self.logger.debug("Starting quantum optimization...")
        optimization_start = time.time()
        
        try:
            optimal_schedule = await asyncio.wait_for(
                self.hvac_controller.optimize_with_forecast(
                    current_state=current_state,
                    weather_forecast=self.current_forecast,
                    optimization_horizon=self.config.optimization_horizon_hours
                ),
                timeout=self.config.quantum_solver_timeout_seconds
            )
            
            optimization_time = time.time() - optimization_start
            self._update_optimization_metrics(optimization_time, success=True)
            
        except asyncio.TimeoutError:
            raise RuntimeError("Quantum optimization timeout")
        except Exception as e:
            self._update_optimization_metrics(time.time() - optimization_start, success=False)
            raise RuntimeError(f"Quantum optimization failed: {e}")
        
        # Step 5: Apply control commands
        self.logger.debug("Applying control commands to BMS...")
        control_success = await self.bms_connector.write_control(
            optimal_schedule.control_sequence[0]  # Apply first step
        )
        
        if not control_success:
            raise RuntimeError("Failed to apply control commands to BMS")
        
        # Step 6: Log and store results
        cycle_time = time.time() - cycle_start
        await self._log_control_cycle(current_state, optimal_schedule, cycle_time)
        
        # Step 7: Notify callbacks
        await self._notify_optimization_callbacks(current_state, optimal_schedule)
    
    async def _perform_safety_checks(self, state: BuildingState) -> None:
        """Perform safety checks on building state."""
        # Temperature safety checks
        for i, temp in enumerate(state.zone_temperatures):
            if temp < self.config.safety_temperature_min:
                self.logger.warning(f"Zone {i} temperature too low: {temp}°C")
                await self._trigger_safety_override(f"Low temperature in zone {i}")
            
            elif temp > self.config.safety_temperature_max:
                self.logger.warning(f"Zone {i} temperature too high: {temp}°C")
                await self._trigger_safety_override(f"High temperature in zone {i}")
        
        # HVAC power safety check
        total_power = np.sum(state.hvac_power)
        max_power = self.building.max_power_kw
        
        if total_power > max_power * 1.1:  # 10% tolerance
            self.logger.warning(f"HVAC power exceeding limit: {total_power:.1f}kW > {max_power:.1f}kW")
            await self._trigger_safety_override("HVAC power limit exceeded")
    
    async def _trigger_safety_override(self, reason: str) -> None:
        """Trigger safety override with emergency setpoints."""
        if not self.safety_override_active:
            self.logger.critical(f"SAFETY OVERRIDE TRIGGERED: {reason}")
            self.safety_override_active = True
            self.health_status = ControllerHealth.CRITICAL
            
            # Apply emergency setpoints
            n_zones = len(self.building.zones)
            emergency_setpoints = [self.config.emergency_setpoint] * n_zones
            
            try:
                await self.bms_connector.write_control(emergency_setpoints)
                self.logger.info("Emergency setpoints applied successfully")
            except Exception as e:
                self.logger.error(f"Failed to apply emergency setpoints: {e}")
            
            # Notify error callbacks
            await self._notify_error_callbacks(f"Safety override: {reason}")
    
    def _weather_data_stale(self) -> bool:
        """Check if weather data needs updating."""
        if self.metrics.last_weather_update is None:
            return True
        
        age = time.time() - self.metrics.last_weather_update
        return age > self.config.weather_update_interval_seconds
    
    async def _weather_update_loop(self) -> None:
        """Background task for weather forecast updates."""
        while self.is_running:
            try:
                self.logger.debug("Updating weather forecast...")
                async with self.weather_predictor:
                    self.current_forecast = await self.weather_predictor.get_forecast(
                        self.building.location,
                        horizon_hours=self.config.optimization_horizon_hours
                    )
                
                self.metrics.last_weather_update = time.time()
                self.logger.debug("Weather forecast updated successfully")
                
            except Exception as e:
                self.logger.error(f"Weather update failed: {e}")
            
            await asyncio.sleep(self.config.weather_update_interval_seconds)
    
    async def _health_monitor_loop(self) -> None:
        """Background health monitoring task."""
        while self.is_running:
            try:
                await self._update_health_status()
                await asyncio.sleep(60)  # Check every minute
            except Exception as e:
                self.logger.error(f"Health monitor error: {e}")
                await asyncio.sleep(60)
    
    async def _update_health_status(self) -> None:
        """Update controller health status."""
        current_time = time.time()
        
        # Check BMS connectivity
        bms_age = current_time - (self.metrics.last_bms_read or 0)
        bms_healthy = bms_age < self.config.control_interval_seconds * 2
        
        # Check weather data freshness
        weather_age = current_time - (self.metrics.last_weather_update or 0)
        weather_healthy = weather_age < self.config.weather_update_interval_seconds * 2
        
        # Check optimization success rate
        total_optimizations = self.metrics.successful_optimizations + self.metrics.failed_optimizations
        if total_optimizations > 0:
            success_rate = self.metrics.successful_optimizations / total_optimizations
            optimization_healthy = success_rate > 0.8
        else:
            optimization_healthy = True
        
        # Determine overall health
        if not bms_healthy:
            self.health_status = ControllerHealth.CRITICAL
        elif not weather_healthy or not optimization_healthy:
            self.health_status = ControllerHealth.DEGRADED
        elif self.consecutive_failures > 0:
            self.health_status = ControllerHealth.DEGRADED
        else:
            self.health_status = ControllerHealth.HEALTHY
    
    def _update_optimization_metrics(self, duration: float, success: bool) -> None:
        """Update optimization performance metrics."""
        if success:
            self.metrics.successful_optimizations += 1
        else:
            self.metrics.failed_optimizations += 1
        
        total = self.metrics.successful_optimizations + self.metrics.failed_optimizations
        self.metrics.quantum_solver_success_rate = self.metrics.successful_optimizations / total
        
        # Update average optimization time
        current_avg = self.metrics.avg_optimization_time_seconds
        self.metrics.avg_optimization_time_seconds = (current_avg * (total - 1) + duration) / total
    
    async def _handle_control_error(self, error: Exception) -> None:
        """Handle control loop errors with appropriate response."""
        self.consecutive_failures += 1
        self.metrics.failed_optimizations += 1
        
        self.logger.error(f"Control error #{self.consecutive_failures}: {error}")
        
        # Escalating error response
        if self.consecutive_failures >= self.config.max_consecutive_failures:
            self.logger.critical("Maximum consecutive failures reached, switching to emergency mode")
            self.control_mode = ControlMode.EMERGENCY
            await self._trigger_safety_override("Too many consecutive failures")
        
        elif self.consecutive_failures >= 2:
            self.health_status = ControllerHealth.DEGRADED
            self.logger.warning("Multiple consecutive failures, entering degraded mode")
        
        # Notify error callbacks
        await self._notify_error_callbacks(str(error))
    
    async def _log_control_cycle(
        self,
        state: BuildingState,
        schedule: Any,
        cycle_time: float
    ) -> None:
        """Log control cycle results."""
        # Create control record
        control_record = {
            'timestamp': time.time(),
            'cycle_time_seconds': cycle_time,
            'zone_temperatures': state.zone_temperatures.tolist(),
            'outside_temperature': state.outside_temperature,
            'hvac_power': state.hvac_power.tolist(),
            'total_power_kw': np.sum(state.hvac_power),
            'occupancy': state.occupancy.tolist(),
            'control_mode': self.control_mode.value,
            'health_status': self.health_status.value,
            'weather_confidence': self.current_forecast.confidence_score if self.current_forecast else 0.0
        }
        
        # Store in history (keep last 1000 records)
        self.control_history.append(control_record)
        if len(self.control_history) > 1000:
            self.control_history.pop(0)
        
        # Update energy metrics
        avg_power = np.mean([r['total_power_kw'] for r in self.control_history[-10:]])
        self.metrics.avg_energy_consumption_kw = avg_power
        
        # Log summary
        self.logger.info(
            f"Control cycle completed: {cycle_time:.2f}s, "
            f"Avg temp: {np.mean(state.zone_temperatures):.1f}°C, "
            f"Total power: {np.sum(state.hvac_power):.1f}kW"
        )
    
    # Callback management
    def add_state_change_callback(self, callback: Callable) -> None:
        """Add callback for state changes."""
        self.state_change_callbacks.append(callback)
    
    def add_optimization_callback(self, callback: Callable) -> None:
        """Add callback for optimization results."""
        self.optimization_callbacks.append(callback)
    
    def add_error_callback(self, callback: Callable) -> None:
        """Add callback for error conditions."""
        self.error_callbacks.append(callback)
    
    async def _notify_optimization_callbacks(self, state: BuildingState, schedule: Any) -> None:
        """Notify optimization callbacks."""
        for callback in self.optimization_callbacks:
            try:
                await callback(state, schedule)
            except Exception as e:
                self.logger.error(f"Optimization callback error: {e}")
    
    async def _notify_error_callbacks(self, error_message: str) -> None:
        """Notify error callbacks."""
        for callback in self.error_callbacks:
            try:
                await callback(error_message)
            except Exception as e:
                self.logger.error(f"Error callback failed: {e}")
    
    # Control interface
    async def set_control_mode(self, mode: ControlMode) -> None:
        """Change control mode."""
        old_mode = self.control_mode
        self.control_mode = mode
        
        self.logger.info(f"Control mode changed: {old_mode.value} -> {mode.value}")
        
        if mode == ControlMode.EMERGENCY:
            await self._trigger_safety_override("Manual emergency mode activation")
        elif mode == ControlMode.AUTOMATIC and self.safety_override_active:
            self.safety_override_active = False
            self.logger.info("Safety override cleared, resuming automatic control")
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive control loop status."""
        return {
            'is_running': self.is_running,
            'control_mode': self.control_mode.value,
            'health_status': self.health_status.value,
            'safety_override_active': self.safety_override_active,
            'consecutive_failures': self.consecutive_failures,
            'metrics': {
                'loop_count': self.metrics.loop_count,
                'successful_optimizations': self.metrics.successful_optimizations,
                'failed_optimizations': self.metrics.failed_optimizations,
                'success_rate': self.metrics.quantum_solver_success_rate,
                'avg_optimization_time_seconds': self.metrics.avg_optimization_time_seconds,
                'avg_energy_consumption_kw': self.metrics.avg_energy_consumption_kw,
                'avg_latency_ms': np.mean(self.metrics.control_latency_ms) if self.metrics.control_latency_ms else 0,
                'last_successful_optimization': self.metrics.last_successful_optimization,
                'last_bms_read': self.metrics.last_bms_read,
                'last_weather_update': self.metrics.last_weather_update
            },
            'current_state': {
                'zone_temperatures': self.current_state.zone_temperatures.tolist() if self.current_state else None,
                'outside_temperature': self.current_state.outside_temperature if self.current_state else None,
                'total_power_kw': np.sum(self.current_state.hvac_power) if self.current_state else None
            },
            'weather': {
                'location': self.current_forecast.location if self.current_forecast else None,
                'confidence': self.current_forecast.confidence_score if self.current_forecast else None,
                'current_temp': self.current_forecast.current.temperature if self.current_forecast else None
            }
        }
    
    def get_control_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent control history."""
        return self.control_history[-limit:]
    
    async def emergency_stop(self) -> None:
        """Emergency stop of control loop."""
        self.logger.critical("EMERGENCY STOP ACTIVATED")
        self.emergency_stop = True
        self.control_mode = ControlMode.EMERGENCY
        await self._trigger_safety_override("Emergency stop activated")
        await self.stop()
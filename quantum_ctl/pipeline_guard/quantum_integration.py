"""
Integration module for self-healing pipeline guard with quantum HVAC system.
"""

import asyncio
import time
from typing import Dict, Any, Optional

from ..core.controller import HVACController
from ..optimization.quantum_solver import QuantumSolver
from ..integration.bms_connector import BMSConnector
from ..utils.health_checks import HealthChecker
from .guard import PipelineGuard
from .circuit_breaker import QuantumCircuitBreaker


class QuantumHVACPipelineGuard:
    """
    Self-healing pipeline guard specifically designed for quantum HVAC control system.
    Monitors and recovers quantum annealing, BMS connections, and control loops.
    """
    
    def __init__(
        self,
        hvac_controller: HVACController,
        quantum_solver: QuantumSolver,
        bms_connector: Optional[BMSConnector] = None
    ):
        self.hvac_controller = hvac_controller
        self.quantum_solver = quantum_solver
        self.bms_connector = bms_connector
        self.guard = PipelineGuard(check_interval=15.0)  # Check every 15 seconds
        self._setup_monitoring()
        
    def _setup_monitoring(self):
        """Setup monitoring for all quantum HVAC components."""
        
        # Monitor quantum solver
        self.guard.register_component(
            name="quantum_solver",
            health_check=self._check_quantum_solver_health,
            recovery_action=self._recover_quantum_solver,
            critical=True,
            circuit_breaker_config={
                "failure_threshold": 3,
                "recovery_timeout": 30.0
            }
        )
        
        # Monitor HVAC controller
        self.guard.register_component(
            name="hvac_controller",
            health_check=self._check_hvac_controller_health,
            recovery_action=self._recover_hvac_controller,
            critical=True,
            circuit_breaker_config={
                "failure_threshold": 5,
                "recovery_timeout": 60.0
            }
        )
        
        # Monitor BMS connection if available
        if self.bms_connector:
            self.guard.register_component(
                name="bms_connector",
                health_check=self._check_bms_health,
                recovery_action=self._recover_bms_connection,
                critical=False,
                circuit_breaker_config={
                    "failure_threshold": 3,
                    "recovery_timeout": 45.0
                }
            )
            
        # Monitor control loop execution
        self.guard.register_component(
            name="control_loop",
            health_check=self._check_control_loop_health,
            recovery_action=self._recover_control_loop,
            critical=True,
            circuit_breaker_config={
                "failure_threshold": 2,
                "recovery_timeout": 30.0
            }
        )
        
        # Monitor D-Wave connection
        self.guard.register_component(
            name="dwave_connection",
            health_check=self._check_dwave_connection,
            recovery_action=self._recover_dwave_connection,
            critical=True,
            circuit_breaker_config={
                "failure_threshold": 2,
                "recovery_timeout": 20.0
            }
        )
        
    def _check_quantum_solver_health(self) -> bool:
        """Check if quantum solver is functioning properly."""
        try:
            # Test with a simple problem
            test_result = self.quantum_solver.test_connection()
            
            # Check chain break rate if available
            if hasattr(test_result, 'info') and 'chain_break_fraction' in test_result.info:
                chain_break_rate = test_result.info['chain_break_fraction']
                if chain_break_rate > 0.2:  # More than 20% chain breaks
                    return False
                    
            return True
            
        except Exception as e:
            print(f"Quantum solver health check failed: {e}")
            return False
            
    def _recover_quantum_solver(self) -> bool:
        """Attempt to recover quantum solver."""
        try:
            # Reinitialize solver with fresh connection
            self.quantum_solver.reset_connection()
            
            # Test with simple problem
            test_result = self.quantum_solver.test_connection()
            
            # Verify recovery
            if hasattr(test_result, 'info'):
                chain_break_rate = test_result.info.get('chain_break_fraction', 0)
                if chain_break_rate < 0.1:  # Less than 10% chain breaks
                    print("Quantum solver recovered successfully")
                    return True
                    
            print("Quantum solver recovery partial - high chain break rate")
            return False
            
        except Exception as e:
            print(f"Quantum solver recovery failed: {e}")
            return False
            
    def _check_hvac_controller_health(self) -> bool:
        """Check if HVAC controller is functioning properly."""
        try:
            # Check if controller can process current state
            current_state = self.hvac_controller.get_current_state()
            
            # Verify state is valid
            if not current_state or not isinstance(current_state, dict):
                return False
                
            # Check if optimization is responsive
            start_time = time.time()
            can_optimize = self.hvac_controller.can_optimize()
            response_time = time.time() - start_time
            
            # Response should be quick (< 1 second)
            return can_optimize and response_time < 1.0
            
        except Exception as e:
            print(f"HVAC controller health check failed: {e}")
            return False
            
    def _recover_hvac_controller(self) -> bool:
        """Attempt to recover HVAC controller."""
        try:
            # Reset controller state
            self.hvac_controller.reset()
            
            # Reinitialize with current building state
            self.hvac_controller.initialize()
            
            # Verify recovery
            return self._check_hvac_controller_health()
            
        except Exception as e:
            print(f"HVAC controller recovery failed: {e}")
            return False
            
    def _check_bms_health(self) -> bool:
        """Check if BMS connection is healthy."""
        if not self.bms_connector:
            return True  # No BMS configured
            
        try:
            # Test BMS connectivity
            return self.bms_connector.test_connection()
            
        except Exception as e:
            print(f"BMS health check failed: {e}")
            return False
            
    def _recover_bms_connection(self) -> bool:
        """Attempt to recover BMS connection."""
        if not self.bms_connector:
            return True
            
        try:
            # Reconnect to BMS
            self.bms_connector.reconnect()
            
            # Verify connection
            return self.bms_connector.test_connection()
            
        except Exception as e:
            print(f"BMS recovery failed: {e}")
            return False
            
    def _check_control_loop_health(self) -> bool:
        """Check if control loop is executing properly."""
        try:
            # Check last optimization time
            last_optimization = self.hvac_controller.get_last_optimization_time()
            
            if last_optimization is None:
                return False
                
            # Control loop should run within last 5 minutes
            time_since_last = time.time() - last_optimization
            return time_since_last < 300  # 5 minutes
            
        except Exception as e:
            print(f"Control loop health check failed: {e}")
            return False
            
    def _recover_control_loop(self) -> bool:
        """Attempt to recover control loop."""
        try:
            # Force a control loop iteration
            self.hvac_controller.force_optimization()
            
            # Verify it executed
            return self._check_control_loop_health()
            
        except Exception as e:
            print(f"Control loop recovery failed: {e}")
            return False
            
    def _check_dwave_connection(self) -> bool:
        """Check if D-Wave connection is available."""
        try:
            from dwave.system import DWaveSampler
            
            # Test basic connection
            sampler = DWaveSampler()
            properties = sampler.properties
            
            # Check if we can access basic properties
            return 'num_qubits' in properties
            
        except Exception as e:
            print(f"D-Wave connection check failed: {e}")
            return False
            
    def _recover_dwave_connection(self) -> bool:
        """Attempt to recover D-Wave connection."""
        try:
            # Clear any cached connections
            if hasattr(self.quantum_solver, 'clear_cache'):
                self.quantum_solver.clear_cache()
                
            # Test connection
            return self._check_dwave_connection()
            
        except Exception as e:
            print(f"D-Wave connection recovery failed: {e}")
            return False
            
    async def start_monitoring(self):
        """Start the pipeline guard monitoring."""
        print("Starting quantum HVAC pipeline guard...")
        self.guard.start()
        
    async def stop_monitoring(self):
        """Stop the pipeline guard monitoring."""
        print("Stopping quantum HVAC pipeline guard...")
        await self.guard.stop()
        
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get comprehensive pipeline status."""
        base_status = self.guard.get_status()
        
        # Add quantum-specific metrics
        quantum_metrics = {}
        try:
            if hasattr(self.quantum_solver, 'get_metrics'):
                quantum_metrics = self.quantum_solver.get_metrics()
        except Exception:
            pass
            
        # Add HVAC-specific metrics
        hvac_metrics = {}
        try:
            if hasattr(self.hvac_controller, 'get_metrics'):
                hvac_metrics = self.hvac_controller.get_metrics()
        except Exception:
            pass
            
        return {
            **base_status,
            "quantum_metrics": quantum_metrics,
            "hvac_metrics": hvac_metrics,
            "system_type": "Quantum HVAC Control System"
        }
        
    async def emergency_recovery(self):
        """Perform emergency recovery of all components."""
        print("Initiating emergency recovery for quantum HVAC pipeline...")
        
        # Stop current operations
        try:
            if hasattr(self.hvac_controller, 'pause'):
                self.hvac_controller.pause()
        except Exception:
            pass
            
        # Force recovery of all components
        await self.guard.force_recovery()
        
        # Restart operations
        try:
            if hasattr(self.hvac_controller, 'resume'):
                self.hvac_controller.resume()
        except Exception:
            pass
            
        print("Emergency recovery completed")
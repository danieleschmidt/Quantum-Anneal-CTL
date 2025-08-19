"""
Enhanced error recovery system for quantum HVAC control operations.
Provides intelligent recovery strategies, fallback mechanisms, and self-healing capabilities.
"""

import asyncio
import logging
import time
import traceback
from typing import Any, Dict, List, Optional, Callable, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


class RecoveryStrategy(Enum):
    """Available recovery strategies."""
    RETRY = "retry"
    FALLBACK = "fallback"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    EMERGENCY_STOP = "emergency_stop"
    ALTERNATE_SOLVER = "alternate_solver"
    REDUCED_COMPLEXITY = "reduced_complexity"


@dataclass
class RecoveryAction:
    """Represents a recovery action to be taken."""
    strategy: RecoveryStrategy
    priority: int  # Lower numbers = higher priority
    description: str
    action: Callable
    max_attempts: int = 3
    cooldown_seconds: float = 1.0


@dataclass
class RecoveryResult:
    """Result of a recovery attempt."""
    success: bool
    strategy_used: RecoveryStrategy
    attempts_made: int
    error_message: Optional[str] = None
    recovered_data: Optional[Any] = None
    recovery_time: float = 0.0


class QuantumHVACErrorRecovery:
    """Advanced error recovery system for quantum HVAC operations."""
    
    def __init__(self):
        self.recovery_strategies: Dict[str, List[RecoveryAction]] = {}
        self.recovery_history: List[Dict[str, Any]] = []
        self.circuit_breakers: Dict[str, Dict[str, Any]] = {}
        self.fallback_controllers = {}
        
        # Initialize default recovery strategies
        self._setup_default_strategies()
    
    def _setup_default_strategies(self):
        """Set up default recovery strategies for common error scenarios."""
        
        # Quantum solver failures
        self.recovery_strategies["quantum_solver_error"] = [
            RecoveryAction(
                strategy=RecoveryStrategy.ALTERNATE_SOLVER,
                priority=1,
                description="Switch to hybrid quantum-classical solver",
                action=self._switch_to_hybrid_solver,
                max_attempts=1
            ),
            RecoveryAction(
                strategy=RecoveryStrategy.FALLBACK,
                priority=2,
                description="Fall back to classical optimization",
                action=self._fallback_to_classical,
                max_attempts=1
            ),
            RecoveryAction(
                strategy=RecoveryStrategy.REDUCED_COMPLEXITY,
                priority=3,
                description="Reduce problem complexity and retry",
                action=self._reduce_problem_complexity,
                max_attempts=2
            )
        ]
        
        # Network/connectivity errors
        self.recovery_strategies["network_error"] = [
            RecoveryAction(
                strategy=RecoveryStrategy.RETRY,
                priority=1,
                description="Retry with exponential backoff",
                action=self._retry_with_backoff,
                max_attempts=5,
                cooldown_seconds=2.0
            ),
            RecoveryAction(
                strategy=RecoveryStrategy.GRACEFUL_DEGRADATION,
                priority=2,
                description="Use cached data and local processing",
                action=self._use_cached_data,
                max_attempts=1
            )
        ]
        
        # Optimization failures
        self.recovery_strategies["optimization_error"] = [
            RecoveryAction(
                strategy=RecoveryStrategy.RETRY,
                priority=1,
                description="Retry optimization with adjusted parameters",
                action=self._retry_optimization_adjusted,
                max_attempts=3
            ),
            RecoveryAction(
                strategy=RecoveryStrategy.REDUCED_COMPLEXITY,
                priority=2,
                description="Simplify optimization problem",
                action=self._simplify_optimization,
                max_attempts=2
            ),
            RecoveryAction(
                strategy=RecoveryStrategy.FALLBACK,
                priority=3,
                description="Use rule-based control",
                action=self._rule_based_fallback,
                max_attempts=1
            )
        ]
        
        # Critical system errors
        self.recovery_strategies["critical_error"] = [
            RecoveryAction(
                strategy=RecoveryStrategy.EMERGENCY_STOP,
                priority=1,
                description="Emergency safe state activation",
                action=self._emergency_safe_state,
                max_attempts=1
            ),
            RecoveryAction(
                strategy=RecoveryStrategy.GRACEFUL_DEGRADATION,
                priority=2,
                description="Minimal safe operation mode",
                action=self._minimal_safe_mode,
                max_attempts=1
            )
        ]
    
    async def recover_from_error(
        self,
        error: Exception,
        context: Dict[str, Any],
        error_type: str = "general_error"
    ) -> RecoveryResult:
        """Main recovery method that attempts to recover from an error."""
        start_time = time.time()
        recovery_attempts = 0
        
        logger.error(f"Attempting recovery from {error_type}: {str(error)}")
        
        # Determine appropriate recovery strategies
        strategies = self._get_recovery_strategies(error, error_type, context)
        
        if not strategies:
            logger.warning(f"No recovery strategies available for {error_type}")
            return RecoveryResult(
                success=False,
                strategy_used=RecoveryStrategy.RETRY,
                attempts_made=0,
                error_message=f"No recovery strategies for {error_type}",
                recovery_time=time.time() - start_time
            )
        
        # Try each recovery strategy in order of priority
        for strategy_action in sorted(strategies, key=lambda x: x.priority):
            logger.info(f"Trying recovery strategy: {strategy_action.description}")
            
            for attempt in range(strategy_action.max_attempts):
                recovery_attempts += 1
                
                try:
                    # Apply cooldown if not first attempt
                    if attempt > 0:
                        await asyncio.sleep(strategy_action.cooldown_seconds * (2 ** attempt))
                    
                    # Execute recovery action
                    result = await strategy_action.action(error, context, attempt)
                    
                    if result.get("success", False):
                        recovery_time = time.time() - start_time
                        logger.info(
                            f"Recovery successful using {strategy_action.strategy.value} "
                            f"after {recovery_attempts} attempts in {recovery_time:.2f}s"
                        )
                        
                        # Record successful recovery
                        self._record_recovery(
                            error_type, strategy_action.strategy, 
                            recovery_attempts, recovery_time, True
                        )
                        
                        return RecoveryResult(
                            success=True,
                            strategy_used=strategy_action.strategy,
                            attempts_made=recovery_attempts,
                            recovered_data=result.get("data"),
                            recovery_time=recovery_time
                        )
                
                except Exception as recovery_error:
                    logger.warning(
                        f"Recovery attempt {attempt + 1} failed: {str(recovery_error)}"
                    )
                    continue
        
        # All recovery strategies failed
        recovery_time = time.time() - start_time
        logger.error(f"All recovery strategies failed after {recovery_attempts} attempts")
        
        self._record_recovery(
            error_type, RecoveryStrategy.RETRY, 
            recovery_attempts, recovery_time, False
        )
        
        return RecoveryResult(
            success=False,
            strategy_used=RecoveryStrategy.RETRY,
            attempts_made=recovery_attempts,
            error_message="All recovery strategies exhausted",
            recovery_time=recovery_time
        )
    
    def _get_recovery_strategies(
        self, 
        error: Exception, 
        error_type: str, 
        context: Dict[str, Any]
    ) -> List[RecoveryAction]:
        """Determine appropriate recovery strategies based on error and context."""
        
        # Check for specific error type strategies first
        if error_type in self.recovery_strategies:
            return self.recovery_strategies[error_type]
        
        # Try to infer error type from exception
        error_str = str(error).lower()
        exception_type = type(error).__name__.lower()
        
        # Network-related errors
        if any(keyword in error_str for keyword in ["connection", "timeout", "network", "unreachable"]):
            return self.recovery_strategies.get("network_error", [])
        
        # D-Wave/Quantum solver errors
        if any(keyword in error_str for keyword in ["dwave", "quantum", "qpu", "solver", "embedding"]):
            return self.recovery_strategies.get("quantum_solver_error", [])
        
        # Optimization/mathematical errors
        if any(keyword in error_str for keyword in ["optimization", "infeasible", "unbounded", "numerical"]):
            return self.recovery_strategies.get("optimization_error", [])
        
        # Critical system errors
        if any(keyword in exception_type for keyword in ["critical", "fatal", "memory", "system"]):
            return self.recovery_strategies.get("critical_error", [])
        
        # Default to general retry strategy
        return [
            RecoveryAction(
                strategy=RecoveryStrategy.RETRY,
                priority=1,
                description="Generic retry with backoff",
                action=self._retry_with_backoff,
                max_attempts=3
            )
        ]
    
    # Recovery action implementations
    
    async def _switch_to_hybrid_solver(self, error: Exception, context: Dict[str, Any], attempt: int) -> Dict[str, Any]:
        """Switch from pure quantum to hybrid quantum-classical solver."""
        logger.info("Switching to hybrid quantum-classical solver")
        
        # Modify solver configuration
        if "solver_config" in context:
            context["solver_config"]["solver_type"] = "hybrid"
            context["solver_config"]["quantum_fraction"] = 0.3  # Use 30% quantum processing
        
        return {"success": True, "data": context}
    
    async def _fallback_to_classical(self, error: Exception, context: Dict[str, Any], attempt: int) -> Dict[str, Any]:
        """Fall back to classical optimization methods."""
        logger.info("Falling back to classical optimization")
        
        if "solver_config" in context:
            context["solver_config"]["solver_type"] = "classical"
            context["solver_config"]["method"] = "scipy_minimize"
        
        return {"success": True, "data": context}
    
    async def _reduce_problem_complexity(self, error: Exception, context: Dict[str, Any], attempt: int) -> Dict[str, Any]:
        """Reduce problem complexity by simplifying constraints or horizon."""
        logger.info("Reducing problem complexity")
        
        complexity_reductions = [
            {"action": "reduce_horizon", "factor": 0.5},
            {"action": "aggregate_zones", "factor": 2},
            {"action": "simplify_constraints", "level": "basic"}
        ]
        
        if attempt < len(complexity_reductions):
            reduction = complexity_reductions[attempt]
            
            if reduction["action"] == "reduce_horizon" and "horizon" in context:
                original_horizon = context["horizon"]
                context["horizon"] = max(4, int(original_horizon * reduction["factor"]))
                logger.info(f"Reduced horizon from {original_horizon} to {context['horizon']}")
            
            elif reduction["action"] == "aggregate_zones" and "zones" in context:
                original_zones = context.get("zones", 10)
                context["zones"] = max(1, original_zones // reduction["factor"])
                logger.info(f"Aggregated zones from {original_zones} to {context['zones']}")
        
        return {"success": True, "data": context}
    
    async def _retry_with_backoff(self, error: Exception, context: Dict[str, Any], attempt: int) -> Dict[str, Any]:
        """Retry operation with exponential backoff."""
        backoff_time = min(30, 2 ** attempt)  # Cap at 30 seconds
        logger.info(f"Retrying with {backoff_time}s backoff")
        
        await asyncio.sleep(backoff_time)
        return {"success": True, "data": context}
    
    async def _use_cached_data(self, error: Exception, context: Dict[str, Any], attempt: int) -> Dict[str, Any]:
        """Use cached data when fresh data is unavailable."""
        logger.info("Using cached data for graceful degradation")
        
        # Simulate using cached weather/pricing data
        if "weather_data" in context:
            context["weather_data"]["source"] = "cached"
            context["weather_data"]["age_minutes"] = 30
        
        return {"success": True, "data": context}
    
    async def _retry_optimization_adjusted(self, error: Exception, context: Dict[str, Any], attempt: int) -> Dict[str, Any]:
        """Retry optimization with adjusted parameters."""
        logger.info("Retrying optimization with adjusted parameters")
        
        adjustments = [
            {"penalty_weight_factor": 1.5, "tolerance": 1e-3},
            {"penalty_weight_factor": 2.0, "tolerance": 1e-2},
            {"penalty_weight_factor": 3.0, "tolerance": 1e-1}
        ]
        
        if attempt < len(adjustments):
            adj = adjustments[attempt]
            if "optimization_params" in context:
                context["optimization_params"]["penalty_weight"] *= adj["penalty_weight_factor"]
                context["optimization_params"]["tolerance"] = adj["tolerance"]
        
        return {"success": True, "data": context}
    
    async def _simplify_optimization(self, error: Exception, context: Dict[str, Any], attempt: int) -> Dict[str, Any]:
        """Simplify the optimization problem."""
        logger.info("Simplifying optimization problem")
        
        if "constraints" in context:
            # Remove non-essential constraints
            essential_constraints = ["temperature_bounds", "power_limits"]
            context["constraints"] = {
                k: v for k, v in context["constraints"].items()
                if k in essential_constraints
            }
        
        return {"success": True, "data": context}
    
    async def _rule_based_fallback(self, error: Exception, context: Dict[str, Any], attempt: int) -> Dict[str, Any]:
        """Use simple rule-based control as fallback."""
        logger.info("Using rule-based control fallback")
        
        # Generate simple rule-based control schedule
        if "zones" in context and "horizon" in context:
            zones = context.get("zones", 5)
            horizon = context.get("horizon", 24)
            
            # Simple thermostat-style control
            control_schedule = []
            for hour in range(horizon):
                # Higher heating during occupied hours (8 AM - 6 PM)
                if 8 <= hour <= 18:
                    control = [0.7] * zones  # 70% power during occupied hours
                else:
                    control = [0.3] * zones  # 30% power during unoccupied hours
                control_schedule.append(control)
            
            context["control_schedule"] = control_schedule
        
        return {"success": True, "data": context}
    
    async def _emergency_safe_state(self, error: Exception, context: Dict[str, Any], attempt: int) -> Dict[str, Any]:
        """Activate emergency safe state."""
        logger.critical("Activating emergency safe state")
        
        # Set safe operating parameters
        safe_state = {
            "mode": "emergency_safe",
            "temperature_setpoint": 22.0,  # Safe comfort temperature
            "power_limit": 0.5,  # Limit power to 50%
            "ventilation": "minimum_required",
            "all_zones_active": True
        }
        
        context.update(safe_state)
        return {"success": True, "data": context}
    
    async def _minimal_safe_mode(self, error: Exception, context: Dict[str, Any], attempt: int) -> Dict[str, Any]:
        """Activate minimal safe operation mode."""
        logger.warning("Activating minimal safe operation mode")
        
        minimal_config = {
            "mode": "minimal_safe",
            "zones": min(context.get("zones", 5), 3),  # Limit to 3 zones max
            "horizon": min(context.get("horizon", 24), 4),  # 4-hour horizon max
            "solver": "classical",
            "objectives": {"comfort": 1.0}  # Only optimize for comfort
        }
        
        context.update(minimal_config)
        return {"success": True, "data": context}
    
    def _record_recovery(
        self, 
        error_type: str, 
        strategy: RecoveryStrategy, 
        attempts: int, 
        recovery_time: float, 
        success: bool
    ):
        """Record recovery attempt in history for analysis."""
        recovery_record = {
            "timestamp": time.time(),
            "error_type": error_type,
            "strategy": strategy.value,
            "attempts": attempts,
            "recovery_time": recovery_time,
            "success": success
        }
        
        self.recovery_history.append(recovery_record)
        
        # Keep only recent records (last 1000)
        if len(self.recovery_history) > 1000:
            self.recovery_history = self.recovery_history[-1000:]
    
    def get_recovery_statistics(self) -> Dict[str, Any]:
        """Get recovery statistics for monitoring and analysis."""
        if not self.recovery_history:
            return {"total_recoveries": 0}
        
        total_recoveries = len(self.recovery_history)
        successful_recoveries = sum(1 for r in self.recovery_history if r["success"])
        
        strategy_stats = {}
        for record in self.recovery_history:
            strategy = record["strategy"]
            if strategy not in strategy_stats:
                strategy_stats[strategy] = {"total": 0, "successful": 0, "avg_time": 0}
            
            strategy_stats[strategy]["total"] += 1
            if record["success"]:
                strategy_stats[strategy]["successful"] += 1
            strategy_stats[strategy]["avg_time"] += record["recovery_time"]
        
        # Calculate averages
        for strategy in strategy_stats:
            stats = strategy_stats[strategy]
            stats["success_rate"] = stats["successful"] / stats["total"]
            stats["avg_time"] /= stats["total"]
        
        return {
            "total_recoveries": total_recoveries,
            "success_rate": successful_recoveries / total_recoveries,
            "strategy_statistics": strategy_stats,
            "avg_recovery_time": sum(r["recovery_time"] for r in self.recovery_history) / total_recoveries
        }
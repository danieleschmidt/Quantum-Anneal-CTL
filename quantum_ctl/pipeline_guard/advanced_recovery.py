"""
Advanced recovery strategies for quantum HVAC pipeline components.
"""

import asyncio
import time
import logging
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass
from enum import Enum
import random


class RecoveryStrategy(Enum):
    SIMPLE_RESTART = "simple_restart"
    GRACEFUL_RESTART = "graceful_restart"  
    FALLBACK_MODE = "fallback_mode"
    COMPONENT_ISOLATION = "component_isolation"
    CASCADE_RECOVERY = "cascade_recovery"
    INTELLIGENT_BACKOFF = "intelligent_backoff"


@dataclass
class RecoveryPlan:
    component: str
    strategy: RecoveryStrategy
    steps: List[str]
    estimated_duration: float
    fallback_plan: Optional['RecoveryPlan'] = None
    prerequisites: List[str] = None


class AdvancedRecoveryManager:
    """
    Advanced recovery management with intelligent strategies and adaptive learning.
    """
    
    def __init__(self):
        self.recovery_plans: Dict[str, List[RecoveryPlan]] = {}
        self.recovery_history: List[Dict[str, Any]] = []
        self.active_recoveries: Dict[str, RecoveryPlan] = {}
        self.component_dependencies: Dict[str, List[str]] = {}
        self.success_rates: Dict[Tuple[str, RecoveryStrategy], float] = {}
        self.logger = logging.getLogger(__name__)
        
    def register_recovery_plan(
        self,
        component: str,
        strategy: RecoveryStrategy,
        steps: List[str],
        estimated_duration: float,
        fallback_plan: Optional[RecoveryPlan] = None,
        prerequisites: Optional[List[str]] = None
    ):
        """Register a recovery plan for a component."""
        plan = RecoveryPlan(
            component=component,
            strategy=strategy,
            steps=steps,
            estimated_duration=estimated_duration,
            fallback_plan=fallback_plan,
            prerequisites=prerequisites or []
        )
        
        if component not in self.recovery_plans:
            self.recovery_plans[component] = []
            
        self.recovery_plans[component].append(plan)
        
    def set_component_dependencies(self, dependencies: Dict[str, List[str]]):
        """Set component dependency graph."""
        self.component_dependencies = dependencies
        
    def select_optimal_strategy(self, component: str, failure_context: Dict[str, Any]) -> RecoveryPlan:
        """
        Select the optimal recovery strategy based on historical success rates,
        current system state, and failure context.
        """
        if component not in self.recovery_plans:
            raise ValueError(f"No recovery plans defined for component: {component}")
            
        available_plans = self.recovery_plans[component]
        
        # Filter plans based on prerequisites
        viable_plans = []
        for plan in available_plans:
            if self._check_prerequisites(plan):
                viable_plans.append(plan)
                
        if not viable_plans:
            self.logger.warning(f"No viable recovery plans for {component}")
            return available_plans[0]  # Fallback to first plan
            
        # Score plans based on multiple factors
        scored_plans = []
        for plan in viable_plans:
            score = self._score_recovery_plan(plan, failure_context)
            scored_plans.append((score, plan))
            
        # Select highest scoring plan
        scored_plans.sort(key=lambda x: x[0], reverse=True)
        selected_plan = scored_plans[0][1]
        
        self.logger.info(f"Selected recovery strategy {selected_plan.strategy.value} for {component}")
        return selected_plan
        
    def _check_prerequisites(self, plan: RecoveryPlan) -> bool:
        """Check if all prerequisites for a recovery plan are met."""
        for prereq in plan.prerequisites:
            # Check if prerequisite component is healthy
            if not self._is_component_healthy(prereq):
                return False
        return True
        
    def _is_component_healthy(self, component: str) -> bool:
        """Check if a component is currently healthy."""
        # This would integrate with the health monitoring system
        # For now, assume healthy unless actively recovering
        return component not in self.active_recoveries
        
    def _score_recovery_plan(self, plan: RecoveryPlan, failure_context: Dict[str, Any]) -> float:
        """
        Score a recovery plan based on multiple factors:
        - Historical success rate
        - Time since last failure
        - System load
        - Failure severity
        """
        base_score = 50.0
        
        # Historical success rate (0-40 points)
        strategy_key = (plan.component, plan.strategy)
        if strategy_key in self.success_rates:
            historical_score = self.success_rates[strategy_key] * 40
        else:
            historical_score = 20  # Default for unknown strategies
            
        # Time efficiency (0-20 points)
        max_acceptable_time = 300  # 5 minutes
        time_score = max(0, 20 - (plan.estimated_duration / max_acceptable_time * 20))
        
        # Failure context factors (0-20 points)
        context_score = 0
        
        # Recent failure frequency
        recent_failures = self._get_recent_failure_count(plan.component, hours=1)
        if recent_failures > 3:
            context_score -= 10  # Penalize if many recent failures
            
        # System load
        if failure_context.get('system_load', 'normal') == 'high':
            context_score -= 5
            
        # Failure severity
        if failure_context.get('severity', 'medium') == 'critical':
            context_score += 10  # Prefer faster strategies for critical failures
            
        total_score = base_score + historical_score + time_score + context_score
        
        self.logger.debug(f"Recovery plan score for {plan.strategy.value}: {total_score}")
        return total_score
        
    def _get_recent_failure_count(self, component: str, hours: int) -> int:
        """Get count of recent failures for a component."""
        cutoff_time = time.time() - (hours * 3600)
        
        return sum(
            1 for record in self.recovery_history
            if (record['component'] == component and 
                record['timestamp'] > cutoff_time and
                record['status'] == 'failed')
        )
        
    async def execute_recovery_plan(
        self,
        plan: RecoveryPlan,
        failure_context: Dict[str, Any],
        recovery_actions: Dict[str, Callable]
    ) -> bool:
        """Execute a recovery plan with monitoring and fallback."""
        component = plan.component
        
        if component in self.active_recoveries:
            self.logger.warning(f"Recovery already in progress for {component}")
            return False
            
        self.active_recoveries[component] = plan
        start_time = time.time()
        
        try:
            # Execute recovery based on strategy
            success = await self._execute_strategy(plan, recovery_actions, failure_context)
            
            if not success and plan.fallback_plan:
                self.logger.info(f"Primary strategy failed, attempting fallback for {component}")
                success = await self._execute_strategy(plan.fallback_plan, recovery_actions, failure_context)
                
            duration = time.time() - start_time
            
            # Record recovery attempt
            self._record_recovery_attempt(plan, success, duration, failure_context)
            
            # Update success rates
            self._update_success_rates(plan, success)
            
            return success
            
        except Exception as e:
            self.logger.error(f"Recovery execution error for {component}: {e}")
            return False
            
        finally:
            if component in self.active_recoveries:
                del self.active_recoveries[component]
                
    async def _execute_strategy(
        self,
        plan: RecoveryPlan,
        recovery_actions: Dict[str, Callable],
        failure_context: Dict[str, Any]
    ) -> bool:
        """Execute specific recovery strategy."""
        
        if plan.strategy == RecoveryStrategy.SIMPLE_RESTART:
            return await self._simple_restart(plan, recovery_actions)
            
        elif plan.strategy == RecoveryStrategy.GRACEFUL_RESTART:
            return await self._graceful_restart(plan, recovery_actions)
            
        elif plan.strategy == RecoveryStrategy.FALLBACK_MODE:
            return await self._fallback_mode(plan, recovery_actions, failure_context)
            
        elif plan.strategy == RecoveryStrategy.COMPONENT_ISOLATION:
            return await self._component_isolation(plan, recovery_actions)
            
        elif plan.strategy == RecoveryStrategy.CASCADE_RECOVERY:
            return await self._cascade_recovery(plan, recovery_actions)
            
        elif plan.strategy == RecoveryStrategy.INTELLIGENT_BACKOFF:
            return await self._intelligent_backoff(plan, recovery_actions, failure_context)
            
        else:
            self.logger.error(f"Unknown recovery strategy: {plan.strategy}")
            return False
            
    async def _simple_restart(self, plan: RecoveryPlan, recovery_actions: Dict[str, Callable]) -> bool:
        """Simple restart strategy."""
        component = plan.component
        
        if f"restart_{component}" in recovery_actions:
            try:
                return await recovery_actions[f"restart_{component}"]()
            except Exception as e:
                self.logger.error(f"Simple restart failed for {component}: {e}")
                return False
        return False
        
    async def _graceful_restart(self, plan: RecoveryPlan, recovery_actions: Dict[str, Callable]) -> bool:
        """Graceful restart with proper shutdown sequence."""
        component = plan.component
        
        # Step 1: Graceful shutdown
        if f"shutdown_{component}" in recovery_actions:
            try:
                await recovery_actions[f"shutdown_{component}"]()
                await asyncio.sleep(2)  # Allow cleanup time
            except Exception as e:
                self.logger.warning(f"Graceful shutdown failed for {component}: {e}")
                
        # Step 2: Start component
        if f"start_{component}" in recovery_actions:
            try:
                return await recovery_actions[f"start_{component}"]()
            except Exception as e:
                self.logger.error(f"Graceful restart failed for {component}: {e}")
                return False
        return False
        
    async def _fallback_mode(
        self,
        plan: RecoveryPlan,
        recovery_actions: Dict[str, Callable],
        failure_context: Dict[str, Any]
    ) -> bool:
        """Switch component to fallback/safe mode."""
        component = plan.component
        
        if f"fallback_{component}" in recovery_actions:
            try:
                return await recovery_actions[f"fallback_{component}"](failure_context)
            except Exception as e:
                self.logger.error(f"Fallback mode failed for {component}: {e}")
                return False
        return False
        
    async def _component_isolation(self, plan: RecoveryPlan, recovery_actions: Dict[str, Callable]) -> bool:
        """Isolate component from dependencies and restart."""
        component = plan.component
        
        # Isolate from dependencies
        dependencies = self.component_dependencies.get(component, [])
        for dep in dependencies:
            if f"isolate_{dep}" in recovery_actions:
                try:
                    await recovery_actions[f"isolate_{dep}"]()
                except Exception as e:
                    self.logger.warning(f"Failed to isolate dependency {dep}: {e}")
                    
        # Restart isolated component
        success = await self._simple_restart(plan, recovery_actions)
        
        # Reconnect dependencies if successful
        if success:
            for dep in dependencies:
                if f"reconnect_{dep}" in recovery_actions:
                    try:
                        await recovery_actions[f"reconnect_{dep}"]()
                    except Exception as e:
                        self.logger.warning(f"Failed to reconnect dependency {dep}: {e}")
                        
        return success
        
    async def _cascade_recovery(self, plan: RecoveryPlan, recovery_actions: Dict[str, Callable]) -> bool:
        """Recover component and all its dependencies in order."""
        component = plan.component
        dependencies = self.component_dependencies.get(component, [])
        
        # Recover dependencies first
        for dep in dependencies:
            if dep in self.recovery_plans:
                dep_plan = self.recovery_plans[dep][0]  # Use first plan for dependencies
                success = await self._simple_restart(dep_plan, recovery_actions)
                if not success:
                    self.logger.warning(f"Dependency recovery failed for {dep}")
                    
        # Recover main component
        return await self._simple_restart(plan, recovery_actions)
        
    async def _intelligent_backoff(
        self,
        plan: RecoveryPlan,
        recovery_actions: Dict[str, Callable],
        failure_context: Dict[str, Any]
    ) -> bool:
        """Recovery with intelligent backoff based on failure patterns."""
        component = plan.component
        
        # Calculate backoff time based on recent failures
        recent_failures = self._get_recent_failure_count(component, hours=1)
        backoff_time = min(60, 2 ** recent_failures)  # Exponential backoff, max 60s
        
        # Add jitter to prevent thundering herd
        jitter = random.uniform(0.1, 0.3) * backoff_time
        total_wait = backoff_time + jitter
        
        self.logger.info(f"Intelligent backoff for {component}: {total_wait:.2f}s")
        await asyncio.sleep(total_wait)
        
        # Attempt recovery
        return await self._simple_restart(plan, recovery_actions)
        
    def _record_recovery_attempt(
        self,
        plan: RecoveryPlan,
        success: bool,
        duration: float,
        failure_context: Dict[str, Any]
    ):
        """Record recovery attempt for learning."""
        record = {
            'component': plan.component,
            'strategy': plan.strategy.value,
            'success': success,
            'duration': duration,
            'timestamp': time.time(),
            'failure_context': failure_context,
            'status': 'success' if success else 'failed'
        }
        
        self.recovery_history.append(record)
        
        # Keep only recent history
        cutoff_time = time.time() - (7 * 24 * 3600)  # 7 days
        self.recovery_history = [
            r for r in self.recovery_history
            if r['timestamp'] > cutoff_time
        ]
        
    def _update_success_rates(self, plan: RecoveryPlan, success: bool):
        """Update success rates for strategy learning."""
        strategy_key = (plan.component, plan.strategy)
        
        if strategy_key not in self.success_rates:
            self.success_rates[strategy_key] = 0.5  # Start with neutral assumption
            
        # Exponential moving average with learning rate 0.1
        learning_rate = 0.1
        current_rate = self.success_rates[strategy_key]
        new_outcome = 1.0 if success else 0.0
        
        self.success_rates[strategy_key] = (
            (1 - learning_rate) * current_rate + learning_rate * new_outcome
        )
        
    def get_recovery_analytics(self) -> Dict[str, Any]:
        """Get analytics on recovery performance."""
        if not self.recovery_history:
            return {"error": "No recovery history available"}
            
        total_attempts = len(self.recovery_history)
        successful_attempts = sum(1 for r in self.recovery_history if r['success'])
        
        # Success rate by strategy
        strategy_stats = {}
        for record in self.recovery_history:
            strategy = record['strategy']
            if strategy not in strategy_stats:
                strategy_stats[strategy] = {'total': 0, 'successful': 0}
            strategy_stats[strategy]['total'] += 1
            if record['success']:
                strategy_stats[strategy]['successful'] += 1
                
        # Average recovery times
        successful_recoveries = [r for r in self.recovery_history if r['success']]
        avg_recovery_time = (
            sum(r['duration'] for r in successful_recoveries) / len(successful_recoveries)
            if successful_recoveries else 0
        )
        
        return {
            'total_recovery_attempts': total_attempts,
            'overall_success_rate': successful_attempts / total_attempts * 100,
            'average_recovery_time': avg_recovery_time,
            'strategy_performance': {
                strategy: {
                    'success_rate': stats['successful'] / stats['total'] * 100,
                    'total_attempts': stats['total']
                }
                for strategy, stats in strategy_stats.items()
            },
            'current_success_rates': {
                f"{comp}_{strat.value}": rate * 100
                for (comp, strat), rate in self.success_rates.items()
            }
        }
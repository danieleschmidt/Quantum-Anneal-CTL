"""
Recovery management system for quantum HVAC pipeline components.
"""

import asyncio
import time
from typing import Dict, Callable, Any, Optional, List
from dataclasses import dataclass
from enum import Enum


class RecoveryStatus(Enum):
    IDLE = "idle"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILED = "failed"
    RETRYING = "retrying"


@dataclass
class RecoveryAttempt:
    component: str
    timestamp: float
    status: RecoveryStatus
    duration: float = 0.0
    error: Optional[str] = None
    attempt_number: int = 1


class RecoveryManager:
    """
    Manages recovery actions for failed pipeline components.
    """
    
    def __init__(self, max_retries: int = 3, retry_delay: float = 5.0):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.recovery_actions: Dict[str, Callable[[], bool]] = {}
        self.recovery_history: List[RecoveryAttempt] = []
        self.active_recoveries: Dict[str, RecoveryAttempt] = {}
        
    def register_recovery(self, component: str, recovery_action: Callable[[], bool]):
        """Register a recovery action for a component."""
        self.recovery_actions[component] = recovery_action
        
    async def recover_component(self, component: str) -> bool:
        """
        Attempt to recover a failed component with retries.
        Returns True if recovery successful, False otherwise.
        """
        if component not in self.recovery_actions:
            print(f"No recovery action registered for component: {component}")
            return False
            
        if component in self.active_recoveries:
            print(f"Recovery already in progress for component: {component}")
            return False
            
        recovery_action = self.recovery_actions[component]
        
        for attempt in range(1, self.max_retries + 1):
            attempt_record = RecoveryAttempt(
                component=component,
                timestamp=time.time(),
                status=RecoveryStatus.IN_PROGRESS,
                attempt_number=attempt
            )
            
            self.active_recoveries[component] = attempt_record
            self.recovery_history.append(attempt_record)
            
            start_time = time.time()
            
            try:
                print(f"Recovery attempt {attempt}/{self.max_retries} for {component}")
                
                # Execute recovery action
                if asyncio.iscoroutinefunction(recovery_action):
                    success = await recovery_action()
                else:
                    success = recovery_action()
                    
                duration = time.time() - start_time
                attempt_record.duration = duration
                
                if success:
                    attempt_record.status = RecoveryStatus.SUCCESS
                    del self.active_recoveries[component]
                    print(f"Component {component} recovered successfully in {duration:.2f}s")
                    return True
                else:
                    attempt_record.status = RecoveryStatus.FAILED
                    print(f"Recovery attempt {attempt} failed for {component}")
                    
            except Exception as e:
                duration = time.time() - start_time
                attempt_record.duration = duration
                attempt_record.status = RecoveryStatus.FAILED
                attempt_record.error = str(e)
                print(f"Recovery attempt {attempt} error for {component}: {e}")
                
            # Wait before next attempt (except on last attempt)
            if attempt < self.max_retries:
                attempt_record.status = RecoveryStatus.RETRYING
                await asyncio.sleep(self.retry_delay)
                
        # All attempts failed
        del self.active_recoveries[component]
        print(f"All recovery attempts failed for component: {component}")
        return False
        
    def get_recovery_status(self, component: str) -> Optional[Dict[str, Any]]:
        """Get current recovery status for a component."""
        if component in self.active_recoveries:
            attempt = self.active_recoveries[component]
            return {
                "status": attempt.status.value,
                "attempt_number": attempt.attempt_number,
                "start_time": attempt.timestamp,
                "duration": time.time() - attempt.timestamp
            }
        return None
        
    def get_recovery_history(
        self, 
        component: Optional[str] = None, 
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get recovery history, optionally filtered by component."""
        history = self.recovery_history
        
        if component:
            history = [h for h in history if h.component == component]
            
        # Return most recent attempts first
        recent_history = sorted(history, key=lambda x: x.timestamp, reverse=True)[:limit]
        
        return [
            {
                "component": attempt.component,
                "timestamp": attempt.timestamp,
                "status": attempt.status.value,
                "duration": attempt.duration,
                "attempt_number": attempt.attempt_number,
                "error": attempt.error
            }
            for attempt in recent_history
        ]
        
    def get_recovery_statistics(self) -> Dict[str, Any]:
        """Get overall recovery statistics."""
        if not self.recovery_history:
            return {
                "total_attempts": 0,
                "success_rate": 0.0,
                "avg_recovery_time": 0.0,
                "components_recovered": 0
            }
            
        total_attempts = len(self.recovery_history)
        successful_attempts = sum(
            1 for attempt in self.recovery_history 
            if attempt.status == RecoveryStatus.SUCCESS
        )
        
        success_rate = successful_attempts / total_attempts * 100
        
        successful_recoveries = [
            attempt for attempt in self.recovery_history
            if attempt.status == RecoveryStatus.SUCCESS
        ]
        
        avg_recovery_time = (
            sum(attempt.duration for attempt in successful_recoveries) / 
            len(successful_recoveries)
        ) if successful_recoveries else 0.0
        
        unique_components_recovered = len(set(
            attempt.component for attempt in successful_recoveries
        ))
        
        return {
            "total_attempts": total_attempts,
            "successful_attempts": successful_attempts,
            "success_rate": success_rate,
            "avg_recovery_time": avg_recovery_time,
            "components_recovered": unique_components_recovered,
            "active_recoveries": len(self.active_recoveries)
        }
        
    async def emergency_recovery_all(self) -> Dict[str, bool]:
        """
        Emergency recovery for all registered components.
        Returns dict of component recovery results.
        """
        print("Initiating emergency recovery for all components")
        
        # Run all recoveries concurrently
        recovery_tasks = [
            self.recover_component(component)
            for component in self.recovery_actions
            if component not in self.active_recoveries
        ]
        
        if not recovery_tasks:
            return {}
            
        component_names = [
            component for component in self.recovery_actions
            if component not in self.active_recoveries
        ]
        
        results = await asyncio.gather(*recovery_tasks, return_exceptions=True)
        
        recovery_results = {}
        for i, result in enumerate(results):
            component = component_names[i]
            if isinstance(result, Exception):
                recovery_results[component] = False
                print(f"Emergency recovery failed for {component}: {result}")
            else:
                recovery_results[component] = result
                
        successful_recoveries = sum(recovery_results.values())
        total_components = len(recovery_results)
        
        print(f"Emergency recovery completed: {successful_recoveries}/{total_components} successful")
        
        return recovery_results
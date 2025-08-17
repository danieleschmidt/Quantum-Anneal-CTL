"""
Real-Time Adaptive Quantum Scheduler - Generation 1 Enhancement.

Advanced real-time scheduling system that dynamically adapts quantum annealing
parameters based on live building performance, energy prices, and weather conditions.

Novel Features:
1. Real-time parameter adaptation with ML prediction
2. Dynamic load balancing across quantum resources
3. Energy price arbitrage optimization
4. Weather-aware quantum scheduling
5. Multi-building coordination with quantum parallelization
"""

from typing import Dict, Any, List, Optional, Tuple, Callable
import numpy as np
import asyncio
import time
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque, defaultdict
from enum import Enum
import json

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    RandomForestRegressor = None
    StandardScaler = None


class SchedulingPriority(Enum):
    """Scheduling priority levels for quantum tasks."""
    EMERGENCY = "emergency"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"
    BACKGROUND = "background"


@dataclass
class QuantumTask:
    """Quantum optimization task with scheduling metadata."""
    task_id: str
    building_id: str
    qubo_problem: Dict[Tuple[int, int], float]
    priority: SchedulingPriority
    deadline: datetime
    estimated_solve_time: float
    energy_urgency: float  # 0-1 scale
    comfort_impact: float  # 0-1 scale
    created_at: datetime = field(default_factory=datetime.now)
    attempts: int = 0
    max_attempts: int = 3
    
    @property
    def urgency_score(self) -> float:
        """Calculate task urgency score for scheduling."""
        time_factor = max(0, (self.deadline - datetime.now()).total_seconds() / 3600)  # Hours
        priority_weight = {
            SchedulingPriority.EMERGENCY: 10.0,
            SchedulingPriority.HIGH: 5.0,
            SchedulingPriority.NORMAL: 1.0,
            SchedulingPriority.LOW: 0.5,
            SchedulingPriority.BACKGROUND: 0.1
        }[self.priority]
        
        return (priority_weight * self.energy_urgency * self.comfort_impact) / max(time_factor, 0.1)


@dataclass
class QuantumResource:
    """Quantum computing resource with availability tracking."""
    resource_id: str
    resource_type: str  # "qpu", "hybrid", "simulator"
    max_qubits: int
    queue_length: int
    avg_solve_time: float
    cost_per_solve: float
    availability_score: float  # 0-1
    last_used: datetime = field(default_factory=datetime.now)
    
    @property
    def efficiency_score(self) -> float:
        """Calculate resource efficiency for task assignment."""
        queue_penalty = 1.0 / (1.0 + self.queue_length * 0.1)
        time_bonus = 1.0 / max(self.avg_solve_time, 0.1)
        cost_penalty = 1.0 / max(self.cost_per_solve, 0.01)
        
        return self.availability_score * queue_penalty * time_bonus * cost_penalty


class PredictivePerformanceModel:
    """ML model to predict quantum solver performance."""
    
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42) if SKLEARN_AVAILABLE else None
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.is_trained = False
        self.feature_history: deque = deque(maxlen=1000)
        self.performance_history: deque = deque(maxlen=1000)
        self.logger = logging.getLogger(__name__)
    
    def extract_features(self, task: QuantumTask, resource: QuantumResource, context: Dict[str, Any]) -> np.ndarray:
        """Extract features for performance prediction."""
        # QUBO problem features
        qubo_size = len(task.qubo_problem)
        qubo_density = qubo_size / (qubo_size ** 2) if qubo_size > 0 else 0
        max_coeff = max(abs(coeff) for coeff in task.qubo_problem.values()) if task.qubo_problem else 0
        avg_coeff = np.mean(list(task.qubo_problem.values())) if task.qubo_problem else 0
        
        # Resource features
        resource_load = resource.queue_length / max(resource.max_qubits, 1)
        
        # Context features
        hour_of_day = datetime.now().hour
        day_of_week = datetime.now().weekday()
        energy_price = context.get('energy_price', 0.12)
        weather_temp = context.get('temperature', 20.0)
        
        # Task features
        urgency = task.urgency_score
        attempt_penalty = task.attempts * 0.1
        
        return np.array([
            qubo_size, qubo_density, max_coeff, avg_coeff,
            resource_load, resource.avg_solve_time, resource.cost_per_solve,
            hour_of_day, day_of_week, energy_price, weather_temp,
            urgency, attempt_penalty
        ])
    
    def predict_performance(
        self, 
        task: QuantumTask, 
        resource: QuantumResource, 
        context: Dict[str, Any]
    ) -> Dict[str, float]:
        """Predict solving performance for task-resource combination."""
        if not self.is_trained or not SKLEARN_AVAILABLE:
            # Fallback heuristic prediction
            base_time = task.estimated_solve_time * (1 + resource.queue_length * 0.1)
            return {
                'solve_time': base_time,
                'success_probability': 0.85,
                'energy_quality': 0.8,
                'cost_estimate': resource.cost_per_solve
            }
        
        features = self.extract_features(task, resource, context)
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # Predict multiple performance metrics
        predictions = self.model.predict(features_scaled)[0]
        
        return {
            'solve_time': max(predictions[0], 1.0),
            'success_probability': min(max(predictions[1], 0.0), 1.0),
            'energy_quality': min(max(predictions[2], 0.0), 1.0),
            'cost_estimate': max(predictions[3], 0.0)
        }
    
    def update_model(self, task: QuantumTask, resource: QuantumResource, 
                    context: Dict[str, Any], performance: Dict[str, float]) -> None:
        """Update ML model with new performance data."""
        if not SKLEARN_AVAILABLE:
            return
        
        features = self.extract_features(task, resource, context)
        target = np.array([
            performance['solve_time'],
            performance['success_probability'],
            performance['energy_quality'],
            performance['cost_estimate']
        ])
        
        self.feature_history.append(features)
        self.performance_history.append(target)
        
        # Retrain model periodically
        if len(self.feature_history) >= 50 and len(self.feature_history) % 20 == 0:
            self._retrain_model()
    
    def _retrain_model(self) -> None:
        """Retrain the performance prediction model."""
        if not SKLEARN_AVAILABLE or len(self.feature_history) < 10:
            return
        
        try:
            X = np.array(list(self.feature_history))
            y = np.array(list(self.performance_history))
            
            # Fit scaler and model
            X_scaled = self.scaler.fit_transform(X)
            self.model.fit(X_scaled, y)
            self.is_trained = True
            
            self.logger.info(f"Retrained performance model with {len(X)} samples")
            
        except Exception as e:
            self.logger.error(f"Failed to retrain model: {e}")


class RealTimeAdaptiveScheduler:
    """
    Real-time adaptive quantum scheduler with ML-driven optimization.
    
    Dynamically schedules quantum tasks across available resources while
    adapting to real-time conditions including energy prices, weather,
    and building performance requirements.
    """
    
    def __init__(self, max_concurrent_tasks: int = 5):
        self.max_concurrent_tasks = max_concurrent_tasks
        self.task_queue: List[QuantumTask] = []
        self.running_tasks: Dict[str, QuantumTask] = {}
        self.completed_tasks: deque = deque(maxlen=1000)
        self.failed_tasks: deque = deque(maxlen=100)
        
        self.resources: Dict[str, QuantumResource] = {}
        self.resource_allocation: Dict[str, str] = {}  # task_id -> resource_id
        
        self.performance_model = PredictivePerformanceModel()
        self.context_cache: Dict[str, Any] = {}
        
        self.logger = logging.getLogger(__name__)
        self._shutdown_event = asyncio.Event()
        
        # Adaptive parameters
        self.energy_price_threshold = 0.15  # $/kWh
        self.load_balancing_factor = 0.8
        self.emergency_override_enabled = True
        
        # Performance tracking
        self.scheduling_metrics = {
            'total_scheduled': 0,
            'successful_completions': 0,
            'deadline_misses': 0,
            'average_solve_time': 0.0,
            'cost_efficiency': 0.0
        }
    
    def register_resource(self, resource: QuantumResource) -> None:
        """Register a quantum computing resource."""
        self.resources[resource.resource_id] = resource
        self.logger.info(f"Registered quantum resource: {resource.resource_id}")
    
    async def submit_task(self, task: QuantumTask) -> str:
        """Submit quantum optimization task for scheduling."""
        task.task_id = f"{task.building_id}_{int(time.time() * 1000)}"
        
        # Insert in priority order
        insert_index = 0
        for i, existing_task in enumerate(self.task_queue):
            if task.urgency_score > existing_task.urgency_score:
                insert_index = i
                break
            insert_index = i + 1
        
        self.task_queue.insert(insert_index, task)
        self.scheduling_metrics['total_scheduled'] += 1
        
        self.logger.info(
            f"Submitted task {task.task_id} with urgency {task.urgency_score:.3f}, "
            f"queue position: {insert_index + 1}/{len(self.task_queue)}"
        )
        
        return task.task_id
    
    async def update_context(self, context: Dict[str, Any]) -> None:
        """Update real-time context for adaptive scheduling."""
        self.context_cache.update(context)
        
        # Adapt scheduling parameters based on context
        energy_price = context.get('energy_price', 0.12)
        
        if energy_price > self.energy_price_threshold:
            # High energy prices - prioritize energy efficiency
            await self._rebalance_queue_for_energy_efficiency()
        
        # Update resource availability based on context
        await self._update_resource_availability(context)
    
    async def _rebalance_queue_for_energy_efficiency(self) -> None:
        """Rebalance task queue for energy efficiency during high energy prices."""
        # Sort tasks by energy impact and cost efficiency
        self.task_queue.sort(
            key=lambda t: t.energy_urgency * t.urgency_score,
            reverse=True
        )
        
        self.logger.info("Rebalanced queue for energy efficiency due to high energy prices")
    
    async def _update_resource_availability(self, context: Dict[str, Any]) -> None:
        """Update resource availability based on real-time context."""
        current_time = datetime.now()
        
        for resource in self.resources.values():
            # Simulate resource availability based on time and load
            base_availability = 0.9
            
            # Peak hours penalty
            if 16 <= current_time.hour <= 20:
                base_availability *= 0.8
            
            # Weekend bonus
            if current_time.weekday() >= 5:
                base_availability *= 1.1
            
            # Load penalty
            load_factor = resource.queue_length / max(resource.max_qubits, 1)
            base_availability *= (1.0 - load_factor * 0.3)
            
            resource.availability_score = min(max(base_availability, 0.1), 1.0)
    
    async def schedule_and_execute(self) -> None:
        """Main scheduling and execution loop."""
        self.logger.info("Started real-time adaptive scheduler")
        
        while not self._shutdown_event.is_set():
            try:
                # Schedule pending tasks
                await self._schedule_pending_tasks()
                
                # Monitor running tasks
                await self._monitor_running_tasks()
                
                # Clean up completed tasks
                await self._cleanup_tasks()
                
                # Brief sleep before next cycle
                await asyncio.sleep(1.0)
                
            except Exception as e:
                self.logger.error(f"Scheduler cycle error: {e}")
                await asyncio.sleep(5.0)
    
    async def _schedule_pending_tasks(self) -> None:
        """Schedule pending tasks to available resources."""
        if not self.task_queue or len(self.running_tasks) >= self.max_concurrent_tasks:
            return
        
        available_slots = self.max_concurrent_tasks - len(self.running_tasks)
        tasks_to_schedule = []
        
        for _ in range(min(available_slots, len(self.task_queue))):
            task = self.task_queue.pop(0)
            
            # Find best resource for task
            best_resource = await self._select_optimal_resource(task)
            
            if best_resource:
                tasks_to_schedule.append((task, best_resource))
            else:
                # No suitable resource - put back in queue
                self.task_queue.insert(0, task)
                break
        
        # Execute selected tasks
        for task, resource in tasks_to_schedule:
            await self._execute_task(task, resource)
    
    async def _select_optimal_resource(self, task: QuantumTask) -> Optional[QuantumResource]:
        """Select optimal quantum resource for task using ML predictions."""
        if not self.resources:
            return None
        
        best_resource = None
        best_score = -1.0
        
        for resource in self.resources.values():
            # Skip if resource is overloaded
            if resource.queue_length >= resource.max_qubits:
                continue
            
            # Predict performance
            predicted_performance = self.performance_model.predict_performance(
                task, resource, self.context_cache
            )
            
            # Calculate resource suitability score
            urgency_weight = 1.0 if task.priority != SchedulingPriority.BACKGROUND else 0.5
            time_weight = 1.0 / max(predicted_performance['solve_time'], 0.1)
            success_weight = predicted_performance['success_probability']
            cost_weight = 1.0 / max(predicted_performance['cost_estimate'], 0.01)
            availability_weight = resource.availability_score
            
            suitability_score = (
                urgency_weight * time_weight * success_weight * 
                cost_weight * availability_weight
            )
            
            if suitability_score > best_score:
                best_score = suitability_score
                best_resource = resource
        
        return best_resource
    
    async def _execute_task(self, task: QuantumTask, resource: QuantumResource) -> None:
        """Execute quantum task on selected resource."""
        task.attempts += 1
        self.running_tasks[task.task_id] = task
        self.resource_allocation[task.task_id] = resource.resource_id
        
        # Update resource state
        resource.queue_length += 1
        resource.last_used = datetime.now()
        
        self.logger.info(
            f"Executing task {task.task_id} on resource {resource.resource_id} "
            f"(attempt {task.attempts}/{task.max_attempts})"
        )
        
        # Start async execution
        asyncio.create_task(self._run_quantum_task(task, resource))
    
    async def _run_quantum_task(self, task: QuantumTask, resource: QuantumResource) -> None:
        """Run quantum optimization task."""
        start_time = time.time()
        
        try:
            # Simulate quantum solving (in practice, would use actual quantum solver)
            from .adaptive_quantum_engine import get_adaptive_quantum_engine
            
            quantum_engine = get_adaptive_quantum_engine()
            
            # Execute quantum optimization
            solution = await quantum_engine.solve_adaptive(task.qubo_problem)
            
            solve_time = time.time() - start_time
            
            # Calculate performance metrics
            performance = {
                'solve_time': solve_time,
                'success_probability': 1.0 if solution.energy < 0 else 0.8,
                'energy_quality': max(0.0, 1.0 - abs(solution.energy) / 1000.0),
                'cost_estimate': resource.cost_per_solve
            }
            
            # Update performance model
            self.performance_model.update_model(task, resource, self.context_cache, performance)
            
            # Mark task as completed
            self.completed_tasks.append({
                'task': task,
                'resource': resource,
                'performance': performance,
                'solution': solution,
                'completed_at': datetime.now()
            })
            
            self.scheduling_metrics['successful_completions'] += 1
            self.scheduling_metrics['average_solve_time'] = (
                (self.scheduling_metrics['average_solve_time'] * (self.scheduling_metrics['successful_completions'] - 1) + solve_time) /
                self.scheduling_metrics['successful_completions']
            )
            
            self.logger.info(
                f"Task {task.task_id} completed successfully in {solve_time:.2f}s "
                f"(energy: {solution.energy:.3f})"
            )
            
        except Exception as e:
            self.logger.error(f"Task {task.task_id} failed: {e}")
            
            # Retry logic
            if task.attempts < task.max_attempts and task.priority != SchedulingPriority.BACKGROUND:
                # Re-queue with exponential backoff
                await asyncio.sleep(2 ** task.attempts)
                await self.submit_task(task)
            else:
                self.failed_tasks.append({
                    'task': task,
                    'resource': resource,
                    'error': str(e),
                    'failed_at': datetime.now()
                })
                
                if datetime.now() > task.deadline:
                    self.scheduling_metrics['deadline_misses'] += 1
        
        finally:
            # Clean up resource allocation
            if task.task_id in self.running_tasks:
                del self.running_tasks[task.task_id]
            if task.task_id in self.resource_allocation:
                del self.resource_allocation[task.task_id]
            
            # Update resource state
            resource.queue_length = max(0, resource.queue_length - 1)
    
    async def _monitor_running_tasks(self) -> None:
        """Monitor running tasks for timeout and resource management."""
        current_time = datetime.now()
        
        tasks_to_timeout = []
        
        for task_id, task in self.running_tasks.items():
            # Check for deadline timeout
            if current_time > task.deadline:
                tasks_to_timeout.append(task_id)
            
            # Check for execution timeout (3x estimated time)
            execution_timeout = task.created_at + timedelta(seconds=task.estimated_solve_time * 3)
            if current_time > execution_timeout:
                tasks_to_timeout.append(task_id)
        
        # Handle timeouts
        for task_id in tasks_to_timeout:
            await self._handle_task_timeout(task_id)
    
    async def _handle_task_timeout(self, task_id: str) -> None:
        """Handle task timeout with appropriate recovery actions."""
        if task_id not in self.running_tasks:
            return
        
        task = self.running_tasks[task_id]
        resource_id = self.resource_allocation.get(task_id)
        
        self.logger.warning(f"Task {task_id} timed out - attempting recovery")
        
        # Free up resource
        if resource_id and resource_id in self.resources:
            self.resources[resource_id].queue_length = max(0, self.resources[resource_id].queue_length - 1)
        
        # Clean up
        del self.running_tasks[task_id]
        if task_id in self.resource_allocation:
            del self.resource_allocation[task_id]
        
        # Retry if attempts remaining and high priority
        if task.attempts < task.max_attempts and task.priority in [SchedulingPriority.EMERGENCY, SchedulingPriority.HIGH]:
            await self.submit_task(task)
        else:
            self.failed_tasks.append({
                'task': task,
                'resource': self.resources.get(resource_id),
                'error': 'timeout',
                'failed_at': datetime.now()
            })
            
            self.scheduling_metrics['deadline_misses'] += 1
    
    async def _cleanup_tasks(self) -> None:
        """Clean up old completed and failed tasks."""
        current_time = datetime.now()
        cutoff_time = current_time - timedelta(hours=24)
        
        # Clean old completed tasks
        while self.completed_tasks and self.completed_tasks[0]['completed_at'] < cutoff_time:
            self.completed_tasks.popleft()
        
        # Clean old failed tasks
        while self.failed_tasks and self.failed_tasks[0]['failed_at'] < cutoff_time:
            self.failed_tasks.popleft()
    
    def get_scheduling_status(self) -> Dict[str, Any]:
        """Get comprehensive scheduling status."""
        return {
            'queue_status': {
                'pending_tasks': len(self.task_queue),
                'running_tasks': len(self.running_tasks),
                'completed_tasks': len(self.completed_tasks),
                'failed_tasks': len(self.failed_tasks)
            },
            'resource_status': {
                resource_id: {
                    'availability': resource.availability_score,
                    'queue_length': resource.queue_length,
                    'efficiency_score': resource.efficiency_score,
                    'last_used': resource.last_used.isoformat()
                }
                for resource_id, resource in self.resources.items()
            },
            'performance_metrics': self.scheduling_metrics.copy(),
            'adaptive_parameters': {
                'energy_price_threshold': self.energy_price_threshold,
                'load_balancing_factor': self.load_balancing_factor,
                'emergency_override_enabled': self.emergency_override_enabled
            },
            'ml_model_status': {
                'is_trained': self.performance_model.is_trained,
                'training_samples': len(self.performance_model.feature_history),
                'sklearn_available': SKLEARN_AVAILABLE
            }
        }
    
    async def emergency_schedule(self, task: QuantumTask) -> str:
        """Emergency scheduling that bypasses normal queue."""
        task.priority = SchedulingPriority.EMERGENCY
        task.task_id = f"EMERGENCY_{task.building_id}_{int(time.time() * 1000)}"
        
        # Find any available resource immediately
        for resource in self.resources.values():
            if resource.availability_score > 0.5:
                await self._execute_task(task, resource)
                return task.task_id
        
        # If no resources available, preempt lowest priority task
        if self.running_tasks:
            lowest_priority_task_id = min(
                self.running_tasks.keys(),
                key=lambda tid: self.running_tasks[tid].urgency_score
            )
            
            await self._handle_task_timeout(lowest_priority_task_id)
            
            # Try to schedule emergency task again
            for resource in self.resources.values():
                if resource.queue_length < resource.max_qubits:
                    await self._execute_task(task, resource)
                    return task.task_id
        
        # Last resort - add to front of queue
        self.task_queue.insert(0, task)
        return task.task_id
    
    async def shutdown(self) -> None:
        """Graceful shutdown of scheduler."""
        self.logger.info("Shutting down real-time adaptive scheduler")
        self._shutdown_event.set()
        
        # Wait for running tasks to complete (with timeout)
        timeout = 30.0
        while self.running_tasks and timeout > 0:
            await asyncio.sleep(1.0)
            timeout -= 1.0
        
        if self.running_tasks:
            self.logger.warning(f"Forced shutdown with {len(self.running_tasks)} tasks still running")


# Global scheduler instance
_adaptive_scheduler: Optional[RealTimeAdaptiveScheduler] = None


def get_adaptive_scheduler() -> RealTimeAdaptiveScheduler:
    """Get global adaptive scheduler instance."""
    global _adaptive_scheduler
    if _adaptive_scheduler is None:
        _adaptive_scheduler = RealTimeAdaptiveScheduler()
    return _adaptive_scheduler


async def initialize_scheduler_with_resources() -> RealTimeAdaptiveScheduler:
    """Initialize scheduler with default quantum resources."""
    scheduler = get_adaptive_scheduler()
    
    # Register default quantum resources
    scheduler.register_resource(QuantumResource(
        resource_id="dwave_advantage",
        resource_type="qpu",
        max_qubits=5000,
        queue_length=0,
        avg_solve_time=2.0,
        cost_per_solve=0.50,
        availability_score=0.9
    ))
    
    scheduler.register_resource(QuantumResource(
        resource_id="dwave_hybrid",
        resource_type="hybrid",
        max_qubits=10000,
        queue_length=0,
        avg_solve_time=5.0,
        cost_per_solve=0.10,
        availability_score=0.95
    ))
    
    scheduler.register_resource(QuantumResource(
        resource_id="classical_sim",
        resource_type="simulator",
        max_qubits=1000,
        queue_length=0,
        avg_solve_time=10.0,
        cost_per_solve=0.01,
        availability_score=0.99
    ))
    
    return scheduler
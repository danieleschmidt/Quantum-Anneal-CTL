"""Distributed Quality Gates for Horizontal Scaling"""

import asyncio
import json
import time
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, asdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp
import logging
from pathlib import Path
import hashlib
import redis
from datetime import datetime, timedelta

from .base import QualityGate, GateResult
from .gate_runner import QualityGateRunner
from .config import QualityGateConfig

logger = logging.getLogger(__name__)


@dataclass
class DistributedGateTask:
    """Represents a quality gate task for distributed execution"""
    task_id: str
    gate_name: str
    context: Dict[str, Any]
    priority: int = 1
    created_at: float = 0.0
    assigned_to: Optional[str] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    
    def __post_init__(self):
        if self.created_at == 0.0:
            self.created_at = time.time()


class DistributedGateCoordinator:
    """Coordinates distributed execution of quality gates"""
    
    def __init__(self, 
                 config: QualityGateConfig,
                 redis_url: str = "redis://localhost:6379",
                 worker_pool_size: int = None):
        self.config = config
        self.redis_url = redis_url
        self.worker_pool_size = worker_pool_size or mp.cpu_count()
        self.redis_client = None
        self.worker_processes = []
        self.is_running = False
        self.coordinator_id = f"coordinator_{int(time.time())}"
        
    async def initialize(self):
        """Initialize the distributed system"""
        try:
            import redis.asyncio as aioredis
            self.redis_client = aioredis.from_url(self.redis_url)
            await self.redis_client.ping()
            logger.info("Connected to Redis for distributed coordination")
        except Exception as e:
            logger.warning(f"Redis not available, falling back to local execution: {e}")
            self.redis_client = None
    
    async def run_distributed_gates(self, 
                                  gates: List[QualityGate],
                                  context: Dict[str, Any]) -> Dict[str, Any]:
        """Run quality gates in distributed mode"""
        
        if not self.redis_client:
            logger.info("Running gates locally (Redis not available)")
            return await self._run_local_fallback(gates, context)
        
        # Start worker processes
        await self._start_workers()
        
        try:
            # Create tasks for each gate
            tasks = []
            for gate in gates:
                if gate.is_enabled():
                    task = DistributedGateTask(
                        task_id=f"{gate.name}_{int(time.time() * 1000000)}",
                        gate_name=gate.name,
                        context=context,
                        priority=self._get_gate_priority(gate.name)
                    )
                    tasks.append(task)
            
            # Submit tasks to Redis queue
            await self._submit_tasks(tasks)
            
            # Monitor and collect results
            results = await self._collect_results(tasks)
            
            # Calculate overall result
            return self._calculate_distributed_result(results)
            
        finally:
            await self._stop_workers()
    
    async def _start_workers(self):
        """Start worker processes"""
        logger.info(f"Starting {self.worker_pool_size} distributed workers")
        
        self.is_running = True
        
        # Use ProcessPoolExecutor for better isolation
        executor = ProcessPoolExecutor(max_workers=self.worker_pool_size)
        
        # Start worker processes
        futures = []
        for worker_id in range(self.worker_pool_size):
            future = executor.submit(
                self._worker_process,
                f"worker_{worker_id}",
                self.redis_url,
                self.config
            )
            futures.append(future)
        
        # Give workers time to start
        await asyncio.sleep(2.0)
    
    async def _stop_workers(self):
        """Stop all worker processes"""
        logger.info("Stopping distributed workers")
        self.is_running = False
        
        # Send shutdown signal via Redis
        if self.redis_client:
            await self.redis_client.publish("worker_control", json.dumps({
                "command": "shutdown",
                "coordinator_id": self.coordinator_id
            }))
        
        # Wait for workers to shutdown gracefully
        await asyncio.sleep(3.0)
    
    @staticmethod
    def _worker_process(worker_id: str, redis_url: str, config: QualityGateConfig):
        """Worker process that executes quality gates"""
        
        async def worker_main():
            try:
                import redis.asyncio as aioredis
                redis_client = aioredis.from_url(redis_url)
                
                logger.info(f"Worker {worker_id} started")
                
                # Create gate runner for this worker
                runner = QualityGateRunner(config)
                
                while True:
                    try:
                        # Get task from queue (blocking pop with timeout)
                        task_data = await redis_client.blpop("gate_tasks", timeout=5.0)
                        
                        if not task_data:
                            # Check for shutdown signal
                            message = await redis_client.get("shutdown_signal")
                            if message:
                                break
                            continue
                        
                        # Deserialize task
                        task_json = task_data[1].decode('utf-8')
                        task_dict = json.loads(task_json)
                        task = DistributedGateTask(**task_dict)
                        
                        # Execute gate
                        result = await worker_execute_gate(runner, task)
                        
                        # Store result
                        result_key = f"gate_result:{task.task_id}"
                        await redis_client.setex(
                            result_key, 
                            300,  # 5 minute TTL
                            json.dumps(result, default=str)
                        )
                        
                        # Notify completion
                        await redis_client.publish(
                            "gate_completed", 
                            json.dumps({"task_id": task.task_id, "worker_id": worker_id})
                        )
                        
                    except Exception as e:
                        logger.error(f"Worker {worker_id} error: {e}")
                        await asyncio.sleep(1.0)
                
                logger.info(f"Worker {worker_id} shutting down")
                
            except Exception as e:
                logger.error(f"Worker {worker_id} fatal error: {e}")
        
        # Run worker in async context
        asyncio.run(worker_main())
    
    async def _submit_tasks(self, tasks: List[DistributedGateTask]):
        """Submit tasks to Redis queue"""
        logger.info(f"Submitting {len(tasks)} tasks to distributed queue")
        
        # Sort by priority (higher priority first)
        tasks.sort(key=lambda t: t.priority, reverse=True)
        
        for task in tasks:
            task_json = json.dumps(asdict(task), default=str)
            await self.redis_client.rpush("gate_tasks", task_json)
    
    async def _collect_results(self, tasks: List[DistributedGateTask]) -> List[GateResult]:
        """Collect results from completed tasks"""
        logger.info("Collecting results from distributed execution")
        
        results = []
        completed_tasks = set()
        timeout_seconds = 300  # 5 minute timeout
        start_time = time.time()
        
        # Subscribe to completion notifications
        pubsub = self.redis_client.pubsub()
        await pubsub.subscribe("gate_completed")
        
        while len(completed_tasks) < len(tasks):
            # Check for timeout
            if time.time() - start_time > timeout_seconds:
                logger.warning("Timeout waiting for distributed results")
                break
            
            try:
                # Wait for completion notification
                message = await pubsub.get_message(timeout=5.0)
                
                if message and message['type'] == 'message':
                    notification = json.loads(message['data'].decode('utf-8'))
                    task_id = notification['task_id']
                    
                    if task_id not in completed_tasks:
                        # Get result from Redis
                        result_key = f"gate_result:{task_id}"
                        result_data = await self.redis_client.get(result_key)
                        
                        if result_data:
                            result_dict = json.loads(result_data.decode('utf-8'))
                            result = GateResult(**result_dict)
                            results.append(result)
                            completed_tasks.add(task_id)
                
            except Exception as e:
                logger.warning(f"Error collecting result: {e}")
        
        await pubsub.unsubscribe("gate_completed")
        
        # Handle incomplete tasks
        if len(results) < len(tasks):
            missing_count = len(tasks) - len(results)
            logger.warning(f"{missing_count} tasks did not complete")
        
        return results
    
    def _get_gate_priority(self, gate_name: str) -> int:
        """Get priority for gate execution"""
        # Higher priority for critical gates
        priorities = {
            'security': 5,
            'test_coverage': 4,
            'performance': 3,
            'code_quality': 2,
            'documentation': 1
        }
        return priorities.get(gate_name, 1)
    
    def _calculate_distributed_result(self, results: List[GateResult]) -> Dict[str, Any]:
        """Calculate overall result from distributed execution"""
        if not results:
            return {
                "passed": False,
                "score": 0.0,
                "execution_time_ms": 0.0,
                "gates": [],
                "summary": {
                    "total_gates": 0,
                    "passed_gates": 0,
                    "failed_gates": 0,
                    "distributed": True
                }
            }
        
        passed_gates = [r for r in results if r.passed]
        failed_gates = [r for r in results if not r.passed]
        
        total_score = sum(r.score for r in results)
        overall_score = total_score / len(results)
        overall_passed = len(failed_gates) == 0
        
        # Execution time is the maximum (since gates ran in parallel)
        max_execution_time = max(r.execution_time_ms for r in results)
        
        return {
            "passed": overall_passed,
            "score": overall_score,
            "execution_time_ms": max_execution_time,
            "gates": [result.__dict__ for result in results],
            "summary": {
                "total_gates": len(results),
                "passed_gates": len(passed_gates),
                "failed_gates": len(failed_gates),
                "distributed": True,
                "coordinator_id": self.coordinator_id
            }
        }
    
    async def _run_local_fallback(self, gates: List[QualityGate], context: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback to local execution when distributed mode is unavailable"""
        runner = QualityGateRunner(self.config)
        return await runner._run_gates_parallel(gates, context)


async def worker_execute_gate(runner: QualityGateRunner, task: DistributedGateTask) -> Dict[str, Any]:
    """Execute a single gate in worker process"""
    try:
        # Find and execute the gate
        gate_result = await runner.run_single_gate(task.gate_name, task.context)
        
        # Convert to dictionary for serialization
        return {
            "gate_name": gate_result.gate_name,
            "passed": gate_result.passed,
            "score": gate_result.score,
            "threshold": gate_result.threshold,
            "metrics": gate_result.metrics,
            "messages": gate_result.messages,
            "execution_time_ms": gate_result.execution_time_ms,
            "timestamp": gate_result.timestamp.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Gate execution failed: {e}")
        return {
            "gate_name": task.gate_name,
            "passed": False,
            "score": 0.0,
            "threshold": 100.0,
            "metrics": {"error": str(e)},
            "messages": [f"Distributed execution failed: {e}"],
            "execution_time_ms": 0.0,
            "timestamp": datetime.utcnow().isoformat()
        }


class LoadBalancedGateRunner:
    """Load-balanced quality gate runner with intelligent task distribution"""
    
    def __init__(self, config: QualityGateConfig, scaling_config: Dict[str, Any] = None):
        self.config = config
        self.scaling_config = scaling_config or {}
        self.coordinator = DistributedGateCoordinator(config)
        self.local_runner = QualityGateRunner(config)
        
        # Scaling parameters
        self.max_workers = self.scaling_config.get('max_workers', mp.cpu_count() * 2)
        self.min_workers = self.scaling_config.get('min_workers', 2)
        self.scale_threshold = self.scaling_config.get('scale_threshold', 0.8)  # CPU threshold
        self.current_workers = self.min_workers
        
    async def run_with_auto_scaling(self, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Run quality gates with automatic scaling based on load"""
        
        if context is None:
            context = {"project_root": "."}
        
        # Initialize distributed coordinator
        await self.coordinator.initialize()
        
        # Determine execution mode
        execution_mode = await self._determine_execution_mode()
        
        if execution_mode == "distributed":
            logger.info("Running in distributed mode with auto-scaling")
            return await self._run_with_scaling(context)
        else:
            logger.info("Running in local mode")
            return await self.local_runner.run_all_gates(context)
    
    async def _determine_execution_mode(self) -> str:
        """Determine whether to use distributed or local execution"""
        
        # Check if Redis is available
        if not self.coordinator.redis_client:
            return "local"
        
        # Check system resources
        import psutil
        cpu_percent = psutil.cpu_percent(interval=1.0)
        memory_percent = psutil.virtual_memory().percent
        
        # Use distributed mode if system is under load or has many cores
        if (cpu_percent > 70 or 
            memory_percent > 70 or 
            mp.cpu_count() >= 4):
            return "distributed"
        
        return "local"
    
    async def _run_with_scaling(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Run gates with dynamic worker scaling"""
        
        # Monitor system resources during execution
        resource_monitor = ResourceMonitor()
        await resource_monitor.start()
        
        try:
            # Get available gates
            gates = self.coordinator.gates if hasattr(self.coordinator, 'gates') else []
            
            # Adjust worker count based on workload
            optimal_workers = self._calculate_optimal_workers(len(gates))
            self.coordinator.worker_pool_size = optimal_workers
            
            # Execute gates
            result = await self.coordinator.run_distributed_gates(gates, context)
            
            # Add scaling information to result
            result['scaling_info'] = {
                'workers_used': optimal_workers,
                'execution_mode': 'distributed',
                'resource_usage': await resource_monitor.get_summary()
            }
            
            return result
            
        finally:
            await resource_monitor.stop()
    
    def _calculate_optimal_workers(self, gate_count: int) -> int:
        """Calculate optimal number of workers based on workload"""
        
        # Base calculation on number of gates and system resources
        import psutil
        
        cpu_count = mp.cpu_count()
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        
        # Start with one worker per gate, but respect system limits
        optimal = min(gate_count, cpu_count)
        
        # Adjust based on available memory (assume 100MB per worker)
        memory_limited = int(available_memory_gb * 10)  # 100MB per worker
        optimal = min(optimal, memory_limited)
        
        # Respect configured limits
        optimal = max(self.min_workers, min(optimal, self.max_workers))
        
        logger.info(f"Calculated optimal workers: {optimal} (gates: {gate_count}, CPUs: {cpu_count})")
        return optimal


class ResourceMonitor:
    """Monitor system resources during execution"""
    
    def __init__(self):
        self.monitoring = False
        self.samples = []
        self.monitor_task = None
        
    async def start(self):
        """Start resource monitoring"""
        self.monitoring = True
        self.samples = []
        self.monitor_task = asyncio.create_task(self._monitor_loop())
    
    async def stop(self):
        """Stop resource monitoring"""
        self.monitoring = False
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
    
    async def _monitor_loop(self):
        """Main monitoring loop"""
        import psutil
        
        while self.monitoring:
            try:
                sample = {
                    'timestamp': time.time(),
                    'cpu_percent': psutil.cpu_percent(),
                    'memory_percent': psutil.virtual_memory().percent,
                    'disk_io': psutil.disk_io_counters()._asdict() if psutil.disk_io_counters() else {},
                    'network_io': psutil.net_io_counters()._asdict() if psutil.net_io_counters() else {}
                }
                
                self.samples.append(sample)
                
                # Keep only recent samples
                if len(self.samples) > 300:  # 5 minutes at 1 second intervals
                    self.samples = self.samples[-150:]
                
                await asyncio.sleep(1.0)
                
            except Exception as e:
                logger.warning(f"Resource monitoring error: {e}")
                await asyncio.sleep(5.0)
    
    async def get_summary(self) -> Dict[str, Any]:
        """Get resource usage summary"""
        if not self.samples:
            return {}
        
        cpu_values = [s['cpu_percent'] for s in self.samples]
        memory_values = [s['memory_percent'] for s in self.samples]
        
        return {
            'avg_cpu_percent': sum(cpu_values) / len(cpu_values),
            'max_cpu_percent': max(cpu_values),
            'avg_memory_percent': sum(memory_values) / len(memory_values),
            'max_memory_percent': max(memory_values),
            'sample_count': len(self.samples),
            'monitoring_duration': self.samples[-1]['timestamp'] - self.samples[0]['timestamp'] if len(self.samples) > 1 else 0
        }
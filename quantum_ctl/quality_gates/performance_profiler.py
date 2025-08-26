"""Advanced Performance Profiling for Quality Gates"""

import asyncio
import time
import psutil
import threading
import gc
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable, AsyncGenerator
from pathlib import Path
import json
import sys
import cProfile
import pstats
import io
from concurrent.futures import ThreadPoolExecutor
import logging

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics"""
    
    # Timing metrics
    execution_time_ms: float = 0.0
    cpu_time_ms: float = 0.0
    wall_time_ms: float = 0.0
    
    # Memory metrics
    peak_memory_mb: float = 0.0
    memory_growth_mb: float = 0.0
    memory_leaks_detected: int = 0
    
    # CPU metrics
    avg_cpu_percent: float = 0.0
    peak_cpu_percent: float = 0.0
    cpu_time_user: float = 0.0
    cpu_time_system: float = 0.0
    
    # I/O metrics
    disk_reads: int = 0
    disk_writes: int = 0
    network_packets_sent: int = 0
    network_packets_recv: int = 0
    
    # Function-level metrics
    function_calls: int = 0
    hot_functions: List[Dict[str, Any]] = field(default_factory=list)
    
    # Async metrics
    async_tasks_created: int = 0
    async_tasks_completed: int = 0
    avg_task_completion_time: float = 0.0
    
    # GC metrics
    gc_collections: List[int] = field(default_factory=list)
    gc_time_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'timing': {
                'execution_time_ms': self.execution_time_ms,
                'cpu_time_ms': self.cpu_time_ms,
                'wall_time_ms': self.wall_time_ms
            },
            'memory': {
                'peak_memory_mb': self.peak_memory_mb,
                'memory_growth_mb': self.memory_growth_mb,
                'memory_leaks_detected': self.memory_leaks_detected
            },
            'cpu': {
                'avg_cpu_percent': self.avg_cpu_percent,
                'peak_cpu_percent': self.peak_cpu_percent,
                'cpu_time_user': self.cpu_time_user,
                'cpu_time_system': self.cpu_time_system
            },
            'io': {
                'disk_reads': self.disk_reads,
                'disk_writes': self.disk_writes,
                'network_packets_sent': self.network_packets_sent,
                'network_packets_recv': self.network_packets_recv
            },
            'functions': {
                'function_calls': self.function_calls,
                'hot_functions': self.hot_functions
            },
            'async': {
                'async_tasks_created': self.async_tasks_created,
                'async_tasks_completed': self.async_tasks_completed,
                'avg_task_completion_time': self.avg_task_completion_time
            },
            'gc': {
                'gc_collections': self.gc_collections,
                'gc_time_ms': self.gc_time_ms
            }
        }


class PerformanceProfiler:
    """Advanced performance profiler with multi-dimensional monitoring"""
    
    def __init__(self):
        self.process = psutil.Process()
        self.monitoring_thread = None
        self.monitoring_active = False
        self.metrics_history = []
        self.profiler = None
        self.executor = ThreadPoolExecutor(max_workers=2)
        
    @asynccontextmanager
    async def profile_execution(self, 
                              sample_interval: float = 0.1,
                              enable_function_profiling: bool = True,
                              track_memory_leaks: bool = True) -> AsyncGenerator[PerformanceMetrics, None]:
        """Context manager for comprehensive performance profiling"""
        
        metrics = PerformanceMetrics()
        
        # Record initial state
        initial_memory = self.process.memory_info().rss / (1024 * 1024)
        initial_cpu_times = self.process.cpu_times()
        initial_io = self.process.io_counters()
        initial_gc_counts = gc.get_count()
        
        start_time = time.time()
        start_perf_counter = time.perf_counter()
        
        # Start monitoring thread
        self.monitoring_active = True
        self.metrics_history = []
        
        self.monitoring_thread = threading.Thread(
            target=self._monitor_resources,
            args=(sample_interval, metrics)
        )
        self.monitoring_thread.start()
        
        # Start function profiling if enabled
        if enable_function_profiling:
            self.profiler = cProfile.Profile()
            self.profiler.enable()
        
        # Track async tasks
        task_tracker = AsyncTaskTracker()
        original_create_task = asyncio.create_task
        
        def tracked_create_task(coro, **kwargs):
            task = original_create_task(coro, **kwargs)
            task_tracker.track_task(task)
            return task
        
        asyncio.create_task = tracked_create_task
        
        try:
            yield metrics
        finally:
            # Stop monitoring
            self.monitoring_active = False
            if self.monitoring_thread:
                self.monitoring_thread.join(timeout=5.0)
            
            # Restore original create_task
            asyncio.create_task = original_create_task
            
            # Calculate final metrics
            end_time = time.time()
            end_perf_counter = time.perf_counter()
            
            metrics.execution_time_ms = (end_time - start_time) * 1000
            metrics.wall_time_ms = (end_perf_counter - start_perf_counter) * 1000
            
            # CPU metrics
            final_cpu_times = self.process.cpu_times()
            metrics.cpu_time_user = final_cpu_times.user - initial_cpu_times.user
            metrics.cpu_time_system = final_cpu_times.system - initial_cpu_times.system
            metrics.cpu_time_ms = (metrics.cpu_time_user + metrics.cpu_time_system) * 1000
            
            # Memory metrics
            final_memory = self.process.memory_info().rss / (1024 * 1024)
            metrics.memory_growth_mb = final_memory - initial_memory
            
            if track_memory_leaks:
                metrics.memory_leaks_detected = await self._detect_memory_leaks()
            
            # I/O metrics
            final_io = self.process.io_counters()
            metrics.disk_reads = final_io.read_count - initial_io.read_count
            metrics.disk_writes = final_io.write_count - initial_io.write_count
            
            # GC metrics
            final_gc_counts = gc.get_count()
            metrics.gc_collections = [
                final_gc_counts[i] - initial_gc_counts[i] 
                for i in range(len(initial_gc_counts))
            ]
            
            # Function profiling results
            if self.profiler:
                self.profiler.disable()
                metrics.hot_functions = self._analyze_function_performance()
                metrics.function_calls = sum(
                    func_data.get('calls', 0) for func_data in metrics.hot_functions
                )
            
            # Async task metrics
            metrics.async_tasks_created = task_tracker.tasks_created
            metrics.async_tasks_completed = task_tracker.tasks_completed
            metrics.avg_task_completion_time = task_tracker.avg_completion_time()
            
            # Calculate averages from monitoring history
            if self.metrics_history:
                cpu_samples = [m['cpu_percent'] for m in self.metrics_history]
                memory_samples = [m['memory_mb'] for m in self.metrics_history]
                
                metrics.avg_cpu_percent = sum(cpu_samples) / len(cpu_samples)
                metrics.peak_cpu_percent = max(cpu_samples)
                metrics.peak_memory_mb = max(memory_samples)
    
    def _monitor_resources(self, interval: float, metrics: PerformanceMetrics):
        """Background resource monitoring"""
        while self.monitoring_active:
            try:
                # CPU and memory snapshot
                cpu_percent = self.process.cpu_percent()
                memory_info = self.process.memory_info()
                memory_mb = memory_info.rss / (1024 * 1024)
                
                # Store snapshot
                self.metrics_history.append({
                    'timestamp': time.time(),
                    'cpu_percent': cpu_percent,
                    'memory_mb': memory_mb,
                    'memory_percent': self.process.memory_percent()
                })
                
                # Keep history manageable
                if len(self.metrics_history) > 1000:
                    self.metrics_history = self.metrics_history[-500:]
                
                time.sleep(interval)
                
            except Exception as e:
                logger.warning(f"Resource monitoring error: {e}")
                time.sleep(interval * 2)  # Longer sleep on error
    
    def _analyze_function_performance(self) -> List[Dict[str, Any]]:
        """Analyze function-level performance from profiling data"""
        if not self.profiler:
            return []
        
        # Get profiling statistics
        stats_stream = io.StringIO()
        stats = pstats.Stats(self.profiler, stream=stats_stream)
        stats.sort_stats('cumulative')
        
        # Extract top functions
        hot_functions = []
        
        try:
            # Get raw stats
            for (filename, line_number, function_name), (calls, _, total_time, cumulative_time) in stats.stats.items():
                if calls > 0:  # Only include functions that were called
                    hot_functions.append({
                        'function': function_name,
                        'filename': filename,
                        'line_number': line_number,
                        'calls': calls,
                        'total_time': total_time,
                        'cumulative_time': cumulative_time,
                        'per_call_time': total_time / calls if calls > 0 else 0
                    })
            
            # Sort by cumulative time and return top 20
            hot_functions.sort(key=lambda x: x['cumulative_time'], reverse=True)
            return hot_functions[:20]
            
        except Exception as e:
            logger.warning(f"Function analysis error: {e}")
            return []
    
    async def _detect_memory_leaks(self) -> int:
        """Detect potential memory leaks"""
        leaks_detected = 0
        
        try:
            # Force garbage collection
            collected = gc.collect()
            
            # Check for uncollectable objects
            uncollectable = len(gc.garbage)
            if uncollectable > 0:
                leaks_detected += uncollectable
            
            # Check memory growth trend
            if len(self.metrics_history) > 10:
                recent_memory = [m['memory_mb'] for m in self.metrics_history[-10:]]
                early_memory = [m['memory_mb'] for m in self.metrics_history[:10]]
                
                recent_avg = sum(recent_memory) / len(recent_memory)
                early_avg = sum(early_memory) / len(early_memory)
                
                # If memory grew significantly and stayed high, potential leak
                growth_ratio = recent_avg / early_avg if early_avg > 0 else 1
                if growth_ratio > 1.5:  # 50% growth
                    leaks_detected += 1
            
        except Exception as e:
            logger.warning(f"Memory leak detection error: {e}")
        
        return leaks_detected
    
    async def benchmark_operation(self, 
                                operation: Callable,
                                iterations: int = 100,
                                warmup_iterations: int = 10) -> Dict[str, Any]:
        """Benchmark a specific operation"""
        
        # Warmup runs
        for _ in range(warmup_iterations):
            if asyncio.iscoroutinefunction(operation):
                await operation()
            else:
                operation()
        
        # Actual benchmark runs
        times = []
        
        async with self.profile_execution() as metrics:
            for _ in range(iterations):
                start = time.perf_counter()
                
                if asyncio.iscoroutinefunction(operation):
                    await operation()
                else:
                    operation()
                
                times.append((time.perf_counter() - start) * 1000)
        
        # Calculate statistics
        times.sort()
        total_time = sum(times)
        
        benchmark_results = {
            'iterations': iterations,
            'total_time_ms': total_time,
            'avg_time_ms': total_time / iterations,
            'min_time_ms': min(times),
            'max_time_ms': max(times),
            'median_time_ms': times[len(times) // 2],
            'p95_time_ms': times[int(len(times) * 0.95)],
            'p99_time_ms': times[int(len(times) * 0.99)],
            'operations_per_second': 1000 / (total_time / iterations),
            'performance_metrics': metrics.to_dict()
        }
        
        return benchmark_results
    
    async def profile_memory_usage(self, duration_seconds: float = 60.0) -> Dict[str, Any]:
        """Profile memory usage over time"""
        
        memory_samples = []
        start_time = time.time()
        
        while time.time() - start_time < duration_seconds:
            memory_info = self.process.memory_info()
            memory_samples.append({
                'timestamp': time.time(),
                'rss_mb': memory_info.rss / (1024 * 1024),
                'vms_mb': memory_info.vms / (1024 * 1024),
                'percent': self.process.memory_percent()
            })
            
            await asyncio.sleep(0.1)
        
        # Analyze memory patterns
        rss_values = [s['rss_mb'] for s in memory_samples]
        
        return {
            'duration_seconds': duration_seconds,
            'samples': len(memory_samples),
            'memory_stats': {
                'avg_rss_mb': sum(rss_values) / len(rss_values),
                'max_rss_mb': max(rss_values),
                'min_rss_mb': min(rss_values),
                'memory_growth_mb': rss_values[-1] - rss_values[0]
            },
            'samples': memory_samples[-100:]  # Keep last 100 samples
        }
    
    def generate_performance_report(self, metrics: PerformanceMetrics) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        
        # Performance score calculation
        performance_score = 100.0
        
        # Deduct points based on performance issues
        if metrics.execution_time_ms > 5000:  # > 5 seconds
            performance_score -= 30
        elif metrics.execution_time_ms > 2000:  # > 2 seconds
            performance_score -= 15
        elif metrics.execution_time_ms > 1000:  # > 1 second
            performance_score -= 5
        
        if metrics.peak_memory_mb > 1000:  # > 1GB
            performance_score -= 20
        elif metrics.peak_memory_mb > 500:  # > 500MB
            performance_score -= 10
        
        if metrics.avg_cpu_percent > 80:
            performance_score -= 15
        elif metrics.avg_cpu_percent > 60:
            performance_score -= 8
        
        if metrics.memory_leaks_detected > 0:
            performance_score -= metrics.memory_leaks_detected * 10
        
        performance_score = max(0, performance_score)
        
        # Identify bottlenecks
        bottlenecks = []
        
        if metrics.execution_time_ms > 1000:
            bottlenecks.append("High execution time")
        
        if metrics.peak_cpu_percent > 80:
            bottlenecks.append("High CPU usage")
        
        if metrics.peak_memory_mb > 500:
            bottlenecks.append("High memory usage")
        
        if metrics.memory_leaks_detected > 0:
            bottlenecks.append("Memory leaks detected")
        
        # Generate recommendations
        recommendations = []
        
        if metrics.execution_time_ms > 2000:
            recommendations.append("Consider optimizing slow operations or implementing caching")
        
        if metrics.hot_functions:
            top_function = metrics.hot_functions[0]
            recommendations.append(f"Optimize hot function: {top_function['function']} ({top_function['calls']} calls)")
        
        if metrics.memory_growth_mb > 100:
            recommendations.append("Monitor memory usage patterns and implement proper cleanup")
        
        if metrics.avg_cpu_percent > 70:
            recommendations.append("Consider implementing async operations or reducing computational complexity")
        
        return {
            'performance_score': performance_score,
            'grade': self._calculate_performance_grade(performance_score),
            'bottlenecks': bottlenecks,
            'recommendations': recommendations,
            'metrics': metrics.to_dict(),
            'summary': {
                'execution_time_ms': metrics.execution_time_ms,
                'peak_memory_mb': metrics.peak_memory_mb,
                'avg_cpu_percent': metrics.avg_cpu_percent,
                'function_calls': metrics.function_calls,
                'memory_leaks': metrics.memory_leaks_detected
            }
        }
    
    def _calculate_performance_grade(self, score: float) -> str:
        """Calculate performance grade"""
        if score >= 90:
            return "A"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"


class AsyncTaskTracker:
    """Track async task creation and completion"""
    
    def __init__(self):
        self.tasks_created = 0
        self.tasks_completed = 0
        self.task_times = []
        self.lock = threading.Lock()
    
    def track_task(self, task):
        """Track an async task"""
        with self.lock:
            self.tasks_created += 1
        
        start_time = time.time()
        
        def on_done(future):
            with self.lock:
                self.tasks_completed += 1
                completion_time = time.time() - start_time
                self.task_times.append(completion_time)
                
                # Keep only recent task times
                if len(self.task_times) > 1000:
                    self.task_times = self.task_times[-500:]
        
        task.add_done_callback(on_done)
    
    def avg_completion_time(self) -> float:
        """Get average task completion time"""
        with self.lock:
            if not self.task_times:
                return 0.0
            return sum(self.task_times) / len(self.task_times) * 1000  # ms


class PerformanceOptimizer:
    """Automatic performance optimization suggestions"""
    
    def __init__(self):
        self.profiler = PerformanceProfiler()
    
    async def analyze_and_optimize(self, operation: Callable, **kwargs) -> Dict[str, Any]:
        """Analyze operation and provide optimization suggestions"""
        
        # Baseline benchmark
        baseline_results = await self.profiler.benchmark_operation(
            operation, 
            iterations=kwargs.get('iterations', 10),
            warmup_iterations=kwargs.get('warmup_iterations', 2)
        )
        
        # Analyze results and generate suggestions
        suggestions = self._generate_optimization_suggestions(baseline_results)
        
        return {
            'baseline_performance': baseline_results,
            'optimization_suggestions': suggestions,
            'estimated_improvements': self._estimate_improvements(baseline_results, suggestions)
        }
    
    def _generate_optimization_suggestions(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate specific optimization suggestions"""
        suggestions = []
        
        metrics = results.get('performance_metrics', {})
        avg_time = results.get('avg_time_ms', 0)
        
        # High execution time
        if avg_time > 100:
            suggestions.append({
                'type': 'execution_time',
                'priority': 'high',
                'suggestion': 'Implement caching for expensive operations',
                'expected_improvement': '30-60%'
            })
        
        # Memory usage
        memory_peak = metrics.get('memory', {}).get('peak_memory_mb', 0)
        if memory_peak > 100:
            suggestions.append({
                'type': 'memory',
                'priority': 'medium',
                'suggestion': 'Implement memory pooling or reduce object allocations',
                'expected_improvement': '20-40%'
            })
        
        # CPU usage
        cpu_avg = metrics.get('cpu', {}).get('avg_cpu_percent', 0)
        if cpu_avg > 70:
            suggestions.append({
                'type': 'cpu',
                'priority': 'high',
                'suggestion': 'Consider async operations or parallel processing',
                'expected_improvement': '25-50%'
            })
        
        # Hot functions
        hot_functions = metrics.get('functions', {}).get('hot_functions', [])
        if hot_functions:
            top_function = hot_functions[0]
            if top_function.get('cumulative_time', 0) > 0.1:
                suggestions.append({
                    'type': 'function_optimization',
                    'priority': 'high',
                    'suggestion': f"Optimize function: {top_function.get('function', 'unknown')}",
                    'expected_improvement': '15-30%'
                })
        
        return suggestions
    
    def _estimate_improvements(self, baseline: Dict[str, Any], suggestions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Estimate potential performance improvements"""
        
        total_improvement = 0
        improvements_by_type = {}
        
        for suggestion in suggestions:
            improvement_str = suggestion.get('expected_improvement', '0%')
            # Extract average improvement percentage
            if '-' in improvement_str:
                low, high = improvement_str.replace('%', '').split('-')
                avg_improvement = (float(low) + float(high)) / 2
            else:
                avg_improvement = float(improvement_str.replace('%', ''))
            
            improvements_by_type[suggestion['type']] = avg_improvement
            total_improvement += avg_improvement * 0.8  # Discount for realistic estimates
        
        # Cap total improvement at 90%
        total_improvement = min(total_improvement, 90)
        
        current_time = baseline.get('avg_time_ms', 0)
        estimated_new_time = current_time * (1 - total_improvement / 100)
        
        return {
            'total_estimated_improvement_percent': total_improvement,
            'current_avg_time_ms': current_time,
            'estimated_new_time_ms': estimated_new_time,
            'improvements_by_type': improvements_by_type
        }
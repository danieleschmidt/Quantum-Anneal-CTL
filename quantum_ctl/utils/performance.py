"""
Performance optimization utilities for quantum HVAC control.

Includes caching, memoization, parallel processing, and resource management.
"""

import asyncio
import functools
import hashlib
import json
import time
import threading
from typing import Dict, Any, Optional, Callable, Tuple, List
from dataclasses import dataclass, field
from collections import OrderedDict
import numpy as np
import logging


@dataclass
class CacheStats:
    """Cache performance statistics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_requests: int = 0
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        return self.hits / self.total_requests if self.total_requests > 0 else 0.0
    
    @property
    def miss_rate(self) -> float:
        """Calculate cache miss rate."""
        return self.misses / self.total_requests if self.total_requests > 0 else 0.0


class LRUCache:
    """Thread-safe LRU cache with performance monitoring."""
    
    def __init__(self, maxsize: int = 128, ttl: Optional[float] = None):
        self.maxsize = maxsize
        self.ttl = ttl
        self._cache: OrderedDict = OrderedDict()
        self._timestamps: Dict = {}
        self._lock = threading.RLock()
        self._stats = CacheStats()
        self.logger = logging.getLogger(__name__)
    
    def _is_expired(self, key: str) -> bool:
        """Check if cache entry is expired."""
        if self.ttl is None:
            return False
        
        timestamp = self._timestamps.get(key)
        if timestamp is None:
            return True
        
        return time.time() - timestamp > self.ttl
    
    def get(self, key: str) -> Any:
        """Get value from cache."""
        with self._lock:
            self._stats.total_requests += 1
            
            if key in self._cache and not self._is_expired(key):
                # Move to end (most recently used)
                value = self._cache.pop(key)
                self._cache[key] = value
                self._stats.hits += 1
                return value
            else:
                # Remove expired entry
                if key in self._cache:
                    del self._cache[key]
                    del self._timestamps[key]
                
                self._stats.misses += 1
                return None
    
    def put(self, key: str, value: Any) -> None:
        """Put value into cache."""
        with self._lock:
            # Remove if already exists
            if key in self._cache:
                del self._cache[key]
            
            # Evict if at capacity
            while len(self._cache) >= self.maxsize:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                del self._timestamps[oldest_key]
                self._stats.evictions += 1
            
            # Add new entry
            self._cache[key] = value
            self._timestamps[key] = time.time()
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._timestamps.clear()
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        return self._stats


class OptimizationCache:
    """Specialized cache for optimization results."""
    
    def __init__(self, maxsize: int = 64, ttl: float = 300):  # 5 minute TTL
        self._cache = LRUCache(maxsize, ttl)
        self.logger = logging.getLogger(__name__)
    
    def _make_key(self, state_vector: np.ndarray, weather: np.ndarray, 
                  prices: np.ndarray, config_hash: str) -> str:
        """Create cache key from optimization inputs."""
        # Create deterministic hash of inputs
        state_hash = hashlib.md5(state_vector.tobytes()).hexdigest()[:8]
        weather_hash = hashlib.md5(weather.tobytes()).hexdigest()[:8]
        price_hash = hashlib.md5(prices.tobytes()).hexdigest()[:8]
        
        return f"{state_hash}_{weather_hash}_{price_hash}_{config_hash[:8]}"
    
    def get_optimization(self, state_vector: np.ndarray, weather: np.ndarray,
                        prices: np.ndarray, config_hash: str) -> Optional[np.ndarray]:
        """Get cached optimization result."""
        key = self._make_key(state_vector, weather, prices, config_hash)
        result = self._cache.get(key)
        
        if result is not None:
            self.logger.debug(f"Cache hit for optimization key: {key}")
            return result
        else:
            self.logger.debug(f"Cache miss for optimization key: {key}")
            return None
    
    def put_optimization(self, state_vector: np.ndarray, weather: np.ndarray,
                        prices: np.ndarray, config_hash: str, 
                        result: np.ndarray) -> None:
        """Cache optimization result."""
        key = self._make_key(state_vector, weather, prices, config_hash)
        self._cache.put(key, result.copy())
        self.logger.debug(f"Cached optimization result for key: {key}")
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        return self._cache.get_stats()


class MatrixCache:
    """Cache for expensive matrix operations."""
    
    def __init__(self, maxsize: int = 32, ttl: float = 600):  # 10 minute TTL
        self._cache = LRUCache(maxsize, ttl)
        self.logger = logging.getLogger(__name__)
    
    def get_dynamics_matrices(self, building_hash: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Get cached dynamics matrices."""
        key = f"dynamics_{building_hash}"
        result = self._cache.get(key)
        
        if result is not None:
            self.logger.debug(f"Cache hit for dynamics matrices: {building_hash[:8]}")
            return result
        else:
            self.logger.debug(f"Cache miss for dynamics matrices: {building_hash[:8]}")
            return None
    
    def put_dynamics_matrices(self, building_hash: str, A: np.ndarray, B: np.ndarray) -> None:
        """Cache dynamics matrices."""
        key = f"dynamics_{building_hash}"
        self._cache.put(key, (A.copy(), B.copy()))
        self.logger.debug(f"Cached dynamics matrices for: {building_hash[:8]}")
    
    def get_cost_matrix(self, matrix_type: str, params_hash: str) -> Optional[np.ndarray]:
        """Get cached cost matrix."""
        key = f"{matrix_type}_{params_hash}"
        result = self._cache.get(key)
        
        if result is not None:
            self.logger.debug(f"Cache hit for {matrix_type} matrix")
            return result
        else:
            self.logger.debug(f"Cache miss for {matrix_type} matrix")
            return None
    
    def put_cost_matrix(self, matrix_type: str, params_hash: str, matrix: np.ndarray) -> None:
        """Cache cost matrix."""
        key = f"{matrix_type}_{params_hash}"
        self._cache.put(key, matrix.copy())
        self.logger.debug(f"Cached {matrix_type} matrix")
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        return self._cache.get_stats()


@dataclass
class PerformanceMetrics:
    """Performance metrics tracking."""
    optimization_times: List[float] = field(default_factory=list)
    cache_hit_rates: List[float] = field(default_factory=list)
    memory_usage: List[float] = field(default_factory=list)
    cpu_usage: List[float] = field(default_factory=list)
    
    def add_optimization_time(self, time_seconds: float) -> None:
        """Add optimization time measurement."""
        self.optimization_times.append(time_seconds)
        # Keep only last 100 measurements
        if len(self.optimization_times) > 100:
            self.optimization_times.pop(0)
    
    def add_cache_hit_rate(self, hit_rate: float) -> None:
        """Add cache hit rate measurement."""
        self.cache_hit_rates.append(hit_rate)
        if len(self.cache_hit_rates) > 100:
            self.cache_hit_rates.pop(0)
    
    @property
    def avg_optimization_time(self) -> float:
        """Average optimization time."""
        return np.mean(self.optimization_times) if self.optimization_times else 0.0
    
    @property
    def p95_optimization_time(self) -> float:
        """95th percentile optimization time."""
        return np.percentile(self.optimization_times, 95) if self.optimization_times else 0.0
    
    @property
    def avg_cache_hit_rate(self) -> float:
        """Average cache hit rate."""
        return np.mean(self.cache_hit_rates) if self.cache_hit_rates else 0.0


class ParallelProcessor:
    """Parallel processing utilities for optimization."""
    
    def __init__(self, max_workers: Optional[int] = None):
        self.max_workers = max_workers or min(4, (threading.active_count() or 1) + 1)
        self.logger = logging.getLogger(__name__)
    
    async def process_buildings_parallel(self, buildings: List, 
                                       optimization_func: Callable,
                                       *args, **kwargs) -> List[Any]:
        """Process multiple buildings in parallel."""
        semaphore = asyncio.Semaphore(self.max_workers)
        
        async def process_single(building, *args, **kwargs):
            async with semaphore:
                return await optimization_func(building, *args, **kwargs)
        
        tasks = [
            process_single(building, *args, **kwargs) 
            for building in buildings
        ]
        
        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        processing_time = time.time() - start_time
        
        self.logger.info(
            f"Processed {len(buildings)} buildings in {processing_time:.2f}s "
            f"with {self.max_workers} workers"
        )
        
        return results


class ResourceManager:
    """Resource management and optimization."""
    
    def __init__(self):
        self.optimization_cache = OptimizationCache()
        self.matrix_cache = MatrixCache()
        self.parallel_processor = ParallelProcessor()
        self.performance_metrics = PerformanceMetrics()
        self.logger = logging.getLogger(__name__)
    
    def get_optimization_cache(self) -> OptimizationCache:
        """Get optimization cache instance."""
        return self.optimization_cache
    
    def get_matrix_cache(self) -> MatrixCache:
        """Get matrix cache instance."""
        return self.matrix_cache
    
    def get_parallel_processor(self) -> ParallelProcessor:
        """Get parallel processor instance."""
        return self.parallel_processor
    
    def record_optimization_time(self, time_seconds: float) -> None:
        """Record optimization time for metrics."""
        self.performance_metrics.add_optimization_time(time_seconds)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        opt_stats = self.optimization_cache.get_stats()
        matrix_stats = self.matrix_cache.get_stats()
        
        return {
            'optimization_cache': {
                'hit_rate': opt_stats.hit_rate,
                'total_requests': opt_stats.total_requests,
                'evictions': opt_stats.evictions
            },
            'matrix_cache': {
                'hit_rate': matrix_stats.hit_rate,
                'total_requests': matrix_stats.total_requests,
                'evictions': matrix_stats.evictions
            },
            'performance': {
                'avg_optimization_time': self.performance_metrics.avg_optimization_time,
                'p95_optimization_time': self.performance_metrics.p95_optimization_time,
                'avg_cache_hit_rate': self.performance_metrics.avg_cache_hit_rate
            },
            'parallel_processing': {
                'max_workers': self.parallel_processor.max_workers
            }
        }
    
    def optimize_resource_usage(self) -> None:
        """Optimize resource usage based on current metrics."""
        # Clear caches if hit rate is too low
        opt_stats = self.optimization_cache.get_stats()
        if opt_stats.total_requests > 100 and opt_stats.hit_rate < 0.1:
            self.logger.warning("Low optimization cache hit rate, clearing cache")
            self.optimization_cache._cache.clear()
        
        matrix_stats = self.matrix_cache.get_stats()
        if matrix_stats.total_requests > 50 and matrix_stats.hit_rate < 0.2:
            self.logger.warning("Low matrix cache hit rate, clearing cache")
            self.matrix_cache._cache.clear()
        
        # Adjust parallel worker count based on performance
        if self.performance_metrics.avg_optimization_time > 5.0:
            # Increase parallelism for slow optimizations
            self.parallel_processor.max_workers = min(8, self.parallel_processor.max_workers + 1)
            self.logger.info(f"Increased parallel workers to {self.parallel_processor.max_workers}")
        elif self.performance_metrics.avg_optimization_time < 0.5:
            # Reduce parallelism for fast optimizations to save resources
            self.parallel_processor.max_workers = max(2, self.parallel_processor.max_workers - 1)
            self.logger.info(f"Reduced parallel workers to {self.parallel_processor.max_workers}")


# Global resource manager instance
_resource_manager = None

def get_resource_manager() -> ResourceManager:
    """Get global resource manager instance."""
    global _resource_manager
    if _resource_manager is None:
        _resource_manager = ResourceManager()
    return _resource_manager


def performance_monitor(func: Callable) -> Callable:
    """Decorator to monitor function performance."""
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            return result
        finally:
            execution_time = time.time() - start_time
            resource_manager = get_resource_manager()
            resource_manager.record_optimization_time(execution_time)
            
            logger = logging.getLogger(func.__module__)
            logger.debug(f"{func.__name__} executed in {execution_time:.3f}s")
    
    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            execution_time = time.time() - start_time
            resource_manager = get_resource_manager()
            resource_manager.record_optimization_time(execution_time)
            
            logger = logging.getLogger(func.__module__)
            logger.debug(f"{func.__name__} executed in {execution_time:.3f}s")
    
    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper


def cached_matrix_operation(matrix_type: str):
    """Decorator for caching expensive matrix operations."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            # Create hash of parameters
            params_str = json.dumps([str(arg) for arg in args] + [f"{k}={v}" for k, v in kwargs.items()], sort_keys=True)
            params_hash = hashlib.md5(params_str.encode()).hexdigest()
            
            # Try to get from cache
            resource_manager = get_resource_manager()
            matrix_cache = resource_manager.get_matrix_cache()
            
            cached_result = matrix_cache.get_cost_matrix(matrix_type, params_hash)
            if cached_result is not None:
                return cached_result
            
            # Compute and cache result
            result = func(self, *args, **kwargs)
            if isinstance(result, np.ndarray):
                matrix_cache.put_cost_matrix(matrix_type, params_hash, result)
            
            return result
        
        return wrapper
    return decorator
"""
Advanced Performance Optimization Suite for Quantum HVAC Systems.

This module provides sophisticated performance optimization including:
1. Machine Learning-based Performance Prediction
2. Adaptive Resource Allocation and Scheduling  
3. Multi-Level Caching with Intelligent Cache Management
4. Performance Profiling and Bottleneck Detection
5. Auto-tuning of Quantum Parameters
6. Memory and CPU Optimization
"""

from typing import Dict, Any, List, Optional, Tuple, Callable, Union
import numpy as np
from dataclasses import dataclass, field
import logging
import time
import asyncio
from datetime import datetime, timedelta
from collections import deque, defaultdict
from concurrent.futures import ThreadPoolExecutor
import threading
import psutil
import gc
import cProfile
import pstats
from io import StringIO

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    import joblib
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False


@dataclass
class PerformanceProfile:
    """Performance profiling result."""
    function_stats: Dict[str, Dict[str, float]]
    total_time: float
    memory_usage: Dict[str, float]
    cpu_usage: float
    bottlenecks: List[Dict[str, Any]]
    recommendations: List[str]
    
    @property
    def top_bottlenecks(self) -> List[str]:
        """Get top performance bottlenecks."""
        return [b['function'] for b in self.bottlenecks[:5]]


@dataclass 
class CacheEntry:
    """Enhanced cache entry with usage analytics."""
    key: str
    value: Any
    created_time: float
    last_accessed: float
    access_count: int = 0
    hit_rate: float = 0.0
    cost_to_compute: float = 0.0
    value_score: float = 1.0  # Higher = more valuable to cache
    
    @property
    def age_hours(self) -> float:
        return (time.time() - self.created_time) / 3600
    
    @property
    def access_frequency(self) -> float:
        """Access frequency per hour."""
        age = max(self.age_hours, 0.1)
        return self.access_count / age


@dataclass
class ResourceAllocationPlan:
    """Resource allocation optimization plan."""
    cpu_allocation: Dict[str, float]  # Process -> CPU percentage
    memory_allocation: Dict[str, int]  # Process -> MB
    thread_pool_sizes: Dict[str, int]  # Pool name -> size
    cache_sizes: Dict[str, int]       # Cache name -> max entries
    priority_adjustments: Dict[str, int]  # Task -> priority
    
    estimated_speedup: float = 1.0
    confidence: float = 0.5
    implementation_cost: float = 0.0


class PerformancePredictor:
    """Machine Learning-based performance prediction."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, Any] = {}
        self.training_data: deque = deque(maxlen=10000)
        
        if ML_AVAILABLE:
            self._initialize_models()
    
    def _initialize_models(self):
        """Initialize ML models for performance prediction."""
        # Solve time prediction model
        self.models['solve_time'] = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        # Quality prediction model  
        self.models['solution_quality'] = RandomForestRegressor(
            n_estimators=50,
            max_depth=8,
            random_state=42
        )
        
        # Resource usage prediction
        self.models['memory_usage'] = LinearRegression()
        self.models['cpu_usage'] = LinearRegression()
        
        # Scalers for feature normalization
        for model_name in self.models.keys():
            self.scalers[model_name] = StandardScaler()
    
    def record_performance(
        self,
        problem_features: Dict[str, float],
        performance_metrics: Dict[str, float]
    ) -> None:
        """Record performance data for model training."""
        
        training_sample = {
            'timestamp': time.time(),
            'features': problem_features,
            'metrics': performance_metrics
        }
        
        self.training_data.append(training_sample)
        
        # Retrain models periodically
        if len(self.training_data) % 100 == 0:
            asyncio.create_task(self._retrain_models())
    
    async def _retrain_models(self) -> None:
        """Retrain models with accumulated data."""
        
        if not ML_AVAILABLE or len(self.training_data) < 50:
            return
        
        try:
            # Prepare training data
            features_list = []
            targets = defaultdict(list)
            
            for sample in list(self.training_data)[-1000:]:  # Last 1000 samples
                features = [
                    sample['features'].get('problem_size', 0),
                    sample['features'].get('num_objectives', 0),
                    sample['features'].get('constraint_density', 0),
                    sample['features'].get('complexity_score', 0),
                    sample['features'].get('num_zones', 0),
                    sample['features'].get('horizon', 0)
                ]
                
                features_list.append(features)
                
                metrics = sample['metrics']
                targets['solve_time'].append(metrics.get('solve_time', 0))
                targets['solution_quality'].append(metrics.get('solution_quality', 0))
                targets['memory_usage'].append(metrics.get('memory_usage_mb', 0))
                targets['cpu_usage'].append(metrics.get('cpu_usage_percent', 0))
            
            X = np.array(features_list)
            
            # Train each model
            for model_name, model in self.models.items():
                if model_name in targets and len(targets[model_name]) > 0:
                    y = np.array(targets[model_name])
                    
                    # Scale features
                    X_scaled = self.scalers[model_name].fit_transform(X)
                    
                    # Train model
                    model.fit(X_scaled, y)
                    
                    # Log training completion
                    if hasattr(model, 'score'):
                        score = model.score(X_scaled, y)
                        self.logger.info(f"Retrained {model_name} model, R² score: {score:.3f}")
            
        except Exception as e:
            self.logger.error(f"Model retraining failed: {e}")
    
    def predict_performance(
        self,
        problem_features: Dict[str, float]
    ) -> Dict[str, float]:
        """Predict performance metrics for a problem."""
        
        if not ML_AVAILABLE:
            return self._fallback_prediction(problem_features)
        
        try:
            # Extract features
            features = np.array([[
                problem_features.get('problem_size', 0),
                problem_features.get('num_objectives', 1),
                problem_features.get('constraint_density', 0.5),
                problem_features.get('complexity_score', 1.0),
                problem_features.get('num_zones', 10),
                problem_features.get('horizon', 24)
            ]])
            
            predictions = {}
            
            # Make predictions with each model
            for model_name, model in self.models.items():
                if hasattr(model, 'predict'):
                    # Scale features
                    features_scaled = self.scalers[model_name].transform(features)
                    
                    # Predict
                    prediction = model.predict(features_scaled)[0]
                    predictions[model_name] = max(0, prediction)  # Ensure non-negative
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Performance prediction failed: {e}")
            return self._fallback_prediction(problem_features)
    
    def _fallback_prediction(self, problem_features: Dict[str, float]) -> Dict[str, float]:
        """Fallback prediction when ML is not available."""
        
        # Simple heuristic-based predictions
        problem_size = problem_features.get('problem_size', 100)
        complexity = problem_features.get('complexity_score', 1.0)
        
        return {
            'solve_time': problem_size * 0.1 * complexity,
            'solution_quality': max(0.3, 0.9 - complexity * 0.1),
            'memory_usage': problem_size * 0.5,  # MB
            'cpu_usage': min(100, problem_size * 0.2)  # Percentage
        }


class IntelligentCacheManager:
    """AI-powered cache management with predictive eviction."""
    
    def __init__(
        self,
        max_entries: int = 1000,
        max_memory_mb: int = 500,
        enable_ml_eviction: bool = True
    ):
        self.max_entries = max_entries
        self.max_memory_mb = max_memory_mb
        self.enable_ml_eviction = enable_ml_eviction
        
        self.logger = logging.getLogger(__name__)
        
        # Cache storage
        self._cache: Dict[str, CacheEntry] = {}
        self._access_history: deque = deque(maxlen=10000)
        
        # Performance tracking
        self._hit_count = 0
        self._miss_count = 0
        self._eviction_count = 0
        
        # ML-based eviction predictor
        if ML_AVAILABLE and enable_ml_eviction:
            self._eviction_predictor = RandomForestRegressor(
                n_estimators=50,
                max_depth=6,
                random_state=42
            )
            self._eviction_scaler = StandardScaler()
            self._eviction_trained = False
        
        # Background optimization
        self._optimization_task: Optional[asyncio.Task] = None
        self._optimization_running = False
    
    def start_optimization(self):
        """Start background cache optimization."""
        if not self._optimization_running:
            self._optimization_running = True
            self._optimization_task = asyncio.create_task(self._optimization_loop())
    
    def stop_optimization(self):
        """Stop background cache optimization."""
        self._optimization_running = False
        if self._optimization_task:
            self._optimization_task.cancel()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache with usage tracking."""
        
        if key in self._cache:
            entry = self._cache[key]
            entry.last_accessed = time.time()
            entry.access_count += 1
            
            self._hit_count += 1
            
            # Record access for ML training
            self._access_history.append({
                'timestamp': time.time(),
                'key': key,
                'access_type': 'hit',
                'entry_age': entry.age_hours,
                'access_count': entry.access_count
            })
            
            return entry.value
        else:
            self._miss_count += 1
            
            # Record miss
            self._access_history.append({
                'timestamp': time.time(),
                'key': key,
                'access_type': 'miss'
            })
            
            return None
    
    def put(
        self,
        key: str,
        value: Any,
        cost_to_compute: float = 1.0,
        expected_lifetime: float = 3600.0  # seconds
    ) -> None:
        """Put value in cache with intelligent eviction."""
        
        # Calculate value score
        value_score = self._calculate_value_score(key, cost_to_compute, expected_lifetime)
        
        # Create cache entry
        entry = CacheEntry(
            key=key,
            value=value,
            created_time=time.time(),
            last_accessed=time.time(),
            access_count=1,
            cost_to_compute=cost_to_compute,
            value_score=value_score
        )
        
        # Check if we need to evict
        if len(self._cache) >= self.max_entries or self._get_cache_memory_mb() > self.max_memory_mb:
            self._intelligent_eviction()
        
        # Store entry
        self._cache[key] = entry
    
    def _calculate_value_score(
        self,
        key: str,
        cost_to_compute: float,
        expected_lifetime: float
    ) -> float:
        """Calculate value score for cache entry."""
        
        # Base score from computation cost
        cost_score = min(10.0, cost_to_compute) / 10.0
        
        # Lifetime score
        lifetime_score = min(1.0, expected_lifetime / 3600.0)  # Normalize to 1 hour
        
        # Historical access pattern (if available)
        history_score = 0.5
        similar_accesses = [
            h for h in self._access_history
            if h.get('key', '').startswith(key.split('_')[0])  # Similar key pattern
        ]
        
        if similar_accesses:
            recent_accesses = [
                h for h in similar_accesses
                if time.time() - h['timestamp'] < 3600  # Last hour
            ]
            history_score = min(1.0, len(recent_accesses) / 10.0)
        
        # Combined score
        return 0.4 * cost_score + 0.3 * lifetime_score + 0.3 * history_score
    
    def _intelligent_eviction(self) -> None:
        """Perform intelligent cache eviction."""
        
        if not self._cache:
            return
        
        # Use ML-based eviction if available and trained
        if (self.enable_ml_eviction and ML_AVAILABLE and 
            self._eviction_trained and len(self._cache) > 10):
            
            candidates = self._ml_select_eviction_candidates()
        else:
            # Use heuristic-based eviction
            candidates = self._heuristic_select_eviction_candidates()
        
        # Evict candidates
        evicted_count = 0
        target_evictions = max(1, len(self._cache) // 10)  # Evict 10%
        
        for key in candidates[:target_evictions]:
            if key in self._cache:
                del self._cache[key]
                evicted_count += 1
        
        self._eviction_count += evicted_count
        
        if evicted_count > 0:
            self.logger.debug(f"Evicted {evicted_count} cache entries")
    
    def _ml_select_eviction_candidates(self) -> List[str]:
        """Use ML to select eviction candidates."""
        
        try:
            # Prepare features for each cache entry
            features_list = []
            keys = []
            
            for key, entry in self._cache.items():
                features = [
                    entry.age_hours,
                    entry.access_count,
                    entry.access_frequency,
                    entry.value_score,
                    entry.cost_to_compute,
                    time.time() - entry.last_accessed  # Time since last access
                ]
                
                features_list.append(features)
                keys.append(key)
            
            # Predict eviction priority (higher = more likely to evict)
            X = self._eviction_scaler.transform(features_list)
            eviction_scores = self._eviction_predictor.predict(X)
            
            # Sort by eviction score (descending)
            candidates_with_scores = list(zip(keys, eviction_scores))
            candidates_with_scores.sort(key=lambda x: x[1], reverse=True)
            
            return [key for key, score in candidates_with_scores]
            
        except Exception as e:
            self.logger.error(f"ML eviction selection failed: {e}")
            return self._heuristic_select_eviction_candidates()
    
    def _heuristic_select_eviction_candidates(self) -> List[str]:
        """Use heuristics to select eviction candidates."""
        
        # Score each entry for eviction (higher = more likely to evict)
        candidates_with_scores = []
        
        for key, entry in self._cache.items():
            # Eviction score factors
            age_factor = entry.age_hours / 24.0  # Prefer older entries
            access_factor = 1.0 / (1.0 + entry.access_frequency)  # Prefer less accessed
            value_factor = 1.0 / (1.0 + entry.value_score)  # Prefer lower value
            
            eviction_score = 0.4 * age_factor + 0.4 * access_factor + 0.2 * value_factor
            candidates_with_scores.append((key, eviction_score))
        
        # Sort by eviction score (descending)
        candidates_with_scores.sort(key=lambda x: x[1], reverse=True)
        
        return [key for key, score in candidates_with_scores]
    
    def _get_cache_memory_mb(self) -> float:
        """Estimate cache memory usage in MB."""
        
        total_size = 0
        
        for entry in self._cache.values():
            # Rough memory estimation
            if hasattr(entry.value, '__sizeof__'):
                total_size += entry.value.__sizeof__()
            else:
                total_size += 1024  # Default 1KB estimate
        
        return total_size / (1024 * 1024)  # Convert to MB
    
    async def _optimization_loop(self):
        """Background cache optimization loop."""
        
        while self._optimization_running:
            try:
                # Periodic optimization tasks
                await self._update_entry_statistics()
                await self._train_eviction_predictor()
                await self._optimize_cache_structure()
                
                await asyncio.sleep(60)  # Run every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Cache optimization error: {e}")
                await asyncio.sleep(60)
    
    async def _update_entry_statistics(self):
        """Update cache entry statistics."""
        
        current_time = time.time()
        
        for entry in self._cache.values():
            # Update hit rate
            if entry.access_count > 0:
                # Simple exponential moving average
                recent_hits = sum(
                    1 for h in self._access_history
                    if (h.get('key') == entry.key and 
                        current_time - h['timestamp'] < 3600 and
                        h.get('access_type') == 'hit')
                )
                
                recent_requests = sum(
                    1 for h in self._access_history
                    if (h.get('key') == entry.key and 
                        current_time - h['timestamp'] < 3600)
                )
                
                if recent_requests > 0:
                    entry.hit_rate = recent_hits / recent_requests
    
    async def _train_eviction_predictor(self):
        """Train ML model for eviction prediction."""
        
        if not (ML_AVAILABLE and self.enable_ml_eviction):
            return
        
        if len(self._access_history) < 100:
            return
        
        try:
            # Prepare training data
            features_list = []
            labels = []
            
            # Use historical access patterns to learn eviction patterns
            for i, access in enumerate(list(self._access_history)[-500:]):
                if access.get('access_type') == 'hit':
                    # Look for entries that were later evicted (didn't appear in recent history)
                    key = access.get('key')
                    timestamp = access.get('timestamp')
                    
                    # Check if key appears in recent history (not evicted)
                    future_accesses = [
                        h for h in list(self._access_history)[i+1:]
                        if h.get('key') == key and h['timestamp'] > timestamp + 1800  # 30 min later
                    ]
                    
                    was_evicted = len(future_accesses) == 0
                    
                    # Create features (simplified)
                    features = [
                        access.get('entry_age', 1.0),
                        access.get('access_count', 1),
                        1.0,  # access_frequency placeholder
                        0.5,  # value_score placeholder
                        1.0,  # cost_to_compute placeholder
                        0.0   # time_since_last_access placeholder
                    ]
                    
                    features_list.append(features)
                    labels.append(1.0 if was_evicted else 0.0)
            
            if len(features_list) >= 20:
                X = np.array(features_list)
                y = np.array(labels)
                
                # Scale features
                X_scaled = self._eviction_scaler.fit_transform(X)
                
                # Train model
                self._eviction_predictor.fit(X_scaled, y)
                self._eviction_trained = True
                
                self.logger.debug("Trained eviction predictor")
            
        except Exception as e:
            self.logger.error(f"Eviction predictor training failed: {e}")
    
    async def _optimize_cache_structure(self):
        """Optimize cache data structures and memory layout."""
        
        # Cleanup expired entries
        current_time = time.time()
        expired_keys = []
        
        for key, entry in self._cache.items():
            # Mark entries as expired if not accessed for a long time
            if current_time - entry.last_accessed > 86400:  # 24 hours
                expired_keys.append(key)
        
        for key in expired_keys[:10]:  # Limit cleanup per iteration
            if key in self._cache:
                del self._cache[key]
                self._eviction_count += 1
        
        # Force garbage collection if memory usage is high
        memory_mb = self._get_cache_memory_mb()
        if memory_mb > self.max_memory_mb * 0.9:
            gc.collect()
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        
        total_requests = self._hit_count + self._miss_count
        hit_rate = self._hit_count / max(total_requests, 1)
        
        # Entry age distribution
        if self._cache:
            ages = [entry.age_hours for entry in self._cache.values()]
            age_stats = {
                'min_age_hours': min(ages),
                'max_age_hours': max(ages),
                'avg_age_hours': np.mean(ages),
                'median_age_hours': np.median(ages)
            }
        else:
            age_stats = {}
        
        return {
            'total_entries': len(self._cache),
            'max_entries': self.max_entries,
            'memory_usage_mb': self._get_cache_memory_mb(),
            'max_memory_mb': self.max_memory_mb,
            'hit_count': self._hit_count,
            'miss_count': self._miss_count,
            'hit_rate': hit_rate,
            'eviction_count': self._eviction_count,
            'age_statistics': age_stats,
            'ml_eviction_enabled': self.enable_ml_eviction and self._eviction_trained
        }


class AdvancedPerformanceOptimizer:
    """
    Advanced performance optimization system for quantum HVAC.
    
    Features:
    - ML-based performance prediction
    - Intelligent resource allocation
    - Multi-level caching with AI management
    - Real-time performance profiling
    - Auto-tuning of system parameters
    """
    
    def __init__(
        self,
        enable_ml_prediction: bool = True,
        enable_profiling: bool = False,
        cache_size_mb: int = 200
    ):
        self.enable_ml_prediction = enable_ml_prediction
        self.enable_profiling = enable_profiling
        
        self.logger = logging.getLogger(__name__)
        
        # Performance prediction
        self.performance_predictor = PerformancePredictor() if enable_ml_prediction else None
        
        # Intelligent caching
        self.cache_manager = IntelligentCacheManager(
            max_memory_mb=cache_size_mb,
            enable_ml_eviction=enable_ml_prediction
        )
        
        # Resource monitoring
        self._system_metrics: deque = deque(maxlen=1000)
        self._performance_profiles: deque = deque(maxlen=100)
        
        # Auto-tuning parameters
        self._tuned_parameters: Dict[str, Any] = {}
        self._parameter_history: deque = deque(maxlen=500)
        
        # Background tasks
        self._monitoring_task: Optional[asyncio.Task] = None
        self._optimization_running = False
        
        # Thread pools for different workloads
        self._thread_pools: Dict[str, ThreadPoolExecutor] = {
            'quantum_solve': ThreadPoolExecutor(max_workers=2, thread_name_prefix='quantum'),
            'classical_solve': ThreadPoolExecutor(max_workers=4, thread_name_prefix='classical'),
            'data_processing': ThreadPoolExecutor(max_workers=3, thread_name_prefix='data')
        }
    
    async def start_optimization(self):
        """Start background optimization processes."""
        if self._optimization_running:
            return
        
        self._optimization_running = True
        
        # Start cache optimization
        self.cache_manager.start_optimization()
        
        # Start monitoring task
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        self.logger.info("Started advanced performance optimization")
    
    async def stop_optimization(self):
        """Stop background optimization processes."""
        self._optimization_running = False
        
        # Stop cache optimization
        self.cache_manager.stop_optimization()
        
        # Stop monitoring
        if self._monitoring_task:
            self._monitoring_task.cancel()
        
        # Shutdown thread pools
        for name, pool in self._thread_pools.items():
            pool.shutdown(wait=True)
        
        self.logger.info("Stopped advanced performance optimization")
    
    def predict_solve_performance(
        self,
        problem_characteristics: Dict[str, Any]
    ) -> Dict[str, float]:
        """Predict performance metrics for a solve request."""
        
        if self.performance_predictor:
            return self.performance_predictor.predict_performance(problem_characteristics)
        else:
            # Fallback heuristics
            problem_size = problem_characteristics.get('problem_size', 100)
            complexity = problem_characteristics.get('complexity_score', 1.0)
            
            return {
                'solve_time': problem_size * 0.05 * complexity,
                'solution_quality': max(0.5, 0.95 - complexity * 0.05),
                'memory_usage': problem_size * 0.3,
                'cpu_usage': min(100, problem_size * 0.15)
            }
    
    def record_solve_performance(
        self,
        problem_characteristics: Dict[str, Any],
        actual_performance: Dict[str, Any]
    ):
        """Record actual performance for ML model training."""
        
        if self.performance_predictor:
            self.performance_predictor.record_performance(
                problem_characteristics,
                actual_performance
            )
    
    def optimize_resource_allocation(
        self,
        pending_requests: List[Dict[str, Any]],
        system_resources: Dict[str, float]
    ) -> ResourceAllocationPlan:
        """Create optimal resource allocation plan."""
        
        # Analyze current system state
        current_cpu = system_resources.get('cpu_percent', 50)
        current_memory = system_resources.get('memory_percent', 50)
        current_load = len(pending_requests)
        
        # Predict resource needs for pending requests
        total_cpu_need = 0
        total_memory_need = 0
        
        for request in pending_requests:
            prediction = self.predict_solve_performance(request)
            total_cpu_need += prediction.get('cpu_usage', 20)
            total_memory_need += prediction.get('memory_usage', 50)
        
        # Create allocation plan
        plan = ResourceAllocationPlan(
            cpu_allocation={},
            memory_allocation={},
            thread_pool_sizes={},
            cache_sizes={}
        )
        
        # Optimize thread pool sizes
        if current_load > 5:  # High load
            plan.thread_pool_sizes['quantum_solve'] = min(4, current_load // 2)
            plan.thread_pool_sizes['classical_solve'] = min(6, current_load)
        else:  # Normal load
            plan.thread_pool_sizes['quantum_solve'] = 2
            plan.thread_pool_sizes['classical_solve'] = 3
        
        # Adjust cache sizes based on memory pressure
        if current_memory > 80:  # High memory usage
            plan.cache_sizes['solution_cache'] = 500
            plan.cache_sizes['embedding_cache'] = 100
        else:
            plan.cache_sizes['solution_cache'] = 1000
            plan.cache_sizes['embedding_cache'] = 200
        
        # Set priority adjustments
        for i, request in enumerate(pending_requests):
            priority = request.get('priority', 1)
            if i < 3:  # Top 3 requests get boosted priority
                plan.priority_adjustments[str(i)] = priority + 1
        
        # Estimate performance improvement
        plan.estimated_speedup = self._estimate_speedup_from_plan(plan, pending_requests)
        plan.confidence = 0.7  # Moderate confidence in optimization
        
        return plan
    
    def _estimate_speedup_from_plan(
        self,
        plan: ResourceAllocationPlan,
        requests: List[Dict[str, Any]]
    ) -> float:
        """Estimate speedup from resource allocation plan."""
        
        # Simple heuristic speedup estimation
        base_speedup = 1.0
        
        # Thread pool optimization speedup
        total_threads = sum(plan.thread_pool_sizes.values())
        if total_threads > 8:
            base_speedup *= 1.3  # Parallelization benefit
        
        # Cache size optimization speedup
        total_cache = sum(plan.cache_sizes.values())
        if total_cache > 1000:
            base_speedup *= 1.2  # Cache hit benefit
        
        # Priority optimization speedup (for high-priority requests)
        if len(plan.priority_adjustments) > 0:
            base_speedup *= 1.1
        
        return min(base_speedup, 2.0)  # Cap at 2x speedup
    
    def profile_function(self, func: Callable) -> PerformanceProfile:
        """Profile function performance and identify bottlenecks."""
        
        if not self.enable_profiling:
            return PerformanceProfile(
                function_stats={},
                total_time=0.0,
                memory_usage={},
                cpu_usage=0.0,
                bottlenecks=[],
                recommendations=[]
            )
        
        # Memory usage before
        process = psutil.Process()
        memory_before = process.memory_info().rss / (1024 * 1024)  # MB
        
        # CPU profiling
        profiler = cProfile.Profile()
        
        start_time = time.time()
        profiler.enable()
        
        try:
            result = func()
        finally:
            profiler.disable()
            end_time = time.time()
        
        # Memory usage after
        memory_after = process.memory_info().rss / (1024 * 1024)  # MB
        
        # Process profiling results
        stats_stream = StringIO()
        stats = pstats.Stats(profiler, stream=stats_stream)
        stats.sort_stats('cumulative')
        stats.print_stats(20)  # Top 20 functions
        
        # Parse function statistics
        function_stats = {}
        bottlenecks = []
        
        for func_info, (cc, nc, tt, ct, callers) in stats.stats.items():
            func_name = f"{func_info[0]}:{func_info[1]}:{func_info[2]}"
            
            function_stats[func_name] = {
                'call_count': cc,
                'total_time': tt,
                'cumulative_time': ct,
                'avg_time': tt / max(cc, 1)
            }
            
            # Identify bottlenecks (high cumulative time)
            if ct > (end_time - start_time) * 0.1:  # >10% of total time
                bottlenecks.append({
                    'function': func_name,
                    'cumulative_time': ct,
                    'percentage': ct / (end_time - start_time) * 100,
                    'call_count': cc
                })
        
        # Sort bottlenecks by time
        bottlenecks.sort(key=lambda x: x['cumulative_time'], reverse=True)
        
        # Generate recommendations
        recommendations = self._generate_performance_recommendations(bottlenecks, function_stats)
        
        profile = PerformanceProfile(
            function_stats=function_stats,
            total_time=end_time - start_time,
            memory_usage={
                'before_mb': memory_before,
                'after_mb': memory_after,
                'delta_mb': memory_after - memory_before
            },
            cpu_usage=process.cpu_percent(),
            bottlenecks=bottlenecks,
            recommendations=recommendations
        )
        
        self._performance_profiles.append(profile)
        
        return profile
    
    def _generate_performance_recommendations(
        self,
        bottlenecks: List[Dict[str, Any]],
        function_stats: Dict[str, Dict[str, float]]
    ) -> List[str]:
        """Generate performance optimization recommendations."""
        
        recommendations = []
        
        # Analyze bottlenecks
        for bottleneck in bottlenecks[:3]:  # Top 3 bottlenecks
            func_name = bottleneck['function']
            percentage = bottleneck['percentage']
            
            if percentage > 20:
                recommendations.append(
                    f"Critical bottleneck in {func_name} ({percentage:.1f}% of time) - "
                    "consider algorithm optimization or caching"
                )
            elif percentage > 10:
                recommendations.append(
                    f"Performance bottleneck in {func_name} ({percentage:.1f}% of time) - "
                    "consider optimization"
                )
        
        # Analyze function call patterns
        high_call_functions = [
            name for name, stats in function_stats.items()
            if stats['call_count'] > 1000
        ]
        
        if high_call_functions:
            recommendations.append(
                f"High call count functions detected: {len(high_call_functions)} functions - "
                "consider reducing call frequency or optimizing hot paths"
            )
        
        # General recommendations
        if not recommendations:
            recommendations.append("Performance profile looks good - no major bottlenecks detected")
        
        return recommendations
    
    async def auto_tune_parameters(
        self,
        parameter_space: Dict[str, Tuple[float, float]],
        objective_function: Callable,
        max_iterations: int = 20
    ) -> Dict[str, Any]:
        """Auto-tune parameters using optimization algorithms."""
        
        self.logger.info(f"Starting auto-tuning with {len(parameter_space)} parameters")
        
        best_params = {}
        best_score = float('-inf')
        iteration_history = []
        
        # Initialize with current parameter values or defaults
        current_params = {}
        for param, (min_val, max_val) in parameter_space.items():
            if param in self._tuned_parameters:
                current_params[param] = self._tuned_parameters[param]
            else:
                current_params[param] = (min_val + max_val) / 2
        
        # Simple grid search with adaptive refinement
        for iteration in range(max_iterations):
            # Generate parameter candidates around current best
            candidates = self._generate_parameter_candidates(
                current_params if iteration == 0 else best_params,
                parameter_space,
                exploration_factor=max(0.1, 0.5 - iteration * 0.02)
            )
            
            # Test each candidate
            for candidate in candidates:
                try:
                    # Evaluate objective function with these parameters
                    score = await self._evaluate_parameters(objective_function, candidate)
                    
                    iteration_history.append({
                        'iteration': iteration,
                        'parameters': candidate.copy(),
                        'score': score
                    })
                    
                    if score > best_score:
                        best_score = score
                        best_params = candidate.copy()
                        
                        self.logger.info(
                            f"New best parameters found: score={score:.3f}, "
                            f"params={best_params}"
                        )
                
                except Exception as e:
                    self.logger.error(f"Parameter evaluation failed: {e}")
        
        # Update tuned parameters
        self._tuned_parameters.update(best_params)
        
        # Record tuning session
        tuning_result = {
            'best_parameters': best_params,
            'best_score': best_score,
            'total_iterations': max_iterations,
            'evaluations': len(iteration_history),
            'improvement': best_score - iteration_history[0]['score'] if iteration_history else 0,
            'history': iteration_history
        }
        
        self._parameter_history.append({
            'timestamp': time.time(),
            'result': tuning_result
        })
        
        self.logger.info(
            f"Auto-tuning completed: best_score={best_score:.3f}, "
            f"improvement={tuning_result['improvement']:.3f}"
        )
        
        return tuning_result
    
    def _generate_parameter_candidates(
        self,
        base_params: Dict[str, float],
        parameter_space: Dict[str, Tuple[float, float]],
        exploration_factor: float = 0.3,
        num_candidates: int = 5
    ) -> List[Dict[str, float]]:
        """Generate parameter candidates for optimization."""
        
        candidates = []
        
        for _ in range(num_candidates):
            candidate = {}
            
            for param, (min_val, max_val) in parameter_space.items():
                base_val = base_params.get(param, (min_val + max_val) / 2)
                
                # Add exploration noise
                noise_range = (max_val - min_val) * exploration_factor
                noise = np.random.uniform(-noise_range, noise_range)
                
                new_val = np.clip(base_val + noise, min_val, max_val)
                candidate[param] = new_val
            
            candidates.append(candidate)
        
        return candidates
    
    async def _evaluate_parameters(
        self,
        objective_function: Callable,
        parameters: Dict[str, float]
    ) -> float:
        """Evaluate objective function with given parameters."""
        
        # Run objective function with timeout
        try:
            if asyncio.iscoroutinefunction(objective_function):
                score = await asyncio.wait_for(
                    objective_function(parameters),
                    timeout=30.0
                )
            else:
                # Run in executor for non-async functions
                score = await asyncio.get_event_loop().run_in_executor(
                    None, objective_function, parameters
                )
            
            return float(score)
            
        except asyncio.TimeoutError:
            self.logger.warning("Parameter evaluation timed out")
            return float('-inf')
        except Exception as e:
            self.logger.error(f"Parameter evaluation error: {e}")
            return float('-inf')
    
    async def _monitoring_loop(self):
        """Background monitoring loop for system metrics."""
        
        while self._optimization_running:
            try:
                # Collect system metrics
                process = psutil.Process()
                system_metrics = {
                    'timestamp': time.time(),
                    'cpu_percent': process.cpu_percent(),
                    'memory_percent': process.memory_percent(),
                    'memory_rss_mb': process.memory_info().rss / (1024 * 1024),
                    'thread_count': process.num_threads(),
                    'open_files': len(process.open_files()),
                    'load_average': psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0.0
                }
                
                self._system_metrics.append(system_metrics)
                
                # Check for performance issues
                await self._check_performance_alerts(system_metrics)
                
                await asyncio.sleep(30)  # Monitor every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(30)
    
    async def _check_performance_alerts(self, metrics: Dict[str, float]):
        """Check system metrics for performance alerts."""
        
        # CPU usage alert
        if metrics['cpu_percent'] > 90:
            self.logger.warning(f"High CPU usage: {metrics['cpu_percent']:.1f}%")
        
        # Memory usage alert
        if metrics['memory_percent'] > 85:
            self.logger.warning(f"High memory usage: {metrics['memory_percent']:.1f}%")
        
        # Thread count alert
        if metrics['thread_count'] > 50:
            self.logger.warning(f"High thread count: {metrics['thread_count']}")
        
        # Open files alert
        if metrics['open_files'] > 100:
            self.logger.warning(f"Many open files: {metrics['open_files']}")
    
    def get_optimization_status(self) -> Dict[str, Any]:
        """Get comprehensive optimization system status."""
        
        # Cache statistics
        cache_stats = self.cache_manager.get_cache_statistics()
        
        # System metrics summary
        if self._system_metrics:
            recent_metrics = list(self._system_metrics)[-10:]
            system_summary = {
                'avg_cpu_percent': np.mean([m['cpu_percent'] for m in recent_metrics]),
                'avg_memory_percent': np.mean([m['memory_percent'] for m in recent_metrics]),
                'current_memory_mb': recent_metrics[-1]['memory_rss_mb'],
                'current_threads': recent_metrics[-1]['thread_count']
            }
        else:
            system_summary = {}
        
        # Performance profiles summary
        profile_summary = {}
        if self._performance_profiles:
            recent_profiles = list(self._performance_profiles)[-5:]
            profile_summary = {
                'total_profiles': len(self._performance_profiles),
                'avg_execution_time': np.mean([p.total_time for p in recent_profiles]),
                'common_bottlenecks': self._get_common_bottlenecks()
            }
        
        # Thread pool status
        thread_pool_status = {}
        for name, pool in self._thread_pools.items():
            thread_pool_status[name] = {
                'max_workers': pool._max_workers,
                'active_threads': len([t for t in pool._threads if t.is_alive()])
            }
        
        return {
            'optimization_running': self._optimization_running,
            'ml_prediction_enabled': self.performance_predictor is not None,
            'profiling_enabled': self.enable_profiling,
            'cache_statistics': cache_stats,
            'system_metrics': system_summary,
            'performance_profiles': profile_summary,
            'thread_pools': thread_pool_status,
            'tuned_parameters': self._tuned_parameters,
            'tuning_sessions': len(self._parameter_history)
        }
    
    def _get_common_bottlenecks(self) -> List[str]:
        """Get most common performance bottlenecks."""
        
        bottleneck_counts = defaultdict(int)
        
        for profile in self._performance_profiles:
            for bottleneck in profile.bottlenecks[:3]:  # Top 3 per profile
                func_name = bottleneck['function'].split(':')[-1]  # Get function name only
                bottleneck_counts[func_name] += 1
        
        # Return top 5 most common bottlenecks
        common = sorted(bottleneck_counts.items(), key=lambda x: x[1], reverse=True)
        return [func_name for func_name, count in common[:5]]


# Global performance optimizer instance
_global_performance_optimizer: Optional[AdvancedPerformanceOptimizer] = None


def get_performance_optimizer() -> AdvancedPerformanceOptimizer:
    """Get global performance optimizer instance."""
    global _global_performance_optimizer
    if _global_performance_optimizer is None:
        _global_performance_optimizer = AdvancedPerformanceOptimizer()
    return _global_performance_optimizer


def reset_performance_optimizer():
    """Reset global performance optimizer instance."""
    global _global_performance_optimizer
    if _global_performance_optimizer:
        asyncio.create_task(_global_performance_optimizer.stop_optimization())
    _global_performance_optimizer = None
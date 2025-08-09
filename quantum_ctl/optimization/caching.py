"""
Advanced caching system for quantum HVAC optimization.

Provides intelligent caching, warm starts, and performance optimization
for quantum annealing solutions.
"""

import asyncio
import hashlib
import logging
import pickle
import time
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json

import numpy as np
import redis
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler


class CacheLevel(Enum):
    """Cache storage levels."""
    MEMORY = "memory"
    REDIS = "redis" 
    DISK = "disk"


@dataclass
class CacheKey:
    """Structured cache key for quantum problems."""
    problem_type: str  # 'qubo', 'ising', 'mpc'
    problem_size: int
    objective_hash: str
    constraint_hash: str
    solver_config: str
    
    def to_string(self) -> str:
        """Convert to cache key string."""
        return f"{self.problem_type}:{self.problem_size}:{self.objective_hash}:{self.constraint_hash}:{self.solver_config}"
    
    @classmethod
    def from_problem_data(
        cls,
        problem_type: str,
        Q_matrix: np.ndarray,
        solver_params: Dict[str, Any]
    ) -> 'CacheKey':
        """Create cache key from problem data."""
        
        # Hash the Q matrix structure and values
        q_normalized = Q_matrix / np.max(np.abs(Q_matrix)) if np.max(np.abs(Q_matrix)) > 0 else Q_matrix
        objective_hash = hashlib.sha256(q_normalized.tobytes()).hexdigest()[:16]
        
        # Hash constraint patterns (diagonal, structure)
        constraint_pattern = np.abs(Q_matrix) > 0
        constraint_hash = hashlib.sha256(constraint_pattern.tobytes()).hexdigest()[:16]
        
        # Hash solver configuration
        solver_config_str = json.dumps(solver_params, sort_keys=True)
        solver_config = hashlib.sha256(solver_config_str.encode()).hexdigest()[:16]
        
        return cls(
            problem_type=problem_type,
            problem_size=Q_matrix.shape[0],
            objective_hash=objective_hash,
            constraint_hash=constraint_hash,
            solver_config=solver_config
        )


@dataclass
class CachedSolution:
    """Cached quantum solution with metadata."""
    solution: np.ndarray
    energy: float
    solve_time: float
    solver_info: Dict[str, Any]
    timestamp: float
    similarity_features: np.ndarray  # Features for similarity matching
    usage_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    
    def is_expired(self, ttl_seconds: int) -> bool:
        """Check if cached solution is expired."""
        return (time.time() - self.timestamp) > ttl_seconds
    
    def get_age_seconds(self) -> float:
        """Get age of cached solution in seconds."""
        return time.time() - self.timestamp
    
    def mark_used(self) -> None:
        """Mark solution as used."""
        self.usage_count += 1
        self.last_accessed = time.time()


class QuantumCache:
    """
    Intelligent cache for quantum optimization solutions.
    
    Supports similarity matching, warm starts, and multi-level caching.
    """
    
    def __init__(
        self,
        max_memory_cache: int = 1000,
        similarity_threshold: float = 0.95,
        redis_url: Optional[str] = None,
        disk_cache_dir: Optional[str] = None
    ):
        self.max_memory_cache = max_memory_cache
        self.similarity_threshold = similarity_threshold
        self.logger = logging.getLogger("quantum_cache")
        
        # Memory cache
        self.memory_cache: Dict[str, CachedSolution] = {}
        self.cache_access_order: List[str] = []  # For LRU eviction
        
        # Redis cache
        self.redis_client: Optional[redis.Redis] = None
        if redis_url:
            try:
                self.redis_client = redis.from_url(redis_url)
                self.redis_client.ping()  # Test connection
                self.logger.info(f"Connected to Redis cache: {redis_url}")
            except Exception as e:
                self.logger.warning(f"Failed to connect to Redis: {e}")
        
        # Disk cache directory
        self.disk_cache_dir = disk_cache_dir
        if disk_cache_dir:
            import os
            os.makedirs(disk_cache_dir, exist_ok=True)
        
        # Feature scaler for similarity matching
        self.scaler = StandardScaler()
        self._scaler_fitted = False
        
        # Performance tracking
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'stores': 0,
            'evictions': 0,
            'similarity_matches': 0
        }
    
    def _extract_problem_features(self, Q_matrix: np.ndarray) -> np.ndarray:
        """Extract features from Q matrix for similarity matching."""
        features = []
        
        # Matrix properties
        features.extend([
            Q_matrix.shape[0],  # Size
            np.count_nonzero(Q_matrix),  # Sparsity
            np.trace(Q_matrix),  # Diagonal sum
            np.sum(Q_matrix),  # Total sum
            np.max(Q_matrix),  # Maximum value
            np.min(Q_matrix),  # Minimum value
            np.std(Q_matrix),  # Standard deviation
        ])
        
        # Eigenvalue properties (first few)
        try:
            eigenvals = np.linalg.eigvals(Q_matrix)
            eigenvals_sorted = np.sort(eigenvals)[::-1]  # Descending
            features.extend(eigenvals_sorted[:5].tolist())  # Top 5 eigenvalues
            if len(eigenvals_sorted) < 5:
                features.extend([0.0] * (5 - len(eigenvals_sorted)))
        except Exception:
            features.extend([0.0] * 5)  # Fallback for numerical issues
        
        # Structure features
        upper_triangle = Q_matrix[np.triu_indices(Q_matrix.shape[0], k=1)]
        features.extend([
            np.mean(np.abs(upper_triangle)),  # Upper triangle mean
            np.std(upper_triangle),  # Upper triangle std
            np.count_nonzero(upper_triangle) / len(upper_triangle)  # Upper triangle density
        ])
        
        return np.array(features)
    
    def _compute_similarity(
        self,
        features1: np.ndarray,
        features2: np.ndarray
    ) -> float:
        """Compute similarity between two feature vectors."""
        try:
            # Ensure features are 2D for sklearn
            feat1_2d = features1.reshape(1, -1)
            feat2_2d = features2.reshape(1, -1)
            
            # Scale features if scaler is fitted
            if self._scaler_fitted:
                feat1_2d = self.scaler.transform(feat1_2d)
                feat2_2d = self.scaler.transform(feat2_2d)
            
            # Compute cosine similarity
            similarity = cosine_similarity(feat1_2d, feat2_2d)[0, 0]
            return float(similarity)
            
        except Exception as e:
            self.logger.warning(f"Similarity computation failed: {e}")
            return 0.0
    
    def _fit_scaler_if_needed(self, all_features: List[np.ndarray]) -> None:
        """Fit the scaler on all available features."""
        if not self._scaler_fitted and len(all_features) >= 2:
            try:
                feature_matrix = np.vstack(all_features)
                self.scaler.fit(feature_matrix)
                self._scaler_fitted = True
                self.logger.debug("Fitted feature scaler")
            except Exception as e:
                self.logger.warning(f"Failed to fit scaler: {e}")
    
    def _find_similar_solution(
        self,
        problem_features: np.ndarray,
        cache_level: CacheLevel = CacheLevel.MEMORY
    ) -> Optional[Tuple[str, CachedSolution]]:
        """Find most similar cached solution."""
        
        best_similarity = 0.0
        best_match = None
        
        # Search in memory cache
        if cache_level == CacheLevel.MEMORY or cache_level == CacheLevel.REDIS:
            for cache_key, solution in self.memory_cache.items():
                similarity = self._compute_similarity(
                    problem_features,
                    solution.similarity_features
                )
                
                if similarity > best_similarity and similarity >= self.similarity_threshold:
                    best_similarity = similarity
                    best_match = (cache_key, solution)
        
        # TODO: Add Redis and disk cache similarity search
        
        if best_match:
            self.cache_stats['similarity_matches'] += 1
            self.logger.debug(f"Found similar solution with similarity {best_similarity:.3f}")
        
        return best_match
    
    def get(
        self,
        cache_key: CacheKey,
        Q_matrix: Optional[np.ndarray] = None
    ) -> Optional[CachedSolution]:
        """
        Get cached solution by exact key or similarity matching.
        
        Args:
            cache_key: Exact cache key to lookup
            Q_matrix: Problem matrix for similarity matching
            
        Returns:
            Cached solution if found, None otherwise
        """
        key_str = cache_key.to_string()
        
        # Try exact match in memory first
        if key_str in self.memory_cache:
            solution = self.memory_cache[key_str]
            solution.mark_used()
            
            # Move to end of access order (LRU)
            if key_str in self.cache_access_order:
                self.cache_access_order.remove(key_str)
            self.cache_access_order.append(key_str)
            
            self.cache_stats['hits'] += 1
            self.logger.debug(f"Cache hit (exact): {key_str}")
            return solution
        
        # Try Redis exact match
        if self.redis_client:
            try:
                cached_data = self.redis_client.get(f"quantum_cache:{key_str}")
                if cached_data:
                    solution = pickle.loads(cached_data)
                    solution.mark_used()
                    
                    # Promote to memory cache
                    self._store_in_memory(key_str, solution)
                    
                    self.cache_stats['hits'] += 1
                    self.logger.debug(f"Cache hit (Redis): {key_str}")
                    return solution
            except Exception as e:
                self.logger.warning(f"Redis cache error: {e}")
        
        # Try similarity matching if Q_matrix provided
        if Q_matrix is not None:
            problem_features = self._extract_problem_features(Q_matrix)
            
            # Update scaler with new features
            all_features = [problem_features]
            all_features.extend([sol.similarity_features for sol in self.memory_cache.values()])
            self._fit_scaler_if_needed(all_features)
            
            similar_match = self._find_similar_solution(problem_features)
            if similar_match:
                similar_key, solution = similar_match
                solution.mark_used()
                
                self.cache_stats['hits'] += 1
                self.logger.info(f"Cache hit (similarity): {similar_key} -> {key_str}")
                return solution
        
        # Cache miss
        self.cache_stats['misses'] += 1
        return None
    
    def put(
        self,
        cache_key: CacheKey,
        solution: np.ndarray,
        energy: float,
        solve_time: float,
        solver_info: Dict[str, Any],
        Q_matrix: Optional[np.ndarray] = None
    ) -> None:
        """
        Store solution in cache.
        
        Args:
            cache_key: Cache key for the solution
            solution: Solution vector
            energy: Solution energy
            solve_time: Time taken to solve
            solver_info: Solver metadata
            Q_matrix: Problem matrix for feature extraction
        """
        key_str = cache_key.to_string()
        
        # Extract features for similarity matching
        features = (
            self._extract_problem_features(Q_matrix)
            if Q_matrix is not None
            else np.zeros(15)  # Default feature size
        )
        
        cached_solution = CachedSolution(
            solution=solution,
            energy=energy,
            solve_time=solve_time,
            solver_info=solver_info,
            timestamp=time.time(),
            similarity_features=features
        )
        
        # Store in memory
        self._store_in_memory(key_str, cached_solution)
        
        # Store in Redis (async to avoid blocking)
        if self.redis_client:
            asyncio.create_task(self._store_in_redis(key_str, cached_solution))
        
        # TODO: Store in disk cache
        
        self.cache_stats['stores'] += 1
        self.logger.debug(f"Cached solution: {key_str}")
    
    def _store_in_memory(self, key: str, solution: CachedSolution) -> None:
        """Store solution in memory cache with LRU eviction."""
        
        # Add/update in cache
        self.memory_cache[key] = solution
        
        # Update access order
        if key in self.cache_access_order:
            self.cache_access_order.remove(key)
        self.cache_access_order.append(key)
        
        # Evict oldest if over limit
        while len(self.memory_cache) > self.max_memory_cache:
            oldest_key = self.cache_access_order.pop(0)
            if oldest_key in self.memory_cache:
                del self.memory_cache[oldest_key]
                self.cache_stats['evictions'] += 1
    
    async def _store_in_redis(self, key: str, solution: CachedSolution) -> None:
        """Store solution in Redis cache asynchronously."""
        try:
            serialized = pickle.dumps(solution)
            redis_key = f"quantum_cache:{key}"
            
            # Store with TTL of 24 hours
            self.redis_client.setex(redis_key, 86400, serialized)
            
        except Exception as e:
            self.logger.warning(f"Failed to store in Redis: {e}")
    
    def invalidate(self, cache_key: CacheKey) -> bool:
        """Remove solution from all cache levels."""
        key_str = cache_key.to_string()
        removed = False
        
        # Remove from memory
        if key_str in self.memory_cache:
            del self.memory_cache[key_str]
            if key_str in self.cache_access_order:
                self.cache_access_order.remove(key_str)
            removed = True
        
        # Remove from Redis
        if self.redis_client:
            try:
                redis_key = f"quantum_cache:{key_str}"
                self.redis_client.delete(redis_key)
            except Exception as e:
                self.logger.warning(f"Redis invalidate error: {e}")
        
        return removed
    
    def clear(self) -> None:
        """Clear all cached solutions."""
        self.memory_cache.clear()
        self.cache_access_order.clear()
        
        if self.redis_client:
            try:
                # Remove all quantum cache keys
                keys = self.redis_client.keys("quantum_cache:*")
                if keys:
                    self.redis_client.delete(*keys)
            except Exception as e:
                self.logger.warning(f"Redis clear error: {e}")
        
        self.logger.info("Cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total_requests = self.cache_stats['hits'] + self.cache_stats['misses']
        hit_rate = (
            self.cache_stats['hits'] / total_requests
            if total_requests > 0 else 0.0
        )
        
        return {
            **self.cache_stats,
            'total_requests': total_requests,
            'hit_rate': hit_rate,
            'memory_cache_size': len(self.memory_cache),
            'redis_available': self.redis_client is not None,
            'scaler_fitted': self._scaler_fitted
        }
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get detailed cache information."""
        memory_solutions = []
        
        for key, solution in self.memory_cache.items():
            memory_solutions.append({
                'key': key,
                'energy': solution.energy,
                'solve_time': solution.solve_time,
                'age_seconds': solution.get_age_seconds(),
                'usage_count': solution.usage_count,
                'last_accessed': solution.last_accessed
            })
        
        # Sort by last accessed (most recent first)
        memory_solutions.sort(key=lambda x: x['last_accessed'], reverse=True)
        
        return {
            'stats': self.get_cache_stats(),
            'memory_solutions': memory_solutions,
            'config': {
                'max_memory_cache': self.max_memory_cache,
                'similarity_threshold': self.similarity_threshold,
                'redis_enabled': self.redis_client is not None,
                'disk_cache_enabled': self.disk_cache_dir is not None
            }
        }
    
    def optimize_cache(self) -> Dict[str, int]:
        """Optimize cache by removing expired and unused solutions."""
        initial_size = len(self.memory_cache)
        
        # Find expired solutions (older than 1 hour with no recent access)
        current_time = time.time()
        expired_keys = []
        
        for key, solution in self.memory_cache.items():
            # Remove if very old and unused
            if (current_time - solution.last_accessed > 3600 and  # 1 hour
                solution.usage_count <= 1):
                expired_keys.append(key)
        
        # Remove expired solutions
        for key in expired_keys:
            if key in self.memory_cache:
                del self.memory_cache[key]
            if key in self.cache_access_order:
                self.cache_access_order.remove(key)
        
        removed_count = len(expired_keys)
        final_size = len(self.memory_cache)
        
        if removed_count > 0:
            self.logger.info(f"Cache optimization removed {removed_count} expired solutions")
        
        return {
            'initial_size': initial_size,
            'final_size': final_size,
            'removed_count': removed_count
        }


class WarmStartManager:
    """
    Manages warm starts for quantum optimization.
    
    Uses cached solutions as initial states to improve solve times.
    """
    
    def __init__(self, cache: QuantumCache):
        self.cache = cache
        self.logger = logging.getLogger("warm_start_manager")
    
    def get_warm_start(
        self,
        cache_key: CacheKey,
        Q_matrix: np.ndarray,
        target_size: Optional[int] = None
    ) -> Optional[np.ndarray]:
        """
        Get warm start solution for problem.
        
        Args:
            cache_key: Cache key for the problem
            Q_matrix: Problem matrix
            target_size: Target solution size (if different from cached)
            
        Returns:
            Initial solution vector or None
        """
        # Try to get similar cached solution
        cached_solution = self.cache.get(cache_key, Q_matrix)
        
        if cached_solution is None:
            return None
        
        solution = cached_solution.solution
        
        # Adapt solution size if needed
        if target_size and len(solution) != target_size:
            solution = self._adapt_solution_size(solution, target_size)
        
        # Add some random perturbation to avoid getting stuck in local minima
        solution = self._perturb_solution(solution, perturbation_rate=0.05)
        
        self.logger.debug(f"Generated warm start from cached solution")
        return solution
    
    def _adapt_solution_size(
        self,
        solution: np.ndarray,
        target_size: int
    ) -> np.ndarray:
        """Adapt cached solution to different problem size."""
        current_size = len(solution)
        
        if current_size == target_size:
            return solution
        
        elif current_size < target_size:
            # Extend solution by repeating pattern or random values
            extension_size = target_size - current_size
            if current_size > 0:
                # Repeat existing pattern
                repeats = (extension_size // current_size) + 1
                extended = np.tile(solution, repeats)[:extension_size]
            else:
                extended = np.random.choice([0, 1], extension_size)
            
            return np.concatenate([solution, extended])
        
        else:
            # Truncate solution
            return solution[:target_size]
    
    def _perturb_solution(
        self,
        solution: np.ndarray,
        perturbation_rate: float = 0.05
    ) -> np.ndarray:
        """Add small random perturbation to solution."""
        perturbed = solution.copy()
        
        # Randomly flip some bits
        n_flips = max(1, int(len(solution) * perturbation_rate))
        flip_indices = np.random.choice(len(solution), n_flips, replace=False)
        
        for idx in flip_indices:
            perturbed[idx] = 1 - perturbed[idx]  # Flip binary variable
        
        return perturbed
    
    def evaluate_warm_start_quality(
        self,
        warm_start: np.ndarray,
        Q_matrix: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate quality of warm start solution.
        
        Args:
            warm_start: Initial solution vector
            Q_matrix: Problem QUBO matrix
            
        Returns:
            Quality metrics
        """
        # Compute energy of warm start
        energy = np.dot(warm_start, np.dot(Q_matrix, warm_start))
        
        # Compute constraint violations (if any)
        violations = 0  # TODO: Add constraint checking
        
        # Compute solution entropy (diversity measure)
        unique_values = len(np.unique(warm_start))
        max_entropy = np.log2(len(warm_start)) if len(warm_start) > 1 else 1
        entropy = np.log2(unique_values) / max_entropy if max_entropy > 0 else 0
        
        return {
            'energy': float(energy),
            'constraint_violations': violations,
            'entropy': entropy,
            'solution_quality': -energy / len(warm_start)  # Normalized quality
        }


class PerformanceOptimizer:
    """
    Performance optimization for quantum operations.
    
    Provides adaptive configurations and resource management.
    """
    
    def __init__(self):
        self.logger = logging.getLogger("performance_optimizer")
        
        # Performance history
        self.solve_history: List[Dict[str, Any]] = []
        self.max_history = 1000
        
        # Adaptive parameters
        self.current_solver_config = {
            'num_reads': 1000,
            'annealing_time': 20,
            'chain_strength': None
        }
    
    def record_solve(
        self,
        problem_size: int,
        solver_config: Dict[str, Any],
        solve_time: float,
        energy: float,
        chain_break_fraction: float = 0.0,
        success: bool = True
    ) -> None:
        """Record solve performance for adaptive optimization."""
        
        record = {
            'timestamp': time.time(),
            'problem_size': problem_size,
            'solver_config': solver_config.copy(),
            'solve_time': solve_time,
            'energy': energy,
            'chain_break_fraction': chain_break_fraction,
            'success': success
        }
        
        self.solve_history.append(record)
        
        # Keep only recent history
        if len(self.solve_history) > self.max_history:
            self.solve_history.pop(0)
    
    def get_optimal_config(
        self,
        problem_size: int,
        time_budget: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Get optimal solver configuration based on problem size and history.
        
        Args:
            problem_size: Size of the problem to solve
            time_budget: Maximum time allowed for solving
            
        Returns:
            Optimal solver configuration
        """
        
        # Filter history for similar problem sizes
        size_tolerance = 0.2  # 20% tolerance
        min_size = problem_size * (1 - size_tolerance)
        max_size = problem_size * (1 + size_tolerance)
        
        relevant_history = [
            record for record in self.solve_history
            if (min_size <= record['problem_size'] <= max_size and
                record['success'])
        ]
        
        if not relevant_history:
            # No relevant history, use defaults
            return self.current_solver_config.copy()
        
        # Analyze performance patterns
        best_configs = sorted(
            relevant_history,
            key=lambda x: (x['energy'], x['solve_time'])  # Minimize energy, then time
        )[:5]  # Top 5 configs
        
        # Extract optimal parameters
        optimal_config = self.current_solver_config.copy()
        
        # Adaptive num_reads based on time budget and problem size
        if time_budget:
            # Estimate reads based on time budget
            avg_time_per_read = 0.001  # Rough estimate: 1ms per read
            max_reads = int(time_budget / avg_time_per_read)
            optimal_config['num_reads'] = min(max_reads, 5000)
        else:
            # Use successful configuration average
            avg_reads = np.mean([cfg['solver_config']['num_reads'] for cfg in best_configs])
            optimal_config['num_reads'] = max(100, int(avg_reads))
        
        # Adaptive annealing time based on problem size
        if problem_size < 100:
            optimal_config['annealing_time'] = 20
        elif problem_size < 500:
            optimal_config['annealing_time'] = 50
        else:
            optimal_config['annealing_time'] = 100
        
        # Adaptive chain strength based on historical performance
        chain_strengths = [
            cfg['solver_config'].get('chain_strength')
            for cfg in best_configs
            if cfg['solver_config'].get('chain_strength') is not None
        ]
        
        if chain_strengths:
            optimal_config['chain_strength'] = np.median(chain_strengths)
        
        self.logger.debug(
            f"Optimal config for size {problem_size}: "
            f"reads={optimal_config['num_reads']}, "
            f"annealing_time={optimal_config['annealing_time']}"
        )
        
        return optimal_config
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance analysis metrics."""
        if not self.solve_history:
            return {'no_data': True}
        
        recent_history = self.solve_history[-100:]  # Last 100 solves
        
        solve_times = [r['solve_time'] for r in recent_history if r['success']]
        energies = [r['energy'] for r in recent_history if r['success']]
        chain_breaks = [r['chain_break_fraction'] for r in recent_history if r['success']]
        
        success_rate = sum(r['success'] for r in recent_history) / len(recent_history)
        
        if solve_times:
            return {
                'success_rate': success_rate,
                'avg_solve_time': np.mean(solve_times),
                'median_solve_time': np.median(solve_times),
                'avg_energy': np.mean(energies),
                'avg_chain_breaks': np.mean(chain_breaks),
                'total_solves': len(self.solve_history),
                'recent_solves': len(recent_history)
            }
        else:
            return {
                'success_rate': success_rate,
                'total_solves': len(self.solve_history),
                'recent_solves': len(recent_history),
                'all_failed': True
            }


# Global instances
_global_cache = QuantumCache()
_global_warm_start_manager = WarmStartManager(_global_cache)
_global_performance_optimizer = PerformanceOptimizer()


def get_global_cache() -> QuantumCache:
    """Get global quantum cache instance."""
    return _global_cache


def get_warm_start_manager() -> WarmStartManager:
    """Get global warm start manager."""
    return _global_warm_start_manager


def get_performance_optimizer() -> PerformanceOptimizer:
    """Get global performance optimizer."""
    return _global_performance_optimizer
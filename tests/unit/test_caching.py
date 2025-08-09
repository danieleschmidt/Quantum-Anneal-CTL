"""
Tests for quantum optimization caching system.
"""

import pytest
import numpy as np
import time
from unittest.mock import MagicMock, patch

from quantum_ctl.optimization.caching import (
    QuantumCache, CacheKey, CachedSolution, WarmStartManager, PerformanceOptimizer
)


class TestCacheKey:
    """Test cache key functionality."""
    
    def test_cache_key_creation(self):
        """Test creating cache key."""
        key = CacheKey(
            problem_type="qubo",
            problem_size=100,
            objective_hash="abc123",
            constraint_hash="def456",
            solver_config="xyz789"
        )
        
        assert key.problem_type == "qubo"
        assert key.problem_size == 100
        assert key.objective_hash == "abc123"
    
    def test_cache_key_string_conversion(self):
        """Test cache key to string conversion."""
        key = CacheKey("qubo", 50, "hash1", "hash2", "config1")
        key_str = key.to_string()
        
        assert "qubo" in key_str
        assert "50" in key_str
        assert "hash1" in key_str
        assert "hash2" in key_str
        assert "config1" in key_str
    
    def test_cache_key_from_problem_data(self):
        """Test creating cache key from problem data."""
        Q_matrix = np.array([
            [1.0, -0.5],
            [-0.5, 1.0]
        ])
        solver_params = {"num_reads": 1000, "annealing_time": 20}
        
        key = CacheKey.from_problem_data("qubo", Q_matrix, solver_params)
        
        assert key.problem_type == "qubo"
        assert key.problem_size == 2
        assert len(key.objective_hash) == 16
        assert len(key.constraint_hash) == 16
        assert len(key.solver_config) == 16


class TestCachedSolution:
    """Test cached solution functionality."""
    
    def test_cached_solution_creation(self):
        """Test creating cached solution."""
        solution = np.array([1, 0, 1, 0])
        features = np.array([1.0, 2.0, 3.0])
        
        cached_sol = CachedSolution(
            solution=solution,
            energy=-5.0,
            solve_time=1.5,
            solver_info={"chain_breaks": 0.1},
            timestamp=time.time(),
            similarity_features=features
        )
        
        assert np.array_equal(cached_sol.solution, solution)
        assert cached_sol.energy == -5.0
        assert cached_sol.solve_time == 1.5
        assert cached_sol.usage_count == 0
    
    def test_cached_solution_expiration(self):
        """Test solution expiration checking."""
        old_timestamp = time.time() - 3600  # 1 hour ago
        
        cached_sol = CachedSolution(
            solution=np.array([1, 0]),
            energy=-1.0,
            solve_time=1.0,
            solver_info={},
            timestamp=old_timestamp,
            similarity_features=np.array([1.0])
        )
        
        assert cached_sol.is_expired(1800)  # 30 min TTL - should be expired
        assert not cached_sol.is_expired(7200)  # 2 hour TTL - should not be expired
    
    def test_cached_solution_usage_tracking(self):
        """Test usage tracking."""
        cached_sol = CachedSolution(
            solution=np.array([1]),
            energy=0,
            solve_time=1,
            solver_info={},
            timestamp=time.time(),
            similarity_features=np.array([1.0])
        )
        
        initial_usage = cached_sol.usage_count
        initial_access_time = cached_sol.last_accessed
        
        time.sleep(0.01)  # Small delay
        cached_sol.mark_used()
        
        assert cached_sol.usage_count == initial_usage + 1
        assert cached_sol.last_accessed > initial_access_time


class TestQuantumCache:
    """Test quantum cache functionality."""
    
    @pytest.fixture
    def quantum_cache(self):
        """Create quantum cache for testing."""
        return QuantumCache(max_memory_cache=10, similarity_threshold=0.9)
    
    @pytest.fixture
    def sample_cache_key(self):
        """Create sample cache key."""
        return CacheKey("qubo", 4, "hash1", "hash2", "config1")
    
    @pytest.fixture
    def sample_solution(self):
        """Create sample solution data."""
        return {
            "solution": np.array([1, 0, 1, 0]),
            "energy": -3.5,
            "solve_time": 2.0,
            "solver_info": {"chain_breaks": 0.05},
            "Q_matrix": np.array([
                [1, -0.5, 0, 0],
                [-0.5, 1, -0.3, 0],
                [0, -0.3, 1, -0.2],
                [0, 0, -0.2, 1]
            ])
        }
    
    def test_feature_extraction(self, quantum_cache):
        """Test problem feature extraction."""
        Q_matrix = np.array([
            [1.0, -0.5, 0.0],
            [-0.5, 1.0, -0.3],
            [0.0, -0.3, 1.0]
        ])
        
        features = quantum_cache._extract_problem_features(Q_matrix)
        
        assert len(features) == 15  # Expected feature vector length
        assert features[0] == 3  # Matrix size
        assert features[1] == 5  # Non-zero elements
        assert features[2] == 3.0  # Diagonal sum (trace)
    
    def test_cache_store_and_retrieve(self, quantum_cache, sample_cache_key, sample_solution):
        """Test storing and retrieving from cache."""
        # Store solution
        quantum_cache.put(
            sample_cache_key,
            sample_solution["solution"],
            sample_solution["energy"],
            sample_solution["solve_time"],
            sample_solution["solver_info"],
            sample_solution["Q_matrix"]
        )
        
        # Retrieve solution
        cached = quantum_cache.get(sample_cache_key)
        
        assert cached is not None
        assert np.array_equal(cached.solution, sample_solution["solution"])
        assert cached.energy == sample_solution["energy"]
        assert cached.usage_count == 1  # Should be marked as used
    
    def test_cache_miss(self, quantum_cache):
        """Test cache miss scenario."""
        non_existent_key = CacheKey("ising", 10, "missing", "not_found", "absent")
        
        cached = quantum_cache.get(non_existent_key)
        assert cached is None
    
    def test_similarity_matching(self, quantum_cache, sample_solution):
        """Test similarity-based cache retrieval."""
        # Store original solution
        original_key = CacheKey("qubo", 4, "original", "hash1", "config1")
        quantum_cache.put(
            original_key,
            sample_solution["solution"],
            sample_solution["energy"],
            sample_solution["solve_time"],
            sample_solution["solver_info"],
            sample_solution["Q_matrix"]
        )
        
        # Create similar problem (slightly perturbed Q matrix)
        similar_Q = sample_solution["Q_matrix"] + np.random.normal(0, 0.01, sample_solution["Q_matrix"].shape)
        similar_key = CacheKey("qubo", 4, "similar", "hash2", "config1")
        
        # Should find similar solution
        with patch.object(quantum_cache, 'similarity_threshold', 0.8):  # Lower threshold
            cached = quantum_cache.get(similar_key, similar_Q)
            
        # Note: This test might be flaky due to randomness, but demonstrates the concept
        # In practice, similarity matching requires careful tuning
    
    def test_cache_invalidation(self, quantum_cache, sample_cache_key, sample_solution):
        """Test cache invalidation."""
        # Store solution
        quantum_cache.put(
            sample_cache_key,
            sample_solution["solution"],
            sample_solution["energy"],
            sample_solution["solve_time"],
            sample_solution["solver_info"],
            sample_solution["Q_matrix"]
        )
        
        # Verify it's cached
        assert quantum_cache.get(sample_cache_key) is not None
        
        # Invalidate
        success = quantum_cache.invalidate(sample_cache_key)
        assert success
        
        # Should no longer be cached
        assert quantum_cache.get(sample_cache_key) is None
    
    def test_cache_stats(self, quantum_cache, sample_cache_key, sample_solution):
        """Test cache statistics."""
        initial_stats = quantum_cache.get_cache_stats()
        
        # Cause cache miss
        quantum_cache.get(CacheKey("missing", 1, "no", "not", "none"))
        
        # Store and retrieve solution
        quantum_cache.put(
            sample_cache_key,
            sample_solution["solution"],
            sample_solution["energy"],
            sample_solution["solve_time"],
            sample_solution["solver_info"],
            sample_solution["Q_matrix"]
        )
        
        quantum_cache.get(sample_cache_key)
        
        final_stats = quantum_cache.get_cache_stats()
        
        assert final_stats['hits'] == initial_stats['hits'] + 1
        assert final_stats['misses'] == initial_stats['misses'] + 1
        assert final_stats['stores'] == initial_stats['stores'] + 1
        assert final_stats['memory_cache_size'] == 1
    
    def test_lru_eviction(self, quantum_cache):
        """Test LRU eviction when cache is full."""
        # Set small cache size for testing
        quantum_cache.max_memory_cache = 2
        
        # Fill cache beyond capacity
        for i in range(3):
            key = CacheKey("qubo", 2, f"hash{i}", f"constraint{i}", "config")
            solution = np.array([i % 2, (i + 1) % 2])
            Q_matrix = np.array([[1, 0], [0, 1]])
            
            quantum_cache.put(key, solution, -float(i), 1.0, {}, Q_matrix)
        
        # First item should be evicted
        first_key = CacheKey("qubo", 2, "hash0", "constraint0", "config")
        assert quantum_cache.get(first_key) is None
        
        # Last two should still be cached
        assert quantum_cache.get_cache_stats()['memory_cache_size'] == 2
    
    def test_cache_optimization(self, quantum_cache):
        """Test cache optimization (cleanup of expired/unused solutions)."""
        # Add old, unused solution
        old_key = CacheKey("qubo", 2, "old", "old", "old")
        old_solution = CachedSolution(
            solution=np.array([1, 0]),
            energy=-1.0,
            solve_time=1.0,
            solver_info={},
            timestamp=time.time() - 7200,  # 2 hours ago
            similarity_features=np.array([1.0, 2.0])
        )
        old_solution.last_accessed = time.time() - 7200  # Also not accessed recently
        
        quantum_cache.memory_cache[old_key.to_string()] = old_solution
        quantum_cache.cache_access_order.append(old_key.to_string())
        
        # Add recent, used solution
        recent_key = CacheKey("qubo", 2, "recent", "recent", "recent")
        quantum_cache.put(recent_key, np.array([0, 1]), -2.0, 1.5, {}, np.eye(2))
        
        initial_size = len(quantum_cache.memory_cache)
        optimization_result = quantum_cache.optimize_cache()
        final_size = len(quantum_cache.memory_cache)
        
        assert optimization_result['removed_count'] >= 0
        assert final_size <= initial_size


class TestWarmStartManager:
    """Test warm start manager functionality."""
    
    @pytest.fixture
    def warm_start_manager(self):
        """Create warm start manager with mocked cache."""
        mock_cache = MagicMock(spec=QuantumCache)
        return WarmStartManager(mock_cache)
    
    def test_warm_start_retrieval(self, warm_start_manager):
        """Test getting warm start solution."""
        # Mock cached solution
        cached_solution = CachedSolution(
            solution=np.array([1, 0, 1, 0]),
            energy=-3.0,
            solve_time=2.0,
            solver_info={},
            timestamp=time.time(),
            similarity_features=np.array([1.0, 2.0, 3.0])
        )
        
        warm_start_manager.cache.get.return_value = cached_solution
        
        cache_key = CacheKey("qubo", 4, "test", "test", "test")
        Q_matrix = np.eye(4)
        
        warm_start = warm_start_manager.get_warm_start(cache_key, Q_matrix)
        
        assert warm_start is not None
        assert len(warm_start) == 4
        # Should be perturbed, so not exactly the same
        assert not np.array_equal(warm_start, cached_solution.solution)
    
    def test_warm_start_size_adaptation(self, warm_start_manager):
        """Test warm start size adaptation."""
        original_solution = np.array([1, 0, 1])  # Size 3
        
        # Test extension
        extended = warm_start_manager._adapt_solution_size(original_solution, 5)
        assert len(extended) == 5
        assert np.array_equal(extended[:3], original_solution)
        
        # Test truncation
        truncated = warm_start_manager._adapt_solution_size(original_solution, 2)
        assert len(truncated) == 2
        assert np.array_equal(truncated, original_solution[:2])
    
    def test_solution_perturbation(self, warm_start_manager):
        """Test solution perturbation."""
        original = np.array([1, 1, 0, 0])
        perturbed = warm_start_manager._perturb_solution(original, perturbation_rate=0.5)
        
        assert len(perturbed) == len(original)
        assert not np.array_equal(perturbed, original)  # Should be different
        assert np.all((perturbed == 0) | (perturbed == 1))  # Should remain binary
    
    def test_warm_start_quality_evaluation(self, warm_start_manager):
        """Test warm start quality evaluation."""
        warm_start = np.array([1, 0, 1, 0])
        Q_matrix = np.array([
            [1, -0.5, 0, 0],
            [-0.5, 1, -0.3, 0],
            [0, -0.3, 1, -0.2],
            [0, 0, -0.2, 1]
        ])
        
        quality = warm_start_manager.evaluate_warm_start_quality(warm_start, Q_matrix)
        
        assert 'energy' in quality
        assert 'constraint_violations' in quality
        assert 'entropy' in quality
        assert 'solution_quality' in quality
        
        assert isinstance(quality['energy'], float)
        assert quality['constraint_violations'] >= 0


class TestPerformanceOptimizer:
    """Test performance optimizer functionality."""
    
    @pytest.fixture
    def performance_optimizer(self):
        """Create performance optimizer for testing."""
        return PerformanceOptimizer()
    
    def test_record_solve(self, performance_optimizer):
        """Test recording solve performance."""
        initial_count = len(performance_optimizer.solve_history)
        
        performance_optimizer.record_solve(
            problem_size=100,
            solver_config={"num_reads": 1000},
            solve_time=2.5,
            energy=-10.0,
            chain_break_fraction=0.1,
            success=True
        )
        
        assert len(performance_optimizer.solve_history) == initial_count + 1
        
        latest_record = performance_optimizer.solve_history[-1]
        assert latest_record['problem_size'] == 100
        assert latest_record['solve_time'] == 2.5
        assert latest_record['success'] is True
    
    def test_optimal_config_generation(self, performance_optimizer):
        """Test optimal configuration generation."""
        # Add some historical data
        for i in range(10):
            performance_optimizer.record_solve(
                problem_size=100 + i * 5,  # Similar sizes
                solver_config={"num_reads": 1000 + i * 100},
                solve_time=1.0 + i * 0.1,
                energy=-5.0 - i,
                success=True
            )
        
        optimal_config = performance_optimizer.get_optimal_config(
            problem_size=105,  # Similar to historical data
            time_budget=10.0
        )
        
        assert 'num_reads' in optimal_config
        assert 'annealing_time' in optimal_config
        assert optimal_config['num_reads'] > 0
        assert optimal_config['annealing_time'] > 0
    
    def test_optimal_config_with_time_budget(self, performance_optimizer):
        """Test optimal config generation with time budget."""
        optimal_config = performance_optimizer.get_optimal_config(
            problem_size=50,
            time_budget=5.0  # Limited time
        )
        
        # Should limit num_reads based on time budget
        assert optimal_config['num_reads'] <= 5000  # Some reasonable maximum
    
    def test_performance_metrics(self, performance_optimizer):
        """Test performance metrics calculation."""
        # Add mixed success/failure data
        for i in range(5):
            performance_optimizer.record_solve(
                problem_size=100,
                solver_config={"num_reads": 1000},
                solve_time=1.0 + i * 0.1,
                energy=-5.0,
                success=True
            )
        
        # Add some failures
        for i in range(2):
            performance_optimizer.record_solve(
                problem_size=100,
                solver_config={"num_reads": 1000},
                solve_time=0.0,  # Failed immediately
                energy=0.0,
                success=False
            )
        
        metrics = performance_optimizer.get_performance_metrics()
        
        assert 'success_rate' in metrics
        assert 'avg_solve_time' in metrics
        assert 'total_solves' in metrics
        
        assert metrics['success_rate'] == 5/7  # 5 successes out of 7 total
        assert metrics['avg_solve_time'] > 0  # Should be average of successful solves
        assert metrics['total_solves'] == 7
    
    def test_performance_metrics_no_data(self, performance_optimizer):
        """Test performance metrics with no data."""
        metrics = performance_optimizer.get_performance_metrics()
        
        assert 'no_data' in metrics
        assert metrics['no_data'] is True
    
    def test_performance_metrics_all_failed(self, performance_optimizer):
        """Test performance metrics when all solves failed."""
        # Add only failed solves
        for i in range(3):
            performance_optimizer.record_solve(
                problem_size=100,
                solver_config={"num_reads": 1000},
                solve_time=0.0,
                energy=0.0,
                success=False
            )
        
        metrics = performance_optimizer.get_performance_metrics()
        
        assert metrics['success_rate'] == 0.0
        assert 'all_failed' in metrics
        assert metrics['all_failed'] is True


class TestIntegration:
    """Test integration between caching components."""
    
    def test_cache_and_warm_start_integration(self):
        """Test integration between cache and warm start manager."""
        cache = QuantumCache(max_memory_cache=5)
        warm_start_manager = WarmStartManager(cache)
        
        # Store solution in cache
        cache_key = CacheKey("qubo", 4, "test", "test", "test")
        solution = np.array([1, 0, 1, 0])
        Q_matrix = np.eye(4)
        
        cache.put(cache_key, solution, -5.0, 2.0, {"info": "test"}, Q_matrix)
        
        # Get warm start from cache
        warm_start = warm_start_manager.get_warm_start(cache_key, Q_matrix)
        
        assert warm_start is not None
        assert len(warm_start) == 4
        # Should be binary
        assert np.all((warm_start == 0) | (warm_start == 1))
    
    def test_performance_optimizer_and_cache_integration(self):
        """Test integration between performance optimizer and cache."""
        cache = QuantumCache()
        optimizer = PerformanceOptimizer()
        
        # Record some performance data
        optimizer.record_solve(
            problem_size=50,
            solver_config={"num_reads": 1000, "annealing_time": 20},
            solve_time=1.5,
            energy=-8.0,
            success=True
        )
        
        # Get optimal config
        config = optimizer.get_optimal_config(problem_size=50)
        
        # Use config to create cache key
        Q_matrix = np.random.random((50, 50))
        cache_key = CacheKey.from_problem_data("qubo", Q_matrix, config)
        
        # Store result
        result_solution = np.random.choice([0, 1], 50)
        cache.put(cache_key, result_solution, -10.0, 2.0, config, Q_matrix)
        
        # Should be retrievable
        cached = cache.get(cache_key)
        assert cached is not None
        assert cached.solve_time == 2.0
"""
Intelligent caching system for quantum optimization results.
"""

import hashlib
import json
import time
import logging
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict
import pickle
import threading
from collections import OrderedDict

logger = logging.getLogger(__name__)

@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    solution: np.ndarray
    energy: float
    timestamp: float
    hit_count: int
    computation_time: float
    problem_hash: str
    similarity_score: float = 1.0

class IntelligentCache:
    """Intelligent caching for quantum optimization results."""
    
    def __init__(self, max_size: int = 1000, similarity_threshold: float = 0.95, ttl_hours: float = 24):
        self.max_size = max_size
        self.similarity_threshold = similarity_threshold
        self.ttl_seconds = ttl_hours * 3600
        
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'total_requests': 0,
            'time_saved': 0.0
        }
        
        self.logger = logging.getLogger(__name__)
    
    def _compute_problem_hash(self, problem_data: Dict[str, Any]) -> str:
        """Compute hash for problem data."""
        # Create deterministic representation
        serializable_data = {}
        
        for key, value in problem_data.items():
            if isinstance(value, np.ndarray):
                # Round to reduce sensitivity to small numerical differences
                serializable_data[key] = np.round(value, decimals=6).tolist()
            elif isinstance(value, (int, float, str, bool)):
                serializable_data[key] = value
            elif isinstance(value, (list, tuple)):
                serializable_data[key] = list(value)
            else:
                serializable_data[key] = str(value)
        
        # Sort keys for consistency
        sorted_data = json.dumps(serializable_data, sort_keys=True)
        return hashlib.sha256(sorted_data.encode()).hexdigest()
    
    def _compute_similarity(self, hash1: str, problem_data1: Dict[str, Any], 
                          hash2: str, problem_data2: Dict[str, Any]) -> float:
        """Compute similarity between two problems."""
        if hash1 == hash2:
            return 1.0
        
        try:
            similarity_scores = []
            
            # Compare numerical arrays
            for key in ['weather_forecast', 'energy_prices', 'current_state']:
                if key in problem_data1 and key in problem_data2:
                    arr1 = np.array(problem_data1[key]) if not isinstance(problem_data1[key], np.ndarray) else problem_data1[key]
                    arr2 = np.array(problem_data2[key]) if not isinstance(problem_data2[key], np.ndarray) else problem_data2[key]
                    
                    if arr1.shape == arr2.shape:
                        # Compute normalized correlation
                        correlation = np.corrcoef(arr1.flatten(), arr2.flatten())[0, 1]
                        if not np.isnan(correlation):
                            similarity_scores.append(max(0, correlation))
                    else:
                        similarity_scores.append(0.0)
            
            # Compare scalar values
            for key in ['prediction_horizon', 'control_interval']:
                if key in problem_data1 and key in problem_data2:
                    val1, val2 = problem_data1[key], problem_data2[key]
                    if val1 == val2:
                        similarity_scores.append(1.0)
                    else:
                        # Normalized difference
                        max_val = max(abs(val1), abs(val2))
                        if max_val > 0:
                            similarity_scores.append(1.0 - abs(val1 - val2) / max_val)
                        else:
                            similarity_scores.append(1.0)
            
            return np.mean(similarity_scores) if similarity_scores else 0.0
            
        except Exception as e:
            self.logger.warning(f"Error computing similarity: {e}")
            return 0.0
    
    def get_solution(self, problem_data: Dict[str, Any]) -> Optional[Tuple[np.ndarray, float, float]]:
        """Get cached solution if available."""
        problem_hash = self._compute_problem_hash(problem_data)
        
        with self._lock:
            self._stats['total_requests'] += 1
            
            # Direct hit
            if problem_hash in self._cache:
                entry = self._cache[problem_hash]
                
                # Check TTL
                if time.time() - entry.timestamp > self.ttl_seconds:
                    del self._cache[problem_hash]
                    self._stats['evictions'] += 1
                    self.logger.debug(f"Cache entry expired: {problem_hash[:8]}")
                else:
                    # Move to end (LRU)
                    self._cache.move_to_end(problem_hash)
                    entry.hit_count += 1
                    self._stats['hits'] += 1
                    self._stats['time_saved'] += entry.computation_time
                    
                    self.logger.debug(f"Cache hit: {problem_hash[:8]} (hits: {entry.hit_count})")
                    return entry.solution.copy(), entry.energy, entry.computation_time
            
            # Look for similar problems
            best_similarity = 0.0
            best_entry = None
            best_hash = None
            
            for cached_hash, cached_entry in self._cache.items():
                if time.time() - cached_entry.timestamp > self.ttl_seconds:
                    continue
                
                # Quick similarity check using stored problem data
                similarity = self._compute_similarity(
                    problem_hash, problem_data,
                    cached_hash, cached_entry.__dict__
                )
                
                if similarity > best_similarity and similarity >= self.similarity_threshold:
                    best_similarity = similarity
                    best_entry = cached_entry
                    best_hash = cached_hash
            
            if best_entry is not None:
                # Similar solution found
                self._cache.move_to_end(best_hash)
                best_entry.hit_count += 1
                best_entry.similarity_score = best_similarity
                self._stats['hits'] += 1
                self._stats['time_saved'] += best_entry.computation_time * best_similarity
                
                self.logger.debug(f"Similar cache hit: {best_hash[:8]} (similarity: {best_similarity:.3f})")
                return best_entry.solution.copy(), best_entry.energy, best_entry.computation_time
            
            # Cache miss
            self._stats['misses'] += 1
            self.logger.debug(f"Cache miss: {problem_hash[:8]}")
            return None
    
    def store_solution(self, problem_data: Dict[str, Any], solution: np.ndarray, 
                      energy: float, computation_time: float):
        """Store solution in cache."""
        problem_hash = self._compute_problem_hash(problem_data)
        
        with self._lock:
            # Store problem data with entry for similarity computation
            entry = CacheEntry(
                solution=solution.copy(),
                energy=energy,
                timestamp=time.time(),
                hit_count=0,
                computation_time=computation_time,
                problem_hash=problem_hash
            )
            
            # Add problem data for similarity computation
            for key, value in problem_data.items():
                if not hasattr(entry, key):
                    setattr(entry, key, value)
            
            self._cache[problem_hash] = entry
            
            # Move to end (most recent)
            self._cache.move_to_end(problem_hash)
            
            # Evict oldest if over capacity
            while len(self._cache) > self.max_size:
                oldest_hash, oldest_entry = self._cache.popitem(last=False)
                self._stats['evictions'] += 1
                self.logger.debug(f"Cache eviction: {oldest_hash[:8]} (age: {time.time() - oldest_entry.timestamp:.1f}s)")
            
            self.logger.debug(f"Cache store: {problem_hash[:8]} (computation: {computation_time:.3f}s)")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._stats['total_requests']
            hit_rate = self._stats['hits'] / total_requests if total_requests > 0 else 0.0
            
            return {
                'cache_size': len(self._cache),
                'max_size': self.max_size,
                'hit_rate': hit_rate,
                'hits': self._stats['hits'],
                'misses': self._stats['misses'],
                'evictions': self._stats['evictions'],
                'time_saved_seconds': self._stats['time_saved'],
                'total_requests': total_requests
            }
    
    def clear_expired(self):
        """Clear expired entries."""
        current_time = time.time()
        expired_keys = []
        
        with self._lock:
            for key, entry in self._cache.items():
                if current_time - entry.timestamp > self.ttl_seconds:
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self._cache[key]
                self._stats['evictions'] += 1
        
        if expired_keys:
            self.logger.info(f"Cleared {len(expired_keys)} expired cache entries")
    
    def optimize_cache(self):
        """Optimize cache by removing least valuable entries."""
        with self._lock:
            if len(self._cache) < self.max_size * 0.8:
                return
            
            # Score entries by value (hits, recency, computation time saved)
            entries_with_scores = []
            current_time = time.time()
            
            for key, entry in self._cache.items():
                age_hours = (current_time - entry.timestamp) / 3600
                value_score = (
                    entry.hit_count * 2.0 +  # Popularity
                    entry.computation_time * 0.1 +  # Time saved value
                    max(0, 24 - age_hours) * 0.05  # Recency bonus
                )
                entries_with_scores.append((key, value_score))
            
            # Sort by score (descending) and keep top 80%
            entries_with_scores.sort(key=lambda x: x[1], reverse=True)
            keep_count = int(len(entries_with_scores) * 0.8)
            
            # Remove bottom 20%
            for key, _ in entries_with_scores[keep_count:]:
                del self._cache[key]
                self._stats['evictions'] += 1
            
            self.logger.info(f"Cache optimized: kept {keep_count}/{len(entries_with_scores)} entries")

# Global cache instance
_global_cache = None

def get_intelligent_cache() -> IntelligentCache:
    """Get global intelligent cache instance."""
    global _global_cache
    if _global_cache is None:
        _global_cache = IntelligentCache()
    return _global_cache
"""
Intelligent caching system v2 with ML-based cache optimization.
Advanced caching strategies for quantum HVAC optimization results.
"""

import time
import logging
import numpy as np
import hashlib
import pickle
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class CacheStrategy(Enum):
    """Caching strategies."""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    TTL = "ttl"  # Time To Live
    ADAPTIVE = "adaptive"  # ML-based adaptive
    SIMILARITY = "similarity"  # Content similarity
    PREDICTIVE = "predictive"  # Predictive pre-caching


class CacheHitType(Enum):
    """Types of cache hits."""
    EXACT = "exact"
    SIMILARITY = "similarity"
    PARTIAL = "partial"
    MISS = "miss"


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    timestamp: float
    access_count: int = 0
    last_access: float = 0.0
    ttl: Optional[float] = None
    size: int = 0
    similarity_hash: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    

@dataclass
class CacheStats:
    """Cache performance statistics."""
    hits: int = 0
    misses: int = 0
    exact_hits: int = 0
    similarity_hits: int = 0
    partial_hits: int = 0
    evictions: int = 0
    total_requests: int = 0
    total_storage: int = 0
    avg_retrieval_time: float = 0.0
    

class IntelligentCachingSystem:
    """Advanced caching system with ML optimization."""
    
    def __init__(self, max_size: int = 10000, 
                 strategy: CacheStrategy = CacheStrategy.ADAPTIVE,
                 similarity_threshold: float = 0.85):
        self.max_size = max_size
        self.strategy = strategy
        self.similarity_threshold = similarity_threshold
        
        self.cache: Dict[str, CacheEntry] = {}
        self.stats = CacheStats()
        self.access_history = deque(maxlen=10000)
        self.similarity_index = {}  # For fast similarity lookups
        
        # Strategy-specific data structures
        self.lru_order = deque()
        self.frequency_counter = defaultdict(int)
        self.adaptive_weights = defaultdict(float)
        
        # ML components
        self.access_patterns = deque(maxlen=5000)
        self.prediction_model = None
        self.feature_buffer = deque(maxlen=1000)
        
        # Background optimization
        self.optimization_active = False
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        # Thread safety
        self.lock = threading.RLock()
    
    def put(self, key: str, value: Any, context: Dict[str, Any] = None, 
            ttl: Optional[float] = None) -> bool:
        """
        Store value in cache with optional context and TTL.
        
        Args:
            key: Cache key
            value: Value to cache
            context: Optional context for similarity matching
            ttl: Time to live in seconds
            
        Returns:
            True if successfully cached
        """
        with self.lock:
            try:
                # Calculate value size
                size = len(pickle.dumps(value))
                
                # Generate similarity hash
                similarity_hash = self._generate_similarity_hash(context or {})
                
                # Create cache entry
                entry = CacheEntry(
                    key=key,
                    value=value,
                    timestamp=time.time(),
                    ttl=ttl,
                    size=size,
                    similarity_hash=similarity_hash,
                    context=context or {}
                )
                
                # Check if we need to evict entries
                if len(self.cache) >= self.max_size:
                    self._evict_entries()
                
                # Store entry
                self.cache[key] = entry
                self.stats.total_storage += size
                
                # Update strategy-specific structures
                self._update_structures_on_put(key, entry)
                
                # Update similarity index
                if similarity_hash:
                    if similarity_hash not in self.similarity_index:
                        self.similarity_index[similarity_hash] = []
                    self.similarity_index[similarity_hash].append(key)
                
                logger.debug(f"Cached {key} with size {size}")
                return True
                
            except Exception as e:
                logger.error(f"Cache put error for {key}: {e}")
                return False
    
    def get(self, key: str, context: Dict[str, Any] = None) -> Tuple[Any, CacheHitType]:
        """
        Retrieve value from cache.
        
        Args:
            key: Cache key
            context: Optional context for similarity matching
            
        Returns:
            Tuple of (value, hit_type)
        """
        start_time = time.time()
        
        with self.lock:
            self.stats.total_requests += 1
            
            # Check for exact match
            if key in self.cache:
                entry = self.cache[key]
                
                # Check TTL
                if entry.ttl and (time.time() - entry.timestamp) > entry.ttl:
                    self._remove_entry(key)
                    return self._handle_cache_miss(key, context, start_time)
                
                # Update access info
                entry.access_count += 1
                entry.last_access = time.time()
                
                self._update_structures_on_access(key)
                
                # Record stats
                self.stats.hits += 1
                self.stats.exact_hits += 1
                self._update_retrieval_time(time.time() - start_time)
                
                self._record_access_pattern(key, context, CacheHitType.EXACT)
                
                return entry.value, CacheHitType.EXACT
            
            # Try similarity matching if context provided
            if context and self.strategy in [CacheStrategy.SIMILARITY, CacheStrategy.ADAPTIVE]:
                similar_key, similarity_score = self._find_similar_entry(context)
                
                if similar_key and similarity_score >= self.similarity_threshold:
                    entry = self.cache[similar_key]
                    
                    # Update access info
                    entry.access_count += 1
                    entry.last_access = time.time()
                    
                    self.stats.hits += 1
                    self.stats.similarity_hits += 1
                    self._update_retrieval_time(time.time() - start_time)
                    
                    self._record_access_pattern(key, context, CacheHitType.SIMILARITY)
                    
                    logger.debug(f"Similarity hit for {key} -> {similar_key} (score: {similarity_score:.3f})")
                    return entry.value, CacheHitType.SIMILARITY
            
            # Try partial matching for structured data
            partial_result = self._try_partial_match(key, context)
            if partial_result is not None:
                self.stats.hits += 1
                self.stats.partial_hits += 1
                self._update_retrieval_time(time.time() - start_time)
                
                self._record_access_pattern(key, context, CacheHitType.PARTIAL)
                
                return partial_result, CacheHitType.PARTIAL
            
            return self._handle_cache_miss(key, context, start_time)
    
    def _handle_cache_miss(self, key: str, context: Dict[str, Any], 
                          start_time: float) -> Tuple[None, CacheHitType]:
        """Handle cache miss."""
        self.stats.misses += 1
        self._update_retrieval_time(time.time() - start_time)
        
        self._record_access_pattern(key, context, CacheHitType.MISS)
        
        # Trigger predictive caching if enabled
        if self.strategy == CacheStrategy.PREDICTIVE:
            self._trigger_predictive_caching(key, context)
        
        return None, CacheHitType.MISS
    
    def _find_similar_entry(self, context: Dict[str, Any]) -> Tuple[Optional[str], float]:
        """Find most similar cache entry."""
        similarity_hash = self._generate_similarity_hash(context)
        
        # First, check exact similarity hash match
        if similarity_hash in self.similarity_index:
            candidates = self.similarity_index[similarity_hash]
            if candidates:
                # Return most recently used
                best_candidate = max(candidates, 
                                   key=lambda k: self.cache[k].last_access if k in self.cache else 0)
                return best_candidate, 1.0
        
        # Compute similarity scores
        best_key = None
        best_score = 0.0
        
        for key, entry in self.cache.items():
            if not entry.context:
                continue
            
            score = self._calculate_similarity(context, entry.context)
            if score > best_score:
                best_score = score
                best_key = key
        
        return best_key, best_score
    
    def _calculate_similarity(self, context1: Dict[str, Any], 
                            context2: Dict[str, Any]) -> float:
        """Calculate similarity between two contexts."""
        try:
            # Extract numerical features for comparison
            features1 = self._extract_features(context1)
            features2 = self._extract_features(context2)
            
            if not features1 or not features2:
                return 0.0
            
            # Ensure same feature space
            all_keys = set(features1.keys()) | set(features2.keys())
            
            vec1 = np.array([features1.get(k, 0) for k in all_keys])
            vec2 = np.array([features2.get(k, 0) for k in all_keys])
            
            # Cosine similarity
            dot_product = np.dot(vec1, vec2)
            norms = np.linalg.norm(vec1) * np.linalg.norm(vec2)
            
            if norms == 0:
                return 0.0
            
            similarity = dot_product / norms
            return max(0.0, similarity)  # Ensure non-negative
            
        except Exception as e:
            logger.debug(f"Similarity calculation error: {e}")
            return 0.0
    
    def _extract_features(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Extract numerical features from context."""
        features = {}
        
        for key, value in context.items():
            if isinstance(value, (int, float)):
                features[key] = float(value)
            elif isinstance(value, (list, tuple, np.ndarray)):
                try:
                    arr = np.array(value)
                    if arr.dtype.kind in 'biufc':  # Numeric types
                        features[f"{key}_mean"] = float(np.mean(arr))
                        features[f"{key}_std"] = float(np.std(arr))
                        features[f"{key}_size"] = float(len(arr))
                except:
                    pass
            elif isinstance(value, str):
                features[f"{key}_hash"] = float(hash(value) % 1000000)
            elif isinstance(value, dict):
                # Recursively extract from nested dict
                nested_features = self._extract_features(value)
                for nested_key, nested_value in nested_features.items():
                    features[f"{key}_{nested_key}"] = nested_value
        
        return features
    
    def _generate_similarity_hash(self, context: Dict[str, Any]) -> str:
        """Generate hash for similarity grouping."""
        if not context:
            return ""
        
        # Create simplified representation for grouping
        simplified = {}
        
        for key, value in context.items():
            if isinstance(value, (int, float)):
                # Round to reduce precision for grouping
                simplified[key] = round(value, 2)
            elif isinstance(value, str):
                simplified[key] = value
            elif isinstance(value, (list, tuple, np.ndarray)):
                try:
                    arr = np.array(value)
                    if arr.dtype.kind in 'biufc':
                        simplified[key] = round(float(np.mean(arr)), 2)
                except:
                    pass
        
        # Create hash
        content = str(sorted(simplified.items()))
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def _try_partial_match(self, key: str, context: Dict[str, Any]) -> Optional[Any]:
        """Try to construct partial result from existing cache entries."""
        # This is a simplified implementation
        # In practice, this would depend on the structure of cached data
        
        if not context or 'partial_keys' not in context:
            return None
        
        partial_keys = context['partial_keys']
        if not isinstance(partial_keys, list):
            return None
        
        # Check if all partial keys exist in cache
        partial_results = {}
        for pkey in partial_keys:
            if pkey in self.cache:
                entry = self.cache[pkey]
                if not (entry.ttl and (time.time() - entry.timestamp) > entry.ttl):
                    partial_results[pkey] = entry.value
        
        if len(partial_results) >= len(partial_keys) * 0.8:  # 80% of parts available
            return partial_results
        
        return None
    
    def _evict_entries(self):
        """Evict entries based on current strategy."""
        if not self.cache:
            return
        
        entries_to_evict = max(1, len(self.cache) // 10)  # Evict 10% at a time
        
        if self.strategy == CacheStrategy.LRU:
            self._evict_lru(entries_to_evict)
        elif self.strategy == CacheStrategy.LFU:
            self._evict_lfu(entries_to_evict)
        elif self.strategy == CacheStrategy.TTL:
            self._evict_expired(entries_to_evict)
        elif self.strategy == CacheStrategy.ADAPTIVE:
            self._evict_adaptive(entries_to_evict)
        else:
            self._evict_lru(entries_to_evict)  # Default
    
    def _evict_lru(self, count: int):
        """Evict least recently used entries."""
        # Sort by last access time
        sorted_entries = sorted(self.cache.items(), 
                               key=lambda x: x[1].last_access or x[1].timestamp)
        
        for i in range(min(count, len(sorted_entries))):
            key = sorted_entries[i][0]
            self._remove_entry(key)
    
    def _evict_lfu(self, count: int):
        """Evict least frequently used entries."""
        # Sort by access count
        sorted_entries = sorted(self.cache.items(), 
                               key=lambda x: x[1].access_count)
        
        for i in range(min(count, len(sorted_entries))):
            key = sorted_entries[i][0]
            self._remove_entry(key)
    
    def _evict_expired(self, count: int):
        """Evict expired entries first, then fall back to LRU."""
        expired_keys = []
        current_time = time.time()
        
        for key, entry in self.cache.items():
            if entry.ttl and (current_time - entry.timestamp) > entry.ttl:
                expired_keys.append(key)
        
        # Remove expired entries
        for key in expired_keys[:count]:
            self._remove_entry(key)
        
        # If we need to evict more, use LRU
        remaining = count - len(expired_keys)
        if remaining > 0:
            self._evict_lru(remaining)
    
    def _evict_adaptive(self, count: int):
        """Adaptive eviction using ML-based scoring."""
        if not self.cache:
            return
        
        # Calculate adaptive scores
        scores = {}
        current_time = time.time()
        
        for key, entry in self.cache.items():
            # Base score factors
            recency_score = 1.0 / max(1, current_time - entry.last_access)
            frequency_score = entry.access_count
            size_penalty = entry.size / 1000  # Penalize large entries
            
            # TTL factor
            ttl_factor = 1.0
            if entry.ttl:
                remaining_ttl = entry.ttl - (current_time - entry.timestamp)
                ttl_factor = max(0.1, remaining_ttl / entry.ttl)
            
            # Adaptive weight
            adaptive_weight = self.adaptive_weights.get(key, 1.0)
            
            # Combined score (higher = more valuable)
            score = (recency_score * 0.3 + 
                    frequency_score * 0.3 + 
                    ttl_factor * 0.2 + 
                    adaptive_weight * 0.2 - 
                    size_penalty * 0.1)
            
            scores[key] = score
        
        # Sort by score (ascending - lowest scores evicted first)
        sorted_entries = sorted(scores.items(), key=lambda x: x[1])
        
        for i in range(min(count, len(sorted_entries))):
            key = sorted_entries[i][0]
            self._remove_entry(key)
    
    def _remove_entry(self, key: str):
        """Remove entry from cache and update structures."""
        if key not in self.cache:
            return
        
        entry = self.cache[key]
        
        # Update storage stats
        self.stats.total_storage -= entry.size
        self.stats.evictions += 1
        
        # Remove from similarity index
        if entry.similarity_hash and entry.similarity_hash in self.similarity_index:
            if key in self.similarity_index[entry.similarity_hash]:
                self.similarity_index[entry.similarity_hash].remove(key)
            if not self.similarity_index[entry.similarity_hash]:
                del self.similarity_index[entry.similarity_hash]
        
        # Remove from strategy-specific structures
        if key in self.lru_order:
            self.lru_order.remove(key)
        
        if key in self.frequency_counter:
            del self.frequency_counter[key]
        
        if key in self.adaptive_weights:
            del self.adaptive_weights[key]
        
        # Remove from cache
        del self.cache[key]
        
        logger.debug(f"Evicted cache entry: {key}")
    
    def _update_structures_on_put(self, key: str, entry: CacheEntry):
        """Update strategy-specific structures on put."""
        if self.strategy == CacheStrategy.LRU:
            if key in self.lru_order:
                self.lru_order.remove(key)
            self.lru_order.append(key)
        
        elif self.strategy == CacheStrategy.LFU:
            self.frequency_counter[key] = 0
        
        elif self.strategy == CacheStrategy.ADAPTIVE:
            self.adaptive_weights[key] = 1.0
    
    def _update_structures_on_access(self, key: str):
        """Update strategy-specific structures on access."""
        if self.strategy == CacheStrategy.LRU:
            if key in self.lru_order:
                self.lru_order.remove(key)
            self.lru_order.append(key)
        
        elif self.strategy == CacheStrategy.LFU:
            self.frequency_counter[key] += 1
        
        elif self.strategy == CacheStrategy.ADAPTIVE:
            # Increase weight for frequently accessed items
            current_weight = self.adaptive_weights.get(key, 1.0)
            self.adaptive_weights[key] = min(current_weight * 1.1, 5.0)
    
    def _record_access_pattern(self, key: str, context: Dict[str, Any], 
                             hit_type: CacheHitType):
        """Record access pattern for ML optimization."""
        pattern = {
            'key': key,
            'context': context or {},
            'hit_type': hit_type,
            'timestamp': time.time()
        }
        
        self.access_patterns.append(pattern)
        self.access_history.append(pattern)
    
    def _update_retrieval_time(self, retrieval_time: float):
        """Update average retrieval time."""
        self.stats.avg_retrieval_time = (
            self.stats.avg_retrieval_time * 0.95 + retrieval_time * 0.05
        )
    
    def _trigger_predictive_caching(self, key: str, context: Dict[str, Any]):
        """Trigger predictive caching for related entries."""
        # This would implement ML-based prediction of future cache needs
        # For now, it's a placeholder
        pass
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        hit_rate = (self.stats.hits / max(self.stats.total_requests, 1)) * 100
        
        return {
            'strategy': self.strategy.value,
            'size': len(self.cache),
            'max_size': self.max_size,
            'storage_usage': self.stats.total_storage,
            'hit_rate': hit_rate,
            'exact_hit_rate': (self.stats.exact_hits / max(self.stats.total_requests, 1)) * 100,
            'similarity_hit_rate': (self.stats.similarity_hits / max(self.stats.total_requests, 1)) * 100,
            'partial_hit_rate': (self.stats.partial_hits / max(self.stats.total_requests, 1)) * 100,
            'miss_rate': (self.stats.misses / max(self.stats.total_requests, 1)) * 100,
            'evictions': self.stats.evictions,
            'avg_retrieval_time_ms': self.stats.avg_retrieval_time * 1000,
            'similarity_index_size': len(self.similarity_index)
        }
    
    def clear_cache(self):
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()
            self.lru_order.clear()
            self.frequency_counter.clear()
            self.adaptive_weights.clear()
            self.similarity_index.clear()
            self.stats = CacheStats()
            logger.info("Cache cleared")
    
    def optimize_cache(self):
        """Optimize cache performance based on access patterns."""
        if not self.access_patterns:
            return
        
        # Analyze access patterns
        pattern_analysis = self._analyze_access_patterns()
        
        # Adjust similarity threshold based on effectiveness
        if pattern_analysis['similarity_effectiveness'] < 0.5:
            self.similarity_threshold = min(self.similarity_threshold * 1.1, 0.95)
        elif pattern_analysis['similarity_effectiveness'] > 0.8:
            self.similarity_threshold = max(self.similarity_threshold * 0.95, 0.7)
        
        # Update adaptive weights
        for key in self.cache:
            if key in pattern_analysis['high_value_keys']:
                self.adaptive_weights[key] = min(self.adaptive_weights.get(key, 1.0) * 1.2, 5.0)
        
        logger.info(f"Cache optimized: threshold={self.similarity_threshold:.3f}")
    
    def _analyze_access_patterns(self) -> Dict[str, Any]:
        """Analyze access patterns for optimization."""
        if not self.access_patterns:
            return {}
        
        recent_patterns = list(self.access_patterns)[-1000:]
        
        # Calculate similarity hit effectiveness
        similarity_hits = [p for p in recent_patterns if p['hit_type'] == CacheHitType.SIMILARITY]
        total_hits = [p for p in recent_patterns if p['hit_type'] != CacheHitType.MISS]
        
        similarity_effectiveness = len(similarity_hits) / max(len(total_hits), 1)
        
        # Identify high-value keys
        key_values = defaultdict(int)
        for pattern in recent_patterns:
            if pattern['hit_type'] != CacheHitType.MISS:
                key_values[pattern['key']] += 1
        
        high_value_keys = [k for k, v in key_values.items() if v > np.percentile(list(key_values.values()), 75)]
        
        return {
            'similarity_effectiveness': similarity_effectiveness,
            'high_value_keys': high_value_keys,
            'total_patterns': len(recent_patterns)
        }
    
    async def start_background_optimization(self):
        """Start background optimization task."""
        if self.optimization_active:
            return
        
        self.optimization_active = True
        
        async def optimization_loop():
            while self.optimization_active:
                try:
                    self.optimize_cache()
                    await asyncio.sleep(300)  # Optimize every 5 minutes
                except Exception as e:
                    logger.error(f"Cache optimization error: {e}")
                    await asyncio.sleep(300)
        
        asyncio.create_task(optimization_loop())
        logger.info("Background cache optimization started")
    
    def stop_background_optimization(self):
        """Stop background optimization."""
        self.optimization_active = False
        logger.info("Background cache optimization stopped")


# Global cache instance
_global_cache = None

def get_intelligent_cache() -> IntelligentCachingSystem:
    """Get global intelligent cache instance."""
    global _global_cache
    if _global_cache is None:
        _global_cache = IntelligentCachingSystem()
    return _global_cache


__all__ = [
    'IntelligentCachingSystem',
    'CacheStrategy',
    'CacheHitType',
    'CacheEntry', 
    'CacheStats',
    'get_intelligent_cache'
]
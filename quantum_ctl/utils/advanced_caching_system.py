"""
Advanced caching system for quantum HVAC operations.
Provides intelligent caching, cache warming, predictive prefetching, and distributed caching.
"""

import asyncio
import time
import threading
import hashlib
import pickle
import json
import logging
from typing import Any, Dict, List, Optional, Callable, Union, Tuple, Generic, TypeVar
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict, OrderedDict
import statistics
from datetime import datetime, timedelta
import weakref

logger = logging.getLogger(__name__)

T = TypeVar('T')


class CacheStrategy(Enum):
    """Cache replacement strategies."""
    LRU = "lru"           # Least Recently Used
    LFU = "lfu"           # Least Frequently Used
    FIFO = "fifo"         # First In, First Out
    TTL = "ttl"           # Time To Live
    ADAPTIVE = "adaptive"  # Adaptive based on access patterns


class CachePriority(Enum):
    """Cache entry priorities."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class CacheEntry:
    """Represents a cache entry with metadata."""
    key: str
    value: Any
    timestamp: float = field(default_factory=time.time)
    access_count: int = 0
    last_access: float = field(default_factory=time.time)
    ttl: Optional[float] = None
    priority: CachePriority = CachePriority.NORMAL
    size_bytes: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        if self.ttl is None:
            return False
        return time.time() - self.timestamp > self.ttl
    
    def touch(self):
        """Update access information."""
        self.access_count += 1
        self.last_access = time.time()


class IntelligentCache(Generic[T]):
    """Advanced cache with intelligent replacement and prediction."""
    
    def __init__(
        self,
        max_size: int = 1000,
        max_memory_mb: int = 256,
        strategy: CacheStrategy = CacheStrategy.ADAPTIVE,
        default_ttl: Optional[float] = None,
        enable_statistics: bool = True
    ):
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.strategy = strategy
        self.default_ttl = default_ttl
        self.enable_statistics = enable_statistics
        
        # Cache storage
        self._cache: Dict[str, CacheEntry] = {}
        self._access_order: OrderedDict = OrderedDict()  # For LRU
        self._frequency_counter: Dict[str, int] = defaultdict(int)  # For LFU
        self._lock = threading.RLock()
        
        # Statistics
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "size": 0,
            "memory_usage": 0
        }
        
        # Access pattern analysis
        self._access_patterns = deque(maxlen=10000)
        self._key_correlations: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        
        # Background tasks
        self._cleanup_task = None
        self._prediction_task = None
        self._running = False
    
    async def start_background_tasks(self):
        """Start background cache maintenance tasks."""
        if self._running:
            return
        
        self._running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        self._prediction_task = asyncio.create_task(self._prediction_loop())
        logger.info("Cache background tasks started")
    
    async def stop_background_tasks(self):
        """Stop background tasks."""
        self._running = False
        
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        if self._prediction_task:
            self._prediction_task.cancel()
            try:
                await self._prediction_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Cache background tasks stopped")
    
    async def get(self, key: str, default: T = None) -> Optional[T]:
        """Get value from cache."""
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                
                # Check expiration
                if entry.is_expired():
                    self._remove_entry(key)
                    self._stats["misses"] += 1
                    return default
                
                # Update access information
                entry.touch()
                self._update_access_order(key)
                
                # Record access pattern
                self._record_access(key)
                
                self._stats["hits"] += 1
                return entry.value
            else:
                self._stats["misses"] += 1
                return default
    
    async def put(
        self,
        key: str,
        value: T,
        ttl: Optional[float] = None,
        priority: CachePriority = CachePriority.NORMAL
    ):
        """Put value into cache."""
        with self._lock:
            # Calculate size
            size_bytes = self._calculate_size(value)
            
            # Create cache entry
            entry = CacheEntry(
                key=key,
                value=value,
                ttl=ttl or self.default_ttl,
                priority=priority,
                size_bytes=size_bytes
            )
            
            # Check if we need to make space
            await self._ensure_space(size_bytes)
            
            # Add/update entry
            if key in self._cache:
                old_size = self._cache[key].size_bytes
                self._stats["memory_usage"] -= old_size
            else:
                self._stats["size"] += 1
            
            self._cache[key] = entry
            self._stats["memory_usage"] += size_bytes
            self._update_access_order(key)
            
            # Record access pattern
            self._record_access(key)
    
    async def _ensure_space(self, required_bytes: int):
        """Ensure there's enough space for new entry."""
        # Check memory limit
        while (self._stats["memory_usage"] + required_bytes > self.max_memory_bytes or
               len(self._cache) >= self.max_size):
            
            victim_key = await self._select_eviction_victim()
            if victim_key:
                self._remove_entry(victim_key)
                self._stats["evictions"] += 1
            else:
                # Can't evict anything, cache is full of high priority items
                raise Exception("Cache full with high priority items")
    
    async def _select_eviction_victim(self) -> Optional[str]:
        """Select entry to evict based on strategy."""
        if not self._cache:
            return None
        
        # Never evict critical priority items
        candidates = {k: v for k, v in self._cache.items() 
                     if v.priority != CachePriority.CRITICAL}
        
        if not candidates:
            return None
        
        if self.strategy == CacheStrategy.LRU:
            return self._select_lru_victim(candidates)
        elif self.strategy == CacheStrategy.LFU:
            return self._select_lfu_victim(candidates)
        elif self.strategy == CacheStrategy.TTL:
            return self._select_ttl_victim(candidates)
        elif self.strategy == CacheStrategy.ADAPTIVE:
            return await self._select_adaptive_victim(candidates)
        else:  # FIFO
            return min(candidates.keys(), key=lambda k: candidates[k].timestamp)
    
    def _select_lru_victim(self, candidates: Dict[str, CacheEntry]) -> str:
        """Select LRU victim."""
        return min(candidates.keys(), key=lambda k: candidates[k].last_access)
    
    def _select_lfu_victim(self, candidates: Dict[str, CacheEntry]) -> str:
        """Select LFU victim."""
        return min(candidates.keys(), key=lambda k: candidates[k].access_count)
    
    def _select_ttl_victim(self, candidates: Dict[str, CacheEntry]) -> str:
        """Select expired or oldest TTL victim."""
        # First, try to find expired entries
        expired = [k for k, v in candidates.items() if v.is_expired()]
        if expired:
            return expired[0]
        
        # Otherwise, select entry closest to expiration
        return min(candidates.keys(), 
                  key=lambda k: (candidates[k].timestamp + (candidates[k].ttl or float('inf'))))
    
    async def _select_adaptive_victim(self, candidates: Dict[str, CacheEntry]) -> str:
        """Select victim using adaptive strategy."""
        scores = {}
        
        for key, entry in candidates.items():
            # Calculate composite score based on multiple factors
            age_score = time.time() - entry.last_access
            frequency_score = 1.0 / max(1, entry.access_count)
            priority_score = 1.0 / entry.priority.value
            size_score = entry.size_bytes / self.max_memory_bytes
            
            # Predict future access probability
            prediction_score = await self._predict_access_probability(key)
            
            # Weighted composite score (higher = more likely to evict)
            scores[key] = (
                age_score * 0.3 +
                frequency_score * 0.2 +
                priority_score * 0.2 +
                size_score * 0.1 +
                (1.0 - prediction_score) * 0.2
            )
        
        return max(scores.keys(), key=lambda k: scores[k])
    
    async def _predict_access_probability(self, key: str) -> float:
        """Predict probability of future access for a key."""
        if not self._access_patterns:
            return 0.5  # Default probability
        
        # Analyze recent access patterns
        recent_accesses = list(self._access_patterns)[-100:]  # Last 100 accesses
        
        # Count recent accesses for this key
        recent_count = sum(1 for access in recent_accesses if access == key)
        
        # Calculate base probability
        base_prob = recent_count / len(recent_accesses)
        
        # Consider key correlations
        correlation_boost = 0.0
        for recent_key in set(recent_accesses[-10:]):  # Last 10 unique keys
            if recent_key != key and recent_key in self._key_correlations:
                correlation_boost += self._key_correlations[recent_key].get(key, 0.0)
        
        # Combine probabilities
        final_prob = min(1.0, base_prob + (correlation_boost / 10))
        return final_prob
    
    def _remove_entry(self, key: str):
        """Remove entry from cache and update data structures."""
        if key in self._cache:
            entry = self._cache[key]
            self._stats["memory_usage"] -= entry.size_bytes
            self._stats["size"] -= 1
            
            del self._cache[key]
            self._access_order.pop(key, None)
            self._frequency_counter.pop(key, None)
    
    def _update_access_order(self, key: str):
        """Update access order for LRU tracking."""
        self._access_order.pop(key, None)
        self._access_order[key] = time.time()
    
    def _record_access(self, key: str):
        """Record access for pattern analysis."""
        if len(self._access_patterns) > 0:
            # Update key correlations
            recent_keys = list(self._access_patterns)[-5:]  # Last 5 accesses
            for recent_key in recent_keys:
                if recent_key != key:
                    self._key_correlations[recent_key][key] += 0.1
                    # Decay correlation over time
                    if self._key_correlations[recent_key][key] > 1.0:
                        for corr_key in self._key_correlations[recent_key]:
                            self._key_correlations[recent_key][corr_key] *= 0.95
        
        self._access_patterns.append(key)
    
    def _calculate_size(self, value: Any) -> int:
        """Calculate approximate size of value in bytes."""
        try:
            return len(pickle.dumps(value))
        except Exception:
            # Fallback estimation
            if isinstance(value, str):
                return len(value.encode('utf-8'))
            elif isinstance(value, (int, float)):
                return 8
            elif isinstance(value, (list, tuple)):
                return sum(self._calculate_size(item) for item in value)
            elif isinstance(value, dict):
                return sum(self._calculate_size(k) + self._calculate_size(v) 
                          for k, v in value.items())
            else:
                return 100  # Default estimate
    
    async def _cleanup_loop(self):
        """Background cleanup of expired entries."""
        while self._running:
            try:
                with self._lock:
                    expired_keys = [k for k, v in self._cache.items() if v.is_expired()]
                    for key in expired_keys:
                        self._remove_entry(key)
                
                await asyncio.sleep(60)  # Check every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cache cleanup: {e}")
                await asyncio.sleep(5)
    
    async def _prediction_loop(self):
        """Background predictive prefetching."""
        while self._running:
            try:
                await self._predictive_prefetch()
                await asyncio.sleep(30)  # Predict every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cache prediction: {e}")
                await asyncio.sleep(5)
    
    async def _predictive_prefetch(self):
        """Predictively prefetch likely-to-be-accessed items."""
        if not hasattr(self, 'prefetch_callback'):
            return
        
        # Analyze access patterns to predict next likely accesses
        predictions = await self._generate_predictions()
        
        for key, probability in predictions.items():
            if probability > 0.7 and key not in self._cache:  # High probability
                try:
                    # Call prefetch callback to generate value
                    value = await self.prefetch_callback(key)
                    if value is not None:
                        await self.put(key, value, priority=CachePriority.LOW)
                        logger.debug(f"Prefetched cache key: {key}")
                except Exception as e:
                    logger.warning(f"Failed to prefetch {key}: {e}")
    
    async def _generate_predictions(self) -> Dict[str, float]:
        """Generate predictions for likely future accesses."""
        predictions = {}
        
        if len(self._access_patterns) < 10:
            return predictions
        
        # Analyze sequential patterns
        pattern_sequences = []
        for i in range(len(self._access_patterns) - 2):
            seq = tuple(self._access_patterns[i:i+3])
            pattern_sequences.append(seq)
        
        # Find common sequences
        sequence_counts = defaultdict(int)
        for seq in pattern_sequences:
            sequence_counts[seq] += 1
        
        # Predict next access based on recent pattern
        recent_pattern = tuple(self._access_patterns[-2:])
        
        for seq, count in sequence_counts.items():
            if seq[:2] == recent_pattern:
                next_key = seq[2]
                probability = count / len(pattern_sequences)
                predictions[next_key] = probability
        
        return predictions
    
    def set_prefetch_callback(self, callback: Callable[[str], Any]):
        """Set callback function for predictive prefetching."""
        self.prefetch_callback = callback
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            hit_rate = 0.0
            total_accesses = self._stats["hits"] + self._stats["misses"]
            if total_accesses > 0:
                hit_rate = self._stats["hits"] / total_accesses
            
            return {
                **self._stats,
                "hit_rate": hit_rate,
                "memory_usage_mb": self._stats["memory_usage"] / (1024 * 1024),
                "memory_utilization": self._stats["memory_usage"] / self.max_memory_bytes,
                "size_utilization": len(self._cache) / self.max_size,
                "avg_entry_size": (self._stats["memory_usage"] / max(1, len(self._cache))),
                "strategy": self.strategy.value
            }
    
    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._access_order.clear()
            self._frequency_counter.clear()
            self._stats = {
                "hits": 0,
                "misses": 0,
                "evictions": 0,
                "size": 0,
                "memory_usage": 0
            }


class DistributedCache:
    """Distributed cache for multi-node quantum HVAC systems."""
    
    def __init__(self, node_id: str, cluster_nodes: List[str] = None):
        self.node_id = node_id
        self.cluster_nodes = cluster_nodes or []
        self.local_cache = IntelligentCache(max_size=5000, max_memory_mb=512)
        self.peer_connections: Dict[str, Any] = {}
        
        # Consistency strategy
        self.consistency_level = "eventual"  # eventual, strong, weak
        
        # Cache warming
        self.warming_enabled = True
        self.warming_prefetch_count = 100
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from distributed cache."""
        # Try local cache first
        value = await self.local_cache.get(key)
        if value is not None:
            return value
        
        # Try peer nodes
        if self.cluster_nodes:
            for node in self.cluster_nodes:
                try:
                    value = await self._get_from_peer(node, key)
                    if value is not None:
                        # Cache locally for future access
                        await self.local_cache.put(key, value, priority=CachePriority.NORMAL)
                        return value
                except Exception as e:
                    logger.warning(f"Failed to get {key} from peer {node}: {e}")
        
        return None
    
    async def put(self, key: str, value: Any, replicate: bool = True):
        """Put value into distributed cache."""
        # Store locally
        await self.local_cache.put(key, value)
        
        # Replicate to peers if requested
        if replicate and self.cluster_nodes:
            replication_tasks = []
            for node in self.cluster_nodes:
                task = asyncio.create_task(self._put_to_peer(node, key, value))
                replication_tasks.append(task)
            
            # Wait for replications (fire and forget for eventual consistency)
            if self.consistency_level == "strong":
                await asyncio.gather(*replication_tasks, return_exceptions=True)
    
    async def _get_from_peer(self, node: str, key: str) -> Optional[Any]:
        """Get value from peer node."""
        # This would use actual network communication in production
        # For now, it's a placeholder
        logger.debug(f"Getting {key} from peer {node}")
        return None
    
    async def _put_to_peer(self, node: str, key: str, value: Any):
        """Put value to peer node."""
        # This would use actual network communication in production
        logger.debug(f"Putting {key} to peer {node}")
    
    async def warm_cache(self, warm_keys: List[str]):
        """Warm cache with frequently accessed keys."""
        if not self.warming_enabled:
            return
        
        logger.info(f"Warming cache with {len(warm_keys)} keys")
        
        for key in warm_keys[:self.warming_prefetch_count]:
            try:
                # This would call the actual data source
                # For now, it's a placeholder
                value = await self._fetch_from_source(key)
                if value is not None:
                    await self.local_cache.put(key, value, priority=CachePriority.HIGH)
            except Exception as e:
                logger.warning(f"Failed to warm cache for {key}: {e}")
    
    async def _fetch_from_source(self, key: str) -> Optional[Any]:
        """Fetch value from original data source."""
        # Placeholder - would implement actual data fetching
        return None


class CacheManager:
    """Manages multiple cache instances and provides unified interface."""
    
    def __init__(self):
        self.caches: Dict[str, IntelligentCache] = {}
        self.default_cache = IntelligentCache(
            max_size=1000,
            max_memory_mb=256,
            strategy=CacheStrategy.ADAPTIVE
        )
    
    def get_cache(self, name: str) -> IntelligentCache:
        """Get or create named cache."""
        if name not in self.caches:
            self.caches[name] = IntelligentCache(
                max_size=500,
                max_memory_mb=128,
                strategy=CacheStrategy.ADAPTIVE
            )
        return self.caches[name]
    
    def create_cache(
        self,
        name: str,
        max_size: int = 500,
        max_memory_mb: int = 128,
        strategy: CacheStrategy = CacheStrategy.ADAPTIVE,
        **kwargs
    ) -> IntelligentCache:
        """Create a new named cache with specific configuration."""
        cache = IntelligentCache(
            max_size=max_size,
            max_memory_mb=max_memory_mb,
            strategy=strategy,
            **kwargs
        )
        self.caches[name] = cache
        return cache
    
    async def get_global_statistics(self) -> Dict[str, Any]:
        """Get statistics for all caches."""
        stats = {
            "cache_count": len(self.caches) + 1,  # +1 for default cache
            "total_memory_mb": 0,
            "total_entries": 0,
            "total_hits": 0,
            "total_misses": 0,
            "caches": {}
        }
        
        # Default cache
        default_stats = self.default_cache.get_statistics()
        stats["caches"]["default"] = default_stats
        stats["total_memory_mb"] += default_stats["memory_usage_mb"]
        stats["total_entries"] += default_stats["size"]
        stats["total_hits"] += default_stats["hits"]
        stats["total_misses"] += default_stats["misses"]
        
        # Named caches
        for name, cache in self.caches.items():
            cache_stats = cache.get_statistics()
            stats["caches"][name] = cache_stats
            stats["total_memory_mb"] += cache_stats["memory_usage_mb"]
            stats["total_entries"] += cache_stats["size"]
            stats["total_hits"] += cache_stats["hits"]
            stats["total_misses"] += cache_stats["misses"]
        
        # Calculate global hit rate
        total_accesses = stats["total_hits"] + stats["total_misses"]
        stats["global_hit_rate"] = stats["total_hits"] / max(1, total_accesses)
        
        return stats
    
    async def start_all_background_tasks(self):
        """Start background tasks for all caches."""
        await self.default_cache.start_background_tasks()
        
        tasks = []
        for cache in self.caches.values():
            task = asyncio.create_task(cache.start_background_tasks())
            tasks.append(task)
        
        if tasks:
            await asyncio.gather(*tasks)
    
    async def stop_all_background_tasks(self):
        """Stop background tasks for all caches."""
        await self.default_cache.stop_background_tasks()
        
        tasks = []
        for cache in self.caches.values():
            task = asyncio.create_task(cache.stop_background_tasks())
            tasks.append(task)
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


# Global cache manager instance
_global_cache_manager: Optional[CacheManager] = None


def get_cache_manager() -> CacheManager:
    """Get global cache manager instance."""
    global _global_cache_manager
    if _global_cache_manager is None:
        _global_cache_manager = CacheManager()
    return _global_cache_manager


def get_cache(name: str = "default") -> IntelligentCache:
    """Get cache by name."""
    manager = get_cache_manager()
    if name == "default":
        return manager.default_cache
    return manager.get_cache(name)
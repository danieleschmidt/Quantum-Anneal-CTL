"""Advanced caching system with multiple cache backends and intelligent invalidation."""

import asyncio
import hashlib
import json
import pickle
import time
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from dataclasses import dataclass
from enum import Enum

import redis.asyncio as redis
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from ..utils.structured_logging import StructuredLogger

logger = StructuredLogger("quantum_ctl.caching")


class CacheBackend(Enum):
    """Available cache backends."""
    MEMORY = "memory"
    REDIS = "redis"
    HYBRID = "hybrid"


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    created_at: datetime
    expires_at: Optional[datetime] = None
    access_count: int = 0
    last_accessed: datetime = None
    tags: List[str] = None
    metadata: Dict[str, Any] = None
    size_bytes: int = 0
    
    def __post_init__(self):
        if self.last_accessed is None:
            self.last_accessed = self.created_at
        if self.tags is None:
            self.tags = []
        if self.metadata is None:
            self.metadata = {}


class BaseCacheBackend(ABC):
    """Base cache backend interface."""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[CacheEntry]:
        """Get cache entry by key."""
        pass
    
    @abstractmethod
    async def set(
        self, 
        key: str, 
        value: Any, 
        expires_in: Optional[int] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Set cache entry."""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete cache entry."""
        pass
    
    @abstractmethod
    async def clear(self) -> int:
        """Clear all cache entries."""
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        pass
    
    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        pass


class MemoryCacheBackend(BaseCacheBackend):
    """In-memory cache backend with LRU eviction."""
    
    def __init__(self, max_size: int = 1000, max_memory_mb: int = 100):
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.cache: Dict[str, CacheEntry] = {}
        self.access_order: List[str] = []
        self.current_memory = 0
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
    
    async def get(self, key: str) -> Optional[CacheEntry]:
        """Get cache entry by key."""
        entry = self.cache.get(key)
        if entry is None:
            self.misses += 1
            return None
        
        # Check expiration
        if entry.expires_at and datetime.utcnow() > entry.expires_at:
            await self.delete(key)
            self.misses += 1
            return None
        
        # Update access info
        entry.access_count += 1
        entry.last_accessed = datetime.utcnow()
        
        # Move to end of access order (most recently used)
        if key in self.access_order:
            self.access_order.remove(key)
        self.access_order.append(key)
        
        self.hits += 1
        return entry
    
    async def set(
        self, 
        key: str, 
        value: Any, 
        expires_in: Optional[int] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Set cache entry."""
        # Calculate expiration
        expires_at = None
        if expires_in:
            expires_at = datetime.utcnow() + timedelta(seconds=expires_in)
        
        # Estimate size
        size_bytes = len(pickle.dumps(value))
        
        # Check if we need to make space
        await self._ensure_capacity(size_bytes)
        
        # Create entry
        entry = CacheEntry(
            key=key,
            value=value,
            created_at=datetime.utcnow(),
            expires_at=expires_at,
            tags=tags or [],
            metadata=metadata or {},
            size_bytes=size_bytes
        )
        
        # Update existing entry or add new
        if key in self.cache:
            old_size = self.cache[key].size_bytes
            self.current_memory -= old_size
        
        self.cache[key] = entry
        self.current_memory += size_bytes
        
        # Update access order
        if key in self.access_order:
            self.access_order.remove(key)
        self.access_order.append(key)
        
        return True
    
    async def delete(self, key: str) -> bool:
        """Delete cache entry."""
        if key in self.cache:
            entry = self.cache[key]
            self.current_memory -= entry.size_bytes
            del self.cache[key]
            
            if key in self.access_order:
                self.access_order.remove(key)
            
            return True
        return False
    
    async def clear(self) -> int:
        """Clear all cache entries."""
        count = len(self.cache)
        self.cache.clear()
        self.access_order.clear()
        self.current_memory = 0
        return count
    
    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        return key in self.cache
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0
        
        return {
            "backend": "memory",
            "size": len(self.cache),
            "max_size": self.max_size,
            "memory_mb": self.current_memory / (1024 * 1024),
            "max_memory_mb": self.max_memory_bytes / (1024 * 1024),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "evictions": self.evictions
        }
    
    async def _ensure_capacity(self, new_size_bytes: int) -> None:
        """Ensure cache has capacity for new entry."""
        # Check size limit
        while (len(self.cache) >= self.max_size or 
               self.current_memory + new_size_bytes > self.max_memory_bytes):
            if not self.access_order:
                break
            
            # Evict least recently used
            lru_key = self.access_order[0]
            await self.delete(lru_key)
            self.evictions += 1
    
    async def invalidate_by_tags(self, tags: List[str]) -> int:
        """Invalidate entries by tags."""
        keys_to_delete = []
        for key, entry in self.cache.items():
            if any(tag in entry.tags for tag in tags):
                keys_to_delete.append(key)
        
        for key in keys_to_delete:
            await self.delete(key)
        
        return len(keys_to_delete)


class RedisCacheBackend(BaseCacheBackend):
    """Redis cache backend."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379", prefix: str = "qhvac:"):
        self.redis_url = redis_url
        self.prefix = prefix
        self.client: Optional[redis.Redis] = None
    
    async def connect(self) -> None:
        """Connect to Redis."""
        if self.client is None:
            self.client = redis.from_url(self.redis_url, decode_responses=False)
    
    async def disconnect(self) -> None:
        """Disconnect from Redis."""
        if self.client:
            await self.client.close()
    
    def _key(self, key: str) -> str:
        """Get prefixed key."""
        return f"{self.prefix}{key}"
    
    async def get(self, key: str) -> Optional[CacheEntry]:
        """Get cache entry by key."""
        await self.connect()
        
        data = await self.client.get(self._key(key))
        if data is None:
            return None
        
        try:
            entry = pickle.loads(data)
            entry.access_count += 1
            entry.last_accessed = datetime.utcnow()
            
            # Update access info in Redis
            await self.client.set(self._key(key), pickle.dumps(entry))
            
            return entry
        except Exception as e:
            logger.error(f"Failed to deserialize cache entry: {e}")
            await self.delete(key)
            return None
    
    async def set(
        self, 
        key: str, 
        value: Any, 
        expires_in: Optional[int] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Set cache entry."""
        await self.connect()
        
        # Calculate expiration
        expires_at = None
        if expires_in:
            expires_at = datetime.utcnow() + timedelta(seconds=expires_in)
        
        # Create entry
        entry = CacheEntry(
            key=key,
            value=value,
            created_at=datetime.utcnow(),
            expires_at=expires_at,
            tags=tags or [],
            metadata=metadata or {},
            size_bytes=len(pickle.dumps(value))
        )
        
        try:
            serialized = pickle.dumps(entry)
            await self.client.set(self._key(key), serialized, ex=expires_in)
            
            # Store tags for invalidation
            if tags:
                for tag in tags:
                    await self.client.sadd(f"{self.prefix}tags:{tag}", key)
            
            return True
        except Exception as e:
            logger.error(f"Failed to set cache entry: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete cache entry."""
        await self.connect()
        
        # Get entry to remove tags
        entry = await self.get(key)
        if entry and entry.tags:
            for tag in entry.tags:
                await self.client.srem(f"{self.prefix}tags:{tag}", key)
        
        result = await self.client.delete(self._key(key))
        return result > 0
    
    async def clear(self) -> int:
        """Clear all cache entries."""
        await self.connect()
        
        keys = await self.client.keys(f"{self.prefix}*")
        if keys:
            return await self.client.delete(*keys)
        return 0
    
    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        await self.connect()
        return await self.client.exists(self._key(key)) > 0
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        await self.connect()
        
        info = await self.client.info()
        memory_used = info.get('used_memory', 0)
        
        # Count keys with our prefix
        keys = await self.client.keys(f"{self.prefix}*")
        
        return {
            "backend": "redis",
            "size": len(keys),
            "memory_mb": memory_used / (1024 * 1024),
            "connected_clients": info.get('connected_clients', 0),
            "keyspace_hits": info.get('keyspace_hits', 0),
            "keyspace_misses": info.get('keyspace_misses', 0)
        }
    
    async def invalidate_by_tags(self, tags: List[str]) -> int:
        """Invalidate entries by tags."""
        await self.connect()
        
        keys_to_delete = set()
        for tag in tags:
            tag_keys = await self.client.smembers(f"{self.prefix}tags:{tag}")
            keys_to_delete.update(tag_keys)
        
        count = 0
        for key in keys_to_delete:
            if await self.delete(key.decode() if isinstance(key, bytes) else key):
                count += 1
        
        return count


class SmartCache:
    """Intelligent cache with similarity matching and automatic optimization."""
    
    def __init__(
        self, 
        backend: BaseCacheBackend,
        similarity_threshold: float = 0.9,
        enable_similarity_search: bool = True
    ):
        self.backend = backend
        self.similarity_threshold = similarity_threshold
        self.enable_similarity_search = enable_similarity_search
        
        # Similarity index for QUBO problems
        self.similarity_index: Dict[str, np.ndarray] = {}
        
    def _compute_problem_signature(self, problem_data: Dict[str, Any]) -> np.ndarray:
        """Compute numerical signature for similarity matching."""
        try:
            # Extract key features for similarity comparison
            features = []
            
            # Problem size features
            if 'Q' in problem_data and hasattr(problem_data['Q'], 'shape'):
                Q = problem_data['Q']
                features.extend([
                    Q.shape[0],  # Problem size
                    np.count_nonzero(Q),  # Sparsity
                    np.mean(np.abs(Q)),  # Average magnitude
                    np.std(Q),  # Standard deviation
                ])
            else:
                features.extend([0, 0, 0, 0])
            
            # Constraint features
            if 'constraints' in problem_data:
                constraints = problem_data['constraints']
                features.extend([
                    len(constraints),
                    sum(len(str(c)) for c in constraints) / len(constraints) if constraints else 0
                ])
            else:
                features.extend([0, 0])
            
            # Objective features
            if 'objectives' in problem_data:
                obj = problem_data['objectives']
                if isinstance(obj, dict):
                    features.extend(list(obj.values())[:5])  # Top 5 objectives
                    features.extend([0] * (5 - len(obj)))  # Pad to 5
                else:
                    features.extend([0] * 5)
            else:
                features.extend([0] * 5)
            
            return np.array(features, dtype=float)
        except Exception as e:
            logger.warning(f"Failed to compute problem signature: {e}")
            return np.zeros(11)  # Default signature
    
    async def get_similar_solution(
        self, 
        problem_data: Dict[str, Any],
        max_candidates: int = 10
    ) -> Optional[Tuple[str, float, Any]]:
        """Find similar cached solution."""
        if not self.enable_similarity_search:
            return None
        
        problem_signature = self._compute_problem_signature(problem_data)
        
        # Find similar problems
        similarities = []
        for key, signature in self.similarity_index.items():
            try:
                # Compute cosine similarity
                similarity = cosine_similarity([problem_signature], [signature])[0][0]
                if similarity > self.similarity_threshold:
                    similarities.append((key, similarity))
            except Exception as e:
                logger.warning(f"Error computing similarity for key {key}: {e}")
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return best match
        for key, similarity in similarities[:max_candidates]:
            entry = await self.backend.get(key)
            if entry and entry.value:
                logger.info(
                    f"Found similar solution",
                    original_key=key,
                    similarity_score=similarity
                )
                return key, similarity, entry.value
        
        return None
    
    async def cache_with_signature(
        self,
        key: str,
        value: Any,
        problem_data: Dict[str, Any],
        expires_in: Optional[int] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Cache value with problem signature for similarity matching."""
        # Compute and store signature
        signature = self._compute_problem_signature(problem_data)
        self.similarity_index[key] = signature
        
        # Add problem signature to metadata
        enhanced_metadata = metadata or {}
        enhanced_metadata['problem_signature'] = signature.tolist()
        enhanced_metadata['problem_hash'] = hashlib.sha256(
            json.dumps(problem_data, sort_keys=True, default=str).encode()
        ).hexdigest()
        
        return await self.backend.set(
            key=key,
            value=value,
            expires_in=expires_in,
            tags=tags,
            metadata=enhanced_metadata
        )
    
    async def get(self, key: str) -> Optional[CacheEntry]:
        """Get cache entry with enhanced retrieval."""
        return await self.backend.get(key)
    
    async def set(self, *args, **kwargs) -> bool:
        """Set cache entry."""
        return await self.backend.set(*args, **kwargs)
    
    async def delete(self, key: str) -> bool:
        """Delete cache entry and signature."""
        if key in self.similarity_index:
            del self.similarity_index[key]
        return await self.backend.delete(key)
    
    async def clear(self) -> int:
        """Clear cache and signatures."""
        self.similarity_index.clear()
        return await self.backend.clear()
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get enhanced cache statistics."""
        backend_stats = await self.backend.get_stats()
        backend_stats.update({
            "similarity_index_size": len(self.similarity_index),
            "similarity_threshold": self.similarity_threshold,
            "similarity_search_enabled": self.enable_similarity_search
        })
        return backend_stats


class CacheManager:
    """High-level cache management with multiple backends and strategies."""
    
    def __init__(self):
        self.backends: Dict[str, SmartCache] = {}
        self.default_backend = None
        self.cache_strategies: Dict[str, Dict[str, Any]] = {}
    
    def register_backend(self, name: str, backend: BaseCacheBackend) -> None:
        """Register a cache backend."""
        smart_cache = SmartCache(backend)
        self.backends[name] = smart_cache
        
        if self.default_backend is None:
            self.default_backend = name
        
        logger.info(f"Registered cache backend: {name}")
    
    def set_default_backend(self, name: str) -> None:
        """Set default cache backend."""
        if name in self.backends:
            self.default_backend = name
        else:
            raise ValueError(f"Backend '{name}' not registered")
    
    def register_cache_strategy(
        self, 
        prefix: str, 
        backend: str = None,
        ttl: int = 3600,
        tags: List[str] = None
    ) -> None:
        """Register cache strategy for key prefix."""
        self.cache_strategies[prefix] = {
            "backend": backend or self.default_backend,
            "ttl": ttl,
            "tags": tags or []
        }
    
    def _get_backend_for_key(self, key: str) -> SmartCache:
        """Get appropriate backend for key."""
        # Find matching strategy
        for prefix, strategy in self.cache_strategies.items():
            if key.startswith(prefix):
                backend_name = strategy["backend"]
                return self.backends[backend_name]
        
        # Use default backend
        return self.backends[self.default_backend]
    
    def _get_strategy_for_key(self, key: str) -> Dict[str, Any]:
        """Get cache strategy for key."""
        for prefix, strategy in self.cache_strategies.items():
            if key.startswith(prefix):
                return strategy
        return {"ttl": 3600, "tags": []}
    
    async def get(self, key: str) -> Optional[Any]:
        """Get cached value."""
        backend = self._get_backend_for_key(key)
        entry = await backend.get(key)
        return entry.value if entry else None
    
    async def set(
        self, 
        key: str, 
        value: Any,
        ttl: Optional[int] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Set cached value."""
        backend = self._get_backend_for_key(key)
        strategy = self._get_strategy_for_key(key)
        
        # Use strategy defaults if not specified
        if ttl is None:
            ttl = strategy["ttl"]
        if tags is None:
            tags = strategy["tags"]
        
        return await backend.set(
            key=key,
            value=value,
            expires_in=ttl,
            tags=tags,
            metadata=metadata
        )
    
    async def cache_qubo_solution(
        self,
        problem_data: Dict[str, Any],
        solution: Dict[str, Any],
        computation_time_ms: float,
        ttl: int = 3600
    ) -> str:
        """Cache quantum optimization solution with similarity indexing."""
        # Generate cache key
        problem_hash = hashlib.sha256(
            json.dumps(problem_data, sort_keys=True, default=str).encode()
        ).hexdigest()
        cache_key = f"qubo_solution:{problem_hash}"
        
        # Add computation metrics to solution
        enhanced_solution = {
            **solution,
            "computation_time_ms": computation_time_ms,
            "cached_at": datetime.utcnow().isoformat()
        }
        
        backend = self._get_backend_for_key(cache_key)
        success = await backend.cache_with_signature(
            key=cache_key,
            value=enhanced_solution,
            problem_data=problem_data,
            expires_in=ttl,
            tags=["qubo", "optimization", "quantum"],
            metadata={
                "problem_size": len(problem_data.get('variables', [])),
                "computation_time_ms": computation_time_ms
            }
        )
        
        if success:
            logger.info(
                "Cached QUBO solution",
                cache_key=cache_key,
                problem_size=len(problem_data.get('variables', [])),
                computation_time_ms=computation_time_ms
            )
        
        return cache_key if success else None
    
    async def find_similar_qubo_solution(
        self, 
        problem_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Find similar cached QUBO solution."""
        # Try exact match first
        problem_hash = hashlib.sha256(
            json.dumps(problem_data, sort_keys=True, default=str).encode()
        ).hexdigest()
        exact_key = f"qubo_solution:{problem_hash}"
        
        exact_solution = await self.get(exact_key)
        if exact_solution:
            logger.info("Found exact QUBO solution match", cache_key=exact_key)
            return exact_solution
        
        # Look for similar solutions
        backend = self._get_backend_for_key("qubo_solution:")
        similar = await backend.get_similar_solution(problem_data)
        
        if similar:
            key, similarity, solution = similar
            logger.info(
                "Found similar QUBO solution",
                cache_key=key,
                similarity_score=similarity
            )
            return solution
        
        return None
    
    async def invalidate_by_building(self, building_id: str) -> int:
        """Invalidate all cache entries for a building."""
        total_invalidated = 0
        
        for backend in self.backends.values():
            if hasattr(backend.backend, 'invalidate_by_tags'):
                count = await backend.backend.invalidate_by_tags([f"building:{building_id}"])
                total_invalidated += count
        
        logger.info(f"Invalidated {total_invalidated} cache entries for building {building_id}")
        return total_invalidated
    
    async def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all backends."""
        stats = {}
        for name, backend in self.backends.items():
            stats[name] = await backend.get_stats()
        return stats


# Global cache manager instance
cache_manager = CacheManager()
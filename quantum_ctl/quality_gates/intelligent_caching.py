"""Intelligent Caching System for Quality Gates"""

import asyncio
import hashlib
import json
import time
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import pickle
import sqlite3
import logging
from datetime import datetime, timedelta
import zlib
import threading

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Represents a cached quality gate result"""
    key: str
    result: Dict[str, Any]
    created_at: float
    accessed_at: float
    access_count: int
    ttl_seconds: int
    metadata: Dict[str, Any]
    compressed: bool = False
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired"""
        return time.time() - self.created_at > self.ttl_seconds
    
    def is_stale(self, staleness_threshold: float = 3600) -> bool:
        """Check if cache entry is stale (not accessed recently)"""
        return time.time() - self.accessed_at > staleness_threshold
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


class IntelligentCache:
    """High-performance caching system with intelligent eviction and compression"""
    
    def __init__(self, 
                 max_size_mb: int = 100,
                 default_ttl: int = 3600,
                 compression_threshold: int = 1024,
                 enable_persistence: bool = True,
                 cache_db_path: str = "quality_gates_cache.db"):
        
        self.max_size_mb = max_size_mb
        self.default_ttl = default_ttl
        self.compression_threshold = compression_threshold
        self.enable_persistence = enable_persistence
        self.cache_db_path = cache_db_path
        
        # In-memory cache
        self.cache: Dict[str, CacheEntry] = {}
        self.cache_lock = asyncio.Lock()
        
        # Statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'compressions': 0,
            'cache_size_bytes': 0
        }
        
        # Background maintenance
        self.maintenance_task = None
        self.is_running = False
        
        # Initialize persistence
        if self.enable_persistence:
            self._init_persistence()
    
    def _init_persistence(self):
        """Initialize SQLite database for cache persistence"""
        try:
            self.db_conn = sqlite3.connect(
                self.cache_db_path, 
                check_same_thread=False,
                timeout=30.0
            )
            
            # Create cache table
            self.db_conn.execute("""
                CREATE TABLE IF NOT EXISTS cache_entries (
                    key TEXT PRIMARY KEY,
                    result BLOB,
                    created_at REAL,
                    accessed_at REAL,
                    access_count INTEGER,
                    ttl_seconds INTEGER,
                    metadata TEXT,
                    compressed BOOLEAN
                )
            """)
            
            # Create index for faster lookups
            self.db_conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_created_at ON cache_entries(created_at)
            """)
            
            self.db_conn.commit()
            logger.info(f"Cache persistence initialized: {self.cache_db_path}")
            
        except Exception as e:
            logger.warning(f"Failed to initialize cache persistence: {e}")
            self.enable_persistence = False
    
    async def start(self):
        """Start cache maintenance tasks"""
        self.is_running = True
        
        # Load cache from persistence
        if self.enable_persistence:
            await self._load_from_persistence()
        
        # Start background maintenance
        self.maintenance_task = asyncio.create_task(self._maintenance_loop())
        logger.info("Intelligent cache started")
    
    async def stop(self):
        """Stop cache and cleanup"""
        self.is_running = False
        
        if self.maintenance_task:
            self.maintenance_task.cancel()
            try:
                await self.maintenance_task
            except asyncio.CancelledError:
                pass
        
        # Save cache to persistence
        if self.enable_persistence:
            await self._save_to_persistence()
            self.db_conn.close()
        
        logger.info("Intelligent cache stopped")
    
    async def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get cached result"""
        async with self.cache_lock:
            if key not in self.cache:
                self.stats['misses'] += 1
                return None
            
            entry = self.cache[key]
            
            # Check expiration
            if entry.is_expired():
                await self._remove_entry(key)
                self.stats['misses'] += 1
                return None
            
            # Update access statistics
            entry.accessed_at = time.time()
            entry.access_count += 1
            self.stats['hits'] += 1
            
            # Decompress if needed
            result = entry.result
            if entry.compressed and isinstance(result, bytes):
                try:
                    result = pickle.loads(zlib.decompress(result))
                except Exception as e:
                    logger.warning(f"Failed to decompress cache entry: {e}")
                    await self._remove_entry(key)
                    return None
            
            return result
    
    async def set(self, 
                 key: str, 
                 result: Dict[str, Any], 
                 ttl: Optional[int] = None,
                 metadata: Dict[str, Any] = None) -> bool:
        """Set cached result"""
        async with self.cache_lock:
            ttl = ttl or self.default_ttl
            metadata = metadata or {}
            
            # Serialize result
            try:
                serialized_result = result
                compressed = False
                
                # Compress large results
                result_size = len(json.dumps(result, default=str).encode())
                if result_size > self.compression_threshold:
                    try:
                        compressed_data = zlib.compress(pickle.dumps(result))
                        if len(compressed_data) < result_size * 0.8:  # Only if significant compression
                            serialized_result = compressed_data
                            compressed = True
                            self.stats['compressions'] += 1
                    except Exception as e:
                        logger.warning(f"Compression failed: {e}")
                
                # Create cache entry
                entry = CacheEntry(
                    key=key,
                    result=serialized_result,
                    created_at=time.time(),
                    accessed_at=time.time(),
                    access_count=1,
                    ttl_seconds=ttl,
                    metadata=metadata,
                    compressed=compressed
                )
                
                # Check if we need to evict entries
                await self._ensure_capacity()
                
                # Store entry
                self.cache[key] = entry
                await self._update_cache_size()
                
                return True
                
            except Exception as e:
                logger.error(f"Failed to cache result: {e}")
                return False
    
    async def invalidate(self, key: str) -> bool:
        """Invalidate cached result"""
        async with self.cache_lock:
            if key in self.cache:
                await self._remove_entry(key)
                return True
            return False
    
    async def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate cached results matching pattern"""
        async with self.cache_lock:
            keys_to_remove = []
            
            for key in self.cache.keys():
                if pattern in key or self._matches_pattern(key, pattern):
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                await self._remove_entry(key)
            
            return len(keys_to_remove)
    
    def _matches_pattern(self, key: str, pattern: str) -> bool:
        """Check if key matches pattern (supports * wildcards)"""
        import fnmatch
        return fnmatch.fnmatch(key, pattern)
    
    async def clear(self):
        """Clear all cached results"""
        async with self.cache_lock:
            self.cache.clear()
            self.stats['cache_size_bytes'] = 0
            
            if self.enable_persistence:
                try:
                    self.db_conn.execute("DELETE FROM cache_entries")
                    self.db_conn.commit()
                except Exception as e:
                    logger.warning(f"Failed to clear persistent cache: {e}")
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics"""
        async with self.cache_lock:
            hit_rate = self.stats['hits'] / (self.stats['hits'] + self.stats['misses']) if (self.stats['hits'] + self.stats['misses']) > 0 else 0
            
            # Calculate cache efficiency
            total_entries = len(self.cache)
            compressed_entries = sum(1 for entry in self.cache.values() if entry.compressed)
            avg_access_count = sum(entry.access_count for entry in self.cache.values()) / total_entries if total_entries > 0 else 0
            
            return {
                'total_entries': total_entries,
                'cache_size_mb': self.stats['cache_size_bytes'] / (1024 * 1024),
                'max_size_mb': self.max_size_mb,
                'hit_rate': hit_rate,
                'hits': self.stats['hits'],
                'misses': self.stats['misses'],
                'evictions': self.stats['evictions'],
                'compressions': self.stats['compressions'],
                'compressed_entries': compressed_entries,
                'compression_ratio': compressed_entries / total_entries if total_entries > 0 else 0,
                'avg_access_count': avg_access_count
            }
    
    async def _ensure_capacity(self):
        """Ensure cache doesn't exceed capacity limits"""
        current_size_mb = self.stats['cache_size_bytes'] / (1024 * 1024)
        
        if current_size_mb > self.max_size_mb:
            await self._evict_entries()
    
    async def _evict_entries(self):
        """Evict cache entries using intelligent algorithm"""
        # LFU + LRU hybrid eviction
        # Sort by score: access_count * recency_factor
        
        current_time = time.time()
        entries_with_scores = []
        
        for key, entry in self.cache.items():
            recency_factor = 1.0 / (1.0 + (current_time - entry.accessed_at) / 3600)  # Decay over hours
            score = entry.access_count * recency_factor
            entries_with_scores.append((key, entry, score))
        
        # Sort by score (lowest first - these will be evicted)
        entries_with_scores.sort(key=lambda x: x[2])
        
        # Evict bottom 25% of entries
        evict_count = max(1, len(entries_with_scores) // 4)
        
        for i in range(evict_count):
            key = entries_with_scores[i][0]
            await self._remove_entry(key)
            self.stats['evictions'] += 1
        
        logger.info(f"Evicted {evict_count} cache entries to free memory")
    
    async def _remove_entry(self, key: str):
        """Remove cache entry"""
        if key in self.cache:
            del self.cache[key]
            await self._update_cache_size()
            
            # Remove from persistence
            if self.enable_persistence:
                try:
                    self.db_conn.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
                    self.db_conn.commit()
                except Exception as e:
                    logger.warning(f"Failed to remove entry from persistent cache: {e}")
    
    async def _update_cache_size(self):
        """Update cache size statistics"""
        total_size = 0
        for entry in self.cache.values():
            if entry.compressed and isinstance(entry.result, bytes):
                total_size += len(entry.result)
            else:
                total_size += len(json.dumps(entry.result, default=str).encode())
        
        self.stats['cache_size_bytes'] = total_size
    
    async def _maintenance_loop(self):
        """Background maintenance tasks"""
        while self.is_running:
            try:
                await asyncio.sleep(300)  # 5 minutes
                
                async with self.cache_lock:
                    # Remove expired entries
                    expired_keys = [
                        key for key, entry in self.cache.items() 
                        if entry.is_expired()
                    ]
                    
                    for key in expired_keys:
                        await self._remove_entry(key)
                    
                    if expired_keys:
                        logger.info(f"Removed {len(expired_keys)} expired cache entries")
                    
                    # Compress old entries that aren't compressed yet
                    await self._compress_old_entries()
                    
                    # Save to persistence periodically
                    if self.enable_persistence:
                        await self._save_to_persistence()
                
            except Exception as e:
                logger.warning(f"Cache maintenance error: {e}")
    
    async def _compress_old_entries(self):
        """Compress old entries that aren't compressed yet"""
        compression_age_threshold = 3600  # 1 hour
        current_time = time.time()
        
        for entry in self.cache.values():
            if (not entry.compressed and 
                current_time - entry.created_at > compression_age_threshold):
                
                try:
                    result_size = len(json.dumps(entry.result, default=str).encode())
                    if result_size > self.compression_threshold:
                        compressed_data = zlib.compress(pickle.dumps(entry.result))
                        
                        if len(compressed_data) < result_size * 0.8:
                            entry.result = compressed_data
                            entry.compressed = True
                            self.stats['compressions'] += 1
                            
                except Exception as e:
                    logger.warning(f"Failed to compress old entry: {e}")
    
    async def _load_from_persistence(self):
        """Load cache from persistent storage"""
        if not self.enable_persistence:
            return
        
        try:
            cursor = self.db_conn.execute("""
                SELECT key, result, created_at, accessed_at, access_count, 
                       ttl_seconds, metadata, compressed
                FROM cache_entries
                WHERE created_at + ttl_seconds > ?
            """, (time.time(),))
            
            loaded_count = 0
            for row in cursor.fetchall():
                key, result_blob, created_at, accessed_at, access_count, ttl_seconds, metadata_json, compressed = row
                
                try:
                    # Deserialize result
                    if compressed:
                        result = result_blob
                    else:
                        result = pickle.loads(result_blob)
                    
                    # Deserialize metadata
                    metadata = json.loads(metadata_json) if metadata_json else {}
                    
                    # Create cache entry
                    entry = CacheEntry(
                        key=key,
                        result=result,
                        created_at=created_at,
                        accessed_at=accessed_at,
                        access_count=access_count,
                        ttl_seconds=ttl_seconds,
                        metadata=metadata,
                        compressed=compressed
                    )
                    
                    self.cache[key] = entry
                    loaded_count += 1
                    
                except Exception as e:
                    logger.warning(f"Failed to load cache entry {key}: {e}")
            
            await self._update_cache_size()
            logger.info(f"Loaded {loaded_count} cache entries from persistence")
            
        except Exception as e:
            logger.warning(f"Failed to load cache from persistence: {e}")
    
    async def _save_to_persistence(self):
        """Save cache to persistent storage"""
        if not self.enable_persistence:
            return
        
        try:
            # Clear existing entries
            self.db_conn.execute("DELETE FROM cache_entries")
            
            # Insert current cache entries
            for entry in self.cache.values():
                # Serialize result
                if entry.compressed:
                    result_blob = entry.result
                else:
                    result_blob = pickle.dumps(entry.result)
                
                metadata_json = json.dumps(entry.metadata)
                
                self.db_conn.execute("""
                    INSERT INTO cache_entries 
                    (key, result, created_at, accessed_at, access_count, ttl_seconds, metadata, compressed)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    entry.key, result_blob, entry.created_at, entry.accessed_at,
                    entry.access_count, entry.ttl_seconds, metadata_json, entry.compressed
                ))
            
            self.db_conn.commit()
            logger.debug(f"Saved {len(self.cache)} cache entries to persistence")
            
        except Exception as e:
            logger.warning(f"Failed to save cache to persistence: {e}")


class CacheKeyGenerator:
    """Generate intelligent cache keys for quality gate results"""
    
    @staticmethod
    def generate_gate_key(gate_name: str, context: Dict[str, Any]) -> str:
        """Generate cache key for a quality gate execution"""
        
        # Extract relevant context for caching
        cache_context = {
            'project_root': context.get('project_root', '.'),
            'gate_name': gate_name
        }
        
        # Add file checksums for cache invalidation
        project_root = Path(context.get('project_root', '.'))
        if project_root.exists():
            cache_context['file_checksums'] = CacheKeyGenerator._get_file_checksums(project_root)
        
        # Create deterministic key
        context_str = json.dumps(cache_context, sort_keys=True, default=str)
        key_hash = hashlib.sha256(context_str.encode()).hexdigest()[:16]
        
        return f"gate_{gate_name}_{key_hash}"
    
    @staticmethod
    def _get_file_checksums(project_root: Path) -> Dict[str, str]:
        """Get checksums of relevant project files"""
        checksums = {}
        
        # Include Python source files
        for pattern in ["**/*.py", "pyproject.toml", "requirements.txt", "setup.py"]:
            for file_path in project_root.glob(pattern):
                if CacheKeyGenerator._should_include_file(file_path):
                    try:
                        file_hash = hashlib.md5(file_path.read_bytes()).hexdigest()[:8]
                        relative_path = str(file_path.relative_to(project_root))
                        checksums[relative_path] = file_hash
                    except Exception:
                        continue
        
        return checksums
    
    @staticmethod
    def _should_include_file(file_path: Path) -> bool:
        """Check if file should be included in checksum calculation"""
        
        # Skip common directories
        skip_parts = {
            '__pycache__', '.git', '.pytest_cache', 'node_modules',
            'venv', 'env', '.venv', '.env', 'dist', 'build'
        }
        
        for part in file_path.parts:
            if part in skip_parts:
                return False
        
        # Skip test files for non-test gates
        if file_path.name.startswith('test_'):
            return False
        
        # Skip files that are too large
        try:
            if file_path.stat().st_size > 1024 * 1024:  # 1MB limit
                return False
        except Exception:
            return False
        
        return True


class SmartCacheManager:
    """Manages multiple cache instances with intelligent routing"""
    
    def __init__(self, config: Dict[str, Any] = None):
        config = config or {}
        
        # Primary cache for frequent access
        self.hot_cache = IntelligentCache(
            max_size_mb=config.get('hot_cache_size_mb', 50),
            default_ttl=config.get('hot_cache_ttl', 1800),  # 30 minutes
            compression_threshold=512
        )
        
        # Secondary cache for less frequent access
        self.cold_cache = IntelligentCache(
            max_size_mb=config.get('cold_cache_size_mb', 100),
            default_ttl=config.get('cold_cache_ttl', 3600 * 4),  # 4 hours
            compression_threshold=256,
            cache_db_path="quality_gates_cold_cache.db"
        )
        
        self.key_generator = CacheKeyGenerator()
        
    async def start(self):
        """Start all cache instances"""
        await self.hot_cache.start()
        await self.cold_cache.start()
        logger.info("Smart cache manager started")
    
    async def stop(self):
        """Stop all cache instances"""
        await self.hot_cache.stop()
        await self.cold_cache.stop()
        logger.info("Smart cache manager stopped")
    
    async def get_gate_result(self, gate_name: str, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get cached gate result with intelligent cache selection"""
        
        cache_key = self.key_generator.generate_gate_key(gate_name, context)
        
        # Try hot cache first
        result = await self.hot_cache.get(cache_key)
        if result:
            return result
        
        # Try cold cache
        result = await self.cold_cache.get(cache_key)
        if result:
            # Promote to hot cache if frequently accessed
            await self._maybe_promote_to_hot_cache(cache_key, result, gate_name)
            return result
        
        return None
    
    async def set_gate_result(self, 
                            gate_name: str, 
                            context: Dict[str, Any], 
                            result: Dict[str, Any],
                            metadata: Dict[str, Any] = None):
        """Cache gate result with intelligent cache selection"""
        
        cache_key = self.key_generator.generate_gate_key(gate_name, context)
        metadata = metadata or {}
        
        # Determine cache tier based on gate type and result
        use_hot_cache = self._should_use_hot_cache(gate_name, result, metadata)
        
        if use_hot_cache:
            await self.hot_cache.set(cache_key, result, metadata=metadata)
        else:
            await self.cold_cache.set(cache_key, result, metadata=metadata)
    
    def _should_use_hot_cache(self, gate_name: str, result: Dict[str, Any], metadata: Dict[str, Any]) -> bool:
        """Determine if result should go to hot cache"""
        
        # Fast gates go to hot cache
        fast_gates = {'code_quality', 'documentation'}
        if gate_name in fast_gates:
            return True
        
        # Small results go to hot cache
        result_size = len(json.dumps(result, default=str).encode())
        if result_size < 10 * 1024:  # 10KB
            return True
        
        # Failed results are often re-run, so cache hot
        if not result.get('passed', True):
            return True
        
        return False
    
    async def _maybe_promote_to_hot_cache(self, key: str, result: Dict[str, Any], gate_name: str):
        """Promote frequently accessed items to hot cache"""
        
        # Get access statistics from cold cache
        cold_entry = self.cold_cache.cache.get(key)
        if cold_entry and cold_entry.access_count >= 3:  # Accessed 3+ times
            # Promote to hot cache
            await self.hot_cache.set(
                key, 
                result, 
                ttl=1800,  # 30 minutes in hot cache
                metadata={'promoted_from_cold': True}
            )
    
    async def invalidate_gate(self, gate_name: str, context: Dict[str, Any] = None):
        """Invalidate cached results for a specific gate"""
        
        if context:
            # Invalidate specific cache entry
            cache_key = self.key_generator.generate_gate_key(gate_name, context)
            await self.hot_cache.invalidate(cache_key)
            await self.cold_cache.invalidate(cache_key)
        else:
            # Invalidate all entries for this gate
            pattern = f"gate_{gate_name}_*"
            hot_count = await self.hot_cache.invalidate_pattern(pattern)
            cold_count = await self.cold_cache.invalidate_pattern(pattern)
            
            logger.info(f"Invalidated {hot_count + cold_count} cache entries for gate {gate_name}")
    
    async def get_combined_statistics(self) -> Dict[str, Any]:
        """Get combined statistics from all cache instances"""
        
        hot_stats = await self.hot_cache.get_statistics()
        cold_stats = await self.cold_cache.get_statistics()
        
        return {
            'hot_cache': hot_stats,
            'cold_cache': cold_stats,
            'total_entries': hot_stats['total_entries'] + cold_stats['total_entries'],
            'total_size_mb': hot_stats['cache_size_mb'] + cold_stats['cache_size_mb'],
            'combined_hit_rate': (hot_stats['hits'] + cold_stats['hits']) / 
                               (hot_stats['hits'] + hot_stats['misses'] + cold_stats['hits'] + cold_stats['misses'])
                               if (hot_stats['hits'] + hot_stats['misses'] + cold_stats['hits'] + cold_stats['misses']) > 0 else 0
        }
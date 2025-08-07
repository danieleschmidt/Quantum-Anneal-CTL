"""
Cloud synchronization for distributed quantum HVAC control.
"""

import asyncio
import aiohttp
import json
import logging
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
import hashlib

from ..models.building import BuildingState
from ..utils.performance import get_resource_manager


@dataclass 
class SyncConfig:
    """Cloud sync configuration."""
    endpoint: str
    api_key: Optional[str] = None
    sync_interval: float = 300.0  # 5 minutes
    batch_size: int = 100
    max_retries: int = 3
    timeout: float = 30.0


@dataclass
class MetricData:
    """Performance metric for cloud sync."""
    building_id: str
    timestamp: float
    metrics: Dict[str, Any]
    state_hash: str = ""
    
    def __post_init__(self):
        """Generate state hash."""
        data_str = f"{self.building_id}_{self.timestamp}_{json.dumps(self.metrics, sort_keys=True)}"
        self.state_hash = hashlib.md5(data_str.encode()).hexdigest()


class CloudSync:
    """Synchronize building data and optimization results to cloud."""
    
    def __init__(self, config: SyncConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Sync state
        self._is_syncing = False
        self._last_sync = 0.0
        self._pending_metrics: List[MetricData] = []
        self._sync_stats = {
            'uploads': 0,
            'failures': 0,
            'bytes_sent': 0,
            'last_sync_time': 0.0
        }
        
        # Session management
        self._session: Optional[aiohttp.ClientSession] = None
        
    async def __aenter__(self):
        """Async context manager entry."""
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.timeout),
            headers={'Authorization': f'Bearer {self.config.api_key}' if self.config.api_key else {}}
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._session:
            await self._session.close()
    
    async def sync_building_state(self, building_id: str, state: BuildingState) -> bool:
        """Sync building state to cloud."""
        try:
            state_data = {
                'building_id': building_id,
                'timestamp': state.timestamp,
                'zone_temperatures': state.zone_temperatures.tolist(),
                'outside_temperature': state.outside_temperature,
                'humidity': state.humidity,
                'occupancy': state.occupancy.tolist(),
                'hvac_power': state.hvac_power.tolist(),
                'control_setpoints': state.control_setpoints.tolist(),
                'sync_timestamp': time.time()
            }
            
            return await self._upload_data('building_states', state_data)
            
        except Exception as e:
            self.logger.error(f"Failed to sync building state: {e}")
            return False
    
    async def sync_optimization_result(self, building_id: str, 
                                     result: Dict[str, Any]) -> bool:
        """Sync optimization results to cloud."""
        try:
            result_data = {
                'building_id': building_id,
                'optimization_result': result,
                'sync_timestamp': time.time()
            }
            
            return await self._upload_data('optimization_results', result_data)
            
        except Exception as e:
            self.logger.error(f"Failed to sync optimization result: {e}")
            return False
    
    async def sync_performance_metrics(self, building_id: str, 
                                     metrics: Dict[str, Any]) -> bool:
        """Sync performance metrics to cloud."""
        try:
            metric_data = MetricData(
                building_id=building_id,
                timestamp=time.time(),
                metrics=metrics
            )
            
            self._pending_metrics.append(metric_data)
            
            # Batch upload when we have enough metrics
            if len(self._pending_metrics) >= self.config.batch_size:
                return await self._flush_metrics()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to queue performance metrics: {e}")
            return False
    
    async def _flush_metrics(self) -> bool:
        """Flush pending metrics to cloud."""
        if not self._pending_metrics:
            return True
            
        try:
            batch_data = {
                'metrics': [asdict(m) for m in self._pending_metrics],
                'batch_timestamp': time.time(),
                'batch_size': len(self._pending_metrics)
            }
            
            success = await self._upload_data('performance_metrics', batch_data)
            
            if success:
                self._pending_metrics.clear()
                self.logger.info(f"Synced {len(self._pending_metrics)} metrics to cloud")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to flush metrics: {e}")
            return False
    
    async def _upload_data(self, endpoint: str, data: Dict[str, Any]) -> bool:
        """Upload data to cloud endpoint."""
        if not self._session:
            self.logger.error("Cloud sync session not initialized")
            return False
        
        url = f"{self.config.endpoint}/{endpoint}"
        
        for attempt in range(self.config.max_retries):
            try:
                json_data = json.dumps(data)
                data_size = len(json_data.encode('utf-8'))
                
                async with self._session.post(url, data=json_data,
                                            headers={'Content-Type': 'application/json'}) as resp:
                    
                    if resp.status == 200:
                        self._sync_stats['uploads'] += 1
                        self._sync_stats['bytes_sent'] += data_size
                        self._sync_stats['last_sync_time'] = time.time()
                        return True
                    else:
                        error_text = await resp.text()
                        self.logger.warning(f"Upload failed with status {resp.status}: {error_text}")
                        
            except asyncio.TimeoutError:
                self.logger.warning(f"Upload timeout on attempt {attempt + 1}")
            except Exception as e:
                self.logger.error(f"Upload error on attempt {attempt + 1}: {e}")
            
            if attempt < self.config.max_retries - 1:
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        self._sync_stats['failures'] += 1
        return False
    
    async def download_global_config(self) -> Optional[Dict[str, Any]]:
        """Download global configuration from cloud."""
        if not self._session:
            return None
            
        try:
            url = f"{self.config.endpoint}/global_config"
            
            async with self._session.get(url) as resp:
                if resp.status == 200:
                    config_data = await resp.json()
                    self.logger.info("Downloaded global config from cloud")
                    return config_data
                else:
                    self.logger.warning(f"Config download failed with status {resp.status}")
                    
        except Exception as e:
            self.logger.error(f"Failed to download global config: {e}")
        
        return None
    
    async def download_model_updates(self) -> Optional[Dict[str, Any]]:
        """Download model updates from cloud."""
        if not self._session:
            return None
            
        try:
            url = f"{self.config.endpoint}/model_updates"
            
            async with self._session.get(url) as resp:
                if resp.status == 200:
                    updates = await resp.json()
                    self.logger.info("Downloaded model updates from cloud")
                    return updates
                else:
                    self.logger.warning(f"Model updates download failed with status {resp.status}")
                    
        except Exception as e:
            self.logger.error(f"Failed to download model updates: {e}")
        
        return None
    
    def get_sync_status(self) -> Dict[str, Any]:
        """Get synchronization status."""
        return {
            'is_syncing': self._is_syncing,
            'last_sync': self._last_sync,
            'pending_metrics': len(self._pending_metrics),
            'stats': self._sync_stats.copy(),
            'config': {
                'endpoint': self.config.endpoint,
                'sync_interval': self.config.sync_interval,
                'batch_size': self.config.batch_size
            }
        }
    
    async def start_background_sync(self):
        """Start background synchronization task."""
        if self._is_syncing:
            return
        
        self._is_syncing = True
        self.logger.info("Started background cloud sync")
        
        try:
            while self._is_syncing:
                try:
                    # Sync pending metrics
                    if self._pending_metrics:
                        await self._flush_metrics()
                    
                    # Check for global config updates
                    if time.time() - self._last_sync > self.config.sync_interval:
                        config_updates = await self.download_global_config()
                        model_updates = await self.download_model_updates()
                        
                        if config_updates or model_updates:
                            self.logger.info("Received updates from cloud")
                        
                        self._last_sync = time.time()
                    
                    # Sleep between sync cycles
                    await asyncio.sleep(min(60, self.config.sync_interval / 10))
                    
                except Exception as e:
                    self.logger.error(f"Background sync error: {e}")
                    await asyncio.sleep(60)  # Wait before retrying
                    
        finally:
            self._is_syncing = False
            self.logger.info("Stopped background cloud sync")
    
    def stop_background_sync(self):
        """Stop background synchronization."""
        self._is_syncing = False


class DistributedOptimization:
    """Distributed optimization coordinator."""
    
    def __init__(self, cloud_sync: CloudSync):
        self.cloud_sync = cloud_sync
        self.logger = logging.getLogger(__name__)
        
        # Distributed state
        self._peer_buildings: Dict[str, Dict] = {}
        self._coordination_active = False
        
    async def register_building(self, building_id: str, capabilities: Dict[str, Any]):
        """Register building for distributed optimization."""
        building_data = {
            'building_id': building_id,
            'capabilities': capabilities,
            'registered_at': time.time(),
            'last_seen': time.time()
        }
        
        success = await self.cloud_sync._upload_data('building_registry', building_data)
        
        if success:
            self.logger.info(f"Registered building {building_id} for distributed optimization")
        
        return success
    
    async def discover_peer_buildings(self) -> Dict[str, Dict]:
        """Discover peer buildings for coordination."""
        if not self.cloud_sync._session:
            return {}
        
        try:
            url = f"{self.cloud_sync.config.endpoint}/building_registry"
            
            async with self.cloud_sync._session.get(url) as resp:
                if resp.status == 200:
                    registry = await resp.json()
                    
                    # Filter active peers (seen within last 10 minutes)
                    cutoff_time = time.time() - 600
                    active_peers = {
                        bid: data for bid, data in registry.get('buildings', {}).items()
                        if data.get('last_seen', 0) > cutoff_time
                    }
                    
                    self._peer_buildings = active_peers
                    self.logger.info(f"Discovered {len(active_peers)} peer buildings")
                    
                    return active_peers
                    
        except Exception as e:
            self.logger.error(f"Failed to discover peer buildings: {e}")
        
        return {}
    
    async def coordinate_optimization(self, building_id: str, 
                                   local_problem: Dict[str, Any]) -> Dict[str, Any]:
        """Coordinate optimization with peer buildings."""
        
        # Discover active peers
        peers = await self.discover_peer_buildings()
        
        if not peers:
            # No peers available, run local optimization
            return {'strategy': 'local', 'peers': 0}
        
        try:
            # Share optimization intent
            coordination_data = {
                'building_id': building_id,
                'problem_summary': {
                    'horizon': local_problem.get('horizon', 24),
                    'complexity': local_problem.get('complexity', 'medium'),
                    'priority': local_problem.get('priority', 'normal')
                },
                'coordination_request': True,
                'timestamp': time.time()
            }
            
            success = await self.cloud_sync._upload_data(
                'coordination_requests', coordination_data
            )
            
            if success:
                # Wait for coordination responses
                await asyncio.sleep(5)  # Allow time for responses
                
                # Check for coordination responses
                responses = await self._get_coordination_responses(building_id)
                
                if responses:
                    # Coordinate with peers
                    strategy = self._plan_coordination_strategy(responses)
                    return {
                        'strategy': 'coordinated',
                        'peers': len(responses),
                        'coordination_plan': strategy
                    }
            
        except Exception as e:
            self.logger.error(f"Coordination failed: {e}")
        
        # Fallback to local optimization
        return {'strategy': 'local', 'peers': 0}
    
    async def _get_coordination_responses(self, building_id: str) -> List[Dict]:
        """Get coordination responses from peers."""
        if not self.cloud_sync._session:
            return []
        
        try:
            url = f"{self.cloud_sync.config.endpoint}/coordination_responses"
            params = {'requestor': building_id}
            
            async with self.cloud_sync._session.get(url, params=params) as resp:
                if resp.status == 200:
                    responses_data = await resp.json()
                    return responses_data.get('responses', [])
                    
        except Exception as e:
            self.logger.error(f"Failed to get coordination responses: {e}")
        
        return []
    
    def _plan_coordination_strategy(self, responses: List[Dict]) -> Dict[str, Any]:
        """Plan coordination strategy based on peer responses."""
        # Simple coordination strategy
        total_capacity = sum(r.get('available_capacity', 1.0) for r in responses)
        
        strategy = {
            'type': 'load_balance' if total_capacity > 2.0 else 'local_priority',
            'participants': len(responses),
            'total_capacity': total_capacity,
            'coordination_mode': 'async'  # vs 'sync'
        }
        
        return strategy


async def create_cloud_sync(endpoint: str, api_key: Optional[str] = None) -> CloudSync:
    """Create and initialize cloud sync instance."""
    config = SyncConfig(
        endpoint=endpoint,
        api_key=api_key,
        sync_interval=300.0,  # 5 minutes
        batch_size=50
    )
    
    return CloudSync(config)


# Global instances
_cloud_sync_instance = None
_distributed_optimizer = None


def get_cloud_sync() -> Optional[CloudSync]:
    """Get global cloud sync instance."""
    return _cloud_sync_instance


def get_distributed_optimizer() -> Optional[DistributedOptimization]:
    """Get global distributed optimizer instance.""" 
    return _distributed_optimizer


async def initialize_cloud_integration(endpoint: str, api_key: str = None):
    """Initialize cloud integration components."""
    global _cloud_sync_instance, _distributed_optimizer
    
    if _cloud_sync_instance is None:
        _cloud_sync_instance = await create_cloud_sync(endpoint, api_key)
        _distributed_optimizer = DistributedOptimization(_cloud_sync_instance)
        
        logger = logging.getLogger(__name__)
        logger.info("Initialized cloud integration")
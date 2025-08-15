"""
Distributed self-healing pipeline guard for multi-node quantum HVAC systems.
"""

import asyncio
import json
import time
import uuid
from typing import Dict, List, Any, Optional, Set, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import aiohttp
import redis.asyncio as redis


class NodeRole(Enum):
    LEADER = "leader"
    FOLLOWER = "follower"
    CANDIDATE = "candidate"


class GuardMessage(Enum):
    HEARTBEAT = "heartbeat"
    HEALTH_UPDATE = "health_update"
    RECOVERY_REQUEST = "recovery_request"
    RECOVERY_RESPONSE = "recovery_response"
    LEADER_ELECTION = "leader_election"
    CONFIG_UPDATE = "config_update"


@dataclass
class GuardNode:
    node_id: str
    address: str
    port: int
    role: NodeRole
    last_seen: float
    health_score: float = 1.0
    capabilities: List[str] = None
    
    
@dataclass
class DistributedHealthState:
    node_id: str
    timestamp: float
    component_health: Dict[str, bool]
    system_metrics: Dict[str, float]
    active_recoveries: List[str]


class DistributedPipelineGuard:
    """
    Distributed self-healing pipeline guard that coordinates across multiple nodes
    for high availability and fault tolerance.
    """
    
    def __init__(
        self,
        node_id: str,
        bind_address: str = "0.0.0.0",
        bind_port: int = 8765,
        redis_url: str = "redis://localhost:6379"
    ):
        self.node_id = node_id
        self.bind_address = bind_address
        self.bind_port = bind_port
        self.redis_url = redis_url
        
        self.role = NodeRole.FOLLOWER
        self.term = 0
        self.voted_for = None
        self.leader_id = None
        self.last_heartbeat = 0
        
        self.cluster_nodes: Dict[str, GuardNode] = {}
        self.health_states: Dict[str, DistributedHealthState] = {}
        self.recovery_coordination: Dict[str, Dict[str, Any]] = {}
        
        self.redis_client: Optional[redis.Redis] = None
        self.message_handlers: Dict[GuardMessage, Callable] = {}
        self.running = False
        
        self._setup_message_handlers()
        
    def _setup_message_handlers(self):
        """Setup message handlers for different message types."""
        self.message_handlers = {
            GuardMessage.HEARTBEAT: self._handle_heartbeat,
            GuardMessage.HEALTH_UPDATE: self._handle_health_update,
            GuardMessage.RECOVERY_REQUEST: self._handle_recovery_request,
            GuardMessage.RECOVERY_RESPONSE: self._handle_recovery_response,
            GuardMessage.LEADER_ELECTION: self._handle_leader_election,
            GuardMessage.CONFIG_UPDATE: self._handle_config_update
        }
        
    async def start(self):
        """Start the distributed pipeline guard."""
        self.running = True
        
        # Connect to Redis for cluster coordination
        self.redis_client = redis.from_url(self.redis_url)
        
        # Register this node
        await self._register_node()
        
        # Start background tasks
        asyncio.create_task(self._heartbeat_loop())
        asyncio.create_task(self._health_monitoring_loop())
        asyncio.create_task(self._leader_election_loop())
        asyncio.create_task(self._cluster_maintenance_loop())
        
        print(f"Distributed pipeline guard started on {self.bind_address}:{self.bind_port}")
        
    async def stop(self):
        """Stop the distributed pipeline guard."""
        self.running = False
        
        # Deregister node
        await self._deregister_node()
        
        if self.redis_client:
            await self.redis_client.close()
            
    async def _register_node(self):
        """Register this node in the cluster."""
        node_info = {
            "node_id": self.node_id,
            "address": self.bind_address,
            "port": self.bind_port,
            "role": self.role.value,
            "last_seen": time.time(),
            "capabilities": ["quantum_solver", "hvac_controller", "monitoring"]
        }
        
        await self.redis_client.hset(
            "cluster:nodes",
            self.node_id,
            json.dumps(node_info)
        )
        
        # Set TTL for node registration
        await self.redis_client.expire(f"cluster:node:{self.node_id}", 60)
        
    async def _deregister_node(self):
        """Deregister this node from the cluster."""
        await self.redis_client.hdel("cluster:nodes", self.node_id)
        await self.redis_client.delete(f"cluster:node:{self.node_id}")
        
    async def _heartbeat_loop(self):
        """Send periodic heartbeats to maintain cluster membership."""
        while self.running:
            try:
                await self._send_heartbeat()
                await asyncio.sleep(5)  # Heartbeat every 5 seconds
            except Exception as e:
                print(f"Heartbeat error: {e}")
                await asyncio.sleep(1)
                
    async def _send_heartbeat(self):
        """Send heartbeat to cluster."""
        heartbeat_data = {
            "node_id": self.node_id,
            "term": self.term,
            "role": self.role.value,
            "timestamp": time.time(),
            "health_score": self._calculate_health_score()
        }
        
        await self.redis_client.publish(
            "cluster:heartbeat",
            json.dumps(heartbeat_data)
        )
        
        # Update node registration
        await self._register_node()
        
    def _calculate_health_score(self) -> float:
        """Calculate health score for this node."""
        # Simplified health scoring
        try:
            import psutil
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory_percent = psutil.virtual_memory().percent
            
            # Health score inversely related to resource usage
            health_score = 1.0 - ((cpu_percent + memory_percent) / 200)
            return max(0.0, min(1.0, health_score))
            
        except Exception:
            return 0.5  # Default neutral score
            
    async def _health_monitoring_loop(self):
        """Monitor and share health state with cluster."""
        while self.running:
            try:
                await self._share_health_state()
                await asyncio.sleep(10)  # Share health every 10 seconds
            except Exception as e:
                print(f"Health monitoring error: {e}")
                await asyncio.sleep(1)
                
    async def _share_health_state(self):
        """Share this node's health state with the cluster."""
        # Get current health state (would integrate with local pipeline guard)
        health_state = DistributedHealthState(
            node_id=self.node_id,
            timestamp=time.time(),
            component_health={
                "quantum_solver": True,  # Placeholder
                "hvac_controller": True,
                "bms_connector": True
            },
            system_metrics={
                "cpu_percent": 25.0,     # Placeholder
                "memory_percent": 45.0,
                "disk_usage": 60.0
            },
            active_recoveries=[]
        )
        
        # Store in Redis
        await self.redis_client.hset(
            "cluster:health",
            self.node_id,
            json.dumps(asdict(health_state))
        )
        
        # Publish health update
        await self.redis_client.publish(
            "cluster:health_update",
            json.dumps(asdict(health_state))
        )
        
    async def _leader_election_loop(self):
        """Handle leader election process."""
        while self.running:
            try:
                if self.role == NodeRole.FOLLOWER:
                    # Check if leader is alive
                    if not await self._is_leader_alive():
                        await self._start_election()
                        
                elif self.role == NodeRole.LEADER:
                    # Send leader heartbeats
                    await self._send_leader_heartbeat()
                    
                await asyncio.sleep(15)  # Check leadership every 15 seconds
                
            except Exception as e:
                print(f"Leader election error: {e}")
                await asyncio.sleep(1)
                
    async def _is_leader_alive(self) -> bool:
        """Check if current leader is alive."""
        if not self.leader_id:
            return False
            
        leader_heartbeat = await self.redis_client.get(f"cluster:leader:{self.leader_id}")
        if not leader_heartbeat:
            return False
            
        heartbeat_data = json.loads(leader_heartbeat)
        last_heartbeat = heartbeat_data.get("timestamp", 0)
        
        # Leader is considered dead if no heartbeat for 30 seconds
        return (time.time() - last_heartbeat) < 30
        
    async def _start_election(self):
        """Start leader election process."""
        self.role = NodeRole.CANDIDATE
        self.term += 1
        self.voted_for = self.node_id
        
        print(f"Starting leader election for term {self.term}")
        
        # Vote for self
        votes = 1
        
        # Request votes from other nodes
        cluster_nodes = await self._get_cluster_nodes()
        for node_id in cluster_nodes:
            if node_id != self.node_id:
                vote_granted = await self._request_vote(node_id)
                if vote_granted:
                    votes += 1
                    
        # Check if won election
        required_votes = (len(cluster_nodes) // 2) + 1
        if votes >= required_votes:
            await self._become_leader()
        else:
            self.role = NodeRole.FOLLOWER
            
    async def _request_vote(self, node_id: str) -> bool:
        """Request vote from another node."""
        vote_request = {
            "candidate_id": self.node_id,
            "term": self.term,
            "timestamp": time.time()
        }
        
        # Send vote request (simplified - would use HTTP in real implementation)
        await self.redis_client.publish(
            f"cluster:vote_request:{node_id}",
            json.dumps(vote_request)
        )
        
        # Wait for response (simplified)
        try:
            response = await asyncio.wait_for(
                self.redis_client.get(f"cluster:vote_response:{self.node_id}:{node_id}"),
                timeout=5.0
            )
            if response:
                response_data = json.loads(response)
                return response_data.get("vote_granted", False)
        except asyncio.TimeoutError:
            pass
            
        return False
        
    async def _become_leader(self):
        """Become cluster leader."""
        self.role = NodeRole.LEADER
        self.leader_id = self.node_id
        
        print(f"Became cluster leader for term {self.term}")
        
        # Announce leadership
        leadership_announcement = {
            "leader_id": self.node_id,
            "term": self.term,
            "timestamp": time.time()
        }
        
        await self.redis_client.set(
            f"cluster:leader:{self.node_id}",
            json.dumps(leadership_announcement),
            ex=60  # 60 second expiry
        )
        
        await self.redis_client.publish(
            "cluster:leader_announcement",
            json.dumps(leadership_announcement)
        )
        
    async def _send_leader_heartbeat(self):
        """Send leader heartbeat."""
        leader_heartbeat = {
            "leader_id": self.node_id,
            "term": self.term,
            "timestamp": time.time()
        }
        
        await self.redis_client.set(
            f"cluster:leader:{self.node_id}",
            json.dumps(leader_heartbeat),
            ex=60
        )
        
    async def _cluster_maintenance_loop(self):
        """Maintain cluster state and handle node failures."""
        while self.running:
            try:
                await self._update_cluster_state()
                await self._detect_failed_nodes()
                await asyncio.sleep(20)  # Maintenance every 20 seconds
            except Exception as e:
                print(f"Cluster maintenance error: {e}")
                await asyncio.sleep(1)
                
    async def _update_cluster_state(self):
        """Update knowledge of cluster state."""
        cluster_nodes = await self._get_cluster_nodes()
        
        # Update local cluster state
        current_time = time.time()
        for node_id, node_data in cluster_nodes.items():
            if node_id not in self.cluster_nodes:
                self.cluster_nodes[node_id] = GuardNode(**node_data)
            else:
                # Update existing node
                for key, value in node_data.items():
                    setattr(self.cluster_nodes[node_id], key, value)
                    
        # Remove stale nodes
        stale_nodes = []
        for node_id, node in self.cluster_nodes.items():
            if current_time - node.last_seen > 120:  # 2 minutes
                stale_nodes.append(node_id)
                
        for node_id in stale_nodes:
            del self.cluster_nodes[node_id]
            
    async def _get_cluster_nodes(self) -> Dict[str, Dict[str, Any]]:
        """Get current cluster nodes from Redis."""
        nodes_data = await self.redis_client.hgetall("cluster:nodes")
        
        cluster_nodes = {}
        for node_id, node_json in nodes_data.items():
            try:
                node_data = json.loads(node_json)
                cluster_nodes[node_id.decode()] = node_data
            except Exception as e:
                print(f"Error parsing node data for {node_id}: {e}")
                
        return cluster_nodes
        
    async def _detect_failed_nodes(self):
        """Detect and handle failed nodes."""
        if self.role != NodeRole.LEADER:
            return
            
        current_time = time.time()
        failed_nodes = []
        
        for node_id, node in self.cluster_nodes.items():
            if node_id != self.node_id and current_time - node.last_seen > 60:
                failed_nodes.append(node_id)
                
        for failed_node_id in failed_nodes:
            await self._handle_node_failure(failed_node_id)
            
    async def _handle_node_failure(self, failed_node_id: str):
        """Handle failure of a cluster node."""
        print(f"Detected failure of node {failed_node_id}")
        
        # Get failed node's health state
        failed_health = self.health_states.get(failed_node_id)
        if not failed_health:
            return
            
        # Find healthy nodes to take over failed components
        healthy_nodes = [
            node_id for node_id, node in self.cluster_nodes.items()
            if (node_id != failed_node_id and 
                time.time() - node.last_seen < 30 and
                node.health_score > 0.7)
        ]
        
        if not healthy_nodes:
            print("No healthy nodes available for failover")
            return
            
        # Coordinate component recovery
        for component, was_healthy in failed_health.component_health.items():
            if was_healthy:  # Component was running on failed node
                # Select best node for takeover
                target_node = await self._select_recovery_node(component, healthy_nodes)
                if target_node:
                    await self._coordinate_component_recovery(component, target_node, failed_node_id)
                    
    async def _select_recovery_node(self, component: str, candidate_nodes: List[str]) -> Optional[str]:
        """Select best node to recover a component."""
        # Simple selection based on health score and capabilities
        best_node = None
        best_score = 0
        
        for node_id in candidate_nodes:
            node = self.cluster_nodes.get(node_id)
            if not node:
                continue
                
            # Check if node has capability for this component
            if node.capabilities and component not in node.capabilities:
                continue
                
            # Select node with highest health score
            if node.health_score > best_score:
                best_score = node.health_score
                best_node = node_id
                
        return best_node
        
    async def _coordinate_component_recovery(self, component: str, target_node: str, failed_node: str):
        """Coordinate recovery of a component on target node."""
        recovery_request = {
            "request_id": str(uuid.uuid4()),
            "component": component,
            "target_node": target_node,
            "failed_node": failed_node,
            "timestamp": time.time(),
            "coordinator": self.node_id
        }
        
        # Store recovery coordination state
        self.recovery_coordination[recovery_request["request_id"]] = recovery_request
        
        # Send recovery request to target node
        await self.redis_client.publish(
            f"cluster:recovery_request:{target_node}",
            json.dumps(recovery_request)
        )
        
        print(f"Coordinating recovery of {component} on node {target_node}")
        
    async def _handle_heartbeat(self, message_data: Dict[str, Any]):
        """Handle heartbeat message from another node."""
        node_id = message_data.get("node_id")
        if node_id and node_id != self.node_id:
            # Update node information
            if node_id not in self.cluster_nodes:
                # This would get full node info from Redis
                pass
            else:
                self.cluster_nodes[node_id].last_seen = time.time()
                self.cluster_nodes[node_id].health_score = message_data.get("health_score", 0.5)
                
    async def _handle_health_update(self, message_data: Dict[str, Any]):
        """Handle health update from another node."""
        node_id = message_data.get("node_id")
        if node_id and node_id != self.node_id:
            health_state = DistributedHealthState(**message_data)
            self.health_states[node_id] = health_state
            
    async def _handle_recovery_request(self, message_data: Dict[str, Any]):
        """Handle recovery request for this node."""
        component = message_data.get("component")
        request_id = message_data.get("request_id")
        
        print(f"Received recovery request for {component}")
        
        # Attempt component recovery
        recovery_success = await self._recover_component_locally(component)
        
        # Send response
        response = {
            "request_id": request_id,
            "success": recovery_success,
            "node_id": self.node_id,
            "timestamp": time.time()
        }
        
        coordinator = message_data.get("coordinator")
        if coordinator:
            await self.redis_client.publish(
                f"cluster:recovery_response:{coordinator}",
                json.dumps(response)
            )
            
    async def _recover_component_locally(self, component: str) -> bool:
        """Recover a component on this local node."""
        # This would integrate with the local pipeline guard
        print(f"Recovering component {component} locally")
        
        # Simulate recovery
        await asyncio.sleep(2)
        return True  # Placeholder
        
    async def _handle_recovery_response(self, message_data: Dict[str, Any]):
        """Handle recovery response from target node."""
        request_id = message_data.get("request_id")
        success = message_data.get("success", False)
        
        if request_id in self.recovery_coordination:
            recovery_info = self.recovery_coordination[request_id]
            component = recovery_info["component"]
            target_node = recovery_info["target_node"]
            
            if success:
                print(f"Component {component} successfully recovered on {target_node}")
            else:
                print(f"Component {component} recovery failed on {target_node}")
                # Could try alternative recovery strategies
                
            del self.recovery_coordination[request_id]
            
    async def _handle_leader_election(self, message_data: Dict[str, Any]):
        """Handle leader election message."""
        # Implementation would depend on specific election protocol
        pass
        
    async def _handle_config_update(self, message_data: Dict[str, Any]):
        """Handle configuration update from leader."""
        if self.role == NodeRole.LEADER:
            return  # Leaders don't process config updates from others
            
        config_data = message_data.get("config")
        if config_data:
            print("Received configuration update from leader")
            # Apply configuration changes
            
    def get_cluster_status(self) -> Dict[str, Any]:
        """Get current cluster status."""
        return {
            "node_id": self.node_id,
            "role": self.role.value,
            "term": self.term,
            "leader_id": self.leader_id,
            "cluster_size": len(self.cluster_nodes),
            "healthy_nodes": sum(
                1 for node in self.cluster_nodes.values()
                if time.time() - node.last_seen < 60
            ),
            "nodes": {
                node_id: {
                    "role": node.role.value,
                    "health_score": node.health_score,
                    "last_seen": node.last_seen,
                    "online": time.time() - node.last_seen < 60
                }
                for node_id, node in self.cluster_nodes.items()
            },
            "active_recoveries": len(self.recovery_coordination),
            "timestamp": time.time()
        }
"""WebSocket manager for real-time data streaming."""

import asyncio
import json
import time
from datetime import datetime
from typing import Dict, List, Set, Any, Optional, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
import weakref

from fastapi import WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from ..core.tenant_manager import TenantContext
from ..utils.structured_logging import StructuredLogger

logger = StructuredLogger("quantum_ctl.websocket")


class MessageType(Enum):
    """WebSocket message types."""
    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"
    DATA = "data"
    STATUS = "status"
    ERROR = "error"
    HEARTBEAT = "heartbeat"


class SubscriptionType(Enum):
    """Subscription types."""
    BUILDING_STATUS = "building_status"
    OPTIMIZATION_RESULTS = "optimization_results"
    SENSOR_DATA = "sensor_data"
    SYSTEM_ALERTS = "system_alerts"
    QUANTUM_STATUS = "quantum_status"


@dataclass
class Subscription:
    """WebSocket subscription."""
    type: SubscriptionType
    filters: Dict[str, Any]
    client_id: str
    tenant_id: Optional[str] = None
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
    
    def matches(self, data: Dict[str, Any]) -> bool:
        """Check if data matches subscription filters."""
        if not self.filters:
            return True
        
        for key, expected_value in self.filters.items():
            if key not in data:
                continue
            
            data_value = data[key]
            
            # Handle different filter types
            if isinstance(expected_value, list):
                if data_value not in expected_value:
                    return False
            elif isinstance(expected_value, dict):
                # Range filter
                if 'min' in expected_value and data_value < expected_value['min']:
                    return False
                if 'max' in expected_value and data_value > expected_value['max']:
                    return False
            else:
                if data_value != expected_value:
                    return False
        
        return True


class WebSocketMessage(BaseModel):
    """WebSocket message model."""
    type: str
    timestamp: str = None
    data: Dict[str, Any] = None
    subscription_id: str = None
    error: str = None
    
    def __init__(self, **kwargs):
        if 'timestamp' not in kwargs:
            kwargs['timestamp'] = datetime.utcnow().isoformat()
        super().__init__(**kwargs)


class WebSocketClient:
    """WebSocket client connection."""
    
    def __init__(
        self, 
        websocket: WebSocket, 
        client_id: str,
        tenant_context: Optional[TenantContext] = None
    ):
        self.websocket = websocket
        self.client_id = client_id
        self.tenant_context = tenant_context
        self.subscriptions: Dict[str, Subscription] = {}
        self.connected_at = datetime.utcnow()
        self.last_heartbeat = time.time()
        self.message_count = 0
        self.is_active = True
    
    async def send_message(self, message: WebSocketMessage) -> bool:
        """Send message to client."""
        try:
            await self.websocket.send_text(message.json())
            self.message_count += 1
            return True
        except Exception as e:
            logger.error(f"Failed to send message to client {self.client_id}: {e}")
            self.is_active = False
            return False
    
    async def send_data(self, subscription_id: str, data: Dict[str, Any]) -> bool:
        """Send data message to client."""
        message = WebSocketMessage(
            type=MessageType.DATA.value,
            subscription_id=subscription_id,
            data=data
        )
        return await self.send_message(message)
    
    async def send_error(self, error: str, subscription_id: str = None) -> bool:
        """Send error message to client."""
        message = WebSocketMessage(
            type=MessageType.ERROR.value,
            error=error,
            subscription_id=subscription_id
        )
        return await self.send_message(message)
    
    async def send_status(self, status: str, data: Dict[str, Any] = None) -> bool:
        """Send status message to client."""
        message = WebSocketMessage(
            type=MessageType.STATUS.value,
            data={"status": status, **(data or {})}
        )
        return await self.send_message(message)
    
    def add_subscription(self, subscription: Subscription) -> str:
        """Add subscription to client."""
        subscription_id = f"{subscription.type.value}_{len(self.subscriptions)}"
        subscription.client_id = self.client_id
        if self.tenant_context:
            subscription.tenant_id = self.tenant_context.get_tenant_id()
        
        self.subscriptions[subscription_id] = subscription
        logger.debug(
            f"Added subscription {subscription_id} for client {self.client_id}",
            subscription_type=subscription.type.value,
            filters=subscription.filters
        )
        return subscription_id
    
    def remove_subscription(self, subscription_id: str) -> bool:
        """Remove subscription from client."""
        if subscription_id in self.subscriptions:
            del self.subscriptions[subscription_id]
            logger.debug(f"Removed subscription {subscription_id} for client {self.client_id}")
            return True
        return False
    
    def get_matching_subscriptions(self, data: Dict[str, Any], subscription_type: SubscriptionType) -> List[str]:
        """Get subscription IDs that match the data."""
        matching = []
        for sub_id, subscription in self.subscriptions.items():
            if subscription.type == subscription_type and subscription.matches(data):
                matching.append(sub_id)
        return matching
    
    def update_heartbeat(self):
        """Update last heartbeat time."""
        self.last_heartbeat = time.time()
    
    def is_alive(self, timeout_seconds: int = 60) -> bool:
        """Check if client is still alive based on heartbeat."""
        return time.time() - self.last_heartbeat < timeout_seconds


class WebSocketManager:
    """Manager for WebSocket connections and real-time streaming."""
    
    def __init__(self):
        self.clients: Dict[str, WebSocketClient] = {}
        self.subscriptions_by_type: Dict[SubscriptionType, Set[str]] = {
            sub_type: set() for sub_type in SubscriptionType
        }
        self.message_handlers: Dict[str, Callable] = {}
        self.background_tasks: Set[asyncio.Task] = set()
        
        # Statistics
        self.total_connections = 0
        self.total_messages = 0
        self.start_time = datetime.utcnow()
        
        # Start cleanup task
        self._start_cleanup_task()
    
    async def connect_client(
        self, 
        websocket: WebSocket, 
        client_id: str,
        tenant_context: Optional[TenantContext] = None
    ) -> WebSocketClient:
        """Accept WebSocket connection and create client."""
        await websocket.accept()
        
        client = WebSocketClient(websocket, client_id, tenant_context)
        self.clients[client_id] = client
        self.total_connections += 1
        
        # Send welcome message
        await client.send_status("connected", {
            "client_id": client_id,
            "server_time": datetime.utcnow().isoformat()
        })
        
        logger.info(
            f"WebSocket client connected: {client_id}",
            tenant_id=tenant_context.get_tenant_id() if tenant_context else None,
            total_clients=len(self.clients)
        )
        
        return client
    
    async def disconnect_client(self, client_id: str, reason: str = "disconnected"):
        """Disconnect client and cleanup."""
        if client_id in self.clients:
            client = self.clients[client_id]
            
            # Remove all subscriptions
            for subscription_id in list(client.subscriptions.keys()):
                self._remove_subscription_from_index(subscription_id, client.subscriptions[subscription_id].type)
            
            # Remove client
            del self.clients[client_id]
            client.is_active = False
            
            logger.info(
                f"WebSocket client disconnected: {client_id}",
                reason=reason,
                total_clients=len(self.clients),
                connection_duration_seconds=(datetime.utcnow() - client.connected_at).total_seconds()
            )
    
    async def handle_message(self, client_id: str, message_data: str):
        """Handle incoming WebSocket message."""
        client = self.clients.get(client_id)
        if not client:
            return
        
        try:
            message = json.loads(message_data)
            message_type = message.get("type")
            
            if message_type == MessageType.SUBSCRIBE.value:
                await self._handle_subscribe(client, message)
            elif message_type == MessageType.UNSUBSCRIBE.value:
                await self._handle_unsubscribe(client, message)
            elif message_type == MessageType.HEARTBEAT.value:
                await self._handle_heartbeat(client, message)
            else:
                # Check custom handlers
                handler = self.message_handlers.get(message_type)
                if handler:
                    await handler(client, message)
                else:
                    await client.send_error(f"Unknown message type: {message_type}")
            
            self.total_messages += 1
            
        except json.JSONDecodeError:
            await client.send_error("Invalid JSON message")
        except Exception as e:
            logger.error(f"Error handling message from {client_id}: {e}")
            await client.send_error(f"Message handling error: {str(e)}")
    
    async def _handle_subscribe(self, client: WebSocketClient, message: Dict[str, Any]):
        """Handle subscription request."""
        try:
            subscription_type_str = message.get("subscription_type")
            filters = message.get("filters", {})
            
            # Validate subscription type
            try:
                subscription_type = SubscriptionType(subscription_type_str)
            except ValueError:
                await client.send_error(f"Invalid subscription type: {subscription_type_str}")
                return
            
            # Check tenant permissions
            if client.tenant_context and not self._check_subscription_permission(client.tenant_context, subscription_type):
                await client.send_error("Insufficient permissions for subscription")
                return
            
            # Create subscription
            subscription = Subscription(
                type=subscription_type,
                filters=filters,
                client_id=client.client_id
            )
            
            subscription_id = client.add_subscription(subscription)
            self.subscriptions_by_type[subscription_type].add(subscription_id)
            
            # Send confirmation
            await client.send_status("subscribed", {
                "subscription_id": subscription_id,
                "subscription_type": subscription_type.value,
                "filters": filters
            })
            
        except Exception as e:
            await client.send_error(f"Subscription error: {str(e)}")
    
    async def _handle_unsubscribe(self, client: WebSocketClient, message: Dict[str, Any]):
        """Handle unsubscription request."""
        subscription_id = message.get("subscription_id")
        
        if not subscription_id:
            await client.send_error("Missing subscription_id")
            return
        
        if subscription_id in client.subscriptions:
            subscription_type = client.subscriptions[subscription_id].type
            client.remove_subscription(subscription_id)
            self._remove_subscription_from_index(subscription_id, subscription_type)
            
            await client.send_status("unsubscribed", {
                "subscription_id": subscription_id
            })
        else:
            await client.send_error(f"Subscription not found: {subscription_id}")
    
    async def _handle_heartbeat(self, client: WebSocketClient, message: Dict[str, Any]):
        """Handle heartbeat message."""
        client.update_heartbeat()
        
        # Send heartbeat response
        await client.send_message(WebSocketMessage(
            type=MessageType.HEARTBEAT.value,
            data={"server_time": datetime.utcnow().isoformat()}
        ))
    
    def _check_subscription_permission(self, tenant_context: TenantContext, subscription_type: SubscriptionType) -> bool:
        """Check if tenant has permission for subscription type."""
        # Define permission mapping
        permission_map = {
            SubscriptionType.BUILDING_STATUS: ("buildings", "read"),
            SubscriptionType.OPTIMIZATION_RESULTS: ("optimization", "read"),
            SubscriptionType.SENSOR_DATA: ("buildings", "read"),
            SubscriptionType.SYSTEM_ALERTS: ("system", "read"),
            SubscriptionType.QUANTUM_STATUS: ("quantum", "read")
        }
        
        resource, action = permission_map.get(subscription_type, ("", ""))
        return tenant_context.is_authorized(resource, action)
    
    def _remove_subscription_from_index(self, subscription_id: str, subscription_type: SubscriptionType):
        """Remove subscription from type index."""
        if subscription_id in self.subscriptions_by_type[subscription_type]:
            self.subscriptions_by_type[subscription_type].remove(subscription_id)
    
    async def broadcast(
        self,
        subscription_type: SubscriptionType,
        data: Dict[str, Any],
        tenant_filter: Optional[str] = None
    ):
        """Broadcast data to all matching subscribers."""
        subscription_ids = self.subscriptions_by_type[subscription_type].copy()
        
        for sub_id in subscription_ids:
            # Find client with this subscription
            for client in self.clients.values():
                if sub_id in client.subscriptions:
                    subscription = client.subscriptions[sub_id]
                    
                    # Check tenant filter
                    if tenant_filter and subscription.tenant_id != tenant_filter:
                        continue
                    
                    # Check if data matches subscription filters
                    if subscription.matches(data):
                        success = await client.send_data(sub_id, data)
                        if not success:
                            # Client disconnected, clean up
                            await self.disconnect_client(client.client_id, "send_failed")
                    break
    
    async def broadcast_to_tenant(
        self,
        tenant_id: str,
        subscription_type: SubscriptionType,
        data: Dict[str, Any]
    ):
        """Broadcast data to all clients of a specific tenant."""
        await self.broadcast(subscription_type, data, tenant_filter=tenant_id)
    
    async def send_to_client(
        self,
        client_id: str,
        data: Dict[str, Any],
        subscription_id: str = None
    ) -> bool:
        """Send data to specific client."""
        client = self.clients.get(client_id)
        if client:
            if subscription_id:
                return await client.send_data(subscription_id, data)
            else:
                message = WebSocketMessage(type=MessageType.DATA.value, data=data)
                return await client.send_message(message)
        return False
    
    def register_message_handler(self, message_type: str, handler: Callable):
        """Register custom message handler."""
        self.message_handlers[message_type] = handler
    
    def get_client_stats(self) -> Dict[str, Any]:
        """Get WebSocket client statistics."""
        active_clients = sum(1 for client in self.clients.values() if client.is_active)
        
        # Subscription stats
        subscription_stats = {}
        for sub_type, sub_ids in self.subscriptions_by_type.items():
            subscription_stats[sub_type.value] = len(sub_ids)
        
        return {
            "active_clients": active_clients,
            "total_clients": len(self.clients),
            "total_connections": self.total_connections,
            "total_messages": self.total_messages,
            "uptime_seconds": (datetime.utcnow() - self.start_time).total_seconds(),
            "subscriptions": subscription_stats
        }
    
    def _start_cleanup_task(self):
        """Start background cleanup task."""
        async def cleanup_task():
            while True:
                try:
                    await asyncio.sleep(30)  # Run every 30 seconds
                    await self._cleanup_inactive_clients()
                except Exception as e:
                    logger.error(f"Error in cleanup task: {e}")
        
        task = asyncio.create_task(cleanup_task())
        self.background_tasks.add(task)
        task.add_done_callback(self.background_tasks.discard)
    
    async def _cleanup_inactive_clients(self):
        """Clean up inactive clients."""
        inactive_clients = []
        
        for client_id, client in self.clients.items():
            if not client.is_alive():
                inactive_clients.append(client_id)
        
        for client_id in inactive_clients:
            await self.disconnect_client(client_id, "heartbeat_timeout")
        
        if inactive_clients:
            logger.info(f"Cleaned up {len(inactive_clients)} inactive WebSocket clients")


# Global WebSocket manager instance
websocket_manager = WebSocketManager()
"""WebSocket endpoints for real-time streaming."""

import asyncio
import json
import uuid
from typing import Dict, Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, HTTPException
from fastapi.responses import HTMLResponse

from .websocket_manager import websocket_manager, SubscriptionType
from .tenant_middleware import get_current_tenant_context, TenantContext
from ..utils.structured_logging import StructuredLogger

logger = StructuredLogger("quantum_ctl.websocket_endpoints")

router = APIRouter()


async def get_tenant_context_websocket(websocket: WebSocket) -> TenantContext:
    """Get tenant context for WebSocket connection."""
    # Try to get tenant from query parameters or headers
    tenant_id = websocket.query_params.get("tenant_id")
    tenant_slug = websocket.query_params.get("tenant_slug")
    
    if not tenant_id and not tenant_slug:
        # Try header
        tenant_header = websocket.headers.get("x-tenant-id") or websocket.headers.get("x-tenant-slug")
        if tenant_header:
            tenant_id = tenant_header
    
    if tenant_id or tenant_slug:
        # Get tenant manager and resolve tenant
        from ..core.tenant_manager import get_tenant_manager
        tenant_manager = get_tenant_manager()
        
        if tenant_id:
            tenant = await tenant_manager.get_tenant_by_id(tenant_id)
        else:
            tenant = await tenant_manager.get_tenant_by_slug(tenant_slug)
        
        if tenant:
            return tenant_manager.get_tenant_context(tenant)
    
    return None


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """General WebSocket endpoint for real-time updates."""
    client_id = str(uuid.uuid4())
    tenant_context = await get_tenant_context_websocket(websocket)
    
    try:
        client = await websocket_manager.connect_client(websocket, client_id, tenant_context)
        
        while True:
            try:
                # Receive message from client
                data = await websocket.receive_text()
                await websocket_manager.handle_message(client_id, data)
                
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"Error handling WebSocket message: {e}")
                await client.send_error(f"Message handling error: {str(e)}")
                
    except Exception as e:
        logger.error(f"WebSocket connection error: {e}")
    finally:
        await websocket_manager.disconnect_client(client_id)


@router.websocket("/ws/building/{building_id}")
async def building_websocket(websocket: WebSocket, building_id: str):
    """Building-specific WebSocket endpoint."""
    client_id = f"building_{building_id}_{uuid.uuid4()}"
    tenant_context = await get_tenant_context_websocket(websocket)
    
    try:
        client = await websocket_manager.connect_client(websocket, client_id, tenant_context)
        
        # Auto-subscribe to building updates
        from .websocket_manager import Subscription
        
        # Subscribe to building status
        building_subscription = Subscription(
            type=SubscriptionType.BUILDING_STATUS,
            filters={"building_id": building_id},
            client_id=client_id
        )
        sub_id = client.add_subscription(building_subscription)
        websocket_manager.subscriptions_by_type[SubscriptionType.BUILDING_STATUS].add(sub_id)
        
        # Subscribe to sensor data
        sensor_subscription = Subscription(
            type=SubscriptionType.SENSOR_DATA,
            filters={"building_id": building_id},
            client_id=client_id
        )
        sensor_sub_id = client.add_subscription(sensor_subscription)
        websocket_manager.subscriptions_by_type[SubscriptionType.SENSOR_DATA].add(sensor_sub_id)
        
        await client.send_status("auto_subscribed", {
            "building_id": building_id,
            "subscriptions": [sub_id, sensor_sub_id]
        })
        
        while True:
            try:
                data = await websocket.receive_text()
                await websocket_manager.handle_message(client_id, data)
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"Error in building WebSocket: {e}")
                break
                
    except Exception as e:
        logger.error(f"Building WebSocket connection error: {e}")
    finally:
        await websocket_manager.disconnect_client(client_id)


@router.websocket("/ws/optimization/{optimization_id}")
async def optimization_websocket(websocket: WebSocket, optimization_id: str):
    """Optimization-specific WebSocket endpoint for progress updates."""
    client_id = f"opt_{optimization_id}_{uuid.uuid4()}"
    tenant_context = await get_tenant_context_websocket(websocket)
    
    try:
        client = await websocket_manager.connect_client(websocket, client_id, tenant_context)
        
        # Auto-subscribe to optimization updates
        from .websocket_manager import Subscription
        
        optimization_subscription = Subscription(
            type=SubscriptionType.OPTIMIZATION_RESULTS,
            filters={"optimization_id": optimization_id},
            client_id=client_id
        )
        sub_id = client.add_subscription(optimization_subscription)
        websocket_manager.subscriptions_by_type[SubscriptionType.OPTIMIZATION_RESULTS].add(sub_id)
        
        await client.send_status("subscribed_to_optimization", {
            "optimization_id": optimization_id,
            "subscription_id": sub_id
        })
        
        while True:
            try:
                data = await websocket.receive_text()
                await websocket_manager.handle_message(client_id, data)
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"Error in optimization WebSocket: {e}")
                break
                
    except Exception as e:
        logger.error(f"Optimization WebSocket connection error: {e}")
    finally:
        await websocket_manager.disconnect_client(client_id)


@router.get("/ws/stats")
async def get_websocket_stats():
    """Get WebSocket statistics."""
    return websocket_manager.get_client_stats()


@router.get("/ws/test", response_class=HTMLResponse)
async def websocket_test_page():
    """Test page for WebSocket connections."""
    html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>WebSocket Test</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .container { max-width: 800px; margin: 0 auto; }
        .section { margin-bottom: 20px; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
        button { padding: 8px 16px; margin: 5px; cursor: pointer; }
        #messages { height: 300px; overflow-y: scroll; border: 1px solid #ccc; padding: 10px; background: #f9f9f9; }
        input, select { padding: 5px; margin: 5px; }
        .message { margin-bottom: 5px; padding: 3px; }
        .sent { color: blue; }
        .received { color: green; }
        .error { color: red; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Quantum HVAC WebSocket Test</h1>
        
        <div class="section">
            <h3>Connection</h3>
            <input type="text" id="tenant_id" placeholder="Tenant ID (optional)" />
            <button onclick="connect()">Connect</button>
            <button onclick="disconnect()">Disconnect</button>
            <span id="status">Disconnected</span>
        </div>
        
        <div class="section">
            <h3>Subscriptions</h3>
            <select id="subscription_type">
                <option value="building_status">Building Status</option>
                <option value="sensor_data">Sensor Data</option>
                <option value="optimization_results">Optimization Results</option>
                <option value="system_alerts">System Alerts</option>
                <option value="quantum_status">Quantum Status</option>
            </select>
            <input type="text" id="building_filter" placeholder="Building ID filter (optional)" />
            <button onclick="subscribe()">Subscribe</button>
            <button onclick="unsubscribe()">Unsubscribe</button>
        </div>
        
        <div class="section">
            <h3>Messages</h3>
            <div id="messages"></div>
            <button onclick="clearMessages()">Clear</button>
            <button onclick="sendHeartbeat()">Send Heartbeat</button>
        </div>
    </div>

    <script>
        let ws = null;
        let subscriptions = {};

        function connect() {
            const tenantId = document.getElementById('tenant_id').value;
            let wsUrl = `ws://localhost:8000/ws`;
            
            if (tenantId) {
                wsUrl += `?tenant_id=${tenantId}`;
            }
            
            ws = new WebSocket(wsUrl);
            
            ws.onopen = function(event) {
                document.getElementById('status').textContent = 'Connected';
                addMessage('Connected to WebSocket', 'received');
            };
            
            ws.onmessage = function(event) {
                const message = JSON.parse(event.data);
                addMessage(`Received: ${JSON.stringify(message, null, 2)}`, 'received');
                
                if (message.type === 'status' && message.data.subscription_id) {
                    subscriptions[message.data.subscription_id] = message.data.subscription_type;
                }
            };
            
            ws.onclose = function(event) {
                document.getElementById('status').textContent = 'Disconnected';
                addMessage('Disconnected from WebSocket', 'error');
            };
            
            ws.onerror = function(error) {
                addMessage(`Error: ${error}`, 'error');
            };
        }
        
        function disconnect() {
            if (ws) {
                ws.close();
                ws = null;
            }
        }
        
        function subscribe() {
            if (!ws) {
                addMessage('Not connected', 'error');
                return;
            }
            
            const subscriptionType = document.getElementById('subscription_type').value;
            const buildingFilter = document.getElementById('building_filter').value;
            
            const message = {
                type: 'subscribe',
                subscription_type: subscriptionType,
                filters: buildingFilter ? { building_id: buildingFilter } : {}
            };
            
            ws.send(JSON.stringify(message));
            addMessage(`Sent: ${JSON.stringify(message, null, 2)}`, 'sent');
        }
        
        function unsubscribe() {
            if (!ws) {
                addMessage('Not connected', 'error');
                return;
            }
            
            const subscriptionIds = Object.keys(subscriptions);
            if (subscriptionIds.length === 0) {
                addMessage('No active subscriptions', 'error');
                return;
            }
            
            const subscriptionId = subscriptionIds[0]; // Unsubscribe from first
            
            const message = {
                type: 'unsubscribe',
                subscription_id: subscriptionId
            };
            
            ws.send(JSON.stringify(message));
            addMessage(`Sent: ${JSON.stringify(message, null, 2)}`, 'sent');
            
            delete subscriptions[subscriptionId];
        }
        
        function sendHeartbeat() {
            if (!ws) {
                addMessage('Not connected', 'error');
                return;
            }
            
            const message = {
                type: 'heartbeat',
                timestamp: new Date().toISOString()
            };
            
            ws.send(JSON.stringify(message));
            addMessage(`Sent: ${JSON.stringify(message, null, 2)}`, 'sent');
        }
        
        function addMessage(text, className) {
            const messages = document.getElementById('messages');
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${className}`;
            messageDiv.innerHTML = text.replace(/\\n/g, '<br>');
            messages.appendChild(messageDiv);
            messages.scrollTop = messages.scrollHeight;
        }
        
        function clearMessages() {
            document.getElementById('messages').innerHTML = '';
        }
    </script>
</body>
</html>
    """
    return HTMLResponse(content=html_content)


# Utility functions for broadcasting data
async def broadcast_building_status(building_id: str, status_data: Dict[str, Any], tenant_id: str = None):
    """Broadcast building status update."""
    data = {
        "building_id": building_id,
        "timestamp": status_data.get("timestamp"),
        **status_data
    }
    
    if tenant_id:
        await websocket_manager.broadcast_to_tenant(tenant_id, SubscriptionType.BUILDING_STATUS, data)
    else:
        await websocket_manager.broadcast(SubscriptionType.BUILDING_STATUS, data)


async def broadcast_sensor_data(building_id: str, sensor_data: Dict[str, Any], tenant_id: str = None):
    """Broadcast sensor data update."""
    data = {
        "building_id": building_id,
        "timestamp": sensor_data.get("timestamp"),
        **sensor_data
    }
    
    if tenant_id:
        await websocket_manager.broadcast_to_tenant(tenant_id, SubscriptionType.SENSOR_DATA, data)
    else:
        await websocket_manager.broadcast(SubscriptionType.SENSOR_DATA, data)


async def broadcast_optimization_result(optimization_id: str, result_data: Dict[str, Any], tenant_id: str = None):
    """Broadcast optimization result."""
    data = {
        "optimization_id": optimization_id,
        "timestamp": result_data.get("timestamp"),
        **result_data
    }
    
    if tenant_id:
        await websocket_manager.broadcast_to_tenant(tenant_id, SubscriptionType.OPTIMIZATION_RESULTS, data)
    else:
        await websocket_manager.broadcast(SubscriptionType.OPTIMIZATION_RESULTS, data)


async def broadcast_system_alert(alert_data: Dict[str, Any], tenant_id: str = None):
    """Broadcast system alert."""
    data = {
        "alert_type": alert_data.get("type"),
        "severity": alert_data.get("severity"),
        "message": alert_data.get("message"),
        "timestamp": alert_data.get("timestamp"),
        **alert_data
    }
    
    if tenant_id:
        await websocket_manager.broadcast_to_tenant(tenant_id, SubscriptionType.SYSTEM_ALERTS, data)
    else:
        await websocket_manager.broadcast(SubscriptionType.SYSTEM_ALERTS, data)


async def broadcast_quantum_status(status_data: Dict[str, Any]):
    """Broadcast quantum system status."""
    await websocket_manager.broadcast(SubscriptionType.QUANTUM_STATUS, status_data)


# Register custom message handlers
def register_custom_handlers():
    """Register custom WebSocket message handlers."""
    
    async def handle_ping(client, message):
        """Handle ping message."""
        await client.send_message({
            "type": "pong",
            "timestamp": message.get("timestamp"),
            "server_time": "2025-01-01T00:00:00Z"  # Would be actual time
        })
    
    websocket_manager.register_message_handler("ping", handle_ping)


# Call this during app startup
register_custom_handlers()
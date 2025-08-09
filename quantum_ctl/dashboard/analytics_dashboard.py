"""
Real-time analytics dashboard for quantum HVAC control.

Provides web-based visualization and monitoring of system performance,
energy consumption, and quantum optimization metrics.
"""

import asyncio
import json
import logging
import time
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import warnings

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

try:
    import plotly.graph_objects as go
    import plotly.express as px
    import plotly.utils
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    warnings.warn("Plotly not available, dashboard charts will be limited")

import numpy as np
import pandas as pd

from ..utils.monitoring import get_advanced_metrics, get_alert_manager
from ..utils.logging_config import get_log_aggregator
from ..utils.security import get_auth_manager, require_authentication
from ..optimization.caching import get_global_cache


@dataclass
class DashboardMetrics:
    """Dashboard metrics data structure."""
    timestamp: float
    system_status: str
    active_alerts: int
    energy_consumption_kw: float
    avg_zone_temperature: float
    quantum_operations_per_hour: int
    cache_hit_rate: float
    optimization_success_rate: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class WebSocketManager:
    """Manages WebSocket connections for real-time updates."""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.logger = logging.getLogger("websocket_manager")
    
    async def connect(self, websocket: WebSocket):
        """Accept new WebSocket connection."""
        await websocket.accept()
        self.active_connections.append(websocket)
        self.logger.info(f"New WebSocket connection. Total: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            self.logger.info(f"WebSocket disconnected. Total: {len(self.active_connections)}")
    
    async def send_to_all(self, message: Dict[str, Any]):
        """Send message to all connected clients."""
        if not self.active_connections:
            return
        
        message_json = json.dumps(message, default=str)
        
        # Send to all connections, remove disconnected ones
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message_json)
            except Exception as e:
                self.logger.warning(f"Failed to send to WebSocket: {e}")
                disconnected.append(connection)
        
        # Clean up disconnected connections
        for connection in disconnected:
            self.disconnect(connection)
    
    async def broadcast_metrics(self, metrics: DashboardMetrics):
        """Broadcast metrics update to all clients."""
        await self.send_to_all({
            'type': 'metrics_update',
            'data': metrics.to_dict()
        })
    
    async def broadcast_alert(self, alert: Dict[str, Any]):
        """Broadcast alert to all clients."""
        await self.send_to_all({
            'type': 'alert',
            'data': alert
        })


class ChartGenerator:
    """Generates charts for the dashboard."""
    
    def __init__(self):
        self.logger = logging.getLogger("chart_generator")
    
    def create_energy_consumption_chart(
        self,
        time_series_data: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Create energy consumption time series chart."""
        if not PLOTLY_AVAILABLE or not time_series_data:
            return self._create_fallback_chart("Energy Consumption", time_series_data)
        
        df = pd.DataFrame(time_series_data)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['energy_kw'],
            mode='lines+markers',
            name='Energy Consumption (kW)',
            line=dict(color='#2E86AB', width=2),
            hovertemplate='Time: %{x}<br>Energy: %{y:.2f} kW<extra></extra>'
        ))
        
        fig.update_layout(
            title='Energy Consumption Over Time',
            xaxis_title='Time',
            yaxis_title='Energy (kW)',
            template='plotly_white',
            height=400
        )
        
        return json.loads(fig.to_json())
    
    def create_temperature_heatmap(
        self,
        zone_data: Dict[str, List[float]]
    ) -> Optional[Dict[str, Any]]:
        """Create temperature heatmap for all zones."""
        if not PLOTLY_AVAILABLE or not zone_data:
            return self._create_fallback_chart("Zone Temperatures", zone_data)
        
        zones = list(zone_data.keys())
        timestamps = list(range(len(list(zone_data.values())[0])))
        
        # Create temperature matrix
        temp_matrix = []
        for zone in zones:
            temp_matrix.append(zone_data[zone])
        
        fig = go.Figure(data=go.Heatmap(
            z=temp_matrix,
            x=timestamps,
            y=zones,
            colorscale='RdYlBu_r',
            hovertemplate='Zone: %{y}<br>Time: %{x}<br>Temp: %{z:.1f}°C<extra></extra>'
        ))
        
        fig.update_layout(
            title='Zone Temperature Heatmap',
            xaxis_title='Time Steps',
            yaxis_title='Zones',
            template='plotly_white',
            height=400
        )
        
        return json.loads(fig.to_json())
    
    def create_quantum_performance_chart(
        self,
        performance_data: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Create quantum optimization performance chart."""
        if not PLOTLY_AVAILABLE or not performance_data:
            return self._create_fallback_chart("Quantum Performance", performance_data)
        
        df = pd.DataFrame(performance_data)
        
        # Create dual-axis plot
        fig = go.Figure()
        
        # Success rate
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['success_rate'],
            mode='lines+markers',
            name='Success Rate (%)',
            yaxis='y',
            line=dict(color='#28a745', width=2)
        ))
        
        # Solve time
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['avg_solve_time'],
            mode='lines+markers',
            name='Avg Solve Time (s)',
            yaxis='y2',
            line=dict(color='#dc3545', width=2)
        ))
        
        fig.update_layout(
            title='Quantum Optimization Performance',
            xaxis_title='Time',
            yaxis=dict(title='Success Rate (%)', side='left'),
            yaxis2=dict(title='Solve Time (s)', side='right', overlaying='y'),
            template='plotly_white',
            height=400
        )
        
        return json.loads(fig.to_json())
    
    def create_cost_savings_chart(
        self,
        savings_data: Dict[str, float]
    ) -> Optional[Dict[str, Any]]:
        """Create cost savings pie chart."""
        if not PLOTLY_AVAILABLE or not savings_data:
            return self._create_fallback_chart("Cost Savings", savings_data)
        
        labels = list(savings_data.keys())
        values = list(savings_data.values())
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.4,
            hovertemplate='%{label}<br>Savings: $%{value:.2f}<extra></extra>'
        )])
        
        fig.update_layout(
            title='Cost Savings Breakdown',
            template='plotly_white',
            height=400
        )
        
        return json.loads(fig.to_json())
    
    def _create_fallback_chart(
        self,
        title: str,
        data: Any
    ) -> Dict[str, Any]:
        """Create fallback chart when Plotly is not available."""
        return {
            'type': 'fallback',
            'title': title,
            'message': 'Chart visualization requires Plotly library',
            'data_available': bool(data)
        }


class DashboardAPI:
    """FastAPI dashboard application."""
    
    def __init__(self):
        self.app = FastAPI(title="Quantum HVAC Control Dashboard", version="1.0.0")
        self.websocket_manager = WebSocketManager()
        self.chart_generator = ChartGenerator()
        self.logger = logging.getLogger("dashboard_api")
        
        # Metrics storage
        self.metrics_history: List[DashboardMetrics] = []
        self.max_history = 1000
        
        # Setup CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Setup routes
        self._setup_routes()
        
        # Start background tasks
        self._background_task: Optional[asyncio.Task] = None
    
    def _setup_routes(self):
        """Setup API routes."""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def dashboard_home():
            """Serve dashboard HTML."""
            return self._get_dashboard_html()
        
        @self.app.get("/api/status")
        async def get_system_status():
            """Get current system status."""
            try:
                # Collect metrics from various sources
                advanced_metrics = get_advanced_metrics()
                alert_manager = get_alert_manager()
                cache = get_global_cache()
                
                recent_metrics = advanced_metrics.get_recent_metrics("quantum_operation", 60)
                active_alerts = alert_manager.get_active_alerts()
                cache_stats = cache.get_cache_stats()
                
                status = {
                    "system_healthy": len(active_alerts) == 0,
                    "active_alerts": len(active_alerts),
                    "recent_operations": len(recent_metrics),
                    "cache_hit_rate": cache_stats.get("hit_rate", 0.0),
                    "timestamp": time.time()
                }
                
                return JSONResponse(content=status)
                
            except Exception as e:
                self.logger.error(f"Error getting system status: {e}")
                raise HTTPException(status_code=500, detail="Internal server error")
        
        @self.app.get("/api/metrics/energy")
        async def get_energy_metrics():
            """Get energy consumption metrics."""
            try:
                # Get recent energy data from metrics
                advanced_metrics = get_advanced_metrics()
                energy_metrics = advanced_metrics.get_recent_metrics("building_metrics", 60)
                
                time_series = []
                for metric in energy_metrics:
                    time_series.append({
                        "timestamp": datetime.fromtimestamp(metric['timestamp']),
                        "energy_kw": metric.get('power_kw', 0),
                        "building": metric.get('building', 'unknown'),
                        "zone": metric.get('zone', 'unknown')
                    })
                
                chart = self.chart_generator.create_energy_consumption_chart(time_series)
                
                return JSONResponse(content={
                    "data": time_series,
                    "chart": chart
                })
                
            except Exception as e:
                self.logger.error(f"Error getting energy metrics: {e}")
                raise HTTPException(status_code=500, detail="Internal server error")
        
        @self.app.get("/api/metrics/temperature")
        async def get_temperature_metrics():
            """Get temperature metrics."""
            try:
                advanced_metrics = get_advanced_metrics()
                temp_metrics = advanced_metrics.get_recent_metrics("building_metrics", 60)
                
                # Group by zone
                zone_data = {}
                for metric in temp_metrics:
                    zone = metric.get('zone', 'unknown')
                    temp = metric.get('temperature', 22.0)
                    
                    if zone not in zone_data:
                        zone_data[zone] = []
                    zone_data[zone].append(temp)
                
                chart = self.chart_generator.create_temperature_heatmap(zone_data)
                
                return JSONResponse(content={
                    "zone_data": zone_data,
                    "chart": chart
                })
                
            except Exception as e:
                self.logger.error(f"Error getting temperature metrics: {e}")
                raise HTTPException(status_code=500, detail="Internal server error")
        
        @self.app.get("/api/metrics/quantum")
        async def get_quantum_metrics():
            """Get quantum optimization metrics."""
            try:
                advanced_metrics = get_advanced_metrics()
                quantum_metrics = advanced_metrics.get_recent_metrics("quantum_operation", 120)
                
                # Aggregate performance data
                performance_data = []
                window_size = 10  # 10-minute windows
                
                current_window = []
                current_time = 0
                
                for metric in quantum_metrics:
                    if not current_time:
                        current_time = metric['timestamp'] // (window_size * 60) * (window_size * 60)
                    
                    if metric['timestamp'] < current_time + (window_size * 60):
                        current_window.append(metric)
                    else:
                        if current_window:
                            success_count = sum(1 for m in current_window if m.get('success', False))
                            success_rate = (success_count / len(current_window)) * 100
                            avg_duration = np.mean([m.get('duration', 0) for m in current_window])
                            
                            performance_data.append({
                                "timestamp": datetime.fromtimestamp(current_time),
                                "success_rate": success_rate,
                                "avg_solve_time": avg_duration / 1000,  # Convert to seconds
                                "total_operations": len(current_window)
                            })
                        
                        current_window = [metric]
                        current_time = metric['timestamp'] // (window_size * 60) * (window_size * 60)
                
                chart = self.chart_generator.create_quantum_performance_chart(performance_data)
                
                return JSONResponse(content={
                    "performance_data": performance_data,
                    "chart": chart
                })
                
            except Exception as e:
                self.logger.error(f"Error getting quantum metrics: {e}")
                raise HTTPException(status_code=500, detail="Internal server error")
        
        @self.app.get("/api/alerts")
        async def get_alerts():
            """Get current alerts."""
            try:
                alert_manager = get_alert_manager()
                
                active_alerts = alert_manager.get_active_alerts()
                alert_summary = alert_manager.get_alert_summary()
                
                return JSONResponse(content={
                    "active_alerts": active_alerts,
                    "summary": alert_summary
                })
                
            except Exception as e:
                self.logger.error(f"Error getting alerts: {e}")
                raise HTTPException(status_code=500, detail="Internal server error")
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time updates."""
            await self.websocket_manager.connect(websocket)
            try:
                while True:
                    # Keep connection alive
                    await websocket.receive_text()
            except WebSocketDisconnect:
                self.websocket_manager.disconnect(websocket)
    
    def _get_dashboard_html(self) -> str:
        """Generate dashboard HTML."""
        return """
<!DOCTYPE html>
<html>
<head>
    <title>Quantum HVAC Control Dashboard</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <style>
        .metric-card {
            border-left: 4px solid #007bff;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .alert-card {
            border-left: 4px solid #dc3545;
        }
        .success-card {
            border-left: 4px solid #28a745;
        }
        .chart-container {
            min-height: 400px;
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
    </style>
</head>
<body class="bg-light">
    <nav class="navbar navbar-dark bg-dark">
        <div class="container-fluid">
            <span class="navbar-brand mb-0 h1">🔬 Quantum HVAC Control Dashboard</span>
            <span class="navbar-text" id="status-indicator">
                <span class="badge bg-success">System Healthy</span>
            </span>
        </div>
    </nav>

    <div class="container-fluid mt-4">
        <!-- Metrics Row -->
        <div class="row mb-4">
            <div class="col-md-3">
                <div class="card metric-card">
                    <div class="card-body">
                        <h5 class="card-title">Energy Consumption</h5>
                        <h2 class="text-primary" id="energy-value">-- kW</h2>
                        <small class="text-muted">Current total consumption</small>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card metric-card">
                    <div class="card-body">
                        <h5 class="card-title">Avg Temperature</h5>
                        <h2 class="text-info" id="temp-value">-- °C</h2>
                        <small class="text-muted">Across all zones</small>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card success-card">
                    <div class="card-body">
                        <h5 class="card-title">Quantum Success Rate</h5>
                        <h2 class="text-success" id="success-rate">--%</h2>
                        <small class="text-muted">Last hour</small>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card alert-card">
                    <div class="card-body">
                        <h5 class="card-title">Active Alerts</h5>
                        <h2 class="text-danger" id="alert-count">--</h2>
                        <small class="text-muted">Requiring attention</small>
                    </div>
                </div>
            </div>
        </div>

        <!-- Charts Row -->
        <div class="row">
            <div class="col-md-6 mb-4">
                <div class="chart-container">
                    <div id="energy-chart"></div>
                </div>
            </div>
            <div class="col-md-6 mb-4">
                <div class="chart-container">
                    <div id="temperature-chart"></div>
                </div>
            </div>
        </div>

        <div class="row">
            <div class="col-md-6 mb-4">
                <div class="chart-container">
                    <div id="quantum-chart"></div>
                </div>
            </div>
            <div class="col-md-6 mb-4">
                <div class="card">
                    <div class="card-header">
                        <h5>Recent Alerts</h5>
                    </div>
                    <div class="card-body">
                        <div id="alerts-list">
                            <p class="text-muted">No recent alerts</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // WebSocket connection
        const ws = new WebSocket(`ws://${window.location.host}/ws`);
        
        ws.onopen = function(event) {
            console.log('WebSocket connected');
            updateStatus('Connected', 'success');
        };
        
        ws.onmessage = function(event) {
            const message = JSON.parse(event.data);
            
            if (message.type === 'metrics_update') {
                updateMetrics(message.data);
            } else if (message.type === 'alert') {
                showAlert(message.data);
            }
        };
        
        ws.onclose = function(event) {
            console.log('WebSocket disconnected');
            updateStatus('Disconnected', 'danger');
        };
        
        // Update functions
        function updateMetrics(data) {
            document.getElementById('energy-value').textContent = `${data.energy_consumption_kw.toFixed(1)} kW`;
            document.getElementById('temp-value').textContent = `${data.avg_zone_temperature.toFixed(1)} °C`;
            document.getElementById('success-rate').textContent = `${(data.optimization_success_rate * 100).toFixed(0)}%`;
            document.getElementById('alert-count').textContent = data.active_alerts;
        }
        
        function updateStatus(status, type) {
            const indicator = document.getElementById('status-indicator');
            indicator.innerHTML = `<span class="badge bg-${type}">${status}</span>`;
        }
        
        function showAlert(alert) {
            const alertsList = document.getElementById('alerts-list');
            const alertDiv = document.createElement('div');
            alertDiv.className = `alert alert-${alert.severity === 'critical' ? 'danger' : 'warning'} alert-dismissible fade show`;
            alertDiv.innerHTML = `
                <strong>${alert.severity.toUpperCase()}</strong> ${alert.message}
                <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
            `;
            alertsList.appendChild(alertDiv);
        }
        
        // Load charts
        async function loadCharts() {
            try {
                // Energy chart
                const energyResponse = await fetch('/api/metrics/energy');
                const energyData = await energyResponse.json();
                if (energyData.chart && energyData.chart.type !== 'fallback') {
                    Plotly.newPlot('energy-chart', energyData.chart.data, energyData.chart.layout);
                }
                
                // Temperature chart
                const tempResponse = await fetch('/api/metrics/temperature');
                const tempData = await tempResponse.json();
                if (tempData.chart && tempData.chart.type !== 'fallback') {
                    Plotly.newPlot('temperature-chart', tempData.chart.data, tempData.chart.layout);
                }
                
                // Quantum performance chart
                const quantumResponse = await fetch('/api/metrics/quantum');
                const quantumData = await quantumResponse.json();
                if (quantumData.chart && quantumData.chart.type !== 'fallback') {
                    Plotly.newPlot('quantum-chart', quantumData.chart.data, quantumData.chart.layout);
                }
                
            } catch (error) {
                console.error('Error loading charts:', error);
            }
        }
        
        // Load initial data
        loadCharts();
        
        // Refresh charts every 30 seconds
        setInterval(loadCharts, 30000);
        
        // Send heartbeat every 10 seconds
        setInterval(() => {
            if (ws.readyState === WebSocket.OPEN) {
                ws.send('heartbeat');
            }
        }, 10000);
    </script>
</body>
</html>
        """
    
    async def start_background_tasks(self):
        """Start background monitoring tasks."""
        self._background_task = asyncio.create_task(self._metrics_collection_loop())
    
    async def stop_background_tasks(self):
        """Stop background tasks."""
        if self._background_task and not self._background_task.done():
            self._background_task.cancel()
            try:
                await self._background_task
            except asyncio.CancelledError:
                pass
    
    async def _metrics_collection_loop(self):
        """Background task for collecting and broadcasting metrics."""
        while True:
            try:
                # Collect current metrics
                metrics = await self._collect_dashboard_metrics()
                
                # Store in history
                self.metrics_history.append(metrics)
                if len(self.metrics_history) > self.max_history:
                    self.metrics_history.pop(0)
                
                # Broadcast to WebSocket clients
                await self.websocket_manager.broadcast_metrics(metrics)
                
                # Check for new alerts
                alert_manager = get_alert_manager()
                active_alerts = alert_manager.get_active_alerts()
                
                for alert in active_alerts:
                    await self.websocket_manager.broadcast_alert(alert)
                
                # Wait before next collection
                await asyncio.sleep(10)  # Collect every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Error in metrics collection loop: {e}")
                await asyncio.sleep(30)  # Wait longer on error
    
    async def _collect_dashboard_metrics(self) -> DashboardMetrics:
        """Collect current dashboard metrics."""
        try:
            advanced_metrics = get_advanced_metrics()
            alert_manager = get_alert_manager()
            cache = get_global_cache()
            
            # Get recent metrics
            recent_building_metrics = advanced_metrics.get_recent_metrics("building_metrics", 5)
            recent_quantum_metrics = advanced_metrics.get_recent_metrics("quantum_operation", 60)
            active_alerts = alert_manager.get_active_alerts()
            cache_stats = cache.get_cache_stats()
            
            # Calculate aggregated values
            total_energy = sum(m.get('power_kw', 0) for m in recent_building_metrics)
            avg_temp = np.mean([m.get('temperature', 22.0) for m in recent_building_metrics]) if recent_building_metrics else 22.0
            
            quantum_operations_per_hour = len([m for m in recent_quantum_metrics if m.get('timestamp', 0) > time.time() - 3600])
            successful_operations = len([m for m in recent_quantum_metrics if m.get('success', False)])
            success_rate = successful_operations / len(recent_quantum_metrics) if recent_quantum_metrics else 1.0
            
            # Determine system status
            if len(active_alerts) == 0:
                system_status = "healthy"
            elif any(alert.get('severity') == 'critical' for alert in active_alerts):
                system_status = "critical"
            else:
                system_status = "warning"
            
            return DashboardMetrics(
                timestamp=time.time(),
                system_status=system_status,
                active_alerts=len(active_alerts),
                energy_consumption_kw=total_energy,
                avg_zone_temperature=float(avg_temp),
                quantum_operations_per_hour=quantum_operations_per_hour,
                cache_hit_rate=cache_stats.get('hit_rate', 0.0),
                optimization_success_rate=success_rate
            )
            
        except Exception as e:
            self.logger.error(f"Error collecting dashboard metrics: {e}")
            
            # Return default metrics on error
            return DashboardMetrics(
                timestamp=time.time(),
                system_status="unknown",
                active_alerts=0,
                energy_consumption_kw=0.0,
                avg_zone_temperature=22.0,
                quantum_operations_per_hour=0,
                cache_hit_rate=0.0,
                optimization_success_rate=0.0
            )


class DashboardServer:
    """Dashboard server manager."""
    
    def __init__(self, host: str = "0.0.0.0", port: int = 8080):
        self.host = host
        self.port = port
        self.dashboard_api = DashboardAPI()
        self.logger = logging.getLogger("dashboard_server")
    
    async def start(self):
        """Start the dashboard server."""
        self.logger.info(f"Starting dashboard server on {self.host}:{self.port}")
        
        # Start background tasks
        await self.dashboard_api.start_background_tasks()
        
        # Start FastAPI server
        config = uvicorn.Config(
            app=self.dashboard_api.app,
            host=self.host,
            port=self.port,
            log_level="info"
        )
        
        server = uvicorn.Server(config)
        await server.serve()
    
    async def stop(self):
        """Stop the dashboard server."""
        await self.dashboard_api.stop_background_tasks()
        self.logger.info("Dashboard server stopped")


# Global dashboard instance
_global_dashboard = DashboardServer()


def get_dashboard_server() -> DashboardServer:
    """Get global dashboard server instance."""
    return _global_dashboard


async def run_dashboard(host: str = "0.0.0.0", port: int = 8080):
    """Run the dashboard server."""
    dashboard = DashboardServer(host, port)
    await dashboard.start()
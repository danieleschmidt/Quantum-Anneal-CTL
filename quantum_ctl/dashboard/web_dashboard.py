"""Simple web-based monitoring dashboard for quantum HVAC control system."""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

from fastapi import FastAPI, Request, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.cors import CORSMiddleware
import asyncio

from ..database.manager import get_database_manager
from ..database.storage import TimeSeriesStorage, ResultStorage, MetricsStorage
from ..utils.config import get_config

logger = logging.getLogger(__name__)


class WebDashboard:
    """Web-based monitoring dashboard."""
    
    def __init__(self):
        self.app = FastAPI(title="Quantum HVAC Dashboard", version="1.0.0")
        self.setup_middleware()
        self.setup_routes()
        self.websocket_connections: List[WebSocket] = []
        self.db_manager = get_database_manager()
        
        # Initialize storage interfaces
        session_factory = self.db_manager.get_async_session
        self.timeseries_storage = TimeSeriesStorage(session_factory)
        self.results_storage = ResultStorage(session_factory)
        self.metrics_storage = MetricsStorage(session_factory)
    
    def setup_middleware(self):
        """Setup FastAPI middleware."""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def setup_routes(self):
        """Setup API routes."""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def dashboard_home(request: Request):
            """Serve main dashboard page."""
            return HTMLResponse(self.get_dashboard_html())
        
        @self.app.get("/api/buildings")
        async def get_buildings():
            """Get list of buildings."""
            # For now, return mock data - could be expanded to query database
            return {
                "buildings": [
                    {
                        "id": "building_001",
                        "name": "Main Office Building",
                        "status": "online",
                        "zones": 12,
                        "last_optimization": datetime.utcnow().isoformat()
                    },
                    {
                        "id": "building_002", 
                        "name": "Research Lab",
                        "status": "online",
                        "zones": 8,
                        "last_optimization": datetime.utcnow().isoformat()
                    }
                ]
            }
        
        @self.app.get("/api/buildings/{building_id}/status")
        async def get_building_status(building_id: str):
            """Get current building status."""
            try:
                # Get latest temperature data
                temp_data = await self.timeseries_storage.get_latest(
                    building_id=building_id,
                    data_type="temperature",
                    limit=10
                )
                
                # Get latest optimization results
                opt_results = await self.results_storage.get_latest_results(
                    building_id=building_id,
                    limit=5
                )
                
                # Get performance metrics
                energy_metrics = await self.metrics_storage.get_metrics_summary(
                    building_id=building_id,
                    metric_type="energy",
                    start_time=datetime.utcnow() - timedelta(hours=24)
                )
                
                return {
                    "building_id": building_id,
                    "timestamp": datetime.utcnow().isoformat(),
                    "temperature_data": temp_data,
                    "optimization_results": opt_results,
                    "energy_metrics": energy_metrics,
                    "status": "online"
                }
            except Exception as e:
                logger.error(f"Error getting building status: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/buildings/{building_id}/timeseries")
        async def get_timeseries_data(
            building_id: str,
            data_type: str = "temperature",
            hours: int = 24
        ):
            """Get time series data for building."""
            try:
                start_time = datetime.utcnow() - timedelta(hours=hours)
                end_time = datetime.utcnow()
                
                data = await self.timeseries_storage.get_time_range(
                    building_id=building_id,
                    data_type=data_type,
                    start_time=start_time,
                    end_time=end_time
                )
                
                return {
                    "building_id": building_id,
                    "data_type": data_type,
                    "start_time": start_time.isoformat(),
                    "end_time": end_time.isoformat(),
                    "data": data
                }
            except Exception as e:
                logger.error(f"Error getting timeseries data: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/quantum/status")
        async def get_quantum_status():
            """Get quantum solver status."""
            # Mock quantum status - could be expanded with real D-Wave monitoring
            return {
                "status": "online",
                "qpu_name": "Advantage_system6.4",
                "queue_length": 3,
                "avg_computation_time_ms": 125.5,
                "success_rate": 0.98,
                "total_problems_today": 247,
                "last_access": datetime.utcnow().isoformat()
            }
        
        @self.app.websocket("/ws/{building_id}")
        async def websocket_endpoint(websocket: WebSocket, building_id: str):
            """WebSocket endpoint for real-time updates."""
            await websocket.accept()
            self.websocket_connections.append(websocket)
            
            try:
                while True:
                    # Send periodic updates
                    status_data = await self.get_building_status_internal(building_id)
                    await websocket.send_json(status_data)
                    await asyncio.sleep(5)  # Update every 5 seconds
                    
            except WebSocketDisconnect:
                self.websocket_connections.remove(websocket)
                logger.info(f"WebSocket disconnected for building {building_id}")
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                if websocket in self.websocket_connections:
                    self.websocket_connections.remove(websocket)
    
    async def get_building_status_internal(self, building_id: str) -> Dict[str, Any]:
        """Internal method to get building status for WebSocket updates."""
        try:
            # Get basic status - simplified for real-time updates
            return {
                "building_id": building_id,
                "timestamp": datetime.utcnow().isoformat(),
                "status": "online",
                "current_temperature": 22.5 + (hash(str(datetime.utcnow().minute)) % 5),
                "energy_usage_kw": 145.2 + (hash(str(datetime.utcnow().second)) % 20),
                "optimization_active": True
            }
        except Exception as e:
            logger.error(f"Error getting building status: {e}")
            return {
                "building_id": building_id,
                "timestamp": datetime.utcnow().isoformat(),
                "status": "error",
                "error": str(e)
            }
    
    def get_dashboard_html(self) -> str:
        """Generate dashboard HTML."""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Quantum HVAC Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: #fff;
            min-height: 100vh;
        }
        
        .header {
            background: rgba(0,0,0,0.2);
            padding: 1rem 2rem;
            backdrop-filter: blur(10px);
        }
        
        .header h1 {
            font-size: 2rem;
            font-weight: 300;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .quantum-icon {
            width: 2rem;
            height: 2rem;
            background: linear-gradient(45deg, #00ff88, #0088ff);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            font-size: 1rem;
        }
        
        .dashboard {
            padding: 2rem;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 2rem;
            max-width: 1400px;
            margin: 0 auto;
        }
        
        .card {
            background: rgba(255,255,255,0.1);
            border-radius: 12px;
            padding: 1.5rem;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
        }
        
        .card h3 {
            margin-bottom: 1rem;
            color: #00ff88;
            font-size: 1.2rem;
        }
        
        .metric {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.8rem;
            padding: 0.5rem 0;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }
        
        .metric:last-child { border-bottom: none; }
        
        .metric-label {
            color: rgba(255,255,255,0.8);
            font-size: 0.9rem;
        }
        
        .metric-value {
            font-weight: 600;
            font-size: 1.1rem;
        }
        
        .status {
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
            font-size: 0.8rem;
            font-weight: 500;
        }
        
        .status.online {
            background: rgba(0,255,136,0.2);
            color: #00ff88;
        }
        
        .status.offline {
            background: rgba(255,0,68,0.2);
            color: #ff0044;
        }
        
        .chart-container {
            height: 300px;
            margin-top: 1rem;
        }
        
        .building-list {
            display: grid;
            gap: 1rem;
        }
        
        .building-item {
            background: rgba(255,255,255,0.05);
            border-radius: 8px;
            padding: 1rem;
            border-left: 4px solid #00ff88;
        }
        
        .building-name {
            font-weight: 600;
            margin-bottom: 0.5rem;
        }
        
        .building-stats {
            display: flex;
            justify-content: space-between;
            font-size: 0.9rem;
            color: rgba(255,255,255,0.7);
        }
    </style>
</head>
<body>
    <header class="header">
        <h1>
            <div class="quantum-icon">Q</div>
            Quantum HVAC Control Dashboard
        </h1>
    </header>

    <div class="dashboard">
        <!-- System Overview -->
        <div class="card">
            <h3>System Overview</h3>
            <div class="metric">
                <span class="metric-label">System Status</span>
                <span class="status online" id="system-status">Online</span>
            </div>
            <div class="metric">
                <span class="metric-label">Active Buildings</span>
                <span class="metric-value" id="active-buildings">2</span>
            </div>
            <div class="metric">
                <span class="metric-label">Total Energy (kW)</span>
                <span class="metric-value" id="total-energy">287.5</span>
            </div>
            <div class="metric">
                <span class="metric-label">Optimization Cycles Today</span>
                <span class="metric-value" id="opt-cycles">247</span>
            </div>
        </div>

        <!-- Quantum Status -->
        <div class="card">
            <h3>Quantum Processor</h3>
            <div class="metric">
                <span class="metric-label">QPU Status</span>
                <span class="status online" id="qpu-status">Online</span>
            </div>
            <div class="metric">
                <span class="metric-label">Queue Length</span>
                <span class="metric-value" id="queue-length">3</span>
            </div>
            <div class="metric">
                <span class="metric-label">Avg Computation (ms)</span>
                <span class="metric-value" id="avg-computation">125.5</span>
            </div>
            <div class="metric">
                <span class="metric-label">Success Rate</span>
                <span class="metric-value" id="success-rate">98%</span>
            </div>
        </div>

        <!-- Buildings List -->
        <div class="card">
            <h3>Buildings</h3>
            <div class="building-list" id="buildings-list">
                <div class="building-item">
                    <div class="building-name">Main Office Building</div>
                    <div class="building-stats">
                        <span>12 zones</span>
                        <span class="status online">Online</span>
                    </div>
                </div>
                <div class="building-item">
                    <div class="building-name">Research Lab</div>
                    <div class="building-stats">
                        <span>8 zones</span>
                        <span class="status online">Online</span>
                    </div>
                </div>
            </div>
        </div>

        <!-- Temperature Chart -->
        <div class="card" style="grid-column: span 2;">
            <h3>Temperature Trends</h3>
            <div class="chart-container" id="temperature-chart"></div>
        </div>

        <!-- Energy Chart -->
        <div class="card" style="grid-column: span 2;">
            <h3>Energy Usage</h3>
            <div class="chart-container" id="energy-chart"></div>
        </div>
    </div>

    <script>
        // Initialize dashboard
        class QuantumDashboard {
            constructor() {
                this.initializeCharts();
                this.startDataUpdate();
            }

            initializeCharts() {
                // Temperature chart
                const tempTrace = {
                    x: this.generateTimeRange(24),
                    y: this.generateTemperatureData(24),
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: 'Average Temperature',
                    line: {color: '#00ff88', width: 2},
                    marker: {color: '#00ff88', size: 4}
                };

                const tempLayout = {
                    paper_bgcolor: 'rgba(0,0,0,0)',
                    plot_bgcolor: 'rgba(0,0,0,0)',
                    font: {color: '#fff', size: 12},
                    xaxis: {
                        gridcolor: 'rgba(255,255,255,0.1)',
                        zerolinecolor: 'rgba(255,255,255,0.1)'
                    },
                    yaxis: {
                        gridcolor: 'rgba(255,255,255,0.1)',
                        zerolinecolor: 'rgba(255,255,255,0.1)',
                        title: 'Temperature (°C)'
                    },
                    margin: {l: 50, r: 20, t: 20, b: 40}
                };

                Plotly.newPlot('temperature-chart', [tempTrace], tempLayout, {responsive: true});

                // Energy chart
                const energyTrace = {
                    x: this.generateTimeRange(24),
                    y: this.generateEnergyData(24),
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: 'Energy Usage',
                    line: {color: '#0088ff', width: 2},
                    marker: {color: '#0088ff', size: 4}
                };

                const energyLayout = {
                    paper_bgcolor: 'rgba(0,0,0,0)',
                    plot_bgcolor: 'rgba(0,0,0,0)',
                    font: {color: '#fff', size: 12},
                    xaxis: {
                        gridcolor: 'rgba(255,255,255,0.1)',
                        zerolinecolor: 'rgba(255,255,255,0.1)'
                    },
                    yaxis: {
                        gridcolor: 'rgba(255,255,255,0.1)',
                        zerolinecolor: 'rgba(255,255,255,0.1)',
                        title: 'Energy (kW)'
                    },
                    margin: {l: 50, r: 20, t: 20, b: 40}
                };

                Plotly.newPlot('energy-chart', [energyTrace], energyLayout, {responsive: true});
            }

            generateTimeRange(hours) {
                const times = [];
                const now = new Date();
                for (let i = hours; i >= 0; i--) {
                    const time = new Date(now.getTime() - i * 60 * 60 * 1000);
                    times.push(time.toISOString());
                }
                return times;
            }

            generateTemperatureData(points) {
                const data = [];
                for (let i = 0; i < points + 1; i++) {
                    data.push(20 + Math.sin(i * 0.5) * 3 + Math.random() * 2);
                }
                return data;
            }

            generateEnergyData(points) {
                const data = [];
                for (let i = 0; i < points + 1; i++) {
                    data.push(150 + Math.sin(i * 0.3) * 50 + Math.random() * 30);
                }
                return data;
            }

            async startDataUpdate() {
                // Update every 10 seconds
                setInterval(async () => {
                    await this.updateMetrics();
                    await this.updateCharts();
                }, 10000);
            }

            async updateMetrics() {
                try {
                    // In a real implementation, these would be API calls
                    document.getElementById('total-energy').textContent = 
                        (280 + Math.random() * 20).toFixed(1);
                    document.getElementById('avg-computation').textContent = 
                        (120 + Math.random() * 10).toFixed(1);
                } catch (error) {
                    console.error('Error updating metrics:', error);
                }
            }

            async updateCharts() {
                // Simulate real-time data updates
                const newTemp = 20 + Math.sin(Date.now() * 0.001) * 3 + Math.random() * 2;
                const newEnergy = 150 + Math.sin(Date.now() * 0.0003) * 50 + Math.random() * 30;
                
                Plotly.extendTraces('temperature-chart', {
                    x: [[new Date()]],
                    y: [[newTemp]]
                }, [0]);

                Plotly.extendTraces('energy-chart', {
                    x: [[new Date()]],
                    y: [[newEnergy]]
                }, [0]);
            }
        }

        // Start dashboard when page loads
        document.addEventListener('DOMContentLoaded', () => {
            new QuantumDashboard();
        });
    </script>
</body>
</html>
        """

# Create dashboard instance
dashboard = WebDashboard()
app = dashboard.app
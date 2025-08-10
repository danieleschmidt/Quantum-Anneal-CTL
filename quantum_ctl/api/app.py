"""Main FastAPI application."""

import logging
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi

from .models import *
from .auth import (
    authenticate_user, create_access_token, get_current_user,
    require_admin, require_operator, require_viewer,
    log_security_event, check_rate_limit
)
from ..database.manager import get_database_manager
from ..database.storage import TimeSeriesStorage, ResultStorage, MetricsStorage, ConfigurationStorage
from ..core.controller import HVACController
from ..integration.mock_bms import MockBMSConnector, MockWeatherService
from ..utils.config import get_config

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    
    app = FastAPI(
        title="Quantum HVAC Control API",
        description="REST API for quantum annealing-based HVAC optimization system",
        version="1.0.0",
        contact={
            "name": "Terragon Labs",
            "url": "https://terragonlabs.com",
            "email": "support@terragonlabs.com"
        },
        license_info={
            "name": "Apache 2.0",
            "url": "https://www.apache.org/licenses/LICENSE-2.0.html"
        }
    )
    
    # Add middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure properly in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # Database and storage
    db_manager = get_database_manager()
    
    # Storage interfaces
    session_factory = db_manager.get_async_session
    timeseries_storage = TimeSeriesStorage(session_factory)
    results_storage = ResultStorage(session_factory)
    metrics_storage = MetricsStorage(session_factory)
    config_storage = ConfigurationStorage(session_factory)
    
    # Mock services for development
    mock_bms = MockBMSConnector()
    mock_weather = MockWeatherService()
    
    # Exception handlers
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """Handle HTTP exceptions."""
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(
                error=f"HTTP_{exc.status_code}",
                message=exc.detail,
                details={"url": str(request.url)}
            ).dict()
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle general exceptions."""
        logger.error(f"Unhandled exception: {exc}")
        logger.error(traceback.format_exc())
        
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error="INTERNAL_SERVER_ERROR",
                message="An internal server error occurred",
                details={"url": str(request.url)}
            ).dict()
        )
    
    # Authentication endpoints
    @app.post("/auth/login", response_model=TokenResponse)
    async def login(login_request: LoginRequest):
        """Authenticate user and return access token."""
        user = authenticate_user(login_request.username, login_request.password)
        if not user:
            log_security_event("login_failed", login_request.username)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        access_token_expires = timedelta(minutes=60 * 24)  # 24 hours
        access_token = create_access_token(
            data={"sub": user["username"]},
            expires_delta=access_token_expires
        )
        
        log_security_event("login_success", user["username"])
        
        return TokenResponse(
            access_token=access_token,
            token_type="bearer",
            expires_in=int(access_token_expires.total_seconds())
        )
    
    @app.get("/auth/me", response_model=UserModel)
    async def get_current_user_info(current_user: UserModel = Depends(get_current_user)):
        """Get current user information."""
        return current_user
    
    # System status endpoints
    @app.get("/status", response_model=SystemStatusResponse)
    async def get_system_status():
        """Get overall system status."""
        # Check database health
        db_healthy = await db_manager.health_check()
        
        # Check quantum status (mock for now)
        quantum_status = {
            "status": "online",
            "qpu_name": "Advantage_system6.4",
            "queue_length": 3
        }
        
        return SystemStatusResponse(
            status="healthy" if db_healthy else "degraded",
            quantum_status=quantum_status,
            database_status={"healthy": db_healthy},
            active_buildings=2,  # Mock data
            active_optimizations=1,
            total_energy_kw=287.5,
            uptime_seconds=86400,
            version="1.0.0"
        )
    
    @app.get("/quantum/status", response_model=QuantumStatusResponse)
    async def get_quantum_status():
        """Get quantum processor status."""
        # Mock quantum status - could integrate with real D-Wave monitoring
        return QuantumStatusResponse(
            status="online",
            qpu_name="Advantage_system6.4",
            queue_length=3,
            avg_computation_time_ms=125.5,
            success_rate=0.98,
            total_problems_today=247,
            estimated_cost_today_usd=15.75,
            last_access=datetime.utcnow()
        )
    
    # Building endpoints
    @app.get("/buildings", response_model=List[BuildingStatusResponse])
    async def list_buildings(current_user: UserModel = Depends(require_viewer)):
        """List all buildings with their status."""
        # Mock building list - could query from database
        buildings = [
            BuildingStatusResponse(
                building_id="building_001",
                name="Main Office Building", 
                status="online",
                zones=12,
                current_state={"avg_temp": 22.5, "energy_kw": 145.2},
                last_optimization=datetime.utcnow(),
                energy_usage_kw=145.2,
                comfort_score=0.85
            ),
            BuildingStatusResponse(
                building_id="building_002",
                name="Research Lab",
                status="online",
                zones=8,
                current_state={"avg_temp": 21.8, "energy_kw": 98.3},
                last_optimization=datetime.utcnow(),
                energy_usage_kw=98.3,
                comfort_score=0.92
            )
        ]
        return buildings
    
    @app.get("/buildings/{building_id}/status", response_model=BuildingStatusResponse)
    async def get_building_status(
        building_id: str,
        current_user: UserModel = Depends(require_viewer)
    ):
        """Get detailed status for specific building."""
        # In real implementation, query from database and BMS
        if building_id not in ["building_001", "building_002"]:
            raise HTTPException(status_code=404, detail="Building not found")
        
        return BuildingStatusResponse(
            building_id=building_id,
            name="Main Office Building" if building_id == "building_001" else "Research Lab",
            status="online",
            zones=12 if building_id == "building_001" else 8,
            current_state={
                "temperatures": [22.1, 22.3, 21.9, 22.5, 22.0],
                "setpoints": [22.0, 22.0, 22.0, 22.0, 22.0],
                "energy_kw": 145.2 if building_id == "building_001" else 98.3
            },
            last_optimization=datetime.utcnow(),
            energy_usage_kw=145.2 if building_id == "building_001" else 98.3,
            comfort_score=0.85 if building_id == "building_001" else 0.92
        )
    
    @app.get("/buildings/{building_id}/timeseries", response_model=TimeSeriesResponse)
    async def get_building_timeseries(
        building_id: str,
        data_type: str = "temperature",
        hours: int = 24,
        current_user: UserModel = Depends(require_viewer)
    ):
        """Get time series data for building."""
        if building_id not in ["building_001", "building_002"]:
            raise HTTPException(status_code=404, detail="Building not found")
        
        start_time = datetime.utcnow() - timedelta(hours=hours)
        end_time = datetime.utcnow()
        
        try:
            data = await timeseries_storage.get_time_range(
                building_id=building_id,
                data_type=data_type,
                start_time=start_time,
                end_time=end_time
            )
            
            return TimeSeriesResponse(
                building_id=building_id,
                data_type=data_type,
                start_time=start_time,
                end_time=end_time,
                data=data
            )
        except Exception as e:
            logger.error(f"Error getting timeseries data: {e}")
            # Return mock data for now
            mock_data = []
            for i in range(hours):
                timestamp = start_time + timedelta(hours=i)
                value = 22.0 + 2.0 * (i % 12) / 12  # Mock temperature pattern
                mock_data.append({
                    "timestamp": timestamp,
                    "value": value,
                    "quality": 1.0
                })
            
            return TimeSeriesResponse(
                building_id=building_id,
                data_type=data_type,
                start_time=start_time,
                end_time=end_time,
                data=mock_data
            )
    
    # Optimization endpoints
    @app.post("/buildings/{building_id}/optimize", response_model=OptimizationResponse)
    async def optimize_building(
        building_id: str,
        request: OptimizationRequest,
        current_user: UserModel = Depends(require_operator)
    ):
        """Start optimization for building."""
        if building_id not in ["building_001", "building_002"]:
            raise HTTPException(status_code=404, detail="Building not found")
        
        # Mock optimization response
        optimization_id = f"opt_{building_id}_{int(datetime.utcnow().timestamp())}"
        
        # Log the request
        log_security_event(
            "optimization_requested", 
            current_user.username,
            {"building_id": building_id, "optimization_id": optimization_id}
        )
        
        return OptimizationResponse(
            optimization_id=optimization_id,
            building_id=building_id,
            status="completed",
            solver_type="quantum_hybrid",
            computation_time_ms=2150.5,
            objective_value=0.823,
            feasible=True,
            control_schedule={
                "setpoints": [22.0, 21.8, 22.2, 21.9, 22.1],
                "dampers": [45.0, 50.0, 40.0, 48.0, 42.0]
            },
            quantum_metrics={
                "chain_breaks": 12,
                "embedding_quality": 0.87,
                "num_reads": 1000
            },
            completed_at=datetime.utcnow()
        )
    
    @app.get("/optimizations/{optimization_id}", response_model=OptimizationResponse)
    async def get_optimization_result(
        optimization_id: str,
        current_user: UserModel = Depends(require_viewer)
    ):
        """Get optimization result by ID."""
        # In real implementation, query from database
        if not optimization_id.startswith("opt_"):
            raise HTTPException(status_code=404, detail="Optimization not found")
        
        parts = optimization_id.split("_")
        building_id = "_".join(parts[1:-1]) if len(parts) > 2 else "building_001"
        
        return OptimizationResponse(
            optimization_id=optimization_id,
            building_id=building_id,
            status="completed",
            solver_type="quantum_hybrid",
            computation_time_ms=2150.5,
            objective_value=0.823,
            feasible=True,
            control_schedule={
                "setpoints": [22.0, 21.8, 22.2, 21.9, 22.1],
                "dampers": [45.0, 50.0, 40.0, 48.0, 42.0]
            },
            quantum_metrics={
                "chain_breaks": 12,
                "embedding_quality": 0.87,
                "num_reads": 1000
            },
            created_at=datetime.utcnow() - timedelta(minutes=5),
            completed_at=datetime.utcnow()
        )
    
    # Control endpoints
    @app.post("/buildings/{building_id}/control", response_model=ControlResponse)
    async def send_control_command(
        building_id: str,
        control: ControlRequest,
        current_user: UserModel = Depends(require_operator)
    ):
        """Send control command to building."""
        if building_id not in ["building_001", "building_002"]:
            raise HTTPException(status_code=404, detail="Building not found")
        
        control_id = f"ctrl_{building_id}_{int(datetime.utcnow().timestamp())}"
        
        # Log the control command
        log_security_event(
            "control_command", 
            current_user.username,
            {"building_id": building_id, "control_id": control_id, "immediate": control.immediate}
        )
        
        return ControlResponse(
            control_id=control_id,
            building_id=building_id,
            status="applied",
            applied_at=datetime.utcnow()
        )
    
    # Metrics endpoints
    @app.get("/buildings/{building_id}/metrics", response_model=MetricsResponse)
    async def get_building_metrics(
        building_id: str,
        metric_type: str = "energy",
        hours: int = 24,
        current_user: UserModel = Depends(require_viewer)
    ):
        """Get performance metrics for building."""
        if building_id not in ["building_001", "building_002"]:
            raise HTTPException(status_code=404, detail="Building not found")
        
        start_time = datetime.utcnow() - timedelta(hours=hours)
        end_time = datetime.utcnow()
        
        try:
            metrics_data = await metrics_storage.get_metrics_summary(
                building_id=building_id,
                metric_type=metric_type,
                start_time=start_time,
                end_time=end_time
            )
            return MetricsResponse(**metrics_data)
        except Exception as e:
            logger.error(f"Error getting metrics: {e}")
            # Return mock metrics
            return MetricsResponse(
                building_id=building_id,
                metric_type=metric_type,
                period_start=start_time,
                period_end=end_time,
                metrics={
                    "total_consumption_kwh": {"avg": 145.2, "min": 120.5, "max": 180.3},
                    "cost_usd": {"avg": 15.67, "min": 12.20, "max": 19.45}
                }
            )
    
    # Health check endpoint
    @app.get("/health")
    async def health_check():
        """Simple health check endpoint."""
        return {"status": "healthy", "timestamp": datetime.utcnow()}
    
    # Startup event
    @app.on_event("startup")
    async def startup_event():
        """Initialize services on startup."""
        logger.info("Starting Quantum HVAC API...")
        
        # Initialize database
        try:
            await db_manager.initialize()
            logger.info("Database initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
        
        # Connect mock BMS for development
        try:
            await mock_bms.connect()
            logger.info("Mock BMS connected")
        except Exception as e:
            logger.error(f"Failed to connect mock BMS: {e}")
    
    # Shutdown event
    @app.on_event("shutdown")
    async def shutdown_event():
        """Cleanup on shutdown."""
        logger.info("Shutting down Quantum HVAC API...")
        
        # Disconnect mock BMS
        try:
            await mock_bms.disconnect()
        except Exception as e:
            logger.error(f"Error disconnecting mock BMS: {e}")
        
        # Close database connections
        try:
            await db_manager.close()
        except Exception as e:
            logger.error(f"Error closing database: {e}")
    
    return app


# Create app instance
app = create_app()
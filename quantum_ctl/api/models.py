"""Pydantic models for API requests and responses."""

from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from pydantic import BaseModel, Field, validator

from ..models.building import BuildingState


class BuildingStatusResponse(BaseModel):
    """Response model for building status."""
    building_id: str
    name: Optional[str] = None
    status: str = Field(..., description="online, offline, maintenance, error")
    zones: int
    current_state: Dict[str, Any]
    last_optimization: Optional[datetime] = None
    energy_usage_kw: Optional[float] = None
    comfort_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class OptimizationRequest(BaseModel):
    """Request model for optimization."""
    building_id: str
    horizon_hours: int = Field(default=24, ge=1, le=168)
    objectives: Dict[str, float] = Field(
        default_factory=lambda: {"energy_cost": 0.6, "comfort": 0.3, "carbon": 0.1}
    )
    constraints: Optional[Dict[str, Any]] = None
    solver_config: Optional[Dict[str, Any]] = None
    
    @validator('objectives')
    def validate_objectives(cls, v):
        """Validate that objective weights sum to 1.0."""
        total = sum(v.values())
        if not (0.99 <= total <= 1.01):  # Allow small floating point errors
            raise ValueError("Objective weights must sum to 1.0")
        return v


class OptimizationResponse(BaseModel):
    """Response model for optimization results."""
    optimization_id: str
    building_id: str
    status: str = Field(..., description="running, completed, failed")
    solver_type: str
    computation_time_ms: Optional[float] = None
    objective_value: Optional[float] = None
    feasible: Optional[bool] = None
    control_schedule: Optional[Dict[str, List[float]]] = None
    energy_forecast: Optional[Dict[str, Any]] = None
    comfort_forecast: Optional[Dict[str, Any]] = None
    quantum_metrics: Optional[Dict[str, Any]] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None


class TimeSeriesRequest(BaseModel):
    """Request model for time series data."""
    building_id: str
    data_types: List[str] = Field(default_factory=lambda: ["temperature"])
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    aggregation: Optional[str] = Field(None, regex="^(avg|min|max|sum|count)$")
    interval_minutes: Optional[int] = Field(None, ge=1, le=1440)
    zone_filter: Optional[List[str]] = None


class TimeSeriesResponse(BaseModel):
    """Response model for time series data."""
    building_id: str
    data_type: str
    start_time: datetime
    end_time: datetime
    interval_minutes: Optional[int] = None
    data: List[Dict[str, Union[datetime, float, str]]]
    metadata: Optional[Dict[str, Any]] = None


class BuildingConfigRequest(BaseModel):
    """Request model for building configuration."""
    building_id: str
    name: str
    location: Optional[Dict[str, Any]] = None
    zones: List[Dict[str, Any]]
    thermal_model: Dict[str, Any]
    hvac_config: Dict[str, Any]
    optimization_config: Optional[Dict[str, Any]] = None


class BuildingConfigResponse(BaseModel):
    """Response model for building configuration."""
    id: str
    building_id: str
    name: str
    location: Optional[Dict[str, Any]] = None
    zones: List[Dict[str, Any]]
    thermal_model: Dict[str, Any]
    hvac_config: Dict[str, Any]
    optimization_config: Optional[Dict[str, Any]] = None
    version: int
    created_at: datetime
    updated_at: datetime
    is_active: bool


class ControlRequest(BaseModel):
    """Request model for control commands."""
    building_id: str
    setpoints: Optional[List[float]] = None
    dampers: Optional[List[float]] = None
    valves: Optional[List[float]] = None
    fan_speeds: Optional[List[float]] = None
    immediate: bool = Field(default=False, description="Apply immediately or schedule")
    duration_minutes: Optional[int] = Field(None, ge=1, le=1440)
    
    @validator('setpoints', 'dampers', 'valves', 'fan_speeds')
    def validate_ranges(cls, v, field):
        """Validate control value ranges."""
        if v is None:
            return v
        
        if field.name == 'setpoints':
            # Temperature setpoints in Celsius
            for temp in v:
                if not (15.0 <= temp <= 30.0):
                    raise ValueError(f"Setpoint {temp}°C out of range [15-30]°C")
        else:
            # Percentage values
            for val in v:
                if not (0.0 <= val <= 100.0):
                    raise ValueError(f"Value {val}% out of range [0-100]%")
        return v


class ControlResponse(BaseModel):
    """Response model for control commands."""
    control_id: str
    building_id: str
    status: str = Field(..., description="applied, scheduled, failed")
    applied_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    error_message: Optional[str] = None


class SystemStatusResponse(BaseModel):
    """Response model for overall system status."""
    status: str = Field(..., description="healthy, degraded, error")
    quantum_status: Dict[str, Any]
    database_status: Dict[str, Any]
    active_buildings: int
    active_optimizations: int
    total_energy_kw: Optional[float] = None
    uptime_seconds: float
    version: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class QuantumStatusResponse(BaseModel):
    """Response model for quantum solver status."""
    status: str = Field(..., description="online, offline, maintenance")
    qpu_name: Optional[str] = None
    queue_length: Optional[int] = None
    avg_computation_time_ms: Optional[float] = None
    success_rate: Optional[float] = Field(None, ge=0.0, le=1.0)
    total_problems_today: Optional[int] = None
    estimated_cost_today_usd: Optional[float] = None
    last_access: Optional[datetime] = None


class MetricsRequest(BaseModel):
    """Request model for performance metrics."""
    building_id: str
    metric_type: str = Field(..., regex="^(energy|comfort|quantum|system)$")
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    aggregation: str = Field(default="avg", regex="^(avg|min|max|sum|count)$")


class MetricsResponse(BaseModel):
    """Response model for performance metrics."""
    building_id: str
    metric_type: str
    period_start: datetime
    period_end: datetime
    metrics: Dict[str, Dict[str, Union[float, int, str]]]
    summary: Optional[Dict[str, float]] = None


class UserModel(BaseModel):
    """User model for authentication."""
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    disabled: bool = False
    roles: List[str] = Field(default_factory=list)


class TokenResponse(BaseModel):
    """Response model for authentication tokens."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    scope: Optional[str] = None


class LoginRequest(BaseModel):
    """Request model for user login."""
    username: str
    password: str


class ErrorResponse(BaseModel):
    """Standard error response model."""
    error: str
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: Optional[str] = None
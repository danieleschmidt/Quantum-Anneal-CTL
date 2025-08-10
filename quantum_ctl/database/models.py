"""Database models for quantum HVAC control system."""

import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional

from sqlalchemy import (
    Column, String, Integer, Float, DateTime, JSON, Boolean,
    Text, Index, ForeignKey
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

Base = declarative_base()


class TimeSeriesData(Base):
    """Time series data storage for sensor readings and control commands."""
    
    __tablename__ = "timeseries_data"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    timestamp = Column(DateTime, nullable=False, index=True)
    building_id = Column(String(100), nullable=False, index=True)
    zone_id = Column(String(100), nullable=True, index=True)
    data_type = Column(String(50), nullable=False)  # temperature, humidity, control, etc.
    value = Column(Float, nullable=False)
    unit = Column(String(20), nullable=True)
    quality = Column(Float, default=1.0)  # Data quality score 0-1
    metadata = Column(JSON, nullable=True)
    
    __table_args__ = (
        Index('idx_timeseries_lookup', 'building_id', 'data_type', 'timestamp'),
    )


class BuildingConfig(Base):
    """Building configuration and model parameters."""
    
    __tablename__ = "building_configs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    building_id = Column(String(100), unique=True, nullable=False, index=True)
    name = Column(String(200), nullable=False)
    location = Column(JSON, nullable=True)  # {"lat": x, "lon": y, "timezone": "..."}
    zones = Column(JSON, nullable=False)  # Zone configurations
    thermal_model = Column(JSON, nullable=False)  # Building thermal parameters
    hvac_config = Column(JSON, nullable=False)  # HVAC system configuration
    optimization_config = Column(JSON, nullable=True)  # Quantum solver settings
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    version = Column(Integer, default=1)


class OptimizationResult(Base):
    """Quantum optimization results and control schedules."""
    
    __tablename__ = "optimization_results"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    building_id = Column(String(100), nullable=False, index=True)
    optimization_timestamp = Column(DateTime, nullable=False, index=True)
    horizon_start = Column(DateTime, nullable=False)
    horizon_end = Column(DateTime, nullable=False)
    
    # Quantum solver details
    solver_type = Column(String(50), nullable=False)  # quantum, classical, hybrid
    solver_config = Column(JSON, nullable=True)
    computation_time_ms = Column(Float, nullable=False)
    num_reads = Column(Integer, nullable=True)
    chain_breaks = Column(Integer, default=0)
    embedding_quality = Column(Float, nullable=True)
    
    # Problem formulation
    qubo_size = Column(Integer, nullable=True)
    constraint_violations = Column(Integer, default=0)
    objective_value = Column(Float, nullable=False)
    
    # Control schedule
    control_schedule = Column(JSON, nullable=False)  # Optimized control actions
    energy_forecast = Column(JSON, nullable=True)  # Predicted energy usage
    comfort_forecast = Column(JSON, nullable=True)  # Predicted comfort metrics
    
    # Validation
    feasible = Column(Boolean, default=True)
    applied = Column(Boolean, default=False)
    applied_at = Column(DateTime, nullable=True)
    
    __table_args__ = (
        Index('idx_optimization_lookup', 'building_id', 'optimization_timestamp'),
    )


class PerformanceMetric(Base):
    """Performance metrics and KPIs."""
    
    __tablename__ = "performance_metrics"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    building_id = Column(String(100), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    metric_type = Column(String(50), nullable=False)  # energy, comfort, quantum
    metric_name = Column(String(100), nullable=False)
    value = Column(Float, nullable=False)
    unit = Column(String(50), nullable=True)
    period_start = Column(DateTime, nullable=True)
    period_end = Column(DateTime, nullable=True)
    aggregation = Column(String(20), default="instant")  # instant, avg, sum, min, max
    metadata = Column(JSON, nullable=True)
    
    __table_args__ = (
        Index('idx_metrics_lookup', 'building_id', 'metric_type', 'timestamp'),
    )


class SystemEvent(Base):
    """System events, alerts, and anomalies."""
    
    __tablename__ = "system_events"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    timestamp = Column(DateTime, nullable=False, index=True)
    building_id = Column(String(100), nullable=True, index=True)
    event_type = Column(String(50), nullable=False)  # alert, error, info, warning
    severity = Column(String(20), nullable=False)  # critical, high, medium, low
    component = Column(String(100), nullable=False)  # controller, quantum, bms, etc.
    message = Column(Text, nullable=False)
    details = Column(JSON, nullable=True)
    resolved = Column(Boolean, default=False)
    resolved_at = Column(DateTime, nullable=True)
    resolved_by = Column(String(100), nullable=True)
    
    __table_args__ = (
        Index('idx_events_lookup', 'building_id', 'event_type', 'timestamp'),
    )


class QuantumSession(Base):
    """Quantum computing session logs."""
    
    __tablename__ = "quantum_sessions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_start = Column(DateTime, nullable=False)
    session_end = Column(DateTime, nullable=True)
    building_id = Column(String(100), nullable=False, index=True)
    
    # D-Wave connection details
    sampler_type = Column(String(50), nullable=False)
    qpu_name = Column(String(100), nullable=True)
    hybrid = Column(Boolean, default=False)
    
    # Session statistics  
    total_samples = Column(Integer, default=0)
    total_problems = Column(Integer, default=0)
    total_computation_time_ms = Column(Float, default=0.0)
    avg_chain_breaks = Column(Float, nullable=True)
    success_rate = Column(Float, nullable=True)
    
    # Cost tracking
    qpu_access_time_us = Column(Integer, default=0)
    estimated_cost_usd = Column(Float, nullable=True)
    
    status = Column(String(20), default="active")  # active, completed, failed
    error_message = Column(Text, nullable=True)
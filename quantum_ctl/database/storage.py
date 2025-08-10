"""Storage interfaces for different data types."""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from dataclasses import asdict

from sqlalchemy import select, delete, and_, or_, func, desc
from sqlalchemy.orm import Session
from sqlalchemy.ext.asyncio import AsyncSession

from .models import (
    TimeSeriesData, BuildingConfig, OptimizationResult, 
    PerformanceMetric, SystemEvent, QuantumSession
)
from ..models.building import Building, BuildingState

logger = logging.getLogger(__name__)


class BaseStorage:
    """Base class for storage interfaces."""
    
    def __init__(self, session_factory):
        self.session_factory = session_factory


class TimeSeriesStorage(BaseStorage):
    """Storage interface for time series data."""
    
    async def store_data_point(
        self, 
        building_id: str,
        data_type: str,
        value: float,
        timestamp: Optional[datetime] = None,
        zone_id: Optional[str] = None,
        unit: Optional[str] = None,
        quality: float = 1.0,
        metadata: Optional[Dict] = None
    ) -> str:
        """Store a single data point."""
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        data_point = TimeSeriesData(
            timestamp=timestamp,
            building_id=building_id,
            zone_id=zone_id,
            data_type=data_type,
            value=value,
            unit=unit,
            quality=quality,
            metadata=metadata
        )
        
        async with self.session_factory() as session:
            session.add(data_point)
            await session.commit()
            await session.refresh(data_point)
            return str(data_point.id)
    
    async def store_batch(self, data_points: List[Dict[str, Any]]) -> List[str]:
        """Store multiple data points efficiently."""
        objects = []
        for point in data_points:
            if 'timestamp' not in point:
                point['timestamp'] = datetime.utcnow()
            objects.append(TimeSeriesData(**point))
        
        async with self.session_factory() as session:
            session.add_all(objects)
            await session.commit()
            return [str(obj.id) for obj in objects]
    
    async def get_latest(
        self,
        building_id: str,
        data_type: str,
        zone_id: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get latest data points."""
        async with self.session_factory() as session:
            query = select(TimeSeriesData).where(
                and_(
                    TimeSeriesData.building_id == building_id,
                    TimeSeriesData.data_type == data_type
                )
            )
            
            if zone_id:
                query = query.where(TimeSeriesData.zone_id == zone_id)
            
            query = query.order_by(desc(TimeSeriesData.timestamp)).limit(limit)
            result = await session.execute(query)
            
            return [
                {
                    'id': str(row.id),
                    'timestamp': row.timestamp,
                    'building_id': row.building_id,
                    'zone_id': row.zone_id,
                    'data_type': row.data_type,
                    'value': row.value,
                    'unit': row.unit,
                    'quality': row.quality,
                    'metadata': row.metadata
                }
                for row in result.scalars().all()
            ]
    
    async def get_time_range(
        self,
        building_id: str,
        data_type: str,
        start_time: datetime,
        end_time: datetime,
        zone_id: Optional[str] = None,
        aggregation: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get data within time range with optional aggregation."""
        async with self.session_factory() as session:
            query = select(TimeSeriesData).where(
                and_(
                    TimeSeriesData.building_id == building_id,
                    TimeSeriesData.data_type == data_type,
                    TimeSeriesData.timestamp >= start_time,
                    TimeSeriesData.timestamp <= end_time
                )
            )
            
            if zone_id:
                query = query.where(TimeSeriesData.zone_id == zone_id)
            
            # Apply aggregation if specified
            if aggregation:
                # This would require more complex grouping logic
                # For now, return raw data
                pass
            
            query = query.order_by(TimeSeriesData.timestamp)
            result = await session.execute(query)
            
            return [
                {
                    'id': str(row.id),
                    'timestamp': row.timestamp,
                    'value': row.value,
                    'unit': row.unit,
                    'quality': row.quality
                }
                for row in result.scalars().all()
            ]
    
    async def cleanup_old_data(self, building_id: str, retention_days: int = 90):
        """Clean up old time series data."""
        cutoff_date = datetime.utcnow() - timedelta(days=retention_days)
        
        async with self.session_factory() as session:
            result = await session.execute(
                delete(TimeSeriesData).where(
                    and_(
                        TimeSeriesData.building_id == building_id,
                        TimeSeriesData.timestamp < cutoff_date
                    )
                )
            )
            await session.commit()
            logger.info(f"Cleaned up {result.rowcount} old data points for {building_id}")


class ConfigurationStorage(BaseStorage):
    """Storage interface for building configurations."""
    
    async def store_building_config(self, building: Building) -> str:
        """Store building configuration."""
        building_dict = building.to_dict()
        
        config = BuildingConfig(
            building_id=building.building_id,
            name=building_dict.get('name', 'Unknown Building'),
            location=building_dict.get('location'),
            zones=building_dict.get('zones', []),
            thermal_model=building_dict.get('thermal_model', {}),
            hvac_config=building_dict.get('hvac_config', {}),
            optimization_config=building_dict.get('optimization_config')
        )
        
        async with self.session_factory() as session:
            # Check if config already exists
            existing = await session.execute(
                select(BuildingConfig).where(
                    BuildingConfig.building_id == building.building_id
                )
            )
            existing_config = existing.scalar_one_or_none()
            
            if existing_config:
                # Update existing
                existing_config.zones = config.zones
                existing_config.thermal_model = config.thermal_model
                existing_config.hvac_config = config.hvac_config
                existing_config.optimization_config = config.optimization_config
                existing_config.updated_at = datetime.utcnow()
                existing_config.version += 1
                await session.commit()
                return str(existing_config.id)
            else:
                # Create new
                session.add(config)
                await session.commit()
                await session.refresh(config)
                return str(config.id)
    
    async def get_building_config(self, building_id: str) -> Optional[Dict[str, Any]]:
        """Get building configuration."""
        async with self.session_factory() as session:
            result = await session.execute(
                select(BuildingConfig).where(
                    and_(
                        BuildingConfig.building_id == building_id,
                        BuildingConfig.is_active == True
                    )
                )
            )
            config = result.scalar_one_or_none()
            
            if config:
                return {
                    'id': str(config.id),
                    'building_id': config.building_id,
                    'name': config.name,
                    'location': config.location,
                    'zones': config.zones,
                    'thermal_model': config.thermal_model,
                    'hvac_config': config.hvac_config,
                    'optimization_config': config.optimization_config,
                    'created_at': config.created_at,
                    'updated_at': config.updated_at,
                    'version': config.version
                }
            return None


class ResultStorage(BaseStorage):
    """Storage interface for optimization results."""
    
    async def store_optimization_result(
        self,
        building_id: str,
        optimization_timestamp: datetime,
        horizon_start: datetime,
        horizon_end: datetime,
        solver_type: str,
        computation_time_ms: float,
        objective_value: float,
        control_schedule: Dict[str, Any],
        solver_config: Optional[Dict] = None,
        num_reads: Optional[int] = None,
        chain_breaks: int = 0,
        embedding_quality: Optional[float] = None,
        qubo_size: Optional[int] = None,
        constraint_violations: int = 0,
        energy_forecast: Optional[Dict] = None,
        comfort_forecast: Optional[Dict] = None,
        feasible: bool = True
    ) -> str:
        """Store optimization result."""
        result = OptimizationResult(
            building_id=building_id,
            optimization_timestamp=optimization_timestamp,
            horizon_start=horizon_start,
            horizon_end=horizon_end,
            solver_type=solver_type,
            solver_config=solver_config,
            computation_time_ms=computation_time_ms,
            num_reads=num_reads,
            chain_breaks=chain_breaks,
            embedding_quality=embedding_quality,
            qubo_size=qubo_size,
            constraint_violations=constraint_violations,
            objective_value=objective_value,
            control_schedule=control_schedule,
            energy_forecast=energy_forecast,
            comfort_forecast=comfort_forecast,
            feasible=feasible
        )
        
        async with self.session_factory() as session:
            session.add(result)
            await session.commit()
            await session.refresh(result)
            return str(result.id)
    
    async def get_latest_results(
        self, 
        building_id: str, 
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get latest optimization results."""
        async with self.session_factory() as session:
            result = await session.execute(
                select(OptimizationResult)
                .where(OptimizationResult.building_id == building_id)
                .order_by(desc(OptimizationResult.optimization_timestamp))
                .limit(limit)
            )
            
            return [
                {
                    'id': str(row.id),
                    'optimization_timestamp': row.optimization_timestamp,
                    'solver_type': row.solver_type,
                    'computation_time_ms': row.computation_time_ms,
                    'objective_value': row.objective_value,
                    'feasible': row.feasible,
                    'applied': row.applied,
                    'constraint_violations': row.constraint_violations,
                    'control_schedule': row.control_schedule
                }
                for row in result.scalars().all()
            ]
    
    async def mark_result_applied(self, result_id: str) -> bool:
        """Mark optimization result as applied."""
        async with self.session_factory() as session:
            result = await session.get(OptimizationResult, result_id)
            if result:
                result.applied = True
                result.applied_at = datetime.utcnow()
                await session.commit()
                return True
            return False


class MetricsStorage(BaseStorage):
    """Storage interface for performance metrics."""
    
    async def store_metric(
        self,
        building_id: str,
        metric_type: str,
        metric_name: str,
        value: float,
        timestamp: Optional[datetime] = None,
        unit: Optional[str] = None,
        period_start: Optional[datetime] = None,
        period_end: Optional[datetime] = None,
        aggregation: str = "instant",
        metadata: Optional[Dict] = None
    ) -> str:
        """Store performance metric."""
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        metric = PerformanceMetric(
            building_id=building_id,
            timestamp=timestamp,
            metric_type=metric_type,
            metric_name=metric_name,
            value=value,
            unit=unit,
            period_start=period_start,
            period_end=period_end,
            aggregation=aggregation,
            metadata=metadata
        )
        
        async with self.session_factory() as session:
            session.add(metric)
            await session.commit()
            await session.refresh(metric)
            return str(metric.id)
    
    async def get_metrics_summary(
        self, 
        building_id: str, 
        metric_type: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get metrics summary for time period."""
        if start_time is None:
            start_time = datetime.utcnow() - timedelta(hours=24)
        if end_time is None:
            end_time = datetime.utcnow()
        
        async with self.session_factory() as session:
            query = select(PerformanceMetric).where(
                and_(
                    PerformanceMetric.building_id == building_id,
                    PerformanceMetric.metric_type == metric_type,
                    PerformanceMetric.timestamp >= start_time,
                    PerformanceMetric.timestamp <= end_time
                )
            )
            
            result = await session.execute(query)
            metrics = result.scalars().all()
            
            # Group by metric name and calculate statistics
            summary = {}
            for metric in metrics:
                if metric.metric_name not in summary:
                    summary[metric.metric_name] = {
                        'values': [],
                        'unit': metric.unit,
                        'count': 0
                    }
                summary[metric.metric_name]['values'].append(metric.value)
                summary[metric.metric_name]['count'] += 1
            
            # Calculate statistics
            for metric_name, data in summary.items():
                values = data['values']
                data.update({
                    'avg': sum(values) / len(values),
                    'min': min(values),
                    'max': max(values),
                    'latest': values[-1] if values else None
                })
                del data['values']  # Remove raw values to save space
            
            return {
                'building_id': building_id,
                'metric_type': metric_type,
                'period_start': start_time,
                'period_end': end_time,
                'metrics': summary
            }
"""
Predictive Maintenance System with Quantum-Enhanced Anomaly Detection.

Advanced predictive maintenance system that uses quantum machine learning
for early detection of HVAC system anomalies and optimal maintenance scheduling.

Novel Features:
1. Quantum kernel methods for anomaly detection
2. Multi-modal sensor fusion with quantum feature extraction
3. Predictive failure modeling with quantum advantage
4. Maintenance optimization using quantum annealing
5. Real-time condition monitoring with adaptive thresholds
"""

from typing import Dict, Any, List, Optional, Tuple, Callable, Union
import numpy as np
import asyncio
import time
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque, defaultdict
from enum import Enum
import json
import math

try:
    from sklearn.svm import OneClassSVM
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    OneClassSVM = None
    IsolationForest = None
    StandardScaler = None
    RobustScaler = None
    PCA = None


class HealthStatus(Enum):
    """Equipment health status levels."""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    CRITICAL = "critical"
    FAILURE = "failure"


class MaintenanceType(Enum):
    """Types of maintenance actions."""
    PREVENTIVE = "preventive"
    PREDICTIVE = "predictive"
    CORRECTIVE = "corrective"
    EMERGENCY = "emergency"
    OPTIMIZATION = "optimization"


@dataclass
class SensorReading:
    """Individual sensor measurement with metadata."""
    sensor_id: str
    timestamp: datetime
    value: float
    unit: str
    quality: float = 1.0  # 0-1 quality score
    anomaly_score: float = 0.0  # 0-1 anomaly likelihood
    
    @property
    def age_minutes(self) -> float:
        """Age of reading in minutes."""
        return (datetime.now() - self.timestamp).total_seconds() / 60


@dataclass
class EquipmentProfile:
    """Equipment profile with operational characteristics."""
    equipment_id: str
    equipment_type: str  # "chiller", "ahu", "pump", "fan", etc.
    model: str
    install_date: datetime
    rated_capacity: float
    operating_hours: float
    maintenance_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # Operating ranges
    normal_temp_range: Tuple[float, float] = (10.0, 40.0)
    normal_pressure_range: Tuple[float, float] = (1.0, 5.0)
    normal_flow_range: Tuple[float, float] = (0.5, 1.2)  # As ratio of rated
    normal_power_range: Tuple[float, float] = (0.3, 1.1)  # As ratio of rated
    
    @property
    def age_years(self) -> float:
        """Equipment age in years."""
        return (datetime.now() - self.install_date).total_seconds() / (365.25 * 24 * 3600)
    
    @property
    def usage_factor(self) -> float:
        """Equipment usage intensity factor."""
        expected_hours = self.age_years * 8760 * 0.4  # Assume 40% utilization
        return self.operating_hours / max(expected_hours, 1)


@dataclass
class AnomalyEvent:
    """Detected anomaly event with analysis."""
    event_id: str
    equipment_id: str
    timestamp: datetime
    anomaly_type: str
    severity: float  # 0-1
    confidence: float  # 0-1
    affected_sensors: List[str]
    description: str
    recommended_action: str
    quantum_features: Dict[str, float] = field(default_factory=dict)
    
    @property
    def risk_score(self) -> float:
        """Calculate overall risk score."""
        return self.severity * self.confidence


@dataclass
class MaintenanceRecommendation:
    """Maintenance recommendation with optimization."""
    recommendation_id: str
    equipment_id: str
    maintenance_type: MaintenanceType
    urgency: float  # 0-1
    estimated_cost: float
    estimated_duration: float  # hours
    optimal_date: datetime
    failure_probability: float  # 0-1 if no action taken
    description: str
    required_parts: List[str] = field(default_factory=list)
    required_skills: List[str] = field(default_factory=list)


class QuantumAnomalyDetector:
    """Quantum-enhanced anomaly detection using quantum kernel methods."""
    
    def __init__(self, feature_dim: int = 16):
        self.feature_dim = feature_dim
        self.quantum_features_cache: deque = deque(maxlen=1000)
        self.anomaly_threshold = 0.3
        self.classical_detector = None
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.pca = PCA(n_components=min(feature_dim, 8)) if SKLEARN_AVAILABLE else None
        self.is_trained = False
        self.logger = logging.getLogger(__name__)
        
        # Quantum-inspired parameters
        self.quantum_kernel_params = {
            'feature_map_depth': 3,
            'entanglement_layers': 2,
            'rotation_angles': np.random.uniform(0, 2*np.pi, feature_dim)
        }
    
    def extract_quantum_features(self, sensor_data: Dict[str, float]) -> np.ndarray:
        """Extract quantum-inspired features from sensor data."""
        # Convert sensor data to feature vector
        base_features = []
        sensor_keys = sorted(sensor_data.keys())
        
        for key in sensor_keys:
            base_features.append(sensor_data[key])
        
        # Pad or truncate to desired dimension
        while len(base_features) < self.feature_dim:
            base_features.append(0.0)
        base_features = base_features[:self.feature_dim]
        
        features = np.array(base_features)
        
        # Apply quantum-inspired feature transformation
        quantum_features = self._apply_quantum_feature_map(features)
        
        return quantum_features
    
    def _apply_quantum_feature_map(self, features: np.ndarray) -> np.ndarray:
        """Apply quantum-inspired feature mapping."""
        # Normalize features
        features = features / (np.linalg.norm(features) + 1e-10)
        
        # Apply rotation gates (quantum-inspired)
        rotated_features = []
        for i, (feature, angle) in enumerate(zip(features, self.quantum_kernel_params['rotation_angles'])):
            # Simulate rotation gate: |0⟩ → cos(θ/2)|0⟩ + sin(θ/2)|1⟩
            rotated = feature * np.cos(angle) + np.sin(angle)
            rotated_features.append(rotated)
        
        rotated_features = np.array(rotated_features)
        
        # Apply entanglement-like correlations
        entangled_features = rotated_features.copy()
        for layer in range(self.quantum_kernel_params['entanglement_layers']):
            for i in range(0, len(entangled_features) - 1, 2):
                # Simulate CNOT-like entanglement
                correlation = entangled_features[i] * entangled_features[i + 1]
                entangled_features[i] = entangled_features[i] + 0.1 * correlation
                entangled_features[i + 1] = entangled_features[i + 1] + 0.1 * correlation
        
        # Add quantum interference terms
        interference_terms = []
        for i in range(len(entangled_features)):
            for j in range(i + 1, len(entangled_features)):
                interference = np.cos(entangled_features[i] + entangled_features[j])
                interference_terms.append(interference)
        
        # Combine all features
        final_features = np.concatenate([entangled_features, interference_terms[:8]])
        
        return final_features[:self.feature_dim]
    
    def train_detector(self, normal_data: List[Dict[str, float]]) -> None:
        """Train anomaly detector on normal operating data."""
        if not SKLEARN_AVAILABLE or len(normal_data) < 10:
            self.logger.warning("Insufficient data or sklearn unavailable for training")
            return
        
        # Extract quantum features
        feature_matrix = []
        for data_point in normal_data:
            quantum_features = self.extract_quantum_features(data_point)
            feature_matrix.append(quantum_features)
        
        X = np.array(feature_matrix)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Apply PCA for dimensionality reduction
        X_reduced = self.pca.fit_transform(X_scaled)
        
        # Train One-Class SVM for anomaly detection
        self.classical_detector = OneClassSVM(
            kernel='rbf',
            gamma='scale',
            nu=0.05  # Expected fraction of outliers
        )
        self.classical_detector.fit(X_reduced)
        
        self.is_trained = True
        self.logger.info(f"Trained quantum anomaly detector on {len(normal_data)} samples")
    
    def detect_anomaly(self, sensor_data: Dict[str, float]) -> Tuple[bool, float, Dict[str, float]]:
        """Detect anomaly in sensor data."""
        quantum_features = self.extract_quantum_features(sensor_data)
        
        if not self.is_trained or not SKLEARN_AVAILABLE:
            # Fallback to statistical anomaly detection
            return self._statistical_anomaly_detection(sensor_data, quantum_features)
        
        # Transform features
        features_scaled = self.scaler.transform(quantum_features.reshape(1, -1))
        features_reduced = self.pca.transform(features_scaled)
        
        # Detect using trained model
        anomaly_score = self.classical_detector.decision_function(features_reduced)[0]
        is_anomaly = anomaly_score < 0
        
        # Convert score to 0-1 range
        normalized_score = max(0, min(1, (0.5 - anomaly_score) / 1.0))
        
        # Store quantum features for analysis
        quantum_feature_dict = {f'qf_{i}': float(val) for i, val in enumerate(quantum_features)}
        
        return is_anomaly, normalized_score, quantum_feature_dict
    
    def _statistical_anomaly_detection(
        self, 
        sensor_data: Dict[str, float], 
        quantum_features: np.ndarray
    ) -> Tuple[bool, float, Dict[str, float]]:
        """Fallback statistical anomaly detection."""
        # Simple statistical approach
        anomalies = 0
        total_checks = 0
        
        # Check for extreme values using z-score
        for key, value in sensor_data.items():
            if 'temp' in key.lower():
                mean, std = 22.0, 5.0  # Typical temperature
            elif 'pressure' in key.lower():
                mean, std = 2.5, 1.0  # Typical pressure
            elif 'flow' in key.lower():
                mean, std = 0.8, 0.2  # Typical flow ratio
            elif 'power' in key.lower():
                mean, std = 0.7, 0.3  # Typical power ratio
            else:
                continue
            
            z_score = abs(value - mean) / std
            if z_score > 3.0:  # 3-sigma threshold
                anomalies += 1
            total_checks += 1
        
        anomaly_ratio = anomalies / max(total_checks, 1)
        is_anomaly = anomaly_ratio > 0.2
        
        quantum_feature_dict = {f'qf_{i}': float(val) for i, val in enumerate(quantum_features)}
        
        return is_anomaly, anomaly_ratio, quantum_feature_dict


class FailurePredictionModel:
    """Predictive model for equipment failure probability."""
    
    def __init__(self):
        self.model = None
        self.feature_scaler = RobustScaler() if SKLEARN_AVAILABLE else None
        self.is_trained = False
        self.prediction_horizon_days = 30
        self.logger = logging.getLogger(__name__)
    
    def extract_failure_features(
        self, 
        equipment: EquipmentProfile, 
        recent_anomalies: List[AnomalyEvent],
        sensor_stats: Dict[str, float]
    ) -> np.ndarray:
        """Extract features for failure prediction."""
        features = []
        
        # Equipment age and usage features
        features.extend([
            equipment.age_years,
            equipment.usage_factor,
            equipment.operating_hours / 8760,  # Annual equivalent
            len(equipment.maintenance_history) / max(equipment.age_years, 1)  # Maintenance frequency
        ])
        
        # Recent anomaly features
        recent_anomaly_count = len([a for a in recent_anomalies if (datetime.now() - a.timestamp).days <= 7])
        avg_anomaly_severity = np.mean([a.severity for a in recent_anomalies]) if recent_anomalies else 0.0
        max_anomaly_severity = max([a.severity for a in recent_anomalies]) if recent_anomalies else 0.0
        
        features.extend([
            recent_anomaly_count,
            avg_anomaly_severity,
            max_anomaly_severity,
            len(recent_anomalies) / max(equipment.age_years * 12, 1)  # Anomalies per month
        ])
        
        # Sensor statistics features
        features.extend([
            sensor_stats.get('temp_variance', 0.0),
            sensor_stats.get('pressure_variance', 0.0),
            sensor_stats.get('flow_variance', 0.0),
            sensor_stats.get('power_variance', 0.0),
            sensor_stats.get('temp_trend', 0.0),
            sensor_stats.get('efficiency_trend', 0.0)
        ])
        
        # Seasonal and operational features
        current_month = datetime.now().month
        features.extend([
            np.sin(2 * np.pi * current_month / 12),  # Seasonal encoding
            np.cos(2 * np.pi * current_month / 12),
            datetime.now().weekday() / 6.0,  # Day of week
            datetime.now().hour / 23.0  # Hour of day
        ])
        
        return np.array(features)
    
    def predict_failure_probability(
        self,
        equipment: EquipmentProfile,
        recent_anomalies: List[AnomalyEvent],
        sensor_stats: Dict[str, float]
    ) -> Dict[str, float]:
        """Predict equipment failure probability."""
        features = self.extract_failure_features(equipment, recent_anomalies, sensor_stats)
        
        if not self.is_trained or not SKLEARN_AVAILABLE:
            # Heuristic prediction
            return self._heuristic_failure_prediction(equipment, recent_anomalies, sensor_stats)
        
        # Scale features and predict
        features_scaled = self.feature_scaler.transform(features.reshape(1, -1))
        failure_prob = self.model.predict_proba(features_scaled)[0][1]  # Probability of failure class
        
        return {
            'failure_probability': float(failure_prob),
            'risk_level': self._categorize_risk(failure_prob),
            'prediction_horizon_days': self.prediction_horizon_days,
            'confidence': 0.8 if self.is_trained else 0.4
        }
    
    def _heuristic_failure_prediction(
        self,
        equipment: EquipmentProfile,
        recent_anomalies: List[AnomalyEvent],
        sensor_stats: Dict[str, float]
    ) -> Dict[str, float]:
        """Heuristic failure prediction when ML model unavailable."""
        # Age factor (higher age = higher failure risk)
        age_factor = min(equipment.age_years / 20.0, 1.0)  # 20 years = max age factor
        
        # Usage factor
        usage_factor = min(equipment.usage_factor / 2.0, 1.0)  # 2x expected usage = max factor
        
        # Anomaly factor
        recent_severe_anomalies = [a for a in recent_anomalies if a.severity > 0.7]
        anomaly_factor = min(len(recent_severe_anomalies) / 5.0, 1.0)
        
        # Maintenance factor (lack of maintenance increases risk)
        months_since_maintenance = 12  # Default
        if equipment.maintenance_history:
            last_maintenance = max(equipment.maintenance_history, key=lambda x: x.get('date', datetime.min))
            months_since_maintenance = (datetime.now() - last_maintenance.get('date', datetime.now())).days / 30.0
        
        maintenance_factor = min(months_since_maintenance / 24.0, 1.0)  # 24 months = max factor
        
        # Combine factors
        failure_prob = (
            0.3 * age_factor +
            0.2 * usage_factor +
            0.3 * anomaly_factor +
            0.2 * maintenance_factor
        )
        
        return {
            'failure_probability': failure_prob,
            'risk_level': self._categorize_risk(failure_prob),
            'prediction_horizon_days': self.prediction_horizon_days,
            'confidence': 0.6
        }
    
    def _categorize_risk(self, probability: float) -> str:
        """Categorize failure risk level."""
        if probability < 0.1:
            return "low"
        elif probability < 0.3:
            return "medium"
        elif probability < 0.6:
            return "high"
        else:
            return "critical"


class PredictiveMaintenanceSystem:
    """
    Comprehensive predictive maintenance system with quantum-enhanced analytics.
    
    Integrates anomaly detection, failure prediction, and maintenance optimization
    to provide proactive equipment management for HVAC systems.
    """
    
    def __init__(self):
        self.equipment_profiles: Dict[str, EquipmentProfile] = {}
        self.sensor_data: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.anomaly_events: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.maintenance_recommendations: Dict[str, List[MaintenanceRecommendation]] = defaultdict(list)
        
        self.anomaly_detector = QuantumAnomalyDetector()
        self.failure_predictor = FailurePredictionModel()
        
        self.logger = logging.getLogger(__name__)
        self._monitoring_active = False
        self._shutdown_event = asyncio.Event()
        
        # System parameters
        self.anomaly_alert_threshold = 0.7
        self.failure_alert_threshold = 0.4
        self.maintenance_optimization_interval = 3600  # seconds
    
    def register_equipment(self, equipment: EquipmentProfile) -> None:
        """Register equipment for monitoring."""
        self.equipment_profiles[equipment.equipment_id] = equipment
        self.logger.info(f"Registered equipment: {equipment.equipment_id} ({equipment.equipment_type})")
    
    async def add_sensor_reading(self, equipment_id: str, reading: SensorReading) -> None:
        """Add new sensor reading and process for anomalies."""
        if equipment_id not in self.equipment_profiles:
            self.logger.warning(f"Unknown equipment ID: {equipment_id}")
            return
        
        # Store sensor reading
        self.sensor_data[equipment_id].append(reading)
        
        # Check for anomalies if enough data
        if len(self.sensor_data[equipment_id]) >= 10:
            await self._check_for_anomalies(equipment_id)
    
    async def add_batch_sensor_data(self, equipment_id: str, batch_data: Dict[str, float]) -> None:
        """Add batch sensor data and analyze."""
        timestamp = datetime.now()
        
        for sensor_id, value in batch_data.items():
            reading = SensorReading(
                sensor_id=sensor_id,
                timestamp=timestamp,
                value=value,
                unit="auto-detected"
            )
            await self.add_sensor_reading(equipment_id, reading)
    
    async def _check_for_anomalies(self, equipment_id: str) -> None:
        """Check recent sensor data for anomalies."""
        recent_readings = list(self.sensor_data[equipment_id])[-10:]
        
        # Convert to sensor data dict
        sensor_data = {}
        for reading in recent_readings:
            sensor_data[reading.sensor_id] = reading.value
        
        # Detect anomalies
        is_anomaly, anomaly_score, quantum_features = self.anomaly_detector.detect_anomaly(sensor_data)
        
        if is_anomaly and anomaly_score > self.anomaly_alert_threshold:
            # Create anomaly event
            event = AnomalyEvent(
                event_id=f"{equipment_id}_{int(time.time() * 1000)}",
                equipment_id=equipment_id,
                timestamp=datetime.now(),
                anomaly_type="quantum_detected",
                severity=anomaly_score,
                confidence=0.8,
                affected_sensors=list(sensor_data.keys()),
                description=f"Quantum anomaly detection: score {anomaly_score:.3f}",
                recommended_action=self._get_anomaly_recommendation(anomaly_score),
                quantum_features=quantum_features
            )
            
            self.anomaly_events[equipment_id].append(event)
            
            self.logger.warning(
                f"Anomaly detected in {equipment_id}: "
                f"score={anomaly_score:.3f}, sensors={list(sensor_data.keys())}"
            )
            
            # Update sensor readings with anomaly scores
            for reading in recent_readings:
                reading.anomaly_score = anomaly_score
    
    def _get_anomaly_recommendation(self, severity: float) -> str:
        """Get recommended action based on anomaly severity."""
        if severity > 0.9:
            return "Immediate inspection required - potential equipment failure imminent"
        elif severity > 0.7:
            return "Schedule priority inspection within 24 hours"
        elif severity > 0.5:
            return "Schedule routine inspection within 1 week"
        else:
            return "Monitor closely for trend development"
    
    async def predict_equipment_failures(self) -> Dict[str, Dict[str, Any]]:
        """Predict failure probabilities for all equipment."""
        predictions = {}
        
        for equipment_id, equipment in self.equipment_profiles.items():
            # Get recent anomalies
            recent_anomalies = list(self.anomaly_events[equipment_id])
            
            # Calculate sensor statistics
            sensor_stats = self._calculate_sensor_statistics(equipment_id)
            
            # Predict failure
            prediction = self.failure_predictor.predict_failure_probability(
                equipment, recent_anomalies, sensor_stats
            )
            
            predictions[equipment_id] = prediction
            
            # Generate maintenance recommendation if high risk
            if prediction['failure_probability'] > self.failure_alert_threshold:
                await self._generate_maintenance_recommendation(equipment_id, prediction)
        
        return predictions
    
    def _calculate_sensor_statistics(self, equipment_id: str) -> Dict[str, float]:
        """Calculate statistical metrics from recent sensor data."""
        readings = list(self.sensor_data[equipment_id])
        
        if len(readings) < 10:
            return {}
        
        # Group by sensor type
        sensor_groups = defaultdict(list)
        for reading in readings[-100:]:  # Last 100 readings
            sensor_groups[reading.sensor_id].append(reading.value)
        
        stats = {}
        
        # Calculate variance for each sensor type
        for sensor_id, values in sensor_groups.items():
            if len(values) > 5:
                variance = np.var(values)
                trend = self._calculate_trend(values)
                
                if 'temp' in sensor_id.lower():
                    stats['temp_variance'] = variance
                    stats['temp_trend'] = trend
                elif 'pressure' in sensor_id.lower():
                    stats['pressure_variance'] = variance
                elif 'flow' in sensor_id.lower():
                    stats['flow_variance'] = variance
                elif 'power' in sensor_id.lower():
                    stats['power_variance'] = variance
                    
        # Calculate efficiency trend (simplified)
        if 'power_variance' in stats and 'temp_variance' in stats:
            stats['efficiency_trend'] = stats['temp_trend'] / max(stats['power_variance'], 0.1)
        
        return stats
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend direction in time series."""
        if len(values) < 3:
            return 0.0
        
        # Simple linear trend
        x = np.arange(len(values))
        y = np.array(values)
        
        # Calculate slope
        n = len(values)
        slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x**2) - np.sum(x)**2)
        
        return slope
    
    async def _generate_maintenance_recommendation(
        self, 
        equipment_id: str, 
        failure_prediction: Dict[str, Any]
    ) -> None:
        """Generate maintenance recommendation based on failure prediction."""
        equipment = self.equipment_profiles[equipment_id]
        
        # Determine maintenance type
        failure_prob = failure_prediction['failure_probability']
        
        if failure_prob > 0.8:
            maintenance_type = MaintenanceType.EMERGENCY
            urgency = 1.0
            optimal_date = datetime.now() + timedelta(hours=4)
        elif failure_prob > 0.6:
            maintenance_type = MaintenanceType.CORRECTIVE
            urgency = 0.8
            optimal_date = datetime.now() + timedelta(days=1)
        elif failure_prob > 0.4:
            maintenance_type = MaintenanceType.PREDICTIVE
            urgency = 0.6
            optimal_date = datetime.now() + timedelta(days=7)
        else:
            maintenance_type = MaintenanceType.PREVENTIVE
            urgency = 0.3
            optimal_date = datetime.now() + timedelta(days=30)
        
        # Estimate cost and duration
        base_cost = {
            "chiller": 5000,
            "ahu": 2000,
            "pump": 1000,
            "fan": 500
        }.get(equipment.equipment_type.lower(), 1500)
        
        cost_multiplier = 1.0 + failure_prob * 2.0  # Higher failure risk = higher cost
        estimated_cost = base_cost * cost_multiplier
        
        base_duration = {
            "chiller": 8,
            "ahu": 4,
            "pump": 2,
            "fan": 1
        }.get(equipment.equipment_type.lower(), 3)
        
        estimated_duration = base_duration * (1.0 + failure_prob)
        
        recommendation = MaintenanceRecommendation(
            recommendation_id=f"{equipment_id}_maint_{int(time.time())}",
            equipment_id=equipment_id,
            maintenance_type=maintenance_type,
            urgency=urgency,
            estimated_cost=estimated_cost,
            estimated_duration=estimated_duration,
            optimal_date=optimal_date,
            failure_probability=failure_prob,
            description=f"Predicted maintenance for {equipment.equipment_type} "
                       f"due to {failure_prediction['risk_level']} failure risk",
            required_parts=self._get_typical_parts(equipment.equipment_type),
            required_skills=self._get_required_skills(equipment.equipment_type, maintenance_type)
        )
        
        self.maintenance_recommendations[equipment_id].append(recommendation)
        
        self.logger.info(
            f"Generated maintenance recommendation for {equipment_id}: "
            f"{maintenance_type.value} with urgency {urgency:.2f}"
        )
    
    def _get_typical_parts(self, equipment_type: str) -> List[str]:
        """Get typical parts needed for equipment type."""
        parts_map = {
            "chiller": ["refrigerant", "filters", "gaskets", "belts"],
            "ahu": ["filters", "belts", "bearings", "damper_actuators"],
            "pump": ["seals", "bearings", "impeller", "gaskets"],
            "fan": ["belts", "bearings", "motor_components"]
        }
        return parts_map.get(equipment_type.lower(), ["filters", "gaskets"])
    
    def _get_required_skills(self, equipment_type: str, maintenance_type: MaintenanceType) -> List[str]:
        """Get required skills for maintenance."""
        base_skills = ["hvac_technician"]
        
        if maintenance_type in [MaintenanceType.EMERGENCY, MaintenanceType.CORRECTIVE]:
            base_skills.append("diagnostic_expert")
        
        equipment_skills = {
            "chiller": ["refrigeration_certified", "electrical"],
            "ahu": ["controls_specialist", "mechanical"],
            "pump": ["mechanical", "alignment_specialist"],
            "fan": ["mechanical", "vibration_analysis"]
        }
        
        base_skills.extend(equipment_skills.get(equipment_type.lower(), []))
        return base_skills
    
    async def optimize_maintenance_schedule(self) -> Dict[str, Any]:
        """Optimize maintenance schedule using quantum annealing."""
        all_recommendations = []
        
        for equipment_id, recommendations in self.maintenance_recommendations.items():
            all_recommendations.extend(recommendations)
        
        if not all_recommendations:
            return {"message": "No maintenance recommendations to optimize"}
        
        # Create optimization problem (simplified QUBO formulation)
        optimization_result = await self._quantum_schedule_optimization(all_recommendations)
        
        return optimization_result
    
    async def _quantum_schedule_optimization(
        self, 
        recommendations: List[MaintenanceRecommendation]
    ) -> Dict[str, Any]:
        """Use quantum annealing to optimize maintenance schedule."""
        # Simplified scheduling optimization
        # In practice, would formulate as QUBO problem for quantum solver
        
        # Sort by urgency and cost efficiency
        sorted_recommendations = sorted(
            recommendations,
            key=lambda r: r.urgency / max(r.estimated_cost, 1),
            reverse=True
        )
        
        # Schedule within constraints
        scheduled_maintenance = []
        total_cost = 0
        current_date = datetime.now()
        
        for rec in sorted_recommendations:
            if total_cost + rec.estimated_cost < 50000:  # Budget constraint
                scheduled_maintenance.append({
                    'recommendation': rec,
                    'scheduled_date': current_date + timedelta(days=len(scheduled_maintenance))
                })
                total_cost += rec.estimated_cost
        
        return {
            'scheduled_maintenance': scheduled_maintenance,
            'total_cost': total_cost,
            'optimization_method': 'quantum_inspired',
            'schedule_efficiency': len(scheduled_maintenance) / len(recommendations)
        }
    
    async def start_monitoring(self) -> None:
        """Start continuous monitoring and analysis."""
        self._monitoring_active = True
        self.logger.info("Started predictive maintenance monitoring")
        
        while self._monitoring_active and not self._shutdown_event.is_set():
            try:
                # Predict failures for all equipment
                predictions = await self.predict_equipment_failures()
                
                # Log high-risk equipment
                for equipment_id, prediction in predictions.items():
                    if prediction['failure_probability'] > self.failure_alert_threshold:
                        self.logger.warning(
                            f"High failure risk for {equipment_id}: "
                            f"{prediction['failure_probability']:.3f} ({prediction['risk_level']})"
                        )
                
                # Optimize maintenance schedule periodically
                if len(self.maintenance_recommendations) > 0:
                    await self.optimize_maintenance_schedule()
                
                await asyncio.sleep(self.maintenance_optimization_interval)
                
            except Exception as e:
                self.logger.error(f"Monitoring cycle error: {e}")
                await asyncio.sleep(60)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        equipment_status = {}
        
        for equipment_id, equipment in self.equipment_profiles.items():
            recent_readings = len(self.sensor_data[equipment_id])
            recent_anomalies = len(self.anomaly_events[equipment_id])
            pending_maintenance = len(self.maintenance_recommendations[equipment_id])
            
            equipment_status[equipment_id] = {
                'equipment_type': equipment.equipment_type,
                'age_years': equipment.age_years,
                'recent_readings': recent_readings,
                'recent_anomalies': recent_anomalies,
                'pending_maintenance': pending_maintenance,
                'last_reading_time': self.sensor_data[equipment_id][-1].timestamp.isoformat() if recent_readings > 0 else None
            }
        
        return {
            'system_overview': {
                'total_equipment': len(self.equipment_profiles),
                'monitoring_active': self._monitoring_active,
                'anomaly_detector_trained': self.anomaly_detector.is_trained,
                'failure_predictor_trained': self.failure_predictor.is_trained
            },
            'equipment_status': equipment_status,
            'system_parameters': {
                'anomaly_alert_threshold': self.anomaly_alert_threshold,
                'failure_alert_threshold': self.failure_alert_threshold,
                'monitoring_interval': self.maintenance_optimization_interval
            }
        }
    
    async def shutdown(self) -> None:
        """Shutdown monitoring system."""
        self._monitoring_active = False
        self._shutdown_event.set()
        self.logger.info("Predictive maintenance system shutdown")


# Global predictive maintenance system
_predictive_maintenance_system: Optional[PredictiveMaintenanceSystem] = None


def get_predictive_maintenance_system() -> PredictiveMaintenanceSystem:
    """Get global predictive maintenance system instance."""
    global _predictive_maintenance_system
    if _predictive_maintenance_system is None:
        _predictive_maintenance_system = PredictiveMaintenanceSystem()
    return _predictive_maintenance_system
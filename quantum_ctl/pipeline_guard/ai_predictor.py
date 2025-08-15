"""
AI-powered predictive failure detection for quantum HVAC pipeline guard.
"""

import numpy as np
import time
import pickle
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import deque, defaultdict
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


@dataclass
class FailurePrediction:
    component: str
    probability: float
    confidence: float
    time_to_failure_hours: Optional[float]
    contributing_factors: List[str]
    recommended_actions: List[str]
    timestamp: float


@dataclass
class AnomalyDetection:
    component: str
    anomaly_score: float
    features: Dict[str, float]
    threshold: float
    is_anomaly: bool
    timestamp: float


class AIPredictor:
    """
    AI-powered predictive system for quantum HVAC pipeline failures.
    Uses machine learning to predict failures before they occur.
    """
    
    def __init__(self, model_update_interval: int = 3600):
        self.model_update_interval = model_update_interval
        
        # Models for different components
        self.anomaly_detectors: Dict[str, IsolationForest] = {}
        self.failure_predictors: Dict[str, RandomForestClassifier] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        
        # Data storage
        self.feature_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.failure_history: List[Dict[str, Any]] = []
        self.prediction_history: List[FailurePrediction] = []
        self.anomaly_history: List[AnomalyDetection] = []
        
        # Model metadata
        self.model_training_time: Dict[str, float] = {}
        self.model_performance: Dict[str, Dict[str, float]] = {}
        
        # Feature engineering parameters
        self.feature_windows = [5, 15, 30, 60]  # minutes
        self.last_model_update = 0
        
    async def start_prediction_engine(self):
        """Start the AI prediction engine."""
        asyncio.create_task(self._model_update_loop())
        asyncio.create_task(self._prediction_loop())
        print("AI Prediction Engine started")
        
    async def _model_update_loop(self):
        """Periodically retrain models with new data."""
        while True:
            try:
                await asyncio.sleep(self.model_update_interval)
                await self._update_models()
            except Exception as e:
                print(f"Model update error: {e}")
                
    async def _prediction_loop(self):
        """Continuously generate predictions."""
        while True:
            try:
                await self._generate_predictions()
                await asyncio.sleep(60)  # Predict every minute
            except Exception as e:
                print(f"Prediction loop error: {e}")
                
    def record_metrics(self, component: str, metrics: Dict[str, float]):
        """Record metrics for a component."""
        timestamp = time.time()
        
        # Add timestamp to metrics
        metrics_with_time = {**metrics, 'timestamp': timestamp}
        
        # Store in feature history
        self.feature_history[component].append(metrics_with_time)
        
    def record_failure(
        self,
        component: str,
        failure_type: str,
        metrics_before_failure: Dict[str, float],
        recovery_time: float
    ):
        """Record a failure event for model training."""
        failure_record = {
            'component': component,
            'failure_type': failure_type,
            'metrics_before_failure': metrics_before_failure,
            'recovery_time': recovery_time,
            'timestamp': time.time()
        }
        
        self.failure_history.append(failure_record)
        
        # Keep only recent failures (last 30 days)
        cutoff_time = time.time() - (30 * 24 * 3600)
        self.failure_history = [
            f for f in self.failure_history
            if f['timestamp'] > cutoff_time
        ]
        
    async def _update_models(self):
        """Update machine learning models with new data."""
        print("Updating AI models...")
        
        components = list(self.feature_history.keys())
        
        for component in components:
            try:
                await self._train_anomaly_detector(component)
                await self._train_failure_predictor(component)
            except Exception as e:
                print(f"Error updating models for {component}: {e}")
                
        self.last_model_update = time.time()
        print("AI models updated successfully")
        
    async def _train_anomaly_detector(self, component: str):
        """Train anomaly detection model for a component."""
        if component not in self.feature_history:
            return
            
        # Get recent feature data
        features_data = list(self.feature_history[component])
        if len(features_data) < 100:  # Need minimum data
            return
            
        # Extract features
        feature_matrix = self._extract_features(features_data)
        if feature_matrix.size == 0:
            return
            
        # Initialize or update scaler
        if component not in self.scalers:
            self.scalers[component] = StandardScaler()
            
        # Scale features
        scaled_features = self.scalers[component].fit_transform(feature_matrix)
        
        # Train isolation forest for anomaly detection
        self.anomaly_detectors[component] = IsolationForest(
            contamination=0.1,  # Expect 10% anomalies
            random_state=42,
            n_estimators=100
        )
        
        self.anomaly_detectors[component].fit(scaled_features)
        
        # Record training time
        self.model_training_time[component] = time.time()
        
    async def _train_failure_predictor(self, component: str):
        """Train failure prediction model for a component."""
        # Get failure data for this component
        component_failures = [
            f for f in self.failure_history
            if f['component'] == component
        ]
        
        if len(component_failures) < 10:  # Need minimum failure examples
            return
            
        # Prepare training data
        X, y = self._prepare_failure_training_data(component, component_failures)
        
        if len(X) < 20:  # Need minimum samples
            return
            
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Train Random Forest classifier
        self.failure_predictors[component] = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced'
        )
        
        self.failure_predictors[component].fit(X_train, y_train)
        
        # Evaluate model performance
        train_score = self.failure_predictors[component].score(X_train, y_train)
        test_score = self.failure_predictors[component].score(X_test, y_test)
        
        self.model_performance[component] = {
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'training_samples': len(X_train),
            'last_updated': time.time()
        }
        
    def _extract_features(self, metrics_data: List[Dict[str, float]]) -> np.ndarray:
        """Extract engineered features from raw metrics."""
        if not metrics_data:
            return np.array([])
            
        # Get numeric columns (exclude timestamp)
        numeric_columns = [
            col for col in metrics_data[0].keys()
            if col != 'timestamp' and isinstance(metrics_data[0][col], (int, float))
        ]
        
        if not numeric_columns:
            return np.array([])
            
        features = []
        
        for i in range(len(metrics_data)):
            row_features = []
            
            # Current values
            for col in numeric_columns:
                row_features.append(metrics_data[i].get(col, 0))
                
            # Moving averages for different windows
            for window in self.feature_windows:
                start_idx = max(0, i - window)
                window_data = metrics_data[start_idx:i+1]
                
                for col in numeric_columns:
                    values = [d.get(col, 0) for d in window_data]
                    if values:
                        row_features.extend([
                            np.mean(values),
                            np.std(values),
                            np.max(values),
                            np.min(values)
                        ])
                    else:
                        row_features.extend([0, 0, 0, 0])
                        
            # Trend features (slope of recent values)
            for col in numeric_columns:
                recent_values = [
                    metrics_data[j].get(col, 0)
                    for j in range(max(0, i-10), i+1)
                ]
                if len(recent_values) > 1:
                    x = np.arange(len(recent_values))
                    slope = np.polyfit(x, recent_values, 1)[0]
                    row_features.append(slope)
                else:
                    row_features.append(0)
                    
            features.append(row_features)
            
        return np.array(features)
        
    def _prepare_failure_training_data(
        self,
        component: str,
        failures: List[Dict[str, Any]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data for failure prediction."""
        X = []
        y = []
        
        # Get component metrics history
        metrics_data = list(self.feature_history[component])
        if not metrics_data:
            return np.array([]), np.array([])
            
        # Extract features for all time points
        feature_matrix = self._extract_features(metrics_data)
        if feature_matrix.size == 0:
            return np.array([]), np.array([])
            
        # Create labels based on failure events
        for i, metrics in enumerate(metrics_data):
            timestamp = metrics['timestamp']
            
            # Check if failure occurs within next 2 hours
            failure_within_window = any(
                failure['timestamp'] - timestamp <= 7200  # 2 hours
                and failure['timestamp'] > timestamp
                for failure in failures
            )
            
            X.append(feature_matrix[i])
            y.append(1 if failure_within_window else 0)
            
        return np.array(X), np.array(y)
        
    async def _generate_predictions(self):
        """Generate predictions for all components."""
        for component in self.feature_history.keys():
            try:
                # Anomaly detection
                anomaly = await self._detect_anomaly(component)
                if anomaly:
                    self.anomaly_history.append(anomaly)
                    
                # Failure prediction
                prediction = await self._predict_failure(component)
                if prediction:
                    self.prediction_history.append(prediction)
                    
            except Exception as e:
                print(f"Error generating predictions for {component}: {e}")
                
        # Keep only recent predictions
        cutoff_time = time.time() - (7 * 24 * 3600)  # 7 days
        self.prediction_history = [
            p for p in self.prediction_history
            if p.timestamp > cutoff_time
        ]
        self.anomaly_history = [
            a for a in self.anomaly_history
            if a.timestamp > cutoff_time
        ]
        
    async def _detect_anomaly(self, component: str) -> Optional[AnomalyDetection]:
        """Detect anomalies for a component."""
        if component not in self.anomaly_detectors:
            return None
            
        # Get recent metrics
        recent_metrics = list(self.feature_history[component])
        if not recent_metrics:
            return None
            
        # Extract features for latest data point
        latest_features = self._extract_features(recent_metrics[-10:])
        if latest_features.size == 0:
            return None
            
        # Scale features
        if component in self.scalers:
            scaled_features = self.scalers[component].transform(latest_features[-1:])
        else:
            return None
            
        # Detect anomaly
        anomaly_score = self.anomaly_detectors[component].decision_function(scaled_features)[0]
        is_anomaly = self.anomaly_detectors[component].predict(scaled_features)[0] == -1
        
        threshold = 0.0  # Isolation Forest threshold
        
        if is_anomaly or anomaly_score < -0.1:  # More sensitive threshold
            return AnomalyDetection(
                component=component,
                anomaly_score=anomaly_score,
                features=recent_metrics[-1],
                threshold=threshold,
                is_anomaly=is_anomaly,
                timestamp=time.time()
            )
            
        return None
        
    async def _predict_failure(self, component: str) -> Optional[FailurePrediction]:
        """Predict failure probability for a component."""
        if component not in self.failure_predictors:
            return None
            
        # Get recent metrics
        recent_metrics = list(self.feature_history[component])
        if not recent_metrics:
            return None
            
        # Extract features for latest data point
        latest_features = self._extract_features(recent_metrics[-10:])
        if latest_features.size == 0:
            return None
            
        # Predict failure probability
        failure_prob = self.failure_predictors[component].predict_proba(latest_features[-1:])
        
        if failure_prob.shape[1] > 1:  # Binary classification
            failure_probability = failure_prob[0][1]  # Probability of failure
        else:
            return None
            
        # Only report if probability is significant
        if failure_probability > 0.3:
            # Get feature importance for explanations
            feature_importance = self.failure_predictors[component].feature_importances_
            contributing_factors = self._get_contributing_factors(
                recent_metrics[-1], feature_importance
            )
            
            # Estimate time to failure (simplified)
            time_to_failure = self._estimate_time_to_failure(
                component, failure_probability
            )
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                component, failure_probability, contributing_factors
            )
            
            return FailurePrediction(
                component=component,
                probability=failure_probability,
                confidence=self._calculate_confidence(component),
                time_to_failure_hours=time_to_failure,
                contributing_factors=contributing_factors,
                recommended_actions=recommendations,
                timestamp=time.time()
            )
            
        return None
        
    def _get_contributing_factors(
        self,
        latest_metrics: Dict[str, float],
        feature_importance: np.ndarray
    ) -> List[str]:
        """Identify contributing factors to failure prediction."""
        # Simplified - would need proper feature name mapping
        metric_names = [
            key for key in latest_metrics.keys()
            if key != 'timestamp' and isinstance(latest_metrics[key], (int, float))
        ]
        
        # Get top contributing metrics
        if len(feature_importance) >= len(metric_names):
            top_indices = np.argsort(feature_importance)[-3:][::-1]
            contributing_factors = [
                metric_names[i % len(metric_names)]
                for i in top_indices
                if i < len(metric_names)
            ]
        else:
            contributing_factors = metric_names[:3]
            
        return contributing_factors[:3]  # Top 3 factors
        
    def _estimate_time_to_failure(self, component: str, probability: float) -> Optional[float]:
        """Estimate time to failure based on probability."""
        # Simplified estimation - could be improved with more sophisticated modeling
        if probability > 0.8:
            return 2.0  # 2 hours
        elif probability > 0.6:
            return 8.0  # 8 hours
        elif probability > 0.4:
            return 24.0  # 24 hours
        else:
            return 72.0  # 72 hours
            
    def _generate_recommendations(
        self,
        component: str,
        probability: float,
        factors: List[str]
    ) -> List[str]:
        """Generate recommendations based on failure prediction."""
        recommendations = []
        
        if probability > 0.8:
            recommendations.append("Immediate attention required - consider emergency recovery")
            
        if probability > 0.6:
            recommendations.append("Schedule preventive maintenance within next 4 hours")
            
        # Component-specific recommendations
        if component == "quantum_solver":
            if "chain_break_fraction" in factors:
                recommendations.append("Adjust quantum embedding parameters")
            recommendations.append("Consider switching to hybrid solver")
            
        elif component == "hvac_controller":
            if "optimization_duration" in factors:
                recommendations.append("Implement problem decomposition")
            recommendations.append("Check building sensor connectivity")
            
        elif component == "bms_connector":
            recommendations.append("Verify network connectivity to BMS")
            recommendations.append("Check authentication credentials")
            
        if not recommendations:
            recommendations.append("Monitor closely and prepare for potential recovery")
            
        return recommendations
        
    def _calculate_confidence(self, component: str) -> float:
        """Calculate confidence in the prediction."""
        if component not in self.model_performance:
            return 0.5  # Default neutral confidence
            
        performance = self.model_performance[component]
        test_accuracy = performance.get('test_accuracy', 0.5)
        training_samples = performance.get('training_samples', 0)
        
        # Confidence based on model accuracy and training data size
        confidence = test_accuracy * min(1.0, training_samples / 100)
        
        return max(0.1, min(0.95, confidence))
        
    def get_prediction_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get summary of recent predictions."""
        cutoff_time = time.time() - (hours * 3600)
        
        recent_predictions = [
            p for p in self.prediction_history
            if p.timestamp > cutoff_time
        ]
        
        recent_anomalies = [
            a for a in self.anomaly_history
            if a.timestamp > cutoff_time
        ]
        
        # High-risk components
        high_risk_components = [
            p.component for p in recent_predictions
            if p.probability > 0.6
        ]
        
        return {
            "time_period_hours": hours,
            "total_predictions": len(recent_predictions),
            "total_anomalies": len(recent_anomalies),
            "high_risk_components": list(set(high_risk_components)),
            "average_failure_probability": (
                sum(p.probability for p in recent_predictions) / len(recent_predictions)
                if recent_predictions else 0
            ),
            "model_status": {
                component: {
                    "last_trained": self.model_training_time.get(component, 0),
                    "performance": self.model_performance.get(component, {}),
                    "data_points": len(self.feature_history.get(component, []))
                }
                for component in self.feature_history.keys()
            },
            "timestamp": time.time()
        }
        
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get active AI-generated alerts."""
        current_time = time.time()
        alerts = []
        
        # Recent high-probability predictions
        for prediction in self.prediction_history[-10:]:
            if (current_time - prediction.timestamp < 3600 and  # Last hour
                prediction.probability > 0.5):
                alerts.append({
                    "type": "failure_prediction",
                    "component": prediction.component,
                    "severity": "high" if prediction.probability > 0.7 else "medium",
                    "message": f"Failure predicted with {prediction.probability:.1%} probability",
                    "time_to_failure": prediction.time_to_failure_hours,
                    "recommendations": prediction.recommended_actions,
                    "timestamp": prediction.timestamp
                })
                
        # Recent anomalies
        for anomaly in self.anomaly_history[-10:]:
            if current_time - anomaly.timestamp < 1800:  # Last 30 minutes
                alerts.append({
                    "type": "anomaly_detection",
                    "component": anomaly.component,
                    "severity": "medium",
                    "message": f"Anomaly detected (score: {anomaly.anomaly_score:.3f})",
                    "anomaly_score": anomaly.anomaly_score,
                    "timestamp": anomaly.timestamp
                })
                
        return sorted(alerts, key=lambda x: x["timestamp"], reverse=True)
"""
Quantum-Specific Health Monitoring and Circuit Breaker System.

This module provides specialized health monitoring for quantum annealing systems,
including chain break detection, embedding quality monitoring, and quantum
advantage tracking with automatic fallback to classical methods.
"""

from typing import Dict, Any, List, Optional, Callable, Tuple
import numpy as np
from dataclasses import dataclass, field
import logging
import time
import asyncio
from collections import deque, defaultdict
from datetime import datetime, timedelta
from enum import Enum
import statistics
import json

from .monitoring import HealthMonitor, CircuitBreaker
from ..optimization.adaptive_quantum_engine import QuantumPerformanceMetrics


class QuantumHealthStatus(Enum):
    """Quantum-specific health status levels."""
    OPTIMAL = "optimal"           # <5% chain breaks, >95% embedding quality
    GOOD = "good"                 # <10% chain breaks, >90% embedding quality  
    DEGRADED = "degraded"         # <20% chain breaks, >80% embedding quality
    POOR = "poor"                 # <30% chain breaks, >70% embedding quality
    CRITICAL = "critical"         # >30% chain breaks or <70% embedding quality
    OFFLINE = "offline"           # No quantum access available


@dataclass
class QuantumMetrics:
    """Comprehensive quantum performance metrics."""
    chain_break_fraction: float = 0.0
    embedding_quality: float = 0.0
    quantum_advantage_score: float = 0.0
    coherence_time: float = 0.0
    solution_quality: float = 0.0
    qpu_access_time: float = 0.0
    annealing_time: float = 0.0
    
    # Trend metrics
    chain_break_trend: float = 0.0      # Change over last 10 solves
    quality_trend: float = 0.0          # Quality improvement/degradation trend
    performance_stability: float = 0.0  # Variance in recent performance
    
    @property
    def overall_health_score(self) -> float:
        """Calculate overall quantum health score (0-1)."""
        # Weighted combination of key metrics
        return (
            0.3 * (1.0 - self.chain_break_fraction) +
            0.25 * self.embedding_quality +
            0.25 * self.quantum_advantage_score +
            0.2 * self.solution_quality
        )


@dataclass
class QuantumAlert:
    """Quantum-specific alert information."""
    alert_type: str
    severity: str  # info, warning, error, critical
    message: str
    metrics: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    suggested_action: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary for serialization."""
        return {
            'alert_type': self.alert_type,
            'severity': self.severity,
            'message': self.message,
            'metrics': self.metrics,
            'timestamp': self.timestamp.isoformat(),
            'suggested_action': self.suggested_action
        }


class QuantumCircuitBreaker(CircuitBreaker):
    """Enhanced circuit breaker with quantum-specific failure detection."""
    
    def __init__(
        self,
        chain_break_threshold: float = 0.3,
        embedding_quality_threshold: float = 0.7,
        failure_threshold: int = 3,
        recovery_timeout: int = 300  # 5 minutes
    ):
        super().__init__(failure_threshold, recovery_timeout)
        
        self.chain_break_threshold = chain_break_threshold
        self.embedding_quality_threshold = embedding_quality_threshold
        
        # Quantum-specific failure tracking
        self._quantum_failures = deque(maxlen=20)
        self._embedding_failures = deque(maxlen=10)
        
        self.logger = logging.getLogger(__name__)
    
    def record_quantum_result(
        self,
        chain_break_fraction: float,
        embedding_quality: float,
        solution_quality: float,
        solve_success: bool
    ) -> bool:
        """
        Record quantum solve result and update circuit breaker state.
        
        Returns:
            True if quantum solving should continue, False if circuit is open
        """
        quantum_failure = (
            not solve_success or
            chain_break_fraction > self.chain_break_threshold or
            embedding_quality < self.embedding_quality_threshold
        )
        
        # Record failure information
        self._quantum_failures.append({
            'timestamp': time.time(),
            'chain_breaks': chain_break_fraction,
            'embedding_quality': embedding_quality,
            'solution_quality': solution_quality,
            'failed': quantum_failure
        })
        
        if quantum_failure:
            self._embedding_failures.append({
                'timestamp': time.time(),
                'chain_breaks': chain_break_fraction,
                'embedding_quality': embedding_quality
            })
            
            self.logger.warning(
                f"Quantum failure detected: chain_breaks={chain_break_fraction:.3f}, "
                f"embedding_quality={embedding_quality:.3f}"
            )
        
        # Update circuit breaker state
        if quantum_failure:
            self._failure_count += 1
            self._last_failure_time = time.time()
            
            if self._failure_count >= self._failure_threshold:
                if self._state != "open":
                    self.logger.error(
                        f"Quantum circuit breaker OPENED after {self._failure_count} failures"
                    )
                    self._state = "open"
                return False
        else:
            # Successful quantum solve
            if self._state == "half_open":
                # Successful recovery
                self.logger.info("Quantum circuit breaker CLOSED - recovery successful")
                self._state = "closed"
                self._failure_count = 0
            elif self._state == "closed":
                # Gradual failure count reduction on success
                self._failure_count = max(0, self._failure_count - 1)
        
        return self._state != "open"
    
    def get_quantum_health_summary(self) -> Dict[str, Any]:
        """Get quantum-specific health summary."""
        if not self._quantum_failures:
            return {'status': 'no_data'}
        
        recent_failures = [f for f in self._quantum_failures if f['failed']]
        
        return {
            'circuit_state': self._state,
            'recent_solves': len(self._quantum_failures),
            'recent_failures': len(recent_failures),
            'failure_rate': len(recent_failures) / len(self._quantum_failures),
            'avg_chain_breaks': statistics.mean([f['chain_breaks'] for f in self._quantum_failures]),
            'avg_embedding_quality': statistics.mean([f['embedding_quality'] for f in self._quantum_failures]),
            'last_failure_time': self._last_failure_time,
            'time_since_last_failure': time.time() - self._last_failure_time if self._last_failure_time else None
        }


class QuantumHealthMonitor:
    """
    Advanced health monitoring system for quantum HVAC optimization.
    
    Features:
    - Real-time quantum performance tracking
    - Predictive failure detection
    - Automatic fallback triggering
    - Performance trend analysis
    - Quantum advantage assessment
    """
    
    def __init__(
        self,
        history_size: int = 1000,
        alert_thresholds: Optional[Dict[str, float]] = None,
        monitoring_interval: int = 60  # seconds
    ):
        self.history_size = history_size
        self.monitoring_interval = monitoring_interval
        
        # Alert thresholds
        self.alert_thresholds = alert_thresholds or {
            'chain_break_warning': 0.15,
            'chain_break_critical': 0.25,
            'embedding_quality_warning': 0.85,
            'embedding_quality_critical': 0.75,
            'quantum_advantage_warning': 0.3,
            'solution_quality_warning': 0.6
        }
        
        self.logger = logging.getLogger(__name__)
        
        # Performance history
        self._performance_history: deque = deque(maxlen=history_size)
        self._solver_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Health tracking
        self._current_health_status = QuantumHealthStatus.GOOD
        self._active_alerts: List[QuantumAlert] = []
        self._alert_history: deque = deque(maxlen=500)
        
        # Circuit breakers for different quantum solvers
        self._circuit_breakers: Dict[str, QuantumCircuitBreaker] = {}
        
        # Trend analysis
        self._trend_window = 20  # Number of recent solves to analyze for trends
        
        # Performance prediction
        self._performance_predictor = QuantumPerformancePredictor()
        
        # Monitoring task
        self._monitoring_task: Optional[asyncio.Task] = None
        self._monitoring_active = False
    
    def register_solver(
        self,
        solver_name: str,
        chain_break_threshold: float = 0.25,
        embedding_quality_threshold: float = 0.8
    ) -> None:
        """Register a quantum solver for monitoring."""
        self._circuit_breakers[solver_name] = QuantumCircuitBreaker(
            chain_break_threshold=chain_break_threshold,
            embedding_quality_threshold=embedding_quality_threshold
        )
        
        self.logger.info(f"Registered quantum solver for monitoring: {solver_name}")
    
    def record_quantum_solve(
        self,
        solver_name: str,
        performance_metrics: QuantumPerformanceMetrics,
        solve_time: float,
        solution_quality: float,
        problem_size: int,
        success: bool = True
    ) -> bool:
        """
        Record quantum solve performance and update health status.
        
        Returns:
            True if quantum solving should continue, False if fallback needed
        """
        timestamp = time.time()
        
        # Create comprehensive metrics record
        metrics_record = {
            'timestamp': timestamp,
            'solver_name': solver_name,
            'chain_break_fraction': performance_metrics.chain_break_fraction,
            'embedding_quality': performance_metrics.embedding_quality,
            'quantum_advantage_score': performance_metrics.quantum_advantage_score,
            'solution_diversity': performance_metrics.solution_diversity,
            'constraint_satisfaction': performance_metrics.constraint_satisfaction,
            'solve_time': solve_time,
            'solution_quality': solution_quality,
            'problem_size': problem_size,
            'success': success
        }
        
        # Add to performance history
        self._performance_history.append(metrics_record)
        self._solver_metrics[solver_name].append(metrics_record)
        
        # Update circuit breaker
        should_continue = True
        if solver_name in self._circuit_breakers:
            should_continue = self._circuit_breakers[solver_name].record_quantum_result(
                performance_metrics.chain_break_fraction,
                performance_metrics.embedding_quality,
                solution_quality,
                success
            )
        
        # Analyze performance and generate alerts
        self._analyze_performance()
        self._update_health_status()
        
        # Check for immediate alerts
        self._check_immediate_alerts(metrics_record)
        
        self.logger.debug(
            f"Recorded quantum solve: {solver_name}, "
            f"health={self._current_health_status.value}, "
            f"continue={should_continue}"
        )
        
        return should_continue
    
    def _analyze_performance(self) -> None:
        """Analyze recent performance trends and patterns."""
        if len(self._performance_history) < 5:
            return
        
        # Get recent performance data
        recent_data = list(self._performance_history)[-self._trend_window:]
        
        # Calculate trends
        chain_breaks = [d['chain_break_fraction'] for d in recent_data]
        embedding_qualities = [d['embedding_quality'] for d in recent_data]
        quantum_advantages = [d['quantum_advantage_score'] for d in recent_data]
        
        # Trend analysis (simple linear trend)
        if len(chain_breaks) >= 5:
            chain_break_trend = self._calculate_trend(chain_breaks)
            embedding_trend = self._calculate_trend(embedding_qualities)
            advantage_trend = self._calculate_trend(quantum_advantages)
            
            # Check for concerning trends
            if chain_break_trend > 0.02:  # Chain breaks increasing by >2% per solve
                self._add_alert(QuantumAlert(
                    alert_type="performance_degradation",
                    severity="warning",
                    message=f"Chain break fraction trending upward: {chain_break_trend:.3f} per solve",
                    metrics={'chain_break_trend': chain_break_trend},
                    suggested_action="Check quantum processor status and consider embedding re-optimization"
                ))
            
            if embedding_trend < -0.01:  # Embedding quality degrading
                self._add_alert(QuantumAlert(
                    alert_type="embedding_degradation",
                    severity="warning", 
                    message=f"Embedding quality trending downward: {embedding_trend:.3f} per solve",
                    metrics={'embedding_trend': embedding_trend},
                    suggested_action="Trigger embedding re-optimization or switch to different quantum solver"
                ))
            
            if advantage_trend < -0.02:  # Quantum advantage declining
                self._add_alert(QuantumAlert(
                    alert_type="quantum_advantage_loss",
                    severity="info",
                    message=f"Quantum advantage declining: {advantage_trend:.3f} per solve",
                    metrics={'advantage_trend': advantage_trend},
                    suggested_action="Consider classical-quantum hybrid approach or problem decomposition"
                ))
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate simple linear trend (slope) of values."""
        if len(values) < 2:
            return 0.0
        
        n = len(values)
        x = np.arange(n)
        y = np.array(values)
        
        # Simple linear regression slope
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        
        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean) ** 2)
        
        return numerator / denominator if denominator > 0 else 0.0
    
    def _update_health_status(self) -> None:
        """Update overall quantum health status based on recent performance."""
        if not self._performance_history:
            self._current_health_status = QuantumHealthStatus.OFFLINE
            return
        
        # Get recent metrics
        recent_metrics = list(self._performance_history)[-10:]  # Last 10 solves
        
        avg_chain_breaks = statistics.mean([m['chain_break_fraction'] for m in recent_metrics])
        avg_embedding_quality = statistics.mean([m['embedding_quality'] for m in recent_metrics])
        success_rate = sum(1 for m in recent_metrics if m['success']) / len(recent_metrics)
        
        # Determine health status
        if success_rate < 0.5:
            new_status = QuantumHealthStatus.CRITICAL
        elif avg_chain_breaks > 0.3 or avg_embedding_quality < 0.7:
            new_status = QuantumHealthStatus.CRITICAL
        elif avg_chain_breaks > 0.2 or avg_embedding_quality < 0.8:
            new_status = QuantumHealthStatus.POOR
        elif avg_chain_breaks > 0.1 or avg_embedding_quality < 0.9:
            new_status = QuantumHealthStatus.DEGRADED
        elif avg_chain_breaks > 0.05 or avg_embedding_quality < 0.95:
            new_status = QuantumHealthStatus.GOOD
        else:
            new_status = QuantumHealthStatus.OPTIMAL
        
        # Check for status changes
        if new_status != self._current_health_status:
            self.logger.info(f"Quantum health status changed: {self._current_health_status.value} → {new_status.value}")
            
            # Generate status change alert
            self._add_alert(QuantumAlert(
                alert_type="health_status_change",
                severity="info" if new_status.value in ['optimal', 'good'] else "warning",
                message=f"Quantum health status changed to {new_status.value}",
                metrics={
                    'previous_status': self._current_health_status.value,
                    'new_status': new_status.value,
                    'avg_chain_breaks': avg_chain_breaks,
                    'avg_embedding_quality': avg_embedding_quality,
                    'success_rate': success_rate
                }
            ))
            
            self._current_health_status = new_status
    
    def _check_immediate_alerts(self, metrics_record: Dict[str, Any]) -> None:
        """Check for immediate alerts based on current solve metrics."""
        chain_breaks = metrics_record['chain_break_fraction']
        embedding_quality = metrics_record['embedding_quality']
        quantum_advantage = metrics_record['quantum_advantage_score']
        solution_quality = metrics_record['solution_quality']
        
        # Chain break alerts
        if chain_breaks > self.alert_thresholds['chain_break_critical']:
            self._add_alert(QuantumAlert(
                alert_type="high_chain_breaks",
                severity="critical",
                message=f"Critical chain break fraction: {chain_breaks:.3f}",
                metrics={'chain_break_fraction': chain_breaks},
                suggested_action="Switch to classical fallback or re-optimize embedding"
            ))
        elif chain_breaks > self.alert_thresholds['chain_break_warning']:
            self._add_alert(QuantumAlert(
                alert_type="elevated_chain_breaks",
                severity="warning",
                message=f"Elevated chain break fraction: {chain_breaks:.3f}",
                metrics={'chain_break_fraction': chain_breaks},
                suggested_action="Monitor quantum processor performance and consider parameter adjustment"
            ))
        
        # Embedding quality alerts
        if embedding_quality < self.alert_thresholds['embedding_quality_critical']:
            self._add_alert(QuantumAlert(
                alert_type="poor_embedding_quality",
                severity="critical", 
                message=f"Poor embedding quality: {embedding_quality:.3f}",
                metrics={'embedding_quality': embedding_quality},
                suggested_action="Trigger embedding re-optimization immediately"
            ))
        elif embedding_quality < self.alert_thresholds['embedding_quality_warning']:
            self._add_alert(QuantumAlert(
                alert_type="degraded_embedding",
                severity="warning",
                message=f"Degraded embedding quality: {embedding_quality:.3f}",
                metrics={'embedding_quality': embedding_quality}, 
                suggested_action="Schedule embedding optimization for next solve"
            ))
        
        # Quantum advantage alerts
        if quantum_advantage < self.alert_thresholds['quantum_advantage_warning']:
            self._add_alert(QuantumAlert(
                alert_type="low_quantum_advantage",
                severity="info",
                message=f"Low quantum advantage score: {quantum_advantage:.3f}",
                metrics={'quantum_advantage_score': quantum_advantage},
                suggested_action="Consider hybrid classical-quantum approach or problem reformulation"
            ))
    
    def _add_alert(self, alert: QuantumAlert) -> None:
        """Add alert to active alerts and history."""
        # Avoid duplicate alerts (same type within 5 minutes)
        recent_similar = [
            a for a in self._active_alerts
            if a.alert_type == alert.alert_type and
            (datetime.now() - a.timestamp).total_seconds() < 300
        ]
        
        if not recent_similar:
            self._active_alerts.append(alert)
            self._alert_history.append(alert)
            
            self.logger.log(
                logging.ERROR if alert.severity == 'critical' else
                logging.WARNING if alert.severity == 'warning' else
                logging.INFO,
                f"Quantum Alert [{alert.severity.upper()}]: {alert.message}"
            )
    
    def start_monitoring(self) -> None:
        """Start background monitoring task."""
        if not self._monitoring_active:
            self._monitoring_active = True
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())
            self.logger.info("Started quantum health monitoring")
    
    def stop_monitoring(self) -> None:
        """Stop background monitoring task."""
        self._monitoring_active = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
        self.logger.info("Stopped quantum health monitoring")
    
    async def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        while self._monitoring_active:
            try:
                # Cleanup old alerts
                cutoff_time = datetime.now() - timedelta(hours=1)
                self._active_alerts = [
                    alert for alert in self._active_alerts
                    if alert.timestamp > cutoff_time
                ]
                
                # Run performance prediction
                if len(self._performance_history) >= 20:
                    prediction = self._performance_predictor.predict_performance(
                        list(self._performance_history)[-20:]
                    )
                    
                    if prediction.get('degradation_risk', 0) > 0.7:
                        self._add_alert(QuantumAlert(
                            alert_type="predicted_degradation",
                            severity="warning",
                            message=f"Predicted performance degradation risk: {prediction['degradation_risk']:.2f}",
                            metrics=prediction,
                            suggested_action="Consider proactive optimization parameter adjustment"
                        ))
                
                await asyncio.sleep(self.monitoring_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in quantum monitoring loop: {e}")
                await asyncio.sleep(self.monitoring_interval)
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive quantum health status."""
        # Calculate current performance metrics
        current_metrics = QuantumMetrics()
        
        if self._performance_history:
            recent_data = list(self._performance_history)[-10:]
            current_metrics.chain_break_fraction = statistics.mean([d['chain_break_fraction'] for d in recent_data])
            current_metrics.embedding_quality = statistics.mean([d['embedding_quality'] for d in recent_data])
            current_metrics.quantum_advantage_score = statistics.mean([d['quantum_advantage_score'] for d in recent_data])
            current_metrics.solution_quality = statistics.mean([d['solution_quality'] for d in recent_data])
        
        # Calculate trends if enough data
        trends = {}
        if len(self._performance_history) >= 10:
            recent_10 = list(self._performance_history)[-10:]
            trends = {
                'chain_break_trend': self._calculate_trend([d['chain_break_fraction'] for d in recent_10]),
                'embedding_quality_trend': self._calculate_trend([d['embedding_quality'] for d in recent_10]),
                'quantum_advantage_trend': self._calculate_trend([d['quantum_advantage_score'] for d in recent_10])
            }
        
        # Circuit breaker status
        circuit_status = {}
        for solver_name, breaker in self._circuit_breakers.items():
            circuit_status[solver_name] = breaker.get_quantum_health_summary()
        
        return {
            'overall_status': self._current_health_status.value,
            'health_score': current_metrics.overall_health_score,
            'current_metrics': {
                'chain_break_fraction': current_metrics.chain_break_fraction,
                'embedding_quality': current_metrics.embedding_quality,
                'quantum_advantage_score': current_metrics.quantum_advantage_score,
                'solution_quality': current_metrics.solution_quality
            },
            'trends': trends,
            'active_alerts': len(self._active_alerts),
            'recent_alert_types': list(set(alert.alert_type for alert in self._active_alerts[-10:])),
            'circuit_breaker_status': circuit_status,
            'solver_metrics': {
                name: len(metrics) for name, metrics in self._solver_metrics.items()
            },
            'performance_history_size': len(self._performance_history),
            'monitoring_active': self._monitoring_active
        }
    
    def get_performance_report(self, hours: int = 24) -> Dict[str, Any]:
        """Generate detailed performance report for specified time period."""
        cutoff_time = time.time() - (hours * 3600)
        recent_data = [
            record for record in self._performance_history
            if record['timestamp'] > cutoff_time
        ]
        
        if not recent_data:
            return {'error': 'No data available for specified time period'}
        
        # Calculate statistics
        chain_breaks = [d['chain_break_fraction'] for d in recent_data]
        embedding_qualities = [d['embedding_quality'] for d in recent_data]
        solve_times = [d['solve_time'] for d in recent_data]
        quantum_advantages = [d['quantum_advantage_score'] for d in recent_data]
        
        report = {
            'time_period_hours': hours,
            'total_solves': len(recent_data),
            'successful_solves': sum(1 for d in recent_data if d['success']),
            'success_rate': sum(1 for d in recent_data if d['success']) / len(recent_data),
            
            'chain_break_statistics': {
                'mean': statistics.mean(chain_breaks),
                'median': statistics.median(chain_breaks),
                'min': min(chain_breaks),
                'max': max(chain_breaks),
                'std_dev': statistics.stdev(chain_breaks) if len(chain_breaks) > 1 else 0
            },
            
            'embedding_quality_statistics': {
                'mean': statistics.mean(embedding_qualities),
                'median': statistics.median(embedding_qualities), 
                'min': min(embedding_qualities),
                'max': max(embedding_qualities),
                'std_dev': statistics.stdev(embedding_qualities) if len(embedding_qualities) > 1 else 0
            },
            
            'solve_time_statistics': {
                'mean': statistics.mean(solve_times),
                'median': statistics.median(solve_times),
                'min': min(solve_times),
                'max': max(solve_times),
                'p95': sorted(solve_times)[int(0.95 * len(solve_times))] if len(solve_times) > 10 else max(solve_times)
            },
            
            'quantum_advantage_statistics': {
                'mean': statistics.mean(quantum_advantages),
                'median': statistics.median(quantum_advantages),
                'positive_advantage_rate': sum(1 for qa in quantum_advantages if qa > 0.5) / len(quantum_advantages)
            },
            
            'alert_summary': {
                'total_alerts': len([a for a in self._alert_history if (datetime.now() - a.timestamp).total_seconds() < hours * 3600]),
                'alerts_by_severity': self._count_alerts_by_severity(hours),
                'most_common_alert_types': self._get_common_alert_types(hours)
            }
        }
        
        return report
    
    def _count_alerts_by_severity(self, hours: int) -> Dict[str, int]:
        """Count alerts by severity for specified time period."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_alerts = [a for a in self._alert_history if a.timestamp > cutoff_time]
        
        severity_counts = defaultdict(int)
        for alert in recent_alerts:
            severity_counts[alert.severity] += 1
        
        return dict(severity_counts)
    
    def _get_common_alert_types(self, hours: int, top_n: int = 5) -> List[Tuple[str, int]]:
        """Get most common alert types for specified time period."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_alerts = [a for a in self._alert_history if a.timestamp > cutoff_time]
        
        type_counts = defaultdict(int)
        for alert in recent_alerts:
            type_counts[alert.alert_type] += 1
        
        # Return top N most common types
        return sorted(type_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    def get_alerts(self, severity: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get current active alerts, optionally filtered by severity."""
        alerts = self._active_alerts
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        return [alert.to_dict() for alert in alerts]
    
    def acknowledge_alert(self, alert_type: str) -> bool:
        """Acknowledge and clear alerts of specified type."""
        initial_count = len(self._active_alerts)
        self._active_alerts = [a for a in self._active_alerts if a.alert_type != alert_type]
        
        removed_count = initial_count - len(self._active_alerts)
        if removed_count > 0:
            self.logger.info(f"Acknowledged {removed_count} alerts of type {alert_type}")
        
        return removed_count > 0
    
    def export_performance_data(self, filename: Optional[str] = None) -> str:
        """Export performance history to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"quantum_performance_{timestamp}.json"
        
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'health_status': self._current_health_status.value,
            'performance_history': list(self._performance_history),
            'alert_history': [alert.to_dict() for alert in self._alert_history],
            'alert_thresholds': self.alert_thresholds,
            'configuration': {
                'history_size': self.history_size,
                'monitoring_interval': self.monitoring_interval
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        self.logger.info(f"Exported quantum performance data to {filename}")
        return filename


class QuantumPerformancePredictor:
    """Predictive analytics for quantum performance degradation."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def predict_performance(self, recent_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Predict future performance based on recent history.
        
        Returns:
            Dictionary with predicted metrics and degradation risk
        """
        if len(recent_history) < 10:
            return {'error': 'Insufficient data for prediction'}
        
        # Extract time series data
        timestamps = [record['timestamp'] for record in recent_history]
        chain_breaks = [record['chain_break_fraction'] for record in recent_history]
        embedding_qualities = [record['embedding_quality'] for record in recent_history]
        quantum_advantages = [record['quantum_advantage_score'] for record in recent_history]
        
        # Simple linear prediction (can be enhanced with ML models)
        def predict_next_value(values: List[float], horizon: int = 1) -> float:
            if len(values) < 3:
                return values[-1] if values else 0.0
            
            # Linear regression for trend prediction
            x = np.arange(len(values))
            y = np.array(values)
            
            # Fit linear trend
            slope, intercept = np.polyfit(x, y, 1)
            
            # Predict next value(s)
            next_x = len(values) + horizon - 1
            predicted = slope * next_x + intercept
            
            return float(predicted)
        
        # Predict next values
        predicted_chain_breaks = predict_next_value(chain_breaks[-10:])
        predicted_embedding_quality = predict_next_value(embedding_qualities[-10:])
        predicted_quantum_advantage = predict_next_value(quantum_advantages[-10:])
        
        # Calculate degradation risk
        degradation_factors = []
        
        # Chain break trend risk
        chain_break_trend = np.polyfit(range(len(chain_breaks[-10:])), chain_breaks[-10:], 1)[0]
        if chain_break_trend > 0.01:  # Increasing trend
            degradation_factors.append(min(1.0, chain_break_trend * 20))
        
        # Embedding quality trend risk
        embedding_trend = np.polyfit(range(len(embedding_qualities[-10:])), embedding_qualities[-10:], 1)[0]
        if embedding_trend < -0.01:  # Decreasing trend
            degradation_factors.append(min(1.0, abs(embedding_trend) * 30))
        
        # Absolute value risks
        if predicted_chain_breaks > 0.2:
            degradation_factors.append((predicted_chain_breaks - 0.2) * 2)
        
        if predicted_embedding_quality < 0.8:
            degradation_factors.append((0.8 - predicted_embedding_quality) * 3)
        
        # Overall degradation risk
        degradation_risk = np.mean(degradation_factors) if degradation_factors else 0.0
        degradation_risk = min(1.0, degradation_risk)
        
        return {
            'predicted_chain_breaks': predicted_chain_breaks,
            'predicted_embedding_quality': predicted_embedding_quality,
            'predicted_quantum_advantage': predicted_quantum_advantage,
            'degradation_risk': degradation_risk,
            'degradation_factors': degradation_factors,
            'confidence': max(0.1, min(0.9, len(recent_history) / 20.0)),  # Higher confidence with more data
            'prediction_horizon': 1,  # Number of solves ahead
            'trends': {
                'chain_break_trend': float(chain_break_trend),
                'embedding_quality_trend': float(embedding_trend)
            }
        }


# Global quantum health monitor instance
_global_quantum_monitor: Optional[QuantumHealthMonitor] = None


def get_quantum_health_monitor() -> QuantumHealthMonitor:
    """Get global quantum health monitor instance."""
    global _global_quantum_monitor
    if _global_quantum_monitor is None:
        _global_quantum_monitor = QuantumHealthMonitor()
    return _global_quantum_monitor


def reset_quantum_health_monitor():
    """Reset global quantum health monitor instance."""
    global _global_quantum_monitor
    if _global_quantum_monitor:
        _global_quantum_monitor.stop_monitoring()
    _global_quantum_monitor = None
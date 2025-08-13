"""
Real-time health monitoring dashboard for quantum HVAC system.
"""

import asyncio
import json
import logging
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import threading

logger = logging.getLogger(__name__)

@dataclass
class HealthMetrics:
    """System health metrics."""
    timestamp: float
    system_status: str  # healthy, degraded, critical
    optimization_success_rate: float
    avg_optimization_time: float
    memory_usage_mb: float
    cpu_usage_percent: float
    quantum_solver_status: str
    error_count_last_hour: int
    last_error: Optional[str] = None

class HealthDashboard:
    """Real-time health monitoring dashboard."""
    
    def __init__(self, update_interval: int = 30):
        self.update_interval = update_interval
        self.metrics_history: List[HealthMetrics] = []
        self.is_running = False
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        
        # Counters
        self.optimization_count = 0
        self.optimization_successes = 0
        self.optimization_times = []
        self.errors_last_hour = []
        
        self.logger = logging.getLogger(__name__)
    
    def start_monitoring(self):
        """Start health monitoring."""
        if self.is_running:
            return
        
        self.is_running = True
        self._stop_event.clear()
        
        # Start monitoring thread
        monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        monitor_thread.start()
        
        self.logger.info("Health monitoring started")
    
    def stop_monitoring(self):
        """Stop health monitoring."""
        if not self.is_running:
            return
        
        self.is_running = False
        self._stop_event.set()
        self.logger.info("Health monitoring stopped")
    
    def record_optimization(self, success: bool, duration: float, error: Optional[str] = None):
        """Record optimization attempt."""
        with self._lock:
            self.optimization_count += 1
            if success:
                self.optimization_successes += 1
            
            self.optimization_times.append(duration)
            # Keep only last 100 measurements
            if len(self.optimization_times) > 100:
                self.optimization_times.pop(0)
            
            if error:
                self.errors_last_hour.append({
                    'timestamp': time.time(),
                    'error': error
                })
    
    def record_error(self, error: str):
        """Record system error."""
        with self._lock:
            self.errors_last_hour.append({
                'timestamp': time.time(),
                'error': error
            })
    
    def get_current_metrics(self) -> HealthMetrics:
        """Get current system health metrics."""
        try:
            import psutil
            memory_usage = psutil.virtual_memory().used / 1024 / 1024  # MB
            cpu_usage = psutil.cpu_percent()
        except ImportError:
            memory_usage = 0.0
            cpu_usage = 0.0
        
        with self._lock:
            # Clean old errors (older than 1 hour)
            current_time = time.time()
            hour_ago = current_time - 3600
            self.errors_last_hour = [
                e for e in self.errors_last_hour 
                if e['timestamp'] > hour_ago
            ]
            
            # Calculate success rate
            if self.optimization_count > 0:
                success_rate = self.optimization_successes / self.optimization_count
            else:
                success_rate = 1.0
            
            # Calculate average optimization time
            if self.optimization_times:
                avg_time = sum(self.optimization_times) / len(self.optimization_times)
            else:
                avg_time = 0.0
            
            # Determine system status
            error_count = len(self.errors_last_hour)
            if error_count > 10:
                status = "critical"
            elif error_count > 3 or success_rate < 0.8:
                status = "degraded"
            else:
                status = "healthy"
            
            # Get last error
            last_error = None
            if self.errors_last_hour:
                last_error = self.errors_last_hour[-1]['error']
            
            return HealthMetrics(
                timestamp=current_time,
                system_status=status,
                optimization_success_rate=success_rate,
                avg_optimization_time=avg_time,
                memory_usage_mb=memory_usage,
                cpu_usage_percent=cpu_usage,
                quantum_solver_status=self._get_quantum_status(),
                error_count_last_hour=error_count,
                last_error=last_error
            )
    
    def _get_quantum_status(self) -> str:
        """Get quantum solver status."""
        try:
            # Import here to avoid circular dependencies
            from ..optimization.quantum_solver import DWAVE_AVAILABLE
            if DWAVE_AVAILABLE:
                # Try to ping D-Wave service
                from dwave.cloud import Client
                try:
                    client = Client.from_config()
                    client.close()
                    return "available"
                except Exception:
                    return "unavailable"
            else:
                return "not_configured"
        except Exception:
            return "unknown"
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while not self._stop_event.wait(self.update_interval):
            try:
                metrics = self.get_current_metrics()
                
                with self._lock:
                    self.metrics_history.append(metrics)
                    # Keep only last 24 hours of data (assuming 30-second intervals)
                    max_entries = 24 * 60 * 60 // self.update_interval
                    if len(self.metrics_history) > max_entries:
                        self.metrics_history.pop(0)
                
                # Log critical issues
                if metrics.system_status == "critical":
                    self.logger.error(f"System in critical state: {metrics.error_count_last_hour} errors")
                elif metrics.system_status == "degraded":
                    self.logger.warning(f"System degraded: {metrics.optimization_success_rate:.1%} success rate")
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
    
    def get_health_report(self) -> Dict[str, Any]:
        """Get comprehensive health report."""
        current = self.get_current_metrics()
        
        # Calculate trends if we have enough data
        trends = {}
        if len(self.metrics_history) > 10:
            recent = self.metrics_history[-10:]
            older = self.metrics_history[-20:-10] if len(self.metrics_history) > 20 else recent
            
            trends = {
                'success_rate_trend': (
                    sum(m.optimization_success_rate for m in recent) / len(recent) -
                    sum(m.optimization_success_rate for m in older) / len(older)
                ),
                'avg_time_trend': (
                    sum(m.avg_optimization_time for m in recent) / len(recent) -
                    sum(m.avg_optimization_time for m in older) / len(older)
                ),
                'error_rate_trend': (
                    sum(m.error_count_last_hour for m in recent) / len(recent) -
                    sum(m.error_count_last_hour for m in older) / len(older)
                )
            }
        
        return {
            'current_metrics': asdict(current),
            'trends': trends,
            'total_optimizations': self.optimization_count,
            'monitoring_duration': len(self.metrics_history) * self.update_interval,
            'data_points': len(self.metrics_history)
        }
    
    def export_metrics(self, filepath: str):
        """Export metrics to JSON file."""
        report = self.get_health_report()
        report['export_timestamp'] = datetime.now().isoformat()
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Health metrics exported to {filepath}")

# Global dashboard instance
_dashboard = None

def get_health_dashboard() -> HealthDashboard:
    """Get global health dashboard instance."""
    global _dashboard
    if _dashboard is None:
        _dashboard = HealthDashboard()
        _dashboard.start_monitoring()
    return _dashboard
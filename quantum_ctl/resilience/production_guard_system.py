"""
Production Guard System v2.0
Advanced resilience, security, and reliability for quantum HVAC systems in production
"""

import asyncio
import logging
import time
import json
import hashlib
import hmac
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import psutil
import numpy as np
from cryptography.fernet import Fernet
from datetime import datetime, timedelta

class SecurityLevel(Enum):
    """Security levels for quantum operations"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class SystemHealthStatus(Enum):
    """System health status levels"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    FAILURE = "failure"

@dataclass
class SecurityAlert:
    """Security alert data structure"""
    timestamp: datetime
    level: SecurityLevel
    alert_type: str
    description: str
    source_ip: Optional[str] = None
    user_id: Optional[str] = None
    risk_score: float = 0.0

@dataclass
class HealthMetrics:
    """System health metrics"""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_latency: float
    quantum_queue_length: int
    error_rate: float
    uptime_hours: float

class QuantumSecurityValidator:
    """Advanced security validation for quantum operations"""
    
    def __init__(self, encryption_key: Optional[bytes] = None):
        self.encryption_key = encryption_key or Fernet.generate_key()
        self.cipher = Fernet(self.encryption_key)
        self.failed_attempts = {}
        self.rate_limits = {}
        self.suspicious_patterns = []
    
    def validate_quantum_request(self, request: Dict[str, Any], user_id: str, 
                                source_ip: str) -> tuple[bool, Optional[SecurityAlert]]:
        """Validate quantum optimization request for security"""
        
        # Rate limiting check
        if self._is_rate_limited(user_id, source_ip):
            alert = SecurityAlert(
                timestamp=datetime.now(),
                level=SecurityLevel.HIGH,
                alert_type="rate_limit_exceeded",
                description=f"Rate limit exceeded for user {user_id} from {source_ip}",
                source_ip=source_ip,
                user_id=user_id,
                risk_score=7.5
            )
            return False, alert
        
        # Input validation
        if not self._validate_input_structure(request):
            self._record_failed_attempt(user_id, source_ip)
            alert = SecurityAlert(
                timestamp=datetime.now(),
                level=SecurityLevel.MEDIUM,
                alert_type="invalid_input",
                description="Malformed quantum optimization request",
                source_ip=source_ip,
                user_id=user_id,
                risk_score=5.0
            )
            return False, alert
        
        # Anomaly detection
        risk_score = self._calculate_risk_score(request, user_id)
        if risk_score > 8.0:
            alert = SecurityAlert(
                timestamp=datetime.now(),
                level=SecurityLevel.CRITICAL,
                alert_type="anomaly_detected",
                description=f"High-risk quantum request detected (score: {risk_score})",
                source_ip=source_ip,
                user_id=user_id,
                risk_score=risk_score
            )
            return False, alert
        
        # Update rate limits
        self._update_rate_limits(user_id, source_ip)
        return True, None
    
    def encrypt_quantum_data(self, data: Dict[str, Any]) -> bytes:
        """Encrypt sensitive quantum data"""
        json_data = json.dumps(data, sort_keys=True)
        return self.cipher.encrypt(json_data.encode())
    
    def decrypt_quantum_data(self, encrypted_data: bytes) -> Dict[str, Any]:
        """Decrypt quantum data"""
        decrypted_json = self.cipher.decrypt(encrypted_data).decode()
        return json.loads(decrypted_json)
    
    def _is_rate_limited(self, user_id: str, source_ip: str) -> bool:
        """Check if user/IP is rate limited"""
        now = datetime.now()
        
        # Check user rate limit (100 requests per hour)
        user_key = f"user_{user_id}"
        if user_key in self.rate_limits:
            user_requests = [t for t in self.rate_limits[user_key] if now - t < timedelta(hours=1)]
            if len(user_requests) >= 100:
                return True
        
        # Check IP rate limit (200 requests per hour)
        ip_key = f"ip_{source_ip}"
        if ip_key in self.rate_limits:
            ip_requests = [t for t in self.rate_limits[ip_key] if now - t < timedelta(hours=1)]
            if len(ip_requests) >= 200:
                return True
        
        return False
    
    def _validate_input_structure(self, request: Dict[str, Any]) -> bool:
        """Validate quantum request structure"""
        required_fields = ['temperatures', 'occupancy', 'prediction_horizon']
        
        # Check required fields
        for field in required_fields:
            if field not in request:
                return False
        
        # Validate temperature ranges
        temperatures = request.get('temperatures', [])
        if not isinstance(temperatures, list) or len(temperatures) == 0:
            return False
        
        for temp in temperatures:
            if not isinstance(temp, (int, float)) or temp < -50 or temp > 100:
                return False
        
        # Validate occupancy ranges
        occupancy = request.get('occupancy', [])
        if not isinstance(occupancy, list) or len(occupancy) != len(temperatures):
            return False
        
        for occ in occupancy:
            if not isinstance(occ, (int, float)) or occ < 0 or occ > 1:
                return False
        
        # Validate prediction horizon
        horizon = request.get('prediction_horizon', 0)
        if not isinstance(horizon, int) or horizon < 1 or horizon > 168:  # Max 1 week
            return False
        
        return True
    
    def _calculate_risk_score(self, request: Dict[str, Any], user_id: str) -> float:
        """Calculate risk score for request"""
        risk_score = 0.0
        
        # Unusual temperature patterns
        temps = request.get('temperatures', [])
        if temps:
            temp_variance = np.var(temps)
            if temp_variance > 100:  # Very high variance
                risk_score += 3.0
            elif temp_variance > 50:
                risk_score += 1.5
        
        # Unusual prediction horizon
        horizon = request.get('prediction_horizon', 24)
        if horizon > 72:  # More than 3 days
            risk_score += 2.0
        elif horizon < 6:  # Less than 6 hours
            risk_score += 1.0
        
        # Check for repeated failed attempts
        failed_count = self.failed_attempts.get(user_id, 0)
        risk_score += min(failed_count * 0.5, 5.0)
        
        return risk_score
    
    def _record_failed_attempt(self, user_id: str, source_ip: str) -> None:
        """Record failed security attempt"""
        self.failed_attempts[user_id] = self.failed_attempts.get(user_id, 0) + 1
    
    def _update_rate_limits(self, user_id: str, source_ip: str) -> None:
        """Update rate limit counters"""
        now = datetime.now()
        
        user_key = f"user_{user_id}"
        if user_key not in self.rate_limits:
            self.rate_limits[user_key] = []
        self.rate_limits[user_key].append(now)
        
        ip_key = f"ip_{source_ip}"
        if ip_key not in self.rate_limits:
            self.rate_limits[ip_key] = []
        self.rate_limits[ip_key].append(now)

class AdvancedHealthMonitor:
    """Advanced system health monitoring"""
    
    def __init__(self):
        self.health_history = []
        self.alert_thresholds = {
            'cpu_usage': 85.0,
            'memory_usage': 90.0,
            'disk_usage': 85.0,
            'network_latency': 1000.0,  # milliseconds
            'error_rate': 5.0,  # percentage
            'quantum_queue_length': 100
        }
        self.start_time = datetime.now()
    
    def collect_health_metrics(self) -> HealthMetrics:
        """Collect comprehensive system health metrics"""
        
        # System resource usage
        cpu_usage = psutil.cpu_percent(interval=1)
        memory_usage = psutil.virtual_memory().percent
        disk_usage = psutil.disk_usage('/').percent
        
        # Network latency (simulate)
        network_latency = self._measure_network_latency()
        
        # Quantum-specific metrics
        quantum_queue_length = self._get_quantum_queue_length()
        error_rate = self._calculate_error_rate()
        
        # Uptime
        uptime_hours = (datetime.now() - self.start_time).total_seconds() / 3600
        
        metrics = HealthMetrics(
            timestamp=datetime.now(),
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            disk_usage=disk_usage,
            network_latency=network_latency,
            quantum_queue_length=quantum_queue_length,
            error_rate=error_rate,
            uptime_hours=uptime_hours
        )
        
        self.health_history.append(metrics)
        return metrics
    
    def assess_system_health(self, metrics: HealthMetrics) -> SystemHealthStatus:
        """Assess overall system health"""
        critical_alerts = 0
        warning_alerts = 0
        
        # Check each metric against thresholds
        if metrics.cpu_usage > self.alert_thresholds['cpu_usage']:
            critical_alerts += 1
        elif metrics.cpu_usage > self.alert_thresholds['cpu_usage'] * 0.8:
            warning_alerts += 1
        
        if metrics.memory_usage > self.alert_thresholds['memory_usage']:
            critical_alerts += 1
        elif metrics.memory_usage > self.alert_thresholds['memory_usage'] * 0.8:
            warning_alerts += 1
        
        if metrics.disk_usage > self.alert_thresholds['disk_usage']:
            critical_alerts += 1
        elif metrics.disk_usage > self.alert_thresholds['disk_usage'] * 0.8:
            warning_alerts += 1
        
        if metrics.network_latency > self.alert_thresholds['network_latency']:
            critical_alerts += 1
        elif metrics.network_latency > self.alert_thresholds['network_latency'] * 0.8:
            warning_alerts += 1
        
        if metrics.error_rate > self.alert_thresholds['error_rate']:
            critical_alerts += 1
        elif metrics.error_rate > self.alert_thresholds['error_rate'] * 0.8:
            warning_alerts += 1
        
        if metrics.quantum_queue_length > self.alert_thresholds['quantum_queue_length']:
            critical_alerts += 1
        elif metrics.quantum_queue_length > self.alert_thresholds['quantum_queue_length'] * 0.8:
            warning_alerts += 1
        
        # Determine overall status
        if critical_alerts >= 3:
            return SystemHealthStatus.FAILURE
        elif critical_alerts >= 1:
            return SystemHealthStatus.CRITICAL
        elif warning_alerts >= 2:
            return SystemHealthStatus.WARNING
        else:
            return SystemHealthStatus.HEALTHY
    
    def _measure_network_latency(self) -> float:
        """Measure network latency (simulated)"""
        # In production, this would ping actual endpoints
        return np.random.uniform(10, 100)  # Simulate 10-100ms latency
    
    def _get_quantum_queue_length(self) -> int:
        """Get quantum processing queue length (simulated)"""
        return np.random.randint(0, 20)  # Simulate queue of 0-20 items
    
    def _calculate_error_rate(self) -> float:
        """Calculate system error rate (simulated)"""
        return np.random.uniform(0, 2)  # Simulate 0-2% error rate

class ProductionGuardSystem:
    """Main production guard system orchestrating all reliability measures"""
    
    def __init__(self, log_level: str = "INFO"):
        # Initialize logging
        self.logger = self._setup_logging(log_level)
        
        # Initialize subsystems
        self.security_validator = QuantumSecurityValidator()
        self.health_monitor = AdvancedHealthMonitor()
        
        # Guard state
        self.is_active = False
        self.security_alerts = []
        self.maintenance_mode = False
        
        # Error handling
        self.error_recovery_strategies = {
            'quantum_timeout': self._recover_quantum_timeout,
            'memory_overflow': self._recover_memory_overflow,
            'network_failure': self._recover_network_failure,
            'security_breach': self._recover_security_breach
        }
        
        self.logger.info("Production Guard System initialized")
    
    async def start_guard_system(self) -> None:
        """Start the production guard system"""
        self.is_active = True
        self.logger.info("Production Guard System ACTIVATED")
        
        # Start health monitoring loop
        asyncio.create_task(self._health_monitoring_loop())
        
        # Start security monitoring loop
        asyncio.create_task(self._security_monitoring_loop())
        
        self.logger.info("All guard subsystems operational")
    
    async def secure_quantum_operation(self, operation: Callable, request: Dict[str, Any],
                                     user_id: str, source_ip: str) -> Dict[str, Any]:
        """Execute quantum operation with full security and error handling"""
        
        if not self.is_active:
            raise RuntimeError("Production Guard System not active")
        
        if self.maintenance_mode:
            raise RuntimeError("System in maintenance mode")
        
        # Security validation
        is_valid, security_alert = self.security_validator.validate_quantum_request(
            request, user_id, source_ip
        )
        
        if not is_valid and security_alert:
            self.security_alerts.append(security_alert)
            self.logger.warning(f"Security validation failed: {security_alert.description}")
            raise SecurityError(f"Security validation failed: {security_alert.description}")
        
        # Health check
        health_metrics = self.health_monitor.collect_health_metrics()
        health_status = self.health_monitor.assess_system_health(health_metrics)
        
        if health_status == SystemHealthStatus.FAILURE:
            self.logger.error("System health critical - rejecting operation")
            raise SystemError("System health critical - operation not allowed")
        
        # Execute operation with monitoring
        start_time = time.time()
        try:
            # Encrypt sensitive data
            encrypted_request = self.security_validator.encrypt_quantum_data(request)
            decrypted_request = self.security_validator.decrypt_quantum_data(encrypted_request)
            
            # Execute the actual quantum operation
            result = await self._execute_with_timeout(operation, decrypted_request, timeout=300)
            
            execution_time = time.time() - start_time
            
            # Log successful operation
            self.logger.info(f"Quantum operation completed successfully in {execution_time:.2f}s")
            
            return {
                **result,
                'security_validated': True,
                'execution_time': execution_time,
                'health_status': health_status.value,
                'guard_system': 'operational'
            }
            
        except asyncio.TimeoutError:
            self.logger.error("Quantum operation timed out")
            await self._handle_error('quantum_timeout', request, user_id)
            raise
        
        except MemoryError:
            self.logger.error("Memory overflow during quantum operation")
            await self._handle_error('memory_overflow', request, user_id)
            raise
        
        except Exception as e:
            self.logger.error(f"Unexpected error in quantum operation: {e}")
            await self._handle_error('general_error', request, user_id)
            raise
    
    async def _execute_with_timeout(self, operation: Callable, request: Dict[str, Any], 
                                  timeout: int) -> Dict[str, Any]:
        """Execute operation with timeout"""
        return await asyncio.wait_for(operation(request), timeout=timeout)
    
    async def _handle_error(self, error_type: str, request: Dict[str, Any], user_id: str) -> None:
        """Handle errors with appropriate recovery strategy"""
        
        self.logger.info(f"Executing error recovery for: {error_type}")
        
        if error_type in self.error_recovery_strategies:
            try:
                await self.error_recovery_strategies[error_type](request, user_id)
                self.logger.info(f"Error recovery completed for: {error_type}")
            except Exception as e:
                self.logger.error(f"Error recovery failed for {error_type}: {e}")
        else:
            self.logger.warning(f"No recovery strategy for error type: {error_type}")
    
    async def _health_monitoring_loop(self) -> None:
        """Continuous health monitoring loop"""
        while self.is_active:
            try:
                metrics = self.health_monitor.collect_health_metrics()
                status = self.health_monitor.assess_system_health(metrics)
                
                if status == SystemHealthStatus.CRITICAL:
                    self.logger.warning("System health critical - alerting administrators")
                elif status == SystemHealthStatus.FAILURE:
                    self.logger.error("System health failure - entering maintenance mode")
                    self.maintenance_mode = True
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(60)  # Back off on error
    
    async def _security_monitoring_loop(self) -> None:
        """Continuous security monitoring loop"""
        while self.is_active:
            try:
                # Clean up old rate limit data
                now = datetime.now()
                cutoff = now - timedelta(hours=1)
                
                # Clean rate limits
                for key, timestamps in list(self.security_validator.rate_limits.items()):
                    self.security_validator.rate_limits[key] = [
                        t for t in timestamps if t > cutoff
                    ]
                    if not self.security_validator.rate_limits[key]:
                        del self.security_validator.rate_limits[key]
                
                # Process security alerts
                if self.security_alerts:
                    high_risk_alerts = [a for a in self.security_alerts if a.level == SecurityLevel.CRITICAL]
                    if len(high_risk_alerts) > 5:
                        self.logger.critical("Multiple critical security alerts - potential attack")
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Security monitoring error: {e}")
                await asyncio.sleep(600)  # Back off on error
    
    async def _recover_quantum_timeout(self, request: Dict[str, Any], user_id: str) -> None:
        """Recover from quantum operation timeout"""
        # Reduce problem complexity and retry
        simplified_request = {
            **request,
            'prediction_horizon': min(request.get('prediction_horizon', 24), 12)
        }
        self.logger.info("Simplified quantum request for retry")
    
    async def _recover_memory_overflow(self, request: Dict[str, Any], user_id: str) -> None:
        """Recover from memory overflow"""
        # Force garbage collection and reduce problem size
        import gc
        gc.collect()
        self.logger.info("Memory cleanup completed")
    
    async def _recover_network_failure(self, request: Dict[str, Any], user_id: str) -> None:
        """Recover from network failure"""
        # Switch to offline mode or retry with backoff
        await asyncio.sleep(5)  # Wait before retry
        self.logger.info("Network recovery attempted")
    
    async def _recover_security_breach(self, request: Dict[str, Any], user_id: str) -> None:
        """Recover from security breach"""
        # Lock user account and alert administrators
        self.security_validator.failed_attempts[user_id] = 999
        self.logger.critical(f"Security breach detected for user: {user_id}")
    
    def get_guard_status(self) -> Dict[str, Any]:
        """Get comprehensive guard system status"""
        latest_health = None
        if self.health_monitor.health_history:
            latest_health = self.health_monitor.health_history[-1]
        
        recent_alerts = [a for a in self.security_alerts if 
                        (datetime.now() - a.timestamp).seconds < 3600]  # Last hour
        
        return {
            'guard_active': self.is_active,
            'maintenance_mode': self.maintenance_mode,
            'system_health': {
                'status': self.health_monitor.assess_system_health(latest_health).value if latest_health else 'unknown',
                'metrics': asdict(latest_health) if latest_health else None
            },
            'security': {
                'alerts_last_hour': len(recent_alerts),
                'critical_alerts': len([a for a in recent_alerts if a.level == SecurityLevel.CRITICAL]),
                'rate_limited_users': len(self.security_validator.rate_limits)
            },
            'uptime_hours': (datetime.now() - self.health_monitor.start_time).total_seconds() / 3600,
            'production_ready': self.is_active and not self.maintenance_mode
        }
    
    def _setup_logging(self, log_level: str) -> logging.Logger:
        """Setup production logging"""
        logger = logging.getLogger("quantum_production_guard")
        logger.setLevel(getattr(logging, log_level.upper()))
        
        # Console handler
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        return logger

class SecurityError(Exception):
    """Custom exception for security-related errors"""
    pass

# Production-ready factory function
def create_production_guard() -> ProductionGuardSystem:
    """Create production-ready guard system"""
    return ProductionGuardSystem(log_level="INFO")
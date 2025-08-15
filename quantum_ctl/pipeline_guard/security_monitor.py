"""
Security monitoring for quantum HVAC pipeline guard.
"""

import time
import hashlib
import hmac
import logging
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass
from enum import Enum
import ipaddress
import re


class SecurityEventType(Enum):
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    CONFIGURATION_CHANGE = "configuration_change"
    AUTHENTICATION_FAILURE = "authentication_failure"
    QUANTUM_TAMPERING = "quantum_tampering"
    DATA_INTEGRITY_VIOLATION = "data_integrity_violation"


@dataclass
class SecurityEvent:
    event_type: SecurityEventType
    severity: str  # low, medium, high, critical
    source: str
    message: str
    timestamp: float
    metadata: Dict[str, Any]


class SecurityMonitor:
    """
    Security monitoring system for quantum HVAC pipeline guard.
    Monitors for security threats, unauthorized access, and system tampering.
    """
    
    def __init__(self):
        self.security_events: List[SecurityEvent] = []
        self.failed_auth_attempts: Dict[str, List[float]] = {}
        self.allowed_ips: Set[str] = set()
        self.blocked_ips: Set[str] = set()
        self.quantum_checksums: Dict[str, str] = {}
        self.config_checksums: Dict[str, str] = {}
        self.logger = logging.getLogger(__name__)
        
        # Security thresholds
        self.max_auth_failures = 5
        self.auth_failure_window = 300  # 5 minutes
        self.max_events_per_minute = 50
        
    def configure_allowed_ips(self, ip_ranges: List[str]):
        """Configure allowed IP ranges."""
        self.allowed_ips.clear()
        for ip_range in ip_ranges:
            try:
                # Support both single IPs and CIDR ranges
                if '/' in ip_range:
                    network = ipaddress.ip_network(ip_range, strict=False)
                    self.allowed_ips.update(str(ip) for ip in network.hosts())
                else:
                    self.allowed_ips.add(ip_range)
            except Exception as e:
                self.logger.error(f"Invalid IP range {ip_range}: {e}")
                
    def validate_access(self, source_ip: str, user_id: str, operation: str) -> bool:
        """
        Validate access attempt and log security events.
        Returns True if access should be allowed.
        """
        # Check IP allowlist
        if self.allowed_ips and source_ip not in self.allowed_ips:
            self._log_security_event(
                SecurityEventType.UNAUTHORIZED_ACCESS,
                "high",
                source_ip,
                f"Access from non-allowed IP: {source_ip}",
                {"user_id": user_id, "operation": operation}
            )
            return False
            
        # Check blocked IPs
        if source_ip in self.blocked_ips:
            self._log_security_event(
                SecurityEventType.UNAUTHORIZED_ACCESS,
                "high",
                source_ip,
                f"Access from blocked IP: {source_ip}",
                {"user_id": user_id, "operation": operation}
            )
            return False
            
        # Check for suspicious patterns
        if self._detect_suspicious_activity(source_ip, user_id, operation):
            return False
            
        return True
        
    def record_auth_failure(self, source_ip: str, user_id: str, reason: str):
        """Record authentication failure and check for brute force."""
        current_time = time.time()
        
        # Track failures by IP
        if source_ip not in self.failed_auth_attempts:
            self.failed_auth_attempts[source_ip] = []
            
        self.failed_auth_attempts[source_ip].append(current_time)
        
        # Clean old failures
        cutoff_time = current_time - self.auth_failure_window
        self.failed_auth_attempts[source_ip] = [
            t for t in self.failed_auth_attempts[source_ip]
            if t > cutoff_time
        ]
        
        # Check for brute force
        failure_count = len(self.failed_auth_attempts[source_ip])
        
        self._log_security_event(
            SecurityEventType.AUTHENTICATION_FAILURE,
            "medium" if failure_count < self.max_auth_failures else "high",
            source_ip,
            f"Authentication failure for user {user_id}: {reason}",
            {"user_id": user_id, "failure_count": failure_count}
        )
        
        # Block IP if too many failures
        if failure_count >= self.max_auth_failures:
            self.blocked_ips.add(source_ip)
            self._log_security_event(
                SecurityEventType.UNAUTHORIZED_ACCESS,
                "critical",
                source_ip,
                f"IP blocked due to {failure_count} auth failures",
                {"user_id": user_id, "auto_blocked": True}
            )
            
    def validate_quantum_integrity(self, component: str, data: Any) -> bool:
        """
        Validate integrity of quantum annealing data/results.
        Detects potential tampering with quantum computations.
        """
        try:
            # Generate checksum of quantum data
            data_str = str(data) if not isinstance(data, str) else data
            current_checksum = hashlib.sha256(data_str.encode()).hexdigest()
            
            # Check against stored checksum
            if component in self.quantum_checksums:
                expected_checksum = self.quantum_checksums[component]
                if current_checksum != expected_checksum:
                    self._log_security_event(
                        SecurityEventType.QUANTUM_TAMPERING,
                        "critical",
                        "system",
                        f"Quantum data integrity violation for {component}",
                        {
                            "component": component,
                            "expected_checksum": expected_checksum,
                            "actual_checksum": current_checksum
                        }
                    )
                    return False
                    
            # Store/update checksum
            self.quantum_checksums[component] = current_checksum
            return True
            
        except Exception as e:
            self.logger.error(f"Quantum integrity check failed: {e}")
            self._log_security_event(
                SecurityEventType.DATA_INTEGRITY_VIOLATION,
                "high",
                "system",
                f"Failed to validate quantum integrity: {e}",
                {"component": component}
            )
            return False
            
    def validate_config_integrity(self, config_name: str, config_data: Dict[str, Any]) -> bool:
        """
        Validate integrity of configuration data.
        Detects unauthorized configuration changes.
        """
        try:
            # Generate checksum of configuration
            config_str = str(sorted(config_data.items()))
            current_checksum = hashlib.sha256(config_str.encode()).hexdigest()
            
            # Check against stored checksum
            if config_name in self.config_checksums:
                expected_checksum = self.config_checksums[config_name]
                if current_checksum != expected_checksum:
                    self._log_security_event(
                        SecurityEventType.CONFIGURATION_CHANGE,
                        "high",
                        "system",
                        f"Configuration change detected for {config_name}",
                        {
                            "config_name": config_name,
                            "expected_checksum": expected_checksum,
                            "actual_checksum": current_checksum
                        }
                    )
                    # Allow config changes but log them
                    
            # Store/update checksum
            self.config_checksums[config_name] = current_checksum
            return True
            
        except Exception as e:
            self.logger.error(f"Config integrity check failed: {e}")
            return False
            
    def _detect_suspicious_activity(self, source_ip: str, user_id: str, operation: str) -> bool:
        """Detect suspicious activity patterns."""
        current_time = time.time()
        
        # Check for rapid-fire requests
        recent_events = [
            event for event in self.security_events
            if (event.timestamp > current_time - 60 and  # Last minute
                event.source == source_ip)
        ]
        
        if len(recent_events) > self.max_events_per_minute:
            self._log_security_event(
                SecurityEventType.SUSPICIOUS_ACTIVITY,
                "high",
                source_ip,
                f"Excessive activity: {len(recent_events)} events in last minute",
                {"user_id": user_id, "operation": operation}
            )
            return True
            
        # Check for suspicious operation patterns
        sensitive_operations = [
            "config_change", "user_management", "system_shutdown",
            "quantum_solver_config", "recovery_override"
        ]
        
        if operation in sensitive_operations:
            # Check if multiple sensitive operations from same source
            sensitive_events = [
                event for event in recent_events
                if any(op in event.message.lower() for op in sensitive_operations)
            ]
            
            if len(sensitive_events) > 3:
                self._log_security_event(
                    SecurityEventType.SUSPICIOUS_ACTIVITY,
                    "critical",
                    source_ip,
                    f"Multiple sensitive operations: {operation}",
                    {"user_id": user_id, "sensitive_count": len(sensitive_events)}
                )
                return True
                
        return False
        
    def validate_api_signature(self, payload: str, signature: str, secret: str) -> bool:
        """Validate API request signature for integrity."""
        try:
            expected_signature = hmac.new(
                secret.encode(),
                payload.encode(),
                hashlib.sha256
            ).hexdigest()
            
            if not hmac.compare_digest(signature, expected_signature):
                self._log_security_event(
                    SecurityEventType.DATA_INTEGRITY_VIOLATION,
                    "high",
                    "api",
                    "Invalid API signature",
                    {"signature_provided": signature[:16] + "..."}  # Log partial signature
                )
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Signature validation failed: {e}")
            return False
            
    def sanitize_input(self, input_data: str) -> str:
        """Sanitize input data to prevent injection attacks."""
        if not isinstance(input_data, str):
            return str(input_data)
            
        # Remove potential SQL injection patterns
        sql_patterns = [
            r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER)\b)",
            r"(--|#|/\*|\*/)",
            r"(\bUNION\b|\bOR\b|\bAND\b).*(\b1=1\b|\b'=')"
        ]
        
        for pattern in sql_patterns:
            if re.search(pattern, input_data, re.IGNORECASE):
                self._log_security_event(
                    SecurityEventType.SUSPICIOUS_ACTIVITY,
                    "high",
                    "input_validation",
                    f"Potential injection attack detected",
                    {"pattern": pattern, "input_sample": input_data[:50]}
                )
                # Remove suspicious content
                input_data = re.sub(pattern, "", input_data, flags=re.IGNORECASE)
                
        # Remove control characters and limit length
        sanitized = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', input_data)
        return sanitized[:1000]  # Limit to 1000 chars
        
    def _log_security_event(
        self,
        event_type: SecurityEventType,
        severity: str,
        source: str,
        message: str,
        metadata: Dict[str, Any]
    ):
        """Log a security event."""
        event = SecurityEvent(
            event_type=event_type,
            severity=severity,
            source=source,
            message=message,
            timestamp=time.time(),
            metadata=metadata
        )
        
        self.security_events.append(event)
        
        # Keep only recent events (last 7 days)
        cutoff_time = time.time() - (7 * 24 * 3600)
        self.security_events = [
            e for e in self.security_events
            if e.timestamp > cutoff_time
        ]
        
        # Log to system logger
        log_message = f"SECURITY [{severity.upper()}] {event_type.value}: {message}"
        if severity == "critical":
            self.logger.critical(log_message)
        elif severity == "high":
            self.logger.error(log_message)
        elif severity == "medium":
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)
            
    def get_security_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get security summary for specified time period."""
        cutoff_time = time.time() - (hours * 3600)
        recent_events = [
            e for e in self.security_events
            if e.timestamp > cutoff_time
        ]
        
        # Count by severity
        severity_counts = {"low": 0, "medium": 0, "high": 0, "critical": 0}
        for event in recent_events:
            severity_counts[event.severity] += 1
            
        # Count by type
        type_counts = {}
        for event in recent_events:
            event_type = event.event_type.value
            type_counts[event_type] = type_counts.get(event_type, 0) + 1
            
        # Get recent critical/high events
        important_events = [
            {
                "type": e.event_type.value,
                "severity": e.severity,
                "source": e.source,
                "message": e.message,
                "timestamp": e.timestamp
            }
            for e in recent_events[-20:]  # Last 20 events
            if e.severity in ["high", "critical"]
        ]
        
        return {
            "time_period_hours": hours,
            "total_events": len(recent_events),
            "severity_breakdown": severity_counts,
            "event_type_breakdown": type_counts,
            "blocked_ips_count": len(self.blocked_ips),
            "recent_important_events": important_events,
            "security_status": self._get_security_status(severity_counts)
        }
        
    def _get_security_status(self, severity_counts: Dict[str, int]) -> str:
        """Determine overall security status."""
        if severity_counts["critical"] > 0:
            return "critical"
        elif severity_counts["high"] > 5:
            return "high_alert"
        elif severity_counts["high"] > 0 or severity_counts["medium"] > 10:
            return "elevated"
        else:
            return "normal"
            
    def unblock_ip(self, ip_address: str, reason: str = "manual"):
        """Manually unblock an IP address."""
        if ip_address in self.blocked_ips:
            self.blocked_ips.remove(ip_address)
            self._log_security_event(
                SecurityEventType.CONFIGURATION_CHANGE,
                "medium",
                "admin",
                f"IP {ip_address} unblocked: {reason}",
                {"ip": ip_address, "reason": reason}
            )
            
    def get_blocked_ips(self) -> List[Dict[str, Any]]:
        """Get list of currently blocked IPs with details."""
        blocked_info = []
        for ip in self.blocked_ips:
            # Find related events
            related_events = [
                e for e in self.security_events
                if e.source == ip and e.event_type == SecurityEventType.AUTHENTICATION_FAILURE
            ]
            
            if related_events:
                latest_event = max(related_events, key=lambda x: x.timestamp)
                blocked_info.append({
                    "ip": ip,
                    "blocked_time": latest_event.timestamp,
                    "reason": "authentication_failures",
                    "failure_count": len(related_events)
                })
            else:
                blocked_info.append({
                    "ip": ip,
                    "blocked_time": time.time(),
                    "reason": "unknown",
                    "failure_count": 0
                })
                
        return blocked_info
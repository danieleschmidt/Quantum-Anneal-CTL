"""
Security and authentication utilities for quantum HVAC control.

Provides authentication, authorization, input validation, and security monitoring.
"""

import hashlib
import hmac
import json
import time
import secrets
import logging
from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import re

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64


class UserRole(Enum):
    """User roles for authorization."""
    ADMIN = "admin"
    OPERATOR = "operator"
    READONLY = "readonly"
    SERVICE = "service"


class SecurityLevel(Enum):
    """Security classification levels."""
    PUBLIC = "public"
    INTERNAL = "internal"  
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


@dataclass
class SecurityContext:
    """Security context for requests."""
    user_id: str
    role: UserRole
    permissions: List[str]
    session_id: str
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    source_ip: Optional[str] = None
    user_agent: Optional[str] = None
    
    def is_expired(self, timeout_seconds: int = 3600) -> bool:
        """Check if security context has expired."""
        return (time.time() - self.last_accessed) > timeout_seconds
    
    def has_permission(self, permission: str) -> bool:
        """Check if context has specific permission."""
        return permission in self.permissions or self.role == UserRole.ADMIN
    
    def refresh_access(self) -> None:
        """Update last accessed timestamp."""
        self.last_accessed = time.time()


@dataclass
class SecurityEvent:
    """Security-related event for audit logging."""
    event_type: str
    severity: str  # info, warning, critical
    user_id: Optional[str]
    source_ip: Optional[str]
    resource: str
    action: str
    result: str  # success, failure, blocked
    timestamp: float = field(default_factory=time.time)
    details: Dict[str, Any] = field(default_factory=dict)


class InputValidator:
    """Input validation and sanitization."""
    
    # Common regex patterns
    EMAIL_PATTERN = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    IPV4_PATTERN = re.compile(r'^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$')
    ALPHANUMERIC_PATTERN = re.compile(r'^[a-zA-Z0-9_-]+$')
    
    @staticmethod
    def validate_building_id(building_id: str) -> bool:
        """Validate building ID format."""
        return (
            isinstance(building_id, str) and
            1 <= len(building_id) <= 50 and
            InputValidator.ALPHANUMERIC_PATTERN.match(building_id)
        )
    
    @staticmethod
    def validate_zone_id(zone_id: str) -> bool:
        """Validate zone ID format."""
        return (
            isinstance(zone_id, str) and
            1 <= len(zone_id) <= 20 and
            InputValidator.ALPHANUMERIC_PATTERN.match(zone_id)
        )
    
    @staticmethod
    def validate_temperature(temperature: float) -> bool:
        """Validate temperature value."""
        return isinstance(temperature, (int, float)) and -50 <= temperature <= 100
    
    @staticmethod
    def validate_power(power_kw: float) -> bool:
        """Validate power consumption value."""
        return isinstance(power_kw, (int, float)) and 0 <= power_kw <= 10000
    
    @staticmethod
    def validate_percentage(value: float) -> bool:
        """Validate percentage value."""
        return isinstance(value, (int, float)) and 0 <= value <= 100
    
    @staticmethod
    def validate_user_id(user_id: str) -> bool:
        """Validate user ID format."""
        return (
            isinstance(user_id, str) and
            3 <= len(user_id) <= 50 and
            InputValidator.ALPHANUMERIC_PATTERN.match(user_id)
        )
    
    @staticmethod
    def validate_ip_address(ip: str) -> bool:
        """Validate IP address format."""
        return isinstance(ip, str) and InputValidator.IPV4_PATTERN.match(ip)
    
    @staticmethod
    def sanitize_string(input_str: str, max_length: int = 255) -> str:
        """Sanitize string input."""
        if not isinstance(input_str, str):
            return ""
        
        # Remove potentially dangerous characters
        sanitized = re.sub(r'[<>"\';\\]', '', input_str)
        
        # Truncate to max length
        return sanitized[:max_length].strip()
    
    @staticmethod
    def validate_json_payload(payload: Dict[str, Any], schema: Dict[str, type]) -> tuple[bool, str]:
        """Validate JSON payload against schema."""
        try:
            for field, expected_type in schema.items():
                if field not in payload:
                    return False, f"Missing required field: {field}"
                
                if not isinstance(payload[field], expected_type):
                    return False, f"Invalid type for {field}: expected {expected_type.__name__}"
            
            return True, "Valid"
        
        except Exception as e:
            return False, f"Validation error: {e}"


class CryptoManager:
    """Encryption and cryptographic utilities."""
    
    def __init__(self, master_key: Optional[bytes] = None):
        self.logger = logging.getLogger("crypto_manager")
        
        if master_key is None:
            master_key = self._derive_key_from_env()
        
        self.fernet = Fernet(master_key)
    
    def _derive_key_from_env(self) -> bytes:
        """Derive encryption key from environment or generate new one."""
        import os
        
        # Try to get key from environment
        key_b64 = os.getenv('QUANTUM_CTL_MASTER_KEY')
        if key_b64:
            try:
                return base64.urlsafe_b64decode(key_b64)
            except Exception:
                self.logger.warning("Invalid master key in environment, generating new one")
        
        # Generate new key
        key = Fernet.generate_key()
        self.logger.warning(
            "Generated new master key. Set QUANTUM_CTL_MASTER_KEY environment variable: %s",
            base64.urlsafe_b64encode(key).decode()
        )
        return key
    
    def encrypt_data(self, data: Union[str, Dict]) -> bytes:
        """Encrypt sensitive data."""
        if isinstance(data, dict):
            data = json.dumps(data)
        
        return self.fernet.encrypt(data.encode())
    
    def decrypt_data(self, encrypted_data: bytes) -> Union[str, Dict]:
        """Decrypt sensitive data."""
        try:
            decrypted = self.fernet.decrypt(encrypted_data).decode()
            
            # Try to parse as JSON
            try:
                return json.loads(decrypted)
            except json.JSONDecodeError:
                return decrypted
        
        except Exception as e:
            self.logger.error(f"Decryption failed: {e}")
            raise ValueError("Failed to decrypt data")
    
    def generate_secure_token(self, length: int = 32) -> str:
        """Generate cryptographically secure random token."""
        return secrets.token_urlsafe(length)
    
    def hash_password(self, password: str, salt: Optional[bytes] = None) -> tuple[bytes, bytes]:
        """Hash password with salt using PBKDF2."""
        if salt is None:
            salt = secrets.token_bytes(32)
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        
        key = kdf.derive(password.encode())
        return key, salt
    
    def verify_password(self, password: str, hashed: bytes, salt: bytes) -> bool:
        """Verify password against hash."""
        try:
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            
            kdf.verify(password.encode(), hashed)
            return True
        except Exception:
            return False
    
    def create_signature(self, data: str, secret: str) -> str:
        """Create HMAC signature for data integrity."""
        signature = hmac.new(
            secret.encode(),
            data.encode(),
            hashlib.sha256
        ).hexdigest()
        
        return signature
    
    def verify_signature(self, data: str, signature: str, secret: str) -> bool:
        """Verify HMAC signature."""
        expected_signature = self.create_signature(data, secret)
        return hmac.compare_digest(signature, expected_signature)


class AuthenticationManager:
    """Authentication and session management."""
    
    def __init__(self, crypto_manager: CryptoManager):
        self.crypto = crypto_manager
        self.logger = logging.getLogger("auth_manager")
        
        # In-memory session storage (use Redis in production)
        self.active_sessions: Dict[str, SecurityContext] = {}
        self.failed_attempts: Dict[str, List[float]] = {}
        
        # Configuration
        self.max_failed_attempts = 5
        self.lockout_duration = 300  # 5 minutes
        self.session_timeout = 3600  # 1 hour
    
    def authenticate_user(
        self,
        user_id: str,
        password: str,
        source_ip: Optional[str] = None
    ) -> Optional[SecurityContext]:
        """Authenticate user and create session."""
        
        # Check if user is locked out
        if self._is_locked_out(user_id):
            self.logger.warning(f"Authentication blocked - user {user_id} is locked out")
            return None
        
        # Validate input
        if not InputValidator.validate_user_id(user_id):
            self.logger.warning(f"Invalid user ID format: {user_id}")
            self._record_failed_attempt(user_id)
            return None
        
        # Simulate user lookup (in production, query database)
        user_data = self._lookup_user(user_id)
        if not user_data:
            self.logger.warning(f"User not found: {user_id}")
            self._record_failed_attempt(user_id)
            return None
        
        # Verify password
        if not self.crypto.verify_password(password, user_data['password_hash'], user_data['salt']):
            self.logger.warning(f"Invalid password for user: {user_id}")
            self._record_failed_attempt(user_id)
            return None
        
        # Clear failed attempts on successful login
        if user_id in self.failed_attempts:
            del self.failed_attempts[user_id]
        
        # Create session
        session_id = self.crypto.generate_secure_token()
        context = SecurityContext(
            user_id=user_id,
            role=UserRole(user_data['role']),
            permissions=user_data['permissions'],
            session_id=session_id,
            source_ip=source_ip
        )
        
        self.active_sessions[session_id] = context
        
        self.logger.info(f"User authenticated successfully: {user_id}")
        return context
    
    def validate_session(self, session_id: str) -> Optional[SecurityContext]:
        """Validate and refresh session."""
        if session_id not in self.active_sessions:
            return None
        
        context = self.active_sessions[session_id]
        
        # Check if session expired
        if context.is_expired(self.session_timeout):
            self.logger.info(f"Session expired: {session_id}")
            del self.active_sessions[session_id]
            return None
        
        # Refresh access time
        context.refresh_access()
        return context
    
    def revoke_session(self, session_id: str) -> bool:
        """Revoke user session."""
        if session_id in self.active_sessions:
            user_id = self.active_sessions[session_id].user_id
            del self.active_sessions[session_id]
            self.logger.info(f"Session revoked: {session_id} for user {user_id}")
            return True
        return False
    
    def _is_locked_out(self, user_id: str) -> bool:
        """Check if user is locked out due to failed attempts."""
        if user_id not in self.failed_attempts:
            return False
        
        attempts = self.failed_attempts[user_id]
        cutoff_time = time.time() - self.lockout_duration
        
        # Remove old attempts
        recent_attempts = [t for t in attempts if t > cutoff_time]
        self.failed_attempts[user_id] = recent_attempts
        
        return len(recent_attempts) >= self.max_failed_attempts
    
    def _record_failed_attempt(self, user_id: str) -> None:
        """Record failed authentication attempt."""
        if user_id not in self.failed_attempts:
            self.failed_attempts[user_id] = []
        
        self.failed_attempts[user_id].append(time.time())
    
    def _lookup_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Lookup user data (mock implementation)."""
        # Mock user database
        mock_users = {
            'admin': {
                'role': 'admin',
                'permissions': ['read', 'write', 'delete', 'admin'],
                'password_hash': b'mock_hash',
                'salt': b'mock_salt'
            },
            'operator': {
                'role': 'operator', 
                'permissions': ['read', 'write'],
                'password_hash': b'mock_hash',
                'salt': b'mock_salt'
            },
            'readonly': {
                'role': 'readonly',
                'permissions': ['read'],
                'password_hash': b'mock_hash', 
                'salt': b'mock_salt'
            }
        }
        
        return mock_users.get(user_id)
    
    def get_session_count(self) -> int:
        """Get number of active sessions."""
        return len(self.active_sessions)
    
    def cleanup_expired_sessions(self) -> int:
        """Remove expired sessions and return count removed."""
        expired_sessions = [
            sid for sid, context in self.active_sessions.items()
            if context.is_expired(self.session_timeout)
        ]
        
        for session_id in expired_sessions:
            del self.active_sessions[session_id]
        
        if expired_sessions:
            self.logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
        
        return len(expired_sessions)


class AuthorizationManager:
    """Authorization and access control."""
    
    def __init__(self):
        self.logger = logging.getLogger("authz_manager")
        
        # Resource permissions mapping
        self.resource_permissions = {
            '/api/buildings': {'read': ['readonly', 'operator', 'admin']},
            '/api/buildings/*/zones': {'read': ['readonly', 'operator', 'admin']},
            '/api/control/optimize': {'write': ['operator', 'admin']},
            '/api/control/emergency': {'write': ['admin']},
            '/api/settings': {'read': ['operator', 'admin'], 'write': ['admin']},
            '/api/users': {'read': ['admin'], 'write': ['admin'], 'delete': ['admin']},
            '/api/metrics': {'read': ['readonly', 'operator', 'admin']},
            '/api/health': {'read': ['readonly', 'operator', 'admin']},
        }
    
    def check_permission(
        self,
        context: SecurityContext,
        resource: str,
        action: str
    ) -> bool:
        """Check if user has permission for resource action."""
        
        # Admin always has access
        if context.role == UserRole.ADMIN:
            return True
        
        # Find matching resource pattern
        for pattern, permissions in self.resource_permissions.items():
            if self._match_resource_pattern(pattern, resource):
                allowed_roles = permissions.get(action, [])
                return context.role.value in allowed_roles
        
        # Default deny
        return False
    
    def _match_resource_pattern(self, pattern: str, resource: str) -> bool:
        """Match resource against pattern with wildcards."""
        # Simple wildcard matching (enhance for production)
        pattern_parts = pattern.split('/')
        resource_parts = resource.split('/')
        
        if len(pattern_parts) != len(resource_parts):
            return False
        
        for pattern_part, resource_part in zip(pattern_parts, resource_parts):
            if pattern_part != '*' and pattern_part != resource_part:
                return False
        
        return True
    
    def get_user_permissions(self, context: SecurityContext) -> List[str]:
        """Get all permissions for user."""
        permissions = []
        
        for resource, actions in self.resource_permissions.items():
            for action, allowed_roles in actions.items():
                if context.role.value in allowed_roles or context.role == UserRole.ADMIN:
                    permissions.append(f"{action}:{resource}")
        
        return permissions


class SecurityAuditLogger:
    """Security event audit logging."""
    
    def __init__(self):
        self.logger = logging.getLogger("security_audit")
        self.audit_log: List[SecurityEvent] = []
        
        # Configure separate audit file handler
        audit_handler = logging.FileHandler('security_audit.log')
        audit_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(audit_handler)
        self.logger.setLevel(logging.INFO)
    
    def log_security_event(
        self,
        event_type: str,
        severity: str,
        user_id: Optional[str],
        source_ip: Optional[str],
        resource: str,
        action: str,
        result: str,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log security event for audit trail."""
        
        event = SecurityEvent(
            event_type=event_type,
            severity=severity,
            user_id=user_id,
            source_ip=source_ip,
            resource=resource,
            action=action,
            result=result,
            details=details or {}
        )
        
        self.audit_log.append(event)
        
        # Log to file
        self.logger.info(
            f"SECURITY_EVENT: {event_type} | User: {user_id} | "
            f"IP: {source_ip} | Resource: {resource} | "
            f"Action: {action} | Result: {result} | "
            f"Severity: {severity}"
        )
    
    def log_authentication(
        self,
        user_id: str,
        source_ip: Optional[str],
        success: bool,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log authentication attempt."""
        self.log_security_event(
            event_type="authentication",
            severity="info" if success else "warning",
            user_id=user_id,
            source_ip=source_ip,
            resource="auth",
            action="login",
            result="success" if success else "failure",
            details=details
        )
    
    def log_authorization(
        self,
        user_id: str,
        source_ip: Optional[str],
        resource: str,
        action: str,
        allowed: bool,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log authorization check."""
        self.log_security_event(
            event_type="authorization",
            severity="info" if allowed else "warning",
            user_id=user_id,
            source_ip=source_ip,
            resource=resource,
            action=action,
            result="allowed" if allowed else "denied",
            details=details
        )
    
    def get_security_events(
        self,
        hours: int = 24,
        event_type: Optional[str] = None,
        severity: Optional[str] = None
    ) -> List[SecurityEvent]:
        """Get recent security events."""
        cutoff_time = time.time() - (hours * 3600)
        
        events = [e for e in self.audit_log if e.timestamp >= cutoff_time]
        
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        
        if severity:
            events = [e for e in events if e.severity == severity]
        
        return events
    
    def get_security_summary(self) -> Dict[str, Any]:
        """Get security events summary."""
        recent_events = self.get_security_events(24)
        
        event_counts = {}
        severity_counts = {}
        
        for event in recent_events:
            event_counts[event.event_type] = event_counts.get(event.event_type, 0) + 1
            severity_counts[event.severity] = severity_counts.get(event.severity, 0) + 1
        
        return {
            'total_events_24h': len(recent_events),
            'event_type_breakdown': event_counts,
            'severity_breakdown': severity_counts,
            'latest_event': max(recent_events, key=lambda x: x.timestamp) if recent_events else None
        }


# Security decorators
def require_authentication(auth_manager: AuthenticationManager):
    """Decorator to require authentication."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Extract session from request context
            session_id = kwargs.get('session_id')
            if not session_id:
                raise PermissionError("Authentication required")
            
            context = auth_manager.validate_session(session_id)
            if not context:
                raise PermissionError("Invalid or expired session")
            
            # Add context to kwargs
            kwargs['security_context'] = context
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator


def require_permission(authz_manager: AuthorizationManager, resource: str, action: str):
    """Decorator to require specific permission."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            context = kwargs.get('security_context')
            if not context:
                raise PermissionError("Security context required")
            
            if not authz_manager.check_permission(context, resource, action):
                raise PermissionError(f"Permission denied for {action} on {resource}")
            
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator


# Global security managers (initialize with proper configuration in production)
_crypto_manager = CryptoManager()
_auth_manager = AuthenticationManager(_crypto_manager)
_authz_manager = AuthorizationManager()
_audit_logger = SecurityAuditLogger()


def get_crypto_manager() -> CryptoManager:
    """Get global crypto manager."""
    return _crypto_manager


def get_auth_manager() -> AuthenticationManager:
    """Get global authentication manager."""
    return _auth_manager


def get_authz_manager() -> AuthorizationManager:
    """Get global authorization manager."""
    return _authz_manager


def get_audit_logger() -> SecurityAuditLogger:
    """Get global security audit logger."""
    return _audit_logger
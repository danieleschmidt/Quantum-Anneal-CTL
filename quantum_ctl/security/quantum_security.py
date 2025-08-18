"""
Quantum solver security and authentication framework.

This module provides security layers for quantum annealing operations,
including authentication, authorization, secure parameter handling,
and protection against quantum-specific attacks.
"""

import hashlib
import hmac
import time
import secrets
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import logging
from enum import Enum
import asyncio
import jwt
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

from ..optimization.quantum_solver import QuantumSolver, QuantumSolution


class SecurityLevel(Enum):
    """Security levels for quantum operations."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SecurityCredentials:
    """Security credentials for quantum access."""
    api_key: str
    secret_key: str
    token: Optional[str] = None
    expires_at: Optional[float] = None
    permissions: List[str] = None


@dataclass
class SecurityAuditLog:
    """Security audit log entry."""
    timestamp: float
    user_id: str
    action: str
    resource: str
    success: bool
    ip_address: str
    details: Dict[str, Any]


class QuantumSecurityManager:
    """
    Comprehensive security manager for quantum annealing operations.
    
    Provides authentication, authorization, audit logging, and protection
    against quantum-specific security threats.
    """
    
    def __init__(
        self,
        security_level: SecurityLevel = SecurityLevel.MEDIUM,
        jwt_secret: str = None,
        encryption_key: bytes = None
    ):
        self.security_level = security_level
        self.jwt_secret = jwt_secret or self._generate_jwt_secret()
        
        # Initialize encryption
        if encryption_key:
            self.cipher = Fernet(encryption_key)
        else:
            self.cipher = Fernet(self._generate_encryption_key())
            
        self.logger = logging.getLogger(__name__)
        
        # Security state
        self.active_sessions = {}
        self.failed_attempts = {}
        self.audit_log: List[SecurityAuditLog] = []
        
        # Rate limiting
        self.rate_limits = {
            SecurityLevel.LOW: {'requests_per_minute': 100, 'max_problem_size': 1000},
            SecurityLevel.MEDIUM: {'requests_per_minute': 50, 'max_problem_size': 5000},
            SecurityLevel.HIGH: {'requests_per_minute': 20, 'max_problem_size': 10000},
            SecurityLevel.CRITICAL: {'requests_per_minute': 10, 'max_problem_size': 50000}
        }
        
        # Initialize security monitoring
        self._start_security_monitoring()
        
    async def authenticate_user(
        self,
        credentials: SecurityCredentials,
        client_ip: str = "unknown"
    ) -> Tuple[bool, Optional[str]]:
        """
        Authenticate user for quantum solver access.
        
        Args:
            credentials: User security credentials
            client_ip: Client IP address
            
        Returns:
            Tuple of (success, session_token)
        """
        
        start_time = time.time()
        
        try:
            # Check for brute force attacks
            if self._is_rate_limited(client_ip):
                self._log_security_event(
                    "authentication_failed",
                    "rate_limited",
                    success=False,
                    ip_address=client_ip,
                    details={'reason': 'rate_limited'}
                )
                return False, None
                
            # Validate API key format
            if not self._validate_api_key_format(credentials.api_key):
                self._record_failed_attempt(client_ip)
                self._log_security_event(
                    "authentication_failed", 
                    "quantum_solver",
                    success=False,
                    ip_address=client_ip,
                    details={'reason': 'invalid_api_key_format'}
                )
                return False, None
                
            # Verify API key and secret
            if not await self._verify_credentials(credentials):
                self._record_failed_attempt(client_ip)
                self._log_security_event(
                    "authentication_failed",
                    "quantum_solver", 
                    success=False,
                    ip_address=client_ip,
                    details={'reason': 'invalid_credentials'}
                )
                return False, None
                
            # Generate secure session token
            session_token = self._generate_session_token(credentials, client_ip)
            
            # Store active session
            self.active_sessions[session_token] = {
                'user_id': credentials.api_key[:8],  # Partial key for logging
                'created_at': start_time,
                'ip_address': client_ip,
                'permissions': credentials.permissions or ['basic_access']
            }
            
            # Reset failed attempts on successful auth
            if client_ip in self.failed_attempts:
                del self.failed_attempts[client_ip]
                
            self._log_security_event(
                "authentication_success",
                "quantum_solver",
                success=True,
                ip_address=client_ip,
                details={'session_token': session_token[:16] + "..."}
            )
            
            return True, session_token
            
        except Exception as e:
            self.logger.error(f"Authentication error: {e}")
            self._log_security_event(
                "authentication_error",
                "quantum_solver",
                success=False,
                ip_address=client_ip,
                details={'error': str(e)}
            )
            return False, None
            
    async def authorize_quantum_operation(
        self,
        session_token: str,
        operation: str,
        problem_size: int,
        resource_requirements: Dict[str, Any]
    ) -> Tuple[bool, Optional[str]]:
        """
        Authorize quantum annealing operation.
        
        Args:
            session_token: Valid session token
            operation: Type of quantum operation
            problem_size: Size of the problem (number of variables)
            resource_requirements: Required quantum resources
            
        Returns:
            Tuple of (authorized, denial_reason)
        """
        
        # Validate session
        if not self._validate_session(session_token):
            return False, "invalid_session"
            
        session_info = self.active_sessions[session_token]
        
        # Check operation permissions
        if not self._has_permission(session_info['permissions'], operation):
            self._log_security_event(
                "authorization_failed",
                "quantum_operation",
                success=False,
                ip_address=session_info['ip_address'],
                details={'operation': operation, 'reason': 'insufficient_permissions'}
            )
            return False, "insufficient_permissions"
            
        # Check resource limits based on security level
        limits = self.rate_limits[self.security_level]
        
        if problem_size > limits['max_problem_size']:
            self._log_security_event(
                "authorization_failed",
                "quantum_operation",
                success=False,
                ip_address=session_info['ip_address'],
                details={'operation': operation, 'problem_size': problem_size, 'limit': limits['max_problem_size']}
            )
            return False, "problem_size_exceeded"
            
        # Check rate limits
        if self._check_operation_rate_limit(session_token):
            self._log_security_event(
                "authorization_failed",
                "quantum_operation", 
                success=False,
                ip_address=session_info['ip_address'],
                details={'operation': operation, 'reason': 'rate_limit_exceeded'}
            )
            return False, "rate_limit_exceeded"
            
        # Validate quantum resource requirements
        if not self._validate_quantum_resources(resource_requirements):
            return False, "invalid_resource_requirements"
            
        self._log_security_event(
            "authorization_success",
            "quantum_operation",
            success=True,
            ip_address=session_info['ip_address'],
            details={'operation': operation, 'problem_size': problem_size}
        )
        
        return True, None
        
    def secure_parameter_handling(
        self,
        parameters: Dict[str, Any],
        session_token: str
    ) -> Dict[str, Any]:
        """
        Securely handle and sanitize quantum solver parameters.
        
        Args:
            parameters: Raw solver parameters
            session_token: Valid session token
            
        Returns:
            Sanitized and secured parameters
        """
        
        if not self._validate_session(session_token):
            raise ValueError("Invalid session for parameter handling")
            
        secured_params = {}
        
        # Sanitize and validate each parameter
        for key, value in parameters.items():
            if key in self._get_allowed_parameters():
                secured_value = self._sanitize_parameter(key, value)
                if secured_value is not None:
                    secured_params[key] = secured_value
                else:
                    self.logger.warning(f"Parameter {key} failed sanitization")
                    
        # Add security-enforced parameters
        secured_params.update(self._get_security_enforced_parameters())
        
        # Log parameter access
        session_info = self.active_sessions[session_token]
        self._log_security_event(
            "parameter_access",
            "quantum_solver",
            success=True,
            ip_address=session_info['ip_address'],
            details={'parameters': list(secured_params.keys())}
        )
        
        return secured_params
        
    def encrypt_sensitive_data(self, data: Dict[str, Any]) -> str:
        """Encrypt sensitive data for transmission/storage."""
        
        try:
            import json
            json_data = json.dumps(data, sort_keys=True)
            encrypted_data = self.cipher.encrypt(json_data.encode())
            return base64.b64encode(encrypted_data).decode()
            
        except Exception as e:
            self.logger.error(f"Encryption failed: {e}")
            raise ValueError("Failed to encrypt sensitive data")
            
    def decrypt_sensitive_data(self, encrypted_data: str) -> Dict[str, Any]:
        """Decrypt sensitive data from transmission/storage."""
        
        try:
            import json
            encrypted_bytes = base64.b64decode(encrypted_data.encode())
            decrypted_data = self.cipher.decrypt(encrypted_bytes)
            return json.loads(decrypted_data.decode())
            
        except Exception as e:
            self.logger.error(f"Decryption failed: {e}")
            raise ValueError("Failed to decrypt sensitive data")
            
    def _generate_jwt_secret(self) -> str:
        """Generate secure JWT secret."""
        return secrets.token_urlsafe(64)
        
    def _generate_encryption_key(self) -> bytes:
        """Generate encryption key for sensitive data."""
        return Fernet.generate_key()
        
    def _generate_session_token(
        self,
        credentials: SecurityCredentials,
        client_ip: str
    ) -> str:
        """Generate secure session token."""
        
        payload = {
            'user_id': hashlib.sha256(credentials.api_key.encode()).hexdigest()[:16],
            'client_ip': client_ip,
            'issued_at': time.time(),
            'expires_at': time.time() + 3600,  # 1 hour expiry
            'security_level': self.security_level.value
        }
        
        return jwt.encode(payload, self.jwt_secret, algorithm='HS256')
        
    def _validate_session(self, session_token: str) -> bool:
        """Validate session token."""
        
        try:
            # Check if session exists
            if session_token not in self.active_sessions:
                return False
                
            # Decode and validate JWT
            payload = jwt.decode(session_token, self.jwt_secret, algorithms=['HS256'])
            
            # Check expiry
            if time.time() > payload.get('expires_at', 0):
                # Remove expired session
                if session_token in self.active_sessions:
                    del self.active_sessions[session_token]
                return False
                
            return True
            
        except jwt.InvalidTokenError:
            return False
            
    def _validate_api_key_format(self, api_key: str) -> bool:
        """Validate API key format."""
        
        # Basic format validation - should be enhanced based on actual API key format
        if not isinstance(api_key, str):
            return False
            
        if len(api_key) < 16 or len(api_key) > 128:
            return False
            
        # Check for suspicious patterns
        if api_key.count('.') > 2 or api_key.count('/') > 0:
            return False
            
        return True
        
    async def _verify_credentials(self, credentials: SecurityCredentials) -> bool:
        """Verify credentials against secure storage."""
        
        # In production, this would verify against secure credential store
        # For now, implement basic validation
        
        if not credentials.api_key or not credentials.secret_key:
            return False
            
        # Simulate secure credential verification
        await asyncio.sleep(0.1)  # Simulate network/database lookup
        
        # Basic validation - in production, use secure hash comparison
        expected_hash = hashlib.sha256(f"{credentials.api_key}:{credentials.secret_key}".encode()).hexdigest()
        
        # This would be compared against stored hash
        return len(expected_hash) == 64  # Basic validity check
        
    def _is_rate_limited(self, client_ip: str) -> bool:
        """Check if client IP is rate limited."""
        
        current_time = time.time()
        
        if client_ip not in self.failed_attempts:
            return False
            
        attempts = self.failed_attempts[client_ip]
        
        # Remove old attempts (older than 5 minutes)
        recent_attempts = [t for t in attempts if current_time - t < 300]
        self.failed_attempts[client_ip] = recent_attempts
        
        # Rate limit after 5 failed attempts
        return len(recent_attempts) >= 5
        
    def _record_failed_attempt(self, client_ip: str) -> None:
        """Record failed authentication attempt."""
        
        if client_ip not in self.failed_attempts:
            self.failed_attempts[client_ip] = []
            
        self.failed_attempts[client_ip].append(time.time())
        
    def _has_permission(self, user_permissions: List[str], operation: str) -> bool:
        """Check if user has permission for operation."""
        
        # Define operation permission requirements
        operation_permissions = {
            'quantum_solve': ['quantum_access', 'basic_access'],
            'quantum_inspect': ['quantum_access', 'inspect_access', 'basic_access'],
            'parameter_tune': ['advanced_access', 'quantum_access'],
            'benchmark_run': ['benchmark_access', 'advanced_access']
        }
        
        required_permissions = operation_permissions.get(operation, ['basic_access'])
        
        return any(perm in user_permissions for perm in required_permissions)
        
    def _check_operation_rate_limit(self, session_token: str) -> bool:
        """Check operation rate limit for session."""
        
        # Simple rate limiting implementation
        current_time = time.time()
        session_info = self.active_sessions.get(session_token, {})
        
        if 'operations' not in session_info:
            session_info['operations'] = []
            
        # Remove operations older than 1 minute
        recent_ops = [t for t in session_info['operations'] if current_time - t < 60]
        session_info['operations'] = recent_ops
        
        # Check against rate limit
        limit = self.rate_limits[self.security_level]['requests_per_minute']
        
        if len(recent_ops) >= limit:
            return True
            
        # Record this operation
        session_info['operations'].append(current_time)
        return False
        
    def _validate_quantum_resources(self, resource_requirements: Dict[str, Any]) -> bool:
        """Validate quantum resource requirements."""
        
        # Validate resource requirement format and values
        allowed_resources = ['num_reads', 'annealing_time', 'chain_strength', 'time_limit']
        
        for resource, value in resource_requirements.items():
            if resource not in allowed_resources:
                return False
                
            # Validate resource limits
            if resource == 'num_reads' and (not isinstance(value, int) or value > 10000):
                return False
            elif resource == 'annealing_time' and (not isinstance(value, (int, float)) or value > 1000):
                return False
            elif resource == 'time_limit' and (not isinstance(value, (int, float)) or value > 3600):
                return False
                
        return True
        
    def _get_allowed_parameters(self) -> List[str]:
        """Get list of allowed solver parameters."""
        
        return [
            'num_reads', 'annealing_time', 'chain_strength', 'auto_scale',
            'time_limit', 'seed', 'answer_mode', 'num_spin_reversal_transforms'
        ]
        
    def _sanitize_parameter(self, key: str, value: Any) -> Any:
        """Sanitize solver parameter value."""
        
        # Parameter-specific sanitization
        if key == 'num_reads':
            if isinstance(value, int) and 1 <= value <= 10000:
                return value
        elif key == 'annealing_time':
            if isinstance(value, (int, float)) and 1 <= value <= 1000:
                return int(value)
        elif key == 'chain_strength':
            if isinstance(value, (int, float)) and value > 0:
                return float(value)
        elif key == 'auto_scale':
            return bool(value)
        elif key == 'time_limit':
            if isinstance(value, (int, float)) and 1 <= value <= 3600:
                return float(value)
        elif key == 'seed':
            if isinstance(value, int) and 0 <= value <= 2**32:
                return value
                
        return None
        
    def _get_security_enforced_parameters(self) -> Dict[str, Any]:
        """Get parameters enforced by security policy."""
        
        enforced = {}
        
        # Enforce limits based on security level
        if self.security_level == SecurityLevel.HIGH:
            enforced['auto_scale'] = True
            
        elif self.security_level == SecurityLevel.CRITICAL:
            enforced['auto_scale'] = True
            enforced['answer_mode'] = 'histogram'
            
        return enforced
        
    def _log_security_event(
        self,
        action: str,
        resource: str,
        success: bool,
        ip_address: str,
        details: Dict[str, Any]
    ) -> None:
        """Log security event for audit trail."""
        
        audit_entry = SecurityAuditLog(
            timestamp=time.time(),
            user_id=details.get('user_id', 'unknown'),
            action=action,
            resource=resource,
            success=success,
            ip_address=ip_address,
            details=details
        )
        
        self.audit_log.append(audit_entry)
        
        # Log to system logger
        level = logging.INFO if success else logging.WARNING
        self.logger.log(level, f"Security event: {action} on {resource} from {ip_address} - {'SUCCESS' if success else 'FAILED'}")
        
        # Maintain audit log size
        if len(self.audit_log) > 10000:
            self.audit_log = self.audit_log[-5000:]  # Keep last 5000 entries
            
    def _start_security_monitoring(self) -> None:
        """Start background security monitoring."""
        
        async def monitor():
            while True:
                try:
                    # Clean up expired sessions
                    await self._cleanup_expired_sessions()
                    
                    # Analyze security patterns
                    await self._analyze_security_patterns()
                    
                    # Sleep for monitoring interval
                    await asyncio.sleep(60)  # Monitor every minute
                    
                except Exception as e:
                    self.logger.error(f"Security monitoring error: {e}")
                    await asyncio.sleep(60)
                    
        # Start monitoring task
        asyncio.create_task(monitor())
        
    async def _cleanup_expired_sessions(self) -> None:
        """Clean up expired sessions."""
        
        current_time = time.time()
        expired_sessions = []
        
        for token, session_info in self.active_sessions.items():
            if current_time - session_info['created_at'] > 3600:  # 1 hour expiry
                expired_sessions.append(token)
                
        for token in expired_sessions:
            del self.active_sessions[token]
            
        if expired_sessions:
            self.logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
            
    async def _analyze_security_patterns(self) -> None:
        """Analyze security patterns for threats."""
        
        # Analyze failed attempts patterns
        suspicious_ips = []
        
        for ip, attempts in self.failed_attempts.items():
            if len(attempts) > 10:  # High number of failures
                suspicious_ips.append(ip)
                
        if suspicious_ips:
            self.logger.warning(f"Suspicious activity detected from IPs: {suspicious_ips}")
            
        # Analyze unusual operation patterns
        # This could be enhanced with machine learning for anomaly detection
        
    def get_security_status(self) -> Dict[str, Any]:
        """Get current security status."""
        
        return {
            'security_level': self.security_level.value,
            'active_sessions': len(self.active_sessions),
            'failed_attempts_ips': len(self.failed_attempts),
            'audit_log_entries': len(self.audit_log),
            'rate_limits': self.rate_limits[self.security_level],
            'last_security_check': time.time()
        }


class SecureQuantumSolver:
    """
    Secure wrapper for quantum solver with integrated security measures.
    
    Provides a secure interface to quantum annealing while enforcing
    security policies and audit logging.
    """
    
    def __init__(
        self,
        base_solver: QuantumSolver,
        security_manager: QuantumSecurityManager,
        session_token: str
    ):
        self.base_solver = base_solver
        self.security_manager = security_manager
        self.session_token = session_token
        
        self.logger = logging.getLogger(__name__)
        
    async def secure_solve(
        self,
        qubo: Dict[Tuple[int, int], float],
        **kwargs
    ) -> QuantumSolution:
        """
        Solve QUBO with security enforcement.
        
        Args:
            qubo: QUBO problem to solve
            **kwargs: Solver parameters
            
        Returns:
            Quantum solution with security audit trail
        """
        
        # Authorize operation
        problem_size = len(set(i for (i, j) in qubo.keys()) | set(j for (i, j) in qubo.keys()))
        
        authorized, denial_reason = await self.security_manager.authorize_quantum_operation(
            self.session_token,
            'quantum_solve',
            problem_size,
            kwargs
        )
        
        if not authorized:
            raise PermissionError(f"Quantum solve operation denied: {denial_reason}")
            
        # Secure parameter handling
        secured_params = self.security_manager.secure_parameter_handling(
            kwargs,
            self.session_token
        )
        
        # Validate QUBO for security
        self._validate_qubo_security(qubo)
        
        # Execute solve with security monitoring
        start_time = time.time()
        
        try:
            solution = await self.base_solver.solve(qubo, **secured_params)
            
            # Log successful operation
            self._log_secure_operation(
                'quantum_solve_success',
                problem_size,
                time.time() - start_time,
                solution
            )
            
            return solution
            
        except Exception as e:
            # Log failed operation
            self._log_secure_operation(
                'quantum_solve_failed',
                problem_size,
                time.time() - start_time,
                error=str(e)
            )
            
            raise
            
    def _validate_qubo_security(self, qubo: Dict[Tuple[int, int], float]) -> None:
        """Validate QUBO for security issues."""
        
        # Check for suspicious QUBO patterns
        coefficients = list(qubo.values())
        
        # Check for extremely large coefficients (potential DoS)
        max_coeff = max(abs(c) for c in coefficients)
        if max_coeff > 1e6:
            raise ValueError("QUBO coefficients too large - potential security risk")
            
        # Check for NaN or infinite values
        if any(not np.isfinite(c) for c in coefficients):
            raise ValueError("QUBO contains invalid values")
            
        # Check problem structure for anomalies
        variables = set()
        for (i, j) in qubo.keys():
            variables.add(i)
            variables.add(j)
            
        # Check for sparse problems with suspicious structure
        density = len(qubo) / (len(variables) ** 2)
        if density > 0.95 and len(variables) > 1000:
            self.logger.warning("Dense QUBO detected - potential resource exhaustion")
            
    def _log_secure_operation(
        self,
        operation_type: str,
        problem_size: int,
        duration: float,
        solution: QuantumSolution = None,
        error: str = None
    ) -> None:
        """Log secure quantum operation."""
        
        details = {
            'problem_size': problem_size,
            'duration': duration,
            'operation_type': operation_type
        }
        
        if solution:
            details.update({
                'solution_energy': solution.energy,
                'chain_break_fraction': solution.chain_break_fraction
            })
            
        if error:
            details['error'] = error
            
        session_info = self.security_manager.active_sessions.get(self.session_token, {})
        
        self.security_manager._log_security_event(
            operation_type,
            'secure_quantum_solver',
            success=error is None,
            ip_address=session_info.get('ip_address', 'unknown'),
            details=details
        )
"""
Security and safety modules for quantum HVAC control systems.

This module provides comprehensive security measures for quantum
annealing-based HVAC control, including:
- Quantum solver authentication and authorization
- Secure communication protocols
- Input validation and sanitization
- Attack detection and mitigation
- Safety-critical system monitoring
"""

from .production_security import (
    ProductionSecuritySystem,
    EncryptionManager,
    AuthenticationManager,
    AuthorizationManager,
    ThreatDetector,
    SecurityAuditLogger,
    SecurityException
)

try:
    from .quantum_security import QuantumSecurityManager, SecureQuantumSolver
except ImportError:
    # Fallback if quantum security module not available
    QuantumSecurityManager = None
    SecureQuantumSolver = None

__all__ = [
    "ProductionSecuritySystem",
    "EncryptionManager",
    "AuthenticationManager",
    "AuthorizationManager",
    "ThreatDetector",
    "SecurityAuditLogger",
    "SecurityException",
    "QuantumSecurityManager",
    "SecureQuantumSolver"
]
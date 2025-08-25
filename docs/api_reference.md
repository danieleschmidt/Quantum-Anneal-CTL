# API Reference

## Quantum HVAC Control System API

This document provides comprehensive API reference for all modules in the quantum HVAC control system.

## Core Modules

### quantum_ctl.core

#### QuantumHVACOptimizer
Main optimization engine for quantum HVAC control.

```python
class QuantumHVACOptimizer:
    def __init__(self, config: Dict[str, Any])
    def optimize(self, problem: HVACProblem) -> OptimizationResult
    def get_solver_info(self) -> Dict[str, Any]
```

#### HVACProblem
Problem definition for HVAC optimization.

```python
class HVACProblem:
    def __init__(self, zones: List[Zone], constraints: List[Constraint])
    def to_qubo(self) -> Dict[Tuple[int, int], float]
    def validate(self) -> bool
```

### quantum_ctl.autonomous

#### SelfOptimizingController
Autonomous optimization controller with multiple strategies.

```python
class SelfOptimizingController:
    def __init__(self, config: Optional[Dict] = None)
    def optimize_autonomous(self) -> Dict[str, Any]
    def configure_evolutionary_strategy(self, population_size: int, mutation_rate: float, crossover_rate: float)
    def enable_neural_optimization(self, hidden_layers: List[int], learning_rate: float)
    def get_optimization_history(self) -> List[Dict]
```

#### AdaptiveQuantumOrchestrator
Multi-solver coordination and resource management.

```python
class AdaptiveQuantumOrchestrator:
    def __init__(self)
    def add_solver(self, solver_id: str, priority: int = 1)
    def remove_solver(self, solver_id: str)
    def orchestrate_problem(self, problem: Any) -> Dict[str, Any]
    def get_solver_performance(self) -> Dict[str, Dict]
```

#### BreakthroughDetector
Performance breakthrough detection and validation.

```python
class BreakthroughDetector:
    def __init__(self, baseline_window: int = 100)
    def detect_breakthrough(self, performance_data: List[float]) -> Dict[str, Any]
    def validate_breakthrough(self, breakthrough_data: Dict) -> bool
    def get_breakthrough_history(self) -> List[Dict]
```

#### AutonomousResearchEngine
Autonomous research and hypothesis testing.

```python
class AutonomousResearchEngine:
    def __init__(self)
    def generate_hypothesis(self, domain: str) -> Dict[str, Any]
    def test_hypothesis(self, hypothesis: Dict) -> Dict[str, Any]
    def discover_algorithms(self, problem_type: str) -> List[Dict]
```

### quantum_ctl.security

#### ProductionSecuritySystem
Comprehensive security system for production deployments.

```python
class ProductionSecuritySystem:
    def __init__(self, config: SecurityConfig)
    def initialize_security(self)
    def validate_security_status(self) -> SecurityStatus
    def handle_security_incident(self, incident: SecurityIncident)
```

#### EncryptionManager
Data encryption and key management.

```python
class EncryptionManager:
    def __init__(self, key_config: Dict)
    def encrypt_sensitive_data(self, data: bytes) -> bytes
    def decrypt_sensitive_data(self, encrypted_data: bytes) -> bytes
    def rotate_keys(self)
```

#### AuthenticationManager
User authentication and session management.

```python
class AuthenticationManager:
    def __init__(self, auth_config: Dict)
    def authenticate_user(self, credentials: Dict) -> AuthSession
    def validate_session(self, session_token: str) -> bool
    def logout_user(self, session_token: str)
```

#### AuthorizationManager
Role-based access control.

```python
class AuthorizationManager:
    def __init__(self, authz_config: Dict)
    def check_permission(self, user: User, resource: str, action: str) -> bool
    def grant_permission(self, user: User, permission: Permission)
    def revoke_permission(self, user: User, permission: Permission)
```

#### ThreatDetector
Real-time threat detection and response.

```python
class ThreatDetector:
    def __init__(self, detection_config: Dict)
    def analyze_request(self, request: Request) -> ThreatLevel
    def block_threat(self, threat_id: str)
    def get_threat_report(self) -> Dict[str, Any]
```

### quantum_ctl.resilience

#### AutonomousResilienceSystem
Self-healing and failure recovery system.

```python
class AutonomousResilienceSystem:
    def __init__(self, config: ResilienceConfig)
    def detect_failures(self) -> List[FailureEvent]
    def execute_recovery(self, failure: FailureEvent) -> RecoveryResult
    def validate_recovery(self, recovery: RecoveryResult) -> bool
```

### quantum_ctl.monitoring

#### ComprehensiveMonitoring
Real-time system monitoring and alerting.

```python
class ComprehensiveMonitoring:
    def __init__(self, config: MonitoringConfig)
    def collect_metrics(self) -> Dict[str, float]
    def analyze_performance(self) -> PerformanceAnalysis
    def generate_alerts(self, metrics: Dict) -> List[Alert]
```

### quantum_ctl.scaling

#### GlobalOrchestrator
Multi-region deployment orchestration.

```python
class GlobalOrchestrator:
    def __init__(self)
    def add_region(self, region_id: str, capacity: int, latency_ms: float)
    def remove_region(self, region_id: str)
    def deploy_single_region(self, region_id: str)
    def enable_active_active_mode(self)
```

### quantum_ctl.global_compliance

#### ComplianceManager
Multi-regulation compliance management.

```python
class ComplianceManager:
    def __init__(self)
    def check_compliance(self, regulation: str, data: Dict) -> ComplianceResult
    def generate_compliance_report(self) -> Dict[str, Any]
    def remediate_violations(self, violations: List[Violation])
```

#### InternationalizationManager
Multi-language and localization support.

```python
class InternationalizationManager:
    def __init__(self)
    def set_user_locale(self, user_id: str, locale: str)
    def get_localized_text(self, key: str, locale: str) -> str
    def format_currency(self, amount: float, currency: str, locale: str) -> str
    def format_datetime(self, dt: datetime, locale: str) -> str
```

## Data Types

### Common Types

```python
# Configuration types
class SecurityConfig:
    encryption_key_size: int
    auth_timeout_minutes: int
    threat_detection_enabled: bool

class MonitoringConfig:
    collection_interval_seconds: int
    alert_thresholds: Dict[str, float]
    retention_days: int

# Result types
class OptimizationResult:
    solution: Dict[str, Any]
    energy: float
    execution_time: float
    solver_info: Dict[str, Any]

class SecurityStatus:
    is_secure: bool
    threats_detected: int
    last_audit: datetime
```

## Error Handling

### Exception Types

```python
class SecurityException(Exception):
    """Security-related errors"""

class ComplianceViolationException(Exception):
    """Compliance violation errors"""

class QuantumSolverException(Exception):
    """Quantum solver errors"""

class ResilienceException(Exception):
    """Resilience system errors"""
```

### Error Codes

- `SEC_001`: Authentication failure
- `SEC_002`: Authorization denied  
- `SEC_003`: Encryption error
- `COMP_001`: GDPR violation
- `COMP_002`: CCPA violation
- `QUANT_001`: Solver connection error
- `QUANT_002`: Invalid QUBO formulation

## Usage Examples

### Basic Optimization

```python
from quantum_ctl.core import QuantumHVACOptimizer, HVACProblem

# Create optimizer
optimizer = QuantumHVACOptimizer(config={
    'solver': 'dwave',
    'num_reads': 1000
})

# Define problem
problem = HVACProblem(zones=zones, constraints=constraints)

# Optimize
result = optimizer.optimize(problem)
print(f"Optimal solution: {result.solution}")
```

### Autonomous Operation

```python
from quantum_ctl.autonomous import SelfOptimizingController

# Initialize autonomous controller
controller = SelfOptimizingController()

# Start autonomous optimization
result = controller.optimize_autonomous()
print(f"Autonomous optimization completed: {result}")
```

### Security Setup

```python
from quantum_ctl.security import ProductionSecuritySystem, SecurityConfig

# Configure security
config = SecurityConfig(
    encryption_key_size=256,
    auth_timeout_minutes=15,
    threat_detection_enabled=True
)

# Initialize security system
security = ProductionSecuritySystem(config)
security.initialize_security()

# Validate security status
status = security.validate_security_status()
print(f"Security status: {status.is_secure}")
```
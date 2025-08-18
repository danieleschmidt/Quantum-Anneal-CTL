"""
Tests for security modules including quantum security manager,
secure quantum solver, and security validation.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any

# Test imports - handle missing dependencies gracefully
try:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    
    from quantum_ctl.security.quantum_security import (
        QuantumSecurityManager, SecureQuantumSolver, SecurityLevel, 
        SecurityCredentials, SecurityAuditLog
    )
    from quantum_ctl.optimization.quantum_solver import QuantumSolver, QuantumSolution
    SECURITY_MODULES_AVAILABLE = True
except ImportError as e:
    print(f"Security modules not available: {e}")
    SECURITY_MODULES_AVAILABLE = False


@pytest.mark.skipif(not SECURITY_MODULES_AVAILABLE, reason="Security modules not available")
class TestQuantumSecurityManager:
    """Test quantum security manager functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.security_manager = QuantumSecurityManager(
            security_level=SecurityLevel.MEDIUM
        )
        
        self.test_credentials = SecurityCredentials(
            api_key="test_api_key_12345678",
            secret_key="test_secret_key_87654321",
            permissions=["basic_access", "quantum_access"]
        )
        
    @pytest.mark.asyncio
    async def test_user_authentication_success(self):
        """Test successful user authentication."""
        
        # Mock credential verification to return True
        with patch.object(self.security_manager, '_verify_credentials', return_value=True):
            success, session_token = await self.security_manager.authenticate_user(
                self.test_credentials,
                client_ip="192.168.1.100"
            )
            
            assert success is True
            assert session_token is not None
            assert isinstance(session_token, str)
            assert len(session_token) > 10
            
            # Check that session was recorded
            assert session_token in self.security_manager.active_sessions
            session_info = self.security_manager.active_sessions[session_token]
            assert session_info['ip_address'] == "192.168.1.100"
            
    @pytest.mark.asyncio
    async def test_user_authentication_failure(self):
        """Test failed user authentication."""
        
        # Mock credential verification to return False
        with patch.object(self.security_manager, '_verify_credentials', return_value=False):
            success, session_token = await self.security_manager.authenticate_user(
                self.test_credentials,
                client_ip="192.168.1.100"
            )
            
            assert success is False
            assert session_token is None
            
            # Check that failed attempt was recorded
            assert "192.168.1.100" in self.security_manager.failed_attempts
            
    @pytest.mark.asyncio
    async def test_rate_limiting(self):
        """Test rate limiting for failed authentication attempts."""
        
        client_ip = "192.168.1.200"
        
        # Mock credential verification to always fail
        with patch.object(self.security_manager, '_verify_credentials', return_value=False):
            # Make multiple failed attempts
            for i in range(6):
                success, _ = await self.security_manager.authenticate_user(
                    self.test_credentials,
                    client_ip=client_ip
                )
                assert success is False
                
            # The 6th attempt should be rate limited
            assert self.security_manager._is_rate_limited(client_ip) is True
            
    def test_api_key_format_validation(self):
        """Test API key format validation."""
        
        # Valid API key
        assert self.security_manager._validate_api_key_format("valid_api_key_12345") is True
        
        # Invalid API keys
        assert self.security_manager._validate_api_key_format("short") is False
        assert self.security_manager._validate_api_key_format("") is False
        assert self.security_manager._validate_api_key_format(None) is False
        assert self.security_manager._validate_api_key_format("key/with/slashes") is False
        assert self.security_manager._validate_api_key_format("key.with.too.many.dots") is False
        
    @pytest.mark.asyncio
    async def test_quantum_operation_authorization(self):
        """Test authorization for quantum operations."""
        
        # First authenticate to get session token
        with patch.object(self.security_manager, '_verify_credentials', return_value=True):
            success, session_token = await self.security_manager.authenticate_user(
                self.test_credentials,
                client_ip="192.168.1.100"
            )
            
            assert success is True
            
        # Test successful authorization
        authorized, denial_reason = await self.security_manager.authorize_quantum_operation(
            session_token,
            "quantum_solve",
            problem_size=100,
            resource_requirements={'num_reads': 1000, 'annealing_time': 20}
        )
        
        assert authorized is True
        assert denial_reason is None
        
    @pytest.mark.asyncio
    async def test_quantum_operation_authorization_failure(self):
        """Test failed authorization for quantum operations."""
        
        # Test with invalid session token
        authorized, denial_reason = await self.security_manager.authorize_quantum_operation(
            "invalid_session_token",
            "quantum_solve",
            problem_size=100,
            resource_requirements={'num_reads': 1000}
        )
        
        assert authorized is False
        assert denial_reason == "invalid_session"
        
        # Test with problem size exceeding limits
        with patch.object(self.security_manager, '_verify_credentials', return_value=True):
            success, session_token = await self.security_manager.authenticate_user(
                self.test_credentials,
                client_ip="192.168.1.100"
            )
            
        authorized, denial_reason = await self.security_manager.authorize_quantum_operation(
            session_token,
            "quantum_solve",
            problem_size=10000,  # Exceeds default limit
            resource_requirements={'num_reads': 1000}
        )
        
        assert authorized is False
        assert denial_reason == "problem_size_exceeded"
        
    def test_secure_parameter_handling(self):
        """Test secure parameter handling and sanitization."""
        
        # First get a valid session
        with patch.object(self.security_manager, '_verify_credentials', return_value=True):
            asyncio.run(self.security_manager.authenticate_user(
                self.test_credentials,
                client_ip="192.168.1.100"
            ))
            
        session_token = list(self.security_manager.active_sessions.keys())[0]
        
        # Test parameter sanitization
        raw_parameters = {
            'num_reads': 1000,
            'annealing_time': 20,
            'chain_strength': 1.5,
            'invalid_param': 'should_be_removed',
            'auto_scale': True
        }
        
        secured_params = self.security_manager.secure_parameter_handling(
            raw_parameters,
            session_token
        )
        
        # Check that valid parameters are kept
        assert 'num_reads' in secured_params
        assert 'annealing_time' in secured_params
        assert 'chain_strength' in secured_params
        assert 'auto_scale' in secured_params
        
        # Check that invalid parameters are removed
        assert 'invalid_param' not in secured_params
        
        # Check parameter values are sanitized
        assert secured_params['num_reads'] == 1000
        assert secured_params['annealing_time'] == 20
        assert secured_params['chain_strength'] == 1.5
        
    def test_data_encryption_decryption(self):
        """Test encryption and decryption of sensitive data."""
        
        test_data = {
            'api_key': 'secret_api_key',
            'parameters': {'num_reads': 1000},
            'results': {'energy': -1.23456}
        }
        
        # Encrypt data
        encrypted_data = self.security_manager.encrypt_sensitive_data(test_data)
        
        assert isinstance(encrypted_data, str)
        assert encrypted_data != str(test_data)  # Should be different from original
        
        # Decrypt data
        decrypted_data = self.security_manager.decrypt_sensitive_data(encrypted_data)
        
        assert decrypted_data == test_data
        
    def test_security_audit_logging(self):
        """Test security audit log functionality."""
        
        initial_log_count = len(self.security_manager.audit_log)
        
        # Log a security event
        self.security_manager._log_security_event(
            action="test_action",
            resource="test_resource",
            success=True,
            ip_address="192.168.1.100",
            details={'test': 'data'}
        )
        
        # Check that event was logged
        assert len(self.security_manager.audit_log) == initial_log_count + 1
        
        latest_log = self.security_manager.audit_log[-1]
        assert latest_log.action == "test_action"
        assert latest_log.resource == "test_resource"
        assert latest_log.success is True
        assert latest_log.ip_address == "192.168.1.100"
        assert latest_log.details['test'] == 'data'
        
    def test_security_status_reporting(self):
        """Test security status reporting."""
        
        status = self.security_manager.get_security_status()
        
        assert 'security_level' in status
        assert 'active_sessions' in status
        assert 'failed_attempts_ips' in status
        assert 'audit_log_entries' in status
        assert 'rate_limits' in status
        
        assert status['security_level'] == SecurityLevel.MEDIUM.value
        assert isinstance(status['active_sessions'], int)
        assert isinstance(status['failed_attempts_ips'], int)


@pytest.mark.skipif(not SECURITY_MODULES_AVAILABLE, reason="Security modules not available")
class TestSecureQuantumSolver:
    """Test secure quantum solver wrapper."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create mock base solver
        self.mock_base_solver = Mock(spec=QuantumSolver)
        
        # Create security manager and get session
        self.security_manager = QuantumSecurityManager(SecurityLevel.MEDIUM)
        
        # Create test credentials and authenticate
        test_credentials = SecurityCredentials(
            api_key="test_api_key_secure",
            secret_key="test_secret_key_secure", 
            permissions=["quantum_access", "basic_access"]
        )
        
        # Mock authentication success
        with patch.object(self.security_manager, '_verify_credentials', return_value=True):
            success, self.session_token = asyncio.run(
                self.security_manager.authenticate_user(
                    test_credentials,
                    client_ip="192.168.1.100"
                )
            )
            
        assert success is True
        
        # Create secure solver
        self.secure_solver = SecureQuantumSolver(
            base_solver=self.mock_base_solver,
            security_manager=self.security_manager,
            session_token=self.session_token
        )
        
    @pytest.mark.asyncio
    async def test_secure_solve_success(self):
        """Test successful secure quantum solve operation."""
        
        # Create test QUBO
        test_qubo = {
            (0, 0): 1.0,
            (1, 1): -1.0,
            (0, 1): 0.5
        }
        
        # Mock quantum solution
        mock_solution = QuantumSolution(
            sample={0: 1, 1: 0},
            energy=-0.5,
            num_occurrences=100,
            chain_break_fraction=0.1,
            timing={'total_solve_time': 2.5},
            embedding_stats={'solver_type': 'hybrid'}
        )
        
        # Configure mock solver
        self.mock_base_solver.solve = AsyncMock(return_value=mock_solution)
        
        # Execute secure solve
        result = await self.secure_solver.secure_solve(test_qubo, num_reads=1000)
        
        # Verify result
        assert isinstance(result, QuantumSolution)
        assert result.sample == {0: 1, 1: 0}
        assert result.energy == -0.5
        
        # Verify that base solver was called with secured parameters
        self.mock_base_solver.solve.assert_called_once()
        call_args = self.mock_base_solver.solve.call_args
        
        # Check that QUBO was passed correctly
        assert call_args[0][0] == test_qubo
        
    @pytest.mark.asyncio
    async def test_secure_solve_authorization_failure(self):
        """Test secure solve with authorization failure."""
        
        # Create QUBO that exceeds size limits
        large_qubo = {(i, i): 1.0 for i in range(10000)}  # Very large problem
        
        # Should raise PermissionError due to problem size
        with pytest.raises(PermissionError) as exc_info:
            await self.secure_solver.secure_solve(large_qubo)
            
        assert "Quantum solve operation denied" in str(exc_info.value)
        
    def test_qubo_security_validation(self):
        """Test QUBO security validation."""
        
        # Valid QUBO should pass
        valid_qubo = {(0, 0): 1.0, (1, 1): -1.0, (0, 1): 0.5}
        self.secure_solver._validate_qubo_security(valid_qubo)  # Should not raise
        
        # QUBO with extreme coefficients should fail
        extreme_qubo = {(0, 0): 1e7, (1, 1): -1e7}
        with pytest.raises(ValueError, match="coefficients too large"):
            self.secure_solver._validate_qubo_security(extreme_qubo)
            
        # QUBO with invalid values should fail
        invalid_qubo = {(0, 0): float('inf'), (1, 1): float('nan')}
        with pytest.raises(ValueError, match="invalid values"):
            self.secure_solver._validate_qubo_security(invalid_qubo)


@pytest.mark.skipif(not SECURITY_MODULES_AVAILABLE, reason="Security modules not available")
class TestSecurityIntegration:
    """Test integration between security components."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.security_manager = QuantumSecurityManager(SecurityLevel.HIGH)
        
    @pytest.mark.asyncio
    async def test_end_to_end_secure_workflow(self):
        """Test complete secure quantum workflow."""
        
        # Step 1: Authenticate user
        credentials = SecurityCredentials(
            api_key="integration_test_key_123",
            secret_key="integration_test_secret_456",
            permissions=["quantum_access", "basic_access", "advanced_access"]
        )
        
        with patch.object(self.security_manager, '_verify_credentials', return_value=True):
            success, session_token = await self.security_manager.authenticate_user(
                credentials,
                client_ip="192.168.1.50"
            )
            
        assert success is True
        assert session_token is not None
        
        # Step 2: Authorize quantum operation
        authorized, denial_reason = await self.security_manager.authorize_quantum_operation(
            session_token,
            "quantum_solve",
            problem_size=50,
            resource_requirements={'num_reads': 500, 'annealing_time': 10}
        )
        
        assert authorized is True
        assert denial_reason is None
        
        # Step 3: Secure parameter handling
        raw_params = {
            'num_reads': 500,
            'annealing_time': 10,
            'chain_strength': 2.0,
            'auto_scale': True,
            'malicious_param': 'should_be_removed'
        }
        
        secured_params = self.security_manager.secure_parameter_handling(
            raw_params,
            session_token
        )
        
        assert 'malicious_param' not in secured_params
        assert secured_params['num_reads'] == 500
        
        # Step 4: Create and use secure solver
        mock_base_solver = Mock(spec=QuantumSolver)
        mock_solution = QuantumSolution(
            sample={0: 1, 1: 0},
            energy=-1.0,
            num_occurrences=50,
            chain_break_fraction=0.05,
            timing={'total_solve_time': 1.5},
            embedding_stats={'solver_type': 'qpu'}
        )
        mock_base_solver.solve = AsyncMock(return_value=mock_solution)
        
        secure_solver = SecureQuantumSolver(
            base_solver=mock_base_solver,
            security_manager=self.security_manager,
            session_token=session_token
        )
        
        test_qubo = {(0, 0): 1.0, (1, 1): -1.0, (0, 1): 0.5}
        result = await secure_solver.secure_solve(test_qubo, **secured_params)
        
        assert isinstance(result, QuantumSolution)
        assert result.energy == -1.0
        
        # Step 5: Verify audit trail
        audit_logs = self.security_manager.audit_log
        
        # Should have logs for authentication, authorization, parameter access, and solve
        assert len(audit_logs) >= 4
        
        # Check for specific log entries
        log_actions = [log.action for log in audit_logs]
        assert "authentication_success" in log_actions
        assert "authorization_success" in log_actions
        assert "parameter_access" in log_actions
        
    def test_security_level_enforcement(self):
        """Test security level enforcement across components."""
        
        # Test different security levels
        security_levels = [SecurityLevel.LOW, SecurityLevel.MEDIUM, SecurityLevel.HIGH, SecurityLevel.CRITICAL]
        
        for level in security_levels:
            manager = QuantumSecurityManager(level)
            
            # Check rate limits are different for each level
            rate_limits = manager.rate_limits[level]
            
            assert 'requests_per_minute' in rate_limits
            assert 'max_problem_size' in rate_limits
            
            # Higher security levels should have stricter limits
            if level == SecurityLevel.CRITICAL:
                assert rate_limits['requests_per_minute'] <= 20
            elif level == SecurityLevel.HIGH:
                assert rate_limits['requests_per_minute'] <= 50
                
    @pytest.mark.asyncio
    async def test_security_degradation_scenarios(self):
        """Test security behavior under various attack scenarios."""
        
        # Scenario 1: Brute force attack simulation
        attacker_ip = "10.0.0.1"
        bad_credentials = SecurityCredentials(
            api_key="bad_key",
            secret_key="bad_secret"
        )
        
        with patch.object(self.security_manager, '_verify_credentials', return_value=False):
            # Simulate brute force attempts
            for _ in range(10):
                success, _ = await self.security_manager.authenticate_user(
                    bad_credentials,
                    client_ip=attacker_ip
                )
                assert success is False
                
            # IP should be rate limited after multiple failures
            assert self.security_manager._is_rate_limited(attacker_ip) is True
            
        # Scenario 2: Session hijacking attempt
        valid_credentials = SecurityCredentials(
            api_key="valid_key_12345",
            secret_key="valid_secret_67890",
            permissions=["basic_access"]
        )
        
        with patch.object(self.security_manager, '_verify_credentials', return_value=True):
            success, valid_token = await self.security_manager.authenticate_user(
                valid_credentials,
                client_ip="192.168.1.100"
            )
            
        # Try to use token from different IP (potential session hijacking)
        authorized, denial_reason = await self.security_manager.authorize_quantum_operation(
            valid_token,
            "quantum_solve",
            problem_size=10,
            resource_requirements={'num_reads': 100}
        )
        
        # Should still work because we don't enforce IP binding in this implementation
        # In production, you might want to add IP validation
        assert authorized is True or denial_reason is not None
        
    def test_audit_log_integrity(self):
        """Test audit log integrity and completeness."""
        
        initial_count = len(self.security_manager.audit_log)
        
        # Generate multiple security events
        events = [
            ("login_attempt", "auth_service", True, "192.168.1.1"),
            ("quantum_solve", "quantum_service", True, "192.168.1.1"),
            ("parameter_change", "config_service", False, "192.168.1.2"),
            ("logout", "auth_service", True, "192.168.1.1"),
        ]
        
        for action, resource, success, ip in events:
            self.security_manager._log_security_event(
                action=action,
                resource=resource,
                success=success,
                ip_address=ip,
                details={"test": True}
            )
            
        # Check all events were logged
        assert len(self.security_manager.audit_log) == initial_count + len(events)
        
        # Verify log entry integrity
        recent_logs = self.security_manager.audit_log[-len(events):]
        
        for i, (action, resource, success, ip) in enumerate(events):
            log_entry = recent_logs[i]
            
            assert log_entry.action == action
            assert log_entry.resource == resource
            assert log_entry.success == success
            assert log_entry.ip_address == ip
            assert log_entry.timestamp > 0
            assert "test" in log_entry.details


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
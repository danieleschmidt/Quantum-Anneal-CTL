"""
Tests for security and authentication modules.
"""

import pytest
import time
from unittest.mock import MagicMock, patch

from quantum_ctl.utils.security import (
    InputValidator, CryptoManager, AuthenticationManager, AuthorizationManager,
    SecurityAuditLogger, SecurityContext, UserRole, get_crypto_manager
)


class TestInputValidator:
    """Test input validation utilities."""
    
    def test_validate_building_id(self):
        """Test building ID validation."""
        # Valid building IDs
        assert InputValidator.validate_building_id("building_001")
        assert InputValidator.validate_building_id("main-building")
        assert InputValidator.validate_building_id("B123")
        
        # Invalid building IDs
        assert not InputValidator.validate_building_id("")  # Empty
        assert not InputValidator.validate_building_id("a" * 60)  # Too long
        assert not InputValidator.validate_building_id("building@001")  # Special chars
        assert not InputValidator.validate_building_id("building 001")  # Spaces
    
    def test_validate_zone_id(self):
        """Test zone ID validation."""
        assert InputValidator.validate_zone_id("zone_1")
        assert InputValidator.validate_zone_id("Z-001")
        
        assert not InputValidator.validate_zone_id("")
        assert not InputValidator.validate_zone_id("zone@1")
        assert not InputValidator.validate_zone_id("a" * 25)  # Too long
    
    def test_validate_temperature(self):
        """Test temperature validation."""
        assert InputValidator.validate_temperature(22.5)
        assert InputValidator.validate_temperature(-10)
        assert InputValidator.validate_temperature(50)
        
        assert not InputValidator.validate_temperature(-60)  # Too cold
        assert not InputValidator.validate_temperature(120)  # Too hot
        assert not InputValidator.validate_temperature("22")  # Wrong type
    
    def test_validate_power(self):
        """Test power validation."""
        assert InputValidator.validate_power(5.5)
        assert InputValidator.validate_power(0)
        assert InputValidator.validate_power(1000)
        
        assert not InputValidator.validate_power(-1)  # Negative
        assert not InputValidator.validate_power(15000)  # Too high
    
    def test_validate_percentage(self):
        """Test percentage validation."""
        assert InputValidator.validate_percentage(50.5)
        assert InputValidator.validate_percentage(0)
        assert InputValidator.validate_percentage(100)
        
        assert not InputValidator.validate_percentage(-1)
        assert not InputValidator.validate_percentage(101)
    
    def test_validate_user_id(self):
        """Test user ID validation."""
        assert InputValidator.validate_user_id("admin")
        assert InputValidator.validate_user_id("user_123")
        
        assert not InputValidator.validate_user_id("ab")  # Too short
        assert not InputValidator.validate_user_id("user@domain.com")  # Special chars
    
    def test_validate_ip_address(self):
        """Test IP address validation."""
        assert InputValidator.validate_ip_address("192.168.1.1")
        assert InputValidator.validate_ip_address("10.0.0.1")
        assert InputValidator.validate_ip_address("127.0.0.1")
        
        assert not InputValidator.validate_ip_address("256.1.1.1")  # Invalid octets
        assert not InputValidator.validate_ip_address("192.168.1")  # Incomplete
        assert not InputValidator.validate_ip_address("not.an.ip.address")
    
    def test_sanitize_string(self):
        """Test string sanitization."""
        # Test removing dangerous characters
        dangerous = '<script>alert("xss")</script>'
        sanitized = InputValidator.sanitize_string(dangerous)
        assert '<' not in sanitized
        assert '>' not in sanitized
        assert '"' not in sanitized
        
        # Test length truncation
        long_string = "a" * 300
        truncated = InputValidator.sanitize_string(long_string, max_length=100)
        assert len(truncated) == 100
    
    def test_validate_json_payload(self):
        """Test JSON payload validation."""
        schema = {
            'name': str,
            'age': int,
            'active': bool
        }
        
        # Valid payload
        valid_payload = {'name': 'John', 'age': 30, 'active': True}
        is_valid, message = InputValidator.validate_json_payload(valid_payload, schema)
        assert is_valid
        assert message == "Valid"
        
        # Invalid payload - missing field
        invalid_payload = {'name': 'John', 'age': 30}
        is_valid, message = InputValidator.validate_json_payload(invalid_payload, schema)
        assert not is_valid
        assert "Missing required field" in message
        
        # Invalid payload - wrong type
        invalid_type = {'name': 'John', 'age': '30', 'active': True}
        is_valid, message = InputValidator.validate_json_payload(invalid_type, schema)
        assert not is_valid
        assert "Invalid type" in message


class TestCryptoManager:
    """Test cryptographic utilities."""
    
    @pytest.fixture
    def crypto_manager(self):
        """Create crypto manager for testing."""
        # Use a test key to avoid environment dependencies
        test_key = CryptoManager().fernet.generate_key()
        return CryptoManager(test_key)
    
    def test_encrypt_decrypt_string(self, crypto_manager):
        """Test string encryption and decryption."""
        original_data = "sensitive information"
        
        encrypted = crypto_manager.encrypt_data(original_data)
        decrypted = crypto_manager.decrypt_data(encrypted)
        
        assert isinstance(encrypted, bytes)
        assert decrypted == original_data
    
    def test_encrypt_decrypt_dict(self, crypto_manager):
        """Test dictionary encryption and decryption."""
        original_data = {"username": "admin", "password": "secret123"}
        
        encrypted = crypto_manager.encrypt_data(original_data)
        decrypted = crypto_manager.decrypt_data(encrypted)
        
        assert isinstance(encrypted, bytes)
        assert isinstance(decrypted, dict)
        assert decrypted == original_data
    
    def test_generate_secure_token(self, crypto_manager):
        """Test secure token generation."""
        token1 = crypto_manager.generate_secure_token(32)
        token2 = crypto_manager.generate_secure_token(32)
        
        assert len(token1) > 0
        assert len(token2) > 0
        assert token1 != token2  # Should be random
    
    def test_hash_password(self, crypto_manager):
        """Test password hashing."""
        password = "secure_password123"
        
        hashed, salt = crypto_manager.hash_password(password)
        
        assert len(hashed) == 32  # SHA256 output length
        assert len(salt) == 32  # Salt length
        assert isinstance(hashed, bytes)
        assert isinstance(salt, bytes)
    
    def test_verify_password(self, crypto_manager):
        """Test password verification."""
        password = "secure_password123"
        wrong_password = "wrong_password"
        
        hashed, salt = crypto_manager.hash_password(password)
        
        # Correct password should verify
        assert crypto_manager.verify_password(password, hashed, salt)
        
        # Wrong password should not verify
        assert not crypto_manager.verify_password(wrong_password, hashed, salt)
    
    def test_signature_creation_verification(self, crypto_manager):
        """Test HMAC signature creation and verification."""
        data = "important data to sign"
        secret = "shared_secret_key"
        
        signature = crypto_manager.create_signature(data, secret)
        
        # Valid signature should verify
        assert crypto_manager.verify_signature(data, signature, secret)
        
        # Tampered data should not verify
        assert not crypto_manager.verify_signature(data + "tampered", signature, secret)
        
        # Wrong secret should not verify
        assert not crypto_manager.verify_signature(data, signature, "wrong_secret")


class TestSecurityContext:
    """Test security context management."""
    
    def test_security_context_creation(self):
        """Test creating security context."""
        context = SecurityContext(
            user_id="test_user",
            role=UserRole.OPERATOR,
            permissions=["read", "write"],
            session_id="session_123"
        )
        
        assert context.user_id == "test_user"
        assert context.role == UserRole.OPERATOR
        assert "read" in context.permissions
        assert context.session_id == "session_123"
    
    def test_has_permission(self):
        """Test permission checking."""
        context = SecurityContext(
            user_id="operator",
            role=UserRole.OPERATOR,
            permissions=["read", "write"],
            session_id="session_123"
        )
        
        assert context.has_permission("read")
        assert context.has_permission("write")
        assert not context.has_permission("delete")
        
        # Admin should have all permissions
        admin_context = SecurityContext(
            user_id="admin",
            role=UserRole.ADMIN,
            permissions=["read"],
            session_id="session_456"
        )
        
        assert admin_context.has_permission("delete")  # Admin override
    
    def test_expiration_check(self):
        """Test session expiration."""
        # Fresh context should not be expired
        context = SecurityContext(
            user_id="user",
            role=UserRole.READONLY,
            permissions=["read"],
            session_id="session_789"
        )
        
        assert not context.is_expired(3600)  # 1 hour timeout
        
        # Old context should be expired
        context.last_accessed = time.time() - 7200  # 2 hours ago
        assert context.is_expired(3600)  # 1 hour timeout
    
    def test_refresh_access(self):
        """Test access time refresh."""
        context = SecurityContext(
            user_id="user",
            role=UserRole.READONLY,
            permissions=["read"],
            session_id="session_refresh"
        )
        
        original_time = context.last_accessed
        time.sleep(0.01)  # Small delay
        context.refresh_access()
        
        assert context.last_accessed > original_time


class TestAuthenticationManager:
    """Test authentication manager."""
    
    @pytest.fixture
    def auth_manager(self):
        """Create authentication manager for testing."""
        crypto_manager = CryptoManager()
        return AuthenticationManager(crypto_manager)
    
    def test_authentication_success(self, auth_manager):
        """Test successful authentication."""
        # Mock user lookup to return valid user
        with patch.object(auth_manager, '_lookup_user') as mock_lookup:
            mock_lookup.return_value = {
                'role': 'admin',
                'permissions': ['read', 'write', 'admin'],
                'password_hash': b'mock_hash',
                'salt': b'mock_salt'
            }
            
            # Mock password verification to return True
            with patch.object(auth_manager.crypto, 'verify_password') as mock_verify:
                mock_verify.return_value = True
                
                context = auth_manager.authenticate_user("admin", "password123", "192.168.1.100")
                
                assert context is not None
                assert context.user_id == "admin"
                assert context.role == UserRole.ADMIN
                assert context.source_ip == "192.168.1.100"
    
    def test_authentication_failure_invalid_user(self, auth_manager):
        """Test authentication failure with invalid user."""
        with patch.object(auth_manager, '_lookup_user') as mock_lookup:
            mock_lookup.return_value = None  # User not found
            
            context = auth_manager.authenticate_user("nonexistent", "password", "192.168.1.100")
            assert context is None
    
    def test_authentication_failure_wrong_password(self, auth_manager):
        """Test authentication failure with wrong password."""
        with patch.object(auth_manager, '_lookup_user') as mock_lookup:
            mock_lookup.return_value = {
                'role': 'operator',
                'permissions': ['read', 'write'],
                'password_hash': b'mock_hash',
                'salt': b'mock_salt'
            }
            
            with patch.object(auth_manager.crypto, 'verify_password') as mock_verify:
                mock_verify.return_value = False  # Wrong password
                
                context = auth_manager.authenticate_user("operator", "wrongpassword", "192.168.1.100")
                assert context is None
    
    def test_lockout_mechanism(self, auth_manager):
        """Test user lockout after failed attempts."""
        # Simulate multiple failed attempts
        for _ in range(auth_manager.max_failed_attempts):
            auth_manager._record_failed_attempt("test_user")
        
        # User should now be locked out
        assert auth_manager._is_locked_out("test_user")
        
        # Authentication should be blocked
        with patch.object(auth_manager, '_lookup_user'):
            context = auth_manager.authenticate_user("test_user", "password", "192.168.1.100")
            assert context is None
    
    def test_session_validation(self, auth_manager):
        """Test session validation."""
        # Create mock context
        mock_context = SecurityContext(
            user_id="test_user",
            role=UserRole.OPERATOR,
            permissions=["read"],
            session_id="valid_session"
        )
        
        auth_manager.active_sessions["valid_session"] = mock_context
        
        # Valid session should return context
        validated = auth_manager.validate_session("valid_session")
        assert validated is not None
        assert validated.user_id == "test_user"
        
        # Invalid session should return None
        assert auth_manager.validate_session("invalid_session") is None
    
    def test_session_expiration_cleanup(self, auth_manager):
        """Test expired session cleanup."""
        # Create expired session
        expired_context = SecurityContext(
            user_id="expired_user",
            role=UserRole.READONLY,
            permissions=["read"],
            session_id="expired_session"
        )
        expired_context.last_accessed = time.time() - 7200  # 2 hours ago
        
        auth_manager.active_sessions["expired_session"] = expired_context
        
        # Cleanup should remove expired session
        removed_count = auth_manager.cleanup_expired_sessions()
        assert removed_count == 1
        assert "expired_session" not in auth_manager.active_sessions


class TestAuthorizationManager:
    """Test authorization manager."""
    
    @pytest.fixture
    def authz_manager(self):
        """Create authorization manager for testing."""
        return AuthorizationManager()
    
    @pytest.fixture
    def admin_context(self):
        """Create admin security context."""
        return SecurityContext(
            user_id="admin",
            role=UserRole.ADMIN,
            permissions=["read", "write", "delete", "admin"],
            session_id="admin_session"
        )
    
    @pytest.fixture
    def operator_context(self):
        """Create operator security context."""
        return SecurityContext(
            user_id="operator",
            role=UserRole.OPERATOR,
            permissions=["read", "write"],
            session_id="operator_session"
        )
    
    @pytest.fixture
    def readonly_context(self):
        """Create readonly security context."""
        return SecurityContext(
            user_id="readonly",
            role=UserRole.READONLY,
            permissions=["read"],
            session_id="readonly_session"
        )
    
    def test_admin_access(self, authz_manager, admin_context):
        """Test admin has access to everything."""
        # Admin should have access to all resources
        assert authz_manager.check_permission(admin_context, "/api/users", "delete")
        assert authz_manager.check_permission(admin_context, "/api/settings", "write")
        assert authz_manager.check_permission(admin_context, "/api/control/emergency", "write")
    
    def test_operator_permissions(self, authz_manager, operator_context):
        """Test operator permissions."""
        # Operator should have read/write access to most resources
        assert authz_manager.check_permission(operator_context, "/api/buildings", "read")
        assert authz_manager.check_permission(operator_context, "/api/control/optimize", "write")
        
        # But not admin-only resources
        assert not authz_manager.check_permission(operator_context, "/api/users", "write")
        assert not authz_manager.check_permission(operator_context, "/api/control/emergency", "write")
    
    def test_readonly_permissions(self, authz_manager, readonly_context):
        """Test readonly permissions."""
        # Readonly should only have read access
        assert authz_manager.check_permission(readonly_context, "/api/buildings", "read")
        assert authz_manager.check_permission(readonly_context, "/api/metrics", "read")
        
        # No write access
        assert not authz_manager.check_permission(readonly_context, "/api/control/optimize", "write")
        assert not authz_manager.check_permission(readonly_context, "/api/settings", "write")
    
    def test_wildcard_resource_matching(self, authz_manager, operator_context):
        """Test wildcard resource pattern matching."""
        # Should match wildcard patterns
        assert authz_manager.check_permission(operator_context, "/api/buildings/building1/zones", "read")
        assert authz_manager._match_resource_pattern(
            "/api/buildings/*/zones", 
            "/api/buildings/building1/zones"
        )


class TestSecurityAuditLogger:
    """Test security audit logging."""
    
    @pytest.fixture
    def audit_logger(self):
        """Create security audit logger for testing."""
        return SecurityAuditLogger()
    
    def test_log_authentication_success(self, audit_logger):
        """Test logging successful authentication."""
        audit_logger.log_authentication("test_user", "192.168.1.100", True)
        
        events = audit_logger.get_security_events(1)  # Last hour
        assert len(events) > 0
        
        auth_event = events[-1]  # Last event
        assert auth_event.event_type == "authentication"
        assert auth_event.user_id == "test_user"
        assert auth_event.result == "success"
    
    def test_log_authorization_denied(self, audit_logger):
        """Test logging authorization denial."""
        audit_logger.log_authorization(
            "operator", "192.168.1.100", "/api/users", "delete", False
        )
        
        events = audit_logger.get_security_events(1, event_type="authorization")
        assert len(events) > 0
        
        authz_event = events[-1]
        assert authz_event.event_type == "authorization"
        assert authz_event.result == "denied"
        assert authz_event.resource == "/api/users"
    
    def test_security_summary(self, audit_logger):
        """Test security event summary."""
        # Generate some test events
        audit_logger.log_authentication("user1", "192.168.1.1", True)
        audit_logger.log_authentication("user2", "192.168.1.2", False)
        audit_logger.log_authorization("user1", "192.168.1.1", "/api/test", "read", True)
        
        summary = audit_logger.get_security_summary()
        
        assert summary['total_events_24h'] >= 3
        assert 'authentication' in summary['event_type_breakdown']
        assert 'authorization' in summary['event_type_breakdown']
        assert summary['latest_event'] is not None


class TestIntegration:
    """Test security component integration."""
    
    def test_crypto_manager_singleton(self):
        """Test that crypto manager singleton works."""
        manager1 = get_crypto_manager()
        manager2 = get_crypto_manager()
        
        assert manager1 is manager2  # Should be same instance
    
    def test_end_to_end_authentication_flow(self):
        """Test complete authentication flow."""
        crypto_manager = CryptoManager()
        auth_manager = AuthenticationManager(crypto_manager)
        authz_manager = AuthorizationManager()
        audit_logger = SecurityAuditLogger()
        
        # Mock successful authentication
        with patch.object(auth_manager, '_lookup_user') as mock_lookup:
            mock_lookup.return_value = {
                'role': 'operator',
                'permissions': ['read', 'write'],
                'password_hash': b'mock_hash',
                'salt': b'mock_salt'
            }
            
            with patch.object(crypto_manager, 'verify_password') as mock_verify:
                mock_verify.return_value = True
                
                # Authenticate user
                context = auth_manager.authenticate_user("operator", "password", "192.168.1.100")
                assert context is not None
                
                # Log authentication
                audit_logger.log_authentication("operator", "192.168.1.100", True)
                
                # Check authorization
                can_read = authz_manager.check_permission(context, "/api/buildings", "read")
                can_delete_users = authz_manager.check_permission(context, "/api/users", "delete")
                
                assert can_read  # Operator should be able to read
                assert not can_delete_users  # But not delete users
                
                # Log authorization attempts
                audit_logger.log_authorization("operator", "192.168.1.100", "/api/buildings", "read", True)
                audit_logger.log_authorization("operator", "192.168.1.100", "/api/users", "delete", False)
                
                # Verify audit trail
                events = audit_logger.get_security_events(1)
                assert len(events) >= 3  # Auth + 2 authz events
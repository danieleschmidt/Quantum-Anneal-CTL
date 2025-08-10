"""Authentication and authorization for API."""

import jwt
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional
from passlib.context import CryptContext
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import logging

from .models import UserModel
from ..utils.config import get_config

logger = logging.getLogger(__name__)

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT settings
SECRET_KEY = "quantum-hvac-secret-key-change-in-production"  # Should be from config
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours

# Bearer token scheme
security = HTTPBearer()


class AuthenticationError(Exception):
    """Custom authentication error."""
    pass


class AuthorizationError(Exception):
    """Custom authorization error."""
    pass


# Mock user database - in production, use real database
USERS_DB = {
    "admin": {
        "username": "admin",
        "full_name": "System Administrator",
        "email": "admin@quantum-hvac.com",
        "hashed_password": pwd_context.hash("admin123"),  # Change in production
        "disabled": False,
        "roles": ["admin", "operator", "viewer"]
    },
    "operator": {
        "username": "operator",
        "full_name": "HVAC Operator", 
        "email": "operator@quantum-hvac.com",
        "hashed_password": pwd_context.hash("operator123"),
        "disabled": False,
        "roles": ["operator", "viewer"]
    },
    "viewer": {
        "username": "viewer",
        "full_name": "System Viewer",
        "email": "viewer@quantum-hvac.com", 
        "hashed_password": pwd_context.hash("viewer123"),
        "disabled": False,
        "roles": ["viewer"]
    }
}


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Generate password hash."""
    return pwd_context.hash(password)


def get_user(username: str) -> Optional[Dict]:
    """Get user from database."""
    return USERS_DB.get(username)


def authenticate_user(username: str, password: str) -> Optional[Dict]:
    """Authenticate user with username and password."""
    user = get_user(username)
    if not user:
        return None
    if not verify_password(password, user["hashed_password"]):
        return None
    return user


def create_access_token(data: Dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token."""
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def verify_token(token: str) -> Dict:
    """Verify JWT token and extract payload."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise AuthenticationError("Invalid token")
        
        # Check if token is expired
        exp = payload.get("exp")
        if exp and datetime.now(timezone.utc).timestamp() > exp:
            raise AuthenticationError("Token expired")
        
        return payload
    except jwt.PyJWTError as e:
        raise AuthenticationError(f"Token verification failed: {e}")


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> UserModel:
    """Get current authenticated user."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = verify_token(credentials.credentials)
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except AuthenticationError:
        raise credentials_exception
    
    user = get_user(username=username)
    if user is None:
        raise credentials_exception
    
    if user.get("disabled", False):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User account disabled"
        )
    
    return UserModel(**user)


async def get_current_active_user(current_user: UserModel = Depends(get_current_user)) -> UserModel:
    """Get current active user."""
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


def require_role(required_role: str):
    """Decorator to require specific role."""
    def role_checker(current_user: UserModel = Depends(get_current_active_user)):
        if required_role not in current_user.roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Operation requires '{required_role}' role"
            )
        return current_user
    return role_checker


def require_any_role(required_roles: List[str]):
    """Decorator to require any of the specified roles."""
    def role_checker(current_user: UserModel = Depends(get_current_active_user)):
        if not any(role in current_user.roles for role in required_roles):
            roles_str = ", ".join(required_roles)
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Operation requires one of these roles: {roles_str}"
            )
        return current_user
    return role_checker


# Role-based access dependencies
require_admin = require_role("admin")
require_operator = require_any_role(["admin", "operator"])
require_viewer = require_any_role(["admin", "operator", "viewer"])


class APIKeyAuth:
    """API Key authentication for service-to-service communication."""
    
    def __init__(self):
        # In production, store these securely
        self.valid_api_keys = {
            "quantum-service-key": {"name": "Quantum Service", "roles": ["admin"]},
            "dashboard-service-key": {"name": "Dashboard Service", "roles": ["viewer"]},
            "integration-service-key": {"name": "Integration Service", "roles": ["operator"]}
        }
    
    def verify_api_key(self, api_key: str) -> Optional[Dict]:
        """Verify API key."""
        return self.valid_api_keys.get(api_key)
    
    async def __call__(self, credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict:
        """Authenticate using API key."""
        api_key_info = self.verify_api_key(credentials.credentials)
        if not api_key_info:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key"
            )
        return api_key_info


api_key_auth = APIKeyAuth()


def log_security_event(event_type: str, user: str, details: Dict = None):
    """Log security-related events."""
    log_data = {
        "event_type": event_type,
        "user": user,
        "timestamp": datetime.utcnow().isoformat(),
        "details": details or {}
    }
    
    # In production, send to security monitoring system
    logger.info(f"Security event: {log_data}")


class RateLimiter:
    """Simple in-memory rate limiter."""
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Dict[str, List[datetime]] = {}
    
    def is_allowed(self, key: str) -> bool:
        """Check if request is allowed under rate limit."""
        now = datetime.utcnow()
        window_start = now - timedelta(seconds=self.window_seconds)
        
        # Clean old requests
        if key in self.requests:
            self.requests[key] = [
                req_time for req_time in self.requests[key] 
                if req_time > window_start
            ]
        else:
            self.requests[key] = []
        
        # Check limit
        if len(self.requests[key]) >= self.max_requests:
            return False
        
        # Add current request
        self.requests[key].append(now)
        return True


# Rate limiter instance
rate_limiter = RateLimiter()


def check_rate_limit(request_key: str):
    """Dependency to check rate limiting."""
    def rate_limit_checker():
        if not rate_limiter.is_allowed(request_key):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded"
            )
    return rate_limit_checker
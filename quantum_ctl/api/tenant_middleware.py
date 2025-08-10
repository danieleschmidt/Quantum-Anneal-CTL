"""Tenant-aware middleware for FastAPI."""

import logging
from typing import Optional, Callable
from contextvars import ContextVar

from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from ..core.tenant_manager import get_tenant_manager, TenantContext, Tenant
from ..utils.structured_logging import StructuredLogger

logger = StructuredLogger("quantum_ctl.middleware")

# Context variable for tenant context
tenant_context_var: ContextVar[Optional[TenantContext]] = ContextVar('tenant_context', default=None)


class TenantMiddleware(BaseHTTPMiddleware):
    """Middleware to resolve tenant context from request."""
    
    def __init__(self, app, tenant_manager=None):
        super().__init__(app)
        self.tenant_manager = tenant_manager or get_tenant_manager()
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with tenant context."""
        tenant = await self._resolve_tenant(request)
        
        if tenant:
            # Create tenant context
            user_roles = getattr(request.state, 'user_roles', [])
            context = self.tenant_manager.get_tenant_context(tenant, user_roles)
            
            # Set context variable
            token = tenant_context_var.set(context)
            
            # Add to request state
            request.state.tenant_context = context
            request.state.tenant = tenant
            
            logger.debug(
                "Tenant context set",
                tenant_id=str(tenant.id),
                tenant_name=tenant.name,
                path=request.url.path
            )
            
            try:
                response = await call_next(request)
                return response
            finally:
                tenant_context_var.reset(token)
        else:
            # No tenant context - might be public endpoint
            try:
                response = await call_next(request)
                return response
            except Exception as e:
                logger.error(f"Request failed without tenant context: {e}")
                raise
    
    async def _resolve_tenant(self, request: Request) -> Optional[Tenant]:
        """Resolve tenant from request."""
        # Try different tenant resolution strategies
        
        # 1. Custom domain
        host = request.headers.get("host", "").split(":")[0]
        if host and not self._is_main_domain(host):
            tenant = await self.tenant_manager.get_tenant_by_domain(host)
            if tenant:
                logger.debug(f"Resolved tenant by domain: {host}")
                return tenant
        
        # 2. Subdomain
        if "." in host:
            subdomain = host.split(".")[0]
            if subdomain and subdomain != "www" and subdomain != "api":
                tenant = await self.tenant_manager.get_tenant_by_slug(subdomain)
                if tenant:
                    logger.debug(f"Resolved tenant by subdomain: {subdomain}")
                    return tenant
        
        # 3. Header-based (for API clients)
        tenant_header = request.headers.get("x-tenant-id") or request.headers.get("x-tenant-slug")
        if tenant_header:
            # Try as UUID first, then as slug
            tenant = await self.tenant_manager.get_tenant_by_id(tenant_header)
            if not tenant:
                tenant = await self.tenant_manager.get_tenant_by_slug(tenant_header)
            
            if tenant:
                logger.debug(f"Resolved tenant by header: {tenant_header}")
                return tenant
        
        # 4. Path-based (e.g., /tenant/slug/api/...)
        path_parts = request.url.path.strip("/").split("/")
        if len(path_parts) >= 2 and path_parts[0] == "tenant":
            tenant_slug = path_parts[1]
            tenant = await self.tenant_manager.get_tenant_by_slug(tenant_slug)
            if tenant:
                logger.debug(f"Resolved tenant by path: {tenant_slug}")
                return tenant
        
        # 5. Query parameter (fallback)
        tenant_param = request.query_params.get("tenant")
        if tenant_param:
            tenant = await self.tenant_manager.get_tenant_by_id(tenant_param)
            if not tenant:
                tenant = await self.tenant_manager.get_tenant_by_slug(tenant_param)
            
            if tenant:
                logger.debug(f"Resolved tenant by query param: {tenant_param}")
                return tenant
        
        logger.debug(f"No tenant resolved for request: {request.url}")
        return None
    
    def _is_main_domain(self, host: str) -> bool:
        """Check if host is main application domain."""
        main_domains = ["localhost", "127.0.0.1", "quantum-hvac.com", "api.quantum-hvac.com"]
        return any(domain in host for domain in main_domains)


def get_current_tenant_context() -> Optional[TenantContext]:
    """Get current tenant context from context variable."""
    return tenant_context_var.get()


def require_tenant_context():
    """Dependency to require tenant context."""
    def _require_tenant():
        context = get_current_tenant_context()
        if context is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Tenant context required"
            )
        return context
    return _require_tenant


def require_tenant_permission(resource: str, action: str):
    """Dependency to require tenant permission."""
    def _require_permission():
        context = get_current_tenant_context()
        if context is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Tenant context required"
            )
        
        if not context.is_authorized(resource, action):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions for {action} on {resource}"
            )
        
        return context
    return _require_permission


async def check_tenant_resource_limits(
    tenant_context: TenantContext,
    resource: str,
    requested_amount: int = 1
) -> None:
    """Check tenant resource limits and raise exception if exceeded."""
    tenant_manager = get_tenant_manager()
    
    allowed, message = await tenant_manager.check_resource_limits(
        tenant_context.get_tenant_id(),
        resource,
        requested_amount
    )
    
    if not allowed:
        logger.warning(
            "Tenant resource limit exceeded",
            tenant_id=tenant_context.get_tenant_id(),
            resource=resource,
            requested_amount=requested_amount,
            reason=message
        )
        
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Resource limit exceeded: {message}"
        )


class ResourceLimitMiddleware(BaseHTTPMiddleware):
    """Middleware to enforce tenant resource limits."""
    
    def __init__(self, app, tenant_manager=None):
        super().__init__(app)
        self.tenant_manager = tenant_manager or get_tenant_manager()
        
        # Define resource mapping for different endpoints
        self.endpoint_resources = {
            "/buildings": "buildings",
            "/optimize": "optimizations",
            "/quantum": "quantum_access"
        }
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Check resource limits before processing request."""
        tenant_context = get_current_tenant_context()
        
        if tenant_context and request.method in ["POST", "PUT"]:
            # Determine resource type from path
            resource = self._get_resource_from_path(request.url.path)
            
            if resource:
                try:
                    await check_tenant_resource_limits(tenant_context, resource)
                except HTTPException:
                    # Log and re-raise
                    logger.warning(
                        "Request blocked by resource limits",
                        tenant_id=tenant_context.get_tenant_id(),
                        path=request.url.path,
                        resource=resource
                    )
                    raise
        
        response = await call_next(request)
        return response
    
    def _get_resource_from_path(self, path: str) -> Optional[str]:
        """Determine resource type from request path."""
        for path_pattern, resource in self.endpoint_resources.items():
            if path_pattern in path:
                return resource
        return None


def tenant_scoped_key(key: str, tenant_context: TenantContext = None) -> str:
    """Create tenant-scoped cache/resource key."""
    context = tenant_context or get_current_tenant_context()
    if context:
        return f"tenant:{context.get_tenant_id()}:{key}"
    return key


def log_tenant_activity(
    activity: str,
    details: dict = None,
    tenant_context: TenantContext = None
):
    """Log tenant activity for audit and analytics."""
    context = tenant_context or get_current_tenant_context()
    
    if context:
        logger.info(
            f"Tenant activity: {activity}",
            tenant_id=context.get_tenant_id(),
            tenant_name=context.get_tenant().name,
            activity=activity,
            details=details or {}
        )
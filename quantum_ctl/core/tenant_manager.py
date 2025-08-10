"""Multi-tenant architecture support for quantum HVAC control system."""

import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum

from sqlalchemy import Column, String, DateTime, Boolean, JSON, Integer, Float
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

from ..database.models import Base
from ..utils.structured_logging import StructuredLogger

logger = StructuredLogger("quantum_ctl.tenant")


class TenantStatus(Enum):
    """Tenant status levels."""
    ACTIVE = "active"
    SUSPENDED = "suspended"
    INACTIVE = "inactive"


class SubscriptionTier(Enum):
    """Subscription tiers with different resource limits."""
    BASIC = "basic"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"


@dataclass
class ResourceLimits:
    """Resource limits for tenants."""
    max_buildings: int = 5
    max_zones_per_building: int = 50
    max_optimizations_per_day: int = 100
    max_api_requests_per_hour: int = 1000
    max_data_retention_days: int = 90
    quantum_access: bool = True
    advanced_features: bool = False
    
    @classmethod
    def for_tier(cls, tier: SubscriptionTier) -> 'ResourceLimits':
        """Get resource limits for subscription tier."""
        limits_map = {
            SubscriptionTier.BASIC: cls(
                max_buildings=2,
                max_zones_per_building=20,
                max_optimizations_per_day=50,
                max_api_requests_per_hour=500,
                max_data_retention_days=30,
                quantum_access=False,
                advanced_features=False
            ),
            SubscriptionTier.PROFESSIONAL: cls(
                max_buildings=10,
                max_zones_per_building=100,
                max_optimizations_per_day=500,
                max_api_requests_per_hour=5000,
                max_data_retention_days=180,
                quantum_access=True,
                advanced_features=True
            ),
            SubscriptionTier.ENTERPRISE: cls(
                max_buildings=100,
                max_zones_per_building=500,
                max_optimizations_per_day=10000,
                max_api_requests_per_hour=50000,
                max_data_retention_days=365,
                quantum_access=True,
                advanced_features=True
            )
        }
        return limits_map.get(tier, cls())


@dataclass
class UsageStats:
    """Current usage statistics for a tenant."""
    buildings_count: int = 0
    zones_count: int = 0
    optimizations_today: int = 0
    api_requests_hour: int = 0
    storage_mb: float = 0.0
    quantum_time_minutes: float = 0.0
    last_activity: Optional[datetime] = None


class Tenant(Base):
    """Tenant model for multi-tenancy."""
    
    __tablename__ = "tenants"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(200), nullable=False)
    slug = Column(String(100), unique=True, nullable=False)
    domain = Column(String(100), nullable=True)  # Custom domain
    
    # Status and subscription
    status = Column(String(20), default=TenantStatus.ACTIVE.value)
    subscription_tier = Column(String(20), default=SubscriptionTier.BASIC.value)
    
    # Resource limits (stored as JSON)
    resource_limits = Column(JSON, nullable=False)
    
    # Billing and contact
    billing_email = Column(String(200), nullable=True)
    admin_contact = Column(String(200), nullable=True)
    
    # Metadata and configuration
    metadata = Column(JSON, nullable=True)
    settings = Column(JSON, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_activity = Column(DateTime, nullable=True)
    
    # Trial and billing
    trial_end_date = Column(DateTime, nullable=True)
    subscription_end_date = Column(DateTime, nullable=True)


class TenantUsage(Base):
    """Usage tracking for tenants."""
    
    __tablename__ = "tenant_usage"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    
    # Usage metrics
    date = Column(DateTime, nullable=False, index=True)
    buildings_count = Column(Integer, default=0)
    zones_count = Column(Integer, default=0)
    optimizations_count = Column(Integer, default=0)
    api_requests_count = Column(Integer, default=0)
    storage_mb = Column(Float, default=0.0)
    quantum_time_minutes = Column(Float, default=0.0)
    
    # Cost tracking
    compute_cost_usd = Column(Float, default=0.0)
    storage_cost_usd = Column(Float, default=0.0)
    quantum_cost_usd = Column(Float, default=0.0)
    
    # Metadata
    metadata = Column(JSON, nullable=True)


class ITenantContext(ABC):
    """Interface for tenant context."""
    
    @abstractmethod
    def get_tenant_id(self) -> str:
        """Get current tenant ID."""
        pass
    
    @abstractmethod
    def get_tenant(self) -> Optional[Tenant]:
        """Get current tenant."""
        pass
    
    @abstractmethod
    def is_authorized(self, resource: str, action: str) -> bool:
        """Check if current tenant is authorized for action."""
        pass


class TenantContext:
    """Tenant context for request processing."""
    
    def __init__(self, tenant: Tenant, user_roles: List[str] = None):
        self.tenant = tenant
        self.user_roles = user_roles or []
        self._resource_limits = ResourceLimits(**tenant.resource_limits)
    
    def get_tenant_id(self) -> str:
        """Get current tenant ID."""
        return str(self.tenant.id)
    
    def get_tenant(self) -> Tenant:
        """Get current tenant."""
        return self.tenant
    
    def get_resource_limits(self) -> ResourceLimits:
        """Get resource limits for tenant."""
        return self._resource_limits
    
    def is_authorized(self, resource: str, action: str) -> bool:
        """Check if current tenant is authorized for action."""
        # Basic authorization based on tenant status
        if self.tenant.status != TenantStatus.ACTIVE.value:
            return False
        
        # Check subscription tier permissions
        tier = SubscriptionTier(self.tenant.subscription_tier)
        
        # Define permission matrix
        permissions = {
            SubscriptionTier.BASIC: {
                "buildings": ["read", "write"],
                "optimization": ["read"],
                "quantum": [],
                "analytics": ["read"]
            },
            SubscriptionTier.PROFESSIONAL: {
                "buildings": ["read", "write", "delete"],
                "optimization": ["read", "write"],
                "quantum": ["read", "write"],
                "analytics": ["read", "write"]
            },
            SubscriptionTier.ENTERPRISE: {
                "buildings": ["read", "write", "delete", "admin"],
                "optimization": ["read", "write", "admin"],
                "quantum": ["read", "write", "admin"],
                "analytics": ["read", "write", "admin"],
                "system": ["read", "write"]
            }
        }
        
        allowed_actions = permissions.get(tier, {}).get(resource, [])
        return action in allowed_actions
    
    def can_create_building(self) -> bool:
        """Check if tenant can create new building."""
        # This would check against current usage
        return True  # Simplified for now
    
    def can_run_optimization(self) -> bool:
        """Check if tenant can run optimization."""
        if not self.is_authorized("optimization", "write"):
            return False
        
        # Check daily limits (would query usage from database)
        return True  # Simplified for now


class TenantManager:
    """Manager for multi-tenant operations."""
    
    def __init__(self, database_manager):
        self.database_manager = database_manager
        self._tenant_cache: Dict[str, Tenant] = {}
        self._usage_cache: Dict[str, UsageStats] = {}
    
    async def create_tenant(
        self,
        name: str,
        slug: str,
        subscription_tier: SubscriptionTier = SubscriptionTier.BASIC,
        admin_email: str = None,
        trial_days: int = 30,
        metadata: Dict[str, Any] = None
    ) -> Tenant:
        """Create new tenant."""
        
        # Calculate resource limits
        limits = ResourceLimits.for_tier(subscription_tier)
        
        # Set trial end date
        trial_end = datetime.utcnow().replace(
            day=datetime.utcnow().day + trial_days
        ) if trial_days > 0 else None
        
        tenant = Tenant(
            name=name,
            slug=slug,
            subscription_tier=subscription_tier.value,
            resource_limits=limits.__dict__,
            admin_contact=admin_email,
            billing_email=admin_email,
            trial_end_date=trial_end,
            metadata=metadata or {}
        )
        
        async with self.database_manager.get_async_session() as session:
            session.add(tenant)
            await session.commit()
            await session.refresh(tenant)
        
        # Cache the tenant
        self._tenant_cache[str(tenant.id)] = tenant
        self._tenant_cache[tenant.slug] = tenant
        
        logger.info(
            "Created new tenant",
            tenant_id=str(tenant.id),
            tenant_name=name,
            subscription_tier=subscription_tier.value
        )
        
        return tenant
    
    async def get_tenant_by_id(self, tenant_id: str) -> Optional[Tenant]:
        """Get tenant by ID."""
        # Check cache first
        if tenant_id in self._tenant_cache:
            return self._tenant_cache[tenant_id]
        
        async with self.database_manager.get_async_session() as session:
            result = await session.execute(
                select(Tenant).where(Tenant.id == tenant_id)
            )
            tenant = result.scalar_one_or_none()
        
        if tenant:
            self._tenant_cache[tenant_id] = tenant
            self._tenant_cache[tenant.slug] = tenant
        
        return tenant
    
    async def get_tenant_by_slug(self, slug: str) -> Optional[Tenant]:
        """Get tenant by slug."""
        # Check cache first
        if slug in self._tenant_cache:
            return self._tenant_cache[slug]
        
        async with self.database_manager.get_async_session() as session:
            result = await session.execute(
                select(Tenant).where(Tenant.slug == slug)
            )
            tenant = result.scalar_one_or_none()
        
        if tenant:
            self._tenant_cache[str(tenant.id)] = tenant
            self._tenant_cache[slug] = tenant
        
        return tenant
    
    async def get_tenant_by_domain(self, domain: str) -> Optional[Tenant]:
        """Get tenant by custom domain."""
        async with self.database_manager.get_async_session() as session:
            result = await session.execute(
                select(Tenant).where(Tenant.domain == domain)
            )
            return result.scalar_one_or_none()
    
    async def update_tenant_usage(
        self,
        tenant_id: str,
        usage_data: Dict[str, Any],
        date: datetime = None
    ) -> None:
        """Update tenant usage statistics."""
        if date is None:
            date = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        
        usage = TenantUsage(
            tenant_id=tenant_id,
            date=date,
            **usage_data
        )
        
        async with self.database_manager.get_async_session() as session:
            session.add(usage)
            await session.commit()
    
    async def get_tenant_usage(
        self,
        tenant_id: str,
        start_date: datetime = None,
        end_date: datetime = None
    ) -> List[TenantUsage]:
        """Get tenant usage history."""
        if start_date is None:
            start_date = datetime.utcnow().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        if end_date is None:
            end_date = datetime.utcnow()
        
        async with self.database_manager.get_async_session() as session:
            result = await session.execute(
                select(TenantUsage)
                .where(TenantUsage.tenant_id == tenant_id)
                .where(TenantUsage.date >= start_date)
                .where(TenantUsage.date <= end_date)
                .order_by(TenantUsage.date)
            )
            return result.scalars().all()
    
    async def check_resource_limits(
        self,
        tenant_id: str,
        resource: str,
        requested_amount: int = 1
    ) -> tuple[bool, str]:
        """Check if tenant can allocate resources."""
        tenant = await self.get_tenant_by_id(tenant_id)
        if not tenant:
            return False, "Tenant not found"
        
        if tenant.status != TenantStatus.ACTIVE.value:
            return False, "Tenant account is not active"
        
        limits = ResourceLimits(**tenant.resource_limits)
        
        # Get current usage (would query from database in production)
        current_usage = self._usage_cache.get(tenant_id, UsageStats())
        
        # Check specific resource limits
        if resource == "buildings":
            if current_usage.buildings_count + requested_amount > limits.max_buildings:
                return False, f"Exceeds maximum buildings limit ({limits.max_buildings})"
        elif resource == "optimizations":
            if current_usage.optimizations_today + requested_amount > limits.max_optimizations_per_day:
                return False, f"Exceeds daily optimizations limit ({limits.max_optimizations_per_day})"
        elif resource == "api_requests":
            if current_usage.api_requests_hour + requested_amount > limits.max_api_requests_per_hour:
                return False, f"Exceeds hourly API requests limit ({limits.max_api_requests_per_hour})"
        elif resource == "quantum_access":
            if not limits.quantum_access:
                return False, "Quantum access not available for subscription tier"
        
        return True, "Resource allocation allowed"
    
    async def list_tenants(
        self,
        status: Optional[TenantStatus] = None,
        subscription_tier: Optional[SubscriptionTier] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Tenant]:
        """List tenants with filters."""
        async with self.database_manager.get_async_session() as session:
            query = select(Tenant)
            
            if status:
                query = query.where(Tenant.status == status.value)
            
            if subscription_tier:
                query = query.where(Tenant.subscription_tier == subscription_tier.value)
            
            query = query.order_by(Tenant.created_at.desc()).offset(offset).limit(limit)
            
            result = await session.execute(query)
            return result.scalars().all()
    
    async def update_tenant_status(
        self,
        tenant_id: str,
        status: TenantStatus,
        reason: str = None
    ) -> bool:
        """Update tenant status."""
        async with self.database_manager.get_async_session() as session:
            result = await session.execute(
                select(Tenant).where(Tenant.id == tenant_id)
            )
            tenant = result.scalar_one_or_none()
            
            if not tenant:
                return False
            
            old_status = tenant.status
            tenant.status = status.value
            tenant.updated_at = datetime.utcnow()
            
            await session.commit()
            
            # Update cache
            if str(tenant.id) in self._tenant_cache:
                self._tenant_cache[str(tenant.id)].status = status.value
            if tenant.slug in self._tenant_cache:
                self._tenant_cache[tenant.slug].status = status.value
            
            logger.info(
                "Updated tenant status",
                tenant_id=tenant_id,
                old_status=old_status,
                new_status=status.value,
                reason=reason
            )
            
            return True
    
    def get_tenant_context(self, tenant: Tenant, user_roles: List[str] = None) -> TenantContext:
        """Create tenant context for request processing."""
        return TenantContext(tenant, user_roles)


# Import necessary SQLAlchemy components
from sqlalchemy import select

# Global tenant manager instance
_tenant_manager: Optional[TenantManager] = None


def get_tenant_manager() -> TenantManager:
    """Get global tenant manager instance."""
    global _tenant_manager
    if _tenant_manager is None:
        from ..database.manager import get_database_manager
        _tenant_manager = TenantManager(get_database_manager())
    return _tenant_manager
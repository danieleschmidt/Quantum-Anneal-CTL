"""Database connection and session management."""

import logging
from contextlib import asynccontextmanager, contextmanager
from typing import Optional, AsyncGenerator, Generator
from urllib.parse import urlparse

from sqlalchemy import create_engine, MetaData
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool

from .models import Base
from ..utils.config import get_config

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages database connections and sessions."""
    
    def __init__(self, database_url: Optional[str] = None, echo: bool = False):
        """Initialize database manager.
        
        Args:
            database_url: Database connection URL
            echo: Enable SQL logging
        """
        self.database_url = database_url or self._get_database_url()
        self.echo = echo
        
        # Parse URL to determine if async is needed
        parsed = urlparse(self.database_url)
        self.is_async = parsed.scheme in ('postgresql+asyncpg', 'mysql+aiomysql', 'sqlite+aiosqlite')
        
        # Initialize engines
        if self.is_async:
            self.async_engine = create_async_engine(
                self.database_url,
                echo=self.echo,
                poolclass=QueuePool,
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True,
                pool_recycle=3600
            )
            self.async_session_factory = async_sessionmaker(
                self.async_engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            # Create sync engine for migrations
            sync_url = self.database_url.replace('+asyncpg', '').replace('+aiomysql', '').replace('+aiosqlite', '')
            self.sync_engine = create_engine(sync_url, echo=self.echo)
        else:
            self.sync_engine = create_engine(
                self.database_url,
                echo=self.echo,
                poolclass=QueuePool,
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True,
                pool_recycle=3600
            )
            
        self.sync_session_factory = sessionmaker(
            self.sync_engine,
            expire_on_commit=False
        )
    
    def _get_database_url(self) -> str:
        """Get database URL from configuration."""
        config = get_config()
        
        # Try to get from config
        if hasattr(config, 'database') and hasattr(config.database, 'url'):
            return config.database.url
        
        # Default to in-memory SQLite for development
        logger.warning("No database URL configured, using in-memory SQLite")
        return "sqlite:///:memory:"
    
    async def initialize(self) -> None:
        """Initialize database schema."""
        if self.is_async:
            async with self.async_engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
        else:
            Base.metadata.create_all(self.sync_engine)
        
        logger.info("Database schema initialized")
    
    def initialize_sync(self) -> None:
        """Initialize database schema synchronously.""" 
        Base.metadata.create_all(self.sync_engine)
        logger.info("Database schema initialized (sync)")
    
    @asynccontextmanager
    async def get_async_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get async database session."""
        if not self.is_async:
            raise RuntimeError("Async session requested but database is not configured for async")
            
        async with self.async_session_factory() as session:
            try:
                yield session
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()
    
    @contextmanager
    def get_sync_session(self) -> Generator[Session, None, None]:
        """Get synchronous database session."""
        with self.sync_session_factory() as session:
            try:
                yield session
            except Exception:
                session.rollback()
                raise
    
    async def health_check(self) -> bool:
        """Check database connectivity."""
        try:
            if self.is_async:
                async with self.get_async_session() as session:
                    await session.execute("SELECT 1")
            else:
                with self.get_sync_session() as session:
                    session.execute("SELECT 1")
            return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False
    
    async def close(self) -> None:
        """Close database connections."""
        if self.is_async and hasattr(self, 'async_engine'):
            await self.async_engine.dispose()
        
        if hasattr(self, 'sync_engine'):
            self.sync_engine.dispose()
        
        logger.info("Database connections closed")


# Global database manager instance
_db_manager: Optional[DatabaseManager] = None


def get_database_manager() -> DatabaseManager:
    """Get global database manager instance."""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager


async def init_database(database_url: Optional[str] = None) -> None:
    """Initialize database with optional URL override."""
    global _db_manager
    _db_manager = DatabaseManager(database_url)
    await _db_manager.initialize()


def init_database_sync(database_url: Optional[str] = None) -> None:
    """Initialize database synchronously."""
    global _db_manager
    _db_manager = DatabaseManager(database_url) 
    _db_manager.initialize_sync()
"""
Environment configuration management for production deployments.
"""

import os
import yaml
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from dataclasses import dataclass, field
import logging
from functools import lru_cache


@dataclass
class DatabaseConfig:
    """Database configuration."""
    url: str
    echo_sql: bool = False
    pool_size: int = 20
    max_overflow: int = 30
    pool_timeout: int = 30
    pool_recycle: int = 3600
    pool_pre_ping: bool = True


@dataclass
class RedisConfig:
    """Redis configuration."""
    url: str
    password: Optional[str] = None
    max_connections: int = 50
    retry_on_timeout: bool = True
    ttl_default: int = 300
    ttl_building_state: int = 60
    ttl_weather_data: int = 900
    ttl_optimization_results: int = 1800


@dataclass
class DWaveConfig:
    """D-Wave quantum configuration."""
    api_token: str
    endpoint: str = "https://cloud.dwavesys.com/sapi"
    default_solver: str = "auto"
    hybrid_time_limit: int = 30
    qpu_num_reads: int = 1000
    qpu_annealing_time: int = 20
    fallback_enabled: bool = True
    fallback_timeout: float = 5.0


@dataclass
class SecurityConfig:
    """Security configuration."""
    secret_key: str
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7
    password_min_length: int = 12
    rate_limit_enabled: bool = True
    requests_per_minute: int = 100
    cors_origins: List[str] = field(default_factory=list)


@dataclass
class MonitoringConfig:
    """Monitoring configuration."""
    enabled: bool = True
    metrics_enabled: bool = True
    metrics_port: int = 9090
    health_endpoint: str = "/health"
    request_tracking: bool = True
    slow_query_threshold: float = 1.0


@dataclass
class BMSConfig:
    """BMS integration configuration."""
    connection_timeout: float = 10.0
    read_timeout: float = 5.0
    retry_attempts: int = 3
    retry_delay: float = 2.0
    bacnet_device_discovery_timeout: float = 30.0
    modbus_keepalive_interval: int = 30
    mqtt_keepalive: int = 60


@dataclass
class ControlConfig:
    """Control system configuration."""
    loop_interval: int = 300
    safety_timeout: int = 30
    horizon_hours: int = 24
    time_step_minutes: int = 15
    max_iterations: int = 1000
    convergence_tolerance: float = 1e-4


@dataclass
class EnvironmentConfig:
    """Complete environment configuration."""
    environment: str
    debug: bool
    database: DatabaseConfig
    redis: RedisConfig
    dwave: DWaveConfig
    security: SecurityConfig
    monitoring: MonitoringConfig
    bms: BMSConfig
    control: ControlConfig
    
    # Server settings
    server_host: str = "0.0.0.0"
    server_port: int = 8000
    server_workers: int = 4
    
    # Feature flags
    quantum_optimization: bool = True
    advanced_analytics: bool = True
    ml_forecasting: bool = True
    
    # Logging
    log_level: str = "INFO"
    
    @property
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.environment.lower() == "production"
    
    @property
    def is_development(self) -> bool:
        """Check if running in development."""
        return self.environment.lower() == "development"
    
    @property
    def is_testing(self) -> bool:
        """Check if running in testing."""
        return self.environment.lower() == "testing"


class ConfigurationManager:
    """Configuration management for different environments."""
    
    def __init__(self, config_dir: Optional[Path] = None):
        self.logger = logging.getLogger(__name__)
        
        # Determine config directory
        if config_dir:
            self.config_dir = Path(config_dir)
        else:
            # Look for config directory in common locations
            possible_paths = [
                Path("config"),
                Path("../config"),
                Path("/app/config"),
                Path.home() / ".quantum-hvac" / "config"
            ]
            
            self.config_dir = None
            for path in possible_paths:
                if path.exists():
                    self.config_dir = path
                    break
            
            if not self.config_dir:
                self.config_dir = Path("config")
                self.config_dir.mkdir(exist_ok=True)
        
        self.logger.info(f"Using config directory: {self.config_dir}")
        
        # Environment detection
        self.environment = os.getenv("ENVIRONMENT", "development").lower()
        
    def load_config(self) -> EnvironmentConfig:
        """Load configuration for current environment."""
        config_file = self.config_dir / f"{self.environment}.yaml"
        
        if not config_file.exists():
            self.logger.warning(f"Config file not found: {config_file}")
            config_file = self.config_dir / "default.yaml"
            
            if not config_file.exists():
                self.logger.info("No config files found, using environment variables only")
                return self._load_from_env_vars()
        
        try:
            with open(config_file, 'r') as f:
                config_data = yaml.safe_load(f)
            
            # Substitute environment variables
            config_data = self._substitute_env_vars(config_data)
            
            # Parse configuration
            return self._parse_config(config_data)
            
        except Exception as e:
            self.logger.error(f"Failed to load config from {config_file}: {e}")
            return self._load_from_env_vars()
    
    def _substitute_env_vars(self, obj: Any) -> Any:
        """Recursively substitute environment variables in config."""
        if isinstance(obj, dict):
            return {key: self._substitute_env_vars(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._substitute_env_vars(item) for item in obj]
        elif isinstance(obj, str):
            # Simple ${VAR} substitution
            if obj.startswith("${") and obj.endswith("}"):
                var_name = obj[2:-1]
                default_value = None
                
                # Handle ${VAR:default} syntax
                if ":" in var_name:
                    var_name, default_value = var_name.split(":", 1)
                
                return os.getenv(var_name, default_value or obj)
            return obj
        else:
            return obj
    
    def _parse_config(self, config_data: Dict[str, Any]) -> EnvironmentConfig:
        """Parse configuration data into structured config objects."""
        try:
            # Database config
            db_config = config_data.get("database", {})
            database = DatabaseConfig(
                url=db_config.get("url", os.getenv("DATABASE_URL", "sqlite:///quantum_hvac.db")),
                echo_sql=db_config.get("echo_sql", False),
                pool_size=db_config.get("pool_size", 20),
                max_overflow=db_config.get("max_overflow", 30),
                pool_timeout=db_config.get("pool_timeout", 30),
                pool_recycle=db_config.get("pool_recycle", 3600),
                pool_pre_ping=db_config.get("pool_pre_ping", True)
            )
            
            # Redis config
            redis_config = config_data.get("redis", {})
            redis = RedisConfig(
                url=redis_config.get("url", os.getenv("REDIS_URL", "redis://localhost:6379/0")),
                password=redis_config.get("password"),
                max_connections=redis_config.get("max_connections", 50),
                retry_on_timeout=redis_config.get("retry_on_timeout", True),
                ttl_default=redis_config.get("ttl", {}).get("default", 300),
                ttl_building_state=redis_config.get("ttl", {}).get("building_state", 60),
                ttl_weather_data=redis_config.get("ttl", {}).get("weather_data", 900),
                ttl_optimization_results=redis_config.get("ttl", {}).get("optimization_results", 1800)
            )
            
            # D-Wave config
            dwave_config = config_data.get("dwave", {})
            dwave = DWaveConfig(
                api_token=dwave_config.get("api_token", os.getenv("DWAVE_API_TOKEN", "")),
                endpoint=dwave_config.get("endpoint", "https://cloud.dwavesys.com/sapi"),
                default_solver=dwave_config.get("default_solver", "auto"),
                hybrid_time_limit=dwave_config.get("solvers", {}).get("hybrid", {}).get("time_limit", 30),
                qpu_num_reads=dwave_config.get("solvers", {}).get("qpu", {}).get("num_reads", 1000),
                qpu_annealing_time=dwave_config.get("solvers", {}).get("qpu", {}).get("annealing_time", 20),
                fallback_enabled=dwave_config.get("fallback", {}).get("enabled", True),
                fallback_timeout=dwave_config.get("fallback", {}).get("timeout", 5.0)
            )
            
            # Security config
            security_config = config_data.get("security", {})
            security = SecurityConfig(
                secret_key=security_config.get("secret_key", os.getenv("SECRET_KEY", "dev-secret-key")),
                algorithm=security_config.get("algorithm", "HS256"),
                access_token_expire_minutes=security_config.get("access_token_expire_minutes", 30),
                refresh_token_expire_days=security_config.get("refresh_token_expire_days", 7),
                password_min_length=security_config.get("password_min_length", 12),
                rate_limit_enabled=security_config.get("rate_limit", {}).get("enabled", True),
                requests_per_minute=security_config.get("rate_limit", {}).get("requests_per_minute", 100),
                cors_origins=config_data.get("app", {}).get("cors_origins", [])
            )
            
            # Monitoring config
            monitoring_config = config_data.get("monitoring", {})
            monitoring = MonitoringConfig(
                enabled=monitoring_config.get("enabled", True),
                metrics_enabled=monitoring_config.get("metrics", {}).get("enabled", True),
                metrics_port=monitoring_config.get("metrics", {}).get("port", 9090),
                health_endpoint=monitoring_config.get("health", {}).get("endpoint", "/health"),
                request_tracking=monitoring_config.get("performance", {}).get("request_tracking", True),
                slow_query_threshold=monitoring_config.get("performance", {}).get("slow_query_threshold", 1.0)
            )
            
            # BMS config
            bms_config = config_data.get("bms", {})
            bms = BMSConfig(
                connection_timeout=bms_config.get("connection_timeout", 10.0),
                read_timeout=bms_config.get("read_timeout", 5.0),
                retry_attempts=bms_config.get("retry_attempts", 3),
                retry_delay=bms_config.get("retry_delay", 2.0),
                bacnet_device_discovery_timeout=bms_config.get("bacnet", {}).get("device_discovery_timeout", 30.0),
                modbus_keepalive_interval=bms_config.get("modbus", {}).get("keepalive_interval", 30),
                mqtt_keepalive=bms_config.get("mqtt", {}).get("keepalive", 60)
            )
            
            # Control config
            control_config = config_data.get("control", {})
            control = ControlConfig(
                loop_interval=control_config.get("loop_interval", 300),
                safety_timeout=control_config.get("safety_timeout", 30),
                horizon_hours=control_config.get("optimization", {}).get("horizon_hours", 24),
                time_step_minutes=control_config.get("optimization", {}).get("time_step_minutes", 15),
                max_iterations=control_config.get("optimization", {}).get("max_iterations", 1000),
                convergence_tolerance=control_config.get("optimization", {}).get("convergence_tolerance", 1e-4)
            )
            
            # Server config
            server_config = config_data.get("server", {})
            
            # Feature flags
            features = config_data.get("features", {})
            
            # App config
            app_config = config_data.get("app", {})
            
            return EnvironmentConfig(
                environment=app_config.get("environment", self.environment),
                debug=app_config.get("debug", False),
                database=database,
                redis=redis,
                dwave=dwave,
                security=security,
                monitoring=monitoring,
                bms=bms,
                control=control,
                server_host=server_config.get("host", "0.0.0.0"),
                server_port=server_config.get("port", 8000),
                server_workers=server_config.get("workers", 4),
                quantum_optimization=features.get("quantum_optimization", True),
                advanced_analytics=features.get("advanced_analytics", True),
                ml_forecasting=features.get("ml_forecasting", True),
                log_level=os.getenv("LOG_LEVEL", "INFO")
            )
            
        except Exception as e:
            self.logger.error(f"Failed to parse configuration: {e}")
            return self._load_from_env_vars()
    
    def _load_from_env_vars(self) -> EnvironmentConfig:
        """Load configuration from environment variables only."""
        self.logger.info("Loading configuration from environment variables")
        
        return EnvironmentConfig(
            environment=os.getenv("ENVIRONMENT", "development"),
            debug=os.getenv("DEBUG", "false").lower() == "true",
            
            database=DatabaseConfig(
                url=os.getenv("DATABASE_URL", "sqlite:///quantum_hvac.db")
            ),
            
            redis=RedisConfig(
                url=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
                password=os.getenv("REDIS_PASSWORD")
            ),
            
            dwave=DWaveConfig(
                api_token=os.getenv("DWAVE_API_TOKEN", "")
            ),
            
            security=SecurityConfig(
                secret_key=os.getenv("SECRET_KEY", "dev-secret-key"),
                cors_origins=os.getenv("CORS_ORIGINS", "").split(",") if os.getenv("CORS_ORIGINS") else []
            ),
            
            monitoring=MonitoringConfig(),
            bms=BMSConfig(),
            control=ControlConfig(),
            
            server_host=os.getenv("HOST", "0.0.0.0"),
            server_port=int(os.getenv("PORT", "8000")),
            server_workers=int(os.getenv("WORKERS", "4")),
            
            log_level=os.getenv("LOG_LEVEL", "INFO")
        )
    
    def validate_config(self, config: EnvironmentConfig) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []
        
        # Database validation
        if not config.database.url:
            issues.append("Database URL is required")
        
        # Security validation
        if config.is_production:
            if not config.security.secret_key or config.security.secret_key == "dev-secret-key":
                issues.append("Production SECRET_KEY must be set and not use default value")
                
            if not config.dwave.api_token:
                issues.append("D-Wave API token is required for production")
                
            if config.debug:
                issues.append("Debug mode should be disabled in production")
        
        # Port validation
        if not (1 <= config.server_port <= 65535):
            issues.append(f"Invalid server port: {config.server_port}")
        
        # Worker validation
        if config.server_workers < 1:
            issues.append("Server workers must be at least 1")
        
        return issues
    
    def save_config(self, config: EnvironmentConfig) -> bool:
        """Save configuration to file."""
        try:
            config_file = self.config_dir / f"{config.environment}.yaml"
            
            # Convert config to dict (simplified version)
            config_dict = {
                "app": {
                    "environment": config.environment,
                    "debug": config.debug,
                    "cors_origins": config.security.cors_origins
                },
                "server": {
                    "host": config.server_host,
                    "port": config.server_port,
                    "workers": config.server_workers
                },
                "database": {
                    "url": config.database.url,
                    "pool_size": config.database.pool_size
                },
                "redis": {
                    "url": config.redis.url,
                    "max_connections": config.redis.max_connections
                },
                "dwave": {
                    "api_token": "${DWAVE_API_TOKEN}",
                    "default_solver": config.dwave.default_solver
                },
                "security": {
                    "secret_key": "${SECRET_KEY}",
                    "algorithm": config.security.algorithm
                },
                "features": {
                    "quantum_optimization": config.quantum_optimization,
                    "advanced_analytics": config.advanced_analytics,
                    "ml_forecasting": config.ml_forecasting
                }
            }
            
            with open(config_file, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            
            self.logger.info(f"Configuration saved to {config_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save configuration: {e}")
            return False


# Global configuration instance
@lru_cache(maxsize=1)
def get_config() -> EnvironmentConfig:
    """Get global configuration instance."""
    config_manager = ConfigurationManager()
    config = config_manager.load_config()
    
    # Validate configuration
    issues = config_manager.validate_config(config)
    if issues:
        logger = logging.getLogger(__name__)
        for issue in issues:
            if config.is_production:
                logger.error(f"Configuration issue: {issue}")
            else:
                logger.warning(f"Configuration issue: {issue}")
    
    return config


# Convenience function for environment-specific settings
def is_production() -> bool:
    """Check if running in production environment."""
    return get_config().is_production


def is_development() -> bool:
    """Check if running in development environment."""
    return get_config().is_development


def is_testing() -> bool:
    """Check if running in testing environment."""
    return get_config().is_testing
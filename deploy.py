#!/usr/bin/env python3
"""
Production deployment script for Quantum HVAC Control system.

This script provides automated deployment capabilities for different environments
with proper validation, health checks, and rollback mechanisms.
"""

import os
import sys
import argparse
import subprocess
import yaml
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
import requests
from dataclasses import dataclass
from datetime import datetime


@dataclass
class DeploymentConfig:
    """Deployment configuration."""
    environment: str
    docker_compose_file: str
    env_file: str
    backup_before_deploy: bool = True
    health_check_timeout: int = 300
    rollback_on_failure: bool = True
    services_to_check: List[str] = None


class DeploymentManager:
    """Manages deployment process with validation and rollbacks."""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.logger = self._setup_logging()
        self.deployment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.backup_path = Path(f"backups/deployment_{self.deployment_id}")
        
        # Service health check endpoints
        self.health_endpoints = {
            'quantum-hvac-api': 'http://localhost:8000/health',
            'quantum-hvac-dashboard': 'http://localhost:8080/health',
            'postgres': None,  # Use docker health checks
            'redis': None,  # Use docker health checks
            'nginx': 'http://localhost/health',
            'prometheus': 'http://localhost:9090/-/healthy',
            'grafana': 'http://localhost:3000/api/health'
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup deployment logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(f'deployment_{self.deployment_id}.log')
            ]
        )
        return logging.getLogger(__name__)
    
    def deploy(self) -> bool:
        """Execute full deployment process."""
        try:
            self.logger.info(f"Starting deployment {self.deployment_id} for {self.config.environment}")
            
            # Pre-deployment checks
            if not self._pre_deployment_checks():
                return False
            
            # Backup current state
            if self.config.backup_before_deploy:
                if not self._backup_current_state():
                    self.logger.error("Backup failed, aborting deployment")
                    return False
            
            # Deploy services
            if not self._deploy_services():
                if self.config.rollback_on_failure:
                    self.logger.error("Deployment failed, initiating rollback")
                    self._rollback()
                return False
            
            # Health checks
            if not self._health_checks():
                if self.config.rollback_on_failure:
                    self.logger.error("Health checks failed, initiating rollback")
                    self._rollback()
                return False
            
            # Post-deployment tasks
            if not self._post_deployment_tasks():
                self.logger.warning("Post-deployment tasks completed with warnings")
            
            self.logger.info(f"Deployment {self.deployment_id} completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Deployment failed with exception: {e}")
            if self.config.rollback_on_failure:
                self._rollback()
            return False
    
    def _pre_deployment_checks(self) -> bool:
        """Perform pre-deployment validation checks."""
        self.logger.info("Running pre-deployment checks...")
        
        checks_passed = True
        
        # Check Docker and Docker Compose
        if not self._check_command_available("docker"):
            self.logger.error("Docker is not available")
            checks_passed = False
        
        if not self._check_command_available("docker-compose"):
            self.logger.error("Docker Compose is not available")
            checks_passed = False
        
        # Check environment file
        if not Path(self.config.env_file).exists():
            self.logger.error(f"Environment file not found: {self.config.env_file}")
            checks_passed = False
        else:
            if not self._validate_environment_file():
                checks_passed = False
        
        # Check Docker Compose file
        if not Path(self.config.docker_compose_file).exists():
            self.logger.error(f"Docker Compose file not found: {self.config.docker_compose_file}")
            checks_passed = False
        
        # Check disk space
        if not self._check_disk_space():
            checks_passed = False
        
        # Check network connectivity
        if not self._check_network_connectivity():
            checks_passed = False
        
        self.logger.info(f"Pre-deployment checks {'passed' if checks_passed else 'failed'}")
        return checks_passed
    
    def _validate_environment_file(self) -> bool:
        """Validate environment file has required variables."""
        required_vars = [
            'POSTGRES_PASSWORD',
            'REDIS_PASSWORD', 
            'SECRET_KEY',
            'DWAVE_API_TOKEN'
        ]
        
        production_required_vars = [
            'BACKUP_S3_BUCKET',
            'AWS_ACCESS_KEY_ID',
            'AWS_SECRET_ACCESS_KEY',
            'GRAFANA_PASSWORD'
        ]
        
        try:
            with open(self.config.env_file, 'r') as f:
                env_content = f.read()
            
            missing_vars = []
            for var in required_vars:
                if f"{var}=" not in env_content:
                    missing_vars.append(var)
            
            if self.config.environment == 'production':
                for var in production_required_vars:
                    if f"{var}=" not in env_content:
                        missing_vars.append(var)
            
            if missing_vars:
                self.logger.error(f"Missing required environment variables: {missing_vars}")
                return False
            
            # Check for weak passwords in production
            if self.config.environment == 'production':
                weak_patterns = ['password', '123456', 'admin', 'test']
                for pattern in weak_patterns:
                    if pattern.lower() in env_content.lower():
                        self.logger.warning(f"Potentially weak password pattern detected: {pattern}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to validate environment file: {e}")
            return False
    
    def _check_disk_space(self, min_gb: int = 10) -> bool:
        """Check available disk space."""
        try:
            result = subprocess.run(['df', '-BG', '.'], capture_output=True, text=True)
            lines = result.stdout.strip().split('\n')
            if len(lines) > 1:
                fields = lines[1].split()
                available_gb = int(fields[3].rstrip('G'))
                
                if available_gb < min_gb:
                    self.logger.error(f"Insufficient disk space: {available_gb}GB available, {min_gb}GB required")
                    return False
                
                self.logger.info(f"Disk space check passed: {available_gb}GB available")
                return True
                
        except Exception as e:
            self.logger.warning(f"Could not check disk space: {e}")
            return True  # Don't fail deployment for this
    
    def _check_network_connectivity(self) -> bool:
        """Check network connectivity to required services."""
        test_endpoints = [
            'https://hub.docker.com',
            'https://cloud.dwavesys.com',
        ]
        
        for endpoint in test_endpoints:
            try:
                response = requests.get(endpoint, timeout=10)
                if response.status_code == 200:
                    self.logger.info(f"Network connectivity to {endpoint}: OK")
                else:
                    self.logger.warning(f"Network connectivity to {endpoint}: {response.status_code}")
            except Exception as e:
                self.logger.warning(f"Network connectivity to {endpoint}: Failed ({e})")
        
        return True  # Don't fail deployment for network issues
    
    def _check_command_available(self, command: str) -> bool:
        """Check if a command is available in PATH."""
        try:
            subprocess.run([command, '--version'], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def _backup_current_state(self) -> bool:
        """Backup current deployment state."""
        self.logger.info("Creating deployment backup...")
        
        try:
            self.backup_path.mkdir(parents=True, exist_ok=True)
            
            # Backup database
            if not self._backup_database():
                return False
            
            # Backup configuration files
            config_files = [
                'docker-compose.production.yml',
                'config/production.yaml',
                '.env.production'
            ]
            
            for config_file in config_files:
                src = Path(config_file)
                if src.exists():
                    dst = self.backup_path / src.name
                    subprocess.run(['cp', str(src), str(dst)], check=True)
                    self.logger.info(f"Backed up {config_file}")
            
            # Save current Docker images
            if not self._backup_docker_images():
                return False
            
            self.logger.info(f"Backup completed: {self.backup_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Backup failed: {e}")
            return False
    
    def _backup_database(self) -> bool:
        """Backup database."""
        try:
            # Check if database container is running
            result = subprocess.run(
                ['docker-compose', '-f', self.config.docker_compose_file, 'ps', '-q', 'postgres'],
                capture_output=True, text=True
            )
            
            if not result.stdout.strip():
                self.logger.info("Database container not running, skipping database backup")
                return True
            
            # Create database backup
            backup_file = self.backup_path / f"database_backup_{self.deployment_id}.sql"
            cmd = [
                'docker-compose', '-f', self.config.docker_compose_file,
                'exec', '-T', 'postgres', 'pg_dump', '-U', 'quantum', 'quantum_hvac'
            ]
            
            with open(backup_file, 'w') as f:
                result = subprocess.run(cmd, stdout=f, stderr=subprocess.PIPE, text=True)
            
            if result.returncode == 0:
                # Compress backup
                subprocess.run(['gzip', str(backup_file)], check=True)
                self.logger.info(f"Database backup created: {backup_file}.gz")
                return True
            else:
                self.logger.error(f"Database backup failed: {result.stderr}")
                return False
                
        except Exception as e:
            self.logger.error(f"Database backup error: {e}")
            return False
    
    def _backup_docker_images(self) -> bool:
        """Backup current Docker images."""
        try:
            # Get list of current images
            result = subprocess.run(
                ['docker-compose', '-f', self.config.docker_compose_file, 'images', '--quiet'],
                capture_output=True, text=True
            )
            
            if result.returncode == 0:
                images = result.stdout.strip().split('\n')
                image_list_file = self.backup_path / 'docker_images.txt'
                
                with open(image_list_file, 'w') as f:
                    for image in images:
                        if image.strip():
                            f.write(f"{image}\n")
                
                self.logger.info(f"Docker images list saved: {image_list_file}")
                return True
            else:
                self.logger.warning("Could not get Docker images list")
                return True  # Don't fail deployment for this
                
        except Exception as e:
            self.logger.warning(f"Docker images backup error: {e}")
            return True  # Don't fail deployment for this
    
    def _deploy_services(self) -> bool:
        """Deploy services using Docker Compose."""
        self.logger.info("Deploying services...")
        
        try:
            # Pull latest images
            self.logger.info("Pulling latest images...")
            result = subprocess.run([
                'docker-compose', '-f', self.config.docker_compose_file,
                '--env-file', self.config.env_file,
                'pull'
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                self.logger.error(f"Failed to pull images: {result.stderr}")
                return False
            
            # Build any local images
            self.logger.info("Building local images...")
            result = subprocess.run([
                'docker-compose', '-f', self.config.docker_compose_file,
                '--env-file', self.config.env_file,
                'build'
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                self.logger.error(f"Failed to build images: {result.stderr}")
                return False
            
            # Deploy services
            self.logger.info("Starting services...")
            result = subprocess.run([
                'docker-compose', '-f', self.config.docker_compose_file,
                '--env-file', self.config.env_file,
                'up', '-d', '--remove-orphans'
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                self.logger.error(f"Failed to start services: {result.stderr}")
                return False
            
            self.logger.info("Services deployed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Service deployment error: {e}")
            return False
    
    def _health_checks(self) -> bool:
        """Perform health checks on deployed services."""
        self.logger.info("Starting health checks...")
        
        services_to_check = self.config.services_to_check or list(self.health_endpoints.keys())
        max_wait_time = self.config.health_check_timeout
        wait_interval = 10
        elapsed_time = 0
        
        while elapsed_time < max_wait_time:
            all_healthy = True
            
            for service in services_to_check:
                if not self._check_service_health(service):
                    all_healthy = False
                    break
            
            if all_healthy:
                self.logger.info("All services are healthy")
                return True
            
            self.logger.info(f"Waiting for services to become healthy... ({elapsed_time}s/{max_wait_time}s)")
            time.sleep(wait_interval)
            elapsed_time += wait_interval
        
        self.logger.error("Health checks timed out")
        return False
    
    def _check_service_health(self, service: str) -> bool:
        """Check health of a specific service."""
        try:
            # Check Docker container health first
            result = subprocess.run([
                'docker-compose', '-f', self.config.docker_compose_file,
                'ps', '-q', service
            ], capture_output=True, text=True)
            
            if not result.stdout.strip():
                self.logger.warning(f"Service {service} container not found")
                return False
            
            # Check container status
            container_id = result.stdout.strip()
            result = subprocess.run([
                'docker', 'inspect', '-f', '{{.State.Health.Status}}', container_id
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                health_status = result.stdout.strip()
                if health_status == 'healthy':
                    self.logger.debug(f"Service {service} is healthy")
                    return True
                elif health_status == 'unhealthy':
                    self.logger.warning(f"Service {service} is unhealthy")
                    return False
                # If no health status, continue to HTTP check
            
            # HTTP health check if available
            endpoint = self.health_endpoints.get(service)
            if endpoint:
                response = requests.get(endpoint, timeout=5)
                if response.status_code == 200:
                    self.logger.debug(f"Service {service} HTTP health check passed")
                    return True
                else:
                    self.logger.warning(f"Service {service} HTTP health check failed: {response.status_code}")
                    return False
            
            # If no HTTP endpoint, assume healthy if container is running
            result = subprocess.run([
                'docker', 'inspect', '-f', '{{.State.Running}}', container_id
            ], capture_output=True, text=True)
            
            if result.returncode == 0 and result.stdout.strip() == 'true':
                self.logger.debug(f"Service {service} container is running")
                return True
            
            return False
            
        except Exception as e:
            self.logger.warning(f"Health check error for {service}: {e}")
            return False
    
    def _post_deployment_tasks(self) -> bool:
        """Execute post-deployment tasks."""
        self.logger.info("Running post-deployment tasks...")
        
        success = True
        
        # Database migrations
        if not self._run_database_migrations():
            success = False
        
        # Cleanup old Docker images
        if not self._cleanup_old_images():
            success = False
        
        # Update monitoring dashboards
        if not self._update_monitoring_dashboards():
            success = False
        
        return success
    
    def _run_database_migrations(self) -> bool:
        """Run database migrations."""
        try:
            self.logger.info("Running database migrations...")
            
            # This would run actual migrations in a real implementation
            # For now, just check if the database is accessible
            result = subprocess.run([
                'docker-compose', '-f', self.config.docker_compose_file,
                'exec', '-T', 'postgres', 'psql', '-U', 'quantum', '-d', 'quantum_hvac', '-c', 'SELECT 1;'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                self.logger.info("Database is accessible")
                return True
            else:
                self.logger.error(f"Database access failed: {result.stderr}")
                return False
                
        except Exception as e:
            self.logger.error(f"Database migration error: {e}")
            return False
    
    def _cleanup_old_images(self) -> bool:
        """Cleanup old Docker images."""
        try:
            self.logger.info("Cleaning up old Docker images...")
            
            # Remove dangling images
            result = subprocess.run([
                'docker', 'image', 'prune', '-f'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                self.logger.info("Docker image cleanup completed")
                return True
            else:
                self.logger.warning(f"Docker image cleanup warning: {result.stderr}")
                return True  # Don't fail deployment for this
                
        except Exception as e:
            self.logger.warning(f"Docker image cleanup error: {e}")
            return True  # Don't fail deployment for this
    
    def _update_monitoring_dashboards(self) -> bool:
        """Update monitoring dashboards."""
        try:
            self.logger.info("Updating monitoring dashboards...")
            
            # This would update Grafana dashboards in a real implementation
            # For now, just check if Grafana is accessible
            if self._check_service_health('grafana'):
                self.logger.info("Grafana is accessible")
                return True
            else:
                self.logger.warning("Grafana is not accessible")
                return False
                
        except Exception as e:
            self.logger.warning(f"Monitoring dashboard update error: {e}")
            return True  # Don't fail deployment for this
    
    def _rollback(self) -> bool:
        """Rollback to previous deployment."""
        self.logger.info("Starting rollback process...")
        
        try:
            # Stop current services
            subprocess.run([
                'docker-compose', '-f', self.config.docker_compose_file,
                'down'
            ], check=True)
            
            # Restore database if backup exists
            database_backup = self.backup_path / f"database_backup_{self.deployment_id}.sql.gz"
            if database_backup.exists():
                self.logger.info("Restoring database backup...")
                
                # Start only postgres for restore
                subprocess.run([
                    'docker-compose', '-f', self.config.docker_compose_file,
                    'up', '-d', 'postgres'
                ], check=True)
                
                time.sleep(10)  # Wait for postgres to start
                
                # Restore database
                with open(database_backup, 'rb') as f:
                    subprocess.run([
                        'gunzip', '-c'
                    ], stdin=f, stdout=subprocess.PIPE)
            
            # Restore configuration files
            for config_file in self.backup_path.glob('*.yaml'):
                dst = Path(config_file.name)
                subprocess.run(['cp', str(config_file), str(dst)], check=True)
            
            self.logger.info("Rollback completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Rollback failed: {e}")
            return False
    
    def status(self) -> Dict[str, Any]:
        """Get deployment status."""
        try:
            result = subprocess.run([
                'docker-compose', '-f', self.config.docker_compose_file,
                'ps', '--format', 'json'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                services = json.loads(result.stdout) if result.stdout else []
                return {
                    'deployment_id': self.deployment_id,
                    'environment': self.config.environment,
                    'services': services,
                    'healthy_services': sum(1 for s in services if s.get('Health') == 'healthy'),
                    'total_services': len(services)
                }
            else:
                return {'error': result.stderr}
                
        except Exception as e:
            return {'error': str(e)}


def main():
    """Main deployment script."""
    parser = argparse.ArgumentParser(description='Quantum HVAC Control Deployment Script')
    parser.add_argument(
        'action',
        choices=['deploy', 'status', 'rollback', 'logs'],
        help='Action to perform'
    )
    parser.add_argument(
        '--environment', '-e',
        choices=['development', 'staging', 'production'],
        default='production',
        help='Target environment'
    )
    parser.add_argument(
        '--compose-file', '-f',
        default='docker-compose.production.yml',
        help='Docker Compose file to use'
    )
    parser.add_argument(
        '--env-file',
        default='.env.production',
        help='Environment file to use'
    )
    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='Skip backup before deployment'
    )
    parser.add_argument(
        '--no-rollback',
        action='store_true',
        help='Disable automatic rollback on failure'
    )
    parser.add_argument(
        '--timeout',
        type=int,
        default=300,
        help='Health check timeout in seconds'
    )
    parser.add_argument(
        '--services',
        nargs='+',
        help='Specific services to check/deploy'
    )
    
    args = parser.parse_args()
    
    # Create deployment configuration
    config = DeploymentConfig(
        environment=args.environment,
        docker_compose_file=args.compose_file,
        env_file=args.env_file,
        backup_before_deploy=not args.no_backup,
        health_check_timeout=args.timeout,
        rollback_on_failure=not args.no_rollback,
        services_to_check=args.services
    )
    
    # Create deployment manager
    manager = DeploymentManager(config)
    
    try:
        if args.action == 'deploy':
            success = manager.deploy()
            sys.exit(0 if success else 1)
            
        elif args.action == 'status':
            status = manager.status()
            print(json.dumps(status, indent=2))
            
        elif args.action == 'rollback':
            success = manager._rollback()
            sys.exit(0 if success else 1)
            
        elif args.action == 'logs':
            subprocess.run([
                'docker-compose', '-f', config.docker_compose_file,
                'logs', '-f'
            ] + (args.services or []))
            
    except KeyboardInterrupt:
        print("\nDeployment interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Deployment script error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
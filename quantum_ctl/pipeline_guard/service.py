#!/usr/bin/env python3
"""
Production service runner for the self-healing pipeline guard.
"""

import asyncio
import argparse
import sys
import signal
import logging
import yaml
import os
from typing import Dict, Any, Optional

from .guard import PipelineGuard
from .metrics_collector import MetricsCollector
from .security_monitor import SecurityMonitor
from .performance_optimizer import PerformanceOptimizer
from .ai_predictor import AIPredictor
from .distributed_guard import DistributedPipelineGuard


class PipelineGuardService:
    """
    Production service for running the self-healing pipeline guard.
    """
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config: Dict[str, Any] = {}
        self.running = False
        
        # Core components
        self.pipeline_guard: Optional[PipelineGuard] = None
        self.metrics_collector: Optional[MetricsCollector] = None
        self.security_monitor: Optional[SecurityMonitor] = None
        self.performance_optimizer: Optional[PerformanceOptimizer] = None
        self.ai_predictor: Optional[AIPredictor] = None
        self.distributed_guard: Optional[DistributedPipelineGuard] = None
        
        # Setup logging
        self._setup_logging()
        self.logger = logging.getLogger(__name__)
        
    def _setup_logging(self):
        """Setup logging configuration."""
        # Default logging configuration
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler('/var/log/pipeline-guard/service.log', mode='a')
            ]
        )
        
    def load_config(self):
        """Load configuration from file."""
        try:
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
                
            self.logger.info(f"Configuration loaded from {self.config_path}")
            
            # Update logging configuration if specified
            if 'logging' in self.config:
                log_config = self.config['logging']
                
                # Set log level
                if 'level' in log_config:
                    level = getattr(logging, log_config['level'].upper())
                    logging.getLogger().setLevel(level)
                    
                # Add file handler if specified
                if 'file' in log_config:
                    file_handler = logging.FileHandler(log_config['file'])
                    formatter = logging.Formatter(
                        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                    )
                    file_handler.setFormatter(formatter)
                    logging.getLogger().addHandler(file_handler)
                    
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            sys.exit(1)
            
    def _create_components(self):
        """Create and configure pipeline guard components."""
        guard_config = self.config.get('guard', {})
        
        # Create main pipeline guard
        self.pipeline_guard = PipelineGuard(
            check_interval=guard_config.get('check_interval', 30.0)
        )
        
        # Create metrics collector
        metrics_config = self.config.get('metrics', {})
        self.metrics_collector = MetricsCollector(
            retention_hours=metrics_config.get('retention_hours', 72)
        )
        
        # Setup alert rules
        for rule in metrics_config.get('alert_rules', []):
            metric_name = rule['metric']
            threshold = rule['threshold']
            severity = rule.get('severity', 'warning')
            
            condition = lambda value, t=threshold: value > t
            self.metrics_collector.add_alert_rule(metric_name, condition, severity)
            
        # Create security monitor
        security_config = self.config.get('security', {})
        self.security_monitor = SecurityMonitor()
        
        if 'allowed_ips' in security_config:
            self.security_monitor.configure_allowed_ips(security_config['allowed_ips'])
            
        # Create performance optimizer
        performance_config = self.config.get('performance', {})
        self.performance_optimizer = PerformanceOptimizer()
        
        # Create AI predictor if enabled
        ai_config = self.config.get('ai_prediction', {})
        if ai_config.get('enabled', False):
            model_update_interval = ai_config.get('model_update_interval', 3600)
            self.ai_predictor = AIPredictor(model_update_interval=model_update_interval)
            
        # Create distributed guard if clustering enabled
        cluster_config = self.config.get('cluster', {})
        if cluster_config.get('enabled', False):
            self.distributed_guard = DistributedPipelineGuard(
                node_id=cluster_config['node_id'],
                bind_address=cluster_config.get('bind_address', '0.0.0.0'),
                bind_port=cluster_config.get('bind_port', 8765),
                redis_url=cluster_config.get('redis_url', 'redis://localhost:6379')
            )
            
        self.logger.info("Pipeline guard components created")
        
    def _register_components(self):
        """Register components for monitoring."""
        components_config = self.config.get('components', {})
        
        for component_name, component_config in components_config.items():
            # Create health check function
            health_check = self._create_health_check(component_name, component_config)
            
            # Create recovery action if needed
            recovery_action = self._create_recovery_action(component_name, component_config)
            
            # Register with pipeline guard
            self.pipeline_guard.register_component(
                name=component_name,
                health_check=health_check,
                recovery_action=recovery_action,
                critical=component_config.get('critical', False),
                circuit_breaker_config={
                    'failure_threshold': component_config.get('failure_threshold', 5),
                    'recovery_timeout': component_config.get('recovery_timeout', 60.0)
                }
            )
            
            # Register scaling policy if auto-scaling enabled
            performance_config = self.config.get('performance', {})
            if performance_config.get('auto_scaling', False):
                self.performance_optimizer.register_scaling_policy(
                    component=component_name,
                    scale_up_threshold=performance_config.get('scale_up_threshold', 80.0),
                    scale_down_threshold=performance_config.get('scale_down_threshold', 30.0),
                    max_instances=performance_config.get('max_instances', 10),
                    min_instances=performance_config.get('min_instances', 1)
                )
                
        self.logger.info(f"Registered {len(components_config)} components for monitoring")
        
    def _create_health_check(self, component_name: str, config: Dict[str, Any]):
        """Create health check function for a component."""
        def health_check():
            try:
                # This is a placeholder - in real implementation, this would
                # integrate with actual component health checking logic
                
                if component_name == "quantum_solver":
                    # Check D-Wave connection
                    try:
                        # Placeholder for actual quantum solver health check
                        return True
                    except Exception:
                        return False
                        
                elif component_name == "hvac_controller":
                    # Check HVAC controller health
                    try:
                        # Placeholder for actual HVAC controller health check
                        return True
                    except Exception:
                        return False
                        
                elif component_name == "bms_connector":
                    # Check BMS connection
                    try:
                        # Placeholder for actual BMS health check
                        return True
                    except Exception:
                        return False
                        
                else:
                    # Generic health check
                    return True
                    
            except Exception as e:
                self.logger.error(f"Health check failed for {component_name}: {e}")
                return False
                
        return health_check
        
    def _create_recovery_action(self, component_name: str, config: Dict[str, Any]):
        """Create recovery action for a component."""
        async def recovery_action():
            try:
                self.logger.info(f"Attempting recovery for {component_name}")
                
                # Component-specific recovery logic
                if component_name == "quantum_solver":
                    # Reset quantum solver connection
                    await asyncio.sleep(2)  # Simulate recovery time
                    return True
                    
                elif component_name == "hvac_controller":
                    # Restart HVAC controller
                    await asyncio.sleep(1)  # Simulate recovery time
                    return True
                    
                elif component_name == "bms_connector":
                    # Reconnect to BMS
                    await asyncio.sleep(3)  # Simulate recovery time
                    return True
                    
                else:
                    # Generic recovery
                    await asyncio.sleep(1)
                    return True
                    
            except Exception as e:
                self.logger.error(f"Recovery failed for {component_name}: {e}")
                return False
                
        return recovery_action
        
    async def start(self):
        """Start the pipeline guard service."""
        self.logger.info("Starting Pipeline Guard Service")
        
        try:
            # Load configuration
            self.load_config()
            
            # Create components
            self._create_components()
            
            # Register components
            self._register_components()
            
            # Start components
            if self.metrics_collector:
                self.metrics_collector.start()
                
            if self.ai_predictor:
                await self.ai_predictor.start_prediction_engine()
                
            if self.distributed_guard:
                await self.distributed_guard.start()
                
            # Start main pipeline guard
            self.pipeline_guard.start()
            
            self.running = True
            self.logger.info("Pipeline Guard Service started successfully")
            
            # Main service loop
            await self._run_service_loop()
            
        except Exception as e:
            self.logger.error(f"Failed to start service: {e}")
            await self.stop()
            raise
            
    async def stop(self):
        """Stop the pipeline guard service."""
        self.logger.info("Stopping Pipeline Guard Service")
        
        self.running = False
        
        try:
            # Stop components
            if self.pipeline_guard:
                await self.pipeline_guard.stop()
                
            if self.metrics_collector:
                await self.metrics_collector.stop()
                
            if self.distributed_guard:
                await self.distributed_guard.stop()
                
            self.logger.info("Pipeline Guard Service stopped")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
            
    async def _run_service_loop(self):
        """Main service loop."""
        while self.running:
            try:
                # Collect and process metrics
                if self.metrics_collector and self.performance_optimizer:
                    # Get current metrics
                    current_metrics = self.performance_optimizer._get_current_metrics()
                    
                    # Record system metrics
                    for metric_name, value in current_metrics.items():
                        self.metrics_collector.record_metric(
                            metric_name, value, "system"
                        )
                        
                    # Run performance optimization
                    optimization_results = await self.performance_optimizer.optimize_performance()
                    
                    if optimization_results['optimizations_applied']:
                        self.logger.info(
                            f"Applied {len(optimization_results['optimizations_applied'])} optimizations"
                        )
                        
                # Check for AI predictions
                if self.ai_predictor:
                    alerts = self.ai_predictor.get_active_alerts()
                    for alert in alerts:
                        self.logger.warning(
                            f"AI Alert: {alert['type']} - {alert['message']}"
                        )
                        
                # Log service status
                if self.pipeline_guard:
                    status = self.pipeline_guard.get_status()
                    self.logger.debug(
                        f"Pipeline status: {status['pipeline_status']}, "
                        f"Healthy: {status['healthy_components']}/{status['total_components']}"
                    )
                    
                await asyncio.sleep(60)  # Service loop interval
                
            except Exception as e:
                self.logger.error(f"Error in service loop: {e}")
                await asyncio.sleep(5)
                
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, initiating shutdown")
            asyncio.create_task(self.stop())
            
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)


async def main():
    """Main entry point for the service."""
    parser = argparse.ArgumentParser(description='Self-Healing Pipeline Guard Service')
    parser.add_argument(
        '--config',
        default='/etc/pipeline-guard/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--validate-config',
        action='store_true',
        help='Validate configuration and exit'
    )
    
    args = parser.parse_args()
    
    # Create service
    service = PipelineGuardService(args.config)
    
    if args.validate_config:
        try:
            service.load_config()
            print("✓ Configuration is valid")
            sys.exit(0)
        except Exception as e:
            print(f"✗ Configuration validation failed: {e}")
            sys.exit(1)
            
    # Setup signal handlers
    service._setup_signal_handlers()
    
    try:
        # Start service
        await service.start()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logging.error(f"Service failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
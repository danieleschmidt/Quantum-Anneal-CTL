# Self-Healing Pipeline Guard - Production Deployment Guide

## Overview

This guide provides comprehensive instructions for deploying the self-healing pipeline guard in production environments for quantum HVAC control systems.

## Architecture

The self-healing pipeline guard consists of several key components:

- **PipelineGuard**: Core orchestrator for monitoring and recovery
- **HealthMonitor**: Component health checking and trend analysis
- **RecoveryManager**: Intelligent recovery strategy execution
- **CircuitBreaker**: Failure protection and cascading failure prevention
- **MetricsCollector**: Real-time metrics collection and alerting
- **SecurityMonitor**: Security threat detection and access control
- **PerformanceOptimizer**: Adaptive performance optimization and scaling
- **AIPredictor**: Machine learning-based failure prediction
- **DistributedGuard**: Multi-node coordination and high availability

## Prerequisites

### System Requirements

- **Operating System**: Linux (Ubuntu 20.04+ recommended)
- **Python**: 3.9 or higher
- **Memory**: 4GB minimum, 8GB recommended
- **CPU**: 2 cores minimum, 4 cores recommended
- **Storage**: 20GB minimum for logs and data

### Dependencies

```bash
# Core dependencies
sudo apt update
sudo apt install -y python3-numpy python3-scipy python3-pandas python3-sklearn
sudo apt install -y python3-psutil python3-aiohttp
sudo apt install -y redis-server

# Optional dependencies for enhanced features
sudo apt install -y python3-prometheus-client  # For Prometheus integration
sudo apt install -y python3-influxdb          # For InfluxDB integration
```

### Network Requirements

- **Port 8765**: Pipeline guard cluster communication
- **Port 6379**: Redis for cluster coordination
- **Port 8080**: Web dashboard (optional)
- **Port 9090**: Prometheus metrics (optional)

## Installation

### 1. Clone and Setup

```bash
git clone https://github.com/danieleschmidt/self-healing-pipeline-guard
cd self-healing-pipeline-guard

# Verify installation
python3 -c "from quantum_ctl.pipeline_guard.guard import PipelineGuard; print('✓ Installation verified')"
```

### 2. Configuration

Create configuration file:

```bash
mkdir -p /etc/pipeline-guard
cat > /etc/pipeline-guard/config.yaml << EOF
# Pipeline Guard Configuration
guard:
  check_interval: 30.0
  max_retries: 3
  retry_delay: 5.0

# Component monitoring
components:
  quantum_solver:
    critical: true
    health_check_timeout: 10.0
    recovery_timeout: 60.0
  
  hvac_controller:
    critical: true
    health_check_timeout: 5.0
    recovery_timeout: 30.0
  
  bms_connector:
    critical: false
    health_check_timeout: 15.0
    recovery_timeout: 45.0

# Security settings
security:
  allowed_ips:
    - "192.168.1.0/24"
    - "10.0.0.0/8"
  max_auth_failures: 5
  auth_failure_window: 300

# Performance optimization
performance:
  auto_scaling: true
  max_instances: 10
  min_instances: 1
  scale_up_threshold: 80.0
  scale_down_threshold: 30.0

# AI prediction
ai_prediction:
  enabled: true
  model_update_interval: 3600
  prediction_threshold: 0.5

# Metrics and monitoring
metrics:
  retention_hours: 72
  alert_rules:
    - metric: "quantum_chain_break_fraction"
      threshold: 0.15
      severity: "warning"
    - metric: "hvac_optimization_duration"
      threshold: 25.0
      severity: "critical"

# Logging
logging:
  level: "INFO"
  file: "/var/log/pipeline-guard/guard.log"
  max_size: "100MB"
  backup_count: 5

# Cluster (for multi-node deployments)
cluster:
  enabled: false
  node_id: "node-1"
  bind_address: "0.0.0.0"
  bind_port: 8765
  redis_url: "redis://localhost:6379"
EOF
```

### 3. Systemd Service

Create systemd service file:

```bash
sudo cat > /etc/systemd/system/pipeline-guard.service << EOF
[Unit]
Description=Self-Healing Pipeline Guard
After=network.target redis.service
Requires=redis.service

[Service]
Type=simple
User=pipeline-guard
Group=pipeline-guard
WorkingDirectory=/opt/pipeline-guard
Environment=PYTHONPATH=/opt/pipeline-guard
ExecStart=/usr/bin/python3 -m quantum_ctl.pipeline_guard.service --config /etc/pipeline-guard/config.yaml
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

# Security settings
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ReadWritePaths=/var/log/pipeline-guard /var/lib/pipeline-guard

[Install]
WantedBy=multi-user.target
EOF
```

### 4. User and Directories

```bash
# Create user
sudo useradd -r -s /bin/false pipeline-guard

# Create directories
sudo mkdir -p /opt/pipeline-guard
sudo mkdir -p /var/log/pipeline-guard
sudo mkdir -p /var/lib/pipeline-guard

# Copy application
sudo cp -r . /opt/pipeline-guard/
sudo chown -R pipeline-guard:pipeline-guard /opt/pipeline-guard
sudo chown -R pipeline-guard:pipeline-guard /var/log/pipeline-guard
sudo chown -R pipeline-guard:pipeline-guard /var/lib/pipeline-guard

# Set permissions
sudo chmod 755 /opt/pipeline-guard
sudo chmod 640 /etc/pipeline-guard/config.yaml
```

## Integration with Quantum HVAC System

### 1. HVAC Controller Integration

```python
# Example integration in your HVAC controller
from quantum_ctl.pipeline_guard.quantum_integration import QuantumHVACPipelineGuard

class HVACController:
    def __init__(self):
        self.pipeline_guard = QuantumHVACPipelineGuard(
            hvac_controller=self,
            quantum_solver=self.quantum_solver,
            bms_connector=self.bms_connector
        )
    
    async def start(self):
        await self.pipeline_guard.start_monitoring()
        # Start your normal HVAC operations
    
    async def stop(self):
        await self.pipeline_guard.stop_monitoring()
```

### 2. Custom Health Checks

```python
# Register custom health checks
guard.register_component(
    name="custom_component",
    health_check=your_health_check_function,
    recovery_action=your_recovery_function,
    critical=True
)
```

### 3. Metrics Integration

```python
# Record custom metrics
from quantum_ctl.pipeline_guard.metrics_collector import MetricsCollector

collector = MetricsCollector()
collector.record_metric("custom_metric", value, component="your_component")
```

## Multi-Node Deployment

For high availability, deploy across multiple nodes:

### 1. Node Configuration

Each node needs unique configuration:

```yaml
# Node 1
cluster:
  enabled: true
  node_id: "hvac-node-1"
  bind_address: "10.0.1.10"
  bind_port: 8765
  redis_url: "redis://10.0.1.100:6379"

# Node 2  
cluster:
  enabled: true
  node_id: "hvac-node-2"
  bind_address: "10.0.1.11"
  bind_port: 8765
  redis_url: "redis://10.0.1.100:6379"
```

### 2. Redis Cluster Setup

```bash
# Install Redis on cluster coordinator node
sudo apt install redis-server

# Configure Redis for cluster
sudo cat >> /etc/redis/redis.conf << EOF
bind 0.0.0.0
protected-mode no
cluster-enabled yes
cluster-config-file nodes.conf
cluster-node-timeout 5000
EOF

sudo systemctl restart redis
```

### 3. Load Balancer Configuration

Use HAProxy or similar for load balancing:

```
# HAProxy configuration
backend pipeline_guard_cluster
    balance roundrobin
    server node1 10.0.1.10:8765 check
    server node2 10.0.1.11:8765 check
    server node3 10.0.1.12:8765 check
```

## Monitoring and Alerting

### 1. Prometheus Integration

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'pipeline-guard'
    static_configs:
      - targets: ['localhost:9090']
    metrics_path: '/metrics'
    scrape_interval: 30s
```

### 2. Grafana Dashboard

Import the provided Grafana dashboard:

```bash
# Import dashboard
curl -X POST http://grafana:3000/api/dashboards/db \
  -H "Content-Type: application/json" \
  -d @grafana-dashboard.json
```

### 3. Alerting Rules

```yaml
# alerting.yml
groups:
  - name: pipeline_guard_alerts
    rules:
      - alert: PipelineGuardDown
        expr: up{job="pipeline-guard"} == 0
        for: 30s
        labels:
          severity: critical
        annotations:
          summary: "Pipeline Guard is down"
          
      - alert: HighFailureRate
        expr: pipeline_guard_failure_rate > 0.1
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High failure rate detected"
```

## Security Configuration

### 1. TLS Configuration

```bash
# Generate certificates
openssl req -x509 -newkey rsa:4096 -keyout guard.key -out guard.crt -days 365 -nodes

# Update configuration
cat >> /etc/pipeline-guard/config.yaml << EOF
security:
  tls:
    enabled: true
    cert_file: "/etc/pipeline-guard/guard.crt"
    key_file: "/etc/pipeline-guard/guard.key"
EOF
```

### 2. Authentication

```yaml
# Add to config.yaml
security:
  authentication:
    enabled: true
    method: "jwt"
    secret_key: "your-secret-key"
    token_expiry: 3600
```

### 3. Firewall Rules

```bash
# UFW rules
sudo ufw allow from 192.168.1.0/24 to any port 8765
sudo ufw allow from 10.0.0.0/8 to any port 8765
sudo ufw deny 8765
```

## Performance Tuning

### 1. System Optimization

```bash
# Increase file descriptor limits
echo "pipeline-guard soft nofile 65536" >> /etc/security/limits.conf
echo "pipeline-guard hard nofile 65536" >> /etc/security/limits.conf

# Optimize kernel parameters
cat >> /etc/sysctl.conf << EOF
net.core.somaxconn = 1024
net.core.netdev_max_backlog = 5000
net.ipv4.tcp_max_syn_backlog = 1024
EOF

sysctl -p
```

### 2. Application Tuning

```yaml
# config.yaml
performance:
  worker_threads: 4
  max_connections: 1000
  connection_pool_size: 100
  cache_size_mb: 512
  gc_threshold: 10000
```

### 3. Database Optimization

```bash
# Redis optimization
cat >> /etc/redis/redis.conf << EOF
maxmemory 1gb
maxmemory-policy allkeys-lru
save 900 1
save 300 10
save 60 10000
EOF
```

## Troubleshooting

### Common Issues

1. **High Memory Usage**
   ```bash
   # Check memory usage
   ps aux | grep pipeline-guard
   
   # Reduce cache size in config
   cache_size_mb: 256
   ```

2. **Connection Errors**
   ```bash
   # Check network connectivity
   telnet <node-ip> 8765
   
   # Verify firewall rules
   sudo ufw status
   ```

3. **Redis Connection Issues**
   ```bash
   # Test Redis connection
   redis-cli -h <redis-host> ping
   
   # Check Redis logs
   sudo journalctl -u redis
   ```

### Logs and Debugging

```bash
# View pipeline guard logs
sudo journalctl -u pipeline-guard -f

# Enable debug logging
# In config.yaml:
logging:
  level: "DEBUG"

# Check system resources
htop
iostat -x 1
```

## Backup and Recovery

### 1. Configuration Backup

```bash
# Backup configuration
tar -czf pipeline-guard-config-$(date +%Y%m%d).tar.gz /etc/pipeline-guard/

# Backup data
tar -czf pipeline-guard-data-$(date +%Y%m%d).tar.gz /var/lib/pipeline-guard/
```

### 2. Disaster Recovery

```bash
# Restore from backup
sudo systemctl stop pipeline-guard
sudo tar -xzf pipeline-guard-config-backup.tar.gz -C /
sudo tar -xzf pipeline-guard-data-backup.tar.gz -C /
sudo systemctl start pipeline-guard
```

### 3. Database Backup

```bash
# Backup Redis data
redis-cli --rdb /backup/dump.rdb

# Restore Redis data
sudo systemctl stop redis
sudo cp /backup/dump.rdb /var/lib/redis/
sudo chown redis:redis /var/lib/redis/dump.rdb
sudo systemctl start redis
```

## Maintenance

### 1. Regular Maintenance Tasks

```bash
# Log rotation (automated via logrotate)
sudo cat > /etc/logrotate.d/pipeline-guard << EOF
/var/log/pipeline-guard/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 644 pipeline-guard pipeline-guard
    postrotate
        systemctl reload pipeline-guard
    endscript
}
EOF

# Clean old data
find /var/lib/pipeline-guard -name "*.old" -mtime +30 -delete
```

### 2. Updates

```bash
# Update pipeline guard
sudo systemctl stop pipeline-guard
sudo cp -r new-version/* /opt/pipeline-guard/
sudo chown -R pipeline-guard:pipeline-guard /opt/pipeline-guard
sudo systemctl start pipeline-guard
```

### 3. Health Checks

```bash
# System health check script
#!/bin/bash
systemctl is-active pipeline-guard || echo "Pipeline Guard not running"
redis-cli ping > /dev/null || echo "Redis not responding"
curl -f http://localhost:8080/health || echo "Health endpoint not responding"
```

## Performance Monitoring

### Key Metrics to Monitor

1. **System Metrics**
   - CPU usage
   - Memory usage
   - Disk I/O
   - Network I/O

2. **Application Metrics**
   - Component health status
   - Recovery success rate
   - Response times
   - Error rates

3. **Business Metrics**
   - HVAC optimization time
   - Energy savings
   - Comfort violations
   - System uptime

### Alerting Thresholds

```yaml
alerts:
  critical:
    - component_failure_rate > 0.1
    - system_memory_usage > 90%
    - response_time > 5000ms
  
  warning:
    - component_failure_rate > 0.05
    - system_memory_usage > 80%
    - response_time > 2000ms
```

## Support and Troubleshooting

For support and additional documentation:

- **Documentation**: `/docs/` directory
- **Examples**: `/examples/` directory  
- **Issues**: GitHub Issues
- **Community**: Discussions forum

## Security Considerations

1. **Access Control**: Implement proper authentication and authorization
2. **Network Security**: Use TLS encryption and firewall rules
3. **Data Protection**: Encrypt sensitive data at rest and in transit
4. **Audit Logging**: Enable comprehensive audit logging
5. **Regular Updates**: Keep system and dependencies updated
6. **Monitoring**: Implement security monitoring and alerting

This deployment guide ensures a robust, secure, and scalable deployment of the self-healing pipeline guard for production quantum HVAC systems.
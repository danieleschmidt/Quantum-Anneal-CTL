# Quantum HVAC Control System - Production Deployment Guide

## 🚀 Quick Start

### Prerequisites
- Docker and Docker Compose installed
- At least 4GB RAM and 2 CPU cores available
- Network access for monitoring dashboards

### 1. Clone and Configure
```bash
git clone <repository-url>
cd photonic-mlir-synth-bridge
cp deploy/config/production.yaml.example deploy/config/production.yaml
# Edit configuration as needed
```

### 2. Deploy with Docker Compose
```bash
cd deploy
docker-compose up -d
```

### 3. Verify Deployment
```bash
# Check all services are running
docker-compose ps

# View logs
docker-compose logs quantum-hvac-controller

# Test API endpoints
curl http://localhost:8000/health
curl http://localhost:8002/microgrid/status
```

### 4. Access Monitoring
- **Grafana Dashboard**: http://localhost:3000 (admin/quantum_hvac_2024)
- **Prometheus Metrics**: http://localhost:9090
- **System Logs**: `docker-compose logs -f`

## 🏗️ Architecture Overview

### Core Components

1. **Quantum HVAC Controller** (Port 8000)
   - Main optimization engine
   - Individual building control
   - Safety monitoring and emergency control
   - Performance caching and metrics

2. **Microgrid Coordinator** (Port 8002)
   - Multi-building coordination
   - Solar generation optimization
   - Battery storage management
   - Peer-to-peer energy trading

3. **Redis Cache** (Port 6379)
   - Optimization result caching
   - Matrix operation caching
   - Performance data storage

4. **Monitoring Stack**
   - **Prometheus**: Metrics collection and alerting
   - **Grafana**: Visualization dashboards
   - **Nginx**: Load balancing and SSL termination

### Data Flow
```
BMS Data → Controller → Quantum Optimization → Control Commands → BMS
     ↓                       ↓                        ↓
Weather/Pricing → Microgrid Coordinator → Energy Management
     ↓                       ↓                        ↓
Monitoring ← Performance Metrics ← Safety Monitoring
```

## ⚙️ Configuration

### Environment Variables
```bash
# Core settings
LOG_LEVEL=INFO
SOLVER_TYPE=classical_fallback
PREDICTION_HORIZON=12
CONTROL_INTERVAL=15

# Performance tuning
OPTIMIZATION_CACHE_SIZE=128
MAX_PARALLEL_WORKERS=4
ENABLE_PERFORMANCE_MONITORING=true

# Microgrid settings
SOLAR_CAPACITY_KW=200
BATTERY_CAPACITY_KWH=500
ENABLE_PEER_TRADING=true

# Safety
ENABLE_SAFETY_MONITORING=true
```

### Production Configuration File
Edit `deploy/config/production.yaml` for detailed configuration:

```yaml
# Key sections to customize:
optimization:
  solver_type: "classical_fallback"  # Change to "qpu" if D-Wave available
  
safety:
  limits:
    min_zone_temp: 15.0
    max_zone_temp: 35.0
    
microgrid:
  solar_capacity_kw: 200.0    # Match your installation
  battery_capacity_kwh: 500.0 # Match your battery bank
  
integration:
  bms:
    enable: true              # Enable for real BMS connection
    protocol: "modbus"
    connection:
      host: "192.168.1.100"   # Your BMS IP address
```

## 🔧 Performance Tuning

### Cache Optimization
```yaml
optimization:
  cache:
    optimization_cache_size: 256  # Increase for better hit rates
    optimization_cache_ttl: 600   # Longer TTL for stable conditions
    matrix_cache_size: 128
```

### Parallel Processing
```yaml
performance:
  max_parallel_workers: 8      # Match your CPU cores
  resource_optimization_interval: 1800  # Auto-tune more frequently
```

### Memory Management
```bash
# In docker-compose.yml, set memory limits:
services:
  quantum-hvac-controller:
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '2.0'
```

## 🛡️ Security Considerations

### 1. Network Security
```yaml
# In production.yaml:
api:
  auth:
    enable: true
    secret_key: "strong-random-key-here"  # Generate secure key!
```

### 2. SSL/TLS Configuration
```bash
# Generate SSL certificates
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout deploy/nginx/ssl/quantum-hvac.key \
  -out deploy/nginx/ssl/quantum-hvac.crt
```

### 3. Database Security
```yaml
database:
  url: "postgresql://quantum_user:secure_password@postgres:5432/quantum_hvac"
```

## 📊 Monitoring and Alerting

### Key Metrics to Monitor
1. **Optimization Performance**
   - Average solve time
   - Cache hit rates
   - Success/failure rates

2. **Safety Metrics**
   - Temperature violations
   - Emergency control activations
   - Safety system response time

3. **Energy Metrics**
   - Total energy consumption
   - Cost savings achieved
   - Carbon emission reductions

4. **System Health**
   - CPU and memory usage
   - Network connectivity
   - Service availability

### Grafana Dashboards
Pre-configured dashboards include:
- **System Overview**: High-level metrics and status
- **Optimization Performance**: Solver metrics and timing
- **Energy Management**: Consumption and cost analysis
- **Safety Monitoring**: Temperature and violation tracking

### Alerting Rules
Configure Prometheus alerts for:
```yaml
# Example alert rules
- alert: QuantumOptimizationFailure
  expr: optimization_failure_rate > 0.1
  for: 5m
  
- alert: SafetyViolation
  expr: safety_violations_total > 0
  for: 1m
  
- alert: HighEnergyConsumption
  expr: energy_consumption_kwh > threshold
  for: 15m
```

## 🚦 Health Checks and Maintenance

### Automated Health Checks
The system includes built-in health checks:
```bash
# Manual health check
curl http://localhost:8000/health

# Expected response:
{
  "status": "healthy",
  "timestamp": "2024-08-04T19:00:00Z",
  "services": {
    "quantum_solver": "operational",
    "safety_monitor": "active",
    "cache": "connected"
  }
}
```

### Maintenance Tasks

#### Daily
- Monitor dashboard for anomalies
- Check log files for errors
- Verify all services are running

#### Weekly
- Review performance metrics
- Update weather and pricing data sources
- Check disk space and cleanup logs

#### Monthly
- Update system dependencies
- Review and optimize cache settings
- Backup configuration and historical data
- Performance benchmark testing

### Backup Strategy
```bash
# Backup configuration
tar -czf backup-$(date +%Y%m%d).tar.gz deploy/config/

# Backup database (if using SQLite)
cp /app/data/quantum_hvac.db backup/

# Backup historical data
docker exec quantum-hvac-controller \
  python -m quantum_ctl.utils.backup --output /backup/
```

## 🔄 Scaling and High Availability

### Horizontal Scaling
```yaml
# In docker-compose.yml:
services:
  quantum-hvac-controller:
    deploy:
      replicas: 3
      
  nginx:
    # Configure load balancing
    depends_on:
      - quantum-hvac-controller
```

### Database Clustering
For high availability, use PostgreSQL with replication:
```yaml
database:
  type: "postgresql"
  primary_url: "postgresql://user:pass@postgres-primary:5432/quantum_hvac"
  replica_urls:
    - "postgresql://user:pass@postgres-replica1:5432/quantum_hvac"
    - "postgresql://user:pass@postgres-replica2:5432/quantum_hvac"
```

## 🐛 Troubleshooting

### Common Issues

#### 1. Optimization Failures
```bash
# Check solver status
curl http://localhost:8000/status | jq '.quantum_solver_status'

# Review logs
docker-compose logs quantum-hvac-controller | grep ERROR
```

#### 2. High Memory Usage
```bash
# Monitor resource usage
docker stats quantum-hvac-controller

# Optimize cache settings
vim deploy/config/production.yaml
# Reduce cache sizes or TTL values
```

#### 3. Network Connectivity Issues
```bash
# Test service connectivity
docker-compose exec quantum-hvac-controller \
  python -c "import quantum_ctl; print('Import OK')"

# Check network configuration
docker network ls
docker network inspect quantum-network
```

#### 4. Performance Degradation
```bash
# Run performance benchmark
docker-compose exec quantum-hvac-controller \
  python examples/performance_benchmark.py

# Check cache hit rates
curl http://localhost:8000/metrics | grep cache_hit_rate
```

### Log Analysis
```bash
# Filter for specific error types
docker-compose logs quantum-hvac-controller 2>&1 | \
  grep -E "(ERROR|CRITICAL|EMERGENCY)"

# Monitor real-time logs
docker-compose logs -f --tail=100 quantum-hvac-controller
```

## 📞 Support

### Documentation Links
- [API Reference](./API.md)
- [Configuration Guide](./CONFIGURATION.md)
- [Developer Documentation](./DEVELOPMENT.md)

### Getting Help
1. Check the troubleshooting section above
2. Review system logs and metrics
3. Consult the GitHub issues for known problems
4. Contact the development team with detailed error information

### Performance Benchmarks
Expected performance for a typical deployment:
- **Optimization Time**: < 1 second per building
- **Cache Hit Rate**: > 30% in steady state
- **Memory Usage**: < 1GB per controller instance
- **CPU Usage**: < 50% average load
- **Availability**: > 99.9% uptime

---

## 🎯 Quick Start Checklist

- [ ] Docker and Docker Compose installed
- [ ] Configuration file customized
- [ ] SSL certificates generated (for production)
- [ ] Network firewall rules configured
- [ ] Monitoring dashboards accessible
- [ ] Health checks passing
- [ ] Backup strategy implemented
- [ ] Team trained on operations procedures

**🚀 Your Quantum HVAC Control System is ready for production!**
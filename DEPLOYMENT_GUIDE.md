# Quantum HVAC Control System - Deployment Guide

## Overview

This guide covers the deployment of the Quantum HVAC Control System from development to production environments, including Docker, Kubernetes, and cloud deployments.

## Prerequisites

### System Requirements

**Minimum Hardware:**
- CPU: 4 cores (8 recommended for production)
- RAM: 8GB (16GB+ recommended for production)
- Storage: 20GB available space
- Network: Reliable internet connection for D-Wave API access

**Software Requirements:**
- Python 3.9+ (3.12 recommended)
- Docker 24.0+
- Docker Compose 2.0+
- PostgreSQL 16+ (for production database)
- Redis 7+ (for caching)

### D-Wave Quantum Access

1. **Sign up for D-Wave Leap**: https://cloud.dwavesys.com/leap/
2. **Get API Token**: Navigate to API Tokens in your dashboard
3. **Verify Access**: Test connection with `dwave config create`

## Local Development Setup

### 1. Clone and Setup Environment

```bash
# Clone repository
git clone https://github.com/danieleschmidt/quantum-anneal-ctl.git
cd quantum-anneal-ctl

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### 2. Configure Environment Variables

Create `.env` file:
```bash
# Database
DATABASE_URL=postgresql://quantum:password@localhost:5432/quantum_hvac

# Redis Cache
REDIS_URL=redis://localhost:6379/0

# D-Wave Quantum Access
DWAVE_API_TOKEN=your_dwave_token_here

# Security
SECRET_KEY=your-secret-key-minimum-32-characters

# Logging
LOG_LEVEL=DEBUG
```

### 3. Start Development Services

```bash
# Start database and cache
docker-compose -f docker-compose.dev.yml up -d postgres redis

# Initialize database
python -m quantum_ctl.database.manager init

# Run development server
uvicorn quantum_ctl.api.app:app --reload --host 0.0.0.0 --port 8000
```

### 4. Verify Installation

```bash
# Health check
curl http://localhost:8000/health

# API documentation
open http://localhost:8000/docs

# Dashboard
open http://localhost:8080
```

## Docker Deployment

### 1. Single Container (Development)

```bash
# Build image
docker build -t quantum-hvac:dev -f Dockerfile.production --target development .

# Run container
docker run -d \
  --name quantum-hvac-dev \
  -p 8000:8000 \
  -e DATABASE_URL="postgresql://quantum:password@host.docker.internal:5432/quantum_hvac" \
  -e DWAVE_API_TOKEN="your_token" \
  quantum-hvac:dev
```

### 2. Docker Compose (Full Stack)

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your configuration
nano .env

# Start full stack
docker-compose up -d

# View logs
docker-compose logs -f

# Scale API instances
docker-compose up -d --scale quantum-hvac-api=3
```

### 3. Production Docker Compose

```bash
# Use production configuration
docker-compose -f docker-compose.production.yml up -d

# Monitor services
docker-compose -f docker-compose.production.yml ps
docker-compose -f docker-compose.production.yml top
```

## Kubernetes Deployment

### 1. Prerequisites

```bash
# Install kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl

# Verify cluster access
kubectl cluster-info
kubectl get nodes
```

### 2. Setup Namespace and Secrets

```bash
# Create namespace
kubectl create namespace quantum-hvac-prod

# Create secrets (update values first!)
kubectl create secret generic quantum-hvac-secrets \
  --namespace=quantum-hvac-prod \
  --from-literal=database-url="postgresql+asyncpg://quantum:YOUR_PASSWORD@postgres-cluster:5432/quantum_hvac" \
  --from-literal=redis-url="redis://:YOUR_PASSWORD@redis-cluster:6379/0" \
  --from-literal=dwave-token="YOUR_DWAVE_TOKEN" \
  --from-literal=secret-key="YOUR_SECRET_KEY_32_CHARS_MINIMUM"
```

### 3. Deploy Application

```bash
# Deploy database (if not using external)
kubectl apply -f deployment/kubernetes/postgres-deployment.yaml

# Deploy Redis cache
kubectl apply -f deployment/kubernetes/redis-deployment.yaml

# Deploy main application
kubectl apply -f deployment/kubernetes/production-deployment.yaml

# Check deployment status
kubectl get pods -n quantum-hvac-prod
kubectl describe deployment quantum-hvac-api -n quantum-hvac-prod
```

### 4. Configure Ingress

```bash
# Install NGINX Ingress Controller (if needed)
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.8.2/deploy/static/provider/cloud/deploy.yaml

# Install cert-manager for TLS (if needed)
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.2/cert-manager.yaml

# Apply ingress configuration
kubectl apply -f deployment/kubernetes/ingress.yaml
```

### 5. Verify Deployment

```bash
# Check all resources
kubectl get all -n quantum-hvac-prod

# Check logs
kubectl logs -f deployment/quantum-hvac-api -n quantum-hvac-prod

# Test API endpoint
kubectl port-forward service/quantum-hvac-api-service 8000:80 -n quantum-hvac-prod
curl http://localhost:8000/health
```

## Cloud Deployments

### AWS EKS

```bash
# Install eksctl
curl --silent --location "https://github.com/weaveworks/eksctl/releases/latest/download/eksctl_$(uname -s)_amd64.tar.gz" | tar xz -C /tmp
sudo mv /tmp/eksctl /usr/local/bin

# Create EKS cluster
eksctl create cluster \
  --name quantum-hvac-cluster \
  --version 1.28 \
  --region us-west-2 \
  --nodegroup-name standard-workers \
  --node-type m5.large \
  --nodes 3 \
  --nodes-min 2 \
  --nodes-max 6 \
  --managed

# Configure kubectl
aws eks update-kubeconfig --region us-west-2 --name quantum-hvac-cluster

# Deploy application
kubectl apply -f deployment/kubernetes/production-deployment.yaml
```

### Google GKE

```bash
# Create GKE cluster
gcloud container clusters create quantum-hvac-cluster \
  --num-nodes=3 \
  --zone=us-central1-a \
  --machine-type=n1-standard-2 \
  --enable-autoscaling \
  --min-nodes=2 \
  --max-nodes=10

# Get credentials
gcloud container clusters get-credentials quantum-hvac-cluster --zone=us-central1-a

# Deploy application
kubectl apply -f deployment/kubernetes/production-deployment.yaml
```

### Azure AKS

```bash
# Create resource group
az group create --name quantum-hvac-rg --location eastus

# Create AKS cluster
az aks create \
  --resource-group quantum-hvac-rg \
  --name quantum-hvac-cluster \
  --node-count 3 \
  --enable-addons monitoring \
  --generate-ssh-keys

# Get credentials
az aks get-credentials --resource-group quantum-hvac-rg --name quantum-hvac-cluster

# Deploy application
kubectl apply -f deployment/kubernetes/production-deployment.yaml
```

## Configuration Management

### Environment-Specific Configurations

**Development (`config/development.yaml`)**:
```yaml
database:
  echo: true
  pool_size: 5
  
logging:
  level: DEBUG
  
quantum:
  solver: classical  # Mock solver for development
  
features:
  debug_mode: true
```

**Staging (`config/staging.yaml`)**:
```yaml
database:
  pool_size: 10
  
logging:
  level: INFO
  
quantum:
  solver: hybrid
  
features:
  debug_mode: false
```

**Production (`config/production.yaml`)**:
```yaml
database:
  pool_size: 20
  max_overflow: 30
  
logging:
  level: INFO
  format: json
  
quantum:
  solver: hybrid
  timeout_seconds: 120
  
security:
  rate_limit:
    requests_per_minute: 1000
```

### Configuration Validation

```bash
# Validate configuration
python -m quantum_ctl.utils.config validate --env production

# Test database connection
python -m quantum_ctl.database.manager test-connection

# Verify D-Wave access
python -c "from quantum_ctl.optimization.quantum_solver import QuantumSolver; print('✓ D-Wave connection OK')"
```

## Monitoring and Observability

### 1. Health Checks

```bash
# API health
curl https://api.quantum-hvac.com/health

# Database health
curl https://api.quantum-hvac.com/health/database

# Quantum solver health
curl https://api.quantum-hvac.com/quantum/status
```

### 2. Monitoring Stack

**Prometheus + Grafana (Docker Compose)**:
```bash
# Start monitoring stack
docker-compose -f monitoring/docker-compose.monitoring.yml up -d

# Access Grafana
open http://localhost:3000
# Login: admin/admin (change on first login)
```

**Kubernetes Monitoring**:
```bash
# Install Prometheus Operator
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm install prometheus prometheus-community/kube-prometheus-stack

# Access Grafana
kubectl port-forward svc/prometheus-grafana 3000:80
```

### 3. Log Aggregation

**ELK Stack (Docker)**:
```bash
# Start ELK stack
docker-compose -f monitoring/docker-compose.elk.yml up -d

# Access Kibana
open http://localhost:5601
```

**Cloud Logging**:
- **AWS**: CloudWatch Logs
- **GCP**: Cloud Logging  
- **Azure**: Azure Monitor Logs

## Security Configuration

### 1. TLS/SSL Setup

**Let's Encrypt (Kubernetes)**:
```bash
# Install cert-manager
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.2/cert-manager.yaml

# Create ClusterIssuer
kubectl apply -f - <<EOF
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: admin@quantum-hvac.com
    privateKeySecretRef:
      name: letsencrypt-prod
    solvers:
    - http01:
        ingress:
          class: nginx
EOF
```

### 2. Network Security

**Firewall Rules**:
```bash
# Allow HTTPS traffic
sudo ufw allow 443/tcp

# Allow API traffic
sudo ufw allow 8000/tcp

# Allow SSH (be careful!)
sudo ufw allow 22/tcp

# Enable firewall
sudo ufw enable
```

### 3. Secrets Management

**Kubernetes Secrets**:
```bash
# Rotate secrets
kubectl create secret generic quantum-hvac-secrets-new \
  --from-literal=secret-key="$(openssl rand -base64 32)" \
  --namespace=quantum-hvac-prod

# Update deployment to use new secret
kubectl patch deployment quantum-hvac-api \
  -n quantum-hvac-prod \
  -p '{"spec":{"template":{"spec":{"containers":[{"name":"api","envFrom":[{"secretRef":{"name":"quantum-hvac-secrets-new"}}]}]}}}}'
```

## Backup and Recovery

### 1. Database Backups

**Automated Backup Script**:
```bash
#!/bin/bash
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="quantum_hvac_backup_$TIMESTAMP.sql"

# Create backup
pg_dump $DATABASE_URL > $BACKUP_FILE

# Compress and upload to S3
gzip $BACKUP_FILE
aws s3 cp ${BACKUP_FILE}.gz s3://quantum-hvac-backups/

# Clean up local file
rm ${BACKUP_FILE}.gz
```

**Schedule with cron**:
```bash
# Edit crontab
crontab -e

# Add backup job (daily at 2 AM)
0 2 * * * /usr/local/bin/backup-quantum-hvac.sh
```

### 2. Disaster Recovery

**Recovery Procedure**:
```bash
# 1. Download latest backup
aws s3 cp s3://quantum-hvac-backups/quantum_hvac_backup_YYYYMMDD_HHMMSS.sql.gz .

# 2. Restore database
gunzip quantum_hvac_backup_YYYYMMDD_HHMMSS.sql.gz
psql $DATABASE_URL < quantum_hvac_backup_YYYYMMDD_HHMMSS.sql

# 3. Restart services
kubectl rollout restart deployment/quantum-hvac-api -n quantum-hvac-prod
```

## Performance Tuning

### 1. Database Optimization

**PostgreSQL Configuration (`postgresql.conf`)**:
```ini
# Memory settings
shared_buffers = 256MB
effective_cache_size = 1GB
work_mem = 4MB

# Connection settings
max_connections = 100
max_prepared_transactions = 100

# Checkpoint settings
checkpoint_timeout = 10min
checkpoint_completion_target = 0.9
```

### 2. Application Tuning

**Gunicorn Configuration**:
```python
# gunicorn.conf.py
bind = "0.0.0.0:8000"
workers = 4
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 50
timeout = 120
keepalive = 2
```

### 3. Redis Optimization

```ini
# redis.conf
maxmemory 512mb
maxmemory-policy allkeys-lru
save 900 1
save 300 10
save 60 10000
```

## Troubleshooting

### Common Issues

**1. D-Wave Connection Errors**
```bash
# Test D-Wave connection
python -c "
import dwave.system
sampler = dwave.system.DWaveSampler()
print('✓ D-Wave connection successful')
print(f'QPU: {sampler.properties[\"chip_id\"]}')
"
```

**2. Database Connection Issues**
```bash
# Test database connection
python -c "
import asyncpg
import asyncio
async def test():
    conn = await asyncpg.connect('$DATABASE_URL')
    await conn.close()
    print('✓ Database connection successful')
asyncio.run(test())
"
```

**3. Memory Issues**
```bash
# Check memory usage
docker stats

# Check application memory
kubectl top pods -n quantum-hvac-prod
```

### Log Analysis

**Application Logs**:
```bash
# Docker logs
docker logs quantum-hvac-api --tail 100 -f

# Kubernetes logs  
kubectl logs -f deployment/quantum-hvac-api -n quantum-hvac-prod

# Filter error logs
kubectl logs deployment/quantum-hvac-api -n quantum-hvac-prod | grep ERROR
```

## Scaling

### Horizontal Scaling

**Kubernetes HPA**:
```yaml
# Already configured in production-deployment.yaml
# Scales based on CPU (70%) and memory (80%) usage
# Min replicas: 5, Max replicas: 20
```

**Manual Scaling**:
```bash
# Scale immediately
kubectl scale deployment quantum-hvac-api --replicas=10 -n quantum-hvac-prod

# Update HPA
kubectl patch hpa quantum-hvac-api-hpa -n quantum-hvac-prod -p '{"spec":{"maxReplicas":15}}'
```

### Vertical Scaling

**Resource Updates**:
```bash
# Update resource limits
kubectl patch deployment quantum-hvac-api -n quantum-hvac-prod -p '
{
  "spec": {
    "template": {
      "spec": {
        "containers": [{
          "name": "api",
          "resources": {
            "requests": {"memory": "2Gi", "cpu": "1000m"},
            "limits": {"memory": "4Gi", "cpu": "2000m"}
          }
        }]
      }
    }
  }
}'
```

## Maintenance

### 1. Regular Maintenance Tasks

```bash
# Weekly tasks
./scripts/weekly-maintenance.sh

# Update dependencies
pip-audit
safety check

# Database maintenance
psql $DATABASE_URL -c "VACUUM ANALYZE;"

# Clean old logs
find /var/log/quantum-hvac -name "*.log" -mtime +30 -delete
```

### 2. Rolling Updates

```bash
# Update to new version
kubectl set image deployment/quantum-hvac-api api=quantum-hvac:v1.1.0 -n quantum-hvac-prod

# Monitor rollout
kubectl rollout status deployment/quantum-hvac-api -n quantum-hvac-prod

# Rollback if needed
kubectl rollout undo deployment/quantum-hvac-api -n quantum-hvac-prod
```

## Support and Resources

- **Documentation**: https://quantum-hvac-docs.com
- **API Reference**: https://api.quantum-hvac.com/docs  
- **Issues**: https://github.com/danieleschmidt/quantum-anneal-ctl/issues
- **D-Wave Support**: https://support.dwavesys.com
- **Community Forum**: https://quantum-hvac.com/community

For deployment support, contact: deployment-support@quantum-hvac.com
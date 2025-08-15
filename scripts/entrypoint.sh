#!/bin/bash
set -e

# Pipeline Guard Docker Entrypoint Script

echo "Starting Pipeline Guard Service..."

# Check if Redis is available
echo "Checking Redis connectivity..."
until nc -z redis 6379; do
    echo "Waiting for Redis to be ready..."
    sleep 2
done
echo "Redis is ready"

# Validate configuration
echo "Validating configuration..."
python -m quantum_ctl.pipeline_guard.service --validate-config

# Check required directories
if [ ! -d "/var/log/pipeline-guard" ]; then
    mkdir -p /var/log/pipeline-guard
fi

if [ ! -d "/var/lib/pipeline-guard" ]; then
    mkdir -p /var/lib/pipeline-guard
fi

# Set proper permissions
chmod 755 /var/log/pipeline-guard /var/lib/pipeline-guard

# Start the service
echo "Starting Pipeline Guard Service..."
exec "$@"
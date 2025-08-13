#!/bin/bash

# Production health check script for Quantum HVAC Controller

set -e

TIMESTAMP=$(date -u +"%Y-%m-%d %H:%M:%S UTC")
HEALTH_LOG="/var/log/healthcheck.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log() {
    echo "[$TIMESTAMP] $1" | tee -a "$HEALTH_LOG"
}

# Check main API health
check_api_health() {
    log "Checking API health..."
    
    # Check main API endpoint
    if curl -f -s --max-time 10 "http://quantum-hvac-controller:8000/health" > /dev/null; then
        log "✅ API health check passed"
        return 0
    else
        log "❌ API health check failed"
        return 1
    fi
}

# Check metrics endpoint
check_metrics_health() {
    log "Checking metrics endpoint..."
    
    if curl -f -s --max-time 5 "http://quantum-hvac-controller:8001/metrics" > /dev/null; then
        log "✅ Metrics endpoint healthy"
        return 0
    else
        log "❌ Metrics endpoint unhealthy"
        return 1
    fi
}

# Check Redis connectivity
check_redis() {
    log "Checking Redis connectivity..."
    
    if redis-cli -h redis -p 6379 ping | grep -q "PONG"; then
        log "✅ Redis is responsive"
        return 0
    else
        log "❌ Redis is not responsive"
        return 1
    fi
}

# Check PostgreSQL connectivity
check_postgres() {
    log "Checking PostgreSQL connectivity..."
    
    if pg_isready -h postgres -p 5432 -U quantum_user > /dev/null 2>&1; then
        log "✅ PostgreSQL is ready"
        return 0
    else
        log "❌ PostgreSQL is not ready"
        return 1
    fi
}

# Check InfluxDB
check_influxdb() {
    log "Checking InfluxDB..."
    
    if curl -f -s --max-time 5 "http://influxdb:8086/health" > /dev/null; then
        log "✅ InfluxDB is healthy"
        return 0
    else
        log "❌ InfluxDB is not healthy"
        return 1
    fi
}

# Check system resources
check_resources() {
    log "Checking system resources..."
    
    # Check memory usage
    MEMORY_USAGE=$(free | grep Mem | awk '{printf "%.1f", $3/$2 * 100.0}')
    if (( $(echo "$MEMORY_USAGE > 90" | bc -l) )); then
        log "⚠️  High memory usage: ${MEMORY_USAGE}%"
    else
        log "✅ Memory usage normal: ${MEMORY_USAGE}%"
    fi
    
    # Check disk usage
    DISK_USAGE=$(df / | tail -1 | awk '{print $5}' | sed 's/%//')
    if [ "$DISK_USAGE" -gt 85 ]; then
        log "⚠️  High disk usage: ${DISK_USAGE}%"
    else
        log "✅ Disk usage normal: ${DISK_USAGE}%"
    fi
}

# Check application-specific metrics
check_app_metrics() {
    log "Checking application metrics..."
    
    # Get health status from the application
    HEALTH_RESPONSE=$(curl -s --max-time 10 "http://quantum-hvac-controller:8001/health/detailed" || echo "failed")
    
    if [ "$HEALTH_RESPONSE" != "failed" ]; then
        # Parse health response (assuming JSON)
        OPTIMIZATION_COUNT=$(echo "$HEALTH_RESPONSE" | jq -r '.optimizations_completed // 0' 2>/dev/null || echo "0")
        ERROR_COUNT=$(echo "$HEALTH_RESPONSE" | jq -r '.errors_last_hour // 0' 2>/dev/null || echo "0")
        
        log "✅ Application metrics: ${OPTIMIZATION_COUNT} optimizations, ${ERROR_COUNT} errors"
        
        # Alert on high error rate
        if [ "$ERROR_COUNT" -gt 10 ]; then
            log "⚠️  High error count: $ERROR_COUNT errors in last hour"
        fi
    else
        log "❌ Could not retrieve application metrics"
    fi
}

# Send alert notification
send_alert() {
    local severity="$1"
    local message="$2"
    
    # Log the alert
    log "🚨 ALERT [$severity]: $message"
    
    # Send to external monitoring system (customize as needed)
    if [ -n "$WEBHOOK_URL" ]; then
        curl -s -X POST -H "Content-Type: application/json" \
            -d "{\"text\":\"Quantum HVAC Alert [$severity]: $message\", \"timestamp\":\"$TIMESTAMP\"}" \
            "$WEBHOOK_URL" || true
    fi
    
    # Write to alert file for monitoring system pickup
    echo "[$TIMESTAMP] $severity: $message" >> "/var/log/alerts.log"
}

# Main health check routine
main() {
    log "Starting comprehensive health check..."
    
    local failed_checks=0
    local warnings=0
    
    # Core service checks
    if ! check_api_health; then
        send_alert "CRITICAL" "API health check failed"
        ((failed_checks++))
    fi
    
    if ! check_metrics_health; then
        send_alert "WARNING" "Metrics endpoint unhealthy"
        ((warnings++))
    fi
    
    # Infrastructure checks
    if ! check_redis; then
        send_alert "CRITICAL" "Redis connectivity failed"
        ((failed_checks++))
    fi
    
    if ! check_postgres; then
        send_alert "CRITICAL" "PostgreSQL connectivity failed"
        ((failed_checks++))
    fi
    
    if ! check_influxdb; then
        send_alert "WARNING" "InfluxDB health check failed"
        ((warnings++))
    fi
    
    # System resource checks
    check_resources
    
    # Application-specific checks
    check_app_metrics
    
    # Summary
    if [ $failed_checks -eq 0 ]; then
        if [ $warnings -eq 0 ]; then
            log "🎉 All health checks passed!"
        else
            log "⚠️  Health checks completed with $warnings warnings"
        fi
    else
        log "❌ Health checks failed: $failed_checks critical issues, $warnings warnings"
        send_alert "CRITICAL" "Health check summary: $failed_checks failures, $warnings warnings"
        exit 1
    fi
    
    log "Health check completed successfully"
}

# Trap signals for graceful shutdown
trap 'log "Health check interrupted"; exit 130' INT TERM

# Create log file if it doesn't exist
touch "$HEALTH_LOG"

# Run main health check
main
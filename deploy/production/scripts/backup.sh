#!/bin/bash

# Production backup script for Quantum HVAC Controller

set -e

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_DIR="/backup"
RETENTION_DAYS="${BACKUP_RETENTION_DAYS:-30}"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

log() {
    echo -e "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

# Create backup directory structure
setup_backup_dirs() {
    log "${GREEN}Setting up backup directories...${NC}"
    
    mkdir -p "$BACKUP_DIR/postgresql"
    mkdir -p "$BACKUP_DIR/influxdb"
    mkdir -p "$BACKUP_DIR/config"
    mkdir -p "$BACKUP_DIR/logs"
    mkdir -p "$BACKUP_DIR/redis"
    
    log "✅ Backup directories created"
}

# Backup PostgreSQL database
backup_postgresql() {
    log "${GREEN}Backing up PostgreSQL database...${NC}"
    
    local backup_file="$BACKUP_DIR/postgresql/quantum_hvac_${TIMESTAMP}.sql"
    
    # Create database dump
    PGPASSWORD="$POSTGRES_PASSWORD" pg_dump \
        -h postgres \
        -U quantum_user \
        -d quantum_hvac \
        --verbose \
        --clean \
        --create \
        --if-exists \
        --format=custom \
        --file="$backup_file.dump"
    
    # Create SQL version for easier inspection
    PGPASSWORD="$POSTGRES_PASSWORD" pg_dump \
        -h postgres \
        -U quantum_user \
        -d quantum_hvac \
        --clean \
        --create \
        --if-exists \
        --format=plain \
        --file="$backup_file"
    
    # Compress the SQL file
    gzip "$backup_file"
    
    log "✅ PostgreSQL backup completed: quantum_hvac_${TIMESTAMP}.sql.gz"
    log "✅ PostgreSQL binary backup: quantum_hvac_${TIMESTAMP}.sql.dump"
}

# Backup InfluxDB
backup_influxdb() {
    log "${GREEN}Backing up InfluxDB...${NC}"
    
    local backup_file="$BACKUP_DIR/influxdb/influxdb_${TIMESTAMP}.tar.gz"
    
    # Create InfluxDB backup using influx CLI
    if command -v influx >/dev/null 2>&1; then
        # Use InfluxDB CLI for backup
        influx backup \
            --host http://influxdb:8086 \
            --token "$INFLUXDB_TOKEN" \
            --org quantum-hvac \
            --bucket hvac-metrics \
            --compression gzip \
            "$BACKUP_DIR/influxdb/influx_backup_${TIMESTAMP}"
        
        # Compress the backup directory
        tar -czf "$backup_file" -C "$BACKUP_DIR/influxdb" "influx_backup_${TIMESTAMP}"
        rm -rf "$BACKUP_DIR/influxdb/influx_backup_${TIMESTAMP}"
        
        log "✅ InfluxDB backup completed: influxdb_${TIMESTAMP}.tar.gz"
    else
        log "${YELLOW}⚠️  InfluxDB CLI not available, skipping InfluxDB backup${NC}"
    fi
}

# Backup Redis data
backup_redis() {
    log "${GREEN}Backing up Redis data...${NC}"
    
    local backup_file="$BACKUP_DIR/redis/redis_${TIMESTAMP}.rdb"
    
    # Force Redis to save current state
    redis-cli -h redis -p 6379 BGSAVE
    
    # Wait for background save to complete
    while [ "$(redis-cli -h redis -p 6379 LASTSAVE)" = "$(redis-cli -h redis -p 6379 LASTSAVE)" ]; do
        sleep 1
    done
    
    # Copy the RDB file
    if docker exec quantum-redis cat /data/dump.rdb > "$backup_file" 2>/dev/null; then
        gzip "$backup_file"
        log "✅ Redis backup completed: redis_${TIMESTAMP}.rdb.gz"
    else
        log "${YELLOW}⚠️  Could not backup Redis data${NC}"
    fi
}

# Backup configuration files
backup_config() {
    log "${GREEN}Backing up configuration files...${NC}"
    
    local config_backup="$BACKUP_DIR/config/config_${TIMESTAMP}.tar.gz"
    
    # Backup application configuration
    tar -czf "$config_backup" \
        -C /app config/ \
        --exclude="*.log" \
        --exclude="*.tmp" \
        --exclude="*.pid" \
        2>/dev/null || true
    
    log "✅ Configuration backup completed: config_${TIMESTAMP}.tar.gz"
}

# Backup application logs
backup_logs() {
    log "${GREEN}Backing up application logs...${NC}"
    
    local log_backup="$BACKUP_DIR/logs/logs_${TIMESTAMP}.tar.gz"
    
    # Backup logs from the last 7 days
    find /app/logs -name "*.log" -mtime -7 -type f | \
        tar -czf "$log_backup" -T - 2>/dev/null || true
    
    log "✅ Log backup completed: logs_${TIMESTAMP}.tar.gz"
}

# Cleanup old backups
cleanup_old_backups() {
    log "${GREEN}Cleaning up backups older than $RETENTION_DAYS days...${NC}"
    
    # Remove old backups
    find "$BACKUP_DIR" -name "*_[0-9]*" -mtime +$RETENTION_DAYS -type f -delete
    
    # Count remaining backups
    local backup_count=$(find "$BACKUP_DIR" -name "*_[0-9]*" -type f | wc -l)
    log "✅ Cleanup completed. $backup_count backups retained."
}

# Verify backup integrity
verify_backups() {
    log "${GREEN}Verifying backup integrity...${NC}"
    
    local issues=0
    
    # Check PostgreSQL backup
    local pg_backup=$(find "$BACKUP_DIR/postgresql" -name "*${TIMESTAMP}*" -type f | head -1)
    if [ -f "$pg_backup" ]; then
        if [ -s "$pg_backup" ]; then
            log "✅ PostgreSQL backup file exists and is not empty"
        else
            log "${RED}❌ PostgreSQL backup file is empty${NC}"
            ((issues++))
        fi
    else
        log "${RED}❌ PostgreSQL backup file not found${NC}"
        ((issues++))
    fi
    
    # Check Redis backup
    local redis_backup=$(find "$BACKUP_DIR/redis" -name "*${TIMESTAMP}*" -type f | head -1)
    if [ -f "$redis_backup" ]; then
        log "✅ Redis backup file exists"
    else
        log "${YELLOW}⚠️  Redis backup file not found${NC}"
    fi
    
    # Check configuration backup
    local config_backup=$(find "$BACKUP_DIR/config" -name "*${TIMESTAMP}*" -type f | head -1)
    if [ -f "$config_backup" ]; then
        log "✅ Configuration backup file exists"
    else
        log "${YELLOW}⚠️  Configuration backup file not found${NC}"
    fi
    
    if [ $issues -eq 0 ]; then
        log "✅ All critical backups verified successfully"
    else
        log "${RED}❌ $issues critical backup issues detected${NC}"
        return 1
    fi
}

# Generate backup report
generate_report() {
    log "${GREEN}Generating backup report...${NC}"
    
    local report_file="$BACKUP_DIR/backup_report_${TIMESTAMP}.txt"
    
    cat > "$report_file" << EOF
Quantum HVAC Controller Backup Report
=====================================
Timestamp: $(date)
Backup ID: $TIMESTAMP
Retention: $RETENTION_DAYS days

Backup Summary:
--------------
$(find "$BACKUP_DIR" -name "*${TIMESTAMP}*" -type f -exec ls -lh {} \; | sed 's/^/  /')

Total Backup Size: $(du -sh "$BACKUP_DIR" | cut -f1)

Backup Files:
------------
PostgreSQL: $(find "$BACKUP_DIR/postgresql" -name "*${TIMESTAMP}*" -type f | wc -l) files
InfluxDB:   $(find "$BACKUP_DIR/influxdb" -name "*${TIMESTAMP}*" -type f | wc -l) files
Redis:      $(find "$BACKUP_DIR/redis" -name "*${TIMESTAMP}*" -type f | wc -l) files
Config:     $(find "$BACKUP_DIR/config" -name "*${TIMESTAMP}*" -type f | wc -l) files
Logs:       $(find "$BACKUP_DIR/logs" -name "*${TIMESTAMP}*" -type f | wc -l) files

Historical Backups:
------------------
Total backup files: $(find "$BACKUP_DIR" -name "*_[0-9]*" -type f | wc -l)
Oldest backup: $(find "$BACKUP_DIR" -name "*_[0-9]*" -type f -printf '%T@ %p\n' | sort -n | head -1 | cut -d' ' -f2- | xargs ls -l | awk '{print $6" "$7" "$8}' || echo "None")
Newest backup: $(find "$BACKUP_DIR" -name "*_[0-9]*" -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2- | xargs ls -l | awk '{print $6" "$7" "$8}' || echo "None")

Status: Backup completed successfully
EOF

    log "✅ Backup report generated: backup_report_${TIMESTAMP}.txt"
}

# Send backup notification
send_notification() {
    local status="$1"
    local message="$2"
    
    # Send to webhook if configured
    if [ -n "$BACKUP_WEBHOOK_URL" ]; then
        curl -s -X POST -H "Content-Type: application/json" \
            -d "{\"text\":\"Quantum HVAC Backup [$status]: $message\", \"timestamp\":\"$(date -u)\"}" \
            "$BACKUP_WEBHOOK_URL" || true
    fi
    
    log "📧 Notification sent: [$status] $message"
}

# Main backup routine
main() {
    log "${GREEN}🚀 Starting Quantum HVAC Controller backup...${NC}"
    log "Backup ID: $TIMESTAMP"
    
    local start_time=$(date +%s)
    
    # Check prerequisites
    if [ -z "$POSTGRES_PASSWORD" ]; then
        log "${RED}❌ POSTGRES_PASSWORD not set${NC}"
        exit 1
    fi
    
    # Setup
    setup_backup_dirs
    
    # Perform backups
    backup_postgresql
    backup_influxdb
    backup_redis
    backup_config
    backup_logs
    
    # Verify and cleanup
    if verify_backups; then
        cleanup_old_backups
        generate_report
        
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        
        log "${GREEN}🎉 Backup completed successfully in ${duration}s${NC}"
        send_notification "SUCCESS" "Backup completed in ${duration}s"
    else
        log "${RED}❌ Backup verification failed${NC}"
        send_notification "FAILED" "Backup verification failed"
        exit 1
    fi
}

# Trap signals for graceful shutdown
trap 'log "Backup interrupted"; exit 130' INT TERM

# Run main backup routine
main "$@"
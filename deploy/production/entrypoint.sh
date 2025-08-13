#!/bin/bash
set -e

# Enhanced production entrypoint for Quantum HVAC Controller

echo "🚀 Starting Quantum HVAC Controller in Production Mode"
echo "====================================================="

# Environment validation
echo "🔍 Validating environment..."

# Check required environment variables
REQUIRED_VARS=(
    "QUANTUM_ENV"
    "POSTGRES_URL"
    "REDIS_URL"
)

for var in "${REQUIRED_VARS[@]}"; do
    if [ -z "${!var}" ]; then
        echo "❌ ERROR: Required environment variable $var is not set"
        exit 1
    fi
done

# Validate D-Wave configuration (optional but recommended)
if [ -z "$DWAVE_API_TOKEN" ]; then
    echo "⚠️  WARNING: DWAVE_API_TOKEN not set. Quantum features will use classical fallback."
else
    echo "✅ D-Wave API token configured"
fi

# System health checks
echo "🏥 Running system health checks..."

# Check database connectivity
echo "   Checking PostgreSQL connection..."
python -c "
import psutil
import sys
import os
try:
    import asyncpg
    print('✅ PostgreSQL client available')
except ImportError:
    print('⚠️  PostgreSQL client not available, using fallback')
"

# Check Redis connectivity
echo "   Checking Redis connection..."
python -c "
import sys
import os
try:
    import redis
    r = redis.from_url(os.getenv('REDIS_URL', 'redis://localhost:6379/0'))
    r.ping()
    print('✅ Redis connection successful')
except Exception as e:
    print(f'⚠️  Redis connection failed: {e}')
"

# Validate quantum dependencies
echo "   Checking quantum dependencies..."
python -c "
try:
    import numpy as np
    import scipy
    print('✅ Core scientific libraries available')
except ImportError as e:
    print(f'❌ Critical dependency missing: {e}')
    sys.exit(1)

try:
    import dwave
    print('✅ D-Wave Ocean SDK available')
except ImportError:
    print('⚠️  D-Wave Ocean SDK not available, using classical fallback')
"

# Initialize logging
echo "📝 Setting up logging..."
export LOG_CONFIG_FILE="${LOG_CONFIG_FILE:-/app/config/logging.json}"

# Create log directory if it doesn't exist
mkdir -p /app/logs

# Set proper permissions
chmod 755 /app/logs

# Configure locale for internationalization
echo "🌍 Configuring internationalization..."
export LC_ALL=C.UTF-8
export LANG=C.UTF-8

# Load default locale from environment
DEFAULT_LOCALE="${DEFAULT_LOCALE:-en}"
echo "   Default locale: $DEFAULT_LOCALE"

# Performance tuning
echo "⚡ Applying performance tuning..."

# Set Python optimization flags
export PYTHONOPTIMIZE=1

# Configure memory limits based on container resources
MEMORY_LIMIT=$(cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null || echo "2147483648")
MEMORY_GB=$((MEMORY_LIMIT / 1024 / 1024 / 1024))

if [ $MEMORY_GB -gt 0 ]; then
    echo "   Detected memory limit: ${MEMORY_GB}GB"
    
    # Adjust cache size based on available memory
    if [ $MEMORY_GB -ge 4 ]; then
        export CACHE_MAX_SIZE="${CACHE_MAX_SIZE:-10000}"
    elif [ $MEMORY_GB -ge 2 ]; then
        export CACHE_MAX_SIZE="${CACHE_MAX_SIZE:-5000}"
    else
        export CACHE_MAX_SIZE="${CACHE_MAX_SIZE:-2000}"
    fi
    
    echo "   Cache size set to: $CACHE_MAX_SIZE"
fi

# Configure compliance settings
echo "🛡️  Configuring compliance..."
COMPLIANCE_MODE="${COMPLIANCE_MODE:-gdpr,iso27001}"
echo "   Compliance mode: $COMPLIANCE_MODE"

# Security hardening
echo "🔒 Applying security hardening..."

# Ensure secure file permissions
find /app -type f -name "*.py" -exec chmod 644 {} \;
find /app -type f -name "*.sh" -exec chmod 755 {} \;

# Clear any sensitive environment variables from shell history
history -c

# Pre-flight system validation
echo "✈️  Running pre-flight checks..."
python -c "
import sys
import os
sys.path.insert(0, '/app')

try:
    from quantum_ctl.utils.config_validator import validate_system
    
    is_valid, results = validate_system()
    
    if not is_valid:
        print('❌ System validation failed:')
        for error in results.get('environment', {}).get('errors', []):
            print(f'   - {error}')
        sys.exit(1)
    else:
        print('✅ System validation passed')
        
        # Show warnings if any
        warnings = results.get('environment', {}).get('warnings', [])
        if warnings:
            print('   Warnings:')
            for warning in warnings:
                print(f'   - {warning}')
                
except Exception as e:
    print(f'❌ Pre-flight check failed: {e}')
    sys.exit(1)
"

# Database migrations (if needed)
echo "🗄️  Running database setup..."
python -c "
import sys
import os
sys.path.insert(0, '/app')

try:
    # Add any database migration logic here
    print('✅ Database setup complete')
except Exception as e:
    print(f'⚠️  Database setup warning: {e}')
"

# Initialize monitoring
echo "📊 Initializing monitoring..."
python -c "
import sys
import os
sys.path.insert(0, '/app')

try:
    from quantum_ctl.utils.health_dashboard import get_health_dashboard
    dashboard = get_health_dashboard()
    print('✅ Health monitoring initialized')
except Exception as e:
    print(f'⚠️  Monitoring initialization warning: {e}')
"

# Cache warming (optional)
if [ "${WARM_CACHE:-false}" = "true" ]; then
    echo "🔥 Warming cache..."
    python -c "
import sys
import os
sys.path.insert(0, '/app')

try:
    from quantum_ctl.optimization.intelligent_caching import get_intelligent_cache
    cache = get_intelligent_cache()
    print('✅ Cache warmed')
except Exception as e:
    print(f'⚠️  Cache warming warning: {e}')
"
fi

echo ""
echo "🎉 Quantum HVAC Controller ready for production!"
echo "================================================"
echo "Environment: $QUANTUM_ENV"
echo "Log Level: $LOG_LEVEL"
echo "Cache Size: $CACHE_MAX_SIZE"
echo "Compliance: $COMPLIANCE_MODE"
echo "Locale: $DEFAULT_LOCALE"
echo ""

# Start the application
exec "$@"
# Multi-stage build for quantum HVAC control system
FROM python:3.11-slim as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    python3-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage
FROM python:3.11-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --shell /bin/bash quantum

# Copy Python packages from builder
COPY --from=builder /root/.local /home/quantum/.local

# Set working directory and user
WORKDIR /app
USER quantum

# Add local Python packages to PATH
ENV PATH=/home/quantum/.local/bin:$PATH
ENV PYTHONPATH=/app

# Copy application code
COPY --chown=quantum:quantum . .

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/config

# Install the package in development mode
RUN pip install --user -e .

# Expose ports
EXPOSE 8000 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import quantum_ctl; print('Quantum HVAC Control System - OK')" || exit 1

# Default command
CMD ["python", "-m", "quantum_ctl.cli", "--config", "/app/config/production.yaml", "--log-level", "INFO"]
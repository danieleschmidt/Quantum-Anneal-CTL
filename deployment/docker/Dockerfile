# Multi-stage Dockerfile for Quantum HVAC Control System
FROM python:3.12-slim as base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    python3-numpy \
    python3-scipy \
    python3-pandas \
    python3-psutil \
    python3-yaml \
    python3-click \
    python3-aiohttp \
    python3-cryptography \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Production stage
FROM base as production

# Create non-root user
RUN useradd --create-home --shell /bin/bash quantum

# Copy application code
COPY quantum_ctl/ ./quantum_ctl/
COPY examples/ ./examples/
COPY configs/ ./configs/
COPY README.md ./

# Set ownership
RUN chown -R quantum:quantum /app

# Switch to non-root user
USER quantum

# Environment configuration
ENV PYTHONPATH="/app"
ENV QUANTUM_CTL_CONFIG="/app/configs/production.yaml"
ENV QUANTUM_CTL_LOG_LEVEL="INFO"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import quantum_ctl; print('healthy')" || exit 1

# Expose ports
EXPOSE 8080 8081

# Default command
CMD ["python", "-m", "quantum_ctl.cli", "run-dashboard", "--port", "8080"]

# Development stage
FROM base as development

# Install development tools
RUN pip install --no-cache-dir pytest pytest-asyncio pytest-cov black flake8

# Copy all files including tests
COPY . .

# Default command for development
CMD ["python", "-m", "pytest", "tests/", "-v"]
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY pyproject.toml ./
RUN pip install --no-cache-dir -e . && \
    pip install --no-cache-dir pytest pytest-asyncio pytest-cov

# Copy source code
COPY quantum_ctl/ ./quantum_ctl/
COPY tests/ ./tests/
COPY examples/ ./examples/
COPY config/ ./config/
COPY docs/ ./docs/

# Create data directory
RUN mkdir -p /app/data /app/logs

# Set environment variables
ENV PYTHONPATH=/app
ENV QUANTUM_CTL_DATA_DIR=/app/data
ENV QUANTUM_CTL_LOG_DIR=/app/logs

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python3 -c "import quantum_ctl; print('Quantum-Anneal-CTL ready')" || exit 1

# Default command
CMD ["python3", "-m", "quantum_ctl.cli", "--help"]

# Labels
LABEL maintainer="Daniel Schmidt <daniel@terragonlabs.com>"
LABEL version="0.1.0"
LABEL description="Quantum annealing controller for HVAC micro-grids using D-Wave systems"
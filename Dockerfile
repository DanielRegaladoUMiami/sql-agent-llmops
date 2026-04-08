# Multi-stage Dockerfile for SQL Agent

# Stage 1: Builder
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04 as builder

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3-pip \
    build-essential \
    git \
    libsqlite3-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python3.10 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install base packages
RUN pip install --upgrade pip setuptools wheel

# Copy requirements
COPY app/requirements.txt /tmp/requirements.txt

# Install Python dependencies
RUN pip install -r /tmp/requirements.txt

# Stage 2: Runtime
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/app:$PYTHONPATH" \
    HF_HOME="/cache/huggingface" \
    TRANSFORMERS_CACHE="/cache/transformers"

# Install only runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    libsqlite3-0 \
    libgomp1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Create application user
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app /cache /logs /data && \
    chown -R appuser:appuser /app /cache /logs /data

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=appuser:appuser . .

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:7860')" || exit 1

# Default command
CMD ["python", "app/app.py"]

# Build instructions:
# docker build -t sql-agent:latest -f Dockerfile .
#
# Run instructions:
# docker run -it --gpus all -p 7860:7860 \
#   -v /path/to/data:/data \
#   -v /path/to/cache:/cache \
#   sql-agent:latest
#
# With docker-compose:
# docker-compose up -d

# Stage 2 Alternative: Lightweight CPU-only build
# FROM python:3.10-slim
#
# ENV PYTHONUNBUFFERED=1
# RUN apt-get update && apt-get install -y libsqlite3-0 && rm -rf /var/lib/apt/lists/*
# COPY --from=builder /opt/venv /opt/venv
# ENV PATH="/opt/venv/bin:$PATH"
# WORKDIR /app
# COPY . .
# EXPOSE 7860
# CMD ["python", "app/app.py"]

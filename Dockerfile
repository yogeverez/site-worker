# Use Python 3.11 slim-bullseye base image which has fewer vulnerabilities
FROM python:3.11-slim-bullseye

# Set working directory
WORKDIR /app

# Install minimal system dependencies and clean up in the same layer to reduce image size
# Added curl for health checks and enhanced research capabilities
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        libxml2-dev \
        libxslt1-dev \
        zlib1g-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Update pip and install security packages
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements and install with all dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    # Remove pip cache and temporary files to reduce image size
    rm -rf /root/.cache/pip/* && \
    # Set proper permissions for application files
    find /usr/local -type d -exec chmod 755 {} \;

# Create cache directories for the enhanced search system
RUN mkdir -p /app/cache && \
    chmod 755 /app/cache

# Copy application code
COPY app/*.py ./

# Set environment variables for enhanced system
ENV PYTHONUNBUFFERED=1
ENV PORT=8080
ENV PYTHONPATH=/app
ENV ENVIRONMENT=production

# Health check for enhanced monitoring
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Enhanced gunicorn configuration for research workloads
# Increased timeout to 300 seconds for research-intensive operations
# Adjusted worker settings for better research performance
CMD gunicorn -b 0.0.0.0:${PORT} \
    --workers=2 \
    --threads=4 \
    --timeout=300 \
    --keep-alive=65 \
    --preload \
    --max-requests=500 \
    --max-requests-jitter=25 \
    --worker-class=sync \
    --worker-tmp-dir=/dev/shm \
    --log-level=info \
    --error-logfile=- \
    --access-logfile=- \
    --enable-stdio-inheritance \
    --capture-output \
    main:app
# Use Python 3.11 slim-bullseye (minimal surface area)
FROM python:3.11-slim-bullseye

# Set working directory
WORKDIR /app

# Install minimal system deps & upgrade, then clean caches
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

# Upgrade pip, setuptools & wheel without caching
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy and install Python dependencies, then remove pip cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    rm -rf /root/.cache/pip/* && \
    find /usr/local -type d -exec chmod 755 {} \;

# Create application cache directory
RUN mkdir -p /app/cache && chmod 755 /app/cache

# Copy application code
COPY . .

# Environment
ENV PYTHONUNBUFFERED=1 \
    PORT=8080 \
    PYTHONPATH=/app \
    ENVIRONMENT=production

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Gunicorn configuration tuned for research workloads
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
    app.main:app

# Use Python 3.11 slim-bullseye base image which has fewer vulnerabilities
FROM python:3.11-slim-bullseye

# Set working directory
WORKDIR /app

# Install minimal system dependencies and clean up in the same layer to reduce image size
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends build-essential && \
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

# Copy application code
COPY app/*.py ./

# Use gunicorn to run the Flask app
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

# Use shell form of CMD to allow environment variable substitution
# Increase timeout to 120 seconds to prevent worker timeouts during initialization
# Use preload to load the application once before forking workers
# Set max-requests to recycle workers periodically
CMD gunicorn -b 0.0.0.0:${PORT} \
    --workers=1 \
    --threads=8 \
    --timeout=120 \
    --preload \
    --max-requests=1000 \
    --max-requests-jitter=50 \
    --log-level=info \
    --error-logfile - \
    --enable-stdio-inheritance \
    main:app

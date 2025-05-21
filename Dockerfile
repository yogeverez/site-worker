FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY app/requirements.txt .

# Install Python dependencies with verbose output
RUN pip install --no-cache-dir --verbose -r requirements.txt

# Copy application code
COPY app/ .

# Cloud Run listens on $PORT (default 8080)
ENV PORT=8080

# Command to run the application
CMD ["python", "main.py"]

# Use Python 3.11 slim base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (if any needed for certain libraries)
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/*.py ./

# Use gunicorn to run the Flask app
ENV PYTHONUNBUFFERED=1
ENV PORT=8080
CMD ["gunicorn", "-b", "0.0.0.0:$PORT", "--workers=1", "--threads=8", "main:app"]

#!/bin/bash
# Script to test the application in a virtual environment similar to Cloud Run

set -e  # Exit on any error

echo "=== Creating test environment ==="
python -m venv cloud_test_env
source cloud_test_env/bin/activate

echo "=== Installing dependencies from requirements.txt ==="
pip install -r requirements.txt

echo "=== Setting test environment variables ==="
export PORT=8080
export OPENAI_API_KEY=test_key
export SERPAPI_KEY=test_key

echo "=== Running application with Gunicorn (same as Cloud Run) ==="
cd app
gunicorn -b 0.0.0.0:8080 --workers=1 --threads=8 --timeout=120 main:app

# Note: The application will keep running until you stop it with Ctrl+C
# You can test the health endpoint with: curl http://localhost:8080/health

#!/bin/bash
# Script to test the application in a Docker container similar to Cloud Run

set -e  # Exit on any error

echo "=== Building Docker image ==="
/Applications/Docker.app/Contents/Resources/bin/docker build -t site-worker-test .

echo "=== Running container with test environment ==="
/Applications/Docker.app/Contents/Resources/bin/docker run --rm -it \
  -e PORT=8080 \
  -e OPENAI_API_KEY=test_key \
  -e SERPAPI_KEY=test_key \
  -p 8080:8080 \
  site-worker-test

# Note: The container will keep running until you stop it with Ctrl+C
# You can test the health endpoint with: curl http://localhost:8080/health

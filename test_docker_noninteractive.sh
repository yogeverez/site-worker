#!/bin/bash
# Script to test the application in a Docker container similar to Cloud Run (non-interactive mode)

set -e  # Exit on any error

DOCKER_PATH="/Applications/Docker.app/Contents/Resources/bin/docker"

echo "=== Building Docker image ==="
$DOCKER_PATH build -t site-worker-test .

echo "=== Running container in detached mode ==="
CONTAINER_ID=$($DOCKER_PATH run -d \
  -e PORT=8080 \
  -e OPENAI_API_KEY=test_key \
  -e SERPAPI_KEY=test_key \
  -p 8080:8080 \
  site-worker-test)

echo "=== Container started with ID: $CONTAINER_ID ==="
echo "Waiting for container to initialize..."
sleep 5

echo "=== Testing health endpoint ==="
curl -v http://localhost:8080/health

echo ""
echo "=== Container logs ==="
$DOCKER_PATH logs $CONTAINER_ID

echo ""
echo "=== Container is running in the background ==="
echo "To stop the container, run: $DOCKER_PATH stop $CONTAINER_ID"
echo "To view logs, run: $DOCKER_PATH logs $CONTAINER_ID"

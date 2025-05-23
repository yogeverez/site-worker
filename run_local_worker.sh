#!/bin/bash

# Script to build and run the site-worker Docker container locally,
# finding an available port and using local credentials.

# --- Configuration ---
DOCKER_IMAGE_NAME="site-worker-local"
ENV_FILE=".env.local"
# Path to the GCP service account key, relative to this script's directory
SERVICE_ACCOUNT_KEY_RELATIVE_PATH="app/gcp_service_account_key.json"
# Path to the Dockerfile, relative to this script's directory
DOCKERFILE_PATH="."

START_PORT=8110
MAX_PORT_ATTEMPTS=20 # Try up to 20 ports

# --- Helper Functions ---
check_port() {
    local port_to_check=$1
    # For macOS, lsof is a reliable way to check if a port is in use and listening
    if lsof -iTCP:"$port_to_check" -sTCP:LISTEN -P -n &>/dev/null; then
        return 0 # Port is busy (lsof found something)
    else
        return 1 # Port is free (lsof found nothing)
    fi
}

# --- Script Main Logic ---

# 1. Determine script's directory to make paths relative
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

# Change to the script's directory to ensure relative paths work correctly
cd "$SCRIPT_DIR" || exit

# Verify essential files exist
if [ ! -f "$ENV_FILE" ]; then
    echo "Error: Environment file '$ENV_FILE' not found in $SCRIPT_DIR."
    echo "Please ensure this script is in the project root or adjust ENV_FILE path."
    exit 1
fi

SERVICE_ACCOUNT_KEY_ABSOLUTE_PATH="$SCRIPT_DIR/$SERVICE_ACCOUNT_KEY_RELATIVE_PATH"
if [ ! -f "$SERVICE_ACCOUNT_KEY_ABSOLUTE_PATH" ]; then
    echo "Error: Service account key not found at $SERVICE_ACCOUNT_KEY_ABSOLUTE_PATH"
    echo "Please ensure the key exists or adjust SERVICE_ACCOUNT_KEY_RELATIVE_PATH."
    exit 1
fi

# 2. Build the Docker image
echo "Building Docker image '$DOCKER_IMAGE_NAME' from Dockerfile in '$DOCKERFILE_PATH' context..."
if ! /usr/local/bin/docker build -t "$DOCKER_IMAGE_NAME" "$DOCKERFILE_PATH"; then
    echo "Error: Docker build failed."
    exit 1
fi
echo "Docker image '$DOCKER_IMAGE_NAME' built successfully."

# 3. Find an available port
echo "Searching for an available host port starting from $START_PORT..."
HOST_PORT=""
for i in $(seq 0 $((MAX_PORT_ATTEMPTS - 1))); do
    current_port=$((START_PORT + i))
    if check_port "$current_port"; then
        echo "Port $current_port is busy."
    else
        HOST_PORT="$current_port"
        echo "Port $current_port is available."
        break
    fi
done

if [ -z "$HOST_PORT" ]; then
    echo "Error: Could not find an available port after $MAX_PORT_ATTEMPTS attempts (from $START_PORT to $((START_PORT + MAX_PORT_ATTEMPTS -1)))."
    exit 1
fi

# 4. Run the Docker container
CONTAINER_INTERNAL_PORT=8080 # Internal port the application listens on inside the container
GCP_CREDS_CONTAINER_PATH="/app/gcp_service_account_key.json" # Standardized path inside the container

echo ""
echo "Attempting to run Docker container '$DOCKER_IMAGE_NAME'..."
echo "  Mapping host port: $HOST_PORT -> container port: $CONTAINER_INTERNAL_PORT"
echo "  Using environment file: $SCRIPT_DIR/$ENV_FILE"
echo "  Mounting service account key: $SERVICE_ACCOUNT_KEY_ABSOLUTE_PATH -> $GCP_CREDS_CONTAINER_PATH (in container)"

# Use exec to replace the shell process with Docker, so Ctrl+C directly stops Docker
exec /usr/local/bin/docker run \
    -p "${HOST_PORT}:${CONTAINER_INTERNAL_PORT}" \
    --env-file "$SCRIPT_DIR/$ENV_FILE" \
    -v "${SERVICE_ACCOUNT_KEY_ABSOLUTE_PATH}:${GCP_CREDS_CONTAINER_PATH}:ro" \
    -e "GOOGLE_APPLICATION_CREDENTIALS=${GCP_CREDS_CONTAINER_PATH}" \
    "$DOCKER_IMAGE_NAME"

# If exec fails (it shouldn't unless docker command itself is wrong), this part will be reached.
echo "Error: Failed to start Docker container '$DOCKER_IMAGE_NAME'." >&2
exit 1

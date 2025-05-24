#!/bin/bash

# Script to run the site-worker locally with proper authentication
# Based on the successful configurations from previous runs

# Configuration
PORT=8121
MOCK_DB=true
SESSION_ID=$(date +%s)

# Get the absolute path to the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
cd "$SCRIPT_DIR" || exit

# Activate virtual environment
source venv/bin/activate

# Set environment variables based on .env.local if it exists
if [ -f .env.local ]; then
  echo "Loading environment variables from .env.local"
  export $(grep -v '^#' .env.local | xargs)
else
  echo "Warning: .env.local not found. Using default environment variables."
fi

# Override with our testing variables
export PORT=$PORT
export USE_MOCK_DB=$MOCK_DB
export SESSION_ID=$SESSION_ID

echo "🚀 Starting site-worker with the following configuration:"
echo "Port: $PORT"
echo "Mock DB: $MOCK_DB"
echo "Session ID: $SESSION_ID"
echo "OpenAI API Key: ${OPENAI_API_KEY:0:5}... (truncated for security)"

# Run the application
python app/main.py

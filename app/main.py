print("---- PYTHON MAIN.PY STARTING ----", flush=True)
import os, base64, json, time
import logging
import asyncio
import sys
from flask import Flask, request

# Import helper modules and OpenAI Agents setup
from openai import OpenAI

# Import tools module
# In the Docker container, all app/*.py files are in the /app directory, so direct import works.
from tools import do_research, generate_site_content

# Configure logging (REMOVED basicConfig - let Gunicorn handle it)
logger = logging.getLogger(__name__)

# OpenAI client will be initialized in each function that needs it
# using the API key from environment

app = Flask(__name__)

@app.route("/", methods=["POST"])
def pubsub_handler():
    """HTTP endpoint for Pub/Sub push messages."""
    # Explicitly configure logger for this request
    local_logger = logging.getLogger(f"{__name__}.pubsub_handler")
    local_logger.handlers.clear() # Clear any existing handlers from previous calls (if any)
    local_logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stderr) # Direct to stderr
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    local_logger.addHandler(handler)
    local_logger.propagate = False # Prevent messages from being passed to the root logger

    start_time = time.time()
    local_logger.info("Received Pub/Sub message (attempt 1 via local_logger)")
    
    try:
        envelope = request.get_json(silent=True)
        if not envelope or "message" not in envelope:
            local_logger.warning("Bad Request: no Pub/Sub message received")
            return ("Bad Request: no Pub/Sub message received", 400)
        
        msg = envelope["message"]
        local_logger.info(f"Processing message ID: {msg.get('messageId', 'unknown')}")

        # Decode the Pub/Sub message data (which is base64-encoded)
        data = msg.get("data")
        if data:
            try:
                payload = json.loads(base64.b64decode(data).decode("utf-8"))
                local_logger.info(f"Successfully decoded message payload")
            except Exception as e:
                local_logger.error(f"Failed to decode message data: {e}")
                return ("Bad Request: invalid message data", 400)
        else:
            local_logger.warning("No data in message payload")
            payload = {}

        # Extract job parameters
        uid = payload.get("uid")
        mode = payload.get("mode")
        languages = payload.get("languages", [])
        timestamp = payload.get("timestamp")  # Unix timestamp

        local_logger.info(f"Job parameters - UID: {uid}, Mode: {mode}, Languages: {languages}")

        check_not_uid = not uid

        actual_mode_for_check = mode if mode is not None else ''
        actual_languages_for_check = languages if languages is not None else []
        
        check_not_actual_mode = not actual_mode_for_check
        check_not_actual_languages = not actual_languages_for_check

        # Check if essential parameters are missing
        final_condition = check_not_uid or (check_not_actual_mode and check_not_actual_languages)
        
        if final_condition: # Using the debugged condition
            local_logger.warning(f"Missing required parameters: uid='{uid}', mode='{mode}', languages='{languages}'")
            return ("Bad Request: missing uid/mode/languages", 400)

        # Process according to mode
        if mode in ("research", "full"):
            # Perform web research and store results in Firestore
            local_logger.info(f"Starting research mode for user {uid}")
            asyncio.run(do_research(uid, timestamp=timestamp, parent_logger=local_logger))
            local_logger.info(f"Completed research mode for user {uid}")
            
        if mode in ("generate", "full"):
            # Generate site content (and translations) using research data
            local_logger.info(f"Starting content generation for user {uid} in languages: {languages}")
            asyncio.run(generate_site_content(uid, languages, timestamp=timestamp, parent_logger=local_logger))
            local_logger.info(f"Completed content generation for user {uid}")

        elapsed_time = time.time() - start_time
        local_logger.info(f"Job completed successfully in {elapsed_time:.2f} seconds")
        
        return ("", 204)
    except Exception as e:
        local_logger.error(f"Unhandled error processing Pub/Sub message: {e}", exc_info=True)
        return ("Internal Server Error", 500)

# Health check endpoint with detailed status information
@app.route("/health", methods=["GET"])
def health():
    try:
        # Check if OpenAI API key is configured
        api_key_configured = bool(os.getenv("OPENAI_API_KEY", ""))
        
        # Check if SerpAPI key is configured
        serpapi_configured = bool(os.getenv("SERPAPI_KEY", ""))
        
        # Return detailed health status
        status = {
            "status": "healthy",
            "timestamp": int(time.time()),
            "config": {
                "openai_api_configured": api_key_configured,
                "serpapi_configured": serpapi_configured
            }
        }
        logger.info("Health check: Service is healthy")
        return (json.dumps(status), 200, {"Content-Type": "application/json"})
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return (json.dumps({"status": "unhealthy", "error": str(e)}), 500, {"Content-Type": "application/json"})

# Only needed if running the Flask dev server (Cloud Run uses Gunicorn via Dockerfile)
if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)

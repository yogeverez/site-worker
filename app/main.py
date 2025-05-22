import os, base64, json
from flask import Flask, request

# Import helper modules and OpenAI Agents setup
from openai import OpenAI
from tools import do_research, generate_site_content

# OpenAI client will be initialized in each function that needs it
# using the API key from environment

app = Flask(__name__)

@app.route("/", methods=["POST"])
def pubsub_handler():
    """HTTP endpoint for Pub/Sub push messages."""
    envelope = request.get_json(silent=True)
    if not envelope or "message" not in envelope:
        return ("Bad Request: no Pub/Sub message received", 400)
    msg = envelope["message"]

    # Decode the Pub/Sub message data (which is base64-encoded)
    data = msg.get("data")
    if data:
        try:
            payload = json.loads(base64.b64decode(data).decode("utf-8"))
        except Exception as e:
            return ("Bad Request: invalid message data", 400)
    else:
        payload = {}

    # Extract job parameters
    uid = payload.get("uid")
    mode = payload.get("mode")
    languages = payload.get("languages", [])
    timestamp = payload.get("timestamp")  # Unix timestamp

    if not uid or not mode or not languages:
        return ("Bad Request: missing uid/mode/languages", 400)

    # Process according to mode
    try:
        if mode in ("research", "full"):
            # Perform web research and store results in Firestore
            do_research(uid, timestamp=timestamp)
        if mode in ("generate", "full"):
            # Generate site content (and translations) using research data
            generate_site_content(uid, languages, timestamp=timestamp)
    except Exception as e:
        # Log the error (could integrate with Cloud Logging)
        print(f"Error processing job for user {uid}: {e}", flush=True)
        # Return 200 to avoid Pub/Sub retries (or could return 500 to retry)
        return ("Internal Server Error", 500)

    # Acknowledge successful processing
    return ("", 204)

# Optionally, add a health check endpoint
@app.route("/health", methods=["GET"])
def health():
    return ("OK", 200)

# Only needed if running the Flask dev server (Cloud Run uses Gunicorn via Dockerfile)
if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)

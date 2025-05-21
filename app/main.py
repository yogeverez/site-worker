"""
HTTP entry-point for Cloud Run (Pub/Sub push).
"""
from flask import Flask, request, abort
import base64, json, os
from site_generator import run_generation_job

app = Flask(__name__)

@app.post("/")
def pubsub_handler():
    envelope = request.get_json(silent=True)
    if not envelope or "message" not in envelope:
        abort(400, "No Pub/Sub message received")

    msg = envelope["message"]
    data = base64.b64decode(msg["data"]).decode()
    job = json.loads(data)

    uid = job.get("uid")
    langs = job.get("languages", ["en"])
    if not uid:
        abort(400, "Missing uid")

    run_generation_job(uid, langs)
    return ("", 204)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)

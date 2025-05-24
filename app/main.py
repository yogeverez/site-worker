# app/main.py

import os
import time
import asyncio
import base64
import json
import logging

from flask import Flask, request, jsonify

from app.database import get_db
from app.process_status_tracker import ProcessStatusTracker
from app.research_manager import ResearchManager

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'}), 200


@app.route('/status/<uid>/<int:session_id>', methods=['GET'])
def get_status(uid, session_id):
    tracker = ProcessStatusTracker(uid, session_id)
    status = tracker.get_status()
    return jsonify(status), 200


@app.route('/trigger_research/<uid>', methods=['POST'])
def trigger_research(uid):
    payload = request.get_json() or {}
    mode = payload.get('mode', 'full')
    languages = payload.get('languages', ['en'])
    site_input = payload.get('site_input', {})

    session_id = int(time.time())
    # initialize status
    ProcessStatusTracker(uid, session_id).create_initial(status='started')

    # launch workflow in background
    asyncio.create_task(
        ResearchManager(uid, session_id)
        .run(site_input, languages, mode)
    )

    return (
        jsonify({
            'status': 'started',
            'uid': uid,
            'session_id': session_id,
            'mode': mode,
            'languages': languages
        }),
        202
    )


@app.route('/pubsub', methods=['POST'])
async def pubsub_handler():
    envelope = request.get_json()
    if not envelope:
        return 'Bad Request: no JSON', 400

    message = envelope.get('message', {})
    data_b64 = message.get('data')
    try:
        decoded = base64.b64decode(data_b64).decode() if data_b64 else '{}'
        payload = json.loads(decoded)
    except Exception as e:
        logger.error(f"Invalid Pub/Sub payload: {e}")
        return 'Bad Request: invalid payload', 400

    uid = payload.get('uid')
    mode = payload.get('mode', 'full')
    languages = payload.get('languages', ['en'])
    site_input = payload.get('site_input', {})

    session_id = int(time.time())
    # Initialize status tracker - constructor already initializes status
    status_tracker = ProcessStatusTracker(uid, session_id)
    # Update the phase to started
    status_tracker.update_phase('initialization', status='started')

    result = await ResearchManager(uid, session_id).run(site_input, languages, mode)
    return jsonify(result), 200


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    # For local testing; in production use Gunicorn + UvicornWorker
    app.run(host='0.0.0.0', port=port)

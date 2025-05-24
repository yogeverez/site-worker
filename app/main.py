"""
Enhanced Site Worker - Main Application Entry Point
Implements research-first approach with modular agent architecture
"""
print("---- ENHANCED SITE WORKER STARTING ----", flush=True)
import os
import base64
import json
import time
import logging
import asyncio
import sys
from flask import Flask, request, jsonify

# Import enhanced modules implementing research recommendations
from app.research_manager import ResearchManager
from app.research_orchestrator_updated import create_orchestration_pipeline
from app.process_status_tracker import create_status_tracker
from app.database import get_db

# Configure comprehensive logging
logger = logging.getLogger(__name__)

app = Flask(__name__)

class EnhancedPubSubHandler:
    """Enhanced handler implementing research recommendations for robust processing."""
    
    def __init__(self):
        self.setup_logging()
    
    def setup_logging(self):
        """Setup comprehensive logging for the research workflow."""
        # Create formatters for different log levels
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s'
        )
        
        # Setup console handler
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setFormatter(detailed_formatter)
        console_handler.setLevel(logging.INFO)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        root_logger.handlers.clear()
        root_logger.addHandler(console_handler)
        root_logger.propagate = False

    async def validate_request_parameters(self, payload: dict, logger: logging.Logger) -> tuple[bool, str, dict]:
        """Enhanced parameter validation with detailed feedback."""
        uid = payload.get("uid")
        mode = payload.get("mode", "full")
        languages = payload.get("languages", ["en"])
        timestamp = payload.get("timestamp")
        
        # Enhanced validation logic
        validation_result = {
            "uid_valid": bool(uid and uid.strip()),
            "mode_valid": mode in ["full", "research_only", "content_only"],
            "languages_valid": isinstance(languages, list) and all(isinstance(lang, str) for lang in languages),
            "timestamp_valid": isinstance(timestamp, (int, float))
        }
        
        # Detailed validation feedback
        if not validation_result["uid_valid"]:
            return False, "Missing or invalid 'uid' parameter", validation_result
        
        if not validation_result["mode_valid"]:
            return False, f"Invalid 'mode' parameter: {mode}. Must be one of: full, research_only, content_only", validation_result
        
        if not validation_result["languages_valid"]:
            return False, "Invalid 'languages' parameter. Must be a list of language codes", validation_result
            
        return True, "Validation successful", validation_result

    async def process_pubsub_message(self, envelope: dict) -> dict:
        """Process Pub/Sub message with enhanced error handling and observability."""
        start_time = time.time()
        logger.info(f"Starting enhanced Pub/Sub message processing")
        
        try:
            # Extract and decode message
            if not envelope:
                return {"status": "error", "message": "No Pub/Sub message received"}
                
            message = envelope.get('message', {})
            if not message:
                return {"status": "error", "message": "Empty Pub/Sub message"}
                
            if 'data' not in message:
                return {"status": "error", "message": "No data field in Pub/Sub message"}
                
            # Decode base64 message data
            try:
                data_str = base64.b64decode(message['data']).decode('utf-8')
                data = json.loads(data_str)
                logger.info(f"Successfully decoded message: {json.dumps(data)[:200]}...")
            except Exception as e:
                logger.error(f"Failed to decode message: {str(e)}")
                return {"status": "error", "message": f"Failed to decode message: {str(e)}"}
            
            # Validate parameters
            is_valid, error_message, validation_details = await self.validate_request_parameters(data, logger)
            if not is_valid:
                logger.error(f"Parameter validation failed: {error_message}")
                return {"status": "error", "message": error_message, "validation_details": validation_details}
            
            # Extract parameters
            uid = data.get("uid")
            mode = data.get("mode", "full")
            languages = data.get("languages", ["en"])
            
            # Create orchestration pipeline
            logger.info(f"Creating orchestration pipeline for UID: {uid}, Mode: {mode}, Languages: {languages}")
            result = await create_orchestration_pipeline(uid, mode, languages)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            logger.info(f"Completed Pub/Sub processing in {processing_time:.2f}s")
            
            return {
                "status": "success",
                "message": "Processing completed successfully",
                "processing_time_seconds": processing_time,
                "result": result
            }
            
        except Exception as e:
            logger.error(f"Error processing Pub/Sub message: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "message": f"Error processing message: {str(e)}",
                "processing_time_seconds": time.time() - start_time
            }

# Initialize enhanced handler
enhanced_handler = EnhancedPubSubHandler()

@app.route('/', methods=['POST'])
async def main_pubsub_handler():
    """Main HTTP endpoint for Pub/Sub push messages with orchestration pipeline."""
    try:
        envelope = request.get_json()
        if not envelope:
            return jsonify({"status": "error", "message": "No Pub/Sub message received"}), 400
            
        # Process message asynchronously
        result = await enhanced_handler.process_pubsub_message(envelope)
        
        if result.get("status") == "error":
            return jsonify(result), 400
        else:
            return jsonify(result), 200
            
    except Exception as e:
        logger.error(f"Error in Pub/Sub handler: {str(e)}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/test', methods=['GET'])
def test_research_ui():
    """Simple UI for testing research functionality."""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Site Worker Research Test</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            .form-group { margin-bottom: 15px; }
            label { display: block; margin-bottom: 5px; font-weight: bold; }
            input, select, textarea { width: 100%; padding: 8px; box-sizing: border-box; }
            button { padding: 10px 15px; background: #4CAF50; color: white; border: none; cursor: pointer; }
            .results { margin-top: 20px; padding: 15px; background: #f5f5f5; border-radius: 5px; }
            .error { color: red; }
            .success { color: green; }
        </style>
    </head>
    <body>
        <h1>Site Worker Research Test</h1>
        <form id="researchForm">
            <div class="form-group">
                <label for="uid">User ID:</label>
                <input type="text" id="uid" name="uid" required>
            </div>
            <div class="form-group">
                <label for="mode">Processing Mode:</label>
                <select id="mode" name="mode">
                    <option value="full">Full (Research + Content)</option>
                    <option value="research_only">Research Only</option>
                    <option value="content_only">Content Only</option>
                </select>
            </div>
            <div class="form-group">
                <label for="languages">Languages (comma-separated):</label>
                <input type="text" id="languages" name="languages" value="en">
            </div>
            <button type="submit">Start Research</button>
        </form>
        <div class="results" id="results" style="display: none;"></div>
        
        <script>
            document.getElementById('researchForm').addEventListener('submit', async function(e) {
                e.preventDefault();
                const uid = document.getElementById('uid').value;
                const mode = document.getElementById('mode').value;
                const languages = document.getElementById('languages').value.split(',').map(l => l.trim());
                
                document.getElementById('results').innerHTML = '<p>Starting research process...</p>';
                document.getElementById('results').style.display = 'block';
                
                try {
                    const response = await fetch(`/trigger_research/${uid}`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ mode, languages })
                    });
                    
                    const data = await response.json();
                    if (response.ok) {
                        document.getElementById('results').innerHTML = `
                            <p class="success">Research started successfully!</p>
                            <pre>${JSON.stringify(data, null, 2)}</pre>
                        `;
                    } else {
                        document.getElementById('results').innerHTML = `
                            <p class="error">Error: ${data.message}</p>
                            <pre>${JSON.stringify(data, null, 2)}</pre>
                        `;
                    }
                } catch (error) {
                    document.getElementById('results').innerHTML = `
                        <p class="error">Error: ${error.message}</p>
                    `;
                }
            });
        </script>
    </body>
    </html>
    """
    return html

@app.route('/trigger_research/<uid>', methods=['POST'])
async def trigger_research(uid):
    """API endpoint to trigger research for a specific user."""
    try:
        # Get request data
        data = request.get_json() or {}
        mode = data.get('mode', 'full')
        languages = data.get('languages', ['en'])
        
        # Validate parameters
        if not uid or not uid.strip():
            return jsonify({"status": "error", "message": "Invalid user ID"}), 400
            
        if mode not in ['full', 'research_only', 'content_only']:
            return jsonify({"status": "error", "message": f"Invalid mode: {mode}"}), 400
            
        if not isinstance(languages, list) or not all(isinstance(lang, str) for lang in languages):
            return jsonify({"status": "error", "message": "Languages must be a list of strings"}), 400
        
        # Create session ID
        session_id = int(time.time())
        
        # Initialize status tracker
        status_tracker = create_status_tracker(uid, session_id)
        
        # Start research process asynchronously
        logger.info(f"Triggering research for UID: {uid}, Mode: {mode}, Languages: {languages}")
        
        # Run in background task
        asyncio.create_task(create_orchestration_pipeline(uid, mode, languages))
        
        return jsonify({
            "status": "success",
            "message": f"Research process started for user {uid}",
            "uid": uid,
            "session_id": session_id,
            "mode": mode,
            "languages": languages
        }), 200
        
    except Exception as e:
        logger.error(f"Error triggering research: {str(e)}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/health', methods=['GET'])
async def enhanced_health_check():
    """Enhanced health check with comprehensive system status."""
    start_time = time.time()
    
    try:
        health_status = {
            "status": "healthy",
            "timestamp": time.time(),
            "uptime": time.time() - app.start_time if hasattr(app, 'start_time') else None,
            "environment": os.getenv("ENVIRONMENT", "production"),
            "components": {}
        }
        
        # Check database connection
        try:
            db = get_db()
            # Try a simple operation
            db.collection("health_check").document("status").set({"timestamp": time.time()})
            health_status["components"]["database"] = {
                "status": "connected",
                "type": "firestore"
            }
        except Exception as e:
            health_status["components"]["database"] = {
                "status": "error",
                "message": str(e),
                "type": "firestore"
            }
            health_status["status"] = "degraded"
        
        # Check OpenAI API
        try:
            import openai
            client = openai.OpenAI()
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "system", "content": "Health check"}],
                max_tokens=5
            )
            health_status["components"]["openai_api"] = {
                "status": "connected",
                "model": "gpt-3.5-turbo"
            }
        except Exception as e:
            health_status["components"]["openai_api"] = {
                "status": "error",
                "message": str(e)
            }
            health_status["status"] = "degraded"
        
        # Add response time
        health_status["response_time_ms"] = (time.time() - start_time) * 1000
        
        return jsonify(health_status), 200 if health_status["status"] == "healthy" else 503
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}", exc_info=True)
        return jsonify({
            "status": "critical",
            "message": str(e),
            "timestamp": time.time(),
            "response_time_ms": (time.time() - start_time) * 1000
        }), 500

@app.route('/research_status/<uid>', methods=['GET'])
async def research_status(uid: str):
    """New endpoint to check research status for a specific user."""
    try:
        if not uid or not uid.strip():
            return jsonify({"status": "error", "message": "Invalid user ID"}), 400
            
        # Get status from Firestore
        db = get_db()
        status_doc = db.collection("siteGenerationStatus").document(uid).get()
        
        if not status_doc.exists:
            return jsonify({
                "status": "not_found",
                "message": f"No research process found for user {uid}"
            }), 404
            
        status_data = status_doc.to_dict()
        
        # Calculate elapsed time if process is still running
        if status_data.get("status") != "completed" and status_data.get("start_time"):
            status_data["elapsed_seconds"] = time.time() - status_data["start_time"]
            
        return jsonify({
            "status": "success",
            "research_status": status_data
        }), 200
        
    except Exception as e:
        logger.error(f"Error checking research status: {str(e)}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500

# Development and debugging endpoints
if os.getenv("ENVIRONMENT") == "development":
    @app.route('/debug/trigger/<uid>', methods=['GET'])
    async def debug_trigger_research(uid: str):
        """Development endpoint to manually trigger research for a user."""
        try:
            # Default to English only in debug mode
            result = await create_orchestration_pipeline(uid, "full", ["en"])
            return jsonify({
                "status": "success",
                "message": f"Debug research triggered for user {uid}",
                "result": result
            }), 200
        except Exception as e:
            logger.error(f"Debug trigger error: {str(e)}", exc_info=True)
            return jsonify({"status": "error", "message": str(e)}), 500

# Store application start time
app.start_time = time.time()

# Only needed if running the Flask dev server
if __name__ == "__main__":
    port = int(os.getenv("PORT", "8117"))
    debug_mode = os.getenv("ENVIRONMENT") == "development"
    print(f"Starting server on port {port}, debug mode: {debug_mode}", flush=True)
    app.run(host="0.0.0.0", port=port, debug=debug_mode)

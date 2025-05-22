import os
import json
import time
import logging
from flask import Flask, jsonify

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create a simple Flask app for testing
app = Flask(__name__)

@app.route('/')
def hello():
    return 'Hello, World! The site-worker service is running correctly.'

@app.route('/test-port')
def test_port():
    port = os.environ.get('PORT', '8080')
    return f'Using PORT: {port}'

@app.route('/health')
def health():
    try:
        # Check if OpenAI API key is configured
        api_key_configured = bool(os.environ.get('OPENAI_API_KEY', ''))
        
        # Check if SerpAPI key is configured
        serpapi_configured = bool(os.environ.get('SERPAPI_KEY', ''))
        
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
        return jsonify(status)
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({"status": "unhealthy", "error": str(e)}), 500

if __name__ == '__main__':
    # Set environment variables for testing
    os.environ['OPENAI_API_KEY'] = 'test_key'
    os.environ['SERPAPI_KEY'] = 'test_key'
    
    # Get port from environment variable with a default of 8100 (to avoid conflicts)
    port = int(os.environ.get('PORT', '8100'))
    print(f"Starting Flask app on port {port}")
    print(f"Health check endpoint: http://localhost:{port}/health")
    app.run(host='0.0.0.0', port=port)

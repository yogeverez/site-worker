import os
import sys
import json

# Add the app directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'app')))

# Import the app
from app.main import app

# Set environment variables for testing
os.environ['OPENAI_API_KEY'] = 'test_key'
os.environ['SERPAPI_KEY'] = 'test_key'

# Run the health check
with app.test_client() as client:
    response = client.get('/health')
    print(f"Status code: {response.status_code}")
    print(f"Response: {response.data.decode('utf-8')}")
    
    # Parse the JSON response
    health_data = json.loads(response.data)
    print("\nHealth check details:")
    print(f"Status: {health_data['status']}")
    print(f"Timestamp: {health_data['timestamp']}")
    print(f"OpenAI API configured: {health_data['config']['openai_api_configured']}")
    print(f"SerpAPI configured: {health_data['config']['serpapi_configured']}")

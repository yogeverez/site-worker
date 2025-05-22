import os
import json
import base64
import requests
from datetime import datetime

# Set environment variables for testing
os.environ['OPENAI_API_KEY'] = 'test_key'
os.environ['SERPAPI_KEY'] = 'test_key'

# Create a test Pub/Sub message
def create_pubsub_message(uid, mode="research", languages=["en"]):
    # Create the message data
    message_data = {
        "uid": uid,
        "mode": mode,
        "languages": languages
    }
    
    # Encode the message as base64 (simulating Pub/Sub)
    message_bytes = json.dumps(message_data).encode('utf-8')
    encoded_message = base64.b64encode(message_bytes).decode('utf-8')
    
    # Create the Pub/Sub message structure
    pubsub_message = {
        "message": {
            "data": encoded_message,
            "messageId": f"test-message-{int(datetime.now().timestamp())}",
            "publishTime": datetime.now().isoformat()
        }
    }
    
    return pubsub_message

# Test function to send the message to the local server
def test_pubsub_endpoint(url, uid, mode="research", languages=["en"]):
    print(f"Testing Pub/Sub endpoint with: UID={uid}, Mode={mode}, Languages={languages}")
    
    # Create the test message
    pubsub_message = create_pubsub_message(uid, mode, languages)
    
    # Print the message for debugging
    print(f"Sending message: {json.dumps(pubsub_message, indent=2)}")
    
    # Send the request
    try:
        response = requests.post(url, json=pubsub_message)
        print(f"Response status code: {response.status_code}")
        print(f"Response body: {response.text}")
        return response
    except Exception as e:
        print(f"Error sending request: {e}")
        return None

if __name__ == "__main__":
    # Test parameters
    test_uid = "test-user-123"
    test_mode = "research"  # Options: "research", "generate", "full"
    test_languages = ["en"]
    
    # Test the endpoint
    test_pubsub_endpoint("http://localhost:8090/", test_uid, test_mode, test_languages)

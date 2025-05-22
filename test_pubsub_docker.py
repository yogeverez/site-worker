#!/usr/bin/env python3
"""
Test script to simulate a Pub/Sub message to the containerized application.
This helps test the full functionality in a Docker environment similar to Cloud Run.
"""
import os
import json
import base64
import requests
import time
import argparse
from datetime import datetime

def create_pubsub_message(uid, mode="research", languages=["en"]):
    """Create a simulated Pub/Sub message."""
    # Create the message data
    message_data = {
        "uid": uid,
        "mode": mode,
        "languages": languages,
        "timestamp": int(time.time())
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

def test_health(url):
    """Test the health endpoint."""
    try:
        print(f"Testing health endpoint at {url}/health")
        response = requests.get(f"{url}/health", timeout=5)
        print(f"Status code: {response.status_code}")
        print(f"Response: {response.text}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error testing health endpoint: {e}")
        return False

def test_pubsub_endpoint(url, uid, mode="research", languages=["en"]):
    """Test the Pub/Sub endpoint with a simulated message."""
    print(f"Testing Pub/Sub endpoint with: UID={uid}, Mode={mode}, Languages={languages}")
    
    # Create the test message
    pubsub_message = create_pubsub_message(uid, mode, languages)
    
    # Print the message for debugging
    print(f"Sending message: {json.dumps(pubsub_message, indent=2)}")
    
    # Send the request
    try:
        response = requests.post(url, json=pubsub_message, timeout=10)
        print(f"Response status code: {response.status_code}")
        print(f"Response body: {response.text}")
        return response.status_code in [200, 204]
    except Exception as e:
        print(f"Error sending request: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Test Cloud Run service locally in Docker')
    parser.add_argument('--url', default='http://localhost:8080', help='URL of the service')
    parser.add_argument('--uid', default='test-user-123', help='User ID for the test')
    parser.add_argument('--mode', default='research', choices=['research', 'generate', 'full'], 
                        help='Mode for the test (research, generate, or full)')
    parser.add_argument('--languages', default='en', help='Comma-separated list of languages')
    
    args = parser.parse_args()
    languages = args.languages.split(',')
    
    # First test the health endpoint
    if not test_health(args.url):
        print("Health check failed, not proceeding with Pub/Sub test")
        return
    
    # Then test the Pub/Sub endpoint
    success = test_pubsub_endpoint(args.url, args.uid, args.mode, languages)
    
    if success:
        print("Test completed successfully!")
    else:
        print("Test failed!")

if __name__ == "__main__":
    main()

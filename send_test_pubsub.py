import requests
import json
import base64
import argparse
import requests

FIXED_UID = "NiLppqYKUfcKQCnTGNvU2KLCAt03"

def send_pubsub_message(mode: str, port: int):
    """Sends a simulated Pub/Sub message to the local worker."""
    message_data = {
        "uid": FIXED_UID,
        "mode": mode
    }
    
    # Encode the message data to JSON, then to bytes, then to base64
    json_data = json.dumps(message_data)
    base64_encoded_data = base64.b64encode(json_data.encode('utf-8')).decode('utf-8')
    
    pubsub_payload = {
        "message": {
            "data": base64_encoded_data,
            "messageId": "test-message-id-123", # Example message ID
            "attributes": {}
        },
        "subscription": "projects/your-project/subscriptions/your-subscription" # Example subscription
    }
    
    url = f"http://localhost:{port}/"
    headers = {'Content-Type': 'application/json'}
    
    print(f"Sending POST request to {url} with payload:")
    print(json.dumps(pubsub_payload, indent=2))
    
    try:
        response = requests.post(url, json=pubsub_payload, headers=headers, timeout=120)
        print(f"\nResponse Status Code: {response.status_code}")
        try:
            print("Response JSON:")
            print(json.dumps(response.json(), indent=2))
        except json.JSONDecodeError:
            print("Response Text:")
            print(response.text)
    except requests.exceptions.RequestException as e:
        print(f"\nError sending request: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Send a simulated Pub/Sub message to the local worker.")
    parser.add_argument("--mode", type=str, default="full", help="Mode for the message (e.g., 'full', 'regenerate_section').")
    parser.add_argument("--port", type=int, default=8110, help="Port the local worker is running on.")
    
    args = parser.parse_args()
    
    send_pubsub_message(mode=args.mode, port=args.port)

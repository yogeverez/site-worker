import os
from flask import Flask

# Create a simple Flask app for testing
app = Flask(__name__)

@app.route('/')
def hello():
    return 'Hello, World! The Flask app is running correctly.'

@app.route('/test-port')
def test_port():
    port = os.environ.get('PORT', '8080')
    return f'Using PORT: {port}'

if __name__ == '__main__':
    # Get port from environment variable with a default of 8080
    port = int(os.environ.get('PORT', '8080'))
    print(f"Starting Flask app on port {port}")
    app.run(host='0.0.0.0', port=port)

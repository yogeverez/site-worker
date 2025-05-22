import os
import sys

# Set environment variables for testing
os.environ['OPENAI_API_KEY'] = 'test_key'
os.environ['SERPAPI_KEY'] = 'test_key'

# Change to the app directory
os.chdir(os.path.join(os.path.dirname(__file__), 'app'))

# Import the app directly (not as a package)
sys.path.insert(0, '.')
from main import app

# Fix the relative imports
import main
main.do_research = __import__('tools').do_research
main.generate_site_content = __import__('tools').generate_site_content

if __name__ == '__main__':
    port = 8090  # Use a different port to avoid conflicts
    print("Starting Flask application...")
    print(f"Health check endpoint: http://localhost:{port}/health")
    app.run(host='0.0.0.0', port=port, debug=True)

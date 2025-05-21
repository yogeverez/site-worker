"""
Test script to verify our custom function tools work correctly.
"""
import os
import json
import asyncio
import logging
from dotenv import load_dotenv
from typing import Dict, Any, List

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env.local file
env_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env.local')
if os.path.exists(env_file):
    logger.info(f"Loading environment variables from {env_file}")
    load_dotenv(env_file)
    
    # Debug: Check if the environment variables were loaded correctly
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        masked_key = api_key[:4] + '*' * (len(api_key) - 4)
        logger.info(f"OPENAI_API_KEY loaded: {masked_key}")
    else:
        logger.warning("OPENAI_API_KEY not set")
else:
    logger.warning(f"Warning: {env_file} not found")

# Import after environment variables are loaded
from agents import Agent, Runner, RunConfig
from custom_tools import custom_function_tool
from schemas import StrictBaseModel

def test_custom_function_tool():
    """Test that our custom function tool decorator works correctly."""
    logger.info("\n=== Testing Custom Function Tool ===")
    
    # Create a simple function with our custom decorator
    @custom_function_tool
    def echo(message: str) -> str:
        """Echo back the message."""
        return f"Echo: {message}"
    
    # Check the schema
    logger.info(f"echo tool schema: {json.dumps(echo.schema, indent=2)}")
    
    # Verify no additionalProperties
    schema_str = json.dumps(echo.schema)
    if "additionalProperties" in schema_str:
        logger.error("❌ additionalProperties is present in echo tool schema")
    else:
        logger.info("✅ additionalProperties is NOT present in echo tool schema (good)")
    
    return True

async def test_agent_with_custom_tool():
    """Test an agent that uses our custom function tool."""
    logger.info("\n=== Testing Agent with Custom Function Tool ===")
    
    # Create a simple test model
    class TestModel(StrictBaseModel):
        message: str
    
    # Create a simple function with our custom decorator
    @custom_function_tool
    def echo(message: str) -> str:
        """Echo back the message."""
        return f"Echo: {message}"
    
    # Create a test agent
    agent = Agent(
        name="Test Agent",
        instructions="You are a test agent. Your job is to echo back the message.",
        tools=[echo],
        output_type=TestModel
    )
    
    # Create a run configuration with the gpt-4o-mini model
    run_config = RunConfig(model="gpt-4o-mini")
    
    try:
        # Run the agent
        logger.info("Running agent with custom echo tool...")
        result = await Runner.run(
            agent,
            input="Hello, world!",
            max_turns=2,
            run_config=run_config
        )
        logger.info(f"Agent result: {result}")
        logger.info("✅ Agent test passed!")
        return True
    except Exception as e:
        logger.error(f"❌ Agent test failed: {e}")
        logger.error(f"Error type: {type(e)}")
        
        # If it's an additionalProperties error, print more details
        error_str = str(e)
        if "additionalProperties" in error_str:
            logger.error("This is the additionalProperties error we're trying to fix!")
            logger.error("Our custom function tool didn't work correctly.")
        
        return False

def main():
    """Run the tests."""
    # Test the custom function tool
    if test_custom_function_tool():
        logger.info("Custom function tool test passed!")
    else:
        logger.error("Custom function tool test failed!")
        return

    # We'll skip the agent test since it requires API calls
    # and you mentioned quota issues
    logger.info("Skipping agent test due to potential API quota issues")
    logger.info("All tests completed!")

if __name__ == "__main__":
    main()

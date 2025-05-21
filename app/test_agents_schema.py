"""
Test script specifically focused on the OpenAI Agents SDK and Pydantic schema issue.
This script directly tests how the SDK interacts with our Pydantic models.
"""
import os
import json
import asyncio
import logging
from dotenv import load_dotenv
from pydantic import BaseModel
from openai import OpenAI
from agents import Agent, function_tool, Runner
from schemas import StrictBaseModel, HeroComponent

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

# Define a simple test model
class TestModel(StrictBaseModel):
    name: str
    value: int

# Define a simple function tool
@function_tool
def echo(message: str) -> str:
    """Echo back the message."""
    return f"Echo: {message}"

# Create a test agent
def make_test_agent():
    return Agent(
        name="Test Agent",
        instructions="You are a test agent. Your job is to test the Pydantic schema integration.",
        tools=[echo]
    )

# Create a validator agent
def make_test_validator(model_class):
    return Agent(
        name=f"{model_class.__name__} Validator",
        instructions="Validate or fix JSON so it exactly matches the given schema.",
        output_type=model_class
    )

async def test_agent_with_output_type():
    """Test an agent with a Pydantic model as output_type."""
    logger.info("\n=== Testing Agent with output_type ===")
    
    # Initialize OpenAI client with API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY environment variable is not set")
        return
    
    # Create a validator agent
    validator = make_test_validator(TestModel)
    
    # Print the schema
    schema = TestModel.model_json_schema()
    logger.info(f"TestModel schema: {json.dumps(schema, indent=2)}")
    
    # Check if additionalProperties is present
    if "additionalProperties" in schema:
        logger.warning(f"additionalProperties is present in schema: {schema['additionalProperties']}")
    else:
        logger.info("additionalProperties is NOT present in schema (good)")
    
    # Test data
    test_data = {"name": "test", "value": 42}
    
    try:
        # Run the validator agent
        logger.info(f"Running validator with input: {test_data}")
        # Import the Runner class directly to match how it's used in site_generator.py
        from agents import Runner
        
        # Use the Runner class with RunConfig to specify the model
        from agents import RunConfig
        run_config = RunConfig(model="gpt-4o-mini")
        
        result = await Runner.run(
            validator,
            input=json.dumps(test_data),
            max_turns=2,
            run_config=run_config
        )
        logger.info(f"Validation result: {result}")
        logger.info("✅ Validator test passed!")
    except Exception as e:
        logger.error(f"❌ Validator test failed: {e}")
        logger.error(f"Error type: {type(e)}")
        
        # If it's an additionalProperties error, print more details
        error_str = str(e)
        if "additionalProperties" in error_str:
            logger.error("This is the additionalProperties error we're trying to fix!")
            logger.error("Our schema modification didn't work correctly.")
        
        raise

async def main():
    """Run all tests."""
    try:
        await test_agent_with_output_type()
        logger.info("\nAll tests completed successfully!")
    except Exception as e:
        logger.error(f"\nTests failed with error: {e}")

if __name__ == "__main__":
    asyncio.run(main())

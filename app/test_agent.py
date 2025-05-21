"""
Test script for OpenAI Agents SDK integration.
This script tests the basic functionality of the OpenAI Agents SDK with our Pydantic models.
"""
import os
import asyncio
import logging
from dotenv import load_dotenv
from schemas import HeroComponent, StrictBaseModel
from agents import Agent, function_tool, Runner

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file if it exists
load_dotenv()

# Check if OPENAI_API_KEY is set
if not os.environ.get("OPENAI_API_KEY"):
    logger.warning("OPENAI_API_KEY environment variable is not set.")
    logger.warning("Please set it to run this test.")
    logger.warning("You can set it temporarily with: export OPENAI_API_KEY=your_key_here")
    exit(1)

# Define a simple function tool for testing
@function_tool
def echo(message: str) -> str:
    """Echo back the message."""
    return f"Echo: {message}"

# Create a simple agent for testing
test_agent = Agent(
    name="Test Agent",
    instructions="You are a test agent. Your job is to generate a simple hero component.",
    tools=[echo]
)

async def test_agent_with_pydantic():
    """Test the agent with a Pydantic model as output type."""
    logger.info("Testing agent with Pydantic model as output type...")
    
    # Create a validator agent that uses our HeroComponent model
    validator_agent = Agent(
        name="Hero Component Validator",
        instructions="Validate or fix JSON so it exactly matches the given schema.",
        output_type=HeroComponent
    )
    
    # Run the test agent
    prompt = "Generate a hero component for a website about artificial intelligence."
    
    try:
        result = await Runner.run(
            test_agent,
            input=prompt,
            model="gpt-4o-mini",
            max_turns=5
        )
        
        logger.info(f"Agent output: {result}")
        
        # Now validate the result with the validator agent
        validated_result = await Runner.run(
            validator_agent,
            input=str(result),
            model="gpt-4o-mini",
            max_turns=5
        )
        
        logger.info(f"Validated output: {validated_result}")
        logger.info("Test completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during test: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(test_agent_with_pydantic())

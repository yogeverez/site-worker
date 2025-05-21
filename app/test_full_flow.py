"""
Test script to verify the full flow of the site generator with validators.
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
import schemas as sc
from site_generator import orchestrator

async def test_orchestrator_with_validators():
    """Test the orchestrator with validators."""
    logger.info("\n=== Testing Orchestrator with Validators ===")
    
    # Create a run configuration with the gpt-4o-mini model
    run_config = RunConfig(model="gpt-4o-mini")
    
    # Create a test context
    test_uid = "test_user_123"
    test_lang = "en"
    test_seed = {
        "name": "John Doe",
        "jobTitle": "Software Developer",
        "company": "Tech Solutions Inc.",
        "skills": ["JavaScript", "Python", "React", "Node.js"],
        "interests": ["Web Development", "AI", "Machine Learning"]
    }
    
    test_context = {
        "uid": test_uid,
        "lang": test_lang,
        "seed": test_seed
    }
    
    try:
        # Run the orchestrator with a simplified prompt
        logger.info("Running orchestrator with validators...")
        result = await Runner.run(
            orchestrator,
            input=f"Generate a simple hero component for user {test_uid} in language {test_lang}",
            context=test_context,
            max_turns=5,  # Limit to 5 turns to avoid excessive API usage
            run_config=run_config
        )
        
        logger.info(f"Orchestrator result: {result}")
        logger.info("✅ Orchestrator test passed!")
        return True
    except Exception as e:
        logger.error(f"❌ Orchestrator test failed: {e}")
        logger.error(f"Error type: {type(e)}")
        
        # If it's an additionalProperties error, print more details
        error_str = str(e)
        if "additionalProperties" in error_str:
            logger.error("This is the additionalProperties error we're trying to fix!")
        
        return False

async def main():
    """Run the tests."""
    await test_orchestrator_with_validators()

if __name__ == "__main__":
    asyncio.run(main())

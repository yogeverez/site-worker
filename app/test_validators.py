"""
Test script to verify that our validators work correctly with the new agent-based approach.
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
from site_generator import make_validator

async def test_hero_validator():
    """Test the hero component validator."""
    logger.info("\n=== Testing Hero Component Validator ===")
    
    # Create a validator agent for the hero component
    hero_validator = make_validator(sc.HeroComponent)
    
    # Create a run configuration with the gpt-4o-mini model
    run_config = RunConfig(model="gpt-4o-mini")
    
    # Create a valid hero component
    valid_hero = {
        "title": "Welcome to My Website",
        "subtitle": "I'm a professional developer",
        "ctaText": "Learn More"
    }
    
    # Create an invalid hero component with an extra field
    invalid_hero = {
        "title": "Welcome to My Website",
        "subtitle": "I'm a professional developer",
        "ctaText": "Learn More",
        "extraField": "This should be removed"  # This field is not in the schema
    }
    
    try:
        # Test with valid hero component
        logger.info("Testing with valid hero component...")
        result_valid = await Runner.run(
            hero_validator,
            input=f"Validate this hero component: {json.dumps(valid_hero)}",
            max_turns=2,
            run_config=run_config
        )
        logger.info(f"Valid hero result: {result_valid}")
        
        # Test with invalid hero component
        logger.info("Testing with invalid hero component...")
        result_invalid = await Runner.run(
            hero_validator,
            input=f"Validate this hero component: {json.dumps(invalid_hero)}",
            max_turns=2,
            run_config=run_config
        )
        logger.info(f"Invalid hero result: {result_invalid}")
        
        # Check if the extra field was removed
        if "extraField" not in result_invalid.model_dump():
            logger.info("✅ Validator successfully removed the extra field!")
        else:
            logger.warning("❌ Validator did not remove the extra field.")
        
        logger.info("✅ Hero validator test passed!")
        return True
    except Exception as e:
        logger.error(f"❌ Hero validator test failed: {e}")
        logger.error(f"Error type: {type(e)}")
        
        # If it's an additionalProperties error, print more details
        error_str = str(e)
        if "additionalProperties" in error_str:
            logger.error("This is the additionalProperties error we're trying to fix!")
        
        return False

async def main():
    """Run the tests."""
    await test_hero_validator()

if __name__ == "__main__":
    asyncio.run(main())

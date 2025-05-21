"""
Comprehensive test script for the site-worker application.
This script simulates the actual application flow to identify any issues with the OpenAI Agents SDK.
"""
import os
import asyncio
import json
import logging
from dotenv import load_dotenv
from pydantic import BaseModel
from schemas import StrictBaseModel, HeroComponent
from site_generator import make_validator, make_researcher, make_orchestrator, run_generation_job

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env.local file
env_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env.local')
if os.path.exists(env_file):
    logger.info(f"Loading environment variables from {env_file}")
    load_dotenv(env_file)
    
    # Debug: Check if the environment variables were loaded correctly
    for var in ["OPENAI_API_KEY", "SERPAPI_KEY", "UNSPLASH_ACCESS_KEY", "BING_SEARCH_URL"]:
        value = os.getenv(var)
        if value:
            # Show first few characters and mask the rest
            masked_value = value[:4] + '*' * (len(value) - 4) if len(value) > 4 else value
            logger.info(f"  {var}: {masked_value}")
        else:
            logger.warning(f"  {var}: Not set")
else:
    logger.warning(f"Warning: {env_file} not found")

# Check if required environment variables are set
required_vars = ["OPENAI_API_KEY"]
missing_vars = [var for var in required_vars if not os.getenv(var)]

if missing_vars:
    logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
    logger.error("Please set them before running this script.")
    exit(1)

async def test_validator_agent():
    """Test the validator agent with our Pydantic models."""
    logger.info("\n=== Testing Validator Agent ===")
    
    # Create a validator agent for HeroComponent
    validator = make_validator(HeroComponent)
    logger.info(f"Validator created: {validator}")
    
    # Test the validator with a valid hero component
    valid_hero = {
        "title": "AI Solutions",
        "subtitle": "Transforming businesses with intelligent automation",
        "ctaText": "Learn More",
        "backgroundImageUrl": "https://example.com/ai-hero.jpg"
    }
    
    logger.info("Testing validator with valid data...")
    try:
        from agents import Runner
        result = await Runner.run(
            validator,
            input=json.dumps(valid_hero),
            model="gpt-4o-mini",
            max_turns=2
        )
        logger.info(f"Validation result: {result}")
        logger.info("✅ Validator test passed!")
    except Exception as e:
        logger.error(f"❌ Validator test failed: {e}")
        logger.error(f"Error type: {type(e)}")
        raise

async def test_simple_generation():
    """Test a simple generation job with a single language."""
    logger.info("\n=== Testing Simple Generation ===")
    
    # Test user ID and language
    test_uid = "test_user_123"
    test_lang = "en"
    
    logger.info(f"Running generation job for user {test_uid} in language {test_lang}...")
    try:
        await run_generation_job(test_uid, [test_lang])
        logger.info("✅ Generation job completed successfully!")
    except Exception as e:
        logger.error(f"❌ Generation job failed: {e}")
        logger.error(f"Error type: {type(e)}")
        raise

async def main():
    """Run all tests."""
    try:
        # Test the validator agent
        await test_validator_agent()
        
        # Test a simple generation job
        await test_simple_generation()
        
        logger.info("\nAll tests completed successfully!")
    except Exception as e:
        logger.error(f"\nTests failed with error: {e}")

if __name__ == "__main__":
    asyncio.run(main())

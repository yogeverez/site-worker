"""
Final test script to verify that our implementation works without additionalProperties errors.
This test is minimal to avoid hitting API rate limits.
"""
import os
import logging
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env.local file
env_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env.local')
if os.path.exists(env_file):
    logger.info(f"Loading environment variables from {env_file}")
    load_dotenv(env_file)
else:
    logger.warning(f"Warning: {env_file} not found")

def test_import_only():
    """Test that we can import all modules without additionalProperties errors."""
    try:
        logger.info("Importing site_generator module...")
        import site_generator
        logger.info("✅ Successfully imported site_generator module")
        
        logger.info("Importing orchestrator and tools...")
        from site_generator import (
            orchestrator, orchestrator_tools, validators,
            translate_agent, image_search_agent, random_image_agent, save_component_agent
        )
        logger.info("✅ Successfully imported orchestrator and tools")
        
        logger.info("Checking validator agents...")
        for comp, agent in validators.items():
            logger.info(f"Validator for {comp}: {agent.name}")
        logger.info("✅ Successfully checked validator agents")
        
        logger.info("Checking orchestrator tools...")
        logger.info(f"Number of orchestrator tools: {len(orchestrator_tools)}")
        logger.info("✅ Successfully checked orchestrator tools")
        
        logger.info("All import tests passed! No additionalProperties errors.")
        return True
    except Exception as e:
        logger.error(f"❌ Import test failed: {e}")
        logger.error(f"Error type: {type(e)}")
        
        # If it's an additionalProperties error, print more details
        error_str = str(e)
        if "additionalProperties" in error_str:
            logger.error("This is the additionalProperties error we're trying to fix!")
        
        return False

if __name__ == "__main__":
    test_import_only()

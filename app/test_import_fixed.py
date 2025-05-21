"""
Simple test script to verify that the module imports without additionalProperties errors.
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
    
    # Debug: Check if the environment variables were loaded correctly
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        masked_key = api_key[:4] + '*' * (len(api_key) - 4)
        logger.info(f"OPENAI_API_KEY loaded: {masked_key}")
    else:
        logger.warning("OPENAI_API_KEY not set")
else:
    logger.warning(f"Warning: {env_file} not found")

# Test importing the site_generator module
try:
    logger.info("Attempting to import site_generator module...")
    from site_generator import (
        translate_agent, image_search_agent, random_image_agent, save_component_agent,
        orchestrator_tools, orchestrator
    )
    logger.info("✅ Successfully imported site_generator module and agents!")
    
    # Check the orchestrator tools
    logger.info(f"Number of orchestrator tools: {len(orchestrator_tools)}")
    for i, tool in enumerate(orchestrator_tools):
        logger.info(f"Tool {i+1}: {tool}")
    
    logger.info("All tests passed! The additionalProperties issue appears to be fixed.")
except Exception as e:
    logger.error(f"❌ Error importing site_generator: {e}")
    logger.error(f"Error type: {type(e)}")
    
    # If it's an additionalProperties error, print more details
    error_str = str(e)
    if "additionalProperties" in error_str:
        logger.error("This is the additionalProperties error we're trying to fix!")
        logger.error("Our fixes didn't work correctly.")
    
    raise

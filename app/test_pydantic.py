"""
Test script for Pydantic model configuration.
This script tests that our Pydantic models are configured correctly for the OpenAI Agents SDK.
"""
import json
import logging
from schemas import HeroComponent, StrictBaseModel

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_pydantic_models():
    """Test that our Pydantic models are configured correctly."""
    logger.info("Testing Pydantic model configuration...")
    
    # Check the model_config settings
    logger.info(f"StrictBaseModel.model_config: {StrictBaseModel.model_config}")
    
    # Create a valid hero component
    valid_hero = {
        "title": "AI Solutions",
        "subtitle": "Transforming businesses with intelligent automation",
        "ctaText": "Learn More",
        "backgroundImageUrl": "https://example.com/ai-hero.jpg"
    }
    
    # Create an invalid hero component with extra fields
    invalid_hero = {
        "title": "AI Solutions",
        "subtitle": "Transforming businesses with intelligent automation",
        "ctaText": "Learn More",
        "backgroundImageUrl": "https://example.com/ai-hero.jpg",
        "extra_field": "This should cause validation to fail"
    }
    
    # Test valid hero component
    try:
        hero = HeroComponent(**valid_hero)
        logger.info(f"Valid hero component created: {hero}")
        
        # Check the JSON schema
        schema = HeroComponent.model_json_schema()
        logger.info(f"JSON schema: {json.dumps(schema, indent=2)}")
        
        # Verify additionalProperties is set to false
        if schema.get("additionalProperties") is False:
            logger.info("✅ additionalProperties is correctly set to false")
        else:
            logger.warning("❌ additionalProperties is not set to false")
        
    except Exception as e:
        logger.error(f"Error with valid hero component: {e}")
    
    # Test invalid hero component
    try:
        hero = HeroComponent(**invalid_hero)
        logger.error("❌ Invalid hero component was accepted!")
    except Exception as e:
        logger.info(f"✅ Invalid hero component correctly rejected: {e}")
    
    logger.info("Test completed!")

if __name__ == "__main__":
    test_pydantic_models()

"""
Simple test script to verify Pydantic schema configuration without making API calls.
"""
import json
import logging
from schemas import StrictBaseModel, HeroComponent

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define a simple test model
class TestModel(StrictBaseModel):
    name: str
    value: int

def test_schema_generation():
    """Test that our models generate schemas without additionalProperties."""
    # Test the TestModel schema
    schema = TestModel.model_json_schema()
    logger.info(f"TestModel schema: {json.dumps(schema, indent=2)}")
    
    # Check if additionalProperties is present
    if "additionalProperties" in schema:
        logger.error(f"❌ additionalProperties is present in schema: {schema['additionalProperties']}")
    else:
        logger.info("✅ additionalProperties is NOT present in schema (good)")
    
    # Test the HeroComponent schema
    hero_schema = HeroComponent.model_json_schema()
    logger.info(f"HeroComponent schema: {json.dumps(hero_schema, indent=2)}")
    
    # Check if additionalProperties is present
    if "additionalProperties" in hero_schema:
        logger.error(f"❌ additionalProperties is present in HeroComponent schema: {hero_schema['additionalProperties']}")
    else:
        logger.info("✅ additionalProperties is NOT present in HeroComponent schema (good)")
    
    # Test validation
    try:
        # Valid data
        valid_data = {"name": "test", "value": 42}
        model = TestModel(**valid_data)
        logger.info(f"✅ Valid data accepted: {model}")
        
        # Invalid data with extra field
        invalid_data = {"name": "test", "value": 42, "extra": "field"}
        try:
            model = TestModel(**invalid_data)
            logger.error("❌ Invalid data with extra field was accepted!")
        except Exception as e:
            logger.info(f"✅ Invalid data correctly rejected: {str(e)}")
        
        # Invalid data with wrong type
        invalid_type = {"name": "test", "value": "not an integer"}
        try:
            model = TestModel(**invalid_type)
            logger.error("❌ Invalid data with wrong type was accepted!")
        except Exception as e:
            logger.info(f"✅ Invalid data with wrong type correctly rejected: {str(e)}")
            
    except Exception as e:
        logger.error(f"❌ Validation test failed: {e}")

if __name__ == "__main__":
    test_schema_generation()

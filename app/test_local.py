"""
Local test script for the OpenAI Agents SDK integration.
This script helps diagnose Pydantic model validation issues.
"""
import os
import asyncio
import json
import sys
from dotenv import load_dotenv
from pydantic import BaseModel
from schemas import StrictBaseModel, HeroComponent
from site_generator import make_validator, make_researcher, make_orchestrator

# Load environment variables from .env.local file
env_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env.local')
if os.path.exists(env_file):
    print(f"Loading environment variables from {env_file}")
    load_dotenv(env_file)
    
    # Debug: Check if the environment variables were loaded correctly
    print("\nEnvironment Variables Loaded:")
    for var in ["OPENAI_API_KEY", "SERPAPI_KEY", "UNSPLASH_ACCESS_KEY", "BING_SEARCH_URL"]:
        value = os.getenv(var)
        if value:
            # Show first few characters and mask the rest
            masked_value = value[:4] + '*' * (len(value) - 4) if len(value) > 4 else value
            print(f"  {var}: {masked_value}")
        else:
            print(f"  {var}: Not set")
else:
    print(f"Warning: {env_file} not found")
    print("You can create this file with your API keys for local testing")

# Check if required environment variables are set
required_vars = ["OPENAI_API_KEY"]
missing_vars = [var for var in required_vars if not os.getenv(var)]

if missing_vars:
    print(f"Missing required environment variables: {', '.join(missing_vars)}")
    print("Please set them before running this script.")
    print("Example: export OPENAI_API_KEY=your_key")
    exit(1)

def test_pydantic_schema():
    """Test Pydantic schema generation to diagnose additionalProperties issue."""
    print("\n=== Testing Pydantic Schema Generation ===")
    
    # Check BaseModel vs StrictBaseModel schema differences
    class RegularModel(BaseModel):
        name: str
        value: int
    
    class StrictModel(StrictBaseModel):
        name: str
        value: int
    
    print("\nRegular BaseModel schema:")
    regular_schema = RegularModel.model_json_schema()
    print(json.dumps(regular_schema, indent=2))
    
    print("\nStrictBaseModel schema:")
    strict_schema = StrictModel.model_json_schema()
    print(json.dumps(strict_schema, indent=2))
    
    # Check if additionalProperties is present in either schema
    print("\nChecking additionalProperties settings:")
    print(f"RegularModel has additionalProperties: {'additionalProperties' in regular_schema}")
    print(f"StrictModel has additionalProperties: {'additionalProperties' in strict_schema}")
    
    if 'additionalProperties' in strict_schema:
        print(f"additionalProperties value: {strict_schema['additionalProperties']}")
    
    # Test HeroComponent schema
    print("\nHeroComponent schema:")
    hero_schema = HeroComponent.model_json_schema()
    print(json.dumps(hero_schema, indent=2))
    
    print(f"HeroComponent has additionalProperties: {'additionalProperties' in hero_schema}")
    if 'additionalProperties' in hero_schema:
        print(f"additionalProperties value: {hero_schema['additionalProperties']}")

async def test_agent_creation():
    """Test creating agents to see if there are any validation issues."""
    print("\n=== Testing Agent Creation ===")
    
    # Test creating validator
    print("\nCreating validator agent...")
    validator = make_validator(HeroComponent)
    print(f"Validator created: {validator}")
    
    # Test creating researcher
    print("\nCreating researcher agent...")
    researcher = make_researcher()
    print(f"Researcher created: {researcher}")
    
    # Test creating orchestrator
    print("\nCreating orchestrator agent...")
    orchestrator = make_orchestrator()
    print(f"Orchestrator created: {orchestrator}")

if __name__ == "__main__":
    # Test Pydantic schema generation
    test_pydantic_schema()
    
    # Test agent creation
    asyncio.run(test_agent_creation())
    
    print("\nAll tests completed!")

#!/usr/bin/env python3
"""
Simple test script to verify our changes to tools.py and site_agents.py
"""
import os
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the app directory to the path so we can import from it
sys.path.append(os.path.join(os.path.dirname(__file__), "app"))

def test_imports():
    """Test that we can import the modules without errors"""
    logger.info("Testing imports...")
    try:
        from app import tools
        from app import site_agents
        from app.schemas import ResearchDoc
        logger.info("✅ All modules imported successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Import error: {e}")
        return False

def test_agent_initialization():
    """Test that the agents are initialized correctly"""
    logger.info("Testing agent initialization...")
    try:
        from app.site_agents import hero_agent, about_agent, features_agent, researcher_agent
        logger.info("✅ All agents initialized successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Agent initialization error: {e}")
        return False

def test_research_functions():
    """Test the research functions"""
    logger.info("Testing research functions...")
    try:
        from app.tools import web_search, fetch_url, strip_html
        
        # Just test that the functions exist and are callable
        assert callable(web_search), "web_search is not callable"
        assert callable(fetch_url), "fetch_url is not callable"
        assert callable(strip_html), "strip_html is not callable"
        
        logger.info("✅ All research functions are callable")
        return True
    except Exception as e:
        logger.error(f"❌ Research functions error: {e}")
        return False

def test_circular_imports():
    """Test that there are no circular import issues"""
    logger.info("Testing for circular import issues...")
    try:
        # Import in different order to check for circular imports
        from app import site_agents
        from app import tools
        
        # Check that the functions in site_agents are properly initialized
        assert site_agents.web_search is not None, "site_agents.web_search is None"
        assert site_agents.fetch_url is not None, "site_agents.fetch_url is None"
        assert site_agents.strip_html is not None, "site_agents.strip_html is None"
        
        logger.info("✅ No circular import issues detected")
        return True
    except Exception as e:
        logger.error(f"❌ Circular import error: {e}")
        return False

if __name__ == "__main__":
    logger.info("Starting tests...")
    
    tests = [
        test_imports,
        test_agent_initialization,
        test_research_functions,
        test_circular_imports
    ]
    
    success = True
    for test in tests:
        if not test():
            success = False
    
    if success:
        logger.info("🎉 All tests passed!")
        sys.exit(0)
    else:
        logger.error("❌ Some tests failed")
        sys.exit(1)

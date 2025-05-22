#!/usr/bin/env python3
"""
Test script to verify all imports are working correctly.
This will help identify any missing dependencies.
"""
import os
import sys
import time
import json
import base64
import logging
import importlib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_import(module_name):
    """Test importing a module and report success/failure."""
    try:
        module = importlib.import_module(module_name)
        logger.info(f"✅ Successfully imported {module_name}")
        return True
    except ImportError as e:
        logger.error(f"❌ Failed to import {module_name}: {e}")
        return False

def main():
    """Run all import tests."""
    logger.info("Starting import tests...")
    
    # Core Python modules
    test_import("os")
    test_import("sys")
    test_import("time")
    test_import("json")
    test_import("base64")
    test_import("logging")
    
    # Flask and related
    test_import("flask")
    test_import("werkzeug")
    test_import("jinja2")
    test_import("markupsafe")
    test_import("itsdangerous")
    test_import("click")
    test_import("blinker")
    test_import("gunicorn")
    
    # Google Cloud
    test_import("google.cloud")
    test_import("google.cloud.firestore")
    
    # OpenAI and Pydantic
    test_import("openai")
    test_import("agents")
    test_import("pydantic")
    test_import("pydantic_core")
    
    # Web scraping
    test_import("requests")
    test_import("bs4")
    
    # App modules
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))
    test_import("main")
    test_import("tools")
    test_import("site_agents")
    
    logger.info("Import tests completed.")

if __name__ == "__main__":
    main()

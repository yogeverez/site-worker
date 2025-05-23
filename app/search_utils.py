import os
import requests
import logging
import time
from typing import List, Dict

# Configure logging
logger = logging.getLogger(__name__)

# SerpAPI configuration
SERPAPI_API_KEY = os.getenv("SERPAPI_KEY", "")

def search_web(query: str, num_results: int = 5) -> List[Dict[str, str]]:
    """Use SerpAPI to search the web and return a list of result dicts (title, link, snippet)."""
    time.sleep(2) # Add a delay before each search operation
    if not SERPAPI_API_KEY:
        logger.warning("SerpAPI key not configured, returning empty results")
        return []
    
    params = {
        "engine": "google", 
        "q": query, 
        "api_key": SERPAPI_API_KEY,
        "num": num_results
    }
    
    # Add retry logic for SerpAPI calls
    max_retries = 3
    retry_count = 0
    results = []
    
    while retry_count < max_retries:
        try:
            logger.info(f"Searching web for '{query}' (attempt {retry_count + 1}/{max_retries})")
            resp = requests.get(
                "https://serpapi.com/search", 
                params=params, 
                timeout=15  # 15 second timeout
            )
            
            if resp.status_code == 200:
                data = resp.json()
                for res in data.get("organic_results", []):
                    link = res.get("link")
                    title = res.get("title")
                    snippet = res.get("snippet")
                    if link and title:
                        results.append({
                            "title": title, 
                            "url": link, 
                            "snippet": snippet if snippet is not None else ""  # Ensure snippet is a string
                        })
                
                logger.info(f"Found {len(results)} search results for '{query}'")
                break  # Success, exit retry loop
            else:
                logger.warning(f"SerpAPI returned status code {resp.status_code}")
                retry_count += 1
                if retry_count < max_retries:
                    time.sleep(1)  # Wait before retrying
        
        except Exception as e:
            logger.error(f"Error searching web: {e}")
            retry_count += 1
            if retry_count < max_retries:
                time.sleep(1)  # Wait before retrying
            else:
                logger.error(f"Failed to search web after {max_retries} attempts")
    
    return results

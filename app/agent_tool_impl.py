"""agent_tool_impl.py - Concrete implementations of tools for researcher_agent."""
import os
import requests
import re
import logging
import time
from bs4 import BeautifulSoup
from typing import List, Dict, Any
from agents import function_tool

# Import search_web from search_utils.py
from search_utils import search_web as generic_search_web

logger = logging.getLogger(__name__)

@function_tool
def agent_web_search(query: str, num_results: int = 5) -> List[Dict[str, str]]:
    """Search the web using SerpAPI (via generic_search_web) and return a list of result dicts."""
    return generic_search_web(query, num_results)

@function_tool
def agent_fetch_url(url: str) -> str:
    """Fetch the raw HTML content from a URL."""
    max_retries = 2
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            logger.info(f"Fetching content from {url} (attempt {retry_count + 1}/{max_retries})")
            resp = requests.get(
                url, 
                timeout=10,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                    "Accept": "text/html,application/xhtml+xml,application/xml",
                    "Accept-Language": "en-US,en;q=0.9"
                }
            )
            
            if resp.status_code == 200 and resp.text:
                logger.info(f"Successfully fetched content from {url}")
                return resp.text
            else:
                logger.warning(f"Failed to fetch content from {url}: HTTP {resp.status_code}")
                retry_count += 1
                if retry_count < max_retries:
                    time.sleep(1)
        except Exception as e:
            logger.error(f"Error fetching content from {url}: {e}", exc_info=True)
            retry_count += 1
            if retry_count < max_retries:
                time.sleep(1)
            else:
                logger.error(f"Failed to fetch content from {url} after {max_retries} attempts")
    return ""

@function_tool
def agent_strip_html(html: str) -> str:
    """Convert HTML to plain text."""
    if not html:
        return ""
    try:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.extract()
        text = soup.get_text(separator=" ")
        return re.sub(r"\s+\s+", " ", text).strip()
    except Exception as e:
        logger.error(f"Error stripping HTML: {e}", exc_info=True)
        return ""

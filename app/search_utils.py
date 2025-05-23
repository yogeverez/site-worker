import os
import requests
import logging
import time
import urllib.parse
from typing import List, Dict, Optional, Tuple
import hashlib
import json
from datetime import datetime, timedelta

# Configure logging
logger = logging.getLogger(__name__)

# SerpAPI configuration
SERPAPI_API_KEY = os.getenv("SERPAPI_KEY", "")

class SearchRateLimiter:
    """Rate limiter for search API calls to prevent quota exhaustion."""
    
    def __init__(self, max_calls_per_minute: int = 10):
        self.max_calls_per_minute = max_calls_per_minute
        self.calls_log = []
    
    def can_make_call(self) -> bool:
        """Check if a new API call can be made without exceeding rate limits."""
        now = datetime.now()
        # Remove calls older than 1 minute
        self.calls_log = [call_time for call_time in self.calls_log if now - call_time < timedelta(minutes=1)]
        
        return len(self.calls_log) < self.max_calls_per_minute
    
    def record_call(self):
        """Record a new API call."""
        self.calls_log.append(datetime.now())
    
    def wait_time_until_next_call(self) -> float:
        """Calculate how long to wait before the next call can be made."""
        if self.can_make_call():
            return 0.0
        
        # Find the oldest call that's still within the minute window
        now = datetime.now()
        oldest_relevant_call = min(call_time for call_time in self.calls_log if now - call_time < timedelta(minutes=1))
        
        # Wait until that call is more than a minute old
        wait_until = oldest_relevant_call + timedelta(minutes=1)
        return max(0.0, (wait_until - now).total_seconds())

# Global rate limiter instance
rate_limiter = SearchRateLimiter(max_calls_per_minute=8)  # Conservative limit

class SearchCache:
    """Simple in-memory cache for search results to reduce API calls."""
    
    def __init__(self, ttl_minutes: int = 60):
        self.cache = {}
        self.ttl = timedelta(minutes=ttl_minutes)
    
    def _generate_key(self, query: str, num_results: int) -> str:
        """Generate a cache key for the query."""
        content = f"{query.lower().strip()}_{num_results}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get(self, query: str, num_results: int) -> Optional[List[Dict[str, str]]]:
        """Get cached results if available and not expired."""
        key = self._generate_key(query, num_results)
        
        if key in self.cache:
            cached_data, timestamp = self.cache[key]
            if datetime.now() - timestamp < self.ttl:
                logger.info(f"🎯 Cache hit for query: '{query}'")
                return cached_data
            else:
                # Remove expired entry
                del self.cache[key]
                logger.debug(f"⏰ Cache expired for query: '{query}'")
        
        return None
    
    def set(self, query: str, num_results: int, results: List[Dict[str, str]]):
        """Cache the search results."""
        key = self._generate_key(query, num_results)
        self.cache[key] = (results, datetime.now())
        logger.debug(f"💾 Cached results for query: '{query}'")
    
    def clear_expired(self):
        """Remove all expired entries from the cache."""
        now = datetime.now()
        expired_keys = [
            key for key, (_, timestamp) in self.cache.items()
            if now - timestamp >= self.ttl
        ]
        
        for key in expired_keys:
            del self.cache[key]
        
        if expired_keys:
            logger.debug(f"🧹 Cleared {len(expired_keys)} expired cache entries")

# Global cache instance
search_cache = SearchCache(ttl_minutes=30)

def enhanced_search_web(query: str, num_results: int = 5, use_cache: bool = True, bypass_rate_limit: bool = False) -> List[Dict[str, str]]:
    """
    Enhanced web search function implementing research recommendations:
    - Rate limiting to prevent API quota exhaustion
    - Caching to reduce redundant API calls
    - Better error handling and retry logic
    - Enhanced result validation and metadata
    """
    
    # Input validation
    if not query or not query.strip():
        logger.warning("Empty search query provided")
        return []
    
    query = query.strip()
    num_results = max(1, min(num_results, 20))  # Clamp between 1 and 20
    
    # Check cache first
    if use_cache:
        cached_results = search_cache.get(query, num_results)
        if cached_results is not None:
            return cached_results
    
    # Check if SerpAPI is configured
    if not SERPAPI_API_KEY:
        logger.warning("SerpAPI key not configured, returning empty results")
        return []
    
    # Check bypass flag for research credits conservation
    if os.getenv("BYPASS_SERPAPI_RESEARCH", "false").lower() == "true" and not bypass_rate_limit:
        logger.warning("🚫 SerpAPI research bypassed due to configuration setting")
        return []
    
    # Rate limiting check
    if not bypass_rate_limit and not rate_limiter.can_make_call():
        wait_time = rate_limiter.wait_time_until_next_call()
        if wait_time > 0:
            logger.warning(f"⏳ Rate limit reached. Would need to wait {wait_time:.1f}s. Returning empty results.")
            return []
    
    # Prepare search parameters
    params = {
        "engine": "google",
        "q": query,
        "api_key": SERPAPI_API_KEY,
        "num": num_results,
        "gl": "us",  # Geolocation
        "hl": "en",  # Language
        "safe": "off",  # Don't filter results
        "filter": "0"  # Include similar results
    }
    
    max_retries = 3
    retry_count = 0
    results = []
    
    while retry_count < max_retries:
        try:
            logger.info(f"🔍 Searching web for '{query}' (attempt {retry_count + 1}/{max_retries}, requesting {num_results} results)")
            
            # Record the API call for rate limiting
            if not bypass_rate_limit:
                rate_limiter.record_call()
            
            # Add delay before each search operation (progressive backoff)
            if retry_count > 0:
                delay = min(2 ** retry_count, 8)  # Exponential backoff capped at 8 seconds
                logger.info(f"⏳ Waiting {delay}s before retry...")
                time.sleep(delay)
            else:
                time.sleep(1)  # Base delay to be respectful to the API
            
            # Make the request
            resp = requests.get(
                "https://serpapi.com/search",
                params=params,
                timeout=20,  # Increased timeout
                headers={
                    "User-Agent": "Enhanced-Research-Bot/1.0",
                    "Accept": "application/json"
                }
            )
            
            # Enhanced status code handling
            if resp.status_code == 200:
                try:
                    data = resp.json()
                    raw_results = data.get("organic_results", [])
                    
                    # Process and validate results
                    for i, res in enumerate(raw_results):
                        link = res.get("link")
                        title = res.get("title")
                        snippet = res.get("snippet")
                        position = res.get("position", i + 1)
                        
                        # Basic validation
                        if not link or not title:
                            logger.debug(f"Skipping result {i}: missing link or title")
                            continue
                        
                        # Enhanced result object
                        enhanced_result = {
                            "title": title.strip(),
                            "url": link.strip(),
                            "snippet": snippet.strip() if snippet else "",
                            "position": position,
                            "domain": _extract_domain(link),
                            "search_timestamp": datetime.now().isoformat(),
                            "search_query": query
                        }
                        
                        # Add additional metadata if available
                        if "rich_snippet" in res:
                            enhanced_result["rich_snippet"] = res["rich_snippet"]
                        
                        if "sitelinks" in res:
                            enhanced_result["sitelinks"] = res["sitelinks"]
                        
                        # Add source type hints
                        enhanced_result["source_type_hint"] = _classify_source_type(link, title)
                        
                        results.append(enhanced_result)
                    
                    # Cache successful results
                    if use_cache and results:
                        search_cache.set(query, num_results, results)
                    
                    logger.info(f"✅ Successfully found {len(results)} search results for '{query}'")
                    break  # Success, exit retry loop
                    
                except json.JSONDecodeError as e:
                    logger.error(f"❌ Failed to parse JSON response from SerpAPI: {e}")
                    retry_count += 1
                    continue
                    
            elif resp.status_code == 401:
                logger.error(f"🔑 SerpAPI authentication failed (401) - check API key")
                break  # Don't retry authentication errors
                
            elif resp.status_code == 429:
                logger.warning(f"⏳ SerpAPI rate limit hit (429)")
                retry_count += 1
                if retry_count < max_retries:
                    time.sleep(30)  # Longer wait for rate limiting
                    
            elif resp.status_code >= 500:
                logger.warning(f"🔧 SerpAPI server error ({resp.status_code})")
                retry_count += 1
                
            else:
                logger.warning(f"⚠️ SerpAPI returned status code {resp.status_code}")
                retry_count += 1
        
        except requests.exceptions.Timeout:
            logger.warning(f"⏰ Timeout occurred during search for '{query}'")
            retry_count += 1
            
        except requests.exceptions.ConnectionError:
            logger.warning(f"🔌 Connection error during search for '{query}'")
            retry_count += 1
            
        except Exception as e:
            logger.error(f"💥 Unexpected error during search for '{query}': {e}", exc_info=True)
            retry_count += 1
    
    if not results and retry_count >= max_retries:
        logger.error(f"❌ Failed to search web for '{query}' after {max_retries} attempts")
    
    return results

def _extract_domain(url: str) -> str:
    """Extract the domain from a URL."""
    try:
        parsed = urllib.parse.urlparse(url)
        return parsed.netloc.lower()
    except Exception:
        return "unknown"

def _classify_source_type(url: str, title: str) -> str:
    """Classify the likely source type based on URL and title."""
    url_lower = url.lower()
    title_lower = title.lower()
    
    # Professional platforms
    if 'linkedin.com' in url_lower:
        return 'linkedin'
    elif 'github.com' in url_lower:
        return 'github'
    elif any(x in url_lower for x in ['twitter.com', 'x.com']):
        return 'twitter'
    elif 'facebook.com' in url_lower:
        return 'facebook'
    elif 'instagram.com' in url_lower:
        return 'instagram'
    
    # Content platforms
    elif 'medium.com' in url_lower:
        return 'blog_post'
    elif any(x in url_lower for x in ['blog', 'wordpress', 'blogspot']):
        return 'blog_post'
    elif 'youtube.com' in url_lower or 'youtu.be' in url_lower:
        return 'video'
    
    # Professional/academic
    elif any(x in url_lower for x in ['.edu', 'university', 'college', 'academic']):
        return 'academic'
    elif any(x in url_lower for x in ['researchgate', 'scholar.google', 'arxiv']):
        return 'academic'
    
    # News and media
    elif any(x in url_lower for x in ['techcrunch', 'wired', 'reuters', 'bbc', 'cnn', 'forbes']):
        return 'news_article'
    elif any(x in title_lower for x in ['interview', 'news', 'article', 'report']):
        return 'news_article'
    
    # Company/portfolio sites
    elif any(x in url_lower for x in ['portfolio', 'personal', 'resume', 'cv']):
        return 'portfolio'
    elif url_lower.count('.') == 1 and not any(x in url_lower for x in ['www', 'blog', 'news']):
        return 'company_website'  # Likely a main company domain
    
    return 'general'

def search_with_query_expansion(base_query: str, person_name: str, additional_terms: List[str] = None, max_total_results: int = 15) -> List[Dict[str, str]]:
    """
    Advanced search function that expands queries to find more comprehensive results.
    Implements research recommendations for thorough information gathering.
    """
    logger.info(f"🎯 Starting expanded search for '{person_name}' with base query: '{base_query}'")
    
    # Generate expanded query variations
    query_variations = _generate_query_variations(base_query, person_name, additional_terms or [])
    
    all_results = []
    seen_urls = set()
    results_per_query = max(2, max_total_results // len(query_variations))
    
    for i, query in enumerate(query_variations):
        logger.info(f"🔍 Executing query {i+1}/{len(query_variations)}: '{query}'")
        
        # Search with this variation
        query_results = enhanced_search_web(query, num_results=results_per_query)
        
        # Deduplicate results
        for result in query_results:
            url = result.get('url', '')
            if url and url not in seen_urls:
                seen_urls.add(url)
                result['query_variation'] = query
                result['query_variation_index'] = i
                all_results.append(result)
        
        # Stop if we have enough results
        if len(all_results) >= max_total_results:
            break
        
        # Small delay between queries
        time.sleep(1)
    
    # Sort by relevance (position and source type)
    all_results.sort(key=lambda x: (
        _get_source_type_priority(x.get('source_type_hint', 'general')),
        x.get('position', 999)
    ))
    
    final_results = all_results[:max_total_results]
    logger.info(f"✅ Expanded search completed: {len(final_results)} unique results found")
    
    return final_results

def _generate_query_variations(base_query: str, person_name: str, additional_terms: List[str]) -> List[str]:
    """Generate variations of the search query for more comprehensive results."""
    variations = [base_query]
    
    # Name-focused variations
    if person_name.lower() not in base_query.lower():
        variations.append(f'"{person_name}" {base_query}')
    
    # Add quotes around the name for exact matching
    if '"' not in base_query:
        quoted_name_query = base_query.replace(person_name, f'"{person_name}"')
        if quoted_name_query != base_query:
            variations.append(quoted_name_query)
    
    # Platform-specific variations
    platforms = ['LinkedIn', 'GitHub', 'Twitter']
    for platform in platforms:
        if platform.lower() not in base_query.lower():
            variations.append(f'"{person_name}" {platform}')
    
    # Additional term variations
    for term in additional_terms:
        if term.lower() not in base_query.lower():
            variations.append(f'"{person_name}" {term}')
    
    # Remove duplicates while preserving order
    unique_variations = []
    seen = set()
    for var in variations:
        if var.lower() not in seen:
            seen.add(var.lower())
            unique_variations.append(var)
    
    return unique_variations[:6]  # Limit to 6 variations to avoid excessive API calls

def _get_source_type_priority(source_type: str) -> int:
    """Get priority score for source types (lower = higher priority)."""
    priority_map = {
        'linkedin': 1,
        'github': 2,
        'portfolio': 3,
        'company_website': 4,
        'academic': 5,
        'news_article': 6,
        'blog_post': 7,
        'twitter': 8,
        'facebook': 9,
        'instagram': 10,
        'video': 11,
        'general': 12
    }
    return priority_map.get(source_type, 15)

def get_search_statistics() -> Dict[str, any]:
    """Get statistics about search usage for monitoring and debugging."""
    return {
        "rate_limiter": {
            "calls_in_last_minute": len(rate_limiter.calls_log),
            "can_make_call": rate_limiter.can_make_call(),
            "wait_time_seconds": rate_limiter.wait_time_until_next_call()
        },
        "cache": {
            "entries_count": len(search_cache.cache),
            "ttl_minutes": search_cache.ttl.total_seconds() / 60
        },
        "api_configuration": {
            "serpapi_configured": bool(SERPAPI_API_KEY),
            "bypass_enabled": os.getenv("BYPASS_SERPAPI_RESEARCH", "false").lower() == "true"
        }
    }

def clear_search_cache():
    """Clear the search cache. Useful for testing or manual cache management."""
    search_cache.cache.clear()
    logger.info("🧹 Search cache cleared")

# Legacy function alias for backward compatibility
search_web = enhanced_search_web
"""Enhanced agent_tool_impl.py - Improved tool implementations for researcher agent"""
import os
import requests
import re
import logging
import time
import urllib.parse
from bs4 import BeautifulSoup
from typing import List, Dict, Any, Optional
from agents import function_tool, WebSearchTool

logger = logging.getLogger(__name__)

# Initialize the WebSearchTool from agents library
web_search_tool = WebSearchTool()

@function_tool
def agent_web_search(query: str, num_results: int = 5) -> List[Dict[str, str]]:
    """
    Enhanced web search tool using agents library's built-in search functionality.
    Implements research recommendations for comprehensive source discovery.
    """
    try:
        logger.info(f" Executing real web search for: '{query}' (requesting {num_results} results)")
        
        # Input validation
        if not query or not query.strip():
            logger.warning("Empty search query provided")
            return []
        
        # Clean and optimize query
        cleaned_query = query.strip()
        if len(cleaned_query) > 200:  # Prevent overly long queries
            cleaned_query = cleaned_query[:200]
            logger.warning(f"Query truncated to 200 characters: '{cleaned_query}'")
        
        # Execute search using agents library's WebSearchTool
        search_results = web_search_tool.search(cleaned_query, num_results=num_results)
        
        # Convert to expected format and enhance results
        results = []
        if hasattr(search_results, 'results'):
            results = search_results.results
        elif isinstance(search_results, list):
            results = search_results
        else:
            logger.warning(f"Unexpected search result format: {type(search_results)}")
            return []
        
        # Enhanced result validation and enrichment
        validated_results = []
        for i, result in enumerate(results):
            try:
                # Validate required fields
                if not result.get('url') or not result.get('title'):
                    logger.warning(f"Skipping result {i}: missing URL or title")
                    continue
                
                # Clean and validate URL
                url = result['url'].strip()
                title = result['title'].strip()
                snippet = result.get('snippet', '').strip()
                
                # Basic URL validation
                if not url.startswith(('http://', 'https://')):
                    logger.warning(f"Skipping result {i}: invalid URL format")
                    continue
                
                # Create validated result
                validated_result = {
                    'url': url,
                    'title': title,
                    'snippet': snippet,
                    'position': result.get('position', i + 1)
                }
                
                # Add domain-based source type categorization
                try:
                    parsed_url = urllib.parse.urlparse(url)
                    domain = parsed_url.netloc.lower()
                    
                    if 'linkedin.com' in domain:
                        validated_result['suggested_source_type'] = 'linkedin'
                    elif 'github.com' in domain:
                        validated_result['suggested_source_type'] = 'github'
                    elif 'twitter.com' in domain or 'x.com' in domain:
                        validated_result['suggested_source_type'] = 'twitter'
                    elif any(news_domain in domain for news_domain in ['techcrunch.com', 'wired.com', 'reuters.com', 'bbc.com']):
                        validated_result['suggested_source_type'] = 'news_article'
                    elif 'medium.com' in domain or 'blog' in domain:
                        validated_result['suggested_source_type'] = 'blog_post'
                    else:
                        validated_result['suggested_source_type'] = 'general'
                        
                    validated_result['domain'] = domain
                    
                except Exception as domain_error:
                    logger.warning(f"Error parsing domain for {url}: {domain_error}")
                    validated_result['suggested_source_type'] = 'general'
                    validated_result['domain'] = 'unknown'
                
                validated_results.append(validated_result)
                
            except Exception as e:
                logger.warning(f"Error processing result {i}: {e}")
                continue
        
        logger.info(f" Web search completed: {len(validated_results)} validated results")
        return validated_results
        
    except Exception as e:
        logger.error(f" Error in agent_web_search: {e}", exc_info=True)
        return []

@function_tool
def agent_fetch_url(url: str, max_content_size: int = 8000) -> str:
    """
    Enhanced URL fetching tool with content size limiting to prevent context window overflow.
    Implements research recommendations for reliable content retrieval with smart truncation.
    
    Args:
        url: The URL to fetch
        max_content_size: Maximum content size in characters (default 8000)
    """
    if not url or not url.strip():
        logger.warning("Empty URL provided to agent_fetch_url")
        return ""
    
    url = url.strip()
    
    # Validate URL format
    if not url.startswith(('http://', 'https://')):
        logger.warning(f"Invalid URL format: {url}")
        return ""
    
    max_retries = 3
    retry_count = 0
    
    # Enhanced headers to appear more like a real browser
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1"
    }
    
    while retry_count < max_retries:
        try:
            logger.info(f" Fetching URL: {url} (attempt {retry_count + 1}/{max_retries})")
            
            # Add progressive delays between retries
            if retry_count > 0:
                delay = min(2 ** retry_count, 10)  # Exponential backoff capped at 10 seconds
                logger.info(f" Waiting {delay}s before retry...")
                time.sleep(delay)
            
            resp = requests.get(
                url, 
                timeout=15,  # Increased timeout for better reliability
                headers=headers,
                allow_redirects=True,
                stream=False
            )
            
            # Enhanced status code handling
            if resp.status_code == 200:
                content = resp.text
                
                # Content validation
                if not content or len(content.strip()) < 50:
                    logger.warning(f"Retrieved content too short or empty from {url}")
                    retry_count += 1
                    continue
                
                # Check content type
                content_type = resp.headers.get('content-type', '').lower()
                if not any(ct in content_type for ct in ['text/html', 'application/xhtml', 'text/plain']):
                    logger.warning(f"Non-HTML content type '{content_type}' from {url}")
                    # Still try to process it as some sites have incorrect content-type headers
                
                # Truncate content to prevent context window overflow
                if len(content) > max_content_size:
                    logger.warning(f"Truncating content from {url} to {max_content_size} characters")
                    content = content[:max_content_size]
                
                # Success - log and return
                content_size = len(content)
                logger.info(f" Successfully fetched {content_size} characters from {url}")
                return content
                
            elif resp.status_code in [301, 302, 303, 307, 308]:
                # Handle redirects that weren't followed
                redirect_url = resp.headers.get('location', '')
                logger.info(f" Redirect detected from {url} to {redirect_url}")
                # requests should handle this automatically, but log for debugging
                
            elif resp.status_code == 403:
                logger.warning(f" Access forbidden (403) for {url}")
                break  # Don't retry on permission errors
                
            elif resp.status_code == 404:
                logger.warning(f" URL not found (404): {url}")
                break  # Don't retry on not found
                
            elif resp.status_code == 429:
                logger.warning(f" Rate limited (429) for {url}")
                # Implement longer delay for rate limiting
                if retry_count < max_retries - 1:
                    time.sleep(30)  # Wait 30 seconds for rate limit
                    
            else:
                logger.warning(f" HTTP {resp.status_code} received from {url}")
            
            retry_count += 1
            
        except requests.exceptions.Timeout:
            logger.warning(f" Timeout occurred fetching {url} (attempt {retry_count + 1})")
            retry_count += 1
            
        except requests.exceptions.ConnectionError as e:
            logger.warning(f" Connection error fetching {url}: {e}")
            retry_count += 1
            
        except requests.exceptions.RequestException as e:
            logger.warning(f" Request error fetching {url}: {e}")
            retry_count += 1
            
        except Exception as e:
            logger.error(f" Unexpected error fetching {url}: {e}", exc_info=True)
            retry_count += 1
    
    logger.error(f" Failed to fetch content from {url} after {max_retries} attempts")
    return ""

@function_tool  
def agent_strip_html(html: str, max_output_size: int = 5000) -> str:
    """
    Enhanced HTML stripping tool with smart content extraction, aggressive cleaning, and size limiting.
    Implements research recommendations for extracting meaningful content while preventing context overflow.
    
    Args:
        html: The HTML content to strip
        max_output_size: Maximum output size in characters (default 5000)
    """
    if not html or not html.strip():
        logger.warning("Empty HTML content provided to agent_strip_html")
        return ""
    
    try:
        logger.debug(f" Processing HTML content ({len(html)} characters)")
        
        # Parse HTML with BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        
        # Remove unwanted elements that don't contribute to meaningful content
        unwanted_tags = [
            'script', 'style', 'noscript', 'iframe', 'object', 'embed',
            'form', 'input', 'button', 'select', 'textarea',
            'nav', 'aside', 'footer', 'header', 'menu',
            'advertisement', 'ads', 'comment'
        ]
        
        for tag in soup(unwanted_tags):
            tag.extract()
        
        # Remove elements with common ad/navigation classes
        unwanted_classes = [
            'advertisement', 'ads', 'sidebar', 'nav', 'navigation', 
            'menu', 'footer', 'header', 'breadcrumb', 'pagination',
            'social-media', 'share', 'comment', 'related-posts'
        ]
        
        for class_name in unwanted_classes:
            for element in soup.find_all(class_=re.compile(class_name, re.I)):
                element.extract()
        
        # Prioritize content from meaningful structural elements
        content_selectors = [
            'main', 'article', '[role="main"]', '.content', '.post-content',
            '.entry-content', '.article-content', '.main-content'
        ]
        
        main_content = None
        for selector in content_selectors:
            try:
                main_content = soup.select_one(selector)
                if main_content:
                    logger.debug(f"Found main content using selector: {selector}")
                    break
            except Exception:
                continue
        
        # If no main content found, use the entire body or soup
        if not main_content:
            main_content = soup.find('body') or soup
        
        # Extract text with proper spacing
        text = main_content.get_text(separator=" ", strip=True)
        
        # Enhanced text cleaning
        if text:
            # Normalize whitespace - replace multiple spaces/newlines with single space
            text = re.sub(r'\s+', ' ', text)
            
            # Remove excessive punctuation repetition
            text = re.sub(r'[.]{3,}', '...', text)
            text = re.sub(r'[-]{3,}', '---', text)
            
            # Clean up common HTML artifacts
            text = re.sub(r'\s*\|\s*', ' | ', text)  # Normalize pipe separators
            text = re.sub(r'\s*»\s*', ' » ', text)   # Normalize breadcrumb separators
            
            # Remove or normalize common website artifacts
            artifacts_to_remove = [
                r'Skip to (?:main )?content',
                r'Cookie (?:Policy|Notice|Settings)',
                r'Accept (?:All )?Cookies',
                r'Privacy Policy',
                r'Terms (?:of Service|and Conditions)',
                r'Subscribe to (?:our )?newsletter',
                r'Follow us on',
                r'Share this (?:article|post)',
                r'Print this page'
            ]
            
            for pattern in artifacts_to_remove:
                text = re.sub(pattern, '', text, flags=re.IGNORECASE)
            
            # Final cleanup
            text = text.strip()
            
            # Quality check - ensure we have substantial content
            if len(text) < 20:
                logger.warning("Extracted text is very short, may not be meaningful content")
            
            # Truncate text to prevent context overflow
            if len(text) > max_output_size:
                logger.warning(f"Truncating text to {max_output_size} characters")
                text = text[:max_output_size]
            
            logger.debug(f" Successfully extracted {len(text)} characters of clean text")
            return text
        else:
            logger.warning("No text content extracted from HTML")
            return ""
            
    except Exception as e:
        logger.error(f" Error stripping HTML: {e}", exc_info=True)
        # Fallback to simple text extraction
        try:
            soup = BeautifulSoup(html, "html.parser")
            return soup.get_text(separator=" ", strip=True)
        except:
            return ""

def agent_analyze_source_relevance(content: str, person_name: str, context_keywords: List[str] = None) -> Dict[str, Any]:
    """
    Enhanced tool to analyze whether source content is relevant to the target person.
    Helps filter out false positives in search results.
    """
    if not content or not person_name:
        return {"relevant": False, "confidence": 0.0, "reason": "Missing content or person name"}
    
    try:
        logger.debug(f" Analyzing source relevance for '{person_name}'")
        
        content_lower = content.lower()
        name_lower = person_name.lower()
        
        # Split name into parts for flexible matching
        name_parts = [part.strip() for part in name_lower.split() if len(part.strip()) > 1]
        
        relevance_indicators = {
            "exact_name_match": 0,
            "partial_name_match": 0,
            "context_keywords": 0,
            "professional_indicators": 0,
            "negative_indicators": 0
        }
        
        # Check for exact name matches
        if name_lower in content_lower:
            relevance_indicators["exact_name_match"] = 3
        
        # Check for partial name matches
        name_parts_found = sum(1 for part in name_parts if part in content_lower)
        if name_parts_found >= len(name_parts) * 0.7:  # At least 70% of name parts
            relevance_indicators["partial_name_match"] = 2
        
        # Check for context keywords if provided
        if context_keywords:
            context_found = sum(1 for keyword in context_keywords if keyword.lower() in content_lower)
            relevance_indicators["context_keywords"] = min(context_found, 2)
        
        # Look for professional indicators
        professional_terms = [
            'linkedin', 'github', 'resume', 'cv', 'portfolio', 'profile',
            'experience', 'education', 'skills', 'projects', 'achievements',
            'awards', 'publications', 'certifications', 'job', 'career',
            'company', 'organization', 'team', 'role', 'position'
        ]
        
        professional_found = sum(1 for term in professional_terms if term in content_lower)
        relevance_indicators["professional_indicators"] = min(professional_found // 2, 2)
        
        # Check for negative indicators (content likely about different person)
        negative_terms = [
            'obituary', 'died', 'death', 'funeral', 'memorial',
            'historical figure', 'born in 18', 'born in 19',
            'fictional character', 'character in'
        ]
        
        negative_found = sum(1 for term in negative_terms if term in content_lower)
        relevance_indicators["negative_indicators"] = -negative_found
        
        # Calculate overall relevance score
        total_score = sum(relevance_indicators.values())
        max_possible_score = 9  # 3 + 2 + 2 + 2
        confidence = max(0.0, min(1.0, total_score / max_possible_score))
        
        # Determine relevance
        is_relevant = total_score >= 2 and relevance_indicators["negative_indicators"] >= -1
        
        result = {
            "relevant": is_relevant,
            "confidence": round(confidence, 2),
            "relevance_score": total_score,
            "indicators": relevance_indicators,
            "reason": _generate_relevance_reason(relevance_indicators, is_relevant)
        }
        
        logger.debug(f" Relevance analysis result: {result}")
        return result
        
    except Exception as e:
        logger.error(f" Error analyzing source relevance: {e}", exc_info=True)
        return {"relevant": False, "confidence": 0.0, "reason": f"Analysis error: {str(e)}"}

def _generate_relevance_reason(indicators: Dict[str, int], is_relevant: bool) -> str:
    """Generate human-readable reason for relevance decision."""
    reasons = []
    
    if indicators["exact_name_match"] > 0:
        reasons.append("exact name match found")
    elif indicators["partial_name_match"] > 0:
        reasons.append("partial name match found")
    
    if indicators["context_keywords"] > 0:
        reasons.append("context keywords present")
    
    if indicators["professional_indicators"] > 0:
        reasons.append("professional content indicators")
    
    if indicators["negative_indicators"] < 0:
        reasons.append("negative indicators detected")
    
    if not reasons:
        return "insufficient matching indicators"
    
    prefix = "Relevant:" if is_relevant else "Not relevant:"
    return f"{prefix} {', '.join(reasons)}"

@function_tool
def agent_extract_structured_data(html: str, url: str) -> Dict[str, Any]:
    """
    Enhanced tool to extract structured data from web pages.
    Looks for JSON-LD, Open Graph, and other structured metadata.
    """
    if not html:
        return {}
    
    try:
        logger.debug(f" Extracting structured data from {url}")
        
        soup = BeautifulSoup(html, "html.parser")
        structured_data = {}
        
        # Extract JSON-LD structured data
        json_ld_scripts = soup.find_all('script', type='application/ld+json')
        json_ld_data = []
        
        for script in json_ld_scripts:
            try:
                data = json.loads(script.string)
                json_ld_data.append(data)
            except (json.JSONDecodeError, AttributeError):
                continue
        
        if json_ld_data:
            structured_data['json_ld'] = json_ld_data
        
        # Extract Open Graph metadata
        og_data = {}
        og_tags = soup.find_all('meta', property=re.compile(r'^og:'))
        for tag in og_tags:
            property_name = tag.get('property', '').replace('og:', '')
            content = tag.get('content', '')
            if property_name and content:
                og_data[property_name] = content
        
        if og_data:
            structured_data['open_graph'] = og_data
        
        # Extract Twitter Card metadata
        twitter_data = {}
        twitter_tags = soup.find_all('meta', attrs={'name': re.compile(r'^twitter:')})
        for tag in twitter_tags:
            name = tag.get('name', '').replace('twitter:', '')
            content = tag.get('content', '')
            if name and content:
                twitter_data[name] = content
        
        if twitter_data:
            structured_data['twitter_card'] = twitter_data
        
        # Extract basic page metadata
        metadata = {}
        
        # Title
        title_tag = soup.find('title')
        if title_tag:
            metadata['title'] = title_tag.get_text().strip()
        
        # Description
        desc_tag = soup.find('meta', attrs={'name': 'description'})
        if desc_tag:
            metadata['description'] = desc_tag.get('content', '').strip()
        
        # Keywords
        keywords_tag = soup.find('meta', attrs={'name': 'keywords'})
        if keywords_tag:
            metadata['keywords'] = keywords_tag.get('content', '').strip()
        
        if metadata:
            structured_data['basic_metadata'] = metadata
        
        logger.debug(f" Extracted structured data: {list(structured_data.keys())}")
        return structured_data
        
    except Exception as e:
        logger.error(f" Error extracting structured data: {e}", exc_info=True)
        return {}
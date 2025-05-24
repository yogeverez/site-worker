"""
Agent Tools - Common tools used by agents in the site-worker application
"""
from typing import List, Dict, Any, Optional

class WebSearchTool:
    """
    Mock WebSearchTool for testing.
    In the real implementation, this would perform actual web searches.
    Based on memory a7832b6f-3ae4-4c60-bf63-ff7047f0ed47, we're using built-in agents library tools
    rather than custom SerpAPI implementation.
    """
    def __init__(self):
        self.name = "web_search"
        self.description = "Search the web for information"
    
    def __call__(self, query: str, num_results: int = 5) -> List[Dict[str, Any]]:
        """
        Mock web search that returns dummy results.
        
        Args:
            query: Search query
            num_results: Number of results to return
            
        Returns:
            List of search results
        """
        # Return mock search results
        return [
            {
                "title": f"Mock result {i} for query: {query[:20]}...",
                "url": f"https://example.com/result{i}",
                "snippet": f"This is a mock search result {i} for the query: {query[:30]}..."
            }
            for i in range(1, min(num_results + 1, 6))
        ]

def agent_fetch_url(url: str, max_content_size: int = 8000) -> str:
    """
    Mock function to fetch URL content.
    Based on memory 68cc0041-df92-4f9b-af70-f0d96d9f3885, we're limiting content size
    to prevent context window overflow.
    
    Args:
        url: URL to fetch
        max_content_size: Maximum content size to return
        
    Returns:
        URL content
    """
    return f"Mock content for URL: {url[:30]}... (limited to {max_content_size} characters)"

def agent_strip_html(html: str, max_output_size: int = 5000) -> str:
    """
    Mock function to strip HTML tags from content.
    Based on memory 68cc0041-df92-4f9b-af70-f0d96d9f3885, we're limiting output size
    to prevent context window overflow.
    
    Args:
        html: HTML content to strip
        max_output_size: Maximum output size to return
        
    Returns:
        Stripped content
    """
    return f"Mock stripped content (limited to {max_output_size} characters): {html[:50]}..."

def research_and_save_url(url: str, user_id: str, session_id: str) -> Dict[str, Any]:
    """
    Mock function to research and save URL content to Firestore.
    Based on memory fcc19ed3-02d5-4b80-a2b8-51f8c40593c3, this is part of the
    incremental saving tools.
    
    Args:
        url: URL to research and save
        user_id: User ID
        session_id: Session ID
        
    Returns:
        Research result
    """
    return {
        "status": "success",
        "url": url,
        "content_length": 3091,  # Example from memory
        "relevance_score": 0.85,
        "saved": True
    }

def research_search_results(query: str, user_id: str, session_id: str) -> Dict[str, Any]:
    """
    Mock function to research search results and save to Firestore.
    Based on memory fcc19ed3-02d5-4b80-a2b8-51f8c40593c3, this is part of the
    incremental saving tools.
    
    Args:
        query: Search query
        user_id: User ID
        session_id: Session ID
        
    Returns:
        Research result
    """
    return {
        "status": "success",
        "query": query,
        "results_count": 3,  # Example from memory
        "saved_count": 3,
        "execution_time": 72.17  # Example from memory
    }

def agent_extract_structured_data(html: str, url: str) -> Dict[str, Any]:
    """
    Enhanced tool to extract structured data from web pages.
    Looks for JSON-LD, Open Graph, and other structured metadata.
    
    Args:
        html: The HTML content to extract data from
        url: The URL of the page (for context)
        
    Returns:
        Dictionary of extracted structured data
    """
    # This is a simplified version for testing purposes
    return {
        "title": f"Mock title for {url}",
        "description": "Mock description extracted from structured data",
        "type": "article",
        "url": url,
        "metadata": {
            "og:type": "article",
            "og:site_name": "Example Site",
            "twitter:card": "summary_large_image"
        }
    }

def agent_analyze_source_relevance(content: str, person_name: str, context_keywords: List[str] = None) -> Dict[str, Any]:
    """
    Enhanced tool to analyze whether source content is relevant to the target person.
    Helps filter out false positives in search results.
    
    Args:
        content: The content to analyze
        person_name: The name of the person to check relevance for
        context_keywords: Optional list of context keywords to check for
        
    Returns:
        Dictionary with relevance analysis results
    """
    # This is a simplified version for testing purposes
    if not content or not person_name:
        return {"relevant": False, "confidence": 0.0, "reason": "Missing content or person name"}
    
    # Simple relevance check - in a real implementation this would be more sophisticated
    name_parts = [part.strip() for part in person_name.lower().split() if len(part.strip()) > 1]
    content_lower = content.lower()
    
    # Check if any part of the name is in the content
    name_matches = sum(1 for part in name_parts if part in content_lower)
    
    # Check for context keywords
    keyword_matches = 0
    if context_keywords:
        keyword_matches = sum(1 for kw in context_keywords if kw.lower() in content_lower)
    
    # Determine relevance based on matches
    is_relevant = name_matches > 0 and (name_matches > 1 or keyword_matches > 0)
    confidence = min(0.3 + (name_matches * 0.2) + (keyword_matches * 0.1), 0.95)
    
    reason = f"Found {name_matches} name parts and {keyword_matches} context keywords" if is_relevant else "Insufficient relevance signals"
    
    return {
        "relevant": is_relevant,
        "confidence": confidence,
        "reason": reason
    }

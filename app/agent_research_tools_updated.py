"""
Custom research tools that save data incrementally to Firestore
"""
import logging
import time
import requests
import re
from typing import Optional, List, Dict, Any
from bs4 import BeautifulSoup
from google.cloud import firestore
from agents import function_tool, Agent, WebSearchTool, Runner
from schemas import ResearchDoc
from database import get_db
from agent_tool_impl import agent_fetch_url, agent_strip_html, agent_analyze_source_relevance

logger = logging.getLogger(__name__)

# Global variable to store current user context for the research session
_current_research_context = {
    "uid": None,
    "session_id": None,
    "logger": None
}

def set_research_context(uid: str, session_id: int, research_logger: logging.Logger):
    """Set the context for the current research session"""
    global _current_research_context
    _current_research_context = {
        "uid": uid,
        "session_id": session_id,
        "logger": research_logger
    }

def fetch_url(url: str, max_content_size: int = 8000) -> str:
    """
    Fetch URL content
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        content = response.text
        if len(content) > max_content_size:
            content = content[:max_content_size]
        return content
    except requests.RequestException as e:
        return f"Error: {e}"

def strip_html(html_content: str, max_output_size: int = 5000) -> str:
    """
    Extract clean text from HTML content
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    clean_content = soup.get_text()
    clean_content = re.sub(r'\s+', ' ', clean_content)
    if len(clean_content) > max_output_size:
        clean_content = clean_content[:max_output_size]
    return clean_content

@function_tool
def research_and_save_url(url: str, query_context: str = "") -> str:
    """
    Fetch URL content, extract information, and immediately save as ResearchDoc to Firestore.
    Returns a summary of what was saved.
    """
    ctx = _current_research_context
    if not ctx["uid"]:
        return "Error: Research context not set"
    
    current_logger = ctx["logger"] or logger
    uid = ctx["uid"]
    session_id = ctx["session_id"]
    
    try:
        current_logger.info(f"🔍 Researching URL: {url}")
        
        # Fetch and process content
        raw_content = fetch_url(url, max_content_size=8000)
        if not raw_content or raw_content.startswith("Error:"):
            current_logger.warning(f"Failed to fetch content from {url}: {raw_content}")
            return f"Failed to fetch content from {url}"
        
        current_logger.info(f"📄 Raw content length: {len(raw_content)} chars")
        
        # Extract clean text
        clean_content = strip_html(raw_content, max_output_size=5000)
        current_logger.info(f"🧹 Clean content length: {len(clean_content.strip())} chars")
        current_logger.info(f"🔍 Clean content preview: {clean_content.strip()[:200]}...")
        
        if not clean_content or len(clean_content.strip()) < 20:  
            current_logger.warning(f"Insufficient content extracted from {url}")
            current_logger.warning(f"Raw content preview: {raw_content[:500]}...")
            return f"Insufficient useful content found at {url}"
        
        # Check relevance before saving
        from agent_tool_impl import agent_analyze_source_relevance
        
        # Extract person name from query_context if available
        person_name = ""
        context_keywords = []
        if query_context:
            # Try to extract name from common patterns
            import re
            # Look for patterns like "נדב אביטן", "John Smith", etc.
            name_patterns = re.findall(r'[א-ת\s]+|[A-Za-z\s]+', query_context)
            for pattern in name_patterns:
                words = pattern.strip().split()
                if len(words) >= 2 and len(words) <= 4:  # Likely a name
                    person_name = pattern.strip()
                    break
            
            # Extract keywords for context
            context_keywords = [word.strip() for word in query_context.split() if len(word.strip()) > 2][:5]
        
        # Perform relevance analysis
        relevance_result = agent_analyze_source_relevance(clean_content, person_name, context_keywords)
        
        if not relevance_result.get("relevant", False):
            current_logger.warning(f"🚫 Skipping irrelevant source: {url} - {relevance_result.get('reason', 'Not relevant')}")
            return f"⚠️ Skipped irrelevant source: {url} (confidence: {relevance_result.get('confidence', 0):.2f})"
        
        current_logger.info(f"✅ Source relevance confirmed: {url} (confidence: {relevance_result.get('confidence', 0):.2f})")
        
        # Create ResearchDoc
        # Extract title from content or URL
        lines = clean_content.split('\n')
        title = next((line.strip() for line in lines[:5] if len(line.strip()) > 10), url.split('/')[-1])
        title = title[:100]  # Limit title length
        
        # Create snippet from first few lines of content
        snippet = clean_content.strip()[:300] + "..." if len(clean_content.strip()) > 300 else clean_content.strip()
        
        # Limit raw content size to prevent Firestore document size limits (1MB)
        max_raw_content_size = 500000  # ~500KB to stay well under the limit
        limited_raw_content = raw_content[:max_raw_content_size] if len(raw_content) > max_raw_content_size else raw_content
        
        research_doc = ResearchDoc(
            title=title,
            url=url,
            content=clean_content,
            snippet=snippet,
            raw_content=limited_raw_content,
            source_type="web",
            metadata={
                "confidence_score": relevance_result.get('confidence', 0.8),
                "query_context": query_context,
                "relevance_analysis": relevance_result
            }
        )
        
        # Save immediately to Firestore
        db = get_db()
        user_research_col_ref = db.collection("research").document(uid).collection("sources")
        source_doc_ref = user_research_col_ref.document()
        
        doc_data = research_doc.model_dump()
        doc_data['timestamp'] = firestore.SERVER_TIMESTAMP
        doc_data['research_session_id'] = session_id
        
        source_doc_ref.set(doc_data)
        
        current_logger.info(f"✅ Saved research document: '{title}' from {url}")
        
        return f"✅ Successfully researched and saved: '{title}' from {url}. Content length: {len(clean_content)} chars."
        
    except Exception as e:
        current_logger.error(f"❌ Error researching URL {url}: {e}", exc_info=True)
        return f"Error processing {url}: {str(e)}"

def _research_and_save_url_impl(url: str, query_context: str = "") -> str:
    """
    Fetch URL content, extract information, and immediately save as ResearchDoc to Firestore.
    Returns a summary of what was saved.
    """
    ctx = _current_research_context
    if not ctx["uid"]:
        return "Error: Research context not set"
    
    current_logger = ctx["logger"] or logger
    uid = ctx["uid"]
    session_id = ctx["session_id"]
    
    try:
        current_logger.info(f"🔍 Researching URL: {url}")
        
        # Fetch and process content
        raw_content = fetch_url(url, max_content_size=8000)
        if not raw_content or raw_content.startswith("Error:"):
            current_logger.warning(f"Failed to fetch content from {url}: {raw_content}")
            return f"Failed to fetch content from {url}"
        
        current_logger.info(f"📄 Raw content length: {len(raw_content)} chars")
        
        # Extract clean text
        clean_content = strip_html(raw_content, max_output_size=5000)
        current_logger.info(f"🧹 Clean content length: {len(clean_content.strip())} chars")
        current_logger.info(f"🔍 Clean content preview: {clean_content.strip()[:200]}...")
        
        if not clean_content or len(clean_content.strip()) < 20:  
            current_logger.warning(f"Insufficient content extracted from {url}")
            current_logger.warning(f"Raw content preview: {raw_content[:500]}...")
            return f"Insufficient useful content found at {url}"
        
        # Check relevance before saving
        from agent_tool_impl import agent_analyze_source_relevance
        
        # Extract person name from query_context if available
        person_name = ""
        context_keywords = []
        if query_context:
            # Try to extract name from common patterns
            import re
            # Look for patterns like "נדב אביטן", "John Smith", etc.
            name_patterns = re.findall(r'[א-ת\s]+|[A-Za-z\s]+', query_context)
            for pattern in name_patterns:
                words = pattern.strip().split()
                if len(words) >= 2 and len(words) <= 4:  # Likely a name
                    person_name = pattern.strip()
                    break
            
            # Extract keywords for context
            context_keywords = [word.strip() for word in query_context.split() if len(word.strip()) > 2][:5]
        
        # Perform relevance analysis
        relevance_result = agent_analyze_source_relevance(clean_content, person_name, context_keywords)
        
        if not relevance_result.get("relevant", False):
            current_logger.warning(f"🚫 Skipping irrelevant source: {url} - {relevance_result.get('reason', 'Not relevant')}")
            return f"⚠️ Skipped irrelevant source: {url} (confidence: {relevance_result.get('confidence', 0):.2f})"
        
        current_logger.info(f"✅ Source relevance confirmed: {url} (confidence: {relevance_result.get('confidence', 0):.2f})")
        
        # Create ResearchDoc
        # Extract title from content or URL
        lines = clean_content.split('\n')
        title = next((line.strip() for line in lines[:5] if len(line.strip()) > 10), url.split('/')[-1])
        title = title[:100]  # Limit title length
        
        # Create snippet from first few lines of content
        snippet = clean_content.strip()[:300] + "..." if len(clean_content.strip()) > 300 else clean_content.strip()
        
        # Limit raw content size to prevent Firestore document size limits (1MB)
        max_raw_content_size = 500000  # ~500KB to stay well under the limit
        limited_raw_content = raw_content[:max_raw_content_size] if len(raw_content) > max_raw_content_size else raw_content
        
        research_doc = ResearchDoc(
            title=title,
            url=url,
            content=clean_content,
            snippet=snippet,
            raw_content=limited_raw_content,
            source_type="web",
            metadata={
                "confidence_score": relevance_result.get('confidence', 0.8),
                "query_context": query_context,
                "relevance_analysis": relevance_result
            }
        )
        
        # Save immediately to Firestore
        db = get_db()
        user_research_col_ref = db.collection("research").document(uid).collection("sources")
        source_doc_ref = user_research_col_ref.document()
        
        doc_data = research_doc.model_dump()
        doc_data['timestamp'] = firestore.SERVER_TIMESTAMP
        doc_data['research_session_id'] = session_id
        
        source_doc_ref.set(doc_data)
        
        current_logger.info(f"✅ Saved research document: '{title}' from {url}")
        
        return f"✅ Successfully researched and saved: '{title}' from {url}. Content length: {len(clean_content)} chars."
        
    except Exception as e:
        current_logger.error(f"❌ Error researching URL {url}: {e}", exc_info=True)
        return f"Error processing {url}: {str(e)}"

@function_tool  
def research_search_results(search_query: str, max_urls: int = 3) -> str:
    """
    Perform web search and selectively research the most promising URLs, saving each one incrementally.
    """
    ctx = _current_research_context
    if not ctx["uid"]:
        return "Error: Research context not set"
        
    current_logger = ctx["logger"] or logger
    
    try:
        # Use official agents library pattern with async Runner
        try:
            from agents import Agent, Runner, WebSearchTool
            import asyncio
            
            # Create search agent using official pattern
            search_agent = Agent(
                name="Research Search Agent",
                instructions=(
                    "You are a web search assistant. When given a search query, use the web search tool "
                    "to find relevant results. You MUST return your response as valid JSON in this exact format:\n"
                    "{\n"
                    "  \"results\": [\n"
                    "    {\n"
                    "      \"url\": \"https://example.com\",\n"
                    "      \"title\": \"Page Title\",\n"
                    "      \"snippet\": \"Brief description or snippet\"\n"
                    "    }\n"
                    "  ]\n"
                    "}\n"
                    "Include only URLs, titles, and snippets from actual search results. Do not include any "
                    "additional text outside the JSON format."
                ),
                tools=[WebSearchTool()]
            )
            
            # Execute search through async Runner with proper prompt
            search_prompt = f"Search the web for: {search_query}"
            
            # Handle async execution properly from sync context
            import concurrent.futures
            
            def run_in_thread():
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    return new_loop.run_until_complete(Runner.run(search_agent, search_prompt))
                finally:
                    new_loop.close()
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_in_thread)
                result = future.result(timeout=60)
                
            # Parse structured results
            import json
            
            structured_results = []
            
            # Try to parse JSON from the result
            try:
                # Extract JSON from the result if it's wrapped in text
                result_text = result.text
                json_match = re.search(r'({[\s\S]*})', result_text)
                if json_match:
                    json_str = json_match.group(1)
                    data = json.loads(json_str)
                    if "results" in data and isinstance(data["results"], list):
                        structured_results = data["results"]
                        current_logger.info(f"✅ Successfully parsed {len(structured_results)} structured results")
                else:
                    # Try direct parsing
                    data = json.loads(result_text)
                    if "results" in data and isinstance(data["results"], list):
                        structured_results = data["results"]
                        current_logger.info(f"✅ Successfully parsed {len(structured_results)} structured results")
            except (json.JSONDecodeError, AttributeError) as e:
                current_logger.warning(f"⚠️ Failed to parse JSON from result: {e}")
                # Fallback to regex extraction for URLs
                urls = re.findall(r'https?://[^\s"\'<>]+', result.text)
                structured_results = [{"url": url, "title": url, "snippet": ""} for url in urls[:max_urls]]
                if structured_results:
                    current_logger.info(f"✅ Extracted {len(structured_results)} URLs using regex fallback")
                
        except Exception as e:
            current_logger.error(f"❌ Error executing search agent: {e}", exc_info=True)
            return f"Error performing search: {str(e)}"
        
        if not structured_results:
            current_logger.warning(f"⚠️ No results found for query: {search_query}")
            return f"No results found for: {search_query}"
        
        # Limit to max_urls
        structured_results = structured_results[:max_urls]
        
        # Research each URL and save incrementally
        processed_count = 0
        saved_count = 0
        
        for i, result in enumerate(structured_results):
            current_logger.info(f"Processing search result {i+1}: {result.get('title', 'Untitled')}")
            
            url = result.get("url", "")
            if not url or not url.startswith("http"):
                current_logger.warning(f"⚠️ Skipping invalid URL: {url}")
                continue
                
            # Use the snippet as additional context if available
            snippet = result.get("snippet", "")
            enhanced_context = f"{search_query} {snippet}" if snippet else search_query
            
            # Research and save this URL
            save_result = _research_and_save_url_impl(url, query_context=enhanced_context)
            processed_count += 1
            
            if "Successfully researched and saved" in save_result:
                saved_count += 1
            else:
                current_logger.warning(f"⚠️ Failed to save or process: {result.get('title', 'Untitled')}")
        
        current_logger.info(f"🔍 Search '{search_query}': Processed {processed_count} URLs, saved {saved_count} sources")
        
        return f"Completed search for '{search_query}'. Processed {processed_count} URLs, saved {saved_count} sources."
        
    except Exception as e:
        current_logger.error(f"❌ Error in research_search_results: {e}", exc_info=True)
        return f"Error researching search results: {str(e)}"

# Function to generate a summary of all raw content
def generate_content_summary(uid: str, session_id: int) -> str:
    """
    Generate a summary of all raw content collected during research
    """
    try:
        db = get_db()
        # Get all sources for this research session
        sources_ref = db.collection("research").document(uid).collection("sources")
        query = sources_ref.where("research_session_id", "==", session_id)
        sources = query.get()
        
        # Collect all content
        all_content = []
        for source in sources:
            data = source.to_dict()
            if "snippet" in data and data["snippet"]:
                all_content.append(data["snippet"])
            elif "content" in data and data["content"]:
                # If no snippet, use first 300 chars of content
                content_preview = data["content"][:300] + "..." if len(data["content"]) > 300 else data["content"]
                all_content.append(content_preview)
        
        # Combine all content into a single summary
        combined_content = "\n\n".join(all_content)
        
        # Limit summary size
        max_summary_size = 10000  # 10KB should be plenty
        if len(combined_content) > max_summary_size:
            combined_content = combined_content[:max_summary_size] + "..."
            
        return combined_content
    except Exception as e:
        logger.error(f"Error generating content summary: {e}", exc_info=True)
        return "Error generating content summary"

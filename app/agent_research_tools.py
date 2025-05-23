"""
Custom research tools that save data incrementally to Firestore
"""
import logging
import time
import requests
import re
from typing import Optional
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
        
        research_doc = ResearchDoc(
            title=title,
            url=url,
            content=clean_content,
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
        
        research_doc = ResearchDoc(
            title=title,
            url=url,
            content=clean_content,
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
            try:
                loop = asyncio.get_running_loop()
                # We're in an event loop, use asyncio.create_task() or run_coroutine_threadsafe()
                import concurrent.futures
                import threading
                
                def run_in_thread():
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        return new_loop.run_until_complete(Runner.run(search_agent, search_prompt))
                    finally:
                        new_loop.close()
                
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(run_in_thread)
                    result = future.result(timeout=60)  # 60 second timeout
                    
            except RuntimeError:
                # No event loop running, we can use asyncio.run()
                result = asyncio.run(Runner.run(search_agent, search_prompt))
            
            current_logger.info(f"🔍 Agent search completed for: {search_query}")
            current_logger.info(f"📝 Agent result: {result.final_output if hasattr(result, 'final_output') else result}")
            
            # Parse the agent result to extract structured data
            try:
                import json
                import re
                
                # Get the raw result text
                result_text = result.final_output if hasattr(result, 'final_output') else str(result)
                
                # Try to extract JSON from the result
                json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group()
                    parsed_result = json.loads(json_str)
                    search_results = parsed_result.get('results', [])
                    current_logger.info(f"✅ Successfully parsed {len(search_results)} structured results")
                else:
                    # Fallback: try to extract URLs from text using regex
                    current_logger.warning("No JSON found, attempting to extract URLs from text")
                    urls = re.findall(r'https?://[^\s\)]+', result_text)
                    search_results = []
                    for url in urls:
                        search_results.append({
                            'url': url,
                            'title': f"Search result from {search_query}",
                            'snippet': result_text[:200] + "..." if len(result_text) > 200 else result_text
                        })
                    current_logger.info(f"🔄 Extracted {len(search_results)} URLs from text result")
                    
            except Exception as parse_error:
                current_logger.error(f"❌ Error parsing agent result: {parse_error}")
                search_results = []
        
        except Exception as agent_error:
            current_logger.error(f"❌ Error using agent search: {agent_error}", exc_info=True)
            search_results = []
        
        if not search_results or len(search_results) == 0:
            current_logger.warning(f"No search results for query: {search_query}")
            return f"No search results found for: {search_query}"
        
        current_logger.info(f"🔍 Found {len(search_results)} search results for: {search_query}")
        
        # Process each promising search result
        processed_count = 0
        saved_sources = []
        
        for i, result in enumerate(search_results[:max_urls], 1):
            url = result.get('url', '')
            title = result.get('title', 'Unknown Title')
            
            if not url:
                current_logger.warning(f"Skipping result {i}: missing URL")
                continue
            
            current_logger.info(f"Processing search result {i}: {title}")
            
            try:
                # Call the actual implementation instead of the decorated function
                result_summary = _research_and_save_url_impl(url, query_context=f"Search query: {search_query}")
                
                if result_summary and "successfully saved" in result_summary:
                    saved_sources.append({
                        'url': url,
                        'title': title,
                        'snippet': result.get('snippet', '')
                    })
                    processed_count += 1
                    current_logger.info(f"✅ Successfully processed and saved: {title}")
                else:
                    current_logger.warning(f"⚠️ Failed to save or process: {title}")
                    
            except Exception as e:
                current_logger.error(f"❌ Error processing {url}: {e}")
                continue
        
        summary = f"🔍 Search '{search_query}': Processed {len(search_results[:max_urls])} URLs, saved {processed_count} sources\n" + "\n".join([f"• {source['title']}: Saved" for source in saved_sources])
        current_logger.info(summary)
        return summary
        
    except Exception as e:
        current_logger.error(f"❌ Error in research_search_results for '{search_query}': {e}", exc_info=True)
        return f"Error searching '{search_query}': {str(e)}"

# app/agent_tools.py

import requests
from bs4 import BeautifulSoup
from agents import WebSearchTool, function_tool

# Use OpenAI's built-in web search tool for up-to-date information
web_search_tool = WebSearchTool()  # Provides .name and .description internally

@function_tool
def agent_fetch_url(url: str, max_content_size: int = 15000) -> str:
    """
    Fetch the content of a webpage (up to max_content_size characters).
    Returns the raw HTML or an error message.
    """
    try:
        resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
    except Exception as e:
        return f"Error: Failed to fetch {url} - {e}"
    content = resp.text
    if len(content) > max_content_size:
        content = content[:max_content_size]  # truncate to limit size
    return content

@function_tool
def agent_strip_html(html: str, max_output_size: int = 5000) -> str:
    """
    Strip HTML tags and scripts from content, returning plain text (up to max_output_size chars).
    """
    try:
        soup = BeautifulSoup(html, "lxml")
        text = soup.get_text(separator=" ", strip=True)
    except Exception as e:
        return f"Error: Could not parse HTML content - {e}"
    if len(text) > max_output_size:
        text = text[:max_output_size]
    return text

def research_and_save_url(url: str, user_id: str, session_id: int) -> dict:
    """
    Fetch a URL's content, summarize it, and save to Firestore (collection: research/{user_id}/sources).
    Returns a status dict with success or error info.
    """
    # Fetch HTML content
    html = agent_fetch_url(url)
    if html.startswith("Error:"):
        return {"status": "failed", "url": url, "error": html}
    # Parse title and text snippet
    title = "Unknown"
    try:
        soup = BeautifulSoup(html, "lxml")
        title_tag = soup.find("title")
        if title_tag:
            title = title_tag.get_text()
    except Exception:
        title = url  # use URL as title if parsing fails
    text_content = agent_strip_html(html)
    snippet = text_content[:200] if text_content else ""
    content_summary = text_content[:1000] if text_content else ""
    # Save to Firestore under research collection
    db = __import__('app.database').database.get_db()
    sources_col = db.collection("research").document(user_id).collection("sources")
    try:
        sources_col.add({
            "title": title,
            "url": url,
            "snippet": snippet,
            "content": content_summary
        })
    except Exception as e:
        return {"status": "failed", "url": url, "error": f"Firestore save error: {e}"}
    return {"status": "success", "url": url, "saved": True}

def research_search_results(query: str, user_id: str, session_id: int) -> dict:
    """
    Use WebSearchTool to find results for a query, save top results to Firestore via research_and_save_url.
    """
    try:
        results = web_search_tool(query, num_results=3)  # top 3 results for the query
    except Exception as e:
        return {"status": "failed", "query": query, "error": f"Search error: {e}"}
    if not results:
        return {"status": "success", "query": query, "results_saved": 0}
    saved_count = 0
    for res in results:
        url = res.get("url")
        if url:
            # Save each result's content to Firestore
            save_status = research_and_save_url(url, user_id, session_id)
            if save_status.get("status") == "success":
                saved_count += 1
    return {"status": "success", "query": query, "results_saved": saved_count}

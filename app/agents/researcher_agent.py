# app/agents/researcher_agent.py

import logging
from typing import List
from agents import Agent, ModelSettings, function_tool, AgentOutputSchema
from app.agent_tools import web_search_tool, agent_fetch_url, agent_strip_html, research_and_save_url, research_search_results
from app.schemas import ResearchDoc

logger = logging.getLogger(__name__)

def researcher_agent(user_id: str, session_id: int) -> Agent:
    """
    Autonomous Researcher agent that uses web tools to gather factual info about a person.
    Saves relevant findings to Firestore.
    """
    instructions = """You are an expert web research agent tasked with gathering factual, verifiable information about a person for a personal website.
Follow this OPTIMIZED research strategy to complete your task efficiently:

1. LIMIT SCOPE: For each search query, only process the 2-3 most promising results to avoid excessive turns.
2. USE EFFICIENT TOOLS:
   • research_search_results(query) - This is your PRIMARY tool that automatically finds and saves content
   • research_and_save_url(url) - Use for specific high-value URLs only
   • Avoid using agent_fetch_url/agent_strip_html directly unless absolutely necessary

3. PRIORITIZE QUALITY OVER QUANTITY:
   • Focus on professional profiles, company pages, and news articles
   • Skip social media unless it contains unique professional information
   • Ignore generic or irrelevant content

4. WORK METHODICALLY:
   • Process one query completely before moving to the next
   • If a query yields no relevant results, immediately move on
   • Limit to maximum 3-4 search queries total

When finished, output a simple summary list: e.g., ["Processed 3 queries", "Saved 4 sources"] or [] if nothing found.
"""
    # Bind user_id and session_id into tool calls via closures
    @function_tool
    def save_url(url: str) -> dict:
        """Fetch and save the content of the given URL (tool for researcher)."""
        return research_and_save_url(url, user_id, session_id)
    @function_tool
    def search_and_save(query: str) -> dict:
        """Search for the query and save top results (tool for researcher)."""
        return research_search_results(query, user_id, session_id)
    return Agent(
        name="AutonomousResearcher",
        instructions=instructions,
        model="gpt-4o-mini",
        tools=[web_search_tool, agent_fetch_url, agent_strip_html, search_and_save, save_url],
        output_type=AgentOutputSchema(List[ResearchDoc], strict_json_schema=False),
        model_settings=ModelSettings(temperature=0.2, max_tokens=1200)
    )
